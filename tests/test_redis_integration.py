import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from urllib.parse import urlparse

import pytest
import redis.asyncio as redis
from agent_contracts import (
    AgentCapabilityMetadata,
    AgentEvent,
    InferenceExecutionTelemetry,
    RootAuthorityRecord,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    TaskResult,
)
from analysis_agent.application.processor import AnalysisProcessor
from analysis_agent.infrastructure.deterministic_analyzer import (
    DeterministicMaterialAnalyzer,
)
from analysis_agent.infrastructure.redis_bus import RedisBus as AnalysisRedisBus
from orchestrator.domain.task import (
    ConflictingContinuationError,
    ConflictingRootAuthorityError,
    ConflictingStageArtifactError,
    ConflictingTaskInputError,
    TaskLineage,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway
from research_agent.domain.evidence import AuthorityTier, EvaluatedEvidence, SourceType
from research_agent.domain.models import ResearchResult
from research_agent.infrastructure.redis_bus import RedisBus as ResearchRedisBus
from worker.infrastructure.redis_bus import RedisBus as WorkerRedisBus

pytestmark = [pytest.mark.anyio, pytest.mark.redis]

REQUIRED_TEST_DB = "15"


def _sanitize_redis_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        if parsed.password:
            user = parsed.username or ""
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            userinfo = f"{user}:***@" if user else ":***@"
            masked_netloc = f"{userinfo}{host}{port}"
            return parsed._replace(netloc=masked_netloc).geturl()
        return url
    except ValueError:
        return "<invalid-url>"


def validate_test_redis_url(url: str) -> None:
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValueError(f"Invalid REDIS_URL format: {exc}") from exc

    if parsed.scheme not in ("redis", "rediss"):
        sanitized = _sanitize_redis_url(url)
        raise ValueError(
            f"Safety check failed: REDIS_URL scheme must be 'redis' or 'rediss', got endpoint: {sanitized}"
        )

    db_segment = parsed.path.strip("/")
    if db_segment != REQUIRED_TEST_DB:
        sanitized = _sanitize_redis_url(url)
        raise ValueError(
            f"Safety check failed: Redis integration tests require database {REQUIRED_TEST_DB} "
            f"to prevent accidental data loss, got endpoint: {sanitized}"
        )


@pytest.fixture
def redis_url() -> str:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        pytest.skip("Redis integration tests disabled: REDIS_URL environment variable is not set")
    try:
        validate_test_redis_url(url)
    except ValueError as exc:
        pytest.fail(str(exc))
    return url


@pytest.fixture
async def clean_redis(redis_url: str) -> AsyncGenerator[str, None]:
    validate_test_redis_url(redis_url)
    client = redis.from_url(redis_url, decode_responses=True)
    sanitized_url = _sanitize_redis_url(redis_url)
    try:
        await asyncio.wait_for(client.ping(), timeout=1.0)
    except (redis.ConnectionError, OSError, TimeoutError) as exc:
        await client.aclose()
        pytest.fail(
            f"Configured Redis endpoint is unreachable at {sanitized_url} within 1.0s timeout: {exc}"
        )

    # Clean before test
    await client.flushdb()
    try:
        yield redis_url
    finally:
        try:
            await client.flushdb()
        finally:
            await client.aclose()


async def test_redis_connection_and_client_lifecycle(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    bus = WorkerRedisBus(clean_redis, queue_name=f"agent.tasks.test_{uuid.uuid4().hex[:8]}")

    try:
        gateway_pong = await gateway._client.ping()
        assert gateway_pong is True

        bus_pong = await bus.client.ping()
        assert bus_pong is True
    finally:
        await gateway.aclose()
        await bus.aclose()


async def test_redis_task_dispatch_and_consumption_round_trip(clean_redis: str) -> None:
    queue_name = f"agent.tasks.test_{uuid.uuid4().hex[:8]}"
    gateway = RedisGateway(clean_redis)
    bus = WorkerRedisBus(clean_redis, queue_name=queue_name)

    try:
        task = TaskRequested.create("Perform integration analysis on real Redis wire protocol")
        task_id = task.task_id

        # 1. Dispatch through RedisGateway (sets task.status:{task_id} and rpush to queue)
        await gateway.dispatch(task, target=queue_name)

        # 2. Verify persisted pending status directly in Redis
        raw_status = await gateway._client.get(f"task.status:{task_id}")
        assert raw_status is not None
        status_data = json.loads(raw_status)
        assert status_data["task_id"] == task_id
        assert status_data["status"] == "pending"

        # 3. Consume through WorkerRedisBus.next_task()
        consumed_task = await asyncio.wait_for(bus.next_task(), timeout=2.0)

        assert consumed_task.task_id == task_id
        assert consumed_task.prompt == "Perform integration analysis on real Redis wire protocol"
        assert consumed_task.created_at == task.created_at
        assert consumed_task.version == task.version

        # 4. Verify queue is now empty
        remaining = await gateway._client.llen(queue_name)
        assert remaining == 0
    finally:
        await gateway.aclose()
        await bus.aclose()


async def test_analysis_agent_dispatch_lifecycle_and_result_round_trip(
    clean_redis: str,
) -> None:
    gateway = RedisGateway(clean_redis)
    bus = AnalysisRedisBus(clean_redis)
    agent_id = f"analysis-agent-{uuid.uuid4().hex[:8]}"
    metadata = AgentCapabilityMetadata(
        agent_id=agent_id,
        agent_type="analysis-agent",
        capabilities=["analysis"],
        providers={"analysis_engine": "deterministic"},
        dispatch_target="agent.tasks.analysis",
    )
    task = TaskRequested.create(
        "The rover maps caves. The rover records samples. Does the map show water?"
    )
    processor = AnalysisProcessor(
        agent_id,
        bus,
        DeterministicMaterialAnalyzer(),
        bus,
    )

    try:
        await bus.register(metadata)
        await bus.publish(AgentEvent.create(agent_id, "idle", "Waiting for work"))
        await gateway.dispatch(task, target="agent.tasks.analysis")

        consumed = await asyncio.wait_for(bus.next_task(), timeout=2.0)
        await processor.run_once(consumed)

        registered = await gateway.agents()
        agent = next(agent for agent in registered if agent["agent_id"] == agent_id)
        assert agent["agent_type"] == "analysis-agent"
        assert agent["capabilities"] == ["analysis"]
        assert agent["dispatch_target"] == "agent.tasks.analysis"

        result = await gateway.get_result(task.task_id)
        assert result is not None
        assert result["status"] == "completed"
        assert result["output"] == (
            "Analysis report\n"
            "Paragraphs: 1\n"
            "Sentences: 3\n"
            "Words: 13\n"
            "Repeated terms: rover (2)\n"
            "Question sentences:\n"
            "- Does the map show water?\n"
            "Scope: Only supplied material was analyzed; no retrieval was performed."
        )

        events = await gateway.events(limit=10)
        task_states = [
            event["state"]
            for event in events
            if event.get("task_id") == task.task_id
        ]
        assert task_states == ["idle", "working"]
    finally:
        await gateway.aclose()
        await bus.aclose()


async def test_redis_result_persistence_and_retrieval_round_trip(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    worker_bus = WorkerRedisBus(clean_redis)
    research_bus = ResearchRedisBus(clean_redis)

    try:
        # 1. Worker TaskResult round trip
        worker_task_id = f"task_worker_{uuid.uuid4().hex[:8]}"
        worker_result = TaskResult(
            task_id=worker_task_id,
            status="completed",
            output="General worker finished successfully",
            acknowledged=True,
        )
        await worker_bus.save(worker_result)

        retrieved_worker = await gateway.get_result(worker_task_id)
        assert retrieved_worker is not None
        assert retrieved_worker["task_id"] == worker_task_id
        assert retrieved_worker["status"] == "completed"
        assert retrieved_worker["output"] == "General worker finished successfully"
        assert retrieved_worker["acknowledged"] is True

        # 2. ResearchResult with sources round trip
        evidence = EvaluatedEvidence(
            url="https://example.org/spec",
            title="Specification Doc",
            content="Details on wire protocol",
            domain="example.org",
            authority_tier=AuthorityTier.HIGH,
            source_type=SourceType.OFFICIAL_DOCS,
            quality_signals=("verified", "official"),
            relevance_signal="high",
            recency_indicator="recent",
            rank_score=0.95,
            retained=True,
        )
        research_task_id_with_sources = f"task_research_src_{uuid.uuid4().hex[:8]}"
        research_result_with_sources = ResearchResult(
            task_id=research_task_id_with_sources,
            output="Detailed synthesis output",
            acknowledged=True,
            truncated=False,
            sources=(evidence,),
        )
        await research_bus.save(research_result_with_sources)

        retrieved_research_1 = await gateway.get_result(research_task_id_with_sources)
        assert retrieved_research_1 is not None
        assert retrieved_research_1["task_id"] == research_task_id_with_sources
        assert retrieved_research_1["status"] == "completed"
        assert retrieved_research_1["output"] == "Detailed synthesis output"
        assert retrieved_research_1["acknowledged"] is True
        assert retrieved_research_1["truncated"] is False
        assert "sources" in retrieved_research_1
        assert len(retrieved_research_1["sources"]) == 1
        assert retrieved_research_1["sources"][0]["url"] == "https://example.org/spec"
        assert retrieved_research_1["sources"][0]["domain"] == "example.org"

        # 3. ResearchResult without sources (conditional source omission)
        research_task_id_no_sources = f"task_research_nosrc_{uuid.uuid4().hex[:8]}"
        research_result_no_sources = ResearchResult(
            task_id=research_task_id_no_sources,
            output="Synthesis without sources",
            acknowledged=True,
            truncated=False,
            sources=(),
        )
        await research_bus.save(research_result_no_sources)

        retrieved_research_2 = await gateway.get_result(research_task_id_no_sources)
        assert retrieved_research_2 is not None
        assert retrieved_research_2["task_id"] == research_task_id_no_sources
        assert retrieved_research_2["status"] == "completed"
        assert retrieved_research_2["output"] == "Synthesis without sources"
        assert retrieved_research_2["acknowledged"] is True
        assert "sources" not in retrieved_research_2
    finally:
        await gateway.aclose()
        await worker_bus.aclose()
        await research_bus.aclose()


async def test_redis_event_and_telemetry_round_trip(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    bus = WorkerRedisBus(clean_redis)

    try:
        agent_id = f"worker_{uuid.uuid4().hex[:8]}"
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        telemetry = InferenceExecutionTelemetry(
            provider_name="cloudflare",
            attempt_index=0,
            latency_ms=88.5,
            status="succeeded",
            model_name="@cf/meta/llama-3.1-8b-instruct",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )
        event = AgentEvent.create(
            agent_id=agent_id,
            state="working",
            detail="Inference execution succeeded",
            task_id=task_id,
            inference_telemetry=telemetry,
        )

        # 1. Publish event to Redis stream and agent state
        await bus.publish(event)

        # 2. Query event stream
        events = await gateway.events(limit=10)
        matching_events = [e for e in events if e.get("task_id") == task_id]
        assert len(matching_events) == 1
        published_event = matching_events[0]
        assert published_event["agent_id"] == agent_id
        assert published_event["state"] == "working"
        assert "-" in published_event["event_id"]

        # 3. Query telemetry parsed from Redis stream
        telemetry_result = await gateway.task_inference_telemetry(task_id)
        assert telemetry_result is not None
        assert telemetry_result["task_id"] == task_id
        telem_data = telemetry_result["inference_telemetry"]
        assert telem_data["provider_name"] == "cloudflare"
        assert telem_data["latency_ms"] == 88.5
        assert telem_data["status"] == "succeeded"
        assert telem_data["model_name"] == "@cf/meta/llama-3.1-8b-instruct"

        # 4. Verify agent state discovery
        agents = await gateway.agents()
        matching_agents = [a for a in agents if a.get("agent_id") == agent_id]
        assert len(matching_agents) == 1
        assert matching_agents[0]["state"] == "working"
    finally:
        await gateway.aclose()
        await bus.aclose()


async def test_redis_root_registration_and_atomic_dispatch_lifecycle(
    clean_redis: str,
) -> None:
    gateway = RedisGateway(clean_redis)
    try:
        root_task = TaskRequested.create("Root task for Redis integration lifecycle")

        # 1. Authoritative registration establishes pending_dispatch, lineage, TaskInput v1, and RootAuthority
        await gateway.register_root_task(root_task)

        status = await gateway.get_status(root_task.task_id)
        assert status == "pending_dispatch"

        lineage = await gateway.get_lineage(root_task.task_id)
        assert lineage is not None
        assert isinstance(lineage, TaskLineage)
        assert lineage.task_id == root_task.task_id
        assert lineage.root_task_id == root_task.task_id
        assert lineage.parent_task_id is None
        assert lineage.is_root is True

        # Verify durable TaskInput v1 and default RootAuthorityRecord persisted
        input_record = await gateway.get_task_input(root_task.task_id, 1)
        assert input_record is not None
        assert input_record.prompt == root_task.prompt
        assert input_record.task_id == root_task.task_id

        authority = await gateway.get_root_authority(root_task.task_id)
        assert authority is not None
        assert authority.root_task_id == root_task.task_id
        assert authority.max_delegation_depth == 1
        assert authority.max_delegation_waves == 1

        # Verify reference-safety: no TTL set on authoritative keys
        assert await gateway._client.ttl(f"task.status:{root_task.task_id}") == -1
        assert await gateway._client.ttl(f"task.lineage:{root_task.task_id}") == -1
        assert await gateway._client.ttl(f"task.input:{root_task.task_id}:1") == -1
        assert await gateway._client.ttl(f"task.authority:{root_task.task_id}") == -1

        # 2. Atomic dispatch pushes to target queue and sets status to pending
        dispatch_target = f"agent.tasks.test_{uuid.uuid4().hex[:8]}"
        await gateway.dispatch(root_task, dispatch_target)

        dispatched_status = await gateway.get_status(root_task.task_id)
        assert dispatched_status == "pending"

        # Verify queue content in Redis
        queued_items = await gateway._client.lrange(dispatch_target, 0, -1)
        assert len(queued_items) == 1
        enqueued_payload = json.loads(queued_items[0])
        assert enqueued_payload["task_id"] == root_task.task_id
        assert enqueued_payload["root_task_id"] == root_task.task_id
        assert enqueued_payload["parent_task_id"] is None
        assert enqueued_payload["version"] == 2

        # 3. Update status transition
        await gateway.update_status(root_task.task_id, "completed")
        completed_status = await gateway.get_status(root_task.task_id)
        assert completed_status == "completed"
    finally:
        await gateway.aclose()


async def test_redis_child_registration_atomicity(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    try:
        root_task = TaskRequested.create("Root task for child atomicity test")
        await gateway.register_root_task(root_task)

        child_id = f"child_{uuid.uuid4().hex[:8]}"
        child_task = TaskRequested(
            task_id=child_id,
            prompt="Child task for atomicity test",
            created_at="2026-11-15T00:00:00+00:00",
            root_task_id=root_task.task_id,
            parent_task_id=root_task.task_id,
            version=2,
        )

        # 1. Authoritative child registration
        await gateway.register_child_task(child_task)

        # Verify child lineage
        child_lineage = await gateway.get_lineage(child_id)
        assert child_lineage is not None
        assert isinstance(child_lineage, TaskLineage)
        assert child_lineage.task_id == child_id
        assert child_lineage.root_task_id == root_task.task_id
        assert child_lineage.parent_task_id == root_task.task_id
        assert child_lineage.is_root is False

        # Verify child status
        child_status = await gateway.get_status(child_id)
        assert child_status == "pending_dispatch"

        # Verify parent index set contains child
        children = await gateway.get_children(root_task.task_id)
        assert children == [child_id]

        # 2. Rejection of invalid tasks
        with pytest.raises(ValueError, match="not a root task"):
            await gateway.register_root_task(child_task)

        with pytest.raises(ValueError, match="not a child task"):
            await gateway.register_child_task(root_task)
    finally:
        await gateway.aclose()


async def test_redis_deterministic_sorted_get_children_retrieval(
    clean_redis: str,
) -> None:
    gateway = RedisGateway(clean_redis)
    try:
        root_task = TaskRequested.create("Root task for sorted children test")
        await gateway.register_root_task(root_task)

        # Register children in deliberately unsorted order
        child_ids = [
            f"child_z_{uuid.uuid4().hex[:6]}",
            f"child_a_{uuid.uuid4().hex[:6]}",
            f"child_m_{uuid.uuid4().hex[:6]}",
        ]
        for cid in child_ids:
            child = TaskRequested(
                task_id=cid,
                prompt=f"Child prompt for {cid}",
                created_at="2026-11-15T00:00:00+00:00",
                root_task_id=root_task.task_id,
                parent_task_id=root_task.task_id,
                version=2,
            )
            await gateway.register_child_task(child)

        # Verify retrieval is strictly sorted lexicographically
        retrieved_children = await gateway.get_children(root_task.task_id)
        assert retrieved_children == sorted(child_ids)

        # Verify set membership duplicate safety
        # Re-registering existing child should not duplicate in SADD
        first_child = TaskRequested(
            task_id=child_ids[0],
            prompt="Duplicate registration test",
            created_at="2026-11-15T00:00:00+00:00",
            root_task_id=root_task.task_id,
            parent_task_id=root_task.task_id,
            version=2,
        )
        await gateway.register_child_task(first_child)
        retrieved_after_duplicate = await gateway.get_children(root_task.task_id)
        assert retrieved_after_duplicate == sorted(child_ids)

        # Unknown parent returns empty list
        empty_children = await gateway.get_children("nonexistent-parent-task-id")
        assert empty_children == []
    finally:
        await gateway.aclose()


async def test_redis_durable_storage_and_reference_safety(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    try:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        now = "2026-11-16T12:00:00+00:00"

        # 1. TaskInput write-once and retrieval
        task_input = TaskInput(
            task_id=task_id,
            prompt="Durable input prompt",
            created_at=now,
            input_version=1,
        )
        await gateway.save_task_input(task_input)

        retrieved_input = await gateway.get_task_input(task_id, 1)
        assert retrieved_input == task_input

        # Idempotent write
        await gateway.save_task_input(task_input)

        # Conflicting write fails
        conflict_input = TaskInput(
            task_id=task_id,
            prompt="Conflicting prompt",
            created_at=now,
            input_version=1,
        )
        with pytest.raises(ConflictingTaskInputError):
            await gateway.save_task_input(conflict_input)

        # 2. StageArtifact write-once and retrieval
        art_id = f"art_{uuid.uuid4().hex[:8]}"
        artifact = StageArtifact(
            artifact_id=art_id,
            producing_task_id=task_id,
            artifact_type="research_material",
            content={"findings": ["alpha", "beta"]},
            created_at=now,
            artifact_version=1,
        )
        await gateway.save_stage_artifact(artifact)

        retrieved_art = await gateway.get_stage_artifact(task_id, art_id, 1)
        assert retrieved_art == artifact

        # Conflicting write fails
        conflict_art = StageArtifact(
            artifact_id=art_id,
            producing_task_id=task_id,
            artifact_type="research_material",
            content={"findings": ["conflict"]},
            created_at=now,
            artifact_version=1,
        )
        with pytest.raises(ConflictingStageArtifactError):
            await gateway.save_stage_artifact(conflict_art)

        # 3. TaskContinuation write-once and retrieval
        child_id = f"child_{uuid.uuid4().hex[:8]}"
        continuation = TaskContinuation(
            parent_task_id=task_id,
            child_task_id=child_id,
            stage="incorporate_analysis_child",
            control_fields={"analysis_focus": "check evidence"},
            continuation_version=1,
            created_at=now,
            updated_at=now,
        )
        await gateway.save_continuation(continuation)

        retrieved_cont = await gateway.get_continuation(task_id, 1)
        assert retrieved_cont == continuation

        # Conflicting write fails
        conflict_cont = TaskContinuation(
            parent_task_id=task_id,
            child_task_id=f"child_{uuid.uuid4().hex[:8]}",
            stage="incorporate_analysis_child",
            control_fields={"analysis_focus": "conflict focus"},
            continuation_version=1,
            created_at=now,
            updated_at=now,
        )
        with pytest.raises(ConflictingContinuationError):
            await gateway.save_continuation(conflict_cont)

        # 4. RootAuthorityRecord write-once and retrieval
        authority = RootAuthorityRecord.create_default(task_id, created_at=now)
        await gateway.save_root_authority(authority)

        retrieved_auth = await gateway.get_root_authority(task_id)
        assert retrieved_auth == authority

        # Conflicting write fails
        conflict_auth = RootAuthorityRecord.create_default(
            root_task_id=task_id,
            created_at="2026-01-01T00:00:00+00:00",
        )
        with pytest.raises(ConflictingRootAuthorityError):
            await gateway.save_root_authority(conflict_auth)

        # 5. Reference safety: verify NO TTL on all authoritative keys
        assert await gateway._client.ttl(f"task.input:{task_id}:1") == -1
        assert await gateway._client.ttl(f"task.artifact:{task_id}:{art_id}:1") == -1
        assert await gateway._client.ttl(f"task.continuation:{task_id}:1") == -1
        assert await gateway._client.ttl(f"task.authority:{task_id}") == -1
    finally:
        await gateway.aclose()
