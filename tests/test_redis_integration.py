import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from urllib.parse import urlparse

import pytest
import redis.asyncio as redis
from agent_contracts import (
    ActiveClaim,
    AgentCapabilityMetadata,
    AgentEvent,
    DelegationGrant,
    InferenceExecutionTelemetry,
    RootAuthorityRecord,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    TaskResult,
    WakeUpHint,
    normalize_canonical_utc_timestamp,
)
from analysis_agent.application.processor import AnalysisProcessor
from analysis_agent.infrastructure.deterministic_analyzer import (
    DeterministicMaterialAnalyzer,
)
from analysis_agent.infrastructure.redis_bus import RedisBus as AnalysisRedisBus
from orchestrator.application.ports import (
    CapabilityMismatchError,
    ClaimDeliveryLeaseExpiredError,
    ClaimNotFoundError,
    FencingEpochRaceError,
    TaskAlreadyClaimedError,
    TaskNotClaimableError,
)
from orchestrator.domain.task import (
    ConflictingAuthorityGrantError,
    ConflictingChildTaskRegistrationError,
    ConflictingContinuationError,
    ConflictingRootAuthorityError,
    ConflictingRootTaskRegistrationError,
    ConflictingStageArtifactError,
    ConflictingTaskInputError,
    CorruptAuthoritativeRecordError,
    NonContiguousVersionError,
    TaskLineage,
)
from orchestrator.infrastructure.redis_adapter import CLAIM_TASK_LUA, RedisGateway
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
    except (redis.ConnectionError, OSError) as exc:
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
        future = "2099-01-01T00:00:00Z"
        claim_art = ActiveClaim(
            task_id=task_id,
            assignment_id="assign-art-1",
            claim_id="claim-art-1",
            root_task_id=task_id,
            turn_id="turn-art-1",
            runtime_principal_id="principal-research",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="a" * 64,
            request_id="req-art-1",
            claimed_at=now,
            delivery_expires_at=future,
            state="acknowledged",
            acknowledged_at=now,
        )
        await gateway._client.set(f"task.claim:{task_id}", claim_art.to_json())
        await gateway._client.set(f"task.fence:{task_id}", "1")

        art_id = f"art_{uuid.uuid4().hex[:8]}"
        artifact = StageArtifact(
            artifact_id=art_id,
            producing_task_id=task_id,
            artifact_type="research_material",
            content={"material": "findings alpha and beta"},
            created_at=now,
            artifact_version=1,
        )
        await gateway.save_stage_artifact(artifact, "assign-art-1", "principal-research", 1)

        retrieved_art = await gateway.get_stage_artifact(task_id, art_id, 1)
        assert retrieved_art == artifact

        # Assert direct Redis wire contents for record payload and index set membership
        raw_art_payload = await gateway._client.get(f"task.artifact:{task_id}:{art_id}:1")
        assert raw_art_payload == artifact.to_json()
        raw_index_members = await gateway._client.smembers(f"task.artifacts:{task_id}")
        assert f"{art_id}:1" in raw_index_members

        # Conflicting write fails
        conflict_art = StageArtifact(
            artifact_id=art_id,
            producing_task_id=task_id,
            artifact_type="research_material",
            content={"material": "conflict findings"},
            created_at=now,
            artifact_version=1,
        )
        with pytest.raises(ConflictingStageArtifactError):
            await gateway.save_stage_artifact(conflict_art, "assign-art-1", "principal-research", 1)

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

        # Assert direct Redis wire contents for continuation record and latest pointer
        raw_cont_payload = await gateway._client.get(f"task.continuation:{task_id}:1")
        assert raw_cont_payload == continuation.to_json()
        assert await gateway._client.get(f"task.continuation_latest:{task_id}") == "1"

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
        assert await gateway._client.ttl(f"task.artifacts:{task_id}") == -1
        assert await gateway._client.ttl(f"task.continuation:{task_id}:1") == -1
        assert await gateway._client.ttl(f"task.continuation_latest:{task_id}") == -1
        assert await gateway._client.ttl(f"task.authority:{task_id}") == -1
    finally:
        await gateway.aclose()


async def test_redis_root_registration_marker_concurrency_and_partial_init(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    try:
        now = "2026-11-16T12:00:00+00:00"
        task_id = f"root_{uuid.uuid4().hex[:8]}"
        root_task = TaskRequested(
            task_id=task_id,
            prompt="Root registration concurrency test",
            created_at=now,
            root_task_id=task_id,
            parent_task_id=None,
            version=2,
        )

        # 1. Partial initialization rejection: status key alone exists
        await gateway._client.set(
            f"task.status:{task_id}",
            json.dumps({"status": "pending_dispatch", "task_id": task_id}),
        )
        with pytest.raises(ConflictingRootTaskRegistrationError, match="inconsistent or partial"):
            await gateway.register_root_task(root_task)
        # Registration marker and remaining bundle keys must NOT have been written
        assert await gateway._client.get(f"task.registration:{task_id}") is None
        assert await gateway._client.get(f"task.lineage:{task_id}") is None
        assert await gateway._client.get(f"task.input:{task_id}:1") is None
        assert await gateway._client.get(f"task.authority:{task_id}") is None
        await gateway._client.delete(f"task.status:{task_id}")

        # Partial initialization rejection: registration marker alone exists
        partial_id = f"root_{uuid.uuid4().hex[:8]}"
        partial_task = TaskRequested(
            task_id=partial_id,
            prompt="Partial marker test",
            created_at=now,
            root_task_id=partial_id,
            parent_task_id=None,
            version=2,
        )
        await gateway._client.set(
            f"task.registration:{partial_id}",
            json.dumps({"registered_at": now, "task_id": partial_id}),
        )
        with pytest.raises(ConflictingRootTaskRegistrationError, match="inconsistent or partial"):
            await gateway.register_root_task(partial_task)
        assert await gateway._client.get(f"task.status:{partial_id}") is None
        assert await gateway._client.get(f"task.lineage:{partial_id}") is None
        assert await gateway._client.get(f"task.input:{partial_id}:1") is None
        assert await gateway._client.get(f"task.authority:{partial_id}") is None
        await gateway._client.delete(f"task.registration:{partial_id}")

        # 2. Identical concurrency: 10 concurrent registrations of the same clean root task
        ident_id = f"root_{uuid.uuid4().hex[:8]}"
        clean_task = TaskRequested(
            task_id=ident_id,
            prompt="Concurrent root registration",
            created_at=now,
            root_task_id=ident_id,
            parent_task_id=None,
            version=2,
        )
        tasks = [gateway.register_root_task(clean_task) for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(r is None for r in results)

        # Verify all authoritative keys exist and have no TTL
        marker_key = f"task.registration:{clean_task.task_id}"
        status_key = f"task.status:{clean_task.task_id}"
        lineage_key = f"task.lineage:{clean_task.task_id}"
        input_key = f"task.input:{clean_task.task_id}:1"
        authority_key = f"task.authority:{clean_task.task_id}"

        raw_marker = await gateway._client.get(marker_key)
        assert raw_marker is not None
        assert await gateway._client.ttl(marker_key) == -1
        assert await gateway._client.ttl(status_key) == -1
        assert await gateway._client.ttl(lineage_key) == -1
        assert await gateway._client.ttl(input_key) == -1
        assert await gateway._client.ttl(authority_key) == -1

        # Marker contains valid JSON with task_id
        marker_data = json.loads(raw_marker)
        assert marker_data["task_id"] == clean_task.task_id

        # 3. Conflicting concurrent initialization: same task ID, conflicting immutable input
        conflict_id = f"root_{uuid.uuid4().hex[:8]}"
        task_a = TaskRequested(
            task_id=conflict_id,
            prompt="Candidate prompt A",
            created_at=now,
            root_task_id=conflict_id,
            parent_task_id=None,
            version=2,
        )
        task_b = TaskRequested(
            task_id=conflict_id,
            prompt="Conflicting candidate prompt B",
            created_at=now,
            root_task_id=conflict_id,
            parent_task_id=None,
            version=2,
        )
        conflict_results = await asyncio.gather(
            gateway.register_root_task(task_a),
            gateway.register_root_task(task_b),
            return_exceptions=True,
        )
        successes = [r for r in conflict_results if r is None]
        failures = [r for r in conflict_results if isinstance(r, ConflictingRootTaskRegistrationError)]
        assert len(successes) == 1
        assert len(failures) == 1

        # Verified stored bundle has exactly one accepted identity
        persisted_input = await gateway.get_task_input(conflict_id, 1)
        assert persisted_input is not None
        assert persisted_input.prompt in (task_a.prompt, task_b.prompt)

        # 4. Status advancement: status advances to 'running', replay must NOT reset it
        await gateway.update_status(clean_task.task_id, "running")
        assert await gateway.get_status(clean_task.task_id) == "running"

        # Replay registration: succeeds idempotently, status remains 'running'
        await gateway.register_root_task(clean_task)
        assert await gateway.get_status(clean_task.task_id) == "running"
    finally:
        await gateway.aclose()


async def test_redis_continuation_latest_pointer_corruption_and_regression(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    try:
        p_id = f"parent_{uuid.uuid4().hex[:8]}"
        c_id = f"child_{uuid.uuid4().hex[:8]}"
        now = "2026-11-16T12:00:00+00:00"

        cont_v1 = TaskContinuation(
            parent_task_id=p_id,
            child_task_id=c_id,
            stage="incorporate_analysis_child",
            control_fields={"analysis_focus": "v1 focus"},
            continuation_version=1,
            created_at=now,
            updated_at=now,
        )
        cont_v2 = TaskContinuation(
            parent_task_id=p_id,
            child_task_id=c_id,
            stage="incorporate_analysis_child",
            control_fields={"analysis_focus": "v2 focus"},
            continuation_version=2,
            created_at=now,
            updated_at=now,
        )

        # 1. v1 creation when latest pointer already exists -> CORRUPT_POINTER
        latest_key = f"task.continuation_latest:{p_id}"
        await gateway._client.set(latest_key, "1")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Continuation latest pointer"):
            await gateway.save_continuation(cont_v1)
        await gateway._client.delete(latest_key)

        # Normal v1 creation succeeds
        await gateway.save_continuation(cont_v1)
        assert await gateway._client.get(latest_key) == "1"

        # 2. v2 creation when latest pointer is missing -> CORRUPT_POINTER
        await gateway._client.delete(latest_key)
        with pytest.raises(CorruptAuthoritativeRecordError, match="Continuation latest pointer"):
            await gateway.save_continuation(cont_v2)

        # 3. v2 creation when latest pointer is non-numeric -> CORRUPT_POINTER
        await gateway._client.set(latest_key, "corrupt_string")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Continuation latest pointer"):
            await gateway.save_continuation(cont_v2)

        # 4. v2 creation when latest pointer >= 2 -> CORRUPT_POINTER
        await gateway._client.set(latest_key, "2")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Continuation latest pointer"):
            await gateway.save_continuation(cont_v2)

        # Reset latest to 1 -> v2 creation succeeds
        await gateway._client.set(latest_key, "1")
        await gateway.save_continuation(cont_v2)
        assert await gateway._client.get(latest_key) == "2"

        # 5. Replay v1 when latest pointer has been corrupted/regressed to "0"
        await gateway._client.set(latest_key, "0")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Continuation latest pointer"):
            await gateway.save_continuation(cont_v1)
        # Replay must NOT silently repair the corrupt pointer
        assert await gateway._client.get(latest_key) == "0"
    finally:
        await gateway.aclose()


async def test_redis_version_gaps_concurrency_and_corrupt_payload_retrieval(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    try:
        t_id = f"task_{uuid.uuid4().hex[:8]}"
        now = "2026-11-16T12:00:00+00:00"

        # 1. Version gaps
        # TaskInput v1 then v3
        inp_v1 = TaskInput(task_id=t_id, prompt="prompt v1", created_at=now, input_version=1)
        inp_v3 = TaskInput(task_id=t_id, prompt="prompt v3", created_at=now, input_version=3)
        await gateway.save_task_input(inp_v1)
        with pytest.raises(NonContiguousVersionError, match="prior version 2 does not exist"):
            await gateway.save_task_input(inp_v3)

        # Establish active acknowledged claim on t_id
        claim_art_gap = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-gap-1",
            claim_id="claim-gap-1",
            root_task_id=t_id,
            turn_id="turn-gap-1",
            runtime_principal_id="principal-research",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="g" * 64,
            request_id="req-gap-1",
            claimed_at=now,
            delivery_expires_at="2099-01-01T00:00:00Z",
            state="acknowledged",
            acknowledged_at=now,
        )
        await gateway._client.set(f"task.claim:{t_id}", claim_art_gap.to_json())
        await gateway._client.set(f"task.fence:{t_id}", "1")

        # StageArtifact v1 then v3
        art_id = f"art_{uuid.uuid4().hex[:8]}"
        art_v1 = StageArtifact(
            artifact_id=art_id,
            producing_task_id=t_id,
            artifact_type="research_material",
            content={"material": "material v1"},
            created_at=now,
            artifact_version=1,
        )
        art_v3 = StageArtifact(
            artifact_id=art_id,
            producing_task_id=t_id,
            artifact_type="research_material",
            content={"material": "material v3"},
            created_at=now,
            artifact_version=3,
        )
        await gateway.save_stage_artifact(art_v1, "assign-gap-1", "principal-research", 1)
        with pytest.raises(NonContiguousVersionError, match="prior version 2 does not exist"):
            await gateway.save_stage_artifact(art_v3, "assign-gap-1", "principal-research", 1)

        # 2. Corrupt payload retrieval
        # Corrupt TaskInput
        await gateway._client.set(f"task.input:{t_id}:1", "invalid { json")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt TaskInput record"):
            await gateway.get_task_input(t_id, 1)

        # Corrupt StageArtifact
        await gateway._client.set(f"task.artifact:{t_id}:{art_id}:1", '{"invalid": "schema"}')
        with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt StageArtifact record"):
            await gateway.get_stage_artifact(t_id, art_id, 1)

        # Corrupt TaskContinuation
        await gateway._client.set(f"task.continuation:{t_id}:1", "corrupt text")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt TaskContinuation record"):
            await gateway.get_continuation(t_id, 1)

        # Corrupt RootAuthorityRecord
        await gateway._client.set(f"task.authority:{t_id}", '{"broken": 999}')
        with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt RootAuthorityRecord"):
            await gateway.get_root_authority(t_id)

        # 3. Concurrency on same vs conflicting version write
        t2_id = f"task_{uuid.uuid4().hex[:8]}"
        inp_a = TaskInput(task_id=t2_id, prompt="concurrent prompt", created_at=now, input_version=1)
        inp_b = TaskInput(task_id=t2_id, prompt="conflicting concurrent", created_at=now, input_version=1)

        results = await asyncio.gather(
            gateway.save_task_input(inp_a),
            gateway.save_task_input(inp_a),
            return_exceptions=True,
        )
        assert all(r is None for r in results)

        # Conflicting concurrent writes
        results_conflict = await asyncio.gather(
            gateway.save_task_input(inp_a),
            gateway.save_task_input(inp_b),
            return_exceptions=True,
        )
        # One succeeded, one raised ConflictingTaskInputError
        successes = [r for r in results_conflict if r is None]
        failures = [r for r in results_conflict if isinstance(r, ConflictingTaskInputError)]
        assert len(successes) == 1
        assert len(failures) == 1
    finally:
        await gateway.aclose()


async def test_redis_authority_grant_write_once_and_conflict(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    try:
        t_id = f"root_{uuid.uuid4().hex[:8]}"
        now = "2026-11-16T12:00:00+00:00"
        grant_id = f"grant_{uuid.uuid4().hex[:8]}"
        grant = DelegationGrant(
            grant_id=grant_id,
            root_task_id=t_id,
            issued_to_assignment_id="assign-001",
            issued_to_principal_id="principal-research-1",
            fencing_epoch=1,
            turn_id="turn-001",
            issued_at=now,
        )

        # 1. First save succeeds
        await gateway.save_authority_grant(grant)

        # 2. Get returns identical grant
        retrieved = await gateway.get_authority_grant(t_id, grant_id)
        assert retrieved == grant

        # 3. Idempotent replay succeeds
        await gateway.save_authority_grant(grant)

        # 4. Conflicting save raises ConflictingAuthorityGrantError
        conflict_grant = DelegationGrant(
            grant_id=grant_id,
            root_task_id=t_id,
            issued_to_assignment_id="assign-002",
            issued_to_principal_id="principal-research-2",
            fencing_epoch=1,
            turn_id="turn-001",
            issued_at=now,
        )
        with pytest.raises(ConflictingAuthorityGrantError, match="already exists with conflicting data"):
            await gateway.save_authority_grant(conflict_grant)

        # 5. Replacement assignment gets distinct grant, both remain retrievable
        grant_id_2 = f"grant_{uuid.uuid4().hex[:8]}"
        grant_2 = DelegationGrant(
            grant_id=grant_id_2,
            root_task_id=t_id,
            issued_to_assignment_id="assign-002",
            issued_to_principal_id="principal-research-2",
            fencing_epoch=2,
            turn_id="turn-002",
            issued_at=now,
        )
        await gateway.save_authority_grant(grant_2)
        assert await gateway.get_authority_grant(t_id, grant_id) == grant
        assert await gateway.get_authority_grant(t_id, grant_id_2) == grant_2

        # 6. Corrupt record raises CorruptAuthoritativeRecordError
        await gateway._client.set(f"task.authority_grant:{t_id}:{grant_id}", "corrupt json string")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt DelegationGrant"):
            await gateway.get_authority_grant(t_id, grant_id)
    finally:
        await gateway.aclose()


async def test_redis_atomic_claim_acknowledgement_and_fencing_lifecycle(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    try:
        t_id = f"task_{uuid.uuid4().hex[:8]}"
        past = "2020-01-01T00:00:00Z"
        now = "2020-01-01T00:00:15Z"
        expired_now = "2020-01-01T00:00:35Z"
        future_expire = "2020-01-01T00:00:30Z"

        root = TaskRequested.create("Real Redis claim lifecycle task")
        object.__setattr__(root, "task_id", t_id)
        object.__setattr__(root, "root_task_id", t_id)
        await gateway.register_root_task(root, initial_status="pending_dispatch", required_capability="research")
        await gateway.update_status(t_id, "pending")

        grant = DelegationGrant(
            grant_id=f"grant_{uuid.uuid4().hex[:8]}",
            root_task_id=t_id,
            issued_to_assignment_id="assign-001",
            issued_to_principal_id="principal-research-1",
            fencing_epoch=1,
            turn_id="turn-001",
            issued_at=past,
        )

        claim_1 = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-001",
            claim_id="claim-001",
            root_task_id=t_id,
            turn_id="turn-001",
            runtime_principal_id="principal-research-1",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="a" * 64,
            request_id="req-001",
            claimed_at=past,
            delivery_expires_at=future_expire,
            delegation_grant_id=grant.grant_id,
        )

        # 1. Atomic claim succeeds
        active, is_replay = await gateway.atomic_claim_task(
            task_id=t_id,
            claim=claim_1,
            grant=grant,
            now=past,
        )
        assert is_replay is False
        assert active.fencing_epoch == 1
        assert active.state == "claimed_unacknowledged"
        assert await gateway.get_status(t_id) == "working"
        assert await gateway.get_fencing_epoch(t_id) == 1
        assert await gateway.get_authority_grant(t_id, grant.grant_id) == grant

        # 2. Idempotent claim replay
        active_replay, is_replay_flag = await gateway.atomic_claim_task(
            task_id=t_id,
            claim=claim_1,
            grant=grant,
            now=now,
        )
        assert is_replay_flag is True
        assert active_replay.fencing_epoch == 1
        assert active_replay.assignment_id == "assign-001"

        # 3. Conflicting claim by another principal fails while active
        other_claim = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-002",
            claim_id="claim-002",
            root_task_id=t_id,
            turn_id="turn-002",
            runtime_principal_id="principal-research-2",
            bound_agent_id="research-agent-2",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=2,
            proof_digest="b" * 64,
            request_id="req-002",
            claimed_at=now,
            delivery_expires_at=future_expire,
        )
        with pytest.raises(TaskAlreadyClaimedError, match="active claim by another runtime"):
            await gateway.atomic_claim_task(
                task_id=t_id,
                claim=other_claim,
                now=now,
            )

        # Non-claimable / missing task raises TaskNotClaimableError
        with pytest.raises(TaskNotClaimableError, match="not found"):
            await gateway.atomic_claim_task(
                task_id="missing-task-id",
                claim=claim_1,
                now=now,
            )

        # Acknowledgement fails if lease expired
        with pytest.raises(ClaimDeliveryLeaseExpiredError, match="Delivery lease"):
            await gateway.acknowledge_claim(t_id, "assign-001", "principal-research-1", expired_now)

        # 4. Delivery lease expiry restores claimability and clears claim
        expired = await gateway.expire_delivery_lease(t_id, now=expired_now)
        assert expired is True
        assert await gateway.get_status(t_id) == "pending"
        assert await gateway.get_active_claim(t_id) is None

        # 5. Subsequent claim increments fencing epoch to 2
        claim_2 = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-003",
            claim_id="claim-003",
            root_task_id=t_id,
            turn_id="turn-003",
            runtime_principal_id="principal-research-2",
            bound_agent_id="research-agent-2",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=2,
            proof_digest="c" * 64,
            request_id="req-003",
            claimed_at=expired_now,
            delivery_expires_at="2020-01-01T00:01:00Z",
        )
        active_2, is_replay_2 = await gateway.atomic_claim_task(
            task_id=t_id,
            claim=claim_2,
            now=expired_now,
        )
        assert is_replay_2 is False
        assert active_2.fencing_epoch == 2
        assert await gateway.get_fencing_epoch(t_id) == 2

        # 6. Acknowledge claim
        ack_time = "2020-01-01T00:00:40Z"
        acked = await gateway.acknowledge_claim(t_id, "assign-003", "principal-research-2", ack_time)
        assert acked.state == "acknowledged"
        assert acked.acknowledged_at == normalize_canonical_utc_timestamp(ack_time)

        # 7. Acknowledge replay is idempotent
        acked_replay = await gateway.acknowledge_claim(t_id, "assign-003", "principal-research-2", ack_time)
        assert acked_replay.state == "acknowledged"

        # 8. Mismatched assignment ID raises ClaimNotFoundError
        with pytest.raises(ClaimNotFoundError, match="does not match"):
            await gateway.acknowledge_claim(t_id, "wrong-assignment", "principal-research-2", ack_time)

        # 8b. Mismatched principal ID raises ClaimNotFoundError
        with pytest.raises(ClaimNotFoundError, match="does not match"):
            await gateway.acknowledge_claim(t_id, "assign-003", "wrong-principal", ack_time)
    finally:
        await gateway.aclose()


@pytest.mark.anyio
async def test_redis_concurrent_claim_attempts_atomicity(redis_url: str) -> None:
    gateway = RedisGateway(redis_url)
    try:
        now = datetime.now(UTC).isoformat()
        future = "2099-01-01T00:00:00Z"

        # Setup root task in pending status
        root = TaskRequested.create("Concurrent claim target")
        t_id = root.task_id
        await gateway.register_root_task(root, required_capability="research")
        await gateway.update_status(t_id, "pending")

        # 5 competing claims from different principals
        competing_claims = [
            ActiveClaim(
                task_id=t_id,
                assignment_id=f"assign-{i}",
                claim_id=f"claim-{i}",
                root_task_id=t_id,
                turn_id=f"turn-{i}",
                runtime_principal_id=f"principal-{i}",
                bound_agent_id=f"agent-{i}",
                assigned_capability="research",
                execution_kind="initial",
                fencing_epoch=1,
                proof_digest=f"{i}" * 64,
                request_id=f"req-{i}",
                claimed_at=now,
                delivery_expires_at=future,
            )
            for i in range(5)
        ]

        results = await asyncio.gather(
            *[
                gateway.atomic_claim_task(task_id=t_id, claim=claim, now=now)
                for claim in competing_claims
            ],
            return_exceptions=True,
        )

        successes = [r for r in results if isinstance(r, tuple)]
        failures = [r for r in results if isinstance(r, Exception)]

        # Exactly one concurrent claim succeeds; all others fail closed
        assert len(successes) == 1
        assert len(failures) == 4

        winner_claim, is_replay = successes[0]
        assert is_replay is False
        assert winner_claim.fencing_epoch == 1
        assert await gateway.get_fencing_epoch(t_id) == 1
        assert await gateway.get_status(t_id) == "working"

        for f in failures:
            assert isinstance(f, TaskAlreadyClaimedError)
    finally:
        await gateway.aclose()


@pytest.mark.anyio
async def test_redis_proof_digest_state_and_idempotent_replay(redis_url: str) -> None:
    gateway = RedisGateway(redis_url)
    try:
        now = datetime.now(UTC).isoformat()
        future = "2099-01-01T00:00:00Z"
        digest_hex = "f" * 64

        root = TaskRequested.create("Replay test")
        t_id = root.task_id
        await gateway.register_root_task(root, required_capability="research")
        await gateway.update_status(t_id, "pending")

        claim = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-orig",
            claim_id="claim-orig",
            root_task_id=t_id,
            turn_id="turn-orig",
            runtime_principal_id="principal-1",
            bound_agent_id="agent-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest=digest_hex,
            request_id="req-unique-123",
            claimed_at=now,
            delivery_expires_at=future,
        )

        # 1. Initial claim
        active, is_replay = await gateway.atomic_claim_task(t_id, claim, now=now)
        assert is_replay is False
        assert active.proof_digest == digest_hex

        # Verify Redis state contains exact 64-char hex digest
        persisted = await gateway.get_active_claim(t_id)
        assert persisted is not None
        assert persisted.proof_digest == digest_hex

        # 2. Idempotent replay does not mutate Redis or advance fence
        active_replay, is_replay_2 = await gateway.atomic_claim_task(t_id, claim, now=now)
        assert is_replay_2 is True
        assert active_replay.assignment_id == "assign-orig"
        assert active_replay.fencing_epoch == 1
        assert await gateway.get_fencing_epoch(t_id) == 1

        # 3. Acknowledge assignment
        await gateway.acknowledge_claim(t_id, "assign-orig", "principal-1", now)

        # 4. Claim attempt after acknowledgement fails closed with TaskAlreadyClaimedError
        with pytest.raises(TaskAlreadyClaimedError, match="already acknowledged"):
            await gateway.atomic_claim_task(t_id, claim, now=now)
    finally:
        await gateway.aclose()


@pytest.mark.anyio
async def test_redis_corrupt_claim_and_fence_state_fails_closed(redis_url: str) -> None:
    gateway = RedisGateway(redis_url)
    client = redis.from_url(redis_url, decode_responses=False)
    try:
        now = datetime.now(UTC).isoformat()
        future = "2099-01-01T00:00:00Z"

        root = TaskRequested.create("Corrupt test")
        t_id = root.task_id
        await gateway.register_root_task(root, required_capability="research")
        await gateway.update_status(t_id, "pending")

        # 1. Corrupt fence record (non-integer string)
        await client.set(f"task.fence:{t_id}", b"not-a-number")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt fencing epoch"):
            await gateway.get_fencing_epoch(t_id)

        claim = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-1",
            claim_id="claim-1",
            root_task_id=t_id,
            turn_id="turn-1",
            runtime_principal_id="p-1",
            bound_agent_id="a-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="a" * 64,
            request_id="req-1",
            claimed_at=now,
            delivery_expires_at=future,
        )
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_FENCE"):
            await gateway.atomic_claim_task(t_id, claim, now=now)

        # Restore valid fence
        await client.set(f"task.fence:{t_id}", b"0")

        # 2. Corrupt claim record (invalid JSON)
        await client.set(f"task.claim:{t_id}", b"this is {not valid json")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt ActiveClaim"):
            await gateway.get_active_claim(t_id)

        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_CLAIM"):
            await gateway.atomic_claim_task(t_id, claim, now=now)

        with pytest.raises(CorruptAuthoritativeRecordError, match=r"Corrupt ActiveClaim"):
            await gateway.acknowledge_claim(t_id, "assign-1", "p-1", now)
    finally:
        await client.aclose()
        await gateway.aclose()

@pytest.mark.anyio
async def test_redis_authoritative_records_have_no_ttl(redis_url: str) -> None:
    gateway = RedisGateway(redis_url)
    client = redis.from_url(redis_url, decode_responses=False)
    try:
        now = datetime.now(UTC).isoformat()
        future = "2099-01-01T00:00:00Z"

        root = TaskRequested.create("TTL persistence test")
        t_id = root.task_id
        grant = DelegationGrant(
            grant_id="grant-ttl-1",
            root_task_id=t_id,
            issued_to_assignment_id="assign-ttl-1",
            issued_to_principal_id="principal-1",
            fencing_epoch=1,
            turn_id="turn-ttl-1",
            issued_at=now,
        )
        claim = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-ttl-1",
            claim_id="claim-ttl-1",
            root_task_id=t_id,
            turn_id="turn-ttl-1",
            runtime_principal_id="principal-1",
            bound_agent_id="agent-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="d" * 64,
            request_id="req-ttl-1",
            claimed_at=now,
            delivery_expires_at=future,
            delegation_grant_id=grant.grant_id,
        )

        await gateway.register_root_task(root, required_capability="research")
        await gateway.update_status(t_id, "pending")
        await gateway.atomic_claim_task(t_id, claim, grant=grant, now=now)

        keys_to_check = [
            f"task.status:{t_id}",
            f"task.claim:{t_id}",
            f"task.fence:{t_id}",
            f"task.authority_grant:{t_id}:{grant.grant_id}",
            f"task.lineage:{t_id}",
            f"task.registration:{t_id}",
            f"task.required_capability:{t_id}",
        ]

        for k in keys_to_check:
            ttl = await client.ttl(k)
            # ttl == -1 indicates the key exists and has no expiration (persistent)
            assert ttl == -1, f"Expected key '{k}' to have no TTL (ttl == -1), got {ttl}"
    finally:
        await client.aclose()
        await gateway.aclose()

@pytest.mark.anyio
async def test_redis_hint_payload_secrecy(clean_redis: str) -> None:
    # Gateway configured in authenticated_claim mode
    gateway = RedisGateway(clean_redis, delivery_mode="authenticated_claim")
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        secret_prompt = "TOP_SECRET_PROMPT_DO_NOT_LEAK_IN_HINT_QUEUE"
        root = TaskRequested.create(secret_prompt)
        await gateway.register_root_task(root)

        legacy_target = "agent.tasks.research"
        authoritative_hint_queue = "agent.hints.research"

        # Dispatch via gateway under authenticated_claim mode targeting legacy queue
        await gateway.dispatch(root, legacy_target)

        # 1. Exactly one hint in the actual authoritative hint queue
        raw_hints = await client.lrange(authoritative_hint_queue, 0, -1)
        assert len(raw_hints) == 1

        payload_str = (
            raw_hints[0].decode("utf-8")
            if isinstance(raw_hints[0], (bytes, bytearray))
            else str(raw_hints[0])
        )

        # 2. Payload parses as WakeUpHint
        hint = WakeUpHint.from_json(payload_str)
        assert hint.capability == "research"
        assert hint.execution_kind == "initial"
        assert hint.task_id == root.task_id

        # 3. Secret prompt absent
        assert secret_prompt not in payload_str
        assert "prompt" not in payload_str

        # 4. No credential, token, proof, verifier digest, grant, or authority payload
        payload_data = json.loads(payload_str)
        disallowed_keys = {
            "token",
            "proof",
            "proof_digest",
            "verifier",
            "grant",
            "authority",
            "credential",
            "secret",
            "input",
            "artifact",
        }
        for key in payload_data:
            assert not any(
                disallowed in key.lower() for disallowed in disallowed_keys
            ), f"Disallowed key in hint: {key}"
        for val in payload_data.values():
            val_str = str(val).lower()
            assert not any(
                disallowed in val_str for disallowed in disallowed_keys
            ), f"Disallowed value in hint: {val}"

        # 5. No TaskRequested duplicate in a legacy task queue
        legacy_items = await client.lrange(legacy_target, 0, -1)
        assert len(legacy_items) == 0
    finally:
        await client.aclose()
        await gateway.aclose()

async def test_redis_atomic_claim_lua_two_phase_zero_mutation_on_conflict(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        t_id = f"root_{uuid.uuid4().hex[:8]}"
        past = "2020-01-01T00:00:00Z"
        now = "2020-01-01T00:00:15Z"
        future = "2099-01-01T00:00:00Z"

        root = TaskRequested.create("Root task for Lua two-phase tests")
        object.__setattr__(root, "task_id", t_id)
        object.__setattr__(root, "root_task_id", t_id)
        await gateway.register_root_task(root, required_capability="research")
        await gateway.update_status(t_id, "pending")

        grant_id_1 = f"grant_{uuid.uuid4().hex[:8]}"
        grant_1 = DelegationGrant(
            grant_id=grant_id_1,
            root_task_id=t_id,
            issued_to_assignment_id="assign-001",
            issued_to_principal_id="principal-research-1",
            fencing_epoch=1,
            turn_id="turn-001",
            issued_at=past,
        )
        claim_1 = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-001",
            claim_id="claim-001",
            root_task_id=t_id,
            turn_id="turn-001",
            runtime_principal_id="principal-research-1",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="a" * 64,
            request_id="req-001",
            claimed_at=past,
            delivery_expires_at=past,  # Expired lease
            delegation_grant_id=grant_1.grant_id,
        )

        # 1. Establish initial expired claim in Redis
        await gateway.atomic_claim_task(t_id, claim_1, grant=grant_1, now=past)
        await gateway.update_status(t_id, "pending")

        raw_claim_1_before = await client.get(f"task.claim:{t_id}")
        raw_fence_before = await client.get(f"task.fence:{t_id}")
        raw_status_before = await client.get(f"task.status:{t_id}")
        raw_grant_1_before = await client.get(f"task.authority_grant:{t_id}:{grant_id_1}")

        # 2. Expired claim + epoch race: leaves expired claim, fence, and status unchanged
        claim_bad_epoch = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-002",
            claim_id="claim-002",
            root_task_id=t_id,
            turn_id="turn-002",
            runtime_principal_id="principal-research-2",
            bound_agent_id="research-agent-2",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=99,  # Epoch race
            proof_digest="b" * 64,
            request_id="req-002",
            claimed_at=now,
            delivery_expires_at=future,
        )
        with pytest.raises(TaskAlreadyClaimedError, match="fencing epoch race"):
            await gateway.atomic_claim_task(t_id, claim_bad_epoch, now=now)

        assert await client.get(f"task.claim:{t_id}") == raw_claim_1_before
        assert await client.get(f"task.fence:{t_id}") == raw_fence_before
        assert await client.get(f"task.status:{t_id}") == raw_status_before

        # 3. Expired claim + corrupt fence: leaves expired claim and status unchanged
        await client.set(f"task.fence:{t_id}", "corrupt_non_numeric")
        claim_candidate = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-002",
            claim_id="claim-002",
            root_task_id=t_id,
            turn_id="turn-002",
            runtime_principal_id="principal-research-2",
            bound_agent_id="research-agent-2",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=2,
            proof_digest="c" * 64,
            request_id="req-002",
            claimed_at=now,
            delivery_expires_at=future,
        )
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_FENCE"):
            await gateway.atomic_claim_task(t_id, claim_candidate, now=now)

        assert await client.get(f"task.claim:{t_id}") == raw_claim_1_before
        assert await client.get(f"task.status:{t_id}") == raw_status_before

        # Restore valid fence
        await client.set(f"task.fence:{t_id}", "1")

        # 4. Conflicting grant: leaves fence, status, active claim, and existing grant byte-for-byte unchanged
        grant_conflict = DelegationGrant(
            grant_id=grant_id_1,  # Same grant_id
            root_task_id=t_id,
            issued_to_assignment_id="assign-002",  # Different assignment
            issued_to_principal_id="principal-research-2",
            fencing_epoch=2,
            turn_id="turn-002",
            issued_at=now,
        )
        with pytest.raises(ConflictingAuthorityGrantError, match="already exists with conflicting data"):
            await gateway.atomic_claim_task(t_id, claim_candidate, grant=grant_conflict, now=now)

        assert await client.get(f"task.fence:{t_id}") == "1"
        assert await client.get(f"task.status:{t_id}") == raw_status_before
        assert await client.get(f"task.claim:{t_id}") == raw_claim_1_before
        assert await client.get(f"task.authority_grant:{t_id}:{grant_id_1}") == raw_grant_1_before

        # 5. Inconsistent grant key/payload fails without mutation
        assert raw_status_before is not None
        res_corrupt_grant = await client.eval(
            CLAIM_TASK_LUA,
            7,
            f"task.status:{t_id}",
            f"task.claim:{t_id}",
            f"task.fence:{t_id}",
            f"task.authority_grant:{t_id}:some_grant",  # grant key provided
            f"task.required_capability:{t_id}",
            f"task.continuation_latest:{t_id}",
            f"task.authority:{t_id}",
            t_id,
            now,
            "principal-research-2",
            "req-002",
            raw_status_before,
            claim_candidate.to_json(),
            "",  # grant payload empty!
            "2",
            "research",
            "initial",
            "1",
            "0",
        )
        assert res_corrupt_grant == ["CORRUPT_GRANT_INPUT"]
        assert await client.get(f"task.fence:{t_id}") == "1"
        assert await client.get(f"task.status:{t_id}") == raw_status_before
        assert await client.get(f"task.claim:{t_id}") == raw_claim_1_before

        # 6. Malformed claim JSON in expiry raises CorruptAuthoritativeRecordError
        await client.set(f"task.claim:{t_id}", "not-valid-json")
        with pytest.raises(CorruptAuthoritativeRecordError, match="during lease expiry"):
            await gateway.expire_delivery_lease(t_id, now=now)

        # 7. Malformed delivery_expires_at type in expiry raises CorruptAuthoritativeRecordError
        bad_claim_dict = claim_1.to_dict()
        bad_claim_dict["delivery_expires_at"] = 12345
        await client.set(f"task.claim:{t_id}", json.dumps(bad_claim_dict))
        with pytest.raises(CorruptAuthoritativeRecordError, match="during lease expiry"):
            await gateway.expire_delivery_lease(t_id, now=now)

        # 8. Unknown active-claim state in expiry raises CorruptAuthoritativeRecordError
        bad_claim_dict["delivery_expires_at"] = past
        bad_claim_dict["state"] = "unknown_state_xyz"
        await client.set(f"task.claim:{t_id}", json.dumps(bad_claim_dict))
        with pytest.raises(CorruptAuthoritativeRecordError, match="during lease expiry"):
            await gateway.expire_delivery_lease(t_id, now=now)

        # 9. Unknown active-claim state in claim raises CorruptAuthoritativeRecordError
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_CLAIM"):
            await gateway.atomic_claim_task(t_id, claim_candidate, now=now)

        # 10. Unknown active-claim state in acknowledge raises CorruptAuthoritativeRecordError
        with pytest.raises(CorruptAuthoritativeRecordError, match=r"Corrupt ActiveClaim"):
            await gateway.acknowledge_claim(t_id, "assign-001", "principal-research-1", now)

        # 11. Successful replacement of an expired claim advances the fence exactly once,
        # stores the replacement claim, changes status to working, and preserves historical grant
        assert raw_claim_1_before is not None
        await client.set(f"task.claim:{t_id}", raw_claim_1_before)  # Restore claim 1
        grant_id_2 = f"grant_{uuid.uuid4().hex[:8]}"
        grant_2 = DelegationGrant(
            grant_id=grant_id_2,
            root_task_id=t_id,
            issued_to_assignment_id="assign-002",
            issued_to_principal_id="principal-research-2",
            fencing_epoch=2,
            turn_id="turn-002",
            issued_at=now,
        )
        claim_2 = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-002",
            claim_id="claim-002",
            root_task_id=t_id,
            turn_id="turn-002",
            runtime_principal_id="principal-research-2",
            bound_agent_id="research-agent-2",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=2,
            proof_digest="d" * 64,
            request_id="req-002",
            claimed_at=now,
            delivery_expires_at=future,
            delegation_grant_id=grant_2.grant_id,
        )
        res_claim, is_replay = await gateway.atomic_claim_task(t_id, claim_2, grant=grant_2, now=now)
        assert is_replay is False
        assert res_claim.fencing_epoch == 2
        assert await client.get(f"task.fence:{t_id}") == "2"
        raw_stat_after = await client.get(f"task.status:{t_id}")
        assert raw_stat_after is not None
        status_after = json.loads(raw_stat_after)
        assert status_after["status"] == "working"
        raw_claim_after = await client.get(f"task.claim:{t_id}")
        assert isinstance(raw_claim_after, str)
        claim_after = ActiveClaim.from_json(raw_claim_after)
        assert claim_after.assignment_id == "assign-002"
        assert claim_after.fencing_epoch == 2
        # Historical grant preserved intact!
        assert await client.get(f"task.authority_grant:{t_id}:{grant_id_1}") == raw_grant_1_before
        # New grant written intact!
        assert await client.get(f"task.authority_grant:{t_id}:{grant_id_2}") == grant_2.to_json()
    finally:
        await client.aclose()
        await gateway.aclose()

@pytest.mark.anyio
async def test_redis_capability_and_lifecycle_atomic_validation(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        t_id = f"task_{uuid.uuid4().hex[:8]}"
        now = datetime.now(UTC).isoformat()
        future = "2099-01-01T00:00:00Z"
        past = "2020-01-01T00:00:00Z"

        # 1. Register root task with required_capability='general'
        root = TaskRequested.create("Root general task")
        object.__setattr__(root, "task_id", t_id)
        object.__setattr__(root, "root_task_id", t_id)
        await gateway.register_root_task(root, initial_status="pending_dispatch", required_capability="general")
        await gateway.update_status(t_id, "pending")

        # Verify task.required_capability:{t_id} exists
        cap_rec = await gateway.get_required_capability(t_id)
        assert cap_rec is not None
        assert cap_rec.required_capability == "general"

        raw_status_before = await client.get(f"task.status:{t_id}")
        raw_fence_before = await client.get(f"task.fence:{t_id}")

        # 2. Attempt to claim with assigned_capability='research' -> CapabilityMismatchError, zero mutations
        claim_mismatch = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-mismatch",
            claim_id="claim-mismatch",
            root_task_id=t_id,
            turn_id="turn-mismatch",
            runtime_principal_id="principal-research",
            bound_agent_id="research-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="a" * 64,
            request_id="req-mismatch",
            claimed_at=now,
            delivery_expires_at=future,
        )
        with pytest.raises(CapabilityMismatchError, match="requires capability 'general'"):
            await gateway.atomic_claim_task(t_id, claim_mismatch, now=now)

        assert await client.get(f"task.status:{t_id}") == raw_status_before
        assert await client.get(f"task.fence:{t_id}") == raw_fence_before
        assert await client.exists(f"task.claim:{t_id}") == 0

        # 3. Attempt resumed_continuation when task status is 'pending' -> TaskNotClaimableError, zero mutations
        claim_resumed_on_pending = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-resumed",
            claim_id="claim-resumed",
            root_task_id=t_id,
            turn_id="turn-resumed",
            runtime_principal_id="principal-general",
            bound_agent_id="general-1",
            assigned_capability="general",
            execution_kind="resumed_continuation",
            continuation_version=1,
            fencing_epoch=1,
            proof_digest="b" * 64,
            request_id="req-resumed",
            claimed_at=now,
            delivery_expires_at=future,
            prior_claimable_status="resumable",
        )
        with pytest.raises(TaskNotClaimableError, match="incompatible with execution_kind"):
            await gateway.atomic_claim_task(
                t_id,
                claim_resumed_on_pending,
                now=now,
                expected_continuation_version=1,
                is_root=True,
            )

        assert await client.get(f"task.status:{t_id}") == raw_status_before
        assert await client.get(f"task.fence:{t_id}") == raw_fence_before
        assert await client.exists(f"task.claim:{t_id}") == 0

        # 4. EXPIRE_DELIVERY_LEASE_LUA restores 'resumable' when prior_claimable_status == 'resumable'
        claim_with_resumable_prior = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-prior-resumable",
            claim_id="claim-prior-resumable",
            root_task_id=t_id,
            turn_id="turn-prior-resumable",
            runtime_principal_id="principal-general",
            bound_agent_id="general-1",
            assigned_capability="general",
            execution_kind="resumed_continuation",
            continuation_version=1,
            fencing_epoch=1,
            proof_digest="c" * 64,
            request_id="req-prior-resumable",
            claimed_at=past,
            delivery_expires_at=past,  # Expired lease
            prior_claimable_status="resumable",
        )
        await client.set(f"task.claim:{t_id}", claim_with_resumable_prior.to_json())
        await client.set(f"task.status:{t_id}", json.dumps({"status": "working", "task_id": t_id}))

        expired_restored = await gateway.expire_delivery_lease(t_id, now=now)
        assert expired_restored is True
        assert await client.exists(f"task.claim:{t_id}") == 0
        raw_stat_exp = await client.get(f"task.status:{t_id}")
        assert raw_stat_exp is not None
        status_after_expire = json.loads(raw_stat_exp)
        assert status_after_expire["status"] == "resumable"
    finally:
        await client.aclose()
        await gateway.aclose()

@pytest.mark.anyio
async def test_redis_stage_artifact_atomic_claim_validation(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        t_id = f"task_{uuid.uuid4().hex[:8]}"
        now = datetime.now(UTC).isoformat()
        future = "2099-01-01T00:00:00Z"

        root = TaskRequested.create("Research root for artifact staging")
        object.__setattr__(root, "task_id", t_id)
        object.__setattr__(root, "root_task_id", t_id)
        await gateway.register_root_task(root, initial_status="pending_dispatch", required_capability="research")
        await gateway.update_status(t_id, "pending")

        claim = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-stage-1",
            claim_id="claim-stage-1",
            root_task_id=t_id,
            turn_id="turn-stage-1",
            runtime_principal_id="principal-research",
            bound_agent_id="research-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="a" * 64,
            request_id="req-stage-1",
            claimed_at=now,
            delivery_expires_at=future,
            prior_claimable_status="pending",
        )
        await gateway.atomic_claim_task(t_id, claim, now=now)

        artifact = StageArtifact(
            artifact_id="art-test-1",
            producing_task_id=t_id,
            artifact_type="research_material",
            content={"material": "Preliminary verified research"},
            created_at=now,
            producing_turn_id="turn-stage-1",
            artifact_version=1,
            sealed=True,
        )

        # 1. Attempt to stage artifact while claim is unacknowledged -> TaskNotClaimableError
        with pytest.raises(TaskNotClaimableError, match="acknowledged"):
            await gateway.save_stage_artifact(
                artifact,
                expected_assignment_id="assign-stage-1",
                expected_principal_id="principal-research",
                expected_fencing_epoch=1,
            )

        # 2. Acknowledge the claim
        await gateway.acknowledge_claim(t_id, "assign-stage-1", "principal-research", now)

        # 3. Attempt with wrong fencing epoch -> FencingEpochRaceError
        with pytest.raises(FencingEpochRaceError, match="Fencing epoch mismatch"):
            await gateway.save_stage_artifact(
                artifact,
                expected_assignment_id="assign-stage-1",
                expected_principal_id="principal-research",
                expected_fencing_epoch=99,
            )

        # 4. Successful stage artifact commit
        await gateway.save_stage_artifact(
            artifact,
            expected_assignment_id="assign-stage-1",
            expected_principal_id="principal-research",
            expected_fencing_epoch=1,
        )

        # 5. Verify persisted artifact
        saved = await gateway.get_stage_artifact(t_id, "art-test-1", 1)
        assert saved is not None
        assert saved.artifact_id == "art-test-1"
        assert saved.content == {"material": "Preliminary verified research"}
    finally:
        await client.aclose()
        await gateway.aclose()

@pytest.mark.anyio
async def test_redis_continuation_version_numeric_hardening_and_zero_mutation(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        t_id = f"cont_root_{uuid.uuid4().hex[:8]}"
        now = "2026-11-17T12:00:00Z"
        future = "2099-01-01T00:00:00Z"

        root = TaskRequested.create("Root for real-Redis continuation hardening")
        object.__setattr__(root, "task_id", t_id)
        object.__setattr__(root, "root_task_id", t_id)
        await gateway.register_root_task(root, initial_status="pending_dispatch", required_capability="general")
        await gateway.update_status(t_id, "resumable")

        claim_candidate = ActiveClaim(
            task_id=t_id,
            assignment_id="assign-resumed-001",
            claim_id="claim-resumed-001",
            root_task_id=t_id,
            turn_id="turn-resumed-001",
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
            assigned_capability="general",
            execution_kind="resumed_continuation",
            fencing_epoch=1,
            proof_digest="a" * 64,
            request_id="req-resumed-001",
            claimed_at=now,
            delivery_expires_at=future,
            prior_claimable_status="resumable",
        )

        raw_status_before = await client.get(f"task.status:{t_id}")
        assert raw_status_before is not None

        # 1. Fractional expected continuation version rejected at Lua boundary, zero mutations
        keys = [
            f"task.status:{t_id}",
            f"task.claim:{t_id}",
            f"task.fence:{t_id}",
            "",
            f"task.required_capability:{t_id}",
            f"task.continuation_latest:{t_id}",
            f"task.authority:{t_id}",
        ]
        res_fractional = await client.eval(
            CLAIM_TASK_LUA,
            7,
            *keys,
            t_id,
            now,
            "principal-worker-1",
            "req-resumed-001",
            raw_status_before,
            claim_candidate.to_json(),
            "",
            "1",
            "general",
            "resumed_continuation",
            "1",
            "1.5",
        )
        assert res_fractional == ["INVALID_CONTINUATION_VERSION"]
        assert await client.get(f"task.status:{t_id}") == raw_status_before
        assert await client.exists(f"task.fence:{t_id}") == 0
        assert await client.exists(f"task.claim:{t_id}") == 0

        # 2. Zero and negative expected continuation versions rejected with ValueError, zero mutations
        for bad_ver in (0, -1, -5):
            with pytest.raises(ValueError, match="expected_continuation_version must be a positive integer"):
                await gateway.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=bad_ver, now=now)

            assert await client.get(f"task.status:{t_id}") == raw_status_before
            assert await client.exists(f"task.fence:{t_id}") == 0
            assert await client.exists(f"task.claim:{t_id}") == 0

        # 3. Nonnumeric authoritative latest-pointer fails closed, zero mutations
        await client.set(f"task.continuation_latest:{t_id}", "corrupt_non_numeric")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt continuation latest record"):
            await gateway.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=1, now=now)

        assert await client.get(f"task.status:{t_id}") == raw_status_before
        assert await client.exists(f"task.fence:{t_id}") == 0
        assert await client.exists(f"task.claim:{t_id}") == 0
        assert await client.get(f"task.continuation_latest:{t_id}") == "corrupt_non_numeric"

        # 4. Fractional authoritative latest-pointer fails closed, zero mutations
        await client.set(f"task.continuation_latest:{t_id}", "1.5")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt continuation latest record"):
            await gateway.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=1, now=now)

        assert await client.get(f"task.status:{t_id}") == raw_status_before
        assert await client.exists(f"task.fence:{t_id}") == 0
        assert await client.exists(f"task.claim:{t_id}") == 0

        # 5. Zero and negative authoritative latest-pointer fail closed, zero mutations
        for bad_ptr in ("0", "-1", "-5"):
            await client.set(f"task.continuation_latest:{t_id}", bad_ptr)
            with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt continuation latest record"):
                await gateway.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=1, now=now)

            assert await client.get(f"task.status:{t_id}") == raw_status_before
            assert await client.exists(f"task.fence:{t_id}") == 0
            assert await client.exists(f"task.claim:{t_id}") == 0

        # 6. Mismatch returns continuation-version conflict and causes zero mutations
        await client.set(f"task.continuation_latest:{t_id}", "1")
        with pytest.raises(TaskNotClaimableError, match="Continuation version mismatch"):
            await gateway.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=2, now=now)

        assert await client.get(f"task.status:{t_id}") == raw_status_before
        assert await client.exists(f"task.fence:{t_id}") == 0
        assert await client.exists(f"task.claim:{t_id}") == 0

        # 7. Valid positive integer matching version succeeds and commits state
        res_claim, is_replay = await gateway.atomic_claim_task(t_id, claim_candidate, expected_continuation_version=1, now=now)
        assert not is_replay
        assert res_claim.fencing_epoch == 1
        assert await client.get(f"task.fence:{t_id}") == "1"
        raw_status_after = await client.get(f"task.status:{t_id}")
        assert raw_status_after is not None
        assert json.loads(raw_status_after)["status"] == "working"
        assert await client.exists(f"task.claim:{t_id}") == 1
    finally:
        await client.aclose()
        await gateway.aclose()

@pytest.mark.anyio
async def test_redis_child_registration_atomic_boundary_and_immutability(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        root_id = f"root_{uuid.uuid4().hex[:8]}"
        child_id = f"child_{uuid.uuid4().hex[:8]}"
        now = "2026-11-17T12:00:00Z"

        root = TaskRequested.create("Root task")
        object.__setattr__(root, "task_id", root_id)
        object.__setattr__(root, "root_task_id", root_id)
        await gateway.register_root_task(root)

        child = TaskRequested(
            task_id=child_id,
            prompt="Child prompt",
            created_at=now,
            root_task_id=root_id,
            parent_task_id=root_id,
            version=2,
        )

        # 1. First registration creates all records atomically
        await gateway.register_child_task(child, initial_status="pending_dispatch", required_capability="analysis")
        assert await client.get(f"task.status:{child_id}") == json.dumps({"status": "pending_dispatch", "task_id": child_id}, separators=(",", ":"))
        lineage_raw = await client.get(f"task.lineage:{child_id}")
        assert lineage_raw is not None
        assert json.loads(lineage_raw) == {"parent_task_id": root_id, "root_task_id": root_id, "task_id": child_id}
        cap_rec = await gateway.get_required_capability(child_id)
        assert cap_rec is not None
        assert cap_rec.required_capability == "analysis"
        assert await client.sismember(f"task.children:{root_id}", child_id) == 1

        # Capability key has no TTL (ttl == -1)
        ttl = await client.ttl(f"task.required_capability:{child_id}")
        assert ttl == -1

        raw_status_before = await client.get(f"task.status:{child_id}")
        raw_lineage_before = await client.get(f"task.lineage:{child_id}")
        raw_cap_before = await client.get(f"task.required_capability:{child_id}")

        # 2. Byte-identical replay succeeds idempotently
        await gateway.register_child_task(child, initial_status="pending_dispatch", required_capability="analysis")
        assert await client.get(f"task.status:{child_id}") == raw_status_before
        assert await client.get(f"task.lineage:{child_id}") == raw_lineage_before
        assert await client.get(f"task.required_capability:{child_id}") == raw_cap_before

        # 3. Conflicting capability replay fails closed with ConflictingChildTaskRegistrationError, zero mutation
        with pytest.raises(ConflictingChildTaskRegistrationError, match="already exists with conflicting state"):
            await gateway.register_child_task(child, initial_status="pending_dispatch", required_capability="research")
        assert await client.get(f"task.status:{child_id}") == raw_status_before
        assert await client.get(f"task.lineage:{child_id}") == raw_lineage_before
        assert await client.get(f"task.required_capability:{child_id}") == raw_cap_before

        # 4. Conflicting lineage replay fails closed with ConflictingChildTaskRegistrationError, zero mutation
        conflicting_child = TaskRequested(
            task_id=child_id,
            prompt="Child prompt",
            created_at=now,
            root_task_id="other-root",
            parent_task_id=root_id,
            version=2,
        )
        with pytest.raises(ConflictingChildTaskRegistrationError, match="already exists with conflicting state"):
            await gateway.register_child_task(conflicting_child, initial_status="pending_dispatch", required_capability="analysis")
        assert await client.get(f"task.status:{child_id}") == raw_status_before
        assert await client.get(f"task.lineage:{child_id}") == raw_lineage_before
        assert await client.get(f"task.required_capability:{child_id}") == raw_cap_before

        # 5. Partial prior state fails closed with ConflictingChildTaskRegistrationError, zero mutation
        partial_child_id = f"child_part_{uuid.uuid4().hex[:8]}"
        await client.set(f"task.status:{partial_child_id}", "pending_dispatch")
        partial_child = TaskRequested(
            task_id=partial_child_id,
            prompt="Partial child",
            created_at=now,
            root_task_id=root_id,
            parent_task_id=root_id,
            version=2,
        )
        with pytest.raises(ConflictingChildTaskRegistrationError, match="inconsistent or partial authoritative state"):
            await gateway.register_child_task(partial_child, initial_status="pending_dispatch", required_capability="analysis")
        assert await client.exists(f"task.lineage:{partial_child_id}") == 0
        assert await client.exists(f"task.required_capability:{partial_child_id}") == 0
        assert await client.sismember(f"task.children:{root_id}", partial_child_id) == 0
    finally:
        await client.aclose()
        await gateway.aclose()

@pytest.mark.anyio
async def test_redis_authenticated_claim_hint_routing_and_legacy_isolation(clean_redis: str) -> None:
    gateway_claim = RedisGateway(clean_redis, delivery_mode="authenticated_claim")
    gateway_legacy = RedisGateway(clean_redis, delivery_mode="legacy_push")
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        t_id_1 = f"task_{uuid.uuid4().hex[:8]}"
        task_1 = TaskRequested.create("Hint routing task 1")
        object.__setattr__(task_1, "task_id", t_id_1)
        object.__setattr__(task_1, "root_task_id", t_id_1)

        # 1. Explicit research capability plus a custom target routes to agent.hints.research
        await gateway_claim.dispatch(task_1, target="custom.legacy.target", capability="research")
        hints_research = await client.lrange("agent.hints.research", 0, -1)
        assert len(hints_research) == 1
        raw_hint_1 = hints_research[0]
        assert isinstance(raw_hint_1, str)
        hint_1 = WakeUpHint.from_json(raw_hint_1)
        assert hint_1.task_id == t_id_1
        assert hint_1.capability == "research"
        # No hint in custom target, and no legacy task payload in custom target
        assert await client.exists("custom.legacy.target") == 0

        # 2. Absent capability plus an exact canonical mapped target succeeds
        t_id_2 = f"task_{uuid.uuid4().hex[:8]}"
        task_2 = TaskRequested.create("Hint routing task 2")
        object.__setattr__(task_2, "task_id", t_id_2)
        object.__setattr__(task_2, "root_task_id", t_id_2)
        await gateway_claim.dispatch(task_2, target="agent.tasks.general")
        hints_general = await client.lrange("agent.hints.general", 0, -1)
        assert len(hints_general) == 1
        raw_hint_2 = hints_general[0]
        assert isinstance(raw_hint_2, str)
        hint_2 = WakeUpHint.from_json(raw_hint_2)
        assert hint_2.task_id == t_id_2
        assert hint_2.capability == "general"
        # No legacy TaskRequested payload published to agent.tasks.general
        assert await client.exists("agent.tasks.general") == 0

        # 3. Absent capability plus an arbitrary custom target fails closed
        t_id_3 = f"task_{uuid.uuid4().hex[:8]}"
        task_3 = TaskRequested.create("Hint routing task 3")
        object.__setattr__(task_3, "task_id", t_id_3)
        object.__setattr__(task_3, "root_task_id", t_id_3)
        with pytest.raises(ValueError, match="Cannot derive recognized canonical capability"):
            await gateway_claim.dispatch(task_3, target="custom.arbitrary.queue")
        assert await client.exists(f"task.status:{t_id_3}") == 0
        assert await client.exists("custom.arbitrary.queue") == 0

        # 4. Unrecognized explicit capability fails closed
        t_id_4 = f"task_{uuid.uuid4().hex[:8]}"
        task_4 = TaskRequested.create("Hint routing task 4")
        object.__setattr__(task_4, "task_id", t_id_4)
        object.__setattr__(task_4, "root_task_id", t_id_4)
        with pytest.raises(ValueError, match="Unrecognized canonical capability"):
            await gateway_claim.dispatch(task_4, target="agent.tasks.general", capability="unrecognized_specialty")
        assert await client.exists(f"task.status:{t_id_4}") == 0

        # 5. Legacy push mode publishes TaskRequested payload to target queue unchanged
        t_id_5 = f"task_{uuid.uuid4().hex[:8]}"
        task_5 = TaskRequested.create("Legacy push task")
        object.__setattr__(task_5, "task_id", t_id_5)
        object.__setattr__(task_5, "root_task_id", t_id_5)
        await gateway_legacy.dispatch(task_5, target="agent.tasks.general")
        legacy_items = await client.lrange("agent.tasks.general", 0, -1)
        assert len(legacy_items) == 1
        raw_pushed = legacy_items[0]
        assert isinstance(raw_pushed, str)
        pushed_task = TaskRequested.from_json(raw_pushed)
        assert pushed_task.task_id == t_id_5
    finally:
        await client.aclose()
        await gateway_claim.aclose()
        await gateway_legacy.aclose()


@pytest.mark.anyio
async def test_redis_registration_capability_validation_and_zero_mutation(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis)
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        now = datetime.now(UTC).isoformat()
        root_id = f"root_{uuid.uuid4().hex[:8]}"
        root_task = TaskRequested(
            task_id=root_id,
            prompt="Root prompt",
            created_at=now,
            root_task_id=root_id,
            parent_task_id=None,
            version=2,
        )

        child_id = f"child_{uuid.uuid4().hex[:8]}"
        child_task = TaskRequested(
            task_id=child_id,
            prompt="Child prompt",
            created_at=now,
            root_task_id=root_id,
            parent_task_id=root_id,
            version=2,
        )

        # 1. Explicit empty root capability raises ValueError before any Redis mutation
        with pytest.raises(ValueError, match="required_capability must be a non-empty string"):
            await gateway.register_root_task(root_task, required_capability="")
        assert await client.exists(f"task.registration:{root_id}") == 0
        assert await client.exists(f"task.status:{root_id}") == 0
        assert await client.exists(f"task.lineage:{root_id}") == 0
        assert await client.exists(f"task.input:{root_id}:1") == 0
        assert await client.exists(f"task.authority:{root_id}") == 0
        assert await client.exists(f"task.required_capability:{root_id}") == 0

        # 2. Whitespace-only root capability raises ValueError
        with pytest.raises(ValueError, match="required_capability must be a non-empty string"):
            await gateway.register_root_task(root_task, required_capability="   ")
        assert await client.exists(f"task.registration:{root_id}") == 0

        # 3. Unknown root capability raises ValueError
        with pytest.raises(ValueError, match="Unrecognized required_capability 'invalid_cap'"):
            await gateway.register_root_task(root_task, required_capability="invalid_cap")
        assert await client.exists(f"task.registration:{root_id}") == 0

        # 4. Explicit empty child capability raises ValueError before any Redis mutation
        with pytest.raises(ValueError, match="required_capability must be a non-empty string"):
            await gateway.register_child_task(child_task, required_capability="")
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.exists(f"task.lineage:{child_id}") == 0
        assert await client.exists(f"task.required_capability:{child_id}") == 0
        assert await client.sismember(f"task.children:{root_id}", child_id) == 0

        # 5. Whitespace-only child capability raises ValueError
        with pytest.raises(ValueError, match="required_capability must be a non-empty string"):
            await gateway.register_child_task(child_task, required_capability="   ")
        assert await client.exists(f"task.status:{child_id}") == 0

        # 6. Unknown child capability raises ValueError
        with pytest.raises(ValueError, match="Unrecognized required_capability 'fast'"):
            await gateway.register_child_task(child_task, required_capability="fast")
        assert await client.exists(f"task.status:{child_id}") == 0

        # 7. Valid default registrations succeed unchanged
        await gateway.register_root_task(root_task)
        root_cap_rec = await gateway.get_required_capability(root_id)
        assert root_cap_rec is not None
        assert root_cap_rec.required_capability == "general"

        await gateway.register_child_task(child_task)
        child_cap_rec = await gateway.get_required_capability(child_id)
        assert child_cap_rec is not None
        assert child_cap_rec.required_capability == "analysis"

        # 8. Valid explicit registrations succeed
        root_res_id = f"root_res_{uuid.uuid4().hex[:8]}"
        root_res = TaskRequested(
            task_id=root_res_id,
            prompt="Root prompt res",
            created_at=now,
            root_task_id=root_res_id,
            parent_task_id=None,
            version=2,
        )
        await gateway.register_root_task(root_res, required_capability="research")
        root_res_cap = await gateway.get_required_capability(root_res_id)
        assert root_res_cap is not None
        assert root_res_cap.required_capability == "research"

        child_res_id = f"child_res_{uuid.uuid4().hex[:8]}"
        child_res = TaskRequested(
            task_id=child_res_id,
            prompt="Child prompt res",
            created_at=now,
            root_task_id=root_id,
            parent_task_id=root_id,
            version=2,
        )
        await gateway.register_child_task(child_res, required_capability="research")
        child_res_cap = await gateway.get_required_capability(child_res_id)
        assert child_res_cap is not None
        assert child_res_cap.required_capability == "research"
    finally:
        await client.aclose()
        await gateway.aclose()
