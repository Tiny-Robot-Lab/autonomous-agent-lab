import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
import pytest
import redis.asyncio as redis
import uvicorn
from agent_contracts import (
    ActiveClaim,
    AgentCapabilityMetadata,
    AgentEvent,
    ArtifactReference,
    AssignmentYieldRecord,
    DelegationGrant,
    DelegationProposalRecord,
    InferenceExecutionTelemetry,
    RootAuthorityRecord,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    TaskResult,
    WakeUpHint,
    compute_proposal_fingerprint,
    normalize_canonical_utc_timestamp,
)
from agent_infrastructure import OrchestratorClient
from analysis_agent.application.processor import AnalysisProcessor
from analysis_agent.infrastructure.deterministic_analyzer import (
    DeterministicMaterialAnalyzer,
)
from analysis_agent.infrastructure.redis_bus import RedisBus as AnalysisRedisBus
from orchestrator.application.ports import (
    CapabilityMismatchError,
    ChildClaimDeadlineExpiredError,
    ClaimDeliveryLeaseExpiredError,
    ClaimNotFoundError,
    DelegationConflictError,
    FencingEpochRaceError,
    GrantAlreadyConsumedError,
    TaskAlreadyClaimedError,
    TaskNotClaimableError,
    WaveBudgetExceededError,
)
from orchestrator.config import Settings
from orchestrator.domain.principal import compute_token_digest
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
from orchestrator.presentation.api import create_app
from research_agent.domain.evidence import AuthorityTier, EvaluatedEvidence, SourceType
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


RUNTIME_CREDENTIAL_PEPPER = "test_redis_runtime_pepper_secret_1234567890"
REDIS_PROOF_PEPPER = "test_redis_proof_pepper_4567890123456789"
RESEARCH_RUNTIME_TOKEN = "secret-research-token-redis-1"
ANALYSIS_RUNTIME_TOKEN = "secret-analysis-token-redis-1"

ANALYSIS_MATERIAL = "The rover maps caves. The rover records samples. Does the map show water?"
EXPECTED_ANALYSIS_REPORT = (
    "Analysis report\n"
    "Paragraphs: 1\n"
    "Sentences: 3\n"
    "Words: 13\n"
    "Repeated terms: rover (2)\n"
    "Question sentences:\n"
    "- Does the map show water?\n"
    "Scope: Only supplied material was analyzed; no retrieval was performed."
)


def _redis_principals_config() -> dict[str, dict[str, object]]:
    return {
        "principal-research-1": {
            "principal_id": "principal-research-1",
            "bound_agent_id": "research-agent-1",
            "authorized_capabilities": ["research"],
            "token_digest": compute_token_digest(RESEARCH_RUNTIME_TOKEN, RUNTIME_CREDENTIAL_PEPPER),
            "active": True,
        },
        "principal-analysis-1": {
            "principal_id": "principal-analysis-1",
            "bound_agent_id": "analysis-agent-1",
            "authorized_capabilities": ["analysis"],
            "token_digest": compute_token_digest(ANALYSIS_RUNTIME_TOKEN, RUNTIME_CREDENTIAL_PEPPER),
            "active": True,
        },
    }


async def _serve_orchestrator(app: object) -> tuple[str, uvicorn.Server, asyncio.Task[None]]:
    """Serve the production Orchestrator app over loopback HTTP for the real client."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    async with asyncio.timeout(15):
        while not server.started:
            await asyncio.sleep(0.01)
    sockets = server.servers[0].sockets if server.servers else []
    assert sockets, "test Orchestrator server failed to bind"
    port = sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}", server, serve_task


async def test_analysis_agent_dispatch_lifecycle_and_result_round_trip(
    clean_redis: str,
) -> None:
    """Complete authenticated Analysis runtime lifecycle on real Redis.

    Exercises the production path end to end: authoritative record setup,
    research pull-claim/acknowledge, artifact staging, delegation acceptance,
    legacy-push child delivery, Analysis pull-claim/acknowledge, processor
    execution, and fenced completion through the live HTTP Orchestrator
    boundary. Asserts the canonical TerminalOutcome, TaskResult, terminal
    status, assignment-completion audit, active-claim retirement, exactly-once
    parent projection with a single Research hint, and that agent events stay
    observational.
    """
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
    parent_id = f"task-redis-parent-{uuid.uuid4().hex[:8]}"

    settings = Settings(
        redis_url=clean_redis,
        app_env="test",
        task_delivery_mode="authenticated_claim",
        claim_delivery_lease_seconds=30.0,
        child_claim_deadline_seconds=120.0,
        runtime_credential_pepper=RUNTIME_CREDENTIAL_PEPPER,
        assignment_proof_pepper=REDIS_PROOF_PEPPER,
        runtime_principals=_redis_principals_config(),
    )
    app = create_app(settings, gateway=gateway, task_store=gateway, dispatcher=gateway)
    server: uvicorn.Server | None = None
    serve_task: asyncio.Task[None] | None = None
    base_url, server, serve_task = await _serve_orchestrator(app)
    research_client = OrchestratorClient(base_url, RESEARCH_RUNTIME_TOKEN)
    analysis_client = OrchestratorClient(base_url, ANALYSIS_RUNTIME_TOKEN)
    processor = AnalysisProcessor(
        agent_id,
        bus,
        DeterministicMaterialAnalyzer(),
        completion_client=analysis_client,
    )

    try:
        # 1. Authoritative parent records required for claim.
        now = datetime.now(UTC).isoformat()
        parent_task = TaskRequested(
            task_id=parent_id,
            root_task_id=parent_id,
            parent_task_id=None,
            prompt="Redis research question",
            created_at=now,
            version=2,
        )
        await gateway.register_root_task(parent_task, required_capability="research")
        await gateway.update_status(parent_id, "pending")

        # 2-3. Research pull-claim and acknowledge through the Orchestrator boundary.
        parent_assignment, parent_proof = await research_client.claim_task(
            parent_id, f"req-{parent_id}-research", "research", "initial"
        )
        await research_client.acknowledge_assignment(
            parent_id, parent_assignment["assignment_id"], parent_proof
        )

        # 4. Stage the sealed research artifact through the production endpoint.
        async with httpx.AsyncClient() as http:
            stage_res = await http.post(
                f"{base_url}/api/tasks/{parent_id}/artifacts",
                json={
                    "assignment_id": parent_assignment["assignment_id"],
                    "artifact_id": f"art-{parent_id}",
                    "artifact_type": "research_material",
                    "content": {"material": ANALYSIS_MATERIAL},
                    "artifact_version": 1,
                    "payload_schema_version": 1,
                },
                headers={
                    "Authorization": f"Bearer {RESEARCH_RUNTIME_TOKEN}",
                    "X-Assignment-Proof": parent_proof,
                },
            )
            assert stage_res.status_code == 200, stage_res.text
            artifact_id = stage_res.json()["artifact"]["artifact_id"]

            # 5. Accept the research -> analysis delegation through the boundary.
            delegation_res = await http.post(
                f"{base_url}/api/tasks/{parent_id}/delegations",
                json={
                    "proposal_id": f"prop-{parent_id}",
                    "assignment_id": parent_assignment["assignment_id"],
                    "child_intent": "Analyze the staged findings",
                    "artifact_reference": {
                        "producing_task_id": parent_id,
                        "artifact_id": artifact_id,
                        "artifact_version": 1,
                    },
                    "target_capability": "analysis",
                },
                headers={
                    "Authorization": f"Bearer {RESEARCH_RUNTIME_TOKEN}",
                    "X-Assignment-Proof": parent_proof,
                },
            )
            assert delegation_res.status_code == 200, delegation_res.text
        child_id = delegation_res.json()["child_task_id"]

        # 6. Seed queue delivery for the delegated child. This rpush stands in for
        # production dispatch so the test isolates the #222 runtime completion
        # lifecycle (pull-claim -> acknowledge -> execute -> fenced completion);
        # complete executable orchestration owns the delivery path (#224).
        child_requested = TaskRequested(
            task_id=child_id,
            root_task_id=parent_id,
            parent_task_id=parent_id,
            prompt=ANALYSIS_MATERIAL,
            created_at=datetime.now(UTC).isoformat(),
            version=2,
        )
        await gateway._client.rpush("agent.tasks.analysis", child_requested.to_json())

        # 7. Real bus registration, idle publication, and queue consumption.
        await bus.register(metadata)
        await bus.publish(AgentEvent.create(agent_id, "idle", "Waiting for work"))
        consumed = await asyncio.wait_for(bus.next_task(), timeout=2.0)
        assert consumed.task_id == child_id
        assert await gateway._client.llen("agent.tasks.analysis") == 0

        # 8-9. Analysis pull-claim and acknowledge; no grant on a delegated-child turn.
        child_assignment, child_proof = await analysis_client.claim_task(
            child_id, f"req-{child_id}-analysis", "analysis", "delegated_child"
        )
        assert child_assignment.get("delegation_grant_id") is None
        await analysis_client.acknowledge_assignment(
            child_id, child_assignment["assignment_id"], child_proof
        )

        # 10. Execute and complete through the production TurnCompletionClient.
        await processor.run_once(
            consumed,
            assignment_id=child_assignment["assignment_id"],
            assignment_proof=child_proof,
            execution_kind="delegated_child",
        )

        registered = await gateway.agents()
        agent = next(agent for agent in registered if agent["agent_id"] == agent_id)
        assert agent["agent_type"] == "analysis-agent"
        assert agent["capabilities"] == ["analysis"]
        assert agent["dispatch_target"] == "agent.tasks.analysis"

        # 11. Canonical completion records are authoritative.
        outcome = await gateway.get_terminal_outcome(child_id)
        assert outcome is not None
        assert outcome.terminal_status == "completed"
        assert outcome.task_id == child_id
        assert outcome.assignment_id == child_assignment["assignment_id"]
        assert outcome.runtime_principal_id == "principal-analysis-1"
        assert outcome.bound_agent_id == "analysis-agent-1"
        assert outcome.execution_kind == "delegated_child"
        assert outcome.outcome_id.startswith("outcome-")
        result = await gateway.get_canonical_result(child_id)
        assert result is not None
        assert result.output == EXPECTED_ANALYSIS_REPORT
        assert await gateway.get_status(child_id) == "completed"
        audit = await gateway.get_assignment_completion(child_id, child_assignment["assignment_id"])
        assert audit is not None
        assert audit.outcome_id == outcome.outcome_id
        assert audit.fencing_epoch == 1
        assert await gateway.get_active_claim(child_id) is None
        record = await gateway.get_completion_record(child_id, audit.completion_command_id)
        assert record is not None
        assert record.outcome_id == outcome.outcome_id
        assert record.semantic_fingerprint == outcome.semantic_fingerprint
        record_json = record.to_json()
        assert "assignment_proof" not in record_json
        assert child_proof not in record_json
        assert "proof_digest" in record_json

        # 12. Exactly-once parent projection with a single Research hint.
        assert await gateway.get_status(parent_id) == "resumable"
        continuation = await gateway.get_continuation(parent_id, 1)
        assert continuation is not None
        assert continuation.child_task_id == child_id
        assert continuation.continuation_version == 1
        research_hints = await gateway._client.lrange("agent.hints.research", 0, -1)
        parent_hints = [h for h in research_hints if f'"{parent_id}"' in h]
        assert len(parent_hints) == 1
        assert '"expected_continuation_version":1' in parent_hints[0]
        assert "assignment_proof" not in parent_hints[0]
        assert await gateway._client.llen("agent.hints.analysis") == 1

        # 13. Agent events stay observational: fixed lifecycle states only.
        events = await gateway.events(limit=10)
        task_states = [
            event["state"]
            for event in events
            if event.get("task_id") == child_id
        ]
        assert task_states == ["idle", "working"]
        for event in events:
            if event.get("task_id") == child_id:
                assert EXPECTED_ANALYSIS_REPORT not in event.get("detail", "")
                assert child_proof not in event.get("detail", "")

        # 14. Canonical reads do not depend on the event stream at all.
        await gateway._client.delete("agent.events")
        reread = await gateway.get_result(child_id)
        assert reread is not None
        assert reread["status"] == "completed"
        assert reread["output"] == EXPECTED_ANALYSIS_REPORT
    finally:
        if server is not None:
            server.should_exit = True
        if serve_task is not None:
            await serve_task
        await gateway.aclose()
        await bus.aclose()


async def test_redis_completion_persistence_and_canonical_retrieval_round_trip(
    clean_redis: str,
) -> None:
    """Canonical completion commits through COMPLETE_TURN_LUA and reads back via get_result.

    No runtime writes task.result directly: every turn completes through the
    fenced completion boundary, and get_result is outcome-driven.
    """
    from agent_contracts import CompletionCommand
    from orchestrator.application.use_cases import AcknowledgeAssignment, ClaimTask
    from orchestrator.domain.principal import RuntimePrincipal

    gateway = RedisGateway(clean_redis)
    proof_pepper = "test_completion_redis_proof_pepper_4567890123456789"
    claim_uc = ClaimTask(task_store=gateway, assignment_proof_pepper=proof_pepper, delivery_lease_seconds=30.0)
    ack_uc = AcknowledgeAssignment(task_store=gateway, assignment_proof_pepper=proof_pepper)

    try:
        # 1. Worker completed turn with sources round trip
        worker_task_id = f"task_worker_{uuid.uuid4().hex[:8]}"
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
        now = datetime.now(UTC).isoformat()
        worker_task = TaskRequested(
            task_id=worker_task_id,
            root_task_id=worker_task_id,
            parent_task_id=None,
            prompt="Redis completion worker task",
            created_at=now,
            version=2,
        )
        await gateway.register_root_task(worker_task, required_capability="general")
        await gateway.update_status(worker_task_id, "pending")
        worker_principal = RuntimePrincipal(
            principal_id="principal-worker-1",
            bound_agent_id="worker-1",
            authorized_capabilities=("general",),
            active=True,
        )
        worker_claim = await claim_uc.execute(
            principal=worker_principal,
            task_id=worker_task_id,
            request_id=f"req-{worker_task_id}",
            capability="general",
        )
        await ack_uc.execute(
            principal=worker_principal,
            task_id=worker_task_id,
            assignment_id=worker_claim.assignment.assignment_id,
            assignment_proof=worker_claim.assignment_proof or "",
        )
        active = await gateway.get_active_claim(worker_task_id)
        assert active is not None
        worker_command = CompletionCommand(
            completion_command_id=f"cmd-{worker_task_id}",
            assignment_id=active.assignment_id,
            execution_kind="initial",
            completed_result=TaskResult(
                task_id=worker_task_id,
                output="General worker finished successfully",
                sources=[evidence.to_dict()],
            ),
        )
        await gateway.complete_turn(
            task_id=worker_task_id,
            command=worker_command,
            proof_digest=active.proof_digest,
            fencing_epoch=active.fencing_epoch,
            parent_task_id=None,
            root_task_id=worker_task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=active.runtime_principal_id,
            bound_agent_id=active.bound_agent_id,
        )

        retrieved_worker = await gateway.get_result(worker_task_id)
        assert retrieved_worker is not None
        assert retrieved_worker["task_id"] == worker_task_id
        assert retrieved_worker["status"] == "completed"
        assert retrieved_worker["output"] == "General worker finished successfully"
        assert retrieved_worker["acknowledged"] is True
        assert len(retrieved_worker["sources"]) == 1
        assert retrieved_worker["sources"][0]["url"] == "https://example.org/spec"
        assert retrieved_worker["sources"][0]["domain"] == "example.org"

        # 2. Failed turn persists the outcome but never fabricates a TaskResult
        failed_task_id = f"task_failed_{uuid.uuid4().hex[:8]}"
        failed_task = TaskRequested(
            task_id=failed_task_id,
            root_task_id=failed_task_id,
            parent_task_id=None,
            prompt="Redis completion failing task",
            created_at=datetime.now(UTC).isoformat(),
            version=2,
        )
        await gateway.register_root_task(failed_task, required_capability="general")
        await gateway.update_status(failed_task_id, "pending")
        failed_claim = await claim_uc.execute(
            principal=worker_principal,
            task_id=failed_task_id,
            request_id=f"req-{failed_task_id}",
            capability="general",
        )
        await ack_uc.execute(
            principal=worker_principal,
            task_id=failed_task_id,
            assignment_id=failed_claim.assignment.assignment_id,
            assignment_proof=failed_claim.assignment_proof or "",
        )
        failed_active = await gateway.get_active_claim(failed_task_id)
        assert failed_active is not None
        await gateway.complete_turn(
            task_id=failed_task_id,
            command=CompletionCommand(
                completion_command_id=f"cmd-{failed_task_id}",
                assignment_id=failed_active.assignment_id,
                execution_kind="initial",
                failure_code="PROVIDER_UNAVAILABLE",
            ),
            proof_digest=failed_active.proof_digest,
            fencing_epoch=failed_active.fencing_epoch,
            parent_task_id=None,
            root_task_id=failed_task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=failed_active.runtime_principal_id,
            bound_agent_id=failed_active.bound_agent_id,
        )
        retrieved_failed = await gateway.get_result(failed_task_id)
        assert retrieved_failed == {
            "task_id": failed_task_id,
            "status": "failed",
            "output": None,
        }

        # 3. Legacy pre-boundary task.result records remain readable
        legacy_task_id = f"task_legacy_{uuid.uuid4().hex[:8]}"
        await gateway._client.set(
            f"task.result:{legacy_task_id}",
            json.dumps(
                {
                    "task_id": legacy_task_id,
                    "status": "completed",
                    "output": "Legacy direct-write output",
                    "acknowledged": True,
                }
            ),
        )
        assert await gateway.get_result(legacy_task_id) == {
            "task_id": legacy_task_id,
            "status": "completed",
            "output": "Legacy direct-write output",
            "acknowledged": True,
        }

        # 4. Unknown tasks resolve to None; agent events are never consulted
        assert await gateway.get_result(f"task_unknown_{uuid.uuid4().hex[:8]}") is None
    finally:
        await gateway.aclose()


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

@pytest.mark.anyio
async def test_redis_delegation_acceptance_and_zero_mutation(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis, delivery_mode="authenticated_claim")
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        now = datetime.now(UTC).isoformat()
        parent_id = f"parent_{uuid.uuid4().hex[:8]}"
        root_task = TaskRequested(
            task_id=parent_id,
            prompt="Research topic",
            created_at=now,
            root_task_id=parent_id,
            parent_task_id=None,
            version=2,
        )
        await gateway.register_root_task(root_task, initial_status="pending", required_capability="research")

        # Claim as Research
        asgn_id = f"asgn_{uuid.uuid4().hex[:8]}"
        claim = ActiveClaim(
            task_id=parent_id,
            assignment_id=asgn_id,
            claim_id=f"claim_{uuid.uuid4().hex[:8]}",
            root_task_id=parent_id,
            turn_id=f"turn_{uuid.uuid4().hex[:8]}",
            runtime_principal_id="research-principal",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="digest_root",
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            claimed_at=now,
            delivery_expires_at="2099-01-01T00:00:00.000000Z",
            prior_claimable_status="pending",
        )
        grant = DelegationGrant(
            grant_id="grant-1",
            root_task_id=parent_id,
            issued_to_assignment_id=asgn_id,
            issued_to_principal_id="research-principal",
            fencing_epoch=1,
            turn_id=claim.turn_id,
            issued_at=now,
        )
        await gateway.atomic_claim_task(parent_id, claim, grant=grant, now=now)
        await gateway.acknowledge_claim(parent_id, asgn_id, "research-principal", now)

        # Stage artifact
        art = StageArtifact(
            artifact_id="art-1",
            producing_task_id=parent_id,
            artifact_type="research_material",
            content={"material": "findings"},
            created_at=now,
            producing_turn_id=claim.turn_id,
            artifact_version=1,
        )
        await gateway.save_stage_artifact(art, asgn_id, "research-principal", 1)

        # Prepare proposal
        art_ref = ArtifactReference(
            artifact_id="art-1",
            artifact_version=1,
            producing_task_id=parent_id,
        )
        fp = compute_proposal_fingerprint(
            parent_task_id=parent_id,
            proposal_id="prop-1",
            assignment_id=asgn_id,
            target_capability="analysis",
            child_intent="Analyze findings",
            artifact_reference=art_ref,
            parent_input_version=1,
        )
        child_id = f"child_{uuid.uuid4().hex[:8]}"
        deadline_iso = "2099-01-01T00:02:00.000000Z"
        proposal = DelegationProposalRecord(
            parent_task_id=parent_id,
            proposal_id="prop-1",
            child_task_id=child_id,
            delegation_decision_id="dec-1",
            assignment_id=asgn_id,
            runtime_principal_id="research-principal",
            bound_agent_id="research-agent-1",
            proof_digest="digest_root",
            proposal_fingerprint=fp,
            delegation_grant_id="grant-1",
            child_claim_deadline=deadline_iso,
            accepted_at=now,
            target_capability="analysis",
            continuation_version=1,
        )
        child_input = TaskInput(
            task_id=child_id,
            prompt="Analyze findings",
            created_at=now,
            input_version=1,
            artifact_references=(art_ref,),
        )
        continuation = TaskContinuation(
            parent_task_id=parent_id,
            child_task_id=child_id,
            created_at=now,
            updated_at=now,
            input_version=1,
            continuation_capability="research",
            stage="incorporate_analysis_child",
            artifact_references=(art_ref,),
            continuation_version=1,
        )
        yield_rec = AssignmentYieldRecord(
            assignment_id=asgn_id,
            parent_task_id=parent_id,
            principal_id="research-principal",
            proposal_id="prop-1",
            child_task_id=child_id,
            delegation_decision_id="dec-1",
            continuation_version=1,
            yielded_at=now,
        )
        hint = WakeUpHint(
            hint_id="hint-1",
            capability="analysis",
            created_at=now,
            execution_kind="delegated_child",
            task_id=child_id,
        )

        # 1. Accept proposal atomically
        accepted, is_rep = await gateway.accept_delegation_proposal(
            parent_task_id=parent_id,
            proposal=proposal,
            child_task_input=child_input,
            parent_continuation=continuation,
            yield_record=yield_rec,
            wake_up_hint=hint,
            expected_fencing_epoch=1,
            now=now,
        )
        assert not is_rep
        assert accepted == proposal

        # Verify all 17 keys committed correctly
        assert await client.exists(f"task.delegation_proposal:{parent_id}:prop-1") == 1
        assert await client.exists(f"task.status:{child_id}") == 1
        assert await client.exists(f"task.lineage:{child_id}") == 1
        assert await client.exists(f"task.required_capability:{child_id}") == 1
        assert await client.exists(f"task.input:{child_id}:1") == 1
        assert await client.exists(f"task.claim_deadline:{child_id}") == 1
        assert await client.sismember(f"task.children:{parent_id}", child_id) == 1
        assert await client.exists(f"task.continuation:{parent_id}:1") == 1
        assert await client.get(f"task.continuation_latest:{parent_id}") == "1"
        parent_status_raw = await client.get(f"task.status:{parent_id}")
        assert parent_status_raw is not None
        assert json.loads(parent_status_raw)["status"] == "waiting_for_children"
        assert await client.exists(f"task.claim:{parent_id}") == 0  # Retired active claim
        assert await client.exists(f"task.assignment_yield:{parent_id}:{asgn_id}") == 1
        assert await client.sismember(f"task.consumed_waves:{parent_id}", "1") == 1
        assert await client.sismember(f"task.consumed_grants:{parent_id}", "grant-1") == 1
        assert await client.sismember(f"task.delegation_decisions:{parent_id}", "dec-1") == 1

        hints = await client.lrange("agent.hints.analysis", 0, -1)
        assert len(hints) == 1

        # 2. Idempotent replay: zero writes, no second hint
        replayed, is_rep2 = await gateway.accept_delegation_proposal(
            parent_task_id=parent_id,
            proposal=proposal,
            child_task_input=child_input,
            parent_continuation=continuation,
            yield_record=yield_rec,
            wake_up_hint=hint,
            expected_fencing_epoch=1,
            now=now,
        )
        assert is_rep2
        assert replayed == proposal
        assert await client.llen("agent.hints.analysis") == 1

        # 3. Wave already consumed rejection
        second_child = f"child2_{uuid.uuid4().hex[:8]}"
        proposal_wave2 = DelegationProposalRecord(
            parent_task_id=parent_id,
            proposal_id="prop-2",
            child_task_id=second_child,
            delegation_decision_id="dec-2",
            assignment_id=asgn_id,
            runtime_principal_id="research-principal",
            bound_agent_id="research-agent-1",
            proof_digest="digest_root",
            proposal_fingerprint="fp2",
            delegation_grant_id="grant-2",
            child_claim_deadline=deadline_iso,
            accepted_at=now,
            target_capability="analysis",
            continuation_version=2,
        )
        child_input_2 = TaskInput(
            task_id=second_child,
            prompt="Second child",
            created_at=now,
            input_version=1,
            artifact_references=(art_ref,),
        )
        continuation_2 = TaskContinuation(
            parent_task_id=parent_id,
            child_task_id=second_child,
            created_at=now,
            updated_at=now,
            input_version=1,
            continuation_capability="research",
            stage="incorporate_analysis_child",
            artifact_references=(art_ref,),
            continuation_version=2,
        )
        yield_rec_2 = AssignmentYieldRecord(
            assignment_id=asgn_id,
            parent_task_id=parent_id,
            principal_id="research-principal",
            proposal_id="prop-2",
            child_task_id=second_child,
            delegation_decision_id="dec-2",
            continuation_version=2,
            yielded_at=now,
        )
        hint_2 = WakeUpHint(
            hint_id="hint-2",
            capability="analysis",
            created_at=now,
            execution_kind="delegated_child",
            task_id=second_child,
        )

        with pytest.raises(TaskNotClaimableError, match="must be in 'working' status"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal_wave2,
                child_task_input=child_input_2,
                parent_continuation=continuation_2,
                yield_record=yield_rec_2,
                wake_up_hint=hint_2,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.exists(f"task.status:{second_child}") == 0

        # 4. Delegated child pull-claim
        claim_child = ActiveClaim(
            task_id=child_id,
            assignment_id=f"asgn_c_{uuid.uuid4().hex[:8]}",
            claim_id=f"claim_c_{uuid.uuid4().hex[:8]}",
            root_task_id=parent_id,
            turn_id=f"turn_c_{uuid.uuid4().hex[:8]}",
            runtime_principal_id="analysis-principal",
            bound_agent_id="analysis-agent-1",
            assigned_capability="analysis",
            execution_kind="delegated_child",
            fencing_epoch=1,
            proof_digest="digest_child",
            request_id=f"req_c_{uuid.uuid4().hex[:8]}",
            claimed_at=now,
            delivery_expires_at="2099-01-01T00:00:00.000000Z",
            prior_claimable_status="pending",
        )
        active_child_claim, _ = await gateway.atomic_claim_task(child_id, claim_child, now=now, is_root=False)
        assert active_child_claim.task_id == child_id
        child_status_raw = await client.get(f"task.status:{child_id}")
        assert child_status_raw is not None
        assert json.loads(child_status_raw)["status"] == "working"

        # 5. Expired child claim deadline is rejected
        child_exp_id = f"child_exp_{uuid.uuid4().hex[:8]}"
        await client.set(f"task.status:{child_exp_id}", json.dumps({"status": "pending", "task_id": child_exp_id}))
        await client.set(f"task.required_capability:{child_exp_id}", json.dumps({"required_capability": "analysis", "task_id": child_exp_id, "created_at": now, "version": 1}))
        await client.set(f"task.claim_deadline:{child_exp_id}", "2020-01-01T00:00:00.000000Z")

        claim_exp = ActiveClaim(
            task_id=child_exp_id,
            assignment_id=f"asgn_exp_{uuid.uuid4().hex[:8]}",
            claim_id=f"claim_exp_{uuid.uuid4().hex[:8]}",
            root_task_id=parent_id,
            turn_id=f"turn_exp_{uuid.uuid4().hex[:8]}",
            runtime_principal_id="analysis-principal",
            bound_agent_id="analysis-agent-1",
            assigned_capability="analysis",
            execution_kind="delegated_child",
            fencing_epoch=1,
            proof_digest="digest_exp",
            request_id=f"req_exp_{uuid.uuid4().hex[:8]}",
            claimed_at=now,
            delivery_expires_at="2099-01-01T00:00:00.000000Z",
            prior_claimable_status="pending",
        )
        with pytest.raises(ChildClaimDeadlineExpiredError, match="Claim deadline for child task"):
            await gateway.atomic_claim_task(child_exp_id, claim_exp, now=now, is_root=False)

        # 6. get_delegation_proposal roundtrip and error handling
        rec = await gateway.get_delegation_proposal(parent_id, "prop-1")
        assert rec is not None
        assert rec.proposal_id == "prop-1"
        assert rec.child_task_id == child_id

        assert await gateway.get_delegation_proposal(parent_id, "nonexistent") is None
        assert await gateway.get_delegation_proposal("", "prop-1") is None

        # Corrupt JSON at proposal key
        await client.set(f"task.delegation_proposal:{parent_id}:corrupt-prop", "invalid-json")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Failed to decode delegation proposal"):
            await gateway.get_delegation_proposal(parent_id, "corrupt-prop")

        # Wrong Redis key type at proposal key
        await client.sadd(f"task.delegation_proposal:{parent_id}:set-prop", "val")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Wrong Redis type"):
            await gateway.get_delegation_proposal(parent_id, "set-prop")
    finally:
        await client.aclose()
        await gateway.aclose()

@pytest.mark.anyio
async def test_redis_lua_delegation_authority_and_artifact_validations(clean_redis: str) -> None:
    client = redis.from_url(clean_redis, decode_responses=True)
    gateway = RedisGateway(clean_redis)
    now = datetime.now(UTC).isoformat()
    parent_id = f"parent_val_{uuid.uuid4().hex[:8]}"

    try:
        # 1. Setup valid root task
        root_task = TaskRequested(
            task_id=parent_id,
            root_task_id=parent_id,
            parent_task_id=None,
            prompt="Initial research task",
            created_at=now,
            version=2,
        )
        await gateway.register_root_task(root_task, initial_status="pending", required_capability="research")
        asgn_id = f"asgn_{uuid.uuid4().hex[:8]}"
        grant = DelegationGrant(
            grant_id="grant-val-1",
            root_task_id=parent_id,
            issued_to_assignment_id=asgn_id,
            issued_to_principal_id="research-principal",
            fencing_epoch=1,
            turn_id="turn-1",
            issued_at=now,
        )
        claim = ActiveClaim(
            task_id=parent_id,
            assignment_id=asgn_id,
            claim_id=f"claim_{uuid.uuid4().hex[:8]}",
            root_task_id=parent_id,
            turn_id=grant.turn_id,
            runtime_principal_id="research-principal",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="digest_root",
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            claimed_at=now,
            delivery_expires_at="2099-01-01T00:00:00.000000Z",
            prior_claimable_status="pending",
            delegation_grant_id=grant.grant_id,
        )
        await gateway.atomic_claim_task(parent_id, claim, grant=grant, now=now)
        await gateway.acknowledge_claim(parent_id, asgn_id, "research-principal", now)

        # Stage artifact
        art = StageArtifact(
            artifact_id="art-1",
            producing_task_id=parent_id,
            artifact_type="research_material",
            content={"material": "findings"},
            created_at=now,
            producing_turn_id="turn-1",
            artifact_version=1,
        )
        await gateway.save_stage_artifact(art, asgn_id, "research-principal", 1)

        art_ref = ArtifactReference(
            artifact_id="art-1",
            artifact_version=1,
            producing_task_id=parent_id,
        )
        fp = compute_proposal_fingerprint(
            parent_task_id=parent_id,
            proposal_id="prop-val-1",
            assignment_id=asgn_id,
            target_capability="analysis",
            child_intent="Analyze valid findings",
            artifact_reference=art_ref,
            parent_input_version=1,
        )
        child_id = f"child_val_{uuid.uuid4().hex[:8]}"
        proposal = DelegationProposalRecord(
            parent_task_id=parent_id,
            proposal_id="prop-val-1",
            child_task_id=child_id,
            delegation_decision_id="dec-val-1",
            assignment_id=asgn_id,
            runtime_principal_id="research-principal",
            bound_agent_id="research-agent-1",
            proof_digest="digest_root",
            proposal_fingerprint=fp,
            delegation_grant_id=grant.grant_id,
            child_claim_deadline="2099-01-01T00:02:00.000000Z",
            accepted_at=now,
            target_capability="analysis",
            continuation_version=1,
        )
        child_input = TaskInput(
            task_id=child_id,
            prompt="Analyze valid findings",
            created_at=now,
            input_version=1,
            artifact_references=(art_ref,),
        )
        continuation = TaskContinuation(
            parent_task_id=parent_id,
            child_task_id=child_id,
            created_at=now,
            updated_at=now,
            input_version=1,
            continuation_capability="research",
            stage="incorporate_analysis_child",
            artifact_references=(art_ref,),
            continuation_version=1,
        )
        yield_rec = AssignmentYieldRecord(
            assignment_id=asgn_id,
            parent_task_id=parent_id,
            principal_id="research-principal",
            proposal_id="prop-val-1",
            child_task_id=child_id,
            delegation_decision_id="dec-val-1",
            continuation_version=1,
            yielded_at=now,
        )
        hint = WakeUpHint(
            hint_id="hint-val-1",
            capability="analysis",
            created_at=now,
            execution_kind="delegated_child",
            task_id=child_id,
        )

        # Test A: Corrupt grant in Redis -> CorruptAuthoritativeRecordError
        await client.set(f"task.authority_grant:{parent_id}:{grant.grant_id}", "corrupt-grant-data")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Authoritative record or type corruption"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )

        # Restore valid grant
        await client.set(f"task.authority_grant:{parent_id}:{grant.grant_id}", grant.to_json())

        # Test B: Grant fencing epoch mismatch -> FencingEpochRaceError
        bad_epoch_grant = DelegationGrant(
            grant_id=grant.grant_id,
            root_task_id=parent_id,
            issued_to_assignment_id=asgn_id,
            issued_to_principal_id="research-principal",
            fencing_epoch=2,  # mismatch with expected 1
            turn_id="turn-1",
            issued_at=now,
        )
        await client.set(f"task.authority_grant:{parent_id}:{grant.grant_id}", bad_epoch_grant.to_json())
        with pytest.raises(FencingEpochRaceError, match="grant mismatch"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )

        # Restore valid grant
        await client.set(f"task.authority_grant:{parent_id}:{grant.grant_id}", grant.to_json())

        # Test C: Corrupt root authority -> CorruptAuthoritativeRecordError
        await client.set(f"task.authority:{parent_id}", "corrupt-auth-json")
        with pytest.raises(CorruptAuthoritativeRecordError, match="Authoritative record or type corruption"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )

        # Restore valid root authority
        valid_auth = RootAuthorityRecord.create_default(parent_id, created_at=now)
        await client.set(f"task.authority:{parent_id}", valid_auth.to_json())

        # Test D: Unsealed artifact -> ValueError
        unsealed_art = {
            "artifact_id": "art-1",
            "producing_task_id": parent_id,
            "artifact_type": "research_material",
            "content": {"material": "findings"},
            "created_at": now,
            "producing_turn_id": "turn-1",
            "artifact_version": 1,
            "payload_schema_version": 1,
            "sealed": False,
            "version": 1,
        }
        await client.set(f"task.artifact:{parent_id}:art-1:1", json.dumps(unsealed_art))
        with pytest.raises(ValueError, match="ARTIFACT_NOT_SEALED"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )

        # Restore valid sealed artifact
        await client.set(f"task.artifact:{parent_id}:art-1:1", art.to_json())

        # Test E: Artifact producing task mismatch -> ValueError
        wrong_producer_art = dict(unsealed_art)
        wrong_producer_art["sealed"] = True
        wrong_producer_art["producing_task_id"] = "other-parent"
        await client.set(f"task.artifact:{parent_id}:art-1:1", json.dumps(wrong_producer_art))
        with pytest.raises(ValueError, match="ARTIFACT_PROVENANCE_MISMATCH"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )

        # Restore valid sealed artifact
        await client.set(f"task.artifact:{parent_id}:art-1:1", art.to_json())

        # Test Lineage 1: Non-null parent_task_id is rejected with TaskNotClaimableError (PARENT_NOT_ROOT)
        non_root_lineage = {
            "task_id": parent_id,
            "root_task_id": parent_id,
            "parent_task_id": "grandparent-task-id",
        }
        await client.set(f"task.lineage:{parent_id}", json.dumps(non_root_lineage))
        with pytest.raises(TaskNotClaimableError, match="PARENT_NOT_ROOT"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        # Verify zero mutations
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.exists(f"task.delegation_proposal:{parent_id}:prop-val-1") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Test Lineage 2: Root task ID mismatch is rejected with zero mutations (CORRUPT_PARENT_LINEAGE)
        mismatched_root_lineage = {
            "task_id": parent_id,
            "root_task_id": "different-root-task-id",
            "parent_task_id": None,
        }
        await client.set(f"task.lineage:{parent_id}", json.dumps(mismatched_root_lineage))
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_PARENT_LINEAGE"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        # Verify zero mutations
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.exists(f"task.delegation_proposal:{parent_id}:prop-val-1") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Test Lineage 3: Task ID mismatch is rejected with zero mutations (CORRUPT_PARENT_LINEAGE)
        mismatched_task_lineage = {
            "task_id": "different-task-id",
            "root_task_id": parent_id,
            "parent_task_id": None,
        }
        await client.set(f"task.lineage:{parent_id}", json.dumps(mismatched_task_lineage))
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_PARENT_LINEAGE"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        # Verify zero mutations
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.exists(f"task.delegation_proposal:{parent_id}:prop-val-1") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Test Lineage 4: Missing task_id is rejected with zero mutations (CORRUPT_PARENT_LINEAGE)
        missing_task_id_lineage = {
            "root_task_id": parent_id,
            "parent_task_id": None,
        }
        await client.set(f"task.lineage:{parent_id}", json.dumps(missing_task_id_lineage))
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_PARENT_LINEAGE"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.exists(f"task.delegation_proposal:{parent_id}:prop-val-1") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Test Lineage 5: Missing root_task_id is rejected with zero mutations (CORRUPT_PARENT_LINEAGE)
        missing_root_id_lineage = {
            "task_id": parent_id,
            "parent_task_id": None,
        }
        await client.set(f"task.lineage:{parent_id}", json.dumps(missing_root_id_lineage))
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_PARENT_LINEAGE"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.exists(f"task.delegation_proposal:{parent_id}:prop-val-1") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Test Lineage 6: Missing parent_task_id field entirely is rejected with zero mutations (CORRUPT_PARENT_LINEAGE)
        missing_parent_id_lineage = {
            "task_id": parent_id,
            "root_task_id": parent_id,
        }
        await client.set(f"task.lineage:{parent_id}", json.dumps(missing_parent_id_lineage))
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_PARENT_LINEAGE"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.exists(f"task.delegation_proposal:{parent_id}:prop-val-1") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Restore canonical root lineage
        canonical_lineage = {
            "task_id": parent_id,
            "root_task_id": parent_id,
            "parent_task_id": None,
        }
        await client.set(f"task.lineage:{parent_id}", json.dumps(canonical_lineage))

        # Test Turn 1: Missing claim turn_id is rejected with CORRUPT_CLAIM (zero mutations)
        corrupt_claim_dict = json.loads(claim.to_json())
        corrupt_claim_dict["state"] = "acknowledged"
        del corrupt_claim_dict["turn_id"]
        await client.set(f"task.claim:{parent_id}", json.dumps(corrupt_claim_dict))
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_CLAIM"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.exists(f"task.delegation_proposal:{parent_id}:prop-val-1") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Restore valid acknowledged claim
        ack_claim_dict = json.loads(claim.to_json())
        ack_claim_dict["state"] = "acknowledged"
        await client.set(f"task.claim:{parent_id}", json.dumps(ack_claim_dict))

        # Test Turn 2: Missing grant turn_id is rejected with CORRUPT_GRANT (zero mutations)
        corrupt_grant_dict = json.loads(grant.to_json())
        del corrupt_grant_dict["turn_id"]
        await client.set(f"task.authority_grant:{parent_id}:{grant.grant_id}", json.dumps(corrupt_grant_dict))
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_GRANT"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.exists(f"task.delegation_proposal:{parent_id}:prop-val-1") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Test Turn 3: Mismatched turn_id between grant and claim is rejected with GRANT_TURN_MISMATCH (zero mutations)
        mismatched_turn_grant = DelegationGrant(
            grant_id=grant.grant_id,
            root_task_id=parent_id,
            issued_to_assignment_id=asgn_id,
            issued_to_principal_id="research-principal",
            fencing_epoch=1,
            turn_id="turn-different-999",
            issued_at=now,
        )
        await client.set(f"task.authority_grant:{parent_id}:{grant.grant_id}", mismatched_turn_grant.to_json())
        with pytest.raises(ClaimNotFoundError, match="GRANT_TURN_MISMATCH"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.exists(f"task.delegation_proposal:{parent_id}:prop-val-1") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Restore valid grant
        await client.set(f"task.authority_grant:{parent_id}:{grant.grant_id}", grant.to_json())

        # Test F: Valid proposal acceptance succeeds!
        accepted, is_rep = await gateway.accept_delegation_proposal(
            parent_task_id=parent_id,
            proposal=proposal,
            child_task_input=child_input,
            parent_continuation=continuation,
            yield_record=yield_rec,
            wake_up_hint=hint,
            expected_fencing_epoch=1,
            now=now,
        )
        assert not is_rep
        assert accepted == proposal
        assert await client.get(f"task.fence:{child_id}") == "0"
    finally:
        await client.aclose()
        await gateway.aclose()

async def assert_zero_delegation_mutations(
    client: redis.Redis,
    parent_id: str,
    proposal_id: str,
    child_id: str,
    seeded_child_keys: set[str] | None = None,
    seeded_waves: set[str] | None = None,
    seeded_grants: set[str] | None = None,
    seeded_continuations: set[str] | None = None,
) -> None:
    seeded_child = seeded_child_keys or set()
    seeded_w = seeded_waves or set()
    seeded_g = seeded_grants or set()
    seeded_c = seeded_continuations or set()

    assert await client.exists(f"task.delegation_proposal:{parent_id}:{proposal_id}") == 0
    for key_suffix in ("status", "lineage", "required_capability", "claim_deadline", "fence"):
        k = f"task.{key_suffix}:{child_id}"
        if k not in seeded_child:
            assert await client.exists(k) == 0
    if f"task.input:{child_id}:1" not in seeded_child:
        assert await client.exists(f"task.input:{child_id}:1") == 0

    assert await client.sismember(f"task.children:{parent_id}", child_id) == 0
    if "1" not in seeded_c:
        assert await client.exists(f"task.continuation:{parent_id}:1") == 0
        assert await client.exists(f"task.continuation_latest:{parent_id}") == 0

    yield_keys = await client.keys(f"task.assignment_yield:{parent_id}:*")
    assert len(yield_keys) == 0

    all_waves = set(await client.smembers(f"task.consumed_waves:{parent_id}"))
    assert all_waves == seeded_w

    all_grants = set(await client.smembers(f"task.consumed_grants:{parent_id}"))
    assert all_grants == seeded_g

    all_decisions = await client.keys(f"task.delegation_decisions:{parent_id}")
    assert len(all_decisions) == 0

    assert await client.llen("agent.hints.analysis") == 0


@pytest.mark.anyio
async def test_redis_lua_delegation_zero_mutation_rejection_coverage(clean_redis: str) -> None:
    client = redis.from_url(clean_redis, decode_responses=True)
    gateway = RedisGateway(clean_redis)
    now = datetime.now(UTC).isoformat()
    parent_id = f"parent_rej_{uuid.uuid4().hex[:8]}"
    prop_id = "prop-rej-1"
    child_id = f"child_rej_{uuid.uuid4().hex[:8]}"

    try:
        root_task = TaskRequested(
            task_id=parent_id,
            root_task_id=parent_id,
            parent_task_id=None,
            prompt="Initial research task",
            created_at=now,
            version=2,
        )
        await gateway.register_root_task(root_task, initial_status="pending", required_capability="research")
        asgn_id = f"asgn_{uuid.uuid4().hex[:8]}"
        grant = DelegationGrant(
            grant_id="grant-rej-1",
            root_task_id=parent_id,
            issued_to_assignment_id=asgn_id,
            issued_to_principal_id="research-principal",
            fencing_epoch=1,
            turn_id="turn-rej-1",
            issued_at=now,
        )
        claim = ActiveClaim(
            task_id=parent_id,
            assignment_id=asgn_id,
            claim_id=f"claim_{uuid.uuid4().hex[:8]}",
            root_task_id=parent_id,
            turn_id=grant.turn_id,
            runtime_principal_id="research-principal",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="digest_root",
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            claimed_at=now,
            delivery_expires_at="2099-01-01T00:00:00.000000Z",
            prior_claimable_status="pending",
            delegation_grant_id=grant.grant_id,
        )
        await gateway.atomic_claim_task(parent_id, claim, grant=grant, now=now)
        await gateway.acknowledge_claim(parent_id, asgn_id, "research-principal", now)

        art = StageArtifact(
            artifact_id="art-1",
            producing_task_id=parent_id,
            artifact_type="research_material",
            content={"material": "findings"},
            created_at=now,
            producing_turn_id="turn-rej-1",
            artifact_version=1,
        )
        await gateway.save_stage_artifact(art, asgn_id, "research-principal", 1)

        art_ref = ArtifactReference(
            artifact_id="art-1",
            artifact_version=1,
            producing_task_id=parent_id,
        )
        fp = compute_proposal_fingerprint(
            parent_task_id=parent_id,
            proposal_id=prop_id,
            assignment_id=asgn_id,
            target_capability="analysis",
            child_intent="Analyze valid findings",
            artifact_reference=art_ref,
            parent_input_version=1,
        )
        proposal = DelegationProposalRecord(
            parent_task_id=parent_id,
            proposal_id=prop_id,
            child_task_id=child_id,
            delegation_decision_id="dec-rej-1",
            assignment_id=asgn_id,
            runtime_principal_id="research-principal",
            bound_agent_id="research-agent-1",
            proof_digest="digest_root",
            proposal_fingerprint=fp,
            delegation_grant_id=grant.grant_id,
            child_claim_deadline="2099-01-01T00:02:00.000000Z",
            accepted_at=now,
            target_capability="analysis",
            continuation_version=1,
        )
        child_input = TaskInput(
            task_id=child_id,
            prompt="Analyze valid findings",
            created_at=now,
            input_version=1,
            artifact_references=(art_ref,),
        )
        continuation = TaskContinuation(
            parent_task_id=parent_id,
            child_task_id=child_id,
            created_at=now,
            updated_at=now,
            input_version=1,
            continuation_capability="research",
            stage="incorporate_analysis_child",
            artifact_references=(art_ref,),
            continuation_version=1,
        )
        yield_rec = AssignmentYieldRecord(
            assignment_id=asgn_id,
            parent_task_id=parent_id,
            principal_id="research-principal",
            proposal_id=prop_id,
            child_task_id=child_id,
            delegation_decision_id="dec-rej-1",
            continuation_version=1,
            yielded_at=now,
        )
        hint = WakeUpHint(
            hint_id="hint-rej-1",
            capability="analysis",
            created_at=now,
            execution_kind="delegated_child",
            task_id=child_id,
        )

        # 1. continuation version 1 already exists -> ConflictingContinuationError
        await client.set(f"task.continuation:{parent_id}:1", continuation.to_json())
        with pytest.raises(ConflictingContinuationError):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id, seeded_continuations={"1"})
        await client.delete(f"task.continuation:{parent_id}:1")

        # 2. child status already exists -> ConflictingChildTaskRegistrationError
        await client.set(f"task.status:{child_id}", json.dumps({"status": "pending", "task_id": child_id}))
        with pytest.raises(ConflictingChildTaskRegistrationError):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id, seeded_child_keys={f"task.status:{child_id}"})
        await client.delete(f"task.status:{child_id}")

        # 3. wave 1 already consumed -> WaveBudgetExceededError
        await client.sadd(f"task.consumed_waves:{parent_id}", "1")
        with pytest.raises(WaveBudgetExceededError):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id, seeded_waves={"1"})
        await client.srem(f"task.consumed_waves:{parent_id}", "1")

        # 4. delegation grant already consumed -> GrantAlreadyConsumedError
        await client.sadd(f"task.consumed_grants:{parent_id}", grant.grant_id)
        with pytest.raises(GrantAlreadyConsumedError):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id, seeded_grants={grant.grant_id})
        await client.srem(f"task.consumed_grants:{parent_id}", grant.grant_id)

        # 5. Wrong Redis type for parent status
        saved_status = await client.get(f"task.status:{parent_id}")
        await client.delete(f"task.status:{parent_id}")
        await client.sadd(f"task.status:{parent_id}", "wrong_set_val")
        with pytest.raises(CorruptAuthoritativeRecordError, match="WRONG_KEY_TYPE_PARENT_STATUS"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        await client.delete(f"task.status:{parent_id}")
        if saved_status:
            await client.set(f"task.status:{parent_id}", saved_status)
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id)

        # 6. Wrong Redis type for parent claim
        saved_claim = await client.get(f"task.claim:{parent_id}")
        await client.delete(f"task.claim:{parent_id}")
        await client.sadd(f"task.claim:{parent_id}", "wrong_set_val")
        with pytest.raises(CorruptAuthoritativeRecordError, match="WRONG_KEY_TYPE_CLAIM"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        await client.delete(f"task.claim:{parent_id}")
        if saved_claim:
            await client.set(f"task.claim:{parent_id}", saved_claim)
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id)

        # 7. Wrong Redis type for root authority
        saved_auth = await client.get(f"task.authority:{parent_id}")
        await client.delete(f"task.authority:{parent_id}")
        await client.sadd(f"task.authority:{parent_id}", "wrong_set_val")
        with pytest.raises(CorruptAuthoritativeRecordError, match="WRONG_KEY_TYPE_ROOT_AUTHORITY"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        await client.delete(f"task.authority:{parent_id}")
        if saved_auth:
            await client.set(f"task.authority:{parent_id}", saved_auth)
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id)

        # 8. Wrong Redis type for delegation grant
        saved_grant = await client.get(f"task.authority_grant:{parent_id}:{grant.grant_id}")
        await client.delete(f"task.authority_grant:{parent_id}:{grant.grant_id}")
        await client.sadd(f"task.authority_grant:{parent_id}:{grant.grant_id}", "wrong_set_val")
        with pytest.raises(CorruptAuthoritativeRecordError, match="WRONG_KEY_TYPE_GRANT_RECORD"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        await client.delete(f"task.authority_grant:{parent_id}:{grant.grant_id}")
        if saved_grant:
            await client.set(f"task.authority_grant:{parent_id}:{grant.grant_id}", saved_grant)
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id)

        # 9. Wrong Redis type for artifact
        saved_art = await client.get(f"task.artifact:{parent_id}:art-1:1")
        await client.delete(f"task.artifact:{parent_id}:art-1:1")
        await client.sadd(f"task.artifact:{parent_id}:art-1:1", "wrong_set_val")
        with pytest.raises(CorruptAuthoritativeRecordError, match="WRONG_KEY_TYPE_ARTIFACT"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        await client.delete(f"task.artifact:{parent_id}:art-1:1")
        if saved_art:
            await client.set(f"task.artifact:{parent_id}:art-1:1", saved_art)
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id)

        # 10. Wrong Redis type for hint queue (string instead of list)
        await client.set("agent.hints.analysis", "string_instead_of_list")
        with pytest.raises(CorruptAuthoritativeRecordError, match="WRONG_KEY_TYPE_HINT_QUEUE"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        await client.delete("agent.hints.analysis")
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id)

        # 11. Corrupt JSON for parent claim
        await client.set(f"task.claim:{parent_id}", "corrupt-non-json")
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_CLAIM"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        if saved_claim:
            await client.set(f"task.claim:{parent_id}", saved_claim)
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id)

        # 12. Corrupt JSON for root authority
        await client.set(f"task.authority:{parent_id}", "corrupt-non-json")
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_ROOT_AUTHORITY"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        if saved_auth:
            await client.set(f"task.authority:{parent_id}", saved_auth)
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id)

        # 13. Corrupt JSON for delegation grant
        await client.set(f"task.authority_grant:{parent_id}:{grant.grant_id}", "corrupt-non-json")
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_GRANT"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        if saved_grant:
            await client.set(f"task.authority_grant:{parent_id}:{grant.grant_id}", saved_grant)
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id)

        # 14. Corrupt JSON for artifact
        await client.set(f"task.artifact:{parent_id}:art-1:1", "corrupt-non-json")
        with pytest.raises(CorruptAuthoritativeRecordError, match="CORRUPT_ARTIFACT"):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        if saved_art:
            await client.set(f"task.artifact:{parent_id}:art-1:1", saved_art)
        await assert_zero_delegation_mutations(client, parent_id, prop_id, child_id)
    finally:
        await client.aclose()
        await gateway.aclose()

@pytest.mark.anyio
async def test_redis_concurrent_first_acceptance_atomicity(clean_redis: str) -> None:
    client = redis.from_url(clean_redis, decode_responses=True)
    gateway = RedisGateway(clean_redis)
    now = datetime.now(UTC).isoformat()
    parent_id = f"parent_conc_{uuid.uuid4().hex[:8]}"
    prop_id = "prop-conc-1"
    child_id = f"child_conc_{uuid.uuid4().hex[:8]}"

    try:
        root_task = TaskRequested(
            task_id=parent_id,
            root_task_id=parent_id,
            parent_task_id=None,
            prompt="Initial research task",
            created_at=now,
            version=2,
        )
        await gateway.register_root_task(root_task, initial_status="pending", required_capability="research")
        asgn_id = f"asgn_{uuid.uuid4().hex[:8]}"
        grant = DelegationGrant(
            grant_id="grant-conc-1",
            root_task_id=parent_id,
            issued_to_assignment_id=asgn_id,
            issued_to_principal_id="research-principal",
            fencing_epoch=1,
            turn_id="turn-conc-1",
            issued_at=now,
        )
        claim = ActiveClaim(
            task_id=parent_id,
            assignment_id=asgn_id,
            claim_id=f"claim_{uuid.uuid4().hex[:8]}",
            root_task_id=parent_id,
            turn_id=grant.turn_id,
            runtime_principal_id="research-principal",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="digest_root",
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            claimed_at=now,
            delivery_expires_at="2099-01-01T00:00:00.000000Z",
            prior_claimable_status="pending",
            delegation_grant_id=grant.grant_id,
        )
        await gateway.atomic_claim_task(parent_id, claim, grant=grant, now=now)
        await gateway.acknowledge_claim(parent_id, asgn_id, "research-principal", now)

        art = StageArtifact(
            artifact_id="art-1",
            producing_task_id=parent_id,
            artifact_type="research_material",
            content={"material": "findings"},
            created_at=now,
            producing_turn_id="turn-conc-1",
            artifact_version=1,
        )
        await gateway.save_stage_artifact(art, asgn_id, "research-principal", 1)

        art_ref = ArtifactReference(
            artifact_id="art-1",
            artifact_version=1,
            producing_task_id=parent_id,
        )
        fp = compute_proposal_fingerprint(
            parent_task_id=parent_id,
            proposal_id=prop_id,
            assignment_id=asgn_id,
            target_capability="analysis",
            child_intent="Analyze valid findings",
            artifact_reference=art_ref,
            parent_input_version=1,
        )
        proposal = DelegationProposalRecord(
            parent_task_id=parent_id,
            proposal_id=prop_id,
            child_task_id=child_id,
            delegation_decision_id="dec-conc-1",
            assignment_id=asgn_id,
            runtime_principal_id="research-principal",
            bound_agent_id="research-agent-1",
            proof_digest="digest_root",
            proposal_fingerprint=fp,
            delegation_grant_id=grant.grant_id,
            child_claim_deadline="2099-01-01T00:02:00.000000Z",
            accepted_at=now,
            target_capability="analysis",
            continuation_version=1,
        )
        child_input = TaskInput(
            task_id=child_id,
            prompt="Analyze valid findings",
            created_at=now,
            input_version=1,
            artifact_references=(art_ref,),
        )
        continuation = TaskContinuation(
            parent_task_id=parent_id,
            child_task_id=child_id,
            created_at=now,
            updated_at=now,
            input_version=1,
            continuation_capability="research",
            stage="incorporate_analysis_child",
            artifact_references=(art_ref,),
            continuation_version=1,
        )
        yield_rec = AssignmentYieldRecord(
            assignment_id=asgn_id,
            parent_task_id=parent_id,
            principal_id="research-principal",
            proposal_id=prop_id,
            child_task_id=child_id,
            delegation_decision_id="dec-conc-1",
            continuation_version=1,
            yielded_at=now,
        )
        hint = WakeUpHint(
            hint_id="hint-conc-1",
            capability="analysis",
            created_at=now,
            execution_kind="delegated_child",
            task_id=child_id,
        )

        # Issue two concurrent identical first-acceptance operations
        (res1, is_rep1), (res2, is_rep2) = await asyncio.gather(
            gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            ),
            gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            ),
        )

        # One call reports first acceptance, the other reports idempotent replay
        assert (not is_rep1 and is_rep2) or (is_rep1 and not is_rep2)

        # Both return identical accepted records
        assert res1.proposal_id == res2.proposal_id == prop_id
        assert res1.child_task_id == res2.child_task_id == child_id
        assert res1.delegation_decision_id == res2.delegation_decision_id == "dec-conc-1"
        assert res1.proposal_fingerprint == res2.proposal_fingerprint == fp

        # Authoritative state invariants: exactly one of each record committed
        assert await client.exists(f"task.delegation_proposal:{parent_id}:{prop_id}") == 1
        assert await client.exists(f"task.status:{child_id}") == 1
        assert await client.get(f"task.fence:{child_id}") == "0"
        assert await client.smembers(f"task.children:{parent_id}") == {child_id}
        assert await client.exists(f"task.continuation:{parent_id}:1") == 1
        assert await client.get(f"task.continuation_latest:{parent_id}") == "1"
        assert await client.exists(f"task.assignment_yield:{parent_id}:{asgn_id}") == 1
        assert await client.smembers(f"task.consumed_waves:{parent_id}") == {"1"}
        assert await client.smembers(f"task.consumed_grants:{parent_id}") == {grant.grant_id}
        assert await client.smembers(f"task.delegation_decisions:{parent_id}") == {"dec-conc-1"}

        # Exactly one Analysis hint emitted
        assert await client.llen("agent.hints.analysis") == 1
    finally:
        await client.aclose()
        await gateway.aclose()

async def _setup_claimed_parent(gateway: RedisGateway) -> tuple[str, str, DelegationGrant, ActiveClaim, ArtifactReference, str]:
    now = datetime.now(UTC).isoformat()
    parent_id = f"parent_{uuid.uuid4().hex[:8]}"
    root_task = TaskRequested(
        task_id=parent_id,
        prompt="Research topic",
        created_at=now,
        root_task_id=parent_id,
        parent_task_id=None,
        version=2,
    )
    await gateway.register_root_task(root_task, initial_status="pending", required_capability="research")

    asgn_id = f"asgn_{uuid.uuid4().hex[:8]}"
    claim = ActiveClaim(
        task_id=parent_id,
        assignment_id=asgn_id,
        claim_id=f"claim_{uuid.uuid4().hex[:8]}",
        root_task_id=parent_id,
        turn_id=f"turn_{uuid.uuid4().hex[:8]}",
        runtime_principal_id="research-principal",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="digest_root",
        request_id=f"req_{uuid.uuid4().hex[:8]}",
        claimed_at=now,
        delivery_expires_at="2099-01-01T00:00:00.000000Z",
        prior_claimable_status="pending",
    )
    grant = DelegationGrant(
        grant_id="grant-1",
        root_task_id=parent_id,
        issued_to_assignment_id=asgn_id,
        issued_to_principal_id="research-principal",
        fencing_epoch=1,
        turn_id=claim.turn_id,
        issued_at=now,
    )
    await gateway.atomic_claim_task(parent_id, claim, grant=grant, now=now)
    await gateway.acknowledge_claim(parent_id, asgn_id, "research-principal", now)

    art = StageArtifact(
        artifact_id="art-1",
        producing_task_id=parent_id,
        artifact_type="research_material",
        content={"material": "findings"},
        created_at=now,
        producing_turn_id=claim.turn_id,
        artifact_version=1,
        payload_schema_version=1,
        sealed=True,
    )
    await gateway.save_stage_artifact(art, asgn_id, "research-principal", 1)

    art_ref = ArtifactReference(
        artifact_id="art-1",
        artifact_version=1,
        producing_task_id=parent_id,
    )
    return parent_id, asgn_id, grant, claim, art_ref, now


def _build_test_proposal(
    parent_id: str,
    asgn_id: str,
    grant: DelegationGrant,
    art_ref: ArtifactReference,
    now: str,
    child_id: str | None = None,
    prop_id: str = "prop-1",
    bound_agent_id: str = "research-agent-1",
) -> tuple[DelegationProposalRecord, TaskInput, TaskContinuation, AssignmentYieldRecord, WakeUpHint, str]:
    child_id = child_id or f"child_{uuid.uuid4().hex[:8]}"
    fp = compute_proposal_fingerprint(
        parent_task_id=parent_id,
        proposal_id=prop_id,
        assignment_id=asgn_id,
        target_capability="analysis",
        child_intent="Analyze valid findings",
        artifact_reference=art_ref,
        parent_input_version=1,
    )
    proposal = DelegationProposalRecord(
        parent_task_id=parent_id,
        proposal_id=prop_id,
        child_task_id=child_id,
        delegation_decision_id="dec-1",
        assignment_id=asgn_id,
        runtime_principal_id="research-principal",
        bound_agent_id=bound_agent_id,
        proof_digest="digest_root",
        proposal_fingerprint=fp,
        delegation_grant_id=grant.grant_id,
        child_claim_deadline="2099-01-01T00:02:00.000000Z",
        accepted_at=now,
        target_capability="analysis",
        continuation_version=1,
    )
    child_input = TaskInput(
        task_id=child_id,
        prompt="Analyze valid findings",
        created_at=now,
        input_version=1,
        artifact_references=(art_ref,),
    )
    continuation = TaskContinuation(
        parent_task_id=parent_id,
        child_task_id=child_id,
        created_at=now,
        updated_at=now,
        input_version=1,
        continuation_version=1,
    )
    yield_rec = AssignmentYieldRecord(
        assignment_id=asgn_id,
        parent_task_id=parent_id,
        principal_id="research-principal",
        proposal_id=prop_id,
        child_task_id=child_id,
        delegation_decision_id="dec-1",
        continuation_version=1,
        yielded_at=now,
    )
    hint = WakeUpHint(
        hint_id=f"hint_{uuid.uuid4().hex[:8]}",
        capability="analysis",
        created_at=now,
        execution_kind="delegated_child",
        task_id=child_id,
    )
    return proposal, child_input, continuation, yield_rec, hint, child_id


async def test_redis_lua_delegation_exact_bounds_enforcement(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis, delivery_mode="authenticated_claim")
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        parent_id, asgn_id, grant, _claim, art_ref, now = await _setup_claimed_parent(gateway)

        # 1. Root authority violations: max_delegation_depth, max_delegation_waves, max_children_per_wave
        authority_key = f"task.authority:{parent_id}"
        canonical_authority = json.loads(await client.get(authority_key) or "{}")

        bad_authority_cases = [
            ("max_delegation_depth", 0),
            ("max_delegation_depth", 2),
            ("max_delegation_depth", 1.5),
            ("max_delegation_depth", "1"),
            ("max_delegation_depth", True),
            ("max_delegation_depth", None),
            ("max_delegation_waves", 0),
            ("max_delegation_waves", 2),
            ("max_delegation_waves", 1.5),
            ("max_delegation_waves", "1"),
            ("max_delegation_waves", True),
            ("max_delegation_waves", None),
            ("max_children_per_wave", 0),
            ("max_children_per_wave", 2),
            ("max_children_per_wave", 1.5),
            ("max_children_per_wave", "1"),
            ("max_children_per_wave", True),
            ("max_children_per_wave", None),
        ]

        for field, bad_val in bad_authority_cases:
            tampered = dict(canonical_authority)
            if bad_val is None:
                tampered.pop(field, None)
            else:
                tampered[field] = bad_val
            await client.set(authority_key, json.dumps(tampered))

            proposal, child_input, continuation, yield_rec, hint, child_id = _build_test_proposal(
                parent_id, asgn_id, grant, art_ref, now, prop_id=f"prop-auth-{field}-{bad_val}"
            )
            with pytest.raises(WaveBudgetExceededError):
                await gateway.accept_delegation_proposal(
                    parent_task_id=parent_id,
                    proposal=proposal,
                    child_task_input=child_input,
                    parent_continuation=continuation,
                    yield_record=yield_rec,
                    wake_up_hint=hint,
                    expected_fencing_epoch=1,
                    now=now,
                )
            assert await client.exists(f"task.status:{child_id}") == 0
            assert await client.llen("agent.hints.analysis") == 0

        # Restore canonical authority
        await client.set(authority_key, json.dumps(canonical_authority))

        # 2. Grant bounds violations: max_depth, max_waves, max_children
        grant_key = f"task.authority_grant:{parent_id}:{grant.grant_id}"
        canonical_grant = json.loads(await client.get(grant_key) or "{}")

        bad_grant_cases = [
            ("max_depth", 0),
            ("max_depth", 2),
            ("max_depth", 1.5),
            ("max_depth", "1"),
            ("max_depth", True),
            ("max_depth", None),
            ("max_waves", 0),
            ("max_waves", 2),
            ("max_waves", 1.5),
            ("max_waves", "1"),
            ("max_waves", True),
            ("max_waves", None),
            ("max_children", 0),
            ("max_children", 2),
            ("max_children", 1.5),
            ("max_children", "1"),
            ("max_children", True),
            ("max_children", None),
        ]

        for field, bad_val in bad_grant_cases:
            tampered_grant = dict(canonical_grant)
            if bad_val is None:
                tampered_grant.pop(field, None)
            else:
                tampered_grant[field] = bad_val
            await client.set(grant_key, json.dumps(tampered_grant))

            proposal, child_input, continuation, yield_rec, hint, child_id = _build_test_proposal(
                parent_id, asgn_id, grant, art_ref, now, prop_id=f"prop-grant-{field}-{bad_val}"
            )
            with pytest.raises(WaveBudgetExceededError):
                await gateway.accept_delegation_proposal(
                    parent_task_id=parent_id,
                    proposal=proposal,
                    child_task_input=child_input,
                    parent_continuation=continuation,
                    yield_record=yield_rec,
                    wake_up_hint=hint,
                    expected_fencing_epoch=1,
                    now=now,
                )
            assert await client.exists(f"task.status:{child_id}") == 0
            assert await client.llen("agent.hints.analysis") == 0

    finally:
        await client.aclose()
        await gateway.aclose()

async def test_redis_lua_delegation_parent_input_validation(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis, delivery_mode="authenticated_claim")
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        parent_id, asgn_id, grant, _claim, art_ref, now = await _setup_claimed_parent(gateway)
        input_key = f"task.input:{parent_id}:1"
        canonical_input = json.loads(await client.get(input_key) or "{}")

        # Case 1: Corrupt non-JSON
        await client.set(input_key, "corrupt-non-json{")
        proposal, child_input, continuation, yield_rec, hint, child_id = _build_test_proposal(
            parent_id, asgn_id, grant, art_ref, now, prop_id="prop-corrupt-json"
        )
        with pytest.raises(CorruptAuthoritativeRecordError):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Case 2: Missing or mismatched task_id
        for bad_task_id, expected_err in [(None, CorruptAuthoritativeRecordError), ("wrong-task-id", ValueError)]:
            tampered = dict(canonical_input)
            if bad_task_id is None:
                tampered.pop("task_id", None)
            else:
                tampered["task_id"] = bad_task_id
            await client.set(input_key, json.dumps(tampered))

            proposal, child_input, continuation, yield_rec, hint, child_id = _build_test_proposal(
                parent_id, asgn_id, grant, art_ref, now, prop_id=f"prop-taskid-{bad_task_id}"
            )
            with pytest.raises(expected_err):
                await gateway.accept_delegation_proposal(
                    parent_task_id=parent_id,
                    proposal=proposal,
                    child_task_input=child_input,
                    parent_continuation=continuation,
                    yield_record=yield_rec,
                    wake_up_hint=hint,
                    expected_fencing_epoch=1,
                    now=now,
                )
            assert await client.exists(f"task.status:{child_id}") == 0
            assert await client.llen("agent.hints.analysis") == 0

        # Case 3: Missing or mismatched input_version
        for bad_version, expected_err in [(None, CorruptAuthoritativeRecordError), (99, ValueError)]:
            tampered = dict(canonical_input)
            if bad_version is None:
                tampered.pop("input_version", None)
            else:
                tampered["input_version"] = bad_version
            await client.set(input_key, json.dumps(tampered))

            proposal, child_input, continuation, yield_rec, hint, child_id = _build_test_proposal(
                parent_id, asgn_id, grant, art_ref, now, prop_id=f"prop-ver-{bad_version}"
            )
            with pytest.raises(expected_err):
                await gateway.accept_delegation_proposal(
                    parent_task_id=parent_id,
                    proposal=proposal,
                    child_task_input=child_input,
                    parent_continuation=continuation,
                    yield_record=yield_rec,
                    wake_up_hint=hint,
                    expected_fencing_epoch=1,
                    now=now,
                )
            assert await client.exists(f"task.status:{child_id}") == 0
            assert await client.llen("agent.hints.analysis") == 0

        # Case 4: Unsupported contract version
        tampered = dict(canonical_input)
        tampered["version"] = 2
        await client.set(input_key, json.dumps(tampered))
        proposal, child_input, continuation, yield_rec, hint, child_id = _build_test_proposal(
            parent_id, asgn_id, grant, art_ref, now, prop_id="prop-contract-ver"
        )
        with pytest.raises(CorruptAuthoritativeRecordError):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # Case 5: Missing or empty prompt
        for bad_prompt in [None, "", "   ", 12345]:
            tampered = dict(canonical_input)
            if bad_prompt is None:
                tampered.pop("prompt", None)
            else:
                tampered["prompt"] = bad_prompt
            await client.set(input_key, json.dumps(tampered))

            proposal, child_input, continuation, yield_rec, hint, child_id = _build_test_proposal(
                parent_id, asgn_id, grant, art_ref, now, prop_id=f"prop-prompt-{bad_prompt}"
            )
            with pytest.raises(CorruptAuthoritativeRecordError):
                await gateway.accept_delegation_proposal(
                    parent_task_id=parent_id,
                    proposal=proposal,
                    child_task_input=child_input,
                    parent_continuation=continuation,
                    yield_record=yield_rec,
                    wake_up_hint=hint,
                    expected_fencing_epoch=1,
                    now=now,
                )
            assert await client.exists(f"task.status:{child_id}") == 0
            assert await client.llen("agent.hints.analysis") == 0

    finally:
        await client.aclose()
        await gateway.aclose()

async def test_redis_lua_delegation_artifact_provenance_validation(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis, delivery_mode="authenticated_claim")
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        parent_id, asgn_id, grant, _claim, art_ref, now = await _setup_claimed_parent(gateway)
        artifact_key = f"task.artifact:{parent_id}:{art_ref.artifact_id}:{art_ref.artifact_version}"
        canonical_artifact = json.loads(await client.get(artifact_key) or "{}")

        bad_artifact_cases = [
            ("producing_turn_id", None),
            ("producing_turn_id", ""),
            ("producing_turn_id", "turn-different-from-claim"),
            ("payload_schema_version", None),
            ("payload_schema_version", 2),
            ("version", None),
            ("version", 2),
            ("created_at", None),
            ("created_at", ""),
            ("content", None),
            ("content", "not-a-dict"),
        ]

        for field, bad_val in bad_artifact_cases:
            tampered = dict(canonical_artifact)
            if bad_val is None:
                tampered.pop(field, None)
            else:
                tampered[field] = bad_val
            await client.set(artifact_key, json.dumps(tampered))

            proposal, child_input, continuation, yield_rec, hint, child_id = _build_test_proposal(
                parent_id, asgn_id, grant, art_ref, now, prop_id=f"prop-art-{field}-{bad_val}"
            )
            with pytest.raises((ValueError, CorruptAuthoritativeRecordError)):
                await gateway.accept_delegation_proposal(
                    parent_task_id=parent_id,
                    proposal=proposal,
                    child_task_input=child_input,
                    parent_continuation=continuation,
                    yield_record=yield_rec,
                    wake_up_hint=hint,
                    expected_fencing_epoch=1,
                    now=now,
                )
            assert await client.exists(f"task.status:{child_id}") == 0
            assert await client.llen("agent.hints.analysis") == 0

    finally:
        await client.aclose()
        await gateway.aclose()

async def test_redis_lua_delegation_bound_agent_id_enforcement(clean_redis: str) -> None:
    gateway = RedisGateway(clean_redis, delivery_mode="authenticated_claim")
    client = redis.from_url(clean_redis, decode_responses=True)
    try:
        parent_id, asgn_id, grant, _claim, art_ref, now = await _setup_claimed_parent(gateway)

        # 1. Claim mismatch: claim is bound to "research-agent-1", proposal supplied for "other-agent"
        mismatched_proposal, child_input, continuation, yield_rec, hint, child_id = _build_test_proposal(
            parent_id, asgn_id, grant, art_ref, now, prop_id="prop-mismatch-1", bound_agent_id="other-agent"
        )
        with pytest.raises(ClaimNotFoundError):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=mismatched_proposal,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.exists(f"task.status:{child_id}") == 0
        assert await client.llen("agent.hints.analysis") == 0

        # 2. Matching first acceptance succeeds
        valid_proposal, child_input, continuation, yield_rec, hint, child_id = _build_test_proposal(
            parent_id, asgn_id, grant, art_ref, now, prop_id="prop-valid-1", bound_agent_id="research-agent-1"
        )
        res, is_rep = await gateway.accept_delegation_proposal(
            parent_task_id=parent_id,
            proposal=valid_proposal,
            child_task_input=child_input,
            parent_continuation=continuation,
            yield_record=yield_rec,
            wake_up_hint=hint,
            expected_fencing_epoch=1,
            now=now,
        )
        assert not is_rep
        assert res.proposal_id == "prop-valid-1"
        assert res.bound_agent_id == "research-agent-1"
        assert await client.llen("agent.hints.analysis") == 1

        # 3. Active replay with matching bound_agent_id succeeds
        replayed, is_rep2 = await gateway.accept_delegation_proposal(
            parent_task_id=parent_id,
            proposal=valid_proposal,
            child_task_input=child_input,
            parent_continuation=continuation,
            yield_record=yield_rec,
            wake_up_hint=hint,
            expected_fencing_epoch=1,
            now=now,
        )
        assert is_rep2
        assert replayed == res
        assert await client.llen("agent.hints.analysis") == 1

        # 4. Replay with conflicting bound_agent_id fails closed
        conflicting_bound = DelegationProposalRecord(
            parent_task_id=valid_proposal.parent_task_id,
            proposal_id=valid_proposal.proposal_id,
            child_task_id=valid_proposal.child_task_id,
            delegation_decision_id=valid_proposal.delegation_decision_id,
            assignment_id=valid_proposal.assignment_id,
            runtime_principal_id=valid_proposal.runtime_principal_id,
            bound_agent_id="different-replaying-agent",
            proof_digest=valid_proposal.proof_digest,
            proposal_fingerprint=valid_proposal.proposal_fingerprint,
            delegation_grant_id=valid_proposal.delegation_grant_id,
            child_claim_deadline=valid_proposal.child_claim_deadline,
            accepted_at=valid_proposal.accepted_at,
            target_capability=valid_proposal.target_capability,
            continuation_version=valid_proposal.continuation_version,
        )
        with pytest.raises(DelegationConflictError):
            await gateway.accept_delegation_proposal(
                parent_task_id=parent_id,
                proposal=conflicting_bound,
                child_task_input=child_input,
                parent_continuation=continuation,
                yield_record=yield_rec,
                wake_up_hint=hint,
                expected_fencing_epoch=1,
                now=now,
            )
        assert await client.llen("agent.hints.analysis") == 1

    finally:
        await client.aclose()
        await gateway.aclose()
