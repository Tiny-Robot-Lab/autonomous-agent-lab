"""Dedicated live Redis 7 integration tests for Issue #223.

Governed by ADR-0005, ADR-0006, and ADR-0007 (Decisions 10, 11, 12, 13):
- Dedicated to database 15 (REDIS_URL=redis://127.0.0.1:6379/15);
- Never runs FLUSHALL (flushes only DB 15);
- Tests child timeout materialization (before, at, after deadline, idempotent replay, claim trigger);
- Tests acknowledged claim execution expiry (before, after, fence invariant, replay);
- Tests recovery resolution (operator authorization, restore_claimable, fence invariant, next claim fence increment);
- Tests targeted expiry evaluation in get_result and get_status;
- Tests end-to-end live HTTP Orchestrator + real Redis DB 15 resumed Research flow.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import pytest
import redis.asyncio as redis
from agent_contracts import (
    ActiveClaim,
    CompletionCommand,
    RecoveryDecisionCommand,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    TaskResult,
    TerminalOutcome,
    derive_terminal_outcome_id,
    normalize_canonical_utc_timestamp,
)
from agent_infrastructure import OrchestratorClient
from orchestrator.application.ports import (
    CapabilityMismatchError,
    ChildClaimDeadlineExpiredError,
    ClaimDeadlineNotExpiredError,
    ClaimTimeoutConflictError,
    CompletionConflictError,
    RecoveryPreconditionFailedError,
    TaskAlreadyClaimedError,
    TaskNotClaimableError,
)
from orchestrator.application.use_cases import EvaluateExpiry, ResolveExpiredClaim
from orchestrator.config import Settings
from orchestrator.domain.principal import (
    RuntimePrincipal,
    compute_token_digest,
    derive_proof_subkeys,
    parse_master_secret,
)
from orchestrator.domain.task import CorruptAuthoritativeRecordError
from orchestrator.infrastructure.redis_adapter import RedisGateway
from orchestrator.presentation.api import create_app
from research_agent.application.processor import ResearchProcessor
from research_agent.domain.models import ResearchResult, SynthesisResult
from research_agent.infrastructure.redis_bus import RedisBus

pytestmark = [pytest.mark.anyio, pytest.mark.redis]

REQUIRED_TEST_DB = "15"
TEST_PEPPER = "test_issue_223_redis_proof_pepper_4567890123456789"


def _validate_test_redis_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("redis", "rediss"):
        raise ValueError(f"Redis integration tests require a redis:// URL, got scheme '{parsed.scheme}'")
    if parsed.path.strip("/") != REQUIRED_TEST_DB:
        raise ValueError(
            "Redis integration tests require database 15 to prevent accidental data loss, "
            f"got '{parsed.path}'"
        )


@pytest.fixture
def redis_url() -> str:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        pytest.skip("Redis integration tests disabled: REDIS_URL environment variable is not set")
    try:
        _validate_test_redis_url(url)
    except ValueError as exc:
        pytest.fail(str(exc))
    return url


@pytest.fixture
async def clean_gateway(redis_url: str) -> AsyncGenerator[RedisGateway, None]:
    gateway = RedisGateway(redis_url)
    try:
        await asyncio.wait_for(gateway._client.ping(), timeout=1.0)
    except (redis.ConnectionError, OSError) as exc:
        await gateway.aclose()
        pytest.fail(f"Configured Redis endpoint is unreachable within 1.0s timeout: {exc}")
    await gateway._client.flushdb()
    try:
        yield gateway
    finally:
        try:
            await gateway._client.flushdb()
        finally:
            await gateway.aclose()


def _research_principal() -> RuntimePrincipal:
    return RuntimePrincipal(
        principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        authorized_capabilities=("research",),
        active=True,
    )


def _analysis_principal() -> RuntimePrincipal:
    return RuntimePrincipal(
        principal_id="principal-analysis-1",
        bound_agent_id="analysis-agent-1",
        authorized_capabilities=("analysis",),
        active=True,
    )


def _recovery_principal() -> RuntimePrincipal:
    return RuntimePrincipal(
        principal_id="principal-recovery-1",
        bound_agent_id="recovery-operator-1",
        authorized_capabilities=("recovery_operator",),
        active=True,
    )


class StubResearchExecutor:
    async def execute(self, task: TaskRequested) -> ResearchResult:
        return ResearchResult(
            task_id=task.task_id,
            output="Primary stub findings on renewable energy trends",
            sources=[{"title": "Stub Energy Report", "url": "https://example.org/energy"}],
        )


class StubInferenceProvider:
    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        return SynthesisResult(
            question=question,
            content=f"Synthesized synthesis for {question}:\n{evidence}",
            evidence=evidence,
            provider="stub",
            model="stub-v1",
        )


async def _setup_root_research_task(gateway: RedisGateway, task_id: str) -> None:
    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    task = TaskRequested(
        task_id=task_id,
        root_task_id=task_id,
        parent_task_id=None,
        prompt="Comprehensive energy research",
        created_at=now_iso,
    )
    await gateway.register_root_task(task, initial_status="pending", required_capability="research")
    task_input = TaskInput(
        task_id=task_id,
        prompt="Comprehensive energy research",
        created_at=now_iso,
        input_version=1,
    )
    await gateway.save_task_input(task_input)


async def _register_child_with_deadline(
    gateway: RedisGateway,
    child_id: str,
    parent_id: str,
    deadline_iso: str,
    capability: str = "analysis",
) -> None:
    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    task = TaskRequested(
        task_id=child_id,
        root_task_id=parent_id,
        parent_task_id=parent_id,
        prompt="Analyze energy findings",
        created_at=now_iso,
    )
    await gateway.update_status(parent_id, "waiting_for_children")
    await gateway.register_child_task(task, initial_status="pending", required_capability=capability)
    await gateway._client.set(f"task.claim_deadline:{child_id}", deadline_iso)
    await gateway._client.sadd(f"task.children:{parent_id}", child_id)
    cont = TaskContinuation(
        parent_task_id=parent_id,
        child_task_id=child_id,
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
    )
    await gateway.save_continuation(cont)
    await gateway._client.set(f"task.continuation_latest:{parent_id}", 1)


async def test_redis_child_claim_timeout_before_deadline_rejects(clean_gateway: RedisGateway) -> None:
    parent_id = f"task-parent-{uuid4().hex[:8]}"
    await _setup_root_research_task(clean_gateway, parent_id)

    # Delegate child
    now = datetime.now(UTC)
    deadline_iso = normalize_canonical_utc_timestamp(now + timedelta(seconds=120))
    child_id = f"task-child-{uuid4().hex[:8]}"
    await _register_child_with_deadline(clean_gateway, child_id, parent_id, deadline_iso)

    # Attempt timeout before deadline: must raise TaskNotClaimableError
    before_now_iso = normalize_canonical_utc_timestamp(now + timedelta(seconds=10))
    with pytest.raises(TaskNotClaimableError, match="has not expired"):
        await clean_gateway.materialize_claim_timeout(child_id, before_now_iso)

    # Status must still be pending and outcome must be absent
    assert await clean_gateway.get_status(child_id) == "pending"
    assert await clean_gateway.get_terminal_outcome(child_id) is None


async def test_redis_child_claim_timeout_after_deadline_and_replay(clean_gateway: RedisGateway) -> None:
    parent_id = f"task-parent-{uuid4().hex[:8]}"
    await _setup_root_research_task(clean_gateway, parent_id)

    now = datetime.now(UTC)
    deadline_iso = normalize_canonical_utc_timestamp(now - timedelta(seconds=10))
    child_id = f"task-child-{uuid4().hex[:8]}"
    await _register_child_with_deadline(clean_gateway, child_id, parent_id, deadline_iso)

    # Materialize timeout after deadline
    after_now_iso = normalize_canonical_utc_timestamp(now)
    outcome, replayed = await clean_gateway.materialize_claim_timeout(child_id, after_now_iso)
    assert not replayed
    assert outcome.terminal_status == "claim_timed_out"
    assert outcome.task_id == child_id
    assert outcome.outcome_id.startswith("outcome-")

    # Idempotent replay: calling again returns exact same outcome with replayed=True
    outcome2, replayed2 = await clean_gateway.materialize_claim_timeout(child_id, after_now_iso)
    assert replayed2
    assert outcome2 == outcome

    # Task status is now claim_timed_out
    assert await clean_gateway.get_status(child_id) == "claim_timed_out"


async def test_redis_atomic_claim_triggers_child_claim_timeout(clean_gateway: RedisGateway) -> None:
    parent_id = f"task-parent-{uuid4().hex[:8]}"
    await _setup_root_research_task(clean_gateway, parent_id)

    now = datetime.now(UTC)
    deadline_iso = normalize_canonical_utc_timestamp(now - timedelta(seconds=30))
    child_id = f"task-child-{uuid4().hex[:8]}"
    await _register_child_with_deadline(clean_gateway, child_id, parent_id, deadline_iso)

    master = parse_master_secret(TEST_PEPPER)
    _tok_key, _ver_key = derive_proof_subkeys(master)
    proof_digest = "digest-1"

    claim = ActiveClaim(
        task_id=child_id,
        assignment_id=f"asg-{uuid4().hex[:8]}",
        runtime_principal_id="principal-analysis-1",
        bound_agent_id="analysis-agent-1",
        assigned_capability="analysis",
        execution_kind="delegated_child",
        proof_digest=proof_digest,
        fencing_epoch=1,
        delivery_expires_at=normalize_canonical_utc_timestamp(now + timedelta(seconds=30)),
        prior_claimable_status="pending",
        claim_id=f"claim-{uuid4().hex[:8]}",
        turn_id="turn-1",
        root_task_id=parent_id,
        request_id=f"req-{uuid4().hex[:8]}",
        claimed_at=normalize_canonical_utc_timestamp(now),
    )

    # Claim attempt after deadline must raise ChildClaimDeadlineExpiredError
    with pytest.raises(ChildClaimDeadlineExpiredError):
        await clean_gateway.atomic_claim_task(
            child_id, claim, now=normalize_canonical_utc_timestamp(now), is_root=False
        )

    # Targeted expiry evaluation: child outcome was materialized
    outcome = await clean_gateway.get_terminal_outcome(child_id)
    assert outcome is not None
    assert outcome.terminal_status == "claim_timed_out"


async def test_redis_acknowledged_claim_execution_expiry_and_fence_invariant(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-research-{uuid4().hex[:8]}"
    await _setup_root_research_task(clean_gateway, task_id)

    now = datetime.now(UTC)
    expires_at_iso = normalize_canonical_utc_timestamp(now + timedelta(seconds=900))

    claim = ActiveClaim(
        task_id=task_id,
        assignment_id=f"asg-{uuid4().hex[:8]}",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        proof_digest="digest-root",
        fencing_epoch=1,
        delivery_expires_at=normalize_canonical_utc_timestamp(now + timedelta(seconds=30)),
        prior_claimable_status="pending",
        claim_id=f"claim-{uuid4().hex[:8]}",
        turn_id="turn-1",
        root_task_id=task_id,
        request_id=f"req-{uuid4().hex[:8]}",
        claimed_at=normalize_canonical_utc_timestamp(now),
        state="acknowledged",
        acknowledged_at=normalize_canonical_utc_timestamp(now),
        execution_expires_at=expires_at_iso,
    )
    await clean_gateway.save_active_claim(claim)
    await clean_gateway.update_status(task_id, "working")
    await clean_gateway._client.set(f"task.fence:{task_id}", 1)

    # 1. Before expiry: transition_expired_claim rejects
    before_now_iso = normalize_canonical_utc_timestamp(now + timedelta(seconds=100))
    with pytest.raises(TaskNotClaimableError, match="has not expired"):
        await clean_gateway.transition_expired_claim(task_id, before_now_iso)

    # 2. After expiry: transitions to recovery_required, active claim deleted, fence UNCHANGED
    after_now_iso = normalize_canonical_utc_timestamp(now + timedelta(seconds=1000))
    expiry_rec, replayed = await clean_gateway.transition_expired_claim(task_id, after_now_iso)
    assert not replayed
    assert expiry_rec.task_id == task_id
    assert expiry_rec.expired_fencing_epoch == 1
    assert expiry_rec.reason_code == "EXECUTION_CLAIM_EXPIRED"

    # Status must be recovery_required
    assert await clean_gateway.get_status(task_id) == "recovery_required"
    # Active claim key deleted
    assert await clean_gateway.get_active_claim(task_id) is None
    # Fence invariant: fence must remain 1
    assert await clean_gateway.get_fencing_epoch(task_id) == 1

    # Idempotent replay:
    expiry_rec2, replayed2 = await clean_gateway.transition_expired_claim(task_id, after_now_iso)
    assert replayed2
    assert expiry_rec2 == expiry_rec
    assert await clean_gateway.get_fencing_epoch(task_id) == 1


async def test_redis_recovery_resolution_and_subsequent_pull_claim_fence_increment(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-research-{uuid4().hex[:8]}"
    await _setup_root_research_task(clean_gateway, task_id)

    now = datetime.now(UTC)
    expires_at_iso = normalize_canonical_utc_timestamp(now - timedelta(seconds=10))

    claim = ActiveClaim(
        task_id=task_id,
        assignment_id=f"asg-{uuid4().hex[:8]}",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        proof_digest="digest-root",
        fencing_epoch=1,
        delivery_expires_at=normalize_canonical_utc_timestamp(now - timedelta(seconds=20)),
        prior_claimable_status="pending",
        claim_id=f"claim-{uuid4().hex[:8]}",
        turn_id="turn-1",
        root_task_id=task_id,
        request_id=f"req-{uuid4().hex[:8]}",
        claimed_at=normalize_canonical_utc_timestamp(now),
        state="acknowledged",
        acknowledged_at=normalize_canonical_utc_timestamp(now - timedelta(seconds=25)),
        execution_expires_at=expires_at_iso,
    )
    await clean_gateway.save_active_claim(claim)
    await clean_gateway.update_status(task_id, "working")
    await clean_gateway._client.set(f"task.fence:{task_id}", 1)

    # Transition to recovery_required
    now_iso = normalize_canonical_utc_timestamp(now)
    expiry_rec, _ = await clean_gateway.transition_expired_claim(task_id, now_iso)
    assert await clean_gateway.get_status(task_id) == "recovery_required"

    # Unauthorized principal cannot resolve recovery
    resolve_uc = ResolveExpiredClaim(task_store=clean_gateway)
    command = RecoveryDecisionCommand(
        recovery_decision_id=f"rec-{uuid4().hex[:8]}",
        task_id=task_id,
        expired_claim_id=expiry_rec.claim_id,
        expired_assignment_id=expiry_rec.assignment_id,
        expired_fencing_epoch=expiry_rec.expired_fencing_epoch,
        disposition="restore_claimable",
    )
    with pytest.raises(CapabilityMismatchError):
        await resolve_uc.execute(principal=_research_principal(), command=command)

    # Conflicting recovery command fails closed
    conflict_cmd = RecoveryDecisionCommand(
        recovery_decision_id=f"rec-{uuid4().hex[:8]}",
        task_id=task_id,
        expired_claim_id="wrong-claim-id",
        expired_assignment_id=expiry_rec.assignment_id,
        expired_fencing_epoch=expiry_rec.expired_fencing_epoch,
        disposition="restore_claimable",
    )
    with pytest.raises(RecoveryPreconditionFailedError, match="EXPIRY_RECORD_MISMATCH"):
        await resolve_uc.execute(principal=_recovery_principal(), command=conflict_cmd, now=now_iso)

    # Authorized recovery_operator resolves expired claim
    rec_decision, replayed = await resolve_uc.execute(
        principal=_recovery_principal(), command=command, now=now_iso
    )
    assert not replayed
    assert rec_decision.disposition == "restore_claimable"
    assert await clean_gateway.get_status(task_id) == "pending"
    # Fence remains strictly 1 during recovery resolution
    assert await clean_gateway.get_fencing_epoch(task_id) == 1

    # Idempotent replay:
    rec_decision2, replayed2 = await resolve_uc.execute(
        principal=_recovery_principal(), command=command, now=now_iso
    )
    assert replayed2
    assert rec_decision2 == rec_decision

    # Next successful pull-claim increments fence to 2!
    next_claim = ActiveClaim(
        task_id=task_id,
        assignment_id=f"asg-next-{uuid4().hex[:8]}",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        proof_digest="digest-root-2",
        fencing_epoch=2,
        delivery_expires_at=normalize_canonical_utc_timestamp(now + timedelta(seconds=30)),
        prior_claimable_status="pending",
        claim_id=f"claim-next-{uuid4().hex[:8]}",
        turn_id="turn-2",
        root_task_id=task_id,
        request_id=f"req-next-{uuid4().hex[:8]}",
        claimed_at=now_iso,
    )
    _res_claim, replayed = await clean_gateway.atomic_claim_task(task_id, next_claim, now=now_iso)
    assert not replayed
    assert await clean_gateway.get_fencing_epoch(task_id) == 2


async def test_redis_targeted_expiry_evaluation_in_get_result_and_status(clean_gateway: RedisGateway) -> None:
    # 1. Test working task with expired execution claim
    task_id = f"task-working-exp-{uuid4().hex[:8]}"
    await _setup_root_research_task(clean_gateway, task_id)

    now = datetime.now(UTC)
    expires_at_iso = normalize_canonical_utc_timestamp(now - timedelta(seconds=10))

    claim = ActiveClaim(
        task_id=task_id,
        assignment_id=f"asg-{uuid4().hex[:8]}",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        proof_digest="digest-root",
        fencing_epoch=1,
        delivery_expires_at=normalize_canonical_utc_timestamp(now - timedelta(seconds=20)),
        prior_claimable_status="pending",
        claim_id=f"claim-{uuid4().hex[:8]}",
        turn_id="turn-1",
        root_task_id=task_id,
        request_id=f"req-{uuid4().hex[:8]}",
        claimed_at=normalize_canonical_utc_timestamp(now),
        state="acknowledged",
        acknowledged_at=normalize_canonical_utc_timestamp(now - timedelta(seconds=25)),
        execution_expires_at=expires_at_iso,
    )
    await clean_gateway.save_active_claim(claim)
    await clean_gateway.update_status(task_id, "working")
    await clean_gateway._client.set(f"task.fence:{task_id}", 1)

    # get_status triggers transition to recovery_required
    status = await clean_gateway.get_status(task_id)
    assert status == "recovery_required"

    # get_result also reflects recovery_required
    result_data = await clean_gateway.get_result(task_id)
    assert result_data is not None
    assert result_data["status"] == "recovery_required"

    # 2. Test pending child with expired claim deadline
    parent_target = f"task-parent-exp-{uuid4().hex[:8]}"
    await _setup_root_research_task(clean_gateway, parent_target)
    child_id = f"task-child-exp-{uuid4().hex[:8]}"
    deadline_iso = normalize_canonical_utc_timestamp(now - timedelta(seconds=5))
    await _register_child_with_deadline(clean_gateway, child_id, parent_target, deadline_iso)

    # get_result on expired pending child triggers materialize_claim_timeout
    child_res = await clean_gateway.get_result(child_id)
    assert child_res is not None
    assert child_res["status"] == "claim_timed_out"
    assert await clean_gateway.get_status(child_id) == "claim_timed_out"


async def test_redis_resumed_research_e2e_flow(clean_gateway: RedisGateway, redis_url: str) -> None:
    """Full end-to-end integration: Live FastAPI Orchestrator + real Redis DB 15 + OrchestratorClient."""
    cred_pepper = "test_issue_223_redis_cred_pepper_1234567890123456"
    research_token = "token-research-1"
    analysis_token = "token-analysis-1"
    settings = Settings(
        redis_url=redis_url,
        task_delivery_mode="authenticated_claim",
        runtime_credential_pepper=cred_pepper,
        assignment_proof_pepper=TEST_PEPPER,
        claim_delivery_lease_seconds=30.0,
        child_claim_deadline_seconds=120.0,
        execution_claim_timeout_seconds=900.0,
        runtime_principals={
            "principal-research-1": {
                "principal_id": "principal-research-1",
                "bound_agent_id": "research-agent-1",
                "authorized_capabilities": ["research"],
                "token_digest": compute_token_digest(research_token, cred_pepper),
                "active": True,
            },
            "principal-analysis-1": {
                "principal_id": "principal-analysis-1",
                "bound_agent_id": "analysis-agent-1",
                "authorized_capabilities": ["analysis"],
                "token_digest": compute_token_digest(analysis_token, cred_pepper),
                "active": True,
            },
        },
    )
    app = create_app(settings=settings, gateway=clean_gateway)
    transport = httpx.ASGITransport(app=app)

    orchestrator_research = OrchestratorClient(
        base_url="http://test",
        runtime_token=research_token,
        transport=transport,
    )
    orchestrator_analysis = OrchestratorClient(
        base_url="http://test",
        runtime_token=analysis_token,
        transport=transport,
    )

    # 1. Create root task
    parent_id = f"task-root-{uuid4().hex[:8]}"
    await _setup_root_research_task(clean_gateway, parent_id)

    # 2. Initial claim by Research Agent
    req_id = f"req-{uuid4().hex[:8]}"
    asg, proof = await orchestrator_research.claim_task(
        task_id=parent_id,
        request_id=req_id,
        capability="research",
        execution_kind="initial",
    )
    await orchestrator_research.acknowledge_assignment(parent_id, asg["assignment_id"], proof)

    # 3. Research Agent produces StageArtifact and delegates to Analysis child
    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    artifact = StageArtifact(
        artifact_id=f"art-{uuid4().hex[:8]}",
        producing_task_id=parent_id,
        artifact_type="research_material",
        content={"material": "Detailed primary research evidence on energy grids."},
        created_at=now_iso,
        sealed=True,
    )
    # Stage artifact via API
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/tasks/{parent_id}/artifacts",
            json={
                "assignment_id": asg["assignment_id"],
                "assignment_proof": proof,
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "content": dict(artifact.content),
                "artifact_version": artifact.artifact_version,
                "payload_schema_version": artifact.payload_schema_version,
            },
            headers={"Authorization": f"Bearer {research_token}", "X-Assignment-Proof": proof},
        )
        assert resp.status_code == 200

        # Accept delegation proposal
        proposal_payload = {
            "proposal_id": f"prop-{uuid4().hex[:8]}",
            "assignment_id": asg["assignment_id"],
            "child_intent": "Analyze grid storage capacity",
            "artifact_reference": {
                "artifact_id": artifact.artifact_id,
                "producing_task_id": parent_id,
                "artifact_version": 1,
            },
            "target_capability": "analysis",
            "parent_input_version": 1,
            "control_fields": {
                "analysis_focus": "storage",
                "material_summary": "Grid evidence",
            },
        }
        del_resp = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=proposal_payload,
            headers={"Authorization": f"Bearer {research_token}", "X-Assignment-Proof": proof},
        )
        assert del_resp.status_code == 200
        del_data = del_resp.json()
        child_id = del_data["child_task_id"]

    # Parent status is now waiting_for_children
    assert await clean_gateway.get_status(parent_id) == "waiting_for_children"

    # 4. Analysis child pulls claim, acknowledges, and completes turn
    asg_child, proof_child = await orchestrator_analysis.claim_task(
        task_id=child_id,
        request_id=f"req-{uuid4().hex[:8]}",
        capability="analysis",
        execution_kind="delegated_child",
    )
    await orchestrator_analysis.acknowledge_assignment(
        child_id, asg_child["assignment_id"], proof_child
    )

    child_command = CompletionCommand(
        completion_command_id=f"comp-{uuid4().hex[:8]}",
        assignment_id=asg_child["assignment_id"],
        execution_kind="delegated_child",
        completed_result=TaskResult(
            task_id=child_id,
            output="Specialist findings: battery storage capacity is 85% viable.",
        ),
    )
    comp_resp = await orchestrator_analysis.submit_completion(child_id, child_command, proof_child)
    assert comp_resp["outcome"]["terminal_status"] == "completed"

    # Parent status is now resumable!
    assert await clean_gateway.get_status(parent_id) == "resumable"

    # 5. Non-secret wake-up hint was written to agent.hints.research
    raw_hint = await clean_gateway._client.rpop("agent.hints.research")
    assert raw_hint is not None

    # 6. Research agent pull-claims resumed parent turn!
    asg_resumed, proof_resumed = await orchestrator_research.claim_task(
        task_id=parent_id,
        request_id=f"req-resumed-{uuid4().hex[:8]}",
        capability="research",
        execution_kind="resumed_continuation",
        expected_continuation_version=1,
    )
    await orchestrator_research.acknowledge_assignment(
        parent_id, asg_resumed["assignment_id"], proof_resumed
    )

    # Fencing epoch has incremented to 2
    assert asg_resumed["fencing_epoch"] == 2
    assert await clean_gateway.get_fencing_epoch(parent_id) == 2

    # 7. Read governed continuation context
    ctx = await orchestrator_research.get_resumed_continuation_context(
        task_id=parent_id,
        assignment_id=asg_resumed["assignment_id"],
        assignment_proof=proof_resumed,
    )
    assert ctx.parent_task_input.task_id == parent_id
    assert ctx.continuation.stage == "incorporate_analysis_child"
    assert ctx.stage_artifact.sealed is True
    assert ctx.child_outcome.terminal_status == "completed"
    assert ctx.child_result is not None
    assert "battery storage capacity is 85% viable" in ctx.child_result.output

    # 8. Run ResearchProcessor for resumed turn
    bus = RedisBus(redis_url)
    executor = StubResearchExecutor()
    provider = StubInferenceProvider()
    processor = ResearchProcessor(
        agent_id="research-agent-1",
        bus=bus,
        executor=executor,
        completion_client=orchestrator_research,
        context_reader=orchestrator_research,
        inference_provider=provider,
    )
    task_req = TaskRequested(
        task_id=parent_id,
        root_task_id=parent_id,
        parent_task_id=None,
        prompt=ctx.parent_task_input.prompt,
        created_at=now_iso,
    )
    await processor.run_once(
        task=task_req,
        assignment_id=asg_resumed["assignment_id"],
        assignment_proof=proof_resumed,
        execution_kind="resumed_continuation",
        continuation_version=1,
    )

    # 9. Verify parent task completed authoritatively
    outcome = await clean_gateway.get_terminal_outcome(parent_id)
    assert outcome is not None
    assert outcome.terminal_status == "completed"
    result = await clean_gateway.get_canonical_result(parent_id)
    assert result is not None
    assert "Synthesized synthesis" in result.output
    assert "battery storage capacity is 85% viable" in result.output
    assert await clean_gateway.get_status(parent_id) == "completed"


async def test_redis_wrong_key_types_and_corrupt_json_propagate_corruption(
    clean_gateway: RedisGateway,
) -> None:
    # 1. Wrong key type (set instead of string) at task.status
    await clean_gateway._client.sadd("task.status:bad-st-type", "dummy")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.get_status("bad-st-type")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.get_result("bad-st-type")

    # 2. Corrupt non-JSON payload at task.status
    await clean_gateway._client.set("task.status:bad-st-json", "{malformed-json")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.get_status("bad-st-json")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.get_result("bad-st-json")

    # 3. Wrong key type at task.claim_deadline
    await clean_gateway._client.set("task.status:bad-dl-type", '{"status": "pending", "task_id": "bad-dl-type"}')
    await clean_gateway._client.sadd("task.claim_deadline:bad-dl-type", "dummy")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.get_status("bad-dl-type")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.get_result("bad-dl-type")

    # 4. Wrong key type at task.claim
    await clean_gateway._client.set("task.status:bad-cl-type", '{"status": "working", "task_id": "bad-cl-type"}')
    await clean_gateway._client.sadd("task.claim:bad-cl-type", "dummy")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.get_status("bad-cl-type")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.get_result("bad-cl-type")

    # 5. Corrupt non-JSON payload at task.claim
    await clean_gateway._client.set("task.claim:bad-cl-json", "not-a-valid-claim-json")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.get_active_claim("bad-cl-json")


async def test_redis_expiry_exact_boundaries_and_zero_mutation_on_rejection(
    clean_gateway: RedisGateway,
) -> None:
    # Set up child task lineage and deadline
    parent_id = f"parent-bound-{uuid4().hex[:8]}"
    child_id = f"child-bound-{uuid4().hex[:8]}"
    deadline_iso = "2026-11-22T12:00:00Z"

    await _setup_root_research_task(clean_gateway, parent_id)
    await _register_child_with_deadline(clean_gateway, child_id, parent_id, deadline_iso)

    # 1. Before deadline: now < deadline -> rejected with ClaimDeadlineNotExpiredError
    before_now = "2026-11-22T11:59:59Z"
    with pytest.raises(ClaimDeadlineNotExpiredError, match="has not expired"):
        await clean_gateway.materialize_claim_timeout(child_id, now=before_now)
    # Zero mutation: outcome must not exist, hints must not exist
    assert await clean_gateway._client.exists(f"task.outcome:{child_id}") == 0
    assert await clean_gateway._client.llen("agent.hints.research") == 0

    # 2. Exact deadline: now == deadline -> rejected with ClaimDeadlineNotExpiredError
    with pytest.raises(ClaimDeadlineNotExpiredError, match="has not expired"):
        await clean_gateway.materialize_claim_timeout(child_id, now=deadline_iso)
    assert await clean_gateway._client.exists(f"task.outcome:{child_id}") == 0
    assert await clean_gateway._client.llen("agent.hints.research") == 0

    # 3. Acknowledged claim expiry boundary: now == execution_expires_at -> rejected
    task_id = "task-exp-bound-1"
    expires_at = "2026-11-22T12:00:00Z"
    claim = ActiveClaim(
        task_id=task_id,
        assignment_id="asgn-b-1",
        claim_id="claim-b-1",
        root_task_id=task_id,
        turn_id="turn-b-1",
        runtime_principal_id="p-1",
        bound_agent_id="a-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="a" * 64,
        request_id="req-b-1",
        claimed_at="2026-11-22T11:45:00Z",
        delivery_expires_at="2026-11-22T11:46:00Z",
        state="acknowledged",
        acknowledged_at="2026-11-22T11:45:05Z",
        execution_expires_at=expires_at,
    )
    await clean_gateway.save_active_claim(claim)
    await clean_gateway._client.set(
        f"task.status:{task_id}", '{"status": "working", "task_id": "' + task_id + '"}'
    )
    await clean_gateway._client.set(f"task.fence:{task_id}", "1")

    with pytest.raises(TaskNotClaimableError, match="has not expired"):
        await clean_gateway.transition_expired_claim(task_id, now=expires_at)
    assert await clean_gateway._client.exists(f"task.claim_expiry:{task_id}") == 0
    assert await clean_gateway.get_status(task_id, now=expires_at) == "working"


async def test_redis_concurrent_analysis_claim_versus_timeout(
    clean_gateway: RedisGateway,
) -> None:
    parent_id = f"parent-conc-{uuid4().hex[:8]}"
    child_id = f"child-conc-{uuid4().hex[:8]}"
    deadline_iso = "2026-11-22T12:00:00Z"
    after_iso = "2026-11-22T12:05:00Z"

    await _setup_root_research_task(clean_gateway, parent_id)
    await _register_child_with_deadline(clean_gateway, child_id, parent_id, deadline_iso)
    # Child was already claimed: update status to "working"
    await clean_gateway.update_status(child_id, "working")

    # Calling materialize_claim_timeout must fail because child is not pending
    with pytest.raises(TaskNotClaimableError, match="not in 'pending' status"):
        await clean_gateway.materialize_claim_timeout(child_id, now=after_iso)

    assert await clean_gateway._client.exists(f"task.outcome:{child_id}") == 0
    assert await clean_gateway._client.llen("agent.hints.research") == 0


async def test_redis_concurrent_completion_versus_expiry(
    clean_gateway: RedisGateway,
) -> None:
    task_id = "task-conc-comp-1"
    expires_at = "2026-11-22T12:00:00Z"
    after_iso = "2026-11-22T12:05:00Z"

    claim = ActiveClaim(
        task_id=task_id,
        assignment_id="asgn-cc-1",
        claim_id="claim-cc-1",
        root_task_id=task_id,
        turn_id="turn-cc-1",
        runtime_principal_id="p-1",
        bound_agent_id="a-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="a" * 64,
        request_id="req-cc-1",
        claimed_at="2026-11-22T11:45:00Z",
        delivery_expires_at="2026-11-22T11:46:00Z",
        state="acknowledged",
        acknowledged_at="2026-11-22T11:45:05Z",
        execution_expires_at=expires_at,
    )
    await clean_gateway.save_active_claim(claim)
    await clean_gateway._client.set(f"task.fence:{task_id}", "1")
    await clean_gateway._client.set(
        f"task.status:{task_id}", '{"status": "working", "task_id": "' + task_id + '"}'
    )

    # Task completes right as expiry is checked: canonical outcome recorded
    fp = "a" * 64
    outcome = TerminalOutcome(
        outcome_id=derive_terminal_outcome_id(fp),
        task_id=task_id,
        assignment_id="asgn-cc-1",
        completion_command_id="cmd-cc-1",
        semantic_fingerprint=fp,
        runtime_principal_id="p-1",
        bound_agent_id="a-1",
        execution_kind="initial",
        terminal_status="completed",
        safe_summary="Task completed successfully.",
        completed_at=after_iso,
    )
    await clean_gateway._client.set(f"task.outcome:{task_id}", outcome.to_json())

    # Expiry transition must detect outcome conflict and raise CompletionConflictError
    with pytest.raises(CompletionConflictError):
        await clean_gateway.transition_expired_claim(task_id, now=after_iso)

    assert await clean_gateway._client.exists(f"task.claim_expiry:{task_id}") == 0


async def test_redis_legacy_acknowledged_claim_unchanged(
    clean_gateway: RedisGateway,
) -> None:
    task_id = "task-legacy-1"
    now_iso = "2026-11-22T12:00:00Z"

    # Legacy claim with execution_expires_at=None
    legacy_claim = ActiveClaim(
        task_id=task_id,
        assignment_id="asgn-leg-1",
        claim_id="claim-leg-1",
        root_task_id=task_id,
        turn_id="turn-leg-1",
        runtime_principal_id="p-1",
        bound_agent_id="a-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="a" * 64,
        request_id="req-leg-1",
        claimed_at=now_iso,
        delivery_expires_at=now_iso,
        state="acknowledged",
        acknowledged_at=now_iso,
        execution_expires_at=None,
    )
    await clean_gateway.save_active_claim(legacy_claim)
    await clean_gateway._client.set(
        f"task.status:{task_id}", '{"status": "working", "task_id": "' + task_id + '"}'
    )
    await clean_gateway._client.set(f"task.fence:{task_id}", "1")

    # 1. get_status returns "working", leaves status unchanged
    st = await clean_gateway.get_status(task_id)
    assert st == "working"

    # 2. get_result returns working status
    res = await clean_gateway.get_result(task_id)
    assert res == {"task_id": task_id, "status": "working", "output": None}

    # 3. EvaluateExpiry use case returns diagnostic action="none", status="working"
    eval_uc = EvaluateExpiry(clean_gateway)
    operator = RuntimePrincipal(
        principal_id="p-op",
        bound_agent_id="a-op",
        authorized_capabilities=("recovery_operator",),
        active=True,
    )
    eval_res = await eval_uc.execute(operator, task_id, now=now_iso)
    assert eval_res.action == "none"
    assert eval_res.status == "working"
    assert eval_res.expiry_record is None

    # Authoritative status in Redis remains working
    assert await clean_gateway.get_status(task_id) == "working"


async def test_redis_queue_fairness_interleaving(
    clean_gateway: RedisGateway,
    redis_url: str,
) -> None:
    from agent_contracts import WakeUpHint

    queue_name = f"fair.tasks.{uuid4().hex[:6]}"
    hint_queue = f"fair.hints.{uuid4().hex[:6]}"

    bus = RedisBus(redis_url, queue_name=queue_name, hint_queue=hint_queue)

    try:
        # Push 3 hints and 3 tasks
        for i in range(3):
            hint = WakeUpHint(
                hint_id=f"hint-{i}",
                capability="research",
                created_at="2026-11-22T12:00:00Z",
                execution_kind="resumed_continuation",
                task_id=f"parent-{i}",
                expected_continuation_version=1,
            )
            await clean_gateway._client.rpush(hint_queue, hint.to_json())

            task = TaskRequested(
                task_id=f"task-{i}",
                root_task_id=f"task-{i}",
                parent_task_id=None,
                prompt=f"prompt-{i}",
                created_at="2026-11-22T12:00:00Z",
            )
            await clean_gateway._client.rpush(queue_name, task.to_json())

        # Read 6 signals: bus must alternate between hints and tasks
        signals = []
        for _ in range(6):
            sig = await bus.next_signal()
            signals.append(sig)

        types = [type(s).__name__ for s in signals]
        # Must be alternating: WakeUpHint, TaskRequested, WakeUpHint, TaskRequested, ...
        assert types == [
            "WakeUpHint",
            "TaskRequested",
            "WakeUpHint",
            "TaskRequested",
            "WakeUpHint",
            "TaskRequested",
        ]
    finally:
        await bus.aclose()
        await clean_gateway._client.delete(queue_name, hint_queue)


def test_redis_stream_bound_maxlen_in_expiry_lua() -> None:
    from orchestrator.infrastructure.redis_adapter import TRANSITION_EXPIRED_CLAIM_LUA

    assert "'XADD', events_key, 'MAXLEN', '~', '500', '*'" in TRANSITION_EXPIRED_CLAIM_LUA


async def test_redis_lua_acknowledged_expiry_exact_equality(
    clean_gateway: RedisGateway,
) -> None:
    task_id = "task-eq-lua-1"
    now_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    expires_iso = normalize_canonical_utc_timestamp(now_dt)
    before_iso = normalize_canonical_utc_timestamp(now_dt - timedelta(seconds=1))
    equal_iso = expires_iso
    after_iso = normalize_canonical_utc_timestamp(now_dt + timedelta(seconds=1))

    # Setup working task with claim expiring at expires_iso
    await clean_gateway.register_root_task(
        TaskRequested(
            task_id=task_id,
            root_task_id=task_id,
            parent_task_id=None,
            prompt="Task for equality test",
            created_at=before_iso,
        ),
        required_capability="research",
    )
    claim = ActiveClaim(
        task_id=task_id,
        assignment_id="asg-eq-1",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        proof_digest="digest-eq",
        fencing_epoch=1,
        delivery_expires_at=expires_iso,
        prior_claimable_status="pending",
        claim_id="claim-eq-1",
        turn_id="turn-1",
        root_task_id=task_id,
        request_id="req-eq-1",
        claimed_at=before_iso,
        state="acknowledged",
        acknowledged_at=before_iso,
        execution_expires_at=expires_iso,
    )
    await clean_gateway.save_active_claim(claim)
    await clean_gateway.update_status(task_id, "working")
    await clean_gateway._client.set(f"task.fence:{task_id}", 1)

    # 1. At exact equality (now == expires_iso), transition_expired_claim fails with TaskNotClaimableError
    with pytest.raises(TaskNotClaimableError, match="has not expired"):
        await clean_gateway.transition_expired_claim(task_id, equal_iso)

    # Status remains working directly and through EvaluateExpiry at equality
    assert await clean_gateway.get_status(task_id, now=equal_iso) == "working"
    eval_exp = EvaluateExpiry(clean_gateway)
    rec_operator = RuntimePrincipal(
        principal_id="rec-op",
        bound_agent_id="operator-1",
        authorized_capabilities=("recovery_operator",),
    )
    res_eq = await eval_exp.execute(rec_operator, task_id, now=equal_iso)
    assert res_eq.action == "none"
    assert res_eq.status == "working"

    # Replay claim by same principal at equality returns ALREADY_ACKNOWLEDGED, NOT expired
    with pytest.raises(TaskAlreadyClaimedError, match="already acknowledged"):
        await clean_gateway.atomic_claim_task(task_id, claim, now=equal_iso, is_root=True)

    # 2. Strictly after (now > expires_iso), expiry succeeds
    rec, is_replay = await clean_gateway.transition_expired_claim(task_id, after_iso)
    assert is_replay is False
    assert rec.task_id == task_id
    assert await clean_gateway.get_status(task_id) == "recovery_required"


async def test_redis_timeout_corruption_and_replay_hardening(
    clean_gateway: RedisGateway,
) -> None:
    parent_id = "parent-harden-1"
    child_id = "child-harden-1"
    now_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    now_iso = normalize_canonical_utc_timestamp(now_dt)
    past_iso = normalize_canonical_utc_timestamp(now_dt - timedelta(minutes=5))

    # Helper to setup base valid state
    async def _setup_valid():
        await clean_gateway._client.flushdb()
        await clean_gateway._client.set(f"task.status:{parent_id}", '{"status":"waiting_for_children"}')
        await clean_gateway._client.sadd(f"task.children:{parent_id}", child_id)
        cont = TaskContinuation(
            parent_task_id=parent_id,
            child_task_id=child_id,
            created_at=now_iso,
            updated_at=now_iso,
            stage="incorporate_analysis_child",
            continuation_version=1,
        )
        await clean_gateway.save_continuation(cont)
        await clean_gateway._client.set(
            f"task.lineage:{child_id}",
            f'{{"task_id":"{child_id}","root_task_id":"{parent_id}","parent_task_id":"{parent_id}"}}',
        )
        await clean_gateway._client.set(f"task.status:{child_id}", f'{{"status":"pending","claim_deadline":"{past_iso}"}}')
        await clean_gateway._client.set(f"task.claim_deadline:{child_id}", past_iso)

    # 1. Corrupt lineage JSON -> CorruptAuthoritativeRecordError, zero mutation
    await _setup_valid()
    await clean_gateway._client.set(f"task.lineage:{child_id}", "NOT_JSON{{{")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.get(f"task.outcome:{child_id}") is None
    assert await clean_gateway._client.llen("agent.hints.research") == 0

    # 2. Continuation parent mismatch -> TaskNotClaimableError, zero mutation
    await _setup_valid()
    bad_cont = TaskContinuation(
        parent_task_id="wrong-parent",
        child_task_id=child_id,
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
    )
    await clean_gateway._client.set(f"task.continuation:{parent_id}:1", bad_cont.to_json())
    with pytest.raises(TaskNotClaimableError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.get(f"task.outcome:{child_id}") is None
    assert await clean_gateway._client.llen("agent.hints.research") == 0

    # 3. Continuation version mismatch -> TaskNotClaimableError, zero mutation
    await _setup_valid()
    ver_cont = TaskContinuation(
        parent_task_id=parent_id,
        child_task_id=child_id,
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=2,
    )
    await clean_gateway._client.set(f"task.continuation:{parent_id}:1", ver_cont.to_json())
    with pytest.raises(TaskNotClaimableError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.get(f"task.outcome:{child_id}") is None
    assert await clean_gateway._client.llen("agent.hints.research") == 0

    # 4. Valid initial materialization + valid exact replay
    await _setup_valid()
    outcome, is_rep = await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert is_rep is False
    assert outcome.task_id == child_id
    assert outcome.terminal_status == "claim_timed_out"
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # Exact replay succeeds read-only with zero additional hints
    outcome_rep, is_rep2 = await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert is_rep2 is True
    assert outcome_rep.outcome_id == outcome.outcome_id
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # Snapshot valid marker
    valid_marker_raw = await clean_gateway._client.get(f"task.child_completion:{child_id}")
    assert valid_marker_raw is not None
    marker_data = json.loads(valid_marker_raw)
    assert set(marker_data.keys()) == {
        "child_task_id",
        "parent_task_id",
        "outcome_id",
        "continuation_version",
        "status",
    }
    assert marker_data["child_task_id"] == child_id
    assert marker_data["parent_task_id"] == parent_id
    assert marker_data["outcome_id"] == outcome.outcome_id
    assert marker_data["continuation_version"] == 1
    assert marker_data["status"] == "claim_timed_out"

    # 5. Missing projection marker -> ClaimTimeoutConflictError
    await clean_gateway._client.delete(f"task.child_completion:{child_id}")
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 5a. Missing parent_task_id in marker -> ClaimTimeoutConflictError
    bad_marker = dict(marker_data)
    del bad_marker["parent_task_id"]
    await clean_gateway._client.set(f"task.child_completion:{child_id}", json.dumps(bad_marker))
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 5b. Missing outcome_id in marker -> ClaimTimeoutConflictError
    bad_marker = dict(marker_data)
    del bad_marker["outcome_id"]
    await clean_gateway._client.set(f"task.child_completion:{child_id}", json.dumps(bad_marker))
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 5c. Missing continuation_version in marker -> ClaimTimeoutConflictError
    bad_marker = dict(marker_data)
    del bad_marker["continuation_version"]
    await clean_gateway._client.set(f"task.child_completion:{child_id}", json.dumps(bad_marker))
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 5d. Permissive alias task_id instead of child_task_id -> ClaimTimeoutConflictError
    bad_marker = dict(marker_data)
    del bad_marker["child_task_id"]
    bad_marker["task_id"] = child_id
    await clean_gateway._client.set(f"task.child_completion:{child_id}", json.dumps(bad_marker))
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 5e. Permissive alias terminal_status instead of status -> ClaimTimeoutConflictError
    bad_marker = dict(marker_data)
    del bad_marker["status"]
    bad_marker["terminal_status"] = "claim_timed_out"
    await clean_gateway._client.set(f"task.child_completion:{child_id}", json.dumps(bad_marker))
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 6. Malformed projection marker -> CorruptAuthoritativeRecordError
    await clean_gateway._client.set(f"task.child_completion:{child_id}", "NOT_JSON{{{")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 7. Wrong parent in marker -> ClaimTimeoutConflictError
    await clean_gateway._client.set(
        f"task.child_completion:{child_id}",
        json.dumps({
            "child_task_id": child_id,
            "parent_task_id": "wrong-parent",
            "outcome_id": outcome.outcome_id,
            "status": "claim_timed_out",
            "continuation_version": 1,
        }),
    )
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 8. Wrong outcome in marker -> ClaimTimeoutConflictError
    await clean_gateway._client.set(
        f"task.child_completion:{child_id}",
        json.dumps({
            "child_task_id": child_id,
            "parent_task_id": parent_id,
            "outcome_id": "wrong-outcome-id",
            "status": "claim_timed_out",
            "continuation_version": 1,
        }),
    )
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # Restore valid marker
    await clean_gateway._client.set(f"task.child_completion:{child_id}", valid_marker_raw)

    # 9. Missing child membership in parent children set -> TaskNotClaimableError
    await clean_gateway._client.srem(f"task.children:{parent_id}", child_id)
    with pytest.raises(TaskNotClaimableError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1
    await clean_gateway._client.sadd(f"task.children:{parent_id}", child_id)

    # 10. Corrupt/mismatched lineage on replay -> CorruptAuthoritativeRecordError / TaskNotClaimableError
    await clean_gateway._client.set(f"task.lineage:{child_id}", "NOT_JSON{{{")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1
    await clean_gateway._client.set(
        f"task.lineage:{child_id}",
        f'{{"task_id":"{child_id}","root_task_id":"{parent_id}","parent_task_id":"other-parent"}}',
    )
    with pytest.raises(TaskNotClaimableError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1
    # Restore valid lineage
    await clean_gateway._client.set(
        f"task.lineage:{child_id}",
        f'{{"task_id":"{child_id}","root_task_id":"{parent_id}","parent_task_id":"{parent_id}"}}',
    )

    # 11. Missing continuation on replay -> TaskNotClaimableError
    await clean_gateway._client.delete(f"task.continuation:{parent_id}:1")
    with pytest.raises(TaskNotClaimableError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 12. Wrong continuation parent on replay -> TaskNotClaimableError
    bad_parent_cont = TaskContinuation(
        parent_task_id="other-parent",
        child_task_id=child_id,
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
    )
    await clean_gateway._client.set(f"task.continuation:{parent_id}:1", bad_parent_cont.to_json())
    with pytest.raises(TaskNotClaimableError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 13. Wrong continuation version / latest on replay -> TaskNotClaimableError
    good_cont = TaskContinuation(
        parent_task_id=parent_id,
        child_task_id=child_id,
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
    )
    await clean_gateway._client.set(f"task.continuation:{parent_id}:1", good_cont.to_json())
    await clean_gateway._client.set(f"task.continuation_latest:{parent_id}", "2")
    with pytest.raises(TaskNotClaimableError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1
    await clean_gateway._client.set(f"task.continuation_latest:{parent_id}", "1")

    # 14. Parent still waiting_for_children on replay -> ClaimTimeoutConflictError
    await clean_gateway._client.set(f"task.status:{parent_id}", '{"status":"waiting_for_children"}')
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1

    # 15. Parent in an unrelated state on replay -> ClaimTimeoutConflictError
    await clean_gateway._client.set(f"task.status:{parent_id}", '{"status":"failed"}')
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1
    # Restore parent status to resumable
    await clean_gateway._client.set(f"task.status:{parent_id}", '{"status":"resumable"}')

    # 16. Unexpected child TaskResult on replay -> ClaimTimeoutConflictError
    await clean_gateway._client.set(f"task.result:{child_id}", '{"task_id":"' + child_id + '","status":"completed"}')
    with pytest.raises(ClaimTimeoutConflictError):
        await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert await clean_gateway._client.llen("agent.hints.research") == 1
    await clean_gateway._client.delete(f"task.result:{child_id}")

    # Replay succeeds again once all invariants are clean
    re_rep, is_rep3 = await clean_gateway.materialize_claim_timeout(child_id, now_iso)
    assert is_rep3 is True
    assert re_rep.outcome_id == outcome.outcome_id
    assert await clean_gateway._client.llen("agent.hints.research") == 1


async def test_redis_evaluate_expiry_child_deadline(
    clean_gateway: RedisGateway,
) -> None:
    parent_id = f"parent-eval-{uuid4().hex[:8]}"
    child_id = f"child-eval-{uuid4().hex[:8]}"
    t_deadline = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    deadline_iso = normalize_canonical_utc_timestamp(t_deadline)
    before_now = normalize_canonical_utc_timestamp(t_deadline - timedelta(seconds=1))
    equal_now = deadline_iso
    after_now = normalize_canonical_utc_timestamp(t_deadline + timedelta(seconds=1))

    await _setup_root_research_task(clean_gateway, parent_id)
    await _register_child_with_deadline(clean_gateway, child_id, parent_id, deadline_iso)

    eval_uc = EvaluateExpiry(clean_gateway)
    operator = RuntimePrincipal(
        principal_id="rec-op",
        bound_agent_id="op-1",
        authorized_capabilities=("recovery_operator",),
    )

    # 1. Before deadline -> action="none", status="pending", zero mutations, zero hints
    res_b = await eval_uc.execute(operator, child_id, now=before_now)
    assert res_b.action == "none"
    assert res_b.status == "pending"
    assert await clean_gateway.get_status(child_id, now=before_now) == "pending"
    assert await clean_gateway.get_status(parent_id, now=before_now) == "waiting_for_children"
    assert await clean_gateway._client.exists(f"task.outcome:{child_id}") == 0
    assert await clean_gateway._client.llen("agent.hints.research") == 0

    # 2. At exact deadline -> action="none", status="pending", zero mutations, zero hints
    res_e = await eval_uc.execute(operator, child_id, now=equal_now)
    assert res_e.action == "none"
    assert res_e.status == "pending"
    assert await clean_gateway.get_status(child_id, now=equal_now) == "pending"
    assert await clean_gateway.get_status(parent_id, now=equal_now) == "waiting_for_children"
    assert await clean_gateway._client.exists(f"task.outcome:{child_id}") == 0
    assert await clean_gateway._client.llen("agent.hints.research") == 0

    # 3. After deadline -> action="claim_timed_out", status="claim_timed_out", mutations occur
    res_a = await eval_uc.execute(operator, child_id, now=after_now)
    assert res_a.action == "claim_timed_out"
    assert res_a.status == "claim_timed_out"
    assert await clean_gateway.get_status(child_id) == "claim_timed_out"
    assert await clean_gateway.get_status(parent_id) == "resumable"
    assert await clean_gateway._client.exists(f"task.outcome:{child_id}") == 1
    assert await clean_gateway._client.llen("agent.hints.research") == 1


async def test_redis_race_pull_claim_vs_child_timeout(
    redis_url: str,
) -> None:
    gw1 = RedisGateway(redis_url)
    gw2 = RedisGateway(redis_url)
    try:
        parent_id = f"parent-race-{uuid4().hex[:8]}"
        child_id = f"child-race-{uuid4().hex[:8]}"
        now_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
        now_iso = normalize_canonical_utc_timestamp(now_dt)
        past_iso = normalize_canonical_utc_timestamp(now_dt - timedelta(seconds=1))

        await gw1._client.set(f"task.status:{parent_id}", '{"status":"waiting_for_children"}')
        await gw1._client.sadd(f"task.children:{parent_id}", child_id)
        cont = TaskContinuation(
            parent_task_id=parent_id,
            child_task_id=child_id,
            created_at=now_iso,
            updated_at=now_iso,
            stage="incorporate_analysis_child",
            continuation_version=1,
        )
        await gw1.save_continuation(cont)
        await gw1._client.set(
            f"task.lineage:{child_id}",
            f'{{"task_id":"{child_id}","root_task_id":"{parent_id}","parent_task_id":"{parent_id}"}}',
        )
        await gw1._client.set(f"task.status:{child_id}", f'{{"status":"pending","claim_deadline":"{past_iso}"}}')
        await gw1._client.set(f"task.claim_deadline:{child_id}", past_iso)
        await gw1._client.set(f"task.required_capability:{child_id}", '{"required_capability":"analysis"}')
        await gw1._client.set(f"task.fence:{child_id}", "1")

        child_claim = ActiveClaim(
            task_id=child_id,
            assignment_id=f"asg-c-{uuid4().hex[:8]}",
            runtime_principal_id="principal-analysis-1",
            bound_agent_id="analysis-agent-1",
            assigned_capability="analysis",
            execution_kind="delegated_child",
            proof_digest="digest-child",
            fencing_epoch=1,
            delivery_expires_at=normalize_canonical_utc_timestamp(now_dt + timedelta(seconds=30)),
            prior_claimable_status="pending",
            claim_id=f"claim-c-{uuid4().hex[:8]}",
            turn_id="turn-1",
            root_task_id=parent_id,
            request_id=f"req-c-{uuid4().hex[:8]}",
            claimed_at=now_iso,
        )

        async def _do_claim():
            return await gw1.atomic_claim_task(child_id, child_claim, now=now_iso, is_root=False)

        async def _do_timeout():
            return await gw2.materialize_claim_timeout(child_id, now_iso)

        res_claim, res_timeout = await asyncio.gather(
            _do_claim(),
            _do_timeout(),
            return_exceptions=True,
        )

        claim_won = not isinstance(res_claim, Exception)
        timeout_won = not isinstance(res_timeout, Exception)
        assert claim_won != timeout_won, f"Expected exactly one winner, got claim={claim_won}, timeout={timeout_won}"

        if claim_won:
            assert isinstance(res_timeout, (TaskAlreadyClaimedError, ClaimTimeoutConflictError, TaskNotClaimableError))
            assert await gw1.get_status(child_id) == "claimed_unacknowledged"
        else:
            assert isinstance(res_claim, (TaskNotClaimableError, ChildClaimDeadlineExpiredError, TaskAlreadyClaimedError))
            assert await gw1.get_status(child_id) == "claim_timed_out"
    finally:
        await gw1.aclose()
        await gw2.aclose()


async def test_redis_race_completion_vs_acknowledged_expiry(
    redis_url: str,
) -> None:
    gw1 = RedisGateway(redis_url)
    gw2 = RedisGateway(redis_url)
    try:
        task_id = f"task-race-comp-{uuid4().hex[:8]}"
        now_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
        now_iso = normalize_canonical_utc_timestamp(now_dt)
        past_iso = normalize_canonical_utc_timestamp(now_dt - timedelta(seconds=1))

        await gw1.register_root_task(
            TaskRequested(
                task_id=task_id,
                root_task_id=task_id,
                parent_task_id=None,
                prompt="Task for comp race",
                created_at=past_iso,
            ),
            required_capability="research",
        )
        claim = ActiveClaim(
            task_id=task_id,
            assignment_id=f"asg-race-{uuid4().hex[:8]}",
            runtime_principal_id="principal-research-1",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            proof_digest="digest-race",
            fencing_epoch=1,
            delivery_expires_at=past_iso,
            prior_claimable_status="pending",
            claim_id=f"claim-race-{uuid4().hex[:8]}",
            turn_id="turn-1",
            root_task_id=task_id,
            request_id=f"req-race-{uuid4().hex[:8]}",
            claimed_at=past_iso,
            state="acknowledged",
            acknowledged_at=past_iso,
            execution_expires_at=past_iso,  # expired!
        )
        await gw1.save_active_claim(claim)
        await gw1.update_status(task_id, "working")
        await gw1._client.set(f"task.fence:{task_id}", 1)

        cmd = CompletionCommand(
            completion_command_id=f"cmd-{uuid4().hex[:8]}",
            assignment_id=claim.assignment_id,
            execution_kind="initial",
            completed_result=TaskResult(task_id=task_id, output="completed output"),
        )

        async def _do_complete():
            return await gw1.complete_turn(
                task_id=task_id,
                command=cmd,
                proof_digest="digest-race",
                fencing_epoch=1,
                parent_task_id=None,
                root_task_id=task_id,
                completed_at=now_iso,
                runtime_principal_id="principal-research-1",
                bound_agent_id="research-agent-1",
            )

        async def _do_expire():
            return await gw2.transition_expired_claim(task_id, now_iso)

        res_comp, res_exp = await asyncio.gather(
            _do_complete(),
            _do_expire(),
            return_exceptions=True,
        )

        comp_won = not isinstance(res_comp, Exception)
        exp_won = not isinstance(res_exp, Exception)
        assert comp_won != exp_won, f"Expected exactly one winner, got comp={comp_won}, exp={exp_won}"

        final_st = await gw1.get_status(task_id)
        assert final_st in ("completed", "recovery_required")
        assert int(await gw1._client.get(f"task.fence:{task_id}")) == 1
    finally:
        await gw1.aclose()
        await gw2.aclose()


async def test_redis_race_recovery_resolution_vs_pull_claim(
    redis_url: str,
) -> None:
    gw1 = RedisGateway(redis_url)
    gw2 = RedisGateway(redis_url)
    try:
        task_id = f"task-race-rec-{uuid4().hex[:8]}"
        now_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
        now_iso = normalize_canonical_utc_timestamp(now_dt)
        past_iso = normalize_canonical_utc_timestamp(now_dt - timedelta(seconds=1))

        await gw1.register_root_task(
            TaskRequested(
                task_id=task_id,
                root_task_id=task_id,
                parent_task_id=None,
                prompt="Task for rec race",
                created_at=past_iso,
            ),
            required_capability="research",
        )
        claim = ActiveClaim(
            task_id=task_id,
            assignment_id=f"asg-rec-{uuid4().hex[:8]}",
            runtime_principal_id="principal-research-1",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            proof_digest="digest-rec",
            fencing_epoch=1,
            delivery_expires_at=past_iso,
            prior_claimable_status="pending",
            claim_id=f"claim-rec-{uuid4().hex[:8]}",
            turn_id="turn-1",
            root_task_id=task_id,
            request_id=f"req-rec-{uuid4().hex[:8]}",
            claimed_at=past_iso,
            state="acknowledged",
            acknowledged_at=past_iso,
            execution_expires_at=past_iso,
        )
        await gw1.save_active_claim(claim)
        await gw1.update_status(task_id, "working")
        await gw1._client.set(f"task.fence:{task_id}", 1)

        rec, _ = await gw1.transition_expired_claim(task_id, now_iso)
        assert await gw1.get_status(task_id) == "recovery_required"
        assert int(await gw1._client.get(f"task.fence:{task_id}")) == 1

        cmd = RecoveryDecisionCommand(
            recovery_decision_id=f"rec-dec-{uuid4().hex[:8]}",
            task_id=task_id,
            expired_claim_id=rec.claim_id,
            expired_assignment_id=rec.assignment_id,
            expired_fencing_epoch=1,
            disposition="restore_claimable",
        )

        next_claim = ActiveClaim(
            task_id=task_id,
            assignment_id=f"asg-next-{uuid4().hex[:8]}",
            runtime_principal_id="principal-research-1",
            bound_agent_id="research-agent-1",
            assigned_capability="research",
            execution_kind="initial",
            proof_digest="digest-root-next",
            fencing_epoch=2,
            delivery_expires_at=normalize_canonical_utc_timestamp(now_dt + timedelta(seconds=30)),
            prior_claimable_status="pending",
            claim_id=f"claim-next-{uuid4().hex[:8]}",
            turn_id="turn-2",
            root_task_id=task_id,
            request_id=f"req-next-{uuid4().hex[:8]}",
            claimed_at=now_iso,
        )

        async def _do_resolve():
            return await gw1.resolve_expired_claim(cmd, "principal-recovery-1", now=now_iso)

        async def _do_claim():
            return await gw2.atomic_claim_task(task_id, next_claim, now=now_iso, is_root=True)

        res_resolve, res_claim = await asyncio.gather(
            _do_resolve(),
            _do_claim(),
            return_exceptions=True,
        )

        assert not isinstance(res_resolve, Exception)
        if not isinstance(res_claim, Exception):
            new_claim, _replayed = res_claim
            assert new_claim.fencing_epoch == 2
            assert int(await gw1._client.get(f"task.fence:{task_id}")) == 2
        else:
            assert isinstance(res_claim, TaskNotClaimableError)
            c2, _replayed = await gw2.atomic_claim_task(task_id, next_claim, now=now_iso, is_root=True)
            assert c2.fencing_epoch == 2
            assert int(await gw1._client.get(f"task.fence:{task_id}")) == 2
    finally:
        await gw1.aclose()
        await gw2.aclose()
