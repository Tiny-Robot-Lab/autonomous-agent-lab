"""Tests for GitHub Issue #223: Resumed-parent pull-claim, Research continuation,

claim expiry/recovery handling, and turn-aware inference telemetry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from agent_contracts import (
    ActiveClaim,
    ArtifactReference,
    ClaimExpiryRecord,
    CompletionCommand,
    FallbackTransition,
    InferenceExecutionTelemetry,
    RecoveryDecisionCommand,
    RecoveryDecisionRecord,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
    TaskResult,
    TerminalOutcome,
    build_claim_timed_out_terminal_outcome,
    derive_terminal_outcome_id,
    normalize_canonical_utc_timestamp,
)
from orchestrator.application.inference_history import (
    TaskEventObservation,
    project_task_inference_history,
)
from orchestrator.application.ports import (
    CapabilityMismatchError,
    ClaimDeadlineNotExpiredError,
    ClaimTimeoutConflictError,
    RecoveryConflictError,
)
from orchestrator.application.use_cases import (
    EvaluateExpiry,
    ResolveExpiredClaim,
)
from orchestrator.config import Settings
from orchestrator.domain.principal import (
    RuntimePrincipal,
    compute_token_digest,
)
from orchestrator.domain.task import TaskLineage
from orchestrator.presentation.api import create_app
from research_agent.application.processor import ResearchExecutionError, ResearchProcessor
from research_agent.domain.models import ResearchResult, SynthesisResult

from tests.test_authoritative_task_store import (
    InMemoryAuthoritativeTaskStore,
    MockAtomicDispatcher,
)

PROOF_PEPPER = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


class DummyMessageBus:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def next_task(self) -> TaskRequested:
        raise NotImplementedError

    async def publish(self, event: Any) -> None:
        self.published.append(event)

    async def register(self, metadata: Any) -> None:
        pass

    async def heartbeat(self, heartbeat: Any) -> None:
        pass


class DummyExecutor:
    async def execute(self, task: TaskRequested) -> ResearchResult:
        return ResearchResult(
            task_id=task.task_id,
            output="Executed research findings",
            acknowledged=True,
            sources=[],
        )


class DummyCompletionClient:
    def __init__(self) -> None:
        self.completions: list[tuple[str, CompletionCommand, str]] = []

    async def submit_completion(
        self,
        task_id: str,
        command: CompletionCommand,
        assignment_proof: str,
    ) -> dict[str, Any]:
        self.completions.append((task_id, command, assignment_proof))
        return {"status": "ok", "outcome_id": "out-1"}


class DummyInferenceProvider:
    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        return SynthesisResult(
            question=question,
            evidence=evidence,
            content=f"Synthesized: {question} with evidence: {evidence}",
            truncated=False,
        )


async def _setup_delegated_pair(
    store: InMemoryAuthoritativeTaskStore,
    parent_id: str = "parent-1",
    child_id: str = "child-1",
    child_deadline_iso: str = "",
) -> None:
    now_dt = datetime.now(UTC)
    now_iso = normalize_canonical_utc_timestamp(now_dt)
    if not child_deadline_iso:
        child_deadline_iso = normalize_canonical_utc_timestamp(now_dt + timedelta(seconds=120))

    # Register root parent
    root_task = TaskRequested(
        task_id=parent_id,
        root_task_id=parent_id,
        parent_task_id=None,
        prompt="Initial prompt",
        created_at=now_iso,
        version=2,
    )
    await store.register_root_task(root_task, required_capability="research")
    await store.update_status(parent_id, "waiting_for_children")

    # Save sealed stage artifact
    art = StageArtifact(
        producing_task_id=parent_id,
        artifact_id="art-1",
        artifact_version=1,
        artifact_type="research_material",
        content={"material": "Primary research evidence from parent"},
        created_at=now_iso,
        sealed=True,
    )
    store.artifacts[(parent_id, "art-1", 1)] = art

    # Save parent continuation v1
    cont = TaskContinuation(
        parent_task_id=parent_id,
        child_task_id=child_id,
        created_at=now_iso,
        updated_at=now_iso,
        continuation_version=1,
        continuation_capability="research",
        stage="incorporate_analysis_child",
        artifact_references=(
            ArtifactReference(
                producing_task_id=parent_id,
                artifact_id="art-1",
                artifact_version=1,
            ),
        ),
    )
    await store.save_continuation(cont)
    store.latest_continuation[parent_id] = 1

    # Register child
    child_task = TaskRequested(
        task_id=child_id,
        root_task_id=parent_id,
        parent_task_id=parent_id,
        prompt="Analyze data",
        created_at=now_iso,
        version=2,
    )
    await store.register_child_task(child_task, required_capability="analysis")
    store.statuses[child_id] = "pending"
    store.claim_deadlines[child_id] = child_deadline_iso
    store.children.setdefault(parent_id, set()).add(child_id)


# ==============================================================================
# Capability A: Pending-child claim_timed_out materialization
# ==============================================================================


@pytest.mark.anyio
async def test_materialize_claim_timeout_inclusive_boundary() -> None:
    store = InMemoryAuthoritativeTaskStore()
    t_deadline = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    t_deadline_iso = normalize_canonical_utc_timestamp(t_deadline)
    await _setup_delegated_pair(store, child_deadline_iso=t_deadline_iso)

    # 1. At now < deadline: timeout rejected (deadline not expired)
    now_before = normalize_canonical_utc_timestamp(t_deadline - timedelta(seconds=1))
    with pytest.raises(ClaimDeadlineNotExpiredError, match="has not expired"):
        await store.materialize_claim_timeout("child-1", now_before)

    # 2. At now == deadline: timeout rejected (deadline is inclusive: claim permitted)
    with pytest.raises(ClaimDeadlineNotExpiredError, match="has not expired"):
        await store.materialize_claim_timeout("child-1", t_deadline_iso)

    # 3. At now > deadline: timeout succeeds
    now_after = normalize_canonical_utc_timestamp(t_deadline + timedelta(seconds=1))
    outcome, is_rep = await store.materialize_claim_timeout("child-1", now_after)
    assert is_rep is False

    assert outcome.task_id == "child-1"
    assert outcome.terminal_status == "claim_timed_out"
    assert outcome.failure_code is None
    assert store.statuses["child-1"] == "claim_timed_out"
    assert store.statuses["parent-1"] == "resumable"
    assert len(store.hints) == 1
    assert store.hints[0].capability == "research"
    assert store.hints[0].task_id == "parent-1"
    assert store.hints[0].execution_kind == "resumed_continuation"

    # 4. Idempotent replay
    outcome2, is_rep2 = await store.materialize_claim_timeout("child-1", now_after)
    assert is_rep2 is True
    assert outcome2.outcome_id == outcome.outcome_id


@pytest.mark.anyio
async def test_materialize_claim_timeout_rejected_if_active_claim_exists() -> None:
    store = InMemoryAuthoritativeTaskStore()
    t_deadline = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    t_deadline_iso = normalize_canonical_utc_timestamp(t_deadline)
    await _setup_delegated_pair(store, child_deadline_iso=t_deadline_iso)

    # Add active acknowledged claim on child
    now_iso = normalize_canonical_utc_timestamp(t_deadline - timedelta(minutes=1))
    store.claims["child-1"] = ActiveClaim(
        task_id="child-1",
        root_task_id="parent-1",
        assignment_id="asgn-child-1",
        claim_id="claim-child-1",
        turn_id="turn-1",
        fencing_epoch=1,
        runtime_principal_id="p-1",
        bound_agent_id="a-1",
        assigned_capability="analysis",
        execution_kind="delegated_child",
        proof_digest="digest",
        request_id="req-1",
        delivery_expires_at=normalize_canonical_utc_timestamp(t_deadline + timedelta(minutes=10)),
        claimed_at=now_iso,
        state="acknowledged",
        acknowledged_at=now_iso,
    )

    now_after = normalize_canonical_utc_timestamp(t_deadline + timedelta(seconds=10))
    with pytest.raises(ClaimTimeoutConflictError, match="active acknowledged claim"):
        await store.materialize_claim_timeout("child-1", now_after)


# ==============================================================================
# Capability C: Acknowledged execution claim expiry & recovery_required transition
# ==============================================================================


@pytest.mark.anyio
async def test_acknowledged_claim_execution_expiry_and_fence_preservation() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    now_iso = normalize_canonical_utc_timestamp(now_dt)

    task_id = "task-exp-1"
    store.statuses[task_id] = "working"
    store.fences[task_id] = 4

    exec_expires = normalize_canonical_utc_timestamp(now_dt + timedelta(seconds=900))
    store.claims[task_id] = ActiveClaim(
        task_id=task_id,
        root_task_id=task_id,
        assignment_id="asgn-exp-1",
        claim_id="claim-exp-1",
        turn_id="turn-1",
        fencing_epoch=4,
        runtime_principal_id="p-1",
        bound_agent_id="a-1",
        assigned_capability="research",
        execution_kind="initial",
        prior_claimable_status="pending",
        proof_digest="digest",
        request_id="req-1",
        delivery_expires_at=normalize_canonical_utc_timestamp(now_dt + timedelta(seconds=30)),
        claimed_at=now_iso,
        state="acknowledged",
        acknowledged_at=now_iso,
        execution_expires_at=exec_expires,
    )

    # 1. Before expiry: transition rejected
    before_dt = normalize_canonical_utc_timestamp(now_dt + timedelta(seconds=899))
    with pytest.raises(ClaimTimeoutConflictError, match="has not expired"):
        await store.transition_expired_claim(task_id, before_dt)

    # 2. At expiry: transition rejected
    with pytest.raises(ClaimTimeoutConflictError, match="has not expired"):
        await store.transition_expired_claim(task_id, exec_expires)

    # 3. After expiry: transition succeeds
    after_dt = normalize_canonical_utc_timestamp(now_dt + timedelta(seconds=901))
    expiry_record, is_rep = await store.transition_expired_claim(task_id, after_dt)
    assert is_rep is False

    assert expiry_record.task_id == task_id
    assert expiry_record.claim_id == "claim-exp-1"
    assert expiry_record.assignment_id == "asgn-exp-1"
    assert expiry_record.expired_fencing_epoch == 4
    assert expiry_record.prior_claimable_status == "pending"
    assert store.statuses[task_id] == "recovery_required"
    assert task_id not in store.claims

    # CRITICAL: Fencing epoch must NOT have been incremented!
    assert store.fences[task_id] == 4

    # Idempotent replay
    replayed, is_rep2 = await store.transition_expired_claim(task_id, after_dt)
    assert is_rep2 is True
    assert replayed.claim_id == expiry_record.claim_id
    assert store.fences[task_id] == 4


# ==============================================================================
# Capability D: Trusted ResolveExpiredClaim operation
# ==============================================================================


@pytest.mark.anyio
async def test_resolve_expired_claim_operator_and_disposition() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    now_iso = normalize_canonical_utc_timestamp(now_dt)

    task_id = "task-rec-1"
    store.statuses[task_id] = "recovery_required"
    store.fences[task_id] = 7

    store.claim_expiries[task_id] = ClaimExpiryRecord(
        task_id=task_id,
        claim_id="claim-rec-1",
        assignment_id="asgn-rec-1",
        principal_id="p-1",
        bound_agent_id="a-1",
        execution_kind="resumed_continuation",
        expired_fencing_epoch=7,
        acknowledged_at=now_iso,
        execution_expires_at=now_iso,
        expired_at=now_iso,
        prior_claimable_status="resumable",
        continuation_version=1,
    )

    uc = ResolveExpiredClaim(store)

    # 1. Non-operator principal rejected
    worker_principal = RuntimePrincipal(
        principal_id="p-worker",
        bound_agent_id="a-worker",
        authorized_capabilities=("research",),
        active=True,
    )
    cmd = RecoveryDecisionCommand(
        task_id=task_id,
        recovery_decision_id="rec-dec-1",
        expired_claim_id="claim-rec-1",
        expired_assignment_id="asgn-rec-1",
        expired_fencing_epoch=7,
        disposition="restore_claimable",
    )
    with pytest.raises(CapabilityMismatchError, match="recovery_operator"):
        await uc.execute(worker_principal, cmd)

    # 2. Operator principal with recovery_operator capability succeeds
    operator_principal = RuntimePrincipal(
        principal_id="p-operator",
        bound_agent_id="operator-1",
        authorized_capabilities=("recovery_operator",),
        active=True,
    )
    record, restored_status = await uc.execute(operator_principal, cmd)

    assert record.task_id == task_id
    assert record.operator_principal_id == "p-operator"
    assert record.disposition == "restore_claimable"
    assert restored_status == "resumable"
    assert store.statuses[task_id] == "resumable"

    # CRITICAL: Recovery resolution does NOT increment task.fence!
    assert store.fences[task_id] == 7

    # 3. Idempotent replay with matching fingerprint
    rec_replayed, st_replayed = await uc.execute(operator_principal, cmd)
    assert rec_replayed.recovery_decision_id == "rec-dec-1"
    assert st_replayed == "resumable"
    assert store.fences[task_id] == 7

    # 4. Conflicting replay with differing parameters
    conflicting_cmd = RecoveryDecisionCommand(
        task_id=task_id,
        recovery_decision_id="rec-dec-1",
        expired_claim_id="claim-rec-1",
        expired_assignment_id="asgn-rec-1",
        expired_fencing_epoch=8,
        disposition="restore_claimable",
    )
    with pytest.raises(RecoveryConflictError):
        await uc.execute(operator_principal, conflicting_cmd)


# ==============================================================================
# Capability B: Resumed-parent pull-claim & Research continuation
# ==============================================================================


@pytest.mark.anyio
async def test_research_processor_resumed_continuation_completed_child() -> None:
    store = InMemoryAuthoritativeTaskStore()
    await _setup_delegated_pair(store, parent_id="parent-c1", child_id="child-c1")

    # Complete child turn
    store.statuses["child-c1"] = "completed"
    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    fp = "f" * 64
    store.terminal_outcomes["child-c1"] = TerminalOutcome(
        outcome_id=derive_terminal_outcome_id(fp),
        task_id="child-c1",
        assignment_id="asgn-child-1",
        completion_command_id="cmd-child-1",
        semantic_fingerprint=fp,
        runtime_principal_id="p-analysis",
        bound_agent_id="analysis-1",
        execution_kind="delegated_child",
        terminal_status="completed",
        safe_summary="Analysis completed successfully.",
        completed_at=now_iso,
    )
    store.canonical_results["child-c1"] = TaskResult(
        task_id="child-c1",
        status="completed",
        output="Specialist statistical analysis confirms hypothesis.",
        sources=[{"url": "https://example.com/data"}],
    )

    bus = DummyMessageBus()
    executor = DummyExecutor()
    completion_client = DummyCompletionClient()
    inference = DummyInferenceProvider()

    processor = ResearchProcessor(
        agent_id="research-1",
        bus=bus,
        executor=executor,
        completion_client=completion_client,
        inference_provider=inference,
        context_reader=store,
    )

    task = TaskRequested(
        task_id="parent-c1",
        root_task_id="parent-c1",
        parent_task_id=None,
        prompt="Initial prompt",
        created_at=normalize_canonical_utc_timestamp(datetime.now(UTC)),
    )

    await processor.run_once(
        task,
        assignment_id="asgn-p1",
        assignment_proof="proof-p1",
        execution_kind="resumed_continuation",
        continuation_version=1,
    )

    assert len(completion_client.completions) == 1
    t_id, cmd, _proof = completion_client.completions[0]
    assert t_id == "parent-c1"
    assert cmd.execution_kind == "resumed_continuation"
    assert cmd.continuation_version == 1
    assert cmd.completed_result is not None
    assert "Specialist statistical analysis" in cmd.completed_result.output
    assert "Primary research evidence" in cmd.completed_result.output


@pytest.mark.anyio
async def test_research_processor_resumed_continuation_timed_out_child_degraded() -> None:
    store = InMemoryAuthoritativeTaskStore()
    await _setup_delegated_pair(store, parent_id="parent-to1", child_id="child-to1")

    # Child timed out
    store.statuses["child-to1"] = "claim_timed_out"
    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    store.terminal_outcomes["child-to1"] = build_claim_timed_out_terminal_outcome(
        task_id="child-to1",
        claim_deadline=now_iso,
        completed_at=now_iso,
    )

    bus = DummyMessageBus()
    executor = DummyExecutor()
    completion_client = DummyCompletionClient()
    inference = DummyInferenceProvider()

    processor = ResearchProcessor(
        agent_id="research-1",
        bus=bus,
        executor=executor,
        completion_client=completion_client,
        inference_provider=inference,
        context_reader=store,
    )

    task = TaskRequested(
        task_id="parent-to1",
        root_task_id="parent-to1",
        parent_task_id=None,
        prompt="Initial prompt",
        created_at=normalize_canonical_utc_timestamp(datetime.now(UTC)),
    )

    await processor.run_once(
        task,
        assignment_id="asgn-p2",
        assignment_proof="proof-p2",
        execution_kind="resumed_continuation",
        continuation_version=1,
    )

    assert len(completion_client.completions) == 1
    t_id, cmd, _proof = completion_client.completions[0]
    assert t_id == "parent-to1"
    assert cmd.execution_kind == "resumed_continuation"
    assert cmd.continuation_version == 1
    assert cmd.completed_result is not None
    assert "Specialist analysis was unavailable (claim_timed_out)" in cmd.completed_result.output


@pytest.mark.anyio
async def test_research_processor_missing_primary_evidence_submits_failure() -> None:
    store = InMemoryAuthoritativeTaskStore()
    await _setup_delegated_pair(store, parent_id="parent-fail1", child_id="child-fail1")

    # Corrupt primary artifact content
    del store.artifacts[("parent-fail1", "art-1", 1)]

    bus = DummyMessageBus()
    executor = DummyExecutor()
    completion_client = DummyCompletionClient()

    processor = ResearchProcessor(
        agent_id="research-1",
        bus=bus,
        executor=executor,
        completion_client=completion_client,
        context_reader=store,
    )

    task = TaskRequested(
        task_id="parent-fail1",
        root_task_id="parent-fail1",
        parent_task_id=None,
        prompt="Initial prompt",
        created_at=normalize_canonical_utc_timestamp(datetime.now(UTC)),
    )

    with pytest.raises(ResearchExecutionError):
        await processor.run_once(
            task,
            assignment_id="asgn-p3",
            assignment_proof="proof-p3",
            execution_kind="resumed_continuation",
            continuation_version=1,
        )

    assert len(completion_client.completions) == 1
    _t_id, cmd, _proof = completion_client.completions[0]
    assert cmd.failure_code == "EXECUTION_FAILED"


# ==============================================================================
# Capability D: Turn-Aware Inference Telemetry Projection
# ==============================================================================


def test_turn_aware_telemetry_attempt_index_resets_per_turn() -> None:
    task_id = "task-turn-telemetry-1"

    # Turn 1: Attempt 0 (failed), Attempt 1 (succeeded)
    t1_att0 = TaskEventObservation(
        event_id="1-0",
        task_id=task_id,
        stream_position=(1, 0),
        stream_order=0,
        turn_id="turn-1",
        assignment_id="asgn-1",
        execution_kind="initial",
        continuation_version=None,
        inference_telemetry=InferenceExecutionTelemetry(
            provider_name="gemini",
            attempt_index=0,
            latency_ms=150.0,
            status="fallback_eligible_failure",
            failure_reason="rate_limit",
            turn_id="turn-1",
            assignment_id="asgn-1",
            execution_kind="initial",
        ),
    )
    t1_tr0 = TaskEventObservation(
        event_id="1-1",
        task_id=task_id,
        stream_position=(1, 1),
        stream_order=1,
        turn_id="turn-1",
        assignment_id="asgn-1",
        fallback_transition=FallbackTransition(
            failed_provider="gemini",
            target_provider="cloudflare",
            reason="rate_limit",
            attempt_index=0,
            turn_id="turn-1",
            assignment_id="asgn-1",
        ),
    )
    t1_att1 = TaskEventObservation(
        event_id="1-2",
        task_id=task_id,
        stream_position=(1, 2),
        stream_order=2,
        turn_id="turn-1",
        assignment_id="asgn-1",
        execution_kind="initial",
        inference_telemetry=InferenceExecutionTelemetry(
            provider_name="cloudflare",
            attempt_index=1,
            latency_ms=200.0,
            status="succeeded",
            turn_id="turn-1",
            assignment_id="asgn-1",
            execution_kind="initial",
        ),
    )

    # Turn 2: Attempt 0 (succeeded) -> index resets to 0!
    t2_att0 = TaskEventObservation(
        event_id="2-0",
        task_id=task_id,
        stream_position=(2, 0),
        stream_order=3,
        turn_id="turn-2",
        assignment_id="asgn-2",
        execution_kind="resumed_continuation",
        continuation_version=1,
        inference_telemetry=InferenceExecutionTelemetry(
            provider_name="gemini",
            attempt_index=0,
            latency_ms=180.0,
            status="succeeded",
            turn_id="turn-2",
            assignment_id="asgn-2",
            execution_kind="resumed_continuation",
            continuation_version=1,
        ),
    )

    observations = [t1_att0, t1_tr0, t1_att1, t2_att0]

    projected = project_task_inference_history(
        task_id,
        observations,
        task_exists_durable=True,
        task_completed=True,
    )

    assert projected is not None
    # No collision, no gaps!
    assert projected["history_status"] == "complete"
    assert projected["final_status"] == "succeeded"

    # Flat chronological view contains all 3 attempts
    assert len(projected["attempts"]) == 3
    assert projected["attempts"][0]["provider_name"] == "gemini"
    assert projected["attempts"][1]["provider_name"] == "cloudflare"
    assert projected["attempts"][2]["provider_name"] == "gemini"

    # Grouped turns projection contains 2 distinct turns
    assert "turns" in projected
    turns = projected["turns"]
    assert len(turns) == 2
    assert turns[0]["turn_id"] == "turn-1"
    assert turns[0]["assignment_id"] == "asgn-1"
    assert len(turns[0]["attempts"]) == 2
    assert len(turns[0]["fallback_transitions"]) == 1

    assert turns[1]["turn_id"] == "turn-2"
    assert turns[1]["assignment_id"] == "asgn-2"
    assert turns[1]["execution_kind"] == "resumed_continuation"
    assert turns[1]["continuation_version"] == 1
    assert len(turns[1]["attempts"]) == 1
    assert turns[1]["attempts"][0]["attempt_index"] == 0


@pytest.mark.anyio
async def test_api_evaluate_expiry_and_resolve_expired_claim() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    now_iso = normalize_canonical_utc_timestamp(now_dt)

    test_settings = Settings(
        redis_url="redis://fake:6379/9",
        app_env="test",
        task_delivery_mode="authenticated_claim",
        runtime_credential_pepper="pepper" * 8,
        assignment_proof_pepper="proof" * 8,
        runtime_principals={
            "p-operator": {
                "principal_id": "p-operator",
                "bound_agent_id": "op-1",
                "authorized_capabilities": ["recovery_operator"],
                "token_digest": compute_token_digest("op-secret", "pepper" * 8),
                "active": True,
            },
            "p-worker": {
                "principal_id": "p-worker",
                "bound_agent_id": "worker-1",
                "authorized_capabilities": ["research"],
                "token_digest": compute_token_digest("worker-secret", "pepper" * 8),
                "active": True,
            },
        },
    )

    app = create_app(
        settings=test_settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    # Setup task with expired claim
    task_id = "task-exp-api-1"
    store.statuses[task_id] = "working"
    store.fences[task_id] = 3
    store.claims[task_id] = ActiveClaim(
        task_id=task_id,
        root_task_id=task_id,
        assignment_id="asgn-1",
        claim_id="claim-1",
        turn_id="turn-1",
        fencing_epoch=3,
        runtime_principal_id="p-worker",
        bound_agent_id="worker-1",
        assigned_capability="research",
        execution_kind="initial",
        prior_claimable_status="pending",
        proof_digest="digest",
        request_id="req-1",
        delivery_expires_at=now_iso,
        claimed_at=now_iso,
        state="acknowledged",
        acknowledged_at=now_iso,
        execution_expires_at="2020-01-01T00:00:00.000000Z",  # in the past
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Evaluate expiry without auth -> 401
        res = await client.post(f"/api/tasks/{task_id}/evaluate-expiry")
        assert res.status_code == 401

        # 2. Evaluate expiry with worker auth (lacks recovery_operator) -> 403
        res = await client.post(
            f"/api/tasks/{task_id}/evaluate-expiry",
            headers={"Authorization": "Bearer worker-secret"},
        )
        assert res.status_code == 403

        # 3. Evaluate expiry with operator auth -> 200
        res = await client.post(
            f"/api/tasks/{task_id}/evaluate-expiry",
            headers={"Authorization": "Bearer op-secret"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["task_id"] == task_id
        assert data["status"] == "recovery_required"
        assert store.statuses[task_id] == "recovery_required"

        # 4. Resolve expired claim with worker auth (lacks recovery_operator) -> 403
        resolve_payload = {
            "recovery_decision_id": "rec-dec-1",
            "expired_claim_id": "claim-1",
            "expired_assignment_id": "asgn-1",
            "expired_fencing_epoch": 3,
            "disposition": "restore_claimable",
        }
        res = await client.post(
            f"/api/tasks/{task_id}/resolve-expired-claim",
            json=resolve_payload,
            headers={"Authorization": "Bearer worker-secret"},
        )
        assert res.status_code == 403

        # 5. Resolve expired claim with operator auth -> 200
        res = await client.post(
            f"/api/tasks/{task_id}/resolve-expired-claim",
            json=resolve_payload,
            headers={"Authorization": "Bearer op-secret"},
        )
        assert res.status_code == 200
        res_data = res.json()
        assert res_data["restored_status"] == "pending"
        assert store.statuses[task_id] == "pending"
        assert store.fences[task_id] == 3  # fence preserved!

        # Extra fields (such as rationale) must be strictly forbidden (422)
        res_extra = await client.post(
            f"/api/tasks/{task_id}/resolve-expired-claim",
            json={**resolve_payload, "rationale": "extra rationale"},
            headers={"Authorization": "Bearer op-secret"},
        )
        assert res_extra.status_code == 422


@pytest.mark.anyio
async def test_evaluate_expiry_child_deadline_parity_and_endpoints() -> None:
    store = InMemoryAuthoritativeTaskStore()
    t_deadline = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    t_deadline_iso = normalize_canonical_utc_timestamp(t_deadline)
    before_iso = normalize_canonical_utc_timestamp(t_deadline - timedelta(seconds=1))
    equal_iso = t_deadline_iso
    after_iso = normalize_canonical_utc_timestamp(t_deadline + timedelta(seconds=1))

    await _setup_delegated_pair(store, child_deadline_iso=t_deadline_iso)
    operator = RuntimePrincipal(
        principal_id="rec-op",
        bound_agent_id="op-1",
        authorized_capabilities=("recovery_operator",),
    )
    uc = EvaluateExpiry(store)

    # 1. Before child deadline -> action="none", status="pending", zero mutations
    res_before = await uc.execute(operator, "child-1", now=before_iso)
    assert res_before.action == "none"
    assert res_before.status == "pending"
    assert store.statuses["child-1"] == "pending"
    assert store.statuses["parent-1"] == "waiting_for_children"
    assert "child-1" not in store.child_markers
    assert len(store.hints) == 0

    # 2. At exact child deadline -> action="none", status="pending", zero mutations
    res_equal = await uc.execute(operator, "child-1", now=equal_iso)
    assert res_equal.action == "none"
    assert res_equal.status == "pending"
    assert store.statuses["child-1"] == "pending"
    assert store.statuses["parent-1"] == "waiting_for_children"
    assert "child-1" not in store.child_markers
    assert len(store.hints) == 0

    # 3. HTTP endpoint parity before/at deadline
    test_settings = Settings(
        redis_url="redis://fake:6379/9",
        app_env="test",
        task_delivery_mode="authenticated_claim",
        runtime_credential_pepper="pepper" * 8,
        assignment_proof_pepper="proof" * 8,
        runtime_principals={
            "p-op": {
                "principal_id": "p-op",
                "bound_agent_id": "op-1",
                "authorized_capabilities": ["recovery_operator"],
                "token_digest": compute_token_digest("op-secret", "pepper" * 8),
                "active": True,
            },
        },
    )
    app = create_app(
        settings=test_settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Before deadline HTTP call
        r_before = await client.post(
            "/api/tasks/child-1/evaluate-expiry",
            json={"now": before_iso},
            headers={"Authorization": "Bearer op-secret"},
        )
        assert r_before.status_code == 200
        assert r_before.json() == {"action": "none", "status": "pending", "task_id": "child-1"}

        # At deadline HTTP call
        r_equal = await client.post(
            "/api/tasks/child-1/evaluate-expiry",
            json={"now": equal_iso},
            headers={"Authorization": "Bearer op-secret"},
        )
        assert r_equal.status_code == 200
        assert r_equal.json() == {"action": "none", "status": "pending", "task_id": "child-1"}

    # 4. After deadline -> action="claim_timed_out", status="claim_timed_out", mutations occur
    res_after = await uc.execute(operator, "child-1", now=after_iso)
    assert res_after.action == "claim_timed_out"
    assert res_after.status == "claim_timed_out"
    assert store.statuses["child-1"] == "claim_timed_out"
    assert store.statuses["parent-1"] == "resumable"
    assert "child-1" in store.child_markers
    assert len(store.hints) == 1


@pytest.mark.anyio
async def test_continuation_context_endpoint_and_use_case() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    now_iso = normalize_canonical_utc_timestamp(now_dt)

    test_settings = Settings(
        redis_url="redis://fake:6379/9",
        app_env="test",
        task_delivery_mode="authenticated_claim",
        runtime_credential_pepper="pepper" * 8,
        assignment_proof_pepper="proof" * 8,
        runtime_principals={
            "p-worker": {
                "principal_id": "p-worker",
                "bound_agent_id": "worker-1",
                "authorized_capabilities": ["research"],
                "token_digest": compute_token_digest("worker-secret", "pepper" * 8),
                "active": True,
            },
            "p-other": {
                "principal_id": "p-other",
                "bound_agent_id": "other-1",
                "authorized_capabilities": ["analysis"],
                "token_digest": compute_token_digest("other-secret", "pepper" * 8),
                "active": True,
            },
        },
    )

    app = create_app(
        settings=test_settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    parent_id = "parent-task-ctx-1"
    child_id = "child-task-ctx-1"
    assignment_id = "asgn-ctx-1"

    # Setup parent input, stage artifact, continuation, child outcome, child result
    store.inputs[(parent_id, 1)] = TaskInput(
        task_id=parent_id,
        prompt="Synthesize research",
        created_at=now_iso,
    )
    art = StageArtifact(
        artifact_id="art-ctx-1",
        producing_task_id=parent_id,
        artifact_type="research_material",
        content={"material": "primary facts"},
        created_at=now_iso,
        sealed=True,
    )
    store.artifacts[(parent_id, "art-ctx-1", 1)] = art
    store.continuations[(parent_id, 1)] = TaskContinuation(
        parent_task_id=parent_id,
        child_task_id=child_id,
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        artifact_references=(
            ArtifactReference(
                producing_task_id=parent_id,
                artifact_id="art-ctx-1",
                artifact_version=1,
            ),
        ),
        continuation_version=1,
    )
    from agent_contracts import build_terminal_outcome
    child_res = TaskResult(task_id=child_id, output="Child analysis findings")
    child_outcome, _ = build_terminal_outcome(
        command=CompletionCommand(
            completion_command_id="cmd-child-1",
            assignment_id="asgn-child-1",
            execution_kind="delegated_child",
            completed_result=child_res,
        ),
        task_id=child_id,
        runtime_principal_id="p-other",
        bound_agent_id="other-1",
        completed_at=now_iso,
    )
    store.terminal_outcomes[child_id] = child_outcome
    store.canonical_results[child_id] = child_res

    from orchestrator.domain.principal import (
        compute_assignment_verifier_digest,
        derive_assignment_proof,
        derive_proof_subkeys,
        encode_canonical_proof_identity,
        parse_master_secret,
    )
    master = parse_master_secret("proof" * 8)
    token_key, verifier_key = derive_proof_subkeys(master)
    canonical_identity = encode_canonical_proof_identity(
        assignment_id=assignment_id,
        claim_id="claim-ctx-1",
        task_id=parent_id,
        root_task_id=parent_id,
        turn_id="turn-ctx-1",
        runtime_principal_id="p-worker",
        bound_agent_id="worker-1",
        assigned_capability="research",
        execution_kind="resumed_continuation",
        continuation_version=1,
        fencing_epoch=2,
        request_id="req-ctx-1",
        delivery_expires_at=now_iso,
        delegation_grant_id=None,
        prior_claimable_status="resumable",
    )
    proof = derive_assignment_proof(token_key, canonical_identity)
    proof_digest = compute_assignment_verifier_digest(verifier_key, proof)

    # Active acknowledged claim for resumed_continuation
    store.claims[parent_id] = ActiveClaim(
        task_id=parent_id,
        root_task_id=parent_id,
        assignment_id=assignment_id,
        claim_id="claim-ctx-1",
        turn_id="turn-ctx-1",
        fencing_epoch=2,
        runtime_principal_id="p-worker",
        bound_agent_id="worker-1",
        assigned_capability="research",
        execution_kind="resumed_continuation",
        continuation_version=1,
        prior_claimable_status="resumable",
        proof_digest=proof_digest,
        request_id="req-ctx-1",
        delivery_expires_at=now_iso,
        claimed_at=now_iso,
        state="acknowledged",
        acknowledged_at=now_iso,
        execution_expires_at=now_iso,
    )
    store.fences[parent_id] = 2

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. No auth -> 401
        res = await client.get(
            f"/api/tasks/{parent_id}/continuation-context?assignment_id={assignment_id}",
            headers={"X-Assignment-Proof": proof},
        )
        assert res.status_code == 401

        # 2. No proof header -> 401
        res = await client.get(
            f"/api/tasks/{parent_id}/continuation-context?assignment_id={assignment_id}",
            headers={"Authorization": "Bearer worker-secret"},
        )
        assert res.status_code == 401

        # 3. Wrong principal capability (analysis agent trying to read research continuation) -> 403
        res = await client.get(
            f"/api/tasks/{parent_id}/continuation-context?assignment_id={assignment_id}",
            headers={"Authorization": "Bearer other-secret", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 403

        # 4. Valid call -> 200
        res = await client.get(
            f"/api/tasks/{parent_id}/continuation-context?assignment_id={assignment_id}",
            headers={"Authorization": "Bearer worker-secret", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["version"] == 1
        assert data["parent_task_input"]["task_id"] == parent_id
        assert data["continuation"]["stage"] == "incorporate_analysis_child"
        assert data["stage_artifact"]["artifact_id"] == "art-ctx-1"
        assert data["child_outcome"]["terminal_status"] == "completed"
        assert data["child_result"]["output"] == "Child analysis findings"


def test_resolve_expired_claim_input_epoch_validation() -> None:
    from orchestrator.presentation.api import ResolveExpiredClaimInput
    from pydantic import ValidationError

    base = {
        "recovery_decision_id": "dec-1",
        "expired_claim_id": "claim-1",
        "expired_assignment_id": "asgn-1",
        "disposition": "restore_claimable",
    }

    # Valid integer >= 1
    valid = ResolveExpiredClaimInput(**base, expired_fencing_epoch=1)
    assert valid.expired_fencing_epoch == 1

    valid2 = ResolveExpiredClaimInput(**base, expired_fencing_epoch=42)
    assert valid2.expired_fencing_epoch == 42

    # Reject epoch 0
    with pytest.raises(ValidationError):
        ResolveExpiredClaimInput(**base, expired_fencing_epoch=0)

    # Reject negative epoch
    with pytest.raises(ValidationError):
        ResolveExpiredClaimInput(**base, expired_fencing_epoch=-1)

    # Reject forbidden extra fields (such as rationale)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResolveExpiredClaimInput(**base, expired_fencing_epoch=1, rationale="not allowed")

    # Reject boolean True (which is isinstance(True, int))
    with pytest.raises((ValidationError, TypeError)):
        ResolveExpiredClaimInput(**base, expired_fencing_epoch=True)

    # Reject boolean False
    with pytest.raises((ValidationError, TypeError)):
        ResolveExpiredClaimInput(**base, expired_fencing_epoch=False)

    # Reject string representation of integer
    with pytest.raises((ValidationError, TypeError)):
        ResolveExpiredClaimInput(**base, expired_fencing_epoch="1")


@pytest.mark.anyio
async def test_redis_bus_bounded_queue_fairness_alternation() -> None:
    from unittest.mock import AsyncMock

    from agent_contracts import WakeUpHint
    from research_agent.infrastructure.redis_bus import RedisBus

    bus = RedisBus("redis://fake:6379/0", queue_name="tasks", hint_queue="hints")

    # Simulate client.blpop returning results from whichever queue was prioritized
    call_log: list[list[str]] = []

    async def fake_blpop(queues: list[str], timeout: int = 2):
        call_log.append(list(queues))
        first_q = queues[0]
        if first_q == "hints":
            hint = WakeUpHint(
                hint_id="h1",
                capability="research",
                created_at="2026-11-22T00:00:00Z",
                execution_kind="resumed_continuation",
                task_id="t1",
                expected_continuation_version=1,
            )
            return ("hints", hint.to_json())
        task = TaskRequested(
            task_id="t2",
            root_task_id="t2",
            parent_task_id=None,
            prompt="p2",
            created_at="2026-11-22T00:00:00Z",
        )
        return ("tasks", task.to_json())

    bus.client.blpop = AsyncMock(side_effect=fake_blpop)

    # First call prefers hints
    sig1 = await bus.next_signal()
    assert isinstance(sig1, WakeUpHint)
    assert call_log[0] == ["hints", "tasks"]

    # Second call alternates to tasks
    sig2 = await bus.next_signal()
    assert isinstance(sig2, TaskRequested)
    assert call_log[1] == ["tasks", "hints"]

    # Third call alternates back to hints
    sig3 = await bus.next_signal()
    assert isinstance(sig3, WakeUpHint)
    assert call_log[2] == ["hints", "tasks"]


@pytest.mark.anyio
async def test_acknowledged_expiry_exact_equality_boundary() -> None:
    store = InMemoryAuthoritativeTaskStore()
    expires_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    expires_iso = normalize_canonical_utc_timestamp(expires_dt)
    before_iso = normalize_canonical_utc_timestamp(expires_dt - timedelta(seconds=1))
    equal_iso = expires_iso
    after_iso = normalize_canonical_utc_timestamp(expires_dt + timedelta(seconds=1))

    task_id = "task-boundary-1"
    store.statuses[task_id] = "working"
    store.fences[task_id] = 1
    claim = ActiveClaim(
        task_id=task_id,
        root_task_id=task_id,
        assignment_id="asgn-b-1",
        claim_id="claim-b-1",
        turn_id="turn-b-1",
        fencing_epoch=1,
        runtime_principal_id="p-w",
        bound_agent_id="a-w",
        assigned_capability="research",
        execution_kind="initial",
        prior_claimable_status="pending",
        proof_digest="digest",
        request_id="req-b-1",
        delivery_expires_at=expires_iso,
        claimed_at=before_iso,
        state="acknowledged",
        acknowledged_at=before_iso,
        execution_expires_at=expires_iso,
    )
    store.claims[task_id] = claim

    op_principal = RuntimePrincipal(
        principal_id="p-op",
        bound_agent_id="a-op",
        authorized_capabilities=("recovery_operator",),
        active=True,
    )
    eval_uc = EvaluateExpiry(store)

    # 1. Before boundary: now < execution_expires_at
    # Status hook does not transition
    assert await store.get_status(task_id, now=before_iso) == "working"
    # Result hook does not transition
    res = await store.get_result(task_id, now=before_iso)
    assert res["status"] == "working"
    # EvaluateExpiry returns action="none"
    ev_res = await eval_uc.execute(op_principal, task_id, now=before_iso)
    assert ev_res.action == "none"
    assert ev_res.status == "working"
    assert store.statuses[task_id] == "working"

    # 2. Exact equality boundary: now == execution_expires_at
    # Must remain valid! Must NOT enter recovery_required
    assert await store.get_status(task_id, now=equal_iso) == "working"
    res_eq = await store.get_result(task_id, now=equal_iso)
    assert res_eq["status"] == "working"
    ev_eq = await eval_uc.execute(op_principal, task_id, now=equal_iso)
    assert ev_eq.action == "none"
    assert ev_eq.status == "working"
    assert store.statuses[task_id] == "working"

    # 3. After boundary: now > execution_expires_at
    # Expiry may materialize!
    ev_after = await eval_uc.execute(op_principal, task_id, now=after_iso)
    assert ev_after.action == "recovery_required"
    assert ev_after.status == "recovery_required"
    assert ev_after.expiry_record is not None
    assert store.statuses[task_id] == "recovery_required"


@pytest.mark.anyio
async def test_evaluate_expiry_action_reporting_coverage() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now_dt = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    now_iso = normalize_canonical_utc_timestamp(now_dt)
    past_iso = normalize_canonical_utc_timestamp(now_dt - timedelta(minutes=5))

    op_principal = RuntimePrincipal(
        principal_id="p-op",
        bound_agent_id="a-op",
        authorized_capabilities=("recovery_operator",),
        active=True,
    )
    eval_uc = EvaluateExpiry(store)

    # Case A: Child timeout directly
    p_id = "p-act-1"
    c_id = "c-act-1"
    store.statuses[p_id] = "waiting_for_children"
    store.children[p_id] = {c_id}
    store.latest_continuation[p_id] = 1
    store.continuations[(p_id, 1)] = TaskContinuation(
        parent_task_id=p_id,
        child_task_id=c_id,
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
    )
    store.statuses[c_id] = "pending"
    store.lineages[c_id] = TaskLineage(task_id=c_id, root_task_id=p_id, parent_task_id=p_id)
    store.claim_deadlines[c_id] = past_iso

    res_timeout = await eval_uc.execute(op_principal, c_id, now=now_iso)
    assert res_timeout.action == "claim_timed_out"
    assert res_timeout.status == "claim_timed_out"
    assert res_timeout.outcome is not None

    # Case B: Replay of child timeout returns action="claim_timed_out"
    res_timeout_rep = await eval_uc.execute(op_principal, c_id, now=now_iso)
    assert res_timeout_rep.action == "claim_timed_out"
    assert res_timeout_rep.status == "claim_timed_out"

    # Case C: Acknowledged expiry directly
    work_id = "work-act-1"
    store.statuses[work_id] = "working"
    store.fences[work_id] = 1
    store.claims[work_id] = ActiveClaim(
        task_id=work_id,
        root_task_id=work_id,
        assignment_id="asgn-w",
        claim_id="claim-w",
        turn_id="turn-w",
        fencing_epoch=1,
        runtime_principal_id="p-w",
        bound_agent_id="a-w",
        assigned_capability="research",
        execution_kind="initial",
        prior_claimable_status="pending",
        proof_digest="digest",
        request_id="req-w",
        delivery_expires_at=past_iso,
        claimed_at=past_iso,
        state="acknowledged",
        acknowledged_at=past_iso,
        execution_expires_at=past_iso,
    )
    res_exp = await eval_uc.execute(op_principal, work_id, now=now_iso)
    assert res_exp.action == "recovery_required"
    assert res_exp.status == "recovery_required"
    assert res_exp.expiry_record is not None

    # Case D: Preceding targeted hook just completed transition -> evaluate_expiry still reports action="recovery_required"
    hook_task = "hook-task-1"
    store.statuses[hook_task] = "working"
    store.fences[hook_task] = 1
    store.claims[hook_task] = ActiveClaim(
        task_id=hook_task,
        root_task_id=hook_task,
        assignment_id="asgn-h",
        claim_id="claim-h",
        turn_id="turn-h",
        fencing_epoch=1,
        runtime_principal_id="p-w",
        bound_agent_id="a-w",
        assigned_capability="research",
        execution_kind="initial",
        prior_claimable_status="pending",
        proof_digest="digest",
        request_id="req-h",
        delivery_expires_at=past_iso,
        claimed_at=past_iso,
        state="acknowledged",
        acknowledged_at=past_iso,
        execution_expires_at=past_iso,
    )
    # Hook triggers transition
    hook_st = await store.get_status(hook_task, now=now_iso)
    assert hook_st == "recovery_required"
    # EvaluateExpiry must report action="recovery_required", NOT "none"
    res_hook = await eval_uc.execute(op_principal, hook_task, now=now_iso)
    assert res_hook.action == "recovery_required"
    assert res_hook.status == "recovery_required"
    assert res_hook.expiry_record is not None

    # Case E: No eligible transition exists (completed task)
    done_id = "done-task-1"
    store.statuses[done_id] = "completed"
    res_done = await eval_uc.execute(op_principal, done_id, now=now_iso)
    assert res_done.action == "none"
    assert res_done.status == "completed"


@pytest.mark.anyio
async def test_resumed_failure_classification_preserves_codes() -> None:
    from agent_contracts import AllInferenceProvidersFailedError

    now_iso = "2026-11-22T00:00:00Z"
    task = TaskRequested(
        task_id="p-fail-1",
        root_task_id="p-fail-1",
        parent_task_id=None,
        prompt="Resumed parent prompt",
        created_at=now_iso,
    )

    # 1. AllInferenceProvidersFailedError -> PROVIDER_UNAVAILABLE
    bus = DummyMessageBus()
    comp_client = DummyCompletionClient()

    class FailingInferenceProvider:
        async def synthesize(self, *args: Any, **kwargs: Any) -> Any:
            raise AllInferenceProvidersFailedError(failures=())

    class StubContextReader:
        async def get_task_input(self, *args: Any, **kwargs: Any) -> Any:
            return TaskInput(task_id="p-fail-1", prompt="p", created_at=now_iso)

        async def get_continuation(self, *args: Any, **kwargs: Any) -> Any:
            return TaskContinuation(
                parent_task_id="p-fail-1",
                child_task_id="c-fail-1",
                created_at=now_iso,
                updated_at=now_iso,
                stage="incorporate_analysis_child",
                continuation_version=1,
                artifact_references=[
                    ArtifactReference(
                        artifact_id="art-f-1",
                        producing_task_id="p-fail-1",
                        artifact_version=1,
                    )
                ],
            )

        async def get_stage_artifact(self, *args: Any, **kwargs: Any) -> Any:
            return StageArtifact(
                artifact_id="art-f-1",
                producing_task_id="p-fail-1",
                artifact_type="research_material",
                content={"material": "primary findings"},
                created_at=now_iso,
                sealed=True,
            )

        async def get_terminal_outcome(self, *args: Any, **kwargs: Any) -> Any:
            return build_claim_timed_out_terminal_outcome(
                task_id="c-fail-1",
                claim_deadline=now_iso,
                completed_at=now_iso,
            )

        async def get_canonical_result(self, *args: Any, **kwargs: Any) -> Any:
            return None

    proc = ResearchProcessor(
        agent_id="res-agent-1",
        bus=bus,
        executor=DummyExecutor(),
        completion_client=comp_client,
        context_reader=StubContextReader(),
        inference_provider=FailingInferenceProvider(),
    )

    with pytest.raises(ResearchExecutionError):
        await proc.run_once(
            task,
            assignment_id="asgn-fail-1",
            assignment_proof="proof-1",
            execution_kind="resumed_continuation",
            continuation_version=1,
        )

    assert len(comp_client.completions) == 1
    _, submitted_cmd, _ = comp_client.completions[0]
    assert submitted_cmd.failure_code == "PROVIDER_UNAVAILABLE"
    assert submitted_cmd.execution_kind == "resumed_continuation"
    assert submitted_cmd.continuation_version == 1

    # 2. Corrupt / ambiguous exception -> EXECUTION_FAILED
    comp_client2 = DummyCompletionClient()

    class BuggyInferenceProvider:
        async def synthesize(self, *args: Any, **kwargs: Any) -> Any:
            raise KeyError("unexpected internal data key")

    proc2 = ResearchProcessor(
        agent_id="res-agent-1",
        bus=bus,
        executor=DummyExecutor(),
        completion_client=comp_client2,
        context_reader=StubContextReader(),
        inference_provider=BuggyInferenceProvider(),
    )

    with pytest.raises(ResearchExecutionError):
        await proc2.run_once(
            task,
            assignment_id="asgn-fail-2",
            assignment_proof="proof-2",
            execution_kind="resumed_continuation",
            continuation_version=1,
        )

    assert len(comp_client2.completions) == 1
    _, submitted_cmd2, _ = comp_client2.completions[0]
    assert submitted_cmd2.failure_code == "EXECUTION_FAILED"


def test_continuation_context_reader_runtime_checkable() -> None:
    from research_agent.application.ports import ContinuationContextReader

    class ValidReader:
        async def get_task_input(self, task_id: str, version: int = 1) -> Any: ...
        async def get_continuation(self, task_id: str, version: int = 1) -> Any: ...
        async def get_stage_artifact(self, producing_task_id: str, artifact_id: str, artifact_version: int = 1) -> Any: ...
        async def get_terminal_outcome(self, task_id: str) -> Any: ...
        async def get_canonical_result(self, task_id: str) -> Any: ...

    class IncompleteReader:
        pass

    assert isinstance(ValidReader(), ContinuationContextReader)
    assert not isinstance(IncompleteReader(), ContinuationContextReader)


def test_ast_no_duplicate_dictionary_keys() -> None:
    import ast
    from pathlib import Path

    adapter_path = Path("services/orchestrator/src/orchestrator/infrastructure/redis_adapter.py")
    tree = ast.parse(adapter_path.read_text(encoding="utf-8"))

    duplicates: list[tuple[str, int]] = []

    class DictKeyChecker(ast.NodeVisitor):
        def visit_Dict(self, node: ast.Dict) -> None:
            seen: set[str] = set()
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if key.value in seen:
                        duplicates.append((key.value, key.lineno))
                    seen.add(key.value)
            self.generic_visit(node)

    DictKeyChecker().visit(tree)
    assert not duplicates, f"Found duplicate dictionary keys in redis_adapter.py: {duplicates}"


def test_ast_no_duplicate_redis_gateway_method_definitions() -> None:
    import ast
    from pathlib import Path

    adapter_path = Path("services/orchestrator/src/orchestrator/infrastructure/redis_adapter.py")
    tree = ast.parse(adapter_path.read_text(encoding="utf-8"))

    methods: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RedisGateway":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)

    duplicate_methods = [m for m in set(methods) if methods.count(m) > 1]
    assert not duplicate_methods, f"Found duplicate RedisGateway methods: {duplicate_methods}"


def test_ast_no_duplicate_top_level_lua_constant_definitions() -> None:
    import ast
    from pathlib import Path

    adapter_path = Path("services/orchestrator/src/orchestrator/infrastructure/redis_adapter.py")
    tree = ast.parse(adapter_path.read_text(encoding="utf-8"))

    constants: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("_LUA"):
                    constants.append(target.id)

    duplicate_constants = [c for c in set(constants) if constants.count(c) > 1]
    assert not duplicate_constants, f"Found duplicate top-level Lua constants: {duplicate_constants}"


@pytest.mark.anyio
async def test_redis_adapter_single_get_calls() -> None:
    from unittest.mock import AsyncMock

    from orchestrator.infrastructure.redis_adapter import RedisGateway

    gw = RedisGateway(url="redis://localhost:6379/0")
    mock_client = AsyncMock()
    gw._client = mock_client

    # 1. get_status: on normal path without outcome, calls task.outcome then task.status exactly once each
    mock_client.get.side_effect = [
        None,  # task.outcome
        '{"task_id": "task-1", "status": "completed"}',  # task.status
    ]
    status = await gw.get_status("task-1")
    assert status == "completed"
    assert mock_client.get.call_count == 2
    mock_client.get.assert_any_call("task.outcome:task-1")
    mock_client.get.assert_any_call("task.status:task-1")

    # 2. get_active_claim: exactly 1 get call
    mock_client.get.reset_mock()
    mock_client.get.side_effect = None
    mock_client.get.return_value = None
    claim = await gw.get_active_claim("task-1")
    assert claim is None
    assert mock_client.get.call_count == 1
    mock_client.get.assert_called_once_with("task.claim:task-1")

    # 3. get_fencing_epoch: exactly 1 get call
    mock_client.get.reset_mock()
    mock_client.get.return_value = "3"
    fence = await gw.get_fencing_epoch("task-1")
    assert fence == 3
    assert mock_client.get.call_count == 1
    mock_client.get.assert_called_once_with("task.fence:task-1")


@pytest.mark.anyio
async def test_redis_adapter_eval_executes_exactly_once() -> None:
    from unittest.mock import AsyncMock

    from agent_contracts import ActiveClaim, RecoveryDecisionCommand
    from orchestrator.infrastructure.redis_adapter import RedisGateway

    gw = RedisGateway(url="redis://localhost:6379/0")
    mock_client = AsyncMock()
    gw._client = mock_client

    # 1. atomic_claim_task
    mock_client.eval.return_value = ["OK", "digest-1"]
    claim = ActiveClaim(
        task_id="task-1",
        assignment_id="asgn-1",
        root_task_id="task-1",
        claim_id="claim-1",
        turn_id="turn-1",
        fencing_epoch=1,
        runtime_principal_id="p-1",
        bound_agent_id="a-1",
        assigned_capability="general",
        execution_kind="initial",
        prior_claimable_status="pending",
        proof_digest="digest-1",
        request_id="req-1",
        delivery_expires_at="2026-11-22T13:00:00.000000Z",
        claimed_at="2026-11-22T12:00:00.000000Z",
    )
    res_claim, _ = await gw.atomic_claim_task("task-1", claim, is_root=True)
    assert res_claim.task_id == "task-1"
    assert mock_client.eval.call_count == 1

    # 2. materialize_claim_timeout
    mock_client.eval.reset_mock()
    mock_client.get.side_effect = [
        '{"parent_task_id": "parent-1", "root_task_id": "root-1", "task_id": "child-1"}',  # lineage
        "2026-11-22T11:00:00.000000Z",  # deadline
    ]
    mock_outcome = build_claim_timed_out_terminal_outcome(
        task_id="child-1",
        claim_deadline="2026-11-22T11:00:00.000000Z",
        completed_at="2026-11-22T12:00:00.000000Z",
    )
    mock_client.eval.return_value = ["OK", mock_outcome.to_json()]
    outcome, _ = await gw.materialize_claim_timeout("child-1", "2026-11-22T12:00:00.000000Z")
    assert outcome.task_id == "child-1"
    assert mock_client.eval.call_count == 1

    # 3. transition_expired_claim
    mock_client.eval.reset_mock()
    ack_claim = ActiveClaim(
        task_id="task-1",
        assignment_id="asgn-1",
        root_task_id="task-1",
        claim_id="claim-1",
        turn_id="turn-1",
        fencing_epoch=1,
        runtime_principal_id="p-1",
        bound_agent_id="a-1",
        assigned_capability="general",
        execution_kind="initial",
        prior_claimable_status="pending",
        proof_digest="digest-1",
        request_id="req-1",
        delivery_expires_at="2026-11-22T13:00:00.000000Z",
        claimed_at="2026-11-22T12:00:00.000000Z",
        state="acknowledged",
        acknowledged_at="2026-11-22T12:00:00.000000Z",
        execution_expires_at="2026-11-22T13:00:00.000000Z",
    )
    mock_client.get.side_effect = [
        ack_claim.to_json(),  # claim
    ]
    mock_expiry = ClaimExpiryRecord(
        task_id="task-1",
        claim_id="claim-1",
        assignment_id="asgn-1",
        principal_id="p-1",
        bound_agent_id="a-1",
        execution_kind="initial",
        expired_fencing_epoch=1,
        acknowledged_at="2026-11-22T12:00:00.000000Z",
        execution_expires_at="2026-11-22T13:00:00.000000Z",
        expired_at="2026-11-22T14:00:00.000000Z",
        prior_claimable_status="pending",
    )
    mock_client.eval.return_value = ["OK", mock_expiry.to_json()]
    rec, _ = await gw.transition_expired_claim("task-1", "2026-11-22T14:00:00.000000Z")
    assert rec.task_id == "task-1"
    assert mock_client.eval.call_count == 1

    # 4. resolve_expired_claim
    mock_client.eval.reset_mock()
    mock_client.get.side_effect = [
        mock_expiry.to_json(),  # claim_expiry
    ]
    cmd = RecoveryDecisionCommand(
        task_id="task-1",
        recovery_decision_id="dec-1",
        expired_claim_id="claim-1",
        expired_assignment_id="asgn-1",
        expired_fencing_epoch=1,
        disposition="restore_claimable",
    )
    mock_record = RecoveryDecisionRecord(
        recovery_decision_id="dec-1",
        task_id="task-1",
        expired_claim_id="claim-1",
        expired_assignment_id="asgn-1",
        expired_fencing_epoch=1,
        disposition="restore_claimable",
        semantic_fingerprint="fingerprint-1",
        prior_status="recovery_required",
        restored_status="pending",
        operator_principal_id="rec-op-1",
        resolved_at="2026-11-22T14:00:00.000000Z",
    )
    mock_client.eval.return_value = ["OK", mock_record.to_json()]
    dec, _ = await gw.resolve_expired_claim(cmd, "rec-op-1", now="2026-11-22T14:00:00.000000Z")
    assert dec.task_id == "task-1"
    assert mock_client.eval.call_count == 1


@pytest.mark.anyio
async def test_timeout_marker_round_trip_and_strict_validation() -> None:
    from orchestrator.application.ports import ClaimTimeoutConflictError

    store = InMemoryAuthoritativeTaskStore()
    t_deadline = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    t_deadline_iso = normalize_canonical_utc_timestamp(t_deadline)
    after_iso = normalize_canonical_utc_timestamp(t_deadline + timedelta(seconds=1))

    await _setup_delegated_pair(store, child_deadline_iso=t_deadline_iso)
    outcome, is_replayed = await store.materialize_claim_timeout("child-1", now=after_iso)
    assert not is_replayed
    assert outcome.task_id == "child-1"

    # Verify exact 5 fields in marker
    marker = store.child_markers.get("child-1")
    assert marker is not None
    assert set(marker.keys()) == {
        "child_task_id",
        "parent_task_id",
        "outcome_id",
        "continuation_version",
        "status",
    }
    assert marker["child_task_id"] == "child-1"
    assert marker["parent_task_id"] == "parent-1"
    assert marker["outcome_id"] == outcome.outcome_id
    assert marker["continuation_version"] == 1
    assert marker["status"] == "claim_timed_out"

    # Idempotent replay succeeds with exact marker
    outcome2, is_replayed2 = await store.materialize_claim_timeout("child-1", now=after_iso)
    assert is_replayed2
    assert outcome2.outcome_id == outcome.outcome_id

    # Missing parent_task_id fails closed
    bad_marker = dict(marker)
    del bad_marker["parent_task_id"]
    store.child_markers["child-1"] = bad_marker
    with pytest.raises(ClaimTimeoutConflictError):
        await store.materialize_claim_timeout("child-1", now=after_iso)

    # Missing outcome_id fails closed
    bad_marker = dict(marker)
    del bad_marker["outcome_id"]
    store.child_markers["child-1"] = bad_marker
    with pytest.raises(ClaimTimeoutConflictError):
        await store.materialize_claim_timeout("child-1", now=after_iso)

    # Missing continuation_version fails closed
    bad_marker = dict(marker)
    del bad_marker["continuation_version"]
    store.child_markers["child-1"] = bad_marker
    with pytest.raises(ClaimTimeoutConflictError):
        await store.materialize_claim_timeout("child-1", now=after_iso)

    # Permissive alias task_id instead of child_task_id fails closed
    bad_marker = dict(marker)
    del bad_marker["child_task_id"]
    bad_marker["task_id"] = "child-1"
    store.child_markers["child-1"] = bad_marker
    with pytest.raises(ClaimTimeoutConflictError):
        await store.materialize_claim_timeout("child-1", now=after_iso)

    # Permissive alias terminal_status instead of status fails closed
    bad_marker = dict(marker)
    del bad_marker["status"]
    bad_marker["terminal_status"] = "claim_timed_out"
    store.child_markers["child-1"] = bad_marker
    with pytest.raises(ClaimTimeoutConflictError):
        await store.materialize_claim_timeout("child-1", now=after_iso)


def test_ast_all_eval_calls_in_try_blocks() -> None:
    import ast
    from pathlib import Path

    path = Path("services/orchestrator/src/orchestrator/infrastructure/redis_adapter.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))

    class EvalVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: dict[ast.AST, ast.AST] = {}
            self.eval_calls: list[ast.Call] = []

        def visit(self, node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
            super().visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "eval"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "_client"
            ):
                self.eval_calls.append(node)
            self.generic_visit(node)

    visitor = EvalVisitor()
    visitor.visit(tree)

    assert len(visitor.eval_calls) == 17, f"Expected 17 direct eval calls, found {len(visitor.eval_calls)}"

    messages: list[str] = []
    for call in visitor.eval_calls:
        curr = call
        caught_types: list[str] = []
        handler_found = False
        while curr in visitor.parents:
            curr = visitor.parents[curr]
            if isinstance(curr, ast.Try):
                for h in curr.handlers:
                    if isinstance(h.type, ast.Name):
                        caught_types.append(h.type.id)
                    elif isinstance(h.type, ast.Tuple):
                        caught_types.extend(elt.id for elt in h.type.elts if isinstance(elt, ast.Name))
                    if isinstance(h.type, ast.Name) and h.type.id == "ResponseError":
                        handler_found = True
                        for stmt in h.body:
                            if isinstance(stmt, ast.Raise):
                                assert isinstance(stmt.cause, ast.Name) and stmt.cause.id == h.name, (
                                    f"eval at line {call.lineno} does not chain 'from {h.name}'"
                                )
                                assert (
                                    isinstance(stmt.exc, ast.Call)
                                    and stmt.exc.func.id == "CorruptAuthoritativeRecordError"
                                ), f"eval at line {call.lineno} does not raise CorruptAuthoritativeRecordError"
                                arg = stmt.exc.args[0]
                                assert isinstance(arg, ast.Constant), (
                                    f"eval at line {call.lineno} message is not a plain string constant"
                                )
                                msg = arg.value
                                assert isinstance(msg, str)
                                messages.append(msg)
                                assert "exc" not in msg, f"eval at line {call.lineno} message contains raw 'exc': {msg}"
                                assert "{" not in msg, f"eval at line {call.lineno} message contains format brace: {msg}"
                                assert msg.startswith("Redis command failed during "), (
                                    f"eval at line {call.lineno} message does not follow bounded pattern: {msg}"
                                )
                break
        assert "ResponseError" in caught_types and handler_found, f"eval at line {call.lineno} does not catch ResponseError"

    assert len(messages) == 17


@pytest.mark.anyio
async def test_redis_adapter_eval_response_error_handling() -> None:
    from unittest.mock import AsyncMock

    from agent_contracts import (
        AssignmentYieldRecord,
        DelegationGrant,
        DelegationProposalRecord,
        RootAuthorityRecord,
        WakeUpHint,
    )
    from orchestrator.domain.task import CorruptAuthoritativeRecordError
    from orchestrator.infrastructure.redis_adapter import RedisGateway
    from redis.exceptions import ResponseError

    gw = RedisGateway(url="redis://localhost:6379/0")
    mock_client = AsyncMock()
    gw._client = mock_client

    def _assert_corrupt_exc(exc_info: Any, op_name: str, sim_msg: str) -> None:
        msg = str(exc_info.value)
        assert op_name in msg
        assert sim_msg not in msg
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ResponseError)
        assert sim_msg in str(exc_info.value.__cause__)

    # 1. acknowledge_claim
    mock_client.eval.reset_mock()
    sim_err = "simulated acknowledge failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.acknowledge_claim("task-1", "asgn-1", "p-1", "2026-11-22T12:00:00Z")
    _assert_corrupt_exc(exc_info, "acknowledge_claim", sim_err)
    assert mock_client.eval.call_count == 1

    # 2. expire_delivery_lease
    mock_client.eval.reset_mock()
    sim_err = "simulated lease expiry failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.expire_delivery_lease("task-1")
    _assert_corrupt_exc(exc_info, "expire_delivery_lease", sim_err)
    assert mock_client.eval.call_count == 1

    # 3. accept_delegation_proposal
    mock_client.eval.reset_mock()
    sim_err = "simulated delegation failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    proposal = DelegationProposalRecord(
        parent_task_id="parent-1",
        proposal_id="prop-1",
        child_task_id="child-1",
        delegation_decision_id="dec-1",
        assignment_id="asgn-1",
        runtime_principal_id="p-1",
        bound_agent_id="agent-1",
        proof_digest="digest-1",
        proposal_fingerprint="fp-1",
        delegation_grant_id="grant-1",
        child_claim_deadline="2026-11-22T13:00:00.000000Z",
        accepted_at="2026-11-22T12:00:00.000000Z",
    )
    art_ref = ArtifactReference(artifact_id="art-1", artifact_version=1, producing_task_id="parent-1")
    child_input = TaskInput(
        task_id="child-1",
        input_version=1,
        prompt="p",
        created_at="2026-11-22T12:00:00.000000Z",
        artifact_references=[art_ref],
    )
    parent_continuation = TaskContinuation(
        parent_task_id="parent-1",
        child_task_id="child-1",
        created_at="2026-11-22T12:00:00.000000Z",
        updated_at="2026-11-22T12:00:00.000000Z",
        continuation_version=1,
        input_version=1,
    )
    yield_rec = AssignmentYieldRecord(
        assignment_id="asgn-1",
        parent_task_id="parent-1",
        principal_id="p-1",
        proposal_id="prop-1",
        child_task_id="child-1",
        delegation_decision_id="dec-1",
        yielded_at="2026-11-22T12:00:00.000000Z",
    )
    hint = WakeUpHint(
        hint_id="hint-1",
        capability="analysis",
        created_at="2026-11-22T12:00:00.000000Z",
        task_id="child-1",
    )
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.accept_delegation_proposal(
            "parent-1",
            proposal,
            child_input,
            parent_continuation,
            yield_rec,
            hint,
            expected_fencing_epoch=1,
            now="2026-11-22T12:00:00.000000Z",
        )
    _assert_corrupt_exc(exc_info, "accept_delegation_proposal", sim_err)
    assert mock_client.eval.call_count == 1

    # 4. complete_turn
    mock_client.eval.reset_mock()
    sim_err = "simulated complete_turn failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    cmd = CompletionCommand(
        completion_command_id="cmd-1",
        assignment_id="asgn-1",
        execution_kind="initial",
        failure_code="EXECUTION_FAILED",
    )
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.complete_turn(
            task_id="task-1",
            command=cmd,
            proof_digest="digest-1",
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id="task-1",
            completed_at="2026-11-22T12:00:00.000000Z",
            runtime_principal_id="p-1",
            bound_agent_id="a-1",
        )
    _assert_corrupt_exc(exc_info, "complete_turn", sim_err)
    assert mock_client.eval.call_count == 1

    # 5. register_root_task
    mock_client.eval.reset_mock()
    sim_err = "simulated register_root_task failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    task_req = TaskRequested(task_id="task-1", root_task_id="task-1", prompt="test", created_at="2026-11-22T12:00:00.000000Z")
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.register_root_task(task_req)
    _assert_corrupt_exc(exc_info, "register_root_task", sim_err)
    assert mock_client.eval.call_count == 1

    # 6. register_child_task
    mock_client.eval.reset_mock()
    sim_err = "simulated register_child_task failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    child_req = TaskRequested(
        task_id="child-1",
        prompt="test",
        parent_task_id="parent-1",
        root_task_id="root-1",
        created_at="2026-11-22T12:00:00.000000Z",
    )
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.register_child_task(child_req)
    _assert_corrupt_exc(exc_info, "register_child_task", sim_err)
    assert mock_client.eval.call_count == 1

    # 7. save_task_input (v1 and v2)
    mock_client.eval.reset_mock()
    sim_err = "simulated save_task_input v1 failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    t_input1 = TaskInput(task_id="task-1", input_version=1, prompt="v1", created_at="2026-11-22T12:00:00.000000Z")
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.save_task_input(t_input1)
    _assert_corrupt_exc(exc_info, "save_task_input", sim_err)
    assert mock_client.eval.call_count == 1

    mock_client.eval.reset_mock()
    sim_err = "simulated save_task_input v2 failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    t_input2 = TaskInput(task_id="task-1", input_version=2, prompt="v2", created_at="2026-11-22T12:00:00.000000Z")
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.save_task_input(t_input2)
    _assert_corrupt_exc(exc_info, "save_task_input", sim_err)
    assert mock_client.eval.call_count == 1

    # 8. save_stage_artifact
    mock_client.eval.reset_mock()
    sim_err = "simulated save_stage_artifact failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    art = StageArtifact(
        artifact_id="art-1",
        producing_task_id="task-1",
        artifact_type="research_material",
        content={"material": "test"},
        created_at="2026-11-22T12:00:00.000000Z",
        artifact_version=1,
    )
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.save_stage_artifact(art, "asgn-1", "p-1", 1)
    _assert_corrupt_exc(exc_info, "save_stage_artifact", sim_err)
    assert mock_client.eval.call_count == 1

    # 9. save_continuation (v1 and v2)
    mock_client.eval.reset_mock()
    sim_err = "simulated save_continuation v1 failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    cont1 = TaskContinuation(
        parent_task_id="parent-1",
        child_task_id="child-1",
        created_at="2026-11-22T12:00:00.000000Z",
        updated_at="2026-11-22T12:00:00.000000Z",
        continuation_version=1,
    )
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.save_continuation(cont1)
    _assert_corrupt_exc(exc_info, "save_continuation", sim_err)
    assert mock_client.eval.call_count == 1

    mock_client.eval.reset_mock()
    sim_err = "simulated save_continuation v2 failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    cont2 = TaskContinuation(
        parent_task_id="parent-1",
        child_task_id="child-1",
        created_at="2026-11-22T12:00:00.000000Z",
        updated_at="2026-11-22T12:00:00.000000Z",
        continuation_version=2,
    )
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.save_continuation(cont2)
    _assert_corrupt_exc(exc_info, "save_continuation", sim_err)
    assert mock_client.eval.call_count == 1

    # 10. save_root_authority
    mock_client.eval.reset_mock()
    sim_err = "simulated save_root_authority failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    r_auth = RootAuthorityRecord(root_task_id="task-1", created_at="2026-11-22T12:00:00.000000Z")
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.save_root_authority(r_auth)
    _assert_corrupt_exc(exc_info, "save_root_authority", sim_err)
    assert mock_client.eval.call_count == 1

    # 11. save_authority_grant
    mock_client.eval.reset_mock()
    sim_err = "simulated save_authority_grant failure"
    mock_client.eval.side_effect = ResponseError(sim_err)
    grant = DelegationGrant(
        grant_id="grant-1",
        root_task_id="task-1",
        issued_to_assignment_id="asgn-1",
        issued_to_principal_id="p-1",
        fencing_epoch=1,
        turn_id="turn-1",
        issued_at="2026-11-22T12:00:00.000000Z",
    )
    with pytest.raises(CorruptAuthoritativeRecordError) as exc_info:
        await gw.save_authority_grant(grant)
    _assert_corrupt_exc(exc_info, "save_authority_grant", sim_err)
    assert mock_client.eval.call_count == 1


@pytest.mark.anyio
async def test_transition_expired_claim_preserves_explicit_now_and_propagates_typeerror() -> None:
    from unittest.mock import AsyncMock

    from agent_contracts import ClaimExpiryRecord
    from orchestrator.infrastructure.redis_adapter import RedisGateway

    gw = RedisGateway(url="redis://localhost:6379/0")
    mock_client = AsyncMock()
    gw._client = mock_client

    # 1. When active claim is absent, replay/status detection uses the explicit time,
    # wall-clock time is never substituted, and existing ClaimExpiryRecord is returned.
    gw.get_active_claim = AsyncMock(return_value=None)
    mock_expiry = ClaimExpiryRecord(
        task_id="task-exp-1",
        claim_id="claim-1",
        assignment_id="asgn-1",
        principal_id="p-1",
        bound_agent_id="a-1",
        execution_kind="initial",
        expired_fencing_epoch=1,
        acknowledged_at="2026-11-22T09:00:00.000000Z",
        execution_expires_at="2026-11-22T09:30:00.000000Z",
        expired_at="2026-11-22T10:00:00.000000Z",
        prior_claimable_status="pending",
    )
    gw.get_claim_expiry = AsyncMock(return_value=mock_expiry)
    gw.get_status = AsyncMock(return_value="recovery_required")

    fixed_now = "2026-11-22T10:00:00.000000Z"
    rec, is_replayed = await gw.transition_expired_claim("task-exp-1", now=fixed_now)

    assert is_replayed is True
    assert rec.task_id == "task-exp-1"
    assert rec.claim_id == "claim-1"
    # Verify get_status was awaited with the exact explicit now_iso timestamp, never substituting wall clock
    gw.get_status.assert_awaited_once_with("task-exp-1", now=fixed_now)

    # 2. When internal get_status raises a TypeError, it is NOT swallowed or retried
    gw.get_status.reset_mock()
    gw.get_status.side_effect = TypeError("internal store TypeError bug")
    with pytest.raises(TypeError, match="internal store TypeError bug"):
        await gw.transition_expired_claim("task-exp-1", now=fixed_now)
    assert gw.get_status.call_count == 1


@pytest.mark.anyio
async def test_api_boundary_corrupt_authoritative_record_error_sanitization() -> None:
    from orchestrator.domain.task import CorruptAuthoritativeRecordError

    store = InMemoryAuthoritativeTaskStore()
    test_settings = Settings(
        redis_url="redis://fake:6379/9",
        app_env="test",
        task_delivery_mode="authenticated_claim",
        runtime_credential_pepper="pepper" * 8,
        assignment_proof_pepper="proof" * 8,
        runtime_principals={
            "p-operator": {
                "principal_id": "p-operator",
                "bound_agent_id": "op-1",
                "authorized_capabilities": ["recovery_operator"],
                "token_digest": compute_token_digest("op-secret", "pepper" * 8),
                "active": True,
            },
        },
    )
    app = create_app(
        settings=test_settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    async def corrupt_get_status(task_id: str, now: str = "") -> str | None:
        raise CorruptAuthoritativeRecordError("Simulated internal corrupt record detail with internal state")

    store.get_status = corrupt_get_status

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        res = await client.post(
            "/api/tasks/task-1/evaluate-expiry",
            headers={"Authorization": "Bearer op-secret"},
            json={"now": "2026-11-22T12:00:00.000000Z"},
        )
        assert res.status_code == 500
        # Verify raw error detail is never leaked across the HTTP API boundary
        data = res.json()
        assert data["detail"] == "Authoritative record corruption detected"
        assert "Simulated internal corrupt record" not in res.text


@pytest.mark.anyio
async def test_reader_optional_now_protocol_and_typeerror_propagation() -> None:
    from unittest.mock import AsyncMock

    from orchestrator.application.queries import GetTaskResult
    from orchestrator.presentation.api import NoOpResultReader

    from tests.test_application import FakeResultReader

    # 1. FakeResultReader accepts optional now and works with or without it
    fake_reader = FakeResultReader(results={"t1": {"task_id": "t1", "status": "completed", "output": "ok"}})
    res_without = await fake_reader.get_result("t1")
    assert res_without == {"task_id": "t1", "status": "completed", "output": "ok"}
    res_with = await fake_reader.get_result("t1", now="2026-11-22T12:00:00.000000Z")
    assert res_with == {"task_id": "t1", "status": "completed", "output": "ok"}

    # 2. NoOpResultReader accepts optional now and works with or without it
    noop = NoOpResultReader()
    assert await noop.get_result("t1") is None
    assert await noop.get_result("t1", now="2026-11-22T12:00:00.000000Z") is None

    # 3. GetTaskResult forwards now parameter
    mock_res_reader = AsyncMock()
    mock_res_reader.get_result.return_value = {"task_id": "t1", "status": "completed"}
    query = GetTaskResult(mock_res_reader)

    res_q1 = await query.execute("t1")
    assert res_q1 == {"task_id": "t1", "status": "completed"}
    mock_res_reader.get_result.assert_awaited_once_with("t1", now="")

    mock_res_reader.get_result.reset_mock()
    res_q2 = await query.execute("t1", now="2026-11-22T14:00:00.000000Z")
    assert res_q2 == {"task_id": "t1", "status": "completed"}
    mock_res_reader.get_result.assert_awaited_once_with("t1", now="2026-11-22T14:00:00.000000Z")

    # 4. EvaluateExpiry lets internal TypeError propagate without swallowing or retrying
    store = InMemoryAuthoritativeTaskStore()

    async def buggy_get_status(task_id: str, now: str = "") -> str | None:
        raise TypeError("internal bug inside get_status implementation")

    store.get_status = buggy_get_status
    use_case = EvaluateExpiry(store)
    op_principal = RuntimePrincipal(
        principal_id="rec-op",
        bound_agent_id="op-1",
        authorized_capabilities=("recovery_operator",),
        active=True,
    )

    with pytest.raises(TypeError, match="internal bug inside get_status implementation"):
        await use_case.execute(op_principal, "task-1", now="2026-11-22T12:00:00.000000Z")

    # 5. EvaluateExpiry deterministic evaluation with explicit now
    eval_store = InMemoryAuthoritativeTaskStore()
    eval_use_case = EvaluateExpiry(eval_store)
    t_deadline = datetime(2026, 11, 22, 12, 0, 0, tzinfo=UTC)
    eval_store.lineages["child-det"] = TaskLineage(
        task_id="child-det",
        root_task_id="parent-det",
        parent_task_id="parent-det",
    )
    eval_store.statuses["child-det"] = "pending"
    eval_store.claim_deadlines["child-det"] = t_deadline.isoformat()
    eval_store.statuses["parent-det"] = "waiting_for_children"
    eval_store.children["parent-det"] = {"child-det"}
    eval_store.latest_continuation["parent-det"] = 1
    eval_store.continuations[("parent-det", 1)] = TaskContinuation(
        parent_task_id="parent-det",
        child_task_id="child-det",
        created_at=t_deadline.isoformat(),
        updated_at=t_deadline.isoformat(),
        continuation_version=1,
    )

    # Pre-deadline
    before = (t_deadline - timedelta(seconds=1)).isoformat()
    res_before = await eval_use_case.execute(op_principal, "child-det", now=before)
    assert res_before.action == "none"
    assert res_before.status == "pending"

    # Post-deadline
    after = (t_deadline + timedelta(seconds=1)).isoformat()
    res_after = await eval_use_case.execute(op_principal, "child-det", now=after)
    assert res_after.action == "claim_timed_out"
    assert res_after.status == "claim_timed_out"
