import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agent_contracts import (
    MAX_WAVE_DURATION_SECONDS,
    MIN_WAVE_DURATION_SECONDS,
    AcceptedDelegationWave,
    ArtifactReference,
    CompletionCommand,
    DelegationWaveProposalItem,
    DelegationWaveProposalRequest,
    RecoveryDecisionCommand,
    RootAuthorityRecord,
    TaskContinuationV2,
    TaskLifecycleStatus,
    TaskRequested,
    TaskResult,
    WaveCancellationCommand,
    WaveCancellationRecord,
    WaveResolution,
    WaveTimeoutEvaluationCommand,
    compute_wave_cancellation_fingerprint,
    compute_wave_timeout_fingerprint,
    derive_wave_deadline,
    normalize_canonical_utc_timestamp,
)
from fastapi.testclient import TestClient
from orchestrator.application.ports import (
    CapabilityMismatchError,
    ClaimNotFoundError,
    CompletionReplayAuthorizationError,
    InvalidClaimProofError,
    RecoveryPreconditionFailedError,
    TaskAlreadyClaimedError,
    TaskNotClaimableError,
    WaveAlreadyResolvedError,
    WaveCancellationConflictError,
    WaveDeadlineNotReachedError,
)
from orchestrator.application.use_cases import (
    AcceptDelegationWave,
    AcknowledgeAssignment,
    CancelDelegationWave,
    ClaimTask,
    CompleteTurn,
    EvaluateExpiry,
    EvaluateWaveTimeout,
    GetResumedContinuationContext,
    ResolveExpiredClaim,
    StageArtifactUseCase,
    UnauthorizedPrincipalError,
)
from orchestrator.config import Settings
from orchestrator.domain.principal import (
    RECOGNIZED_CAPABILITIES,
    RuntimePrincipal,
    compute_token_digest,
)
from orchestrator.domain.task import CorruptAuthoritativeRecordError, TaskLineage
from orchestrator.presentation.api import create_app

from tests.test_authoritative_task_store import (
    InMemoryAuthoritativeTaskStore,
)

PEPPER = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
CRED_PEPPER = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"


def _principal(
    principal_id: str,
    bound_agent_id: str,
    capabilities: tuple[str, ...],
    *,
    active: bool = True,
) -> RuntimePrincipal:
    return RuntimePrincipal(
        principal_id=principal_id,
        bound_agent_id=bound_agent_id,
        authorized_capabilities=capabilities,
        active=active,
    )


def _wave_items(
    parent_id: str,
    *,
    count: int = 2,
    art_id: str = "art-stage-1",
    control_fields: dict[str, Any] | None = None,
) -> tuple[DelegationWaveProposalItem, ...]:
    ref = ArtifactReference(artifact_id=art_id, producing_task_id=parent_id, artifact_version=1)
    return tuple(
        DelegationWaveProposalItem(
            item_id=f"item-{idx}",
            child_intent=f"Analyze branch {idx}",
            artifact_reference=ref,
            control_fields=dict(control_fields or {}),
        )
        for idx in range(1, count + 1)
    )


async def _setup_acknowledged_wave_parent(
    store: Any,
    *,
    parent_id: str = "root-inc4-1",
    max_children: int = 4,
    now_dt: datetime,
    set_authority_fn: Any = None,
) -> tuple[RuntimePrincipal, str, str, str, ArtifactReference]:
    principal = _principal("prin-res-1", "agent-res-1", ("research",))
    await store.register_root_task(
        TaskRequested(
            task_id=parent_id,
            root_task_id=parent_id,
            parent_task_id=None,
            prompt="Research parent for increment 4",
            created_at=normalize_canonical_utc_timestamp(now_dt),
            version=2,
        ),
        initial_status="pending",
        required_capability="research",
    )
    root_auth = RootAuthorityRecord.create_wave_baseline(
        parent_id,
        max_children_per_wave=max_children,
        created_at=normalize_canonical_utc_timestamp(now_dt),
    )
    if set_authority_fn is not None:
        await set_authority_fn(parent_id, root_auth)
    else:
        store.authorities[parent_id] = root_auth

    claim_uc = ClaimTask(
        store,
        PEPPER,
        clock=lambda: now_dt,
        allow_wave_delegation_grant=True,
    )
    claimed = await claim_uc.execute(principal, parent_id, f"req-{parent_id}", "research", "initial")
    ack_uc = AcknowledgeAssignment(store, PEPPER, clock=lambda: now_dt + timedelta(seconds=1))
    await ack_uc.execute(
        principal, parent_id, claimed.assignment.assignment_id, claimed.assignment_proof
    )

    stage_uc = StageArtifactUseCase(store, PEPPER, clock=lambda: now_dt + timedelta(seconds=1))
    art = await stage_uc.execute(
        principal=principal,
        task_id=parent_id,
        assignment_id=claimed.assignment.assignment_id,
        assignment_proof=claimed.assignment_proof,
        artifact_id=f"art-{parent_id}",
        artifact_type="research_material",
        content={"material": "wave input"},
    )
    ref = ArtifactReference(artifact_id=art.artifact_id, producing_task_id=parent_id, artifact_version=1)
    return principal, parent_id, claimed.assignment.assignment_id, claimed.assignment_proof, ref


async def _accept_wave(
    store: Any,
    *,
    parent_id: str,
    max_children: int = 3,
    barrier_policy: str = "WAIT_ALL",
    required_success_fraction: dict[str, int] | None = None,
    proposal_id: str = "prop-inc4-1",
    now_dt: datetime,
    child_claim_deadline_seconds: int = 300,
    set_authority_fn: Any = None,
) -> dict[str, Any]:
    res_p, pid, assign_id, proof, ref = await _setup_acknowledged_wave_parent(
        store,
        parent_id=parent_id,
        max_children=max(2, max_children),
        now_dt=now_dt,
        set_authority_fn=set_authority_fn,
    )
    wave_uc = AcceptDelegationWave(
        store,
        PEPPER,
        child_claim_deadline_seconds=child_claim_deadline_seconds,
        clock=lambda: now_dt + timedelta(seconds=2),
    )
    req = DelegationWaveProposalRequest(
        proposal_id=proposal_id,
        items=_wave_items(pid, count=max_children, art_id=ref.artifact_id),
        barrier_policy=barrier_policy,
        required_success_fraction=required_success_fraction,
    )
    resp = await wave_uc.execute(res_p, pid, assign_id, proof, req)
    return {
        "principal": res_p,
        "parent_id": pid,
        "assign_id": assign_id,
        "proof": proof,
        "art_ref": ref,
        "proposal": req,
        "response": resp,
        "wave_id": resp.wave_id,
        "children": resp.required_child_task_ids,
    }


async def _claim_and_ack_child(
    store: Any,
    child_id: str,
    now_dt: datetime,
    *,
    principal_suffix: str = "1",
    execution_claim_timeout_seconds: int = 120,
) -> tuple[RuntimePrincipal, Any]:
    ap = _principal(
        f"prin-analysis-{principal_suffix}",
        f"agent-analysis-{principal_suffix}",
        ("analysis",),
    )
    claimed = await ClaimTask(store, PEPPER, clock=lambda: now_dt).execute(
        ap, child_id, f"req-{child_id}", "analysis", "delegated_child"
    )
    await AcknowledgeAssignment(
        store,
        PEPPER,
        execution_claim_timeout_seconds=execution_claim_timeout_seconds,
        clock=lambda: now_dt + timedelta(seconds=1),
    ).execute(ap, child_id, claimed.assignment.assignment_id, claimed.assignment_proof)
    return ap, claimed


# ---------------------------------------------------------------------------
# A. Contracts, Capabilities, and TIMED_OUT Wave Deadline Evaluation
# ---------------------------------------------------------------------------


def test_increment_4_contracts_and_capabilities() -> None:
    assert "wave_cancel_operator" in RECOGNIZED_CAPABILITIES
    assert TaskLifecycleStatus.CANCELLED.value == "cancelled"

    timeout_cmd = WaveTimeoutEvaluationCommand(parent_task_id="root-1", wave_id="wave-1")
    assert timeout_cmd.semantic_fingerprint() == compute_wave_timeout_fingerprint(
        parent_task_id="root-1", wave_id="wave-1"
    )
    with pytest.raises(ValueError):
        WaveTimeoutEvaluationCommand.from_dict(
            {"parent_task_id": "root-1", "wave_id": "wave-1", "wave_deadline": "2026-11-27T12:00:00Z"}
        )

    cancel_cmd = WaveCancellationCommand(
        cancellation_decision_id="cancel-1",
        parent_task_id="root-1",
        wave_id="wave-1",
    )
    assert cancel_cmd.semantic_fingerprint() == compute_wave_cancellation_fingerprint(
        parent_task_id="root-1", wave_id="wave-1"
    )
    with pytest.raises(ValueError):
        WaveCancellationCommand.from_dict(
            {
                "cancellation_decision_id": "cancel-1",
                "parent_task_id": "root-1",
                "wave_id": "wave-1",
                "status": "cancelled",
            }
        )

    rec = WaveCancellationRecord(
        cancellation_decision_id="cancel-1",
        parent_task_id="root-1",
        wave_id="wave-1",
        operator_principal_id="op-cancel-1",
        semantic_fingerprint=cancel_cmd.semantic_fingerprint(),
        cancelled_at="2026-11-27T10:05:00Z",
    )
    assert WaveCancellationRecord.from_json(rec.to_json()) == rec
    with pytest.raises(ValueError):
        WaveCancellationRecord(
            cancellation_decision_id="cancel-1",
            parent_task_id="root-1",
            wave_id="wave-1",
            operator_principal_id="op-cancel-1",
            semantic_fingerprint="bad-fp",
            cancelled_at="2026-11-27T10:05:00Z",
        )


@pytest.mark.anyio
async def test_evaluate_wave_timeout_authorization_early_rejection_and_atomic_resolution() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    ctx = await _accept_wave(store, parent_id="root-to-1", max_children=3, now_dt=dt)
    parent_id = ctx["parent_id"]
    wave_id = ctx["wave_id"]
    resp = ctx["response"]
    c1, c2, c3 = ctx["children"]

    # Complete child 1; claim+ack child 2; leave child 3 pending (fence 0).
    p1, cl1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=5), principal_suffix="1")
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=8)).execute(
        p1,
        c1,
        CompletionCommand(
            completion_command_id="cmd-to-c1",
            assignment_id=cl1.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c1, output="branch 1 done"),
        ).to_dict(),
        cl1.assignment_proof,
    )
    await _claim_and_ack_child(store, c2, dt + timedelta(seconds=6), principal_suffix="2")
    assert store.fences[c1] == 1
    assert store.fences[c2] == 1
    assert store.fences[c3] == 0

    rec_op = _principal("prin-rec-op", "agent-rec-op", ("recovery_operator",))
    non_op = _principal("prin-cancel-only", "agent-cancel", ("wave_cancel_operator",))
    research_p = _principal("prin-res-x", "agent-res-x", ("research",))
    inactive_op = _principal("prin-inactive", "agent-inactive", ("recovery_operator",), active=False)

    # Authorization checks
    eval_uc = EvaluateWaveTimeout(store)
    with pytest.raises(UnauthorizedPrincipalError):
        await eval_uc.execute(inactive_op, parent_id, wave_id, now=resp.wave_deadline)
    with pytest.raises(CapabilityMismatchError):
        await eval_uc.execute(non_op, parent_id, wave_id, now=resp.wave_deadline)
    with pytest.raises(CapabilityMismatchError):
        await eval_uc.execute(research_p, parent_id, wave_id, now=resp.wave_deadline)

    # Early evaluation (strictly before frozen wave_deadline) fails closed with zero mutation
    early_dt = datetime.fromisoformat(resp.wave_deadline) - timedelta(seconds=1)
    hints_before = len(store.hints)
    with pytest.raises(WaveDeadlineNotReachedError):
        await eval_uc.execute(
            rec_op,
            parent_id,
            wave_id,
            now=normalize_canonical_utc_timestamp(early_dt),
        )
    assert store.statuses[parent_id] == "waiting_for_children"
    assert (parent_id, wave_id) not in store.resolved_waves
    assert len(store.hints) == hints_before

    # Exact deadline boundary triggers TIMED_OUT resolution
    parent_fence_before = store.fences[parent_id]
    rw, replayed = await eval_uc.execute(rec_op, parent_id, wave_id, now=resp.wave_deadline)
    assert replayed is False
    assert rw.resolution == WaveResolution.TIMED_OUT.value
    assert rw.completed_count == 1
    assert rw.terminal_count == 1
    assert rw.unresolved_count == 2
    assert rw.residual_child_task_ids == (c2, c3)
    assert rw.residual_disposition == "ended / barrier_resolved"
    assert rw.triggering_child_task_id is None
    assert rw.triggering_assignment_id is None
    assert rw.triggering_fencing_epoch is None
    assert rw.child_outcomes[c1]["status"] == "completed"
    assert rw.child_outcomes[c2] == {"outcome_id": None, "status": "ended / barrier_resolved"}
    assert rw.child_outcomes[c3] == {"outcome_id": None, "status": "ended / barrier_resolved"}

    # Atomic residual closure & fence preservation for fence=0 never-claimed child
    assert store.statuses[c1] == "completed"
    assert store.statuses[c2] == "ended / barrier_resolved"
    assert store.statuses[c3] == "ended / barrier_resolved"
    assert c2 not in store.claims and c3 not in store.claims
    assert store.fences[c2] == 2  # positive fence bumped
    assert store.fences[c3] == 0  # never-claimed fence zero preserved!
    assert store.fences[parent_id] == parent_fence_before
    assert parent_id not in store.claims
    assert parent_id not in store.terminal_outcomes
    assert parent_id not in store.canonical_results
    assert c2 not in store.terminal_outcomes and c3 not in store.terminal_outcomes
    assert c2 not in store.claim_expiries and c3 not in store.claim_expiries

    # Parent resumable + 1 Research hint + released continuation v2
    assert store.statuses[parent_id] == "resumable"
    cont = store.continuations[(parent_id, 2)]
    assert cont.state == "released"
    assert cont.wave_resolution == "TIMED_OUT"
    assert len(store.hints) == hints_before + 1
    assert store.hints[-1].execution_kind == "resumed_continuation"
    assert store.hints[-1].expected_continuation_version == 2

    # Exact replay returns same TIMED_OUT decision with zero extra hints
    rw2, replayed2 = await eval_uc.execute(
        rec_op,
        parent_id,
        wave_id,
        now=normalize_canonical_utc_timestamp(
            datetime.fromisoformat(resp.wave_deadline) + timedelta(seconds=30)
        ),
    )
    assert replayed2 is True
    assert rw2 == rw
    assert len(store.hints) == hints_before + 1

    # Post-TIMED_OUT cancellation must fail with WaveAlreadyResolvedError
    cancel_op = _principal("prin-cancel-op", "agent-cancel-op", ("wave_cancel_operator",))
    with pytest.raises(WaveAlreadyResolvedError):
        await CancelDelegationWave(store).execute(
            cancel_op,
            WaveCancellationCommand(
                cancellation_decision_id="cancel-after-to",
                parent_task_id=parent_id,
                wave_id=wave_id,
            ),
        )


# ---------------------------------------------------------------------------
# B. Authorized Cancellation & Terminal 'cancelled' Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cancel_delegation_wave_authorization_lifecycle_and_replay() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    ctx = await _accept_wave(store, parent_id="root-cancel-1", max_children=3, now_dt=dt)
    parent_id = ctx["parent_id"]
    wave_id = ctx["wave_id"]
    c1, c2, c3 = ctx["children"]

    # Claim+ack child 1 & complete it; claim+ack child 2; leave child 3 pending (fence 0).
    p1, cl1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=5), principal_suffix="1")
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=7)).execute(
        p1,
        c1,
        CompletionCommand(
            completion_command_id="cmd-cancel-c1",
            assignment_id=cl1.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c1, output="c1 done"),
        ).to_dict(),
        cl1.assignment_proof,
    )
    await _claim_and_ack_child(store, c2, dt + timedelta(seconds=6), principal_suffix="2")

    cancel_uc = CancelDelegationWave(store, clock=lambda: dt + timedelta(seconds=20))
    cmd = WaveCancellationCommand(
        cancellation_decision_id="cancel-dec-1",
        parent_task_id=parent_id,
        wave_id=wave_id,
    )

    # Reject recovery_operator, research, analysis, worker, inactive
    for bad_p in (
        _principal("p-rec", "a-rec", ("recovery_operator",)),
        _principal("p-res", "a-res", ("research",)),
        _principal("p-ana", "a-ana", ("analysis",)),
        _principal("p-wrk", "a-wrk", ("worker",)),
    ):
        with pytest.raises(CapabilityMismatchError):
            await cancel_uc.execute(bad_p, cmd)
    with pytest.raises(UnauthorizedPrincipalError):
        await cancel_uc.execute(
            _principal("p-inact", "a-inact", ("wave_cancel_operator",), active=False),
            cmd,
        )

    cancel_op = _principal("prin-wave-cancel-1", "agent-cancel-1", ("wave_cancel_operator",))
    hints_before = len(store.hints)
    dec, rw, replayed = await cancel_uc.execute(cancel_op, cmd)
    assert replayed is False
    assert isinstance(dec, WaveCancellationRecord)
    assert dec.cancellation_decision_id == "cancel-dec-1"
    assert dec.operator_principal_id == "prin-wave-cancel-1"
    assert dec.prior_parent_status == "waiting_for_children"
    assert dec.terminal_parent_status == "cancelled"
    assert dec.resolution == "CANCELLED"

    assert rw.resolution == "CANCELLED"
    assert rw.residual_child_task_ids == (c2, c3)
    assert rw.child_outcomes[c1]["status"] == "completed"
    assert rw.child_outcomes[c2] == {"outcome_id": None, "status": "ended / barrier_resolved"}
    assert rw.child_outcomes[c3] == {"outcome_id": None, "status": "ended / barrier_resolved"}

    # Parent cancelled cleanly with zero hints, outcomes, or results
    assert await store.get_status(parent_id) == "cancelled"
    assert await store.get_result(parent_id) == {
        "task_id": parent_id,
        "status": "cancelled",
        "output": None,
    }
    assert parent_id not in store.terminal_outcomes
    assert parent_id not in store.canonical_results
    assert parent_id not in store.claims
    assert len(store.hints) == hints_before

    # Continuation closed and non-claimable
    cont = store.continuations[(parent_id, 2)]
    assert cont.state == "closed"
    assert cont.wave_resolution is None

    # Residual fences: claimed child bumped, never-claimed fence 0 preserved
    assert store.fences[c2] == 2
    assert store.fences[c3] == 0
    assert store.statuses[c2] == "ended / barrier_resolved"
    assert store.statuses[c3] == "ended / barrier_resolved"
    assert c2 not in store.claims and c3 not in store.claims

    # Exact replay with same operator + decision ID + wave returns existing decision
    dec2, rw2, replayed2 = await cancel_uc.execute(cancel_op, cmd)
    assert replayed2 is True
    assert dec2 == dec
    assert rw2 == rw
    assert len(store.hints) == hints_before

    # Different operator with same decision_id -> conflict
    other_cancel_op = _principal("prin-wave-cancel-2", "agent-cancel-2", ("wave_cancel_operator",))
    with pytest.raises(WaveCancellationConflictError):
        await cancel_uc.execute(other_cancel_op, cmd)

    # New decision_id on already-cancelled wave -> WaveAlreadyResolvedError
    with pytest.raises(WaveAlreadyResolvedError):
        await cancel_uc.execute(
            cancel_op,
            WaveCancellationCommand(
                cancellation_decision_id="cancel-dec-2",
                parent_task_id=parent_id,
                wave_id=wave_id,
            ),
        )

    # Cancelled parent cannot be claimed via resumed_continuation or initial or delegated_child
    for kind, cap, ver in (
        ("resumed_continuation", "research", 2),
        ("initial", "research", None),
        ("delegated_child", "analysis", None),
    ):
        with pytest.raises((TaskNotClaimableError, CapabilityMismatchError)):
            await ClaimTask(store, PEPPER, clock=lambda: dt + timedelta(seconds=25)).execute(
                _principal("p-any", "a-any", (cap,)),
                parent_id,
                f"req-reject-{kind}",
                cap,
                kind,
                expected_continuation_version=ver,
            )


def test_api_routes_wave_timeout_and_cancellation() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)

    async def _prep() -> dict[str, Any]:
        return await _accept_wave(store, parent_id="root-api-1", max_children=2, now_dt=dt)

    ctx = asyncio.run(_prep())
    parent_id = ctx["parent_id"]
    wave_id = ctx["wave_id"]

    principals = {
        "op-rec": {
            "bound_agent_id": "agent-op-rec",
            "authorized_capabilities": ("recovery_operator",),
            "token_digest": compute_token_digest("tok-rec", CRED_PEPPER),
            "active": True,
        },
        "op-can": {
            "bound_agent_id": "agent-op-can",
            "authorized_capabilities": ("wave_cancel_operator",),
            "token_digest": compute_token_digest("tok-can", CRED_PEPPER),
            "active": True,
        },
    }
    settings = Settings(
        assignment_proof_pepper=PEPPER,
        runtime_credential_pepper=CRED_PEPPER,
        task_delivery_mode="authenticated_claim",
        runtime_principals=principals,
    )
    app = create_app(settings=settings, task_store=store, task_queue=store)
    client = TestClient(app)

    # Missing auth -> 401
    r401 = client.post(f"/api/tasks/{parent_id}/waves/{wave_id}/evaluate-timeout", json={})
    assert r401.status_code == 401

    # Wrong capability for cancel -> 403
    r403 = client.post(
        f"/api/tasks/{parent_id}/waves/{wave_id}/cancel",
        headers={"Authorization": "Bearer tok-rec"},
        json={"cancellation_decision_id": "api-cancel-1"},
    )
    assert r403.status_code == 403

    # Wrong capability for timeout -> 403
    r403_to = client.post(
        f"/api/tasks/{parent_id}/waves/{wave_id}/evaluate-timeout",
        headers={"Authorization": "Bearer tok-can"},
        json={},
    )
    assert r403_to.status_code == 403

    # Early timeout -> 409
    r409 = client.post(
        f"/api/tasks/{parent_id}/waves/{wave_id}/evaluate-timeout",
        headers={"Authorization": "Bearer tok-rec"},
        json={"now": normalize_canonical_utc_timestamp(dt + timedelta(seconds=5))},
    )
    assert r409.status_code == 409

    # Cancel authorized -> 200 and replay 200
    r200 = client.post(
        f"/api/tasks/{parent_id}/waves/{wave_id}/cancel",
        headers={"Authorization": "Bearer tok-can"},
        json={"cancellation_decision_id": "api-cancel-1"},
    )
    assert r200.status_code == 200
    body = r200.json()
    assert body["replayed"] is False
    assert body["resolved_wave"]["resolution"] == "CANCELLED"
    assert body["cancellation"]["terminal_parent_status"] == "cancelled"

    r200_rep = client.post(
        f"/api/tasks/{parent_id}/waves/{wave_id}/cancel",
        headers={"Authorization": "Bearer tok-can"},
        json={"cancellation_decision_id": "api-cancel-1"},
    )
    assert r200_rep.status_code == 200
    assert r200_rep.json()["replayed"] is True

    # Result endpoint reports cancelled cleanly
    r_res = client.get(f"/api/tasks/{parent_id}/result")
    assert r_res.status_code == 200
    assert r_res.json() == {"task_id": parent_id, "status": "cancelled", "output": None}


# ---------------------------------------------------------------------------
# C. Wave-Aware Child Claim Timeout & Barrier Evaluation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_wave_child_claim_timeout_non_decisive_and_decisive_partial_unsat() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    ctx = await _accept_wave(
        store,
        parent_id="root-ct-1",
        max_children=3,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 2, "denominator": 3},
        now_dt=dt,
        child_claim_deadline_seconds=60,
    )
    parent_id = ctx["parent_id"]
    wave_id = ctx["wave_id"]
    resp = ctx["response"]
    c1, c2, c3 = ctx["children"]

    # Child 1 claims before deadline and stays working (fence=1). Child 2 & 3 remain pending (fence=0).
    await _claim_and_ack_child(store, c1, dt + timedelta(seconds=10), principal_suffix="1")
    hints_before = len(store.hints)

    # Deadline exact timestamp is not expired (now <= deadline)
    rec_op = _principal("op-ct", "agent-op-ct", ("recovery_operator",))
    eval_exp = EvaluateExpiry(store, clock=lambda: datetime.fromisoformat(resp.child_claim_deadline))
    res_not_exp = await eval_exp.execute(rec_op, c2)
    assert res_not_exp.action == "none"
    assert res_not_exp.status == "pending"

    # Past deadline: Child 2 claim times out -> 1 failure still leaves threshold 2/3 reachable (1 working + 1 pending >= 2)
    after_dl = datetime.fromisoformat(resp.child_claim_deadline) + timedelta(seconds=5)
    out2, replayed2 = await store.materialize_claim_timeout(
        c2, now=normalize_canonical_utc_timestamp(after_dl)
    )
    assert replayed2 is False
    assert out2.terminal_status == "claim_timed_out"
    assert c2 not in store.canonical_results
    assert store.child_markers[c2]["continuation_version"] == 2
    assert store.statuses[parent_id] == "waiting_for_children"
    assert (parent_id, wave_id) not in store.resolved_waves
    assert len(store.hints) == hints_before

    # Replay before wave resolution
    out2_rep, rep2_flag = await store.materialize_claim_timeout(
        c2, now=normalize_canonical_utc_timestamp(after_dl + timedelta(seconds=1))
    )
    assert rep2_flag is True
    assert out2_rep == out2

    # Second claim timeout on Child 3 -> now max possible successes = 1 (< 2) -> PARTIAL barrier UNSATISFIABLE!
    out3, replayed3 = await store.materialize_claim_timeout(
        c3, now=normalize_canonical_utc_timestamp(after_dl + timedelta(seconds=2))
    )
    assert replayed3 is False
    assert out3.terminal_status == "claim_timed_out"
    rw = store.resolved_waves[(parent_id, wave_id)]
    assert rw.resolution == "UNSATISFIABLE"
    assert rw.triggering_child_task_id == c3
    assert rw.triggering_assignment_id is None
    assert rw.triggering_fencing_epoch is None
    assert rw.residual_child_task_ids == (c1,)
    assert store.statuses[c1] == "ended / barrier_resolved"
    assert c1 not in store.claims
    assert store.fences[c1] == 2  # active fence bumped
    assert store.statuses[parent_id] == "resumable"
    assert store.continuations[(parent_id, 2)].wave_resolution == "UNSATISFIABLE"
    assert len(store.hints) == hints_before + 1

    # Post-resolution exact replay on c2 & c3 returns stored outcome without another hint
    _, rep2_post = await store.materialize_claim_timeout(c2, now=normalize_canonical_utc_timestamp(after_dl + timedelta(seconds=5)))
    _, rep3_post = await store.materialize_claim_timeout(c3, now=normalize_canonical_utc_timestamp(after_dl + timedelta(seconds=5)))
    assert rep2_post is True and rep3_post is True
    assert len(store.hints) == hints_before + 1


@pytest.mark.anyio
async def test_wave_child_claim_timeout_wait_all_mixed_completion_and_timeout_resolves_satisfied() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    ctx = await _accept_wave(
        store,
        parent_id="root-ct-wa",
        max_children=2,
        barrier_policy="WAIT_ALL",
        now_dt=dt,
        child_claim_deadline_seconds=60,
    )
    parent_id = ctx["parent_id"]
    wave_id = ctx["wave_id"]
    resp = ctx["response"]
    c1, c2 = ctx["children"]

    p1, cl1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=5), principal_suffix="1")
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=10)).execute(
        p1,
        c1,
        CompletionCommand(
            completion_command_id="cmd-wa-1",
            assignment_id=cl1.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c1, output="c1 ok"),
        ).to_dict(),
        cl1.assignment_proof,
    )
    assert store.statuses[parent_id] == "waiting_for_children"

    after_dl = datetime.fromisoformat(resp.child_claim_deadline) + timedelta(seconds=5)
    out2, _ = await store.materialize_claim_timeout(c2, now=normalize_canonical_utc_timestamp(after_dl))
    assert out2.terminal_status == "claim_timed_out"
    rw = store.resolved_waves[(parent_id, wave_id)]
    assert rw.resolution == "SATISFIED"
    assert rw.completed_count == 1
    assert rw.terminal_count == 2
    assert rw.unresolved_count == 0
    assert rw.residual_child_task_ids == ()
    assert store.statuses[parent_id] == "resumable"


# ---------------------------------------------------------------------------
# D & E. Wave-Aware Execution Expiry, Recovery, Acceptance Replay, Resumed V2
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_wave_child_execution_expiry_recovery_reclaim_and_acceptance_replay() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    ctx = await _accept_wave(
        store,
        parent_id="root-exp-1",
        max_children=2,
        barrier_policy="WAIT_ALL",
        now_dt=dt,
        child_claim_deadline_seconds=600,
    )
    parent_id = ctx["parent_id"]
    wave_id = ctx["wave_id"]
    c1, c2 = ctx["children"]

    _, cl1 = await _claim_and_ack_child(
        store, c1, dt + timedelta(seconds=5), principal_suffix="1", execution_claim_timeout_seconds=30
    )
    rec_op = _principal("prin-rec-op", "agent-rec-op", ("recovery_operator",))

    # Execution expires on c1 -> c1 becomes recovery_required; parent stays waiting_for_children!
    hints_before = len(store.hints)
    exp_time = dt + timedelta(seconds=60)
    ev_res = await EvaluateExpiry(store, clock=lambda: exp_time).execute(rec_op, c1)
    assert ev_res.action == "recovery_required"
    assert ev_res.status == "recovery_required"
    assert ev_res.expiry_record is not None
    assert store.statuses[parent_id] == "waiting_for_children"
    assert store.fences[c1] == 1
    assert c1 not in store.claims
    assert len(store.hints) == hints_before

    # Acceptance replay succeeds with c1 in recovery_required (carry-forward #2)!
    rep_resp, rep_flag = await store.accept_delegation_wave(
        parent_task_id=parent_id,
        proposal=store.wave_proposals[(parent_id, ctx["proposal"].proposal_id)],
        accepted_wave=store.accepted_waves[(parent_id, wave_id)],
        child_task_inputs=[store.inputs[(cid, 1)] for cid in ctx["children"]],
        parent_continuation=store.continuations[(parent_id, 2)],
        yield_record=store.yield_records[(parent_id, ctx["assign_id"])],
        wake_up_hints=[],
        expected_fencing_epoch=1,
        now=normalize_canonical_utc_timestamp(dt + timedelta(seconds=2)),
    )
    assert rep_flag is True and rep_resp.proposal_id == ctx["proposal"].proposal_id

    # Recover c1 -> pending (fence unchanged until reclaim)
    dec_cmd = RecoveryDecisionCommand(
        task_id=c1,
        recovery_decision_id="rec-wave-c1",
        expired_claim_id=cl1.assignment.claim_id,
        expired_assignment_id=cl1.assignment.assignment_id,
        expired_fencing_epoch=1,
        disposition="restore_claimable",
    )
    _dec_rec, restored_st = await ResolveExpiredClaim(store, clock=lambda: exp_time + timedelta(seconds=5)).execute(
        rec_op, dec_cmd
    )
    assert restored_st == "pending"
    assert store.fences[c1] == 1

    # Reclaim c1 at fencing_epoch=2 and complete both children -> SATISFIED -> parent resumes!
    p1_b, cl1_b = await _claim_and_ack_child(store, c1, exp_time + timedelta(seconds=10), principal_suffix="1b")
    assert cl1_b.assignment.fencing_epoch == 2
    p2, cl2 = await _claim_and_ack_child(store, c2, exp_time + timedelta(seconds=11), principal_suffix="2")
    complete_uc = CompleteTurn(store, store, PEPPER, clock=lambda: exp_time + timedelta(seconds=15))
    await complete_uc.execute(
        p1_b,
        c1,
        CompletionCommand(
            completion_command_id="cmd-rec-c1",
            assignment_id=cl1_b.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c1, output="c1 recovered ok"),
        ).to_dict(),
        cl1_b.assignment_proof,
    )
    await complete_uc.execute(
        p2,
        c2,
        CompletionCommand(
            completion_command_id="cmd-rec-c2",
            assignment_id=cl2.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c2, output="c2 ok"),
        ).to_dict(),
        cl2.assignment_proof,
    )
    assert store.resolved_waves[(parent_id, wave_id)].resolution == "SATISFIED"
    assert store.statuses[parent_id] == "resumable"


@pytest.mark.anyio
async def test_wave_expiry_and_recovery_fail_closed_after_wave_resolution() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    ctx = await _accept_wave(store, parent_id="root-fail-post", max_children=2, now_dt=dt)
    parent_id = ctx["parent_id"]
    wave_id = ctx["wave_id"]
    c1, c2 = ctx["children"]

    # c1 expires -> recovery_required; c2 still working -> then wave is cancelled!
    _, cl1 = await _claim_and_ack_child(
        store, c1, dt + timedelta(seconds=5), principal_suffix="1", execution_claim_timeout_seconds=10
    )
    _, _cl2 = await _claim_and_ack_child(
        store, c2, dt + timedelta(seconds=5), principal_suffix="2", execution_claim_timeout_seconds=10
    )
    rec_op = _principal("prin-rec-op", "agent-rec-op", ("recovery_operator",))
    await store.transition_expired_claim(c1, now=normalize_canonical_utc_timestamp(dt + timedelta(seconds=25)))

    cancel_op = _principal("prin-can-op", "agent-can-op", ("wave_cancel_operator",))
    await CancelDelegationWave(store, clock=lambda: dt + timedelta(seconds=30)).execute(
        cancel_op,
        WaveCancellationCommand(
            cancellation_decision_id="can-post-exp",
            parent_task_id=parent_id,
            wave_id=wave_id,
        ),
    )

    # Recovery on c1 must fail closed now that wave is resolved/cancelled and c1 is ended / barrier_resolved
    with pytest.raises(RecoveryPreconditionFailedError):
        await ResolveExpiredClaim(store, clock=lambda: dt + timedelta(seconds=35)).execute(
            rec_op,
            RecoveryDecisionCommand(
                task_id=c1,
                recovery_decision_id="rec-too-late",
                expired_claim_id=cl1.assignment.claim_id,
                expired_assignment_id=cl1.assignment.assignment_id,
                expired_fencing_epoch=1,
                disposition="restore_claimable",
            ),
        )
    # Expiry transition on c2 must fail closed
    with pytest.raises((WaveAlreadyResolvedError, TaskNotClaimableError, ClaimNotFoundError)):
        await store.transition_expired_claim(c2, now=normalize_canonical_utc_timestamp(dt + timedelta(seconds=35)))
    # Acceptance replay still works after CANCELLED
    _, rep = await store.accept_delegation_wave(
        parent_task_id=parent_id,
        proposal=store.wave_proposals[(parent_id, ctx["proposal"].proposal_id)],
        accepted_wave=store.accepted_waves[(parent_id, wave_id)],
        child_task_inputs=[store.inputs[(cid, 1)] for cid in ctx["children"]],
        parent_continuation=store.continuations[(parent_id, 2)],
        yield_record=store.yield_records[(parent_id, ctx["assign_id"])],
        wake_up_hints=[],
        expected_fencing_epoch=1,
        now=normalize_canonical_utc_timestamp(dt + timedelta(seconds=2)),
    )
    assert rep is True


@pytest.mark.anyio
async def test_v2_resumed_claim_allowed_for_timed_out_and_completes_parent() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    ctx = await _accept_wave(store, parent_id="root-res-to", max_children=2, now_dt=dt)
    parent_id = ctx["parent_id"]
    wave_id = ctx["wave_id"]
    resp = ctx["response"]

    rec_op = _principal("prin-rec-op", "agent-rec-op", ("recovery_operator",))
    await EvaluateWaveTimeout(store).execute(rec_op, parent_id, wave_id, now=resp.wave_deadline)

    res_p2 = _principal("prin-res-2", "agent-res-2", ("research",))
    res_claim = await ClaimTask(store, PEPPER, clock=lambda: dt + timedelta(seconds=900)).execute(
        res_p2,
        parent_id,
        "req-res-to",
        "research",
        "resumed_continuation",
        expected_continuation_version=2,
    )
    await AcknowledgeAssignment(store, PEPPER, clock=lambda: dt + timedelta(seconds=901)).execute(
        res_p2, parent_id, res_claim.assignment.assignment_id, res_claim.assignment_proof
    )
    assert res_claim.assignment.continuation_version == 2
    rw_to = await store.get_resolved_delegation_wave(parent_id, wave_id)
    assert rw_to is not None and rw_to.resolution == "TIMED_OUT"
    ctx_to = await GetResumedContinuationContext(store, PEPPER).execute(
        res_p2, parent_id, res_claim.assignment.assignment_id, res_claim.assignment_proof
    )
    assert ctx_to.version == 2
    assert ctx_to.root_task_id == parent_id
    assert ctx_to.assignment_id == res_claim.assignment.assignment_id
    assert ctx_to.fencing_epoch == res_claim.assignment.fencing_epoch
    assert ctx_to.resolved_wave is not None and ctx_to.resolved_wave.resolution == "TIMED_OUT"
    assert ctx_to.child_outcome is None and ctx_to.child_result is None

    done = await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=905)).execute(
        res_p2,
        parent_id,
        CompletionCommand(
            completion_command_id="cmd-parent-after-to",
            assignment_id=res_claim.assignment.assignment_id,
            execution_kind="resumed_continuation",
            continuation_version=2,
            completed_result=TaskResult(task_id=parent_id, output="synthesized partial timeout result"),
        ).to_dict(),
        res_claim.assignment_proof,
    )
    assert done.outcome.terminal_status == "completed"
    assert store.statuses[parent_id] == "completed"


@pytest.mark.anyio
async def test_first_resolution_wins_concurrent_completion_vs_cancellation_and_timeout() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    ctx = await _accept_wave(
        store,
        parent_id="root-race-1",
        max_children=2,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 1, "denominator": 2},
        now_dt=dt,
    )
    parent_id = ctx["parent_id"]
    wave_id = ctx["wave_id"]
    resp = ctx["response"]
    c1, _c2 = ctx["children"]
    p1, cl1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=5), principal_suffix="1")
    cancel_op = _principal("prin-can-race", "agent-can-race", ("wave_cancel_operator",))

    r_comp, r_cancel = await asyncio.gather(
        CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=100)).execute(
            p1,
            c1,
            CompletionCommand(
                completion_command_id="cmd-race-c1",
                assignment_id=cl1.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=c1, output="winner"),
            ).to_dict(),
            cl1.assignment_proof,
        ),
        CancelDelegationWave(store, clock=lambda: dt + timedelta(seconds=100)).execute(
            cancel_op,
            WaveCancellationCommand(
                cancellation_decision_id="can-race-1",
                parent_task_id=parent_id,
                wave_id=wave_id,
            ),
        ),
        return_exceptions=True,
    )
    assert sum(1 for r in (r_comp, r_cancel) if not isinstance(r, BaseException)) == 1
    assert (parent_id, wave_id) in store.resolved_waves
    assert store.resolved_waves[(parent_id, wave_id)].resolution in ("SATISFIED", "CANCELLED")
    # Post-resolution timeout evaluation loses cleanly
    rec_op = _principal("prin-rec-race", "agent-rec-race", ("recovery_operator",))
    with pytest.raises(WaveAlreadyResolvedError):
        await EvaluateWaveTimeout(store).execute(rec_op, parent_id, wave_id, now=resp.wave_deadline)


# ---------------------------------------------------------------------------
# I. Mocked RedisGateway Error & Wrong-Type Fail-Closed Checks
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_redis_gateway_wave_timeout_and_cancellation_fail_closed_on_wrong_type() -> None:
    from orchestrator.infrastructure.redis_adapter import RedisGateway
    from redis.exceptions import ResponseError

    gw = RedisGateway("redis://localhost:6379/15", allow_unauthenticated_local_test=True)
    mock_client = AsyncMock()
    gw._client = mock_client

    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    t0 = normalize_canonical_utc_timestamp(dt)
    ref = ArtifactReference(artifact_id="art-1", producing_task_id="p-mock", artifact_version=1)
    dd = derive_wave_deadline(
        work_complexity="medium",
        uncertainty_level="medium",
        required_child_count=2,
        planned_parallelism=2,
        min_duration_seconds=MIN_WAVE_DURATION_SECONDS,
        max_duration_seconds=MAX_WAVE_DURATION_SECONDS,
        accepted_at=t0,
    )
    cont = TaskContinuationV2(
        parent_task_id="p-mock",
        wave_id="wave-mock",
        required_child_task_ids=("c1", "c2"),
        created_at=t0,
        updated_at=t0,
        input_version=1,
        continuation_capability="research",
        stage="incorporate_analysis_child",
        control_fields={},
        artifact_references=(ref,),
        continuation_version=2,
        state="active",
        wave_resolution=None,
    )
    wave = AcceptedDelegationWave(
        wave_id="wave-mock",
        parent_task_id="p-mock",
        required_child_task_ids=("c1", "c2"),
        barrier_policy="WAIT_ALL",
        deadline_derivation=dd,
        accepted_at=t0,
    )
    gw.get_lineage = AsyncMock(  # type: ignore[method-assign]
        return_value=TaskLineage(task_id="p-mock", root_task_id="p-mock", parent_task_id=None)
    )
    gw.get_continuation = AsyncMock(return_value=cont)  # type: ignore[method-assign]
    gw.get_accepted_delegation_wave = AsyncMock(return_value=wave)  # type: ignore[method-assign]

    mock_client.eval.return_value = ["WRONG_KEY_TYPE_task.delegation_wave_resolution:p-mock:wave-mock", "hash"]
    with pytest.raises(CorruptAuthoritativeRecordError):
        await gw.evaluate_wave_timeout(
            WaveTimeoutEvaluationCommand(parent_task_id="p-mock", wave_id="wave-mock"),
            now=dd.wave_deadline,
        )

    mock_client.eval.side_effect = ResponseError("WRONGTYPE")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await gw.cancel_delegation_wave(
            WaveCancellationCommand(
                cancellation_decision_id="can-m1",
                parent_task_id="p-mock",
                wave_id="wave-mock",
            ),
            operator_principal_id="op-can",
            now=t0,
        )


# ---------------------------------------------------------------------------
# J. Live Redis ACL Gateway Verification (4-Child Paths, Corruption, Replay, Cancellation)
# ---------------------------------------------------------------------------


@pytest.mark.live_redis_acl
@pytest.mark.anyio
async def test_live_redis_gateway_increment_4_timeout_cancellation_expiry_recovery() -> None:
    import redis.asyncio as aioredis
    from orchestrator.infrastructure.redis_adapter import RedisGateway

    from scripts.redis_acl_live_harness import LiveRedisAclCluster

    cluster = LiveRedisAclCluster.start(build_service_images=False, publish_host_port=True)
    gw = RedisGateway(
        cluster.redis_url,
        username="orchestrator",
        password=cluster.secrets_map["orchestrator"],
    )
    admin = aioredis.from_url(
        cluster.redis_url,
        username="fixture-admin",
        password=cluster.admin_secret,
        decode_responses=True,
    )
    try:
        dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)

        async def _set_admin_auth(pid: str, rec: RootAuthorityRecord) -> None:
            await admin.set(f"task.authority:{pid}", rec.to_json())

        # ------------------------------------------------------------------
        # 1. Four-child TIMED_OUT lifecycle + fence zero preservation + replay + resumed v2 claim
        # ------------------------------------------------------------------
        to_ctx = await _accept_wave(
            gw,
            parent_id="live-inc4-to-4c",
            max_children=4,
            barrier_policy="WAIT_ALL",
            proposal_id="prop-live-to-4c",
            now_dt=dt,
            set_authority_fn=_set_admin_auth,
        )
        tp = to_ctx["parent_id"]
        tw = to_ctx["wave_id"]
        tw_dl = to_ctx["response"].wave_deadline
        tc1, tc2, tc3, tc4 = to_ctx["children"]

        # c1 completes; c2 active working; c3 expired -> recovery_required; c4 never claimed (fence=0)
        p_tc1, cl_tc1 = await _claim_and_ack_child(gw, tc1, dt + timedelta(seconds=5), principal_suffix="t1")
        await CompleteTurn(gw, gw, PEPPER, clock=lambda: dt + timedelta(seconds=10)).execute(
            p_tc1,
            tc1,
            CompletionCommand(
                completion_command_id="cmd-live-tc1",
                assignment_id=cl_tc1.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=tc1, output="tc1 ok"),
            ).to_dict(),
            cl_tc1.assignment_proof,
        )
        p_tc2, cl_tc2 = await _claim_and_ack_child(gw, tc2, dt + timedelta(seconds=6), principal_suffix="t2")
        _p_tc3, cl_tc3 = await _claim_and_ack_child(
            gw, tc3, dt + timedelta(seconds=6), principal_suffix="t3", execution_claim_timeout_seconds=15
        )
        rec_op = _principal("prin-live-rec", "agent-live-rec", ("recovery_operator",))
        await EvaluateExpiry(gw, clock=lambda: dt + timedelta(seconds=40)).execute(rec_op, tc3)
        assert await gw.get_status(tc3) == "recovery_required"

        # Acceptance replay after progression to recovery_required (carry-forward #2)
        rep_wave_uc = AcceptDelegationWave(gw, PEPPER, clock=lambda: dt + timedelta(seconds=2))
        rep_resp = await rep_wave_uc.execute(
            to_ctx["principal"], tp, to_ctx["assign_id"], to_ctx["proof"], to_ctx["proposal"]
        )
        assert rep_resp.wave_id == tw

        # Early evaluation fails closed with zero mutation
        hints_before = await admin.llen("agent.hints.research")
        with pytest.raises(WaveDeadlineNotReachedError):
            await EvaluateWaveTimeout(gw).execute(
                rec_op,
                tp,
                tw,
                now=normalize_canonical_utc_timestamp(dt + timedelta(seconds=50)),
            )
        assert await admin.llen("agent.hints.research") == hints_before

        # Corrupt projection test on unresolved wave -> zero mutation!
        saved_cont_raw = await admin.get(f"task.continuation:{tp}:2")
        await admin.set(f"task.continuation:{tp}:2", "{not-json")
        with pytest.raises(CorruptAuthoritativeRecordError):
            await EvaluateWaveTimeout(gw).execute(rec_op, tp, tw, now=tw_dl)
        assert await admin.get(f"task.delegation_wave_resolution:{tp}:{tw}") is None
        assert await admin.llen("agent.hints.research") == hints_before
        await admin.set(f"task.continuation:{tp}:2", saved_cont_raw)

        # Resolve TIMED_OUT on 4-child wave at deadline
        rw_to, to_rep = await EvaluateWaveTimeout(gw).execute(rec_op, tp, tw, now=tw_dl)
        assert to_rep is False
        assert rw_to.resolution == "TIMED_OUT"
        assert rw_to.residual_child_task_ids == (tc2, tc3, tc4)
        assert await admin.llen("agent.hints.research") == hints_before + 1
        assert await gw.get_status(tc2) == "ended / barrier_resolved"
        assert await gw.get_status(tc3) == "ended / barrier_resolved"
        assert await gw.get_status(tc4) == "ended / barrier_resolved"
        assert await gw.get_fencing_epoch(tc2) == 2
        assert await gw.get_fencing_epoch(tc3) == 2
        assert await gw.get_fencing_epoch(tc4) == 0  # never-claimed fence 0 preserved!
        assert await gw.get_active_claim(tc2) is None

        # Post-resolution exact replay & acceptance replay after TIMED_OUT
        rw_to_2, to_rep_2 = await EvaluateWaveTimeout(gw).execute(rec_op, tp, tw, now=tw_dl)
        assert to_rep_2 is True and rw_to_2 == rw_to
        assert await admin.llen("agent.hints.research") == hints_before + 1
        rep_resp2 = await rep_wave_uc.execute(
            to_ctx["principal"], tp, to_ctx["assign_id"], to_ctx["proof"], to_ctx["proposal"]
        )
        assert rep_resp2.wave_id == tw

        # Stale losers after closure fail closed without mutation
        with pytest.raises(CompletionReplayAuthorizationError):
            await CompleteTurn(gw, gw, PEPPER, clock=lambda: dt + timedelta(seconds=900)).execute(
                p_tc2,
                tc2,
                CompletionCommand(
                    completion_command_id="cmd-stale-tc2",
                    assignment_id=cl_tc2.assignment.assignment_id,
                    execution_kind="delegated_child",
                    completed_result=TaskResult(task_id=tc2, output="stale"),
                ).to_dict(),
                cl_tc2.assignment_proof,
            )
        with pytest.raises((WaveAlreadyResolvedError, TaskNotClaimableError)):
            await gw.materialize_claim_timeout(tc4, now=normalize_canonical_utc_timestamp(dt + timedelta(seconds=900)))
        with pytest.raises(RecoveryPreconditionFailedError):
            await gw.resolve_expired_claim(
                RecoveryDecisionCommand(
                    task_id=tc3,
                    recovery_decision_id="rec-stale-tc3",
                    expired_claim_id=cl_tc3.assignment.claim_id,
                    expired_assignment_id=cl_tc3.assignment.assignment_id,
                    expired_fencing_epoch=1,
                    disposition="restore_claimable",
                ),
                operator_principal_id="prin-live-rec",
                now=normalize_canonical_utc_timestamp(dt + timedelta(seconds=900)),
            )

        # Resume parent v2 on TIMED_OUT wave -> complete root
        rp_res = _principal("prin-live-res-to", "agent-res-to", ("research",))
        r_claim = await ClaimTask(gw, PEPPER, clock=lambda: dt + timedelta(seconds=901)).execute(
            rp_res, tp, "req-live-res-to", "research", "resumed_continuation", expected_continuation_version=2
        )
        await AcknowledgeAssignment(gw, PEPPER, clock=lambda: dt + timedelta(seconds=902)).execute(
            rp_res, tp, r_claim.assignment.assignment_id, r_claim.assignment_proof
        )
        live_ctx_to = await GetResumedContinuationContext(gw, PEPPER).execute(
            rp_res, tp, r_claim.assignment.assignment_id, r_claim.assignment_proof
        )
        assert live_ctx_to.version == 2
        assert live_ctx_to.root_task_id == tp
        assert live_ctx_to.assignment_id == r_claim.assignment.assignment_id
        assert live_ctx_to.fencing_epoch == r_claim.assignment.fencing_epoch
        assert live_ctx_to.continuation.wave_id == tw
        assert live_ctx_to.resolved_wave == rw_to
        assert live_ctx_to.child_outcome is None and live_ctx_to.child_result is None
        r_done = await CompleteTurn(gw, gw, PEPPER, clock=lambda: dt + timedelta(seconds=905)).execute(
            rp_res,
            tp,
            CompletionCommand(
                completion_command_id="cmd-live-res-to-done",
                assignment_id=r_claim.assignment.assignment_id,
                execution_kind="resumed_continuation",
                continuation_version=2,
                completed_result=TaskResult(task_id=tp, output="timed out wave resumed"),
            ).to_dict(),
            r_claim.assignment_proof,
        )
        assert r_done.outcome.terminal_status == "completed"

        # ------------------------------------------------------------------
        # 2. Four-child PARTIAL UNSATISFIABLE via claim timeout + Wrong Redis types
        # ------------------------------------------------------------------
        ct_ctx = await _accept_wave(
            gw,
            parent_id="live-inc4-ct-4c",
            max_children=4,
            barrier_policy="PARTIAL",
            required_success_fraction={"numerator": 3, "denominator": 4},
            proposal_id="prop-live-ct-4c",
            now_dt=dt,
            child_claim_deadline_seconds=30,
            set_authority_fn=_set_admin_auth,
        )
        cp = ct_ctx["parent_id"]
        cw = ct_ctx["wave_id"]
        cc_dl = ct_ctx["response"].child_claim_deadline
        c4_1, c4_2, c4_3, c4_4 = ct_ctx["children"]
        await _claim_and_ack_child(gw, c4_1, dt + timedelta(seconds=5), principal_suffix="c41")

        # Wrong Redis type on resolved_wave key -> CorruptAuthoritativeRecordError & zero mutation
        after_cc = normalize_canonical_utc_timestamp(
            datetime.fromisoformat(cc_dl) + timedelta(seconds=10)
        )
        rw_key = f"task.delegation_wave_resolution:{cp}:{cw}"
        await admin.hset(rw_key, "bad", "type")
        with pytest.raises(CorruptAuthoritativeRecordError):
            await gw.materialize_claim_timeout(c4_2, now=after_cc)
        assert await gw.get_status(c4_2, now=normalize_canonical_utc_timestamp(dt + timedelta(seconds=10))) == "pending"
        await admin.delete(rw_key)

        # c4_2 claim timeout (1 fail -> still 3 left, unresolved); c4_3 claim timeout (2 fails -> max 2 < 3 -> UNSATISFIABLE)
        out_42, _ = await gw.materialize_claim_timeout(c4_2, now=after_cc)
        assert out_42.terminal_status == "claim_timed_out"
        assert await gw.get_status(cp) == "waiting_for_children"
        out_43, _ = await gw.materialize_claim_timeout(c4_3, now=after_cc)
        assert out_43.terminal_status == "claim_timed_out"
        rw_ct = await gw.get_resolved_delegation_wave(cp, cw)
        assert rw_ct is not None and rw_ct.resolution == "UNSATISFIABLE"
        assert await gw.get_status(c4_1) == "ended / barrier_resolved"
        assert await gw.get_status(c4_4) == "ended / barrier_resolved"
        assert await gw.get_fencing_epoch(c4_1) == 2
        assert await gw.get_fencing_epoch(c4_4) == 0  # never-claimed fence zero preserved!

        # Wrong Redis type on resolved_wave key AFTER UNSATISFIABLE fails read_resumed_continuation_v2_context without mutation
        ru_p = _principal("prin-live-res-unsat", "agent-res-unsat", ("research",))
        ru_claim = await ClaimTask(gw, PEPPER, clock=lambda: dt + timedelta(seconds=50)).execute(
            ru_p, cp, "req-live-res-unsat", "research", "resumed_continuation", expected_continuation_version=2
        )
        await AcknowledgeAssignment(gw, PEPPER, clock=lambda: dt + timedelta(seconds=51)).execute(
            ru_p, cp, ru_claim.assignment.assignment_id, ru_claim.assignment_proof
        )
        live_ctx_unsat = await GetResumedContinuationContext(gw, PEPPER).execute(
            ru_p, cp, ru_claim.assignment.assignment_id, ru_claim.assignment_proof
        )
        assert live_ctx_unsat.version == 2
        assert live_ctx_unsat.resolved_wave is not None
        assert live_ctx_unsat.resolved_wave.resolution == "UNSATISFIABLE"
        saved_rw_raw = await admin.get(rw_key)
        await admin.delete(rw_key)
        await admin.hset(rw_key, "bad", "type")
        with pytest.raises(CorruptAuthoritativeRecordError):
            await GetResumedContinuationContext(gw, PEPPER).execute(
                ru_p, cp, ru_claim.assignment.assignment_id, ru_claim.assignment_proof
            )
        await admin.delete(rw_key)
        await admin.set(rw_key, saved_rw_raw)

        # ------------------------------------------------------------------
        # 3. Four-child Expiry -> Recovery -> Reclaim + Cancellation & CANCELLED Resumed Claim Rejection
        # ------------------------------------------------------------------
        can_ctx = await _accept_wave(
            gw,
            parent_id="live-inc4-cancel-4c",
            max_children=4,
            barrier_policy="WAIT_ALL",
            proposal_id="prop-live-can-4c",
            now_dt=dt,
            set_authority_fn=_set_admin_auth,
        )
        kp = can_ctx["parent_id"]
        kw = can_ctx["wave_id"]
        kc1, kc2, kc3, kc4 = can_ctx["children"]
        _kp1, kcl1 = await _claim_and_ack_child(
            gw, kc1, dt + timedelta(seconds=5), principal_suffix="k1", execution_claim_timeout_seconds=15
        )
        await EvaluateExpiry(gw, clock=lambda: dt + timedelta(seconds=30)).execute(rec_op, kc1)
        await ResolveExpiredClaim(gw, clock=lambda: dt + timedelta(seconds=35)).execute(
            rec_op,
            RecoveryDecisionCommand(
                task_id=kc1,
                recovery_decision_id="rec-live-kc1",
                expired_claim_id=kcl1.assignment.claim_id,
                expired_assignment_id=kcl1.assignment.assignment_id,
                expired_fencing_epoch=1,
                disposition="restore_claimable",
            ),
        )
        # Reclaim kc1 (epoch=2) and complete it!
        kp1b, kcl1b = await _claim_and_ack_child(gw, kc1, dt + timedelta(seconds=40), principal_suffix="k1b")
        assert kcl1b.assignment.fencing_epoch == 2
        await CompleteTurn(gw, gw, PEPPER, clock=lambda: dt + timedelta(seconds=45)).execute(
            kp1b,
            kc1,
            CompletionCommand(
                completion_command_id="cmd-live-kc1b",
                assignment_id=kcl1b.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=kc1, output="kc1 recovered"),
            ).to_dict(),
            kcl1b.assignment_proof,
        )
        await _claim_and_ack_child(gw, kc2, dt + timedelta(seconds=46), principal_suffix="k2")

        cancel_op = _principal("prin-live-cancel", "agent-live-cancel", ("wave_cancel_operator",))
        hints_pre_can = await admin.llen("agent.hints.research")
        cdec, crw, crep = await CancelDelegationWave(gw, clock=lambda: dt + timedelta(seconds=50)).execute(
            cancel_op,
            WaveCancellationCommand(
                cancellation_decision_id="can-live-1",
                parent_task_id=kp,
                wave_id=kw,
            ),
        )
        assert crep is False
        assert crw.resolution == "CANCELLED"
        assert cdec.terminal_parent_status == "cancelled"
        assert await admin.llen("agent.hints.research") == hints_pre_can
        assert await gw.get_status(kp) == "cancelled"
        assert await gw.get_result(kp) == {"task_id": kp, "status": "cancelled", "output": None}
        assert await gw.get_fencing_epoch(kc2) == 2
        assert await gw.get_fencing_epoch(kc3) == 0 and await gw.get_fencing_epoch(kc4) == 0

        # Exact replay + acceptance replay after CANCELLED
        cdec2, crw2, crep2 = await CancelDelegationWave(gw, clock=lambda: dt + timedelta(seconds=55)).execute(
            cancel_op,
            WaveCancellationCommand(
                cancellation_decision_id="can-live-1",
                parent_task_id=kp,
                wave_id=kw,
            ),
        )
        assert crep2 is True and cdec2 == cdec and crw2 == crw
        rep_can = await AcceptDelegationWave(gw, PEPPER, clock=lambda: dt + timedelta(seconds=2)).execute(
            can_ctx["principal"], kp, can_ctx["assign_id"], can_ctx["proof"], can_ctx["proposal"]
        )
        assert rep_can.wave_id == kw

        # Explicit CANCELLED resumed-claim rejection in live Redis Lua!
        with pytest.raises(TaskNotClaimableError):
            await ClaimTask(gw, PEPPER, clock=lambda: dt + timedelta(seconds=60)).execute(
                _principal("prin-live-res-can", "agent-res-can", ("research",)),
                kp,
                "req-reject-cancelled-res",
                "research",
                "resumed_continuation",
                expected_continuation_version=2,
            )
    finally:
        await admin.aclose()
        await gw.aclose()
        cluster.cleanup()


# ---------------------------------------------------------------------------
# K. Versioned v2 ResumedContinuationContext, Boundary, API & Parity Coverage
# ---------------------------------------------------------------------------


def test_resumed_continuation_context_v2_contract_strict_validation() -> None:
    from dataclasses import FrozenInstanceError

    from agent_contracts import (
        CONTINUATION_CONTEXT_V2_CONTRACT_VERSION,
        RequiredSuccessFraction,
        ResolvedDelegationWave,
        ResumedContinuationContext,
        StageArtifact,
        TaskContinuationV2,
        TaskInput,
        TaskResult,
        build_claim_timed_out_terminal_outcome,
    )

    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    t0 = normalize_canonical_utc_timestamp(dt)
    t1 = normalize_canonical_utc_timestamp(dt + timedelta(seconds=30))
    ref = ArtifactReference(artifact_id="art-ctx-v2", producing_task_id="parent-v2", artifact_version=1)
    dd = derive_wave_deadline(
        work_complexity="medium",
        uncertainty_level="low",
        required_child_count=2,
        planned_parallelism=2,
        min_duration_seconds=MIN_WAVE_DURATION_SECONDS,
        max_duration_seconds=MAX_WAVE_DURATION_SECONDS,
        accepted_at=t0,
    )
    parent_input = TaskInput(
        task_id="parent-v2",
        prompt="Synthesize wave",
        created_at=t0,
        input_version=1,
        artifact_references=(ref,),
    )
    stage_artifact = StageArtifact(
        artifact_id="art-ctx-v2",
        producing_task_id="parent-v2",
        artifact_type="research_material",
        content={"material": "wave seed"},
        created_at=t0,
        producing_turn_id="turn-1",
        artifact_version=1,
        payload_schema_version=1,
        sealed=True,
    )
    accepted_wave = AcceptedDelegationWave(
        wave_id="wave-ctx-1",
        parent_task_id="parent-v2",
        required_child_task_ids=("parent-v2-c1", "parent-v2-c2"),
        barrier_policy="PARTIAL",
        required_success_fraction=RequiredSuccessFraction(1, 2),
        required_success_count=1,
        deadline_derivation=dd,
        accepted_at=t0,
    )

    for resolution in ("SATISFIED", "UNSATISFIABLE", "TIMED_OUT"):
        cont = TaskContinuationV2(
            parent_task_id="parent-v2",
            wave_id="wave-ctx-1",
            required_child_task_ids=("parent-v2-c1", "parent-v2-c2"),
            created_at=t0,
            updated_at=t1,
            input_version=1,
            continuation_capability="research",
            stage="incorporate_analysis_child",
            control_fields={"analysis_focus": "battery viability"},
            artifact_references=(ref,),
            continuation_version=2,
            state="released",
            wave_resolution=resolution,
        )
        if resolution == "UNSATISFIABLE":
            child_outcomes_val = {
                "parent-v2-c1": {"status": "claim_timed_out", "outcome_id": "out-c1"},
                "parent-v2-c2": {"status": "failed", "outcome_id": "out-c2"},
            }
            residual_ids_val: tuple[str, ...] = ()
            c_cnt, t_cnt, u_cnt = 0, 2, 0
        elif resolution == "SATISFIED":
            child_outcomes_val = {
                "parent-v2-c1": {"status": "completed", "outcome_id": "out-c1"},
                "parent-v2-c2": {"status": "ended / barrier_resolved", "outcome_id": None},
            }
            residual_ids_val = ("parent-v2-c2",)
            c_cnt, t_cnt, u_cnt = 1, 1, 1
        else:
            child_outcomes_val = {
                "parent-v2-c1": {"status": "claim_timed_out", "outcome_id": "out-c1"},
                "parent-v2-c2": {"status": "ended / barrier_resolved", "outcome_id": None},
            }
            residual_ids_val = ("parent-v2-c2",)
            c_cnt, t_cnt, u_cnt = 0, 1, 1
        rw = ResolvedDelegationWave(
            version=2,
            wave_id="wave-ctx-1",
            parent_task_id="parent-v2",
            required_child_task_ids=("parent-v2-c1", "parent-v2-c2"),
            accepted_at=t0,
            accepted_wave_fingerprint=accepted_wave.semantic_fingerprint(),
            resolution=resolution,
            resolved_at=t1,
            barrier_policy="PARTIAL",
            required_success_fraction=RequiredSuccessFraction(1, 2),
            required_success_count=1,
            completed_count=c_cnt,
            terminal_count=t_cnt,
            unresolved_count=u_cnt,
            child_outcomes=child_outcomes_val,
            residual_child_task_ids=residual_ids_val,
            residual_disposition="ended / barrier_resolved",
            triggering_child_task_id="parent-v2-c1" if resolution != "TIMED_OUT" else None,
            triggering_assignment_id="asgn-c1" if resolution == "SATISFIED" else None,
            triggering_fencing_epoch=1 if resolution == "SATISFIED" else None,
            wave_deadline=dd.wave_deadline,
            deadline_policy_id=dd.policy_id,
            deadline_policy_version=dd.policy_version,
        )
        ctx = ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=cont,
            stage_artifact=stage_artifact,
            resolved_wave=rw,
            root_task_id="parent-v2",
            assignment_id="asgn-res-2",
            fencing_epoch=2,
            version=CONTINUATION_CONTEXT_V2_CONTRACT_VERSION,
        )
        raw_dict = ctx.to_dict()
        assert set(raw_dict.keys()) == {
            "version",
            "root_task_id",
            "assignment_id",
            "fencing_epoch",
            "parent_task_input",
            "continuation",
            "stage_artifact",
            "resolved_wave",
        }
        assert ResumedContinuationContext.from_dict(raw_dict) == ctx
        assert ResumedContinuationContext.from_json(ctx.to_json()) == ctx

    # Immutability check
    with pytest.raises(FrozenInstanceError):
        ctx.fencing_epoch = 3  # type: ignore[misc]

    # Unsupported versions, bool-as-int, unknown fields, missing fields, and forbidden v1 fields
    with pytest.raises(ValueError, match="Unsupported contract version"):
        ResumedContinuationContext.from_dict({**raw_dict, "version": 3})
    with pytest.raises(TypeError, match="bool"):
        ResumedContinuationContext.from_dict({**raw_dict, "fencing_epoch": True})
    with pytest.raises(TypeError, match="unexpected ResumedContinuationContext fields"):
        ResumedContinuationContext.from_dict({**raw_dict, "child_result": None})
    with pytest.raises(TypeError, match="unexpected ResumedContinuationContext fields"):
        ResumedContinuationContext.from_dict({**raw_dict, "extra": "bad"})
    for req_field in ("root_task_id", "assignment_id", "fencing_epoch", "resolved_wave"):
        missing = dict(raw_dict)
        del missing[req_field]
        with pytest.raises(ValueError, match="is required"):
            ResumedContinuationContext.from_dict(missing)
    with pytest.raises(ValueError, match="root_task_id"):
        ResumedContinuationContext.from_dict({**raw_dict, "root_task_id": "x" * 300})

    # Active or closed/CANCELLED rejected
    active_cont = TaskContinuationV2(
        parent_task_id="parent-v2",
        wave_id="wave-ctx-1",
        required_child_task_ids=("parent-v2-c1", "parent-v2-c2"),
        created_at=t0,
        updated_at=t1,
        input_version=1,
        continuation_capability="research",
        stage="incorporate_analysis_child",
        control_fields={},
        artifact_references=(ref,),
        continuation_version=2,
        state="active",
        wave_resolution=None,
    )
    with pytest.raises(ValueError, match="released"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=active_cont,
            stage_artifact=stage_artifact,
            resolved_wave=rw,
            root_task_id="parent-v2",
            assignment_id="asgn-res-2",
            fencing_epoch=2,
            version=2,
        )
    with pytest.raises(ValueError, match="child_outcome must be absent"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=cont,
            stage_artifact=stage_artifact,
            resolved_wave=rw,
            child_outcome=build_claim_timed_out_terminal_outcome(
                task_id="c1",
                claim_deadline=t0,
                completed_at=t1,
            ),
            root_task_id="parent-v2",
            assignment_id="asgn-res-2",
            fencing_epoch=2,
            version=2,
        )
    with pytest.raises(ValueError, match="child_result must be absent"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=cont,
            stage_artifact=stage_artifact,
            resolved_wave=rw,
            child_result=TaskResult(task_id="parent-v2-c1", output="bad"),
            root_task_id="parent-v2",
            assignment_id="asgn-res-2",
            fencing_epoch=2,
            version=2,
        )


@pytest.mark.anyio
async def test_v2_resumed_continuation_context_boundary_and_public_api_roundtrip() -> None:
    import httpx
    from agent_infrastructure import OrchestratorClient

    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)

    # Setup SATISFIED wave via PARTIAL 1/2
    ctx = await _accept_wave(
        store,
        parent_id="root-api-ctx-v2",
        max_children=2,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 1, "denominator": 2},
        now_dt=dt,
    )
    parent_id = ctx["parent_id"]
    wave_id = ctx["wave_id"]
    c1, _c2 = ctx["children"]
    p1, cl1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=5), principal_suffix="1")
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=8)).execute(
        p1,
        c1,
        CompletionCommand(
            completion_command_id="cmd-api-ctx-c1",
            assignment_id=cl1.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c1, output="child result stays out of v2 context"),
        ).to_dict(),
        cl1.assignment_proof,
    )

    res_p2 = _principal("prin-res-api-2", "agent-res-api-2", ("research",))
    res_claim = await ClaimTask(store, PEPPER, clock=lambda: dt + timedelta(seconds=12)).execute(
        res_p2,
        parent_id,
        "req-api-ctx-v2",
        "research",
        "resumed_continuation",
        expected_continuation_version=2,
    )
    await AcknowledgeAssignment(store, PEPPER, clock=lambda: dt + timedelta(seconds=13)).execute(
        res_p2, parent_id, res_claim.assignment.assignment_id, res_claim.assignment_proof
    )

    # Authorization & non-disclosure checks before exposing context
    get_ctx_uc = GetResumedContinuationContext(store, PEPPER)
    wrong_cap = _principal("prin-ana-wrong", "agent-ana-wrong", ("analysis",))
    other_res = _principal("prin-res-other", "agent-res-other", ("research",))
    wrong_agent = _principal("prin-res-api-2", "agent-other-bound", ("research",))
    with pytest.raises(CapabilityMismatchError):
        await get_ctx_uc.execute(wrong_cap, parent_id, res_claim.assignment.assignment_id, res_claim.assignment_proof)
    with pytest.raises(ClaimNotFoundError):
        await get_ctx_uc.execute(other_res, parent_id, res_claim.assignment.assignment_id, res_claim.assignment_proof)
    with pytest.raises(ClaimNotFoundError):
        await get_ctx_uc.execute(wrong_agent, parent_id, res_claim.assignment.assignment_id, res_claim.assignment_proof)
    with pytest.raises(InvalidClaimProofError):
        await get_ctx_uc.execute(res_p2, parent_id, res_claim.assignment.assignment_id, "0" * 64)

    # Corrupt resolved wave fails closed (never reveals partial state)
    saved_rw = store.resolved_waves.pop((parent_id, wave_id))
    with pytest.raises(CorruptAuthoritativeRecordError):
        await get_ctx_uc.execute(res_p2, parent_id, res_claim.assignment.assignment_id, res_claim.assignment_proof)
    store.resolved_waves[(parent_id, wave_id)] = saved_rw

    # Public API + OrchestratorClient cross-instance round trip (SATISFIED v2)
    principals = {
        "prin-res-api-2": {
            "bound_agent_id": "agent-res-api-2",
            "authorized_capabilities": ("research",),
            "token_digest": compute_token_digest("tok-res-api-2", CRED_PEPPER),
            "active": True,
        },
        "prin-ana-api": {
            "bound_agent_id": "agent-ana-api",
            "authorized_capabilities": ("analysis",),
            "token_digest": compute_token_digest("tok-ana-api", CRED_PEPPER),
            "active": True,
        },
    }
    settings = Settings(
        assignment_proof_pepper=PEPPER,
        runtime_credential_pepper=CRED_PEPPER,
        task_delivery_mode="authenticated_claim",
        runtime_principals=principals,
    )
    app = create_app(settings=settings, task_store=store, task_queue=store)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http_client:
        url = f"/api/tasks/{parent_id}/continuation-context?assignment_id={res_claim.assignment.assignment_id}"
        assert (await http_client.get(url, headers={"X-Assignment-Proof": res_claim.assignment_proof})).status_code == 401
        assert (
            await http_client.get(
                url,
                headers={"Authorization": "Bearer tok-ana-api", "X-Assignment-Proof": res_claim.assignment_proof},
            )
        ).status_code == 403
        assert (
            await http_client.get(
                url,
                headers={"Authorization": "Bearer tok-res-api-2", "X-Assignment-Proof": "f" * 64},
            )
        ).status_code == 403
        r_wrong_asgn = await http_client.get(
            f"/api/tasks/{parent_id}/continuation-context?assignment_id=assign-missing",
            headers={"Authorization": "Bearer tok-res-api-2", "X-Assignment-Proof": res_claim.assignment_proof},
        )
        assert r_wrong_asgn.status_code == 404

        r_ok = await http_client.get(
            url,
            headers={"Authorization": "Bearer tok-res-api-2", "X-Assignment-Proof": res_claim.assignment_proof},
        )
        assert r_ok.status_code == 200
        body = r_ok.json()
        assert body["version"] == 2
        assert body["root_task_id"] == parent_id
        assert body["assignment_id"] == res_claim.assignment.assignment_id
        assert body["fencing_epoch"] == 2
        assert body["resolved_wave"]["resolution"] == "SATISFIED"
        assert "child_result" not in body
        assert "child result stays out of v2 context" not in json.dumps(body)

    client = OrchestratorClient(
        base_url="http://test",
        runtime_token="tok-res-api-2",
        transport=httpx.ASGITransport(app=app),
    )
    parsed_ctx = await client.get_resumed_continuation_context(
        parent_id, res_claim.assignment.assignment_id, res_claim.assignment_proof
    )
    assert parsed_ctx.version == 2
    assert parsed_ctx.resolved_wave is not None
    assert parsed_ctx.resolved_wave.wave_id == wave_id
    assert await client.get_continuation(parent_id, version=2) == parsed_ctx.continuation

    # Cancelled parent cannot obtain resumed context (even if corrupted claim existed)
    assert await client.get_terminal_outcome(parent_id) is None
    store.fences[parent_id] = 3
    with pytest.raises(TaskAlreadyClaimedError):
        await get_ctx_uc.execute(res_p2, parent_id, res_claim.assignment.assignment_id, res_claim.assignment_proof)
    can_ctx = await _accept_wave(store, parent_id="root-api-cancel-ctx", max_children=2, now_dt=dt)
    await CancelDelegationWave(store, clock=lambda: dt + timedelta(seconds=20)).execute(
        _principal("op-can-ctx", "agent-can-ctx", ("wave_cancel_operator",)),
        WaveCancellationCommand(
            cancellation_decision_id="can-api-ctx",
            parent_task_id=can_ctx["parent_id"],
            wave_id=can_ctx["wave_id"],
        ),
    )
    assert await store.get_status(can_ctx["parent_id"]) == "cancelled"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http_client:
        r_can = await http_client.get(
            f"/api/tasks/{can_ctx['parent_id']}/continuation-context?assignment_id=any-assign",
            headers={"Authorization": "Bearer tok-res-api-2", "X-Assignment-Proof": "a" * 64},
        )
        assert r_can.status_code == 404


@pytest.mark.anyio
async def test_redis_gateway_read_resumed_continuation_v2_context_fails_closed_on_wrong_types() -> None:
    from orchestrator.infrastructure.redis_adapter import RedisGateway
    from redis.exceptions import ResponseError

    gw = RedisGateway("redis://localhost:6379/15", allow_unauthenticated_local_test=True)
    mock_client = AsyncMock()
    gw._client = mock_client

    # Simulate wrong key type returned by READ_RESUMED_CONTINUATION_V2_LUA
    mock_client.get.side_effect = ResponseError("WRONGTYPE")
    mock_client.eval.return_value = ["WRONG_KEY_TYPE_task.continuation:parent-mock:2", "hash"]
    with pytest.raises(CorruptAuthoritativeRecordError):
        await gw.read_resumed_continuation_v2_context(
            parent_task_id="parent-mock",
            assignment_id="asgn-mock",
            principal_id="prin-mock",
            bound_agent_id="agent-mock",
            expected_fencing_epoch=2,
        )

    # Simulate closed/cancelled continuation returned by Lua
    mock_client.get.side_effect = None
    mock_client.get.return_value = None
    mock_client.eval.return_value = ["CONTINUATION_CLOSED"]
    with pytest.raises(TaskNotClaimableError):
        await gw.read_resumed_continuation_v2_context(
            parent_task_id="parent-mock",
            assignment_id="asgn-mock",
            principal_id="prin-mock",
            bound_agent_id="agent-mock",
            expected_fencing_epoch=2,
        )


@pytest.mark.anyio
async def test_orchestrator_client_v2_resumed_context_terminal_outcome_lookup() -> None:
    from agent_contracts import (
        CONTINUATION_CONTEXT_V2_CONTRACT_VERSION,
        RequiredSuccessFraction,
        ResolvedDelegationWave,
        ResumedContinuationContext,
        StageArtifact,
        TaskContinuation,
        TaskInput,
        build_claim_timed_out_terminal_outcome,
    )
    from agent_infrastructure.orchestrator_client import OrchestratorClient

    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    t0 = normalize_canonical_utc_timestamp(dt)
    t1 = normalize_canonical_utc_timestamp(dt + timedelta(seconds=30))
    ref = ArtifactReference(artifact_id="art-client-1", producing_task_id="parent-v2", artifact_version=1)
    dd = derive_wave_deadline(
        work_complexity="medium",
        uncertainty_level="low",
        required_child_count=2,
        planned_parallelism=2,
        min_duration_seconds=MIN_WAVE_DURATION_SECONDS,
        max_duration_seconds=MAX_WAVE_DURATION_SECONDS,
        accepted_at=t0,
    )
    parent_input = TaskInput(
        task_id="parent-v2",
        prompt="Synthesize wave",
        created_at=t0,
        input_version=1,
        artifact_references=(ref,),
    )
    stage_artifact = StageArtifact(
        artifact_id="art-client-1",
        producing_task_id="parent-v2",
        artifact_type="research_material",
        content={"material": "wave seed"},
        created_at=t0,
        producing_turn_id="turn-1",
        artifact_version=1,
        payload_schema_version=1,
        sealed=True,
    )
    accepted_wave = AcceptedDelegationWave(
        wave_id="wave-client-1",
        parent_task_id="parent-v2",
        required_child_task_ids=("parent-v2-c1", "parent-v2-c2"),
        barrier_policy="PARTIAL",
        required_success_fraction=RequiredSuccessFraction(1, 2),
        required_success_count=1,
        deadline_derivation=dd,
        accepted_at=t0,
    )
    cont_v2 = TaskContinuationV2(
        parent_task_id="parent-v2",
        wave_id="wave-client-1",
        required_child_task_ids=("parent-v2-c1", "parent-v2-c2"),
        created_at=t0,
        updated_at=t1,
        input_version=1,
        continuation_capability="research",
        stage="incorporate_analysis_child",
        control_fields={"material_summary": "seed"},
        artifact_references=(ref,),
        continuation_version=2,
        state="released",
        wave_resolution="SATISFIED",
    )
    rw = ResolvedDelegationWave(
        version=2,
        wave_id="wave-client-1",
        parent_task_id="parent-v2",
        required_child_task_ids=("parent-v2-c1", "parent-v2-c2"),
        accepted_at=t0,
        accepted_wave_fingerprint=accepted_wave.semantic_fingerprint(),
        resolution="SATISFIED",
        resolved_at=t1,
        barrier_policy="PARTIAL",
        required_success_fraction=RequiredSuccessFraction(1, 2),
        required_success_count=1,
        completed_count=1,
        terminal_count=1,
        unresolved_count=1,
        child_outcomes={
            "parent-v2-c1": {"status": "completed", "outcome_id": "out-c1"},
            "parent-v2-c2": {"status": "ended / barrier_resolved", "outcome_id": None},
        },
        residual_child_task_ids=("parent-v2-c2",),
        residual_disposition="ended / barrier_resolved",
        triggering_child_task_id="parent-v2-c1",
        triggering_assignment_id="asgn-c1",
        triggering_fencing_epoch=1,
        wave_deadline=dd.wave_deadline,
        deadline_policy_id=dd.policy_id,
        deadline_policy_version=dd.policy_version,
    )
    v2_ctx = ResumedContinuationContext(
        parent_task_input=parent_input,
        continuation=cont_v2,
        stage_artifact=stage_artifact,
        resolved_wave=rw,
        root_task_id="parent-v2",
        assignment_id="asgn-res-2",
        fencing_epoch=2,
        version=CONTINUATION_CONTEXT_V2_CONTRACT_VERSION,
    )

    client = OrchestratorClient(base_url="http://test", runtime_token="tok-client")

    # 1. Cache a valid v2 ResumedContinuationContext.
    client._cached_contexts["parent-v2"] = v2_ctx

    # 2. Call get_terminal_outcome for any task.
    # 3. No AttributeError occurs.
    # 4. The method returns the established absence value.
    assert await client.get_terminal_outcome("parent-v2") is None
    assert await client.get_terminal_outcome("parent-v2-c1") is None
    assert await client.get_terminal_outcome("other-task") is None

    # Also verify get_continuation returns TaskContinuationV2
    assert await client.get_continuation("parent-v2", version=2) == cont_v2

    # 5. Existing v1 cached-outcome lookup remains unchanged.
    ref_v1 = ArtifactReference(artifact_id="art-v1", producing_task_id="parent-v1", artifact_version=1)
    parent_input_v1 = TaskInput(
        task_id="parent-v1",
        prompt="Process child",
        created_at=t0,
        input_version=1,
        artifact_references=(ref_v1,),
    )
    stage_artifact_v1 = StageArtifact(
        artifact_id="art-v1",
        producing_task_id="parent-v1",
        artifact_type="research_material",
        content={"material": "v1 seed"},
        created_at=t0,
        producing_turn_id="turn-1",
        artifact_version=1,
        payload_schema_version=1,
        sealed=True,
    )
    cont_v1 = TaskContinuation(
        parent_task_id="parent-v1",
        child_task_id="child-1",
        created_at=t0,
        updated_at=t1,
        input_version=1,
        continuation_capability="research",
        stage="incorporate_analysis_child",
        control_fields={"material_summary": "v1 summary"},
        artifact_references=(ref_v1,),
        continuation_version=1,
    )
    v1_outcome = build_claim_timed_out_terminal_outcome(
        task_id="child-1",
        claim_deadline=t0,
        completed_at=t1,
    )
    v1_ctx = ResumedContinuationContext(
        parent_task_input=parent_input_v1,
        continuation=cont_v1,
        stage_artifact=stage_artifact_v1,
        child_outcome=v1_outcome,
        child_result=None,
        version=1,
    )
    client._cached_contexts["parent-v1"] = v1_ctx

    assert await client.get_terminal_outcome("child-1") == v1_outcome
    assert await client.get_terminal_outcome("parent-v1") is None
    assert await client.get_continuation("parent-v1", version=1) == cont_v1
