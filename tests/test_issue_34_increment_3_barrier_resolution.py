import asyncio
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import pytest
from agent_contracts import (
    MAX_CONTROL_RECORD_BYTES,
    MAX_WAVE_IDENTIFIER_BYTES,
    RESOLVED_DELEGATION_WAVE_V2_CONTRACT_VERSION,
    ActiveClaim,
    ArtifactReference,
    CompletionCommand,
    DelegationWaveProposalItem,
    DelegationWaveProposalRequest,
    RequiredSuccessFraction,
    ResolvedDelegationWave,
    RootAuthorityRecord,
    StageArtifact,
    TaskContinuationV2,
    TaskRequested,
    TaskResult,
    evaluate_barrier,
    normalize_canonical_utc_timestamp,
)
from orchestrator.application.ports import (
    CompletionConflictError,
    DelegationConflictError,
    TaskNotClaimableError,
)
from orchestrator.application.use_cases import (
    AcceptDelegationWave,
    AcknowledgeAssignment,
    ClaimTask,
    CompleteTurn,
    CompletionReplayAuthorizationError,
    DelegationReplayAuthorizationError,
)
from orchestrator.domain.principal import RuntimePrincipal
from orchestrator.domain.task import CorruptAuthoritativeRecordError

from tests.test_authoritative_task_store import InMemoryAuthoritativeTaskStore

PEPPER = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def _principal(
    principal_id: str = "principal-research-1",
    agent_id: str = "agent-research-1",
    capabilities: tuple[str, ...] = ("research",),
) -> RuntimePrincipal:
    return RuntimePrincipal(
        principal_id=principal_id,
        bound_agent_id=agent_id,
        authorized_capabilities=capabilities,
        active=True,
    )


def _wave_items(
    parent_id: str,
    count: int = 3,
    artifact_id: str = "art-wave-1",
) -> tuple[DelegationWaveProposalItem, ...]:
    ref = ArtifactReference(
        producing_task_id=parent_id,
        artifact_id=artifact_id,
        artifact_version=1,
    )
    return tuple(
        DelegationWaveProposalItem(
            item_id=f"item-{idx}",
            child_intent=f"Analyze angle {idx} for {parent_id}",
            artifact_reference=ref,
        )
        for idx in range(1, count + 1)
    )


async def _setup_accepted_wave(
    store: Any,
    *,
    parent_id: str,
    count: int,
    barrier_policy: str,
    required_success_fraction: dict[str, int] | RequiredSuccessFraction | None = None,
    dt: datetime | None = None,
    set_authority_fn: Any | None = None,
) -> dict[str, Any]:
    base_dt = dt or datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    now_iso = normalize_canonical_utc_timestamp(base_dt)
    await store.register_root_task(
        TaskRequested(
            task_id=parent_id,
            root_task_id=parent_id,
            parent_task_id=None,
            prompt="Parent wave task",
            created_at=now_iso,
        ),
        initial_status="pending",
        required_capability="research",
    )
    wave_auth = RootAuthorityRecord.create_wave_baseline(
        parent_id, max_children_per_wave=max(2, count), created_at=now_iso
    )
    if set_authority_fn is not None:
        await set_authority_fn(parent_id, wave_auth)
    else:
        store.authorities[parent_id] = wave_auth

    res_principal = _principal()
    claim_uc = ClaimTask(
        store,
        PEPPER,
        clock=lambda: base_dt,
        allow_wave_delegation_grant=True,
    )
    p_claim = await claim_uc.execute(
        principal=res_principal,
        task_id=parent_id,
        request_id=f"req-{uuid4()}",
        capability="research",
        execution_kind="initial",
    )
    ack_uc = AcknowledgeAssignment(store, PEPPER, clock=lambda: base_dt + timedelta(seconds=1))
    await ack_uc.execute(
        res_principal, parent_id, p_claim.assignment.assignment_id, p_claim.assignment_proof
    )
    await store.save_stage_artifact(
        StageArtifact(
            producing_task_id=parent_id,
            artifact_id="art-wave-1",
            artifact_version=1,
            artifact_type="research_material",
            producing_turn_id=p_claim.assignment.turn_id,
            created_at=now_iso,
            payload_schema_version=1,
            sealed=True,
            content={"material": f"Wave material for {parent_id}"},
        ),
        p_claim.assignment.assignment_id,
        res_principal.principal_id,
        p_claim.assignment.fencing_epoch,
    )
    req = DelegationWaveProposalRequest(
        proposal_id=f"prop-{parent_id}",
        items=_wave_items(parent_id, count=count),
        barrier_policy=barrier_policy,
        required_success_fraction=required_success_fraction,
    )
    wave_uc = AcceptDelegationWave(store, PEPPER, clock=lambda: base_dt + timedelta(seconds=2))
    wave_resp = await wave_uc.execute(
        res_principal,
        parent_id,
        p_claim.assignment.assignment_id,
        p_claim.assignment_proof,
        req,
    )
    return {
        "dt": base_dt,
        "parent_id": parent_id,
        "principal": res_principal,
        "parent_claim": p_claim,
        "request": req,
        "wave_resp": wave_resp,
        "wave_id": wave_resp.wave_id,
        "children": wave_resp.required_child_task_ids,
    }


async def _claim_and_ack_child(
    store: Any,
    child_id: str,
    dt: datetime,
    *,
    principal_suffix: str = "1",
) -> tuple[RuntimePrincipal, Any]:
    prin = _principal(
        principal_id=f"prin-analysis-{principal_suffix}",
        agent_id=f"agent-analysis-{principal_suffix}",
        capabilities=("analysis",),
    )
    c_claim = await ClaimTask(store, PEPPER, clock=lambda: dt).execute(
        prin, child_id, f"req-{child_id}-{uuid4()}", "analysis", "delegated_child"
    )
    await AcknowledgeAssignment(store, PEPPER, clock=lambda: dt + timedelta(seconds=1)).execute(
        prin, child_id, c_claim.assignment.assignment_id, c_claim.assignment_proof
    )
    return prin, c_claim


def _deep_freeze_for_snapshot(value: Any) -> Any:
    if isinstance(value, (dict, MappingProxyType)):
        return {k: _deep_freeze_for_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_freeze_for_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_deep_freeze_for_snapshot(v) for v in value)
    if isinstance(value, set):
        return {_deep_freeze_for_snapshot(v) for v in value}
    return value


def _snapshot_store(store: InMemoryAuthoritativeTaskStore) -> dict[str, Any]:
    return _deep_freeze_for_snapshot(
        {
            "statuses": store.statuses,
            "registrations": store.registrations,
            "lineages": store.lineages,
            "children": store.children,
            "inputs": store.inputs,
            "artifacts": store.artifacts,
            "continuations": store.continuations,
            "latest_continuation": store.latest_continuation,
            "authorities": store.authorities,
            "authority_grants": store.authority_grants,
            "required_capabilities": store.required_capabilities,
            "claims": store.claims,
            "fences": store.fences,
            "proposals": store.proposals,
            "wave_proposals": store.wave_proposals,
            "accepted_waves": store.accepted_waves,
            "resolved_waves": store.resolved_waves,
            "yield_records": store.yield_records,
            "consumed_waves": store.consumed_waves,
            "consumed_grants": store.consumed_grants,
            "claim_deadlines": store.claim_deadlines,
            "hints": store.hints,
            "delegation_decisions": store.delegation_decisions,
            "completion_records": store.completion_records,
            "terminal_outcomes": store.terminal_outcomes,
            "canonical_results": store.canonical_results,
            "assignment_completions": store.assignment_completions,
            "child_markers": store.child_markers,
        }
    )


def _valid_v2_resolved_wave(**overrides: Any) -> ResolvedDelegationWave:
    ts = "2026-11-27T10:00:00Z"
    ts_res = "2026-11-27T10:00:10Z"
    ts_dl = "2026-11-27T10:05:00Z"
    kwargs: dict[str, Any] = {
        "version": RESOLVED_DELEGATION_WAVE_V2_CONTRACT_VERSION,
        "wave_id": "wave-v2-1",
        "parent_task_id": "parent-1",
        "required_child_task_ids": ("child-1", "child-2", "child-3"),
        "accepted_at": ts,
        "accepted_wave_fingerprint": "a" * 64,
        "resolution": "SATISFIED",
        "resolved_at": ts_res,
        "barrier_policy": "PARTIAL",
        "required_success_fraction": {"numerator": 2, "denominator": 3},
        "required_success_count": 2,
        "completed_count": 2,
        "terminal_count": 2,
        "unresolved_count": 1,
        "child_outcomes": {
            "child-1": {"status": "completed", "outcome_id": "out-1"},
            "child-2": {"status": "completed", "outcome_id": "out-2"},
            "child-3": {"status": "ended / barrier_resolved", "outcome_id": None},
        },
        "residual_child_task_ids": ("child-3",),
        "residual_disposition": "ended / barrier_resolved",
        "triggering_child_task_id": "child-2",
        "triggering_assignment_id": "assign-c2",
        "triggering_fencing_epoch": 1,
        "wave_deadline": ts_dl,
        "deadline_policy_id": "deterministic_bounded_baseline_v1",
        "deadline_policy_version": 1,
    }
    kwargs.update(overrides)
    return ResolvedDelegationWave(**kwargs)


# ---------------------------------------------------------------------------
# A. Contracts & Bounds
# ---------------------------------------------------------------------------


def test_resolved_delegation_wave_v2_schema_immutability_and_bounds() -> None:
    rw = _valid_v2_resolved_wave()
    assert rw.version == 2
    assert rw.required_success_fraction == RequiredSuccessFraction(2, 3)
    assert rw.child_outcomes["child-1"]["outcome_id"] == "out-1"
    assert rw.residual_child_task_ids == ("child-3",)

    roundtrip = ResolvedDelegationWave.from_json(rw.to_json())
    assert roundtrip == rw

    # Deep immutability on returned mapping / tuple
    with pytest.raises(TypeError):
        rw.child_outcomes["child-1"] = {"status": "failed", "outcome_id": "x"}  # type: ignore[index]
    with pytest.raises(TypeError):
        rw.child_outcomes["child-1"]["status"] = "failed"  # type: ignore[index]

    # V1 rejects v2-only fields; v2 rejects missing v2 fields
    with pytest.raises(ValueError, match="does not permit v2 fields"):
        ResolvedDelegationWave(
            version=1,
            wave_id="w1",
            parent_task_id="p1",
            required_child_task_ids=("c1", "c2"),
            barrier_policy="WAIT_ALL",
            resolution="SATISFIED",
            resolved_at="2026-11-27T10:00:10Z",
            wave_deadline="2026-11-27T10:05:00Z",
            completed_count=2,
        )
    with pytest.raises(ValueError, match="is required for ResolvedDelegationWave v2"):
        ResolvedDelegationWave(
            version=2,
            wave_id="w1",
            parent_task_id="p1",
            required_child_task_ids=("c1", "c2"),
            barrier_policy="WAIT_ALL",
            resolution="SATISFIED",
            resolved_at="2026-11-27T10:00:10Z",
            wave_deadline="2026-11-27T10:05:00Z",
        )

    # Consistent count and policy guards
    with pytest.raises(ValueError, match="WAIT_ALL wave cannot resolve as UNSATISFIABLE"):
        _valid_v2_resolved_wave(
            barrier_policy="WAIT_ALL",
            resolution="UNSATISFIABLE",
            required_success_fraction=None,
            required_success_count=None,
        )
    with pytest.raises(ValueError, match="PARTIAL SATISFIED resolution requires completed_count"):
        _valid_v2_resolved_wave(
            completed_count=1,
            terminal_count=2,
            unresolved_count=1,
            child_outcomes={
                "child-1": {"status": "completed", "outcome_id": "out-1"},
                "child-2": {"status": "failed", "outcome_id": "out-2"},
                "child-3": {"status": "ended / barrier_resolved", "outcome_id": None},
            },
        )
    with pytest.raises(ValueError, match="PARTIAL UNSATISFIABLE resolution inconsistent with counts"):
        _valid_v2_resolved_wave(
            resolution="UNSATISFIABLE",
            completed_count=2,
            terminal_count=2,
            unresolved_count=1,
        )

    # UTF-8 256-byte identifier bounds & 16 KiB payload bound
    with pytest.raises(ValueError, match="UTF-8 byte bound"):
        _valid_v2_resolved_wave(wave_id="w" * (MAX_WAVE_IDENTIFIER_BYTES + 1))
    with pytest.raises(ValueError, match="UTF-8 byte bound"):
        TaskContinuationV2(
            parent_task_id="p1",
            wave_id="w" * (MAX_WAVE_IDENTIFIER_BYTES + 1),
            required_child_task_ids=("c1", "c2"),
            created_at="2026-11-27T10:00:00Z",
            updated_at="2026-11-27T10:00:00Z",
            artifact_references=(ArtifactReference("p1", "art-1", 1),),
        )
    assert MAX_CONTROL_RECORD_BYTES == 16384


# ---------------------------------------------------------------------------
# B. WAIT_ALL Progression & Terminal-Mix Resolution
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("child_count", [2, 3, 4])
async def test_wait_all_incremental_progression_and_mixed_terminal_resolution(
    child_count: int,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await _setup_accepted_wave(
        store,
        parent_id=f"root-wait-all-{child_count}",
        count=child_count,
        barrier_policy="WAIT_ALL",
    )
    dt = ctx["dt"]
    parent_id = ctx["parent_id"]
    children = ctx["children"]
    initial_hints = len(store.hints)

    # All non-final children complete (mix of success and failure) -> wave stays UNRESOLVED
    for idx, cid in enumerate(children[:-1]):
        prin, c_claim = await _claim_and_ack_child(store, cid, dt + timedelta(seconds=5 + idx))
        complete_uc = CompleteTurn(
            store, store, PEPPER, clock=lambda s=idx: dt + timedelta(seconds=10 + s)
        )
        cmd = (
            CompletionCommand(
                completion_command_id=f"cmd-{cid}",
                assignment_id=c_claim.assignment.assignment_id,
                execution_kind="delegated_child",
                failure_code="EXECUTION_FAILED",
            )
            if idx % 2 == 1
            else CompletionCommand(
                completion_command_id=f"cmd-{cid}",
                assignment_id=c_claim.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=cid, output=f"Output {cid}"),
            )
        )
        resp = await complete_uc.execute(prin, cid, cmd.to_dict(), c_claim.assignment_proof)
        assert not resp.replayed
        assert store.statuses[parent_id] == "waiting_for_children"
        cont = await store.get_continuation(parent_id, 2)
        assert isinstance(cont, TaskContinuationV2)
        assert cont.state == "active" and cont.wave_resolution is None
        assert await store.get_resolved_delegation_wave(parent_id, ctx["wave_id"]) is None
        assert len(store.hints) == initial_hints
        # Last sibling remains pending and claimable
        assert store.statuses[children[-1]] == "pending"

    # Final child completes -> atomic WAIT_ALL SATISFIED resolution
    final_cid = children[-1]
    prin, final_claim = await _claim_and_ack_child(
        store, final_cid, dt + timedelta(seconds=20), principal_suffix="final"
    )
    final_uc = CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=25))
    final_resp = await final_uc.execute(
        prin,
        final_cid,
        CompletionCommand(
            completion_command_id=f"cmd-{final_cid}",
            assignment_id=final_claim.assignment.assignment_id,
            execution_kind="delegated_child",
            failure_code="EXECUTION_FAILED" if child_count == 2 else None,
            completed_result=(
                None
                if child_count == 2
                else TaskResult(task_id=final_cid, output=f"Output {final_cid}")
            ),
        ).to_dict(),
        final_claim.assignment_proof,
    )
    assert not final_resp.replayed
    assert store.statuses[parent_id] == "resumable"
    assert store.fences[parent_id] == 1  # parent fence NOT advanced by child completion
    assert parent_id not in store.claims  # parent NOT claimed automatically

    cont = await store.get_continuation(parent_id, 2)
    assert isinstance(cont, TaskContinuationV2)
    assert cont.state == "released" and cont.wave_resolution == "SATISFIED"

    rw = await store.get_resolved_delegation_wave(parent_id, ctx["wave_id"])
    assert rw is not None
    assert rw.version == 2
    assert rw.resolution == "SATISFIED"
    assert rw.residual_child_task_ids == ()
    assert rw.unresolved_count == 0
    assert rw.terminal_count == child_count
    assert rw.triggering_child_task_id == final_cid
    assert rw.triggering_assignment_id == final_claim.assignment.assignment_id
    assert rw.triggering_fencing_epoch == 1

    # Accepted wave stays immutable with state="unresolved"
    accepted = await store.get_accepted_delegation_wave(parent_id, ctx["wave_id"])
    assert accepted is not None and accepted.state == "unresolved"

    # Exactly one Research wake-up hint emitted
    assert len(store.hints) == initial_hints + 1
    hint = store.hints[-1]
    assert hint.capability == "research"
    assert hint.execution_kind == "resumed_continuation"
    assert hint.expected_continuation_version == 2
    assert hint.task_id == parent_id


# ---------------------------------------------------------------------------
# C. PARTIAL SATISFIED & Residual Closure (pending + active working)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_partial_satisfied_closes_pending_and_active_working_residuals() -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await _setup_accepted_wave(
        store,
        parent_id="root-partial-sat",
        count=4,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 1, "denominator": 2},
    )
    dt = ctx["dt"]
    parent_id = ctx["parent_id"]
    c1, c2, c3, c4 = ctx["children"]
    assert ctx["wave_resp"].required_success_count == 2
    initial_hints = len(store.hints)

    # c3 becomes active WORKING before c1/c2 resolve the wave; c4 remains PENDING
    prin3, claim3 = await _claim_and_ack_child(store, c3, dt + timedelta(seconds=4), principal_suffix="3")
    assert store.statuses[c3] == "working"
    assert store.fences[c3] == 1

    # c1 fails -> still unresolved (0 success + 3 remaining >= 2)
    prin1, claim1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=5), principal_suffix="1")
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=6)).execute(
        prin1,
        c1,
        CompletionCommand(
            completion_command_id="cmd-c1",
            assignment_id=claim1.assignment.assignment_id,
            execution_kind="delegated_child",
            failure_code="EXECUTION_FAILED",
        ).to_dict(),
        claim1.assignment_proof,
    )
    assert store.statuses[parent_id] == "waiting_for_children"

    # c2 succeeds -> count=1 (<2), c3 working, c4 pending -> still unresolved!
    prin2, claim2 = await _claim_and_ack_child(store, c2, dt + timedelta(seconds=7), principal_suffix="2")
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=8)).execute(
        prin2,
        c2,
        CompletionCommand(
            completion_command_id="cmd-c2",
            assignment_id=claim2.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c2, output="c2 ok"),
        ).to_dict(),
        claim2.assignment_proof,
    )
    assert store.statuses[parent_id] == "waiting_for_children"

    # c3 succeeds -> reaches 2 successes -> PARTIAL SATISFIED, closing pending residual c4!
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=10)).execute(
        prin3,
        c3,
        CompletionCommand(
            completion_command_id="cmd-c3",
            assignment_id=claim3.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c3, output="c3 ok"),
        ).to_dict(),
        claim3.assignment_proof,
    )
    rw = await store.get_resolved_delegation_wave(parent_id, ctx["wave_id"])
    assert rw is not None and rw.resolution == "SATISFIED"
    assert rw.residual_child_task_ids == (c4,)
    assert store.statuses[c4] == "ended / barrier_resolved"
    assert c4 not in store.terminal_outcomes and c4 not in store.canonical_results
    assert store.statuses[parent_id] == "resumable"
    assert len(store.hints) == initial_hints + 1

    # Pending residual c4 is permanently non-claimable
    with pytest.raises(TaskNotClaimableError):
        await ClaimTask(store, PEPPER, clock=lambda: dt + timedelta(seconds=12)).execute(
            _principal("prin-late", "agent-late", ("analysis",)),
            c4,
            "req-c4-late",
            "analysis",
            "delegated_child",
        )


@pytest.mark.anyio
async def test_partial_satisfied_closes_active_working_residual_and_invalidates_fence() -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await _setup_accepted_wave(
        store,
        parent_id="root-partial-working-res",
        count=3,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 2, "denominator": 3},
    )
    dt = ctx["dt"]
    parent_id = ctx["parent_id"]
    c1, c2, c3 = ctx["children"]
    initial_hints = len(store.hints)

    prin1, claim1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=4), principal_suffix="1")
    prin2, claim2 = await _claim_and_ack_child(store, c2, dt + timedelta(seconds=4), principal_suffix="2")
    prin3, claim3 = await _claim_and_ack_child(store, c3, dt + timedelta(seconds=4), principal_suffix="3")
    assert store.fences[c3] == 1

    # c1 and c2 succeed while c3 is actively working
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=6)).execute(
        prin1,
        c1,
        CompletionCommand(
            completion_command_id="cmd-c1",
            assignment_id=claim1.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c1, output="c1"),
        ).to_dict(),
        claim1.assignment_proof,
    )
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=8)).execute(
        prin2,
        c2,
        CompletionCommand(
            completion_command_id="cmd-c2",
            assignment_id=claim2.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c2, output="c2"),
        ).to_dict(),
        claim2.assignment_proof,
    )

    assert store.statuses[c3] == "ended / barrier_resolved"
    assert c3 not in store.claims  # active claim removed
    assert store.fences[c3] == 2  # active fence invalidated!
    assert c3 not in store.terminal_outcomes
    assert c3 not in store.canonical_results
    assert len(store.hints) == initial_hints + 1

    # Active worker on c3 tries to complete later -> rejected uniformly via CompleteTurn
    with pytest.raises(CompletionReplayAuthorizationError):
        await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=12)).execute(
            prin3,
            c3,
            CompletionCommand(
                completion_command_id="cmd-c3-late",
                assignment_id=claim3.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=c3, output="c3 late"),
            ).to_dict(),
            claim3.assignment_proof,
        )
    # Direct store call also fails closed on ended / barrier_resolved status
    with pytest.raises(TaskNotClaimableError):
        await store.complete_turn(
            c3,
            CompletionCommand(
                completion_command_id="cmd-c3-direct",
                assignment_id=claim3.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=c3, output="c3 direct"),
            ),
            claim3.assignment.proof_digest if hasattr(claim3.assignment, "proof_digest") else "a" * 64,
            1,
            parent_id,
            parent_id,
            normalize_canonical_utc_timestamp(dt + timedelta(seconds=12)),
            prin3.principal_id,
            prin3.bound_agent_id,
        )
    assert store.statuses[c3] == "ended / barrier_resolved"
    assert len(store.hints) == initial_hints + 1


# ---------------------------------------------------------------------------
# D. PARTIAL UNSATISFIABLE & Residual Closure
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_partial_unsatisfiable_resolves_early_and_closes_residuals() -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await _setup_accepted_wave(
        store,
        parent_id="root-partial-unsat",
        count=3,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 2, "denominator": 3},
    )
    dt = ctx["dt"]
    parent_id = ctx["parent_id"]
    c1, c2, c3 = ctx["children"]
    initial_hints = len(store.hints)

    prin1, claim1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=4), principal_suffix="1")
    prin2, claim2 = await _claim_and_ack_child(store, c2, dt + timedelta(seconds=4), principal_suffix="2")
    prin3, claim3 = await _claim_and_ack_child(store, c3, dt + timedelta(seconds=4), principal_suffix="3")

    # c1 fails -> 0 + 2 >= 2 (still reachable)
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=6)).execute(
        prin1,
        c1,
        CompletionCommand(
            completion_command_id="cmd-c1",
            assignment_id=claim1.assignment.assignment_id,
            execution_kind="delegated_child",
            failure_code="EXECUTION_FAILED",
        ).to_dict(),
        claim1.assignment_proof,
    )
    assert store.statuses[parent_id] == "waiting_for_children"

    # c2 fails -> 0 + 1 < 2 -> immediate UNSATISFIABLE, closing active c3!
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=8)).execute(
        prin2,
        c2,
        CompletionCommand(
            completion_command_id="cmd-c2",
            assignment_id=claim2.assignment.assignment_id,
            execution_kind="delegated_child",
            failure_code="PROVIDER_UNAVAILABLE",
        ).to_dict(),
        claim2.assignment_proof,
    )
    rw = await store.get_resolved_delegation_wave(parent_id, ctx["wave_id"])
    assert rw is not None
    assert rw.resolution == "UNSATISFIABLE"
    assert rw.completed_count == 0
    assert rw.terminal_count == 2
    assert rw.unresolved_count == 1
    assert rw.residual_child_task_ids == (c3,)
    assert store.statuses[c3] == "ended / barrier_resolved"
    assert store.fences[c3] == 2 and c3 not in store.claims
    assert store.statuses[parent_id] == "resumable"
    cont = await store.get_continuation(parent_id, 2)
    assert isinstance(cont, TaskContinuationV2)
    assert cont.state == "released" and cont.wave_resolution == "UNSATISFIABLE"
    assert len(store.hints) == initial_hints + 1

    # c3 late completion rejected cleanly without mutating state
    with pytest.raises(CompletionReplayAuthorizationError):
        await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=10)).execute(
            prin3,
            c3,
            CompletionCommand(
                completion_command_id="cmd-c3",
                assignment_id=claim3.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=c3, output="too late"),
            ).to_dict(),
            claim3.assignment_proof,
        )


# ---------------------------------------------------------------------------
# E. Concurrency, Exact Replay, Changed Replay, & Authorization
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_concurrent_final_siblings_first_authoritative_resolution_wins() -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await _setup_accepted_wave(
        store,
        parent_id="root-race",
        count=2,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 1, "denominator": 2},
    )
    dt = ctx["dt"]
    parent_id = ctx["parent_id"]
    c1, c2 = ctx["children"]
    initial_hints = len(store.hints)

    prin1, claim1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=4), principal_suffix="1")
    prin2, claim2 = await _claim_and_ack_child(store, c2, dt + timedelta(seconds=4), principal_suffix="2")

    uc1 = CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=7))
    uc2 = CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=7))
    res1, res2 = await asyncio.gather(
        uc1.execute(
            prin1,
            c1,
            CompletionCommand(
                completion_command_id="cmd-r1",
                assignment_id=claim1.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=c1, output="winner 1"),
            ).to_dict(),
            claim1.assignment_proof,
        ),
        uc2.execute(
            prin2,
            c2,
            CompletionCommand(
                completion_command_id="cmd-r2",
                assignment_id=claim2.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=c2, output="winner 2"),
            ).to_dict(),
            claim2.assignment_proof,
        ),
        return_exceptions=True,
    )
    successes = [r for r in (res1, res2) if not isinstance(r, BaseException)]
    failures = [r for r in (res1, res2) if isinstance(r, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], (CompletionReplayAuthorizationError, TaskNotClaimableError))
    assert len(store.hints) == initial_hints + 1
    rw = await store.get_resolved_delegation_wave(parent_id, ctx["wave_id"])
    assert rw is not None and len(rw.residual_child_task_ids) == 1


@pytest.mark.anyio
async def test_completion_replay_before_and_after_resolution_and_conflict_auth() -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await _setup_accepted_wave(
        store,
        parent_id="root-replay",
        count=2,
        barrier_policy="WAIT_ALL",
    )
    dt = ctx["dt"]
    parent_id = ctx["parent_id"]
    c1, c2 = ctx["children"]
    initial_hints = len(store.hints)

    prin1, claim1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=4), principal_suffix="1")
    cmd1 = CompletionCommand(
        completion_command_id="cmd-rep-1",
        assignment_id=claim1.assignment.assignment_id,
        execution_kind="delegated_child",
        completed_result=TaskResult(task_id=c1, output="c1 ok"),
    )
    uc = CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=7))
    first = await uc.execute(prin1, c1, cmd1.to_dict(), claim1.assignment_proof)
    # Replay before wave resolution
    rep_before = await uc.execute(prin1, c1, cmd1.to_dict(), claim1.assignment_proof)
    assert rep_before.replayed and rep_before.outcome == first.outcome
    assert store.statuses[parent_id] == "waiting_for_children"
    assert len(store.hints) == initial_hints

    # Second child resolves wave
    prin2, claim2 = await _claim_and_ack_child(store, c2, dt + timedelta(seconds=8), principal_suffix="2")
    cmd2 = CompletionCommand(
        completion_command_id="cmd-rep-2",
        assignment_id=claim2.assignment.assignment_id,
        execution_kind="delegated_child",
        completed_result=TaskResult(task_id=c2, output="c2 ok"),
    )
    second = await uc.execute(prin2, c2, cmd2.to_dict(), claim2.assignment_proof)
    assert not second.replayed
    assert len(store.hints) == initial_hints + 1

    # Replay BOTH children after wave resolution -> zero new hints/resolutions
    rep1_after = await uc.execute(prin1, c1, cmd1.to_dict(), claim1.assignment_proof)
    rep2_after = await uc.execute(prin2, c2, cmd2.to_dict(), claim2.assignment_proof)
    assert rep1_after.replayed and rep2_after.replayed
    assert rep1_after.outcome == first.outcome and rep2_after.outcome == second.outcome
    assert len(store.hints) == initial_hints + 1

    # Changed semantic payload -> conflict at store boundary, uniform auth failure via CompleteTurn
    bad_cmd = CompletionCommand(
        completion_command_id="cmd-rep-1",
        assignment_id=claim1.assignment.assignment_id,
        execution_kind="delegated_child",
        completed_result=TaskResult(task_id=c1, output="tampered"),
    )
    with pytest.raises(CompletionConflictError):
        await store.complete_turn(
            c1,
            bad_cmd,
            "0" * 64,
            1,
            parent_id,
            parent_id,
            normalize_canonical_utc_timestamp(dt + timedelta(seconds=10)),
            prin1.principal_id,
            prin1.bound_agent_id,
        )
    with pytest.raises(CompletionReplayAuthorizationError):
        await uc.execute(prin1, c1, bad_cmd.to_dict(), claim1.assignment_proof)
    # Unauthorized caller / bad proof
    with pytest.raises(CompletionReplayAuthorizationError):
        await uc.execute(
            _principal("wrong-prin", "agent-analysis-1", ("analysis",)),
            c1,
            cmd1.to_dict(),
            claim1.assignment_proof,
        )


# ---------------------------------------------------------------------------
# F. Corruption & Zero-Mutation Matrix
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_corrupt_wave_state_fails_closed_with_zero_mutation() -> None:
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    now_iso = normalize_canonical_utc_timestamp(dt)

    for scenario in (
        "missing_accepted_wave",
        "missing_cont_v2",
        "wrong_latest_ver",
        "missing_sibling_lineage",
        "pre_existing_resolved_wave",
        "parent_not_waiting",
        "child_set_mismatch",
        "bad_barrier_policy",
    ):
        store = InMemoryAuthoritativeTaskStore()
        ctx = await _setup_accepted_wave(
            store, parent_id=f"root-corrupt-{scenario}", count=2, barrier_policy="WAIT_ALL", dt=dt
        )
        parent_id = ctx["parent_id"]
        c1, c2 = ctx["children"]
        prin1, claim1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=4))

        if scenario == "missing_accepted_wave":
            del store.accepted_waves[(parent_id, ctx["wave_id"])]
        elif scenario == "missing_cont_v2":
            del store.continuations[(parent_id, 2)]
        elif scenario == "wrong_latest_ver":
            store.latest_continuation[parent_id] = 1
        elif scenario == "missing_sibling_lineage":
            del store.lineages[c2]
        elif scenario == "pre_existing_resolved_wave":
            store.resolved_waves[(parent_id, ctx["wave_id"])] = _valid_v2_resolved_wave(
                wave_id=ctx["wave_id"],
                parent_task_id=parent_id,
                required_child_task_ids=(c1, c2),
                barrier_policy="WAIT_ALL",
                required_success_fraction=None,
                required_success_count=None,
                completed_count=2,
                terminal_count=2,
                unresolved_count=0,
                residual_child_task_ids=(),
                child_outcomes={
                    c1: {"status": "completed", "outcome_id": "o1"},
                    c2: {"status": "completed", "outcome_id": "o2"},
                },
                triggering_child_task_id=c2,
                triggering_assignment_id="a2",
            )
        elif scenario == "parent_not_waiting":
            store.statuses[parent_id] = "working"
        elif scenario == "child_set_mismatch":
            store.children[parent_id].add("rogue-child")
        elif scenario == "bad_barrier_policy":
            sw = store.accepted_waves[(parent_id, ctx["wave_id"])]
            object.__setattr__(sw, "barrier_policy", "UNKNOWN")

        snap = _snapshot_store(store)
        with pytest.raises((CorruptAuthoritativeRecordError, TaskNotClaimableError, ValueError)):
            await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=6)).execute(
                prin1,
                c1,
                CompletionCommand(
                    completion_command_id="cmd-corrupt",
                    assignment_id=claim1.assignment.assignment_id,
                    execution_kind="delegated_child",
                    completed_result=TaskResult(task_id=c1, output="x"),
                ).to_dict(),
                claim1.assignment_proof,
            )
        # Re-read snapshots match zero mutation (timestamp normalized check)
        assert _snapshot_store(store) == snap, f"Mutation leaked in scenario '{scenario}'"
        assert now_iso  # keep lint clean


# ---------------------------------------------------------------------------
# G. Acceptance Replay After Child Progression & Fraction Parity
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_acceptance_replay_after_progression_and_fraction_parity() -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await _setup_accepted_wave(
        store,
        parent_id="root-acc-prog",
        count=3,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 1, "denominator": 2},
    )
    dt = ctx["dt"]
    parent_id = ctx["parent_id"]
    p_claim = ctx["parent_claim"]
    wave_uc = AcceptDelegationWave(store, PEPPER, clock=lambda: dt + timedelta(seconds=20))
    c1, c2, c3 = ctx["children"]

    # Equivalent fraction 2/4 reduces to canonical 1/2 -> idempotent replay succeeds
    req_equiv_frac = DelegationWaveProposalRequest(
        proposal_id=ctx["request"].proposal_id,
        items=ctx["request"].items,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 2, "denominator": 4},
    )
    equiv_resp = await wave_uc.execute(
        ctx["principal"],
        parent_id,
        p_claim.assignment.assignment_id,
        p_claim.assignment_proof,
        req_equiv_frac,
    )
    assert equiv_resp == ctx["wave_resp"]

    # Non-equivalent fraction 2/3 vs 1/2 -> DelegationReplayAuthorizationError at retired use-case boundary
    req_diff_frac = DelegationWaveProposalRequest(
        proposal_id=ctx["request"].proposal_id,
        items=ctx["request"].items,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 2, "denominator": 3},
    )
    with pytest.raises(DelegationReplayAuthorizationError):
        await wave_uc.execute(
            ctx["principal"],
            parent_id,
            p_claim.assignment.assignment_id,
            p_claim.assignment_proof,
            req_diff_frac,
        )

    # Also verify store-level fraction parity rejection -> DelegationConflictError
    stored_prop = await store.get_delegation_wave_proposal(parent_id, ctx["request"].proposal_id)
    assert stored_prop is not None
    stored_wave = await store.get_accepted_delegation_wave(parent_id, ctx["wave_id"])
    assert stored_wave is not None
    stored_cont = await store.get_continuation(parent_id, 2)
    assert isinstance(stored_cont, TaskContinuationV2)
    stored_yield = await store.get_wave_yield(parent_id, p_claim.assignment.assignment_id)
    assert stored_yield is not None
    mismatched_prop = type(stored_prop).from_dict({
        **stored_prop.to_dict(),
        "required_success_fraction": {"numerator": 2, "denominator": 3},
        "required_success_count": 2,
    })
    with pytest.raises(DelegationConflictError):
        await store.accept_delegation_wave(
            parent_task_id=parent_id,
            proposal=mismatched_prop,
            accepted_wave=stored_wave,
            child_task_inputs=[],
            parent_continuation=stored_cont,
            yield_record=stored_yield,
            wake_up_hints=[],
            expected_fencing_epoch=p_claim.assignment.fencing_epoch,
            now=stored_prop.accepted_at,
        )

    # 1. Replay when c1 is working
    prin1, claim1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=4), principal_suffix="1")
    r1 = await wave_uc.execute(
        ctx["principal"], parent_id, p_claim.assignment.assignment_id, p_claim.assignment_proof, ctx["request"]
    )
    assert r1 == ctx["wave_resp"]

    # 2. Replay when c1 is completed, c2 working, c3 pending
    prin2, claim2 = await _claim_and_ack_child(store, c2, dt + timedelta(seconds=5), principal_suffix="2")
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=6)).execute(
        prin1,
        c1,
        CompletionCommand(
            completion_command_id="cmd-c1",
            assignment_id=claim1.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c1, output="c1"),
        ).to_dict(),
        claim1.assignment_proof,
    )
    r2 = await wave_uc.execute(
        ctx["principal"], parent_id, p_claim.assignment.assignment_id, p_claim.assignment_proof, ctx["request"]
    )
    assert r2 == ctx["wave_resp"]

    # 3. Replay after wave resolved (c1 completed, c2 completed, c3 ended / barrier_resolved, parent resumable)
    await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=8)).execute(
        prin2,
        c2,
        CompletionCommand(
            completion_command_id="cmd-c2",
            assignment_id=claim2.assignment.assignment_id,
            execution_kind="delegated_child",
            completed_result=TaskResult(task_id=c2, output="c2"),
        ).to_dict(),
        claim2.assignment_proof,
    )
    assert store.statuses[c3] == "ended / barrier_resolved"
    r3 = await wave_uc.execute(
        ctx["principal"], parent_id, p_claim.assignment.assignment_id, p_claim.assignment_proof, ctx["request"]
    )
    assert r3 == ctx["wave_resp"]

    # 4. Contradictory progression: remove c1 result -> CorruptAuthoritativeRecordError
    del store.canonical_results[c1]
    with pytest.raises(CorruptAuthoritativeRecordError):
        await wave_uc.execute(
            ctx["principal"], parent_id, p_claim.assignment.assignment_id, p_claim.assignment_proof, ctx["request"]
        )


# ---------------------------------------------------------------------------
# H. V2 Resumed Claim Validation & End-to-End Parent Resumption
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize("target_resolution", ["SATISFIED", "UNSATISFIABLE"])
async def test_v2_resumed_claim_happy_path_and_fail_closed_checks(target_resolution: str) -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await _setup_accepted_wave(
        store,
        parent_id=f"root-resume-{target_resolution}",
        count=2,
        barrier_policy="PARTIAL",
        required_success_fraction={"numerator": 2, "denominator": 2},
    )
    dt = ctx["dt"]
    parent_id = ctx["parent_id"]
    c1, c2 = ctx["children"]

    # Claiming parent with resumed_continuation v2 BEFORE wave resolves fails closed
    res_prin2 = _principal("principal-research-2", "agent-research-2", ("research",))
    with pytest.raises(TaskNotClaimableError):
        await ClaimTask(store, PEPPER, clock=lambda: dt + timedelta(seconds=4)).execute(
            res_prin2,
            parent_id,
            "req-early-resume",
            "research",
            "resumed_continuation",
            expected_continuation_version=2,
        )

    prin1, claim1 = await _claim_and_ack_child(store, c1, dt + timedelta(seconds=5), principal_suffix="1")
    if target_resolution == "SATISFIED":
        prin2, claim2 = await _claim_and_ack_child(store, c2, dt + timedelta(seconds=5), principal_suffix="2")
        for cid, pr, cl in ((c1, prin1, claim1), (c2, prin2, claim2)):
            await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=6)).execute(
                pr,
                cid,
                CompletionCommand(
                    completion_command_id=f"cmd-{cid}",
                    assignment_id=cl.assignment.assignment_id,
                    execution_kind="delegated_child",
                    completed_result=TaskResult(task_id=cid, output=f"out-{cid}"),
                ).to_dict(),
                cl.assignment_proof,
            )
    else:
        await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=6)).execute(
            prin1,
            c1,
            CompletionCommand(
                completion_command_id="cmd-c1-fail",
                assignment_id=claim1.assignment.assignment_id,
                execution_kind="delegated_child",
                failure_code="EXECUTION_FAILED",
            ).to_dict(),
            claim1.assignment_proof,
        )

    assert store.statuses[parent_id] == "resumable"

    # TOCTOU direct-store rejection checks on v2 resumed claim
    bad_claim = ActiveClaim(
        task_id=parent_id,
        assignment_id="assign-toctou",
        claim_id="claim-toctou",
        root_task_id=parent_id,
        turn_id="turn-toctou",
        runtime_principal_id=res_prin2.principal_id,
        bound_agent_id=res_prin2.bound_agent_id,
        assigned_capability="research",
        execution_kind="resumed_continuation",
        continuation_version=2,
        fencing_epoch=2,
        proof_digest="9" * 64,
        request_id="req-toctou",
        claimed_at=normalize_canonical_utc_timestamp(dt + timedelta(seconds=10)),
        delivery_expires_at=normalize_canonical_utc_timestamp(dt + timedelta(seconds=60)),
        prior_claimable_status="resumable",
    )
    saved_rw = store.resolved_waves[(parent_id, ctx["wave_id"])]
    del store.resolved_waves[(parent_id, ctx["wave_id"])]
    with pytest.raises(CorruptAuthoritativeRecordError):
        await store.atomic_claim_task(
            parent_id,
            bad_claim,
            now=normalize_canonical_utc_timestamp(dt + timedelta(seconds=10)),
            expected_continuation_version=2,
            is_root=True,
        )
    store.resolved_waves[(parent_id, ctx["wave_id"])] = saved_rw

    # Reject TIMED_OUT or CANCELLED continuation state
    saved_cont = store.continuations[(parent_id, 2)]
    store.continuations[(parent_id, 2)] = TaskContinuationV2(
        parent_task_id=saved_cont.parent_task_id,
        wave_id=saved_cont.wave_id,
        required_child_task_ids=saved_cont.required_child_task_ids,
        created_at=saved_cont.created_at,
        updated_at=saved_cont.updated_at,
        input_version=saved_cont.input_version,
        continuation_capability=saved_cont.continuation_capability,
        stage=saved_cont.stage,
        control_fields=dict(saved_cont.control_fields),
        artifact_references=saved_cont.artifact_references,
        continuation_version=2,
        state="released",
        wave_resolution="TIMED_OUT",
    )
    with pytest.raises(TaskNotClaimableError):
        await store.atomic_claim_task(
            parent_id,
            bad_claim,
            now=normalize_canonical_utc_timestamp(dt + timedelta(seconds=10)),
            expected_continuation_version=2,
            is_root=True,
        )
    store.continuations[(parent_id, 2)] = saved_cont

    # Reject stale expected_continuation_version=1
    with pytest.raises(TaskNotClaimableError):
        await ClaimTask(store, PEPPER, clock=lambda: dt + timedelta(seconds=10)).execute(
            res_prin2,
            parent_id,
            "req-stale-ver",
            "research",
            "resumed_continuation",
            expected_continuation_version=1,
        )

    # Happy path: claim, read context, acknowledge, complete root parent
    res_claim = await ClaimTask(store, PEPPER, clock=lambda: dt + timedelta(seconds=11)).execute(
        res_prin2,
        parent_id,
        "req-resume-ok",
        "research",
        "resumed_continuation",
        expected_continuation_version=2,
    )
    assert res_claim.assignment.fencing_epoch == 2
    assert res_claim.assignment.continuation_version == 2
    assert store.statuses[parent_id] == "working"
    await AcknowledgeAssignment(store, PEPPER, clock=lambda: dt + timedelta(seconds=12)).execute(
        res_prin2, parent_id, res_claim.assignment.assignment_id, res_claim.assignment_proof
    )
    cont_resumed = await store.get_continuation(parent_id, 2)
    assert isinstance(cont_resumed, TaskContinuationV2)
    assert cont_resumed.state == "released" and cont_resumed.wave_resolution == target_resolution
    rw_resumed = await store.get_resolved_delegation_wave(parent_id, ctx["wave_id"])
    assert rw_resumed is not None and rw_resumed.resolution == target_resolution

    root_done = await CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=15)).execute(
        res_prin2,
        parent_id,
        CompletionCommand(
            completion_command_id="cmd-root-done",
            assignment_id=res_claim.assignment.assignment_id,
            execution_kind="resumed_continuation",
            continuation_version=2,
            completed_result=TaskResult(task_id=parent_id, output=f"final-{target_resolution}"),
        ).to_dict(),
        res_claim.assignment_proof,
    )
    assert root_done.outcome.terminal_status == "completed"
    assert store.statuses[parent_id] == "completed"


# ---------------------------------------------------------------------------
# I. Pure Barrier Evaluator Parity Sanity
# ---------------------------------------------------------------------------


def test_pure_barrier_evaluator_fraction_ceil_and_unsatisfiable_parity() -> None:
    res = evaluate_barrier(
        required_child_task_ids=("c1", "c2", "c3", "c4"),
        child_states={"c1": "completed", "c2": "completed", "c3": "failed", "c4": "pending"},
        barrier_policy="PARTIAL",
        required_success_fraction=RequiredSuccessFraction(2, 3),
    )
    assert res.required_success_count == 3  # ceil(4 * 2 / 3) = ceil(2.666...) = 3
    assert res.outcome.value == "UNRESOLVED"  # 2 + 1 >= 3

    res_unsat = evaluate_barrier(
        required_child_task_ids=("c1", "c2", "c3", "c4"),
        child_states={"c1": "completed", "c2": "failed", "c3": "failed", "c4": "pending"},
        barrier_policy="PARTIAL",
        required_success_fraction=RequiredSuccessFraction(2, 3),
    )
    assert res_unsat.outcome.value == "UNSATISFIABLE"  # 1 + 1 < 3


# ---------------------------------------------------------------------------
# J. Live Redis ACL Gateway Verification (WAIT_ALL, PARTIAL SATISFIED/UNSATISFIABLE, Race, Claim)
# ---------------------------------------------------------------------------


@pytest.mark.live_redis_acl
@pytest.mark.anyio
async def test_live_redis_gateway_increment_3_barrier_resolution() -> None:
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

        # 1. Live Redis WAIT_ALL (3 children, mixed completed + failed -> SATISFIED)
        w_ctx = await _setup_accepted_wave(
            gw,
            parent_id="live-inc3-wait-all",
            count=3,
            barrier_policy="WAIT_ALL",
            dt=dt,
            set_authority_fn=_set_admin_auth,
        )
        res_hints_before = await admin.llen("agent.hints.research")
        for idx, cid in enumerate(w_ctx["children"]):
            pr, cl = await _claim_and_ack_child(gw, cid, dt + timedelta(seconds=5 + idx), principal_suffix=str(idx))
            cmd = (
                CompletionCommand(
                    completion_command_id=f"cmd-live-wa-{idx}",
                    assignment_id=cl.assignment.assignment_id,
                    execution_kind="delegated_child",
                    failure_code="EXECUTION_FAILED" if idx == 1 else None,
                    completed_result=None if idx == 1 else TaskResult(task_id=cid, output=f"ok-{idx}"),
                )
            )
            await CompleteTurn(gw, gw, PEPPER, clock=lambda: dt + timedelta(seconds=10)).execute(
                pr, cid, cmd.to_dict(), cl.assignment_proof
            )
        assert await admin.llen("agent.hints.research") == res_hints_before + 1
        rw_wa = await gw.get_resolved_delegation_wave("live-inc3-wait-all", w_ctx["wave_id"])
        assert rw_wa is not None and rw_wa.resolution == "SATISFIED" and rw_wa.completed_count == 2

        # 2. Live Redis PARTIAL SATISFIED with active working residual + fence invalidation
        p_sat = await _setup_accepted_wave(
            gw,
            parent_id="live-inc3-partial-sat",
            count=3,
            barrier_policy="PARTIAL",
            required_success_fraction={"numerator": 1, "denominator": 2},
            dt=dt,
            set_authority_fn=_set_admin_auth,
        )
        sc1, sc2, sc3 = p_sat["children"]
        sp1, scl1 = await _claim_and_ack_child(gw, sc1, dt + timedelta(seconds=4), principal_suffix="s1")
        sp2, scl2 = await _claim_and_ack_child(gw, sc2, dt + timedelta(seconds=4), principal_suffix="s2")
        sp3, scl3 = await _claim_and_ack_child(gw, sc3, dt + timedelta(seconds=4), principal_suffix="s3")
        uc = CompleteTurn(gw, gw, PEPPER, clock=lambda: dt + timedelta(seconds=8))
        await uc.execute(
            sp1,
            sc1,
            CompletionCommand(
                completion_command_id="cmd-ps-1",
                assignment_id=scl1.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=sc1, output="ps1"),
            ).to_dict(),
            scl1.assignment_proof,
        )
        await uc.execute(
            sp2,
            sc2,
            CompletionCommand(
                completion_command_id="cmd-ps-2",
                assignment_id=scl2.assignment.assignment_id,
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=sc2, output="ps2"),
            ).to_dict(),
            scl2.assignment_proof,
        )
        assert await gw.get_status(sc3) == "ended / barrier_resolved"
        assert await gw.get_fencing_epoch(sc3) == 2
        with pytest.raises(CompletionReplayAuthorizationError):
            await uc.execute(
                sp3,
                sc3,
                CompletionCommand(
                    completion_command_id="cmd-ps-3-late",
                    assignment_id=scl3.assignment.assignment_id,
                    execution_kind="delegated_child",
                    completed_result=TaskResult(task_id=sc3, output="late"),
                ).to_dict(),
                scl3.assignment_proof,
            )

        # Resume parent v2 on live Redis
        rp2 = _principal("prin-res-live-2", "agent-res-live-2", ("research",))
        r_claim = await ClaimTask(gw, PEPPER, clock=lambda: dt + timedelta(seconds=12)).execute(
            rp2,
            "live-inc3-partial-sat",
            "req-live-res",
            "research",
            "resumed_continuation",
            expected_continuation_version=2,
        )
        await AcknowledgeAssignment(gw, PEPPER, clock=lambda: dt + timedelta(seconds=13)).execute(
            rp2, "live-inc3-partial-sat", r_claim.assignment.assignment_id, r_claim.assignment_proof
        )
        done_root = await CompleteTurn(gw, gw, PEPPER, clock=lambda: dt + timedelta(seconds=15)).execute(
            rp2,
            "live-inc3-partial-sat",
            CompletionCommand(
                completion_command_id="cmd-live-root-done",
                assignment_id=r_claim.assignment.assignment_id,
                execution_kind="resumed_continuation",
                continuation_version=2,
                completed_result=TaskResult(task_id="live-inc3-partial-sat", output="all done"),
            ).to_dict(),
            r_claim.assignment_proof,
        )
        assert done_root.outcome.terminal_status == "completed"

        # 3. Live Redis PARTIAL UNSATISFIABLE + Concurrent race
        p_unsat = await _setup_accepted_wave(
            gw,
            parent_id="live-inc3-partial-unsat",
            count=2,
            barrier_policy="PARTIAL",
            required_success_fraction={"numerator": 2, "denominator": 2},
            dt=dt,
            set_authority_fn=_set_admin_auth,
        )
        uc1, uc2 = p_unsat["children"]
        up1, ucl1 = await _claim_and_ack_child(gw, uc1, dt + timedelta(seconds=4), principal_suffix="u1")
        up2, ucl2 = await _claim_and_ack_child(gw, uc2, dt + timedelta(seconds=4), principal_suffix="u2")
        uc_live = CompleteTurn(gw, gw, PEPPER, clock=lambda: dt + timedelta(seconds=9))
        r_a, r_b = await asyncio.gather(
            uc_live.execute(
                up1,
                uc1,
                CompletionCommand(
                    completion_command_id="cmd-pu-1",
                    assignment_id=ucl1.assignment.assignment_id,
                    execution_kind="delegated_child",
                    failure_code="EXECUTION_FAILED",
                ).to_dict(),
                ucl1.assignment_proof,
            ),
            uc_live.execute(
                up2,
                uc2,
                CompletionCommand(
                    completion_command_id="cmd-pu-2",
                    assignment_id=ucl2.assignment.assignment_id,
                    execution_kind="delegated_child",
                    failure_code="PROVIDER_UNAVAILABLE",
                ).to_dict(),
                ucl2.assignment_proof,
            ),
            return_exceptions=True,
        )
        assert sum(1 for item in (r_a, r_b) if not isinstance(item, BaseException)) == 1
        rw_u = await gw.get_resolved_delegation_wave("live-inc3-partial-unsat", p_unsat["wave_id"])
        assert rw_u is not None and rw_u.resolution == "UNSATISFIABLE"
    finally:
        await admin.aclose()
        await gw.aclose()
        cluster.cleanup()
