from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from agent_contracts import (
    MAX_ACCEPTED_WAVE_BYTES,
    MAX_CHILD_INTENT_BYTES,
    MAX_CONTROL_RECORD_BYTES,
    MAX_FRACTION_VALUE,
    MAX_ITEM_CONTROL_FIELDS_BYTES,
    MAX_ITEM_ID_BYTES,
    MAX_WAVE_CHILDREN,
    MAX_WAVE_IDENTIFIER_BYTES,
    MIN_WAVE_CHILDREN,
    AcceptedDelegationWave,
    ArtifactReference,
    CompletionCommand,
    DelegationGrant,
    DelegationGrantV2,
    DelegationWaveProposalItem,
    DelegationWaveProposalRecord,
    DelegationWaveProposalRequest,
    DelegationWaveResponse,
    RequiredSuccessFraction,
    ResolvedDelegationWave,
    RootAuthorityRecord,
    StageArtifact,
    TaskContinuation,
    TaskContinuationV2,
    TaskInput,
    TaskRequested,
    WakeUpHint,
    WaveYieldRecord,
    compute_wave_proposal_fingerprint,
    derive_wave_deadline,
    deserialize_continuation,
    deserialize_delegation_grant,
    normalize_canonical_utc_timestamp,
)
from orchestrator.application.ports import (
    ClaimNotFoundError,
    ClaimTimeoutConflictError,
    DelegationConflictError,
    TaskNotClaimableError,
)
from orchestrator.application.use_cases import (
    AcceptDelegationProposal,
    AcceptDelegationWave,
    AcknowledgeAssignment,
    ClaimTask,
    CompleteTurn,
    DelegationReplayAuthorizationError,
)
from orchestrator.domain.principal import RuntimePrincipal
from orchestrator.domain.task import (
    ConflictingChildTaskRegistrationError,
    CorruptAuthoritativeRecordError,
)

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


async def _setup_acknowledged_wave_parent(
    store: InMemoryAuthoritativeTaskStore,
    *,
    parent_id: str = "root-wave-1",
    max_children: int = MAX_WAVE_CHILDREN,
    now_dt: datetime | None = None,
) -> tuple[RuntimePrincipal, str, str, str, StageArtifact]:
    dt = now_dt or datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    now_iso = normalize_canonical_utc_timestamp(dt)
    req = TaskRequested(
        task_id=parent_id,
        root_task_id=parent_id,
        parent_task_id=None,
        prompt="Research multi-angle wave topic",
        created_at=now_iso,
    )
    await store.register_root_task(req, initial_status="pending", required_capability="research")
    # Override baseline v1 authority with explicit v2 wave authority for Increment 2 tests
    store.authorities[parent_id] = RootAuthorityRecord.create_wave_baseline(
        parent_id, max_children_per_wave=max_children, created_at=now_iso
    )

    principal = _principal()
    claim_uc = ClaimTask(
        store,
        PEPPER,
        clock=lambda: dt,
        allow_wave_delegation_grant=True,
    )
    claim_resp = await claim_uc.execute(
        principal=principal,
        task_id=parent_id,
        request_id=f"req-{uuid4()}",
        capability="research",
        execution_kind="initial",
    )
    ack_uc = AcknowledgeAssignment(store, PEPPER, clock=lambda: dt + timedelta(seconds=1))
    await ack_uc.execute(
        principal=principal,
        task_id=parent_id,
        assignment_id=claim_resp.assignment.assignment_id,
        assignment_proof=claim_resp.assignment_proof,
    )

    active_claim = await store.get_active_claim(parent_id)
    assert active_claim is not None
    artifact = StageArtifact(
        producing_task_id=parent_id,
        artifact_id="art-research-1",
        artifact_version=1,
        artifact_type="research_material",
        producing_turn_id=active_claim.turn_id,
        created_at=now_iso,
        payload_schema_version=1,
        content={"material": "verified findings"},
        sealed=True,
    )
    await store.save_stage_artifact(
        artifact,
        active_claim.assignment_id,
        principal.principal_id,
        active_claim.fencing_epoch,
    )
    return (
        principal,
        parent_id,
        claim_resp.assignment.assignment_id,
        claim_resp.assignment_proof,
        artifact,
    )


def _wave_items(
    parent_id: str,
    count: int = 2,
    artifact_id: str = "art-research-1",
) -> tuple[DelegationWaveProposalItem, ...]:
    ref = ArtifactReference(parent_id, artifact_id, 1)
    return tuple(
        DelegationWaveProposalItem(
            item_id=f"item-{i + 1}",
            child_intent=f"Analyze angle {i + 1}",
            artifact_reference=ref,
            control_fields={"analysis_focus": f"angle_{i + 1}", "min_findings": 1},
        )
        for i in range(count)
    )


# ==============================================================================
# 1. Strict V2 Contracts, Bounds, and Deserialization
# ==============================================================================


def test_delegation_grant_v2_and_root_authority_v2_roundtrip_and_bounds() -> None:
    ts = "2026-11-27T10:00:00Z"
    grant_v2 = DelegationGrantV2(
        grant_id="grant-1",
        root_task_id="root-1",
        issued_to_assignment_id="assign-1",
        issued_to_principal_id="principal-1",
        fencing_epoch=1,
        turn_id="turn-1",
        issued_at=ts,
        max_children=4,
    )
    assert deserialize_delegation_grant(grant_v2.to_json()) == grant_v2

    grant_v1 = DelegationGrant(
        grant_id="grant-v1",
        root_task_id="root-1",
        issued_to_assignment_id="assign-1",
        issued_to_principal_id="principal-1",
        fencing_epoch=1,
        turn_id="turn-1",
        issued_at=ts,
    )
    assert deserialize_delegation_grant(grant_v1.to_json()) == grant_v1

    with pytest.raises(ValueError, match="between 2 and 4"):
        DelegationGrantV2(
            grant_id="grant-bad",
            root_task_id="root-1",
            issued_to_assignment_id="assign-1",
            issued_to_principal_id="principal-1",
            fencing_epoch=1,
            turn_id="turn-1",
            issued_at=ts,
            max_children=5,
        )
    with pytest.raises(TypeError, match="max_children must be an int"):
        DelegationGrantV2(
            grant_id="grant-bad-bool",
            root_task_id="root-1",
            issued_to_assignment_id="assign-1",
            issued_to_principal_id="principal-1",
            fencing_epoch=1,
            turn_id="turn-1",
            issued_at=ts,
            max_children=True,  # type: ignore[arg-type]
        )

    auth_v2 = RootAuthorityRecord.create_wave_baseline("root-1", max_children_per_wave=3, created_at=ts)
    assert auth_v2.version == 2
    assert auth_v2.max_children_per_wave == 3
    assert RootAuthorityRecord.from_json(auth_v2.to_json()) == auth_v2

    with pytest.raises(ValueError, match="between 2 and 4"):
        RootAuthorityRecord.create_wave_baseline("root-1", max_children_per_wave=5, created_at=ts)


def test_wave_proposal_request_bounds_canonicalization_and_forbidden_fields() -> None:
    items_2 = _wave_items("root-1", count=2)
    req = DelegationWaveProposalRequest(
        proposal_id="prop-1",
        items=items_2,
        barrier_policy="PARTIAL",
        required_success_fraction=RequiredSuccessFraction(2, 4),
        work_complexity="medium",
        uncertainty_level="low",
    )
    assert req.required_success_fraction == RequiredSuccessFraction(1, 2)
    assert DelegationWaveProposalRequest.from_json(req.to_json()) == req

    # Fingerprint changes if item order changes
    req_reordered = DelegationWaveProposalRequest(
        proposal_id="prop-1",
        items=(items_2[1], items_2[0]),
        barrier_policy="PARTIAL",
        required_success_fraction=RequiredSuccessFraction(1, 2),
        work_complexity="medium",
        uncertainty_level="low",
    )
    fp1 = compute_wave_proposal_fingerprint(parent_task_id="root-1", assignment_id="a-1", request=req)
    fp2 = compute_wave_proposal_fingerprint(
        parent_task_id="root-1", assignment_id="a-1", request=req_reordered
    )
    assert fp1 != fp2

    # Child count bounds: 1 and 5 rejected
    with pytest.raises(ValueError, match=f"{MIN_WAVE_CHILDREN} to {MAX_WAVE_CHILDREN}"):
        DelegationWaveProposalRequest(
            proposal_id="prop-1",
            items=_wave_items("root-1", count=1),
            barrier_policy="WAIT_ALL",
        )
    with pytest.raises(ValueError, match=f"{MIN_WAVE_CHILDREN} to {MAX_WAVE_CHILDREN}"):
        DelegationWaveProposalRequest(
            proposal_id="prop-1",
            items=_wave_items("root-1", count=5),
            barrier_policy="WAIT_ALL",
        )

    # Duplicate item_id rejected
    with pytest.raises(ValueError, match="Duplicate item_id"):
        DelegationWaveProposalRequest(
            proposal_id="prop-1",
            items=(items_2[0], items_2[0]),
            barrier_policy="WAIT_ALL",
        )

    # Fraction > 10,000 rejected
    with pytest.raises(ValueError, match=str(MAX_FRACTION_VALUE)):
        RequiredSuccessFraction(1, MAX_FRACTION_VALUE + 1)

    # Forbidden authority fields rejected
    with pytest.raises(ValueError, match="Forbidden authority fields"):
        DelegationWaveProposalRequest.from_dict({
            **req.to_dict(),
            "wave_id": "caller-forged-wave",
        })
    with pytest.raises(ValueError, match="Forbidden authority fields"):
        DelegationWaveProposalItem.from_dict({
            **items_2[0].to_dict(),
            "child_task_id": "caller-forged-child",
        })


def test_continuation_deserialization_v1_v2_and_unknown_fails_closed() -> None:
    ts = "2026-11-27T10:00:00Z"
    v1 = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        created_at=ts,
        updated_at=ts,
        continuation_version=1,
    )
    v2 = TaskContinuationV2(
        parent_task_id="p-1",
        wave_id="w-1",
        required_child_task_ids=("c-1", "c-2"),
        created_at=ts,
        updated_at=ts,
        continuation_version=2,
    )
    assert isinstance(deserialize_continuation(v1.to_json(), expected_continuation_version=1), TaskContinuation)
    assert isinstance(deserialize_continuation(v2.to_json(), expected_continuation_version=2), TaskContinuationV2)

    with pytest.raises(ValueError, match="Unsupported contract version"):
        deserialize_continuation({**v2.to_dict(), "version": 3})
    with pytest.raises(ValueError, match="Continuation version mismatch"):
        deserialize_continuation(v2.to_dict(), expected_continuation_version=1)
    with pytest.raises(ValueError, match="version is required"):
        deserialize_continuation({"parent_task_id": "p-1"})


def test_accepted_wave_size_bound_and_record_roundtrips() -> None:
    assert MAX_ACCEPTED_WAVE_BYTES == 16 * 1024
    assert MAX_CONTROL_RECORD_BYTES == 16 * 1024
    ts = "2026-11-27T10:00:00Z"
    rec = DelegationWaveProposalRecord(
        parent_task_id="p-1",
        proposal_id="prop-1",
        wave_id="w-1",
        delegation_decision_id="dec-1",
        assignment_id="a-1",
        runtime_principal_id="prin-1",
        bound_agent_id="agent-1",
        fencing_epoch=1,
        proof_digest="digest",
        proposal_fingerprint="fp",
        delegation_grant_id="grant-1",
        required_child_task_ids=("c-2", "c-1"),
        barrier_policy="WAIT_ALL",
        accepted_at=ts,
        wave_deadline="2026-11-27T10:01:00Z",
        child_claim_deadline="2026-11-27T10:02:00Z",
    )
    assert rec.required_child_task_ids == ("c-1", "c-2")
    assert DelegationWaveProposalRecord.from_json(rec.to_json()) == rec

    y = WaveYieldRecord(
        assignment_id="a-1",
        parent_task_id="p-1",
        principal_id="prin-1",
        proposal_id="prop-1",
        wave_id="w-1",
        required_child_task_ids=("c-2", "c-1"),
        delegation_decision_id="dec-1",
        yielded_at=ts,
    )
    assert y.required_child_task_ids == ("c-1", "c-2")
    assert WaveYieldRecord.from_json(y.to_json()) == y
    with pytest.raises(ValueError, match="is required"):
        WaveYieldRecord.from_dict({"assignment_id": "a-1"})

    resp = DelegationWaveResponse(
        parent_task_id="p-1",
        proposal_id="prop-1",
        wave_id="w-1",
        delegation_decision_id="dec-1",
        required_child_task_ids=("c-1", "c-2"),
        barrier_policy="WAIT_ALL",
        accepted_at=ts,
        wave_deadline="2026-11-27T10:01:00Z",
        child_claim_deadline="2026-11-27T10:02:00Z",
    )
    assert DelegationWaveResponse.from_json(resp.to_json()) == resp
    with pytest.raises(ValueError, match="continuation_version must be 2"):
        DelegationWaveResponse.from_dict({**resp.to_dict(), "continuation_version": 3})


# ==============================================================================
# 2. Wave Acceptance, Consistency, Replay, Races, and Sibling Claimability
# ==============================================================================


@pytest.mark.anyio
@pytest.mark.parametrize("child_count", [2, 3, 4])
async def test_wait_all_wave_acceptance_and_sibling_claimability(child_count: int) -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    principal, parent_id, assign_id, proof, _art = await _setup_acknowledged_wave_parent(
        store, parent_id=f"root-wait-{child_count}", max_children=4, now_dt=dt
    )

    wave_uc = AcceptDelegationWave(store, PEPPER, clock=lambda: dt + timedelta(seconds=2))
    req = DelegationWaveProposalRequest(
        proposal_id=f"prop-wait-{child_count}",
        items=_wave_items(parent_id, count=child_count),
        barrier_policy="WAIT_ALL",
        work_complexity="medium",
        uncertainty_level="low",
    )
    resp = await wave_uc.execute(
        principal=principal,
        parent_task_id=parent_id,
        assignment_id=assign_id,
        assignment_proof=proof,
        request=req,
    )
    assert isinstance(resp, DelegationWaveResponse)
    assert resp.status == "accepted"
    assert resp.barrier_policy == "WAIT_ALL"
    assert len(resp.required_child_task_ids) == child_count
    assert resp.required_child_task_ids == tuple(sorted(resp.required_child_task_ids))
    assert resp.continuation_version == 2

    # Parent lifecycle, retired claim, unchanged fence, consumed wave/grant, no Research hint
    assert store.statuses[parent_id] == "waiting_for_children"
    assert await store.get_active_claim(parent_id) is None
    assert await store.get_fencing_epoch(parent_id) == 1
    assert "1" in store.consumed_waves[parent_id]
    assert store.latest_continuation[parent_id] == 2
    assert len(store.hints) == child_count
    assert all(h.capability == "analysis" and h.execution_kind == "delegated_child" for h in store.hints)

    # AcceptedDelegationWave & TaskContinuationV2 persisted and consistent
    wave = await store.get_accepted_delegation_wave(parent_id, resp.wave_id)
    assert isinstance(wave, AcceptedDelegationWave)
    assert wave.state == "unresolved"
    assert wave.required_child_task_ids == resp.required_child_task_ids
    cont = await store.get_continuation(parent_id, 2)
    assert isinstance(cont, TaskContinuationV2)
    assert cont.wave_id == resp.wave_id
    assert cont.required_child_task_ids == resp.required_child_task_ids
    assert cont.state == "active"
    assert cont.is_claimable is False

    # Every sibling is independently claimable as delegated_child / analysis
    analysis_claim_uc = ClaimTask(store, PEPPER, clock=lambda: dt + timedelta(seconds=5))
    analysis_principal = _principal("prin-analysis-1", "agent-analysis-1", ("analysis",))
    for child_id in resp.required_child_task_ids:
        assert store.statuses[child_id] == "pending"
        assert await store.get_fencing_epoch(child_id) == 0
        child_claim = await analysis_claim_uc.execute(
            principal=analysis_principal,
            task_id=child_id,
            request_id=f"claim-{child_id}",
            capability="analysis",
            execution_kind="delegated_child",
        )
        assert child_claim.assignment.task_id == child_id
        assert child_claim.assignment.fencing_epoch == 1
        assert child_claim.assignment.delegation_grant_id is None

        # Child cannot be claimed as initial or resumed_continuation
        with pytest.raises(TaskNotClaimableError):
            await analysis_claim_uc.execute(
                principal=analysis_principal,
                task_id=child_id,
                request_id=f"bad-init-{child_id}",
                capability="analysis",
                execution_kind="initial",
            )


@pytest.mark.anyio
async def test_partial_wave_acceptance_and_exact_replay_no_duplicate_hints() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    principal, parent_id, assign_id, proof, _art = await _setup_acknowledged_wave_parent(
        store, parent_id="root-partial-1", max_children=3, now_dt=dt
    )

    wave_uc = AcceptDelegationWave(store, PEPPER, clock=lambda: dt + timedelta(seconds=2))
    req = DelegationWaveProposalRequest(
        proposal_id="prop-partial",
        items=_wave_items(parent_id, count=3),
        barrier_policy="PARTIAL",
        required_success_fraction=RequiredSuccessFraction(2, 3),
    )
    first = await wave_uc.execute(
        principal=principal,
        parent_task_id=parent_id,
        assignment_id=assign_id,
        assignment_proof=proof,
        request=req,
    )
    assert first.barrier_policy == "PARTIAL"
    assert first.required_success_count == 2
    assert len(store.hints) == 3

    # Exact replay after claim retirement returns identical wave & child IDs with no new hints
    replay = await wave_uc.execute(
        principal=principal,
        parent_task_id=parent_id,
        assignment_id=assign_id,
        assignment_proof=proof,
        request=req,
    )
    assert replay == first
    assert len(store.hints) == 3

    # Changed semantic payload under same proposal_id fails closed
    changed_req = DelegationWaveProposalRequest(
        proposal_id="prop-partial",
        items=_wave_items(parent_id, count=3),
        barrier_policy="WAIT_ALL",
    )
    with pytest.raises(DelegationReplayAuthorizationError):
        await wave_uc.execute(
            principal=principal,
            parent_task_id=parent_id,
            assignment_id=assign_id,
            assignment_proof=proof,
            request=changed_req,
        )

    # Competing proposal_id after first wave fails closed
    competing_req = DelegationWaveProposalRequest(
        proposal_id="prop-competing",
        items=_wave_items(parent_id, count=2),
        barrier_policy="WAIT_ALL",
    )
    with pytest.raises(DelegationReplayAuthorizationError):
        await wave_uc.execute(
            principal=principal,
            parent_task_id=parent_id,
            assignment_id=assign_id,
            assignment_proof=proof,
            request=competing_req,
        )


@pytest.mark.anyio
async def test_replay_fails_closed_on_wave_continuation_inconsistency() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    principal, parent_id, assign_id, proof, _art = await _setup_acknowledged_wave_parent(
        store, parent_id="root-inconsistent", max_children=2, now_dt=dt
    )
    wave_uc = AcceptDelegationWave(store, PEPPER, clock=lambda: dt + timedelta(seconds=2))
    req = DelegationWaveProposalRequest(
        proposal_id="prop-1",
        items=_wave_items(parent_id, count=2),
        barrier_policy="WAIT_ALL",
    )
    resp = await wave_uc.execute(
        principal=principal,
        parent_task_id=parent_id,
        assignment_id=assign_id,
        assignment_proof=proof,
        request=req,
    )

    # Corrupt TaskContinuationV2 wave_id binding
    bad_cont = TaskContinuationV2(
        parent_task_id=parent_id,
        wave_id="wave-wrong",
        required_child_task_ids=resp.required_child_task_ids,
        created_at=resp.accepted_at,
        updated_at=resp.accepted_at,
        continuation_version=2,
    )
    store.continuations[(parent_id, 2)] = bad_cont

    with pytest.raises(CorruptAuthoritativeRecordError, match="inconsistent"):
        await wave_uc.execute(
            principal=principal,
            parent_task_id=parent_id,
            assignment_id=assign_id,
            assignment_proof=proof,
            request=req,
        )


@pytest.mark.anyio
async def test_all_or_nothing_rejection_on_authorization_and_target_conflicts() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    principal, parent_id, assign_id, proof, _art = await _setup_acknowledged_wave_parent(
        store, parent_id="root-fail-closed", max_children=2, now_dt=dt
    )
    wave_uc = AcceptDelegationWave(store, PEPPER, clock=lambda: dt + timedelta(seconds=2))
    req = DelegationWaveProposalRequest(
        proposal_id="prop-fail",
        items=_wave_items(parent_id, count=2),
        barrier_policy="WAIT_ALL",
    )

    # Wrong principal / proof
    wrong_principal = _principal("other-prin", "other-agent")
    with pytest.raises(ClaimNotFoundError):
        await wave_uc.execute(
            principal=wrong_principal,
            parent_task_id=parent_id,
            assignment_id=assign_id,
            assignment_proof=proof,
            request=req,
        )
    assert store.statuses[parent_id] == "working"
    assert len(store.hints) == 0
    assert len(store.accepted_waves) == 0

    # Unsealed artifact fails closed with zero mutation
    bad_items = _wave_items(parent_id, count=2, artifact_id="missing-art")
    bad_req = DelegationWaveProposalRequest(
        proposal_id="prop-bad-art",
        items=bad_items,
        barrier_policy="WAIT_ALL",
    )
    with pytest.raises(ValueError, match="Referenced artifact"):
        await wave_uc.execute(
            principal=principal,
            parent_task_id=parent_id,
            assignment_id=assign_id,
            assignment_proof=proof,
            request=bad_req,
        )
    assert store.statuses[parent_id] == "working"
    assert len(store.hints) == 0
    assert len(store.accepted_waves) == 0

    # Pre-existing child key conflict in store.accept_delegation_wave leaves zero partial state
    store.statuses["preexisting-child"] = "pending"
    with pytest.raises(ConflictingChildTaskRegistrationError):
        bad_wave = AcceptedDelegationWave(
            wave_id="wave-conflict",
            parent_task_id=parent_id,
            required_child_task_ids=("other-child", "preexisting-child"),
            barrier_policy="WAIT_ALL",
            deadline_derivation=AcceptDelegationWave(store, PEPPER).execute.__globals__["derive_wave_deadline"](
                work_complexity="medium",
                uncertainty_level="low",
                required_child_count=2,
                planned_parallelism=2,
                min_duration_seconds=30,
                max_duration_seconds=3600,
                accepted_at="2026-11-27T10:00:02Z",
            ),
            accepted_at="2026-11-27T10:00:02Z",
        )
        await store.accept_delegation_wave(
            parent_task_id=parent_id,
            proposal=DelegationWaveProposalRecord(
                parent_task_id=parent_id,
                proposal_id="prop-conflict",
                wave_id="wave-conflict",
                delegation_decision_id="dec-conflict",
                assignment_id=assign_id,
                runtime_principal_id=principal.principal_id,
                bound_agent_id=principal.bound_agent_id,
                fencing_epoch=1,
                proof_digest=(await store.get_active_claim(parent_id)).proof_digest,  # type: ignore[union-attr]
                proposal_fingerprint="fp",
                delegation_grant_id=(await store.get_active_claim(parent_id)).delegation_grant_id,  # type: ignore[union-attr,arg-type]
                required_child_task_ids=bad_wave.required_child_task_ids,
                barrier_policy="WAIT_ALL",
                accepted_at="2026-11-27T10:00:02Z",
                wave_deadline=bad_wave.wave_deadline or "2026-11-27T10:01:02Z",
                child_claim_deadline="2026-11-27T10:02:02Z",
            ),
            accepted_wave=bad_wave,
            child_task_inputs=[
                TaskInput("other-child", "p1", "2026-11-27T10:00:02Z", 1, (_art_ref(parent_id),)),
                TaskInput("preexisting-child", "p2", "2026-11-27T10:00:02Z", 1, (_art_ref(parent_id),)),
            ],
            parent_continuation=TaskContinuationV2(
                parent_task_id=parent_id,
                wave_id="wave-conflict",
                required_child_task_ids=bad_wave.required_child_task_ids,
                created_at="2026-11-27T10:00:02Z",
                updated_at="2026-11-27T10:00:02Z",
                continuation_version=2,
            ),
            yield_record=WaveYieldRecord(
                assignment_id=assign_id,
                parent_task_id=parent_id,
                principal_id=principal.principal_id,
                proposal_id="prop-conflict",
                wave_id="wave-conflict",
                required_child_task_ids=bad_wave.required_child_task_ids,
                delegation_decision_id="dec-conflict",
                yielded_at="2026-11-27T10:00:02Z",
            ),
            wake_up_hints=[
                WakeUpHint("h1", "analysis", "2026-11-27T10:00:02Z", "delegated_child", "other-child"),
                WakeUpHint("h2", "analysis", "2026-11-27T10:00:02Z", "delegated_child", "preexisting-child"),
            ],
            expected_fencing_epoch=1,
            now="2026-11-27T10:00:02Z",
        )
    assert "other-child" not in store.statuses
    assert store.statuses[parent_id] == "working"


def _art_ref(parent_id: str) -> ArtifactReference:
    return ArtifactReference(parent_id, "art-research-1", 1)


@pytest.mark.anyio
async def test_v1_single_child_path_remains_unchanged_when_opt_in_false() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    now_iso = normalize_canonical_utc_timestamp(dt)
    parent_id = "root-v1-compat"
    await store.register_root_task(
        TaskRequested(
            task_id=parent_id,
            prompt="V1 task",
            created_at=now_iso,
            root_task_id=parent_id,
            parent_task_id=None,
        ),
        initial_status="pending",
        required_capability="research",
    )
    principal = _principal()
    claim_uc = ClaimTask(store, PEPPER, clock=lambda: dt)  # default allow_wave_delegation_grant=False
    claim_resp = await claim_uc.execute(principal, parent_id, "req-v1", "research", "initial")
    assert claim_resp.assignment.delegation_grant_id is not None
    assert isinstance(
        await store.get_authority_grant(parent_id, claim_resp.assignment.delegation_grant_id),
        DelegationGrant,
    )

    ack_uc = AcknowledgeAssignment(store, PEPPER, clock=lambda: dt + timedelta(seconds=1))
    await ack_uc.execute(principal, parent_id, claim_resp.assignment.assignment_id, claim_resp.assignment_proof)
    active_claim = await store.get_active_claim(parent_id)
    assert active_claim is not None
    await store.save_stage_artifact(
        StageArtifact(
            producing_task_id=parent_id,
            artifact_id="art-v1",
            artifact_version=1,
            artifact_type="research_material",
            producing_turn_id=active_claim.turn_id,
            created_at=now_iso,
            payload_schema_version=1,
            content={"material": "ok"},
            sealed=True,
        ),
        active_claim.assignment_id,
        principal.principal_id,
        active_claim.fencing_epoch,
    )

    prop_uc = AcceptDelegationProposal(store, PEPPER, clock=lambda: dt + timedelta(seconds=2))
    v1_resp = await prop_uc.execute(
        principal=principal,
        parent_task_id=parent_id,
        proposal_id="prop-v1",
        assignment_id=claim_resp.assignment.assignment_id,
        assignment_proof=claim_resp.assignment_proof,
        child_intent="Single child",
        artifact_reference=ArtifactReference(parent_id, "art-v1", 1),
    )
    assert v1_resp.continuation_version == 1
    assert isinstance(await store.get_continuation(parent_id, 1), TaskContinuation)


def test_root_authority_bool_and_utf8_payload_bounds_and_deadline_overflow() -> None:
    ts = "2026-11-27T10:00:00Z"
    with pytest.raises(TypeError, match="max_delegation_depth must be an int"):
        RootAuthorityRecord(
            root_task_id="r-1",
            created_at=ts,
            max_delegation_depth=True,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="max_delegation_waves must be an int"):
        RootAuthorityRecord(
            root_task_id="r-1",
            created_at=ts,
            max_delegation_waves=True,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="max_children_per_wave must be an int"):
        RootAuthorityRecord(
            root_task_id="r-1",
            created_at=ts,
            max_children_per_wave=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Unsupported contract version"):
        RootAuthorityRecord(root_task_id="r-1", created_at=ts, version=3)

    ref = ArtifactReference("r-1", "art-1", 1)
    # Multibyte UTF-8 child_intent (<2000 chars, >2000 UTF-8 bytes)
    multibyte_intent = "\U0001f680" * (MAX_CHILD_INTENT_BYTES // 4 + 1)
    assert len(multibyte_intent) < MAX_CHILD_INTENT_BYTES
    with pytest.raises(ValueError, match="UTF-8 byte bound"):
        DelegationWaveProposalItem(
            item_id="i-1",
            child_intent=multibyte_intent,
            artifact_reference=ref,
        )

    with pytest.raises(ValueError, match="UTF-8 byte bound"):
        DelegationWaveProposalItem(
            item_id="x" * (MAX_ITEM_ID_BYTES + 1),
            child_intent="Analyze",
            artifact_reference=ref,
        )

    with pytest.raises(ValueError, match="exceeds maximum bound"):
        DelegationWaveProposalItem(
            item_id="i-1",
            child_intent="Analyze",
            artifact_reference=ref,
            control_fields={"material_summary": "a" * (MAX_ITEM_CONTROL_FIELDS_BYTES + 1)},
        )

    with pytest.raises(ValueError, match="UTF-8 byte bound"):
        DelegationWaveProposalRequest(
            proposal_id="p" * (MAX_WAVE_IDENTIFIER_BYTES + 1),
            items=_wave_items("r-1", count=2),
            barrier_policy="WAIT_ALL",
        )

    # Aggregate 4-item request > 16 KiB
    big_items = tuple(
        DelegationWaveProposalItem(
            item_id=f"item-{i}",
            child_intent="x" * 1800,
            artifact_reference=ref,
            control_fields={"material_summary": "y" * 3500},
        )
        for i in range(4)
    )
    with pytest.raises(ValueError, match="exceeds maximum bound"):
        DelegationWaveProposalRequest(
            proposal_id="prop-agg",
            items=big_items,
            barrier_policy="WAIT_ALL",
        )

    # OverflowError in deadline derivation converted to deliberate ValueError
    with pytest.raises(ValueError, match="overflows supported datetime range"):
        derive_wave_deadline(
            work_complexity="low",
            uncertainty_level="low",
            required_child_count=1,
            planned_parallelism=1,
            min_duration_seconds=10**12,
            max_duration_seconds=10**12,
            accepted_at=ts,
        )

    # ResolvedDelegationWave 16 KiB bound
    with pytest.raises(ValueError, match="exceeds maximum bound"):
        ResolvedDelegationWave(
            wave_id="w" * (MAX_CONTROL_RECORD_BYTES + 1),
            parent_task_id="p-1",
            required_child_task_ids=("c-1", "c-2"),
            barrier_policy="WAIT_ALL",
            resolution="SATISFIED",
            resolved_at=ts,
            wave_deadline=ts,
        )


@pytest.mark.anyio
async def test_standalone_save_continuation_rejects_v2_and_claim_validates_grant_bounds() -> None:
    store = InMemoryAuthoritativeTaskStore()
    ts = "2026-11-27T10:00:00Z"
    v1 = TaskContinuation("p-1", "c-1", ts, ts, continuation_version=1)
    await store.save_continuation(v1)
    assert await store.get_continuation("p-1", 1) == v1

    v2 = TaskContinuationV2(
        parent_task_id="p-2",
        wave_id="w-1",
        required_child_task_ids=("c-1", "c-2"),
        created_at=ts,
        updated_at=ts,
        continuation_version=2,
    )
    with pytest.raises(ValueError, match="v1 only"):
        await store.save_continuation(v2)  # type: ignore[arg-type]
    assert ("p-2", 2) not in store.continuations
    assert "p-2" not in store.latest_continuation

    # atomic_claim_task rejects v2 grant when root authority is v1 or max_children exceeds auth
    await store.register_root_task(
        TaskRequested(task_id="r-v1", prompt="p", created_at=ts, root_task_id="r-v1", parent_task_id=None),
        initial_status="pending",
        required_capability="research",
    )
    principal = _principal()
    claim_uc = ClaimTask(store, PEPPER, clock=lambda: datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC))
    resp = await claim_uc.execute(principal, "r-v1", "req-1", "research", "initial")
    assert resp.assignment.fencing_epoch == 1

    await store.register_root_task(
        TaskRequested(task_id="r-v2", prompt="p", created_at=ts, root_task_id="r-v2", parent_task_id=None),
        initial_status="pending",
        required_capability="research",
    )
    store.authorities["r-v2"] = RootAuthorityRecord.create_wave_baseline("r-v2", max_children_per_wave=2, created_at=ts)
    active = await store.get_active_claim("r-v1")
    assert active is not None
    bad_grant = DelegationGrantV2(
        grant_id="g-bad",
        root_task_id="r-v2",
        issued_to_assignment_id="a-bad",
        issued_to_principal_id=principal.principal_id,
        fencing_epoch=1,
        turn_id="t-bad",
        issued_at=ts,
        max_children=3,  # exceeds root_authority max_children_per_wave=2
    )
    from agent_contracts import ActiveClaim
    bad_claim = ActiveClaim(
        task_id="r-v2",
        assignment_id="a-bad",
        claim_id="c-bad",
        root_task_id="r-v2",
        turn_id="t-bad",
        runtime_principal_id=principal.principal_id,
        bound_agent_id=principal.bound_agent_id,
        assigned_capability="research",
        execution_kind="initial",
        continuation_version=None,
        fencing_epoch=1,
        proof_digest=active.proof_digest,
        request_id="req-bad",
        claimed_at=ts,
        delivery_expires_at="2026-11-27T10:01:00Z",
        delegation_grant_id="g-bad",
    )
    with pytest.raises(DelegationConflictError):
        await store.atomic_claim_task("r-v2", bad_claim, grant=bad_grant, now=ts, is_root=True)
    assert store.statuses["r-v2"] == "pending"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "corruption_kind",
    [
        "missing_wave",
        "missing_continuation",
        "bad_continuation_latest",
        "bad_parent_status",
        "extra_child",
        "missing_yield",
        "bad_child_status",
        "bad_child_fence",
    ],
)
async def test_replay_projection_corruption_matrix_fails_closed(corruption_kind: str) -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    principal, parent_id, assign_id, proof, _art = await _setup_acknowledged_wave_parent(
        store, parent_id=f"root-corrupt-{corruption_kind}", max_children=2, now_dt=dt
    )
    wave_uc = AcceptDelegationWave(store, PEPPER, clock=lambda: dt + timedelta(seconds=2))
    req = DelegationWaveProposalRequest(
        proposal_id="prop-corrupt",
        items=_wave_items(parent_id, count=2),
        barrier_policy="WAIT_ALL",
    )
    resp = await wave_uc.execute(principal, parent_id, assign_id, proof, req)

    if corruption_kind == "missing_wave":
        del store.accepted_waves[(parent_id, resp.wave_id)]
    elif corruption_kind == "missing_continuation":
        del store.continuations[(parent_id, 2)]
    elif corruption_kind == "bad_continuation_latest":
        store.latest_continuation[parent_id] = 1
    elif corruption_kind == "bad_parent_status":
        store.statuses[parent_id] = "resumable"
    elif corruption_kind == "extra_child":
        store.children[parent_id].add("rogue-child")
    elif corruption_kind == "missing_yield":
        del store.yield_records[(parent_id, assign_id)]
    elif corruption_kind == "bad_child_status":
        store.statuses[resp.required_child_task_ids[0]] = "completed"
    elif corruption_kind == "bad_child_fence":
        store.fences[resp.required_child_task_ids[0]] = 1

    with pytest.raises(CorruptAuthoritativeRecordError):
        await wave_uc.execute(principal, parent_id, assign_id, proof, req)


@pytest.mark.anyio
async def test_v2_child_completion_and_timeout_fail_closed_without_parent_resumption() -> None:
    store = InMemoryAuthoritativeTaskStore()
    dt = datetime(2026, 11, 27, 10, 0, 0, tzinfo=UTC)
    principal, parent_id, assign_id, proof, _art = await _setup_acknowledged_wave_parent(
        store, parent_id="root-v2-term", max_children=2, now_dt=dt
    )
    wave_uc = AcceptDelegationWave(store, PEPPER, clock=lambda: dt + timedelta(seconds=2))
    resp = await wave_uc.execute(
        principal,
        parent_id,
        assign_id,
        proof,
        DelegationWaveProposalRequest(
            proposal_id="prop-term",
            items=_wave_items(parent_id, count=2),
            barrier_policy="WAIT_ALL",
        ),
    )
    c1, c2 = resp.required_child_task_ids
    initial_hints = len(store.hints)

    # Claim and acknowledge child 1, then complete child 1 via WAIT_ALL projection ->
    # child 1 reaches failed outcome without releasing parent because child 2 is still pending.
    analysis_prin = _principal("prin-analysis-1", "agent-analysis-1", ("analysis",))
    claim_uc = ClaimTask(store, PEPPER, clock=lambda: dt + timedelta(seconds=5))
    c1_claim = await claim_uc.execute(
        analysis_prin, c1, "req-c1", "analysis", "delegated_child"
    )
    ack_uc = AcknowledgeAssignment(store, PEPPER, clock=lambda: dt + timedelta(seconds=6))
    await ack_uc.execute(analysis_prin, c1, c1_claim.assignment.assignment_id, c1_claim.assignment_proof)

    complete_uc = CompleteTurn(store, store, PEPPER, clock=lambda: dt + timedelta(seconds=7))
    cmd = CompletionCommand(
        completion_command_id="cmd-c1",
        assignment_id=c1_claim.assignment.assignment_id,
        execution_kind="delegated_child",
        failure_code="EXECUTION_FAILED",
    )
    c1_resp = await complete_uc.execute(
        principal=analysis_prin,
        task_id=c1,
        assignment_proof=c1_claim.assignment_proof,
        command_payload=cmd.to_dict(),
    )
    assert c1_resp.outcome.terminal_status == "failed"
    assert store.statuses[parent_id] == "waiting_for_children"
    assert store.statuses[c1] == "failed"
    assert c1 in store.terminal_outcomes
    assert (parent_id, resp.wave_id) not in store.resolved_waves
    assert len(store.hints) == initial_hints

    # Attempt v1 claim_timeout on unclaimed child 2 after deadline -> must still fail closed (Increment 4)
    with pytest.raises((ClaimTimeoutConflictError, TaskNotClaimableError, CorruptAuthoritativeRecordError)):
        await store.materialize_claim_timeout(c2, now="2026-11-27T10:10:00Z")
    assert store.statuses[parent_id] == "waiting_for_children"
    assert store.statuses[c2] == "pending"
    assert c2 not in store.terminal_outcomes
    assert len(store.hints) == initial_hints


@pytest.mark.live_redis_acl
@pytest.mark.anyio
async def test_live_redis_gateway_wave_acceptance_claim_replay_and_fail_closed() -> None:
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
        now_iso = normalize_canonical_utc_timestamp(dt)
        principal = _principal()

        # 1. V1 Research root claim still succeeds
        v1_root = "live-v1-root"
        await gw.register_root_task(
            TaskRequested(task_id=v1_root, prompt="v1", created_at=now_iso, root_task_id=v1_root, parent_task_id=None),
            initial_status="pending",
            required_capability="research",
        )
        v1_claim_uc = ClaimTask(gw, PEPPER, clock=lambda: dt)
        v1_claim = await v1_claim_uc.execute(principal, v1_root, "req-v1", "research", "initial")
        assert v1_claim.assignment.delegation_grant_id is not None
        assert isinstance(await gw.get_authority_grant(v1_root, v1_claim.assignment.delegation_grant_id), DelegationGrant)

        # 2. V2 opted-in Research root claim succeeds with v2 authority and fails on mismatched grant/authority
        for child_count in (2, 3, 4):
            parent_id = f"live-v2-root-{child_count}"
            await gw.register_root_task(
                TaskRequested(task_id=parent_id, prompt="v2 wave", created_at=now_iso, root_task_id=parent_id, parent_task_id=None),
                initial_status="pending",
                required_capability="research",
            )
            await admin.set(
                f"task.authority:{parent_id}",
                RootAuthorityRecord.create_wave_baseline(parent_id, max_children_per_wave=4, created_at=now_iso).to_json(),
            )
            v2_claim_uc = ClaimTask(gw, PEPPER, clock=lambda: dt, allow_wave_delegation_grant=True)
            v2_claim = await v2_claim_uc.execute(principal, parent_id, f"req-v2-{child_count}", "research", "initial")
            assert v2_claim.assignment.delegation_grant_id is not None
            grant = await gw.get_authority_grant(parent_id, v2_claim.assignment.delegation_grant_id)
            assert isinstance(grant, DelegationGrantV2)

            await AcknowledgeAssignment(gw, PEPPER, clock=lambda: dt + timedelta(seconds=1)).execute(
                principal, parent_id, v2_claim.assignment.assignment_id, v2_claim.assignment_proof
            )
            active_claim = await gw.get_active_claim(parent_id)
            assert active_claim is not None
            await gw.save_stage_artifact(
                StageArtifact(
                    producing_task_id=parent_id,
                    artifact_id="art-live-1",
                    artifact_version=1,
                    artifact_type="research_material",
                    producing_turn_id=active_claim.turn_id,
                    created_at=now_iso,
                    payload_schema_version=1,
                    content={"material": "live findings"},
                    sealed=True,
                ),
                active_claim.assignment_id,
                principal.principal_id,
                active_claim.fencing_epoch,
            )

            wave_uc = AcceptDelegationWave(gw, PEPPER, clock=lambda: dt + timedelta(seconds=2))
            req = DelegationWaveProposalRequest(
                proposal_id=f"prop-live-{child_count}",
                items=_wave_items(parent_id, count=child_count, artifact_id="art-live-1"),
                barrier_policy="PARTIAL" if child_count == 3 else "WAIT_ALL",
                required_success_fraction=RequiredSuccessFraction(2, 3) if child_count == 3 else None,
            )
            resp = await wave_uc.execute(
                principal, parent_id, v2_claim.assignment.assignment_id, v2_claim.assignment_proof, req
            )
            assert len(resp.required_child_task_ids) == child_count
            assert await gw.get_status(parent_id) == "waiting_for_children"
            assert await gw.get_latest_continuation_version(parent_id) == 2
            cont = await gw.get_continuation(parent_id, 2)
            assert isinstance(cont, TaskContinuationV2)
            assert cont.wave_id == resp.wave_id

            # Exact replay returns identical wave & child IDs without duplicate hints
            hint_count_before = await admin.llen("agent.hints.analysis")
            replay = await wave_uc.execute(
                principal, parent_id, v2_claim.assignment.assignment_id, v2_claim.assignment_proof, req
            )
            assert replay == resp
            assert await admin.llen("agent.hints.analysis") == hint_count_before

            # Standalone save_continuation(v2) rejected
            with pytest.raises(ValueError, match="v1 only"):
                await gw.save_continuation(cont)  # type: ignore[arg-type]

        # 3. First v2 WAIT_ALL child completion projects failed child without parent resumption; timeout fails closed
        p2 = "live-v2-root-2"
        children_p2 = await gw.get_children(p2)
        c1, c2 = sorted(children_p2)
        analysis_prin = _principal("prin-analysis-live", "agent-analysis-live", ("analysis",))
        c1_claim = await ClaimTask(gw, PEPPER, clock=lambda: dt + timedelta(seconds=5)).execute(
            analysis_prin, c1, "req-live-c1", "analysis", "delegated_child"
        )
        await AcknowledgeAssignment(gw, PEPPER, clock=lambda: dt + timedelta(seconds=6)).execute(
            analysis_prin, c1, c1_claim.assignment.assignment_id, c1_claim.assignment_proof
        )
        complete_uc = CompleteTurn(gw, gw, PEPPER, clock=lambda: dt + timedelta(seconds=7))
        c1_live_resp = await complete_uc.execute(
            principal=analysis_prin,
            task_id=c1,
            assignment_proof=c1_claim.assignment_proof,
            command_payload=CompletionCommand(
                completion_command_id="cmd-live-c1",
                assignment_id=c1_claim.assignment.assignment_id,
                execution_kind="delegated_child",
                failure_code="EXECUTION_FAILED",
            ).to_dict(),
        )
        assert c1_live_resp.outcome.terminal_status == "failed"
        assert await gw.get_status(p2) == "waiting_for_children"
        assert await gw.get_status(c1) == "failed"

        with pytest.raises((ClaimTimeoutConflictError, TaskNotClaimableError, CorruptAuthoritativeRecordError)):
            await gw.materialize_claim_timeout(c2, now="2026-11-27T10:10:00Z")
        assert await gw.get_status(p2) == "waiting_for_children"
        assert await gw.get_status(c2) == "pending"
    finally:
        await admin.aclose()
        await gw.aclose()
        cluster.cleanup()
