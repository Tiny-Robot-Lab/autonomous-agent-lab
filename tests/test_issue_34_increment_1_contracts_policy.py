from datetime import UTC, datetime
from types import MappingProxyType

import pytest
from agent_contracts import (
    CONTINUATION_V2_STATE_ACTIVE,
    CONTINUATION_V2_STATE_CLOSED,
    CONTINUATION_V2_STATE_RELEASED,
    MAX_CONTINUATION_BYTES,
    RESIDUAL_DISPOSITION_ENDED_BARRIER_RESOLVED,
    AcceptedDelegationWave,
    ArtifactReference,
    BarrierEvaluationOutcome,
    BarrierPolicy,
    RequiredSuccessFraction,
    ResolvedDelegationWave,
    TaskContinuation,
    TaskContinuationV2,
    WaveDeadlineDerivation,
    WaveResolution,
    derive_wave_deadline,
    evaluate_barrier,
)

# ==============================================================================
# 1. TaskContinuation V1 Compatibility & V2 Evolution
# ==============================================================================


def test_task_continuation_v1_shape_and_roundtrip_unchanged() -> None:
    ref = ArtifactReference("parent-1", "art-1", 1)
    v1 = TaskContinuation.create(
        parent_task_id="parent-1",
        child_task_id="child-1",
        control_fields={"analysis_focus": "latency", "min_findings": 1},
        artifact_references=(ref,),
    )
    assert v1.version == 1
    assert v1.child_task_id == "child-1"
    roundtrip = TaskContinuation.from_json(v1.to_json())
    assert roundtrip == v1

    with pytest.raises(ValueError, match="Unsupported contract version"):
        TaskContinuation.from_dict({**v1.to_dict(), "version": 2})


def test_task_continuation_v2_roundtrip_and_canonical_membership() -> None:
    ref = ArtifactReference("parent-1", "art-1", 1)
    c1 = TaskContinuationV2(
        parent_task_id="parent-1",
        wave_id="wave-1",
        required_child_task_ids=("child-b", "child-a", "child-c"),
        created_at="2026-11-26T10:00:00+00:00",
        updated_at="2026-11-26T10:00:00+00:00",
        control_fields={"analysis_focus": "reliability", "max_findings": 5},
        artifact_references=(ref,),
    )
    c2 = TaskContinuationV2(
        parent_task_id="parent-1",
        wave_id="wave-1",
        required_child_task_ids=("child-c", "child-a", "child-b"),
        created_at="2026-11-26T10:00:00Z",
        updated_at="2026-11-26T10:00:00Z",
        control_fields={"max_findings": 5, "analysis_focus": "reliability"},
        artifact_references=(ref,),
    )

    assert c1.version == 2
    assert c1.required_child_task_ids == ("child-a", "child-b", "child-c")
    assert c1.to_json() == c2.to_json()
    assert c1.semantic_fingerprint() == c2.semantic_fingerprint()
    assert TaskContinuationV2.from_json(c1.to_json()) == c1
    assert isinstance(c1.control_fields, MappingProxyType)
    assert c1.is_claimable is False
    assert c1.is_closed is False


def test_task_continuation_v2_membership_and_validation_rejections() -> None:
    ts = "2026-11-26T10:00:00Z"
    # Empty membership
    with pytest.raises(ValueError, match="must not be empty"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=(),
            created_at=ts,
            updated_at=ts,
        )

    # Duplicate child IDs
    with pytest.raises(ValueError, match="Duplicate child task ID"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1", " c-1 "),
            created_at=ts,
            updated_at=ts,
        )

    # Blank child ID
    with pytest.raises(ValueError, match="must be a non-empty string"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1", "   "),
            created_at=ts,
            updated_at=ts,
        )

    # Self-child (parent ID in required children)
    with pytest.raises(ValueError, match="must not contain parent_task_id"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1", "p-1"),
            created_at=ts,
            updated_at=ts,
        )

    # Blank parent_task_id / wave_id
    with pytest.raises(ValueError, match="parent_task_id must be a non-empty string"):
        TaskContinuationV2(
            parent_task_id="  ",
            wave_id="w-1",
            required_child_task_ids=("c-1",),
            created_at=ts,
            updated_at=ts,
        )
    with pytest.raises(ValueError, match="wave_id must be a non-empty string"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="  ",
            required_child_task_ids=("c-1",),
            created_at=ts,
            updated_at=ts,
        )

    # Unsupported version and bool version
    with pytest.raises(ValueError, match="Unsupported contract version"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1",),
            created_at=ts,
            updated_at=ts,
            version=1,
        )
    with pytest.raises(TypeError, match="version must be an int"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1",),
            created_at=ts,
            updated_at=ts,
            version=True,  # type: ignore[arg-type]
        )


def test_task_continuation_v2_resolutions_and_states() -> None:
    ts = "2026-11-26T10:00:00Z"
    for resolution in ("SATISFIED", "UNSATISFIABLE", "TIMED_OUT"):
        released = TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1",),
            created_at=ts,
            updated_at=ts,
            state=CONTINUATION_V2_STATE_RELEASED,
            wave_resolution=resolution,
        )
        assert released.is_claimable is True
        assert released.is_closed is False
        assert TaskContinuationV2.from_json(released.to_json()) == released

    closed = TaskContinuationV2(
        parent_task_id="p-1",
        wave_id="w-1",
        required_child_task_ids=("c-1",),
        created_at=ts,
        updated_at=ts,
        state=CONTINUATION_V2_STATE_CLOSED,
        wave_resolution=None,
    )
    assert closed.is_claimable is False
    assert closed.is_closed is True

    # Invalid combinations
    with pytest.raises(ValueError, match="must not expose wave_resolution before release"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1",),
            created_at=ts,
            updated_at=ts,
            state=CONTINUATION_V2_STATE_ACTIVE,
            wave_resolution="SATISFIED",
        )

    with pytest.raises(ValueError, match="Released continuation must expose"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1",),
            created_at=ts,
            updated_at=ts,
            state=CONTINUATION_V2_STATE_RELEASED,
            wave_resolution=None,
        )

    with pytest.raises(ValueError, match="Closed continuation must not expose"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1",),
            created_at=ts,
            updated_at=ts,
            state=CONTINUATION_V2_STATE_CLOSED,
            wave_resolution="SATISFIED",
        )

    with pytest.raises(ValueError, match="CANCELLED is not permitted"):
        TaskContinuationV2(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1",),
            created_at=ts,
            updated_at=ts,
            state=CONTINUATION_V2_STATE_RELEASED,
            wave_resolution="CANCELLED",
        )


def test_task_continuation_v2_unknown_fields_and_16kib_bound() -> None:
    valid = TaskContinuationV2.create(
        parent_task_id="p-1",
        wave_id="w-1",
        required_child_task_ids=("c-1", "c-2"),
    )
    with pytest.raises(TypeError, match="unexpected TaskContinuationV2 fields"):
        TaskContinuationV2.from_dict({**valid.to_dict(), "completed_count": 1})

    with pytest.raises(ValueError, match="exceeds maximum bound"):
        TaskContinuationV2.create(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1",),
            control_fields={"material_summary": "x" * (MAX_CONTINUATION_BYTES + 1)},
        )


# ==============================================================================
# 2. Delegation-Wave Contracts
# ==============================================================================


def test_required_success_fraction_normalization_and_bounds() -> None:
    f_reduced = RequiredSuccessFraction(2, 4)
    assert f_reduced == RequiredSuccessFraction(1, 2)
    assert f_reduced.numerator == 1
    assert f_reduced.denominator == 2

    # Boundary fractions 1/N and 1
    f_one_of_three = RequiredSuccessFraction(1, 3)
    assert f_one_of_three.required_success_count(3) == 1
    assert f_one_of_three.required_success_count(4) == 2

    f_full = RequiredSuccessFraction(1, 1)
    assert f_full.required_success_count(1) == 1
    assert f_full.required_success_count(5) == 5

    # Invalid cases
    with pytest.raises(ValueError, match="numerator must be >= 1"):
        RequiredSuccessFraction(0, 2)
    with pytest.raises(ValueError, match="denominator must be >= 1"):
        RequiredSuccessFraction(1, 0)
    with pytest.raises(ValueError, match="must be <= 1"):
        RequiredSuccessFraction(3, 2)
    with pytest.raises(TypeError, match="numerator must be an int"):
        RequiredSuccessFraction(True, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="denominator must be an int"):
        RequiredSuccessFraction(1, False)  # type: ignore[arg-type]


def test_accepted_and_resolved_wave_roundtrip_and_invariants() -> None:
    accepted_at = "2026-11-26T12:00:00Z"
    derivation = derive_wave_deadline(
        work_complexity="medium",
        uncertainty_level="low",
        required_child_count=3,
        planned_parallelism=2,
        min_duration_seconds=30,
        max_duration_seconds=600,
        accepted_at=accepted_at,
    )

    wait_all_wave = AcceptedDelegationWave(
        wave_id="w-1",
        parent_task_id="p-1",
        required_child_task_ids=("c-3", "c-1", "c-2"),
        barrier_policy=BarrierPolicy.WAIT_ALL.value,
        deadline_derivation=derivation,
        accepted_at=accepted_at,
    )
    assert wait_all_wave.required_child_task_ids == ("c-1", "c-2", "c-3")
    assert wait_all_wave.state == "unresolved"
    assert wait_all_wave.required_success_fraction is None
    assert wait_all_wave.required_success_count is None
    assert AcceptedDelegationWave.from_json(wait_all_wave.to_json()) == wait_all_wave
    assert wait_all_wave.semantic_fingerprint() == AcceptedDelegationWave.from_json(
        wait_all_wave.to_json()
    ).semantic_fingerprint()

    partial_wave = AcceptedDelegationWave(
        wave_id="w-2",
        parent_task_id="p-1",
        required_child_task_ids=("c-1", "c-2", "c-3"),
        barrier_policy=BarrierPolicy.PARTIAL.value,
        required_success_fraction=RequiredSuccessFraction(2, 3),
        deadline_derivation=derivation,
        accepted_at=accepted_at,
    )
    assert partial_wave.required_success_count == 2
    assert AcceptedDelegationWave.from_json(partial_wave.to_json()) == partial_wave

    # Accepted wave must be strictly unresolved
    for resolved_state in ("SATISFIED", "UNSATISFIABLE", "TIMED_OUT", "CANCELLED"):
        with pytest.raises(ValueError, match="strictly 'unresolved'"):
            AcceptedDelegationWave(
                wave_id="w-1",
                parent_task_id="p-1",
                required_child_task_ids=("c-1", "c-2", "c-3"),
                barrier_policy="WAIT_ALL",
                deadline_derivation=derivation,
                accepted_at=accepted_at,
                state=resolved_state,
            )

    # Caller-supplied mismatched required_success_count rejected
    with pytest.raises(ValueError, match="does not match derived threshold"):
        AcceptedDelegationWave(
            wave_id="w-2",
            parent_task_id="p-1",
            required_child_task_ids=("c-1", "c-2", "c-3"),
            barrier_policy="PARTIAL",
            required_success_fraction=RequiredSuccessFraction(2, 3),
            required_success_count=1,
            deadline_derivation=derivation,
            accepted_at=accepted_at,
        )

    # WAIT_ALL must not carry partial fraction or count
    with pytest.raises(ValueError, match="WAIT_ALL barrier must not carry"):
        AcceptedDelegationWave(
            wave_id="w-1",
            parent_task_id="p-1",
            required_child_task_ids=("c-1", "c-2", "c-3"),
            barrier_policy="WAIT_ALL",
            required_success_fraction=RequiredSuccessFraction(1, 2),
            deadline_derivation=derivation,
            accepted_at=accepted_at,
        )

    # Unknown field & unsupported version rejected
    with pytest.raises(TypeError, match="unexpected AcceptedDelegationWave fields"):
        AcceptedDelegationWave.from_dict({**wait_all_wave.to_dict(), "extra": 1})
    with pytest.raises(ValueError, match="Unsupported contract version"):
        AcceptedDelegationWave.from_dict({**wait_all_wave.to_dict(), "version": 2})

    # ResolvedDelegationWave round-trip
    resolved = ResolvedDelegationWave(
        wave_id="w-2",
        parent_task_id="p-1",
        required_child_task_ids=("c-1", "c-2", "c-3"),
        barrier_policy="PARTIAL",
        required_success_fraction=RequiredSuccessFraction(2, 3),
        resolution=WaveResolution.SATISFIED.value,
        resolved_at="2026-11-26T12:01:00Z",
        wave_deadline=derivation.wave_deadline,
        residual_disposition=RESIDUAL_DISPOSITION_ENDED_BARRIER_RESOLVED,
    )
    assert ResolvedDelegationWave.from_json(resolved.to_json()) == resolved


# ==============================================================================
# 3. Pure Canonical Barrier Evaluation
# ==============================================================================


def test_barrier_evaluation_wait_all() -> None:
    children = ("c-1", "c-2", "c-3")

    # Unresolved while recovery_required or pending exists
    res_unresolved = evaluate_barrier(
        required_child_task_ids=children,
        child_states={"c-1": "completed", "c-2": "failed", "c-3": "recovery_required"},
        barrier_policy="WAIT_ALL",
    )
    assert res_unresolved.outcome == BarrierEvaluationOutcome.UNRESOLVED
    assert res_unresolved.is_resolved is False

    # ended / barrier_resolved does not newly satisfy WAIT_ALL
    res_residual = evaluate_barrier(
        required_child_task_ids=children,
        child_states={
            "c-1": "completed",
            "c-2": "completed",
            "c-3": RESIDUAL_DISPOSITION_ENDED_BARRIER_RESOLVED,
        },
        barrier_policy="WAIT_ALL",
    )
    assert res_residual.outcome == BarrierEvaluationOutcome.UNRESOLVED

    # Satisfied when all terminal (completed, failed, claim_timed_out)
    res_satisfied = evaluate_barrier(
        required_child_task_ids=children,
        child_states={"c-1": "completed", "c-2": "failed", "c-3": "claim_timed_out"},
        barrier_policy="WAIT_ALL",
    )
    assert res_satisfied.outcome == BarrierEvaluationOutcome.SATISFIED
    assert res_satisfied.is_resolved is True


def test_barrier_evaluation_partial() -> None:
    children = ("c-1", "c-2", "c-3")
    fraction = RequiredSuccessFraction(2, 3)  # requires 2 of 3

    # Satisfied when completed >= 2
    res_sat = evaluate_barrier(
        required_child_task_ids=children,
        child_states={"c-1": "completed", "c-2": "completed", "c-3": "pending"},
        barrier_policy="PARTIAL",
        required_success_fraction=fraction,
    )
    assert res_sat.outcome == BarrierEvaluationOutcome.SATISFIED
    assert res_sat.required_success_count == 2

    # Unresolved when recovery_required keeps threshold reachable
    res_unres = evaluate_barrier(
        required_child_task_ids=children,
        child_states={"c-1": "completed", "c-2": "failed", "c-3": "recovery_required"},
        barrier_policy="PARTIAL",
        required_success_fraction=fraction,
    )
    assert res_unres.outcome == BarrierEvaluationOutcome.UNRESOLVED

    # Unsatisfiable when completed + potentially_successful < 2
    res_unsat = evaluate_barrier(
        required_child_task_ids=children,
        child_states={"c-1": "completed", "c-2": "failed", "c-3": "claim_timed_out"},
        barrier_policy="PARTIAL",
        required_success_fraction=fraction,
    )
    assert res_unsat.outcome == BarrierEvaluationOutcome.UNSATISFIABLE


def test_barrier_evaluation_fail_closed_cases() -> None:
    # Missing required child
    with pytest.raises(ValueError, match="Missing required child states"):
        evaluate_barrier(
            required_child_task_ids=("c-1", "c-2"),
            child_states={"c-1": "completed"},
            barrier_policy="WAIT_ALL",
        )

    # Extra child record
    with pytest.raises(ValueError, match="Unexpected child records"):
        evaluate_barrier(
            required_child_task_ids=("c-1",),
            child_states={"c-1": "completed", "c-extra": "completed"},
            barrier_policy="WAIT_ALL",
        )

    # Duplicate state entry in sequence form
    with pytest.raises(ValueError, match="Duplicate child state entry"):
        evaluate_barrier(
            required_child_task_ids=("c-1",),
            child_states=[("c-1", "completed"), ("c-1", "failed")],
            barrier_policy="WAIT_ALL",
        )

    # Unknown child state
    with pytest.raises(ValueError, match="Unknown child state/outcome"):
        evaluate_barrier(
            required_child_task_ids=("c-1",),
            child_states={"c-1": "invented_state"},
            barrier_policy="WAIT_ALL",
        )

    # Policy / threshold mismatch
    with pytest.raises(ValueError, match="PARTIAL barrier requires"):
        evaluate_barrier(
            required_child_task_ids=("c-1",),
            child_states={"c-1": "completed"},
            barrier_policy="PARTIAL",
        )


# ==============================================================================
# 4. Deterministic Deadline Derivation
# ==============================================================================


def test_deadline_derivation_determinism_floor_range_and_rejection() -> None:
    accepted = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)

    # Below-minimum floor applied
    d_floor = derive_wave_deadline(
        work_complexity="low",  # 30s
        uncertainty_level="low",  # +0% -> raw 30s
        required_child_count=1,
        planned_parallelism=1,
        min_duration_seconds=60,
        max_duration_seconds=300,
        accepted_at=accepted,
    )
    assert d_floor.derived_duration_seconds == 60
    assert d_floor.wave_deadline == "2026-11-26T12:01:00.000000Z"

    # In-range derivation (deterministic repeated calls)
    d_range_1 = derive_wave_deadline(
        work_complexity="medium",  # 60s
        uncertainty_level="medium",  # +25% -> 2 batches * 60 = 120; 120 + 30 = 150s
        required_child_count=3,
        planned_parallelism=2,
        min_duration_seconds=60,
        max_duration_seconds=300,
        accepted_at="2026-11-26T12:00:00+00:00",
    )
    d_range_2 = derive_wave_deadline(
        work_complexity="medium",
        uncertainty_level="medium",
        required_child_count=3,
        planned_parallelism=2,
        min_duration_seconds=60,
        max_duration_seconds=300,
        accepted_at="2026-11-26T12:00:00Z",
    )
    assert d_range_1 == d_range_2
    assert d_range_1.derived_duration_seconds == 150
    assert d_range_1.wave_deadline == "2026-11-26T12:02:30.000000Z"
    assert WaveDeadlineDerivation.from_json(d_range_1.to_json()) == d_range_1
    assert d_range_1.semantic_fingerprint() == d_range_2.semantic_fingerprint()

    # Above-maximum rejected without clamping
    with pytest.raises(ValueError, match="rejected without clamping"):
        derive_wave_deadline(
            work_complexity="high",  # 120s
            uncertainty_level="high",  # +50% -> 3*120 = 360 -> 540s > 300s
            required_child_count=3,
            planned_parallelism=1,
            min_duration_seconds=60,
            max_duration_seconds=300,
            accepted_at=accepted,
        )


def test_deadline_derivation_invalid_inputs_rejected() -> None:
    # Naive timestamp
    with pytest.raises(ValueError, match="Naive timestamps"):
        derive_wave_deadline(
            work_complexity="low",
            uncertainty_level="low",
            required_child_count=1,
            planned_parallelism=1,
            min_duration_seconds=10,
            max_duration_seconds=100,
            accepted_at="2026-11-26T12:00:00",
        )

    # Bool as int
    with pytest.raises(TypeError, match="required_child_count must be an int"):
        derive_wave_deadline(
            work_complexity="low",
            uncertainty_level="low",
            required_child_count=True,  # type: ignore[arg-type]
            planned_parallelism=1,
            min_duration_seconds=10,
            max_duration_seconds=100,
            accepted_at="2026-11-26T12:00:00Z",
        )

    # Invalid bounds (max < min)
    with pytest.raises(ValueError, match="max_duration_seconds"):
        derive_wave_deadline(
            work_complexity="low",
            uncertainty_level="low",
            required_child_count=1,
            planned_parallelism=1,
            min_duration_seconds=100,
            max_duration_seconds=50,
            accepted_at="2026-11-26T12:00:00Z",
        )

    # Unsupported version and unknown fields
    valid = derive_wave_deadline(
        work_complexity="low",
        uncertainty_level="low",
        required_child_count=1,
        planned_parallelism=1,
        min_duration_seconds=10,
        max_duration_seconds=100,
        accepted_at="2026-11-26T12:00:00Z",
    )
    with pytest.raises(ValueError, match="Unsupported contract version"):
        WaveDeadlineDerivation.from_dict({**valid.to_dict(), "version": 2})
    with pytest.raises(TypeError, match="unexpected WaveDeadlineDerivation fields"):
        WaveDeadlineDerivation.from_dict({**valid.to_dict(), "unknown_field": 1})


# ==============================================================================
# 5. Security & Serialization
# ==============================================================================


def test_credential_bearing_nested_keys_rejected_in_v2_continuation() -> None:
    with pytest.raises(ValueError, match="Unallowlisted control fields|Credential-bearing key"):
        TaskContinuationV2.create(
            parent_task_id="p-1",
            wave_id="w-1",
            required_child_task_ids=("c-1",),
            control_fields={"api_key": "secret"},
        )
