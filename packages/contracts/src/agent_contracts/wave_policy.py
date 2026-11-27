from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .delegation_wave import (
    RESIDUAL_DISPOSITION_ENDED_BARRIER_RESOLVED,
    BarrierPolicy,
    RequiredSuccessFraction,
    WaveDeadlineDerivation,
    _validate_barrier_and_threshold,
    compute_expected_wave_duration_seconds,
)
from .lifecycle import TaskLifecycleStatus
from .serialization import normalize_canonical_utc_timestamp

CANONICAL_TERMINAL_BARRIER_CHILD_STATES: frozenset[str] = frozenset({
    TaskLifecycleStatus.COMPLETED.value,
    TaskLifecycleStatus.FAILED.value,
    TaskLifecycleStatus.CLAIM_TIMED_OUT.value,
    TaskLifecycleStatus.DISPATCH_FAILED.value,
})

UNRESOLVED_POTENTIALLY_SUCCESSFUL_CHILD_STATES: frozenset[str] = frozenset({
    TaskLifecycleStatus.PENDING_DISPATCH.value,
    TaskLifecycleStatus.PENDING.value,
    TaskLifecycleStatus.WORKING.value,
    TaskLifecycleStatus.RECOVERY_REQUIRED.value,
})

ALLOWED_BARRIER_CHILD_STATES: frozenset[str] = (
    CANONICAL_TERMINAL_BARRIER_CHILD_STATES
    | UNRESOLVED_POTENTIALLY_SUCCESSFUL_CHILD_STATES
    | frozenset({RESIDUAL_DISPOSITION_ENDED_BARRIER_RESOLVED})
)


class BarrierEvaluationOutcome(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    SATISFIED = "SATISFIED"
    UNSATISFIABLE = "UNSATISFIABLE"


@dataclass(frozen=True)
class BarrierEvaluationResult:
    """Explicit result of pure canonical barrier evaluation."""

    outcome: BarrierEvaluationOutcome
    barrier_policy: str
    required_child_count: int
    completed_count: int
    terminal_count: int
    unresolved_count: int
    potentially_successful_count: int
    required_success_count: int | None = None

    @property
    def is_resolved(self) -> bool:
        return self.outcome != BarrierEvaluationOutcome.UNRESOLVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "barrier_policy": self.barrier_policy,
            "completed_count": self.completed_count,
            "outcome": self.outcome.value,
            "potentially_successful_count": self.potentially_successful_count,
            "required_child_count": self.required_child_count,
            "required_success_count": self.required_success_count,
            "terminal_count": self.terminal_count,
            "unresolved_count": self.unresolved_count,
        }


def _normalize_membership_for_evaluation(required_child_task_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(required_child_task_ids, (str, bytes)) or not isinstance(
        required_child_task_ids, Sequence
    ):
        raise TypeError(
            "required_child_task_ids must be a sequence of strings, "
            f"got {type(required_child_task_ids).__name__}"
        )
    if len(required_child_task_ids) == 0:
        raise ValueError("required_child_task_ids must not be empty")

    seen: set[str] = set()
    cleaned: list[str] = []
    for idx, child_id in enumerate(required_child_task_ids):
        if not isinstance(child_id, str) or not child_id.strip():
            raise ValueError(f"required_child_task_ids[{idx}] must be a non-empty string")
        clean_id = child_id.strip()
        if clean_id in seen:
            raise ValueError(f"Duplicate child task ID in required_child_task_ids: '{clean_id}'")
        seen.add(clean_id)
        cleaned.append(clean_id)
    return tuple(cleaned)


def _normalize_child_states(
    child_states: Mapping[str, str] | Sequence[tuple[str, str]],
) -> dict[str, str]:
    items: list[tuple[Any, Any]]
    if isinstance(child_states, Mapping):
        items = list(child_states.items())
    elif isinstance(child_states, Sequence) and not isinstance(child_states, (str, bytes)):
        items = []
        for entry in child_states:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("Sequence child_states entries must be (child_task_id, status) tuples")
            items.append(entry)
    else:
        raise TypeError(
            "child_states must be a Mapping or sequence of (child_task_id, status) pairs, "
            f"got {type(child_states).__name__}"
        )

    normalized: dict[str, str] = {}
    for raw_id, raw_status in items:
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("child_states keys must be non-empty strings")
        clean_id = raw_id.strip()
        if clean_id in normalized:
            raise ValueError(f"Duplicate child state entry for '{clean_id}'")
        if not isinstance(raw_status, str) or not raw_status.strip():
            raise ValueError(f"Child status for '{clean_id}' must be a non-empty string")
        clean_status = raw_status.strip()
        if clean_status not in ALLOWED_BARRIER_CHILD_STATES:
            raise ValueError(
                f"Unknown child state/outcome '{clean_status}' for '{clean_id}'. "
                f"Allowed: {sorted(ALLOWED_BARRIER_CHILD_STATES)}"
            )
        normalized[clean_id] = clean_status
    return normalized


def evaluate_barrier(
    *,
    required_child_task_ids: Sequence[str],
    child_states: Mapping[str, str] | Sequence[tuple[str, str]],
    barrier_policy: str,
    required_success_fraction: RequiredSuccessFraction | dict[str, Any] | None = None,
    required_success_count: int | None = None,
) -> BarrierEvaluationResult:
    """Pure canonical barrier evaluation over child lifecycle/outcome categories only.

    Fails closed on empty/duplicate membership, missing required children,
    extra unknown children, unknown child state/outcome categories, or
    policy/threshold mismatch. Never inspects TaskResult content.
    """
    membership = _normalize_membership_for_evaluation(required_child_task_ids)
    clean_policy, _, derived_count = _validate_barrier_and_threshold(
        barrier_policy,
        len(membership),
        required_success_fraction,
        required_success_count,
    )
    normalized_states = _normalize_child_states(child_states)

    membership_set = set(membership)
    state_keys = set(normalized_states.keys())
    missing = membership_set - state_keys
    if missing:
        raise ValueError(f"Missing required child states: {sorted(missing)}")
    extra = state_keys - membership_set
    if extra:
        raise ValueError(f"Unexpected child records not in required membership: {sorted(extra)}")

    completed_count = 0
    terminal_count = 0
    unresolved_count = 0
    potentially_successful_count = 0

    for child_id in membership:
        status = normalized_states[child_id]
        if status == TaskLifecycleStatus.COMPLETED.value:
            completed_count += 1
            terminal_count += 1
        elif status in CANONICAL_TERMINAL_BARRIER_CHILD_STATES:
            terminal_count += 1
        elif status in UNRESOLVED_POTENTIALLY_SUCCESSFUL_CHILD_STATES:
            unresolved_count += 1
            potentially_successful_count += 1
        else:
            # RESIDUAL_DISPOSITION_ENDED_BARRIER_RESOLVED:
            # not a canonical terminal outcome and no longer potentially successful.
            unresolved_count += 1

    if clean_policy == BarrierPolicy.WAIT_ALL.value:
        outcome = (
            BarrierEvaluationOutcome.SATISFIED
            if terminal_count == len(membership)
            else BarrierEvaluationOutcome.UNRESOLVED
        )
    else:
        assert derived_count is not None
        if completed_count >= derived_count:
            outcome = BarrierEvaluationOutcome.SATISFIED
        elif completed_count + potentially_successful_count < derived_count:
            outcome = BarrierEvaluationOutcome.UNSATISFIABLE
        else:
            outcome = BarrierEvaluationOutcome.UNRESOLVED

    return BarrierEvaluationResult(
        outcome=outcome,
        barrier_policy=clean_policy,
        required_child_count=len(membership),
        completed_count=completed_count,
        terminal_count=terminal_count,
        unresolved_count=unresolved_count,
        potentially_successful_count=potentially_successful_count,
        required_success_count=derived_count,
    )


def derive_wave_deadline(
    *,
    work_complexity: str,
    uncertainty_level: str,
    required_child_count: int,
    planned_parallelism: int,
    min_duration_seconds: int,
    max_duration_seconds: int,
    accepted_at: str | datetime,
    policy_id: str = "deterministic_bounded_baseline_v1",
    policy_version: int = 1,
) -> WaveDeadlineDerivation:
    """Pure deterministic bounded wave-deadline derivation.

    Uses trusted supplied accepted_at without reading wall clock or using floats.
    """
    from datetime import timedelta

    clean_accepted = normalize_canonical_utc_timestamp(accepted_at)
    duration = compute_expected_wave_duration_seconds(
        work_complexity=work_complexity,
        uncertainty_level=uncertainty_level,
        required_child_count=required_child_count,
        planned_parallelism=planned_parallelism,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
    )
    deadline_utc = normalize_canonical_utc_timestamp(
        datetime.fromisoformat(clean_accepted) + timedelta(seconds=duration)
    )
    return WaveDeadlineDerivation(
        policy_id=policy_id,
        policy_version=policy_version,
        work_complexity=work_complexity,
        uncertainty_level=uncertainty_level,
        required_child_count=required_child_count,
        planned_parallelism=planned_parallelism,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
        accepted_at=clean_accepted,
        derived_duration_seconds=duration,
        wave_deadline=deadline_utc,
    )
