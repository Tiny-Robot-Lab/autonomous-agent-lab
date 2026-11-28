from enum import Enum


class TaskLifecycleStatus(str, Enum):
    PENDING_DISPATCH = "pending_dispatch"
    PENDING = "pending"
    WORKING = "working"
    WAITING_FOR_CHILDREN = "waiting_for_children"
    RESUMABLE = "resumable"
    RECOVERY_REQUIRED = "recovery_required"
    COMPLETED = "completed"
    FAILED = "failed"
    DISPATCH_FAILED = "dispatch_failed"
    CLAIM_TIMED_OUT = "claim_timed_out"
    CANCELLED = "cancelled"


PARENT_LIFECYCLE_STATUSES: frozenset[str] = frozenset({
    TaskLifecycleStatus.PENDING_DISPATCH.value,
    TaskLifecycleStatus.PENDING.value,
    TaskLifecycleStatus.WORKING.value,
    TaskLifecycleStatus.WAITING_FOR_CHILDREN.value,
    TaskLifecycleStatus.RESUMABLE.value,
    TaskLifecycleStatus.RECOVERY_REQUIRED.value,
    TaskLifecycleStatus.COMPLETED.value,
    TaskLifecycleStatus.FAILED.value,
    TaskLifecycleStatus.DISPATCH_FAILED.value,
    TaskLifecycleStatus.CANCELLED.value,
})

CHILD_LIFECYCLE_STATUSES: frozenset[str] = frozenset({
    TaskLifecycleStatus.PENDING_DISPATCH.value,
    TaskLifecycleStatus.PENDING.value,
    TaskLifecycleStatus.WORKING.value,
    TaskLifecycleStatus.RECOVERY_REQUIRED.value,
    TaskLifecycleStatus.COMPLETED.value,
    TaskLifecycleStatus.FAILED.value,
    TaskLifecycleStatus.CLAIM_TIMED_OUT.value,
    TaskLifecycleStatus.DISPATCH_FAILED.value,
})

ALL_TASK_LIFECYCLE_STATUSES: frozenset[str] = frozenset(
    s.value for s in TaskLifecycleStatus
)

TERMINAL_TASK_LIFECYCLE_STATUSES: frozenset[str] = frozenset({
    TaskLifecycleStatus.COMPLETED.value,
    TaskLifecycleStatus.FAILED.value,
    TaskLifecycleStatus.CLAIM_TIMED_OUT.value,
    TaskLifecycleStatus.DISPATCH_FAILED.value,
    TaskLifecycleStatus.CANCELLED.value,
})

NONTERMINAL_TASK_LIFECYCLE_STATUSES: frozenset[str] = frozenset(
    ALL_TASK_LIFECYCLE_STATUSES - TERMINAL_TASK_LIFECYCLE_STATUSES
)


def validate_lifecycle_status(status: str) -> str:
    if not isinstance(status, str):
        raise TypeError(f"status must be a str, got {type(status).__name__}")
    clean = status.strip()
    if clean not in ALL_TASK_LIFECYCLE_STATUSES:
        raise ValueError(
            f"Invalid task lifecycle status: '{clean}'. Allowed statuses: {sorted(ALL_TASK_LIFECYCLE_STATUSES)}"
        )
    return clean
