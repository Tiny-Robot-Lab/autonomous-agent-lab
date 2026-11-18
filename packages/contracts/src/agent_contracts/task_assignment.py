import json
from dataclasses import dataclass
from typing import Any

from .artifact_reference import ArtifactReference
from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    normalize_canonical_utc_timestamp,
)
from .task_input import TaskInputReference

TASK_ASSIGNMENT_CONTRACT_VERSION = 1
ALLOWED_EXECUTION_KINDS = frozenset({"initial", "delegated_child", "resumed_continuation"})


@dataclass(frozen=True)
class TaskAssignment:
    """Private, versioned task assignment record returned over the authenticated claim response.

    Governed by ADR-0007 (Decision 7) and Issue #220:
    - Binds assignment, principal, task, continuation_version, fencing_epoch, capability, and grant;
    - Immutable and private; returned only over authenticated runtime responses;
    - Never carries detached assignment proof;
    - Never carries mutable acknowledgement state.
    """

    assignment_id: str
    claim_id: str
    task_id: str
    root_task_id: str
    turn_id: str
    runtime_principal_id: str
    bound_agent_id: str
    execution_kind: str
    assigned_capability: str
    fencing_epoch: int
    claimed_at: str
    delivery_expires_at: str
    task_input_reference: TaskInputReference
    continuation_version: int | None = None
    pinned_artifact_references: tuple[ArtifactReference, ...] = ()
    delegation_grant_id: str | None = None
    version: int = TASK_ASSIGNMENT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != TASK_ASSIGNMENT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "assignment_id",
            "claim_id",
            "task_id",
            "root_task_id",
            "turn_id",
            "runtime_principal_id",
            "bound_agent_id",
            "execution_kind",
            "assigned_capability",
            "claimed_at",
            "delivery_expires_at",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        object.__setattr__(
            self, "claimed_at", normalize_canonical_utc_timestamp(self.claimed_at)
        )
        object.__setattr__(
            self,
            "delivery_expires_at",
            normalize_canonical_utc_timestamp(self.delivery_expires_at),
        )

        if self.execution_kind not in ALLOWED_EXECUTION_KINDS:
            raise ValueError(
                f"Invalid execution_kind: '{self.execution_kind}'. Allowed: {sorted(ALLOWED_EXECUTION_KINDS)}"
            )

        if type(self.fencing_epoch) is bool or not isinstance(self.fencing_epoch, int):
            raise TypeError(f"fencing_epoch must be an int, got {type(self.fencing_epoch).__name__}")
        if self.fencing_epoch < 1:
            raise ValueError(f"fencing_epoch must be >= 1, got {self.fencing_epoch}")

        if not isinstance(self.task_input_reference, TaskInputReference):
            if isinstance(self.task_input_reference, dict):
                object.__setattr__(
                    self,
                    "task_input_reference",
                    TaskInputReference.from_dict(self.task_input_reference),
                )
            else:
                raise TypeError(
                    f"task_input_reference must be a TaskInputReference, got {type(self.task_input_reference).__name__}"
                )

        if self.continuation_version is not None:
            if type(self.continuation_version) is bool or not isinstance(self.continuation_version, int):
                raise TypeError(
                    f"continuation_version must be an int or None, got {type(self.continuation_version).__name__}"
                )
            if self.continuation_version < 1:
                raise ValueError(
                    f"continuation_version must be >= 1 when set, got {self.continuation_version}"
                )

        if isinstance(self.pinned_artifact_references, (list, tuple)):
            clean_refs: list[ArtifactReference] = []
            for ref in self.pinned_artifact_references:
                if isinstance(ref, ArtifactReference):
                    clean_refs.append(ref)
                elif isinstance(ref, dict):
                    clean_refs.append(ArtifactReference.from_dict(ref))
                else:
                    raise TypeError(
                        f"pinned_artifact_references element must be ArtifactReference or dict, got {type(ref).__name__}"
                    )
            object.__setattr__(self, "pinned_artifact_references", tuple(clean_refs))
        else:
            raise TypeError(
                f"pinned_artifact_references must be a tuple or list, got {type(self.pinned_artifact_references).__name__}"
            )

        if self.delegation_grant_id is not None:
            if not isinstance(self.delegation_grant_id, str) or not self.delegation_grant_id.strip():
                raise ValueError("delegation_grant_id must be a non-empty string when present")
            object.__setattr__(self, "delegation_grant_id", self.delegation_grant_id.strip())

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "assignment_id": self.assignment_id,
            "assigned_capability": self.assigned_capability,
            "bound_agent_id": self.bound_agent_id,
            "claim_id": self.claim_id,
            "claimed_at": self.claimed_at,
            "continuation_version": self.continuation_version,
            "delegation_grant_id": self.delegation_grant_id,
            "delivery_expires_at": self.delivery_expires_at,
            "execution_kind": self.execution_kind,
            "fencing_epoch": self.fencing_epoch,
            "pinned_artifact_references": [ref.to_dict() for ref in self.pinned_artifact_references],
            "root_task_id": self.root_task_id,
            "runtime_principal_id": self.runtime_principal_id,
            "task_id": self.task_id,
            "task_input_reference": self.task_input_reference.to_dict(),
            "turn_id": self.turn_id,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="TaskAssignment.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskAssignment":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "assignment_id",
            "assigned_capability",
            "bound_agent_id",
            "claim_id",
            "claimed_at",
            "continuation_version",
            "delegation_grant_id",
            "delivery_expires_at",
            "execution_kind",
            "fencing_epoch",
            "pinned_artifact_references",
            "root_task_id",
            "runtime_principal_id",
            "task_id",
            "task_input_reference",
            "turn_id",
            "version",
        }
        unexpected = data.keys() - allowed_fields
        if unexpected:
            raise TypeError(f"unexpected TaskAssignment fields: {sorted(unexpected)}")

        required = (
            "assignment_id",
            "claim_id",
            "task_id",
            "root_task_id",
            "turn_id",
            "runtime_principal_id",
            "bound_agent_id",
            "execution_kind",
            "assigned_capability",
            "fencing_epoch",
            "claimed_at",
            "delivery_expires_at",
            "task_input_reference",
        )
        for req in required:
            if req not in data:
                raise ValueError(f"{req} is required")

        version = data.get("version", TASK_ASSIGNMENT_CONTRACT_VERSION)
        if type(version) is bool or not isinstance(version, int):
            raise TypeError(f"version must be an int, got {type(version).__name__}")
        if version != TASK_ASSIGNMENT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {version}")

        raw_input_ref = data["task_input_reference"]
        input_ref = (
            raw_input_ref
            if isinstance(raw_input_ref, TaskInputReference)
            else TaskInputReference.from_dict(raw_input_ref)
        )

        raw_artifacts = data.get("pinned_artifact_references", ())
        if not isinstance(raw_artifacts, (list, tuple)):
            raise TypeError("pinned_artifact_references must be a list or tuple")

        return cls(
            assignment_id=data["assignment_id"],
            claim_id=data["claim_id"],
            task_id=data["task_id"],
            root_task_id=data["root_task_id"],
            turn_id=data["turn_id"],
            runtime_principal_id=data["runtime_principal_id"],
            bound_agent_id=data["bound_agent_id"],
            execution_kind=data["execution_kind"],
            assigned_capability=data["assigned_capability"],
            fencing_epoch=data["fencing_epoch"],
            claimed_at=data["claimed_at"],
            delivery_expires_at=data["delivery_expires_at"],
            task_input_reference=input_ref,
            continuation_version=data.get("continuation_version"),
            pinned_artifact_references=tuple(raw_artifacts),
            delegation_grant_id=data.get("delegation_grant_id"),
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "TaskAssignment":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ClaimTaskResponse:
    """Protected response envelope for the authenticated claim operation.

    Governed by ADR-0007 (Decision 7) and Issue #220:
    - Returns private TaskAssignment and detached secret assignment proof;
    - Proof is deterministically re-derived and returned on valid unacknowledged idempotent replays while delivery lease remains valid;
    - Under no circumstances is assignment_proof persisted in public logs or task contracts.
    """

    assignment: TaskAssignment
    assignment_proof: str | None = None
