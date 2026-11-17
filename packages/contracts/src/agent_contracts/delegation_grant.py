import json
from dataclasses import dataclass
from typing import Any

from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    normalize_canonical_utc_timestamp,
)

DELEGATION_GRANT_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class DelegationGrant:
    """Narrow one-turn delegation grant issued to an eligible initial Research root assignment.

    Governed by ADR-0006 (Items 8, 19), ADR-0007 (Decision 8), and Issue #220:
    - Fixed to research -> analysis, depth 1, 1 wave, 1 child;
    - Bound to root_task_id, issued_to_assignment_id, issued_to_principal_id, fencing_epoch, and turn_id;
    - Immutable; contains no credentials or secret proofs;
    - Stored immutably at task.authority_grant:{root_task_id}:{grant_id};
    - Grant consumption state is tracked separately in Issue #221.
    """

    grant_id: str
    root_task_id: str
    issued_to_assignment_id: str
    issued_to_principal_id: str
    fencing_epoch: int
    turn_id: str
    issued_at: str
    source_capability: str = "research"
    target_capability: str = "analysis"
    max_children: int = 1
    max_depth: int = 1
    max_waves: int = 1
    version: int = DELEGATION_GRANT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != DELEGATION_GRANT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        if type(self.fencing_epoch) is bool or not isinstance(self.fencing_epoch, int):
            raise TypeError(f"fencing_epoch must be an int, got {type(self.fencing_epoch).__name__}")
        if self.fencing_epoch < 1:
            raise ValueError(f"fencing_epoch must be >= 1, got {self.fencing_epoch}")

        for field_name in (
            "grant_id",
            "root_task_id",
            "issued_to_assignment_id",
            "issued_to_principal_id",
            "turn_id",
            "issued_at",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, val.strip())

        object.__setattr__(
            self, "issued_at", normalize_canonical_utc_timestamp(self.issued_at)
        )

        if self.source_capability != "research":
            raise ValueError(
                f"DelegationGrant source_capability must be 'research', got '{self.source_capability}'"
            )
        if self.target_capability != "analysis":
            raise ValueError(
                f"DelegationGrant target_capability must be 'analysis', got '{self.target_capability}'"
            )
        if self.max_children != 1:
            raise ValueError(f"DelegationGrant max_children must be 1, got {self.max_children}")
        if self.max_depth != 1:
            raise ValueError(f"DelegationGrant max_depth must be 1, got {self.max_depth}")
        if self.max_waves != 1:
            raise ValueError(f"DelegationGrant max_waves must be 1, got {self.max_waves}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "fencing_epoch": self.fencing_epoch,
            "grant_id": self.grant_id,
            "issued_at": self.issued_at,
            "issued_to_assignment_id": self.issued_to_assignment_id,
            "issued_to_principal_id": self.issued_to_principal_id,
            "max_children": self.max_children,
            "max_depth": self.max_depth,
            "max_waves": self.max_waves,
            "root_task_id": self.root_task_id,
            "source_capability": self.source_capability,
            "target_capability": self.target_capability,
            "turn_id": self.turn_id,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="DelegationGrant.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegationGrant":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "fencing_epoch",
            "grant_id",
            "issued_at",
            "issued_to_assignment_id",
            "issued_to_principal_id",
            "max_children",
            "max_depth",
            "max_waves",
            "root_task_id",
            "source_capability",
            "target_capability",
            "turn_id",
            "version",
        }
        unexpected = data.keys() - allowed_fields
        if unexpected:
            raise TypeError(f"unexpected DelegationGrant fields: {sorted(unexpected)}")

        required = (
            "grant_id",
            "root_task_id",
            "issued_to_assignment_id",
            "issued_to_principal_id",
            "fencing_epoch",
            "turn_id",
            "issued_at",
        )
        for req in required:
            if req not in data:
                raise ValueError(f"{req} is required")

        fencing_epoch = data["fencing_epoch"]
        if type(fencing_epoch) is bool or not isinstance(fencing_epoch, int):
            raise TypeError(f"fencing_epoch must be an int, got {type(fencing_epoch).__name__}")

        version = data.get("version", DELEGATION_GRANT_CONTRACT_VERSION)
        if type(version) is bool or not isinstance(version, int):
            raise TypeError(f"version must be an int, got {type(version).__name__}")
        if version != DELEGATION_GRANT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {version}")

        return cls(
            grant_id=data["grant_id"],
            root_task_id=data["root_task_id"],
            issued_to_assignment_id=data["issued_to_assignment_id"],
            issued_to_principal_id=data["issued_to_principal_id"],
            fencing_epoch=fencing_epoch,
            turn_id=data["turn_id"],
            issued_at=data["issued_at"],
            source_capability=data.get("source_capability", "research"),
            target_capability=data.get("target_capability", "analysis"),
            max_children=data.get("max_children", 1),
            max_depth=data.get("max_depth", 1),
            max_waves=data.get("max_waves", 1),
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "DelegationGrant":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))
