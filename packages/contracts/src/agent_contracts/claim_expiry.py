import json
from dataclasses import dataclass
from typing import Any

from .active_claim import ALLOWED_CLAIM_EXECUTION_KINDS, ALLOWED_PRIOR_CLAIMABLE_STATUSES
from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    normalize_canonical_utc_timestamp,
)

CLAIM_EXPIRY_CONTRACT_VERSION: int = 1


@dataclass(frozen=True)
class ClaimExpiryRecord:
    """Authoritative Redis-persisted record of an expired execution claim transition to recovery_required.

    Governed by ADR-0007 (Decision 12) and Issue #223:
    - Persisted at task.claim_expiry:{task_id};
    - Immutably binds expired claim, assignment, principal, agent, and fencing epoch;
    - Preserves prior_claimable_status for subsequent recovery resolution;
    - Contains no secrets, bearer tokens, or detached proofs.
    """

    task_id: str
    claim_id: str
    assignment_id: str
    principal_id: str
    bound_agent_id: str
    execution_kind: str
    expired_fencing_epoch: int
    acknowledged_at: str
    execution_expires_at: str
    expired_at: str
    prior_claimable_status: str
    continuation_version: int | None = None
    reason_code: str = "EXECUTION_CLAIM_EXPIRED"
    version: int = CLAIM_EXPIRY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != CLAIM_EXPIRY_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "task_id",
            "claim_id",
            "assignment_id",
            "principal_id",
            "bound_agent_id",
            "execution_kind",
            "acknowledged_at",
            "execution_expires_at",
            "expired_at",
            "prior_claimable_status",
            "reason_code",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if self.execution_kind not in ALLOWED_CLAIM_EXECUTION_KINDS:
            raise ValueError(
                f"Invalid execution_kind '{self.execution_kind}'. Allowed: {sorted(ALLOWED_CLAIM_EXECUTION_KINDS)}"
            )

        if self.prior_claimable_status not in ALLOWED_PRIOR_CLAIMABLE_STATUSES:
            raise ValueError(
                f"Invalid prior_claimable_status '{self.prior_claimable_status}'. Allowed: {sorted(ALLOWED_PRIOR_CLAIMABLE_STATUSES)}"
            )

        if self.execution_kind in ("initial", "delegated_child") and self.prior_claimable_status != "pending":
            raise ValueError(
                f"prior_claimable_status must be 'pending' for {self.execution_kind}, got '{self.prior_claimable_status}'"
            )
        if self.execution_kind == "resumed_continuation" and self.prior_claimable_status != "resumable":
            raise ValueError(
                f"prior_claimable_status must be 'resumable' for resumed_continuation, got '{self.prior_claimable_status}'"
            )

        if type(self.expired_fencing_epoch) is bool or not isinstance(self.expired_fencing_epoch, int):
            raise TypeError(f"expired_fencing_epoch must be an int, got {type(self.expired_fencing_epoch).__name__}")
        if self.expired_fencing_epoch < 1:
            raise ValueError(f"expired_fencing_epoch must be >= 1, got {self.expired_fencing_epoch}")

        if self.continuation_version is not None:
            if type(self.continuation_version) is bool or not isinstance(self.continuation_version, int):
                raise TypeError(
                    f"continuation_version must be an int or None, got {type(self.continuation_version).__name__}"
                )
            if self.continuation_version < 1:
                raise ValueError(f"continuation_version must be >= 1, got {self.continuation_version}")

        if self.execution_kind == "resumed_continuation" and self.continuation_version is None:
            raise ValueError("continuation_version is required for resumed_continuation")
        elif self.execution_kind != "resumed_continuation" and self.continuation_version is not None:
            raise ValueError(f"continuation_version must be None for {self.execution_kind}")

        object.__setattr__(
            self, "acknowledged_at", normalize_canonical_utc_timestamp(self.acknowledged_at)
        )
        object.__setattr__(
            self, "execution_expires_at", normalize_canonical_utc_timestamp(self.execution_expires_at)
        )
        object.__setattr__(
            self, "expired_at", normalize_canonical_utc_timestamp(self.expired_at)
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "acknowledged_at": self.acknowledged_at,
            "assignment_id": self.assignment_id,
            "bound_agent_id": self.bound_agent_id,
            "claim_id": self.claim_id,
            "continuation_version": self.continuation_version,
            "execution_expires_at": self.execution_expires_at,
            "execution_kind": self.execution_kind,
            "expired_at": self.expired_at,
            "expired_fencing_epoch": self.expired_fencing_epoch,
            "principal_id": self.principal_id,
            "prior_claimable_status": self.prior_claimable_status,
            "reason_code": self.reason_code,
            "task_id": self.task_id,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="ClaimExpiryRecord.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaimExpiryRecord":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "acknowledged_at",
            "assignment_id",
            "bound_agent_id",
            "claim_id",
            "continuation_version",
            "execution_expires_at",
            "execution_kind",
            "expired_at",
            "expired_fencing_epoch",
            "principal_id",
            "runtime_principal_id",
            "prior_claimable_status",
            "reason_code",
            "task_id",
            "version",
        }
        unexpected = data.keys() - allowed_fields
        if unexpected:
            raise TypeError(f"unexpected ClaimExpiryRecord fields: {sorted(unexpected)}")

        principal_id = data.get("principal_id") or data.get("runtime_principal_id")
        if not principal_id:
            raise ValueError("principal_id is required")

        required = (
            "task_id",
            "claim_id",
            "assignment_id",
            "bound_agent_id",
            "execution_kind",
            "expired_fencing_epoch",
            "acknowledged_at",
            "execution_expires_at",
            "expired_at",
            "prior_claimable_status",
        )
        for req in required:
            if req not in data:
                raise ValueError(f"{req} is required")

        version = data.get("version", CLAIM_EXPIRY_CONTRACT_VERSION)
        if type(version) is bool or not isinstance(version, int):
            raise TypeError(f"version must be an int, got {type(version).__name__}")
        if version != CLAIM_EXPIRY_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {version}")

        return cls(
            task_id=data["task_id"],
            claim_id=data["claim_id"],
            assignment_id=data["assignment_id"],
            principal_id=str(principal_id).strip(),
            bound_agent_id=data["bound_agent_id"],
            execution_kind=data["execution_kind"],
            expired_fencing_epoch=data["expired_fencing_epoch"],
            acknowledged_at=data["acknowledged_at"],
            execution_expires_at=data["execution_expires_at"],
            expired_at=data["expired_at"],
            prior_claimable_status=data["prior_claimable_status"],
            continuation_version=data.get("continuation_version"),
            reason_code=data.get("reason_code", "EXECUTION_CLAIM_EXPIRED"),
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "ClaimExpiryRecord":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))
