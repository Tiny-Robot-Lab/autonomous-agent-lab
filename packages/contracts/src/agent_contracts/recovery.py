import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .active_claim import ALLOWED_PRIOR_CLAIMABLE_STATUSES
from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    normalize_canonical_utc_timestamp,
)

RECOVERY_CONTRACT_VERSION: int = 1
ALLOWED_RECOVERY_DISPOSITIONS: frozenset[str] = frozenset({"restore_claimable"})


def compute_recovery_decision_fingerprint(
    *,
    task_id: str,
    expired_claim_id: str,
    expired_assignment_id: str,
    expired_fencing_epoch: int,
    disposition: str,
) -> str:
    """Compute deterministic canonical SHA-256 fingerprint for a recovery decision."""
    payload = {
        "disposition": str(disposition).strip(),
        "expired_assignment_id": str(expired_assignment_id).strip(),
        "expired_claim_id": str(expired_claim_id).strip(),
        "expired_fencing_epoch": int(expired_fencing_epoch),
        "task_id": str(task_id).strip(),
    }
    canonical_bytes = canonical_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


@dataclass(frozen=True)
class RecoveryDecisionCommand:
    """Operator command to resolve a task in recovery_required.

    Governed by ADR-0007 (Decision 12) and Issue #223:
    - Binds recovery_decision_id, task, expired claim, assignment, epoch, and disposition;
    - Strictly forbids caller-supplied new fence, assignment, proof, or target status;
    - Only 'restore_claimable' disposition is supported.
    """

    recovery_decision_id: str
    task_id: str
    expired_claim_id: str
    expired_assignment_id: str
    expired_fencing_epoch: int
    disposition: str = "restore_claimable"
    version: int = RECOVERY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != RECOVERY_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "recovery_decision_id",
            "task_id",
            "expired_claim_id",
            "expired_assignment_id",
            "disposition",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if self.disposition not in ALLOWED_RECOVERY_DISPOSITIONS:
            raise ValueError(
                f"Unsupported recovery disposition '{self.disposition}'. Allowed: {sorted(ALLOWED_RECOVERY_DISPOSITIONS)}"
            )

        if type(self.expired_fencing_epoch) is bool or not isinstance(self.expired_fencing_epoch, int):
            raise TypeError(f"expired_fencing_epoch must be an int, got {type(self.expired_fencing_epoch).__name__}")
        if self.expired_fencing_epoch < 1:
            raise ValueError(f"expired_fencing_epoch must be >= 1, got {self.expired_fencing_epoch}")

    def semantic_fingerprint(self) -> str:
        return compute_recovery_decision_fingerprint(
            task_id=self.task_id,
            expired_claim_id=self.expired_claim_id,
            expired_assignment_id=self.expired_assignment_id,
            expired_fencing_epoch=self.expired_fencing_epoch,
            disposition=self.disposition,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "disposition": self.disposition,
            "expired_assignment_id": self.expired_assignment_id,
            "expired_claim_id": self.expired_claim_id,
            "expired_fencing_epoch": self.expired_fencing_epoch,
            "recovery_decision_id": self.recovery_decision_id,
            "task_id": self.task_id,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="RecoveryDecisionCommand.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoveryDecisionCommand":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {
            "disposition",
            "expired_assignment_id",
            "expired_claim_id",
            "expired_fencing_epoch",
            "recovery_decision_id",
            "task_id",
            "version",
        }
        unexpected = data.keys() - allowed
        if unexpected:
            raise TypeError(f"unexpected RecoveryDecisionCommand fields: {sorted(unexpected)}")

        required = (
            "recovery_decision_id",
            "task_id",
            "expired_claim_id",
            "expired_assignment_id",
            "expired_fencing_epoch",
        )
        for req in required:
            if req not in data:
                raise ValueError(f"{req} is required")

        version = data.get("version", RECOVERY_CONTRACT_VERSION)
        if type(version) is bool or not isinstance(version, int):
            raise TypeError(f"version must be an int, got {type(version).__name__}")
        if version != RECOVERY_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {version}")

        return cls(
            recovery_decision_id=data["recovery_decision_id"],
            task_id=data["task_id"],
            expired_claim_id=data["expired_claim_id"],
            expired_assignment_id=data["expired_assignment_id"],
            expired_fencing_epoch=data["expired_fencing_epoch"],
            disposition=data.get("disposition", "restore_claimable"),
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "RecoveryDecisionCommand":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class RecoveryDecisionRecord:
    """Authoritative Redis-persisted record of an executed recovery resolution.

    Governed by ADR-0007 (Decision 12) and Issue #223:
    - Persisted at task.recovery_decision:{task_id}:{recovery_decision_id};
    - Binds recovery_decision_id, task, expired claim, assignment, epoch, disposition, and semantic fingerprint;
    - Records the operator principal and resulting restored status;
    - Serves as the authoritative source for idempotent replay.
    """

    recovery_decision_id: str
    task_id: str
    expired_claim_id: str
    expired_assignment_id: str
    expired_fencing_epoch: int
    disposition: str
    semantic_fingerprint: str
    prior_status: str
    restored_status: str
    operator_principal_id: str
    resolved_at: str
    version: int = RECOVERY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != RECOVERY_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "recovery_decision_id",
            "task_id",
            "expired_claim_id",
            "expired_assignment_id",
            "disposition",
            "semantic_fingerprint",
            "prior_status",
            "restored_status",
            "operator_principal_id",
            "resolved_at",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if self.disposition not in ALLOWED_RECOVERY_DISPOSITIONS:
            raise ValueError(
                f"Unsupported recovery disposition '{self.disposition}'. Allowed: {sorted(ALLOWED_RECOVERY_DISPOSITIONS)}"
            )

        if self.restored_status not in ALLOWED_PRIOR_CLAIMABLE_STATUSES:
            raise ValueError(
                f"Invalid restored_status '{self.restored_status}'. Allowed: {sorted(ALLOWED_PRIOR_CLAIMABLE_STATUSES)}"
            )

        if type(self.expired_fencing_epoch) is bool or not isinstance(self.expired_fencing_epoch, int):
            raise TypeError(f"expired_fencing_epoch must be an int, got {type(self.expired_fencing_epoch).__name__}")
        if self.expired_fencing_epoch < 1:
            raise ValueError(f"expired_fencing_epoch must be >= 1, got {self.expired_fencing_epoch}")

        object.__setattr__(
            self, "resolved_at", normalize_canonical_utc_timestamp(self.resolved_at)
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "disposition": self.disposition,
            "expired_assignment_id": self.expired_assignment_id,
            "expired_claim_id": self.expired_claim_id,
            "expired_fencing_epoch": self.expired_fencing_epoch,
            "operator_principal_id": self.operator_principal_id,
            "prior_status": self.prior_status,
            "recovery_decision_id": self.recovery_decision_id,
            "resolved_at": self.resolved_at,
            "restored_status": self.restored_status,
            "semantic_fingerprint": self.semantic_fingerprint,
            "task_id": self.task_id,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="RecoveryDecisionRecord.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoveryDecisionRecord":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {
            "disposition",
            "expired_assignment_id",
            "expired_claim_id",
            "expired_fencing_epoch",
            "operator_principal_id",
            "prior_status",
            "recovery_decision_id",
            "resolved_at",
            "restored_status",
            "semantic_fingerprint",
            "task_id",
            "version",
        }
        unexpected = data.keys() - allowed
        if unexpected:
            raise TypeError(f"unexpected RecoveryDecisionRecord fields: {sorted(unexpected)}")

        required = (
            "recovery_decision_id",
            "task_id",
            "expired_claim_id",
            "expired_assignment_id",
            "expired_fencing_epoch",
            "disposition",
            "semantic_fingerprint",
            "prior_status",
            "restored_status",
            "operator_principal_id",
            "resolved_at",
        )
        for req in required:
            if req not in data:
                raise ValueError(f"{req} is required")

        version = data.get("version", RECOVERY_CONTRACT_VERSION)
        if type(version) is bool or not isinstance(version, int):
            raise TypeError(f"version must be an int, got {type(version).__name__}")
        if version != RECOVERY_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {version}")

        return cls(
            recovery_decision_id=data["recovery_decision_id"],
            task_id=data["task_id"],
            expired_claim_id=data["expired_claim_id"],
            expired_assignment_id=data["expired_assignment_id"],
            expired_fencing_epoch=data["expired_fencing_epoch"],
            disposition=data["disposition"],
            semantic_fingerprint=data["semantic_fingerprint"],
            prior_status=data["prior_status"],
            restored_status=data["restored_status"],
            operator_principal_id=data["operator_principal_id"],
            resolved_at=data["resolved_at"],
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "RecoveryDecisionRecord":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))
