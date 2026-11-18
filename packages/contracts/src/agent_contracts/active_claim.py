import json
from dataclasses import dataclass
from typing import Any

from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    normalize_canonical_utc_timestamp,
)

ACTIVE_CLAIM_CONTRACT_VERSION = 1
ALLOWED_CLAIM_STATES = frozenset({"claimed_unacknowledged", "acknowledged"})
ALLOWED_CLAIM_EXECUTION_KINDS = frozenset({"initial", "delegated_child", "resumed_continuation"})
ALLOWED_PRIOR_CLAIMABLE_STATUSES = frozenset({"pending", "resumable"})


@dataclass(frozen=True)
class ActiveClaim:
    """Authoritative Redis-persisted active claim and fencing record.

    Governed by ADR-0007 (Decisions 7, 12, 13) and Issue #220:
    - Binds task, assignment, runtime principal, bound agent, and fencing epoch;
    - Stores proof_digest (HMAC-SHA256 of raw proof) rather than plaintext proof;
    - Tracks claim state (claimed_unacknowledged, acknowledged);
    - Idempotency keyed by (runtime_principal_id, request_id).
    """

    task_id: str
    assignment_id: str
    claim_id: str
    root_task_id: str
    turn_id: str
    runtime_principal_id: str
    bound_agent_id: str
    assigned_capability: str
    execution_kind: str
    fencing_epoch: int
    proof_digest: str
    request_id: str
    claimed_at: str
    delivery_expires_at: str
    prior_claimable_status: str = "pending"
    state: str = "claimed_unacknowledged"
    continuation_version: int | None = None
    delegation_grant_id: str | None = None
    acknowledged_at: str | None = None
    version: int = ACTIVE_CLAIM_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != ACTIVE_CLAIM_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "task_id",
            "assignment_id",
            "claim_id",
            "root_task_id",
            "turn_id",
            "runtime_principal_id",
            "bound_agent_id",
            "assigned_capability",
            "execution_kind",
            "proof_digest",
            "request_id",
            "claimed_at",
            "delivery_expires_at",
            "state",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if self.state not in ALLOWED_CLAIM_STATES:
            raise ValueError(
                f"Invalid claim state: '{self.state}'. Allowed states: {sorted(ALLOWED_CLAIM_STATES)}"
            )

        if self.prior_claimable_status not in ALLOWED_PRIOR_CLAIMABLE_STATUSES:
            raise ValueError(
                f"Invalid prior_claimable_status: '{self.prior_claimable_status}'. Allowed: {sorted(ALLOWED_PRIOR_CLAIMABLE_STATUSES)}"
            )

        if self.state == "acknowledged" and self.acknowledged_at is None:
            raise ValueError("acknowledged_at must be set when state is 'acknowledged'")
        if self.state == "claimed_unacknowledged" and self.acknowledged_at is not None:
            raise ValueError("acknowledged_at must be None when state is 'claimed_unacknowledged'")

        if self.execution_kind not in ALLOWED_CLAIM_EXECUTION_KINDS:
            raise ValueError(
                f"Invalid execution_kind: '{self.execution_kind}'. Allowed: {sorted(ALLOWED_CLAIM_EXECUTION_KINDS)}"
            )

        if type(self.fencing_epoch) is bool or not isinstance(self.fencing_epoch, int):
            raise TypeError(f"fencing_epoch must be an int, got {type(self.fencing_epoch).__name__}")
        if self.fencing_epoch < 1:
            raise ValueError(f"fencing_epoch must be >= 1, got {self.fencing_epoch}")

        if self.continuation_version is not None:
            if type(self.continuation_version) is bool or not isinstance(self.continuation_version, int):
                raise TypeError(
                    f"continuation_version must be an int or None, got {type(self.continuation_version).__name__}"
                )
            if self.continuation_version < 1:
                raise ValueError(
                    f"continuation_version must be >= 1 when set, got {self.continuation_version}"
                )

        if self.delegation_grant_id is not None:
            if not isinstance(self.delegation_grant_id, str) or not self.delegation_grant_id.strip():
                raise ValueError("delegation_grant_id must be a non-empty string when present")
            object.__setattr__(self, "delegation_grant_id", self.delegation_grant_id.strip())

        object.__setattr__(
            self, "claimed_at", normalize_canonical_utc_timestamp(self.claimed_at)
        )
        object.__setattr__(
            self,
            "delivery_expires_at",
            normalize_canonical_utc_timestamp(self.delivery_expires_at),
        )

        if self.acknowledged_at is not None:
            if not isinstance(self.acknowledged_at, str) or not self.acknowledged_at.strip():
                raise ValueError("acknowledged_at must be a non-empty string when present")
            object.__setattr__(
                self,
                "acknowledged_at",
                normalize_canonical_utc_timestamp(self.acknowledged_at),
            )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "acknowledged_at": self.acknowledged_at,
            "assigned_capability": self.assigned_capability,
            "assignment_id": self.assignment_id,
            "bound_agent_id": self.bound_agent_id,
            "claim_id": self.claim_id,
            "claimed_at": self.claimed_at,
            "continuation_version": self.continuation_version,
            "delegation_grant_id": self.delegation_grant_id,
            "delivery_expires_at": self.delivery_expires_at,
            "execution_kind": self.execution_kind,
            "fencing_epoch": self.fencing_epoch,
            "prior_claimable_status": self.prior_claimable_status,
            "proof_digest": self.proof_digest,
            "request_id": self.request_id,
            "root_task_id": self.root_task_id,
            "runtime_principal_id": self.runtime_principal_id,
            "state": self.state,
            "task_id": self.task_id,
            "turn_id": self.turn_id,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="ActiveClaim.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActiveClaim":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "acknowledged_at",
            "assigned_capability",
            "assignment_id",
            "bound_agent_id",
            "claim_id",
            "claimed_at",
            "continuation_version",
            "delegation_grant_id",
            "delivery_expires_at",
            "execution_kind",
            "fencing_epoch",
            "prior_claimable_status",
            "proof_digest",
            "request_id",
            "root_task_id",
            "runtime_principal_id",
            "state",
            "task_id",
            "turn_id",
            "version",
        }
        unexpected = data.keys() - allowed_fields
        if unexpected:
            raise TypeError(f"unexpected ActiveClaim fields: {sorted(unexpected)}")

        required = (
            "task_id",
            "assignment_id",
            "claim_id",
            "root_task_id",
            "turn_id",
            "runtime_principal_id",
            "bound_agent_id",
            "assigned_capability",
            "execution_kind",
            "fencing_epoch",
            "prior_claimable_status",
            "proof_digest",
            "request_id",
            "claimed_at",
            "delivery_expires_at",
        )
        for req in required:
            if req not in data:
                raise ValueError(f"{req} is required")

        version = data.get("version", ACTIVE_CLAIM_CONTRACT_VERSION)
        if type(version) is bool or not isinstance(version, int):
            raise TypeError(f"version must be an int, got {type(version).__name__}")
        if version != ACTIVE_CLAIM_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {version}")

        return cls(
            task_id=data["task_id"],
            assignment_id=data["assignment_id"],
            claim_id=data["claim_id"],
            root_task_id=data["root_task_id"],
            turn_id=data["turn_id"],
            runtime_principal_id=data["runtime_principal_id"],
            bound_agent_id=data["bound_agent_id"],
            assigned_capability=data["assigned_capability"],
            execution_kind=data["execution_kind"],
            fencing_epoch=data["fencing_epoch"],
            proof_digest=data["proof_digest"],
            request_id=data["request_id"],
            claimed_at=data["claimed_at"],
            delivery_expires_at=data["delivery_expires_at"],
            prior_claimable_status=data.get("prior_claimable_status", "pending"),
            state=data.get("state", "claimed_unacknowledged"),
            continuation_version=data.get("continuation_version"),
            delegation_grant_id=data.get("delegation_grant_id"),
            acknowledged_at=data.get("acknowledged_at"),
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "ActiveClaim":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))
