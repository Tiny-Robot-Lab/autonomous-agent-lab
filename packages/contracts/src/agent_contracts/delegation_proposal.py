import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .artifact_reference import ArtifactReference
from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    freeze_value,
    normalize_canonical_utc_timestamp,
    to_plain_json_structure,
)

DELEGATION_PROPOSAL_CONTRACT_VERSION: int = 1
ASSIGNMENT_YIELD_CONTRACT_VERSION: int = 1
DELEGATION_RESPONSE_CONTRACT_VERSION: int = 1

ALLOWED_TARGET_CAPABILITIES: frozenset[str] = frozenset({"analysis"})

# Allowlist for stage-specific control fields (aligned with TaskContinuation)
ALLOWED_CONTROL_FIELDS: frozenset[str] = frozenset({
    "analysis_focus",
    "child_intent",
    "material_summary",
    "min_findings",
    "max_findings",
})

FORBIDDEN_AUTHORITY_FIELDS: frozenset[str] = frozenset({
    "child_task_id",
    "target_principal",
    "target_agent_id",
    "queue_name",
    "root_task_id",
    "parent_task_id",
    "fencing_epoch",
    "delegation_grant",
    "delegation_grant_id",
    "delegation_decision_id",
    "claim_deadline",
    "created_at",
    "status",
    "wave",
    "budget",
    "grant",
})


def compute_proposal_fingerprint(
    parent_task_id: str,
    proposal_id: str,
    assignment_id: str,
    target_capability: str,
    child_intent: str,
    artifact_reference: ArtifactReference,
    parent_input_version: int = 1,
    control_fields: Mapping[str, Any] | None = None,
) -> str:
    """Compute deterministic canonical SHA-256 proposal fingerprint over semantic fields."""
    normalized_control = to_plain_json_structure(control_fields or {})
    payload = {
        "artifact_reference": artifact_reference.to_dict(),
        "assignment_id": str(assignment_id).strip(),
        "child_intent": str(child_intent).strip(),
        "control_fields": normalized_control,
        "parent_input_version": int(parent_input_version),
        "parent_task_id": str(parent_task_id).strip(),
        "proposal_id": str(proposal_id).strip(),
        "target_capability": str(target_capability).strip(),
    }
    canonical_bytes = canonical_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


@dataclass(frozen=True)
class DelegationProposalRequest:
    """Caller-supplied delegation proposal from an active Research turn.

    Governed by ADR-0006 and ADR-0007:
    - Strictly scoped to Research -> Analysis delegation;
    - Requires caller proposal_id, target_capability (analysis), child_intent,
      and reference to sealed research artifact;
    - Forbids caller-asserted platform authority fields (child_task_id, principal, epoch, etc.).
    """

    proposal_id: str
    child_intent: str
    artifact_reference: ArtifactReference
    target_capability: str = "analysis"
    parent_input_version: int = 1
    control_fields: Mapping[str, Any] = field(default_factory=dict)
    version: int = DELEGATION_PROPOSAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != DELEGATION_PROPOSAL_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        if not isinstance(self.proposal_id, str) or not self.proposal_id.strip():
            raise ValueError("proposal_id must be a non-empty string")
        object.__setattr__(self, "proposal_id", self.proposal_id.strip())

        if not isinstance(self.target_capability, str) or not self.target_capability.strip():
            raise ValueError("target_capability must be a non-empty string")
        clean_cap = self.target_capability.strip()
        if clean_cap not in ALLOWED_TARGET_CAPABILITIES:
            raise ValueError(
                f"Unsupported target_capability '{clean_cap}'. Allowed: {sorted(ALLOWED_TARGET_CAPABILITIES)}"
            )
        object.__setattr__(self, "target_capability", clean_cap)

        if not isinstance(self.child_intent, str):
            raise TypeError(f"child_intent must be a str, got {type(self.child_intent).__name__}")
        clean_intent = self.child_intent.strip()
        if not clean_intent:
            raise ValueError("child_intent must not be empty")
        if len(self.child_intent) > 2000 or len(clean_intent) > 2000:
            raise ValueError("child_intent must not exceed 2000 characters")
        object.__setattr__(self, "child_intent", clean_intent)

        if not isinstance(self.artifact_reference, ArtifactReference):
            raise TypeError(
                f"artifact_reference must be an ArtifactReference, got {type(self.artifact_reference).__name__}"
            )

        if type(self.parent_input_version) is bool or not isinstance(self.parent_input_version, int):
            raise TypeError(
                f"parent_input_version must be an int, got {type(self.parent_input_version).__name__}"
            )
        if self.parent_input_version < 1:
            raise ValueError(f"parent_input_version must be >= 1, got {self.parent_input_version}")

        if not isinstance(self.control_fields, Mapping):
            raise TypeError(
                f"control_fields must be a Mapping, got {type(self.control_fields).__name__}"
            )

        unexpected_controls = set(self.control_fields.keys()) - ALLOWED_CONTROL_FIELDS
        if unexpected_controls:
            raise ValueError(
                f"Unexpected control_fields for delegation proposal: {sorted(unexpected_controls)}"
            )

        check_no_credential_bearing_keys(dict(self.control_fields), "DelegationProposalRequest.control_fields")
        object.__setattr__(self, "control_fields", freeze_value(dict(self.control_fields)))

    def to_dict(self) -> dict[str, Any]:
        data = {
            "artifact_reference": self.artifact_reference.to_dict(),
            "child_intent": self.child_intent,
            "control_fields": to_plain_json_structure(self.control_fields),
            "parent_input_version": self.parent_input_version,
            "proposal_id": self.proposal_id,
            "target_capability": self.target_capability,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, "DelegationProposalRequest.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegationProposalRequest":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        check_no_credential_bearing_keys(data, "DelegationProposalRequest payload")

        forbidden_present = set(data.keys()) & FORBIDDEN_AUTHORITY_FIELDS
        if forbidden_present:
            raise ValueError(
                f"Forbidden authority fields supplied by caller: {sorted(forbidden_present)}"
            )

        allowed_fields = {
            "artifact_reference",
            "child_intent",
            "control_fields",
            "parent_input_version",
            "proposal_id",
            "target_capability",
            "version",
        }
        unexpected = set(data.keys()) - allowed_fields
        if unexpected:
            raise ValueError(f"Unexpected fields in DelegationProposalRequest: {sorted(unexpected)}")

        raw_ref = data.get("artifact_reference")
        if isinstance(raw_ref, dict):
            ref = ArtifactReference.from_dict(raw_ref)
        elif isinstance(raw_ref, ArtifactReference):
            ref = raw_ref
        else:
            raise TypeError("artifact_reference must be a dict or ArtifactReference")

        return cls(
            proposal_id=data.get("proposal_id", ""),
            child_intent=data.get("child_intent", ""),
            artifact_reference=ref,
            target_capability=data.get("target_capability", "analysis"),
            parent_input_version=data.get("parent_input_version", 1),
            control_fields=data.get("control_fields", {}),
            version=data.get("version", DELEGATION_PROPOSAL_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> "DelegationProposalRequest":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class DelegationProposalRecord:
    """Authoritative Redis-persisted record of an accepted delegation proposal.

    Governed by ADR-0006 and ADR-0007:
    - Persisted at task.delegation_proposal:{parent_task_id}:{proposal_id};
    - Contains fingerprint over semantic caller fields;
    - Immutably binds parent turn, assignment, principal, and platform-generated child;
    - Serves as the authoritative source for idempotent replay.
    """

    parent_task_id: str
    proposal_id: str
    child_task_id: str
    delegation_decision_id: str
    assignment_id: str
    runtime_principal_id: str
    bound_agent_id: str
    proof_digest: str
    proposal_fingerprint: str
    delegation_grant_id: str
    child_claim_deadline: str
    accepted_at: str
    target_capability: str = "analysis"
    continuation_version: int = 1
    version: int = DELEGATION_PROPOSAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != DELEGATION_PROPOSAL_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "parent_task_id",
            "proposal_id",
            "child_task_id",
            "delegation_decision_id",
            "assignment_id",
            "runtime_principal_id",
            "bound_agent_id",
            "proof_digest",
            "proposal_fingerprint",
            "delegation_grant_id",
            "child_claim_deadline",
            "accepted_at",
            "target_capability",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if self.target_capability not in ALLOWED_TARGET_CAPABILITIES:
            raise ValueError(f"Unsupported target_capability '{self.target_capability}'")

        if type(self.continuation_version) is bool or not isinstance(self.continuation_version, int):
            raise TypeError(f"continuation_version must be an int, got {type(self.continuation_version).__name__}")
        if self.continuation_version < 1:
            raise ValueError(f"continuation_version must be >= 1, got {self.continuation_version}")

        object.__setattr__(
            self, "child_claim_deadline", normalize_canonical_utc_timestamp(self.child_claim_deadline)
        )
        object.__setattr__(
            self, "accepted_at", normalize_canonical_utc_timestamp(self.accepted_at)
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "accepted_at": self.accepted_at,
            "assignment_id": self.assignment_id,
            "bound_agent_id": self.bound_agent_id,
            "child_claim_deadline": self.child_claim_deadline,
            "child_task_id": self.child_task_id,
            "continuation_version": self.continuation_version,
            "delegation_decision_id": self.delegation_decision_id,
            "delegation_grant_id": self.delegation_grant_id,
            "parent_task_id": self.parent_task_id,
            "proof_digest": self.proof_digest,
            "proposal_fingerprint": self.proposal_fingerprint,
            "proposal_id": self.proposal_id,
            "runtime_principal_id": self.runtime_principal_id,
            "target_capability": self.target_capability,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, "DelegationProposalRecord.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegationProposalRecord":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        check_no_credential_bearing_keys(data, "DelegationProposalRecord payload")
        allowed = {
            "accepted_at",
            "assignment_id",
            "bound_agent_id",
            "child_claim_deadline",
            "child_task_id",
            "continuation_version",
            "delegation_decision_id",
            "delegation_grant_id",
            "parent_task_id",
            "proof_digest",
            "proposal_fingerprint",
            "proposal_id",
            "runtime_principal_id",
            "target_capability",
            "version",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise ValueError(f"Unexpected fields in DelegationProposalRecord: {sorted(unexpected)}")

        return cls(
            parent_task_id=data["parent_task_id"],
            proposal_id=data["proposal_id"],
            child_task_id=data["child_task_id"],
            delegation_decision_id=data["delegation_decision_id"],
            assignment_id=data["assignment_id"],
            runtime_principal_id=data["runtime_principal_id"],
            bound_agent_id=data["bound_agent_id"],
            proof_digest=data["proof_digest"],
            proposal_fingerprint=data["proposal_fingerprint"],
            delegation_grant_id=data["delegation_grant_id"],
            child_claim_deadline=data["child_claim_deadline"],
            accepted_at=data["accepted_at"],
            target_capability=data.get("target_capability", "analysis"),
            continuation_version=data.get("continuation_version", 1),
            version=data.get("version", DELEGATION_PROPOSAL_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> "DelegationProposalRecord":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class AssignmentYieldRecord:
    """Authoritative audit record created when an active assignment yields turn for delegation.

    Governed by ADR-0007:
    - Persisted at task.assignment_yield:{parent_task_id}:{assignment_id};
    - Explains why task.claim:{parent_task_id} was retired;
    - Links the retiring assignment to the accepted proposal and resulting child task.
    """

    assignment_id: str
    parent_task_id: str
    principal_id: str
    proposal_id: str
    child_task_id: str
    delegation_decision_id: str
    yielded_at: str
    yield_reason: str = "delegation_proposal"
    continuation_version: int = 1
    version: int = ASSIGNMENT_YIELD_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != ASSIGNMENT_YIELD_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "assignment_id",
            "parent_task_id",
            "principal_id",
            "proposal_id",
            "child_task_id",
            "delegation_decision_id",
            "yield_reason",
            "yielded_at",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if self.yield_reason != "delegation_proposal":
            raise ValueError(f"Unsupported yield_reason: '{self.yield_reason}'")

        if type(self.continuation_version) is bool or not isinstance(self.continuation_version, int):
            raise TypeError(f"continuation_version must be an int, got {type(self.continuation_version).__name__}")
        if self.continuation_version < 1:
            raise ValueError(f"continuation_version must be >= 1, got {self.continuation_version}")

        object.__setattr__(
            self, "yielded_at", normalize_canonical_utc_timestamp(self.yielded_at)
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "assignment_id": self.assignment_id,
            "child_task_id": self.child_task_id,
            "continuation_version": self.continuation_version,
            "delegation_decision_id": self.delegation_decision_id,
            "parent_task_id": self.parent_task_id,
            "principal_id": self.principal_id,
            "proposal_id": self.proposal_id,
            "version": self.version,
            "yield_reason": self.yield_reason,
            "yielded_at": self.yielded_at,
        }
        check_no_credential_bearing_keys(data, "AssignmentYieldRecord.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssignmentYieldRecord":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        check_no_credential_bearing_keys(data, "AssignmentYieldRecord payload")
        allowed = {
            "assignment_id",
            "child_task_id",
            "continuation_version",
            "delegation_decision_id",
            "parent_task_id",
            "principal_id",
            "proposal_id",
            "version",
            "yield_reason",
            "yielded_at",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise ValueError(f"Unexpected fields in AssignmentYieldRecord: {sorted(unexpected)}")

        return cls(
            assignment_id=data["assignment_id"],
            parent_task_id=data["parent_task_id"],
            principal_id=data["principal_id"],
            proposal_id=data["proposal_id"],
            child_task_id=data["child_task_id"],
            delegation_decision_id=data["delegation_decision_id"],
            yielded_at=data["yielded_at"],
            yield_reason=data.get("yield_reason", "delegation_proposal"),
            continuation_version=data.get("continuation_version", 1),
            version=data.get("version", ASSIGNMENT_YIELD_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> "AssignmentYieldRecord":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class DelegationProposalResponse:
    """Platform response returned on successful proposal acceptance or idempotent replay."""

    parent_task_id: str
    proposal_id: str
    child_task_id: str
    delegation_decision_id: str
    child_claim_deadline: str
    accepted_at: str
    target_capability: str = "analysis"
    continuation_version: int = 1
    status: str = "accepted"
    version: int = DELEGATION_RESPONSE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != DELEGATION_RESPONSE_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "parent_task_id",
            "proposal_id",
            "child_task_id",
            "delegation_decision_id",
            "child_claim_deadline",
            "accepted_at",
            "target_capability",
            "status",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if type(self.continuation_version) is bool or not isinstance(self.continuation_version, int):
            raise TypeError(f"continuation_version must be an int, got {type(self.continuation_version).__name__}")
        if self.continuation_version < 1:
            raise ValueError(f"continuation_version must be >= 1, got {self.continuation_version}")

        object.__setattr__(
            self, "child_claim_deadline", normalize_canonical_utc_timestamp(self.child_claim_deadline)
        )
        object.__setattr__(
            self, "accepted_at", normalize_canonical_utc_timestamp(self.accepted_at)
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "accepted_at": self.accepted_at,
            "child_claim_deadline": self.child_claim_deadline,
            "child_task_id": self.child_task_id,
            "continuation_version": self.continuation_version,
            "delegation_decision_id": self.delegation_decision_id,
            "parent_task_id": self.parent_task_id,
            "proposal_id": self.proposal_id,
            "status": self.status,
            "target_capability": self.target_capability,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, "DelegationProposalResponse.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegationProposalResponse":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        check_no_credential_bearing_keys(data, "DelegationProposalResponse payload")
        allowed = {
            "accepted_at",
            "child_claim_deadline",
            "child_task_id",
            "continuation_version",
            "delegation_decision_id",
            "parent_task_id",
            "proposal_id",
            "status",
            "target_capability",
            "version",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise ValueError(f"Unexpected fields in DelegationProposalResponse: {sorted(unexpected)}")

        return cls(
            parent_task_id=data["parent_task_id"],
            proposal_id=data["proposal_id"],
            child_task_id=data["child_task_id"],
            delegation_decision_id=data["delegation_decision_id"],
            child_claim_deadline=data["child_claim_deadline"],
            accepted_at=data["accepted_at"],
            target_capability=data.get("target_capability", "analysis"),
            continuation_version=data.get("continuation_version", 1),
            status=data.get("status", "accepted"),
            version=data.get("version", DELEGATION_RESPONSE_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> "DelegationProposalResponse":
        return cls.from_dict(json.loads(value))
