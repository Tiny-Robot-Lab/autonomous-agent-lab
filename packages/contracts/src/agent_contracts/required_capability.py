import json
from dataclasses import dataclass
from typing import Any

from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    normalize_canonical_utc_timestamp,
)

REQUIRED_CAPABILITY_CONTRACT_VERSION = 1
ALLOWED_REQUIRED_CAPABILITIES: frozenset[str] = frozenset({
    "general",
    "research",
    "analysis",
})


@dataclass(frozen=True)
class RequiredCapabilityRecord:
    """Immutable authoritative record of a task's required capability.

    Governed by ADR-0007 (Decision 7) and Issue #220:
    - Persisted at task registration (root or child);
    - Stored at task.required_capability:{task_id};
    - Distinct from mutable task status;
    - Validated atomically in CLAIM_TASK_LUA prior to claim mutation;
    - Free of credentials, secret proofs, or runtime identities.
    """

    task_id: str
    required_capability: str
    created_at: str
    version: int = REQUIRED_CAPABILITY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != REQUIRED_CAPABILITY_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        object.__setattr__(self, "task_id", self.task_id.strip())

        if not isinstance(self.required_capability, str) or not self.required_capability.strip():
            raise ValueError("required_capability must be a non-empty string")
        clean_cap = self.required_capability.strip()
        if clean_cap not in ALLOWED_REQUIRED_CAPABILITIES:
            raise ValueError(
                f"Unrecognized required_capability '{clean_cap}'; must be one of {sorted(ALLOWED_REQUIRED_CAPABILITIES)}"
            )
        object.__setattr__(self, "required_capability", clean_cap)

        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty string")
        object.__setattr__(
            self, "created_at", normalize_canonical_utc_timestamp(self.created_at)
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "created_at": self.created_at,
            "required_capability": self.required_capability,
            "task_id": self.task_id,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, "RequiredCapabilityRecord.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequiredCapabilityRecord":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        check_no_credential_bearing_keys(data, "RequiredCapabilityRecord payload")

        allowed_fields = {
            "created_at",
            "required_capability",
            "task_id",
            "version",
        }
        unexpected = data.keys() - allowed_fields
        if unexpected:
            raise TypeError(f"unexpected RequiredCapabilityRecord fields: {sorted(unexpected)}")

        required = ("task_id", "required_capability", "created_at")
        for req in required:
            if req not in data:
                raise ValueError(f"{req} is required")

        if "version" in data:
            version = data["version"]
            if type(version) is bool or not isinstance(version, int):
                raise TypeError(f"version must be an int, got {type(version).__name__}")
            if version != REQUIRED_CAPABILITY_CONTRACT_VERSION:
                raise ValueError(f"Unsupported contract version: {version}")
        else:
            version = REQUIRED_CAPABILITY_CONTRACT_VERSION

        return cls(
            task_id=data["task_id"],
            required_capability=data["required_capability"],
            created_at=data["created_at"],
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "RequiredCapabilityRecord":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))
