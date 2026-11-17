import json
from dataclasses import dataclass
from typing import Any

from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    normalize_canonical_utc_timestamp,
)

WAKE_UP_HINT_CONTRACT_VERSION = 1
ALLOWED_HINT_EXECUTION_KINDS = frozenset({"initial", "delegated_child", "resumed_continuation"})


@dataclass(frozen=True)
class WakeUpHint:
    """Non-secret, non-authoritative wake-up notification published to capability queues.

    Governed by ADR-0007 (Decision 6) and Issue #220:
    - Published to agent.hints.{capability} queues under authenticated_claim delivery mode;
    - Grants zero execution authority or task ownership;
    - May be capability-level or carry task_id when applicable;
    - Contains no assignment proof, credential, prompt, TaskInput, artifact, or authority grant.
    """

    hint_id: str
    capability: str
    created_at: str
    execution_kind: str = "initial"
    task_id: str | None = None
    expected_continuation_version: int | None = None
    version: int = WAKE_UP_HINT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != WAKE_UP_HINT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in ("hint_id", "capability", "created_at", "execution_kind"):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        object.__setattr__(
            self, "created_at", normalize_canonical_utc_timestamp(self.created_at)
        )

        if self.execution_kind not in ALLOWED_HINT_EXECUTION_KINDS:
            raise ValueError(
                f"Invalid execution_kind: '{self.execution_kind}'. Allowed: {sorted(ALLOWED_HINT_EXECUTION_KINDS)}"
            )

        if self.task_id is not None:
            if not isinstance(self.task_id, str) or not self.task_id.strip():
                raise ValueError("task_id must be a non-empty string when present")
            object.__setattr__(self, "task_id", self.task_id.strip())

        if self.expected_continuation_version is not None:
            if type(self.expected_continuation_version) is bool or not isinstance(
                self.expected_continuation_version, int
            ):
                raise TypeError(
                    f"expected_continuation_version must be an int or None, got {type(self.expected_continuation_version).__name__}"
                )
            if self.expected_continuation_version < 1:
                raise ValueError(
                    f"expected_continuation_version must be >= 1 when set, got {self.expected_continuation_version}"
                )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "capability": self.capability,
            "created_at": self.created_at,
            "execution_kind": self.execution_kind,
            "expected_continuation_version": self.expected_continuation_version,
            "hint_id": self.hint_id,
            "task_id": self.task_id,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="WakeUpHint.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WakeUpHint":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "capability",
            "created_at",
            "execution_kind",
            "expected_continuation_version",
            "hint_id",
            "task_id",
            "version",
        }
        unexpected = data.keys() - allowed_fields
        if unexpected:
            raise TypeError(f"unexpected WakeUpHint fields: {sorted(unexpected)}")

        required = ("hint_id", "capability", "created_at")
        for req in required:
            if req not in data:
                raise ValueError(f"{req} is required")

        version = data.get("version", WAKE_UP_HINT_CONTRACT_VERSION)
        if type(version) is bool or not isinstance(version, int):
            raise TypeError(f"version must be an int, got {type(version).__name__}")
        if version != WAKE_UP_HINT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {version}")

        return cls(
            hint_id=data["hint_id"],
            capability=data["capability"],
            created_at=data["created_at"],
            execution_kind=data.get("execution_kind", "initial"),
            task_id=data.get("task_id"),
            expected_continuation_version=data.get("expected_continuation_version"),
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "WakeUpHint":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))
