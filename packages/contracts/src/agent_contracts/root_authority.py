import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .task_input import FORBIDDEN_SECRET_KEY_SUBSTRINGS


@dataclass(frozen=True)
class RootAuthorityRecord:
    """Non-secret root-authority record persisted at root task ingestion.

    Governed by ADR-0007:
    - Persisted at root ingestion bound to root_task_id;
    - Records baseline delegation policy context, depth, waves, and child bounds;
    - Tracks grant issuance and wave consumption;
    - Stores non-secret policy state only: contains NO credentials, API keys, or secret tokens.
    """

    root_task_id: str
    created_at: str
    updated_at: str
    policy_context: str = "default"
    max_delegation_depth: int = 1
    max_delegation_waves: int = 1
    max_children_per_wave: int = 1
    allowed_delegation_edges: tuple[tuple[str, str], ...] = (("research", "analysis"),)
    grants_issued: int = 0
    waves_consumed: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != 1:
            raise ValueError(f"Unsupported contract version: {self.version}")

        if not isinstance(self.root_task_id, str) or not self.root_task_id.strip():
            raise ValueError("root_task_id must be a non-empty string")
        object.__setattr__(self, "root_task_id", self.root_task_id.strip())

        if not isinstance(self.policy_context, str) or not self.policy_context.strip():
            raise ValueError("policy_context must be a non-empty string")
        clean_policy = self.policy_context.strip()
        lower_policy = clean_policy.lower()
        if any(substr in lower_policy for substr in FORBIDDEN_SECRET_KEY_SUBSTRINGS):
            raise ValueError(f"policy_context contains forbidden secret-bearing token: '{clean_policy}'")
        object.__setattr__(self, "policy_context", clean_policy)

        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty string")
        object.__setattr__(self, "created_at", self.created_at.strip())

        if not isinstance(self.updated_at, str) or not self.updated_at.strip():
            raise ValueError("updated_at must be a non-empty string")
        object.__setattr__(self, "updated_at", self.updated_at.strip())

        for name, val, min_val in (
            ("max_delegation_depth", self.max_delegation_depth, 1),
            ("max_delegation_waves", self.max_delegation_waves, 1),
            ("max_children_per_wave", self.max_children_per_wave, 1),
            ("grants_issued", self.grants_issued, 0),
            ("waves_consumed", self.waves_consumed, 0),
        ):
            if type(val) is bool or not isinstance(val, int):
                raise TypeError(f"{name} must be an int, got {type(val).__name__}")
            if val < min_val:
                raise ValueError(f"{name} must be >= {min_val}, got {val}")

        if not isinstance(self.allowed_delegation_edges, (list, tuple)):
            raise TypeError("allowed_delegation_edges must be a tuple or list of 2-tuples")
        clean_edges: list[tuple[str, str]] = []
        for edge in self.allowed_delegation_edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ValueError("each element of allowed_delegation_edges must be a (caller, target) pair")
            caller, target = edge
            if not isinstance(caller, str) or not caller.strip() or not isinstance(target, str) or not target.strip():
                raise ValueError("delegation edge elements must be non-empty strings")
            clean_edges.append((caller.strip(), target.strip()))
        object.__setattr__(self, "allowed_delegation_edges", tuple(clean_edges))

    @classmethod
    def create_default(
        cls,
        root_task_id: str,
        created_at: str | None = None,
    ) -> "RootAuthorityRecord":
        now = created_at or datetime.now(UTC).isoformat()
        return cls(
            root_task_id=root_task_id,
            policy_context="default",
            max_delegation_depth=1,
            max_delegation_waves=1,
            max_children_per_wave=1,
            allowed_delegation_edges=(("research", "analysis"),),
            grants_issued=0,
            waves_consumed=0,
            created_at=now,
            updated_at=now,
            version=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_task_id": self.root_task_id,
            "policy_context": self.policy_context,
            "max_delegation_depth": self.max_delegation_depth,
            "max_delegation_waves": self.max_delegation_waves,
            "max_children_per_wave": self.max_children_per_wave,
            "allowed_delegation_edges": [list(edge) for edge in self.allowed_delegation_edges],
            "grants_issued": self.grants_issued,
            "waves_consumed": self.waves_consumed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RootAuthorityRecord":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "root_task_id",
            "policy_context",
            "max_delegation_depth",
            "max_delegation_waves",
            "max_children_per_wave",
            "allowed_delegation_edges",
            "grants_issued",
            "waves_consumed",
            "created_at",
            "updated_at",
            "version",
        }
        unexpected_fields = data.keys() - allowed_fields
        if unexpected_fields:
            raise TypeError(f"unexpected RootAuthorityRecord fields: {sorted(unexpected_fields)}")

        if "version" in data:
            version = data["version"]
            if type(version) is bool or not isinstance(version, int):
                raise TypeError(f"version must be an int, got {type(version).__name__}")
            if version != 1:
                raise ValueError(f"Unsupported contract version: {version}")
        else:
            version = 1

        required = ("root_task_id", "created_at", "updated_at")
        for field_name in required:
            if field_name not in data:
                raise ValueError(f"{field_name} is required")

        raw_edges = data.get("allowed_delegation_edges", (("research", "analysis"),))
        if not isinstance(raw_edges, (list, tuple)):
            raise TypeError("allowed_delegation_edges must be a list or tuple")

        edges = tuple(tuple(e) for e in raw_edges)

        return cls(
            root_task_id=data["root_task_id"],
            policy_context=data.get("policy_context", "default"),
            max_delegation_depth=data.get("max_delegation_depth", 1),
            max_delegation_waves=data.get("max_delegation_waves", 1),
            max_children_per_wave=data.get("max_children_per_wave", 1),
            allowed_delegation_edges=edges,
            grants_issued=data.get("grants_issued", 0),
            waves_consumed=data.get("waves_consumed", 0),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "RootAuthorityRecord":
        return cls.from_dict(json.loads(value))
