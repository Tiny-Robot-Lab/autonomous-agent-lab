import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .serialization import canonical_json_dumps

BASELINE_POLICY_CONTEXT = "default"
BASELINE_MAX_DEPTH = 1
BASELINE_MAX_WAVES = 1
BASELINE_MAX_CHILDREN_PER_WAVE = 1
BASELINE_DELEGATION_EDGES: tuple[tuple[str, str], ...] = (("research", "analysis"),)


@dataclass(frozen=True)
class RootAuthorityRecord:
    """Non-secret root-authority record persisted at root task ingestion.

    Governed by ADR-0007:
    - Persisted at root ingestion bound to root_task_id;
    - Strictly constrained to the accepted baseline: depth 1, 1 wave, 1 child, research->analysis;
    - Tracks baseline authority with no credentials, API keys, or secret proofs;
    - Fixed baseline in #219; no authority escalation permitted.
    """

    root_task_id: str
    created_at: str
    updated_at: str
    policy_context: str = BASELINE_POLICY_CONTEXT
    max_delegation_depth: int = BASELINE_MAX_DEPTH
    max_delegation_waves: int = BASELINE_MAX_WAVES
    max_children_per_wave: int = BASELINE_MAX_CHILDREN_PER_WAVE
    allowed_delegation_edges: tuple[tuple[str, str], ...] = BASELINE_DELEGATION_EDGES
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

        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty string")
        object.__setattr__(self, "created_at", self.created_at.strip())

        if not isinstance(self.updated_at, str) or not self.updated_at.strip():
            raise ValueError("updated_at must be a non-empty string")
        object.__setattr__(self, "updated_at", self.updated_at.strip())

        if self.policy_context != BASELINE_POLICY_CONTEXT:
            raise ValueError(
                f"RootAuthorityRecord policy_context must be '{BASELINE_POLICY_CONTEXT}', got '{self.policy_context}'"
            )

        if self.max_delegation_depth != BASELINE_MAX_DEPTH:
            raise ValueError(
                f"RootAuthorityRecord max_delegation_depth must be {BASELINE_MAX_DEPTH}, got {self.max_delegation_depth}"
            )

        if self.max_delegation_waves != BASELINE_MAX_WAVES:
            raise ValueError(
                f"RootAuthorityRecord max_delegation_waves must be {BASELINE_MAX_WAVES}, got {self.max_delegation_waves}"
            )

        if self.max_children_per_wave != BASELINE_MAX_CHILDREN_PER_WAVE:
            raise ValueError(
                f"RootAuthorityRecord max_children_per_wave must be {BASELINE_MAX_CHILDREN_PER_WAVE}, "
                f"got {self.max_children_per_wave}"
            )

        if self.grants_issued != 0:
            raise ValueError(
                f"RootAuthorityRecord baseline grants_issued must be 0 at ingestion, got {self.grants_issued}"
            )

        if self.waves_consumed != 0:
            raise ValueError(
                f"RootAuthorityRecord baseline waves_consumed must be 0 at ingestion, got {self.waves_consumed}"
            )

        if not isinstance(self.allowed_delegation_edges, (list, tuple)):
            raise TypeError("allowed_delegation_edges must be a tuple or list of 2-tuples")
        clean_edges: list[tuple[str, str]] = []
        for edge in self.allowed_delegation_edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ValueError("each element of allowed_delegation_edges must be a (caller, target) pair")
            caller, target = edge
            if not isinstance(caller, str) or not isinstance(target, str):
                raise TypeError("delegation edge elements must be strings")
            clean_edges.append((caller.strip(), target.strip()))
        tuple_edges = tuple(clean_edges)
        if tuple_edges != BASELINE_DELEGATION_EDGES:
            raise ValueError(
                f"RootAuthorityRecord allowed_delegation_edges must be {BASELINE_DELEGATION_EDGES}, got {tuple_edges}"
            )
        object.__setattr__(self, "allowed_delegation_edges", tuple_edges)

    @classmethod
    def create_default(
        cls,
        root_task_id: str,
        created_at: str | None = None,
    ) -> "RootAuthorityRecord":
        now = created_at or datetime.now(UTC).isoformat()
        return cls(
            root_task_id=root_task_id,
            policy_context=BASELINE_POLICY_CONTEXT,
            max_delegation_depth=BASELINE_MAX_DEPTH,
            max_delegation_waves=BASELINE_MAX_WAVES,
            max_children_per_wave=BASELINE_MAX_CHILDREN_PER_WAVE,
            allowed_delegation_edges=BASELINE_DELEGATION_EDGES,
            grants_issued=0,
            waves_consumed=0,
            created_at=now,
            updated_at=now,
            version=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_delegation_edges": [list(edge) for edge in self.allowed_delegation_edges],
            "created_at": self.created_at,
            "grants_issued": self.grants_issued,
            "max_children_per_wave": self.max_children_per_wave,
            "max_delegation_depth": self.max_delegation_depth,
            "max_delegation_waves": self.max_delegation_waves,
            "policy_context": self.policy_context,
            "root_task_id": self.root_task_id,
            "updated_at": self.updated_at,
            "version": self.version,
            "waves_consumed": self.waves_consumed,
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RootAuthorityRecord":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "allowed_delegation_edges",
            "created_at",
            "grants_issued",
            "max_children_per_wave",
            "max_delegation_depth",
            "max_delegation_waves",
            "policy_context",
            "root_task_id",
            "updated_at",
            "version",
            "waves_consumed",
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

        raw_edges = data.get("allowed_delegation_edges", BASELINE_DELEGATION_EDGES)
        if not isinstance(raw_edges, (list, tuple)):
            raise TypeError("allowed_delegation_edges must be a list or tuple")

        edges = tuple(tuple(e) for e in raw_edges)

        return cls(
            root_task_id=data["root_task_id"],
            policy_context=data.get("policy_context", BASELINE_POLICY_CONTEXT),
            max_delegation_depth=data.get("max_delegation_depth", BASELINE_MAX_DEPTH),
            max_delegation_waves=data.get("max_delegation_waves", BASELINE_MAX_WAVES),
            max_children_per_wave=data.get("max_children_per_wave", BASELINE_MAX_CHILDREN_PER_WAVE),
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
