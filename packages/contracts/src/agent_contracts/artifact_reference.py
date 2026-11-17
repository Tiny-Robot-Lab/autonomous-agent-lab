from dataclasses import dataclass
from typing import Any

from .serialization import canonical_json_dumps


@dataclass(frozen=True)
class ArtifactReference:
    """Explicit, versioned, task-scoped reference to a StageArtifact.

    Governed by ADR-0007:
    - Identifies a stage artifact by producing task, artifact ID, and version;
    - Immutable and canonical;
    - References stage-required evidence rather than duplicating payloads into lineage or continuations.
    """

    producing_task_id: str
    artifact_id: str
    artifact_version: int = 1

    def __post_init__(self) -> None:
        if type(self.artifact_version) is bool or not isinstance(self.artifact_version, int):
            raise TypeError(f"artifact_version must be an int, got {type(self.artifact_version).__name__}")
        if self.artifact_version < 1:
            raise ValueError(f"artifact_version must be >= 1, got {self.artifact_version}")

        if not isinstance(self.producing_task_id, str) or not self.producing_task_id.strip():
            raise ValueError("producing_task_id must be a non-empty string")
        object.__setattr__(self, "producing_task_id", self.producing_task_id.strip())

        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        object.__setattr__(self, "artifact_id", self.artifact_id.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "producing_task_id": self.producing_task_id,
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactReference":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {"producing_task_id", "artifact_id", "artifact_version"}
        unexpected = data.keys() - allowed
        if unexpected:
            raise TypeError(f"unexpected ArtifactReference fields: {sorted(unexpected)}")
        if "producing_task_id" not in data:
            raise ValueError("producing_task_id is required")
        if "artifact_id" not in data:
            raise ValueError("artifact_id is required")
        return cls(
            producing_task_id=data["producing_task_id"],
            artifact_id=data["artifact_id"],
            artifact_version=data.get("artifact_version", 1),
        )

    @classmethod
    def from_json(cls, value: str) -> "ArtifactReference":
        import json

        return cls.from_dict(json.loads(value))
