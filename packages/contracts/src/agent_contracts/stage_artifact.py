import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .task_input import _check_no_secret_bearing_keys


@dataclass(frozen=True)
class StageArtifact:
    """Immutable, versioned, task-scoped stage artifact with provenance.

    Governed by ADR-0007:
    - Retains intermediate research or turn material across turns;
    - Keyed by (producing_task_id, artifact_id, artifact_version);
    - Sealed and write-once;
    - Provenance metadata identifies producing task, turn/assignment, schema, and version;
    - Referenced by continuation and child TaskInput rather than copying content into control state.
    """

    artifact_id: str
    producing_task_id: str
    artifact_type: str
    content: dict[str, Any] | str
    created_at: str
    producing_turn_id: str | None = None
    artifact_version: int = 1
    sealed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != 1:
            raise ValueError(f"Unsupported contract version: {self.version}")

        if type(self.artifact_version) is bool or not isinstance(self.artifact_version, int):
            raise TypeError(f"artifact_version must be an int, got {type(self.artifact_version).__name__}")
        if self.artifact_version < 1:
            raise ValueError(f"artifact_version must be >= 1, got {self.artifact_version}")

        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        object.__setattr__(self, "artifact_id", self.artifact_id.strip())

        if not isinstance(self.producing_task_id, str) or not self.producing_task_id.strip():
            raise ValueError("producing_task_id must be a non-empty string")
        object.__setattr__(self, "producing_task_id", self.producing_task_id.strip())

        if self.producing_turn_id is not None:
            if not isinstance(self.producing_turn_id, str) or not self.producing_turn_id.strip():
                raise ValueError("producing_turn_id must be a non-empty string or None")
            object.__setattr__(self, "producing_turn_id", self.producing_turn_id.strip())

        if not isinstance(self.artifact_type, str) or not self.artifact_type.strip():
            raise ValueError("artifact_type must be a non-empty string")
        object.__setattr__(self, "artifact_type", self.artifact_type.strip())

        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty string")
        object.__setattr__(self, "created_at", self.created_at.strip())

        if type(self.sealed) is not bool or not self.sealed:
            raise ValueError("sealed must be True for committed stage artifacts")

        if isinstance(self.content, str):
            clean_content = self.content.strip()
            if not clean_content:
                raise ValueError("content string must not be empty")
            object.__setattr__(self, "content", clean_content)
        elif isinstance(self.content, dict):
            try:
                json.dumps(self.content)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"content dict must be JSON-serializable: {exc}") from exc
            object.__setattr__(self, "content", dict(self.content))
        else:
            raise TypeError(f"content must be a dict or str, got {type(self.content).__name__}")

        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(self.metadata).__name__}")
        _check_no_secret_bearing_keys(self.metadata, "StageArtifact metadata")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def create(
        cls,
        artifact_id: str,
        producing_task_id: str,
        artifact_type: str,
        content: dict[str, Any] | str,
        producing_turn_id: str | None = None,
        artifact_version: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> "StageArtifact":
        return cls(
            artifact_id=artifact_id,
            producing_task_id=producing_task_id,
            producing_turn_id=producing_turn_id,
            artifact_type=artifact_type,
            content=content,
            created_at=datetime.now(UTC).isoformat(),
            artifact_version=artifact_version,
            sealed=True,
            metadata=dict(metadata or {}),
            version=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "producing_task_id": self.producing_task_id,
            "producing_turn_id": self.producing_turn_id,
            "artifact_type": self.artifact_type,
            "artifact_version": self.artifact_version,
            "created_at": self.created_at,
            "sealed": self.sealed,
            "content": self.content if isinstance(self.content, str) else dict(self.content),
            "metadata": dict(self.metadata),
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageArtifact":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "artifact_id",
            "producing_task_id",
            "producing_turn_id",
            "artifact_type",
            "artifact_version",
            "created_at",
            "sealed",
            "content",
            "metadata",
            "version",
        }
        unexpected_fields = data.keys() - allowed_fields
        if unexpected_fields:
            raise TypeError(f"unexpected StageArtifact fields: {sorted(unexpected_fields)}")

        if "version" in data:
            version = data["version"]
            if type(version) is bool or not isinstance(version, int):
                raise TypeError(f"version must be an int, got {type(version).__name__}")
            if version != 1:
                raise ValueError(f"Unsupported contract version: {version}")
        else:
            version = 1

        required = ("artifact_id", "producing_task_id", "artifact_type", "content", "created_at")
        for field_name in required:
            if field_name not in data:
                raise ValueError(f"{field_name} is required")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")

        return cls(
            artifact_id=data["artifact_id"],
            producing_task_id=data["producing_task_id"],
            producing_turn_id=data.get("producing_turn_id"),
            artifact_type=data["artifact_type"],
            content=data["content"],
            created_at=data["created_at"],
            artifact_version=data.get("artifact_version", 1),
            sealed=data.get("sealed", True),
            metadata=dict(metadata),
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "StageArtifact":
        return cls.from_dict(json.loads(value))
