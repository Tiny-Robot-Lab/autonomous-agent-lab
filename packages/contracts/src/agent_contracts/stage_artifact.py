import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    freeze_value,
    to_plain_json_structure,
)

MAX_STAGE_ARTIFACT_BYTES: int = 64 * 1024  # 64 KiB canonical UTF-8 limit

ALLOWED_ARTIFACT_TYPES: frozenset[str] = frozenset({
    "research_material",
    "analysis_summary",
})


@dataclass(frozen=True)
class StageArtifact:
    """Immutable, versioned, task-scoped stage artifact with provenance.

    Governed by ADR-0007:
    - Retains intermediate research or turn material across turns;
    - Keyed by (producing_task_id, artifact_id, artifact_version);
    - Sealed and write-once;
    - Bounded to 64 KiB canonical serialized UTF-8;
    - Constrained to allowlisted artifact types (research_material, analysis_summary);
    - Provenance metadata identifies producing task, turn/assignment, schema, and version;
    - Referenced by continuation and child TaskInput rather than copying content into control state.
    """

    artifact_id: str
    producing_task_id: str
    artifact_type: str
    content: Mapping[str, Any]
    created_at: str
    producing_turn_id: str | None = None
    artifact_version: int = 1
    payload_schema_version: int = 1
    sealed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != 1:
            raise ValueError(f"Unsupported contract version: {self.version}")

        if type(self.payload_schema_version) is bool or not isinstance(self.payload_schema_version, int):
            raise TypeError(f"payload_schema_version must be an int, got {type(self.payload_schema_version).__name__}")
        if self.payload_schema_version != 1:
            raise ValueError(f"Unsupported payload_schema_version: {self.payload_schema_version}")

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
        clean_type = self.artifact_type.strip()
        if clean_type not in ALLOWED_ARTIFACT_TYPES:
            raise ValueError(
                f"Unknown artifact_type: '{clean_type}'. Allowed types: {sorted(ALLOWED_ARTIFACT_TYPES)}"
            )
        object.__setattr__(self, "artifact_type", clean_type)

        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty string")
        object.__setattr__(self, "created_at", self.created_at.strip())

        if type(self.sealed) is not bool or not self.sealed:
            raise ValueError("sealed must be True for committed stage artifacts")

        if not isinstance(self.content, (dict, Mapping)):
            raise TypeError(f"content must be a dict or Mapping, got {type(self.content).__name__}")
        if not self.content:
            raise ValueError("content must not be empty")

        if clean_type == "research_material":
            if set(self.content.keys()) != {"material"}:
                raise ValueError(
                    f"research_material content must contain only 'material' field, got: {sorted(self.content.keys())}"
                )
            if not isinstance(self.content["material"], str) or not self.content["material"].strip():
                raise ValueError("research_material content 'material' must be a non-empty string")
        elif clean_type == "analysis_summary":
            if set(self.content.keys()) != {"summary"}:
                raise ValueError(
                    f"analysis_summary content must contain only 'summary' field, got: {sorted(self.content.keys())}"
                )
            if not isinstance(self.content["summary"], str) or not self.content["summary"].strip():
                raise ValueError("analysis_summary content 'summary' must be a non-empty string")

        check_no_credential_bearing_keys(self.content, "StageArtifact content")
        object.__setattr__(self, "content", freeze_value(self.content))

        if not isinstance(self.metadata, (dict, Mapping)):
            raise TypeError(f"metadata must be a dict or Mapping, got {type(self.metadata).__name__}")
        check_no_credential_bearing_keys(self.metadata, "StageArtifact metadata")
        object.__setattr__(self, "metadata", freeze_value(self.metadata))

        # Size bound enforcement
        serialized = self.to_json()
        byte_len = len(serialized.encode("utf-8"))
        if byte_len > MAX_STAGE_ARTIFACT_BYTES:
            raise ValueError(
                f"StageArtifact byte size ({byte_len} bytes) exceeds maximum allowed bound "
                f"of {MAX_STAGE_ARTIFACT_BYTES} bytes"
            )

    def byte_size(self) -> int:
        return len(self.to_json().encode("utf-8"))

    @classmethod
    def create(
        cls,
        artifact_id: str,
        producing_task_id: str,
        artifact_type: str,
        content: Mapping[str, Any],
        producing_turn_id: str | None = None,
        artifact_version: int = 1,
        payload_schema_version: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StageArtifact":
        return cls(
            artifact_id=artifact_id,
            producing_task_id=producing_task_id,
            producing_turn_id=producing_turn_id,
            artifact_type=artifact_type,
            content=content,
            created_at=datetime.now(UTC).isoformat(),
            artifact_version=artifact_version,
            payload_schema_version=payload_schema_version,
            sealed=True,
            metadata=metadata or {},
            version=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "artifact_version": self.artifact_version,
            "content": to_plain_json_structure(self.content),
            "created_at": self.created_at,
            "metadata": to_plain_json_structure(self.metadata),
            "payload_schema_version": self.payload_schema_version,
            "producing_task_id": self.producing_task_id,
            "producing_turn_id": self.producing_turn_id,
            "sealed": self.sealed,
            "version": self.version,
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageArtifact":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "artifact_id",
            "artifact_type",
            "artifact_version",
            "content",
            "created_at",
            "metadata",
            "payload_schema_version",
            "producing_task_id",
            "producing_turn_id",
            "sealed",
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

        if "payload_schema_version" in data:
            payload_schema_version = data["payload_schema_version"]
            if type(payload_schema_version) is bool or not isinstance(payload_schema_version, int):
                raise TypeError(f"payload_schema_version must be an int, got {type(payload_schema_version).__name__}")
            if payload_schema_version != 1:
                raise ValueError(f"Unsupported payload_schema_version: {payload_schema_version}")
        else:
            payload_schema_version = 1

        required = ("artifact_id", "producing_task_id", "artifact_type", "content", "created_at")
        for field_name in required:
            if field_name not in data:
                raise ValueError(f"{field_name} is required")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, (dict, Mapping)):
            raise TypeError("metadata must be a dict or Mapping")

        content = data["content"]
        if not isinstance(content, (dict, Mapping)):
            raise TypeError(f"content must be a dict or Mapping, got {type(content).__name__}")

        return cls(
            artifact_id=data["artifact_id"],
            producing_task_id=data["producing_task_id"],
            producing_turn_id=data.get("producing_turn_id"),
            artifact_type=data["artifact_type"],
            content=content,
            created_at=data["created_at"],
            artifact_version=data.get("artifact_version", 1),
            payload_schema_version=payload_schema_version,
            sealed=data.get("sealed", True),
            metadata=metadata,
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "StageArtifact":
        return cls.from_dict(json.loads(value))
