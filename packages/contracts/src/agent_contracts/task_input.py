import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .artifact_reference import ArtifactReference
from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    freeze_value,
    to_plain_json_structure,
)


@dataclass(frozen=True)
class TaskInputReference:
    """Explicit, versioned reference to an immutable TaskInput record.

    Governed by ADR-0007 and Issue #220:
    - References durable TaskInput by task_id and input_version;
    - Immutable and canonical.
    """

    task_id: str
    input_version: int = 1

    def __post_init__(self) -> None:
        if type(self.input_version) is bool or not isinstance(self.input_version, int):
            raise TypeError(f"input_version must be an int, got {type(self.input_version).__name__}")
        if self.input_version < 1:
            raise ValueError(f"input_version must be >= 1, got {self.input_version}")

        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        object.__setattr__(self, "task_id", self.task_id.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_version": self.input_version,
            "task_id": self.task_id,
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInputReference":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {"task_id", "input_version"}
        unexpected = data.keys() - allowed
        if unexpected:
            raise TypeError(f"unexpected TaskInputReference fields: {sorted(unexpected)}")
        if "task_id" not in data:
            raise ValueError("task_id is required")
        return cls(
            task_id=data["task_id"],
            input_version=data.get("input_version", 1),
        )

    @classmethod
    def from_json(cls, value: str) -> "TaskInputReference":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class TaskInput:
    """Immutable, versioned record of task input.

    Governed by ADR-0007:
    - Retains durable access to original task input independently of transient queue payloads;
    - Keyed by (task_id, input_version);
    - Write-once immutability per version;
    - May reference immutable task-scoped stage artifacts.
    """

    task_id: str
    prompt: str
    created_at: str
    input_version: int = 1
    artifact_references: tuple[ArtifactReference, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != 1:
            raise ValueError(f"Unsupported contract version: {self.version}")

        if type(self.input_version) is bool or not isinstance(self.input_version, int):
            raise TypeError(f"input_version must be an int, got {type(self.input_version).__name__}")
        if self.input_version < 1:
            raise ValueError(f"input_version must be >= 1, got {self.input_version}")

        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        object.__setattr__(self, "task_id", self.task_id.strip())

        if not isinstance(self.prompt, str):
            raise TypeError(f"prompt must be a str, got {type(self.prompt).__name__}")
        clean_prompt = self.prompt.strip()
        if not clean_prompt:
            raise ValueError("prompt must not be empty")
        if len(self.prompt) > 2000 or len(clean_prompt) > 2000:
            raise ValueError("prompt must not exceed 2000 characters")
        object.__setattr__(self, "prompt", clean_prompt)

        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty string")
        object.__setattr__(self, "created_at", self.created_at.strip())

        if isinstance(self.artifact_references, (list, tuple)):
            clean_refs: list[ArtifactReference] = []
            for ref in self.artifact_references:
                if isinstance(ref, ArtifactReference):
                    clean_refs.append(ref)
                elif isinstance(ref, dict):
                    clean_refs.append(ArtifactReference.from_dict(ref))
                else:
                    raise TypeError(
                        f"artifact_references elements must be ArtifactReference or dict, got {type(ref).__name__}"
                    )
            object.__setattr__(self, "artifact_references", tuple(clean_refs))
        else:
            raise TypeError(
                f"artifact_references must be a tuple or list, got {type(self.artifact_references).__name__}"
            )

        if not isinstance(self.metadata, (dict, Mapping)):
            raise TypeError(f"metadata must be a dict or Mapping, got {type(self.metadata).__name__}")
        check_no_credential_bearing_keys(self.metadata, "TaskInput metadata")
        object.__setattr__(self, "metadata", freeze_value(self.metadata))

    @classmethod
    def create(
        cls,
        task_id: str,
        prompt: str,
        input_version: int = 1,
        artifact_references: tuple[ArtifactReference, ...] | list[Any] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "TaskInput":
        return cls(
            task_id=task_id,
            prompt=prompt,
            created_at=datetime.now(UTC).isoformat(),
            input_version=input_version,
            artifact_references=tuple(artifact_references),
            metadata=metadata or {},
            version=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_references": [ref.to_dict() for ref in self.artifact_references],
            "created_at": self.created_at,
            "input_version": self.input_version,
            "metadata": to_plain_json_structure(self.metadata),
            "prompt": self.prompt,
            "task_id": self.task_id,
            "version": self.version,
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInput":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "artifact_references",
            "created_at",
            "input_version",
            "metadata",
            "prompt",
            "task_id",
            "version",
        }
        unexpected_fields = data.keys() - allowed_fields
        if unexpected_fields:
            raise TypeError(f"unexpected TaskInput fields: {sorted(unexpected_fields)}")

        if "version" in data:
            version = data["version"]
            if type(version) is bool or not isinstance(version, int):
                raise TypeError(f"version must be an int, got {type(version).__name__}")
            if version != 1:
                raise ValueError(f"Unsupported contract version: {version}")
        else:
            version = 1

        if "task_id" not in data:
            raise ValueError("task_id is required")
        if "prompt" not in data:
            raise ValueError("prompt is required")
        if "created_at" not in data:
            raise ValueError("created_at is required")

        raw_refs = data.get("artifact_references", ())
        if not isinstance(raw_refs, (list, tuple)):
            raise TypeError("artifact_references must be a list or tuple")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, (dict, Mapping)):
            raise TypeError("metadata must be a dict or Mapping")

        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            created_at=data["created_at"],
            input_version=data.get("input_version", 1),
            artifact_references=tuple(raw_refs),
            metadata=metadata,
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "TaskInput":
        return cls.from_dict(json.loads(value))
