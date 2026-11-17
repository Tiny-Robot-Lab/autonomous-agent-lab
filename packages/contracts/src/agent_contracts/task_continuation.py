import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .task_input import _check_no_secret_bearing_keys

DEFAULT_MAX_CONTINUATION_BYTES: int = 16 * 1024  # 16 KiB provisional default

ALLOWED_CONTINUATION_STAGES: frozenset[str] = frozenset({
    "incorporate_analysis_child",
})

STAGE_CONTROL_FIELDS_ALLOWLIST: dict[str, frozenset[str]] = {
    "incorporate_analysis_child": frozenset({
        "analysis_focus",
        "child_intent",
        "material_summary",
        "min_findings",
        "max_findings",
    }),
}


@dataclass(frozen=True)
class TaskContinuation:
    """Bounded, provider-neutral record of task continuation state.

    Governed by ADR-0007:
    - Task-owned continuation checkpoint;
    - Distinct from TaskRequested and TaskResult;
    - Bounded size (16 KiB UTF-8 default, test-enforced);
    - Stage-specific allowlist of control fields;
    - Strictly references TaskInput and stage artifacts rather than duplicating payloads;
    - Forbids process memory, credentials, raw prompts, child outputs, and secret-bearing keys;
    - Versioned and write-once per continuation_version.
    """

    parent_task_id: str
    child_task_id: str
    created_at: str
    updated_at: str
    input_version: int = 1
    continuation_capability: str = "research"
    stage: str = "incorporate_analysis_child"
    control_fields: dict[str, Any] = field(default_factory=dict)
    artifact_references: tuple[str, ...] = ()
    continuation_version: int = 1
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != 1:
            raise ValueError(f"Unsupported contract version: {self.version}")

        if type(self.continuation_version) is bool or not isinstance(self.continuation_version, int):
            raise TypeError(f"continuation_version must be an int, got {type(self.continuation_version).__name__}")
        if self.continuation_version < 1:
            raise ValueError(f"continuation_version must be >= 1, got {self.continuation_version}")

        if type(self.input_version) is bool or not isinstance(self.input_version, int):
            raise TypeError(f"input_version must be an int, got {type(self.input_version).__name__}")
        if self.input_version < 1:
            raise ValueError(f"input_version must be >= 1, got {self.input_version}")

        if not isinstance(self.parent_task_id, str) or not self.parent_task_id.strip():
            raise ValueError("parent_task_id must be a non-empty string")
        clean_parent = self.parent_task_id.strip()
        object.__setattr__(self, "parent_task_id", clean_parent)

        if not isinstance(self.child_task_id, str) or not self.child_task_id.strip():
            raise ValueError("child_task_id must be a non-empty string")
        clean_child = self.child_task_id.strip()
        object.__setattr__(self, "child_task_id", clean_child)

        if clean_child == clean_parent:
            raise ValueError(
                f"child_task_id must not equal parent_task_id: '{clean_parent}'"
            )

        if not isinstance(self.continuation_capability, str) or not self.continuation_capability.strip():
            raise ValueError("continuation_capability must be a non-empty string")
        object.__setattr__(self, "continuation_capability", self.continuation_capability.strip())

        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty string")
        clean_stage = self.stage.strip()
        if clean_stage not in ALLOWED_CONTINUATION_STAGES:
            raise ValueError(
                f"Unknown continuation stage: '{clean_stage}'. Allowed stages: {sorted(ALLOWED_CONTINUATION_STAGES)}"
            )
        object.__setattr__(self, "stage", clean_stage)

        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a non-empty string")
        object.__setattr__(self, "created_at", self.created_at.strip())

        if not isinstance(self.updated_at, str) or not self.updated_at.strip():
            raise ValueError("updated_at must be a non-empty string")
        object.__setattr__(self, "updated_at", self.updated_at.strip())

        if isinstance(self.artifact_references, (list, tuple)):
            clean_refs: list[str] = []
            for ref in self.artifact_references:
                if not isinstance(ref, str) or not ref.strip():
                    raise ValueError("artifact_references elements must be non-empty strings")
                clean_refs.append(ref.strip())
            object.__setattr__(self, "artifact_references", tuple(clean_refs))
        else:
            raise TypeError(
                f"artifact_references must be a tuple or list, got {type(self.artifact_references).__name__}"
            )

        if not isinstance(self.control_fields, dict):
            raise TypeError(f"control_fields must be a dict, got {type(self.control_fields).__name__}")

        allowed_controls = STAGE_CONTROL_FIELDS_ALLOWLIST.get(clean_stage, frozenset())
        unallowlisted = self.control_fields.keys() - allowed_controls
        if unallowlisted:
            raise ValueError(
                f"Unallowlisted control fields for stage '{clean_stage}': {sorted(unallowlisted)}. "
                f"Allowed control fields: {sorted(allowed_controls)}"
            )

        _check_no_secret_bearing_keys(self.control_fields, "TaskContinuation control_fields")
        object.__setattr__(self, "control_fields", dict(self.control_fields))

        # Size bound enforcement
        serialized = self.to_json()
        byte_len = len(serialized.encode("utf-8"))
        if byte_len > DEFAULT_MAX_CONTINUATION_BYTES:
            raise ValueError(
                f"TaskContinuation byte size ({byte_len} bytes) exceeds maximum bound "
                f"of {DEFAULT_MAX_CONTINUATION_BYTES} bytes"
            )

    def byte_size(self) -> int:
        return len(self.to_json().encode("utf-8"))

    @classmethod
    def create(
        cls,
        parent_task_id: str,
        child_task_id: str,
        input_version: int = 1,
        continuation_capability: str = "research",
        stage: str = "incorporate_analysis_child",
        control_fields: dict[str, Any] | None = None,
        artifact_references: tuple[str, ...] | list[str] = (),
        continuation_version: int = 1,
    ) -> "TaskContinuation":
        now = datetime.now(UTC).isoformat()
        return cls(
            parent_task_id=parent_task_id,
            child_task_id=child_task_id,
            input_version=input_version,
            continuation_capability=continuation_capability,
            stage=stage,
            control_fields=dict(control_fields or {}),
            artifact_references=tuple(artifact_references),
            continuation_version=continuation_version,
            created_at=now,
            updated_at=now,
            version=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_task_id": self.parent_task_id,
            "child_task_id": self.child_task_id,
            "input_version": self.input_version,
            "continuation_capability": self.continuation_capability,
            "stage": self.stage,
            "control_fields": dict(self.control_fields),
            "artifact_references": list(self.artifact_references),
            "continuation_version": self.continuation_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], max_bytes: int = DEFAULT_MAX_CONTINUATION_BYTES
    ) -> "TaskContinuation":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "parent_task_id",
            "child_task_id",
            "input_version",
            "continuation_capability",
            "stage",
            "control_fields",
            "artifact_references",
            "continuation_version",
            "created_at",
            "updated_at",
            "version",
        }
        unexpected_fields = data.keys() - allowed_fields
        if unexpected_fields:
            raise TypeError(f"unexpected TaskContinuation fields: {sorted(unexpected_fields)}")

        if "version" in data:
            version = data["version"]
            if type(version) is bool or not isinstance(version, int):
                raise TypeError(f"version must be an int, got {type(version).__name__}")
            if version != 1:
                raise ValueError(f"Unsupported contract version: {version}")
        else:
            version = 1

        required = ("parent_task_id", "child_task_id", "created_at", "updated_at")
        for field_name in required:
            if field_name not in data:
                raise ValueError(f"{field_name} is required")

        raw_refs = data.get("artifact_references", ())
        if not isinstance(raw_refs, (list, tuple)):
            raise TypeError("artifact_references must be a list or tuple")

        control_fields = data.get("control_fields", {})
        if not isinstance(control_fields, dict):
            raise TypeError("control_fields must be a dict")

        record = cls(
            parent_task_id=data["parent_task_id"],
            child_task_id=data["child_task_id"],
            input_version=data.get("input_version", 1),
            continuation_capability=data.get("continuation_capability", "research"),
            stage=data.get("stage", "incorporate_analysis_child"),
            control_fields=dict(control_fields),
            artifact_references=tuple(raw_refs),
            continuation_version=data.get("continuation_version", 1),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            version=version,
        )

        if record.byte_size() > max_bytes:
            raise ValueError(
                f"TaskContinuation byte size ({record.byte_size()} bytes) exceeds maximum bound "
                f"of {max_bytes} bytes"
            )
        return record

    @classmethod
    def from_json(
        cls, value: str, max_bytes: int = DEFAULT_MAX_CONTINUATION_BYTES
    ) -> "TaskContinuation":
        byte_len = len(value.encode("utf-8"))
        if byte_len > max_bytes:
            raise ValueError(
                f"TaskContinuation JSON byte size ({byte_len} bytes) exceeds maximum bound "
                f"of {max_bytes} bytes"
            )
        return cls.from_dict(json.loads(value), max_bytes=max_bytes)
