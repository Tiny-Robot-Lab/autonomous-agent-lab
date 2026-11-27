import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .artifact_reference import ArtifactReference
from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    freeze_value,
    normalize_canonical_utc_timestamp,
    to_plain_json_structure,
)

MAX_CONTINUATION_BYTES: int = 16 * 1024  # 16 KiB fixed canonical UTF-8 limit

TASK_CONTINUATION_V2_CONTRACT_VERSION: int = 2

CONTINUATION_V2_STATE_ACTIVE: str = "active"
CONTINUATION_V2_STATE_RELEASED: str = "released"
CONTINUATION_V2_STATE_CLOSED: str = "closed"

ALLOWED_CONTINUATION_V2_STATES: frozenset[str] = frozenset({
    CONTINUATION_V2_STATE_ACTIVE,
    CONTINUATION_V2_STATE_RELEASED,
    CONTINUATION_V2_STATE_CLOSED,
})

RELEASED_CONTINUATION_RESOLUTIONS: frozenset[str] = frozenset({
    "SATISFIED",
    "UNSATISFIABLE",
    "TIMED_OUT",
})

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


def _validate_int_at_least(value: Any, field_name: str, minimum: int = 1) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _validate_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_stage_and_controls(
    stage: Any,
    control_fields: Any,
    context: str,
) -> tuple[str, Any]:
    clean_stage = _validate_nonempty_str(stage, "stage")
    if clean_stage not in ALLOWED_CONTINUATION_STAGES:
        raise ValueError(
            f"Unknown continuation stage: '{clean_stage}'. Allowed stages: {sorted(ALLOWED_CONTINUATION_STAGES)}"
        )

    if not isinstance(control_fields, (dict, Mapping)):
        raise TypeError(f"control_fields must be a dict or Mapping, got {type(control_fields).__name__}")

    allowed_controls = STAGE_CONTROL_FIELDS_ALLOWLIST.get(clean_stage, frozenset())
    unallowlisted = control_fields.keys() - allowed_controls
    if unallowlisted:
        raise ValueError(
            f"Unallowlisted control fields for stage '{clean_stage}': {sorted(unallowlisted)}. "
            f"Allowed control fields: {sorted(allowed_controls)}"
        )

    check_no_credential_bearing_keys(control_fields, context)
    return clean_stage, freeze_value(control_fields)


def _validate_artifact_references(artifact_references: Any) -> tuple[ArtifactReference, ...]:
    if not isinstance(artifact_references, (list, tuple)):
        raise TypeError(
            f"artifact_references must be a tuple or list, got {type(artifact_references).__name__}"
        )
    clean_refs: list[ArtifactReference] = []
    for ref in artifact_references:
        if isinstance(ref, ArtifactReference):
            parsed_ref = ref
        elif isinstance(ref, dict):
            parsed_ref = ArtifactReference.from_dict(ref)
        else:
            raise TypeError(
                f"artifact_references elements must be ArtifactReference or dict, got {type(ref).__name__}"
            )
        clean_refs.append(parsed_ref)
    return tuple(clean_refs)


def _normalize_required_child_ids(
    raw_ids: Any,
    parent_task_id: str,
) -> tuple[str, ...]:
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
        raise TypeError(
            f"required_child_task_ids must be a sequence of strings, got {type(raw_ids).__name__}"
        )
    if len(raw_ids) == 0:
        raise ValueError("required_child_task_ids must not be empty")

    seen: set[str] = set()
    cleaned: list[str] = []
    for idx, child_id in enumerate(raw_ids):
        if not isinstance(child_id, str) or not child_id.strip():
            raise ValueError(f"required_child_task_ids[{idx}] must be a non-empty string")
        clean_child = child_id.strip()
        if clean_child == parent_task_id:
            raise ValueError(
                f"required_child_task_ids must not contain parent_task_id: '{parent_task_id}'"
            )
        if clean_child in seen:
            raise ValueError(f"Duplicate child task ID in required_child_task_ids: '{clean_child}'")
        seen.add(clean_child)
        cleaned.append(clean_child)

    # Canonical sorted order for stable serialization and semantic identity
    return tuple(sorted(cleaned))


@dataclass(frozen=True)
class TaskContinuation:
    """Bounded, provider-neutral record of task continuation state.

    Governed by ADR-0007:
    - Task-owned continuation checkpoint;
    - Distinct from TaskRequested and TaskResult;
    - Bounded to fixed 16 KiB canonical UTF-8;
    - Stage-specific allowlist of control fields;
    - Strictly references TaskInput and stage artifacts rather than duplicating payloads;
    - ArtifactReference validates structure only; contextual authorization belongs to later acceptance policy;
    - Forbids process memory, credentials, raw prompts, child outputs, and credential-bearing keys;
    - Versioned and write-once per continuation_version.
    """

    parent_task_id: str
    child_task_id: str
    created_at: str
    updated_at: str
    input_version: int = 1
    continuation_capability: str = "research"
    stage: str = "incorporate_analysis_child"
    control_fields: Mapping[str, Any] = field(default_factory=dict)
    artifact_references: tuple[ArtifactReference, ...] = ()
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
            clean_refs: list[ArtifactReference] = []
            for ref in self.artifact_references:
                if isinstance(ref, ArtifactReference):
                    parsed_ref = ref
                elif isinstance(ref, dict):
                    parsed_ref = ArtifactReference.from_dict(ref)
                else:
                    raise TypeError(
                        f"artifact_references elements must be ArtifactReference or dict, got {type(ref).__name__}"
                    )
                clean_refs.append(parsed_ref)
            object.__setattr__(self, "artifact_references", tuple(clean_refs))
        else:
            raise TypeError(
                f"artifact_references must be a tuple or list, got {type(self.artifact_references).__name__}"
            )

        if not isinstance(self.control_fields, (dict, Mapping)):
            raise TypeError(f"control_fields must be a dict or Mapping, got {type(self.control_fields).__name__}")

        allowed_controls = STAGE_CONTROL_FIELDS_ALLOWLIST.get(clean_stage, frozenset())
        unallowlisted = self.control_fields.keys() - allowed_controls
        if unallowlisted:
            raise ValueError(
                f"Unallowlisted control fields for stage '{clean_stage}': {sorted(unallowlisted)}. "
                f"Allowed control fields: {sorted(allowed_controls)}"
            )

        check_no_credential_bearing_keys(self.control_fields, "TaskContinuation control_fields")
        object.__setattr__(self, "control_fields", freeze_value(self.control_fields))

        # Fixed 16 KiB size bound enforcement
        serialized = self.to_json()
        byte_len = len(serialized.encode("utf-8"))
        if byte_len > MAX_CONTINUATION_BYTES:
            raise ValueError(
                f"TaskContinuation byte size ({byte_len} bytes) exceeds maximum bound "
                f"of {MAX_CONTINUATION_BYTES} bytes"
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
        control_fields: Mapping[str, Any] | None = None,
        artifact_references: tuple[ArtifactReference, ...] | list[Any] = (),
        continuation_version: int = 1,
    ) -> "TaskContinuation":
        now = datetime.now(UTC).isoformat()
        return cls(
            parent_task_id=parent_task_id,
            child_task_id=child_task_id,
            input_version=input_version,
            continuation_capability=continuation_capability,
            stage=stage,
            control_fields=control_fields or {},
            artifact_references=tuple(artifact_references),
            continuation_version=continuation_version,
            created_at=now,
            updated_at=now,
            version=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_references": [ref.to_dict() for ref in self.artifact_references],
            "child_task_id": self.child_task_id,
            "continuation_capability": self.continuation_capability,
            "continuation_version": self.continuation_version,
            "control_fields": to_plain_json_structure(self.control_fields),
            "created_at": self.created_at,
            "input_version": self.input_version,
            "parent_task_id": self.parent_task_id,
            "stage": self.stage,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskContinuation":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "artifact_references",
            "child_task_id",
            "continuation_capability",
            "continuation_version",
            "control_fields",
            "created_at",
            "input_version",
            "parent_task_id",
            "stage",
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
        if not isinstance(control_fields, (dict, Mapping)):
            raise TypeError("control_fields must be a dict or Mapping")

        return cls(
            parent_task_id=data["parent_task_id"],
            child_task_id=data["child_task_id"],
            input_version=data.get("input_version", 1),
            continuation_capability=data.get("continuation_capability", "research"),
            stage=data.get("stage", "incorporate_analysis_child"),
            control_fields=control_fields,
            artifact_references=tuple(raw_refs),
            continuation_version=data.get("continuation_version", 1),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "TaskContinuation":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class TaskContinuationV2:
    """Version-2 task-owned continuation contract for bounded multi-child delegation waves.

    Governed by ADR-0007 and ADR-0011:
    - Preserves TaskContinuation (v1) unchanged for single-child compatibility;
    - Binds parent_task_id to authoritative wave_id and canonical immutable required_child_task_ids;
    - Does not embed mutable barrier counters or inferred child status;
    - Exposes canonical wave_resolution ('SATISFIED', 'UNSATISFIABLE', 'TIMED_OUT') only when state == 'released';
    - Supports closed/non-claimable state ('closed') after CANCELLED without a resumable resolution;
    - Bounded to fixed 16 KiB canonical UTF-8.
    """

    parent_task_id: str
    wave_id: str
    required_child_task_ids: tuple[str, ...]
    created_at: str
    updated_at: str
    input_version: int = 1
    continuation_capability: str = "research"
    stage: str = "incorporate_analysis_child"
    control_fields: Mapping[str, Any] = field(default_factory=dict)
    artifact_references: tuple[ArtifactReference, ...] = ()
    continuation_version: int = 1
    state: str = CONTINUATION_V2_STATE_ACTIVE
    wave_resolution: str | None = None
    version: int = TASK_CONTINUATION_V2_CONTRACT_VERSION

    def __post_init__(self) -> None:
        ver = _validate_int_at_least(self.version, "version", minimum=1)
        if ver != TASK_CONTINUATION_V2_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {ver}")

        _validate_int_at_least(self.continuation_version, "continuation_version", minimum=1)
        _validate_int_at_least(self.input_version, "input_version", minimum=1)

        clean_parent = _validate_nonempty_str(self.parent_task_id, "parent_task_id")
        clean_wave = _validate_nonempty_str(self.wave_id, "wave_id")
        clean_cap = _validate_nonempty_str(self.continuation_capability, "continuation_capability")
        object.__setattr__(self, "parent_task_id", clean_parent)
        object.__setattr__(self, "wave_id", clean_wave)
        object.__setattr__(self, "continuation_capability", clean_cap)

        clean_children = _normalize_required_child_ids(
            self.required_child_task_ids, clean_parent
        )
        object.__setattr__(self, "required_child_task_ids", clean_children)

        clean_stage, frozen_controls = _validate_stage_and_controls(
            self.stage, self.control_fields, "TaskContinuationV2 control_fields"
        )
        object.__setattr__(self, "stage", clean_stage)
        object.__setattr__(self, "control_fields", frozen_controls)

        clean_refs = _validate_artifact_references(self.artifact_references)
        object.__setattr__(self, "artifact_references", clean_refs)

        object.__setattr__(
            self, "created_at", normalize_canonical_utc_timestamp(self.created_at)
        )
        object.__setattr__(
            self, "updated_at", normalize_canonical_utc_timestamp(self.updated_at)
        )

        clean_state = _validate_nonempty_str(self.state, "state")
        if clean_state not in ALLOWED_CONTINUATION_V2_STATES:
            raise ValueError(
                f"Invalid continuation state '{clean_state}'. "
                f"Allowed: {sorted(ALLOWED_CONTINUATION_V2_STATES)}"
            )
        object.__setattr__(self, "state", clean_state)

        clean_resolution: str | None
        if self.wave_resolution is None:
            clean_resolution = None
        else:
            clean_resolution = _validate_nonempty_str(self.wave_resolution, "wave_resolution")
            if clean_resolution == "CANCELLED":
                raise ValueError(
                    "CANCELLED is not permitted as a resumable released wave_resolution"
                )
            if clean_resolution not in RELEASED_CONTINUATION_RESOLUTIONS:
                raise ValueError(
                    f"Invalid wave_resolution '{clean_resolution}'. "
                    f"Allowed: {sorted(RELEASED_CONTINUATION_RESOLUTIONS)}"
                )
        object.__setattr__(self, "wave_resolution", clean_resolution)

        if clean_state == CONTINUATION_V2_STATE_ACTIVE and clean_resolution is not None:
            raise ValueError(
                "Active continuation must not expose wave_resolution before release"
            )
        if clean_state == CONTINUATION_V2_STATE_RELEASED and clean_resolution is None:
            raise ValueError(
                "Released continuation must expose a canonical wave_resolution"
            )
        if clean_state == CONTINUATION_V2_STATE_CLOSED and clean_resolution is not None:
            raise ValueError(
                "Closed continuation must not expose a resumable wave_resolution"
            )

        serialized = self.to_json()
        byte_len = len(serialized.encode("utf-8"))
        if byte_len > MAX_CONTINUATION_BYTES:
            raise ValueError(
                f"TaskContinuationV2 byte size ({byte_len} bytes) exceeds maximum bound "
                f"of {MAX_CONTINUATION_BYTES} bytes"
            )

    @property
    def is_claimable(self) -> bool:
        return self.state == CONTINUATION_V2_STATE_RELEASED

    @property
    def is_closed(self) -> bool:
        return self.state == CONTINUATION_V2_STATE_CLOSED

    def byte_size(self) -> int:
        return len(self.to_json().encode("utf-8"))

    def semantic_fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        parent_task_id: str,
        wave_id: str,
        required_child_task_ids: Sequence[str],
        input_version: int = 1,
        continuation_capability: str = "research",
        stage: str = "incorporate_analysis_child",
        control_fields: Mapping[str, Any] | None = None,
        artifact_references: tuple[ArtifactReference, ...] | list[Any] = (),
        continuation_version: int = 1,
        state: str = CONTINUATION_V2_STATE_ACTIVE,
        wave_resolution: str | None = None,
    ) -> "TaskContinuationV2":
        now = normalize_canonical_utc_timestamp(datetime.now(UTC))
        return cls(
            parent_task_id=parent_task_id,
            wave_id=wave_id,
            required_child_task_ids=tuple(required_child_task_ids),
            input_version=input_version,
            continuation_capability=continuation_capability,
            stage=stage,
            control_fields=control_fields or {},
            artifact_references=tuple(artifact_references),
            continuation_version=continuation_version,
            state=state,
            wave_resolution=wave_resolution,
            created_at=now,
            updated_at=now,
            version=TASK_CONTINUATION_V2_CONTRACT_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_references": [ref.to_dict() for ref in self.artifact_references],
            "continuation_capability": self.continuation_capability,
            "continuation_version": self.continuation_version,
            "control_fields": to_plain_json_structure(self.control_fields),
            "created_at": self.created_at,
            "input_version": self.input_version,
            "parent_task_id": self.parent_task_id,
            "required_child_task_ids": list(self.required_child_task_ids),
            "stage": self.stage,
            "state": self.state,
            "updated_at": self.updated_at,
            "version": self.version,
            "wave_id": self.wave_id,
            "wave_resolution": self.wave_resolution,
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskContinuationV2":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        allowed_fields = {
            "artifact_references",
            "continuation_capability",
            "continuation_version",
            "control_fields",
            "created_at",
            "input_version",
            "parent_task_id",
            "required_child_task_ids",
            "stage",
            "state",
            "updated_at",
            "version",
            "wave_id",
            "wave_resolution",
        }
        unexpected_fields = data.keys() - allowed_fields
        if unexpected_fields:
            raise TypeError(f"unexpected TaskContinuationV2 fields: {sorted(unexpected_fields)}")

        if "version" in data:
            version = _validate_int_at_least(data["version"], "version", minimum=1)
            if version != TASK_CONTINUATION_V2_CONTRACT_VERSION:
                raise ValueError(f"Unsupported contract version: {version}")
        else:
            version = TASK_CONTINUATION_V2_CONTRACT_VERSION

        required = (
            "parent_task_id",
            "wave_id",
            "required_child_task_ids",
            "created_at",
            "updated_at",
        )
        for field_name in required:
            if field_name not in data:
                raise ValueError(f"{field_name} is required")

        raw_refs = data.get("artifact_references", ())
        if not isinstance(raw_refs, (list, tuple)):
            raise TypeError("artifact_references must be a list or tuple")

        control_fields = data.get("control_fields", {})
        if not isinstance(control_fields, (dict, Mapping)):
            raise TypeError("control_fields must be a dict or Mapping")

        return cls(
            parent_task_id=data["parent_task_id"],
            wave_id=data["wave_id"],
            required_child_task_ids=data["required_child_task_ids"],
            input_version=data.get("input_version", 1),
            continuation_capability=data.get("continuation_capability", "research"),
            stage=data.get("stage", "incorporate_analysis_child"),
            control_fields=control_fields,
            artifact_references=tuple(raw_refs),
            continuation_version=data.get("continuation_version", 1),
            state=data.get("state", CONTINUATION_V2_STATE_ACTIVE),
            wave_resolution=data.get("wave_resolution"),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "TaskContinuationV2":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


def deserialize_continuation(
    data: dict[str, Any] | str,
    *,
    expected_continuation_version: int | None = None,
) -> TaskContinuation | TaskContinuationV2:
    """Version-aware fail-closed deserialization for TaskContinuation v1 and v2."""
    parsed = json.loads(data) if isinstance(data, str) else data
    if not isinstance(parsed, dict):
        raise TypeError("data must be a dict")

    if "version" not in parsed:
        raise ValueError("version is required")
    version = parsed["version"]
    if type(version) is bool or not isinstance(version, int):
        raise TypeError(f"version must be an int, got {type(version).__name__}")

    continuation: TaskContinuation | TaskContinuationV2
    if version == 1:
        continuation = TaskContinuation.from_dict(parsed)
    elif version == TASK_CONTINUATION_V2_CONTRACT_VERSION:
        continuation = TaskContinuationV2.from_dict(parsed)
    else:
        raise ValueError(f"Unsupported contract version: {version}")

    if expected_continuation_version is not None:
        if (
            type(expected_continuation_version) is bool
            or not isinstance(expected_continuation_version, int)
        ):
            raise TypeError("expected_continuation_version must be an int")
        if continuation.continuation_version != expected_continuation_version:
            raise ValueError(
                f"Continuation version mismatch: expected {expected_continuation_version}, "
                f"got {continuation.continuation_version}"
            )
    return continuation
