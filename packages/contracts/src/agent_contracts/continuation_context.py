import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .delegation_wave import (
    MAX_WAVE_IDENTIFIER_BYTES,
    RESOLVED_DELEGATION_WAVE_V2_CONTRACT_VERSION,
    ResolvedDelegationWave,
)
from .results import TaskResult
from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
)
from .stage_artifact import StageArtifact
from .task_continuation import (
    CONTINUATION_V2_STATE_RELEASED,
    MAX_CONTINUATION_BYTES,
    TaskContinuation,
    TaskContinuationV2,
)
from .task_input import TaskInput
from .terminal_completion import TerminalOutcome

CONTINUATION_CONTEXT_CONTRACT_VERSION: int = 1
CONTINUATION_CONTEXT_V2_CONTRACT_VERSION: int = 2
MAX_CONTINUATION_CONTEXT_V2_BYTES: int = MAX_CONTINUATION_BYTES * 4
ALLOWED_V2_CONTEXT_WAVE_RESOLUTIONS: frozenset[str] = frozenset(
    {"SATISFIED", "UNSATISFIABLE", "TIMED_OUT"}
)


def _validate_nonempty_bounded_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str, got {type(value).__name__}")
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field_name} must not be empty")
    if len(clean.encode("utf-8")) > MAX_WAVE_IDENTIFIER_BYTES:
        raise ValueError(
            f"{field_name} exceeds maximum UTF-8 length of {MAX_WAVE_IDENTIFIER_BYTES} bytes"
        )
    return clean


@dataclass(frozen=True)
class ResumedContinuationContext:
    """Governed continuation context envelope returned by the continuation-context boundary.

    Governed by ADR-0006, ADR-0007 (Decision 10), Issue #223, and ADR-0011 (Issue #34):
    - version=1: retains authoritative parent input, v1 continuation, sealed stage
      artifact, child outcome, and (for completed child tasks) canonical child result;
    - version=2: retains authoritative parent input, v2 released continuation, sealed
      stage artifact, bound root/assignment/fence identity, and immutable v2
      ResolvedDelegationWave (SATISFIED, UNSATISFIABLE, or TIMED_OUT) without
      aggregating or exposing child result content;
    - Strictly task-scoped, immutable, and free of credentials.
    """

    parent_task_input: TaskInput
    continuation: TaskContinuation | TaskContinuationV2
    stage_artifact: StageArtifact
    child_outcome: TerminalOutcome | None = None
    child_result: TaskResult | None = None
    resolved_wave: ResolvedDelegationWave | None = None
    root_task_id: str | None = None
    assignment_id: str | None = None
    fencing_epoch: int | None = None
    version: int = CONTINUATION_CONTEXT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version not in (
            CONTINUATION_CONTEXT_CONTRACT_VERSION,
            CONTINUATION_CONTEXT_V2_CONTRACT_VERSION,
        ):
            raise ValueError(f"Unsupported contract version: {self.version}")

        if not isinstance(self.parent_task_input, TaskInput):
            raise TypeError(
                f"parent_task_input must be a TaskInput, got {type(self.parent_task_input).__name__}"
            )

        if not isinstance(self.stage_artifact, StageArtifact):
            raise TypeError(
                f"stage_artifact must be a StageArtifact, got {type(self.stage_artifact).__name__}"
            )

        if self.version == CONTINUATION_CONTEXT_V2_CONTRACT_VERSION:
            self._validate_v2()
            return

        if self.resolved_wave is not None:
            raise ValueError("resolved_wave is not supported for version=1 continuation context")
        if self.root_task_id is not None:
            raise ValueError("root_task_id is not supported for version=1 continuation context")
        if self.assignment_id is not None:
            raise ValueError("assignment_id is not supported for version=1 continuation context")
        if self.fencing_epoch is not None:
            raise ValueError("fencing_epoch is not supported for version=1 continuation context")

        if not isinstance(self.continuation, TaskContinuation):
            raise TypeError(
                f"continuation must be a TaskContinuation, got {type(self.continuation).__name__}"
            )

        if self.child_outcome is None:
            raise ValueError("child_outcome is required for version=1 continuation context")
        if not isinstance(self.child_outcome, TerminalOutcome):
            raise TypeError(
                f"child_outcome must be a TerminalOutcome, got {type(self.child_outcome).__name__}"
            )

        if self.child_result is not None and not isinstance(self.child_result, TaskResult):
            raise TypeError(
                f"child_result must be a TaskResult or None, got {type(self.child_result).__name__}"
            )

        # Relational validations
        if self.continuation.stage != "incorporate_analysis_child":
            raise ValueError(
                f"continuation.stage must be 'incorporate_analysis_child', got '{self.continuation.stage}'"
            )

        if self.continuation.continuation_version != 1:
            raise ValueError(
                f"continuation.continuation_version must be 1, got {self.continuation.continuation_version}"
            )

        if self.parent_task_input.task_id != self.continuation.parent_task_id:
            raise ValueError(
                f"parent_task_input.task_id '{self.parent_task_input.task_id}' "
                f"does not match continuation.parent_task_id '{self.continuation.parent_task_id}'"
            )

        if not self.stage_artifact.sealed:
            raise ValueError("stage_artifact must be sealed")

        if self.stage_artifact.producing_task_id != self.continuation.parent_task_id:
            raise ValueError(
                f"stage_artifact.producing_task_id '{self.stage_artifact.producing_task_id}' "
                f"does not match continuation.parent_task_id '{self.continuation.parent_task_id}'"
            )

        # Artifact reference binding:
        if not self.continuation.artifact_references:
            raise ValueError("continuation has no artifact references")

        matching_refs = [
            ref
            for ref in self.continuation.artifact_references
            if ref.producing_task_id == self.stage_artifact.producing_task_id
            and ref.artifact_id == self.stage_artifact.artifact_id
            and ref.artifact_version == self.stage_artifact.artifact_version
        ]
        if len(matching_refs) == 0:
            raise ValueError(
                f"stage_artifact '{self.stage_artifact.artifact_id}' v{self.stage_artifact.artifact_version} "
                "does not match any artifact reference in continuation"
            )
        if len(matching_refs) > 1:
            raise ValueError(
                f"stage_artifact matches multiple ({len(matching_refs)}) duplicate artifact references in continuation"
            )
        matched_ref = matching_refs[0]
        if matched_ref.producing_task_id != self.continuation.parent_task_id:
            raise ValueError(
                f"matched artifact reference producing_task_id '{matched_ref.producing_task_id}' "
                f"does not match continuation.parent_task_id '{self.continuation.parent_task_id}'"
            )

        if self.child_outcome.task_id != self.continuation.child_task_id:
            raise ValueError(
                f"child_outcome.task_id '{self.child_outcome.task_id}' "
                f"does not match continuation.child_task_id '{self.continuation.child_task_id}'"
            )

        # Relational consistency across child outcome and child result:
        child_status = self.child_outcome.terminal_status
        if child_status == "completed":
            if self.child_result is None:
                raise ValueError("child_result is required when child_outcome.terminal_status is 'completed'")
            if self.child_result.status != "completed":
                raise ValueError(
                    f"child_result.status must be 'completed', got '{self.child_result.status}'"
                )
            if self.child_result.task_id != self.continuation.child_task_id:
                raise ValueError(
                    f"child_result.task_id '{self.child_result.task_id}' "
                    f"does not match continuation.child_task_id '{self.continuation.child_task_id}'"
                )
        elif child_status in ("failed", "claim_timed_out"):
            if self.child_result is not None:
                raise ValueError(
                    f"child_result must be absent when child_outcome.terminal_status is '{child_status}'"
                )
        else:
            raise ValueError(f"Unsupported child_outcome.terminal_status: '{child_status}'")

    def _validate_v2(self) -> None:
        if self.child_outcome is not None:
            raise ValueError("child_outcome must be absent for version=2 continuation context")
        if self.child_result is not None:
            raise ValueError("child_result must be absent for version=2 continuation context")

        if not isinstance(self.continuation, TaskContinuationV2):
            raise TypeError(
                f"continuation must be a TaskContinuationV2 for version=2, got {type(self.continuation).__name__}"
            )
        if not isinstance(self.resolved_wave, ResolvedDelegationWave):
            raise TypeError(
                "resolved_wave must be a ResolvedDelegationWave for version=2"
                if self.resolved_wave is not None
                else "resolved_wave is required for version=2 continuation context"
            )

        if self.root_task_id is None:
            raise ValueError("root_task_id is required for version=2 continuation context")
        clean_root = _validate_nonempty_bounded_identifier(self.root_task_id, "root_task_id")
        object.__setattr__(self, "root_task_id", clean_root)

        if self.assignment_id is None:
            raise ValueError("assignment_id is required for version=2 continuation context")
        clean_assignment = _validate_nonempty_bounded_identifier(self.assignment_id, "assignment_id")
        object.__setattr__(self, "assignment_id", clean_assignment)

        if self.fencing_epoch is None:
            raise ValueError("fencing_epoch is required for version=2 continuation context")
        if type(self.fencing_epoch) is bool or not isinstance(self.fencing_epoch, int):
            raise TypeError(
                f"fencing_epoch must be an int, got {type(self.fencing_epoch).__name__}"
            )
        if self.fencing_epoch < 1:
            raise ValueError(f"fencing_epoch must be >= 1, got {self.fencing_epoch}")

        cont = self.continuation
        rw = self.resolved_wave
        if cont.version != 2:
            raise ValueError(f"continuation.version must be 2, got {cont.version}")
        if cont.continuation_version != 2:
            raise ValueError(f"continuation.continuation_version must be 2, got {cont.continuation_version}")
        if cont.stage != "incorporate_analysis_child":
            raise ValueError(
                f"continuation.stage must be 'incorporate_analysis_child', got '{cont.stage}'"
            )
        if cont.continuation_capability != "research":
            raise ValueError(
                f"continuation.continuation_capability must be 'research', got '{cont.continuation_capability}'"
            )
        if cont.state != CONTINUATION_V2_STATE_RELEASED:
            raise ValueError(
                f"continuation.state must be 'released', got '{cont.state}'"
            )
        if cont.wave_resolution not in ALLOWED_V2_CONTEXT_WAVE_RESOLUTIONS:
            raise ValueError(
                f"Unsupported continuation.wave_resolution for v2 resumed context: '{cont.wave_resolution}'"
            )

        if self.parent_task_input.task_id != cont.parent_task_id:
            raise ValueError(
                f"parent_task_input.task_id '{self.parent_task_input.task_id}' "
                f"does not match continuation.parent_task_id '{cont.parent_task_id}'"
            )
        if self.parent_task_input.input_version != cont.input_version:
            raise ValueError(
                f"parent_task_input.input_version {self.parent_task_input.input_version} "
                f"does not match continuation.input_version {cont.input_version}"
            )

        if rw.version != RESOLVED_DELEGATION_WAVE_V2_CONTRACT_VERSION:
            raise ValueError(
                f"resolved_wave.version must be {RESOLVED_DELEGATION_WAVE_V2_CONTRACT_VERSION}, got {rw.version}"
            )
        if rw.parent_task_id != cont.parent_task_id:
            raise ValueError(
                f"resolved_wave.parent_task_id '{rw.parent_task_id}' "
                f"does not match continuation.parent_task_id '{cont.parent_task_id}'"
            )
        if rw.wave_id != cont.wave_id:
            raise ValueError(
                f"resolved_wave.wave_id '{rw.wave_id}' does not match continuation.wave_id '{cont.wave_id}'"
            )
        if rw.required_child_task_ids != cont.required_child_task_ids:
            raise ValueError("resolved_wave.required_child_task_ids must match continuation")
        if rw.resolution != cont.wave_resolution:
            raise ValueError(
                f"resolved_wave.resolution '{rw.resolution}' "
                f"does not match continuation.wave_resolution '{cont.wave_resolution}'"
            )
        if rw.resolution not in ALLOWED_V2_CONTEXT_WAVE_RESOLUTIONS:
            raise ValueError(f"Unsupported resolved_wave.resolution '{rw.resolution}'")

        if not self.stage_artifact.sealed:
            raise ValueError("stage_artifact must be sealed")
        if self.stage_artifact.producing_task_id != cont.parent_task_id:
            raise ValueError(
                f"stage_artifact.producing_task_id '{self.stage_artifact.producing_task_id}' "
                f"does not match continuation.parent_task_id '{cont.parent_task_id}'"
            )
        if not cont.artifact_references:
            raise ValueError("continuation has no artifact references")
        matching_refs = [
            ref
            for ref in cont.artifact_references
            if ref.producing_task_id == self.stage_artifact.producing_task_id
            and ref.artifact_id == self.stage_artifact.artifact_id
            and ref.artifact_version == self.stage_artifact.artifact_version
        ]
        if len(matching_refs) != 1:
            raise ValueError(
                "stage_artifact must match exactly one artifact reference in continuation"
            )

        byte_len = len(self.to_json().encode("utf-8"))
        if byte_len > MAX_CONTINUATION_CONTEXT_V2_BYTES:
            raise ValueError(
                f"ResumedContinuationContext v2 byte size ({byte_len} bytes) exceeds maximum bound "
                f"of {MAX_CONTINUATION_CONTEXT_V2_BYTES} bytes"
            )

    def to_dict(self) -> dict[str, Any]:
        if self.version == CONTINUATION_CONTEXT_V2_CONTRACT_VERSION:
            assert isinstance(self.continuation, TaskContinuationV2)
            assert self.resolved_wave is not None
            data: dict[str, Any] = {
                "version": self.version,
                "root_task_id": self.root_task_id,
                "assignment_id": self.assignment_id,
                "fencing_epoch": self.fencing_epoch,
                "parent_task_input": self.parent_task_input.to_dict(),
                "continuation": self.continuation.to_dict(),
                "stage_artifact": self.stage_artifact.to_dict(),
                "resolved_wave": self.resolved_wave.to_dict(),
            }
            check_no_credential_bearing_keys(data, "ResumedContinuationContext")
            return data

        assert self.child_outcome is not None
        data = {
            "version": self.version,
            "parent_task_input": self.parent_task_input.to_dict(),
            "continuation": self.continuation.to_dict(),
            "stage_artifact": self.stage_artifact.to_dict(),
            "child_outcome": self.child_outcome.to_dict(),
            "child_result": self.child_result.to_dict() if self.child_result is not None else None,
        }
        check_no_credential_bearing_keys(data, "ResumedContinuationContext")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResumedContinuationContext":
        if not isinstance(data, Mapping):
            raise TypeError(f"data must be a dict, got {type(data).__name__}")

        check_no_credential_bearing_keys(data, "ResumedContinuationContext")

        raw_version = data.get("version", CONTINUATION_CONTEXT_CONTRACT_VERSION)
        if type(raw_version) is bool or not isinstance(raw_version, int):
            raise TypeError(f"version must be an int, got {type(raw_version).__name__}")

        if raw_version == CONTINUATION_CONTEXT_V2_CONTRACT_VERSION:
            allowed_v2 = {
                "version",
                "root_task_id",
                "assignment_id",
                "fencing_epoch",
                "parent_task_input",
                "continuation",
                "stage_artifact",
                "resolved_wave",
            }
            unexpected = data.keys() - allowed_v2
            if unexpected:
                raise TypeError(f"unexpected ResumedContinuationContext fields: {sorted(unexpected)}")
            for req in allowed_v2:
                if req not in data:
                    raise ValueError(f"{req} is required")

            parent_input_val = data["parent_task_input"]
            parent_task_input = (
                parent_input_val
                if isinstance(parent_input_val, TaskInput)
                else TaskInput.from_dict(parent_input_val)
            )
            cont_val = data["continuation"]
            continuation = (
                cont_val
                if isinstance(cont_val, TaskContinuationV2)
                else TaskContinuationV2.from_dict(cont_val)
            )
            stage_val = data["stage_artifact"]
            stage_artifact = (
                stage_val
                if isinstance(stage_val, StageArtifact)
                else StageArtifact.from_dict(stage_val)
            )
            rw_val = data["resolved_wave"]
            resolved_wave = (
                rw_val
                if isinstance(rw_val, ResolvedDelegationWave)
                else ResolvedDelegationWave.from_dict(rw_val)
            )
            return cls(
                parent_task_input=parent_task_input,
                continuation=continuation,
                stage_artifact=stage_artifact,
                resolved_wave=resolved_wave,
                root_task_id=data["root_task_id"],
                assignment_id=data["assignment_id"],
                fencing_epoch=data["fencing_epoch"],
                version=raw_version,
            )

        if raw_version != CONTINUATION_CONTEXT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {raw_version}")

        allowed = {
            "version",
            "parent_task_input",
            "continuation",
            "stage_artifact",
            "child_outcome",
            "child_result",
        }
        unexpected = data.keys() - allowed
        if unexpected:
            raise TypeError(f"unexpected ResumedContinuationContext fields: {sorted(unexpected)}")

        if "parent_task_input" not in data:
            raise ValueError("parent_task_input is required")
        if "continuation" not in data:
            raise ValueError("continuation is required")
        if "stage_artifact" not in data:
            raise ValueError("stage_artifact is required")
        if "child_outcome" not in data:
            raise ValueError("child_outcome is required")

        parent_input_val = data["parent_task_input"]
        parent_task_input = (
            parent_input_val
            if isinstance(parent_input_val, TaskInput)
            else TaskInput.from_dict(parent_input_val)
        )

        continuation_val = data["continuation"]
        continuation = (
            continuation_val
            if isinstance(continuation_val, TaskContinuation)
            else TaskContinuation.from_dict(continuation_val)
        )

        stage_artifact_val = data["stage_artifact"]
        stage_artifact = (
            stage_artifact_val
            if isinstance(stage_artifact_val, StageArtifact)
            else StageArtifact.from_dict(stage_artifact_val)
        )

        child_outcome_val = data["child_outcome"]
        child_outcome = (
            child_outcome_val
            if isinstance(child_outcome_val, TerminalOutcome)
            else TerminalOutcome.from_dict(child_outcome_val)
        )

        child_result_val = data.get("child_result")
        child_result: TaskResult | None = None
        if child_result_val is not None:
            child_result = (
                child_result_val
                if isinstance(child_result_val, TaskResult)
                else TaskResult.from_dict(child_result_val)
            )

        return cls(
            parent_task_input=parent_task_input,
            continuation=continuation,
            stage_artifact=stage_artifact,
            child_outcome=child_outcome,
            child_result=child_result,
            version=raw_version,
        )

    @classmethod
    def from_json(cls, value: str) -> "ResumedContinuationContext":
        if not isinstance(value, str):
            raise TypeError("value must be a JSON string")
        return cls.from_dict(json.loads(value))
