import json
from dataclasses import dataclass
from typing import Any

from .results import TaskResult
from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
)
from .stage_artifact import StageArtifact
from .task_continuation import TaskContinuation
from .task_input import TaskInput
from .terminal_completion import TerminalOutcome

CONTINUATION_CONTEXT_CONTRACT_VERSION: int = 1


@dataclass(frozen=True)
class ResumedContinuationContext:
    """Governed continuation context envelope returned by the continuation-context boundary.

    Governed by ADR-0006, ADR-0007 (Decision 10), and Issue #223:
    - Retains authoritative parent input, v1 continuation, sealed stage artifact,
      child outcome, and (for completed child tasks) the canonical child result;
    - Strictly task-scoped and validated against parent/child task identity;
    - Immutable and free of process memory, bearer tokens, and credentials.
    """

    parent_task_input: TaskInput
    continuation: TaskContinuation
    stage_artifact: StageArtifact
    child_outcome: TerminalOutcome
    child_result: TaskResult | None = None
    version: int = CONTINUATION_CONTEXT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != CONTINUATION_CONTEXT_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        if not isinstance(self.parent_task_input, TaskInput):
            raise TypeError(
                f"parent_task_input must be a TaskInput, got {type(self.parent_task_input).__name__}"
            )

        if not isinstance(self.continuation, TaskContinuation):
            raise TypeError(
                f"continuation must be a TaskContinuation, got {type(self.continuation).__name__}"
            )

        if not isinstance(self.stage_artifact, StageArtifact):
            raise TypeError(
                f"stage_artifact must be a StageArtifact, got {type(self.stage_artifact).__name__}"
            )

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

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
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
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data).__name__}")
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

        check_no_credential_bearing_keys(data, "ResumedContinuationContext")

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

        raw_version = data.get("version", CONTINUATION_CONTEXT_CONTRACT_VERSION)
        if type(raw_version) is bool or not isinstance(raw_version, int):
            raise TypeError(f"version must be an int, got {type(raw_version).__name__}")

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
