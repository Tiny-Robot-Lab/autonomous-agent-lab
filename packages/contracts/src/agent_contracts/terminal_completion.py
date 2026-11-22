"""Universal fenced completion boundary contracts (Issue #222).

Governed by ADR-0006 and ADR-0007 (Decision 11):
- Every assignment-backed execution turn (initial, delegated_child,
  resumed_continuation) commits canonical completion only through the trusted
  Orchestrator completion boundary;
- The runtime submits a stable caller-generated CompletionCommand carrying
  exactly one semantic branch (completed TaskResult or safe failure code);
- The platform persists a canonical TerminalOutcome with a deterministic
  outcome_id derived from the semantic fingerprint, plus a versioned
  AssignmentCompletionRecord audit explaining why the active claim retired.

Never persisted here or downstream: plaintext assignment proof, prompts, raw
exceptions, tracebacks, provider error bodies, credentials, or unrestricted
result content beyond the bounded canonical TaskResult.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .delegation_proposal import FORBIDDEN_AUTHORITY_FIELDS
from .errors import AllInferenceProvidersFailedError
from .results import TaskResult
from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    normalize_canonical_utc_timestamp,
)

TERMINAL_COMPLETION_CONTRACT_VERSION: int = 1

COMPLETION_OUTCOME_ID_SEPARATOR: str = "tiny-agent-lab:terminal-outcome:v1:"

COMPLETION_OUTCOME_ID_PREFIX: str = "outcome-"

MAX_COMPLETION_RESULT_BYTES: int = 65_536

MAX_COMPLETION_COMMAND_ID_LENGTH: int = 256

ALLOWED_COMPLETION_FAILURE_CODES: frozenset[str] = frozenset({
    "EXECUTION_FAILED",
    "PROVIDER_UNAVAILABLE",
    "TOOL_DENIED",
    "INVALID_INPUT",
})

SAFE_COMPLETION_SUMMARIES: dict[str, str] = {
    "EXECUTION_FAILED": "Task execution failed.",
    "PROVIDER_UNAVAILABLE": "Task execution failed because no inference provider was available.",
    "TOOL_DENIED": "Task execution failed because a required tool invocation was denied.",
    "INVALID_INPUT": "Task execution failed because the task input was invalid.",
}

COMPLETION_FORBIDDEN_AUTHORITY_FIELDS: frozenset[str] = (
    FORBIDDEN_AUTHORITY_FIELDS
    | frozenset({
        "assignment_proof",
        "bound_agent_id",
        "completed_at",
        "failure_summary",
        "fencing_epoch",
        "outcome_id",
        "proof_digest",
        "runtime_principal_id",
        "safe_summary",
        "task_id",
    })
)

RAW_FAILURE_TEXT_FIELDS: frozenset[str] = frozenset({
    "error_body",
    "error_message",
    "error_text",
    "exception",
    "exception_type",
    "failure_text",
    "provider_body",
    "provider_error",
    "raw_error",
    "raw_exception",
    "stack_trace",
    "stacktrace",
    "traceback",
})

COMPLETION_COMMAND_ALLOWED_FIELDS: frozenset[str] = frozenset({
    "assignment_id",
    "completion_command_id",
    "continuation_version",
    "execution_kind",
    "failure_code",
    "result",
    "version",
})

# Raw failure-text and secret-shaped field names that must never appear at any
# nesting level inside the completed-result subtree (including sources entries).
# Top-level command fields are already covered by unknown-field rejection, the
# RAW_FAILURE_TEXT_FIELDS check, and the recursive credential-bearing-key scan.
NESTED_FORBIDDEN_RESULT_FIELDS: frozenset[str] = (
    RAW_FAILURE_TEXT_FIELDS
    | frozenset({
        "error",
        "error_detail",
        "prompt",
        "credentials",
        "assignment_proof",
        "bearer_token",
    })
)


def _reject_forbidden_nested_fields(value: Any, context: str, _depth: int = 0) -> None:
    """Recursively reject raw failure-text and secret-shaped keys at any depth."""
    if _depth > 5:
        raise ValueError(f"{context} exceeds maximum structural nesting depth of 5")
    if isinstance(value, Mapping):
        for k, v in value.items():
            if isinstance(k, str) and k.strip() in NESTED_FORBIDDEN_RESULT_FIELDS:
                raise ValueError(f"Forbidden nested field '{k.strip()}' in {context}")
            _reject_forbidden_nested_fields(v, context, _depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden_nested_fields(item, context, _depth + 1)

ALLOWED_COMPLETION_EXECUTION_KINDS: frozenset[str] = frozenset({
    "initial",
    "delegated_child",
    "resumed_continuation",
})

ALLOWED_COMPLETION_TERMINAL_STATUSES: frozenset[str] = frozenset({
    "completed",
    "failed",
})


def classify_completion_failure(exc: BaseException) -> str:
    """Map an execution exception to a platform-owned safe failure code.

    Provider-neutral and deterministic: raw exception text never leaves the
    runtime; only the returned allowlisted code is submitted as canonical
    completion data. Unwraps runtime execution-error wrappers via the
    ``original_error`` attribute when present.

    Mapping:
    - AllInferenceProvidersFailedError -> PROVIDER_UNAVAILABLE
    - everything else -> EXECUTION_FAILED

    Only AllInferenceProvidersFailedError is an explicit repository-supported
    exhaustion signal. PermissionError, ValueError, and TypeError are ambiguous
    built-ins that the repository also raises for internal defects and
    configuration errors, and no raisable tool-denial or input-validation
    exception exists in the live taxonomy (ToolError is a non-raisable
    dataclass; DENIED is an authorization-decision outcome, not an exception),
    so ambiguous built-ins default to EXECUTION_FAILED rather than guessing a
    narrower meaning. TOOL_DENIED and INVALID_INPUT remain submittable codes
    when the caller positively establishes those semantics.

    Timeout/expiry semantics belong to issue #223; they must not be
    submitted as a distinct completion code here.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, AllInferenceProvidersFailedError):
            return "PROVIDER_UNAVAILABLE"
        current = getattr(current, "original_error", None)
        if not isinstance(current, BaseException):
            current = None
    return "EXECUTION_FAILED"


def compute_completion_fingerprint(
    *,
    assignment_id: str,
    execution_kind: str,
    continuation_version: int | None,
    completed_result: TaskResult | dict[str, Any] | None = None,
    failure_code: str | None = None,
) -> str:
    """Compute the deterministic canonical semantic completion fingerprint.

    The fingerprint binds exactly the semantic completion payload: assignment,
    execution kind, continuation version, and one success/failure branch. It
    excludes transport identity (completion_command_id) so identical semantic
    replays produce identical fingerprints and outcome IDs.
    """
    if (completed_result is None) == (failure_code is None):
        raise ValueError("Exactly one of completed_result or failure_code is required")
    if completed_result is not None:
        result = (
            completed_result
            if isinstance(completed_result, TaskResult)
            else TaskResult.from_dict(completed_result)
        )
        if result.status != "completed":
            raise ValueError("completed branch requires TaskResult with status 'completed'")
        branch: dict[str, Any] = {
            "status": "completed",
            "result": result.to_dict(),
        }
    else:
        assert failure_code is not None
        clean_code = failure_code.strip()
        if clean_code not in ALLOWED_COMPLETION_FAILURE_CODES:
            raise ValueError(f"Unsupported failure code: '{failure_code}'")
        branch = {"status": "failed", "failure_code": clean_code}
    payload = {
        "assignment_id": str(assignment_id).strip(),
        "continuation_version": continuation_version,
        "execution_kind": str(execution_kind).strip(),
        "outcome": branch,
    }
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


def derive_terminal_outcome_id(semantic_fingerprint: str) -> str:
    """Derive a stable outcome_id from the semantic fingerprint.

    Uses the ``tiny-agent-lab:terminal-outcome:v1:`` domain separator so
    outcome IDs are namespaced and cannot collide with unrelated digests.
    """
    clean = str(semantic_fingerprint).strip().lower()
    if len(clean) != 64 or any(c not in "0123456789abcdef" for c in clean):
        raise ValueError("semantic_fingerprint must be a 64-character lowercase hex digest")
    digest = hashlib.sha256(
        (COMPLETION_OUTCOME_ID_SEPARATOR + clean).encode("utf-8")
    ).hexdigest()
    return f"{COMPLETION_OUTCOME_ID_PREFIX}{digest}"


def _coerce_result(value: TaskResult | dict[str, Any]) -> TaskResult:
    if isinstance(value, TaskResult):
        return value
    if isinstance(value, dict):
        raw_output = value.get("output")
        if not isinstance(raw_output, str):
            raise TypeError(
                f"result output must be a str, got {type(raw_output).__name__}"
            )
        return TaskResult.from_dict(value)
    raise TypeError(f"result must be a TaskResult or dict, got {type(value).__name__}")


@dataclass(frozen=True)
class CompletionCommand:
    """Caller-controlled immutable completion request (contract version 1).

    Carries exactly one semantic branch: either a completed TaskResult or a
    safe allowlisted failure_code. Transport identity (completion_command_id)
    provides idempotency; it is excluded from the semantic fingerprint.
    """

    completion_command_id: str
    assignment_id: str
    execution_kind: str
    continuation_version: int | None = None
    completed_result: TaskResult | None = None
    failure_code: str | None = None
    version: int = TERMINAL_COMPLETION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != TERMINAL_COMPLETION_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in ("completion_command_id", "assignment_id", "execution_kind"):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if len(self.completion_command_id) > MAX_COMPLETION_COMMAND_ID_LENGTH:
            raise ValueError(
                "completion_command_id must not exceed "
                f"{MAX_COMPLETION_COMMAND_ID_LENGTH} characters"
            )

        if self.execution_kind not in ALLOWED_COMPLETION_EXECUTION_KINDS:
            raise ValueError(
                f"Invalid execution_kind: '{self.execution_kind}'. "
                f"Allowed: {sorted(ALLOWED_COMPLETION_EXECUTION_KINDS)}"
            )

        if self.continuation_version is not None:
            if type(self.continuation_version) is bool or not isinstance(
                self.continuation_version, int
            ):
                raise TypeError(
                    "continuation_version must be an int or None, "
                    f"got {type(self.continuation_version).__name__}"
                )
            if self.continuation_version < 1:
                raise ValueError(
                    f"continuation_version must be >= 1 when set, got {self.continuation_version}"
                )

        if self.execution_kind == "resumed_continuation":
            if self.continuation_version is None:
                raise ValueError(
                    "continuation_version is required for resumed_continuation"
                )
        elif self.continuation_version is not None:
            raise ValueError(
                f"continuation_version must not be provided for {self.execution_kind} execution_kind"
            )

        has_result = self.completed_result is not None
        has_failure = self.failure_code is not None
        if has_result == has_failure:
            raise ValueError(
                "CompletionCommand requires exactly one of completed result or failure_code"
            )

        if has_result:
            assert self.completed_result is not None
            result = self.completed_result
            if not isinstance(result, TaskResult):
                raise TypeError(
                    f"completed_result must be a TaskResult, got {type(result).__name__}"
                )
            if result.status != "completed":
                raise ValueError("completed branch requires TaskResult with status 'completed'")
            check_no_credential_bearing_keys(
                result.to_dict(), "CompletionCommand.completed_result"
            )
            _reject_forbidden_nested_fields(
                result.to_dict(), "CompletionCommand.completed_result"
            )
            serialized = canonical_json_dumps(result.to_dict())
            byte_len = len(serialized.encode("utf-8"))
            if byte_len > MAX_COMPLETION_RESULT_BYTES:
                raise ValueError(
                    f"Canonical serialized TaskResult byte size ({byte_len} bytes) exceeds "
                    f"maximum bound of {MAX_COMPLETION_RESULT_BYTES} bytes"
                )

        if has_failure:
            assert self.failure_code is not None
            if not isinstance(self.failure_code, str) or not self.failure_code.strip():
                raise ValueError("failure_code must be a non-empty string")
            clean_code = self.failure_code.strip()
            if clean_code not in ALLOWED_COMPLETION_FAILURE_CODES:
                raise ValueError(
                    f"Unsupported failure code: '{self.failure_code}'. "
                    f"Allowed: {sorted(ALLOWED_COMPLETION_FAILURE_CODES)}"
                )
            object.__setattr__(self, "failure_code", clean_code)

    def semantic_fingerprint(self) -> str:
        """Return the deterministic canonical semantic completion fingerprint."""
        return compute_completion_fingerprint(
            assignment_id=self.assignment_id,
            execution_kind=self.execution_kind,
            continuation_version=self.continuation_version,
            completed_result=self.completed_result,
            failure_code=self.failure_code,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "assignment_id": self.assignment_id,
            "completion_command_id": self.completion_command_id,
            "continuation_version": self.continuation_version,
            "execution_kind": self.execution_kind,
            "failure_code": self.failure_code,
            "result": (
                self.completed_result.to_dict() if self.completed_result is not None else None
            ),
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="CompletionCommand.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionCommand:
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        if "assignment_proof" in data:
            raise ValueError(
                "assignment_proof must not be provided in the completion JSON body; "
                "use the X-Assignment-Proof header"
            )
        forbidden = set(data.keys()) & COMPLETION_FORBIDDEN_AUTHORITY_FIELDS
        if forbidden:
            raise ValueError(
                f"Forbidden authority fields supplied by caller: {sorted(forbidden)}"
            )
        raw_text_fields = set(data.keys()) & RAW_FAILURE_TEXT_FIELDS
        if raw_text_fields:
            raise ValueError(
                f"Raw failure text fields are forbidden in CompletionCommand: "
                f"{sorted(raw_text_fields)}"
            )
        check_no_credential_bearing_keys(data, "CompletionCommand payload")
        unexpected = set(data.keys()) - COMPLETION_COMMAND_ALLOWED_FIELDS
        if unexpected:
            raise ValueError(f"Unexpected fields in CompletionCommand: {sorted(unexpected)}")

        raw_result = data.get("result")
        result: TaskResult | None = None
        if raw_result is not None:
            result = _coerce_result(raw_result)

        return cls(
            completion_command_id=data.get("completion_command_id", ""),
            assignment_id=data.get("assignment_id", ""),
            execution_kind=data.get("execution_kind", ""),
            continuation_version=data.get("continuation_version"),
            completed_result=result,
            failure_code=data.get("failure_code"),
            version=data.get("version", TERMINAL_COMPLETION_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> CompletionCommand:
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class TerminalOutcome:
    """Canonical immutable terminal outcome (contract version 1).

    The outcome_id is deterministically derived from the semantic completion
    fingerprint, so identical semantic replays resolve to the same outcome.
    Contains no proof material, prompts, exceptions, or provider bodies.
    """

    outcome_id: str
    task_id: str
    assignment_id: str
    completion_command_id: str
    semantic_fingerprint: str
    runtime_principal_id: str
    bound_agent_id: str
    execution_kind: str
    terminal_status: str
    safe_summary: str
    completed_at: str
    continuation_version: int | None = None
    failure_code: str | None = None
    version: int = TERMINAL_COMPLETION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != TERMINAL_COMPLETION_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "outcome_id",
            "task_id",
            "assignment_id",
            "completion_command_id",
            "semantic_fingerprint",
            "runtime_principal_id",
            "bound_agent_id",
            "execution_kind",
            "terminal_status",
            "safe_summary",
            "completed_at",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if self.execution_kind not in ALLOWED_COMPLETION_EXECUTION_KINDS:
            raise ValueError(
                f"Invalid execution_kind: '{self.execution_kind}'. "
                f"Allowed: {sorted(ALLOWED_COMPLETION_EXECUTION_KINDS)}"
            )

        if self.terminal_status not in ALLOWED_COMPLETION_TERMINAL_STATUSES:
            raise ValueError(
                f"Invalid terminal_status: '{self.terminal_status}'. "
                f"Allowed: {sorted(ALLOWED_COMPLETION_TERMINAL_STATUSES)}"
            )

        if self.continuation_version is not None:
            if type(self.continuation_version) is bool or not isinstance(
                self.continuation_version, int
            ):
                raise TypeError(
                    "continuation_version must be an int or None, "
                    f"got {type(self.continuation_version).__name__}"
                )
            if self.continuation_version < 1:
                raise ValueError(
                    f"continuation_version must be >= 1 when set, got {self.continuation_version}"
                )

        if self.terminal_status == "completed":
            if self.failure_code is not None:
                raise ValueError("failure_code must be absent for completed outcomes")
        else:
            if self.failure_code is None:
                raise ValueError("failure_code is required for failed outcomes")
            clean_code = self.failure_code.strip()
            if clean_code not in ALLOWED_COMPLETION_FAILURE_CODES:
                raise ValueError(f"Unsupported failure code: '{self.failure_code}'")
            object.__setattr__(self, "failure_code", clean_code)
            expected_summary = SAFE_COMPLETION_SUMMARIES[clean_code]
            if self.safe_summary != expected_summary:
                raise ValueError(
                    f"safe_summary must be the fixed platform summary for '{clean_code}'"
                )

        expected_outcome_id = derive_terminal_outcome_id(self.semantic_fingerprint)
        if self.outcome_id != expected_outcome_id:
            raise ValueError("outcome_id does not match the semantic fingerprint derivation")

        object.__setattr__(
            self, "completed_at", normalize_canonical_utc_timestamp(self.completed_at)
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "assignment_id": self.assignment_id,
            "bound_agent_id": self.bound_agent_id,
            "completed_at": self.completed_at,
            "completion_command_id": self.completion_command_id,
            "continuation_version": self.continuation_version,
            "execution_kind": self.execution_kind,
            "failure_code": self.failure_code,
            "outcome_id": self.outcome_id,
            "runtime_principal_id": self.runtime_principal_id,
            "safe_summary": self.safe_summary,
            "semantic_fingerprint": self.semantic_fingerprint,
            "task_id": self.task_id,
            "terminal_status": self.terminal_status,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="TerminalOutcome.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TerminalOutcome:
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {
            "assignment_id",
            "bound_agent_id",
            "completed_at",
            "completion_command_id",
            "continuation_version",
            "execution_kind",
            "failure_code",
            "outcome_id",
            "runtime_principal_id",
            "safe_summary",
            "semantic_fingerprint",
            "task_id",
            "terminal_status",
            "version",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise ValueError(f"Unexpected fields in TerminalOutcome: {sorted(unexpected)}")
        check_no_credential_bearing_keys(data, "TerminalOutcome payload")
        return cls(
            outcome_id=data.get("outcome_id", ""),
            task_id=data.get("task_id", ""),
            assignment_id=data.get("assignment_id", ""),
            completion_command_id=data.get("completion_command_id", ""),
            semantic_fingerprint=data.get("semantic_fingerprint", ""),
            runtime_principal_id=data.get("runtime_principal_id", ""),
            bound_agent_id=data.get("bound_agent_id", ""),
            execution_kind=data.get("execution_kind", ""),
            terminal_status=data.get("terminal_status", ""),
            safe_summary=data.get("safe_summary", ""),
            completed_at=data.get("completed_at", ""),
            continuation_version=data.get("continuation_version"),
            failure_code=data.get("failure_code"),
            version=data.get("version", TERMINAL_COMPLETION_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> TerminalOutcome:
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class CompletionRecord:
    """Durable completion-command replay record (contract version 1).

    Persisted at task.completion:{task_id}:{completion_command_id} in the
    same atomic commit as the canonical outcome. Binds the transport command
    to the semantic fingerprint, outcome, and stored proof digest so that an
    identical retry returns the recorded outcome with zero writes, while any
    changed semantic payload under the same command ID fails closed. Retired
    replays authorize against this record plus its canonical outcome.
    """

    completion_command_id: str
    task_id: str
    assignment_id: str
    runtime_principal_id: str
    bound_agent_id: str
    execution_kind: str
    semantic_fingerprint: str
    proof_digest: str
    outcome_id: str
    continuation_version: int | None = None
    version: int = TERMINAL_COMPLETION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != TERMINAL_COMPLETION_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "completion_command_id",
            "task_id",
            "assignment_id",
            "runtime_principal_id",
            "bound_agent_id",
            "execution_kind",
            "semantic_fingerprint",
            "proof_digest",
            "outcome_id",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if self.execution_kind not in ALLOWED_COMPLETION_EXECUTION_KINDS:
            raise ValueError(
                f"Invalid execution_kind: '{self.execution_kind}'. "
                f"Allowed: {sorted(ALLOWED_COMPLETION_EXECUTION_KINDS)}"
            )

        if self.continuation_version is not None:
            if type(self.continuation_version) is bool or not isinstance(
                self.continuation_version, int
            ):
                raise TypeError(
                    "continuation_version must be an int or None, "
                    f"got {type(self.continuation_version).__name__}"
                )
            if self.continuation_version < 1:
                raise ValueError(
                    f"continuation_version must be >= 1 when set, got {self.continuation_version}"
                )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "assignment_id": self.assignment_id,
            "bound_agent_id": self.bound_agent_id,
            "completion_command_id": self.completion_command_id,
            "continuation_version": self.continuation_version,
            "execution_kind": self.execution_kind,
            "outcome_id": self.outcome_id,
            "proof_digest": self.proof_digest,
            "runtime_principal_id": self.runtime_principal_id,
            "semantic_fingerprint": self.semantic_fingerprint,
            "task_id": self.task_id,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="CompletionRecord.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionRecord:
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {
            "assignment_id",
            "bound_agent_id",
            "completion_command_id",
            "continuation_version",
            "execution_kind",
            "outcome_id",
            "proof_digest",
            "runtime_principal_id",
            "semantic_fingerprint",
            "task_id",
            "version",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise ValueError(f"Unexpected fields in CompletionRecord: {sorted(unexpected)}")
        check_no_credential_bearing_keys(data, "CompletionRecord payload")
        return cls(
            completion_command_id=data.get("completion_command_id", ""),
            task_id=data.get("task_id", ""),
            assignment_id=data.get("assignment_id", ""),
            runtime_principal_id=data.get("runtime_principal_id", ""),
            bound_agent_id=data.get("bound_agent_id", ""),
            execution_kind=data.get("execution_kind", ""),
            semantic_fingerprint=data.get("semantic_fingerprint", ""),
            proof_digest=data.get("proof_digest", ""),
            outcome_id=data.get("outcome_id", ""),
            continuation_version=data.get("continuation_version"),
            version=data.get("version", TERMINAL_COMPLETION_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> CompletionRecord:
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class AssignmentCompletionRecord:
    """Versioned immutable audit explaining why an active claim retired.

    Follows the AssignmentYieldRecord pattern: persisted at
    task.assignment_completion:{task_id}:{assignment_id} in the same atomic
    commit as the canonical outcome, binding the retiring assignment to the
    accepted completion and its fencing epoch.
    """

    assignment_id: str
    task_id: str
    runtime_principal_id: str
    bound_agent_id: str
    outcome_id: str
    completion_command_id: str
    fencing_epoch: int
    execution_kind: str
    completed_at: str
    continuation_version: int | None = None
    version: int = TERMINAL_COMPLETION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is bool or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version != TERMINAL_COMPLETION_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {self.version}")

        for str_field in (
            "assignment_id",
            "task_id",
            "runtime_principal_id",
            "bound_agent_id",
            "outcome_id",
            "completion_command_id",
            "execution_kind",
            "completed_at",
        ):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
            object.__setattr__(self, str_field, val.strip())

        if self.execution_kind not in ALLOWED_COMPLETION_EXECUTION_KINDS:
            raise ValueError(
                f"Invalid execution_kind: '{self.execution_kind}'. "
                f"Allowed: {sorted(ALLOWED_COMPLETION_EXECUTION_KINDS)}"
            )

        if type(self.fencing_epoch) is bool or not isinstance(self.fencing_epoch, int):
            raise TypeError(
                f"fencing_epoch must be an int, got {type(self.fencing_epoch).__name__}"
            )
        if self.fencing_epoch < 1:
            raise ValueError(f"fencing_epoch must be >= 1, got {self.fencing_epoch}")

        if self.continuation_version is not None:
            if type(self.continuation_version) is bool or not isinstance(
                self.continuation_version, int
            ):
                raise TypeError(
                    "continuation_version must be an int or None, "
                    f"got {type(self.continuation_version).__name__}"
                )
            if self.continuation_version < 1:
                raise ValueError(
                    f"continuation_version must be >= 1 when set, got {self.continuation_version}"
                )

        object.__setattr__(
            self, "completed_at", normalize_canonical_utc_timestamp(self.completed_at)
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "assignment_id": self.assignment_id,
            "bound_agent_id": self.bound_agent_id,
            "completed_at": self.completed_at,
            "completion_command_id": self.completion_command_id,
            "continuation_version": self.continuation_version,
            "execution_kind": self.execution_kind,
            "fencing_epoch": self.fencing_epoch,
            "outcome_id": self.outcome_id,
            "runtime_principal_id": self.runtime_principal_id,
            "task_id": self.task_id,
            "version": self.version,
        }
        check_no_credential_bearing_keys(data, context="AssignmentCompletionRecord.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssignmentCompletionRecord:
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {
            "assignment_id",
            "bound_agent_id",
            "completed_at",
            "completion_command_id",
            "continuation_version",
            "execution_kind",
            "fencing_epoch",
            "outcome_id",
            "runtime_principal_id",
            "task_id",
            "version",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise ValueError(
                f"Unexpected fields in AssignmentCompletionRecord: {sorted(unexpected)}"
            )
        check_no_credential_bearing_keys(data, "AssignmentCompletionRecord payload")
        return cls(
            assignment_id=data.get("assignment_id", ""),
            task_id=data.get("task_id", ""),
            runtime_principal_id=data.get("runtime_principal_id", ""),
            bound_agent_id=data.get("bound_agent_id", ""),
            outcome_id=data.get("outcome_id", ""),
            completion_command_id=data.get("completion_command_id", ""),
            fencing_epoch=data.get("fencing_epoch", 0),
            execution_kind=data.get("execution_kind", ""),
            completed_at=data.get("completed_at", ""),
            continuation_version=data.get("continuation_version"),
            version=data.get("version", TERMINAL_COMPLETION_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> AssignmentCompletionRecord:
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class CompletionResponse:
    """Deterministic transport-neutral completion result.

    Carries the canonical TerminalOutcome, whether the request replayed an
    already-accepted command, and the canonical TaskResult for completed
    outcomes only (failed outcomes never fabricate a TaskResult).
    """

    outcome: TerminalOutcome
    replayed: bool
    result: TaskResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, TerminalOutcome):
            raise TypeError(
                f"outcome must be a TerminalOutcome, got {type(self.outcome).__name__}"
            )
        if type(self.replayed) is not bool:
            raise TypeError(f"replayed must be a bool, got {type(self.replayed).__name__}")
        if self.result is not None and not isinstance(self.result, TaskResult):
            raise TypeError(
                f"result must be a TaskResult or None, got {type(self.result).__name__}"
            )
        if self.outcome.terminal_status == "completed":
            if self.result is None:
                raise ValueError("result is required for completed outcomes")
            if self.result.status != "completed":
                raise ValueError("completed outcomes require a completed TaskResult")
        elif self.result is not None:
            raise ValueError("failed outcomes must not carry a TaskResult")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "outcome": self.outcome.to_dict(),
            "replayed": self.replayed,
        }
        if self.result is not None:
            data["result"] = self.result.to_dict()
        check_no_credential_bearing_keys(data, context="CompletionResponse.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())


def build_terminal_outcome(
    *,
    command: CompletionCommand,
    task_id: str,
    runtime_principal_id: str,
    bound_agent_id: str,
    completed_at: str,
) -> tuple[TerminalOutcome, TaskResult | None]:
    """Build the canonical TerminalOutcome (and completed TaskResult, if any).

    Pure constructor shared by every store backend so Redis and in-memory
    projections persist byte-identical canonical records.
    """
    fingerprint = command.semantic_fingerprint()
    outcome_id = derive_terminal_outcome_id(fingerprint)
    if command.completed_result is not None:
        status = "completed"
        failure_code = None
        safe_summary = "Task completed successfully."
        result = command.completed_result
    else:
        assert command.failure_code is not None
        status = "failed"
        failure_code = command.failure_code
        safe_summary = SAFE_COMPLETION_SUMMARIES[failure_code]
        result = None
    outcome = TerminalOutcome(
        outcome_id=outcome_id,
        task_id=str(task_id).strip(),
        assignment_id=command.assignment_id,
        completion_command_id=command.completion_command_id,
        semantic_fingerprint=fingerprint,
        runtime_principal_id=str(runtime_principal_id).strip(),
        bound_agent_id=str(bound_agent_id).strip(),
        execution_kind=command.execution_kind,
        terminal_status=status,
        safe_summary=safe_summary,
        completed_at=completed_at,
        continuation_version=command.continuation_version,
        failure_code=failure_code,
    )
    return outcome, result


def build_completion_audit(
    *,
    outcome: TerminalOutcome,
    fencing_epoch: int,
) -> AssignmentCompletionRecord:
    """Build the assignment-completion audit record for a canonical outcome."""
    return AssignmentCompletionRecord(
        assignment_id=outcome.assignment_id,
        task_id=outcome.task_id,
        runtime_principal_id=outcome.runtime_principal_id,
        bound_agent_id=outcome.bound_agent_id,
        outcome_id=outcome.outcome_id,
        completion_command_id=outcome.completion_command_id,
        fencing_epoch=fencing_epoch,
        execution_kind=outcome.execution_kind,
        completed_at=outcome.completed_at,
        continuation_version=outcome.continuation_version,
    )
