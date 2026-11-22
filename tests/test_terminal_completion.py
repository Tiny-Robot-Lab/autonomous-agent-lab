"""Contract tests for the universal fenced completion boundary (Issue #222).

Covers CompletionCommand v1, TerminalOutcome v1, CompletionResponse,
AssignmentCompletionRecord, the safe failure-code allowlist, the 65,536-byte
canonical result bound, deterministic fingerprint/outcome-ID derivation, and
rejection of proofs, authority fields, raw failure text, and secret-shaped
keys in the ordinary JSON contract.
"""

import dataclasses
import hashlib

import pytest
from agent_contracts import (
    ALLOWED_COMPLETION_FAILURE_CODES,
    COMPLETION_OUTCOME_ID_PREFIX,
    COMPLETION_OUTCOME_ID_SEPARATOR,
    MAX_COMPLETION_COMMAND_ID_LENGTH,
    MAX_COMPLETION_RESULT_BYTES,
    SAFE_COMPLETION_SUMMARIES,
    TERMINAL_COMPLETION_CONTRACT_VERSION,
    AllInferenceProvidersFailedError,
    AssignmentCompletionRecord,
    CompletionCommand,
    CompletionResponse,
    ProviderFailureSummary,
    TaskResult,
    TerminalOutcome,
    build_completion_audit,
    build_terminal_outcome,
    canonical_json_dumps,
    classify_completion_failure,
    compute_completion_fingerprint,
    derive_terminal_outcome_id,
)


def _completed_result(output: str = "done", task_id: str = "task-1") -> TaskResult:
    return TaskResult(task_id=task_id, output=output)


def _success_command(**overrides: object) -> CompletionCommand:
    fields: dict[str, object] = {
        "completion_command_id": "cmd-1",
        "assignment_id": "assign-1",
        "execution_kind": "initial",
        "completed_result": _completed_result(),
    }
    fields.update(overrides)
    return CompletionCommand(**fields)  # type: ignore[arg-type]


def _failure_command(**overrides: object) -> CompletionCommand:
    fields: dict[str, object] = {
        "completion_command_id": "cmd-1",
        "assignment_id": "assign-1",
        "execution_kind": "initial",
        "failure_code": "EXECUTION_FAILED",
    }
    fields.update(overrides)
    return CompletionCommand(**fields)  # type: ignore[arg-type]


def _terminal_outcome(command: CompletionCommand) -> tuple[TerminalOutcome, TaskResult | None]:
    return build_terminal_outcome(
        command=command,
        task_id="task-1",
        runtime_principal_id="principal-1",
        bound_agent_id="agent-1",
        completed_at="2026-11-20T00:00:00.000000Z",
    )


# --- Success / failure branches -------------------------------------------------


def test_success_command_requires_completed_task_result() -> None:
    cmd = _success_command()
    assert cmd.completed_result is not None
    assert cmd.completed_result.status == "completed"
    assert cmd.failure_code is None
    assert cmd.version == TERMINAL_COMPLETION_CONTRACT_VERSION


def test_failure_command_for_each_approved_code() -> None:
    for code in sorted(ALLOWED_COMPLETION_FAILURE_CODES):
        cmd = _failure_command(failure_code=code)
        assert cmd.failure_code == code
        assert cmd.completed_result is None


def test_exactly_one_branch_both_missing_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        CompletionCommand(
            completion_command_id="cmd-1",
            assignment_id="assign-1",
            execution_kind="initial",
        )


def test_exactly_one_branch_both_present_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        CompletionCommand(
            completion_command_id="cmd-1",
            assignment_id="assign-1",
            execution_kind="initial",
            completed_result=_completed_result(),
            failure_code="EXECUTION_FAILED",
        )


def test_completed_branch_rejects_non_completed_task_result() -> None:
    bad = TaskResult(task_id="task-1", output="x", status="failed")
    with pytest.raises(ValueError, match="status 'completed'"):
        _success_command(completed_result=bad)


def test_rejected_failure_codes() -> None:
    for code in ("DEADLINE_EXCEEDED", "execution_failed", "", "TIMEOUT", "UNKNOWN"):
        with pytest.raises(ValueError, match="[Uu]nsupported failure code|non-empty"):
            _failure_command(failure_code=code)


def test_from_dict_rejects_failure_code_case_variants() -> None:
    payload = {
        "completion_command_id": "cmd-1",
        "assignment_id": "assign-1",
        "execution_kind": "initial",
        "failure_code": "execution_failed",
    }
    with pytest.raises(ValueError, match="[Uu]nsupported failure code"):
        CompletionCommand.from_dict(payload)


# --- Canonical byte bound -------------------------------------------------------


def _result_at_canonical_size(target_bytes: int) -> TaskResult:
    overhead = len(
        canonical_json_dumps(_completed_result(output="").to_dict()).encode("utf-8")
    )
    padding = target_bytes - overhead
    assert padding >= 0
    return _completed_result(output="x" * padding)


def test_result_bound_measures_canonical_serialized_bytes() -> None:
    result = _result_at_canonical_size(MAX_COMPLETION_RESULT_BYTES)
    canonical = canonical_json_dumps(result.to_dict())
    assert len(canonical.encode("utf-8")) == MAX_COMPLETION_RESULT_BYTES
    # Python character count alone is not the measure; assert the byte measure path.
    assert len(canonical) == MAX_COMPLETION_RESULT_BYTES
    _success_command(completed_result=result)


@pytest.mark.parametrize("delta", (-1, 0))
def test_result_bound_accepts_at_or_below_limit(delta: int) -> None:
    result = _result_at_canonical_size(MAX_COMPLETION_RESULT_BYTES + delta)
    _success_command(completed_result=result)


def test_result_bound_rejects_single_byte_over() -> None:
    result = _result_at_canonical_size(MAX_COMPLETION_RESULT_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds maximum bound"):
        _success_command(completed_result=result)


def test_from_dict_enforces_result_byte_bound() -> None:
    result = _result_at_canonical_size(MAX_COMPLETION_RESULT_BYTES + 1)
    payload = {
        "completion_command_id": "cmd-1",
        "assignment_id": "assign-1",
        "execution_kind": "initial",
        "result": result.to_dict(),
    }
    with pytest.raises(ValueError, match="exceeds maximum bound"):
        CompletionCommand.from_dict(payload)


# --- Rejections ------------------------------------------------------------------


def test_assignment_proof_rejected_in_json_body() -> None:
    payload = {
        "completion_command_id": "cmd-1",
        "assignment_id": "assign-1",
        "execution_kind": "initial",
        "failure_code": "EXECUTION_FAILED",
        "assignment_proof": "deadbeef",
    }
    with pytest.raises(ValueError, match="X-Assignment-Proof"):
        CompletionCommand.from_dict(payload)


def test_caller_supplied_authority_fields_rejected() -> None:
    base = {
        "completion_command_id": "cmd-1",
        "assignment_id": "assign-1",
        "execution_kind": "initial",
        "failure_code": "EXECUTION_FAILED",
    }
    for field in (
        "outcome_id",
        "proof_digest",
        "fencing_epoch",
        "runtime_principal_id",
        "bound_agent_id",
        "task_id",
        "completed_at",
        "safe_summary",
        "child_task_id",
        "delegation_grant_id",
    ):
        payload = dict(base)
        payload[field] = "smuggled"
        with pytest.raises(ValueError, match="[Ff]orbidden authority"):
            CompletionCommand.from_dict(payload)


def test_unknown_fields_rejected() -> None:
    payload = {
        "completion_command_id": "cmd-1",
        "assignment_id": "assign-1",
        "execution_kind": "initial",
        "failure_code": "EXECUTION_FAILED",
        "mystery": "field",
    }
    with pytest.raises(ValueError, match="[Uu]nexpected fields"):
        CompletionCommand.from_dict(payload)


@pytest.mark.parametrize("field", ["exception", "traceback", "stack_trace", "provider_error"])
def test_raw_failure_text_fields_rejected(field: str) -> None:
    payload = {
        "completion_command_id": "cmd-1",
        "assignment_id": "assign-1",
        "execution_kind": "initial",
        "failure_code": "EXECUTION_FAILED",
        field: "boom",
    }
    with pytest.raises(ValueError, match="[Rr]aw failure text"):
        CompletionCommand.from_dict(payload)


def test_secret_shaped_top_level_key_rejected() -> None:
    payload = {
        "completion_command_id": "cmd-1",
        "assignment_id": "assign-1",
        "execution_kind": "initial",
        "failure_code": "EXECUTION_FAILED",
        "api_key": "secret-value",
    }
    with pytest.raises(ValueError, match="[Cc]redential"):
        CompletionCommand.from_dict(payload)


def test_secret_shaped_nested_source_key_rejected() -> None:
    result = TaskResult(
        task_id="task-1",
        output="done",
        sources=[{"url": "https://example.test", "auth_token": "secret"}],
    )
    with pytest.raises(ValueError, match="[Cc]redential"):
        _success_command(completed_result=result)


def test_task_result_rejects_unknown_fields() -> None:
    payload = {
        "task_id": "task-1",
        "output": "done",
        "status": "completed",
        "exception": "boom",
    }
    with pytest.raises(ValueError, match="[Uu]nexpected fields"):
        TaskResult.from_dict(payload)


def _nested_result_payload(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {"task_id": "task-1", "output": "done", "status": "completed"}
    result.update(overrides)
    return {
        "completion_command_id": "cmd-nested-1",
        "assignment_id": "assign-nested-1",
        "execution_kind": "initial",
        "result": result,
    }


@pytest.mark.parametrize(
    "field",
    [
        "exception",
        "traceback",
        "stack_trace",
        "provider_error",
        "error_message",
        "error_detail",
        "stacktrace",
        "raw_error",
        "provider_body",
        "failure_text",
        "error_body",
        "error_text",
        "error",
        "prompt",
        "credentials",
        "assignment_proof",
        "bearer_token",
    ],
)
def test_nested_raw_and_secret_fields_rejected_in_result(field: str) -> None:
    payload = _nested_result_payload(**{field: "smuggled"})
    with pytest.raises(
        ValueError, match="[Ff]orbidden nested field|[Uu]nexpected fields|[Cc]redential"
    ):
        CompletionCommand.from_dict(payload)


def test_nested_raw_fields_rejected_inside_sources() -> None:
    payload = _nested_result_payload(
        sources=[
            {"url": "https://example.test", "content": "evidence"},
            {"url": "https://example.test/2", "traceback": "frame 0"},
        ]
    )
    with pytest.raises(ValueError, match="[Ff]orbidden nested field"):
        CompletionCommand.from_dict(payload)


def test_nested_secret_fields_rejected_inside_sources() -> None:
    payload = _nested_result_payload(
        sources=[{"url": "https://example.test", "api_key": "secret-value"}]
    )
    with pytest.raises(ValueError, match="[Cc]redential"):
        CompletionCommand.from_dict(payload)


def test_nested_error_detail_and_prompt_rejected_in_sources() -> None:
    for key in ("error_detail", "prompt", "assignment_proof", "bearer_token", "credentials"):
        payload = _nested_result_payload(sources=[{"url": "https://x.test", key: "x"}])
        with pytest.raises(ValueError):
            CompletionCommand.from_dict(payload)


def test_legitimate_result_and_source_fields_accepted() -> None:
    payload = _nested_result_payload(
        acknowledged=True,
        sources=[
            {
                "url": "https://example.test/report",
                "title": "Findings",
                "content": "Key findings text",
                "domain": "example.test",
                "rank_score": 0.9,
                "retained": True,
            }
        ],
    )
    command = CompletionCommand.from_dict(payload)
    assert command.completed_result is not None
    assert command.completed_result.task_id == "task-1"
    assert command.completed_result.sources is not None
    assert len(command.completed_result.sources) == 1


def test_completion_command_id_bounded_length() -> None:
    ok_id = "c" * MAX_COMPLETION_COMMAND_ID_LENGTH
    _success_command(completion_command_id=ok_id)
    with pytest.raises(ValueError, match="must not exceed"):
        _success_command(completion_command_id="c" * (MAX_COMPLETION_COMMAND_ID_LENGTH + 1))


def test_empty_and_unknown_execution_kinds_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _failure_command(execution_kind="  ")
    with pytest.raises(ValueError, match="[Ii]nvalid execution_kind"):
        _failure_command(execution_kind="background")


# --- Continuation version rules ---------------------------------------------------


def test_continuation_version_required_for_resumed() -> None:
    cmd = CompletionCommand(
        completion_command_id="cmd-r",
        assignment_id="assign-r",
        execution_kind="resumed_continuation",
        continuation_version=1,
        failure_code="EXECUTION_FAILED",
    )
    assert cmd.continuation_version == 1


def test_continuation_version_missing_for_resumed_rejected() -> None:
    with pytest.raises(ValueError, match="required for resumed_continuation"):
        CompletionCommand(
            completion_command_id="cmd-r",
            assignment_id="assign-r",
            execution_kind="resumed_continuation",
            failure_code="EXECUTION_FAILED",
        )


@pytest.mark.parametrize("kind", ["initial", "delegated_child"])
def test_continuation_version_forbidden_for_non_resumed(kind: str) -> None:
    with pytest.raises(ValueError, match="must not be provided"):
        CompletionCommand(
            completion_command_id="cmd-1",
            assignment_id="assign-1",
            execution_kind=kind,
            continuation_version=1,
            failure_code="EXECUTION_FAILED",
        )


# --- Deterministic fingerprint ----------------------------------------------------


def test_fingerprint_stable_and_transport_independent() -> None:
    first = _success_command().semantic_fingerprint()
    assert len(first) == 64
    assert _success_command().semantic_fingerprint() == first
    renamed = _success_command(completion_command_id="cmd-2")
    assert renamed.semantic_fingerprint() == first


def test_fingerprint_changes_with_semantic_payload() -> None:
    base = _success_command().semantic_fingerprint()
    assert _success_command(completed_result=_completed_result(output="other")).semantic_fingerprint() != base
    assert _failure_command().semantic_fingerprint() != base
    assert _failure_command(failure_code="TOOL_DENIED").semantic_fingerprint() != _failure_command().semantic_fingerprint()
    assert _failure_command(assignment_id="assign-2").semantic_fingerprint() != base


def test_fingerprint_rejects_mixed_or_unknown_branch() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        compute_completion_fingerprint(
            assignment_id="a", execution_kind="initial", continuation_version=None
        )
    with pytest.raises(ValueError, match="[Uu]nsupported failure code"):
        compute_completion_fingerprint(
            assignment_id="a",
            execution_kind="initial",
            continuation_version=None,
            failure_code="DEADLINE_EXCEEDED",
        )


# --- Outcome ID derivation ----------------------------------------------------------


def test_outcome_id_stable_prefixed_and_namespaced() -> None:
    fingerprint = _success_command().semantic_fingerprint()
    first = derive_terminal_outcome_id(fingerprint)
    assert first.startswith(COMPLETION_OUTCOME_ID_PREFIX)
    assert len(first) == len(COMPLETION_OUTCOME_ID_PREFIX) + 64
    assert derive_terminal_outcome_id(fingerprint) == first
    expected = (
        COMPLETION_OUTCOME_ID_PREFIX
        + hashlib.sha256(
            (COMPLETION_OUTCOME_ID_SEPARATOR + fingerprint).encode("utf-8")
        ).hexdigest()
    )
    assert first == expected


def test_same_semantic_replay_returns_same_outcome_id() -> None:
    first = derive_terminal_outcome_id(_success_command().semantic_fingerprint())
    replay = derive_terminal_outcome_id(
        _success_command(completion_command_id="cmd-other").semantic_fingerprint()
    )
    assert replay == first


def test_outcome_id_rejects_malformed_fingerprint() -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        derive_terminal_outcome_id("not-a-digest")


# --- TerminalOutcome ------------------------------------------------------------------


def test_completed_outcome_build_and_roundtrip() -> None:
    outcome, result = _terminal_outcome(_success_command())
    assert outcome.terminal_status == "completed"
    assert outcome.failure_code is None
    assert result is not None
    assert outcome.outcome_id == derive_terminal_outcome_id(outcome.semantic_fingerprint)
    clone = TerminalOutcome.from_json(outcome.to_json())
    assert clone == outcome
    assert TerminalOutcome.from_dict(outcome.to_dict()) == outcome


def test_failed_outcome_requires_fixed_safe_summary() -> None:
    for code in sorted(ALLOWED_COMPLETION_FAILURE_CODES):
        outcome, result = _terminal_outcome(_failure_command(failure_code=code))
        assert outcome.terminal_status == "failed"
        assert outcome.failure_code == code
        assert outcome.safe_summary == SAFE_COMPLETION_SUMMARIES[code]
        assert result is None


def test_failed_outcome_rejects_wrong_summary() -> None:
    outcome, _ = _terminal_outcome(_failure_command())
    tampered = dict(outcome.to_dict())
    tampered["safe_summary"] = "custom operator text"
    with pytest.raises(ValueError, match="fixed platform summary"):
        TerminalOutcome.from_dict(tampered)


def test_completed_outcome_rejects_failure_code() -> None:
    outcome, _ = _terminal_outcome(_success_command())
    tampered = dict(outcome.to_dict())
    tampered["failure_code"] = "EXECUTION_FAILED"
    with pytest.raises(ValueError, match="must be absent"):
        TerminalOutcome.from_dict(tampered)


def test_failed_outcome_requires_failure_code() -> None:
    outcome, _ = _terminal_outcome(_failure_command())
    tampered = dict(outcome.to_dict())
    tampered["failure_code"] = None
    with pytest.raises(ValueError, match="required for failed"):
        TerminalOutcome.from_dict(tampered)


def test_outcome_rejects_mismatched_outcome_id() -> None:
    outcome, _ = _terminal_outcome(_success_command())
    tampered = dict(outcome.to_dict())
    tampered["outcome_id"] = "outcome-" + "0" * 64
    with pytest.raises(ValueError, match="outcome_id"):
        TerminalOutcome.from_dict(tampered)


def test_outcome_rejects_unknown_fields() -> None:
    outcome, _ = _terminal_outcome(_success_command())
    tampered = dict(outcome.to_dict())
    tampered["prompt"] = "must never persist"
    with pytest.raises(ValueError, match="[Uu]nexpected fields"):
        TerminalOutcome.from_dict(tampered)


# --- CompletionResponse -----------------------------------------------------------------


def test_completion_response_completed_carries_result() -> None:
    outcome, result = _terminal_outcome(_success_command())
    assert result is not None
    response = CompletionResponse(outcome=outcome, replayed=False, result=result)
    assert response.to_dict()["result"]["output"] == "done"


def test_completion_response_failed_carries_no_result() -> None:
    outcome, result = _terminal_outcome(_failure_command())
    assert result is None
    response = CompletionResponse(outcome=outcome, replayed=True)
    assert "result" not in response.to_dict()
    with pytest.raises(ValueError, match="must not carry"):
        CompletionResponse(
            outcome=outcome, replayed=False, result=_completed_result()
        )


def test_completion_response_completed_requires_result() -> None:
    outcome, _ = _terminal_outcome(_success_command())
    with pytest.raises(ValueError, match="result is required"):
        CompletionResponse(outcome=outcome, replayed=False)


# --- Assignment completion audit ----------------------------------------------------------


def test_assignment_completion_audit_roundtrip() -> None:
    outcome, _ = _terminal_outcome(_success_command())
    audit = build_completion_audit(outcome=outcome, fencing_epoch=3)
    assert audit.fencing_epoch == 3
    assert audit.outcome_id == outcome.outcome_id
    assert audit.completion_command_id == outcome.completion_command_id
    clone = AssignmentCompletionRecord.from_json(audit.to_json())
    assert clone == audit


def test_assignment_completion_audit_rejects_bad_epoch() -> None:
    outcome, _ = _terminal_outcome(_success_command())
    with pytest.raises(ValueError, match="fencing_epoch"):
        build_completion_audit(outcome=outcome, fencing_epoch=0)


# --- Failure classification -----------------------------------------------------------------


def test_classify_completion_failure_mapping() -> None:
    assert (
        classify_completion_failure(
            AllInferenceProvidersFailedError(
                failures=(ProviderFailureSummary(provider_name="p", reason="down"),)
            )
        )
        == "PROVIDER_UNAVAILABLE"
    )
    # Ambiguous built-ins default to EXECUTION_FAILED: the repository also
    # raises them for internal defects, and no raisable tool-denial or
    # input-validation exception exists in the live taxonomy.
    assert classify_completion_failure(PermissionError("denied")) == "EXECUTION_FAILED"
    assert classify_completion_failure(ValueError("bad")) == "EXECUTION_FAILED"
    assert classify_completion_failure(TypeError("bad")) == "EXECUTION_FAILED"
    assert classify_completion_failure(RuntimeError("boom")) == "EXECUTION_FAILED"
    assert classify_completion_failure(TimeoutError("slow")) == "EXECUTION_FAILED"


def test_classify_completion_failure_unwraps_runtime_wrappers() -> None:
    class Wrapper(RuntimeError):
        def __init__(self, original: BaseException) -> None:
            super().__init__("wrapped")
            self.original_error = original

    assert classify_completion_failure(Wrapper(ValueError("bad"))) == "EXECUTION_FAILED"
    assert (
        classify_completion_failure(Wrapper(PermissionError("denied"))) == "EXECUTION_FAILED"
    )
    assert (
        classify_completion_failure(
            Wrapper(
                AllInferenceProvidersFailedError(
                    failures=(ProviderFailureSummary(provider_name="p", reason="down"),)
                )
            )
        )
        == "PROVIDER_UNAVAILABLE"
    )


# --- Serialization immutability ---------------------------------------------------------------


def test_contracts_are_frozen_and_canonical() -> None:
    cmd = _success_command()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.assignment_id = "mutated"  # type: ignore[misc]
    outcome, _ = _terminal_outcome(cmd)
    first = outcome.to_json()
    assert TerminalOutcome.from_json(first).to_json() == first
    assert cmd.to_json() == CompletionCommand.from_json(cmd.to_json()).to_json()
