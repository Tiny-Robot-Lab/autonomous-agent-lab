from datetime import UTC, datetime

import pytest
from agent_contracts import (
    ActiveClaim,
    ArtifactReference,
    ClaimExpiryRecord,
    RecoveryDecisionCommand,
    RecoveryDecisionRecord,
    ResumedContinuationContext,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TerminalOutcome,
    build_claim_timed_out_terminal_outcome,
    compute_recovery_decision_fingerprint,
)
from orchestrator.config import Settings
from orchestrator.domain.principal import RECOGNIZED_CAPABILITIES, RuntimePrincipal


def test_active_claim_execution_expires_at() -> None:
    now_iso = datetime.now(UTC).isoformat()
    claim = ActiveClaim(
        task_id="task-1",
        assignment_id="assign-1",
        claim_id="claim-1",
        root_task_id="task-1",
        turn_id="turn-1",
        runtime_principal_id="principal-1",
        bound_agent_id="agent-1",
        assigned_capability="general",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="a" * 64,
        request_id="req-1",
        claimed_at=now_iso,
        delivery_expires_at=now_iso,
        state="acknowledged",
        acknowledged_at=now_iso,
        execution_expires_at=now_iso,
    )
    assert claim.execution_expires_at is not None
    data = claim.to_dict()
    assert "execution_expires_at" in data
    rehydrated = ActiveClaim.from_dict(data)
    assert rehydrated.execution_expires_at == claim.execution_expires_at

    # Backward compatibility: legacy claim without execution_expires_at
    data.pop("execution_expires_at")
    legacy = ActiveClaim.from_dict(data)
    assert legacy.execution_expires_at is None


def test_claim_timed_out_terminal_outcome() -> None:
    now_iso = datetime.now(UTC).isoformat()
    outcome = build_claim_timed_out_terminal_outcome(
        task_id="child-1",
        claim_deadline=now_iso,
        completed_at=now_iso,
    )
    assert outcome.terminal_status == "claim_timed_out"
    assert outcome.failure_code is None
    assert "claim deadline expired" in outcome.safe_summary.lower()
    assert outcome.execution_kind == "delegated_child"
    assert outcome.task_id == "child-1"

    data = outcome.to_dict()
    rehydrated = TerminalOutcome.from_dict(data)
    assert rehydrated.terminal_status == "claim_timed_out"
    assert rehydrated.outcome_id == outcome.outcome_id


def test_claim_expiry_record_validation() -> None:
    now_iso = datetime.now(UTC).isoformat()
    rec = ClaimExpiryRecord(
        task_id="task-1",
        claim_id="claim-1",
        assignment_id="assign-1",
        principal_id="principal-1",
        bound_agent_id="agent-1",
        execution_kind="initial",
        expired_fencing_epoch=1,
        acknowledged_at=now_iso,
        execution_expires_at=now_iso,
        expired_at=now_iso,
        prior_claimable_status="pending",
    )
    assert rec.task_id == "task-1"
    assert rec.prior_claimable_status == "pending"

    # Mismatch between execution_kind and prior_claimable_status must fail
    with pytest.raises(ValueError, match="prior_claimable_status must be 'pending'"):
        ClaimExpiryRecord(
            task_id="task-1",
            claim_id="claim-1",
            assignment_id="assign-1",
            principal_id="principal-1",
            bound_agent_id="agent-1",
            execution_kind="initial",
            expired_fencing_epoch=1,
            acknowledged_at=now_iso,
            execution_expires_at=now_iso,
            expired_at=now_iso,
            prior_claimable_status="resumable",
        )

    # Resumed continuation must have continuation_version
    with pytest.raises(ValueError, match="continuation_version is required"):
        ClaimExpiryRecord(
            task_id="task-1",
            claim_id="claim-1",
            assignment_id="assign-1",
            principal_id="principal-1",
            bound_agent_id="agent-1",
            execution_kind="resumed_continuation",
            expired_fencing_epoch=1,
            acknowledged_at=now_iso,
            execution_expires_at=now_iso,
            expired_at=now_iso,
            prior_claimable_status="resumable",
            continuation_version=None,
        )

    # Roundtrip
    data = rec.to_dict()
    rehydrated = ClaimExpiryRecord.from_dict(data)
    assert rehydrated == rec


def test_recovery_decision_command_and_record() -> None:
    now_iso = datetime.now(UTC).isoformat()
    cmd = RecoveryDecisionCommand(
        recovery_decision_id="dec-1",
        task_id="task-1",
        expired_claim_id="claim-1",
        expired_assignment_id="assign-1",
        expired_fencing_epoch=1,
        disposition="restore_claimable",
    )
    assert cmd.disposition == "restore_claimable"
    fp = cmd.semantic_fingerprint()
    assert fp == compute_recovery_decision_fingerprint(
        task_id="task-1",
        expired_claim_id="claim-1",
        expired_assignment_id="assign-1",
        expired_fencing_epoch=1,
        disposition="restore_claimable",
    )

    # Unsupported disposition must fail closed
    with pytest.raises(ValueError, match="Unsupported recovery disposition"):
        RecoveryDecisionCommand(
            recovery_decision_id="dec-1",
            task_id="task-1",
            expired_claim_id="claim-1",
            expired_assignment_id="assign-1",
            expired_fencing_epoch=1,
            disposition="abort_failed",
        )

    rec = RecoveryDecisionRecord(
        recovery_decision_id="dec-1",
        task_id="task-1",
        expired_claim_id="claim-1",
        expired_assignment_id="assign-1",
        expired_fencing_epoch=1,
        disposition="restore_claimable",
        semantic_fingerprint=fp,
        prior_status="recovery_required",
        restored_status="pending",
        operator_principal_id="operator-1",
        resolved_at=now_iso,
    )
    assert rec.restored_status == "pending"
    data = rec.to_dict()
    rehydrated = RecoveryDecisionRecord.from_dict(data)
    assert rehydrated == rec


def test_recovery_operator_capability_recognition() -> None:
    assert "recovery_operator" in RECOGNIZED_CAPABILITIES
    principal = RuntimePrincipal(
        principal_id="rec-op",
        bound_agent_id="operator-agent",
        authorized_capabilities=("recovery_operator",),
    )
    assert principal.has_capability("recovery_operator")
    assert not principal.has_capability("research")


def test_settings_execution_claim_timeout_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default is 900
    s = Settings()
    assert s.execution_claim_timeout_seconds == 900.0

    # Bounds: 60 to 7200
    s_valid = Settings(execution_claim_timeout_seconds=60.0)
    assert s_valid.execution_claim_timeout_seconds == 60.0
    s_valid2 = Settings(execution_claim_timeout_seconds=7200.0)
    assert s_valid2.execution_claim_timeout_seconds == 7200.0

    with pytest.raises(ValueError, match="execution_claim_timeout_seconds must be a number between 60.0 and 7200.0"):
        Settings(execution_claim_timeout_seconds=59.0)

    with pytest.raises(ValueError, match="execution_claim_timeout_seconds must be a number between 60.0 and 7200.0"):
        Settings(execution_claim_timeout_seconds=7201.0)

    monkeypatch.setenv("EXECUTION_CLAIM_TIMEOUT_SECONDS", "300")
    s_env = Settings.from_env()
    assert s_env.execution_claim_timeout_seconds == 300.0


def test_resumed_continuation_context_contract() -> None:
    from agent_contracts import (
        ResumedContinuationContext,
        StageArtifact,
        TaskContinuation,
        TaskInput,
        TaskResult,
        build_claim_timed_out_terminal_outcome,
    )

    now_iso = datetime.now(UTC).isoformat()
    parent_input = TaskInput(
        task_id="parent-1",
        prompt="Synthesize findings",
        created_at=now_iso,
    )
    continuation = TaskContinuation(
        parent_task_id="parent-1",
        child_task_id="child-1",
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
        artifact_references=(
            ArtifactReference(
                producing_task_id="parent-1",
                artifact_id="art-1",
                artifact_version=1,
            ),
        ),
    )
    stage_artifact = StageArtifact(
        artifact_id="art-1",
        producing_task_id="parent-1",
        artifact_type="research_material",
        content={"material": "primary facts"},
        created_at=now_iso,
        sealed=True,
    )
    child_outcome = build_claim_timed_out_terminal_outcome(
        task_id="child-1",
        claim_deadline=now_iso,
        completed_at=now_iso,
    )

    ctx = ResumedContinuationContext(
        parent_task_input=parent_input,
        continuation=continuation,
        stage_artifact=stage_artifact,
        child_outcome=child_outcome,
    )
    assert ctx.version == 1
    assert ctx.child_result is None

    # Serialization roundtrip
    d = ctx.to_dict()
    assert d["parent_task_input"]["task_id"] == "parent-1"
    assert d["continuation"]["stage"] == "incorporate_analysis_child"
    rehydrated = ResumedContinuationContext.from_dict(d)
    assert rehydrated == ctx
    json_str = ctx.to_json()
    assert ResumedContinuationContext.from_json(json_str) == ctx

    # Test claim_timed_out with child_result MUST be rejected
    child_res = TaskResult(task_id="child-1", output="findings", status="completed")
    with pytest.raises(ValueError, match="child_result must be absent when child_outcome.terminal_status is 'claim_timed_out'"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=continuation,
            stage_artifact=stage_artifact,
            child_outcome=child_outcome,
            child_result=child_res,
        )

    # Test completed child outcome requires valid child_result
    from agent_contracts import TerminalOutcome, derive_terminal_outcome_id
    completed_outcome = TerminalOutcome(
        outcome_id=derive_terminal_outcome_id("0" * 64),
        task_id="child-1",
        assignment_id="asgn-child-1",
        completion_command_id="cmd-child-1",
        semantic_fingerprint="0" * 64,
        runtime_principal_id="p-analysis",
        bound_agent_id="agent-analysis",
        terminal_status="completed",
        safe_summary="Child analysis completed",
        completed_at=now_iso,
        execution_kind="delegated_child",
    )
    with pytest.raises(ValueError, match="child_result is required when child_outcome.terminal_status is 'completed'"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=continuation,
            stage_artifact=stage_artifact,
            child_outcome=completed_outcome,
            child_result=None,
        )

    ctx_completed = ResumedContinuationContext(
        parent_task_input=parent_input,
        continuation=continuation,
        stage_artifact=stage_artifact,
        child_outcome=completed_outcome,
        child_result=child_res,
    )
    assert ctx_completed.child_result is not None
    assert ResumedContinuationContext.from_dict(ctx_completed.to_dict()) == ctx_completed
    assert ResumedContinuationContext.from_json(ctx_completed.to_json()) == ctx_completed

    # Test failed child outcome rejects child_result
    failed_outcome = TerminalOutcome(
        outcome_id=derive_terminal_outcome_id("1" * 64),
        task_id="child-1",
        assignment_id="asgn-child-1",
        completion_command_id="cmd-child-fail-1",
        semantic_fingerprint="1" * 64,
        runtime_principal_id="p-analysis",
        bound_agent_id="agent-analysis",
        terminal_status="failed",
        failure_code="EXECUTION_FAILED",
        safe_summary="Task execution failed.",
        completed_at=now_iso,
        execution_kind="delegated_child",
    )
    with pytest.raises(ValueError, match="child_result must be absent when child_outcome.terminal_status is 'failed'"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=continuation,
            stage_artifact=stage_artifact,
            child_outcome=failed_outcome,
            child_result=child_res,
        )

    # Reject non-integer versions
    for bad_ver in ("1", True, False, 1.0):
        with pytest.raises(TypeError, match="version must be an int"):
            ResumedContinuationContext(
                parent_task_input=parent_input,
                continuation=continuation,
                stage_artifact=stage_artifact,
                child_outcome=child_outcome,
                version=bad_ver,  # type: ignore[arg-type]
            )
        bad_dict = ctx.to_dict()
        bad_dict["version"] = bad_ver
        with pytest.raises(TypeError, match="version must be an int"):
            ResumedContinuationContext.from_dict(bad_dict)

    # Validation errors: task ID mismatch
    with pytest.raises(ValueError, match="does not match continuation.parent_task_id"):
        ResumedContinuationContext(
            parent_task_input=TaskInput(task_id="other", prompt="p", created_at=now_iso),
            continuation=continuation,
            stage_artifact=stage_artifact,
            child_outcome=child_outcome,
        )

    with pytest.raises(ValueError, match="does not match continuation.child_task_id"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=continuation,
            stage_artifact=stage_artifact,
            child_outcome=build_claim_timed_out_terminal_outcome(
                task_id="other-child",
                claim_deadline=now_iso,
                completed_at=now_iso,
            ),
        )


def test_resumed_continuation_context_artifact_reference_invariants() -> None:
    now_iso = datetime.now(UTC).isoformat()
    parent_input = TaskInput(task_id="p-1", prompt="prompt", created_at=now_iso)
    sealed_art = StageArtifact(
        artifact_id="art-1",
        producing_task_id="p-1",
        artifact_type="research_material",
        content={"material": "test facts"},
        created_at=now_iso,
        artifact_version=1,
        sealed=True,
    )
    outcome = build_claim_timed_out_terminal_outcome(
        task_id="c-1",
        claim_deadline=now_iso,
        completed_at=now_iso,
    )

    # 1. Valid matching artifact
    cont_valid = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
        artifact_references=(
            ArtifactReference(producing_task_id="p-1", artifact_id="art-1", artifact_version=1),
        ),
    )
    ctx = ResumedContinuationContext(
        parent_task_input=parent_input,
        continuation=cont_valid,
        stage_artifact=sealed_art,
        child_outcome=outcome,
    )
    assert ctx.stage_artifact.artifact_id == "art-1"

    # 2. Empty artifact reference list
    cont_empty = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
        artifact_references=(),
    )
    with pytest.raises(ValueError, match="continuation has no artifact references"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=cont_empty,
            stage_artifact=sealed_art,
            child_outcome=outcome,
        )

    # 3. Wrong artifact ID
    cont_wrong_id = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
        artifact_references=(
            ArtifactReference(producing_task_id="p-1", artifact_id="art-other", artifact_version=1),
        ),
    )
    with pytest.raises(ValueError, match="does not match any artifact reference in continuation"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=cont_wrong_id,
            stage_artifact=sealed_art,
            child_outcome=outcome,
        )

    # 4. Wrong artifact version
    cont_wrong_ver = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
        artifact_references=(
            ArtifactReference(producing_task_id="p-1", artifact_id="art-1", artifact_version=2),
        ),
    )
    with pytest.raises(ValueError, match="does not match any artifact reference in continuation"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=cont_wrong_ver,
            stage_artifact=sealed_art,
            child_outcome=outcome,
        )

    # 5. Wrong producing task in reference
    cont_wrong_task = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
        artifact_references=(
            ArtifactReference(producing_task_id="p-other", artifact_id="art-1", artifact_version=1),
        ),
    )
    with pytest.raises(ValueError, match="does not match any artifact reference in continuation"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=cont_wrong_task,
            stage_artifact=sealed_art,
            child_outcome=outcome,
        )

    # 6. Unsealed artifact rejected
    with pytest.raises(ValueError, match="sealed must be True"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="p-1",
            artifact_type="research_material",
            content={"material": "test"},
            created_at=now_iso,
            artifact_version=1,
            sealed=False,
        )

    # 7. Multiple references with no match
    cont_multi_no_match = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
        artifact_references=(
            ArtifactReference(producing_task_id="p-1", artifact_id="art-other-1", artifact_version=1),
            ArtifactReference(producing_task_id="p-1", artifact_id="art-other-2", artifact_version=1),
        ),
    )
    with pytest.raises(ValueError, match="does not match any artifact reference in continuation"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=cont_multi_no_match,
            stage_artifact=sealed_art,
            child_outcome=outcome,
        )

    # 8. Ambiguous duplicate matching references
    cont_duplicate_match = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        created_at=now_iso,
        updated_at=now_iso,
        stage="incorporate_analysis_child",
        continuation_version=1,
        artifact_references=(
            ArtifactReference(producing_task_id="p-1", artifact_id="art-1", artifact_version=1),
            ArtifactReference(producing_task_id="p-1", artifact_id="art-1", artifact_version=1),
        ),
    )
    with pytest.raises(ValueError, match="matches multiple \\(2\\) duplicate artifact references"):
        ResumedContinuationContext(
            parent_task_input=parent_input,
            continuation=cont_duplicate_match,
            stage_artifact=sealed_art,
            child_outcome=outcome,
        )


def test_completion_command_rejects_claim_timed_out_and_terminal_status() -> None:
    from agent_contracts import CompletionCommand

    # 1. failure_code="claim_timed_out" must be rejected: only platform can materialize claim_timed_out
    with pytest.raises(ValueError, match="Unsupported failure code: 'claim_timed_out'"):
        CompletionCommand(
            completion_command_id="cmd-1",
            assignment_id="asgn-1",
            execution_kind="delegated_child",
            failure_code="claim_timed_out",
        )

    # 2. Attempting to smuggle terminal_status into command must fail (unknown field)
    with pytest.raises(ValueError, match="Unexpected fields in CompletionCommand"):
        CompletionCommand.from_dict({
            "completion_command_id": "cmd-1",
            "assignment_id": "asgn-1",
            "execution_kind": "delegated_child",
            "failure_code": "EXECUTION_FAILED",
            "terminal_status": "claim_timed_out",
        })


def test_claim_expiry_record_alias_parity_and_canonical_emitted_fields() -> None:
    now_iso = datetime.now(UTC).isoformat()
    raw_dict = {
        "task_id": "task-alias-1",
        "claim_id": "claim-alias-1",
        "assignment_id": "asgn-alias-1",
        "runtime_principal_id": "p-alias-1",
        "bound_agent_id": "agent-alias-1",
        "execution_kind": "initial",
        "expired_fencing_epoch": 2,
        "acknowledged_at": now_iso,
        "execution_expires_at": now_iso,
        "expired_at": now_iso,
        "prior_claimable_status": "pending",
    }
    rec = ClaimExpiryRecord.from_dict(raw_dict)
    assert rec.principal_id == "p-alias-1"

    emitted = rec.to_dict()
    assert emitted["principal_id"] == "p-alias-1"
    assert "runtime_principal_id" not in emitted

    # Roundtrip from canonical emitted dict
    rehydrated = ClaimExpiryRecord.from_dict(emitted)
    assert rehydrated == rec
