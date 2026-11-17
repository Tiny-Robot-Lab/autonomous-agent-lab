import json
from datetime import UTC, datetime

import pytest
from agent_contracts import (
    ALL_TASK_LIFECYCLE_STATUSES,
    ALLOWED_CONTINUATION_STAGES,
    CHILD_LIFECYCLE_STATUSES,
    DEFAULT_MAX_CONTINUATION_BYTES,
    FORBIDDEN_SECRET_KEY_SUBSTRINGS,
    NONTERMINAL_TASK_LIFECYCLE_STATUSES,
    PARENT_LIFECYCLE_STATUSES,
    STAGE_CONTROL_FIELDS_ALLOWLIST,
    TERMINAL_TASK_LIFECYCLE_STATUSES,
    RootAuthorityRecord,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskLifecycleStatus,
    validate_lifecycle_status,
)

# ==============================================================================
# 1. TaskLifecycleStatus Tests
# ==============================================================================


def test_task_lifecycle_statuses_defined_and_categorized() -> None:
    assert TaskLifecycleStatus.PENDING == "pending"
    assert TaskLifecycleStatus.WAITING_FOR_CHILDREN == "waiting_for_children"
    assert TaskLifecycleStatus.RESUMABLE == "resumable"
    assert TaskLifecycleStatus.CLAIM_TIMED_OUT == "claim_timed_out"

    # Parent statuses per ADR 0007 Decision 3
    expected_parent = {
        "pending_dispatch",
        "pending",
        "working",
        "waiting_for_children",
        "resumable",
        "recovery_required",
        "completed",
        "failed",
        "dispatch_failed",
    }
    assert PARENT_LIFECYCLE_STATUSES == expected_parent

    # Child statuses per ADR 0007 Decision 3
    expected_child = {
        "pending_dispatch",
        "pending",
        "working",
        "recovery_required",
        "completed",
        "failed",
        "claim_timed_out",
        "dispatch_failed",
    }
    assert CHILD_LIFECYCLE_STATUSES == expected_child

    # All statuses
    assert ALL_TASK_LIFECYCLE_STATUSES == expected_parent | expected_child

    # Terminal vs Nonterminal
    assert "claim_timed_out" in TERMINAL_TASK_LIFECYCLE_STATUSES
    assert "completed" in TERMINAL_TASK_LIFECYCLE_STATUSES
    assert "failed" in TERMINAL_TASK_LIFECYCLE_STATUSES
    assert "waiting_for_children" in NONTERMINAL_TASK_LIFECYCLE_STATUSES
    assert "resumable" in NONTERMINAL_TASK_LIFECYCLE_STATUSES
    assert "working" in NONTERMINAL_TASK_LIFECYCLE_STATUSES


def test_validate_lifecycle_status() -> None:
    assert validate_lifecycle_status("waiting_for_children") == "waiting_for_children"
    assert validate_lifecycle_status("claim_timed_out") == "claim_timed_out"
    assert validate_lifecycle_status("  resumable  ") == "resumable"

    with pytest.raises(ValueError, match="Invalid task lifecycle status"):
        validate_lifecycle_status("non_existent_status")

    with pytest.raises(TypeError, match="status must be a str"):
        validate_lifecycle_status(123)  # type: ignore[arg-type]


# ==============================================================================
# 2. TaskInput Contract Tests
# ==============================================================================


def test_task_input_valid_creation_and_roundtrip() -> None:
    now = datetime.now(UTC).isoformat()
    record = TaskInput(
        task_id="task-123",
        prompt="Analyze system telemetry",
        created_at=now,
        input_version=1,
        artifact_references=("art-1", "art-2"),
        metadata={"priority": "high", "domain": "infra"},
        version=1,
    )

    assert record.task_id == "task-123"
    assert record.prompt == "Analyze system telemetry"
    assert record.input_version == 1
    assert record.artifact_references == ("art-1", "art-2")
    assert record.metadata == {"priority": "high", "domain": "infra"}
    assert record.version == 1

    d = record.to_dict()
    assert d["task_id"] == "task-123"
    assert d["artifact_references"] == ["art-1", "art-2"]
    assert d["input_version"] == 1

    from_dict_record = TaskInput.from_dict(d)
    assert from_dict_record == record

    from_json_record = TaskInput.from_json(record.to_json())
    assert from_json_record == record


def test_task_input_create_factory() -> None:
    record = TaskInput.create(
        task_id="task-456",
        prompt="Execute evaluation benchmark",
        input_version=2,
        artifact_references=["art-abc"],
    )
    assert record.task_id == "task-456"
    assert record.input_version == 2
    assert record.artifact_references == ("art-abc",)
    assert record.version == 1
    assert record.metadata == {}


def test_task_input_validation_errors() -> None:
    now = datetime.now(UTC).isoformat()

    # Empty task_id
    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        TaskInput(task_id="   ", prompt="hello", created_at=now)

    # Empty prompt
    with pytest.raises(ValueError, match="prompt must not be empty"):
        TaskInput(task_id="task-1", prompt="   ", created_at=now)

    # Prompt > 2000 chars
    with pytest.raises(ValueError, match="prompt must not exceed 2000 characters"):
        TaskInput(task_id="task-1", prompt="x" * 2001, created_at=now)

    # Invalid input_version
    with pytest.raises(ValueError, match="input_version must be >= 1"):
        TaskInput(task_id="task-1", prompt="hello", created_at=now, input_version=0)

    # Boolean input_version rejected
    with pytest.raises(TypeError, match="input_version must be an int"):
        TaskInput(task_id="task-1", prompt="hello", created_at=now, input_version=True)  # type: ignore[arg-type]

    # Unsupported contract version
    with pytest.raises(ValueError, match="Unsupported contract version"):
        TaskInput(task_id="task-1", prompt="hello", created_at=now, version=2)

    # Boolean version rejected
    with pytest.raises(TypeError, match="version must be an int"):
        TaskInput(task_id="task-1", prompt="hello", created_at=now, version=True)  # type: ignore[arg-type]


def test_task_input_rejects_secret_bearing_metadata_keys() -> None:
    now = datetime.now(UTC).isoformat()
    for token in FORBIDDEN_SECRET_KEY_SUBSTRINGS:
        with pytest.raises(ValueError, match="Secret-bearing key"):
            TaskInput(
                task_id="task-1",
                prompt="hello",
                created_at=now,
                metadata={f"user_{token}_id": "confidential"},
            )


def test_task_input_from_dict_rejects_unexpected_fields() -> None:
    now = datetime.now(UTC).isoformat()
    payload = {
        "task_id": "task-1",
        "prompt": "hello",
        "created_at": now,
        "input_version": 1,
        "version": 1,
        "unexpected_field": "disallowed",
    }
    with pytest.raises(TypeError, match="unexpected TaskInput fields"):
        TaskInput.from_dict(payload)


# ==============================================================================
# 3. StageArtifact Contract Tests
# ==============================================================================


def test_stage_artifact_valid_creation_and_roundtrip() -> None:
    now = datetime.now(UTC).isoformat()
    artifact = StageArtifact(
        artifact_id="art-999",
        producing_task_id="task-parent-1",
        producing_turn_id="turn-claim-1",
        artifact_type="research_material",
        artifact_version=1,
        created_at=now,
        sealed=True,
        content={"findings": ["finding-1", "finding-2"], "count": 2},
        metadata={"schema_version": "v1.0"},
        version=1,
    )

    assert artifact.artifact_id == "art-999"
    assert artifact.producing_task_id == "task-parent-1"
    assert artifact.producing_turn_id == "turn-claim-1"
    assert artifact.sealed is True
    assert artifact.content == {"findings": ["finding-1", "finding-2"], "count": 2}

    d = artifact.to_dict()
    assert d["artifact_id"] == "art-999"
    assert d["sealed"] is True

    from_dict_artifact = StageArtifact.from_dict(d)
    assert from_dict_artifact == artifact

    from_json_artifact = StageArtifact.from_json(artifact.to_json())
    assert from_json_artifact == artifact


def test_stage_artifact_string_content_and_create_factory() -> None:
    artifact = StageArtifact.create(
        artifact_id="art-text-1",
        producing_task_id="task-p2",
        artifact_type="text_summary",
        content="Summarized intermediate findings from turn 1.",
        producing_turn_id="turn-abc",
    )
    assert artifact.artifact_id == "art-text-1"
    assert artifact.content == "Summarized intermediate findings from turn 1."
    assert artifact.sealed is True


def test_stage_artifact_validation_errors() -> None:
    now = datetime.now(UTC).isoformat()

    # Sealed must be True
    with pytest.raises(ValueError, match="sealed must be True"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="task-1",
            artifact_type="type",
            content="abc",
            created_at=now,
            sealed=False,
        )

    # Empty content
    with pytest.raises(ValueError, match="content string must not be empty"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="task-1",
            artifact_type="type",
            content="   ",
            created_at=now,
        )

    # Empty artifact_id
    with pytest.raises(ValueError, match="artifact_id must be a non-empty string"):
        StageArtifact(
            artifact_id="  ",
            producing_task_id="task-1",
            artifact_type="type",
            content="abc",
            created_at=now,
        )

    # Secret-bearing metadata rejected
    with pytest.raises(ValueError, match="Secret-bearing key"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="task-1",
            artifact_type="type",
            content="abc",
            created_at=now,
            metadata={"auth_token": "secret"},
        )


# ==============================================================================
# 4. TaskContinuation Contract Tests
# ==============================================================================


def test_task_continuation_valid_creation_and_roundtrip() -> None:
    now = datetime.now(UTC).isoformat()
    continuation = TaskContinuation(
        parent_task_id="parent-123",
        child_task_id="child-456",
        input_version=1,
        continuation_capability="research",
        stage="incorporate_analysis_child",
        control_fields={
            "analysis_focus": "key claims verification",
            "child_intent": "analyze evidence density",
            "min_findings": 3,
        },
        artifact_references=("art-1",),
        continuation_version=1,
        created_at=now,
        updated_at=now,
        version=1,
    )

    assert continuation.parent_task_id == "parent-123"
    assert continuation.child_task_id == "child-456"
    assert continuation.stage == "incorporate_analysis_child"
    assert continuation.continuation_version == 1
    assert continuation.byte_size() <= DEFAULT_MAX_CONTINUATION_BYTES

    d = continuation.to_dict()
    assert d["parent_task_id"] == "parent-123"
    assert d["control_fields"]["min_findings"] == 3

    from_dict_cont = TaskContinuation.from_dict(d)
    assert from_dict_cont == continuation

    from_json_cont = TaskContinuation.from_json(continuation.to_json())
    assert from_json_cont == continuation


def test_task_continuation_stage_allowlist() -> None:
    now = datetime.now(UTC).isoformat()

    assert "incorporate_analysis_child" in ALLOWED_CONTINUATION_STAGES

    with pytest.raises(ValueError, match="Unknown continuation stage"):
        TaskContinuation(
            parent_task_id="parent-1",
            child_task_id="child-1",
            stage="arbitrary_unapproved_stage",
            created_at=now,
            updated_at=now,
        )


def test_task_continuation_control_fields_allowlist() -> None:
    now = datetime.now(UTC).isoformat()
    allowed = STAGE_CONTROL_FIELDS_ALLOWLIST["incorporate_analysis_child"]
    assert "analysis_focus" in allowed
    assert "material_summary" in allowed

    # Disallowed control field
    with pytest.raises(ValueError, match="Unallowlisted control fields for stage"):
        TaskContinuation(
            parent_task_id="parent-1",
            child_task_id="child-1",
            stage="incorporate_analysis_child",
            control_fields={"unapproved_control_parameter": "value"},
            created_at=now,
            updated_at=now,
        )


def test_task_continuation_rejects_secret_bearing_control_fields() -> None:
    now = datetime.now(UTC).isoformat()
    # If someone tries to pass a field with 'key', 'token', 'auth'
    with pytest.raises(ValueError):
        TaskContinuation(
            parent_task_id="parent-1",
            child_task_id="child-1",
            stage="incorporate_analysis_child",
            control_fields={"analysis_focus": "test", "auth_token": "leak"},
            created_at=now,
            updated_at=now,
        )


def test_task_continuation_rejects_self_child_parent() -> None:
    now = datetime.now(UTC).isoformat()
    with pytest.raises(ValueError, match="child_task_id must not equal parent_task_id"):
        TaskContinuation(
            parent_task_id="same-task-id",
            child_task_id="same-task-id",
            created_at=now,
            updated_at=now,
        )


def test_task_continuation_size_bound_enforcement() -> None:
    now = datetime.now(UTC).isoformat()

    # Fits within 16 KiB
    valid = TaskContinuation(
        parent_task_id="parent-1",
        child_task_id="child-1",
        control_fields={"material_summary": "x" * 500},
        created_at=now,
        updated_at=now,
    )
    assert valid.byte_size() < DEFAULT_MAX_CONTINUATION_BYTES

    # Exceeds 16 KiB (16384 bytes)
    with pytest.raises(ValueError, match="exceeds maximum bound of 16384 bytes"):
        TaskContinuation(
            parent_task_id="parent-1",
            child_task_id="child-1",
            control_fields={"material_summary": "x" * 17000},
            created_at=now,
            updated_at=now,
        )


def test_task_continuation_from_dict_and_from_json_size_bound() -> None:
    now = datetime.now(UTC).isoformat()
    payload = {
        "parent_task_id": "parent-1",
        "child_task_id": "child-1",
        "input_version": 1,
        "continuation_capability": "research",
        "stage": "incorporate_analysis_child",
        "control_fields": {"material_summary": "x" * 500},
        "artifact_references": [],
        "continuation_version": 1,
        "created_at": now,
        "updated_at": now,
        "version": 1,
    }

    # Custom strict max_bytes (e.g. 100 bytes)
    with pytest.raises(ValueError, match="exceeds maximum bound of 100 bytes"):
        TaskContinuation.from_dict(payload, max_bytes=100)

    with pytest.raises(ValueError, match="exceeds maximum bound of 100 bytes"):
        TaskContinuation.from_json(json.dumps(payload), max_bytes=100)


# ==============================================================================
# 5. RootAuthorityRecord Contract Tests
# ==============================================================================


def test_root_authority_record_valid_creation_and_roundtrip() -> None:
    now = datetime.now(UTC).isoformat()
    auth = RootAuthorityRecord(
        root_task_id="root-100",
        policy_context="default",
        max_delegation_depth=1,
        max_delegation_waves=1,
        max_children_per_wave=1,
        allowed_delegation_edges=(("research", "analysis"),),
        grants_issued=0,
        waves_consumed=0,
        created_at=now,
        updated_at=now,
        version=1,
    )

    assert auth.root_task_id == "root-100"
    assert auth.max_delegation_depth == 1
    assert auth.max_delegation_waves == 1
    assert auth.max_children_per_wave == 1
    assert auth.allowed_delegation_edges == (("research", "analysis"),)
    assert auth.grants_issued == 0
    assert auth.waves_consumed == 0

    d = auth.to_dict()
    assert d["root_task_id"] == "root-100"
    assert d["allowed_delegation_edges"] == [["research", "analysis"]]

    from_dict_auth = RootAuthorityRecord.from_dict(d)
    assert from_dict_auth == auth

    from_json_auth = RootAuthorityRecord.from_json(auth.to_json())
    assert from_json_auth == auth


def test_root_authority_create_default() -> None:
    auth = RootAuthorityRecord.create_default("root-200")
    assert auth.root_task_id == "root-200"
    assert auth.policy_context == "default"
    assert auth.max_delegation_depth == 1
    assert auth.max_delegation_waves == 1
    assert auth.max_children_per_wave == 1
    assert auth.allowed_delegation_edges == (("research", "analysis"),)
    assert auth.grants_issued == 0
    assert auth.waves_consumed == 0
    assert auth.version == 1


def test_root_authority_validation_errors() -> None:
    now = datetime.now(UTC).isoformat()

    # Secret in policy_context rejected
    with pytest.raises(ValueError, match="policy_context contains forbidden secret-bearing token"):
        RootAuthorityRecord(
            root_task_id="root-1",
            policy_context="contains_api_token_here",
            created_at=now,
            updated_at=now,
        )

    # Negative bounds rejected
    with pytest.raises(ValueError, match="max_delegation_depth must be >= 1"):
        RootAuthorityRecord(
            root_task_id="root-1",
            max_delegation_depth=0,
            created_at=now,
            updated_at=now,
        )

    with pytest.raises(ValueError, match="grants_issued must be >= 0"):
        RootAuthorityRecord(
            root_task_id="root-1",
            grants_issued=-1,
            created_at=now,
            updated_at=now,
        )

    # Empty edge element rejected
    with pytest.raises(ValueError, match="delegation edge elements must be non-empty strings"):
        RootAuthorityRecord(
            root_task_id="root-1",
            allowed_delegation_edges=(("research", "   "),),
            created_at=now,
            updated_at=now,
        )
