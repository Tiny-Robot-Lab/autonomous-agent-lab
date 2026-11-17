from datetime import UTC, datetime
from types import MappingProxyType

import pytest
from agent_contracts import (
    ALL_TASK_LIFECYCLE_STATUSES,
    ALLOWED_ARTIFACT_TYPES,
    ALLOWED_CONTINUATION_STAGES,
    CHILD_LIFECYCLE_STATUSES,
    MAX_CONTINUATION_BYTES,
    MAX_STAGE_ARTIFACT_BYTES,
    NONTERMINAL_TASK_LIFECYCLE_STATUSES,
    PARENT_LIFECYCLE_STATUSES,
    STAGE_CONTROL_FIELDS_ALLOWLIST,
    TERMINAL_TASK_LIFECYCLE_STATUSES,
    ArtifactReference,
    RootAuthorityRecord,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskLifecycleStatus,
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    freeze_value,
    is_credential_bearing_key,
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
# 2. Serialization and Credential Detection Tests
# ==============================================================================


def test_canonical_json_dumps_deterministic_and_compact() -> None:
    data = {"z": 1, "a": 2, "m": {"b": 3, "a": 4}}
    serialized = canonical_json_dumps(data)
    # Keys must be sorted and separators compact (no spaces)
    assert serialized == '{"a":2,"m":{"a":4,"b":3},"z":1}'


def test_canonical_json_dumps_rejects_nan_and_inf() -> None:
    # NaN rejected
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_dumps({"nan_val": float("nan")})

    # +Infinity rejected
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_dumps({"inf_val": float("inf")})

    # -Infinity rejected in nested structure
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_dumps({"nested": [1, 2, {"neg_inf": float("-inf")}]})


def test_credential_bearing_key_detection() -> None:
    # Forbidden tokens and compound keys
    assert is_credential_bearing_key("api_key") is True
    assert is_credential_bearing_key("apiKey") is True
    assert is_credential_bearing_key("auth_token") is True
    assert is_credential_bearing_key("client_secret") is True
    assert is_credential_bearing_key("user_password") is True
    assert is_credential_bearing_key("bearer") is True
    assert is_credential_bearing_key("secret_field") is True

    # Legitimate domain words must NOT be flagged
    assert is_credential_bearing_key("author") is False
    assert is_credential_bearing_key("authenticity") is False
    assert is_credential_bearing_key("keyboard") is False
    assert is_credential_bearing_key("key_findings") is False
    assert is_credential_bearing_key("auth_method") is False
    assert is_credential_bearing_key("author_name") is False


def test_check_no_credential_bearing_keys_nested() -> None:
    # Safe structure
    safe_data = {
        "author": "Alice",
        "key_findings": ["finding1", "finding2"],
        "nested": {"auth_method": "deterministic_check", "count": 10},
    }
    check_no_credential_bearing_keys(safe_data, context="test")

    # Unsafe top-level
    with pytest.raises(ValueError, match="Credential-bearing key 'api_key'"):
        check_no_credential_bearing_keys({"api_key": "secret"}, context="test")

    # Unsafe nested
    with pytest.raises(ValueError, match="Credential-bearing key 'access_token'"):
        check_no_credential_bearing_keys({"level1": {"access_token": "secret"}}, context="test")


def test_freeze_value_deep_immutability() -> None:
    mutable = {"a": [1, 2, {"b": 3}], "c": "text"}
    frozen = freeze_value(mutable)
    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["a"], tuple)
    assert isinstance(frozen["a"][2], MappingProxyType)

    with pytest.raises(TypeError):
        frozen["c"] = "mutated"  # type: ignore[index]


# ==============================================================================
# 3. ArtifactReference Tests
# ==============================================================================


def test_artifact_reference_valid_creation_and_roundtrip() -> None:
    ref = ArtifactReference("p-task-1", "art-100", 2)
    assert ref.producing_task_id == "p-task-1"
    assert ref.artifact_id == "art-100"
    assert ref.artifact_version == 2

    d = ref.to_dict()
    assert d == {
        "artifact_id": "art-100",
        "artifact_version": 2,
        "producing_task_id": "p-task-1",
    }
    from_dict_ref = ArtifactReference.from_dict(d)
    assert from_dict_ref == ref


def test_artifact_reference_validation_errors() -> None:
    with pytest.raises(ValueError, match="producing_task_id must be a non-empty string"):
        ArtifactReference("   ", "art-1", 1)

    with pytest.raises(ValueError, match="artifact_id must be a non-empty string"):
        ArtifactReference("p-1", "   ", 1)

    with pytest.raises(ValueError, match="artifact_version must be >= 1"):
        ArtifactReference("p-1", "art-1", 0)

    with pytest.raises(TypeError, match="artifact_version must be an int"):
        ArtifactReference("p-1", "art-1", True)  # type: ignore[arg-type]

    # Required pinned integer version >= 1: implicit latest resolution prohibited
    with pytest.raises(ValueError, match="artifact_version is required"):
        ArtifactReference.from_dict({"producing_task_id": "p-1", "artifact_id": "art-1"})


# ==============================================================================
# 4. TaskInput Contract Tests
# ==============================================================================


def test_task_input_valid_creation_and_roundtrip() -> None:
    now = datetime.now(UTC).isoformat()
    ref1 = ArtifactReference("p-task-1", "art-1", 1)
    ref2 = ArtifactReference("p-task-1", "art-2", 1)
    record = TaskInput(
        task_id="task-123",
        prompt="Analyze system telemetry",
        created_at=now,
        input_version=1,
        artifact_references=(ref1, ref2),
        metadata={"priority": "high", "domain": "infra", "author": "system"},
        version=1,
    )

    assert record.task_id == "task-123"
    assert record.prompt == "Analyze system telemetry"
    assert record.input_version == 1
    assert record.artifact_references == (ref1, ref2)
    assert record.metadata["priority"] == "high"
    assert record.version == 1

    # Immutability check
    with pytest.raises(TypeError):
        record.metadata["new_key"] = "forbidden"  # type: ignore[index]

    d = record.to_dict()
    assert d["task_id"] == "task-123"
    assert len(d["artifact_references"]) == 2
    assert d["artifact_references"][0]["artifact_id"] == "art-1"
    assert d["input_version"] == 1

    from_dict_record = TaskInput.from_dict(d)
    assert from_dict_record == record

    from_json_record = TaskInput.from_json(record.to_json())
    assert from_json_record == record


def test_task_input_create_factory() -> None:
    ref = ArtifactReference("task-456", "art-abc", 1)
    record = TaskInput.create(
        task_id="task-456",
        prompt="Execute evaluation benchmark",
        input_version=2,
        artifact_references=[ref],
    )
    assert record.task_id == "task-456"
    assert record.input_version == 2
    assert record.artifact_references == (ref,)
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


def test_task_input_credential_detection() -> None:
    now = datetime.now(UTC).isoformat()
    # Forbidden key
    with pytest.raises(ValueError, match="Credential-bearing key 'api_key'"):
        TaskInput(
            task_id="task-1",
            prompt="hello",
            created_at=now,
            metadata={"api_key": "confidential"},
        )

    # Allowed domain keys containing 'key' or 'auth'
    record = TaskInput(
        task_id="task-1",
        prompt="hello",
        created_at=now,
        metadata={"author": "alice", "keyboard": "us", "auth_method": "deterministic"},
    )
    assert record.metadata["author"] == "alice"


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
# 5. StageArtifact Contract Tests
# ==============================================================================


def test_stage_artifact_valid_creation_and_roundtrip() -> None:
    now = datetime.now(UTC).isoformat()
    artifact = StageArtifact(
        artifact_id="art-999",
        producing_task_id="task-parent-1",
        producing_turn_id="turn-claim-1",
        artifact_type="research_material",
        artifact_version=1,
        payload_schema_version=1,
        created_at=now,
        sealed=True,
        content={"material": "verified research material content"},
        metadata={"schema_version": "v1.0"},
        version=1,
    )

    assert artifact.artifact_id == "art-999"
    assert artifact.producing_task_id == "task-parent-1"
    assert artifact.producing_turn_id == "turn-claim-1"
    assert artifact.sealed is True
    assert artifact.artifact_type == "research_material"
    assert artifact.payload_schema_version == 1
    assert artifact.content == {"material": "verified research material content"}

    # Deep immutability
    with pytest.raises(TypeError):
        artifact.content["new_key"] = "fail"  # type: ignore[index]

    d = artifact.to_dict()
    assert d["artifact_id"] == "art-999"
    assert d["sealed"] is True
    assert d["artifact_type"] == "research_material"
    assert d["payload_schema_version"] == 1

    from_dict_artifact = StageArtifact.from_dict(d)
    assert from_dict_artifact == artifact

    from_json_artifact = StageArtifact.from_json(artifact.to_json())
    assert from_json_artifact == artifact


def test_stage_artifact_payload_schema_version_enforced() -> None:
    now = datetime.now(UTC).isoformat()

    # payload_schema_version != 1 rejected
    with pytest.raises(ValueError, match="Unsupported payload_schema_version: 2"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="task-1",
            artifact_type="research_material",
            content={"material": "data"},
            payload_schema_version=2,
            created_at=now,
        )

    # non-int payload_schema_version rejected
    with pytest.raises(TypeError, match="payload_schema_version must be an int"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="task-1",
            artifact_type="research_material",
            content={"material": "data"},
            payload_schema_version=True,  # type: ignore[arg-type]
            created_at=now,
        )


def test_stage_artifact_content_schemas_and_allowlisted_types() -> None:
    now = datetime.now(UTC).isoformat()

    assert "research_material" in ALLOWED_ARTIFACT_TYPES
    assert "analysis_summary" in ALLOWED_ARTIFACT_TYPES

    # research_material requires non-empty material string and rejects any other keys
    res_art = StageArtifact(
        artifact_id="art-res",
        producing_task_id="task-1",
        artifact_type="research_material",
        content={"material": "valid research material"},
        created_at=now,
    )
    assert res_art.content["material"] == "valid research material"

    with pytest.raises(ValueError, match="research_material content must contain only 'material' field"):
        StageArtifact(
            artifact_id="art-res-bad",
            producing_task_id="task-1",
            artifact_type="research_material",
            content={"material": "text", "extra_field": "disallowed"},
            created_at=now,
        )

    with pytest.raises(ValueError, match="research_material content 'material' must be a non-empty string"):
        StageArtifact(
            artifact_id="art-res-empty",
            producing_task_id="task-1",
            artifact_type="research_material",
            content={"material": "   "},
            created_at=now,
        )

    # analysis_summary requires non-empty summary string and rejects any other keys
    ana_art = StageArtifact(
        artifact_id="art-ana",
        producing_task_id="task-1",
        artifact_type="analysis_summary",
        content={"summary": "valid analysis summary"},
        created_at=now,
    )
    assert ana_art.content["summary"] == "valid analysis summary"

    with pytest.raises(ValueError, match="analysis_summary content must contain only 'summary' field"):
        StageArtifact(
            artifact_id="art-ana-bad",
            producing_task_id="task-1",
            artifact_type="analysis_summary",
            content={"summary": "text", "extra_field": "disallowed"},
            created_at=now,
        )

    with pytest.raises(ValueError, match="analysis_summary content 'summary' must be a non-empty string"):
        StageArtifact(
            artifact_id="art-ana-empty",
            producing_task_id="task-1",
            artifact_type="analysis_summary",
            content={"summary": ""},
            created_at=now,
        )

    # Unallowlisted artifact type
    with pytest.raises(ValueError, match="Unknown artifact_type"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="task-1",
            artifact_type="arbitrary_unallowlisted_type",
            content={"material": "data"},
            created_at=now,
        )


def test_stage_artifact_content_must_be_non_empty_dict() -> None:
    now = datetime.now(UTC).isoformat()

    # Non-dict content rejected
    with pytest.raises(TypeError, match="content must be a dict"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="task-1",
            artifact_type="research_material",
            content="string_content",  # type: ignore[arg-type]
            created_at=now,
        )

    # Empty dict rejected
    with pytest.raises(ValueError, match="content must not be empty"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="task-1",
            artifact_type="research_material",
            content={},
            created_at=now,
        )


def test_stage_artifact_size_bound_64kib() -> None:
    now = datetime.now(UTC).isoformat()
    assert MAX_STAGE_ARTIFACT_BYTES == 65536

    # Under 64 KiB
    valid = StageArtifact(
        artifact_id="art-1",
        producing_task_id="task-1",
        artifact_type="research_material",
        content={"material": "a" * 1000},
        created_at=now,
    )
    assert valid.byte_size() <= MAX_STAGE_ARTIFACT_BYTES

    # Over 64 KiB
    with pytest.raises(ValueError, match="exceeds maximum allowed bound of 65536 bytes"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="task-1",
            artifact_type="research_material",
            content={"material": "a" * 66000},
            created_at=now,
        )


def test_stage_artifact_credential_detection() -> None:
    now = datetime.now(UTC).isoformat()

    # Metadata with credential key
    with pytest.raises(ValueError, match="Credential-bearing key 'access_token'"):
        StageArtifact(
            artifact_id="art-1",
            producing_task_id="task-1",
            artifact_type="research_material",
            content={"material": "content"},
            metadata={"access_token": "leak"},
            created_at=now,
        )


# ==============================================================================
# 6. TaskContinuation Contract Tests
# ==============================================================================


def test_task_continuation_valid_creation_and_roundtrip() -> None:
    now = datetime.now(UTC).isoformat()
    ref = ArtifactReference("parent-123", "art-1", 1)
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
        artifact_references=(ref,),
        continuation_version=1,
        created_at=now,
        updated_at=now,
        version=1,
    )

    assert continuation.parent_task_id == "parent-123"
    assert continuation.child_task_id == "child-456"
    assert continuation.stage == "incorporate_analysis_child"
    assert continuation.continuation_version == 1
    assert continuation.byte_size() <= MAX_CONTINUATION_BYTES

    # Deep immutability
    with pytest.raises(TypeError):
        continuation.control_fields["extra"] = "fail"  # type: ignore[index]

    d = continuation.to_dict()
    assert d["parent_task_id"] == "parent-123"
    assert d["control_fields"]["min_findings"] == 3
    assert len(d["artifact_references"]) == 1
    assert d["artifact_references"][0]["producing_task_id"] == "parent-123"

    from_dict_cont = TaskContinuation.from_dict(d)
    assert from_dict_cont == continuation

    from_json_cont = TaskContinuation.from_json(continuation.to_json())
    assert from_json_cont == continuation


def test_task_continuation_structural_artifact_references_accepted() -> None:
    now = datetime.now(UTC).isoformat()
    # ArtifactReference validates structure only; contextual cross-task authorization belongs to #221
    foreign_ref = ArtifactReference("other-task-999", "art-1", 1)
    continuation = TaskContinuation(
        parent_task_id="parent-123",
        child_task_id="child-456",
        control_fields={"analysis_focus": "test"},
        artifact_references=(foreign_ref,),
        created_at=now,
        updated_at=now,
    )
    assert continuation.artifact_references == (foreign_ref,)


def test_task_continuation_stage_and_control_fields_allowlist() -> None:
    now = datetime.now(UTC).isoformat()

    assert "incorporate_analysis_child" in ALLOWED_CONTINUATION_STAGES
    assert "incorporate_analysis_child" in STAGE_CONTROL_FIELDS_ALLOWLIST

    with pytest.raises(ValueError, match="Unknown continuation stage"):
        TaskContinuation(
            parent_task_id="parent-1",
            child_task_id="child-1",
            stage="arbitrary_unapproved_stage",
            created_at=now,
            updated_at=now,
        )

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


def test_task_continuation_size_bound_16kib() -> None:
    now = datetime.now(UTC).isoformat()
    assert MAX_CONTINUATION_BYTES == 16384

    # Fits within 16 KiB
    valid = TaskContinuation(
        parent_task_id="parent-1",
        child_task_id="child-1",
        control_fields={"material_summary": "x" * 500},
        created_at=now,
        updated_at=now,
    )
    assert valid.byte_size() <= MAX_CONTINUATION_BYTES

    # Exceeds 16 KiB
    with pytest.raises(ValueError, match="exceeds maximum bound of 16384 bytes"):
        TaskContinuation(
            parent_task_id="parent-1",
            child_task_id="child-1",
            control_fields={"material_summary": "x" * 17000},
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


# ==============================================================================
# 7. RootAuthorityRecord Contract Tests
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
        created_at=now,
        version=1,
    )

    assert auth.root_task_id == "root-100"
    assert auth.max_delegation_depth == 1
    assert auth.max_delegation_waves == 1
    assert auth.max_children_per_wave == 1
    assert auth.allowed_delegation_edges == (("research", "analysis"),)

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
    assert auth.version == 1


def test_root_authority_baseline_invariants_enforced() -> None:
    now = datetime.now(UTC).isoformat()

    # policy_context != 'default'
    with pytest.raises(ValueError, match="policy_context must be 'default'"):
        RootAuthorityRecord(
            root_task_id="root-1",
            policy_context="escalated_context",
            created_at=now,
        )

    # max_delegation_depth != 1
    with pytest.raises(ValueError, match="max_delegation_depth must be 1"):
        RootAuthorityRecord(
            root_task_id="root-1",
            max_delegation_depth=2,
            created_at=now,
        )

    # max_delegation_waves != 1
    with pytest.raises(ValueError, match="max_delegation_waves must be 1"):
        RootAuthorityRecord(
            root_task_id="root-1",
            max_delegation_waves=2,
            created_at=now,
        )

    # max_children_per_wave != 1
    with pytest.raises(ValueError, match="max_children_per_wave must be 1"):
        RootAuthorityRecord(
            root_task_id="root-1",
            max_children_per_wave=2,
            created_at=now,
        )

    # allowed_delegation_edges != (("research", "analysis"),)
    with pytest.raises(ValueError, match="allowed_delegation_edges must be"):
        RootAuthorityRecord(
            root_task_id="root-1",
            allowed_delegation_edges=(("general", "analysis"),),
            created_at=now,
        )
