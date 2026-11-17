import base64
import re
from datetime import UTC, datetime

import pytest
from agent_contracts import (
    ACTIVE_CLAIM_CONTRACT_VERSION,
    DELEGATION_GRANT_CONTRACT_VERSION,
    TASK_ASSIGNMENT_CONTRACT_VERSION,
    WAKE_UP_HINT_CONTRACT_VERSION,
    ActiveClaim,
    ArtifactReference,
    DelegationGrant,
    TaskAssignment,
    TaskInputReference,
    WakeUpHint,
    normalize_canonical_utc_timestamp,
)
from orchestrator.domain.principal import (
    compute_assignment_verifier_digest,
    derive_assignment_proof,
    derive_proof_subkeys,
    encode_canonical_proof_identity,
    parse_master_secret,
    verify_assignment_proof,
)


def test_delegation_grant_valid_and_canonical() -> None:
    now = datetime.now(UTC).isoformat()
    grant = DelegationGrant(
        grant_id="grant-101",
        root_task_id="root-task-1",
        issued_to_assignment_id="assign-001",
        issued_to_principal_id="principal-research-1",
        fencing_epoch=1,
        turn_id="turn-001",
        issued_at=now,
    )
    assert grant.source_capability == "research"
    assert grant.target_capability == "analysis"
    assert grant.max_children == 1
    assert grant.max_depth == 1
    assert grant.max_waves == 1
    assert grant.fencing_epoch == 1
    assert grant.turn_id == "turn-001"
    assert grant.version == DELEGATION_GRANT_CONTRACT_VERSION

    payload_json = grant.to_json()
    round_trip = DelegationGrant.from_json(payload_json)
    assert round_trip == grant
    assert round_trip.to_dict() == grant.to_dict()


def test_delegation_grant_invariants() -> None:
    now = datetime.now(UTC).isoformat()
    # Invalid source capability
    with pytest.raises(ValueError, match="source_capability must be 'research'"):
        DelegationGrant(
            grant_id="g1",
            root_task_id="r1",
            issued_to_assignment_id="a1",
            issued_to_principal_id="p1",
            fencing_epoch=1,
            turn_id="turn-1",
            issued_at=now,
            source_capability="general",
        )

    # Invalid target capability
    with pytest.raises(ValueError, match="target_capability must be 'analysis'"):
        DelegationGrant(
            grant_id="g1",
            root_task_id="r1",
            issued_to_assignment_id="a1",
            issued_to_principal_id="p1",
            fencing_epoch=1,
            turn_id="turn-1",
            issued_at=now,
            target_capability="research",
        )

    # Invalid max_children
    with pytest.raises(ValueError, match="max_children must be 1"):
        DelegationGrant(
            grant_id="g1",
            root_task_id="r1",
            issued_to_assignment_id="a1",
            issued_to_principal_id="p1",
            fencing_epoch=1,
            turn_id="turn-1",
            issued_at=now,
            max_children=2,
        )

    # Invalid version
    with pytest.raises(ValueError, match="Unsupported contract version"):
        DelegationGrant(
            grant_id="g1",
            root_task_id="r1",
            issued_to_assignment_id="a1",
            issued_to_principal_id="p1",
            fencing_epoch=1,
            turn_id="turn-1",
            issued_at=now,
            version=2,
        )

    # Invalid fencing_epoch < 1
    with pytest.raises(ValueError, match="fencing_epoch must be >= 1"):
        DelegationGrant(
            grant_id="g1",
            root_task_id="r1",
            issued_to_assignment_id="a1",
            issued_to_principal_id="p1",
            fencing_epoch=0,
            turn_id="turn-1",
            issued_at=now,
        )

    # Invalid fencing_epoch bool
    with pytest.raises(TypeError, match="fencing_epoch must be an int"):
        DelegationGrant(
            grant_id="g1",
            root_task_id="r1",
            issued_to_assignment_id="a1",
            issued_to_principal_id="p1",
            fencing_epoch=True,  # type: ignore[arg-type]
            turn_id="turn-1",
            issued_at=now,
        )

    # Invalid turn_id empty
    with pytest.raises(ValueError, match="turn_id must be a non-empty string"):
        DelegationGrant(
            grant_id="g1",
            root_task_id="r1",
            issued_to_assignment_id="a1",
            issued_to_principal_id="p1",
            fencing_epoch=1,
            turn_id="   ",
            issued_at=now,
        )


def test_task_input_reference() -> None:
    ref = TaskInputReference(task_id="task-123", input_version=1)
    assert ref.task_id == "task-123"
    assert ref.input_version == 1

    d = ref.to_dict()
    assert d == {"input_version": 1, "task_id": "task-123"}
    assert TaskInputReference.from_dict(d) == ref
    assert TaskInputReference.from_json(ref.to_json()) == ref

    with pytest.raises(ValueError, match="input_version must be >= 1"):
        TaskInputReference(task_id="task-1", input_version=0)
    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        TaskInputReference(task_id="   ", input_version=1)


def test_task_assignment_valid_and_canonical() -> None:
    now = datetime.now(UTC).isoformat()
    input_ref = TaskInputReference(task_id="task-root-1", input_version=1)
    artifact_ref = ArtifactReference(
        producing_task_id="task-root-1",
        artifact_id="art-1",
        artifact_version=1,
    )

    assignment = TaskAssignment(
        assignment_id="assign-001",
        claim_id="claim-001",
        task_id="task-root-1",
        root_task_id="task-root-1",
        turn_id="turn-001",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        execution_kind="initial",
        assigned_capability="research",
        fencing_epoch=1,
        claimed_at=now,
        delivery_expires_at=now,
        task_input_reference=input_ref,
        continuation_version=None,
        pinned_artifact_references=(artifact_ref,),
        delegation_grant_id="grant-001",
    )

    assert assignment.version == TASK_ASSIGNMENT_CONTRACT_VERSION
    assert assignment.continuation_version is None
    assert assignment.fencing_epoch == 1
    assert assignment.delegation_grant_id == "grant-001"

    # Verify no acknowledged_at or assignment_proof fields exist on TaskAssignment
    assert not hasattr(assignment, "acknowledged_at")
    assert not hasattr(assignment, "assignment_proof")

    raw_json = assignment.to_json()
    round_trip = TaskAssignment.from_json(raw_json)
    assert round_trip == assignment
    assert round_trip.to_dict() == assignment.to_dict()


def test_task_assignment_validation() -> None:
    now = datetime.now(UTC).isoformat()
    input_ref = TaskInputReference(task_id="task-root-1", input_version=1)

    # Invalid execution_kind
    with pytest.raises(ValueError, match="Invalid execution_kind"):
        TaskAssignment(
            assignment_id="a1",
            claim_id="c1",
            task_id="t1",
            root_task_id="r1",
            turn_id="turn1",
            runtime_principal_id="p1",
            bound_agent_id="ag1",
            execution_kind="arbitrary_kind",
            assigned_capability="research",
            fencing_epoch=1,
            claimed_at=now,
            delivery_expires_at=now,
            task_input_reference=input_ref,
        )

    # Invalid fencing_epoch < 1
    with pytest.raises(ValueError, match="fencing_epoch must be >= 1"):
        TaskAssignment(
            assignment_id="a1",
            claim_id="c1",
            task_id="t1",
            root_task_id="r1",
            turn_id="turn1",
            runtime_principal_id="p1",
            bound_agent_id="ag1",
            execution_kind="initial",
            assigned_capability="research",
            fencing_epoch=0,
            claimed_at=now,
            delivery_expires_at=now,
            task_input_reference=input_ref,
        )

    # Invalid continuation_version < 1 when set
    with pytest.raises(ValueError, match="continuation_version must be >= 1"):
        TaskAssignment(
            assignment_id="a1",
            claim_id="c1",
            task_id="t1",
            root_task_id="r1",
            turn_id="turn1",
            runtime_principal_id="p1",
            bound_agent_id="ag1",
            execution_kind="initial",
            assigned_capability="research",
            fencing_epoch=1,
            claimed_at=now,
            delivery_expires_at=now,
            task_input_reference=input_ref,
            continuation_version=0,
        )


def test_active_claim_lifecycle_and_validation() -> None:
    now = datetime.now(UTC).isoformat()
    claim = ActiveClaim(
        task_id="task-root-1",
        assignment_id="assign-001",
        claim_id="claim-001",
        root_task_id="task-root-1",
        turn_id="turn-001",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="hmac-sha256-proof-digest-hex",
        request_id="req-123",
        claimed_at=now,
        delivery_expires_at=now,
        state="claimed_unacknowledged",
    )

    assert claim.state == "claimed_unacknowledged"
    assert claim.fencing_epoch == 1
    assert claim.acknowledged_at is None
    assert claim.version == ACTIVE_CLAIM_CONTRACT_VERSION

    # Serialization roundtrip
    round_trip = ActiveClaim.from_json(claim.to_json())
    assert round_trip == claim
    assert round_trip.to_dict() == claim.to_dict()

    # Acknowledged state
    ack_claim = ActiveClaim(
        task_id="task-root-1",
        assignment_id="assign-001",
        claim_id="claim-001",
        root_task_id="task-root-1",
        turn_id="turn-001",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="hmac-sha256-proof-digest-hex",
        request_id="req-123",
        claimed_at=now,
        delivery_expires_at=now,
        state="acknowledged",
        acknowledged_at=now,
    )
    assert ack_claim.state == "acknowledged"
    assert ack_claim.acknowledged_at == normalize_canonical_utc_timestamp(now)

    # Invalid state
    with pytest.raises(ValueError, match="Invalid claim state"):
        ActiveClaim(
            task_id="task-1",
            assignment_id="a1",
            claim_id="c1",
            root_task_id="r1",
            turn_id="t1",
            runtime_principal_id="p1",
            bound_agent_id="ag1",
            assigned_capability="general",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="digest",
            request_id="req-1",
            claimed_at=now,
            delivery_expires_at=now,
            state="invalid_state",
        )


def test_wake_up_hint() -> None:
    now = datetime.now(UTC).isoformat()
    # Capability-level hint without task_id
    hint_cap = WakeUpHint(
        hint_id="hint-001",
        capability="research",
        created_at=now,
        execution_kind="initial",
    )
    assert hint_cap.task_id is None
    assert hint_cap.capability == "research"
    assert hint_cap.version == WAKE_UP_HINT_CONTRACT_VERSION
    assert WakeUpHint.from_json(hint_cap.to_json()) == hint_cap

    # Task-scoped hint
    hint_task = WakeUpHint(
        hint_id="hint-002",
        capability="research",
        created_at=now,
        execution_kind="resumed_continuation",
        task_id="task-456",
        expected_continuation_version=1,
    )
    assert hint_task.task_id == "task-456"
    assert hint_task.expected_continuation_version == 1
    assert WakeUpHint.from_json(hint_task.to_json()) == hint_task


def test_parse_master_secret_and_derive_subkeys() -> None:
    # 1. Base64 encoded secret >= 32 bytes
    raw_b64 = base64.b64encode(b"01234567890123456789012345678901").decode("ascii")
    parsed_1 = parse_master_secret(raw_b64)
    assert len(parsed_1) == 32

    # 2. Raw string >= 32 bytes
    parsed_2 = parse_master_secret("a" * 32)
    assert len(parsed_2) == 32

    # 3. Raw bytes >= 32 bytes
    parsed_3 = parse_master_secret(b"b" * 32)
    assert len(parsed_3) == 32

    # 4. Short strings / bytes fail closed
    with pytest.raises(ValueError, match="at least 32 bytes"):
        parse_master_secret("short")
    with pytest.raises(ValueError, match="at least 32 bytes"):
        parse_master_secret(b"short")
    with pytest.raises(ValueError, match="non-empty string"):
        parse_master_secret("   ")

    # 5. Subkey derivation produces 32-byte distinct keys
    token_key, verifier_key = derive_proof_subkeys(parsed_1)
    assert len(token_key) == 32
    assert len(verifier_key) == 32
    assert token_key != verifier_key


def test_deterministic_assignment_proof_and_verifier_lifecycle() -> None:
    master_secret = parse_master_secret("a" * 32)
    token_key, verifier_key = derive_proof_subkeys(master_secret)

    identity_1 = encode_canonical_proof_identity(
        assignment_id="assign-001",
        claim_id="claim-001",
        task_id="task-001",
        root_task_id="task-001",
        turn_id="turn-001",
        runtime_principal_id="principal-1",
        bound_agent_id="agent-1",
        assigned_capability="research",
        execution_kind="initial",
        continuation_version=None,
        fencing_epoch=1,
        request_id="req-001",
        delivery_expires_at="2099-01-01T00:00:00Z",
        delegation_grant_id=None,
    )

    proof_1 = derive_assignment_proof(token_key, identity_1)
    assert len(proof_1) == 64
    assert re.match(r"^[0-9a-f]{64}$", proof_1)

    # Identical input produces identical proof (deterministic)
    proof_1_repeat = derive_assignment_proof(token_key, identity_1)
    assert proof_1_repeat == proof_1

    verifier_digest_1 = compute_assignment_verifier_digest(verifier_key, proof_1)
    assert len(verifier_digest_1) == 64
    assert re.match(r"^[0-9a-f]{64}$", verifier_digest_1)

    # Verifier matches valid proof
    assert verify_assignment_proof(verifier_key, proof_1, verifier_digest_1) is True

    # Verifier rejects tampered/wrong proof
    wrong_proof = "0" * 64
    assert verify_assignment_proof(verifier_key, wrong_proof, verifier_digest_1) is False

    # Verifier rejects malformed proof
    assert verify_assignment_proof(verifier_key, "not-a-hex", verifier_digest_1) is False

    # Different epoch produces different identity and different proof
    identity_epoch_2 = encode_canonical_proof_identity(
        assignment_id="assign-002",
        claim_id="claim-002",
        task_id="task-001",
        root_task_id="task-001",
        turn_id="turn-002",
        runtime_principal_id="principal-1",
        bound_agent_id="agent-1",
        assigned_capability="research",
        execution_kind="initial",
        continuation_version=None,
        fencing_epoch=2,
        request_id="req-002",
        delivery_expires_at="2099-01-01T00:00:00Z",
        delegation_grant_id=None,
    )
    proof_2 = derive_assignment_proof(token_key, identity_epoch_2)
    assert proof_2 != proof_1


def test_canonical_utc_timestamp_normalization_and_comparison_safety() -> None:
    # 1. Normalization from datetime with UTC
    dt_now = datetime(2026, 6, 15, 12, 30, 45, 123456, tzinfo=UTC)
    norm_dt = normalize_canonical_utc_timestamp(dt_now)
    assert norm_dt == "2026-06-15T12:30:45.123456Z"
    assert len(norm_dt) == 27

    # 2. Normalization from ISO string with Z
    assert normalize_canonical_utc_timestamp("2026-01-01T12:00:00Z") == "2026-01-01T12:00:00.000000Z"

    # 3. Normalization from ISO string with +00:00
    assert normalize_canonical_utc_timestamp("2026-01-01T12:00:00+00:00") == "2026-01-01T12:00:00.000000Z"

    # 4. Normalization from ISO string with non-zero timezone offset
    assert normalize_canonical_utc_timestamp("2026-01-01T17:00:00+05:00") == "2026-01-01T12:00:00.000000Z"

    # 5. Preserves microsecond precision
    assert normalize_canonical_utc_timestamp("2026-01-01T12:00:00.123456Z") == "2026-01-01T12:00:00.123456Z"

    # 6. Expands milliseconds to 6-digit microseconds
    assert normalize_canonical_utc_timestamp("2026-01-01T12:00:00.123Z") == "2026-01-01T12:00:00.123000Z"

    # 7. Naive datetime raises ValueError
    naive_dt = datetime(2026, 1, 1, 12, 0, 0)
    with pytest.raises(ValueError, match="Naive timestamps without timezone are not permitted"):
        normalize_canonical_utc_timestamp(naive_dt)

    # 8. Naive ISO string raises ValueError
    with pytest.raises(ValueError, match="Naive timestamps without timezone are not permitted"):
        normalize_canonical_utc_timestamp("2026-01-01T12:00:00")

    # 9. Invalid ISO format raises ValueError
    with pytest.raises(ValueError, match="Invalid ISO timestamp"):
        normalize_canonical_utc_timestamp("not-an-iso-timestamp")

    # 10. Empty string raises ValueError
    with pytest.raises(ValueError, match="Timestamp string cannot be empty"):
        normalize_canonical_utc_timestamp("   ")

    # 11. Non-string, non-datetime raises TypeError
    with pytest.raises(TypeError, match="Expected str or datetime"):
        normalize_canonical_utc_timestamp(123456789)  # type: ignore[arg-type]

    # 12. Mathematical proof of safe lexicographical comparison across mixed formats
    # Mixed formats (+00:00 vs Z) without normalization:
    raw_earlier_plus = "2026-01-01T12:00:00+00:00"
    raw_exact_z = "2026-01-01T12:00:00Z"
    # In raw ASCII, '+' (43) < 'Z' (90), so raw_earlier_plus < raw_exact_z even though chronologically equal!
    assert (raw_earlier_plus < raw_exact_z) is True

    # With canonical normalization:
    canon_earlier = normalize_canonical_utc_timestamp(raw_earlier_plus)
    canon_exact = normalize_canonical_utc_timestamp(raw_exact_z)
    canon_later = normalize_canonical_utc_timestamp("2026-01-01T12:00:00.000001Z")

    assert canon_earlier == canon_exact == "2026-01-01T12:00:00.000000Z"
    # Strict lexicographical ordering matches chronological ordering:
    assert (canon_earlier <= canon_exact) is True
    assert (canon_exact <= canon_earlier) is True
    assert (canon_earlier < canon_later) is True
    assert (canon_later > canon_earlier) is True

    # 13. ActiveClaim automatically normalizes timestamps upon construction
    claim = ActiveClaim(
        task_id="t1",
        assignment_id="a1",
        claim_id="c1",
        root_task_id="r1",
        turn_id="turn1",
        runtime_principal_id="p1",
        bound_agent_id="ag1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="a" * 64,
        request_id="req1",
        claimed_at="2026-01-01T12:00:00+00:00",
        delivery_expires_at="2026-01-01T12:00:30Z",
        acknowledged_at="2026-01-01T12:00:10+05:00",
        state="acknowledged",
    )
    assert claim.claimed_at == "2026-01-01T12:00:00.000000Z"
    assert claim.delivery_expires_at == "2026-01-01T12:00:30.000000Z"
    assert claim.acknowledged_at == "2026-01-01T07:00:10.000000Z"

    # ActiveClaim rejects naive timestamp
    with pytest.raises(ValueError, match="Naive timestamps without timezone"):
        ActiveClaim(
            task_id="t1",
            assignment_id="a1",
            claim_id="c1",
            root_task_id="r1",
            turn_id="turn1",
            runtime_principal_id="p1",
            bound_agent_id="ag1",
            assigned_capability="research",
            execution_kind="initial",
            fencing_epoch=1,
            proof_digest="a" * 64,
            request_id="req1",
            claimed_at="2026-01-01 12:00:00",
            delivery_expires_at="2026-01-01T12:00:30Z",
        )
