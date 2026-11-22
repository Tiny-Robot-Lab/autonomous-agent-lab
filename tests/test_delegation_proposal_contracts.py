import os

import pytest
from agent_contracts import (
    ArtifactReference,
    AssignmentYieldRecord,
    DelegationProposalRecord,
    DelegationProposalRequest,
    DelegationProposalResponse,
    compute_proposal_fingerprint,
)
from orchestrator.config import Settings


def test_delegation_proposal_request_roundtrip():
    ref = ArtifactReference(
        artifact_id="art-1",
        artifact_version=1,
        producing_task_id="task-root-1",
    )
    req = DelegationProposalRequest(
        proposal_id="prop-1",
        child_intent="Analyze the research material for anomalies",
        artifact_reference=ref,
        target_capability="analysis",
        parent_input_version=1,
        control_fields={"analysis_focus": "anomalies", "min_findings": 1},
    )

    d = req.to_dict()
    assert d["proposal_id"] == "prop-1"
    assert d["target_capability"] == "analysis"
    assert d["child_intent"] == "Analyze the research material for anomalies"
    assert d["artifact_reference"]["artifact_id"] == "art-1"
    assert d["control_fields"]["analysis_focus"] == "anomalies"

    json_str = req.to_json()
    reconstructed = DelegationProposalRequest.from_json(json_str)
    assert reconstructed == req


def test_delegation_proposal_request_validations():
    ref = ArtifactReference(
        artifact_id="art-1",
        artifact_version=1,
        producing_task_id="task-root-1",
    )

    # Empty proposal_id
    with pytest.raises(ValueError, match="proposal_id must be a non-empty string"):
        DelegationProposalRequest(
            proposal_id="   ",
            child_intent="valid intent",
            artifact_reference=ref,
        )

    # Target capability not analysis
    with pytest.raises(ValueError, match="Unsupported target_capability 'research'"):
        DelegationProposalRequest(
            proposal_id="prop-1",
            child_intent="valid intent",
            artifact_reference=ref,
            target_capability="research",
        )

    # Empty child_intent
    with pytest.raises(ValueError, match="child_intent must not be empty"):
        DelegationProposalRequest(
            proposal_id="prop-1",
            child_intent="",
            artifact_reference=ref,
        )

    # child_intent exceeding 2000 characters
    with pytest.raises(ValueError, match="child_intent must not exceed 2000 characters"):
        DelegationProposalRequest(
            proposal_id="prop-1",
            child_intent="x" * 2001,
            artifact_reference=ref,
        )

    # Invalid artifact_reference type
    with pytest.raises(TypeError, match="artifact_reference must be an ArtifactReference"):
        DelegationProposalRequest(
            proposal_id="prop-1",
            child_intent="valid intent",
            artifact_reference="not-a-ref",  # type: ignore[arg-type]
        )

    # Invalid parent_input_version
    with pytest.raises(ValueError, match="parent_input_version must be >= 1"):
        DelegationProposalRequest(
            proposal_id="prop-1",
            child_intent="valid intent",
            artifact_reference=ref,
            parent_input_version=0,
        )

    # Unexpected control fields
    with pytest.raises(ValueError, match="Unexpected control_fields"):
        DelegationProposalRequest(
            proposal_id="prop-1",
            child_intent="valid intent",
            artifact_reference=ref,
            control_fields={"unsupported_field": "val"},
        )


def test_delegation_proposal_request_rejects_forbidden_authority_fields():
    ref_dict = {
        "artifact_id": "art-1",
        "artifact_version": 1,
        "producing_task_id": "task-root-1",
    }
    raw_payload = {
        "proposal_id": "prop-1",
        "child_intent": "intent",
        "artifact_reference": ref_dict,
        "target_capability": "analysis",
        "child_task_id": "attempt-to-forge-child-id",
    }
    with pytest.raises(ValueError, match="Forbidden authority fields supplied by caller"):
        DelegationProposalRequest.from_dict(raw_payload)


def test_compute_proposal_fingerprint_deterministic_and_sensitive():
    ref = ArtifactReference(
        artifact_id="art-1",
        artifact_version=1,
        producing_task_id="task-root-1",
    )
    fp1 = compute_proposal_fingerprint(
        parent_task_id="parent-1",
        proposal_id="prop-1",
        assignment_id="asgn-1",
        target_capability="analysis",
        child_intent="intent 1",
        artifact_reference=ref,
        parent_input_version=1,
        control_fields={"analysis_focus": "data"},
    )
    fp2 = compute_proposal_fingerprint(
        parent_task_id="parent-1",
        proposal_id="prop-1",
        assignment_id="asgn-1",
        target_capability="analysis",
        child_intent="intent 1",
        artifact_reference=ref,
        parent_input_version=1,
        control_fields={"analysis_focus": "data"},
    )
    assert fp1 == fp2

    # Change assignment_id
    fp3 = compute_proposal_fingerprint(
        parent_task_id="parent-1",
        proposal_id="prop-1",
        assignment_id="asgn-2",
        target_capability="analysis",
        child_intent="intent 1",
        artifact_reference=ref,
        parent_input_version=1,
        control_fields={"analysis_focus": "data"},
    )
    assert fp1 != fp3


def test_delegation_proposal_record_roundtrip():
    rec = DelegationProposalRecord(
        parent_task_id="parent-1",
        proposal_id="prop-1",
        child_task_id="child-1",
        delegation_decision_id="dec-1",
        assignment_id="asgn-1",
        runtime_principal_id="research-agent-1",
        bound_agent_id="research-agent-1",
        proof_digest="digest-123",
        proposal_fingerprint="fp-456",
        delegation_grant_id="grant-1",
        child_claim_deadline="2026-11-21T00:02:00.000000Z",
        accepted_at="2026-11-21T00:00:00.000000Z",
        target_capability="analysis",
        continuation_version=1,
    )
    d = rec.to_dict()
    assert d["parent_task_id"] == "parent-1"
    assert d["child_task_id"] == "child-1"
    assert d["bound_agent_id"] == "research-agent-1"

    rec2 = DelegationProposalRecord.from_dict(d)
    assert rec == rec2
    assert DelegationProposalRecord.from_json(rec.to_json()) == rec

    # Validation of bound_agent_id
    with pytest.raises(ValueError, match="bound_agent_id must be a non-empty string"):
        DelegationProposalRecord(
            parent_task_id="parent-1",
            proposal_id="prop-1",
            child_task_id="child-1",
            delegation_decision_id="dec-1",
            assignment_id="asgn-1",
            runtime_principal_id="research-agent-1",
            bound_agent_id="",
            proof_digest="digest-123",
            proposal_fingerprint="fp-456",
            delegation_grant_id="grant-1",
            child_claim_deadline="2026-11-21T00:02:00.000000Z",
            accepted_at="2026-11-21T00:00:00.000000Z",
        )


def test_assignment_yield_record_roundtrip():
    yield_rec = AssignmentYieldRecord(
        assignment_id="asgn-1",
        parent_task_id="parent-1",
        principal_id="research-agent-1",
        proposal_id="prop-1",
        child_task_id="child-1",
        delegation_decision_id="dec-1",
        continuation_version=1,
        yielded_at="2026-11-21T00:00:00.000000Z",
    )
    d = yield_rec.to_dict()
    assert d["yield_reason"] == "delegation_proposal"
    assert d["assignment_id"] == "asgn-1"

    yield_rec2 = AssignmentYieldRecord.from_dict(d)
    assert yield_rec == yield_rec2
    assert AssignmentYieldRecord.from_json(yield_rec.to_json()) == yield_rec


def test_delegation_proposal_response_roundtrip():
    resp = DelegationProposalResponse(
        parent_task_id="parent-1",
        proposal_id="prop-1",
        child_task_id="child-1",
        delegation_decision_id="dec-1",
        child_claim_deadline="2026-11-21T00:02:00.000000Z",
        accepted_at="2026-11-21T00:00:00.000000Z",
        target_capability="analysis",
        continuation_version=1,
        status="accepted",
    )
    d = resp.to_dict()
    assert d["status"] == "accepted"
    assert DelegationProposalResponse.from_dict(d) == resp
    assert DelegationProposalResponse.from_json(resp.to_json()) == resp


def test_settings_child_claim_deadline():
    s = Settings()
    assert s.child_claim_deadline_seconds == 120.0

    s_custom = Settings(child_claim_deadline_seconds=300.0)
    assert s_custom.child_claim_deadline_seconds == 300.0

    # Bounds: [10.0, 3600.0]
    with pytest.raises(ValueError, match="child_claim_deadline_seconds must be a number between 10.0 and 3600.0"):
        Settings(child_claim_deadline_seconds=5.0)

    with pytest.raises(ValueError, match="child_claim_deadline_seconds must be a number between 10.0 and 3600.0"):
        Settings(child_claim_deadline_seconds=4000.0)

    # from_env
    os.environ["CHILD_CLAIM_DEADLINE_SECONDS"] = "240"
    try:
        s_env = Settings.from_env()
        assert s_env.child_claim_deadline_seconds == 240.0
    finally:
        os.environ.pop("CHILD_CLAIM_DEADLINE_SECONDS", None)

    # from_env invalid
    os.environ["CHILD_CLAIM_DEADLINE_SECONDS"] = "not-a-number"
    try:
        with pytest.raises(ValueError, match="Invalid CHILD_CLAIM_DEADLINE_SECONDS configuration"):
            Settings.from_env()
    finally:
        os.environ.pop("CHILD_CLAIM_DEADLINE_SECONDS", None)

    os.environ["CHILD_CLAIM_DEADLINE_SECONDS"] = "5.0"
    try:
        with pytest.raises(ValueError, match="Invalid CHILD_CLAIM_DEADLINE_SECONDS configuration"):
            Settings.from_env()
    finally:
        os.environ.pop("CHILD_CLAIM_DEADLINE_SECONDS", None)
