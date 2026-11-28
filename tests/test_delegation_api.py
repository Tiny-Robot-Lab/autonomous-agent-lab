from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from agent_contracts import (
    ArtifactReference,
    DelegationGrant,
    DelegationGrantV2,
    RootAuthorityRecord,
    StageArtifact,
    TaskRequested,
)
from orchestrator.config import Settings
from orchestrator.domain.principal import (
    compute_assignment_verifier_digest,
    compute_token_digest,
    derive_assignment_proof,
    derive_proof_subkeys,
    encode_canonical_proof_identity,
    parse_master_secret,
)
from orchestrator.presentation.api import create_app

from tests.test_authoritative_task_store import InMemoryAuthoritativeTaskStore, MockAtomicDispatcher

PEPPER = "test_runtime_pepper_secret_1234567890"
PROOF_PEPPER = "test_assignment_proof_pepper_456"
RESEARCH_TOKEN = "secret-token-research-1"
RESEARCH_TOKEN_2 = "secret-token-research-2"
WORKER_TOKEN = "secret-token-worker-1"
ANALYSIS_TOKEN = "secret-token-analysis-1"

PRINCIPALS_CONFIG: dict[str, dict[str, Any]] = {
    "principal-research-1": {
        "principal_id": "principal-research-1",
        "bound_agent_id": "research-agent-1",
        "authorized_capabilities": ["research"],
        "token_digest": compute_token_digest(RESEARCH_TOKEN, PEPPER),
        "active": True,
    },
    "principal-research-2": {
        "principal_id": "principal-research-2",
        "bound_agent_id": "research-agent-2",
        "authorized_capabilities": ["research"],
        "token_digest": compute_token_digest(RESEARCH_TOKEN_2, PEPPER),
        "active": True,
    },
    "principal-worker-1": {
        "principal_id": "principal-worker-1",
        "bound_agent_id": "worker-1",
        "authorized_capabilities": ["general"],
        "token_digest": compute_token_digest(WORKER_TOKEN, PEPPER),
        "active": True,
    },
    "principal-analysis-1": {
        "principal_id": "principal-analysis-1",
        "bound_agent_id": "analysis-agent-1",
        "authorized_capabilities": ["analysis"],
        "token_digest": compute_token_digest(ANALYSIS_TOKEN, PEPPER),
        "active": True,
    },
    "principal-inactive-1": {
        "principal_id": "principal-inactive-1",
        "bound_agent_id": "inactive-1",
        "authorized_capabilities": ["research"],
        "token_digest": compute_token_digest("inactive-token", PEPPER),
        "active": False,
    },
}


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        redis_url="redis://fake:6379/9",
        app_env="test",
        task_delivery_mode="authenticated_claim",
        claim_delivery_lease_seconds=30.0,
        child_claim_deadline_seconds=120.0,
        runtime_credential_pepper=PEPPER,
        assignment_proof_pepper=PROOF_PEPPER,
        runtime_principals=PRINCIPALS_CONFIG,
    )


async def setup_acknowledged_research_claim(
    store: InMemoryAuthoritativeTaskStore,
    parent_id: str = "task-parent-1",
    grant_v2: bool = False,
) -> tuple[str, str, ArtifactReference]:
    master = parse_master_secret(PROOF_PEPPER)
    token_key, verifier_key = derive_proof_subkeys(master)

    now_iso = datetime.now(UTC).isoformat()
    root_task = TaskRequested(
        task_id=parent_id,
        root_task_id=parent_id,
        parent_task_id=None,
        prompt="Do initial research",
        created_at=now_iso,
        version=2,
    )
    await store.register_root_task(root_task, required_capability="research")
    if grant_v2:
        store.authorities[parent_id] = RootAuthorityRecord.create_wave_baseline(
            parent_id, max_children_per_wave=4, created_at=now_iso
        )
    # Set status to working (simulating claim and acknowledge)
    await store.update_status(parent_id, "working")

    assignment_id = f"asgn-{parent_id}"
    if grant_v2:
        grant = DelegationGrantV2(
            grant_id=f"grant-{parent_id}",
            root_task_id=parent_id,
            issued_to_assignment_id=assignment_id,
            issued_to_principal_id="principal-research-1",
            fencing_epoch=1,
            turn_id="turn-1",
            issued_at=now_iso,
            max_children=4,
        )
    else:
        grant = DelegationGrant(
            grant_id=f"grant-{parent_id}",
            root_task_id=parent_id,
            issued_to_assignment_id=assignment_id,
            issued_to_principal_id="principal-research-1",
            fencing_epoch=1,
            turn_id="turn-1",
            issued_at=now_iso,
        )
    claim_id = f"claim-{parent_id}"
    delivery_expires_at = (datetime.now(UTC) + timedelta(minutes=10)).isoformat()
    canonical_identity = encode_canonical_proof_identity(
        assignment_id=assignment_id,
        claim_id=claim_id,
        task_id=parent_id,
        root_task_id=parent_id,
        turn_id="turn-1",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        continuation_version=None,
        fencing_epoch=1,
        request_id="req-1",
        delivery_expires_at=delivery_expires_at,
        delegation_grant_id=grant.grant_id,
        prior_claimable_status="pending",
    )
    proof = derive_assignment_proof(token_key, canonical_identity)
    proof_digest = compute_assignment_verifier_digest(verifier_key, proof)

    from agent_contracts import ActiveClaim
    claim = ActiveClaim(
        task_id=parent_id,
        assignment_id=assignment_id,
        claim_id=claim_id,
        root_task_id=parent_id,
        turn_id="turn-1",
        request_id="req-1",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        state="acknowledged",
        claimed_at=now_iso,
        delivery_expires_at=delivery_expires_at,
        fencing_epoch=1,
        execution_kind="initial",
        proof_digest=proof_digest,
        acknowledged_at=now_iso,
        delegation_grant_id=grant.grant_id,
        prior_claimable_status="pending",
    )
    await store.save_active_claim(claim)
    await store.save_authority_grant(grant)
    store.fences[parent_id] = 1

    # Save a stage artifact for the research task
    art_ref = ArtifactReference(
        artifact_id=f"art-{parent_id}",
        artifact_version=1,
        producing_task_id=parent_id,
    )
    artifact = StageArtifact(
        artifact_id=art_ref.artifact_id,
        producing_task_id=parent_id,
        artifact_type="research_material",
        content={"material": "Key findings from initial research"},
        created_at=now_iso,
        producing_turn_id="turn-1",
        artifact_version=1,
        payload_schema_version=1,
        sealed=True,
    )
    await store.save_stage_artifact(
        artifact,
        expected_assignment_id=assignment_id,
        expected_principal_id="principal-research-1",
        expected_fencing_epoch=1,
    )

    return assignment_id, proof, art_ref


@pytest.mark.anyio
async def test_delegation_api_authentication_and_authorization(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-auth"
    assignment_id, proof, art_ref = await setup_acknowledged_research_claim(store, parent_id)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "proposal_id": "prop-1",
            "assignment_id": assignment_id,
            "child_intent": "Analyze findings",
            "artifact_reference": art_ref.to_dict(),
            "target_capability": "analysis",
        }

        # 1. Missing Authorization header -> 401
        res = await client.post(f"/api/tasks/{parent_id}/delegations", json=payload)
        assert res.status_code == 401
        assert "Authorization" in res.json()["detail"]

        # 2. Invalid bearer token -> 401
        res = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": "Bearer bad-token", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 401

        # 3. Missing assignment proof (neither header nor body) -> 401
        res = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res.status_code == 401
        assert "Missing assignment proof" in res.json()["detail"]

        # 4. Inactive principal -> 403
        res = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": "Bearer inactive-token", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 403

        # 5. Non-research principal (e.g. analysis principal) -> 403
        res = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {ANALYSIS_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 403
        assert "lacks capability 'research'" in res.json()["detail"]

        # 6. Invalid proof (tampered proof) -> 403
        res = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": "tampered-proof-value"},
        )
        assert res.status_code == 403
        assert "verification failed" in res.json()["detail"]

        # 7. Non-existent parent task -> 403 (normalized retired-replay failure, no oracle)
        res = await client.post(
            "/api/tasks/nonexistent-task/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 403
        assert res.json()["detail"] == "Delegation replay authorization failed"


@pytest.mark.anyio
async def test_delegation_api_rejects_caller_supplied_authority_fields(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-forbid"
    assignment_id, proof, art_ref = await setup_acknowledged_research_claim(store, parent_id)

    forbidden_cases = [
        ("child_task_id", "forged-child-id"),
        ("root_task_id", "forged-root-id"),
        ("parent_task_id", "forged-parent-id"),
        ("created_at", "2026-01-01T00:00:00Z"),
        ("status", "pending"),
        ("wave", 2),
        ("budget", 10),
        ("grant", "grant-123"),
        ("fencing_epoch", 5),
    ]

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for field_name, bad_value in forbidden_cases:
            payload = {
                "proposal_id": f"prop-{field_name}",
                "assignment_id": assignment_id,
                "child_intent": "Analyze findings",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
                field_name: bad_value,
            }
            res = await client.post(
                f"/api/tasks/{parent_id}/delegations",
                json=payload,
                headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
            )
            assert res.status_code == 400, f"Expected 400 for forbidden field '{field_name}', got {res.status_code}"
            assert "Forbidden" in res.json()["detail"] or "Unexpected" in res.json()["detail"]


@pytest.mark.anyio
async def test_delegation_api_success_and_idempotent_replay(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-success"
    assignment_id, proof, art_ref = await setup_acknowledged_research_claim(store, parent_id)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "proposal_id": "prop-success-1",
            "assignment_id": assignment_id,
            "child_intent": "Perform deep statistical analysis",
            "artifact_reference": art_ref.to_dict(),
            "target_capability": "analysis",
            "control_fields": {"analysis_focus": "variance"},
        }

        # 1. First acceptance
        res = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 200, res.text
        data = res.json()

        assert data["status"] == "accepted"
        assert data["target_capability"] == "analysis"
        assert data["continuation_version"] == 1
        assert data["child_task_id"].startswith("task-analysis-")
        assert "child_claim_deadline" in data
        assert "accepted_at" in data
        assert "delegation_decision_id" in data

        child_id = data["child_task_id"]
        deadline = data["child_claim_deadline"]

        # Verify parent claim retired in store
        active_claim = await store.get_active_claim(parent_id)
        assert active_claim is None

        # Verify child task registered and pending
        child_status = await store.get_status(child_id)
        assert child_status is not None
        assert child_status == "pending"

        # Verify hints emitted
        assert len(store.hints) == 1
        hint = store.hints[0]
        assert hint.capability == "analysis"
        assert hint.task_id == child_id
        assert hint.execution_kind == "delegated_child"

        # 2. Idempotent replay: exact same payload
        res_replay = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res_replay.status_code == 200
        data_replay = res_replay.json()

        assert data_replay == data
        assert data_replay["child_claim_deadline"] == deadline

        # Ensure no duplicate hint was emitted
        assert len(store.hints) == 1

        # 2b. Replay with principal bound to different agent_id fails closed with 403
        res_diff_bound = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN_2}", "X-Assignment-Proof": proof},
        )
        assert res_diff_bound.status_code == 403
        assert res_diff_bound.json()["detail"] == "Delegation replay authorization failed"
        assert "child_task_id" not in res_diff_bound.json()
        assert "delegation_decision_id" not in res_diff_bound.json()

        # 3. Conflicting replay with same proposal_id but different child_intent
        # Note: Since the parent claim was already retired upon first acceptance (claim is None),
        # any replay where the fingerprint does not match raises DelegationReplayAuthorizationError -> 403,
        # preventing information oracle leakage per ADR-0007 / Muse Code review.
        conflicting_payload = dict(payload)
        conflicting_payload["child_intent"] = "A completely different intent"
        res_conflict = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=conflicting_payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res_conflict.status_code == 403
        assert res_conflict.json()["detail"] == "Delegation replay authorization failed"


@pytest.mark.anyio
async def test_delegation_api_proof_in_body_rejected_and_header_required(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-body-proof"
    assignment_id, proof, art_ref = await setup_acknowledged_research_claim(store, parent_id)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Body containing assignment_proof must be rejected with 400
        payload_with_body_proof = {
            "proposal_id": "prop-body-1",
            "assignment_id": assignment_id,
            "child_intent": "Analyze findings via body proof",
            "artifact_reference": art_ref.to_dict(),
            "target_capability": "analysis",
            "assignment_proof": proof,
        }
        res_body = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload_with_body_proof,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res_body.status_code == 400
        assert "assignment_proof must not be provided in request body" in res_body.json()["detail"]

        # 2. Missing X-Assignment-Proof header must be rejected with 401
        valid_payload = {
            "proposal_id": "prop-body-2",
            "assignment_id": assignment_id,
            "child_intent": "Analyze findings via valid header",
            "artifact_reference": art_ref.to_dict(),
            "target_capability": "analysis",
        }
        res_no_header = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=valid_payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_no_header.status_code == 401
        assert "Missing assignment proof in X-Assignment-Proof header" in res_no_header.json()["detail"]

        # 3. Supplying proof strictly via X-Assignment-Proof header succeeds with 200
        res_ok = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=valid_payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res_ok.status_code == 200
        assert res_ok.json()["status"] == "accepted"


@pytest.mark.anyio
async def test_delegation_api_replay_without_active_claim_via_read_only_get(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-replay-retired"
    assignment_id, proof, art_ref = await setup_acknowledged_research_claim(store, parent_id)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "proposal_id": "prop-ret-1",
            "assignment_id": assignment_id,
            "child_intent": "Analyze findings",
            "artifact_reference": art_ref.to_dict(),
            "target_capability": "analysis",
        }

        # 1. First acceptance
        res1 = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res1.status_code == 200
        data1 = res1.json()

        # Active claim is now retired
        assert (await store.get_active_claim(parent_id)) is None
        assert len(store.hints) == 1

        # 2. Replay with active claim retired: must succeed via get_delegation_proposal with zero writes
        res_rep = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res_rep.status_code == 200
        assert res_rep.json() == data1
        # Zero duplicate hints emitted
        assert len(store.hints) == 1

        # Authorization failure cases must all normalize to 403 "Delegation replay authorization failed"
        # without leaking proposal existence or proposal details.

        # Case 1: Non-existent proposal replay -> 403
        nonexistent_payload = dict(payload)
        nonexistent_payload["proposal_id"] = "prop-nonexistent"
        res_nonexistent = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=nonexistent_payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res_nonexistent.status_code == 403
        assert res_nonexistent.json()["detail"] == "Delegation replay authorization failed"
        assert "child_task_id" not in res_nonexistent.json()
        assert "delegation_decision_id" not in res_nonexistent.json()

        # Case 2: Accepted proposal + invalid proof -> 403
        res_bad_proof = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": "tampered-proof-value"},
        )
        assert res_bad_proof.status_code == 403
        assert res_bad_proof.json()["detail"] == "Delegation replay authorization failed"
        assert "child_task_id" not in res_bad_proof.json()

        # Case 3: Accepted proposal + wrong assignment -> 403
        wrong_asgn_payload = dict(payload)
        wrong_asgn_payload["assignment_id"] = "asgn-wrong-assignment"
        res_wrong_asgn = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=wrong_asgn_payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res_wrong_asgn.status_code == 403
        assert res_wrong_asgn.json()["detail"] == "Delegation replay authorization failed"
        assert "child_task_id" not in res_wrong_asgn.json()

        # Case 4: Accepted proposal + wrong principal binding -> 403
        res_wrong_principal = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN_2}", "X-Assignment-Proof": proof},
        )
        assert res_wrong_principal.status_code == 403
        assert res_wrong_principal.json()["detail"] == "Delegation replay authorization failed"
        assert "child_task_id" not in res_wrong_principal.json()

        # Case 5: Accepted proposal + changed semantic fingerprint -> 403
        conflicting_payload = dict(payload)
        conflicting_payload["child_intent"] = "Tampered child intent"
        res_conf = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=conflicting_payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res_conf.status_code == 403
        assert res_conf.json()["detail"] == "Delegation replay authorization failed"
        assert "child_task_id" not in res_conf.json()


@pytest.mark.anyio
async def test_no_plaintext_assignment_proof_stored_or_leaked(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-no-leak"
    assignment_id, proof, art_ref = await setup_acknowledged_research_claim(store, parent_id)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "proposal_id": "prop-leak-1",
            "assignment_id": assignment_id,
            "child_intent": "Check for secret leakage",
            "artifact_reference": art_ref.to_dict(),
            "target_capability": "analysis",
        }
        res = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 200
        data = res.json()

        # Plaintext proof does not appear anywhere in response JSON
        assert proof not in str(data)

        # Inspect persisted proposal record: only proof_digest is retained
        record = await store.get_delegation_proposal(parent_id, "prop-leak-1")
        assert record is not None
        assert proof not in record.to_json()
        assert not hasattr(record, "assignment_proof")
        assert hasattr(record, "proof_digest")

        # Inspect wake-up hint: no proof
        hint = store.hints[0]
        assert proof not in hint.to_json()

        # Inspect yield record: no proof
        yield_rec = store.yield_records.get((parent_id, assignment_id))
        assert yield_rec is not None
        assert proof not in yield_rec.to_json()


@pytest.mark.anyio
async def test_child_claim_deadline_enforcement(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-deadline"
    assignment_id, proof, art_ref = await setup_acknowledged_research_claim(store, parent_id)

    # Accept delegation to create pending child task
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"/api/tasks/{parent_id}/delegations",
            json={
                "proposal_id": "prop-deadline-1",
                "assignment_id": assignment_id,
                "child_intent": "Analyze findings",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 200
        child_task_id = res.json()["child_task_id"]

        # Case 1: Claim before deadline expires -> succeeds
        claim_payload = {
            "task_id": child_task_id,
            "request_id": "req-claim-1",
            "capability": "analysis",
            "execution_kind": "delegated_child",
        }
        res_claim = await client.post(
            "/api/tasks/claim",
            json=claim_payload,
            headers={"Authorization": f"Bearer {ANALYSIS_TOKEN}"},
        )
        assert res_claim.status_code == 200
        assert res_claim.json()["assignment"]["task_id"] == child_task_id

    # Case 2: Expired deadline
    parent_id_2 = "task-parent-expired"
    assignment_id_2, proof_2, art_ref_2 = await setup_acknowledged_research_claim(store, parent_id_2)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"/api/tasks/{parent_id_2}/delegations",
            json={
                "proposal_id": "prop-expired-1",
                "assignment_id": assignment_id_2,
                "child_intent": "Analyze findings later",
                "artifact_reference": art_ref_2.to_dict(),
                "target_capability": "analysis",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof_2},
        )
        assert res.status_code == 200
        child_task_id_2 = res.json()["child_task_id"]

        # Manually backdate the child claim deadline in the store
        store.claim_deadlines[child_task_id_2] = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()

        # Attempt to claim child task after deadline -> 409 Conflict (ChildClaimDeadlineExpiredError)
        claim_payload = {
            "task_id": child_task_id_2,
            "request_id": "req-claim-2",
            "capability": "analysis",
            "execution_kind": "delegated_child",
        }
        res_claim_expired = await client.post(
            "/api/tasks/claim",
            json=claim_payload,
            headers={"Authorization": f"Bearer {ANALYSIS_TOKEN}"},
        )
        assert res_claim_expired.status_code == 409
        assert "has expired" in res_claim_expired.json()["detail"].lower()


@pytest.mark.anyio
async def test_propose_delegation_wave_success(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-wave-1"
    assignment_id, proof, art_ref = await setup_acknowledged_research_claim(store, parent_id, grant_v2=True)

    payload = {
        "proposal_id": "wave-prop-1",
        "assignment_id": assignment_id,
        "barrier_policy": "WAIT_ALL",
        "parent_input_version": 1,
        "version": 2,
        "items": [
            {
                "item_id": "analysis-security",
                "child_intent": "Security analysis",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
            },
            {
                "item_id": "analysis-reliability",
                "child_intent": "Reliability analysis",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
            },
            {
                "item_id": "analysis-operations",
                "child_intent": "Operations and capacity analysis",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
            },
        ],
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"/api/tasks/{parent_id}/delegation-waves",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "accepted"
        assert data["parent_task_id"] == parent_id
        assert data["proposal_id"] == "wave-prop-1"
        assert data["barrier_policy"] == "WAIT_ALL"
        assert data["continuation_version"] == 2
        assert len(data["required_child_task_ids"]) == 3
        assert proof not in str(data)


@pytest.mark.anyio
async def test_propose_delegation_wave_missing_proof_rejected(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-wave-noproof"
    assignment_id, _proof, art_ref = await setup_acknowledged_research_claim(store, parent_id, grant_v2=True)

    payload = {
        "proposal_id": "wave-prop-noproof",
        "assignment_id": assignment_id,
        "barrier_policy": "WAIT_ALL",
        "parent_input_version": 1,
        "version": 2,
        "items": [
            {
                "item_id": "analysis-security",
                "child_intent": "Security analysis",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
            },
            {
                "item_id": "analysis-reliability",
                "child_intent": "Reliability analysis",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
            },
        ],
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"/api/tasks/{parent_id}/delegation-waves",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res.status_code == 401
        assert "missing assignment proof" in res.json()["detail"].lower()


@pytest.mark.anyio
async def test_propose_delegation_wave_extra_field_rejected(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-wave-extra"
    assignment_id, proof, art_ref = await setup_acknowledged_research_claim(store, parent_id, grant_v2=True)

    payload = {
        "proposal_id": "wave-prop-extra",
        "assignment_id": assignment_id,
        "barrier_policy": "WAIT_ALL",
        "parent_input_version": 1,
        "version": 2,
        "items": [
            {
                "item_id": "analysis-security",
                "child_intent": "Security analysis",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
            },
            {
                "item_id": "analysis-reliability",
                "child_intent": "Reliability analysis",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
            },
        ],
        "forbidden_authority": "inject",
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"/api/tasks/{parent_id}/delegation-waves",
            json=payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 400


@pytest.mark.anyio
async def test_propose_delegation_wave_capability_mismatch(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    parent_id = "task-parent-wave-cap"
    assignment_id, proof, art_ref = await setup_acknowledged_research_claim(store, parent_id, grant_v2=True)

    payload = {
        "proposal_id": "wave-prop-cap",
        "assignment_id": assignment_id,
        "barrier_policy": "WAIT_ALL",
        "parent_input_version": 1,
        "version": 2,
        "items": [
            {
                "item_id": "analysis-security",
                "child_intent": "Security analysis",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
            },
            {
                "item_id": "analysis-reliability",
                "child_intent": "Reliability analysis",
                "artifact_reference": art_ref.to_dict(),
                "target_capability": "analysis",
            },
        ],
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"/api/tasks/{parent_id}/delegation-waves",
            json=payload,
            headers={"Authorization": f"Bearer {WORKER_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert res.status_code == 403


def test_index_html_analysis_agent_friendly_name() -> None:
    from pathlib import Path
    index_path = Path("services/orchestrator/src/orchestrator/presentation/index.html")
    content = index_path.read_text(encoding="utf-8")
    assert "if (type === 'analysis-agent' || id.includes('analysis')) return 'Analysis Agent';" in content
