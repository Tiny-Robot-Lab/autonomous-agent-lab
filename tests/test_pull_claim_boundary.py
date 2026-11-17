import json
import re
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from agent_contracts import TaskRequested
from orchestrator.config import Settings
from orchestrator.domain.principal import (
    compute_token_digest,
)
from orchestrator.presentation.api import create_app

from tests.test_application import FakeStateReader, FakeTaskQueue
from tests.test_authoritative_task_store import InMemoryAuthoritativeTaskStore, MockAtomicDispatcher

PEPPER = "test_runtime_pepper_secret_1234567890"
PROOF_PEPPER = "test_assignment_proof_pepper_456"
RESEARCH_TOKEN = "secret-token-research-1"
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
        "authorized_capabilities": ["general"],
        "token_digest": compute_token_digest("inactive-token", PEPPER),
        "active": False,
    },
}


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        redis_url="redis://fake:6379/9",
        app_env="test",
        claim_delivery_lease_seconds=30.0,
        runtime_credential_pepper=PEPPER,
        assignment_proof_pepper=PROOF_PEPPER,
        runtime_principals=PRINCIPALS_CONFIG,
    )


@pytest.mark.anyio
async def test_claim_authentication_failures(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "task_id": "task-1",
            "request_id": "req-1",
            "capability": "research",
            "execution_kind": "initial",
        }

        # 1. Missing Authorization header -> 401
        res = await client.post("/api/tasks/claim", json=payload)
        assert res.status_code == 401
        assert "Authorization" in res.json()["detail"]

        # 2. Invalid bearer token -> 401
        res = await client.post(
            "/api/tasks/claim",
            json=payload,
            headers={"Authorization": "Bearer bad-token"},
        )
        assert res.status_code == 401
        assert "Invalid runtime credentials" in res.json()["detail"]

        # 3. Inactive principal -> 403
        res = await client.post(
            "/api/tasks/claim",
            json=payload,
            headers={"Authorization": "Bearer inactive-token"},
        )
        assert res.status_code == 403
        assert "inactive" in res.json()["detail"].lower()


@pytest.mark.anyio
async def test_research_initial_root_claim_receives_grant_and_detached_proof(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    root = TaskRequested.create("Research root task")
    await store.register_root_task(root)
    await store.update_status(root.task_id, "pending")

    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        claim_payload = {
            "task_id": root.task_id,
            "request_id": "req-research-001",
            "capability": "research",
            "execution_kind": "initial",
        }

        # Successful Research claim
        res = await client.post(
            "/api/tasks/claim",
            json=claim_payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "assignment" in data
        assert "assignment_proof" in data
        raw_proof = data["assignment_proof"]
        assert len(raw_proof) == 64
        assert re.match(r"^[0-9a-f]{64}$", raw_proof)

        assignment = data["assignment"]
        assert assignment["task_id"] == root.task_id
        assert assignment["assigned_capability"] == "research"
        assert assignment["fencing_epoch"] == 1
        assert assignment["runtime_principal_id"] == "principal-research-1"
        assert assignment["bound_agent_id"] == "research-agent-1"
        # Narrow one-turn research -> analysis grant is issued
        assert assignment["delegation_grant_id"] is not None
        grant_id = assignment["delegation_grant_id"]

        # Verify authority grant is recorded durably
        grant = await store.get_authority_grant(root.task_id, grant_id)
        assert grant is not None
        assert grant.grant_id == grant_id
        assert grant.source_capability == "research"
        assert grant.target_capability == "analysis"
        assert grant.max_children == 1
        assert grant.fencing_epoch == 1
        assert grant.turn_id == assignment["turn_id"]

        # Verify task is now working
        assert await store.get_status(root.task_id) == "working"

        # Idempotent retry returns same assignment and usable proof while unacknowledged
        res_replay = await client.post(
            "/api/tasks/claim",
            json=claim_payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_replay.status_code == 200
        replay_data = res_replay.json()
        assert replay_data["assignment"]["assignment_id"] == assignment["assignment_id"]
        assert replay_data["assignment_proof"] == raw_proof

        # Another principal cannot claim while active -> 409
        res_conflict = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-other-001",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_conflict.status_code == 409


@pytest.mark.anyio
async def test_worker_initial_root_claim_receives_no_grant(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    root = TaskRequested.create("General root task")
    await store.register_root_task(root)
    await store.update_status(root.task_id, "pending")

    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Worker claims general task
        res = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-worker-001",
                "capability": "general",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["assignment"]["assigned_capability"] == "general"
        assert data["assignment"]["delegation_grant_id"] is None

        # Verify no authority grant was issued
        assert len(store.authority_grants) == 0
        assert await store.get_authority_grant(root.task_id, "nonexistent-grant") is None


@pytest.mark.anyio
async def test_capability_mismatch_fails_closed(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    root = TaskRequested.create("General root task")
    await store.register_root_task(root)
    await store.update_status(root.task_id, "pending")

    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Worker attempts to claim research capability -> 403
        res = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-worker-mismatch",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res.status_code == 403
        assert "not authorized for capability" in res.json()["detail"]


@pytest.mark.anyio
async def test_acknowledgement_and_artifact_staging_lifecycle(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    root = TaskRequested.create("Research root task")
    await store.register_root_task(root)
    await store.update_status(root.task_id, "pending")

    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Claim task
        claim_res = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-1",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        claim_data = claim_res.json()
        assignment = claim_data["assignment"]
        assignment_id = assignment["assignment_id"]
        proof = claim_data["assignment_proof"]

        # 2. Attempt staging artifact before acknowledgement -> 409
        stage_payload = {
            "assignment_id": assignment_id,
            "assignment_proof": proof,
            "artifact_id": "material-1",
            "artifact_type": "research_material",
            "content": {
                "material": "Extracted evidence and synthesized source material for analysis",
            },
            "artifact_version": 1,
            "payload_schema_version": 1,
        }
        res_early = await client.post(
            f"/api/tasks/{root.task_id}/artifacts",
            json=stage_payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_early.status_code == 409
        assert "acknowledged" in res_early.json()["detail"].lower()

        # 3. Acknowledge with invalid proof -> 403
        res_bad_proof = await client.post(
            f"/api/assignments/{assignment_id}/acknowledge",
            json={"task_id": root.task_id, "assignment_proof": "wrong-proof"},
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_bad_proof.status_code == 403

        # 3b. Acknowledge with mismatched authenticated principal -> 404
        res_wrong_principal = await client.post(
            f"/api/assignments/{assignment_id}/acknowledge",
            json={"task_id": root.task_id, "assignment_proof": proof},
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res_wrong_principal.status_code == 404

        # 4. Acknowledge with valid proof -> 200
        res_ack = await client.post(
            f"/api/assignments/{assignment_id}/acknowledge",
            json={"task_id": root.task_id, "assignment_proof": proof},
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_ack.status_code == 200
        assert res_ack.json()["status"] == "acknowledged"

        # 5. Stage artifact after acknowledgement -> 200
        res_stage = await client.post(
            f"/api/tasks/{root.task_id}/artifacts",
            json=stage_payload,
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_stage.status_code == 200
        art_data = res_stage.json()["artifact"]
        assert art_data["artifact_id"] == "material-1"
        assert art_data["artifact_type"] == "research_material"
        assert art_data["sealed"] is True

        # Platform-stamped provenance validation
        metadata = art_data["metadata"]
        assert metadata["producing_task_id"] == root.task_id
        assert metadata["assignment_id"] == assignment_id
        assert metadata["runtime_principal_id"] == "principal-research-1"
        assert metadata["bound_agent_id"] == "research-agent-1"
        assert metadata["fencing_epoch"] == 1
        assert "stamped_at" in metadata

        # 6. Claim retry after acknowledgement fails closed -> 409
        res_post_ack_claim = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-1",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_post_ack_claim.status_code == 409
        assert "already acknowledged" in res_post_ack_claim.json()["detail"].lower()


@pytest.mark.anyio
async def test_delivery_lease_expiry_restores_claimability_via_api() -> None:
    # Configure with very short delivery lease (0.1s)
    short_lease_settings = Settings(
        redis_url="redis://fake:6379/9",
        app_env="test",
        claim_delivery_lease_seconds=0.1,
        runtime_credential_pepper=PEPPER,
        assignment_proof_pepper=PROOF_PEPPER,
        runtime_principals=PRINCIPALS_CONFIG,
    )
    store = InMemoryAuthoritativeTaskStore()
    root = TaskRequested.create("Root task with short lease")
    await store.register_root_task(root)
    await store.update_status(root.task_id, "pending")

    app = create_app(
        settings=short_lease_settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. First claim
        claim_res = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-1",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert claim_res.status_code == 200
        a_id = claim_res.json()["assignment"]["assignment_id"]
        proof = claim_res.json()["assignment_proof"]
        assert claim_res.json()["assignment"]["fencing_epoch"] == 1

        # 2. Let delivery lease expire
        import anyio

        await anyio.sleep(0.15)

        # 3. Attempt acknowledgement after expiry -> 409
        res_ack = await client.post(
            f"/api/assignments/{a_id}/acknowledge",
            json={"task_id": root.task_id, "assignment_proof": proof},
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_ack.status_code == 409
        assert "expired" in res_ack.json()["detail"].lower()

        # 4. Re-claiming the task succeeds: delivery lease expiry restored prior claimability
        # and fencing epoch is incremented to 2!
        claim_res_2 = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-2",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert claim_res_2.status_code == 200
        assignment_2 = claim_res_2.json()["assignment"]
        assert assignment_2["fencing_epoch"] == 2
        assert assignment_2["assignment_id"] != a_id


@pytest.mark.anyio
async def test_claim_mode_hints_published_under_authenticated_claim_mode() -> None:
    claim_mode_settings = Settings(
        redis_url="redis://fake:6379/9",
        app_env="test",
        task_delivery_mode="authenticated_claim",
        claim_delivery_lease_seconds=30.0,
        runtime_credential_pepper=PEPPER,
        assignment_proof_pepper=PROOF_PEPPER,
        runtime_principals=PRINCIPALS_CONFIG,
    )
    store = FakeTaskQueue(delivery_mode="authenticated_claim")
    fake_reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "research-1",
                "capabilities": ["research"],
                "state": "idle",
                "dispatch_target": "agent.tasks.research",
                "updated_at": datetime.now(UTC).isoformat(),
            }
        ]
    )
    app = create_app(
        settings=claim_mode_settings,
        task_store=store,
        dispatcher=store,
        state_reader=fake_reader,
        enforce_availability=True,
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/api/tasks", json={"prompt": "Research the latest quantum developments"})
        assert res.status_code == 202
        task_id = res.json()["task_id"]

        # In authenticated_claim mode:
        # 1. No full TaskRequested payload was pushed to dispatches
        assert len(store.dispatches) == 0
        assert len(store.hints) == 1
        hint, target = store.hints[0]
        assert target == "agent.hints.research"
        assert hint.capability == "research"
        assert hint.task_id == task_id
        assert hint.execution_kind == "initial"
        assert not hasattr(hint, "prompt")


@pytest.mark.anyio
async def test_expired_assignment_preserves_immutable_grants_and_advances_fencing() -> None:
    # 1. Setup store with root task (register_root_task automatically sets up default root authority with research -> analysis)
    store = InMemoryAuthoritativeTaskStore()
    root = TaskRequested.create("Root task with delegation grant")
    await store.register_root_task(root)
    await store.update_status(root.task_id, "pending")

    short_lease_settings = Settings(
        redis_url="redis://fake:6379/9",
        app_env="test",
        claim_delivery_lease_seconds=0.1,
        runtime_credential_pepper=PEPPER,
        assignment_proof_pepper=PROOF_PEPPER,
        runtime_principals=PRINCIPALS_CONFIG,
    )
    app = create_app(
        settings=short_lease_settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 2. Initial claim receives assignment A and grant A
        res_a = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-a",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_a.status_code == 200
        assign_a = res_a.json()["assignment"]
        grant_id_a = assign_a["delegation_grant_id"]
        assert grant_id_a is not None
        assert assign_a["fencing_epoch"] == 1

        # Check grant A in store
        grant_a = await store.get_authority_grant(root.task_id, grant_id_a)
        assert grant_a is not None
        assert grant_a.grant_id == grant_id_a
        assert grant_a.issued_to_assignment_id == assign_a["assignment_id"]
        assert grant_a.fencing_epoch == 1
        assert grant_a.turn_id == assign_a["turn_id"]

        # 3. Allow delivery lease to expire without acknowledgement
        import anyio

        await anyio.sleep(0.15)

        # 4. Claim replacement assignment B
        res_b = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-b",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_b.status_code == 200
        assign_b = res_b.json()["assignment"]
        grant_id_b = assign_b["delegation_grant_id"]
        assert grant_id_b is not None
        assert grant_id_b != grant_id_a
        assert assign_b["fencing_epoch"] == 2
        assert assign_b["assignment_id"] != assign_a["assignment_id"]

        # 5. Check both grants are independently retrievable by exact grant_id and immutable
        grant_a_after = await store.get_authority_grant(root.task_id, grant_id_a)
        grant_b = await store.get_authority_grant(root.task_id, grant_id_b)
        assert grant_a_after == grant_a  # Expired assignment A leaves grant A immutable
        assert grant_b is not None
        assert grant_b.grant_id == grant_id_b
        assert grant_b.issued_to_assignment_id == assign_b["assignment_id"]
        assert grant_b.fencing_epoch == 2
        assert grant_b.turn_id == assign_b["turn_id"]

        # Grants differ in assignment_id and fencing_epoch
        assert grant_a.issued_to_assignment_id != grant_b.issued_to_assignment_id
        assert grant_a.fencing_epoch != grant_b.fencing_epoch
        assert grant_a.turn_id != grant_b.turn_id

        # 6. Stale grant A cannot match current active claim or fence
        cur_claim = await store.get_active_claim(root.task_id)
        assert cur_claim is not None
        assert cur_claim.assignment_id == assign_b["assignment_id"]
        assert cur_claim.fencing_epoch == 2
        assert grant_a.issued_to_assignment_id != cur_claim.assignment_id
        assert grant_a.fencing_epoch != cur_claim.fencing_epoch


def test_settings_startup_validation_and_cryptographic_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. legacy_push starts cleanly with default Settings() (no secrets or principals required)
    default_settings = Settings()
    assert default_settings.task_delivery_mode == "legacy_push"
    assert default_settings.runtime_credential_pepper == ""
    assert default_settings.assignment_proof_pepper == ""
    assert len(default_settings.runtime_principals) == 0

    # app starts cleanly under legacy_push with empty secrets
    app = create_app(settings=default_settings)
    assert app.state.claim_task is None
    assert app.state.acknowledge_assignment is None
    assert app.state.stage_artifact is None

    # 2. authenticated_claim fails startup for missing runtime_credential_pepper
    with pytest.raises(ValueError) as exc_info:
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper="",
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=PRINCIPALS_CONFIG,
        )
    assert "runtime_credential_pepper must be configured" in str(exc_info.value)
    assert PROOF_PEPPER not in str(exc_info.value)

    # 2b. authenticated_claim fails startup for short (<32 bytes) runtime_credential_pepper
    short_cred_pepper = "short_cred_pepper_under_32"
    with pytest.raises(ValueError) as exc_info:
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=short_cred_pepper,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=PRINCIPALS_CONFIG,
        )
    assert "Secret pepper must provide at least 32 bytes" in str(exc_info.value)
    assert short_cred_pepper not in str(exc_info.value)

    # 3. authenticated_claim fails startup for missing assignment_proof_pepper
    with pytest.raises(ValueError) as exc_info:
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper="",
            runtime_principals=PRINCIPALS_CONFIG,
        )
    assert "assignment_proof_pepper must be configured" in str(exc_info.value)
    assert PEPPER not in str(exc_info.value)

    # 4. authenticated_claim fails startup for short (<32 bytes) assignment_proof_pepper
    short_secret = "short_secret_under_32_bytes"
    with pytest.raises(ValueError) as exc_info:
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=short_secret,
            runtime_principals=PRINCIPALS_CONFIG,
        )
    assert "Secret pepper must provide at least 32 bytes" in str(exc_info.value)
    assert short_secret not in str(exc_info.value)
    assert PEPPER not in str(exc_info.value)

    # 5. authenticated_claim fails startup for empty runtime_principals
    with pytest.raises(ValueError) as exc_info:
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals={},
        )
    assert "runtime_principals must contain at least one principal" in str(exc_info.value)

    # 6. authenticated_claim fails startup when all runtime_principals are inactive
    inactive_principals = {
        "p1": {"principal_id": "p1", "token_digest": "abc", "active": False}
    }
    with pytest.raises(ValueError) as exc_info:
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=inactive_principals,
        )
    assert "runtime_principals must contain at least one active principal" in str(exc_info.value)

    # 7. Environment parsing: malformed lease values fail startup with generic ValueError
    monkeypatch.setenv("CLAIM_DELIVERY_LEASE_SECONDS", "not-a-number")
    with pytest.raises(ValueError) as exc_info:
        Settings.from_env()
    assert "Invalid CLAIM_DELIVERY_LEASE_SECONDS configuration" in str(exc_info.value)
    assert "not-a-number" not in str(exc_info.value)

    monkeypatch.setenv("CLAIM_DELIVERY_LEASE_SECONDS", "-5.0")
    with pytest.raises(ValueError) as exc_info:
        Settings.from_env()
    assert "Invalid CLAIM_DELIVERY_LEASE_SECONDS configuration" in str(exc_info.value)
    assert "-5.0" not in str(exc_info.value)
    monkeypatch.delenv("CLAIM_DELIVERY_LEASE_SECONDS", raising=False)

    # 8. Environment parsing: malformed RUNTIME_PRINCIPALS_JSON fails with generic ValueError
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", "{invalid-json")
    with pytest.raises(ValueError) as exc_info:
        Settings.from_env()
    assert "Invalid RUNTIME_PRINCIPALS_JSON configuration" in str(exc_info.value)
    assert "{invalid-json" not in str(exc_info.value)
    monkeypatch.delenv("RUNTIME_PRINCIPALS_JSON", raising=False)

    # 9. Environment parsing: unsupported RUNTIME_PRINCIPALS_JSON structure fails with ValueError
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", "[1, 2, 3]")
    with pytest.raises(ValueError) as exc_info:
        Settings.from_env()
    assert "Unsupported RUNTIME_PRINCIPALS_JSON structure" in str(exc_info.value)
    assert "[1, 2, 3]" not in str(exc_info.value)

    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", '{"p1": "not-a-dict"}')
    with pytest.raises(ValueError) as exc_info:
        Settings.from_env()
    assert "Unsupported RUNTIME_PRINCIPALS_JSON structure" in str(exc_info.value)
    assert "not-a-dict" not in str(exc_info.value)
    monkeypatch.delenv("RUNTIME_PRINCIPALS_JSON", raising=False)

    # 10. legacy_push may omit all claim configuration from environment
    monkeypatch.delenv("TASK_DELIVERY_MODE", raising=False)
    monkeypatch.delenv("CLAIM_DELIVERY_LEASE_SECONDS", raising=False)
    monkeypatch.delenv("RUNTIME_CREDENTIAL_PEPPER", raising=False)
    monkeypatch.delenv("ASSIGNMENT_PROOF_PEPPER", raising=False)
    monkeypatch.delenv("RUNTIME_PRINCIPALS_JSON", raising=False)
    legacy_env_settings = Settings.from_env()
    assert legacy_env_settings.task_delivery_mode == "legacy_push"
    assert legacy_env_settings.claim_delivery_lease_seconds == 30.0
    assert legacy_env_settings.runtime_credential_pepper == ""
    assert legacy_env_settings.assignment_proof_pepper == ""
    assert len(legacy_env_settings.runtime_principals) == 0

    # 11. Valid RUNTIME_PRINCIPALS_JSON loads cleanly
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", json.dumps(PRINCIPALS_CONFIG))
    loaded_settings = Settings.from_env()
    assert len(loaded_settings.runtime_principals) == len(PRINCIPALS_CONFIG)
    monkeypatch.delenv("RUNTIME_PRINCIPALS_JSON", raising=False)
