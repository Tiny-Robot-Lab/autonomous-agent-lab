import json
import re
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from agent_contracts import (
    RequiredCapabilityRecord,
    TaskContinuation,
    TaskInput,
    TaskRequested,
)
from orchestrator.config import Settings
from orchestrator.domain.principal import (
    compute_token_digest,
)
from orchestrator.domain.task import TaskLineage
from orchestrator.presentation.api import create_app

from tests.test_application import FakeStateReader, FakeTaskQueue
from tests.test_authoritative_task_store import InMemoryAuthoritativeTaskStore, MockAtomicDispatcher

PEPPER = "test_runtime_pepper_secret_1234567890"
PROOF_PEPPER = "test_assignment_proof_pepper_456"
RESEARCH_TOKEN = "secret-token-research-1"
WORKER_TOKEN = "secret-token-worker-1"
ANALYSIS_TOKEN = "secret-token-analysis-1"
ANALYSIS_MULTI_TOKEN = "secret-token-analysis-multi"

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
    "principal-analysis-multi": {
        "principal_id": "principal-analysis-multi",
        "bound_agent_id": "analysis-agent-multi",
        "authorized_capabilities": ["analysis", "research"],
        "token_digest": compute_token_digest(ANALYSIS_MULTI_TOKEN, PEPPER),
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
    await store.register_root_task(root, required_capability="research")
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
    await store.register_root_task(root, required_capability="research")
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
    await store.register_root_task(root, required_capability="research")
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
    await store.register_root_task(root, required_capability="research")
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
        "p1": {
            "principal_id": "p1",
            "bound_agent_id": "agent-1",
            "token_digest": "a" * 64,
            "authorized_capabilities": ["general"],
            "active": False,
        }
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


@pytest.mark.anyio
async def test_cross_capability_hijack_prevention(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))

    # 1. Root task registered with required capability 'research'
    research_task = TaskRequested.create("Research Task")
    await store.register_root_task(research_task, required_capability="research")
    await store.update_status(research_task.task_id, "pending")

    # 2. Root task registered with required capability 'general'
    general_task = TaskRequested.create("General Task")
    await store.register_root_task(general_task, required_capability="general")
    await store.update_status(general_task.task_id, "pending")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # General principal attempts to claim research task -> 403 Forbidden
        res_hijack_1 = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": research_task.task_id,
                "request_id": "req-hijack-1",
                "capability": "general",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res_hijack_1.status_code == 403
        assert "requires capability 'research'" in res_hijack_1.json()["detail"]

        # Research principal attempts to claim general task -> 403 Forbidden
        res_hijack_2 = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": general_task.task_id,
                "request_id": "req-hijack-2",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_hijack_2.status_code == 403
        assert "requires capability 'general'" in res_hijack_2.json()["detail"]


@pytest.mark.anyio
async def test_execution_kind_lifecycle_compatibility_via_api(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))

    root = TaskRequested.create("Root Task")
    await store.register_root_task(root, required_capability="general")
    await store.update_status(root.task_id, "pending")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Initial claim with expected_continuation_version provided -> 400 Bad Request
        res_bad_cont = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-bad-cont",
                "capability": "general",
                "execution_kind": "initial",
                "expected_continuation_version": 1,
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res_bad_cont.status_code == 400
        assert "expected_continuation_version must not be provided" in res_bad_cont.json()["detail"]

        # 2. Resumed continuation without expected_continuation_version -> 400 Bad Request
        res_missing_ver = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-missing-ver",
                "capability": "general",
                "execution_kind": "resumed_continuation",
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res_missing_ver.status_code == 400
        assert "expected_continuation_version must be an integer >= 1" in res_missing_ver.json()["detail"]

        # 3. Continuation not found -> 404 Not Found
        res_cont_not_found = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-resumed-not-found",
                "capability": "general",
                "execution_kind": "resumed_continuation",
                "expected_continuation_version": 1,
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res_cont_not_found.status_code == 404
        assert "not found" in res_cont_not_found.json()["detail"].lower()

        # Save valid continuation 1
        now_str = datetime.now(UTC).isoformat()
        continuation = TaskContinuation(
            parent_task_id=root.task_id,
            child_task_id="child-123",
            input_version=1,
            continuation_capability="general",
            stage="incorporate_analysis_child",
            continuation_version=1,
            created_at=now_str,
            updated_at=now_str,
        )
        await store.save_continuation(continuation)

        # 4. Resumed continuation on task with status 'pending' -> 409 Conflict
        res_resumed_on_pending = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-resumed-pending",
                "capability": "general",
                "execution_kind": "resumed_continuation",
                "expected_continuation_version": 1,
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res_resumed_on_pending.status_code == 409
        assert "lifecycle mismatch" in res_resumed_on_pending.json()["detail"].lower()

        # 5. Initial claim on task with status 'resumable' -> 409 Conflict
        await store.update_status(root.task_id, "resumable")
        res_initial_on_resumable = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-initial-resumable",
                "capability": "general",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res_initial_on_resumable.status_code == 409
        assert "lifecycle mismatch" in res_initial_on_resumable.json()["detail"].lower()

        # 6. Delegated child claim on root task -> 409 Conflict
        await store.update_status(root.task_id, "pending")
        res_child_on_root = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-child-root",
                "capability": "general",
                "execution_kind": "delegated_child",
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res_child_on_root.status_code == 409
        assert "not a delegated child" in res_child_on_root.json()["detail"].lower()


@pytest.mark.anyio
async def test_delegated_child_claim_receives_no_delegation_grant(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))

    now = datetime.now(UTC).isoformat()
    root_task = TaskRequested.create("Root Research Task")
    parent_id = root_task.task_id
    child_id = "child-analysis-1"

    # 1. Setup root research task and delegated child task
    await store.register_root_task(root_task, required_capability="research")
    await store.update_status(parent_id, "waiting_for_children")

    child_lineage = TaskLineage(
        task_id=child_id,
        root_task_id=parent_id,
        parent_task_id=parent_id,
    )
    store.lineages[child_id] = child_lineage
    store.statuses[child_id] = "pending"
    store.required_capabilities[child_id] = RequiredCapabilityRecord(
        task_id=child_id,
        required_capability="analysis",
        created_at=now,
    )
    store.inputs[(child_id, 1)] = TaskInput(
        task_id=child_id,
        prompt="Analyze data",
        created_at=now,
        input_version=1,
    )
    store.children.setdefault(parent_id, set()).add(child_id)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 2. General principal attempts to claim Analysis child -> 403 Forbidden
        res_worker = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": child_id,
                "request_id": "req-worker-child",
                "capability": "general",
                "execution_kind": "delegated_child",
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res_worker.status_code == 403
        assert "requires capability 'analysis'" in res_worker.json()["detail"]

        # 3. Research principal attempts to claim Analysis child -> 403 Forbidden
        res_research = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": child_id,
                "request_id": "req-research-child",
                "capability": "research",
                "execution_kind": "delegated_child",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res_research.status_code == 403
        assert "requires capability 'analysis'" in res_research.json()["detail"]

        # 4. Analysis principal advertising extra capability claiming child with unrelated capability -> 403
        res_multi_wrong = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": child_id,
                "request_id": "req-multi-wrong",
                "capability": "research",
                "execution_kind": "delegated_child",
            },
            headers={"Authorization": f"Bearer {ANALYSIS_MULTI_TOKEN}"},
        )
        assert res_multi_wrong.status_code == 403
        assert "requires capability 'analysis'" in res_multi_wrong.json()["detail"]

        # 5. Analysis principal claiming child with capability 'analysis' succeeds
        res_ok = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": child_id,
                "request_id": "req-analysis-ok",
                "capability": "analysis",
                "execution_kind": "delegated_child",
            },
            headers={"Authorization": f"Bearer {ANALYSIS_MULTI_TOKEN}"},
        )
        assert res_ok.status_code == 200
        data = res_ok.json()
        assert "assignment" in data
        assert "assignment_proof" in data
        assignment = data["assignment"]

        # Strict checks on resulting TaskAssignment
        assert assignment["execution_kind"] == "delegated_child"
        assert assignment["assigned_capability"] == "analysis"
        assert assignment.get("delegation_grant_id") is None  # Analysis child receives NO delegation grant

        # Verify active claim in authoritative store
        active_claim = await store.get_active_claim(child_id)
        assert active_claim is not None
        assert active_claim.assigned_capability == "analysis"
        assert active_claim.execution_kind == "delegated_child"
        assert active_claim.delegation_grant_id is None


@pytest.mark.anyio
async def test_missing_authoritative_task_input_rejects_claim(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(settings=test_settings, task_store=store, dispatcher=MockAtomicDispatcher(store))

    root = TaskRequested.create("Root Task Without Input")
    await store.register_root_task(root, required_capability="general")
    await store.update_status(root.task_id, "pending")

    # Manually remove TaskInput to simulate missing input record
    del store.inputs[(root.task_id, 1)]

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-missing-input",
                "capability": "general",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
        )
        assert res.status_code == 409
        assert "no authoritative input record" in res.json()["detail"].lower()


def test_principal_registry_startup_validation() -> None:
    # 1. Malformed token_digest (not 64 lowercase hex)
    bad_digest_cfg = {
        "p1": {
            "principal_id": "p1",
            "bound_agent_id": "ag1",
            "token_digest": "not-valid-hex-64-characters",
            "authorized_capabilities": ["general"],
            "active": True,
        }
    }
    with pytest.raises(ValueError, match="token_digest must be a 64-character lowercase hexadecimal string"):
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=bad_digest_cfg,
        )

    # 2. Duplicate token_digest across principals
    dup_digest_cfg = {
        "p1": {
            "principal_id": "p1",
            "bound_agent_id": "ag1",
            "token_digest": "a" * 64,
            "authorized_capabilities": ["general"],
            "active": True,
        },
        "p2": {
            "principal_id": "p2",
            "bound_agent_id": "ag2",
            "token_digest": "a" * 64,
            "authorized_capabilities": ["general"],
            "active": True,
        },
    }
    with pytest.raises(ValueError, match="duplicate token_digest"):
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=dup_digest_cfg,
        )

    # 3. Duplicate active bound_agent_id across active principals
    dup_agent_cfg = {
        "p1": {
            "principal_id": "p1",
            "bound_agent_id": "shared-agent",
            "token_digest": "a" * 64,
            "authorized_capabilities": ["general"],
            "active": True,
        },
        "p2": {
            "principal_id": "p2",
            "bound_agent_id": "shared-agent",
            "token_digest": "b" * 64,
            "authorized_capabilities": ["research"],
            "active": True,
        },
    }
    with pytest.raises(ValueError, match="duplicate bound_agent_id for active principal"):
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=dup_agent_cfg,
        )

    # 4. Unrecognized capability in authorized_capabilities
    unknown_cap_cfg = {
        "p1": {
            "principal_id": "p1",
            "bound_agent_id": "ag1",
            "token_digest": "a" * 64,
            "authorized_capabilities": ["unrecognized_capability"],
            "active": True,
        }
    }
    with pytest.raises(ValueError, match="unrecognized or empty capability in authorized_capabilities"):
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=unknown_cap_cfg,
        )
