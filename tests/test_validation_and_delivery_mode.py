import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from agent_contracts import (
    AgentEvent,
    CompletionCommand,
    TaskRequested,
    TaskResult,
)
from agent_infrastructure import OrchestratorClient
from analysis_agent.application.processor import AnalysisProcessor
from analysis_agent.infrastructure.deterministic_analyzer import (
    DeterministicMaterialAnalyzer,
)
from orchestrator.config import Settings
from orchestrator.domain.principal import (
    compute_token_digest,
    parse_master_secret,
    validate_principal_config_registry,
    verify_token,
)
from orchestrator.presentation.api import create_app
from research_agent.application.processor import ResearchProcessor
from research_agent.infrastructure.stub_executor import StubResearchExecutor
from worker.application.processor import TaskProcessor
from worker.infrastructure.stub_inference import StubInferenceProvider

from tests.test_application import FakeTaskQueue
from tests.test_authoritative_task_store import (
    InMemoryAuthoritativeTaskStore,
    MockAtomicDispatcher,
)

PEPPER = "test_runtime_pepper_secret_1234567890"
PROOF_PEPPER = "test_assignment_proof_pepper_4567890"
RESEARCH_TOKEN = "secret-token-research-1"
WORKER_TOKEN = "secret-token-worker-1"
ANALYSIS_TOKEN = "secret-token-analysis-1"
OPERATOR_TOKEN = "secret-token-operator-1"

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
    "principal-operator-1": {
        "principal_id": "principal-operator-1",
        "bound_agent_id": "operator-1",
        "authorized_capabilities": ["recovery_operator"],
        "token_digest": compute_token_digest(OPERATOR_TOKEN, PEPPER),
        "active": True,
    },
}


class FakeBus:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []

    async def publish(self, event: AgentEvent) -> None:
        self.published.append(event)


def _make_auth_settings(delivery_mode: str = "authenticated_claim") -> Settings:
    return Settings(
        redis_url="redis://fake:6379/9",
        app_env="test",
        task_delivery_mode=delivery_mode,
        claim_delivery_lease_seconds=30.0,
        child_claim_deadline_seconds=120.0,
        execution_claim_timeout_seconds=900.0,
        runtime_credential_pepper=PEPPER,
        assignment_proof_pepper=PROOF_PEPPER,
        runtime_principals=PRINCIPALS_CONFIG,
    )


@pytest.mark.anyio
async def test_artifact_body_proof_smuggling_rejected_and_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(
        settings=_make_auth_settings(),
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    root = TaskRequested.create("Research task")
    await store.register_root_task(root, required_capability="research")
    await store.update_status(root.task_id, "pending")

    secret_proof = "smuggled-proof-secret-do-not-leak-12345"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # First claim and acknowledge task lawfully
        claim_res = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-auth-1",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert claim_res.status_code == 200
        claim_data = claim_res.json()
        assignment_id = claim_data["assignment"]["assignment_id"]
        valid_proof = claim_data["assignment_proof"]

        ack_res = await client.post(
            f"/api/assignments/{assignment_id}/acknowledge",
            json={"task_id": root.task_id, "assignment_proof": valid_proof},
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert ack_res.status_code == 200

        # Attempt staging artifact with smuggled assignment_proof in body
        with caplog.at_level(logging.DEBUG):
            res_smuggle = await client.post(
                f"/api/tasks/{root.task_id}/artifacts",
                json={
                    "assignment_id": assignment_id,
                    "artifact_id": "art-smuggle-1",
                    "artifact_type": "research_material",
                    "content": {"material": "test"},
                    "assignment_proof": secret_proof,
                },
                headers={
                    "Authorization": f"Bearer {RESEARCH_TOKEN}",
                    "X-Assignment-Proof": valid_proof,
                },
            )

        # 1. Status is 422 Unprocessable Entity
        assert res_smuggle.status_code == 422

        # 2. Secret proof string is NOT echoed anywhere
        assert secret_proof not in res_smuggle.text
        assert secret_proof not in caplog.text

        # 3. Validation detail has safe location and type
        errors = res_smuggle.json()["detail"]
        assert len(errors) == 1
        err = errors[0]
        assert err["loc"] == ["body", "assignment_proof"]
        assert err["type"] == "extra_forbidden"
        assert "input" not in err

        # 4. Store is unchanged (no artifacts staged)
        assert len(store.artifacts) == 0


@pytest.mark.anyio
async def test_nested_credential_shaped_extra_value_is_recursively_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(
        settings=_make_auth_settings(),
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    nested_secret = "highly-confidential-bearer-token-abc999"
    nested_private_key = "---BEGIN-PRIVATE-KEY---fake-key---END---"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        with caplog.at_level(logging.DEBUG):
            res = await client.post(
                "/api/tasks/task-dummy/artifacts",
                json={
                    "assignment_id": "as-1",
                    "artifact_id": "art-1",
                    "artifact_type": "research_material",
                    "content": {"material": "test"},
                    "credentials": {
                        "secret_token": nested_secret,
                        "nested": {"private_key": nested_private_key},
                    },
                },
                headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
            )

        assert res.status_code == 422
        assert nested_secret not in res.text
        assert nested_private_key not in res.text
        assert nested_secret not in caplog.text
        assert nested_private_key not in caplog.text

        errors = res.json()["detail"]
        assert len(errors) == 1
        err = errors[0]
        assert err["loc"] == ["body", "credentials"]
        assert err["type"] == "extra_forbidden"
        assert "input" not in err


@pytest.mark.anyio
async def test_safe_validation_error_control_preserves_safe_fields() -> None:
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(
        settings=_make_auth_settings(),
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Missing required assignment_id
        res = await client.post(
            "/api/tasks/task-dummy/artifacts",
            json={
                "artifact_id": "art-1",
                "artifact_type": "research_material",
                "content": {"material": "test"},
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert res.status_code == 422
        errors = res.json()["detail"]
        assert len(errors) == 1
        err = errors[0]
        assert err["loc"] == ["body", "assignment_id"]
        assert err["type"] == "missing"
        assert "Field required" in err["msg"]
        assert "input" not in err


@pytest.mark.anyio
async def test_delivery_mode_case_1_legacy_push_unconfigured() -> None:
    settings_legacy_empty = Settings(task_delivery_mode="legacy_push")
    store = InMemoryAuthoritativeTaskStore()
    queue = FakeTaskQueue()
    app = create_app(
        settings=settings_legacy_empty,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
        task_queue=queue,
    )

    assert not app.state.assignment_lifecycle_configured
    assert not app.state.executable_delegation_enabled

    assert app.state.claim_task is None
    assert app.state.acknowledge_assignment is None
    assert app.state.stage_artifact is None
    assert app.state.accept_delegation is None
    assert app.state.complete_turn is None
    assert app.state.get_resumed_continuation_context is None
    assert app.state.get_child_task_input is None
    assert app.state.evaluate_expiry is None
    assert app.state.resolve_expired_claim is None

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Legacy routes remain operational
        r_health = await client.get("/health")
        assert r_health.status_code == 200

        r_submit = await client.post("/api/tasks", json={"prompt": "legacy prompt"})
        assert r_submit.status_code == 202
        assert len(queue.enqueued) == 1

        # All principal-authenticated endpoints return 401 without auth header
        for endpoint, method, payload in [
            ("/api/tasks/claim", "post", {"task_id": "t1", "request_id": "r1", "capability": "general"}),
            ("/api/assignments/as-1/acknowledge", "post", {"task_id": "t1", "assignment_proof": "p1"}),
            ("/api/tasks/t1/artifacts", "post", {"assignment_id": "as-1", "artifact_id": "a1", "artifact_type": "art", "content": {}}),
            ("/api/tasks/t1/delegations", "post", {"proposal_id": "p1", "assignment_id": "as-1", "child_intent": "intent", "artifact_reference": {"artifact_id": "a1", "artifact_version": 1, "producing_task_id": "t1", "artifact_type": "art"}}),
            ("/api/tasks/t1/completion", "post", {"assignment_id": "as-1", "execution_kind": "initial"}),
            ("/api/tasks/t1/continuation-context?assignment_id=as-1", "get", None),
            ("/api/tasks/t1/input?assignment_id=as-1", "get", None),
            ("/api/tasks/t1/evaluate-expiry", "post", {}),
            ("/api/tasks/t1/resolve-expired-claim", "post", {"recovery_decision_id": "rec-1", "expired_claim_id": "c-1", "expired_assignment_id": "as-1", "expired_fencing_epoch": 1, "disposition": "retry"}),
        ]:
            if method == "post":
                resp = await client.post(endpoint, json=payload)
            else:
                resp = await client.get(endpoint)
            assert resp.status_code == 401, f"{endpoint} expected 401, got {resp.status_code}"


@pytest.mark.anyio
async def test_delivery_mode_case_2_legacy_push_with_valid_lifecycle_configuration() -> None:
    settings_legacy_auth = _make_auth_settings(delivery_mode="legacy_push")
    store = InMemoryAuthoritativeTaskStore()
    queue = FakeTaskQueue()
    app = create_app(
        settings=settings_legacy_auth,
        task_store=store,
        result_reader=store,
        dispatcher=MockAtomicDispatcher(store),
        task_queue=queue,
    )

    # Assignment lifecycle is configured; executable delegation is disabled
    assert app.state.assignment_lifecycle_configured is True
    assert app.state.executable_delegation_enabled is False

    assert app.state.claim_task is not None
    assert app.state.acknowledge_assignment is not None
    assert app.state.complete_turn is not None
    assert app.state.evaluate_expiry is not None
    assert app.state.resolve_expired_claim is not None

    assert app.state.stage_artifact is None
    assert app.state.accept_delegation is None
    assert app.state.get_resumed_continuation_context is None
    assert app.state.get_child_task_input is None

    root = TaskRequested.create("Research root task")
    await store.register_root_task(root, required_capability="research")
    await store.update_status(root.task_id, "pending")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Claim task with execution_kind="initial" succeeds
        claim_res = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-leg-1",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert claim_res.status_code == 200
        claim_data = claim_res.json()
        assignment_id = claim_data["assignment"]["assignment_id"]
        proof = claim_data["assignment_proof"]

        # 2. Acknowledge assignment delivery succeeds
        ack_res = await client.post(
            f"/api/assignments/{assignment_id}/acknowledge",
            json={"task_id": root.task_id, "assignment_proof": proof},
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert ack_res.status_code == 200

        # 3. Evaluate expiry endpoint is operational with authorized recovery operator principal
        eval_res = await client.post(
            f"/api/tasks/{root.task_id}/evaluate-expiry",
            json={},
            headers={"Authorization": f"Bearer {OPERATOR_TOKEN}"},
        )
        assert eval_res.status_code == 200
        assert eval_res.json()["action"] == "none"

        # 4. Complete turn through fenced completion succeeds
        comp_cmd = CompletionCommand(
            completion_command_id="cmd-leg-1",
            assignment_id=assignment_id,
            execution_kind="initial",
            completed_result=TaskResult(
                task_id=root.task_id,
                status="completed",
                output="Canonical research result",
            ),
        )
        comp_res = await client.post(
            f"/api/tasks/{root.task_id}/completion",
            json=comp_cmd.to_dict(),
            headers={
                "Authorization": f"Bearer {RESEARCH_TOKEN}",
                "X-Assignment-Proof": proof,
            },
        )
        assert comp_res.status_code == 200

        # 5. Result is queryable
        res_res = await client.get(f"/api/tasks/{root.task_id}/result")
        assert res_res.status_code == 200
        assert res_res.json()["status"] == "completed"
        assert res_res.json()["output"] == "Canonical research result"

        # 6. Delegation-only endpoints return 503
        r_del = await client.post(
            f"/api/tasks/{root.task_id}/delegations",
            json={
                "proposal_id": "prop-1",
                "assignment_id": assignment_id,
                "child_intent": "Analyze findings",
                "artifact_reference": {
                    "artifact_id": "art-1",
                    "artifact_version": 1,
                    "producing_task_id": root.task_id,
                    "artifact_type": "research_material",
                },
            },
            headers={
                "Authorization": f"Bearer {RESEARCH_TOKEN}",
                "X-Assignment-Proof": proof,
            },
        )
        assert r_del.status_code == 503
        assert "Delegation service is not enabled" in r_del.json()["detail"]

        r_art = await client.post(
            f"/api/tasks/{root.task_id}/artifacts",
            json={
                "assignment_id": assignment_id,
                "artifact_id": "art-1",
                "artifact_type": "research_material",
                "content": {"material": "test"},
            },
            headers={
                "Authorization": f"Bearer {RESEARCH_TOKEN}",
                "X-Assignment-Proof": proof,
            },
        )
        assert r_art.status_code == 503
        assert "Artifact staging service is not enabled" in r_art.json()["detail"]

        r_ctx = await client.get(
            f"/api/tasks/{root.task_id}/continuation-context?assignment_id={assignment_id}",
            headers={
                "Authorization": f"Bearer {RESEARCH_TOKEN}",
                "X-Assignment-Proof": proof,
            },
        )
        assert r_ctx.status_code == 503
        assert "Continuation context service is not enabled" in r_ctx.json()["detail"]

        r_inp = await client.get(
            f"/api/tasks/{root.task_id}/input?assignment_id={assignment_id}",
            headers={
                "Authorization": f"Bearer {RESEARCH_TOKEN}",
                "X-Assignment-Proof": proof,
            },
        )
        assert r_inp.status_code == 503
        assert "Child task input service is not enabled" in r_inp.json()["detail"]


@pytest.mark.anyio
async def test_delivery_mode_case_3_authenticated_claim_incomplete_configuration_fails_startup() -> None:
    # 1. Missing runtime_credential_pepper
    with pytest.raises(ValueError, match="runtime_credential_pepper must be configured"):
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper="",
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=PRINCIPALS_CONFIG,
        )

    # 2. Missing assignment_proof_pepper
    with pytest.raises(ValueError, match="assignment_proof_pepper must be configured"):
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper="",
            runtime_principals=PRINCIPALS_CONFIG,
        )

    # 3. Missing runtime_principals
    with pytest.raises(ValueError, match="runtime_principals must contain at least one principal"):
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals={},
        )

    # 4. All-inactive runtime_principals
    inactive_principals = {
        "p-in": {
            "principal_id": "p-in",
            "bound_agent_id": "agent-in",
            "token_digest": "a" * 64,
            "authorized_capabilities": ["general"],
            "active": False,
        }
    }
    with pytest.raises(ValueError, match="runtime_principals must contain at least one active principal"):
        Settings(
            task_delivery_mode="authenticated_claim",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=inactive_principals,
        )


@pytest.mark.anyio
async def test_delivery_mode_case_4_authenticated_claim_valid_configuration_enables_all() -> None:
    settings_valid = _make_auth_settings(delivery_mode="authenticated_claim")
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(
        settings=settings_valid,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    assert app.state.assignment_lifecycle_configured is True
    assert app.state.executable_delegation_enabled is True

    assert app.state.claim_task is not None
    assert app.state.acknowledge_assignment is not None
    assert app.state.stage_artifact is not None
    assert app.state.accept_delegation is not None
    assert app.state.complete_turn is not None
    assert app.state.get_resumed_continuation_context is not None
    assert app.state.get_child_task_input is not None
    assert app.state.evaluate_expiry is not None
    assert app.state.resolve_expired_claim is not None

    root = TaskRequested.create("Research task")
    await store.register_root_task(root, required_capability="research")
    await store.update_status(root.task_id, "pending")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        claim_res = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-auth-claim-1",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert claim_res.status_code == 200
        claim_data = claim_res.json()
        assignment_id = claim_data["assignment"]["assignment_id"]
        proof = claim_data["assignment_proof"]

        ack_res = await client.post(
            f"/api/assignments/{assignment_id}/acknowledge",
            json={"task_id": root.task_id, "assignment_proof": proof},
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert ack_res.status_code == 200

        # Artifact staging is enabled and succeeds (200, not 503)
        stage_res = await client.post(
            f"/api/tasks/{root.task_id}/artifacts",
            json={
                "assignment_id": assignment_id,
                "artifact_id": "art-auth-1",
                "artifact_type": "research_material",
                "content": {"material": "evidence"},
            },
            headers={
                "Authorization": f"Bearer {RESEARCH_TOKEN}",
                "X-Assignment-Proof": proof,
            },
        )
        assert stage_res.status_code == 200
        assert len(store.artifacts) == 1


@pytest.mark.anyio
async def test_delivery_mode_case_5_zero_mutation_on_disabled_delegation() -> None:
    settings_legacy_auth = _make_auth_settings(delivery_mode="legacy_push")
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(
        settings=settings_legacy_auth,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    root = TaskRequested.create("Research root task")
    await store.register_root_task(root, required_capability="research")
    await store.update_status(root.task_id, "pending")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Claim and acknowledge lawfully
        claim_res = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-mut-1",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert claim_res.status_code == 200
        assignment_id = claim_res.json()["assignment"]["assignment_id"]
        proof = claim_res.json()["assignment_proof"]

        ack_res = await client.post(
            f"/api/assignments/{assignment_id}/acknowledge",
            json={"task_id": root.task_id, "assignment_proof": proof},
            headers={"Authorization": f"Bearer {RESEARCH_TOKEN}"},
        )
        assert ack_res.status_code == 200

        # Attempt all delegation-only endpoints
        r_stage = await client.post(
            f"/api/tasks/{root.task_id}/artifacts",
            json={
                "assignment_id": assignment_id,
                "artifact_id": "art-attempt-1",
                "artifact_type": "research_material",
                "content": {"material": "findings"},
            },
            headers={
                "Authorization": f"Bearer {RESEARCH_TOKEN}",
                "X-Assignment-Proof": proof,
            },
        )
        assert r_stage.status_code == 503

        r_prop = await client.post(
            f"/api/tasks/{root.task_id}/delegations",
            json={
                "proposal_id": "prop-attempt-1",
                "assignment_id": assignment_id,
                "child_intent": "Analyze findings",
                "artifact_reference": {
                    "artifact_id": "art-attempt-1",
                    "artifact_version": 1,
                    "producing_task_id": root.task_id,
                    "artifact_type": "research_material",
                },
            },
            headers={
                "Authorization": f"Bearer {RESEARCH_TOKEN}",
                "X-Assignment-Proof": proof,
            },
        )
        assert r_prop.status_code == 503

        r_ctx = await client.get(
            f"/api/tasks/{root.task_id}/continuation-context?assignment_id={assignment_id}",
            headers={
                "Authorization": f"Bearer {RESEARCH_TOKEN}",
                "X-Assignment-Proof": proof,
            },
        )
        assert r_ctx.status_code == 503

        r_inp = await client.get(
            f"/api/tasks/{root.task_id}/input?assignment_id={assignment_id}",
            headers={
                "Authorization": f"Bearer {RESEARCH_TOKEN}",
                "X-Assignment-Proof": proof,
            },
        )
        assert r_inp.status_code == 503

        # Assert zero side-effects or store mutations occurred:
        assert len(store.artifacts) == 0
        assert len(store.proposals) == 0
        assert len(store.children.get(root.task_id, [])) == 0
        assert len(store.continuations.get(root.task_id, [])) == 0
        active_claim = await store.get_active_claim(root.task_id)
        assert active_claim is not None
        assert active_claim.assignment_id == assignment_id
        assert active_claim.fencing_epoch == 1
        status = await store.get_status(root.task_id)
        assert status == "working"


@pytest.mark.anyio
async def test_legacy_push_worker_runtime_compatibility() -> None:
    settings = _make_auth_settings(delivery_mode="legacy_push")
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(
        settings=settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    root = TaskRequested.create("Worker legacy task")
    await store.register_root_task(root, required_capability="general")
    await store.update_status(root.task_id, "pending")

    client = OrchestratorClient(
        base_url="http://test",
        runtime_token=WORKER_TOKEN,
        transport=httpx.ASGITransport(app=app),
    )

    # Worker runtime flow: claim task -> acknowledge -> processor.run_once
    assignment, proof = await client.claim_task(
        task_id=root.task_id,
        request_id="req-worker-leg",
        capability="general",
        execution_kind="initial",
    )
    assert assignment["assignment_id"]
    assert proof

    await client.acknowledge_assignment(
        root.task_id, assignment["assignment_id"], proof
    )

    bus = FakeBus()
    processor = TaskProcessor(
        agent_id="worker-1",
        bus=bus,
        inference_provider=StubInferenceProvider(),
        completion_client=client,
    )
    await processor.run_once(
        root,
        assignment_id=assignment["assignment_id"],
        assignment_proof=proof,
        execution_kind="initial",
    )

    # Verify task completed canonically in store
    status = await store.get_status(root.task_id)
    assert status == "completed"
    result = await store.get_result(root.task_id)
    assert result is not None
    assert result["status"] == "completed"
    assert "Worker legacy task" in (result["output"] or "")


@pytest.mark.anyio
async def test_legacy_push_research_runtime_compatibility_complete_locally() -> None:
    settings = _make_auth_settings(delivery_mode="legacy_push")
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(
        settings=settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    root = TaskRequested.create("Research legacy task")
    await store.register_root_task(root, required_capability="research")
    await store.update_status(root.task_id, "pending")

    client = OrchestratorClient(
        base_url="http://test",
        runtime_token=RESEARCH_TOKEN,
        transport=httpx.ASGITransport(app=app),
    )

    # Research runtime flow with default complete_locally
    assignment, proof = await client.claim_task(
        task_id=root.task_id,
        request_id="req-research-leg",
        capability="research",
        execution_kind="initial",
    )
    assert assignment["assignment_id"]
    assert proof

    await client.acknowledge_assignment(
        root.task_id, assignment["assignment_id"], proof
    )

    bus = FakeBus()
    # Default processor without injected delegation coordinator defaults to "complete_locally"
    processor = ResearchProcessor(
        agent_id="research-agent-1",
        bus=bus,
        executor=StubResearchExecutor(),
        completion_client=client,
    )
    await processor.run_once(
        root,
        assignment_id=assignment["assignment_id"],
        assignment_proof=proof,
        execution_kind="initial",
    )

    # Verify task completed directly without creating artifacts or delegations
    status = await store.get_status(root.task_id)
    assert status == "completed"
    assert len(store.artifacts) == 0
    assert len(store.proposals) == 0
    assert len(store.children.get(root.task_id, [])) == 0
    result = await store.get_result(root.task_id)
    assert result is not None
    assert result["status"] == "completed"


@pytest.mark.anyio
async def test_legacy_push_analysis_runtime_compatibility_direct_task() -> None:
    settings = _make_auth_settings(delivery_mode="legacy_push")
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(
        settings=settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    root = TaskRequested.create("Analysis direct legacy task")
    await store.register_root_task(root, required_capability="analysis")
    await store.update_status(root.task_id, "pending")

    client = OrchestratorClient(
        base_url="http://test",
        runtime_token=ANALYSIS_TOKEN,
        transport=httpx.ASGITransport(app=app),
    )

    # Analysis direct task flow: claim as execution_kind="initial" -> acknowledge -> process
    assignment, proof = await client.claim_task(
        task_id=root.task_id,
        request_id="req-analysis-leg",
        capability="analysis",
        execution_kind="initial",
    )
    assert assignment["assignment_id"]
    assert proof

    await client.acknowledge_assignment(
        root.task_id, assignment["assignment_id"], proof
    )

    bus = FakeBus()
    processor = AnalysisProcessor(
        agent_id="analysis-agent-1",
        bus=bus,
        analyzer=DeterministicMaterialAnalyzer(),
        completion_client=client,
    )
    await processor.run_once(
        root,
        assignment_id=assignment["assignment_id"],
        assignment_proof=proof,
        execution_kind="initial",
    )

    # Verify task completed canonically and child input was never requested
    status = await store.get_status(root.task_id)
    assert status == "completed"
    result = await store.get_result(root.task_id)
    assert result is not None
    assert result["status"] == "completed"
    assert "Analysis report" in (result["output"] or "")


def test_legacy_push_partial_configuration_fails_startup() -> None:
    # 1. Proof pepper only
    with pytest.raises(
        ValueError,
        match="runtime_credential_pepper must be configured when lifecycle configuration is present",
    ):
        Settings(
            task_delivery_mode="legacy_push",
            assignment_proof_pepper=PROOF_PEPPER,
        )

    # 2. Runtime pepper only
    with pytest.raises(
        ValueError,
        match="assignment_proof_pepper must be configured when lifecycle configuration is present",
    ):
        Settings(
            task_delivery_mode="legacy_push",
            runtime_credential_pepper=PEPPER,
        )

    # 3. Principal registry only
    with pytest.raises(
        ValueError,
        match="runtime_credential_pepper must be configured when lifecycle configuration is present",
    ):
        Settings(
            task_delivery_mode="legacy_push",
            runtime_principals=PRINCIPALS_CONFIG,
        )

    # 4. Both peppers, no registry
    with pytest.raises(
        ValueError,
        match="runtime_principals must contain at least one principal when lifecycle configuration is present",
    ):
        Settings(
            task_delivery_mode="legacy_push",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals={},
        )

    # 5. Registry plus only runtime pepper
    with pytest.raises(
        ValueError,
        match="assignment_proof_pepper must be configured when lifecycle configuration is present",
    ):
        Settings(
            task_delivery_mode="legacy_push",
            runtime_credential_pepper=PEPPER,
            runtime_principals=PRINCIPALS_CONFIG,
        )

    # 6. Registry plus only proof pepper
    with pytest.raises(
        ValueError,
        match="runtime_credential_pepper must be configured when lifecycle configuration is present",
    ):
        Settings(
            task_delivery_mode="legacy_push",
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=PRINCIPALS_CONFIG,
        )

    # 7. Complete configuration with malformed principal
    with pytest.raises(ValueError, match="Invalid principal configuration"):
        Settings(
            task_delivery_mode="legacy_push",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals={
                "p-malformed": {
                    "principal_id": "p-malformed",
                    "authorized_capabilities": ["research"],
                    "token_digest": "a" * 64,
                    "active": True,
                }
            },
        )


def test_principal_entry_validation_independent_cases() -> None:
    valid_base = {
        "principal_id": "p-test",
        "bound_agent_id": "agent-valid",
        "token_digest": "a" * 64,
        "authorized_capabilities": ["research"],
        "active": True,
    }

    def check_invalid(entry: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_principal_config_registry({"p-test": entry})

    # 1. Missing bound_agent_id
    entry = dict(valid_base)
    del entry["bound_agent_id"]
    check_invalid(entry, "bound_agent_id must be a non-empty string")

    # 2. Empty bound_agent_id (empty string and whitespace)
    check_invalid(dict(valid_base, bound_agent_id=""), "bound_agent_id must be a non-empty string")
    check_invalid(dict(valid_base, bound_agent_id="   "), "bound_agent_id must be a non-empty string")

    # 3. Missing token_digest
    entry = dict(valid_base)
    del entry["token_digest"]
    check_invalid(entry, "token_digest must be a 64-character lowercase hexadecimal string")

    # 4. Malformed token_digest
    check_invalid(dict(valid_base, token_digest="a" * 63), "64-character lowercase hexadecimal string")
    check_invalid(dict(valid_base, token_digest="a" * 65), "64-character lowercase hexadecimal string")
    check_invalid(dict(valid_base, token_digest="A" * 64), "64-character lowercase hexadecimal string")
    check_invalid(dict(valid_base, token_digest="g" * 64), "64-character lowercase hexadecimal string")

    # 5. Empty capabilities
    entry = dict(valid_base)
    del entry["authorized_capabilities"]
    check_invalid(entry, "authorized_capabilities must be a non-empty collection")
    check_invalid(dict(valid_base, authorized_capabilities=[]), "authorized_capabilities must be a non-empty collection")

    # 6. Unsupported capability
    check_invalid(dict(valid_base, authorized_capabilities=["unsupported_cap"]), "unrecognized or empty capability")
    check_invalid(dict(valid_base, authorized_capabilities=[""]), "unrecognized or empty capability")

    # 7. Duplicate token digest
    reg_dup_digest = {
        "p1": dict(valid_base, principal_id="p1", bound_agent_id="a1", token_digest="f" * 64),
        "p2": dict(valid_base, principal_id="p2", bound_agent_id="a2", token_digest="f" * 64),
    }
    with pytest.raises(ValueError, match="duplicate token_digest"):
        validate_principal_config_registry(reg_dup_digest)

    # 8. Duplicate active bound_agent_id
    reg_dup_agent = {
        "p1": dict(valid_base, principal_id="p1", bound_agent_id="shared-agent", token_digest="1" * 64),
        "p2": dict(valid_base, principal_id="p2", bound_agent_id="shared-agent", token_digest="2" * 64),
    }
    with pytest.raises(ValueError, match="duplicate bound_agent_id for active principal"):
        validate_principal_config_registry(reg_dup_agent)

    # 9. Inactive principal behavior
    # All inactive fails startup
    reg_all_inactive = {
        "p1": dict(valid_base, principal_id="p1", active=False),
    }
    with pytest.raises(ValueError, match="must contain at least one active principal"):
        validate_principal_config_registry(reg_all_inactive)

    # Mixed active and inactive succeeds startup
    reg_mixed = {
        "p1": dict(valid_base, principal_id="p1", bound_agent_id="a1", token_digest="1" * 64, active=True),
        "p2": dict(valid_base, principal_id="p2", bound_agent_id="a2", token_digest="2" * 64, active=False),
    }
    validated = validate_principal_config_registry(reg_mixed)
    assert len(validated) == 2
    assert validated["p1"]["active"] is True
    assert validated["p2"]["active"] is False

    # 10. Unknown fields
    entry_unknown = dict(valid_base, unknown_field="forbidden")
    check_invalid(entry_unknown, "unexpected field\\(s\\)")

    # 11. Missing active field
    entry_no_active = dict(valid_base)
    del entry_no_active["active"]
    check_invalid(entry_no_active, "active field must be explicitly configured")

    # 12. Non-bool active field
    check_invalid(dict(valid_base, active="true"), "active must be a boolean")
    check_invalid(dict(valid_base, active=1), "active must be a boolean")


@pytest.mark.anyio
async def test_authentication_and_principal_binding_regressions() -> None:
    # 1. A principal missing bound_agent_id cannot authenticate because startup fails
    malformed_registry = {
        "p-malformed": {
            "principal_id": "p-malformed",
            "token_digest": compute_token_digest("token-malformed", PEPPER),
            "authorized_capabilities": ["research"],
            "active": True,
        }
    }
    with pytest.raises(ValueError, match="bound_agent_id must be a non-empty string"):
        Settings(
            task_delivery_mode="legacy_push",
            runtime_credential_pepper=PEPPER,
            assignment_proof_pepper=PROOF_PEPPER,
            runtime_principals=malformed_registry,
        )

    # 2. No fallback binds principal_id as agent_id; valid principal authenticates to exact configured bound_agent_id
    distinct_token = "secret-distinct-token"
    inactive_token = "secret-inactive-token"
    research_only_token = "secret-research-only-token"
    operator_only_token = "secret-operator-only-token"

    test_principals: dict[str, dict[str, Any]] = {
        "principal-uuid-999": {
            "principal_id": "principal-uuid-999",
            "bound_agent_id": "agent-distinct-name-888",
            "token_digest": compute_token_digest(distinct_token, PEPPER),
            "authorized_capabilities": ["research", "general"],
            "active": True,
        },
        "principal-inactive": {
            "principal_id": "principal-inactive",
            "bound_agent_id": "agent-dormant",
            "token_digest": compute_token_digest(inactive_token, PEPPER),
            "authorized_capabilities": ["research"],
            "active": False,
        },
        "principal-research-only": {
            "principal_id": "principal-research-only",
            "bound_agent_id": "research-worker-1",
            "token_digest": compute_token_digest(research_only_token, PEPPER),
            "authorized_capabilities": ["research"],
            "active": True,
        },
        "principal-op": {
            "principal_id": "principal-op",
            "bound_agent_id": "operator-runtime",
            "token_digest": compute_token_digest(operator_only_token, PEPPER),
            "authorized_capabilities": ["recovery_operator"],
            "active": True,
        },
    }

    settings = Settings(
        task_delivery_mode="legacy_push",
        runtime_credential_pepper=PEPPER,
        assignment_proof_pepper=PROOF_PEPPER,
        runtime_principals=test_principals,
    )
    store = InMemoryAuthoritativeTaskStore()
    app = create_app(
        settings=settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
    )

    root = TaskRequested.create("Research test prompt")
    await store.register_root_task(root, required_capability="research")
    await store.update_status(root.task_id, "pending")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 3. Valid principal claims task: assigned_agent_id matches bound_agent_id, NOT principal_id
        claim_res = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-distinct-1",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {distinct_token}"},
        )
        assert claim_res.status_code == 200
        assigned_agent = claim_res.json()["assignment"]["bound_agent_id"]
        runtime_principal = claim_res.json()["assignment"]["runtime_principal_id"]
        assert assigned_agent == "agent-distinct-name-888"
        assert runtime_principal == "principal-uuid-999"
        assert assigned_agent != runtime_principal

        # 4. Wrong token returns 401
        res_wrong = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-wrong-1",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": "Bearer totally-invalid-token-12345"},
        )
        assert res_wrong.status_code == 401

        # 5. Inactive principal returns 403
        res_inactive = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root.task_id,
                "request_id": "req-inactive-1",
                "capability": "research",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {inactive_token}"},
        )
        assert res_inactive.status_code == 403
        assert "Principal is inactive" in res_inactive.json()["detail"]

        # 6. Capability checks enforced: research-only principal claiming analysis returns 403
        root_analysis = TaskRequested.create("Analysis test prompt")
        await store.register_root_task(root_analysis, required_capability="analysis")
        await store.update_status(root_analysis.task_id, "pending")

        res_unauth_cap = await client.post(
            "/api/tasks/claim",
            json={
                "task_id": root_analysis.task_id,
                "request_id": "req-cap-1",
                "capability": "analysis",
                "execution_kind": "initial",
            },
            headers={"Authorization": f"Bearer {research_only_token}"},
        )
        assert res_unauth_cap.status_code == 403

        # 7. recovery_operator required for expiry/recovery
        # Research principal (not recovery_operator) calling evaluate-expiry returns 403
        res_eval_unauth = await client.post(
            f"/api/tasks/{root.task_id}/evaluate-expiry",
            json={},
            headers={"Authorization": f"Bearer {research_only_token}"},
        )
        assert res_eval_unauth.status_code == 403

        # Research principal calling resolve-expired-claim returns 403
        res_res_unauth = await client.post(
            f"/api/tasks/{root.task_id}/resolve-expired-claim",
            json={
                "recovery_decision_id": "rec-1",
                "expired_claim_id": "c-1",
                "expired_assignment_id": "as-1",
                "expired_fencing_epoch": 1,
                "disposition": "restore_claimable",
            },
            headers={"Authorization": f"Bearer {research_only_token}"},
        )
        assert res_res_unauth.status_code == 403

        # Principal with recovery_operator calling evaluate-expiry succeeds
        res_eval_auth = await client.post(
            f"/api/tasks/{root.task_id}/evaluate-expiry",
            json={},
            headers={"Authorization": f"Bearer {operator_only_token}"},
        )
        assert res_eval_auth.status_code == 200


def test_lifecycle_env_whitespace_and_absent_matrix(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Helper to clean environment before each scenario
    def clear_lifecycle_env() -> None:
        monkeypatch.delenv("RUNTIME_CREDENTIAL_PEPPER", raising=False)
        monkeypatch.delenv("ASSIGNMENT_PROOF_PEPPER", raising=False)
        monkeypatch.delenv("RUNTIME_PRINCIPALS_JSON", raising=False)

    valid_principals_json = json.dumps(PRINCIPALS_CONFIG)

    # ==================== legacy_push Matrix ====================
    monkeypatch.setenv("TASK_DELIVERY_MODE", "legacy_push")

    # 1. All three variables genuinely absent: startup succeeds unconfigured
    clear_lifecycle_env()
    s1 = Settings.from_env()
    assert s1.task_delivery_mode == "legacy_push"
    assert not s1.assignment_lifecycle_configured
    assert not s1.executable_delegation_enabled

    # 2. Runtime pepper present as "": startup fails
    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", "")
    with pytest.raises(ValueError) as exc:
        Settings.from_env()
    assert "Invalid RUNTIME_CREDENTIAL_PEPPER configuration" in str(exc.value)

    # 3. Runtime pepper present as whitespace: startup fails
    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", "   \t  ")
    with pytest.raises(ValueError) as exc:
        Settings.from_env()
    assert "Invalid RUNTIME_CREDENTIAL_PEPPER configuration" in str(exc.value)

    # 3b. Runtime pepper present with leading/trailing whitespace: startup fails
    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", " " + PEPPER)
    with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
        Settings.from_env()

    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", PEPPER + " ")
    with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
        Settings.from_env()

    # 4. Proof pepper present as "": startup fails
    clear_lifecycle_env()
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", "")
    with pytest.raises(ValueError) as exc:
        Settings.from_env()
    assert "Invalid ASSIGNMENT_PROOF_PEPPER configuration" in str(exc.value)

    # 5. Proof pepper present as whitespace: startup fails
    clear_lifecycle_env()
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", "   ")
    with pytest.raises(ValueError) as exc:
        Settings.from_env()
    assert "Invalid ASSIGNMENT_PROOF_PEPPER configuration" in str(exc.value)

    # 5b. Proof pepper present with leading/trailing whitespace: startup fails
    clear_lifecycle_env()
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", " " + PROOF_PEPPER)
    with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
        Settings.from_env()

    clear_lifecycle_env()
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", PROOF_PEPPER + " ")
    with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
        Settings.from_env()

    # 6. Principal registry present as "": startup fails
    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", "")
    with pytest.raises(ValueError) as exc:
        Settings.from_env()
    assert "Invalid RUNTIME_PRINCIPALS_JSON configuration" in str(exc.value)

    # 7. Principal registry present as whitespace: startup fails
    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", "   ")
    with pytest.raises(ValueError) as exc:
        Settings.from_env()
    assert "Invalid RUNTIME_PRINCIPALS_JSON configuration" in str(exc.value)

    # 8. Principal registry present as malformed JSON: startup fails without echoing raw value
    clear_lifecycle_env()
    malformed_json_str = "{this is completely invalid json 12345"
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", malformed_json_str)
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as exc:
        Settings.from_env()
    assert "Invalid RUNTIME_PRINCIPALS_JSON configuration" in str(exc.value)
    assert malformed_json_str not in str(exc.value)
    assert malformed_json_str not in caplog.text

    # 9. Complete valid group under legacy_push: startup succeeds configured
    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", PEPPER)
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", PROOF_PEPPER)
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", valid_principals_json)
    s9 = Settings.from_env()
    assert s9.task_delivery_mode == "legacy_push"
    assert s9.assignment_lifecycle_configured is True
    assert s9.executable_delegation_enabled is False

    # ==================== authenticated_claim Matrix ====================
    monkeypatch.setenv("TASK_DELIVERY_MODE", "authenticated_claim")

    # 10. Truly absent under authenticated_claim: startup fails
    clear_lifecycle_env()
    with pytest.raises(
        ValueError,
        match="runtime_credential_pepper must be configured when task_delivery_mode is 'authenticated_claim'",
    ):
        Settings.from_env()

    # 11. Representative whitespace-only cases for authenticated_claim:
    # 11a. Runtime pepper whitespace
    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", "   ")
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", PROOF_PEPPER)
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", valid_principals_json)
    with pytest.raises(ValueError, match="Invalid RUNTIME_CREDENTIAL_PEPPER configuration"):
        Settings.from_env()

    # 11b. Proof pepper whitespace
    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", PEPPER)
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", "   ")
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", valid_principals_json)
    with pytest.raises(ValueError, match="Invalid ASSIGNMENT_PROOF_PEPPER configuration"):
        Settings.from_env()

    # 11c. Principal registry whitespace
    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", PEPPER)
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", PROOF_PEPPER)
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", "   ")
    with pytest.raises(ValueError, match="Invalid RUNTIME_PRINCIPALS_JSON configuration"):
        Settings.from_env()

    # 12. Valid group under authenticated_claim: startup succeeds
    clear_lifecycle_env()
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", PEPPER)
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", PROOF_PEPPER)
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", valid_principals_json)
    s12 = Settings.from_env()
    assert s12.task_delivery_mode == "authenticated_claim"
    assert s12.assignment_lifecycle_configured is True
    assert s12.executable_delegation_enabled is True


def test_never_normalize_cryptographic_pepper_bytes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 1. parse_master_secret rejects empty and surrounding whitespace
    with pytest.raises(ValueError, match="Secret pepper must be a non-empty string"):
        parse_master_secret("")

    with pytest.raises(ValueError, match="Secret pepper must be a non-empty string"):
        parse_master_secret("   ")

    with pytest.raises(
        ValueError, match="Secret pepper cannot contain leading or trailing whitespace"
    ):
        parse_master_secret(" " + PROOF_PEPPER)

    with pytest.raises(
        ValueError, match="Secret pepper cannot contain leading or trailing whitespace"
    ):
        parse_master_secret(PROOF_PEPPER + " ")

    with pytest.raises(
        ValueError, match="Secret pepper cannot contain leading or trailing whitespace"
    ):
        parse_master_secret("   " + PROOF_PEPPER + "   ")

    # 2. Both peppers in Settings direct construction reject surrounding whitespace
    with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
        Settings(runtime_credential_pepper=" " + PEPPER)

    with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
        Settings(runtime_credential_pepper=PEPPER + " ")

    with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
        Settings(assignment_proof_pepper=" " + PROOF_PEPPER)

    with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
        Settings(assignment_proof_pepper=PROOF_PEPPER + " ")

    # 3. Environment loading rejects leading, trailing, surrounding whitespace without leaking secrets
    secret_candidate = "secret_with_trailing_space_test_12345"
    monkeypatch.setenv("TASK_DELIVERY_MODE", "legacy_push")
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", secret_candidate + " ")
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as exc_info:
        Settings.from_env()
    assert "cannot contain leading or trailing whitespace" in str(exc_info.value)
    assert secret_candidate not in str(exc_info.value)
    assert secret_candidate not in caplog.text

    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", PEPPER)
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", secret_candidate + " ")
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as exc_info:
        Settings.from_env()
    assert "cannot contain leading or trailing whitespace" in str(exc_info.value)
    assert secret_candidate not in str(exc_info.value)
    assert secret_candidate not in caplog.text

    # 4. Valid pepper is preserved exactly without silent transformation
    monkeypatch.setenv("RUNTIME_CREDENTIAL_PEPPER", PEPPER)
    monkeypatch.setenv("ASSIGNMENT_PROOF_PEPPER", PROOF_PEPPER)
    monkeypatch.setenv("RUNTIME_PRINCIPALS_JSON", json.dumps(PRINCIPALS_CONFIG))
    s = Settings.from_env()
    assert s.runtime_credential_pepper == PEPPER
    assert s.assignment_proof_pepper == PROOF_PEPPER

    # 5. compute_token_digest and verify_token use the exact accepted pepper
    token = "test-token-exact-1"
    digest = compute_token_digest(token, s.runtime_credential_pepper)
    assert verify_token(token, s.runtime_credential_pepper, digest) is True
    # Tampering with pepper even by one byte or whitespace fails verification
    assert verify_token(token, s.runtime_credential_pepper + "x", digest) is False


def test_compose_yaml_lifecycle_omission_and_passthrough() -> None:
    compose_path = Path(__file__).resolve().parent.parent / "compose.yaml"
    assert compose_path.exists(), "compose.yaml not found"

    with open(compose_path) as f:
        compose_content = f.read()

    # 1. Deterministic text inspection: ensure no empty fallback defaults (:- or :-) exist for lifecycle variables
    assert "RUNTIME_CREDENTIAL_PEPPER: ${RUNTIME_CREDENTIAL_PEPPER:-}" not in compose_content
    assert "ASSIGNMENT_PROOF_PEPPER: ${ASSIGNMENT_PROOF_PEPPER:-}" not in compose_content
    assert "RUNTIME_PRINCIPALS_JSON: ${RUNTIME_PRINCIPALS_JSON:-}" not in compose_content

    # 2. PyYAML parsing verification: pass-through keys parse as None in environment map
    parsed = yaml.safe_load(compose_content)
    orch_env = parsed["services"]["orchestrator"]["environment"]
    assert orch_env["RUNTIME_CREDENTIAL_PEPPER"] is None
    assert orch_env["ASSIGNMENT_PROOF_PEPPER"] is None
    assert orch_env["RUNTIME_PRINCIPALS_JSON"] is None

    # 3. If docker compose is executable, verify rendered JSON and pass-through behavior
    compose_bin = shutil.which("docker")
    if compose_bin:
        # A. Unset lifecycle variables: keys render as null (omitted from container environment)
        res_unset = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_path),
                "--env-file",
                "/dev/null",
                "config",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_unset.returncode == 0:
            data_unset = json.loads(res_unset.stdout)
            env_unset = data_unset["services"]["orchestrator"]["environment"]
            assert env_unset["RUNTIME_CREDENTIAL_PEPPER"] is None
            assert env_unset["ASSIGNMENT_PROOF_PEPPER"] is None
            assert env_unset["RUNTIME_PRINCIPALS_JSON"] is None
            # Filtered active container environment omits all three keys
            active_keys = {k for k, v in env_unset.items() if v is not None}
            assert "RUNTIME_CREDENTIAL_PEPPER" not in active_keys
            assert "ASSIGNMENT_PROOF_PEPPER" not in active_keys
            assert "RUNTIME_PRINCIPALS_JSON" not in active_keys

        # B. Set lifecycle variables: rendered configuration includes exact supplied values
        custom_env = dict(
            os.environ,
            RUNTIME_CREDENTIAL_PEPPER=PEPPER,
            ASSIGNMENT_PROOF_PEPPER=PROOF_PEPPER,
            RUNTIME_PRINCIPALS_JSON=json.dumps(PRINCIPALS_CONFIG),
        )
        res_set = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_path),
                "--env-file",
                "/dev/null",
                "config",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            env=custom_env,
            check=False,
        )
        if res_set.returncode == 0:
            data_set = json.loads(res_set.stdout)
            env_set = data_set["services"]["orchestrator"]["environment"]
            assert env_set["RUNTIME_CREDENTIAL_PEPPER"] == PEPPER
            assert env_set["ASSIGNMENT_PROOF_PEPPER"] == PROOF_PEPPER
            assert env_set["RUNTIME_PRINCIPALS_JSON"] == json.dumps(PRINCIPALS_CONFIG)
