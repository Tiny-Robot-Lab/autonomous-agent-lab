"""API, authentication, and authorization tests for fenced turn completion (#222).

Exercises POST /api/tasks/{task_id}/completion through the authenticated
CompleteTurn use case against the deterministic in-memory store: bearer auth,
detached X-Assignment-Proof binding, identity/epoch/version matching, the
canonical byte bound, the safe failure-code allowlist, and the uniform generic
403 for every failed retired-replay authorization (no proof, secret, result,
outcome, task-state, or existence leakage).
"""

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from agent_contracts import (
    ArtifactReference,
    CompletionCommand,
    StageArtifact,
    TaskContinuation,
    TaskRequested,
    TaskResult,
    build_completion_audit,
    build_terminal_outcome,
    derive_terminal_outcome_id,
)
from orchestrator.application.ports import (
    ClaimNotFoundError,
    CompletionConflictError,
    TaskNotClaimableError,
)
from orchestrator.application.use_cases import (
    AcceptDelegationProposal,
    AcknowledgeAssignment,
    ClaimTask,
    CompleteTurn,
)
from orchestrator.config import Settings
from orchestrator.domain.principal import RuntimePrincipal, compute_token_digest
from orchestrator.domain.task import CorruptAuthoritativeRecordError
from orchestrator.presentation.api import create_app

from tests.test_authoritative_task_store import InMemoryAuthoritativeTaskStore, MockAtomicDispatcher

PEPPER = "test_completion_pepper_secret_1234567890"
PROOF_PEPPER = "test_completion_proof_pepper_4567890"
RESEARCH_TOKEN = "secret-token-completion-research-1"
WORKER_TOKEN = "secret-token-completion-worker-1"
ANALYSIS_TOKEN = "secret-token-completion-analysis-1"

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
        "token_digest": compute_token_digest("inactive-completion-token", PEPPER),
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


def make_client(
    store: InMemoryAuthoritativeTaskStore, settings: Settings
) -> httpx.AsyncClient:
    app = create_app(settings=settings, task_store=store, dispatcher=MockAtomicDispatcher(store))
    assert app.state.complete_turn is not None
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def register_pending_root(
    store: InMemoryAuthoritativeTaskStore,
    task_id: str,
    capability: str,
    prompt: str = "Complete this task",
) -> None:
    now_iso = datetime.now(UTC).isoformat()
    task = TaskRequested(
        task_id=task_id,
        root_task_id=task_id,
        parent_task_id=None,
        prompt=prompt,
        created_at=now_iso,
        version=2,
    )
    await store.register_root_task(task, required_capability=capability)
    await store.update_status(task_id, "pending")


async def claim_and_acknowledge(
    client: httpx.AsyncClient,
    token: str,
    task_id: str,
    capability: str,
    execution_kind: str = "initial",
    expected_continuation_version: int | None = None,
) -> tuple[dict[str, Any], str]:
    claim_payload: dict[str, Any] = {
        "task_id": task_id,
        "request_id": f"req-{task_id}",
        "capability": capability,
        "execution_kind": execution_kind,
    }
    if expected_continuation_version is not None:
        claim_payload["expected_continuation_version"] = expected_continuation_version
    res = await client.post(
        "/api/tasks/claim",
        json=claim_payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assignment = body["assignment"]
    proof = body["assignment_proof"]
    ack = await client.post(
        f"/api/assignments/{assignment['assignment_id']}/acknowledge",
        json={"task_id": task_id, "assignment_proof": proof},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert ack.status_code == 200, ack.text
    return assignment, proof


async def post_completion(
    client: httpx.AsyncClient,
    token: str | None,
    task_id: str,
    command: dict[str, Any],
    proof: str | None,
) -> httpx.Response:
    headers: dict[str, str] = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if proof is not None:
        headers["X-Assignment-Proof"] = proof
    return await client.post(f"/api/tasks/{task_id}/completion", json=command, headers=headers)


def success_command(assignment_id: str, output: str = "done") -> dict[str, Any]:
    return {
        "completion_command_id": "cmd-1",
        "assignment_id": assignment_id,
        "execution_kind": "initial",
        "result": {"task_id": "placeholder", "output": output, "status": "completed"},
    }


# --- Authentication -----------------------------------------------------------------


@pytest.mark.anyio
async def test_completion_requires_bearer_auth(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    async with make_client(store, test_settings) as client:
        res = await client.post("/api/tasks/task-x/completion", json={})
        assert res.status_code == 401


@pytest.mark.anyio
async def test_completion_rejects_invalid_token(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    async with make_client(store, test_settings) as client:
        res = await post_completion(client, "wrong-token", "task-x", {}, "0" * 64)
        assert res.status_code == 401


@pytest.mark.anyio
async def test_completion_rejects_inactive_principal(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    async with make_client(store, test_settings) as client:
        res = await post_completion(client, "inactive-completion-token", "task-x", {}, "0" * 64)
        assert res.status_code == 403


@pytest.mark.anyio
async def test_completion_requires_detached_proof_header(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-1", "general")
    async with make_client(store, test_settings) as client:
        assignment, _proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-auth-1", "general")
        command = success_command(assignment["assignment_id"])
        res = await post_completion(client, WORKER_TOKEN, "task-auth-1", command, None)
        assert res.status_code == 401
        assert "X-Assignment-Proof" in res.json()["detail"]


@pytest.mark.anyio
async def test_completion_rejects_invalid_proof(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-2", "general")
    async with make_client(store, test_settings) as client:
        assignment, _proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-auth-2", "general")
        command = success_command(assignment["assignment_id"])
        command["result"]["task_id"] = "task-auth-2"
        res = await post_completion(client, WORKER_TOKEN, "task-auth-2", command, "0" * 64)
        assert res.status_code == 403


@pytest.mark.anyio
async def test_completion_wrong_principal_is_not_found(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-3", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-auth-3", "general")
        command = success_command(assignment["assignment_id"])
        command["result"]["task_id"] = "task-auth-3"
        # Research principal presents the worker's proof: no matching active claim.
        res = await post_completion(client, RESEARCH_TOKEN, "task-auth-3", command, proof)
        assert res.status_code == 404


@pytest.mark.anyio
async def test_completion_wrong_assignment_is_generic_forbidden(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-4", "general")
    async with make_client(store, test_settings) as client:
        _assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-auth-4", "general")
        command = success_command("assign-unknown")
        command["result"]["task_id"] = "task-auth-4"
        res = await post_completion(client, WORKER_TOKEN, "task-auth-4", command, proof)
        assert res.status_code == 403
        assert res.json() == {"detail": "Completion replay authorization failed"}


@pytest.mark.anyio
async def test_completion_wrong_task_is_generic_forbidden(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-5", "general")
    await register_pending_root(store, "task-auth-6", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-auth-5", "general")
        command = success_command(assignment["assignment_id"])
        command["result"]["task_id"] = "task-auth-6"
        res = await post_completion(client, WORKER_TOKEN, "task-auth-6", command, proof)
        assert res.status_code == 403
        assert res.json() == {"detail": "Completion replay authorization failed"}
        assert await store.get_terminal_outcome("task-auth-6") is None


@pytest.mark.anyio
async def test_completion_wrong_execution_kind_is_forbidden(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-7", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-auth-7", "general")
        command = success_command(assignment["assignment_id"])
        command["result"]["task_id"] = "task-auth-7"
        command["execution_kind"] = "delegated_child"
        res = await post_completion(client, WORKER_TOKEN, "task-auth-7", command, proof)
        assert res.status_code == 403


@pytest.mark.anyio
async def test_completion_rejects_oversized_result(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-8", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-auth-8", "general")
        command = success_command(assignment["assignment_id"], output="y" * 70_000)
        res = await post_completion(client, WORKER_TOKEN, "task-auth-8", command, proof)
        assert res.status_code == 400
        assert await store.get_terminal_outcome("task-auth-8") is None


@pytest.mark.anyio
async def test_completion_rejects_unsupported_failure_code(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-9", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-auth-9", "general")
        command = {
            "completion_command_id": "cmd-9",
            "assignment_id": assignment["assignment_id"],
            "execution_kind": "initial",
            "failure_code": "DEADLINE_EXCEEDED",
        }
        res = await post_completion(client, WORKER_TOKEN, "task-auth-9", command, proof)
        assert res.status_code == 400
        assert await store.get_terminal_outcome("task-auth-9") is None


@pytest.mark.anyio
async def test_completion_rejects_proof_in_json_body(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-10", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-auth-10", "general")
        command = success_command(assignment["assignment_id"])
        command["assignment_proof"] = proof
        res = await post_completion(client, WORKER_TOKEN, "task-auth-10", command, proof)
        assert res.status_code == 400


@pytest.mark.anyio
async def test_completion_rejects_stale_fence(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-11", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-auth-11", "general")
        store.fences["task-auth-11"] = 99
        command = success_command(assignment["assignment_id"])
        command["result"]["task_id"] = "task-auth-11"
        res = await post_completion(client, WORKER_TOKEN, "task-auth-11", command, proof)
        assert res.status_code == 409
        assert await store.get_terminal_outcome("task-auth-11") is None


@pytest.mark.anyio
async def test_completion_wrong_bound_agent_is_not_found(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-auth-12", "general")
    claim_uc = ClaimTask(
        task_store=store,
        assignment_proof_pepper=PROOF_PEPPER,
        delivery_lease_seconds=30.0,
    )
    ack_uc = AcknowledgeAssignment(task_store=store, assignment_proof_pepper=PROOF_PEPPER)
    principal = RuntimePrincipal(
        principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        authorized_capabilities=("general",),
        active=True,
    )
    claim_res = await claim_uc.execute(
        principal=principal,
        task_id="task-auth-12",
        request_id="req-x",
        capability="general",
    )
    await ack_uc.execute(
        principal=principal,
        task_id="task-auth-12",
        assignment_id=claim_res.assignment.assignment_id,
        assignment_proof=claim_res.assignment_proof or "",
    )
    complete_uc = CompleteTurn(
        task_store=store,
        completion_store=store,
        assignment_proof_pepper=PROOF_PEPPER,
    )
    impostor = RuntimePrincipal(
        principal_id="principal-worker-1",
        bound_agent_id="worker-impostor",
        authorized_capabilities=("general",),
        active=True,
    )
    command = CompletionCommand(
        completion_command_id="cmd-12",
        assignment_id=claim_res.assignment.assignment_id,
        execution_kind="initial",
        completed_result=TaskResult(task_id="task-auth-12", output="done"),
    )

    with pytest.raises(ClaimNotFoundError):
        await complete_uc.execute(
            principal=impostor,
            task_id="task-auth-12",
            command_payload=command.to_dict(),
            assignment_proof=claim_res.assignment_proof or "",
        )
    assert await store.get_terminal_outcome("task-auth-12") is None


# --- Successful first completions -------------------------------------------------------


@pytest.mark.anyio
async def test_worker_initial_completed_is_canonical(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-worker-1", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-worker-1", "general")
        command = success_command(assignment["assignment_id"], output="worker output")
        command["result"]["task_id"] = "task-worker-1"
        res = await post_completion(client, WORKER_TOKEN, "task-worker-1", command, proof)
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["replayed"] is False
        outcome = body["outcome"]
        assert outcome["terminal_status"] == "completed"
        assert outcome["task_id"] == "task-worker-1"
        assert outcome["assignment_id"] == assignment["assignment_id"]
        assert outcome["runtime_principal_id"] == "principal-worker-1"
        assert outcome["bound_agent_id"] == "worker-1"
        assert outcome["execution_kind"] == "initial"
        assert outcome["completion_command_id"] == "cmd-1"
        assert "failure_code" not in outcome or outcome["failure_code"] is None
        assert body["result"]["output"] == "worker output"
        assert outcome["outcome_id"] == derive_terminal_outcome_id(outcome["semantic_fingerprint"])
        # No secret material anywhere in the response.
        assert proof not in res.text
        assert "assignment_proof" not in body
        assert "proof_digest" not in body["outcome"]

        assert await store.get_status("task-worker-1") == "completed"
        assert await store.get_active_claim("task-worker-1") is None
        stored_outcome = await store.get_terminal_outcome("task-worker-1")
        assert stored_outcome is not None
        assert stored_outcome.outcome_id == outcome["outcome_id"]
        stored_result = await store.get_canonical_result("task-worker-1")
        assert stored_result is not None
        assert stored_result.output == "worker output"
        audit = await store.get_assignment_completion("task-worker-1", assignment["assignment_id"])
        assert audit is not None
        assert audit.outcome_id == outcome["outcome_id"]
        assert audit.fencing_epoch == 1
        assert audit.completion_command_id == "cmd-1"


@pytest.mark.anyio
async def test_research_initial_completed(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-research-1", "research")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(
            client, RESEARCH_TOKEN, "task-research-1", "research"
        )
        command = success_command(assignment["assignment_id"], output="research synthesis")
        command["result"]["task_id"] = "task-research-1"
        res = await post_completion(client, RESEARCH_TOKEN, "task-research-1", command, proof)
        assert res.status_code == 200, res.text
        assert res.json()["outcome"]["terminal_status"] == "completed"
        assert await store.get_status("task-research-1") == "completed"


@pytest.mark.anyio
async def test_failed_turn_persists_outcome_without_result(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-fail-1", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-fail-1", "general")
        command = {
            "completion_command_id": "cmd-fail-1",
            "assignment_id": assignment["assignment_id"],
            "execution_kind": "initial",
            "failure_code": "TOOL_DENIED",
        }
        res = await post_completion(client, WORKER_TOKEN, "task-fail-1", command, proof)
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["replayed"] is False
        assert body["outcome"]["terminal_status"] == "failed"
        assert body["outcome"]["failure_code"] == "TOOL_DENIED"
        assert body["outcome"]["safe_summary"] == (
            "Task execution failed because a required tool invocation was denied."
        )
        assert "result" not in body
        assert await store.get_status("task-fail-1") == "failed"
        assert await store.get_active_claim("task-fail-1") is None
        assert await store.get_canonical_result("task-fail-1") is None
        audit = await store.get_assignment_completion("task-fail-1", assignment["assignment_id"])
        assert audit is not None


@pytest.mark.anyio
async def test_resumed_research_completed_with_continuation_version_1(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-resumed-1", "research")
    await store.update_status("task-resumed-1", "resumable")
    now_iso = datetime.now(UTC).isoformat()
    continuation = TaskContinuation(
        parent_task_id="task-resumed-1",
        child_task_id="child-prior-1",
        input_version=1,
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "prior"},
        continuation_version=1,
        created_at=now_iso,
        updated_at=now_iso,
    )
    await store.save_continuation(continuation)
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(
            client,
            RESEARCH_TOKEN,
            "task-resumed-1",
            "research",
            execution_kind="resumed_continuation",
            expected_continuation_version=1,
        )
        command = {
            "completion_command_id": "cmd-resumed-1",
            "assignment_id": assignment["assignment_id"],
            "execution_kind": "resumed_continuation",
            "continuation_version": 1,
            "result": {"task_id": "task-resumed-1", "output": "resumed synthesis", "status": "completed"},
        }
        res = await post_completion(client, RESEARCH_TOKEN, "task-resumed-1", command, proof)
        assert res.status_code == 200, res.text
        outcome = res.json()["outcome"]
        assert outcome["terminal_status"] == "completed"
        assert outcome["execution_kind"] == "resumed_continuation"
        assert outcome["continuation_version"] == 1
        assert await store.get_status("task-resumed-1") == "completed"
        audit = await store.get_assignment_completion("task-resumed-1", assignment["assignment_id"])
        assert audit is not None
        assert audit.continuation_version == 1


@pytest.mark.anyio
async def test_resumed_completion_rejects_wrong_continuation_version(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-resumed-2", "research")
    await store.update_status("task-resumed-2", "resumable")
    now_iso = datetime.now(UTC).isoformat()
    await store.save_continuation(
        TaskContinuation(
            parent_task_id="task-resumed-2",
            child_task_id="child-prior-2",
            input_version=1,
            stage="incorporate_analysis_child",
            control_fields={"analysis_focus": "prior"},
            continuation_version=1,
            created_at=now_iso,
            updated_at=now_iso,
        )
    )
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(
            client,
            RESEARCH_TOKEN,
            "task-resumed-2",
            "research",
            execution_kind="resumed_continuation",
            expected_continuation_version=1,
        )
        command = {
            "completion_command_id": "cmd-resumed-2",
            "assignment_id": assignment["assignment_id"],
            "execution_kind": "resumed_continuation",
            "continuation_version": 2,
            "failure_code": "EXECUTION_FAILED",
        }
        res = await post_completion(client, RESEARCH_TOKEN, "task-resumed-2", command, proof)
        assert res.status_code == 409
        assert await store.get_terminal_outcome("task-resumed-2") is None


# --- Delegated child completion with parent projection --------------------------------------


async def setup_delegated_child_turn(
    client: httpx.AsyncClient,
    store: InMemoryAuthoritativeTaskStore,
    parent_id: str = "task-parent-1",
) -> dict[str, Any]:
    """Drive research claim/ack, artifact staging, and delegation via HTTP.

    Returns parent/child assignment and proof material for completion tests.
    """
    await register_pending_root(store, parent_id, "research", prompt="Research question")
    assignment, proof = await claim_and_acknowledge(client, RESEARCH_TOKEN, parent_id, "research")
    auth_headers = {"Authorization": f"Bearer {RESEARCH_TOKEN}"}
    artifact_id = f"art-{parent_id}"
    stage_res = await client.post(
        f"/api/tasks/{parent_id}/artifacts",
        json={
            "assignment_id": assignment["assignment_id"],
            "artifact_id": artifact_id,
            "artifact_type": "research_material",
            "content": {"material": "key findings"},
            "artifact_version": 1,
            "payload_schema_version": 1,
        },
        headers={**auth_headers, "X-Assignment-Proof": proof},
    )
    assert stage_res.status_code == 200, stage_res.text
    delegation_res = await client.post(
        f"/api/tasks/{parent_id}/delegations",
        json={
            "proposal_id": f"prop-{parent_id}",
            "assignment_id": assignment["assignment_id"],
            "child_intent": "Analyze the staged findings",
            "artifact_reference": {
                "producing_task_id": parent_id,
                "artifact_id": artifact_id,
                "artifact_version": 1,
            },
            "target_capability": "analysis",
        },
        headers={**auth_headers, "X-Assignment-Proof": proof},
    )
    assert delegation_res.status_code == 200, delegation_res.text
    child_id = delegation_res.json()["child_task_id"]
    child_assignment, child_proof = await claim_and_acknowledge(
        client, ANALYSIS_TOKEN, child_id, "analysis", execution_kind="delegated_child"
    )
    return {
        "parent_id": parent_id,
        "parent_assignment": assignment,
        "parent_proof": proof,
        "child_id": child_id,
        "child_assignment": child_assignment,
        "child_proof": child_proof,
    }


@pytest.mark.anyio
async def test_delegated_child_completed_projects_parent_resumable(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    async with make_client(store, test_settings) as client:
        ctx = await setup_delegated_child_turn(client, store)
        child_id = ctx["child_id"]
        parent_id = ctx["parent_id"]
        hints_before = len(store.hints)

        command = {
            "completion_command_id": "cmd-child-1",
            "assignment_id": ctx["child_assignment"]["assignment_id"],
            "execution_kind": "delegated_child",
            "result": {"task_id": child_id, "output": "analysis report", "status": "completed"},
        }
        res = await post_completion(client, ANALYSIS_TOKEN, child_id, command, ctx["child_proof"])
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["replayed"] is False
        assert body["outcome"]["terminal_status"] == "completed"

        assert await store.get_status(child_id) == "completed"
        assert await store.get_active_claim(child_id) is None
        # Parent projection: dependency satisfied, version preserved, resumable.
        assert await store.get_status(parent_id) == "resumable"
        continuation = await store.get_continuation(parent_id, 1)
        assert continuation is not None
        assert continuation.child_task_id == child_id
        assert continuation.continuation_version == 1
        assert child_id in store.child_markers
        assert child_id in await store.get_children(parent_id)
        # Exactly one non-secret Research wake-up hint.
        assert len(store.hints) == hints_before + 1
        hint = store.hints[-1]
        assert hint.capability == "research"
        assert hint.task_id == parent_id
        assert hint.expected_continuation_version == 1
        assert hint.execution_kind == "resumed_continuation"
        # Canonical records, no secret material.
        assert await store.get_canonical_result(child_id) is not None
        audit = await store.get_assignment_completion(child_id, ctx["child_assignment"]["assignment_id"])
        assert audit is not None
        assert audit.fencing_epoch == 1


@pytest.mark.anyio
async def test_delegated_child_failed_projects_parent_without_result(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    async with make_client(store, test_settings) as client:
        ctx = await setup_delegated_child_turn(client, store, parent_id="task-parent-fail")
        child_id = ctx["child_id"]
        parent_id = ctx["parent_id"]
        hints_before = len(store.hints)

        command = {
            "completion_command_id": "cmd-child-fail-1",
            "assignment_id": ctx["child_assignment"]["assignment_id"],
            "execution_kind": "delegated_child",
            "failure_code": "PROVIDER_UNAVAILABLE",
        }
        res = await post_completion(client, ANALYSIS_TOKEN, child_id, command, ctx["child_proof"])
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["outcome"]["terminal_status"] == "failed"
        assert "result" not in body

        assert await store.get_status(child_id) == "failed"
        assert await store.get_canonical_result(child_id) is None
        assert await store.get_status(parent_id) == "resumable"
        assert len(store.hints) == hints_before + 1
        assert store.hints[-1].task_id == parent_id


# --- Replay and conflict --------------------------------------------------------------------


@pytest.mark.anyio
async def test_identical_replay_returns_recorded_outcome_with_zero_writes(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-replay-1", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-replay-1", "general")
        command = success_command(assignment["assignment_id"], output="stable")
        command["result"]["task_id"] = "task-replay-1"
        first = await post_completion(client, WORKER_TOKEN, "task-replay-1", command, proof)
        assert first.status_code == 200
        assert first.json()["replayed"] is False

        snapshot = (
            len(store.terminal_outcomes),
            len(store.canonical_results),
            len(store.assignment_completions),
            len(store.completion_records),
            len(store.hints),
            await store.get_status("task-replay-1"),
        )
        second = await post_completion(client, WORKER_TOKEN, "task-replay-1", command, proof)
        assert second.status_code == 200
        assert second.json()["replayed"] is True
        assert second.json()["outcome"] == first.json()["outcome"]
        assert second.json()["result"] == first.json()["result"]
        assert (
            len(store.terminal_outcomes),
            len(store.canonical_results),
            len(store.assignment_completions),
            len(store.completion_records),
            len(store.hints),
            await store.get_status("task-replay-1"),
        ) == snapshot


@pytest.mark.anyio
async def test_changed_payload_under_same_command_id_is_generic_forbidden(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-replay-2", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-replay-2", "general")
        command = success_command(assignment["assignment_id"], output="original")
        command["result"]["task_id"] = "task-replay-2"
        first = await post_completion(client, WORKER_TOKEN, "task-replay-2", command, proof)
        assert first.status_code == 200

        tampered = success_command(assignment["assignment_id"], output="tampered")
        tampered["result"]["task_id"] = "task-replay-2"
        res = await post_completion(client, WORKER_TOKEN, "task-replay-2", tampered, proof)
        assert res.status_code == 403
        assert res.json() == {"detail": "Completion replay authorization failed"}
        stored = await store.get_canonical_result("task-replay-2")
        assert stored is not None and stored.output == "original"


@pytest.mark.anyio
async def test_different_command_after_terminal_state_is_rejected(
    test_settings: Settings,
) -> None:

    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-replay-3", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-replay-3", "general")
        command = success_command(assignment["assignment_id"], output="first")
        command["result"]["task_id"] = "task-replay-3"
        first = await post_completion(client, WORKER_TOKEN, "task-replay-3", command, proof)
        assert first.status_code == 200

        other = success_command(assignment["assignment_id"], output="second")
        other["completion_command_id"] = "cmd-other"
        other["result"]["task_id"] = "task-replay-3"
        res = await post_completion(client, WORKER_TOKEN, "task-replay-3", other, proof)
        assert res.status_code == 403
        assert res.json() == {"detail": "Completion replay authorization failed"}

        # Store level fails closed with a conflict even when invoked directly.
        other_cmd = CompletionCommand(
            completion_command_id="cmd-other",
            assignment_id=assignment["assignment_id"],
            execution_kind="initial",
            completed_result=TaskResult(task_id="task-replay-3", output="second"),
        )
        with pytest.raises(CompletionConflictError):
            await store.complete_turn(
                task_id="task-replay-3",
                command=other_cmd,
                proof_digest="0" * 64,
                fencing_epoch=1,
                parent_task_id=None,
                root_task_id="task-replay-3",
                completed_at=datetime.now(UTC).isoformat(),
                runtime_principal_id="principal-worker-1",
                bound_agent_id="worker-1",
            )


@pytest.mark.anyio
async def test_store_replay_after_retirement_writes_nothing(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-replay-4", "general")
    async with make_client(store, test_settings):
        pass
    claim_uc = ClaimTask(task_store=store, assignment_proof_pepper=PROOF_PEPPER, delivery_lease_seconds=30.0)
    ack_uc = AcknowledgeAssignment(task_store=store, assignment_proof_pepper=PROOF_PEPPER)
    principal = RuntimePrincipal(
        principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        authorized_capabilities=("general",),
        active=True,
    )
    claim_res = await claim_uc.execute(
        principal=principal, task_id="task-replay-4", request_id="req-r4", capability="general"
    )
    claim = await store.get_active_claim("task-replay-4")
    assert claim is not None
    await ack_uc.execute(
        principal=principal,
        task_id="task-replay-4",
        assignment_id=claim_res.assignment.assignment_id,
        assignment_proof=claim_res.assignment_proof or "",
    )
    command = CompletionCommand(
        completion_command_id="cmd-r4",
        assignment_id=claim_res.assignment.assignment_id,
        execution_kind="initial",
        completed_result=TaskResult(task_id="task-replay-4", output="done"),
    )
    outcome, result, replayed = await store.complete_turn(
        task_id="task-replay-4",
        command=command,
        proof_digest=claim.proof_digest,
        fencing_epoch=claim.fencing_epoch,
        parent_task_id=None,
        root_task_id="task-replay-4",
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id=claim.runtime_principal_id,
        bound_agent_id=claim.bound_agent_id,
    )
    assert replayed is False
    assert result is not None

    counts_before = (
        len(store.completion_records),
        len(store.terminal_outcomes),
        len(store.canonical_results),
        len(store.assignment_completions),
        len(store.hints),
    )
    # Claim is retired; the identical command replays from the record alone.
    outcome2, result2, replayed2 = await store.complete_turn(
        task_id="task-replay-4",
        command=command,
        proof_digest=claim.proof_digest,
        fencing_epoch=claim.fencing_epoch,
        parent_task_id=None,
        root_task_id="task-replay-4",
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id=claim.runtime_principal_id,
        bound_agent_id=claim.bound_agent_id,
    )
    assert replayed2 is True
    assert outcome2 == outcome
    assert result2 == result
    assert (
        len(store.completion_records),
        len(store.terminal_outcomes),
        len(store.canonical_results),
        len(store.assignment_completions),
        len(store.hints),
    ) == counts_before


@pytest.mark.anyio
async def test_concurrent_identical_completions_converge(test_settings: Settings) -> None:
    import asyncio

    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-race-1", "general")
    async with make_client(store, test_settings):
        pass
    claim_uc = ClaimTask(task_store=store, assignment_proof_pepper=PROOF_PEPPER, delivery_lease_seconds=30.0)
    ack_uc = AcknowledgeAssignment(task_store=store, assignment_proof_pepper=PROOF_PEPPER)
    principal = RuntimePrincipal(
        principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        authorized_capabilities=("general",),
        active=True,
    )
    claim_res = await claim_uc.execute(
        principal=principal, task_id="task-race-1", request_id="req-race", capability="general"
    )
    claim = await store.get_active_claim("task-race-1")
    assert claim is not None
    await ack_uc.execute(
        principal=principal,
        task_id="task-race-1",
        assignment_id=claim_res.assignment.assignment_id,
        assignment_proof=claim_res.assignment_proof or "",
    )
    command = CompletionCommand(
        completion_command_id="cmd-race-1",
        assignment_id=claim_res.assignment.assignment_id,
        execution_kind="initial",
        completed_result=TaskResult(task_id="task-race-1", output="raced"),
    )

    async def attempt() -> tuple[bool, str]:
        outcome, _result, replayed = await store.complete_turn(
            task_id="task-race-1",
            command=command,
            proof_digest=claim.proof_digest,
            fencing_epoch=claim.fencing_epoch,
            parent_task_id=None,
            root_task_id="task-race-1",
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
        )
        return replayed, outcome.outcome_id

    first, second = await asyncio.gather(attempt(), attempt())
    assert sorted([first[0], second[0]]) == [False, True]
    assert first[1] == second[1]


@pytest.mark.anyio
async def test_concurrent_conflicting_completion_fails_closed(test_settings: Settings) -> None:
    import asyncio


    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-race-2", "general")
    async with make_client(store, test_settings):
        pass
    claim_uc = ClaimTask(task_store=store, assignment_proof_pepper=PROOF_PEPPER, delivery_lease_seconds=30.0)
    ack_uc = AcknowledgeAssignment(task_store=store, assignment_proof_pepper=PROOF_PEPPER)
    principal = RuntimePrincipal(
        principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        authorized_capabilities=("general",),
        active=True,
    )
    claim_res = await claim_uc.execute(
        principal=principal, task_id="task-race-2", request_id="req-race2", capability="general"
    )
    claim = await store.get_active_claim("task-race-2")
    assert claim is not None
    await ack_uc.execute(
        principal=principal,
        task_id="task-race-2",
        assignment_id=claim_res.assignment.assignment_id,
        assignment_proof=claim_res.assignment_proof or "",
    )

    async def attempt(output: str) -> str:
        cmd = CompletionCommand(
            completion_command_id="cmd-race-2",
            assignment_id=claim_res.assignment.assignment_id,
            execution_kind="initial",
            completed_result=TaskResult(task_id="task-race-2", output=output),
        )
        _outcome, _result, replayed = await store.complete_turn(
            task_id="task-race-2",
            command=cmd,
            proof_digest=claim.proof_digest,
            fencing_epoch=claim.fencing_epoch,
            parent_task_id=None,
            root_task_id="task-race-2",
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
        )
        return "replayed" if replayed else "accepted"

    results = await asyncio.gather(attempt("winner"), attempt("loser"), return_exceptions=True)
    kinds = sorted(
        "conflict" if isinstance(r, CompletionConflictError) else r for r in results  # type: ignore[union-attr]
    )
    assert kinds == ["accepted", "conflict"]
    stored = await store.get_canonical_result("task-race-2")
    assert stored is not None and stored.output in ("winner", "loser")


# --- Zero-mutation failures -------------------------------------------------------------------


def snapshot_store(store: InMemoryAuthoritativeTaskStore) -> dict[str, object]:
    return {
        "statuses": dict(store.statuses),
        "claims": dict(store.claims),
        "fences": dict(store.fences),
        "outcomes": dict(store.terminal_outcomes),
        "results": dict(store.canonical_results),
        "audits": dict(store.assignment_completions),
        "records": dict(store.completion_records),
        "markers": dict(store.child_markers),
        "hints": list(store.hints),
        "lineages": dict(store.lineages),
        "children": {k: set(v) for k, v in store.children.items()},
        "continuations": dict(store.continuations),
        "latest_continuation": dict(store.latest_continuation),
    }


async def setup_claimed_root(
    store: InMemoryAuthoritativeTaskStore, task_id: str, capability: str, token: str
) -> tuple[Any, Any, Any, str]:
    await register_pending_root(store, task_id, capability)
    claim_uc = ClaimTask(task_store=store, assignment_proof_pepper=PROOF_PEPPER, delivery_lease_seconds=30.0)
    ack_uc = AcknowledgeAssignment(task_store=store, assignment_proof_pepper=PROOF_PEPPER)
    principal = RuntimePrincipal(
        principal_id=f"principal-{capability}-1" if capability != "general" else "principal-worker-1",
        bound_agent_id=f"{capability}-agent-1" if capability != "general" else "worker-1",
        authorized_capabilities=(capability,),
        active=True,
    )
    claim_res = await claim_uc.execute(
        principal=principal, task_id=task_id, request_id=f"req-{task_id}", capability=capability
    )
    await ack_uc.execute(
        principal=principal,
        task_id=task_id,
        assignment_id=claim_res.assignment.assignment_id,
        assignment_proof=claim_res.assignment_proof or "",
    )
    claim = await store.get_active_claim(task_id)
    assert claim is not None
    return claim_uc, claim_res, claim, token


@pytest.mark.anyio
async def test_zero_mutation_on_missing_claim(test_settings: Settings) -> None:

    store = InMemoryAuthoritativeTaskStore()
    _uc, claim_res, claim, _tok = await setup_claimed_root(store, "task-zero-1", "general", WORKER_TOKEN)
    del store.claims["task-zero-1"]
    before = snapshot_store(store)
    command = CompletionCommand(
        completion_command_id="cmd-zero-1",
        assignment_id=claim_res.assignment.assignment_id,
        execution_kind="initial",
        failure_code="EXECUTION_FAILED",
    )
    with pytest.raises(ClaimNotFoundError):
        await store.complete_turn(
            task_id="task-zero-1",
            command=command,
            proof_digest=claim.proof_digest,
            fencing_epoch=claim.fencing_epoch,
            parent_task_id=None,
            root_task_id="task-zero-1",
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
        )
    assert snapshot_store(store) == before


@pytest.mark.anyio
async def test_zero_mutation_on_unacknowledged_claim(test_settings: Settings) -> None:

    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-zero-2", "general")
    claim_uc = ClaimTask(task_store=store, assignment_proof_pepper=PROOF_PEPPER, delivery_lease_seconds=30.0)
    principal = RuntimePrincipal(
        principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        authorized_capabilities=("general",),
        active=True,
    )
    claim_res = await claim_uc.execute(
        principal=principal, task_id="task-zero-2", request_id="req-zero-2", capability="general"
    )
    claim = await store.get_active_claim("task-zero-2")
    assert claim is not None
    before = snapshot_store(store)
    command = CompletionCommand(
        completion_command_id="cmd-zero-2",
        assignment_id=claim_res.assignment.assignment_id,
        execution_kind="initial",
        failure_code="EXECUTION_FAILED",
    )
    with pytest.raises(ClaimNotFoundError):
        await store.complete_turn(
            task_id="task-zero-2",
            command=command,
            proof_digest=claim.proof_digest,
            fencing_epoch=claim.fencing_epoch,
            parent_task_id=None,
            root_task_id="task-zero-2",
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
        )
    assert snapshot_store(store) == before


@pytest.mark.anyio
async def test_zero_mutation_on_existing_conflicting_records(test_settings: Settings) -> None:

    for seed in ("outcome", "result", "audit"):
        store = InMemoryAuthoritativeTaskStore()
        _uc, claim_res, claim, _tok = await setup_claimed_root(store, "task-zero-3", "general", WORKER_TOKEN)
        seed_command = CompletionCommand(
            completion_command_id="cmd-seed",
            assignment_id=claim_res.assignment.assignment_id,
            execution_kind="initial",
            failure_code="EXECUTION_FAILED",
        )
        seed_outcome, _ = build_terminal_outcome(
            command=seed_command,
            task_id="task-zero-3",
            runtime_principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
            completed_at=datetime.now(UTC).isoformat(),
        )
        if seed == "outcome":
            store.terminal_outcomes["task-zero-3"] = seed_outcome
        elif seed == "result":
            store.canonical_results["task-zero-3"] = TaskResult(task_id="task-zero-3", output="legacy")
        else:
            store.assignment_completions[("task-zero-3", claim_res.assignment.assignment_id)] = (
                build_completion_audit(outcome=seed_outcome, fencing_epoch=claim.fencing_epoch)
            )
        before = snapshot_store(store)
        command = CompletionCommand(
            completion_command_id="cmd-zero-3",
            assignment_id=claim_res.assignment.assignment_id,
            execution_kind="initial",
            failure_code="EXECUTION_FAILED",
        )
        with pytest.raises(CompletionConflictError):
            await store.complete_turn(
                task_id="task-zero-3",
                command=command,
                proof_digest=claim.proof_digest,
                fencing_epoch=claim.fencing_epoch,
                parent_task_id=None,
                root_task_id="task-zero-3",
                completed_at=datetime.now(UTC).isoformat(),
                runtime_principal_id=claim.runtime_principal_id,
                bound_agent_id=claim.bound_agent_id,
            )
        assert snapshot_store(store) == before


async def setup_store_child_turn(
    store: InMemoryAuthoritativeTaskStore, parent_id: str
) -> dict[str, Any]:
    """Drive claim/ack, artifact staging, delegation, and child claim via use cases."""
    await register_pending_root(store, parent_id, "research", prompt="Research question")
    research = RuntimePrincipal(
        principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        authorized_capabilities=("research",),
        active=True,
    )
    analysis = RuntimePrincipal(
        principal_id="principal-analysis-1",
        bound_agent_id="analysis-agent-1",
        authorized_capabilities=("analysis",),
        active=True,
    )
    claim_uc = ClaimTask(task_store=store, assignment_proof_pepper=PROOF_PEPPER, delivery_lease_seconds=30.0)
    ack_uc = AcknowledgeAssignment(task_store=store, assignment_proof_pepper=PROOF_PEPPER)
    parent_res = await claim_uc.execute(
        principal=research, task_id=parent_id, request_id=f"req-{parent_id}", capability="research"
    )
    await ack_uc.execute(
        principal=research,
        task_id=parent_id,
        assignment_id=parent_res.assignment.assignment_id,
        assignment_proof=parent_res.assignment_proof or "",
    )
    now_iso = datetime.now(UTC).isoformat()
    artifact = StageArtifact(
        artifact_id=f"art-{parent_id}",
        producing_task_id=parent_id,
        artifact_type="research_material",
        content={"material": "key findings"},
        created_at=now_iso,
        producing_turn_id=parent_res.assignment.turn_id,
        artifact_version=1,
        payload_schema_version=1,
        sealed=True,
    )
    await store.save_stage_artifact(
        artifact,
        parent_res.assignment.assignment_id,
        "principal-research-1",
        parent_res.assignment.fencing_epoch,
    )
    accept_uc = AcceptDelegationProposal(
        task_store=store,
        assignment_proof_pepper=PROOF_PEPPER,
        child_claim_deadline_seconds=120.0,
    )
    delegation = await accept_uc.execute(
        principal=research,
        parent_task_id=parent_id,
        proposal_id=f"prop-{parent_id}",
        assignment_id=parent_res.assignment.assignment_id,
        assignment_proof=parent_res.assignment_proof or "",
        child_intent="Analyze the staged findings",
        artifact_reference=ArtifactReference(
            producing_task_id=parent_id,
            artifact_id=f"art-{parent_id}",
            artifact_version=1,
        ),
        target_capability="analysis",
        parent_input_version=1,
        control_fields={},
    )
    child_id = delegation.child_task_id
    child_res = await claim_uc.execute(
        principal=analysis,
        task_id=child_id,
        request_id=f"req-{child_id}",
        capability="analysis",
        execution_kind="delegated_child",
    )
    await ack_uc.execute(
        principal=analysis,
        task_id=child_id,
        assignment_id=child_res.assignment.assignment_id,
        assignment_proof=child_res.assignment_proof or "",
    )
    child_claim = await store.get_active_claim(child_id)
    assert child_claim is not None
    return {
        "parent_id": parent_id,
        "child_id": child_id,
        "child_assignment_id": child_res.assignment.assignment_id,
        "child_claim": child_claim,
    }


def child_success_command(child_id: str, assignment_id: str) -> CompletionCommand:
    return CompletionCommand(
        completion_command_id=f"cmd-{child_id}",
        assignment_id=assignment_id,
        execution_kind="delegated_child",
        completed_result=TaskResult(task_id=child_id, output="analysis report"),
    )


@pytest.mark.anyio
async def test_zero_mutation_on_missing_parent_continuation(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await setup_store_child_turn(store, "task-pzero-1")
    del store.continuations[(ctx["parent_id"], 1)]
    del store.latest_continuation[ctx["parent_id"]]
    before = snapshot_store(store)
    with pytest.raises(TaskNotClaimableError, match="MISSING_CONTINUATION"):
        await store.complete_turn(
            task_id=ctx["child_id"],
            command=child_success_command(ctx["child_id"], ctx["child_assignment_id"]),
            proof_digest=ctx["child_claim"].proof_digest,
            fencing_epoch=ctx["child_claim"].fencing_epoch,
            parent_task_id=ctx["parent_id"],
            root_task_id=ctx["parent_id"],
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=ctx["child_claim"].runtime_principal_id,
            bound_agent_id=ctx["child_claim"].bound_agent_id,
        )
    assert snapshot_store(store) == before


@pytest.mark.anyio
async def test_zero_mutation_on_wrong_required_child(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await setup_store_child_turn(store, "task-pzero-2")
    current = store.continuations[(ctx["parent_id"], 1)]
    store.continuations[(ctx["parent_id"], 1)] = TaskContinuation(
        parent_task_id=current.parent_task_id,
        child_task_id="child-someone-else",
        input_version=current.input_version,
        continuation_capability=current.continuation_capability,
        stage=current.stage,
        control_fields=dict(current.control_fields),
        artifact_references=tuple(current.artifact_references),
        continuation_version=current.continuation_version,
        created_at=current.created_at,
        updated_at=current.updated_at,
        version=1,
    )
    before = snapshot_store(store)
    with pytest.raises(TaskNotClaimableError, match="CONTINUATION_CHILD_MISMATCH"):
        await store.complete_turn(
            task_id=ctx["child_id"],
            command=child_success_command(ctx["child_id"], ctx["child_assignment_id"]),
            proof_digest=ctx["child_claim"].proof_digest,
            fencing_epoch=ctx["child_claim"].fencing_epoch,
            parent_task_id=ctx["parent_id"],
            root_task_id=ctx["parent_id"],
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=ctx["child_claim"].runtime_principal_id,
            bound_agent_id=ctx["child_claim"].bound_agent_id,
        )
    assert snapshot_store(store) == before


@pytest.mark.anyio
async def test_zero_mutation_on_parent_wrong_state(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await setup_store_child_turn(store, "task-pzero-3")
    store.statuses[ctx["parent_id"]] = "working"
    before = snapshot_store(store)
    with pytest.raises(TaskNotClaimableError, match="PARENT_NOT_WAITING"):
        await store.complete_turn(
            task_id=ctx["child_id"],
            command=child_success_command(ctx["child_id"], ctx["child_assignment_id"]),
            proof_digest=ctx["child_claim"].proof_digest,
            fencing_epoch=ctx["child_claim"].fencing_epoch,
            parent_task_id=ctx["parent_id"],
            root_task_id=ctx["parent_id"],
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=ctx["child_claim"].runtime_principal_id,
            bound_agent_id=ctx["child_claim"].bound_agent_id,
        )
    assert snapshot_store(store) == before


@pytest.mark.anyio
async def test_zero_mutation_on_continuation_version_mismatch(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await setup_store_child_turn(store, "task-pzero-4")
    store.latest_continuation[ctx["parent_id"]] = 2
    before = snapshot_store(store)
    with pytest.raises(TaskNotClaimableError, match="CONTINUATION_LATEST_MISMATCH"):
        await store.complete_turn(
            task_id=ctx["child_id"],
            command=child_success_command(ctx["child_id"], ctx["child_assignment_id"]),
            proof_digest=ctx["child_claim"].proof_digest,
            fencing_epoch=ctx["child_claim"].fencing_epoch,
            parent_task_id=ctx["parent_id"],
            root_task_id=ctx["parent_id"],
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=ctx["child_claim"].runtime_principal_id,
            bound_agent_id=ctx["child_claim"].bound_agent_id,
        )
    assert snapshot_store(store) == before


@pytest.mark.anyio
async def test_zero_mutation_on_projection_marker_conflict(test_settings: Settings) -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await setup_store_child_turn(store, "task-pzero-5")
    store.child_markers[ctx["child_id"]] = {"child_task_id": ctx["child_id"]}
    before = snapshot_store(store)
    with pytest.raises(CompletionConflictError, match="MARKER_CONFLICT"):
        await store.complete_turn(
            task_id=ctx["child_id"],
            command=child_success_command(ctx["child_id"], ctx["child_assignment_id"]),
            proof_digest=ctx["child_claim"].proof_digest,
            fencing_epoch=ctx["child_claim"].fencing_epoch,
            parent_task_id=ctx["parent_id"],
            root_task_id=ctx["parent_id"],
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=ctx["child_claim"].runtime_principal_id,
            bound_agent_id=ctx["child_claim"].bound_agent_id,
        )
    assert snapshot_store(store) == before


# --- Completed-result task binding (Finding 1) ------------------------------------------


@pytest.mark.anyio
async def test_completed_result_task_id_mismatch_rejected_by_api(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-bind-1", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-bind-1", "general")
        before = snapshot_store(store)
        command = success_command(assignment["assignment_id"])
        command["result"]["task_id"] = "task-someone-else"
        res = await post_completion(client, WORKER_TOKEN, "task-bind-1", command, proof)
        assert res.status_code == 400
        assert "does not match" in res.json()["detail"]
        assert "outcome" not in res.json()
        assert "result" not in res.json()
        assert snapshot_store(store) == before
        assert await store.get_terminal_outcome("task-bind-1") is None
        assert await store.get_active_claim("task-bind-1") is not None


@pytest.mark.anyio
async def test_completed_result_task_id_mismatch_rejected_by_store(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    _uc, claim_res, claim, _tok = await setup_claimed_root(store, "task-bind-2", "general", WORKER_TOKEN)
    before = snapshot_store(store)
    command = CompletionCommand(
        completion_command_id="cmd-bind-2",
        assignment_id=claim_res.assignment.assignment_id,
        execution_kind="initial",
        completed_result=TaskResult(task_id="task-elsewhere", output="misbound"),
    )
    with pytest.raises(ValueError, match="does not match"):
        await store.complete_turn(
            task_id="task-bind-2",
            command=command,
            proof_digest=claim.proof_digest,
            fencing_epoch=claim.fencing_epoch,
            parent_task_id=None,
            root_task_id="task-bind-2",
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
        )
    assert snapshot_store(store) == before
    assert await store.get_terminal_outcome("task-bind-2") is None


@pytest.mark.anyio
async def test_store_replay_with_tampered_stored_result_fails_closed(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    _uc, claim_res, claim, _tok = await setup_claimed_root(store, "task-bind-3", "general", WORKER_TOKEN)
    command = CompletionCommand(
        completion_command_id="cmd-bind-3",
        assignment_id=claim_res.assignment.assignment_id,
        execution_kind="initial",
        completed_result=TaskResult(task_id="task-bind-3", output="canonical"),
    )
    outcome, _result, replayed = await store.complete_turn(
        task_id="task-bind-3",
        command=command,
        proof_digest=claim.proof_digest,
        fencing_epoch=claim.fencing_epoch,
        parent_task_id=None,
        root_task_id="task-bind-3",
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id=claim.runtime_principal_id,
        bound_agent_id=claim.bound_agent_id,
    )
    assert replayed is False
    counts_before = (
        len(store.terminal_outcomes),
        len(store.canonical_results),
        len(store.assignment_completions),
        len(store.completion_records),
        len(store.hints),
    )

    async def replay() -> None:
        await store.complete_turn(
            task_id="task-bind-3",
            command=command,
            proof_digest=claim.proof_digest,
            fencing_epoch=claim.fencing_epoch,
            parent_task_id=None,
            root_task_id="task-bind-3",
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
        )

    # Tampered stored result bound to another task fails closed on replay.
    store.canonical_results["task-bind-3"] = TaskResult(task_id="task-other", output="canonical")
    with pytest.raises(CorruptAuthoritativeRecordError, match="inconsistent"):
        await replay()
    assert await store.get_terminal_outcome("task-bind-3") == outcome

    # Tampered stored result with non-completed status fails closed on replay.
    store.canonical_results["task-bind-3"] = TaskResult(
        task_id="task-bind-3", output="canonical", status="failed"
    )
    with pytest.raises(CorruptAuthoritativeRecordError, match="inconsistent"):
        await replay()

    # Tampered stored outcome bound to another task fails closed on replay.
    tampered_outcome, _ = build_terminal_outcome(
        command=command,
        task_id="task-other",
        runtime_principal_id=claim.runtime_principal_id,
        bound_agent_id=claim.bound_agent_id,
        completed_at=datetime.now(UTC).isoformat(),
    )
    assert tampered_outcome.outcome_id == outcome.outcome_id
    store.canonical_results["task-bind-3"] = TaskResult(task_id="task-bind-3", output="canonical")
    store.terminal_outcomes["task-bind-3"] = tampered_outcome
    with pytest.raises(CorruptAuthoritativeRecordError, match="task mismatch"):
        await replay()

    assert (
        len(store.terminal_outcomes),
        len(store.canonical_results),
        len(store.assignment_completions),
        len(store.completion_records),
        len(store.hints),
    ) == counts_before


@pytest.mark.anyio
async def test_store_replay_of_failed_outcome_with_unexpected_result_fails_closed(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    _uc, claim_res, claim, _tok = await setup_claimed_root(store, "task-bind-5", "general", WORKER_TOKEN)
    command = CompletionCommand(
        completion_command_id="cmd-bind-5",
        assignment_id=claim_res.assignment.assignment_id,
        execution_kind="initial",
        failure_code="EXECUTION_FAILED",
    )

    async def do_complete() -> None:
        await store.complete_turn(
            task_id="task-bind-5",
            command=command,
            proof_digest=claim.proof_digest,
            fencing_epoch=claim.fencing_epoch,
            parent_task_id=None,
            root_task_id="task-bind-5",
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
        )

    outcome, _result, replayed = await store.complete_turn(
        task_id="task-bind-5",
        command=command,
        proof_digest=claim.proof_digest,
        fencing_epoch=claim.fencing_epoch,
        parent_task_id=None,
        root_task_id="task-bind-5",
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id=claim.runtime_principal_id,
        bound_agent_id=claim.bound_agent_id,
    )
    assert replayed is False
    assert outcome.terminal_status == "failed"

    # A canonical result must never coexist with a failed outcome: fail closed.
    store.canonical_results["task-bind-5"] = TaskResult(task_id="task-bind-5", output="rogue")
    before = snapshot_store(store)
    with pytest.raises(CorruptAuthoritativeRecordError, match="Unexpected canonical result"):
        await do_complete()
    assert snapshot_store(store) == before
    assert await store.get_terminal_outcome("task-bind-5") == outcome
    assert await store.get_status("task-bind-5") == "failed"

    # Removing the rogue record restores identical replay with zero writes.
    del store.canonical_results["task-bind-5"]
    _o2, r2, replayed2 = await store.complete_turn(
        task_id="task-bind-5",
        command=command,
        proof_digest=claim.proof_digest,
        fencing_epoch=claim.fencing_epoch,
        parent_task_id=None,
        root_task_id="task-bind-5",
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id=claim.runtime_principal_id,
        bound_agent_id=claim.bound_agent_id,
    )
    assert replayed2 is True
    assert r2 is None


@pytest.mark.anyio
async def test_store_replay_of_completed_outcome_with_missing_result_fails_closed(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    _uc, claim_res, claim, _tok = await setup_claimed_root(store, "task-bind-6", "general", WORKER_TOKEN)
    command = CompletionCommand(
        completion_command_id="cmd-bind-6",
        assignment_id=claim_res.assignment.assignment_id,
        execution_kind="initial",
        completed_result=TaskResult(task_id="task-bind-6", output="canonical"),
    )
    await store.complete_turn(
        task_id="task-bind-6",
        command=command,
        proof_digest=claim.proof_digest,
        fencing_epoch=claim.fencing_epoch,
        parent_task_id=None,
        root_task_id="task-bind-6",
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id=claim.runtime_principal_id,
        bound_agent_id=claim.bound_agent_id,
    )
    del store.canonical_results["task-bind-6"]
    before = snapshot_store(store)
    with pytest.raises(CorruptAuthoritativeRecordError, match="refusing to fabricate"):
        await store.complete_turn(
            task_id="task-bind-6",
            command=command,
            proof_digest=claim.proof_digest,
            fencing_epoch=claim.fencing_epoch,
            parent_task_id=None,
            root_task_id="task-bind-6",
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=claim.runtime_principal_id,
            bound_agent_id=claim.bound_agent_id,
        )
    assert snapshot_store(store) == before


@pytest.mark.anyio
async def test_retired_replay_with_mismatched_result_task_id_does_not_leak(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-bind-4", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-bind-4", "general")
        command = success_command(assignment["assignment_id"], output="original")
        command["result"]["task_id"] = "task-bind-4"
        first = await post_completion(client, WORKER_TOKEN, "task-bind-4", command, proof)
        assert first.status_code == 200

        replay = success_command(assignment["assignment_id"], output="original")
        replay["result"]["task_id"] = "task-not-this-task"
        res = await post_completion(client, WORKER_TOKEN, "task-bind-4", replay, proof)
        assert res.status_code == 400
        assert res.json() == {
            "detail": (
                "Completed TaskResult task_id 'task-not-this-task' "
                "does not match route task 'task-bind-4'"
            )
        }
        stored = await store.get_canonical_result("task-bind-4")
        assert stored is not None and stored.output == "original"


@pytest.mark.anyio
async def test_retired_replay_of_failed_outcome_with_unexpected_result_is_generic_500(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-bind-7", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-bind-7", "general")
        failed = {
            "completion_command_id": "cmd-bind-7",
            "assignment_id": assignment["assignment_id"],
            "execution_kind": "initial",
            "failure_code": "EXECUTION_FAILED",
        }
        first = await post_completion(client, WORKER_TOKEN, "task-bind-7", failed, proof)
        assert first.status_code == 200
        assert "result" not in first.json()

        # Rogue canonical result coexisting with a failed outcome fails closed
        # as corruption: generic 500 with no outcome, result, or record content.
        store.canonical_results["task-bind-7"] = TaskResult(task_id="task-bind-7", output="rogue")
        before = snapshot_store(store)
        res = await post_completion(client, WORKER_TOKEN, "task-bind-7", failed, proof)
        assert res.status_code == 500
        assert res.json() == {"detail": "Authoritative record corruption detected"}
        assert snapshot_store(store) == before
        assert await store.get_terminal_outcome("task-bind-7") is not None
        assert await store.get_status("task-bind-7") == "failed"
        assert len(store.hints) == 0


@pytest.mark.anyio
async def test_delegated_child_result_task_id_mismatch_rejected_with_zero_projection(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    ctx = await setup_store_child_turn(store, "task-pbind-1")
    before = snapshot_store(store)
    command = CompletionCommand(
        completion_command_id=f"cmd-{ctx['child_id']}",
        assignment_id=ctx["child_assignment_id"],
        execution_kind="delegated_child",
        completed_result=TaskResult(task_id=ctx["parent_id"], output="misbound report"),
    )
    with pytest.raises(ValueError, match="does not match"):
        await store.complete_turn(
            task_id=ctx["child_id"],
            command=command,
            proof_digest=ctx["child_claim"].proof_digest,
            fencing_epoch=ctx["child_claim"].fencing_epoch,
            parent_task_id=ctx["parent_id"],
            root_task_id=ctx["parent_id"],
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id=ctx["child_claim"].runtime_principal_id,
            bound_agent_id=ctx["child_claim"].bound_agent_id,
        )
    assert snapshot_store(store) == before
    assert await store.get_status(ctx["child_id"]) == "working"
    assert await store.get_status(ctx["parent_id"]) == "waiting_for_children"
    assert ctx["child_id"] not in store.child_markers
    assert await store.get_active_claim(ctx["child_id"]) is not None


# --- Nested result field rejection (Finding 4) --------------------------------------------


@pytest.mark.anyio
async def test_completion_rejects_nested_raw_field_in_result(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-nested-1", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-nested-1", "general")
        before = snapshot_store(store)
        command = success_command(assignment["assignment_id"])
        command["result"]["task_id"] = "task-nested-1"
        command["result"]["exception"] = "Traceback (most recent call last): ..."
        res = await post_completion(client, WORKER_TOKEN, "task-nested-1", command, proof)
        assert res.status_code == 400
        assert snapshot_store(store) == before
        assert await store.get_terminal_outcome("task-nested-1") is None


@pytest.mark.anyio
async def test_completion_rejects_nested_secret_field_in_sources(
    test_settings: Settings,
) -> None:
    store = InMemoryAuthoritativeTaskStore()
    await register_pending_root(store, "task-nested-2", "general")
    async with make_client(store, test_settings) as client:
        assignment, proof = await claim_and_acknowledge(client, WORKER_TOKEN, "task-nested-2", "general")
        before = snapshot_store(store)
        command = success_command(assignment["assignment_id"])
        command["result"]["task_id"] = "task-nested-2"
        command["result"]["sources"] = [
            {"url": "https://example.test", "bearer_token": "secret-value"}
        ]
        res = await post_completion(client, WORKER_TOKEN, "task-nested-2", command, proof)
        assert res.status_code == 400
        assert snapshot_store(store) == before
        assert await store.get_terminal_outcome("task-nested-2") is None
