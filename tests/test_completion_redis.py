"""Real-Redis atomicity tests for fenced turn completion (Issue #222).

Requires REDIS_URL pointing at database 15 (e.g. redis://localhost:6379/15).
Without REDIS_URL these tests skip; they never touch any other database.

Exercises COMPLETE_TURN_LUA through RedisGateway.complete_turn: first
completed/failed commits, exact replay with zero writes, conflicting replay,
stale fence, concurrent identical/conflicting completions, delegated-child
projection (completed and failed) with exactly one parent hint, and zero
mutation across wrong key types, corrupt JSON, and every major rejection
branch.
"""

import asyncio
import json
import os
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from urllib.parse import urlparse

import pytest
import redis.asyncio as redis
from agent_contracts import (
    ArtifactReference,
    CompletionCommand,
    TaskRequested,
    TaskResult,
)
from orchestrator.application.ports import (
    ClaimNotFoundError,
    CompletionConflictError,
    TaskAlreadyClaimedError,
)
from orchestrator.application.use_cases import (
    AcceptDelegationProposal,
    AcknowledgeAssignment,
    ClaimTask,
)
from orchestrator.domain.principal import RuntimePrincipal
from orchestrator.domain.task import CorruptAuthoritativeRecordError
from orchestrator.infrastructure.redis_adapter import (
    COMPLETE_TURN_LUA,
    RedisGateway,
)

pytestmark = [pytest.mark.anyio, pytest.mark.redis]

REQUIRED_TEST_DB = "15"
PROOF_PEPPER = "test_completion_redis_proof_pepper_4567890123456789"


def _validate_test_redis_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("redis", "rediss"):
        raise ValueError(f"Redis integration tests require a redis:// URL, got scheme '{parsed.scheme}'")
    if parsed.path.strip("/") != REQUIRED_TEST_DB:
        raise ValueError(
            "Redis integration tests require database 15 to prevent accidental data loss, "
            f"got '{parsed.path}'"
        )


@pytest.fixture
def redis_url() -> str:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        pytest.skip("Redis integration tests disabled: REDIS_URL environment variable is not set")
    try:
        _validate_test_redis_url(url)
    except ValueError as exc:
        pytest.fail(str(exc))
    return url


@pytest.fixture
async def clean_gateway(redis_url: str) -> AsyncGenerator[RedisGateway, None]:
    gateway = RedisGateway(redis_url, allow_unauthenticated_local_test=True)
    try:
        await asyncio.wait_for(gateway._client.ping(), timeout=1.0)
    except (redis.ConnectionError, OSError) as exc:
        await gateway.aclose()
        pytest.fail(f"Configured Redis endpoint is unreachable within 1.0s timeout: {exc}")
    await gateway._client.flushdb()
    try:
        yield gateway
    finally:
        try:
            await gateway._client.flushdb()
        finally:
            await gateway.aclose()


def _research_principal() -> RuntimePrincipal:
    return RuntimePrincipal(
        principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        authorized_capabilities=("research",),
        active=True,
    )


def _worker_principal() -> RuntimePrincipal:
    return RuntimePrincipal(
        principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        authorized_capabilities=("general",),
        active=True,
    )


def _analysis_principal() -> RuntimePrincipal:
    return RuntimePrincipal(
        principal_id="principal-analysis-1",
        bound_agent_id="analysis-agent-1",
        authorized_capabilities=("analysis",),
        active=True,
    )


async def _register_pending(gateway: RedisGateway, task_id: str, capability: str) -> None:
    task = TaskRequested(
        task_id=task_id,
        root_task_id=task_id,
        parent_task_id=None,
        prompt=f"Redis completion task {task_id}",
        created_at=datetime.now(UTC).isoformat(),
        version=2,
    )
    await gateway.register_root_task(task, required_capability=capability)
    await gateway.update_status(task_id, "pending")


def _claim_use_cases(gateway: RedisGateway) -> tuple[ClaimTask, AcknowledgeAssignment]:
    return (
        ClaimTask(task_store=gateway, assignment_proof_pepper=PROOF_PEPPER, delivery_lease_seconds=30.0),
        AcknowledgeAssignment(task_store=gateway, assignment_proof_pepper=PROOF_PEPPER),
    )


async def _claimed_turn(
    gateway: RedisGateway,
    task_id: str,
    capability: str,
    principal: RuntimePrincipal,
    execution_kind: str = "initial",
    expected_continuation_version: int | None = None,
) -> tuple[str, str, str]:
    """Claim and acknowledge a task; returns (assignment_id, proof, claim_proof_digest)."""
    claim_uc, ack_uc = _claim_use_cases(gateway)
    claim_res = await claim_uc.execute(
        principal=principal,
        task_id=task_id,
        request_id=f"req-{task_id}-{uuid.uuid4().hex[:8]}",
        capability=capability,
        execution_kind=execution_kind,
        expected_continuation_version=expected_continuation_version,
    )
    await ack_uc.execute(
        principal=principal,
        task_id=task_id,
        assignment_id=claim_res.assignment.assignment_id,
        assignment_proof=claim_res.assignment_proof or "",
    )
    claim = await gateway.get_active_claim(task_id)
    assert claim is not None
    return claim_res.assignment.assignment_id, claim_res.assignment_proof or "", claim.proof_digest


async def _delegated_child_turn(gateway: RedisGateway, parent_id: str) -> dict[str, str]:
    """Full research -> analysis delegation on real Redis; returns child turn material."""
    from orchestrator.application.use_cases import StageArtifactUseCase

    research = _research_principal()
    analysis = _analysis_principal()
    await _register_pending(gateway, parent_id, "research")
    assignment_id, proof, _digest = await _claimed_turn(gateway, parent_id, "research", research)

    stage_uc = StageArtifactUseCase(task_store=gateway, assignment_proof_pepper=PROOF_PEPPER)
    artifact = await stage_uc.execute(
        principal=research,
        task_id=parent_id,
        assignment_id=assignment_id,
        assignment_proof=proof,
        artifact_id=f"art-{parent_id}",
        artifact_type="research_material",
        content={"material": "key findings"},
        artifact_version=1,
        payload_schema_version=1,
    )
    accept_uc = AcceptDelegationProposal(
        task_store=gateway,
        assignment_proof_pepper=PROOF_PEPPER,
        child_claim_deadline_seconds=120.0,
    )
    delegation = await accept_uc.execute(
        principal=research,
        parent_task_id=parent_id,
        proposal_id=f"prop-{parent_id}",
        assignment_id=assignment_id,
        assignment_proof=proof,
        child_intent="Analyze the staged findings",
        artifact_reference=ArtifactReference(
            producing_task_id=parent_id,
            artifact_id=artifact.artifact_id,
            artifact_version=1,
        ),
        target_capability="analysis",
        parent_input_version=1,
        control_fields={},
    )
    child_id = delegation.child_task_id
    child_assignment_id, child_proof, child_digest = await _claimed_turn(
        gateway, child_id, "analysis", analysis, execution_kind="delegated_child"
    )
    return {
        "parent_id": parent_id,
        "child_id": child_id,
        "child_assignment_id": child_assignment_id,
        "child_proof": child_proof,
        "child_digest": child_digest,
    }


def _completed_command(task_id: str, assignment_id: str, kind: str, output: str = "done") -> CompletionCommand:
    return CompletionCommand(
        completion_command_id=f"cmd-{task_id}",
        assignment_id=assignment_id,
        execution_kind=kind,
        completed_result=TaskResult(task_id=task_id, output=output),
    )


# --- First commits ----------------------------------------------------------------------------


async def test_redis_first_completed_commits_canonical_records(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-ok-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())

    outcome, result, replayed = await clean_gateway.complete_turn(
        task_id=task_id,
        command=_completed_command(task_id, assignment_id, "initial", output="redis output"),
        proof_digest=digest,
        fencing_epoch=1,
        parent_task_id=None,
        root_task_id=task_id,
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id="principal-worker-1",
        bound_agent_id="worker-1",
    )
    assert replayed is False
    assert outcome.terminal_status == "completed"
    assert result is not None and result.output == "redis output"

    client = clean_gateway._client
    assert await client.get(f"task.outcome:{task_id}") is not None
    assert await client.get(f"task.result:{task_id}") is not None
    assert await client.get(f"task.completion:{task_id}:cmd-{task_id}") is not None
    assert await client.get(f"task.assignment_completion:{task_id}:{assignment_id}") is not None
    raw_status = await client.get(f"task.status:{task_id}")
    assert raw_status is not None and '"completed"' in raw_status
    assert await client.get(f"task.claim:{task_id}") is None


async def test_redis_first_failed_writes_no_result(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-fail-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())

    command = CompletionCommand(
        completion_command_id=f"cmd-{task_id}",
        assignment_id=assignment_id,
        execution_kind="initial",
        failure_code="EXECUTION_FAILED",
    )
    outcome, result, replayed = await clean_gateway.complete_turn(
        task_id=task_id,
        command=command,
        proof_digest=digest,
        fencing_epoch=1,
        parent_task_id=None,
        root_task_id=task_id,
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id="principal-worker-1",
        bound_agent_id="worker-1",
    )
    assert replayed is False
    assert outcome.terminal_status == "failed"
    assert result is None
    assert await clean_gateway._client.get(f"task.result:{task_id}") is None


# --- Replay and conflict ----------------------------------------------------------------------


async def test_redis_exact_replay_returns_identical_records(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-replay-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())
    command = _completed_command(task_id, assignment_id, "initial")
    kwargs = {
        "task_id": task_id,
        "command": command,
        "proof_digest": digest,
        "fencing_epoch": 1,
        "parent_task_id": None,
        "root_task_id": task_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_principal_id": "principal-worker-1",
        "bound_agent_id": "worker-1",
    }
    outcome, result, replayed = await clean_gateway.complete_turn(**kwargs)
    assert replayed is False
    before = {
        key: await clean_gateway._client.get(key)
        for key in (
            f"task.outcome:{task_id}",
            f"task.result:{task_id}",
            f"task.completion:{task_id}:cmd-{task_id}",
            f"task.assignment_completion:{task_id}:{assignment_id}",
            f"task.status:{task_id}",
        )
    }
    outcome2, result2, replayed2 = await clean_gateway.complete_turn(**kwargs)
    assert replayed2 is True
    assert outcome2 == outcome
    assert result2 == result
    for key, value in before.items():
        assert await clean_gateway._client.get(key) == value


async def test_redis_conflicting_replay_fails_closed(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-conflict-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())
    await clean_gateway.complete_turn(
        task_id=task_id,
        command=_completed_command(task_id, assignment_id, "initial", output="original"),
        proof_digest=digest,
        fencing_epoch=1,
        parent_task_id=None,
        root_task_id=task_id,
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id="principal-worker-1",
        bound_agent_id="worker-1",
    )
    tampered = CompletionCommand(
        completion_command_id=f"cmd-{task_id}",
        assignment_id=assignment_id,
        execution_kind="initial",
        completed_result=TaskResult(task_id=task_id, output="tampered"),
    )
    with pytest.raises(CompletionConflictError):
        await clean_gateway.complete_turn(
            task_id=task_id,
            command=tampered,
            proof_digest=digest,
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id=task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
        )
    stored = await clean_gateway.get_canonical_result(task_id)
    assert stored is not None and stored.output == "original"


async def test_redis_stale_fence_is_rejected(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-fence-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())
    await clean_gateway._client.set(f"task.fence:{task_id}", "99")
    with pytest.raises(TaskAlreadyClaimedError):
        await clean_gateway.complete_turn(
            task_id=task_id,
            command=_completed_command(task_id, assignment_id, "initial"),
            proof_digest=digest,
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id=task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
        )
    assert await clean_gateway.get_terminal_outcome(task_id) is None


async def test_redis_concurrent_identical_completions_converge(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-race-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())
    command = _completed_command(task_id, assignment_id, "initial", output="raced")

    async def attempt() -> tuple[bool, str]:
        outcome, _result, replayed = await clean_gateway.complete_turn(
            task_id=task_id,
            command=command,
            proof_digest=digest,
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id=task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
        )
        return replayed, outcome.outcome_id

    first, second = await asyncio.gather(attempt(), attempt())
    assert sorted([first[0], second[0]]) == [False, True]
    assert first[1] == second[1]


async def test_redis_concurrent_conflicting_completion_fails_closed(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-racec-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())

    async def attempt(output: str) -> str:
        _outcome, _result, replayed = await clean_gateway.complete_turn(
            task_id=task_id,
            command=_completed_command(task_id, assignment_id, "initial", output=output),
            proof_digest=digest,
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id=task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
        )
        return "replayed" if replayed else "accepted"

    results = await asyncio.gather(attempt("winner"), attempt("loser"), return_exceptions=True)
    kinds = sorted("conflict" if isinstance(r, CompletionConflictError) else r for r in results)
    assert kinds == ["accepted", "conflict"]
    stored = await clean_gateway.get_canonical_result(task_id)
    assert stored is not None and stored.output in ("winner", "loser")


# --- Child projection ---------------------------------------------------------------------------


async def test_redis_child_completed_projection_emits_single_hint(clean_gateway: RedisGateway) -> None:
    parent_id = f"task-redis-parent-{uuid.uuid4().hex[:8]}"
    ctx = await _delegated_child_turn(clean_gateway, parent_id)
    child_id = ctx["child_id"]
    command = _completed_command(child_id, ctx["child_assignment_id"], "delegated_child", output="report")

    outcome, result, replayed = await clean_gateway.complete_turn(
        task_id=child_id,
        command=command,
        proof_digest=ctx["child_digest"],
        fencing_epoch=1,
        parent_task_id=parent_id,
        root_task_id=parent_id,
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id="principal-analysis-1",
        bound_agent_id="analysis-agent-1",
    )
    assert replayed is False
    assert outcome.terminal_status == "completed"
    assert result is not None

    client = clean_gateway._client
    raw_parent = await client.get(f"task.status:{parent_id}")
    assert raw_parent is not None and '"resumable"' in raw_parent
    raw_child = await client.get(f"task.status:{child_id}")
    assert raw_child is not None and '"completed"' in raw_child
    assert await client.get(f"task.child_completion:{child_id}") is not None
    raw_continuation = await client.get(f"task.continuation:{parent_id}:1")
    assert raw_continuation is not None
    assert '"continuation_version":1' in raw_continuation
    assert f'"{child_id}"' in raw_continuation
    hints = await client.lrange("agent.hints.research", 0, -1)
    parent_hints = [h for h in hints if f'"{parent_id}"' in h]
    assert len(parent_hints) == 1
    assert '"expected_continuation_version":1' in parent_hints[0]


async def test_redis_child_failed_projection_writes_no_result(clean_gateway: RedisGateway) -> None:
    parent_id = f"task-redis-parentf-{uuid.uuid4().hex[:8]}"
    ctx = await _delegated_child_turn(clean_gateway, parent_id)
    child_id = ctx["child_id"]
    command = CompletionCommand(
        completion_command_id=f"cmd-{child_id}",
        assignment_id=ctx["child_assignment_id"],
        execution_kind="delegated_child",
        failure_code="PROVIDER_UNAVAILABLE",
    )
    outcome, result, replayed = await clean_gateway.complete_turn(
        task_id=child_id,
        command=command,
        proof_digest=ctx["child_digest"],
        fencing_epoch=1,
        parent_task_id=parent_id,
        root_task_id=parent_id,
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id="principal-analysis-1",
        bound_agent_id="analysis-agent-1",
    )
    assert replayed is False
    assert outcome.terminal_status == "failed"
    assert result is None
    assert await clean_gateway._client.get(f"task.result:{child_id}") is None
    raw_parent = await clean_gateway._client.get(f"task.status:{parent_id}")
    assert raw_parent is not None and '"resumable"' in raw_parent


# --- Zero mutation ----------------------------------------------------------------------------------


async def _canonical_keys_absent(clean_gateway: RedisGateway, task_id: str, assignment_id: str) -> bool:
    client = clean_gateway._client
    keys = [
        f"task.outcome:{task_id}",
        f"task.result:{task_id}",
        f"task.completion:{task_id}:cmd-{task_id}",
        f"task.assignment_completion:{task_id}:{assignment_id}",
        f"task.child_completion:{task_id}",
    ]
    existing = [k for k in keys if await client.exists(k) == 1]
    return not existing


async def test_redis_zero_mutation_on_wrong_status_key_type(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-wrongtype-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())
    hints_before = await clean_gateway._client.llen("agent.hints.research")

    client = clean_gateway._client
    await client.delete(f"task.status:{task_id}")
    await client.rpush(f"task.status:{task_id}", "not-a-status-object")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.complete_turn(
            task_id=task_id,
            command=_completed_command(task_id, assignment_id, "initial"),
            proof_digest=digest,
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id=task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
        )
    assert await _canonical_keys_absent(clean_gateway, task_id, assignment_id)
    assert await client.llen("agent.hints.research") == hints_before


async def test_redis_zero_mutation_on_corrupt_status_json(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-corrupt-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())
    hints_before = await clean_gateway._client.llen("agent.hints.research")

    await clean_gateway._client.set(f"task.status:{task_id}", "not-json{{{")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.complete_turn(
            task_id=task_id,
            command=_completed_command(task_id, assignment_id, "initial"),
            proof_digest=digest,
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id=task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
        )
    assert await _canonical_keys_absent(clean_gateway, task_id, assignment_id)
    assert await clean_gateway._client.llen("agent.hints.research") == hints_before


async def test_redis_zero_mutation_on_missing_claim(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-noclaim-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())
    hints_before = await clean_gateway._client.llen("agent.hints.research")
    await clean_gateway._client.delete(f"task.claim:{task_id}")

    with pytest.raises(ClaimNotFoundError):
        await clean_gateway.complete_turn(
            task_id=task_id,
            command=_completed_command(task_id, assignment_id, "initial", output="x"),
            proof_digest=digest,
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id=task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
        )
    assert await _canonical_keys_absent(clean_gateway, task_id, assignment_id)
    assert await clean_gateway._client.llen("agent.hints.research") == hints_before


async def test_redis_zero_mutation_on_existing_conflicting_outcome(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-outcome-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())

    conflicting = CompletionCommand(
        completion_command_id="cmd-seed",
        assignment_id=assignment_id,
        execution_kind="initial",
        failure_code="EXECUTION_FAILED",
    )
    from agent_contracts import build_terminal_outcome

    outcome, _ = build_terminal_outcome(
        command=conflicting,
        task_id=task_id,
        runtime_principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        completed_at=datetime.now(UTC).isoformat(),
    )
    await clean_gateway._client.set(f"task.outcome:{task_id}", outcome.to_json())
    hints_before = await clean_gateway._client.llen("agent.hints.research")

    with pytest.raises(CompletionConflictError):
        await clean_gateway.complete_turn(
            task_id=task_id,
            command=_completed_command(task_id, assignment_id, "initial"),
            proof_digest=digest,
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id=task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
        )
    assert await clean_gateway.get_canonical_result(task_id) is None
    assert await clean_gateway.get_completion_record(task_id, f"cmd-{task_id}") is None
    assert await clean_gateway._client.llen("agent.hints.research") == hints_before


async def test_redis_zero_mutation_on_hint_queue_wrong_type(clean_gateway: RedisGateway) -> None:
    parent_id = f"task-redis-hinttype-{uuid.uuid4().hex[:8]}"
    ctx = await _delegated_child_turn(clean_gateway, parent_id)
    child_id = ctx["child_id"]
    await clean_gateway._client.delete("agent.hints.research")
    await clean_gateway._client.set("agent.hints.research", "not-a-list")

    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.complete_turn(
            task_id=child_id,
            command=_completed_command(child_id, ctx["child_assignment_id"], "delegated_child"),
            proof_digest=ctx["child_digest"],
            fencing_epoch=1,
            parent_task_id=parent_id,
            root_task_id=parent_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-analysis-1",
            bound_agent_id="analysis-agent-1",
        )
    assert await _canonical_keys_absent(clean_gateway, child_id, ctx["child_assignment_id"])
    assert await clean_gateway.get_terminal_outcome(child_id) is None
    await clean_gateway._client.delete("agent.hints.research")
    assert await clean_gateway._client.llen("agent.hints.research") == 0


# --- Finding 1: completed-result task binding on real Redis ----------------------------


async def _eval_complete_turn_raw(
    gateway: RedisGateway,
    *,
    task_id: str,
    assignment_id: str,
    principal_id: str,
    bound_agent_id: str,
    execution_kind: str,
    epoch: str,
    command_id: str,
    fingerprint: str,
    proof_digest: str,
    terminal_status: str,
    failure_code: str,
    outcome_json: str,
    result_json: str,
    audit_json: str,
    parent_task_id: str,
    root_task_id: str,
    binding_json: str,
    status_json: str,
) -> list:
    """Invoke COMPLETE_TURN_LUA directly, mirroring RedisGateway.complete_turn key order."""
    keys = [
        f"task.completion:{task_id}:{command_id}",
        f"task.status:{task_id}",
        f"task.claim:{task_id}",
        f"task.fence:{task_id}",
        f"task.outcome:{task_id}",
        f"task.result:{task_id}",
        f"task.assignment_completion:{task_id}:{assignment_id}",
        f"task.lineage:{task_id}",
        f"task.status:{task_id}",
        f"task.children:{task_id}",
        f"task.continuation:{task_id}:1",
        f"task.continuation_latest:{task_id}",
        f"task.child_completion:{task_id}",
        "agent.hints.research",
    ]
    argv = [
        task_id,
        assignment_id,
        principal_id,
        bound_agent_id,
        execution_kind,
        "",
        epoch,
        command_id,
        fingerprint,
        proof_digest,
        terminal_status,
        failure_code,
        outcome_json,
        result_json,
        audit_json,
        parent_task_id,
        root_task_id,
        binding_json,
        "",
        "",
        status_json,
        "",
    ]
    raw = await gateway._client.eval(COMPLETE_TURN_LUA, len(keys), *keys, *argv)
    return [r if isinstance(r, str) else r.decode("utf-8") for r in raw]


async def test_redis_lua_rejects_mismatched_result_task_id(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-bind-{uuid.uuid4().hex[:8]}"
    result_json = TaskResult(task_id="task-someone-else", output="misbound").to_json()
    status_json = json.dumps({"status": "working", "task_id": task_id})

    res = await _eval_complete_turn_raw(
        clean_gateway,
        task_id=task_id,
        assignment_id="assign-bind-1",
        principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        execution_kind="initial",
        epoch="1",
        command_id="cmd-bind-1",
        fingerprint="f" * 64,
        proof_digest="d" * 64,
        terminal_status="completed",
        failure_code="",
        outcome_json="{}",
        result_json=result_json,
        audit_json="{}",
        parent_task_id="",
        root_task_id=task_id,
        binding_json="{}",
        status_json=status_json,
    )
    assert res[0] == "RESULT_TASK_MISMATCH"
    assert await clean_gateway._client.dbsize() == 0
    assert await clean_gateway.get_terminal_outcome(task_id) is None
    assert await clean_gateway.get_canonical_result(task_id) is None


async def test_redis_lua_rejects_non_completed_result_status(clean_gateway: RedisGateway) -> None:
    task_id = f"task-redis-bindstatus-{uuid.uuid4().hex[:8]}"
    result_json = TaskResult(task_id=task_id, output="x", status="failed").to_json()
    status_json = json.dumps({"status": "working", "task_id": task_id})

    res = await _eval_complete_turn_raw(
        clean_gateway,
        task_id=task_id,
        assignment_id="assign-bind-2",
        principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        execution_kind="initial",
        epoch="1",
        command_id="cmd-bind-2",
        fingerprint="f" * 64,
        proof_digest="d" * 64,
        terminal_status="completed",
        failure_code="",
        outcome_json="{}",
        result_json=result_json,
        audit_json="{}",
        parent_task_id="",
        root_task_id=task_id,
        binding_json="{}",
        status_json=status_json,
    )
    assert res[0] == "RESULT_STATUS_MISMATCH"
    assert await clean_gateway._client.dbsize() == 0


async def test_redis_mismatched_result_task_id_rejected_via_adapter(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-bindadapter-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    _aid, _proof, digest = await _claimed_turn(clean_gateway, task_id, "general", _worker_principal())
    command = CompletionCommand(
        completion_command_id=f"cmd-{task_id}",
        assignment_id=_aid,
        execution_kind="initial",
        completed_result=TaskResult(task_id="task-elsewhere", output="misbound"),
    )
    with pytest.raises(ValueError, match="does not match"):
        await clean_gateway.complete_turn(
            task_id=task_id,
            command=command,
            proof_digest=digest,
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id=task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
        )
    assert await _canonical_keys_absent(clean_gateway, task_id, _aid)
    assert await clean_gateway.get_status(task_id) == "working"
    assert await clean_gateway.get_active_claim(task_id) is not None
    assert await clean_gateway._client.llen("agent.hints.research") == 0


async def test_redis_replay_with_tampered_stored_records_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-tamper-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(
        clean_gateway, task_id, "general", _worker_principal()
    )
    command = _completed_command(task_id, assignment_id, "initial", output="canonical")
    kwargs = {
        "task_id": task_id,
        "command": command,
        "proof_digest": digest,
        "fencing_epoch": 1,
        "parent_task_id": None,
        "root_task_id": task_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_principal_id": "principal-worker-1",
        "bound_agent_id": "worker-1",
    }
    _outcome, _result, replayed = await clean_gateway.complete_turn(**kwargs)
    assert replayed is False

    async def replay() -> None:
        await clean_gateway.complete_turn(**kwargs)

    client = clean_gateway._client
    pristine_outcome = await client.get(f"task.outcome:{task_id}")
    pristine_result = await client.get(f"task.result:{task_id}")
    assert pristine_outcome is not None and pristine_result is not None

    # Tampered stored result bound to another task fails closed on replay.
    await client.set(
        f"task.result:{task_id}",
        TaskResult(task_id="task-other", output="canonical").to_json(),
    )
    with pytest.raises(CorruptAuthoritativeRecordError):
        await replay()

    # Tampered stored result with non-completed status fails closed on replay.
    await client.set(
        f"task.result:{task_id}",
        TaskResult(task_id=task_id, output="canonical", status="failed").to_json(),
    )
    with pytest.raises(CorruptAuthoritativeRecordError):
        await replay()

    # Tampered stored outcome bound to another task fails closed on replay.
    tampered = json.loads(pristine_outcome)
    tampered["task_id"] = "task-other"
    await client.set(f"task.outcome:{task_id}", json.dumps(tampered))
    await client.set(
        f"task.result:{task_id}",
        TaskResult(task_id=task_id, output="canonical").to_json(),
    )
    with pytest.raises(CorruptAuthoritativeRecordError):
        await replay()

    # Restoring canonical records unblocks identical replay (no stuck state).
    await client.set(f"task.outcome:{task_id}", pristine_outcome)
    await client.set(f"task.result:{task_id}", pristine_result)
    _o2, _r2, replayed2 = await clean_gateway.complete_turn(**kwargs)
    assert replayed2 is True


# --- Finding 2: fail-closed canonical reads on real Redis -------------------------------


async def _commit_completed(clean_gateway: RedisGateway, task_id: str) -> str:
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(
        clean_gateway, task_id, "general", _worker_principal()
    )
    await clean_gateway.complete_turn(
        task_id=task_id,
        command=_completed_command(task_id, assignment_id, "initial", output="canonical read"),
        proof_digest=digest,
        fencing_epoch=1,
        parent_task_id=None,
        root_task_id=task_id,
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id="principal-worker-1",
        bound_agent_id="worker-1",
    )
    return assignment_id


async def test_redis_get_result_completed_missing_result_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-noresult-{uuid.uuid4().hex[:8]}"
    await _commit_completed(clean_gateway, task_id)
    await clean_gateway._client.delete(f"task.result:{task_id}")
    with pytest.raises(CorruptAuthoritativeRecordError, match="refusing to fabricate"):
        await clean_gateway.get_result(task_id)


async def test_redis_get_result_completed_corrupt_result_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-badresult-{uuid.uuid4().hex[:8]}"
    await _commit_completed(clean_gateway, task_id)
    await clean_gateway._client.set(f"task.result:{task_id}", "not-json{{{")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt canonical result"):
        await clean_gateway.get_result(task_id)


async def test_redis_get_result_completed_wrong_task_result_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-wrongtask-{uuid.uuid4().hex[:8]}"
    await _commit_completed(clean_gateway, task_id)
    await clean_gateway._client.set(
        f"task.result:{task_id}",
        TaskResult(task_id="task-elsewhere", output="canonical read").to_json(),
    )
    with pytest.raises(CorruptAuthoritativeRecordError, match="not bound"):
        await clean_gateway.get_result(task_id)


async def test_redis_get_result_completed_non_completed_result_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-badstatus-{uuid.uuid4().hex[:8]}"
    await _commit_completed(clean_gateway, task_id)
    await clean_gateway._client.set(
        f"task.result:{task_id}",
        TaskResult(task_id=task_id, output="canonical read", status="failed").to_json(),
    )
    with pytest.raises(CorruptAuthoritativeRecordError, match="not bound"):
        await clean_gateway.get_result(task_id)


async def test_redis_get_result_corrupt_outcome_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-badoutcome-{uuid.uuid4().hex[:8]}"
    await _commit_completed(clean_gateway, task_id)
    await clean_gateway._client.set(f"task.outcome:{task_id}", "not-json{{{")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt terminal outcome"):
        await clean_gateway.get_result(task_id)


async def test_redis_get_result_mismatched_outcome_task_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-outcometask-{uuid.uuid4().hex[:8]}"
    await _commit_completed(clean_gateway, task_id)
    raw_outcome = await clean_gateway._client.get(f"task.outcome:{task_id}")
    assert raw_outcome is not None
    tampered = json.loads(raw_outcome)
    tampered["task_id"] = "task-elsewhere"
    await clean_gateway._client.set(f"task.outcome:{task_id}", json.dumps(tampered))
    with pytest.raises(CorruptAuthoritativeRecordError, match="not bound"):
        await clean_gateway.get_result(task_id)


async def test_redis_get_result_failed_with_unexpected_result_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-failresult-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(
        clean_gateway, task_id, "general", _worker_principal()
    )
    command = CompletionCommand(
        completion_command_id=f"cmd-{task_id}",
        assignment_id=assignment_id,
        execution_kind="initial",
        failure_code="EXECUTION_FAILED",
    )
    await clean_gateway.complete_turn(
        task_id=task_id,
        command=command,
        proof_digest=digest,
        fencing_epoch=1,
        parent_task_id=None,
        root_task_id=task_id,
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id="principal-worker-1",
        bound_agent_id="worker-1",
    )
    failed = await clean_gateway.get_result(task_id)
    assert failed == {"task_id": task_id, "status": "failed", "output": None}
    await clean_gateway._client.set(
        f"task.result:{task_id}",
        TaskResult(task_id=task_id, output="unexpected").to_json(),
    )
    with pytest.raises(CorruptAuthoritativeRecordError, match="Unexpected canonical result"):
        await clean_gateway.get_result(task_id)


async def test_redis_get_result_legacy_valid_and_invalid_records(
    clean_gateway: RedisGateway,
) -> None:
    valid_id = f"task-redis-legacyok-{uuid.uuid4().hex[:8]}"
    await clean_gateway._client.set(
        f"task.result:{valid_id}",
        json.dumps({"task_id": valid_id, "status": "completed", "output": "legacy output"}),
    )
    assert await clean_gateway.get_result(valid_id) == {
        "task_id": valid_id,
        "status": "completed",
        "output": "legacy output",
    }
    invalid_id = f"task-redis-legacybad-{uuid.uuid4().hex[:8]}"
    await clean_gateway._client.set(
        f"task.result:{invalid_id}",
        json.dumps({"task_id": "task-elsewhere", "output": "legacy output"}),
    )
    with pytest.raises(CorruptAuthoritativeRecordError, match="not bound"):
        await clean_gateway.get_result(invalid_id)


# --- Replay outcome/result consistency on real Redis (Finding 1) --------------------------


async def _canonical_snapshot(
    gateway: RedisGateway, task_id: str, assignment_id: str
) -> dict[str, object]:
    client = gateway._client
    return {
        "outcome": await client.get(f"task.outcome:{task_id}"),
        "result": await client.get(f"task.result:{task_id}"),
        "completion": await client.get(f"task.completion:{task_id}:cmd-{task_id}"),
        "audit": await client.get(f"task.assignment_completion:{task_id}:{assignment_id}"),
        "status": await client.get(f"task.status:{task_id}"),
        "dbsize": await client.dbsize(),
        "hints": await client.llen("agent.hints.research"),
    }


async def test_redis_replay_failed_outcome_with_unexpected_result_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-failreplay-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(
        clean_gateway, task_id, "general", _worker_principal()
    )
    await clean_gateway.complete_turn(
        task_id=task_id,
        command=CompletionCommand(
            completion_command_id=f"cmd-{task_id}",
            assignment_id=assignment_id,
            execution_kind="initial",
            failure_code="EXECUTION_FAILED",
        ),
        proof_digest=digest,
        fencing_epoch=1,
        parent_task_id=None,
        root_task_id=task_id,
        completed_at=datetime.now(UTC).isoformat(),
        runtime_principal_id="principal-worker-1",
        bound_agent_id="worker-1",
    )
    await clean_gateway._client.set(
        f"task.result:{task_id}",
        TaskResult(task_id=task_id, output="rogue").to_json(),
    )
    before = await _canonical_snapshot(clean_gateway, task_id, assignment_id)
    command = CompletionCommand(
        completion_command_id=f"cmd-{task_id}",
        assignment_id=assignment_id,
        execution_kind="initial",
        failure_code="EXECUTION_FAILED",
    )
    claim = await clean_gateway.get_active_claim(task_id)
    assert claim is None
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.complete_turn(
            task_id=task_id,
            command=command,
            proof_digest=digest,
            fencing_epoch=1,
            parent_task_id=None,
            root_task_id=task_id,
            completed_at=datetime.now(UTC).isoformat(),
            runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1",
        )
    assert await _canonical_snapshot(clean_gateway, task_id, assignment_id) == before


async def test_redis_replay_completed_outcome_with_missing_result_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-missingreplay-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(
        clean_gateway, task_id, "general", _worker_principal()
    )
    command = _completed_command(task_id, assignment_id, "initial")
    kwargs = {
        "task_id": task_id,
        "command": command,
        "proof_digest": digest,
        "fencing_epoch": 1,
        "parent_task_id": None,
        "root_task_id": task_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_principal_id": "principal-worker-1",
        "bound_agent_id": "worker-1",
    }
    _o, _r, replayed = await clean_gateway.complete_turn(**kwargs)
    assert replayed is False
    await clean_gateway._client.delete(f"task.result:{task_id}")
    before = await _canonical_snapshot(clean_gateway, task_id, assignment_id)
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.complete_turn(**kwargs)
    assert await _canonical_snapshot(clean_gateway, task_id, assignment_id) == before


async def test_redis_replay_completed_outcome_with_corrupt_result_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-corruptreplay-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(
        clean_gateway, task_id, "general", _worker_principal()
    )
    command = _completed_command(task_id, assignment_id, "initial")
    kwargs = {
        "task_id": task_id,
        "command": command,
        "proof_digest": digest,
        "fencing_epoch": 1,
        "parent_task_id": None,
        "root_task_id": task_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_principal_id": "principal-worker-1",
        "bound_agent_id": "worker-1",
    }
    await clean_gateway.complete_turn(**kwargs)
    await clean_gateway._client.set(f"task.result:{task_id}", "not-json{{{")
    before = await _canonical_snapshot(clean_gateway, task_id, assignment_id)
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.complete_turn(**kwargs)
    assert await _canonical_snapshot(clean_gateway, task_id, assignment_id) == before


async def test_redis_replay_completed_outcome_with_wrong_task_result_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-wrongtask-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(
        clean_gateway, task_id, "general", _worker_principal()
    )
    command = _completed_command(task_id, assignment_id, "initial")
    kwargs = {
        "task_id": task_id,
        "command": command,
        "proof_digest": digest,
        "fencing_epoch": 1,
        "parent_task_id": None,
        "root_task_id": task_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_principal_id": "principal-worker-1",
        "bound_agent_id": "worker-1",
    }
    await clean_gateway.complete_turn(**kwargs)
    await clean_gateway._client.set(
        f"task.result:{task_id}",
        TaskResult(task_id="task-elsewhere", output="canonical").to_json(),
    )
    before = await _canonical_snapshot(clean_gateway, task_id, assignment_id)
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.complete_turn(**kwargs)
    assert await _canonical_snapshot(clean_gateway, task_id, assignment_id) == before


async def test_redis_replay_completed_outcome_with_non_completed_result_fails_closed(
    clean_gateway: RedisGateway,
) -> None:
    task_id = f"task-redis-badstatus-{uuid.uuid4().hex[:8]}"
    await _register_pending(clean_gateway, task_id, "general")
    assignment_id, _proof, digest = await _claimed_turn(
        clean_gateway, task_id, "general", _worker_principal()
    )
    command = _completed_command(task_id, assignment_id, "initial")
    kwargs = {
        "task_id": task_id,
        "command": command,
        "proof_digest": digest,
        "fencing_epoch": 1,
        "parent_task_id": None,
        "root_task_id": task_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_principal_id": "principal-worker-1",
        "bound_agent_id": "worker-1",
    }
    await clean_gateway.complete_turn(**kwargs)
    await clean_gateway._client.set(
        f"task.result:{task_id}",
        TaskResult(task_id=task_id, output="canonical", status="failed").to_json(),
    )
    before = await _canonical_snapshot(clean_gateway, task_id, assignment_id)
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.complete_turn(**kwargs)
    assert await _canonical_snapshot(clean_gateway, task_id, assignment_id) == before


async def test_redis_child_failed_replay_with_unexpected_result_emits_no_hint(
    clean_gateway: RedisGateway,
) -> None:
    parent_id = f"task-redis-projfail-{uuid.uuid4().hex[:8]}"
    ctx = await _delegated_child_turn(clean_gateway, parent_id)
    child_id = ctx["child_id"]
    child_assignment_id = ctx["child_assignment_id"]
    fail_command = CompletionCommand(
        completion_command_id=f"cmd-{child_id}",
        assignment_id=child_assignment_id,
        execution_kind="delegated_child",
        failure_code="EXECUTION_FAILED",
    )
    fail_kwargs = {
        "task_id": child_id,
        "command": fail_command,
        "proof_digest": ctx["child_digest"],
        "fencing_epoch": 1,
        "parent_task_id": parent_id,
        "root_task_id": parent_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_principal_id": "principal-analysis-1",
        "bound_agent_id": "analysis-agent-1",
    }
    _o, _r, replayed = await clean_gateway.complete_turn(**fail_kwargs)
    assert replayed is False
    hints_before = await clean_gateway._client.llen("agent.hints.research")
    await clean_gateway._client.set(
        f"task.result:{child_id}",
        TaskResult(task_id=child_id, output="rogue").to_json(),
    )
    before = await _canonical_snapshot(clean_gateway, child_id, child_assignment_id)
    with pytest.raises(CorruptAuthoritativeRecordError):
        await clean_gateway.complete_turn(**fail_kwargs)
    assert await _canonical_snapshot(clean_gateway, child_id, child_assignment_id) == before
    assert await clean_gateway._client.llen("agent.hints.research") == hints_before
    assert await clean_gateway.get_status(parent_id) == "resumable"
