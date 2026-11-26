import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_contracts import (
    AgentAvailability,
    AgentAvailabilityState,
    AgentCandidate,
    AgentCapabilityMetadata,
    AgentEvent,
    AgentExecutionState,
    TaskRequested,
    WakeUpHint,
)
from orchestrator.application import (
    FindCandidatesByCapability,
    SubmitTask,
    TaskDispatcher,
)
from orchestrator.config import Settings
from orchestrator.domain import (
    AgentCapability,
    CandidateSelectionPolicy,
    CapabilityUnavailableError,
    DeterministicSelectionPolicy,
)
from orchestrator.infrastructure.redis_adapter import (
    CAPABILITY_QUEUE_MAP,
    RedisGateway,
)
from orchestrator.presentation.api import TaskInput, create_app, create_task
from research_agent.infrastructure.redis_bus import RedisBus as ResearchRedisBus
from starlette.requests import Request
from worker.infrastructure.redis_bus import RedisBus as WorkerRedisBus

from tests.test_application import FakeStateReader, FakeTaskQueue


class FakeCandidateFinder:
    def __init__(
        self, candidates_by_capability: dict[str, list[AgentCandidate]] | None = None
    ) -> None:
        self.candidates_by_capability = candidates_by_capability or {}
        self.calls: list[str] = []

    async def execute(
        self, capability: AgentCapability | str, as_of: Any = None
    ) -> list[AgentCandidate]:
        cap_str = capability.value if isinstance(capability, AgentCapability) else str(capability)
        self.calls.append(cap_str)
        return list(self.candidates_by_capability.get(cap_str, []))


class FakeEventPublisher:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []

    async def publish(self, event: AgentEvent) -> None:
        self.published.append(event)


def make_candidate(
    agent_id: str,
    capability: str = "general",
    dispatch_target: str | None = None,
    state: AgentAvailabilityState = AgentAvailabilityState.READY,
    execution_state: AgentExecutionState = AgentExecutionState.IDLE,
) -> AgentCandidate:
    now_str = datetime.now(UTC).isoformat()
    return AgentCandidate(
        agent_id=agent_id,
        capabilities=[capability],
        availability=AgentAvailability(
            state=state,
            execution_state=execution_state,
            heartbeat_age_seconds=1.0,
            last_seen_at=now_str,
            as_of=now_str,
            reason="healthy",
        ),
        agent_type="worker",
        execution_state=execution_state,
        dispatch_target=dispatch_target,
    )


# ==============================================================================
# 1. Metadata Contracts & Validation Tests
# ==============================================================================


def test_agent_capability_metadata_dispatch_target_defaults_to_none() -> None:
    meta = AgentCapabilityMetadata(
        agent_id="agent-1",
        capabilities=["general"],
    )
    assert meta.dispatch_target is None
    data = meta.to_dict()
    assert "dispatch_target" not in data or data["dispatch_target"] is None


def test_agent_capability_metadata_stores_and_serializes_dispatch_target() -> None:
    meta = AgentCapabilityMetadata(
        agent_id="worker-1",
        capabilities=["general"],
        dispatch_target="agent.tasks.general",
    )
    assert meta.dispatch_target == "agent.tasks.general"

    # Serialization round-trip
    data = meta.to_dict()
    assert data["dispatch_target"] == "agent.tasks.general"

    restored = AgentCapabilityMetadata.from_dict(data)
    assert restored.dispatch_target == "agent.tasks.general"
    assert restored == meta

    json_str = meta.to_json()
    restored_json = AgentCapabilityMetadata.from_json(json_str)
    assert restored_json.dispatch_target == "agent.tasks.general"
    assert restored_json == meta


def test_agent_capability_metadata_rejects_empty_dispatch_target() -> None:
    with pytest.raises(ValueError, match="dispatch_target must be a non-empty string"):
        AgentCapabilityMetadata(
            agent_id="worker-1",
            capabilities=["general"],
            dispatch_target="   ",
        )


def test_agent_candidate_stores_and_serializes_dispatch_target() -> None:
    candidate = make_candidate("worker-1", dispatch_target="agent.tasks.worker-1")
    assert candidate.dispatch_target == "agent.tasks.worker-1"

    data = candidate.to_dict()
    assert data["dispatch_target"] == "agent.tasks.worker-1"

    restored = AgentCandidate.from_dict(data)
    assert restored.dispatch_target == "agent.tasks.worker-1"
    assert restored.agent_id == "worker-1"


def test_agent_candidate_rejects_empty_dispatch_target() -> None:
    with pytest.raises(ValueError, match="dispatch_target must be a non-empty string"):
        make_candidate("worker-1", dispatch_target=" \t ")


# ==============================================================================
# 2. Agent Registration & Advertisement Tests
# ==============================================================================


@pytest.mark.anyio
async def test_worker_registers_configured_dispatch_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_TOKEN", "test-token")
    monkeypatch.setenv("REDIS_AUTH_MODE", "unauthenticated_local_test")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD_FILE", raising=False)
    monkeypatch.setenv("AGENT_ID", "worker-prod-1")
    monkeypatch.setenv("TASK_QUEUE", "agent.tasks.custom-worker-queue")

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus.register.assert_awaited_once()
        metadata: AgentCapabilityMetadata = mock_bus.register.call_args[0][0]
        assert metadata.agent_id == "worker-prod-1"
        assert metadata.dispatch_target == "agent.tasks.custom-worker-queue"
        assert "general" in metadata.capabilities


@pytest.mark.anyio
async def test_worker_default_dispatch_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_TOKEN", "test-token")
    monkeypatch.setenv("REDIS_AUTH_MODE", "unauthenticated_local_test")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD_FILE", raising=False)
    monkeypatch.setenv("AGENT_ID", "worker-default")
    monkeypatch.delenv("TASK_QUEUE", raising=False)

    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from worker.main import run_worker

        with pytest.raises(asyncio.CancelledError):
            await run_worker()

        mock_bus.register.assert_awaited_once()
        metadata: AgentCapabilityMetadata = mock_bus.register.call_args[0][0]
        assert metadata.dispatch_target == "agent.tasks.general"


@pytest.mark.anyio
async def test_research_agent_registers_configured_dispatch_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_TOKEN", "test-token")
    monkeypatch.setenv("REDIS_AUTH_MODE", "unauthenticated_local_test")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD_FILE", raising=False)
    monkeypatch.setenv("AGENT_ID", "research-prod-1")
    monkeypatch.setenv("TASK_QUEUE", "agent.tasks.custom-research-queue")

    with patch("research_agent.main.RedisBus") as mock_bus_cls, patch(
        "research_agent.main.ResearchProcessor"
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_bus.register.assert_awaited_once()
        metadata: AgentCapabilityMetadata = mock_bus.register.call_args[0][0]
        assert metadata.agent_id == "research-prod-1"
        assert metadata.dispatch_target == "agent.tasks.custom-research-queue"
        assert "research" in metadata.capabilities


@pytest.mark.anyio
async def test_research_agent_default_dispatch_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_TOKEN", "test-token")
    monkeypatch.setenv("REDIS_AUTH_MODE", "unauthenticated_local_test")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD_FILE", raising=False)
    monkeypatch.setenv("AGENT_ID", "research-default")
    monkeypatch.delenv("TASK_QUEUE", raising=False)

    with patch("research_agent.main.RedisBus") as mock_bus_cls, patch(
        "research_agent.main.ResearchProcessor"
    ):
        mock_bus = AsyncMock()
        mock_bus_cls.return_value = mock_bus
        mock_bus.next_task.side_effect = asyncio.CancelledError()

        from research_agent.main import run_research_agent

        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()

        mock_bus.register.assert_awaited_once()
        metadata: AgentCapabilityMetadata = mock_bus.register.call_args[0][0]
        assert metadata.dispatch_target == "agent.tasks.research"


@pytest.mark.anyio
async def test_worker_redis_bus_persists_dispatch_target_in_hash() -> None:
    bus = WorkerRedisBus("redis://localhost:6379/0", queue_name="agent.tasks.worker-pool-1", allow_unauthenticated_local_test=True)
    bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    bus.client.hset = AsyncMock()  # type: ignore[method-assign]

    meta = AgentCapabilityMetadata(
        agent_id="worker-pool-1",
        capabilities=["general"],
        agent_type="worker",
        dispatch_target="agent.tasks.worker-pool-1",
    )
    await bus.register(meta)

    bus.client.hset.assert_awaited_once()
    call_kwargs = bus.client.hset.call_args[1]
    mapping = call_kwargs["mapping"]
    assert mapping["dispatch_target"] == "agent.tasks.worker-pool-1"


@pytest.mark.anyio
async def test_research_redis_bus_persists_dispatch_target_in_hash() -> None:
    bus = ResearchRedisBus("redis://localhost:6379/0", queue_name="agent.tasks.research-alpha", allow_unauthenticated_local_test=True)
    bus.client.sadd = AsyncMock()  # type: ignore[method-assign]
    bus.client.hset = AsyncMock()  # type: ignore[method-assign]

    meta = AgentCapabilityMetadata(
        agent_id="research-alpha",
        capabilities=["research"],
        agent_type="research-agent",
        dispatch_target="agent.tasks.research-alpha",
    )
    await bus.register(meta)

    bus.client.hset.assert_awaited_once()
    call_kwargs = bus.client.hset.call_args[1]
    mapping = call_kwargs["mapping"]
    assert mapping["dispatch_target"] == "agent.tasks.research-alpha"


# ==============================================================================
# 3. Candidate Discovery & Metadata Propagation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_find_candidates_propagates_advertised_dispatch_target() -> None:
    now = datetime.now(UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-42",
                "agent_type": "worker",
                "capabilities": ["general"],
                "dispatch_target": "agent.tasks.worker-42-custom",
                "state": "idle",
                "updated_at": now.isoformat(),
            }
        ]
    )
    query = FindCandidatesByCapability(reader)
    candidates = await query.execute(AgentCapability.GENERAL, as_of=now + timedelta(seconds=1))

    assert len(candidates) == 1
    assert candidates[0].agent_id == "worker-42"
    assert candidates[0].dispatch_target == "agent.tasks.worker-42-custom"


@pytest.mark.anyio
async def test_find_candidates_handles_missing_dispatch_target_gracefully() -> None:
    now = datetime.now(UTC)
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "legacy-worker",
                "agent_type": "worker",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": now.isoformat(),
            }
        ]
    )
    query = FindCandidatesByCapability(reader)
    candidates = await query.execute(AgentCapability.GENERAL, as_of=now + timedelta(seconds=1))

    assert len(candidates) == 1
    assert candidates[0].agent_id == "legacy-worker"
    assert candidates[0].dispatch_target is None


# ==============================================================================
# 4. Selection Policy Tests
# ==============================================================================


def test_deterministic_selection_policy_empty_candidates_raises() -> None:
    policy = DeterministicSelectionPolicy()
    with pytest.raises(ValueError, match="Cannot select candidate from an empty candidate list"):
        policy.select([])


def test_deterministic_selection_policy_single_candidate() -> None:
    policy = DeterministicSelectionPolicy()
    c = make_candidate("worker-1")
    assert policy.select([c]) == c


def test_deterministic_selection_policy_multiple_candidates_is_order_independent() -> None:
    policy = DeterministicSelectionPolicy()
    c1 = make_candidate("agent-bravo", dispatch_target="queue-bravo")
    c2 = make_candidate("agent-alpha", dispatch_target="queue-alpha")
    c3 = make_candidate("agent-charlie", dispatch_target="queue-charlie")

    # Regardless of input permutation, "agent-alpha" is lexicographically first
    assert policy.select([c1, c2, c3]).agent_id == "agent-alpha"
    assert policy.select([c3, c1, c2]).agent_id == "agent-alpha"
    assert policy.select([c2, c3, c1]).agent_id == "agent-alpha"


def test_deterministic_selection_policy_arbitrary_ids() -> None:
    policy = DeterministicSelectionPolicy()
    c_uuid1 = make_candidate("00000000-0000-0000-0000-000000000002")
    c_uuid2 = make_candidate("00000000-0000-0000-0000-000000000001")

    selected = policy.select([c_uuid1, c_uuid2])
    assert selected.agent_id == "00000000-0000-0000-0000-000000000001"


def test_custom_selection_policy_extension() -> None:
    class ReverseAlphabeticalPolicy:
        def select(self, candidates: list[AgentCandidate]) -> AgentCandidate:
            return max(candidates, key=lambda c: c.agent_id)

    policy: CandidateSelectionPolicy = ReverseAlphabeticalPolicy()
    c1 = make_candidate("agent-a")
    c2 = make_candidate("agent-z")
    assert policy.select([c1, c2]).agent_id == "agent-z"


# ==============================================================================
# 5. Orchestrator Dynamic Dispatch Tests
# ==============================================================================


@pytest.mark.anyio
async def test_orchestrator_dispatches_to_selected_candidate_advertised_target() -> None:
    queue = FakeTaskQueue()
    target_queue = "agent.tasks.custom-worker-cluster"
    candidate = make_candidate("worker-1", capability="general", dispatch_target=target_queue)
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(task_store=queue, queue=queue, candidate_finder=finder)
    task = await submit_task.execute("Process background task")

    # Verify task was dispatched to the advertised target queue
    assert len(queue.dispatches) == 1
    dispatched_task, destination = queue.dispatches[0]
    assert dispatched_task.task_id == task.task_id
    assert destination == target_queue


@pytest.mark.anyio
async def test_dynamic_queue_change_requires_no_orchestrator_topology_changes() -> None:
    queue = FakeTaskQueue()
    # A completely unexpected, dynamically registered queue name
    dynamic_queue = "agent.tasks.ephemeral-worker-node-883"
    candidate = make_candidate(
        "worker-ephemeral-883", capability="general", dispatch_target=dynamic_queue
    )
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(task_store=queue, queue=queue, candidate_finder=finder)
    task = await submit_task.execute("Process background batch job")

    assert len(queue.dispatches) == 1
    assert queue.dispatches[0][1] == dynamic_queue
    assert queue.dispatches[0][0].task_id == task.task_id


@pytest.mark.anyio
async def test_dynamic_dispatch_does_not_depend_on_capability_queue_map() -> None:
    queue = FakeTaskQueue()
    target = "agent.tasks.dynamic-research-instance"
    candidate = make_candidate("research-agent-dynamic", capability="research", dispatch_target=target)
    finder = FakeCandidateFinder({"research": [candidate]})

    submit_task = SubmitTask(task_store=queue, queue=queue, candidate_finder=finder)

    # Temporarily clear the static queue map to prove zero runtime dependency during dynamic dispatch
    with patch.dict(CAPABILITY_QUEUE_MAP, {}, clear=True):
        assert len(CAPABILITY_QUEUE_MAP) == 0
        task = await submit_task.execute(
            "Survey distributed tracing protocols",
            capability=AgentCapability.RESEARCH,
        )

    assert len(queue.dispatches) == 1
    assert queue.dispatches[0][1] == target
    assert queue.dispatches[0][0].task_id == task.task_id


@pytest.mark.anyio
async def test_dispatch_with_multiple_candidates_selects_deterministically() -> None:
    queue = FakeTaskQueue()
    c_beta = make_candidate("worker-beta", capability="general", dispatch_target="queue.worker.beta")
    c_alpha = make_candidate(
        "worker-alpha", capability="general", dispatch_target="queue.worker.alpha"
    )
    c_gamma = make_candidate(
        "worker-gamma", capability="general", dispatch_target="queue.worker.gamma"
    )

    # Provide in non-alphabetical order
    finder = FakeCandidateFinder({"general": [c_beta, c_alpha, c_gamma]})
    submit_task = SubmitTask(task_store=queue, queue=queue, candidate_finder=finder)

    task = await submit_task.execute("Compile documentation")

    # worker-alpha is selected deterministically
    assert len(queue.dispatches) == 1
    dispatched_task, destination = queue.dispatches[0]
    assert dispatched_task.task_id == task.task_id
    assert destination == "queue.worker.alpha"


@pytest.mark.anyio
async def test_rejection_preserved_when_zero_candidates() -> None:
    queue = FakeTaskQueue()
    finder = FakeCandidateFinder({"general": []})
    events = FakeEventPublisher()

    submit_task = SubmitTask(task_store=queue, queue=queue, candidate_finder=finder, events=events)

    with pytest.raises(CapabilityUnavailableError) as exc_info:
        await submit_task.execute("Execute standard batch task")

    assert exc_info.value.capability == AgentCapability.GENERAL
    assert "No eligible agent is currently available" in str(exc_info.value.message)

    # Rejection event published
    assert len(events.published) == 1
    assert events.published[0].state == "error"
    assert events.published[0].agent_id == "orchestrator"

    # Nothing dispatched or enqueued
    assert len(queue.dispatches) == 0
    assert len(queue.enqueued) == 0


@pytest.mark.anyio
async def test_ineligible_candidates_are_excluded_by_candidate_finder() -> None:
    now = datetime.now(UTC)
    # Register agents that are BUSY, STALE, and OFFLINE
    reader = FakeStateReader(
        agents_data=[
            {
                "agent_id": "worker-busy",
                "capabilities": ["general"],
                "state": "working",
                "updated_at": now.isoformat(),
                "dispatch_target": "queue.busy",
            },
            {
                "agent_id": "worker-stale",
                "capabilities": ["general"],
                "state": "idle",
                "updated_at": (now - timedelta(seconds=120)).isoformat(),
                "dispatch_target": "queue.stale",
            },
            {
                "agent_id": "worker-offline",
                "capabilities": ["general"],
                "state": "offline",
                "updated_at": now.isoformat(),
                "dispatch_target": "queue.offline",
            },
        ]
    )
    finder = FindCandidatesByCapability(reader)
    queue = FakeTaskQueue()
    submit_task = SubmitTask(task_store=queue, queue=queue, candidate_finder=finder)

    # Zero eligible candidates should be found, resulting in rejection
    with pytest.raises(CapabilityUnavailableError):
        await submit_task.execute("Analyze telemetry", capability=AgentCapability.GENERAL)

    assert len(queue.dispatches) == 0


@pytest.mark.anyio
async def test_candidate_without_dispatch_target_fails_explicitly_without_queue_map_fallback() -> None:
    queue = FakeTaskQueue()
    # Candidate advertised no dispatch target
    candidate = make_candidate("worker-legacy", capability="general", dispatch_target=None)
    # Ensure dispatch_target is explicitly None
    object.__setattr__(candidate, "dispatch_target", None)
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(task_store=queue, queue=queue, candidate_finder=finder)

    with (
        patch.dict(CAPABILITY_QUEUE_MAP, {AgentCapability.GENERAL: "agent.tasks.general"}),
        pytest.raises(
            ValueError,
            match="Selected candidate 'worker-legacy' does not have a valid dispatch target advertised",
        ),
    ):
        await submit_task.execute("Process data pipeline")

    # Proves no silent fallback occurred:
    assert len(queue.dispatches) == 0
    assert len(queue.enqueued) == 0


@pytest.mark.anyio
async def test_candidate_with_blank_dispatch_target_fails_explicitly() -> None:
    queue = FakeTaskQueue()
    # Candidate with blank/empty string dispatch target
    candidate = make_candidate("worker-blank", capability="general")
    object.__setattr__(candidate, "dispatch_target", "   ")
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(task_store=queue, queue=queue, candidate_finder=finder)

    with pytest.raises(
        ValueError,
        match="Selected candidate 'worker-blank' does not have a valid dispatch target advertised",
    ):
        await submit_task.execute("Process data pipeline")

    assert len(queue.dispatches) == 0
    assert len(queue.enqueued) == 0


@pytest.mark.anyio
async def test_production_dynamic_dispatch_via_redis_gateway_never_depends_on_capability_queue_map() -> None:
    custom_target = "agent.tasks.production-custom-node"
    gateway = RedisGateway(
        "redis://localhost:6379/0",
        allow_unauthenticated_local_test=True,
        capability_queue_contract={"general": custom_target},
    )
    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)
    gateway._client.pipeline = MagicMock(return_value=mock_pipe)
    gateway._client.eval = AsyncMock(return_value="OK")

    candidate = make_candidate("worker-prod", capability="general", dispatch_target=custom_target)
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(task_store=gateway, dispatcher=gateway, candidate_finder=finder)

    # Completely clear the capability queue map
    with patch.dict(CAPABILITY_QUEUE_MAP, {}, clear=True):
        assert len(CAPABILITY_QUEUE_MAP) == 0
        task = await submit_task.execute("Process production event")

    # Dynamic dispatch succeeded strictly using the candidate's advertised target via atomic pipeline
    mock_pipe.rpush.assert_called_once_with(custom_target, task.to_json())
    mock_pipe.set.assert_any_call(
        f"task.status:{task.task_id}",
        json.dumps({"task_id": task.task_id, "status": "pending"}),
    )


@pytest.mark.anyio
async def test_runtime_dispatch_prefers_explicit_task_dispatcher() -> None:
    class PureDispatcher:
        def __init__(self) -> None:
            self.dispatches: list[tuple[TaskRequested, str]] = []

        async def dispatch(
            self,
            task: TaskRequested,
            target: str,
            capability: str | None = None,
        ) -> None:
            self.dispatches.append((task, target))

    dispatcher = PureDispatcher()
    assert isinstance(dispatcher, TaskDispatcher)
    candidate = make_candidate("worker-dispatch", capability="general", dispatch_target="queue.target")
    finder = FakeCandidateFinder({"general": [candidate]})

    submit_task = SubmitTask(
        task_store=FakeTaskQueue(), dispatcher=dispatcher, candidate_finder=finder
    )
    task = await submit_task.execute("Process batch job")

    assert len(dispatcher.dispatches) == 1
    assert dispatcher.dispatches[0][0].task_id == task.task_id
    assert dispatcher.dispatches[0][1] == "queue.target"


@pytest.mark.anyio
async def test_legacy_non_dynamic_path_preserves_enqueue_for_isolated_callers() -> None:
    queue = FakeTaskQueue()
    # Non-dynamic mode: candidate_finder is None
    submit_task = SubmitTask(task_store=queue, queue=queue, candidate_finder=None)
    task = await submit_task.execute("Process legacy task")

    assert len(queue.enqueued) == 1
    assert queue.enqueued[0][0].task_id == task.task_id
    assert queue.enqueued[0][1] == AgentCapability.GENERAL


# ==============================================================================
# 6. RedisGateway & Task Status / Result Lifecycle Tests
# ==============================================================================


@pytest.mark.anyio
async def test_redis_gateway_dispatch_sets_pending_status_and_pushes_to_target() -> None:
    gateway = RedisGateway("redis://localhost:6379/0", allow_unauthenticated_local_test=True)
    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)
    gateway._client.pipeline = MagicMock(return_value=mock_pipe)
    gateway._client.eval = AsyncMock(return_value="OK")

    task = TaskRequested.create("Execute runtime task")
    target = "agent.tasks.general"

    await gateway.dispatch(task, target=target, capability="general")

    # Sets status: pending and pushes to target via atomic pipeline
    mock_pipe.set.assert_called_once_with(
        f"task.status:{task.task_id}",
        json.dumps({"task_id": task.task_id, "status": "pending"}),
    )
    mock_pipe.rpush.assert_called_once_with("agent.tasks.general", task.to_json())
    mock_pipe.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_gateway_dispatch_rejects_empty_target() -> None:
    gateway = RedisGateway("redis://localhost:6379/0", allow_unauthenticated_local_test=True)
    task = TaskRequested.create("Some prompt")

    with pytest.raises(ValueError, match="Authoritative dispatch capability is required"):
        await gateway.dispatch(task, target="   ")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "target",
    [
        "",
        " ",
        "\n",
        "agent.tasks.general ",
        "agent.tasks.general-extra",
        "task.status:task-1",
        "agent.events",
        "agent.ids",
        "agent.state:worker-1",
        "agent.hints.general",
        "agent.tasks.research",
        "agent.tasks.*",
    ],
)
async def test_redis_gateway_dispatch_rejects_malformed_or_cross_capability_target(
    target: object,
) -> None:
    gateway = RedisGateway("redis://localhost:6379/0", allow_unauthenticated_local_test=True)
    task = TaskRequested.create("Execute general task")

    gateway._client.pipeline = MagicMock(
        side_effect=AssertionError("invalid destination must not construct a pipeline")
    )
    with pytest.raises(ValueError):
        await gateway.dispatch(task, target=target, capability="general")  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_redis_gateway_dispatch_accepts_injected_finite_custom_contract() -> None:
    gateway = RedisGateway(
        "redis://localhost:6379/0",
        allow_unauthenticated_local_test=True,
        capability_queue_contract={"general": "agent.tasks.custom-general"},
    )
    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)
    gateway._client.pipeline = MagicMock(return_value=mock_pipe)

    task = TaskRequested.create("Execute custom task")
    await gateway.dispatch(
        task,
        target="agent.tasks.custom-general",
        capability="general",
    )

    mock_pipe.rpush.assert_called_once_with(
        "agent.tasks.custom-general",
        task.to_json(),
    )


@pytest.mark.anyio
async def test_redis_gateway_result_lifecycle_after_dispatch() -> None:
    gateway = RedisGateway("redis://localhost:6379/0", allow_unauthenticated_local_test=True)
    task_id = "test-task-lifecycle-123"

    # 1. While task is pending (written by dispatch)
    gateway._client.get = AsyncMock(
        side_effect=lambda key: (
            json.dumps({"task_id": task_id, "status": "pending"})
            if key == f"task.status:{task_id}"
            else None
        )
    )
    res_pending = await gateway.get_result(task_id)
    assert res_pending is not None
    assert res_pending["status"] == "pending"
    assert res_pending["output"] is None

    # 2. When task has completed result
    completed_data = {
        "task_id": task_id,
        "status": "completed",
        "output": "Execution successful",
        "acknowledged": True,
    }
    gateway._client.get = AsyncMock(
        side_effect=lambda key: (
            json.dumps(completed_data) if key == f"task.result:{task_id}" else None
        )
    )
    res_completed = await gateway.get_result(task_id)
    assert res_completed is not None
    assert res_completed["status"] == "completed"
    assert res_completed["output"] == "Execution successful"


# ==============================================================================
# 7. Presentation & API Route Tests
# ==============================================================================


@pytest.mark.anyio
async def test_api_create_task_dispatches_to_advertised_target() -> None:
    queue = FakeTaskQueue()
    target_queue = "agent.tasks.api-worker"
    candidate = make_candidate("api-worker", capability="general", dispatch_target=target_queue)
    finder = FakeCandidateFinder({"general": [candidate]})

    app = create_app(
        settings=Settings(),
        task_queue=queue,
        candidate_finder=finder,
        enforce_availability=True,
    )

    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "app": app,
            "path": "/api/tasks",
            "headers": [],
        }
    )

    response = await create_task(TaskInput(prompt="Check system status"), request)
    assert response["status"] == "accepted"
    task_id = response["task_id"]

    assert len(queue.dispatches) == 1
    dispatched_task, destination = queue.dispatches[0]
    assert dispatched_task.task_id == task_id
    assert destination == target_queue


@pytest.mark.anyio
async def test_api_create_task_returns_503_when_no_candidates() -> None:
    queue = FakeTaskQueue()
    finder = FakeCandidateFinder({"general": []})

    app = create_app(
        settings=Settings(),
        task_queue=queue,
        candidate_finder=finder,
        enforce_availability=True,
    )

    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "app": app,
            "path": "/api/tasks",
            "headers": [],
        }
    )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await create_task(TaskInput(prompt="Check system status"), request)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error"] == "capability_unavailable"
    assert exc_info.value.detail["capability"] == "general"


# ==============================================================================
# 8. Governed dispatch_hint finite capability-to-hint contract (Issue #229 1C/B2)
# ==============================================================================


def _make_hint(capability: str = "general") -> WakeUpHint:
    return WakeUpHint(
        hint_id="hint-1",
        capability=capability,
        created_at="2026-11-24T00:00:00Z",
        execution_kind="initial",
        task_id="task-1",
    )


def _hint_gateway(**kwargs: object) -> RedisGateway:
    return RedisGateway(
        "redis://localhost:6379/0",
        allow_unauthenticated_local_test=True,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("capability", "destination"),
    [
        ("general", "agent.hints.general"),
        ("research", "agent.hints.research"),
        ("analysis", "agent.hints.analysis"),
    ],
)
async def test_dispatch_hint_canonical_mappings_succeed(
    capability: str, destination: str
) -> None:
    gateway = _hint_gateway()
    gateway._client.rpush = AsyncMock()
    hint = _make_hint(capability)

    await gateway.dispatch_hint(hint, destination, capability=capability)

    gateway._client.rpush.assert_awaited_once_with(destination, hint.to_json())


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("capability", "destination"),
    [
        ("general", "agent.hints.general"),
        ("research", "agent.hints.research"),
        ("analysis", "agent.hints.analysis"),
    ],
)
async def test_dispatch_hint_none_target_resolves_from_capability(
    capability: str, destination: str
) -> None:
    gateway = _hint_gateway()
    gateway._client.rpush = AsyncMock()
    hint = _make_hint(capability)

    await gateway.dispatch_hint(hint, None, capability=capability)

    gateway._client.rpush.assert_awaited_once_with(destination, hint.to_json())


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("capability", "destination"),
    [
        ("general", "agent.hints.research"),
        ("general", "agent.hints.analysis"),
        ("research", "agent.hints.general"),
        ("research", "agent.hints.analysis"),
        ("analysis", "agent.hints.general"),
        ("analysis", "agent.hints.research"),
    ],
)
async def test_dispatch_hint_rejects_cross_capability_pair(
    capability: str, destination: str
) -> None:
    gateway = _hint_gateway()
    gateway._client.rpush = AsyncMock()

    with pytest.raises(ValueError, match="does not match authoritative capability"):
        await gateway.dispatch_hint(_make_hint(capability), destination, capability=capability)

    gateway._client.rpush.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("capability", [None, "", "unknown", 123, ["general"]])
async def test_dispatch_hint_rejects_missing_or_unknown_capability(
    capability: object,
) -> None:
    gateway = _hint_gateway()
    gateway._client.rpush = AsyncMock()

    with pytest.raises(ValueError, match="[Aa]uthoritative hint capability|Unrecognized"):
        await gateway.dispatch_hint(
            _make_hint(), "agent.hints.general", capability=capability  # type: ignore[arg-type]
        )

    gateway._client.rpush.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "target",
    [
        "",
        "   ",
        "\n",
        "\t",
        " agent.hints.general",
        "agent.hints.general ",
        " agent.hints.general ",
        "agent.hints.general\n",
        "agent.hints.gener*",
        "agent.hints.*",
        "agent.hints.general?",
        "agent.hints",
        "agent.hints.generalx",
        "xagent.hints.general",
        "agent.hints.general/extra",
        "agent.tasks.general",
        "agent.tasks.research",
        "task.status:task-1",
        "task.routing:task-1",
        "task.*",
        "agent.events",
        "agent.ids",
        "agent.state:worker-1",
        "agent.state:*",
        "agent\\.hints\\.general",
        "AGENT.HINTS.GENERAL",
        "Agent.Hints.General",
    ],
)
async def test_dispatch_hint_rejects_untrusted_target_variants(target: object) -> None:
    gateway = _hint_gateway()
    gateway._client.rpush = AsyncMock()

    with pytest.raises(ValueError):
        await gateway.dispatch_hint(
            _make_hint("general"), target, capability="general"  # type: ignore[arg-type]
        )

    gateway._client.rpush.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("target", [123, 4.5, b"agent.hints.general", ["agent.hints.general"]])
async def test_dispatch_hint_rejects_non_string_target(target: object) -> None:
    gateway = _hint_gateway()
    gateway._client.rpush = AsyncMock()

    with pytest.raises(ValueError, match="must be a string or omitted"):
        await gateway.dispatch_hint(
            _make_hint("general"), target, capability="general"  # type: ignore[arg-type]
        )

    gateway._client.rpush.assert_not_awaited()


@pytest.mark.anyio
async def test_dispatch_hint_control_character_target_fails_closed() -> None:
    gateway = _hint_gateway()
    gateway._client.rpush = AsyncMock()

    with pytest.raises(ValueError):
        await gateway.dispatch_hint(
            _make_hint("general"), "agent.hints.general\x00", capability="general"
        )
    with pytest.raises(ValueError):
        await gateway.dispatch_hint(
            _make_hint("general"), "agent.hints.general\x07", capability="general"
        )

    gateway._client.rpush.assert_not_awaited()


@pytest.mark.anyio
async def test_dispatch_hint_sends_canonical_constant_not_raw_input() -> None:
    gateway = _hint_gateway()
    gateway._client.rpush = AsyncMock()
    hint = _make_hint("research")
    raw = "agent.hints." + "research"
    assert raw == "agent.hints.research"
    assert raw is not gateway._capability_hint_contract["research"]

    await gateway.dispatch_hint(hint, raw, capability="research")

    (destination, _payload), _kwargs = gateway._client.rpush.call_args
    assert destination == "agent.hints.research"
    assert destination is gateway._capability_hint_contract["research"]


@pytest.mark.anyio
async def test_dispatch_hint_accepts_injected_finite_custom_contract() -> None:
    gateway = _hint_gateway(
        capability_hint_contract={"custom": "agent.hints.custom"},
    )
    gateway._client.rpush = AsyncMock()
    hint = _make_hint("general")

    await gateway.dispatch_hint(hint, "agent.hints.custom", capability="custom")

    gateway._client.rpush.assert_awaited_once_with("agent.hints.custom", hint.to_json())


@pytest.mark.anyio
@pytest.mark.parametrize(
    "contract",
    [
        {},
        {"": "agent.hints.empty-capability"},
        {"custom": ""},
        {"custom": "agent.hints.*"},
        {"custom": "agent.hints.custom*"},
        {"custom": "agent.hints.custom\x00"},
        {"custom": "agent.hints.custom\n"},
    ],
)
async def test_dispatch_hint_rejects_invalid_custom_contract(
    contract: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        _hint_gateway(capability_hint_contract=contract)


@pytest.mark.anyio
async def test_submit_task_preserves_raw_whitespace_padded_target_and_fails_closed() -> None:
    gateway = RedisGateway("redis://localhost:6379/0", allow_unauthenticated_local_test=True)
    gateway._client.eval = AsyncMock(return_value="OK")
    gateway._client.set = AsyncMock(return_value=True)
    gateway._client.pipeline = MagicMock(
        side_effect=AssertionError("whitespace-padded target must never construct a pipeline")
    )

    candidate = make_candidate(
        "worker-padded",
        capability="general",
        dispatch_target=" agent.tasks.general ",
    )
    finder = FakeCandidateFinder({"general": [candidate]})
    submit_task = SubmitTask(task_store=gateway, dispatcher=gateway, candidate_finder=finder)

    with pytest.raises(ValueError, match="does not match authoritative capability"):
        await submit_task.execute("Execute general task")

    gateway._client.set.assert_awaited_once()
    set_key, set_val = gateway._client.set.call_args.args
    assert set_key.startswith("task.status:")
    assert json.loads(set_val)["status"] == "dispatch_failed"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: __import__("redis.exceptions", fromlist=["NoPermissionError"]).NoPermissionError(
            "NOPERM this user has no permissions"
        ),
        lambda: __import__(
            "redis.exceptions", fromlist=["AuthenticationError"]
        ).AuthenticationError("WRONGPASS invalid username-password pair"),
    ],
)
async def test_redis_gateway_preserves_nopermission_and_authentication_errors(
    exc_factory: Any,
) -> None:
    from orchestrator.domain.task import CorruptAuthoritativeRecordError

    exc_type = type(exc_factory())
    gateway = RedisGateway("redis://localhost:6379/0", allow_unauthenticated_local_test=True)
    gateway._client.eval = AsyncMock(side_effect=exc_factory())
    gateway._client.get = AsyncMock(side_effect=exc_factory())

    task = TaskRequested.create("Test auth exception preservation")
    with pytest.raises(exc_type):
        await gateway.register_root_task(task, required_capability="general")

    for method in (
        gateway.get_status,
        gateway.get_lineage,
        gateway.get_required_capability,
        gateway.get_result,
        gateway.get_claim_expiry,
    ):
        with pytest.raises(exc_type):
            await method("task-auth-exc-1")

    # Malformed payload still maps to CorruptAuthoritativeRecordError
    gateway._client.get = AsyncMock(return_value="{not-valid-json")
    with pytest.raises(CorruptAuthoritativeRecordError):
        await gateway.get_status("task-corrupt-1")
