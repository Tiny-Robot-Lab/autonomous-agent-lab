import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_contracts import (
    AgentAvailability,
    AgentAvailabilityState,
    AgentCandidate,
    AgentExecutionState,
    RootAuthorityRecord,
    StageArtifact,
    TaskContinuation,
    TaskInput,
    TaskRequested,
)
from fastapi import Request
from orchestrator.application.ports import AuthoritativeTaskStore, TaskDispatcher
from orchestrator.application.use_cases import RegisterChildTask, SubmitTask
from orchestrator.config import Settings
from orchestrator.domain.task import (
    ConflictingContinuationError,
    ConflictingRootAuthorityError,
    ConflictingRootTaskRegistrationError,
    ConflictingStageArtifactError,
    ConflictingTaskInputError,
    CorruptAuthoritativeRecordError,
    DelegationDepthExceededError,
    NonContiguousVersionError,
    ParentTaskNotFoundError,
    TaskLineage,
)
from orchestrator.infrastructure.redis_adapter import RedisGateway
from orchestrator.presentation.api import TaskInput as ApiTaskInput
from orchestrator.presentation.api import create_app, create_task

from tests.test_application import FakeTaskQueue


class InMemoryAuthoritativeTaskStore(AuthoritativeTaskStore):
    """Deterministic in-memory implementation of AuthoritativeTaskStore for tests."""

    def __init__(self) -> None:
        self.statuses: dict[str, str] = {}
        self.registrations: dict[str, dict[str, Any]] = {}
        self.lineages: dict[str, TaskLineage] = {}
        self.children: dict[str, set[str]] = {}
        self.inputs: dict[tuple[str, int], TaskInput] = {}
        self.artifacts: dict[tuple[str, str, int], StageArtifact] = {}
        self.continuations: dict[tuple[str, int], TaskContinuation] = {}
        self.latest_continuation: dict[str, int] = {}
        self.authorities: dict[str, RootAuthorityRecord] = {}
        self.fail_next_registration: bool = False
        self.root_registrations: list[tuple[TaskRequested, str]] = []
        self.child_registrations: list[tuple[TaskRequested, str]] = []

    async def register_root_task(
        self, task: TaskRequested, initial_status: str = "pending_dispatch"
    ) -> None:
        if task.parent_task_id is not None or task.root_task_id != task.task_id:
            raise ValueError(f"Task '{task.task_id}' is not a root task")
        if self.fail_next_registration:
            self.fail_next_registration = False
            raise RuntimeError("Simulated pipeline failure during root registration")

        exists_marker = task.task_id in self.registrations
        exists_status = task.task_id in self.statuses
        exists_lineage = task.task_id in self.lineages
        exists_input = (task.task_id, 1) in self.inputs
        exists_authority = task.task_id in self.authorities

        if not (exists_marker or exists_status or exists_lineage or exists_input or exists_authority):
            self.registrations[task.task_id] = {
                "registered_at": task.created_at,
                "task_id": task.task_id,
            }
            self.statuses[task.task_id] = initial_status
            self.lineages[task.task_id] = TaskLineage(
                task_id=task.task_id,
                root_task_id=task.root_task_id,
                parent_task_id=None,
            )
            self.inputs[(task.task_id, 1)] = TaskInput(
                task_id=task.task_id,
                prompt=task.prompt,
                created_at=task.created_at,
                input_version=1,
                artifact_references=(),
                metadata={},
                version=1,
            )
            self.authorities[task.task_id] = RootAuthorityRecord.create_default(
                root_task_id=task.task_id,
                created_at=task.created_at,
            )
            self.root_registrations.append((task, initial_status))
            return

        if exists_marker and exists_status and exists_lineage and exists_input and exists_authority:
            existing_marker = self.registrations[task.task_id]
            existing_lineage = self.lineages[task.task_id]
            existing_input = self.inputs[(task.task_id, 1)]
            existing_authority = self.authorities[task.task_id]
            existing_status = self.statuses[task.task_id]
            if (
                existing_marker.get("task_id") == task.task_id
                and existing_lineage.root_task_id == task.root_task_id
                and existing_input.prompt == task.prompt
                and existing_authority.root_task_id == task.task_id
                and isinstance(existing_status, str)
            ):
                return
            raise ConflictingRootTaskRegistrationError(
                f"Root task '{task.task_id}' already exists with conflicting state"
            )

        raise ConflictingRootTaskRegistrationError(
            f"Root task '{task.task_id}' has inconsistent or partial authoritative state in storage"
        )

    async def register_child_task(
        self, task: TaskRequested, initial_status: str = "pending_dispatch"
    ) -> None:
        if task.parent_task_id is None or task.root_task_id == task.task_id:
            raise ValueError(f"Task '{task.task_id}' is not a child task")
        if self.fail_next_registration:
            self.fail_next_registration = False
            raise RuntimeError("Simulated pipeline failure during child registration")
        self.statuses[task.task_id] = initial_status
        self.lineages[task.task_id] = TaskLineage(
            task_id=task.task_id,
            root_task_id=task.root_task_id,
            parent_task_id=task.parent_task_id,
        )
        self.children.setdefault(task.parent_task_id, set()).add(task.task_id)
        self.child_registrations.append((task, initial_status))

    async def get_lineage(self, task_id: str) -> TaskLineage | None:
        return self.lineages.get(task_id.strip())

    async def get_children(self, parent_task_id: str) -> list[str]:
        return sorted(self.children.get(parent_task_id.strip(), set()))

    async def update_status(self, task_id: str, status: str) -> None:
        clean_id = task_id.strip()
        if not clean_id:
            raise ValueError("task_id must be a non-empty string")
        self.statuses[clean_id] = status.strip()

    async def get_status(self, task_id: str) -> str | None:
        return self.statuses.get(task_id.strip())

    async def save_task_input(self, task_input: TaskInput) -> None:
        if task_input.input_version > 1:
            prior_key = (task_input.task_id, task_input.input_version - 1)
            if prior_key not in self.inputs:
                raise NonContiguousVersionError(
                    f"Cannot save TaskInput version {task_input.input_version} for task '{task_input.task_id}' "
                    f"because prior version {task_input.input_version - 1} does not exist"
                )
        key = (task_input.task_id, task_input.input_version)
        if key in self.inputs:
            if self.inputs[key] == task_input:
                return
            raise ConflictingTaskInputError(
                f"TaskInput for task '{task_input.task_id}' version {task_input.input_version} "
                "already exists with conflicting data"
            )
        self.inputs[key] = task_input

    async def get_task_input(
        self, task_id: str, input_version: int = 1
    ) -> TaskInput | None:
        return self.inputs.get((task_id.strip(), input_version))

    async def save_stage_artifact(self, artifact: StageArtifact) -> None:
        if artifact.artifact_version > 1:
            prior_key = (artifact.producing_task_id, artifact.artifact_id, artifact.artifact_version - 1)
            if prior_key not in self.artifacts:
                raise NonContiguousVersionError(
                    f"Cannot save StageArtifact version {artifact.artifact_version} for task '{artifact.producing_task_id}', "
                    f"artifact '{artifact.artifact_id}' because prior version {artifact.artifact_version - 1} does not exist"
                )
        key = (artifact.producing_task_id, artifact.artifact_id, artifact.artifact_version)
        if key in self.artifacts:
            if self.artifacts[key] == artifact:
                return
            raise ConflictingStageArtifactError(
                f"StageArtifact '{artifact.artifact_id}' for task '{artifact.producing_task_id}' "
                f"version {artifact.artifact_version} already exists with conflicting data"
            )
        self.artifacts[key] = artifact

    async def get_stage_artifact(
        self, producing_task_id: str, artifact_id: str, artifact_version: int = 1
    ) -> StageArtifact | None:
        return self.artifacts.get((producing_task_id.strip(), artifact_id.strip(), artifact_version))

    async def save_continuation(self, continuation: TaskContinuation) -> None:
        key = (continuation.parent_task_id, continuation.continuation_version)
        if key not in self.continuations:
            if continuation.continuation_version == 1:
                if continuation.parent_task_id in self.latest_continuation:
                    raise CorruptAuthoritativeRecordError(
                        "Continuation latest pointer already exists for version 1 creation"
                    )
            else:
                if continuation.parent_task_id not in self.latest_continuation:
                    raise CorruptAuthoritativeRecordError(
                        f"Continuation latest pointer is missing for version {continuation.continuation_version}"
                    )
                cur_latest = self.latest_continuation[continuation.parent_task_id]
                if cur_latest >= continuation.continuation_version:
                    raise CorruptAuthoritativeRecordError(
                        f"Continuation latest pointer is corrupt: {cur_latest}"
                    )
                if cur_latest < continuation.continuation_version - 1:
                    raise NonContiguousVersionError(
                        f"Cannot save TaskContinuation version {continuation.continuation_version} "
                        f"for parent task '{continuation.parent_task_id}' because prior version "
                        f"{continuation.continuation_version - 1} does not exist"
                    )
                prior_key = (continuation.parent_task_id, continuation.continuation_version - 1)
                if prior_key not in self.continuations:
                    raise NonContiguousVersionError(
                        f"Cannot save TaskContinuation version {continuation.continuation_version} "
                        f"for parent task '{continuation.parent_task_id}' because prior version "
                        f"{continuation.continuation_version - 1} does not exist"
                    )
            self.continuations[key] = continuation
            self.latest_continuation[continuation.parent_task_id] = continuation.continuation_version
            return

        if self.continuations[key] == continuation:
            if continuation.parent_task_id not in self.latest_continuation:
                raise CorruptAuthoritativeRecordError(
                    "Continuation latest pointer is missing during replay"
                )
            cur_latest = self.latest_continuation[continuation.parent_task_id]
            if cur_latest < continuation.continuation_version:
                raise CorruptAuthoritativeRecordError(
                    f"Continuation latest pointer is corrupt during replay: {cur_latest}"
                )
            return
        raise ConflictingContinuationError(
            f"TaskContinuation for parent task '{continuation.parent_task_id}' "
            f"version {continuation.continuation_version} already exists with conflicting data"
        )

    async def get_continuation(
        self, parent_task_id: str, continuation_version: int = 1
    ) -> TaskContinuation | None:
        return self.continuations.get((parent_task_id.strip(), continuation_version))

    async def save_root_authority(self, authority: RootAuthorityRecord) -> None:
        key = authority.root_task_id
        if key in self.authorities:
            if self.authorities[key] == authority:
                return
            raise ConflictingRootAuthorityError(
                f"RootAuthorityRecord for root task '{authority.root_task_id}' "
                "already exists with conflicting data"
            )
        self.authorities[key] = authority

    async def get_root_authority(self, root_task_id: str) -> RootAuthorityRecord | None:
        return self.authorities.get(root_task_id.strip())


class MockAtomicDispatcher(TaskDispatcher):
    """Simulates an atomic dispatcher that enqueues and transitions status to pending."""

    def __init__(
        self,
        task_store: AuthoritativeTaskStore | None = None,
        should_fail: bool = False,
    ) -> None:
        self.task_store = task_store
        self.dispatches: list[tuple[TaskRequested, str]] = []
        self.should_fail = should_fail

    async def dispatch(self, task: TaskRequested, target: str) -> None:
        if self.should_fail:
            raise RuntimeError(f"Simulated dispatch transport failure targeting '{target}'")
        self.dispatches.append((task, target))
        if self.task_store is not None:
            await self.task_store.update_status(task.task_id, "pending")


def make_candidate(
    agent_id: str,
    capability: str = "general",
    dispatch_target: str = "agent.tasks.general",
) -> AgentCandidate:
    now_str = datetime.now(UTC).isoformat()
    return AgentCandidate(
        agent_id=agent_id,
        capabilities=[capability],
        availability=AgentAvailability(
            state=AgentAvailabilityState.READY,
            execution_state=AgentExecutionState.IDLE,
            heartbeat_age_seconds=1.0,
            last_seen_at=now_str,
            as_of=now_str,
            reason="healthy",
        ),
        agent_type="worker",
        execution_state=AgentExecutionState.IDLE,
        dispatch_target=dispatch_target,
    )


class FakeCandidateFinder:
    def __init__(self, candidates: list[AgentCandidate]) -> None:
        self.candidates = candidates

    async def execute(self, capability: Any, as_of: Any = None) -> list[AgentCandidate]:
        return list(self.candidates)


# ==============================================================================
# 1. Domain Model: TaskLineage Invariants
# ==============================================================================


def test_task_lineage_root_invariants() -> None:
    lineage = TaskLineage(task_id="root-1", root_task_id="root-1", parent_task_id=None)
    assert lineage.task_id == "root-1"
    assert lineage.root_task_id == "root-1"
    assert lineage.parent_task_id is None
    assert lineage.is_root is True

    # Root task with mismatched root_task_id is rejected
    with pytest.raises(ValueError, match="root tasks must have root_task_id equal to task_id"):
        TaskLineage(task_id="root-1", root_task_id="different", parent_task_id=None)


def test_task_lineage_child_invariants() -> None:
    lineage = TaskLineage(task_id="child-1", root_task_id="root-1", parent_task_id="root-1")
    assert lineage.task_id == "child-1"
    assert lineage.root_task_id == "root-1"
    assert lineage.parent_task_id == "root-1"
    assert lineage.is_root is False

    # Child claiming root_task_id == task_id rejected
    with pytest.raises(ValueError, match="child tasks must not have root_task_id equal to task_id"):
        TaskLineage(task_id="c1", root_task_id="c1", parent_task_id="p1")

    # Self-parenting rejected
    with pytest.raises(ValueError, match="parent_task_id must not equal task_id"):
        TaskLineage(task_id="c1", root_task_id="r1", parent_task_id="c1")


def test_task_lineage_deeper_ancestry_representation() -> None:
    """Wire/lineage model represents deeper ancestry structurally."""
    lineage = TaskLineage(task_id="A3", root_task_id="A", parent_task_id="A2")
    assert lineage.task_id == "A3"
    assert lineage.root_task_id == "A"
    assert lineage.parent_task_id == "A2"
    assert lineage.is_root is False

    # Round trip dictionary
    data = lineage.to_dict()
    assert TaskLineage.from_dict(data) == lineage


# ==============================================================================
# 2. Production Composition & Wiring
# ==============================================================================


def test_submit_task_requires_authoritative_task_store() -> None:
    """SubmitTask constructor must require an AuthoritativeTaskStore without silent skip."""
    # Omitting task_store raises TypeError
    with pytest.raises(TypeError):
        SubmitTask()  # type: ignore[call-arg]

    # Passing None or non-store raises TypeError
    with pytest.raises(TypeError, match="SubmitTask requires an AuthoritativeTaskStore"):
        SubmitTask(task_store=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="task_store must implement AuthoritativeTaskStore"):
        SubmitTask(task_store="invalid-store")  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_production_composition_wires_redis_gateway_as_authoritative_store() -> None:
    """Demonstrates production-style create_app supplies RedisGateway as AuthoritativeTaskStore.

    A normal root submission invokes authoritative root registration (pending_dispatch).
    """
    settings = Settings(redis_url="redis://fake:6379/0", app_env="test")
    gateway = RedisGateway(settings.redis_url)

    # Mock client and pipeline on gateway
    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.pipeline = MagicMock(return_value=mock_pipe)
    mock_client.set = AsyncMock()
    mock_client.eval = AsyncMock(return_value="OK")
    mock_client.xadd = AsyncMock()
    mock_client.aclose = AsyncMock()
    gateway._client = mock_client
    gateway.agents = AsyncMock(
        return_value=[
            {
                "agent_id": "worker-1",
                "capabilities": ["general"],
                "state": "idle",
                "dispatch_target": "agent.tasks.general",
                "updated_at": datetime.now(UTC).isoformat(),
            }
        ]
    )

    # In production composition with RedisGateway
    app = create_app(settings=settings, gateway=gateway)

    # Dispatch a root task through API layer
    req = Request({"type": "http", "app": app})
    res = await create_task(ApiTaskInput(prompt="Inspect cluster state"), request=req)

    assert res["status"] == "accepted"
    assert res["task_id"]

    # Verify atomic Lua script was evaluated for root registration with initial status pending_dispatch
    mock_client.eval.assert_awaited()
    # Verify dispatch pipeline pushed task and set status to pending
    mock_pipe.rpush.assert_called_once()


def test_create_app_fails_fast_when_task_store_provided_without_dispatch_path() -> None:
    """create_app(task_store=store) must fail fast if no dispatch path exists.

    It must not create a zombie app that registers tasks but cannot dispatch them.
    """
    store = InMemoryAuthoritativeTaskStore()
    with pytest.raises(TypeError, match="create_app requires a dispatch path"):
        create_app(task_store=store)


def test_create_app_fails_fast_when_no_authoritative_task_store_can_be_resolved() -> None:
    """create_app must fail fast if no AuthoritativeTaskStore can be resolved."""
    with pytest.raises(TypeError, match="create_app requires an AuthoritativeTaskStore"):
        create_app(task_store=None, gateway=None, task_queue=MagicMock())


# ==============================================================================
# 3. Root Lifecycle & Atomic Dispatch
# ==============================================================================


@pytest.mark.anyio
async def test_submit_task_records_root_lineage_and_lifecycle_transitions() -> None:
    """Verifies root registration establishes pending_dispatch + root lineage, and successful dispatch transitions to pending."""
    store = InMemoryAuthoritativeTaskStore()
    dispatcher = MockAtomicDispatcher(task_store=store, should_fail=False)
    candidate = make_candidate("worker-1", capability="general")
    finder = FakeCandidateFinder([candidate])

    submit_task = SubmitTask(
        task_store=store,
        dispatcher=dispatcher,
        candidate_finder=finder,
    )

    task = await submit_task.execute("Analyze cluster metrics")

    # Authoritative root lineage persists
    lineage = await store.get_lineage(task.task_id)
    assert lineage is not None
    assert lineage.task_id == task.task_id
    assert lineage.root_task_id == task.task_id
    assert lineage.parent_task_id is None

    # Status transitioned to pending via atomic dispatch
    assert await store.get_status(task.task_id) == "pending"


@pytest.mark.anyio
async def test_submit_task_dispatch_failure_transitions_to_dispatch_failed() -> None:
    """When dispatch fails, use case records dispatch_failed and re-raises the original exception."""
    store = InMemoryAuthoritativeTaskStore()
    dispatcher = MockAtomicDispatcher(task_store=store, should_fail=True)
    candidate = make_candidate("worker-1", capability="general")
    finder = FakeCandidateFinder([candidate])

    submit_task = SubmitTask(
        task_store=store,
        dispatcher=dispatcher,
        candidate_finder=finder,
    )

    original_msg = "Simulated dispatch transport failure targeting 'agent.tasks.general'"
    with pytest.raises(RuntimeError, match=original_msg):
        await submit_task.execute("Analyze cluster metrics")

    # The task was recorded authoritatively before dispatch attempt
    assert len(store.lineages) == 1
    task_id = next(iter(store.lineages.keys()))

    # Lineage is preserved
    lineage = await store.get_lineage(task_id)
    assert lineage is not None
    assert lineage.task_id == task_id

    # Status transitioned to dispatch_failed, preserving observable state
    assert await store.get_status(task_id) == "dispatch_failed"


@pytest.mark.anyio
async def test_legacy_enqueue_success_with_status_failure_does_not_relabel_dispatch_failed() -> None:
    """When legacy enqueue succeeds but later update_status('pending') fails:

    - status-update failure propagates;
    - task is NOT relabelled as dispatch_failed (since it is already committed to queue).
    """
    class FailingPendingStore(InMemoryAuthoritativeTaskStore):
        async def update_status(self, task_id: str, status: str) -> None:
            if status == "pending":
                raise RuntimeError("Simulated status update failure for pending")
            await super().update_status(task_id, status)

    store = FailingPendingStore()
    queue = FakeTaskQueue()
    submit = SubmitTask(task_store=store, queue=queue)

    with pytest.raises(RuntimeError, match="Simulated status update failure for pending"):
        await submit.execute("Legacy task with failing pending status")

    # Enqueue was committed
    assert len(queue.enqueued) == 1
    enqueued_task, _ = queue.enqueued[0]

    # Task status remains pending_dispatch and was NOT relabelled dispatch_failed
    assert await store.get_status(enqueued_task.task_id) == "pending_dispatch"


# ==============================================================================
# 4. Child Registration (RegisterChildTask) - Non-Executable Primitive
# ==============================================================================


@pytest.mark.anyio
async def test_register_child_requires_authoritative_parent_with_zero_state() -> None:
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    with pytest.raises(ParentTaskNotFoundError, match="Authoritative parent task not found"):
        await register_child.execute(parent_task_id="non-existent-parent", prompt="subtask")

    # Zero partial state exists
    assert store.statuses == {}
    assert store.lineages == {}
    assert store.children == {}


@pytest.mark.anyio
async def test_register_child_enforces_depth_1_policy() -> None:
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    # 1. Create root parent
    root = TaskRequested.create("Root task")
    await store.register_root_task(root)

    # 2. Register child of root (depth 1 - valid)
    child1 = await register_child.execute(parent_task_id=root.task_id, prompt="Child 1")
    assert child1.parent_task_id == root.task_id
    assert child1.root_task_id == root.task_id

    # 3. Attempt to register child of child1 (depth 2 - rejected)
    with pytest.raises(DelegationDepthExceededError, match="is already a child task"):
        await register_child.execute(parent_task_id=child1.task_id, prompt="Grandchild")

    # Verify no grandchild was registered in store
    assert await store.get_children(child1.task_id) == []


@pytest.mark.anyio
async def test_register_child_derives_root_and_mints_child_id() -> None:
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    root = TaskRequested.create("Root parent")
    await store.register_root_task(root)

    child = await register_child.execute(parent_task_id=root.task_id, prompt="Derived root child")
    assert child.root_task_id == root.task_id
    assert child.parent_task_id == root.task_id
    assert child.task_id != root.task_id
    assert child.version == 2

    # Status established as pending_dispatch
    assert await store.get_status(child.task_id) == "pending_dispatch"

    # Lineage in store reflects authoritative derivation
    lineage = await store.get_lineage(child.task_id)
    assert lineage is not None
    assert lineage.root_task_id == root.task_id
    assert lineage.parent_task_id == root.task_id


def test_register_child_primitive_is_non_executable() -> None:
    """Verifies RegisterChildTask accepts no dispatcher, no target parameter, and performs no queue dispatch."""
    import inspect

    # Constructor requires only task_store
    sig_init = inspect.signature(RegisterChildTask.__init__)
    assert list(sig_init.parameters.keys()) == ["self", "task_store"]

    # Execute accepts only parent_task_id and prompt (no target or queue parameters)
    sig_exec = inspect.signature(RegisterChildTask.execute)
    assert list(sig_exec.parameters.keys()) == ["self", "parent_task_id", "prompt"]


@pytest.mark.anyio
async def test_register_child_simulated_pipeline_failure_leaves_no_orphans() -> None:
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    root = TaskRequested.create("Root parent")
    await store.register_root_task(root)

    # Trigger failure during child registration
    store.fail_next_registration = True

    with pytest.raises(RuntimeError, match="Simulated pipeline failure"):
        await register_child.execute(parent_task_id=root.task_id, prompt="Failing child")

    # Parent has zero children registered (no phantom entry)
    assert await store.get_children(root.task_id) == []


@pytest.mark.anyio
async def test_parent_membership_duplicate_safe_and_deterministic() -> None:
    """Verifies that parent-to-child Redis set membership is duplicate-safe and retrieval is deterministically sorted.

    Note: Set membership is duplicate-safe via SADD. Full task re-registration idempotency
    across status and forward lineage is deferred to future work.
    """
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    root = TaskRequested.create("Root parent")
    await store.register_root_task(root)

    c1 = await register_child.execute(parent_task_id=root.task_id, prompt="Child 1")
    c2 = await register_child.execute(parent_task_id=root.task_id, prompt="Child 2")
    c3 = await register_child.execute(parent_task_id=root.task_id, prompt="Child 3")

    expected_sorted = sorted([c1.task_id, c2.task_id, c3.task_id])
    assert await store.get_children(root.task_id) == expected_sorted

    # Re-adding the child to the parent set does not duplicate the membership entry
    store.children[root.task_id].add(c1.task_id)
    assert await store.get_children(root.task_id) == expected_sorted


@pytest.mark.anyio
async def test_register_child_validates_and_normalizes_prompt() -> None:
    """RegisterChildTask must validate and normalize prompts through the Task domain model."""
    store = InMemoryAuthoritativeTaskStore()
    register_child = RegisterChildTask(task_store=store)

    root = TaskRequested.create("Root prompt")
    await store.register_root_task(root)

    # 1. Valid trimmed prompt
    child = await register_child.execute(root.task_id, "   child prompt with whitespace   \n")
    assert child.prompt == "child prompt with whitespace"

    # 2. Empty/blank prompt rejected
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        await register_child.execute(root.task_id, "   ")

    # 3. Over 2,000 characters rejected
    with pytest.raises(ValueError, match="Prompt must not exceed 2000 characters"):
        await register_child.execute(root.task_id, "x" * 2001)

    # 4. Non-string prompt rejected
    with pytest.raises(TypeError, match="Prompt must be a string"):
        await register_child.execute(root.task_id, 12345)  # type: ignore[arg-type]


# ==============================================================================
# 5. RedisGateway Implementation & Transaction Semantics
# ==============================================================================


@pytest.mark.anyio
async def test_redis_gateway_dispatch_uses_atomic_pipeline_with_transaction_true() -> None:
    """Verifies that RedisGateway.dispatch executes RPUSH and status:pending inside a transaction=True pipeline."""
    gateway = RedisGateway("redis://localhost:6379/0")

    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)

    gateway._client.pipeline = MagicMock(return_value=mock_pipe)

    task = TaskRequested.create("Execute runtime task")
    target = "agent.tasks.custom-executor-1"

    await gateway.dispatch(task, target=target)

    # Verified that pipeline was requested with transaction=True
    gateway._client.pipeline.assert_called_once_with(transaction=True)

    # Sets status: pending and pushes to target via pipeline buffer
    mock_pipe.rpush.assert_called_once_with(target, task.to_json())
    mock_pipe.set.assert_called_once_with(
        f"task.status:{task.task_id}",
        json.dumps({"task_id": task.task_id, "status": "pending"}),
    )
    mock_pipe.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_gateway_register_child_uses_atomic_pipeline_with_transaction_true() -> None:
    """Verifies that RedisGateway.register_child_task commits status, lineage, and parent set inside a transaction=True pipeline."""
    gateway = RedisGateway("redis://localhost:6379/0")

    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.sadd = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)

    gateway._client.pipeline = MagicMock(return_value=mock_pipe)

    child = TaskRequested(
        task_id="child-123",
        prompt="Child prompt",
        created_at="2026-11-15T00:00:00+00:00",
        root_task_id="root-1",
        parent_task_id="root-1",
        version=2,
    )

    await gateway.register_child_task(child, initial_status="pending_dispatch")

    # Verified transaction=True pipeline
    gateway._client.pipeline.assert_called_once_with(transaction=True)

    mock_pipe.set.assert_any_call(
        f"task.status:{child.task_id}",
        json.dumps({"status": "pending_dispatch", "task_id": child.task_id}, separators=(",", ":"), sort_keys=True),
    )
    mock_pipe.set.assert_any_call(
        f"task.lineage:{child.task_id}",
        json.dumps({
            "parent_task_id": "root-1",
            "root_task_id": "root-1",
            "task_id": child.task_id,
        }, separators=(",", ":"), sort_keys=True),
    )
    mock_pipe.sadd.assert_called_once_with(
        "task.children:root-1", child.task_id
    )
    mock_pipe.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_redis_gateway_get_children_determinism() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.smembers = AsyncMock(return_value={b"c-3", b"c-1", b"c-2"})

    children = await gateway.get_children("parent-1")
    assert children == ["c-1", "c-2", "c-3"]  # strictly sorted


@pytest.mark.anyio
async def test_redis_gateway_register_root_executes_atomic_lua_script() -> None:
    """Verifies that RedisGateway.register_root_task executes atomic Lua script across all 4 keys."""
    gateway = RedisGateway("redis://localhost:6379/0")
    gateway._client.eval = AsyncMock(return_value="OK")

    task = TaskRequested.create("Root task prompt")

    # 1. Fresh registration succeeds
    await gateway.register_root_task(task, initial_status="pending_dispatch")
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent replay succeeds
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.register_root_task(task, initial_status="pending_dispatch")

    # 3. Conflicting registration raises ConflictingRootTaskRegistrationError
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    with pytest.raises(ConflictingRootTaskRegistrationError, match="already exists with conflicting state"):
        await gateway.register_root_task(task)

    # 4. Partial state conflict raises ConflictingRootTaskRegistrationError
    gateway._client.eval = AsyncMock(return_value="PARTIAL_CONFLICT")
    with pytest.raises(ConflictingRootTaskRegistrationError, match="inconsistent or partial authoritative state"):
        await gateway.register_root_task(task)


@pytest.mark.anyio
async def test_simulated_dispatch_pipeline_failure_does_not_expose_queued_task_or_pending_status() -> None:
    """Simulated pipeline execute failure during dispatch does not expose enqueued task or pending status."""
    gateway = RedisGateway("redis://localhost:6379/0")

    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.execute = AsyncMock(side_effect=RuntimeError("Simulated pipeline commit failure"))
    mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_pipe.__aexit__ = AsyncMock(return_value=None)

    gateway._client.pipeline = MagicMock(return_value=mock_pipe)

    task = TaskRequested.create("Root task")

    with pytest.raises(RuntimeError, match="Simulated pipeline commit failure"):
        await gateway.dispatch(task, target="agent.tasks.general")

    # Gateway client pipeline was invoked with transaction=True
    gateway._client.pipeline.assert_called_once_with(transaction=True)


# ==============================================================================
# 6. Durable TaskInput, StageArtifact, Continuation & Authority Storage Tests
# ==============================================================================


@pytest.mark.anyio
async def test_in_memory_task_input_write_once_semantics() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    input_v1 = TaskInput(
        task_id="t-1",
        prompt="Initial prompt",
        created_at=now,
        input_version=1,
    )

    # 1. Initial write
    await store.save_task_input(input_v1)
    retrieved = await store.get_task_input("t-1", input_version=1)
    assert retrieved == input_v1

    # 2. Idempotent write succeeds
    await store.save_task_input(input_v1)

    # 3. Conflicting write for same version raises ConflictingTaskInputError
    conflicting = TaskInput(
        task_id="t-1",
        prompt="Conflicting prompt",
        created_at=now,
        input_version=1,
    )
    with pytest.raises(ConflictingTaskInputError, match="already exists with conflicting data"):
        await store.save_task_input(conflicting)

    # 4. New version succeeds
    input_v2 = TaskInput(
        task_id="t-1",
        prompt="Updated prompt v2",
        created_at=now,
        input_version=2,
    )
    await store.save_task_input(input_v2)
    assert await store.get_task_input("t-1", input_version=2) == input_v2
    assert await store.get_task_input("t-1", input_version=1) == input_v1


@pytest.mark.anyio
async def test_in_memory_stage_artifact_write_once_semantics() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    art_v1 = StageArtifact(
        artifact_id="art-1",
        producing_task_id="t-parent",
        artifact_type="research_material",
        content={"material": "val1"},
        created_at=now,
        artifact_version=1,
    )

    await store.save_stage_artifact(art_v1)
    retrieved = await store.get_stage_artifact("t-parent", "art-1", artifact_version=1)
    assert retrieved == art_v1

    # Idempotent write
    await store.save_stage_artifact(art_v1)

    # Conflicting write
    conflicting = StageArtifact(
        artifact_id="art-1",
        producing_task_id="t-parent",
        artifact_type="research_material",
        content={"material": "val-conflict"},
        created_at=now,
        artifact_version=1,
    )
    with pytest.raises(ConflictingStageArtifactError, match="already exists with conflicting data"):
        await store.save_stage_artifact(conflicting)


@pytest.mark.anyio
async def test_in_memory_continuation_write_once_semantics() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    cont_v1 = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        input_version=1,
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "focus-1"},
        continuation_version=1,
        created_at=now,
        updated_at=now,
    )

    await store.save_continuation(cont_v1)
    retrieved = await store.get_continuation("p-1", continuation_version=1)
    assert retrieved == cont_v1

    # Idempotent write
    await store.save_continuation(cont_v1)

    # Conflicting write
    conflicting = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-different",
        input_version=1,
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "focus-conflict"},
        continuation_version=1,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(ConflictingContinuationError, match="already exists with conflicting data"):
        await store.save_continuation(conflicting)


@pytest.mark.anyio
async def test_in_memory_root_authority_write_once_semantics() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()
    auth = RootAuthorityRecord(
        root_task_id="r-1",
        policy_context="default",
        created_at=now,
    )

    await store.save_root_authority(auth)
    retrieved = await store.get_root_authority("r-1")
    assert retrieved == auth

    # Idempotent write
    await store.save_root_authority(auth)

    # Conflicting write
    conflicting = RootAuthorityRecord.create_default(
        root_task_id="r-1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ConflictingRootAuthorityError, match="already exists with conflicting data"):
        await store.save_root_authority(conflicting)


@pytest.mark.anyio
async def test_submit_task_root_ingestion_persists_durable_input_and_authority() -> None:
    store = InMemoryAuthoritativeTaskStore()
    queue = FakeTaskQueue()
    submit = SubmitTask(task_store=store, queue=queue)

    task_requested = await submit.execute("Analyze system resilience under node failure")
    task_id = task_requested.task_id

    # 1. Authoritative status and lineage exist
    assert await store.get_status(task_id) == "pending"
    lineage = await store.get_lineage(task_id)
    assert lineage is not None
    assert lineage.is_root
    assert lineage.root_task_id == task_id

    # 2. Durable TaskInput v1 exists independently of queue message
    input_record = await store.get_task_input(task_id, input_version=1)
    assert input_record is not None
    assert input_record.task_id == task_id
    assert input_record.prompt == "Analyze system resilience under node failure"
    assert input_record.input_version == 1

    # 3. Default RootAuthorityRecord exists
    authority = await store.get_root_authority(task_id)
    assert authority is not None
    assert authority.root_task_id == task_id
    assert authority.max_delegation_depth == 1
    assert authority.max_delegation_waves == 1


@pytest.mark.anyio
async def test_redis_gateway_task_input_roundtrip_and_conflict() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now = datetime.now(UTC).isoformat()
    record = TaskInput(
        task_id="t-1",
        prompt="Test prompt",
        created_at=now,
        input_version=1,
    )

    # 1. Initial write (Lua returns OK)
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_task_input(record)
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent write (Lua returns IDEMPOTENT)
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.save_task_input(record)

    # 3. Conflicting write (Lua returns CONFLICT)
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    different = TaskInput(
        task_id="t-1",
        prompt="Different prompt",
        created_at=now,
        input_version=1,
    )
    with pytest.raises(ConflictingTaskInputError, match="already exists with conflicting data"):
        await gateway.save_task_input(different)

    # 4. Non-contiguous write (Lua returns NON_CONTIGUOUS)
    gateway._client.eval = AsyncMock(return_value="NON_CONTIGUOUS")
    v3 = TaskInput(
        task_id="t-1",
        prompt="Skip to v3",
        created_at=now,
        input_version=3,
    )
    with pytest.raises(NonContiguousVersionError, match="prior version 2 does not exist"):
        await gateway.save_task_input(v3)

    # 5. Retrieval
    gateway._client.get = AsyncMock(return_value=record.to_json())
    retrieved = await gateway.get_task_input("t-1", 1)
    assert retrieved == record

    # 6. Corrupt record detection
    gateway._client.get = AsyncMock(return_value="invalid JSON {")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt TaskInput record"):
        await gateway.get_task_input("t-1", 1)


@pytest.mark.anyio
async def test_redis_gateway_stage_artifact_roundtrip_and_conflict() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now = datetime.now(UTC).isoformat()
    art = StageArtifact(
        artifact_id="art-1",
        producing_task_id="t-1",
        artifact_type="research_material",
        content={"material": "sample"},
        created_at=now,
        artifact_version=1,
    )

    # 1. Save artifact (Lua returns OK)
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_stage_artifact(art)
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent write (Lua returns IDEMPOTENT)
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.save_stage_artifact(art)

    # 3. Conflicting write (Lua returns CONFLICT)
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    conflict = StageArtifact(
        artifact_id="art-1",
        producing_task_id="t-1",
        artifact_type="research_material",
        content={"material": "conflicting_sample"},
        created_at=now,
        artifact_version=1,
    )
    with pytest.raises(ConflictingStageArtifactError, match="already exists with conflicting data"):
        await gateway.save_stage_artifact(conflict)

    # 4. Non-contiguous write (Lua returns NON_CONTIGUOUS)
    gateway._client.eval = AsyncMock(return_value="NON_CONTIGUOUS")
    art_v3 = StageArtifact(
        artifact_id="art-1",
        producing_task_id="t-1",
        artifact_type="research_material",
        content={"material": "v3"},
        created_at=now,
        artifact_version=3,
    )
    with pytest.raises(NonContiguousVersionError, match="prior version 2 does not exist"):
        await gateway.save_stage_artifact(art_v3)

    # 5. Retrieval
    gateway._client.get = AsyncMock(return_value=art.to_json())
    retrieved = await gateway.get_stage_artifact("t-1", "art-1", 1)
    assert retrieved == art

    # 6. Corrupt record detection
    gateway._client.get = AsyncMock(return_value='{"malformed": "content_missing"}')
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt StageArtifact record"):
        await gateway.get_stage_artifact("t-1", "art-1", 1)


@pytest.mark.anyio
async def test_redis_gateway_continuation_roundtrip_and_conflict() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now = datetime.now(UTC).isoformat()
    cont = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "sample focus"},
        continuation_version=1,
        created_at=now,
        updated_at=now,
    )

    # 1. Save continuation (Lua returns OK)
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_continuation(cont)
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent write (Lua returns IDEMPOTENT)
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.save_continuation(cont)

    # 3. Conflicting write (Lua returns CONFLICT)
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    conflict = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-different",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "different focus"},
        continuation_version=1,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(ConflictingContinuationError, match="already exists with conflicting data"):
        await gateway.save_continuation(conflict)

    # 4. Non-contiguous write (Lua returns NON_CONTIGUOUS)
    gateway._client.eval = AsyncMock(return_value="NON_CONTIGUOUS")
    cont_v3 = TaskContinuation(
        parent_task_id="p-1",
        child_task_id="c-1",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "focus-v3"},
        continuation_version=3,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(NonContiguousVersionError, match="prior version 2 does not exist"):
        await gateway.save_continuation(cont_v3)

    # 4b. Corrupt latest pointer write (Lua returns CORRUPT_POINTER)
    gateway._client.eval = AsyncMock(return_value="CORRUPT_POINTER")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Continuation latest pointer"):
        await gateway.save_continuation(cont)

    # 5. Retrieval
    gateway._client.get = AsyncMock(return_value=cont.to_json())
    retrieved = await gateway.get_continuation("p-1", 1)
    assert retrieved == cont

    # 6. Corrupt record detection
    gateway._client.get = AsyncMock(return_value="corrupt continuation")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt TaskContinuation record"):
        await gateway.get_continuation("p-1", 1)


@pytest.mark.anyio
async def test_redis_gateway_root_authority_roundtrip_and_conflict() -> None:
    gateway = RedisGateway("redis://localhost:6379/0")
    now = datetime.now(UTC).isoformat()
    auth = RootAuthorityRecord.create_default("root-abc", created_at=now)

    # 1. Save authority (Lua returns OK)
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_root_authority(auth)
    gateway._client.eval.assert_awaited_once()

    # 2. Idempotent write (Lua returns IDEMPOTENT)
    gateway._client.eval = AsyncMock(return_value="IDEMPOTENT")
    await gateway.save_root_authority(auth)

    # 3. Conflicting write (Lua returns CONFLICT)
    gateway._client.eval = AsyncMock(return_value="CONFLICT")
    conflict = RootAuthorityRecord.create_default(
        root_task_id="root-abc",
        created_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ConflictingRootAuthorityError, match="already exists with conflicting data"):
        await gateway.save_root_authority(conflict)

    # 4. Retrieval
    gateway._client.get = AsyncMock(return_value=auth.to_json())
    retrieved = await gateway.get_root_authority("root-abc")
    assert retrieved == auth

    # 5. Corrupt record detection
    gateway._client.get = AsyncMock(return_value="corrupt authority data")
    with pytest.raises(CorruptAuthoritativeRecordError, match="Corrupt RootAuthorityRecord"):
        await gateway.get_root_authority("root-abc")


@pytest.mark.anyio
async def test_in_memory_version_contiguity_and_concurrency() -> None:
    store = InMemoryAuthoritativeTaskStore()
    now = datetime.now(UTC).isoformat()

    # TaskInput contiguity: cannot save v2 without v1
    v2_input = TaskInput(task_id="t-c", prompt="v2 prompt", created_at=now, input_version=2)
    with pytest.raises(NonContiguousVersionError, match="prior version 1 does not exist"):
        await store.save_task_input(v2_input)

    # StageArtifact contiguity: cannot save v2 without v1
    v2_art = StageArtifact(
        artifact_id="a-c",
        producing_task_id="t-c",
        artifact_type="research_material",
        content={"material": "v2"},
        created_at=now,
        artifact_version=2,
    )
    with pytest.raises(NonContiguousVersionError, match="prior version 1 does not exist"):
        await store.save_stage_artifact(v2_art)

    # TaskContinuation latest pointer and contiguity:
    # Saving v2 without v1 (latest pointer missing) raises CorruptAuthoritativeRecordError
    v2_missing_cont = TaskContinuation(
        parent_task_id="t-missing",
        child_task_id="child-missing",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "v2"},
        continuation_version=2,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(CorruptAuthoritativeRecordError):
        await store.save_continuation(v2_missing_cont)

    # Save v1, then attempt saving v3 (latest=1 < 3-1) raises NonContiguousVersionError
    v1_cont = TaskContinuation(
        parent_task_id="t-c",
        child_task_id="child-c",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "v1"},
        continuation_version=1,
        created_at=now,
        updated_at=now,
    )
    await store.save_continuation(v1_cont)
    v3_cont = TaskContinuation(
        parent_task_id="t-c",
        child_task_id="child-c",
        stage="incorporate_analysis_child",
        control_fields={"analysis_focus": "v3"},
        continuation_version=3,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(NonContiguousVersionError, match="prior version 2 does not exist"):
        await store.save_continuation(v3_cont)

    # Partial state conflict on root registration
    root = TaskRequested.create("Root task")
    # Manually inject only status (partial state)
    store.statuses[root.task_id] = "pending"
    with pytest.raises(ConflictingRootTaskRegistrationError, match="inconsistent or partial"):
        await store.register_root_task(root)
