from typing import Any

import pytest
from agent_contracts import TaskRequested
from orchestrator.application.ports import AuthoritativeTaskStore
from orchestrator.application.queries import (
    GetAgentStates,
    GetRecentEvents,
    GetTaskResult,
)
from orchestrator.application.use_cases import SubmitTask
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.task import TaskLineage


class FakeTaskQueue(AuthoritativeTaskStore):
    def __init__(self) -> None:
        self.enqueued: list[tuple[TaskRequested, AgentCapability]] = []
        self.dispatches: list[tuple[TaskRequested, str]] = []
        self.statuses: dict[str, str] = {}
        self.lineages: dict[str, TaskLineage] = {}
        self.children: dict[str, set[str]] = {}
        self.fail_get_lineage: bool = False
        self.fail_get_children: bool = False

    async def register_root_task(
        self, task: TaskRequested, initial_status: str = "pending_dispatch"
    ) -> None:
        self.statuses[task.task_id] = initial_status
        self.lineages[task.task_id] = TaskLineage(
            task_id=task.task_id,
            root_task_id=task.root_task_id,
            parent_task_id=None,
        )

    async def register_child_task(
        self, task: TaskRequested, initial_status: str = "pending_dispatch"
    ) -> None:
        self.statuses[task.task_id] = initial_status
        self.lineages[task.task_id] = TaskLineage(
            task_id=task.task_id,
            root_task_id=task.root_task_id,
            parent_task_id=task.parent_task_id,
        )
        if task.parent_task_id:
            self.children.setdefault(task.parent_task_id, set()).add(task.task_id)

    async def get_lineage(self, task_id: str) -> TaskLineage | None:
        if self.fail_get_lineage:
            raise RuntimeError("Simulated get_lineage storage failure")
        return self.lineages.get(task_id.strip())

    async def get_children(self, parent_task_id: str) -> list[str]:
        if self.fail_get_children:
            raise RuntimeError("Simulated get_children storage failure")
        return sorted(self.children.get(parent_task_id.strip(), set()))

    async def update_status(self, task_id: str, status: str) -> None:
        self.statuses[task_id.strip()] = status.strip()

    async def get_status(self, task_id: str) -> str | None:
        return self.statuses.get(task_id.strip())

    async def enqueue(
        self,
        task: TaskRequested,
        capability: AgentCapability = AgentCapability.GENERAL,
        target: str | None = None,
    ) -> None:
        self.enqueued.append((task, capability))
        if target:
            self.dispatches.append((task, target))

    async def dispatch(self, task: TaskRequested, target: str) -> None:
        self.dispatches.append((task, target))
        capability = (
            AgentCapability.RESEARCH if "research" in target else AgentCapability.GENERAL
        )
        self.enqueued.append((task, capability))
        self.statuses[task.task_id] = "pending"


class FakeStateReader:
    def __init__(
        self,
        agents_data: list[dict[str, str]] | None = None,
        events_data: list[dict[str, str]] | None = None,
        routing_decisions: dict[str, dict[str, Any]] | None = None,
        task_results: dict[str, Any] | None = None,
        task_statuses: dict[str, Any] | None = None,
    ) -> None:
        self.agents_data = agents_data or []
        self.events_data = events_data or []
        self.routing_decisions = routing_decisions or {}
        self.task_results = task_results or {}
        self.task_statuses = task_statuses or {}
        self.last_events_limit: int | None = None

    async def agents(self) -> list[dict[str, str]]:
        return self.agents_data

    async def events(self, limit: int = 50) -> list[dict[str, str]]:
        self.last_events_limit = limit
        return self.events_data[:limit]

    async def routing_decision(self, task_id: str) -> dict[str, Any] | None:
        if task_id in self.routing_decisions:
            return self.routing_decisions[task_id]
        for event in self.events_data:
            if event.get("task_id") == task_id and event.get("routing_decision"):
                rd = event["routing_decision"]
                import json

                return json.loads(rd) if isinstance(rd, str) else rd
        return None

    async def task_inference_telemetry(self, task_id: str) -> dict[str, Any] | None:
        import json

        from agent_contracts import FallbackTransition, InferenceExecutionTelemetry
        from orchestrator.application.inference_history import (
            TaskEventObservation,
            project_task_inference_history,
        )

        clean_task_id = str(task_id).strip()
        if not clean_task_id:
            return None

        matching = [e for e in self.events_data if e.get("task_id") == clean_task_id]
        task_completed = clean_task_id in self.task_results
        task_exists_durable = (
            task_completed
            or (clean_task_id in self.routing_decisions)
            or (clean_task_id in self.task_statuses)
        )
        task_failed = False
        if clean_task_id in self.task_statuses:
            s = self.task_statuses[clean_task_id]
            if (isinstance(s, dict) and s.get("status") in ("error", "failed")) or s in (
                "error",
                "failed",
            ):
                task_failed = True

        observations: list[TaskEventObservation] = []
        for stream_order, e in enumerate(matching):
            event_id = str(e.get("event_id", f"1-{stream_order}"))
            state = e.get("state")
            is_terminal_lifecycle = state in ("idle", "completed")
            is_terminal_error = state == "error"

            parsed_telem: InferenceExecutionTelemetry | None = None
            raw_telem = e.get("inference_telemetry")
            is_malformed = False
            if raw_telem:
                try:
                    telem_data = json.loads(raw_telem) if isinstance(raw_telem, str) else raw_telem
                    if isinstance(telem_data, dict):
                        parsed_telem = InferenceExecutionTelemetry.from_dict(telem_data)
                    else:
                        is_malformed = True
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    is_malformed = True

            parsed_trans: FallbackTransition | None = None
            raw_trans = e.get("fallback_transition")
            if raw_trans:
                try:
                    trans_data = json.loads(raw_trans) if isinstance(raw_trans, str) else raw_trans
                    if isinstance(trans_data, dict):
                        parsed_trans = FallbackTransition.from_dict(trans_data)
                    else:
                        is_malformed = True
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    is_malformed = True

            observations.append(
                TaskEventObservation(
                    event_id=event_id,
                    task_id=clean_task_id,
                    stream_position=(
                        int(event_id.split("-")[0]) if event_id.split("-")[0].isdigit() else 0,
                        int(event_id.split("-")[1]) if len(event_id.split("-")) > 1 and event_id.split("-")[1].isdigit() else 0,
                    ),
                    stream_order=stream_order,
                    state=state,
                    inference_telemetry=parsed_telem,
                    fallback_transition=parsed_trans,
                    is_terminal_lifecycle=is_terminal_lifecycle,
                    is_terminal_error=is_terminal_error,
                    is_malformed=is_malformed,
                )
            )

        return project_task_inference_history(
            task_id=clean_task_id,
            observations=observations,
            task_exists_durable=task_exists_durable,
            task_completed=task_completed,
            task_failed=task_failed,
        )


class FakeResultReader:
    def __init__(
        self,
        results: dict[str, dict[str, Any]] | None = None,
        pending: set[str] | None = None,
    ) -> None:
        self.results = results or {}
        self.pending = pending or set()

    async def get_result(self, task_id: str) -> dict[str, Any] | None:
        if task_id in self.results:
            return self.results[task_id]
        if task_id in self.pending:
            return {"task_id": task_id, "status": "pending", "output": None}
        return None


@pytest.mark.anyio
async def test_submit_task_use_case() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)
    task = await use_case.execute("test task prompt")

    assert task.prompt == "test task prompt"
    assert len(queue.enqueued) == 1

    enqueued_task, capability = queue.enqueued[0]

    assert enqueued_task.prompt == "test task prompt"
    assert capability == AgentCapability.GENERAL

@pytest.mark.anyio
async def test_get_agent_states_query() -> None:
    expected = [{"agent_id": "worker-1", "state": "idle", "detail": "Waiting"}]
    reader = FakeStateReader(agents_data=expected)
    query = GetAgentStates(reader)

    result = await query.execute()
    assert result == expected


@pytest.mark.anyio
async def test_get_recent_events_bounds_limits() -> None:
    sample_events = [{"event_id": str(i), "detail": f"event {i}"} for i in range(150)]
    reader = FakeStateReader(events_data=sample_events)
    query = GetRecentEvents(reader)

    # Test default
    await query.execute()
    assert reader.last_events_limit == 50

    # Test lower bound (<= 0 maps to 1)
    await query.execute(limit=0)
    assert reader.last_events_limit == 1

    await query.execute(limit=-10)
    assert reader.last_events_limit == 1

    # Test upper bound (> 100 maps to 100)
    await query.execute(limit=250)
    assert reader.last_events_limit == 100

    # Test valid within-range limit
    await query.execute(limit=25)
    assert reader.last_events_limit == 25

@pytest.mark.anyio
async def test_submit_research_task_routes_to_research_capability() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)

    task = await use_case.execute(
        "Compare local and cloud LLMs",
        capability=AgentCapability.RESEARCH,
    )

    assert len(queue.enqueued) == 1

    enqueued_task, capability = queue.enqueued[0]

    assert enqueued_task == task
    assert capability == AgentCapability.RESEARCH


@pytest.mark.anyio
async def test_submit_task_automatic_routing_research() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)

    # Under Phase B routing policy, genuine research prompts require external retrieval/freshness/source evidence rather than weak verbs alone
    task = await use_case.execute("Compare current features and pricing of local and cloud LLMs")

    assert len(queue.enqueued) == 1
    enqueued_task, capability = queue.enqueued[0]
    assert enqueued_task == task
    assert capability == AgentCapability.RESEARCH


@pytest.mark.anyio
async def test_submit_task_automatic_routing_general() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)

    task = await use_case.execute("build agent")

    assert len(queue.enqueued) == 1
    enqueued_task, capability = queue.enqueued[0]
    assert enqueued_task == task
    assert capability == AgentCapability.GENERAL


@pytest.mark.anyio
async def test_submit_task_routing_v2_freshness_and_explanatory() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)

    # Freshness prompt automatically resolves to RESEARCH
    task_research = await use_case.execute(
        "What are the latest developments in battery technology?"
    )
    assert queue.enqueued[0] == (task_research, AgentCapability.RESEARCH)

    # General explanatory prompt automatically resolves to GENERAL
    task_general = await use_case.execute(
        "Explain the difference between containers and virtual machines."
    )
    assert queue.enqueued[1] == (task_general, AgentCapability.GENERAL)



@pytest.mark.anyio
async def test_submit_task_preserves_explicit_capability_override() -> None:
    queue = FakeTaskQueue()
    use_case = SubmitTask(task_store=queue, queue=queue)

    # Research prompt explicitly directed to GENERAL
    task1 = await use_case.execute(
        "Investigate system logs",
        capability=AgentCapability.GENERAL,
    )
    assert queue.enqueued[0] == (task1, AgentCapability.GENERAL)

    # General prompt explicitly directed to RESEARCH
    task2 = await use_case.execute(
        "build agent",
        capability=AgentCapability.RESEARCH,
    )
    assert queue.enqueued[1] == (task2, AgentCapability.RESEARCH)


class FakeCapabilityRouter:
    def __init__(self, capability_to_return: AgentCapability) -> None:
        self.capability_to_return = capability_to_return
        self.routed_prompts: list[str] = []

    def route(self, prompt: str) -> AgentCapability:
        self.routed_prompts.append(prompt)
        return self.capability_to_return


@pytest.mark.anyio
async def test_submit_task_uses_injected_router() -> None:
    queue = FakeTaskQueue()
    fake_router = FakeCapabilityRouter(AgentCapability.RESEARCH)
    use_case = SubmitTask(task_store=queue, queue=queue, router=fake_router)

    # Non-research prompt routed to RESEARCH via injected router
    task = await use_case.execute("simple task")

    assert fake_router.routed_prompts == ["simple task"]
    assert queue.enqueued[0] == (task, AgentCapability.RESEARCH)


@pytest.mark.anyio
async def test_get_task_result_query_completed() -> None:
    expected = {
        "task_id": "task-1",
        "status": "completed",
        "output": "research findings",
    }
    reader = FakeResultReader(results={"task-1": expected})
    query = GetTaskResult(reader)

    result = await query.execute("task-1")
    assert result == expected


@pytest.mark.anyio
async def test_get_task_result_query_pending() -> None:
    reader = FakeResultReader(pending={"task-pending-1"})
    query = GetTaskResult(reader)

    result = await query.execute("task-pending-1")
    assert result == {"task_id": "task-pending-1", "status": "pending", "output": None}


@pytest.mark.anyio
async def test_get_task_result_query_not_found() -> None:
    reader = FakeResultReader()
    query = GetTaskResult(reader)

    result = await query.execute("non-existent-task")
    assert result is None


@pytest.mark.anyio
async def test_get_task_inference_telemetry_query_success() -> None:
    from orchestrator.application.queries import GetTaskInferenceTelemetry

    expected_telem = {
        "provider_name": "cloudflare",
        "attempt_index": 0,
        "latency_ms": 94.3,
        "status": "succeeded",
        "model_name": "@cf/meta/llama-3.1-8b-instruct",
        "finish_reason": "stop",
    }
    reader = FakeStateReader(
        events_data=[
            {
                "task_id": "task-telemetry-1",
                "inference_telemetry": expected_telem,
            }
        ]
    )
    query = GetTaskInferenceTelemetry(reader)
    result = await query.execute("task-telemetry-1")
    assert result is not None
    assert result["task_id"] == "task-telemetry-1"
    assert result["inference_telemetry"]["provider_name"] == "cloudflare"
    assert result["inference_telemetry"]["status"] == "succeeded"


@pytest.mark.anyio
async def test_get_task_inference_telemetry_query_not_found() -> None:
    from orchestrator.application.queries import GetTaskInferenceTelemetry

    reader = FakeStateReader(events_data=[])
    query = GetTaskInferenceTelemetry(reader)
    result = await query.execute("task-missing")
    assert result is None


@pytest.mark.anyio
async def test_get_task_inference_telemetry_query_empty_task_id_raises() -> None:
    from orchestrator.application.queries import GetTaskInferenceTelemetry

    reader = FakeStateReader()
    query = GetTaskInferenceTelemetry(reader)
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("")
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("   ")


@pytest.mark.anyio
async def test_get_task_children_query_lifecycle() -> None:
    from orchestrator.application.queries import GetTaskChildren

    store = FakeTaskQueue()
    root = TaskRequested.create("Root task")
    await store.register_root_task(root)

    child1 = TaskRequested(
        task_id="child-b",
        prompt="Child B",
        created_at=root.created_at,
        root_task_id=root.task_id,
        parent_task_id=root.task_id,
        version=2,
    )
    child2 = TaskRequested(
        task_id="child-a",
        prompt="Child A",
        created_at=root.created_at,
        root_task_id=root.task_id,
        parent_task_id=root.task_id,
        version=2,
    )
    await store.register_child_task(child1)
    await store.register_child_task(child2)

    query = GetTaskChildren(store)

    # 1. Successful query with lexicographical sorting
    children = await query.execute(root.task_id)
    assert children == ["child-a", "child-b"]

    # 2. Existing parent with zero children returns []
    empty_root = TaskRequested.create("Empty root")
    await store.register_root_task(empty_root)
    assert await query.execute(empty_root.task_id) == []

    # 3. Child queried as parent returns []
    assert await query.execute(child1.task_id) == []

    # 4. Unknown parent returns None
    assert await query.execute("unknown-parent") is None

    # 5. Empty/blank task_id raises ValueError
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("")
    with pytest.raises(ValueError, match="task_id must not be empty"):
        await query.execute("   ")

    # 6. Stripping surrounding whitespace
    assert await query.execute(f"  {root.task_id}  ") == ["child-a", "child-b"]


@pytest.mark.anyio
async def test_get_task_children_get_lineage_failure_propagates() -> None:
    from orchestrator.application.queries import GetTaskChildren

    store = FakeTaskQueue()
    store.fail_get_lineage = True

    query = GetTaskChildren(store)
    with pytest.raises(RuntimeError, match="Simulated get_lineage storage failure"):
        await query.execute("root-1")


@pytest.mark.anyio
async def test_get_task_children_get_children_failure_propagates() -> None:
    from orchestrator.application.queries import GetTaskChildren

    store = FakeTaskQueue()
    root = TaskRequested.create("Root task")
    await store.register_root_task(root)
    store.fail_get_children = True

    query = GetTaskChildren(store)
    with pytest.raises(RuntimeError, match="Simulated get_children storage failure"):
        await query.execute(root.task_id)

