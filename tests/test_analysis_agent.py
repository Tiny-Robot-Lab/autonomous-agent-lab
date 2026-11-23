from typing import Any

import pytest
from agent_contracts import AgentEvent, CompletionCommand, TaskRequested, TaskResult
from analysis_agent.application.processor import AnalysisExecutionError, AnalysisProcessor
from analysis_agent.domain.analysis import AnalysisReport, analyze_material
from analysis_agent.infrastructure.deterministic_analyzer import (
    DeterministicMaterialAnalyzer,
)


class FakeMessageBus:
    def __init__(self) -> None:
        self.published: list[AgentEvent] = []

    async def publish(self, event: AgentEvent) -> None:
        self.published.append(event)


class FakeCompletionClient:
    """Captures fenced CompletionCommands instead of writing canonical keys."""

    def __init__(self) -> None:
        self.submitted: list[CompletionCommand] = []
        self.proofs: list[str] = []

    async def submit_completion(
        self,
        task_id: str,
        command: CompletionCommand,
        assignment_proof: str,
    ) -> dict[str, Any]:
        self.submitted.append(command)
        self.proofs.append(assignment_proof)
        return {"outcome": {}, "replayed": False}


ASSIGNMENT_ID = "assign-analysis-1"
ASSIGNMENT_PROOF = "proof-analysis-1"


def test_analyze_material_reports_structure_terms_and_questions() -> None:
    report = analyze_material(
        "The rover maps caves. The rover records samples. "
        "Does the map show water?"
    )

    assert report == AnalysisReport(
        paragraph_count=1,
        sentence_count=3,
        word_count=13,
        repeated_terms=(("rover", 2),),
        question_sentences=("Does the map show water?",),
    )
    assert report.render() == (
        "Analysis report\n"
        "Paragraphs: 1\n"
        "Sentences: 3\n"
        "Words: 13\n"
        "Repeated terms: rover (2)\n"
        "Question sentences:\n"
        "- Does the map show water?\n"
        "Scope: Only supplied material was analyzed; no retrieval was performed."
    )


def test_analyze_material_counts_nonempty_paragraphs_and_orders_terms() -> None:
    report = analyze_material(
        "Delta alpha alpha.\n\n"
        "Beta beta beta? Alpha delta."
    )

    assert report.paragraph_count == 2
    assert report.sentence_count == 3
    assert report.repeated_terms == (("alpha", 3), ("beta", 3), ("delta", 2))
    assert report.question_sentences == ("Beta beta beta?",)


def test_analyze_material_limits_repeated_terms_to_five() -> None:
    report = analyze_material(
        "alpha alpha bravo bravo charlie charlie delta delta "
        "echo echo foxtrot foxtrot golf golf"
    )

    assert report.repeated_terms == (
        ("alpha", 2),
        ("bravo", 2),
        ("charlie", 2),
        ("delta", 2),
        ("echo", 2),
    )


@pytest.mark.parametrize("material", ["", " \n \t"])
def test_analyze_material_handles_empty_material(material: str) -> None:
    report = analyze_material(material)

    assert report.paragraph_count == 0
    assert report.sentence_count == 0
    assert report.word_count == 0
    assert report.repeated_terms == ()
    assert report.question_sentences == ()
    assert "Repeated terms: None" in report.render()
    assert "Question sentences:\nNone" in report.render()


def test_analyze_material_rejects_non_string_material() -> None:
    with pytest.raises(TypeError, match="material must be a str"):
        analyze_material(None)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_analysis_processor_submits_delegated_child_completion() -> None:
    bus = FakeMessageBus()
    completion_client = FakeCompletionClient()
    task = TaskRequested.create("The rover maps caves. The rover records samples?")
    processor = AnalysisProcessor(
        "analysis-agent-1",
        bus,
        DeterministicMaterialAnalyzer(),
        completion_client=completion_client,
    )

    await processor.run_once(task, assignment_id=ASSIGNMENT_ID, assignment_proof=ASSIGNMENT_PROOF)

    assert [event.state for event in bus.published] == ["working", "idle"]
    assert [event.task_id for event in bus.published] == [task.task_id, task.task_id]
    assert len(completion_client.submitted) == 1
    command = completion_client.submitted[0]
    assert command.execution_kind == "delegated_child"
    assert command.failure_code is None
    assert command.completed_result == TaskResult(
        task_id=task.task_id,
        output=(
            "Analysis report\n"
            "Paragraphs: 1\n"
            "Sentences: 2\n"
            "Words: 8\n"
            "Repeated terms: rover (2)\n"
            "Question sentences:\n"
            "- The rover records samples?\n"
            "Scope: Only supplied material was analyzed; no retrieval was performed."
        ),
    )
    assert completion_client.proofs == [ASSIGNMENT_PROOF]


@pytest.mark.anyio
async def test_analysis_processor_failure_emits_error_and_safe_failure_code() -> None:
    class FailingAnalyzer:
        def analyze(self, material: str) -> AnalysisReport:
            raise RuntimeError("analysis failed")

    bus = FakeMessageBus()
    completion_client = FakeCompletionClient()
    task = TaskRequested.create("Material that cannot be analyzed.")
    processor = AnalysisProcessor(
        "analysis-agent-1", bus, FailingAnalyzer(), completion_client=completion_client
    )

    with pytest.raises(AnalysisExecutionError, match=task.task_id):
        await processor.run_once(task, assignment_id=ASSIGNMENT_ID, assignment_proof=ASSIGNMENT_PROOF)

    assert [event.state for event in bus.published] == ["working", "error"]
    assert [event.task_id for event in bus.published] == [task.task_id, task.task_id]
    assert len(completion_client.submitted) == 1
    command = completion_client.submitted[0]
    assert command.failure_code == "EXECUTION_FAILED"
    assert command.completed_result is None
    assert "analysis failed" not in command.to_json()


def _make_analysis_hint(**overrides: Any) -> Any:
    from agent_contracts import WakeUpHint
    hint = WakeUpHint(
        hint_id="h-base",
        capability="analysis",
        created_at="2026-11-22T00:00:00Z",
        execution_kind="delegated_child",
        task_id="t-base",
    )
    for k, v in overrides.items():
        object.__setattr__(hint, k, v)
    return hint


@pytest.mark.anyio
async def test_analysis_agent_malformed_hints_never_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, MagicMock, patch

    from agent_contracts import WakeUpHint
    from analysis_agent.main import run_analysis_agent

    now_iso = datetime.now(UTC).isoformat()
    mock_bus = AsyncMock()
    mock_orch = AsyncMock()

    malformed_hints = [
        _make_analysis_hint(task_id=""),
        _make_analysis_hint(task_id="   "),
        WakeUpHint(hint_id="h-missing", capability="analysis", created_at=now_iso, execution_kind="delegated_child", task_id=None),
        WakeUpHint(hint_id="h3", task_id="t3", capability="analysis", execution_kind="initial", created_at=now_iso),
        WakeUpHint(hint_id="h4", task_id="t4", capability="analysis", execution_kind="resumed_continuation", created_at=now_iso),
        WakeUpHint(hint_id="h5", task_id="t5", capability="research", execution_kind="delegated_child", created_at=now_iso),
        _make_analysis_hint(execution_kind=None),
    ]

    # Return all malformed hints, then raise CancelledError to exit loop
    mock_bus.next_signal = AsyncMock(side_effect=[*malformed_hints, TimeoutError("exit")])

    with (
        patch("analysis_agent.main.RedisBus", return_value=mock_bus),
        patch("analysis_agent.main.OrchestratorClient", return_value=mock_orch),
        patch("analysis_agent.main.HeartbeatRunner") as mock_hb,
    ):
        mock_hb.return_value.start = MagicMock()
        mock_hb.return_value.stop = AsyncMock()
        monkeypatch.setenv("RUNTIME_TOKEN", "valid-token")

        with pytest.raises(TimeoutError, match="exit"):
            await run_analysis_agent()

    mock_orch.claim_task.assert_not_called()
    mock_orch.get_child_task_input.assert_not_called()


@pytest.mark.anyio
async def test_analysis_agent_valid_hint_claims_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, MagicMock, patch

    from agent_contracts import TaskInput, WakeUpHint
    from analysis_agent.main import run_analysis_agent

    now_iso = datetime.now(UTC).isoformat()
    mock_bus = AsyncMock()
    mock_orch = AsyncMock()
    mock_proc = AsyncMock()

    valid_hint = WakeUpHint(
        hint_id="h-valid",
        task_id="task-child-10",
        capability="analysis",
        execution_kind="delegated_child",
        created_at=now_iso,
    )
    mock_bus.next_signal = AsyncMock(side_effect=[valid_hint, TimeoutError("exit")])

    async def fake_hint_claim_task(
        task_id: str, request_id: str, capability: str, execution_kind: str = "initial"
    ):
        if task_id == "task-root-1" and execution_kind == "delegated_child":
            from orchestrator.application.ports import TaskNotClaimableError

            raise TaskNotClaimableError(f"Task '{task_id}' is a root task, not a delegated child")
        if task_id == "task-child-10" and execution_kind == "delegated_child":
            return (
                {
                    "assignment_id": "asgn-10",
                    "root_task_id": "task-root-1",
                    "parent_task_id": "task-root-1",
                },
                "proof-10",
            )
        raise ValueError(f"Unexpected claim: {task_id}, {execution_kind}")

    mock_orch.claim_task = AsyncMock(side_effect=fake_hint_claim_task)
    mock_orch.acknowledge_assignment = AsyncMock()
    mock_orch.get_child_task_input = AsyncMock(
        return_value=TaskInput(
            task_id="task-child-10",
            prompt="Analyze rover sensor patterns for water signatures.",
            created_at="2026-11-22T00:00:00Z",
        )
    )

    with (
        patch("analysis_agent.main.RedisBus", return_value=mock_bus),
        patch("analysis_agent.main.OrchestratorClient", return_value=mock_orch),
        patch("analysis_agent.main.AnalysisProcessor", return_value=mock_proc),
        patch("analysis_agent.main.HeartbeatRunner") as mock_hb,
    ):
        mock_hb.return_value.start = MagicMock()
        mock_hb.return_value.stop = AsyncMock()
        monkeypatch.setenv("RUNTIME_TOKEN", "valid-token")

        with pytest.raises(TimeoutError, match="exit"):
            await run_analysis_agent()

    # Verify root task rejects delegated_child
    from orchestrator.application.ports import TaskNotClaimableError

    with pytest.raises(TaskNotClaimableError, match="root task, not a delegated child"):
        await fake_hint_claim_task("task-root-1", "req-0", "analysis", "delegated_child")

    mock_orch.claim_task.assert_awaited_once()
    assert mock_orch.claim_task.call_args.kwargs["task_id"] == "task-child-10"
    assert mock_orch.claim_task.call_args.kwargs["capability"] == "analysis"
    assert mock_orch.claim_task.call_args.kwargs["execution_kind"] == "delegated_child"

    mock_orch.acknowledge_assignment.assert_awaited_once_with(
        "task-child-10", "asgn-10", "proof-10"
    )
    mock_orch.get_child_task_input.assert_awaited_once_with(
        "task-child-10", "asgn-10", "proof-10"
    )

    mock_proc.run_once.assert_awaited_once()
    passed_task = mock_proc.run_once.call_args.args[0]
    assert passed_task.task_id == "task-child-10"
    assert passed_task.root_task_id == "task-root-1"
    assert passed_task.prompt == "Analyze rover sensor patterns for water signatures."
    assert mock_proc.run_once.call_args.kwargs["assignment_id"] == "asgn-10"
    assert mock_proc.run_once.call_args.kwargs["assignment_proof"] == "proof-10"
    assert mock_proc.run_once.call_args.kwargs["execution_kind"] == "delegated_child"


@pytest.mark.anyio
async def test_analysis_agent_direct_task_requested_still_processed(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    from analysis_agent.main import run_analysis_agent
    from orchestrator.application.ports import TaskNotClaimableError

    mock_bus = AsyncMock()
    mock_orch = AsyncMock()
    mock_proc = AsyncMock()

    direct_task = TaskRequested.create("Direct task prompt")
    mock_bus.next_signal = AsyncMock(side_effect=[direct_task, TimeoutError("exit")])

    async def fake_claim_task(
        task_id: str, request_id: str, capability: str, execution_kind: str = "initial"
    ):
        if execution_kind == "delegated_child":
            raise TaskNotClaimableError(f"Task '{task_id}' is a root task, not a delegated child")
        if execution_kind == "initial":
            return (
                {
                    "assignment_id": "asgn-direct-1",
                    "root_task_id": direct_task.task_id,
                },
                "proof-direct-1",
            )
        raise ValueError(f"Unexpected execution_kind: {execution_kind}")

    mock_orch.claim_task = AsyncMock(side_effect=fake_claim_task)
    mock_orch.acknowledge_assignment = AsyncMock()
    mock_orch.get_child_task_input = AsyncMock()

    with (
        patch("analysis_agent.main.RedisBus", return_value=mock_bus),
        patch("analysis_agent.main.OrchestratorClient", return_value=mock_orch),
        patch("analysis_agent.main.AnalysisProcessor", return_value=mock_proc),
        patch("analysis_agent.main.HeartbeatRunner") as mock_hb,
    ):
        mock_hb.return_value.start = MagicMock()
        mock_hb.return_value.stop = AsyncMock()
        monkeypatch.setenv("RUNTIME_TOKEN", "valid-token")

        with pytest.raises(TimeoutError, match="exit"):
            await run_analysis_agent()

    mock_orch.claim_task.assert_awaited_once_with(
        direct_task.task_id, mock_orch.claim_task.call_args.args[1], "analysis", "initial"
    )
    mock_orch.acknowledge_assignment.assert_awaited_once_with(
        direct_task.task_id, "asgn-direct-1", "proof-direct-1"
    )
    mock_orch.get_child_task_input.assert_not_called()
    mock_proc.run_once.assert_awaited_once_with(
        direct_task,
        assignment_id="asgn-direct-1",
        assignment_proof="proof-direct-1",
        execution_kind="initial",
    )
    passed_task = mock_proc.run_once.call_args.args[0]
    assert passed_task.prompt == "Direct task prompt"


@pytest.mark.anyio
async def test_analysis_agent_direct_root_task_with_authoritative_claim_boundary() -> None:
    from analysis_agent.application.processor import AnalysisProcessor
    from analysis_agent.infrastructure.deterministic_analyzer import DeterministicMaterialAnalyzer
    from orchestrator.application.ports import TaskNotClaimableError
    from orchestrator.application.use_cases import AcknowledgeAssignment, ClaimTask, CompleteTurn
    from orchestrator.domain.principal import RuntimePrincipal

    from tests.test_authoritative_task_store import InMemoryAuthoritativeTaskStore

    store = InMemoryAuthoritativeTaskStore()
    pepper = "assignment_proof_pepper_test_secret_32b"
    claim_uc = ClaimTask(
        task_store=store, assignment_proof_pepper=pepper, delivery_lease_seconds=30.0
    )
    ack_uc = AcknowledgeAssignment(
        task_store=store,
        assignment_proof_pepper=pepper,
        execution_claim_timeout_seconds=900.0,
    )
    complete_uc = CompleteTurn(
        task_store=store, completion_store=store, assignment_proof_pepper=pepper
    )

    principal = RuntimePrincipal(
        principal_id="principal-analysis-1",
        bound_agent_id="analysis-agent-1",
        authorized_capabilities=("analysis",),
        active=True,
    )

    root_task = TaskRequested.create("Rover sensor patterns: rover detected water.")
    await store.register_root_task(root_task, required_capability="analysis")
    await store.update_status(root_task.task_id, "pending")

    # 1. Claiming root task with delegated_child fails closed with TaskNotClaimableError
    with pytest.raises(TaskNotClaimableError, match="root task, not a delegated child"):
        await claim_uc.execute(
            principal=principal,
            task_id=root_task.task_id,
            request_id="req-root-dc",
            capability="analysis",
            execution_kind="delegated_child",
        )

    # 2. Claiming root task with initial succeeds
    claim_res = await claim_uc.execute(
        principal=principal,
        task_id=root_task.task_id,
        request_id="req-root-init",
        capability="analysis",
        execution_kind="initial",
    )
    assignment = claim_res.assignment
    proof = claim_res.assignment_proof
    assert assignment.execution_kind == "initial"
    assert proof is not None

    # 3. Acknowledge assignment
    await ack_uc.execute(
        principal=principal,
        task_id=root_task.task_id,
        assignment_id=assignment.assignment_id,
        assignment_proof=proof,
    )

    # 4. Run processor directly with execution_kind="initial"
    bus = FakeMessageBus()

    class DirectCompletionClient:
        async def submit_completion(
            self, task_id: str, command: CompletionCommand, assignment_proof: str
        ) -> dict[str, Any]:
            resp = await complete_uc.execute(
                principal=principal,
                task_id=task_id,
                command_payload=command.to_dict(),
                assignment_proof=assignment_proof,
            )
            return {"outcome": resp.outcome.to_dict(), "replayed": resp.replayed}

    processor = AnalysisProcessor(
        agent_id="analysis-agent-1",
        bus=bus,
        analyzer=DeterministicMaterialAnalyzer(),
        completion_client=DirectCompletionClient(),
    )

    await processor.run_once(
        root_task,
        assignment_id=assignment.assignment_id,
        assignment_proof=proof,
        execution_kind="initial",
    )

    # 5. Assert store state
    assert store.statuses[root_task.task_id] == "completed"
    canonical_result = store.canonical_results[root_task.task_id]
    assert "Analysis report" in canonical_result.output
    assert "rover" in canonical_result.output.lower()


@pytest.mark.anyio
async def test_analysis_bus_alternation_fairness() -> None:
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, patch

    from agent_contracts import WakeUpHint
    from analysis_agent.infrastructure.redis_bus import RedisBus

    now_iso = datetime.now(UTC).isoformat()
    hint = WakeUpHint(hint_id="h1", task_id="t1", capability="analysis", execution_kind="delegated_child", created_at=now_iso)
    task = TaskRequested.create("Sample prompt")

    with patch("redis.asyncio.from_url") as mock_redis_factory:
        mock_client = AsyncMock()
        mock_redis_factory.return_value = mock_client

        bus = RedisBus("redis://fake", queue_name="agent.tasks.analysis", hint_queue="agent.hints.analysis")

        # Mock blpop returning hint first, then task
        mock_client.blpop = AsyncMock(
            side_effect=[
                ("agent.hints.analysis", hint.to_json()),
                ("agent.tasks.analysis", task.to_json()),
            ]
        )

        sig1 = await bus.next_signal()
        assert isinstance(sig1, WakeUpHint)
        assert sig1.task_id == "t1"

        # Check that blpop queues alternated
        call1_queues = mock_client.blpop.call_args_list[0].args[0]
        assert call1_queues == ["agent.hints.analysis", "agent.tasks.analysis"]

        sig2 = await bus.next_signal()
        assert isinstance(sig2, TaskRequested)
        assert sig2.prompt == "Sample prompt"

        call2_queues = mock_client.blpop.call_args_list[1].args[0]
        assert call2_queues == ["agent.tasks.analysis", "agent.hints.analysis"]


@pytest.mark.anyio
async def test_analysis_bus_idle_timeout_does_not_toggle_precedence() -> None:
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, patch

    from agent_contracts import TaskRequested, WakeUpHint
    from analysis_agent.infrastructure.redis_bus import RedisBus

    now_iso = datetime.now(UTC).isoformat()
    hint = WakeUpHint(
        hint_id="h-timeout-1",
        task_id="t-timeout-1",
        capability="analysis",
        execution_kind="delegated_child",
        created_at=now_iso,
    )
    task = TaskRequested.create("Timeout test prompt")

    with patch("redis.asyncio.from_url") as mock_redis_factory:
        mock_client = AsyncMock()
        mock_redis_factory.return_value = mock_client

        bus = RedisBus(
            "redis://fake",
            queue_name="agent.tasks.analysis",
            hint_queue="agent.hints.analysis",
        )

        # Initial state: hints preferred
        assert bus._prefer_hints is True

        # 1. BLPOP times out on first attempt (returns None)
        # Next fetch returns hint, then another fetch times out, then next returns task
        mock_client.blpop = AsyncMock(
            side_effect=[
                None,  # Timeout 1: returns None, must NOT toggle precedence
                ("agent.hints.analysis", hint.to_json()),  # Successful hint fetch: toggles
                None,  # Timeout 2: returns None, must NOT toggle precedence
                ("agent.tasks.analysis", task.to_json()),  # Successful task fetch: toggles
                ("agent.hints.analysis", hint.to_json()),  # Alternate back to hint: toggles
            ]
        )

        # Signal 1: first blpop timed out (internal loop retries and succeeds on 2nd blpop)
        sig1 = await bus.next_signal()
        assert isinstance(sig1, WakeUpHint)
        assert sig1.task_id == "t-timeout-1"

        # Check call 0 (timed out) and call 1 (succeeded): BOTH had hint queue first!
        assert mock_client.blpop.call_args_list[0].args[0] == ["agent.hints.analysis", "agent.tasks.analysis"]
        assert mock_client.blpop.call_args_list[1].args[0] == ["agent.hints.analysis", "agent.tasks.analysis"]

        # After successful fetch of hint, preference toggled to tasks
        assert bus._prefer_hints is False

        # Signal 2: next blpop timed out (call 2), retry succeeded (call 3)
        sig2 = await bus.next_signal()
        assert isinstance(sig2, TaskRequested)
        assert sig2.prompt == "Timeout test prompt"

        # Check call 2 (timed out) and call 3 (succeeded): BOTH had task queue first!
        assert mock_client.blpop.call_args_list[2].args[0] == ["agent.tasks.analysis", "agent.hints.analysis"]
        assert mock_client.blpop.call_args_list[3].args[0] == ["agent.tasks.analysis", "agent.hints.analysis"]

        # After successful fetch of task, preference toggled back to hints
        assert bus._prefer_hints is True

        # Signal 3: immediate success (call 4), preferred queue is hint
        sig3 = await bus.next_signal()
        assert isinstance(sig3, WakeUpHint)
        assert mock_client.blpop.call_args_list[4].args[0] == ["agent.hints.analysis", "agent.tasks.analysis"]
        assert bus._prefer_hints is False
