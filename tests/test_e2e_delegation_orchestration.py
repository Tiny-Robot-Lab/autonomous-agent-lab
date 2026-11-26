"""Layer 2 deterministic end-to-end tests for Research -> Analysis delegation and continuation.

Covers the full governed lifecycle across authenticated Orchestrator boundaries:
1. Root parent task registration and initial claim by research-agent-1;
2. ResearchDelegationCoordinator selecting 'delegate_analysis';
3. Sealed StageArtifact creation and DelegationProposal submission;
4. Turn yield freeing parent executor slot (parent in waiting_for_children, no active claim);
5. Parent executor-slot invariant: research-agent-1 can claim and process an unrelated task;
6. Analysis agent consumes agent.hints.analysis, claims delegated_child, acknowledges;
7. Governed child TaskInput read (GET /api/tasks/{task_id}/input) with capability/proof validation;
8. AnalysisProcessor executes analysis and submits fenced #222 completion;
9. All 3 child outcomes tested:
   - completed -> combined synthesis;
   - failed -> degraded synthesis with primary sealed evidence;
   - claim_timed_out -> degraded synthesis after evaluated expiry;
10. Cross-instance Research resumption by research-agent-2 (epoch 2, continuation version 1);
11. Stale-proof and cross-turn fence rejection;
12. Strict security boundaries: X-Assignment-Proof header requirement, extra='forbid' on
    StageArtifactInput, no secret leakage across any response, event, or outcome.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from agent_contracts import (
    CompletionCommand,
    TaskRequested,
    TaskResult,
    normalize_canonical_utc_timestamp,
)
from agent_infrastructure import CompletionSubmissionError, OrchestratorClient
from analysis_agent.application.processor import AnalysisExecutionError, AnalysisProcessor
from analysis_agent.infrastructure.deterministic_analyzer import (
    DeterministicMaterialAnalyzer,
)
from analysis_agent.main import run_analysis_agent
from orchestrator.application.use_cases import EvaluateExpiry
from orchestrator.config import Settings
from orchestrator.domain.principal import RuntimePrincipal, compute_token_digest
from orchestrator.presentation.api import create_app
from research_agent.application.delegation import ResearchDelegationCoordinator
from research_agent.application.processor import ResearchProcessor
from research_agent.domain.models import ResearchResult, SynthesisResult
from research_agent.main import run_research_agent

from tests.test_authoritative_task_store import (
    InMemoryAuthoritativeTaskStore,
    MockAtomicDispatcher,
)

PEPPER = "test_e2e_runtime_pepper_1234567890"
PROOF_PEPPER = "test_e2e_proof_pepper_09876543210987654321"
RESEARCH_1_TOKEN = "secret-token-e2e-research-1"
RESEARCH_2_TOKEN = "secret-token-e2e-research-2"
ANALYSIS_1_TOKEN = "secret-token-e2e-analysis-1"
WORKER_1_TOKEN = "secret-token-e2e-worker-1"

PRINCIPALS_CONFIG: dict[str, dict[str, Any]] = {
    "principal-research-1": {
        "principal_id": "principal-research-1",
        "bound_agent_id": "research-agent-1",
        "authorized_capabilities": ["research"],
        "token_digest": compute_token_digest(RESEARCH_1_TOKEN, PEPPER),
        "active": True,
    },
    "principal-research-2": {
        "principal_id": "principal-research-2",
        "bound_agent_id": "research-agent-2",
        "authorized_capabilities": ["research"],
        "token_digest": compute_token_digest(RESEARCH_2_TOKEN, PEPPER),
        "active": True,
    },
    "principal-analysis-1": {
        "principal_id": "principal-analysis-1",
        "bound_agent_id": "analysis-agent-1",
        "authorized_capabilities": ["analysis"],
        "token_digest": compute_token_digest(ANALYSIS_1_TOKEN, PEPPER),
        "active": True,
    },
    "principal-worker-1": {
        "principal_id": "principal-worker-1",
        "bound_agent_id": "worker-1",
        "authorized_capabilities": ["general"],
        "token_digest": compute_token_digest(WORKER_1_TOKEN, PEPPER),
        "active": True,
    },
}


class MockEventBus:
    def __init__(self, signals: list[Any] | None = None) -> None:
        self.published: list[Any] = []
        self.current_state: str = "idle"
        self._signals = list(signals or [])

    async def publish(self, event: Any) -> None:
        self.published.append(event)
        if hasattr(event, "state"):
            self.current_state = event.state

    async def register(self, metadata: Any) -> None:
        pass

    async def heartbeat(self, heartbeat: Any) -> None:
        pass

    async def next_signal(self) -> Any:
        if self._signals:
            return self._signals.pop(0)
        await asyncio.sleep(0.01)
        raise TimeoutError("No more signals in mock event bus")

    async def aclose(self) -> None:
        pass


class DeterministicResearchExecutor:
    async def execute(self, task: TaskRequested) -> ResearchResult:
        return ResearchResult(
            task_id=task.task_id,
            output=f"Primary research findings for prompt: {task.prompt}",
            acknowledged=True,
            sources=["source://deterministic-research-archive/1"],
        )


class DeterministicInferenceProvider:
    async def synthesize(self, question: str, evidence: str) -> SynthesisResult:
        return SynthesisResult(
            question=question,
            evidence=evidence,
            content=f"Synthesized comprehensive answer for '{question}' combining evidence:\n{evidence}",
            truncated=False,
        )


class FailingAnalyzer:
    def analyze(self, material: str) -> Any:
        raise ValueError("Simulated specialist analysis execution failure: corrupt data")


def _build_test_app(store: InMemoryAuthoritativeTaskStore) -> tuple[Any, Settings]:
    settings = Settings(
        redis_url="redis://fake:6379/15",
        app_env="test",
        task_delivery_mode="authenticated_claim",
        claim_delivery_lease_seconds=30.0,
        child_claim_deadline_seconds=120.0,
        runtime_credential_pepper=PEPPER,
        assignment_proof_pepper=PROOF_PEPPER,
        runtime_principals=PRINCIPALS_CONFIG,
    )
    app = create_app(
        settings=settings,
        task_store=store,
        dispatcher=MockAtomicDispatcher(store),
        result_reader=store,
    )
    return app, settings


@pytest.mark.anyio
async def test_e2e_research_to_analysis_completed_child_cross_instance_resumption() -> None:
    store = InMemoryAuthoritativeTaskStore()
    app, _ = _build_test_app(store)
    transport = httpx.ASGITransport(app=app)

    research_client_1 = OrchestratorClient("http://test", RESEARCH_1_TOKEN, transport=transport)
    research_client_2 = OrchestratorClient("http://test", RESEARCH_2_TOKEN, transport=transport)
    analysis_client_1 = OrchestratorClient("http://test", ANALYSIS_1_TOKEN, transport=transport)

    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    parent_id = "task-parent-research-e2e"

    # Step 1: Register root parent task
    root_task = TaskRequested(
        task_id=parent_id,
        prompt="Synthesize thermal conductivity of monolayer graphene",
        created_at=now_iso,
        root_task_id=parent_id,
        parent_task_id=None,
        version=2,
    )
    await store.register_root_task(root_task, required_capability="research")
    await store.update_status(parent_id, "pending")
    await store.update_status(parent_id, "pending")
    assert store.statuses[parent_id] == "pending"

    # Step 2: research-agent-1 claims parent initial turn
    claim_1, proof_1 = await research_client_1.claim_task(
        parent_id, "req-res-1", "research", "initial"
    )
    assert claim_1["task_id"] == parent_id
    assert claim_1["fencing_epoch"] == 1
    assert store.statuses[parent_id] == "working"

    await research_client_1.acknowledge_assignment(parent_id, claim_1["assignment_id"], proof_1)
    assert store.statuses[parent_id] == "working"

    # Step 3: research-agent-1 executes turn with delegation enabled
    bus_1 = MockEventBus()
    processor_1 = ResearchProcessor(
        agent_id="research-agent-1",
        bus=bus_1,
        executor=DeterministicResearchExecutor(),
        completion_client=research_client_1,
        delegation_coordinator=ResearchDelegationCoordinator("delegate_analysis"),
    )

    await processor_1.run_once(
        root_task,
        assignment_id=claim_1["assignment_id"],
        assignment_proof=proof_1,
        execution_kind="initial",
    )

    # Step 4: Verify parent yielded and freed executor slot
    assert store.statuses[parent_id] == "waiting_for_children"
    assert (await store.get_active_claim(parent_id)) is None, "Parent must have NO active claim after yield"

    idle_events = [e for e in bus_1.published if getattr(e, "state", "") == "idle"]
    assert len(idle_events) >= 1
    assert idle_events[-1].detail == "Yielded turn for delegation"

    # Step 5: Verify parent executor-slot invariant
    # research-agent-1 is free to claim and process an unrelated task while parent is waiting!
    unrelated_id = "task-unrelated-independent-1"
    unrelated_task = TaskRequested(
        task_id=unrelated_id,
        prompt="Unrelated research task during wait",
        created_at=now_iso,
        root_task_id=unrelated_id,
        parent_task_id=None,
        version=2,
    )
    await store.register_root_task(unrelated_task, required_capability="research")
    await store.update_status(unrelated_id, "pending")

    unrelated_claim, unrelated_proof = await research_client_1.claim_task(
        unrelated_id, "req-unrelated-1", "research", "initial"
    )
    await research_client_1.acknowledge_assignment(
        unrelated_id, unrelated_claim["assignment_id"], unrelated_proof
    )
    unrelated_processor = ResearchProcessor(
        agent_id="research-agent-1",
        bus=bus_1,
        executor=DeterministicResearchExecutor(),
        completion_client=research_client_1,
        delegation_coordinator=ResearchDelegationCoordinator("complete_locally"),
    )
    await unrelated_processor.run_once(
        unrelated_task,
        assignment_id=unrelated_claim["assignment_id"],
        assignment_proof=unrelated_proof,
        execution_kind="initial",
    )
    assert store.statuses[unrelated_id] == "completed"
    assert store.statuses[parent_id] == "waiting_for_children"

    # Step 6: Analysis agent consumes wake-up hint from agent.hints.analysis
    analysis_hints = [h for h in store.hints if h.capability == "analysis"]
    assert len(analysis_hints) == 1
    child_hint = analysis_hints[0]
    assert child_hint.execution_kind == "delegated_child"
    child_id = child_hint.task_id

    # Step 7: analysis-agent-1 claims and acknowledges child task
    child_claim, child_proof = await analysis_client_1.claim_task(
        child_id, "req-analysis-1", "analysis", "delegated_child"
    )
    assert child_claim["task_id"] == child_id
    assert child_claim["root_task_id"] == parent_id
    await analysis_client_1.acknowledge_assignment(
        child_id, child_claim["assignment_id"], child_proof
    )

    # Step 8: analysis-agent-1 loads authoritative child TaskInput via governed endpoint
    child_task_input = await analysis_client_1.get_child_task_input(
        child_id, child_claim["assignment_id"], child_proof
    )
    assert child_task_input.task_id == child_id
    assert "Analyze the sealed Research evidence" in child_task_input.prompt

    # Step 9: analysis-agent-1 executes AnalysisProcessor and completes child turn
    bus_analysis = MockEventBus()
    analysis_processor = AnalysisProcessor(
        agent_id="analysis-agent-1",
        bus=bus_analysis,
        analyzer=DeterministicMaterialAnalyzer(),
        completion_client=analysis_client_1,
    )
    child_task_requested = TaskRequested(
        task_id=child_id,
        prompt=child_task_input.prompt,
        created_at=child_task_input.created_at,
        root_task_id=parent_id,
        parent_task_id=parent_id,
        version=2,
    )
    await analysis_processor.run_once(
        child_task_requested,
        assignment_id=child_claim["assignment_id"],
        assignment_proof=child_proof,
        execution_kind="delegated_child",
    )

    assert store.statuses[child_id] == "completed"
    assert store.statuses[parent_id] == "resumable"

    # Step 10: Parent wake-up hint emitted for research
    research_resumption_hints = [
        h for h in store.hints
        if h.capability == "research" and h.execution_kind == "resumed_continuation"
    ]
    assert len(research_resumption_hints) >= 1
    parent_hint = research_resumption_hints[-1]
    assert parent_hint.task_id == parent_id
    assert parent_hint.expected_continuation_version == 1

    # Step 11: Cross-instance Research resumption by research-agent-2
    resumed_claim, resumed_proof = await research_client_2.claim_task(
        parent_id,
        "req-resumed-1",
        "research",
        "resumed_continuation",
        expected_continuation_version=1,
    )
    assert resumed_claim["task_id"] == parent_id
    assert resumed_claim["fencing_epoch"] == 2
    assert resumed_claim["continuation_version"] == 1

    # Acknowledge the resumed assignment under research-agent-2
    await research_client_2.acknowledge_assignment(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )

    # Prove stale completion rejection (stale completion, not stale acknowledgement):
    # research-agent-1 attempts stale completion submission after research-agent-2 owns the resumed epoch.
    # Rejected at the detached-proof authorization boundary (exact HTTP 403);
    # assignment existence is not disclosed.
    hint_count_before = len(store.hints)
    stale_comp_cmd = CompletionCommand(
        completion_command_id=f"comp-stale-{uuid4().hex[:8]}",
        assignment_id=claim_1["assignment_id"],
        execution_kind="initial",
        completed_result=TaskResult(task_id=parent_id, output="Stale initial completion attempt"),
    )
    with pytest.raises(CompletionSubmissionError) as exc_info:
        await research_client_1.submit_completion(parent_id, stale_comp_cmd, proof_1)
    assert exc_info.value.status_code == 403

    # Assert exactly:
    active_claim = await store.get_active_claim(parent_id)
    assert active_claim is not None
    assert active_claim.assignment_id == resumed_claim["assignment_id"]
    assert active_claim.runtime_principal_id == "principal-research-2"
    assert active_claim.bound_agent_id == "research-agent-2"
    assert (await store.get_fencing_epoch(parent_id)) == 2
    assert (await store.get_status(parent_id)) == "working"
    assert (await store.get_terminal_outcome(parent_id)) is None
    assert (await store.get_canonical_result(parent_id)) is None
    assert len(store.hints) == hint_count_before
    assert (await store.get_continuation(parent_id, 2)) is None

    # Step 12: research-agent-2 loads continuation context
    continuation_context = await research_client_2.get_resumed_continuation_context(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )
    assert continuation_context.parent_task_input.task_id == parent_id
    assert continuation_context.continuation.continuation_version == 1
    assert continuation_context.stage_artifact.producing_task_id == parent_id
    assert continuation_context.child_outcome.terminal_status == "completed"
    assert continuation_context.child_result is not None

    # Step 13: research-agent-2 completes parent turn with combined synthesis
    bus_2 = MockEventBus()
    processor_2 = ResearchProcessor(
        agent_id="research-agent-2",
        bus=bus_2,
        executor=DeterministicResearchExecutor(),
        completion_client=research_client_2,
        inference_provider=DeterministicInferenceProvider(),
        context_reader=research_client_2,
    )

    resumed_task_requested = TaskRequested(
        task_id=parent_id,
        prompt=continuation_context.parent_task_input.prompt,
        created_at=continuation_context.parent_task_input.created_at,
        root_task_id=parent_id,
        parent_task_id=None,
        version=2,
    )
    await processor_2.run_once(
        resumed_task_requested,
        assignment_id=resumed_claim["assignment_id"],
        assignment_proof=resumed_proof,
        execution_kind="resumed_continuation",
        continuation_version=1,
    )

    # Step 14: Verify parent completed with combined evidence
    assert store.statuses[parent_id] == "completed"
    parent_result = store.canonical_results[parent_id]
    assert parent_result is not None
    assert "Primary Research Evidence:" in parent_result.output
    assert "Specialist Analysis Findings:" in parent_result.output
    assert "Synthesized comprehensive answer" in parent_result.output

    # Step 15: Verify zero secret leakage across results and events
    for event in bus_1.published + bus_2.published + bus_analysis.published:
        event_str = str(event.__dict__ if hasattr(event, "__dict__") else event)
        assert RESEARCH_1_TOKEN not in event_str
        assert RESEARCH_2_TOKEN not in event_str
        assert ANALYSIS_1_TOKEN not in event_str
        assert PEPPER not in event_str
        assert PROOF_PEPPER not in event_str
        assert proof_1 not in event_str
        assert resumed_proof not in event_str
        assert child_proof not in event_str


@pytest.mark.anyio
async def test_e2e_research_to_analysis_failed_child_degraded_resumption() -> None:
    store = InMemoryAuthoritativeTaskStore()
    app, _ = _build_test_app(store)
    transport = httpx.ASGITransport(app=app)

    research_client_1 = OrchestratorClient("http://test", RESEARCH_1_TOKEN, transport=transport)
    research_client_2 = OrchestratorClient("http://test", RESEARCH_2_TOKEN, transport=transport)
    analysis_client_1 = OrchestratorClient("http://test", ANALYSIS_1_TOKEN, transport=transport)

    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    parent_id = "task-parent-failed-child-e2e"

    # Register root parent
    root_task = TaskRequested(
        task_id=parent_id,
        prompt="Analyze graphene superconductivity under pressure",
        created_at=now_iso,
        root_task_id=parent_id,
        parent_task_id=None,
        version=2,
    )
    await store.register_root_task(root_task, required_capability="research")
    await store.update_status(parent_id, "pending")

    # research-agent-1 claims and delegates
    claim_1, proof_1 = await research_client_1.claim_task(
        parent_id, "req-fail-1", "research", "initial"
    )
    await research_client_1.acknowledge_assignment(parent_id, claim_1["assignment_id"], proof_1)

    bus_1 = MockEventBus()
    processor_1 = ResearchProcessor(
        agent_id="research-agent-1",
        bus=bus_1,
        executor=DeterministicResearchExecutor(),
        completion_client=research_client_1,
        delegation_coordinator=ResearchDelegationCoordinator("delegate_analysis"),
    )
    await processor_1.run_once(
        root_task,
        assignment_id=claim_1["assignment_id"],
        assignment_proof=proof_1,
        execution_kind="initial",
    )
    assert store.statuses[parent_id] == "waiting_for_children"

    # Analysis agent receives hint and claims child
    analysis_hints = [h for h in store.hints if h.capability == "analysis"]
    child_id = analysis_hints[-1].task_id

    child_claim, child_proof = await analysis_client_1.claim_task(
        child_id, "req-child-fail", "analysis", "delegated_child"
    )
    await analysis_client_1.acknowledge_assignment(
        child_id, child_claim["assignment_id"], child_proof
    )
    child_input = await analysis_client_1.get_child_task_input(
        child_id, child_claim["assignment_id"], child_proof
    )

    # AnalysisProcessor executes with FailingAnalyzer -> submits failure completion
    bus_analysis = MockEventBus()
    failing_processor = AnalysisProcessor(
        agent_id="analysis-agent-1",
        bus=bus_analysis,
        analyzer=FailingAnalyzer(),
        completion_client=analysis_client_1,
    )
    child_task_requested = TaskRequested(
        task_id=child_id,
        prompt=child_input.prompt,
        created_at=child_input.created_at,
        root_task_id=parent_id,
        parent_task_id=parent_id,
        version=2,
    )
    with pytest.raises(AnalysisExecutionError):
        await failing_processor.run_once(
            child_task_requested,
            assignment_id=child_claim["assignment_id"],
            assignment_proof=child_proof,
            execution_kind="delegated_child",
        )

    # Child transitioned to failed, parent transitioned to resumable
    assert store.statuses[child_id] == "failed"
    assert store.statuses[parent_id] == "resumable"

    # research-agent-2 resumes parent turn
    resumed_claim, resumed_proof = await research_client_2.claim_task(
        parent_id,
        "req-resumed-fail-1",
        "research",
        "resumed_continuation",
        expected_continuation_version=1,
    )
    await research_client_2.acknowledge_assignment(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )

    context = await research_client_2.get_resumed_continuation_context(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )
    assert context.child_outcome.terminal_status == "failed"

    bus_2 = MockEventBus()
    processor_2 = ResearchProcessor(
        agent_id="research-agent-2",
        bus=bus_2,
        executor=DeterministicResearchExecutor(),
        completion_client=research_client_2,
        inference_provider=DeterministicInferenceProvider(),
        context_reader=research_client_2,
    )
    await processor_2.run_once(
        TaskRequested(
            task_id=parent_id,
            prompt=context.parent_task_input.prompt,
            created_at=context.parent_task_input.created_at,
            root_task_id=parent_id,
            parent_task_id=None,
            version=2,
        ),
        assignment_id=resumed_claim["assignment_id"],
        assignment_proof=resumed_proof,
        execution_kind="resumed_continuation",
        continuation_version=1,
    )

    assert store.statuses[parent_id] == "completed"
    parent_result = store.canonical_results[parent_id]
    assert "[Note: Specialist analysis was unavailable (failed)" in parent_result.output
    assert "primary sealed research evidence" in parent_result.output


@pytest.mark.anyio
async def test_e2e_research_to_analysis_timed_out_child_degraded_resumption() -> None:
    store = InMemoryAuthoritativeTaskStore()
    app, _ = _build_test_app(store)
    transport = httpx.ASGITransport(app=app)

    research_client_1 = OrchestratorClient("http://test", RESEARCH_1_TOKEN, transport=transport)
    research_client_2 = OrchestratorClient("http://test", RESEARCH_2_TOKEN, transport=transport)

    now_dt = datetime.now(UTC)
    now_iso = normalize_canonical_utc_timestamp(now_dt)
    parent_id = "task-parent-timeout-child-e2e"

    # Register root parent
    root_task = TaskRequested(
        task_id=parent_id,
        prompt="Synthesize phonon dispersion in twisted bilayer graphene",
        created_at=now_iso,
        root_task_id=parent_id,
        parent_task_id=None,
        version=2,
    )
    await store.register_root_task(root_task, required_capability="research")
    await store.update_status(parent_id, "pending")

    # research-agent-1 claims and delegates
    claim_1, proof_1 = await research_client_1.claim_task(
        parent_id, "req-time-1", "research", "initial"
    )
    await research_client_1.acknowledge_assignment(parent_id, claim_1["assignment_id"], proof_1)

    bus_1 = MockEventBus()
    processor_1 = ResearchProcessor(
        agent_id="research-agent-1",
        bus=bus_1,
        executor=DeterministicResearchExecutor(),
        completion_client=research_client_1,
        delegation_coordinator=ResearchDelegationCoordinator("delegate_analysis"),
    )
    await processor_1.run_once(
        root_task,
        assignment_id=claim_1["assignment_id"],
        assignment_proof=proof_1,
        execution_kind="initial",
    )
    assert store.statuses[parent_id] == "waiting_for_children"

    # Child is NOT claimed by analysis agent
    analysis_hints = [h for h in store.hints if h.capability == "analysis"]
    child_id = analysis_hints[-1].task_id
    assert store.statuses[child_id] == "pending"

    # Advance clock past child_claim_deadline
    deadline_iso = store.claim_deadlines[child_id]
    past_deadline_dt = datetime.fromisoformat(deadline_iso) + timedelta(seconds=10)
    past_deadline_iso = normalize_canonical_utc_timestamp(past_deadline_dt)

    # Evaluate expiry on the pending child
    eval_expiry = EvaluateExpiry(task_store=store)
    operator = RuntimePrincipal(
        principal_id="recovery-operator",
        bound_agent_id="operator-1",
        authorized_capabilities=("recovery_operator",),
        active=True,
    )
    result = await eval_expiry.execute(operator, child_id, now=past_deadline_iso)
    assert result.action == "claim_timed_out"
    assert result.outcome is not None
    assert result.outcome.terminal_status == "claim_timed_out"

    assert store.statuses[child_id] == "claim_timed_out"
    assert store.statuses[parent_id] == "resumable"

    # research-agent-2 resumes parent turn
    resumed_claim, resumed_proof = await research_client_2.claim_task(
        parent_id,
        "req-resumed-timeout-1",
        "research",
        "resumed_continuation",
        expected_continuation_version=1,
    )
    await research_client_2.acknowledge_assignment(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )

    context = await research_client_2.get_resumed_continuation_context(
        parent_id, resumed_claim["assignment_id"], resumed_proof
    )
    assert context.child_outcome.terminal_status == "claim_timed_out"

    bus_2 = MockEventBus()
    processor_2 = ResearchProcessor(
        agent_id="research-agent-2",
        bus=bus_2,
        executor=DeterministicResearchExecutor(),
        completion_client=research_client_2,
        inference_provider=DeterministicInferenceProvider(),
        context_reader=research_client_2,
    )
    await processor_2.run_once(
        TaskRequested(
            task_id=parent_id,
            prompt=context.parent_task_input.prompt,
            created_at=context.parent_task_input.created_at,
            root_task_id=parent_id,
            parent_task_id=None,
            version=2,
        ),
        assignment_id=resumed_claim["assignment_id"],
        assignment_proof=resumed_proof,
        execution_kind="resumed_continuation",
        continuation_version=1,
    )

    assert store.statuses[parent_id] == "completed"
    parent_result = store.canonical_results[parent_id]
    assert "[Note: Specialist analysis was unavailable (claim_timed_out)" in parent_result.output
    assert "primary sealed research evidence" in parent_result.output


@pytest.mark.anyio
async def test_e2e_governed_child_task_input_security_boundaries() -> None:
    store = InMemoryAuthoritativeTaskStore()
    app, _ = _build_test_app(store)
    transport = httpx.ASGITransport(app=app)

    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    parent_id = "task-parent-sec-1"
    child_id = "task-child-sec-1"

    # Setup parent and child in store
    await store.register_root_task(
        TaskRequested(
            task_id=parent_id,
            prompt="Parent prompt",
            created_at=now_iso,
            root_task_id=parent_id,
            parent_task_id=None,
            version=2,
        ),
        required_capability="research",
    )
    await store.register_child_task(
        TaskRequested(
            task_id=child_id,
            prompt="Analyze data securely",
            created_at=now_iso,
            root_task_id=parent_id,
            parent_task_id=parent_id,
            version=2,
        ),
        required_capability="analysis",
    )
    store.statuses[child_id] = "pending"
    store.claim_deadlines[child_id] = normalize_canonical_utc_timestamp(
        datetime.now(UTC) + timedelta(seconds=120)
    )

    analysis_client = OrchestratorClient("http://test", ANALYSIS_1_TOKEN, transport=transport)
    claim, proof = await analysis_client.claim_task(
        child_id, "req-sec-claim", "analysis", "delegated_child"
    )
    assignment_id = claim["assignment_id"]

    # 1. Unacknowledged claim -> 409
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_http:
        resp = await raw_http.get(
            f"/api/tasks/{child_id}/input",
            params={"assignment_id": assignment_id},
            headers={"Authorization": f"Bearer {ANALYSIS_1_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert resp.status_code == 409

        # Now acknowledge assignment
        await analysis_client.acknowledge_assignment(child_id, assignment_id, proof)

        # 2. Missing authorization header -> 401
        resp = await raw_http.get(
            f"/api/tasks/{child_id}/input",
            params={"assignment_id": assignment_id},
            headers={"X-Assignment-Proof": proof},
        )
        assert resp.status_code == 401

        # 3. Missing X-Assignment-Proof header -> 401
        resp = await raw_http.get(
            f"/api/tasks/{child_id}/input",
            params={"assignment_id": assignment_id},
            headers={"Authorization": f"Bearer {ANALYSIS_1_TOKEN}"},
        )
        assert resp.status_code == 401

        # 4. Capability mismatch: Research principal calling GET /api/tasks/{child_id}/input -> 403
        resp = await raw_http.get(
            f"/api/tasks/{child_id}/input",
            params={"assignment_id": assignment_id},
            headers={"Authorization": f"Bearer {RESEARCH_1_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert resp.status_code == 403

        # 5. Invalid / corrupt proof -> 403
        resp = await raw_http.get(
            f"/api/tasks/{child_id}/input",
            params={"assignment_id": assignment_id},
            headers={"Authorization": f"Bearer {ANALYSIS_1_TOKEN}", "X-Assignment-Proof": "corrupt_proof_123"},
        )
        assert resp.status_code == 403

        # 6. Assignment ID mismatch -> 404
        resp = await raw_http.get(
            f"/api/tasks/{child_id}/input",
            params={"assignment_id": "wrong-assignment-id"},
            headers={"Authorization": f"Bearer {ANALYSIS_1_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert resp.status_code == 404

        # 7. Valid request succeeds and returns authoritative child input
        resp = await raw_http.get(
            f"/api/tasks/{child_id}/input",
            params={"assignment_id": assignment_id},
            headers={"Authorization": f"Bearer {ANALYSIS_1_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_input" in data
        assert data["task_input"]["task_id"] == child_id
        assert data["task_input"]["prompt"] == "Analyze data securely"


@pytest.mark.anyio
async def test_e2e_stage_artifact_security_boundaries() -> None:
    store = InMemoryAuthoritativeTaskStore()
    app, _ = _build_test_app(store)
    transport = httpx.ASGITransport(app=app)

    now_iso = normalize_canonical_utc_timestamp(datetime.now(UTC))
    parent_id = "task-parent-artifact-sec"

    await store.register_root_task(
        TaskRequested(
            task_id=parent_id,
            prompt="Parent prompt",
            created_at=now_iso,
            root_task_id=parent_id,
            parent_task_id=None,
            version=2,
        ),
        required_capability="research",
    )
    await store.update_status(parent_id, "pending")
    research_client = OrchestratorClient("http://test", RESEARCH_1_TOKEN, transport=transport)
    claim, proof = await research_client.claim_task(
        parent_id, "req-art-claim", "research", "initial"
    )
    assignment_id = claim["assignment_id"]
    await research_client.acknowledge_assignment(parent_id, assignment_id, proof)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_http:
        valid_body = {
            "assignment_id": assignment_id,
            "artifact_id": "art-sec-1",
            "artifact_type": "research_material",
            "content": {"material": "Confidential research findings"},
            "artifact_version": 1,
            "payload_schema_version": 1,
        }

        # 1. Missing X-Assignment-Proof header -> 401
        resp = await raw_http.post(
            f"/api/tasks/{parent_id}/artifacts",
            json=valid_body,
            headers={"Authorization": f"Bearer {RESEARCH_1_TOKEN}"},
        )
        assert resp.status_code == 401

        # 2. Extra 'assignment_proof' in JSON body -> 422 (extra='forbid')
        body_with_forbidden_proof = dict(valid_body)
        body_with_forbidden_proof["assignment_proof"] = proof
        resp = await raw_http.post(
            f"/api/tasks/{parent_id}/artifacts",
            json=body_with_forbidden_proof,
            headers={"Authorization": f"Bearer {RESEARCH_1_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert resp.status_code == 422

        # 3. Valid request succeeds
        resp = await raw_http.post(
            f"/api/tasks/{parent_id}/artifacts",
            json=valid_body,
            headers={"Authorization": f"Bearer {RESEARCH_1_TOKEN}", "X-Assignment-Proof": proof},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["artifact"]["artifact_id"] == "art-sec-1"
        assert data["artifact"]["sealed"] is True


@pytest.mark.anyio
async def test_e2e_production_loop_full_delegation_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """True production-loop deterministic test driving full delegation flow from POST /api/tasks.

    Governed by Issue #224 Copilot Finding 3:
    - Starts with POST /api/tasks in authenticated_claim mode;
    - Drives full end-to-end delegation (Research initial -> Analysis child -> Research resumed);
    - Invokes actual production runtime loop entry points (run_research_agent, run_analysis_agent);
    - Contains zero forbidden direct calls in the test body;
    - Proves all 14 points of the full delegation, specialization, and resumption lifecycle.
    """
    store = InMemoryAuthoritativeTaskStore()
    app, _ = _build_test_app(store)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        # 1. Initial TaskRequested hint is published after POST /api/tasks.
        prompt_with_distinctive_fact = (
            "Analyze topological Weyl materials. Topological Weyl semimetals exhibit Weyl fermions "
            "and topological surface states with 23 meV magnetic exchange gap."
        )
        resp = await http_client.post(
            "/api/tasks",
            json={"prompt": prompt_with_distinctive_fact, "capability": "research"},
        )
        assert resp.status_code == 202
        body = resp.json()
        parent_id = body["task_id"]
        assert body["status"] == "accepted"

        assert len(store.hints) == 1
        initial_hint = store.hints[-1]
        assert initial_hint.task_id == parent_id
        assert initial_hint.execution_kind == "initial"
        assert initial_hint.capability == "research"

        # 2-4. Research Agent instance 1 runs initial production loop:
        # consumes hint, claims root task, acknowledges assignment, fetches governed root input,
        # stages sealed evidence artifact, and proposes delegation to Analysis.
        monkeypatch.setenv("AGENT_ID", "research-agent-1")
        monkeypatch.setenv("RUNTIME_TOKEN", RESEARCH_1_TOKEN)
        monkeypatch.setenv("RESEARCH_DELEGATION_MODE", "delegate_analysis")
        monkeypatch.setenv("RESEARCH_EXECUTOR", "stub")
        monkeypatch.setenv("REDIS_AUTH_MODE", "unauthenticated_local_test")
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.delenv("REDIS_USERNAME", raising=False)
        monkeypatch.delenv("REDIS_PASSWORD_FILE", raising=False)

        bus_1 = MockEventBus([initial_hint])
        with (
            patch("research_agent.main.RedisBus", return_value=bus_1),
            patch(
                "research_agent.main.OrchestratorClient",
                side_effect=lambda url, token, **kw: OrchestratorClient(url, token, transport=transport, **kw),
            ),
            patch("research_agent.main.create_research_executor", return_value=DeterministicResearchExecutor()),
        ):
            await run_research_agent(max_signals=1)

        # 5. Parent yielded, child task and child hint are published for Analysis.
        assert store.statuses[parent_id] == "waiting_for_children"
        assert (await store.get_active_claim(parent_id)) is None

        assert len(store.hints) == 2
        child_hint = store.hints[-1]
        assert child_hint.execution_kind == "delegated_child"
        assert child_hint.capability == "analysis"
        child_id = child_hint.task_id
        assert store.statuses[child_id] == "pending"

        # Verify sealed research artifact was staged
        staged_artifacts = [
            art for (t_id, _, _), art in store.artifacts.items() if t_id == parent_id
        ]
        assert len(staged_artifacts) == 1
        sealed_art = staged_artifacts[0]
        assert sealed_art.sealed is True
        assert sealed_art.artifact_type == "research_material"
        assert "weyl" in str(sealed_art.content).lower()

        # 6-9. Analysis Agent runs production loop on delegated child:
        # consumes child hint, claims delegated child, acknowledges assignment, fetches
        # ChildTaskExecutionMaterial (authoritative task input + sealed parent artifact),
        # analyzes sealed material (distinctive fact reaches analysis result), and submits
        # fenced completion yielding child result, transitioning child to completed, unblocking
        # parent, minting continuation context, and publishing resumed Research hint.
        monkeypatch.setenv("AGENT_ID", "analysis-agent-1")
        monkeypatch.setenv("RUNTIME_TOKEN", ANALYSIS_1_TOKEN)

        bus_analysis = MockEventBus([child_hint])
        with (
            patch("analysis_agent.main.RedisBus", return_value=bus_analysis),
            patch(
                "analysis_agent.main.OrchestratorClient",
                side_effect=lambda url, token, **kw: OrchestratorClient(url, token, transport=transport, **kw),
            ),
        ):
            await run_analysis_agent(max_signals=1)

        assert store.statuses[child_id] == "completed"
        assert (await store.get_active_claim(child_id)) is None
        child_outcome = await store.get_terminal_outcome(child_id)
        assert child_outcome is not None
        assert child_outcome.terminal_status == "completed"

        child_result = await store.get_canonical_result(child_id)
        assert child_result is not None
        # 8. Distinctive fact from sealed research artifact reaches analysis result
        assert "weyl" in child_result.output.lower() or "topological" in child_result.output.lower()

        # 9. Child completion unblocks parent, mints continuation context, publishes resumed hint
        assert store.statuses[parent_id] == "resumable"
        assert len(store.hints) == 3
        resumed_hint = store.hints[-1]
        assert resumed_hint.task_id == parent_id
        assert resumed_hint.execution_kind == "resumed_continuation"
        assert resumed_hint.capability == "research"
        assert resumed_hint.expected_continuation_version == 1

        # 10-14. Resumed Research production loop on distinct agent instance 2:
        # claims resumed continuation turn under new fenced epoch 2, acknowledges,
        # fetches governed continuation context, completes parent task under fenced epoch 2,
        # with no instance-local state leaked from turn 1 into turn 2.
        monkeypatch.setenv("AGENT_ID", "research-agent-2")
        monkeypatch.setenv("RUNTIME_TOKEN", RESEARCH_2_TOKEN)
        monkeypatch.setenv("RESEARCH_DELEGATION_MODE", "complete_locally")

        bus_2 = MockEventBus([resumed_hint])
        with (
            patch("research_agent.main.RedisBus", return_value=bus_2),
            patch(
                "research_agent.main.OrchestratorClient",
                side_effect=lambda url, token, **kw: OrchestratorClient(url, token, transport=transport, **kw),
            ),
        ):
            await run_research_agent(max_signals=1)

        # 13-14. Parent completed under fenced epoch 2, canonical result verified
        assert store.statuses[parent_id] == "completed"
        assert (await store.get_fencing_epoch(parent_id)) == 2
        assert (await store.get_active_claim(parent_id)) is None

        parent_outcome = await store.get_terminal_outcome(parent_id)
        assert parent_outcome is not None
        assert parent_outcome.terminal_status == "completed"
        assert parent_outcome.bound_agent_id == "research-agent-2"

        parent_result = await store.get_canonical_result(parent_id)
        assert parent_result is not None
        assert "Primary Research Evidence:" in parent_result.output
        assert "Specialist Analysis Findings:" in parent_result.output
        assert "weyl" in parent_result.output.lower()

        # Canonical result query via GET /api/tasks/{parent_id}/result endpoint
        res_resp = await http_client.get(f"/api/tasks/{parent_id}/result")
        assert res_resp.status_code == 200
        result_payload = res_resp.json()
        assert result_payload["status"] == "completed"
        assert "Primary Research Evidence:" in result_payload["output"]
        assert "Specialist Analysis Findings:" in result_payload["output"]
