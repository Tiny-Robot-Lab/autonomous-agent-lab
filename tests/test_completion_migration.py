"""Migration proofs for assignment-backed runtime completion (Issue #222).

Static and explicit behavioral tests proving that the General Worker (and, in
later increments, the Research and Analysis agents):
- no longer write canonical result/outcome keys directly;
- submit every turn completion through the Orchestrator completion boundary;
- submit only allowlisted safe failure codes and never raw exception text;
- emit agent events as observational lifecycle only, never as canonical
  completion evidence.
"""

from pathlib import Path

import pytest
from agent_contracts import TaskRequested

WORKER_SRC = Path("services/worker/src/worker")
RESEARCH_SRC = Path("services/research_agent/src/research_agent")
ANALYSIS_SRC = Path("services/analysis_agent/src/analysis_agent")

FORBIDDEN_KEY_PREFIXES = (
    "task.result:",
    "task.outcome:",
    "task.completion:",
    "task.assignment_completion:",
)


def _iter_source_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def test_worker_source_contains_no_direct_canonical_writes() -> None:
    offenders: list[str] = []
    for path in _iter_source_files(WORKER_SRC):
        text = path.read_text(encoding="utf-8")
        for prefix in FORBIDDEN_KEY_PREFIXES:
            if prefix in text:
                offenders.append(f"{path}: {prefix}")
    assert offenders == []


def test_worker_bus_has_no_direct_result_write_seam() -> None:
    from worker.infrastructure.redis_bus import RedisBus

    assert not hasattr(RedisBus, "save")


def test_worker_processor_module_has_no_result_store() -> None:
    processor_src = (WORKER_SRC / "application" / "processor.py").read_text(encoding="utf-8")
    assert "ResultStore" not in processor_src
    assert "result_store" not in processor_src


def test_research_source_contains_no_direct_canonical_writes() -> None:
    offenders: list[str] = []
    for path in _iter_source_files(RESEARCH_SRC):
        text = path.read_text(encoding="utf-8")
        for prefix in FORBIDDEN_KEY_PREFIXES:
            if prefix in text:
                offenders.append(f"{path}: {prefix}")
    assert offenders == []


def test_research_bus_has_no_direct_result_write_seam() -> None:
    from research_agent.infrastructure.redis_bus import RedisBus

    assert not hasattr(RedisBus, "save")


def test_research_processor_module_has_no_result_store() -> None:
    processor_src = (RESEARCH_SRC / "application" / "processor.py").read_text(encoding="utf-8")
    assert "ResultStore" not in processor_src
    assert "result_store" not in processor_src


def test_analysis_source_contains_no_direct_canonical_writes() -> None:
    offenders: list[str] = []
    for path in _iter_source_files(ANALYSIS_SRC):
        text = path.read_text(encoding="utf-8")
        for prefix in FORBIDDEN_KEY_PREFIXES:
            if prefix in text:
                offenders.append(f"{path}: {prefix}")
    assert offenders == []


def test_analysis_bus_has_no_direct_result_write_seam() -> None:
    from analysis_agent.infrastructure.redis_bus import RedisBus

    assert not hasattr(RedisBus, "save")


def test_analysis_processor_module_has_no_result_store() -> None:
    processor_src = (ANALYSIS_SRC / "application" / "processor.py").read_text(encoding="utf-8")
    assert "ResultStore" not in processor_src
    assert "result_store" not in processor_src


@pytest.mark.anyio
async def test_worker_success_submits_completed_completion_command() -> None:
    from worker.application.processor import TaskProcessor

    from tests.test_worker import FakeCompletionClient, FakeMessageBus

    bus = FakeMessageBus()
    client = FakeCompletionClient()
    processor = TaskProcessor("worker-1", bus, completion_client=client)
    task = TaskRequested.create("migration probe")

    await processor.run_once(task, assignment_id="assign-mig-1", assignment_proof="proof-mig-1")

    assert len(client.submitted) == 1
    command = client.submitted[0]
    assert command.execution_kind == "initial"
    assert command.completed_result is not None
    assert command.completed_result.task_id == task.task_id
    assert command.failure_code is None


@pytest.mark.anyio
async def test_worker_failure_submits_only_allowlisted_failure_code() -> None:
    from agent_contracts import ALLOWED_COMPLETION_FAILURE_CODES
    from worker.application.processor import TaskExecutionError

    from tests.test_worker import FailingTaskProcessor, FakeCompletionClient, FakeMessageBus

    bus = FakeMessageBus()
    client = FakeCompletionClient()
    secret_message = "super-secret-provider-boom"
    processor = FailingTaskProcessor(
        "worker-1", bus, RuntimeError(secret_message), completion_client=client
    )
    task = TaskRequested.create("failing migration probe")

    with pytest.raises(TaskExecutionError):
        await processor.run_once(task, assignment_id="assign-mig-2", assignment_proof="proof-mig-2")

    assert len(client.submitted) == 1
    command = client.submitted[0]
    assert command.failure_code in ALLOWED_COMPLETION_FAILURE_CODES
    assert command.completed_result is None
    assert secret_message not in command.to_json()


@pytest.mark.anyio
async def test_research_success_submits_completed_completion_command() -> None:
    from research_agent.application.processor import ResearchProcessor
    from research_agent.infrastructure.stub_executor import StubResearchExecutor

    from tests.test_research_agent import FakeCompletionClient, FakeMessageBus

    bus = FakeMessageBus()
    client = FakeCompletionClient()
    processor = ResearchProcessor(
        "research-agent-1", bus, StubResearchExecutor(), completion_client=client
    )
    task = TaskRequested.create("migration research probe")

    await processor.run_once(task, assignment_id="assign-mig-r1", assignment_proof="proof-mig-r1")

    assert len(client.submitted) == 1
    command = client.submitted[0]
    assert command.execution_kind == "initial"
    assert command.completed_result is not None
    assert command.completed_result.task_id == task.task_id
    assert command.failure_code is None


@pytest.mark.anyio
async def test_research_failure_submits_only_allowlisted_failure_code() -> None:
    from agent_contracts import ALLOWED_COMPLETION_FAILURE_CODES
    from research_agent.application.processor import ResearchExecutionError, ResearchProcessor
    from research_agent.infrastructure.stub_executor import StubResearchExecutor

    from tests.test_research_agent import FakeCompletionClient, FakeMessageBus

    bus = FakeMessageBus()
    client = FakeCompletionClient()
    executor = StubResearchExecutor(failure_trigger="migration-fail")
    processor = ResearchProcessor(
        "research-agent-1", bus, executor, completion_client=client
    )
    task = TaskRequested.create("sensitive task containing migration-fail in prompt")

    with pytest.raises(ResearchExecutionError):
        await processor.run_once(task, assignment_id="assign-mig-r2", assignment_proof="proof-mig-r2")

    assert len(client.submitted) == 1
    command = client.submitted[0]
    assert command.failure_code in ALLOWED_COMPLETION_FAILURE_CODES
    assert command.completed_result is None
    assert "Simulated execution failure" not in command.to_json()
    assert "sensitive task containing" not in command.to_json()


@pytest.mark.anyio
async def test_analysis_delegated_child_submits_completed_command() -> None:
    from analysis_agent.application.processor import AnalysisProcessor
    from analysis_agent.infrastructure.deterministic_analyzer import (
        DeterministicMaterialAnalyzer,
    )

    from tests.test_analysis_agent import FakeCompletionClient, FakeMessageBus

    bus = FakeMessageBus()
    client = FakeCompletionClient()
    processor = AnalysisProcessor(
        "analysis-agent-1",
        bus,
        DeterministicMaterialAnalyzer(),
        completion_client=client,
    )
    task = TaskRequested.create("migration analysis material")

    await processor.run_once(task, assignment_id="assign-mig-a1", assignment_proof="proof-mig-a1")

    assert len(client.submitted) == 1
    command = client.submitted[0]
    assert command.execution_kind == "delegated_child"
    assert command.completed_result is not None
    assert command.completed_result.task_id == task.task_id
    assert command.failure_code is None


@pytest.mark.anyio
async def test_analysis_failure_submits_only_allowlisted_failure_code() -> None:
    from agent_contracts import ALLOWED_COMPLETION_FAILURE_CODES
    from analysis_agent.application.processor import AnalysisExecutionError, AnalysisProcessor

    from tests.test_analysis_agent import FakeCompletionClient, FakeMessageBus

    class FailingAnalyzer:
        def analyze(self, material: str) -> object:
            raise RuntimeError("analysis backend exploded")

    bus = FakeMessageBus()
    client = FakeCompletionClient()
    processor = AnalysisProcessor(
        "analysis-agent-1", bus, FailingAnalyzer(), completion_client=client
    )
    task = TaskRequested.create("unanalyzable migration material")

    with pytest.raises(AnalysisExecutionError):
        await processor.run_once(task, assignment_id="assign-mig-a2", assignment_proof="proof-mig-a2")

    assert len(client.submitted) == 1
    command = client.submitted[0]
    assert command.execution_kind == "delegated_child"
    assert command.failure_code in ALLOWED_COMPLETION_FAILURE_CODES
    assert command.completed_result is None
    assert "analysis backend exploded" not in command.to_json()
