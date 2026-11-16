import pytest
from agent_contracts import AgentEvent, TaskRequested, TaskResult
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


class FakeResultStore:
    def __init__(self) -> None:
        self.saved: list[TaskResult] = []

    async def save(self, result: TaskResult) -> None:
        self.saved.append(result)


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
async def test_analysis_processor_persists_result_between_working_and_idle() -> None:
    bus = FakeMessageBus()
    store = FakeResultStore()
    task = TaskRequested.create("The rover maps caves. The rover records samples?")
    processor = AnalysisProcessor(
        "analysis-agent-1",
        bus,
        DeterministicMaterialAnalyzer(),
        store,
    )

    await processor.run_once(task)

    assert [event.state for event in bus.published] == ["working", "idle"]
    assert [event.task_id for event in bus.published] == [task.task_id, task.task_id]
    assert store.saved == [
        TaskResult(
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
    ]


@pytest.mark.anyio
async def test_analysis_processor_failure_emits_error_without_idle_or_result() -> None:
    class FailingAnalyzer:
        def analyze(self, material: str) -> AnalysisReport:
            raise RuntimeError("analysis failed")

    bus = FakeMessageBus()
    store = FakeResultStore()
    task = TaskRequested.create("Material that cannot be analyzed.")
    processor = AnalysisProcessor("analysis-agent-1", bus, FailingAnalyzer(), store)

    with pytest.raises(AnalysisExecutionError, match=task.task_id):
        await processor.run_once(task)

    assert [event.state for event in bus.published] == ["working", "error"]
    assert [event.task_id for event in bus.published] == [task.task_id, task.task_id]
    assert store.saved == []
