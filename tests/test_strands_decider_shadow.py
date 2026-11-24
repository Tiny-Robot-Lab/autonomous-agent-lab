"""Deterministic tests for bounded Strands Decider shadow routing (Issue #237)."""

from __future__ import annotations

import json
import urllib.error
from typing import Self

import pytest
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.router import DeterministicCapabilityRouter

from evaluation.routing.strands_shadow import (
    CANDIDATE_CHOICES,
    CANONICAL_SHADOW_INPUT_VERSION,
    DEFAULT_SHADOW_CONFIDENCE_THRESHOLD,
    DEFAULT_STRANDS_DECIDER_ENDPOINT,
    DEFAULT_STRANDS_DECIDER_MODEL_ID,
    STRANDS_ROUTE_CRITERIA,
    STRANDS_ROUTE_INSTRUCTIONS,
    CanonicalShadowInput,
    DeciderMalformedResponseError,
    DeciderRawResult,
    DeciderTimeoutError,
    DeciderUnavailableError,
    LocalStrandsDeciderHttpAdapter,
    build_canonical_shadow_input,
    build_systemone_request_payload,
    evaluate_shadow_case,
    load_shadow_evaluation_corpus,
    run_shadow_corpus_evaluation,
    validate_loopback_endpoint,
)


class _FakeHttpResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class StubDeciderAdapter:
    """Deterministic stub adapter for testing shadow-routing behavior."""

    def __init__(self, handler, model_id: str = "stub-decider-v1") -> None:
        self._handler = handler
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def decide(self, canonical_input: CanonicalShadowInput) -> DeciderRawResult:
        return self._handler(canonical_input)


def test_canonical_shadow_input_deterministic_serialization() -> None:
    inp1 = build_canonical_shadow_input("eval-001", "  What is 15% of 850?  ")
    inp2 = build_canonical_shadow_input("eval-001", "What is 15% of 850?")

    assert inp1.version == CANONICAL_SHADOW_INPUT_VERSION
    assert inp1.candidate_choices == CANDIDATE_CHOICES
    assert inp1.to_canonical_json() == inp2.to_canonical_json()
    assert '"version":"shadow_routing_input.v1"' in inp1.to_canonical_json()
    assert '"candidate_choices":["general","research","none"]' in inp1.to_canonical_json()


def test_canonical_shadow_input_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unsupported canonical shadow input version"):
        CanonicalShadowInput(case_id="c1", prompt="hello", version="v0")

    with pytest.raises(ValueError, match="case_id must be a non-empty string"):
        build_canonical_shadow_input("   ", "hello")

    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        build_canonical_shadow_input("c1", "   ")

    with pytest.raises(ValueError, match="candidate_choices must match"):
        CanonicalShadowInput(
            case_id="c1",
            prompt="hello",
            candidate_choices=("general", "research"),
        )


def test_shadow_disabled_preserves_authoritative_route_without_calling_adapter() -> None:
    called = False

    def should_not_run(_inp: CanonicalShadowInput) -> DeciderRawResult:
        nonlocal called
        called = True
        return DeciderRawResult(choice="research", confidence=0.99)

    router = DeterministicCapabilityRouter()
    prompt = "What is 15% of 850?"
    expected = router.route(prompt)

    route, obs = evaluate_shadow_case(
        "eval-001",
        prompt,
        router=router,
        adapter=StubDeciderAdapter(should_not_run),
        enabled=False,
    )

    assert called is False
    assert route == expected == AgentCapability.GENERAL
    assert obs.execution_status == "disabled"
    assert obs.comparison == "disabled"
    assert obs.effective_shadow_decision is None
    assert obs.raw_decider_choice is None
    assert obs.confidence is None
    assert obs.authoritative_route == "general"


def test_local_http_adapter_loopback_endpoint_validation() -> None:
    assert validate_loopback_endpoint(DEFAULT_STRANDS_DECIDER_ENDPOINT) == "http://127.0.0.1:8099/v1/systemone"
    assert validate_loopback_endpoint("http://localhost:8099/v1/systemone") == "http://localhost:8099/v1/systemone"
    assert validate_loopback_endpoint("http://[::1]:8099/v1/systemone") == "http://[::1]:8099/v1/systemone"

    for bad_url in (
        "http://example.com:8099/v1/systemone",
        "http://10.0.0.5:8099/v1/systemone",
        "ftp://127.0.0.1:8099/v1/systemone",
        "http://user:secret@127.0.0.1:8099/v1/systemone",
        "   ",
    ):
        with pytest.raises(ValueError):
            LocalStrandsDeciderHttpAdapter(endpoint_url=bad_url)


def test_local_http_adapter_request_payload_and_response_parsing() -> None:
    captured: dict[str, object] = {}

    def fake_opener(req, timeout: float):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeHttpResponse(
            json.dumps({
                "model": DEFAULT_STRANDS_DECIDER_MODEL_ID,
                "latency_ms": 42.5,
                "usage": {"prompt_tokens": 25, "total_tokens": 26},
                "answers": {
                    "route": {
                        "choice": "research",
                        "confidence": 0.91,
                        "probabilities": {
                            "general": 0.07,
                            "research": 0.91,
                            "none": 0.02,
                        },
                    }
                },
            })
        )

    router = DeterministicCapabilityRouter()
    prompt = "What are the latest pricing tiers for Claude 3.5 Sonnet on Anthropic's API?"
    adapter = LocalStrandsDeciderHttpAdapter(opener=fake_opener)
    assert adapter.model_id == DEFAULT_STRANDS_DECIDER_MODEL_ID == "StrandsAgents/strands-decider-E2B-gemma4-v2-2611"

    route, obs = evaluate_shadow_case(
        "eval-006",
        prompt,
        router=router,
        adapter=adapter,
        enabled=True,
    )

    assert captured["url"] == DEFAULT_STRANDS_DECIDER_ENDPOINT
    assert captured["method"] == "POST"
    assert captured["payload"] == build_systemone_request_payload(
        build_canonical_shadow_input("eval-006", prompt),
        model_id=DEFAULT_STRANDS_DECIDER_MODEL_ID,
    )
    assert captured["payload"]["questions"]["route"]["instructions"] == STRANDS_ROUTE_INSTRUCTIONS == "Which Tiny Agent Lab route should handle this request?"
    assert captured["payload"]["questions"]["route"]["criteria"] == STRANDS_ROUTE_CRITERIA
    assert route == AgentCapability.RESEARCH
    assert obs.execution_status == "succeeded"
    assert obs.comparison == "agreed"
    assert obs.raw_decider_choice == "research"
    assert obs.confidence == 0.91
    assert obs.probabilities == {"general": 0.07, "research": 0.91, "none": 0.02}
    assert obs.server_latency_ms == 42.5
    assert obs.latency_ns > 0
    assert obs.model_id == DEFAULT_STRANDS_DECIDER_MODEL_ID


def test_local_http_adapter_handles_unavailable_timeout_and_malformed_json() -> None:
    router = DeterministicCapabilityRouter()
    prompt = "What are the latest pricing tiers for Claude 3.5 Sonnet on Anthropic's API?"

    def refused_opener(_req, timeout: float):
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    route_unavail, obs_unavail = evaluate_shadow_case(
        "eval-006",
        prompt,
        router=router,
        adapter=LocalStrandsDeciderHttpAdapter(opener=refused_opener),
    )
    assert route_unavail == AgentCapability.RESEARCH
    assert obs_unavail.execution_status == "unavailable"
    assert obs_unavail.comparison == "execution_failed"
    assert obs_unavail.effective_shadow_decision is None

    def timeout_opener(_req, timeout: float):
        raise TimeoutError("timed out")

    route_to, obs_to = evaluate_shadow_case(
        "eval-006",
        prompt,
        router=router,
        adapter=LocalStrandsDeciderHttpAdapter(opener=timeout_opener),
    )
    assert route_to == AgentCapability.RESEARCH
    assert obs_to.execution_status == "timeout"
    assert obs_to.effective_shadow_decision is None

    def malformed_opener(_req, timeout: float):
        return _FakeHttpResponse("{not-valid-json")

    route_bad, obs_bad = evaluate_shadow_case(
        "eval-006",
        prompt,
        router=router,
        adapter=LocalStrandsDeciderHttpAdapter(opener=malformed_opener),
    )
    assert route_bad == AgentCapability.RESEARCH
    assert obs_bad.execution_status == "malformed"
    assert obs_bad.effective_shadow_decision is None


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (DeciderUnavailableError("model weights missing"), "unavailable"),
        (DeciderTimeoutError("deadline exceeded"), "timeout"),
        (TimeoutError("socket timeout"), "timeout"),
        (DeciderMalformedResponseError("invalid json"), "malformed"),
    ],
)
def test_decider_failures_never_convert_to_none_or_alter_authoritative_route(
    exc: Exception,
    expected_status: str,
) -> None:
    def failing_handler(_inp: CanonicalShadowInput) -> DeciderRawResult:
        raise exc

    router = DeterministicCapabilityRouter()
    prompt = "Find recent developments from the latest UN Climate Conference."
    route, obs = evaluate_shadow_case(
        "eval-007",
        prompt,
        router=router,
        adapter=StubDeciderAdapter(failing_handler),
    )

    assert route == AgentCapability.RESEARCH
    assert obs.authoritative_route == "research"
    assert obs.execution_status == expected_status
    assert obs.comparison == "execution_failed"
    assert obs.effective_shadow_decision is None
    assert obs.failure_reason == str(exc)


@pytest.mark.parametrize(
    "bad_result",
    [
        "not-a-result",
        DeciderRawResult(choice="", confidence=0.8),
        DeciderRawResult(choice="general", confidence=float("nan")),
        DeciderRawResult(choice="general", confidence=1.2),
        DeciderRawResult(choice="general", confidence=-0.1),
        DeciderRawResult(choice="general", confidence=True),  # type: ignore[arg-type]
    ],
)
def test_malformed_raw_results_marked_malformed_and_authoritative_unchanged(
    bad_result: object,
) -> None:
    router = DeterministicCapabilityRouter()
    route, obs = evaluate_shadow_case(
        "eval-001",
        "What is 15% of 850?",
        router=router,
        adapter=StubDeciderAdapter(lambda _inp: bad_result),
    )

    assert route == AgentCapability.GENERAL
    assert obs.execution_status == "malformed"
    assert obs.comparison == "execution_failed"
    assert obs.effective_shadow_decision is None


def test_unknown_choice_distinct_from_deliberate_none_and_failures() -> None:
    router = DeterministicCapabilityRouter()
    route, obs = evaluate_shadow_case(
        "eval-001",
        "What is 15% of 850?",
        router=router,
        adapter=StubDeciderAdapter(
            lambda _inp: DeciderRawResult(choice="analysis", confidence=0.95)
        ),
    )

    assert route == AgentCapability.GENERAL
    assert obs.execution_status == "unknown_choice"
    assert obs.comparison == "execution_failed"
    assert obs.raw_decider_choice == "analysis"
    assert obs.confidence == 0.95
    assert obs.effective_shadow_decision is None
    assert obs.failure_reason is not None and "Unknown candidate choice" in obs.failure_reason


def test_below_threshold_vs_deliberate_none_distinction() -> None:
    router = DeterministicCapabilityRouter()
    prompt = "Should I use PostgreSQL or MongoDB for my new web application?"

    route_low, obs_low = evaluate_shadow_case(
        "eval-011",
        prompt,
        router=router,
        adapter=StubDeciderAdapter(
            lambda _inp: DeciderRawResult(choice="research", confidence=0.69)
        ),
        threshold=DEFAULT_SHADOW_CONFIDENCE_THRESHOLD,
    )
    assert route_low == AgentCapability.GENERAL
    assert obs_low.execution_status == "succeeded"
    assert obs_low.raw_decider_choice == "research"
    assert obs_low.confidence == 0.69
    assert obs_low.threshold == 0.70
    assert obs_low.effective_shadow_decision == "none"
    assert obs_low.comparison == "below_threshold"

    route_none, obs_none = evaluate_shadow_case(
        "shadow-none-001",
        "Handle this request appropriately.",
        router=router,
        adapter=StubDeciderAdapter(
            lambda _inp: DeciderRawResult(choice="none", confidence=0.88)
        ),
        threshold=DEFAULT_SHADOW_CONFIDENCE_THRESHOLD,
    )
    assert route_none == AgentCapability.GENERAL
    assert obs_none.execution_status == "succeeded"
    assert obs_none.raw_decider_choice == "none"
    assert obs_none.confidence == 0.88
    assert obs_none.effective_shadow_decision == "none"
    assert obs_none.comparison == "deliberate_none"


def test_disagreement_never_alters_authoritative_route() -> None:
    router = DeterministicCapabilityRouter()
    prompt = "What is 15% of 850?"
    route, obs = evaluate_shadow_case(
        "eval-001",
        prompt,
        router=router,
        adapter=StubDeciderAdapter(
            lambda _inp: DeciderRawResult(choice="research", confidence=0.92)
        ),
    )

    assert route == AgentCapability.GENERAL
    assert obs.authoritative_route == "general"
    assert obs.raw_decider_choice == "research"
    assert obs.effective_shadow_decision == "research"
    assert obs.comparison == "disagreed"
    assert obs.execution_status == "succeeded"


def test_programming_error_in_adapter_not_swallowed_as_classifier_failure() -> None:
    def buggy_adapter(_inp: CanonicalShadowInput) -> DeciderRawResult:
        raise ZeroDivisionError("unexpected internal bug")

    with pytest.raises(ZeroDivisionError, match="unexpected internal bug"):
        evaluate_shadow_case(
            "eval-001",
            "What is 15% of 850?",
            adapter=StubDeciderAdapter(buggy_adapter),
        )


def test_shadow_corpus_loads_and_reproduces_cleanly() -> None:
    corpus = load_shadow_evaluation_corpus()
    assert len(corpus) == 17
    ids = {c.id for c in corpus}
    assert {"eval-001", "eval-006", "eval-011", "val-001", "val-025", "shadow-none-001"} <= ids

    def deterministic_stub(inp: CanonicalShadowInput) -> DeciderRawResult:
        if inp.case_id.startswith("shadow-none"):
            return DeciderRawResult(choice="none", confidence=0.85)
        if inp.case_id in {"eval-006", "eval-007", "val-005", "val-012", "val-025"}:
            return DeciderRawResult(choice="research", confidence=0.90)
        if inp.case_id in {"eval-013", "eval-014"}:
            return DeciderRawResult(choice="research", confidence=0.55)
        return DeciderRawResult(choice="general", confidence=0.85)

    adapter = StubDeciderAdapter(deterministic_stub)
    report_1 = run_shadow_corpus_evaluation(adapter=adapter)
    report_2 = run_shadow_corpus_evaluation(adapter=adapter)

    assert report_1.total_cases == 17
    assert report_1.deliberate_none_count == 2
    assert report_1.below_threshold_count == 2
    assert report_1.execution_failed_count == 0
    assert [
        (o.case_id, o.canonical_input_json, o.authoritative_route, o.effective_shadow_decision, o.comparison)
        for o in report_1.observations
    ] == [
        (o.case_id, o.canonical_input_json, o.authoritative_route, o.effective_shadow_decision, o.comparison)
        for o in report_2.observations
    ]


def test_corpus_runner_restricts_to_repository_controlled_offline_corpus() -> None:
    adapter = StubDeciderAdapter(lambda _inp: DeciderRawResult(choice="general", confidence=0.90))

    subset = run_shadow_corpus_evaluation(case_ids=("eval-001", "eval-006"), adapter=adapter)
    assert subset.total_cases == 2
    assert [o.case_id for o in subset.observations] == ["eval-001", "eval-006"]

    with pytest.raises(ValueError, match="only repository-controlled offline corpus cases are supported"):
        run_shadow_corpus_evaluation(case_ids=("arbitrary-live-prompt-1",), adapter=adapter)

    with pytest.raises(TypeError):
        run_shadow_corpus_evaluation([], adapter=adapter)  # type: ignore[misc]
