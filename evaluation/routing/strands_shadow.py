"""Bounded shadow-evaluation harness for Strands Decider routing experiments (Issue #237).

The existing DeterministicCapabilityRouter remains strictly authoritative.
Strands Decider runs only in shadow mode against a bounded, deterministically
serialized canonical input and never alters authoritative routing, task state,
permissions, delegation, or terminal outcomes.
"""

from __future__ import annotations

import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from agent_contracts.serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
)
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.router import DeterministicCapabilityRouter

from evaluation.routing.models import EvaluationCase

CANONICAL_SHADOW_INPUT_VERSION = "shadow_routing_input.v1"
SHADOW_CHOICE_NONE = "none"
SPECIALIST_CHOICES: frozenset[str] = frozenset({
    AgentCapability.GENERAL.value,
    AgentCapability.RESEARCH.value,
})
CANDIDATE_CHOICES: tuple[str, ...] = (
    AgentCapability.GENERAL.value,
    AgentCapability.RESEARCH.value,
    SHADOW_CHOICE_NONE,
)

# Provisional experiment-only threshold frozen prior to corpus evaluation.
# In a 3-way classification (general, research, none), 0.70 requires clear
# majority calibrated confidence before accepting a specialist shadow choice.
DEFAULT_SHADOW_CONFIDENCE_THRESHOLD = 0.70

DeciderExecutionStatus = Literal[
    "disabled",
    "succeeded",
    "unavailable",
    "timeout",
    "malformed",
    "unknown_choice",
]
ShadowComparisonOutcome = Literal[
    "disabled",
    "agreed",
    "disagreed",
    "deliberate_none",
    "below_threshold",
    "execution_failed",
]

# Default Strands Decider checkpoint and bounded loopback HTTP endpoint for issue #237.
DEFAULT_STRANDS_DECIDER_MODEL_ID = "StrandsAgents/strands-decider-E2B-gemma4-v2-2611"
DEFAULT_STRANDS_DECIDER_ENDPOINT = "http://127.0.0.1:8099/v1/systemone"
ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
STRANDS_ROUTE_INSTRUCTIONS = "Which Tiny Agent Lab route should handle this request?"
STRANDS_ROUTE_CRITERIA: dict[str, str] = {
    AgentCapability.GENERAL.value: "general task that does not require research",
    AgentCapability.RESEARCH.value: "task requiring research or external evidence",
    SHADOW_CHOICE_NONE: "none of the offered specialist routes is appropriate",
}

# Representative repository-grounded case IDs from dataset.json and validation_dataset.json.
DEFAULT_DATASET_CASE_IDS: tuple[str, ...] = (
    "eval-001",  # clear_general: arithmetic
    "eval-002",  # clear_general: python slicing
    "eval-006",  # clear_research: live pricing
    "eval-007",  # clear_research: latest UN Climate Conference
    "eval-011",  # ambiguous_borderline: PostgreSQL vs MongoDB
    "eval-013",  # ambiguous_borderline: quantum computing advancements
    "eval-014",  # ambiguous_borderline: AI impacting software engineering jobs
)
DEFAULT_VALIDATION_CASE_IDS: tuple[str, ...] = (
    "val-001",  # weak_verb_supplied_artifact: compare config files
    "val-003",  # weak_verb_supplied_artifact: research infinite loop
    "val-005",  # freshness_current_info: today's AWS Lambda pricing
    "val-012",  # changing_prices_availability: recent GitHub Actions outages
    "val-015",  # historical_established_advances: database indexing theory
    "val-020",  # negative_current_electrical: direct vs alternating current
    "val-025",  # mixed_supplied_and_external: database config vs cloud vendor limits
    "val-028",  # weak_verb_alone: benchmark redis throughput
)


class DeciderError(Exception):
    """Base exception for Strands Decider shadow-adapter failures."""


class DeciderUnavailableError(DeciderError):
    """Raised when Strands Decider package, runtime, or model is unavailable."""


class DeciderTimeoutError(DeciderError):
    """Raised when Strands Decider exceeds its execution deadline."""


class DeciderMalformedResponseError(DeciderError):
    """Raised when Strands Decider returns a malformed or invalid payload."""


@dataclass(frozen=True)
class CanonicalShadowInput:
    """Bounded, deterministically serialized input for shadow routing classification.

    Excludes credentials, tokens, proofs, sealed artifacts, unrestricted
    conversation history, and private platform state.
    """

    case_id: str
    prompt: str
    version: str = CANONICAL_SHADOW_INPUT_VERSION
    candidate_choices: tuple[str, ...] = CANDIDATE_CHOICES
    has_supplied_artifact_context: bool = False

    def __post_init__(self) -> None:
        if self.version != CANONICAL_SHADOW_INPUT_VERSION:
            raise ValueError(
                f"Unsupported canonical shadow input version: {self.version!r}"
            )
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if type(self.has_supplied_artifact_context) is not bool:
            raise TypeError("has_supplied_artifact_context must be a bool")
        if tuple(self.candidate_choices) != CANDIDATE_CHOICES:
            raise ValueError(
                f"candidate_choices must match {CANDIDATE_CHOICES!r}, got {self.candidate_choices!r}"
            )

        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "prompt", self.prompt.strip())
        object.__setattr__(self, "candidate_choices", tuple(self.candidate_choices))

        payload = self.to_dict()
        check_no_credential_bearing_keys(payload, context="CanonicalShadowInput")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_choices": list(self.candidate_choices),
            "case_id": self.case_id,
            "has_supplied_artifact_context": self.has_supplied_artifact_context,
            "prompt": self.prompt,
            "version": self.version,
        }

    def to_canonical_json(self) -> str:
        """Return deterministic canonical UTF-8 JSON representation."""
        return canonical_json_dumps(self.to_dict())


@dataclass(frozen=True)
class DeciderRawResult:
    """Raw output returned by a shadow Decider adapter before thresholding."""

    choice: str
    confidence: float
    model_id: str = ""
    probabilities: dict[str, float] | None = None
    server_latency_ms: float | None = None
    usage: dict[str, Any] | None = None


class ShadowDeciderAdapter(Protocol):
    """Adapter boundary isolating Strands Decider from Tiny Agent Lab routing."""

    @property
    def model_id(self) -> str: ...

    def decide(self, canonical_input: CanonicalShadowInput) -> DeciderRawResult: ...


def validate_loopback_endpoint(endpoint_url: str) -> str:
    """Validate that the Decider HTTP endpoint targets a permitted loopback host."""
    if not isinstance(endpoint_url, str) or not endpoint_url.strip():
        raise ValueError("endpoint_url must be a non-empty string")
    cleaned = endpoint_url.strip()
    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"endpoint_url scheme must be http or https, got {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint_url must not contain credentials")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_LOOPBACK_HOSTS:
        raise ValueError(
            f"Only loopback endpoints {sorted(ALLOWED_LOOPBACK_HOSTS)} are permitted, got {host!r}"
        )
    return cleaned


def build_systemone_request_payload(
    canonical_input: CanonicalShadowInput,
    *,
    model_id: str = DEFAULT_STRANDS_DECIDER_MODEL_ID,
) -> dict[str, Any]:
    """Map CanonicalShadowInput to the Strands /v1/systemone choice request payload."""
    return {
        "model": model_id,
        "state": canonical_input.prompt,
        "questions": {
            "route": {
                "type": "choice",
                "instructions": STRANDS_ROUTE_INSTRUCTIONS,
                "criteria": dict(STRANDS_ROUTE_CRITERIA),
            }
        },
    }


def _parse_systemone_response(
    raw_payload: Any,
    *,
    default_model_id: str,
) -> DeciderRawResult:
    """Strictly validate and parse untrusted /v1/systemone JSON response."""
    if not isinstance(raw_payload, dict):
        raise DeciderMalformedResponseError(
            f"Expected dict response from Decider, got {type(raw_payload).__name__}"
        )

    answers = raw_payload.get("answers")
    if not isinstance(answers, dict):
        raise DeciderMalformedResponseError("Decider response missing dict 'answers'")

    route_answer = answers.get("route")
    if not isinstance(route_answer, dict):
        raise DeciderMalformedResponseError("Decider response missing dict 'answers.route'")

    choice = route_answer.get("choice")
    if not isinstance(choice, str) or not choice.strip():
        raise DeciderMalformedResponseError(
            "Decider response missing valid string 'answers.route.choice'"
        )

    confidence = route_answer.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise DeciderMalformedResponseError(
            f"Decider response 'answers.route.confidence' must be numeric, got {type(confidence).__name__}"
        )
    conf_float = float(confidence)
    if not math.isfinite(conf_float) or conf_float < 0.0 or conf_float > 1.0:
        raise DeciderMalformedResponseError(
            f"Decider response 'answers.route.confidence' out of [0.0, 1.0]: {conf_float!r}"
        )

    probabilities: dict[str, float] | None = None
    raw_probs = route_answer.get("probabilities")
    if raw_probs is not None:
        if not isinstance(raw_probs, dict):
            raise DeciderMalformedResponseError(
                "Decider response 'answers.route.probabilities' must be a dict when present"
            )
        parsed_probs: dict[str, float] = {}
        for k, v in raw_probs.items():
            if not isinstance(k, str) or not k.strip():
                raise DeciderMalformedResponseError("Probability key must be a non-empty string")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise DeciderMalformedResponseError(
                    f"Probability value for {k!r} must be numeric, got {type(v).__name__}"
                )
            p_val = float(v)
            if not math.isfinite(p_val) or p_val < 0.0 or p_val > 1.0:
                raise DeciderMalformedResponseError(
                    f"Probability value for {k!r} out of [0.0, 1.0]: {p_val!r}"
                )
            parsed_probs[k] = p_val
        probabilities = parsed_probs

    server_latency_ms: float | None = None
    raw_latency = raw_payload.get("latency_ms")
    if raw_latency is not None:
        if isinstance(raw_latency, bool) or not isinstance(raw_latency, (int, float)):
            raise DeciderMalformedResponseError("Decider response 'latency_ms' must be numeric")
        lat_val = float(raw_latency)
        if not math.isfinite(lat_val) or lat_val < 0.0:
            raise DeciderMalformedResponseError(
                f"Decider response 'latency_ms' must be finite and non-negative, got {lat_val!r}"
            )
        server_latency_ms = lat_val

    usage: dict[str, Any] | None = None
    raw_usage = raw_payload.get("usage")
    if raw_usage is not None:
        if not isinstance(raw_usage, dict):
            raise DeciderMalformedResponseError("Decider response 'usage' must be a dict")
        usage = dict(raw_usage)

    raw_model = raw_payload.get("model")
    if raw_model is not None:
        if not isinstance(raw_model, str) or not raw_model.strip():
            raise DeciderMalformedResponseError(
                "Decider response 'model' must be a non-empty string when present"
            )
        model_id = raw_model.strip()
    else:
        model_id = default_model_id

    return DeciderRawResult(
        choice=choice.strip(),
        confidence=conf_float,
        model_id=model_id,
        probabilities=probabilities,
        server_latency_ms=server_latency_ms,
        usage=usage,
    )


class LocalStrandsDeciderHttpAdapter:
    """Bounded localhost HTTP adapter calling `POST /v1/systemone`.

    Uses Python standard-library HTTP facilities and enforces loopback-only
    endpoint validation. Does not manage the external Decider server lifecycle.
    """

    def __init__(
        self,
        endpoint_url: str = DEFAULT_STRANDS_DECIDER_ENDPOINT,
        model_id: str = DEFAULT_STRANDS_DECIDER_MODEL_ID,
        timeout_seconds: float = 30.0,
        opener: Any | None = None,
    ) -> None:
        self._endpoint_url = validate_loopback_endpoint(endpoint_url)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive number")
        self._model_id = model_id.strip()
        self._timeout_seconds = float(timeout_seconds)
        self._opener = opener or urllib.request.urlopen

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def endpoint_url(self) -> str:
        return self._endpoint_url

    def decide(self, canonical_input: CanonicalShadowInput) -> DeciderRawResult:
        payload = build_systemone_request_payload(
            canonical_input, model_id=self._model_id
        )
        req = urllib.request.Request(
            self._endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(req, timeout=self._timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except TimeoutError as exc:
            raise DeciderTimeoutError(f"Strands Decider HTTP request timed out: {exc}") from exc
        except urllib.error.HTTPError as exc:
            raise DeciderUnavailableError(
                f"Strands Decider HTTP server returned status {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise DeciderTimeoutError(
                    f"Strands Decider HTTP request timed out: {exc.reason}"
                ) from exc
            raise DeciderUnavailableError(
                f"Strands Decider HTTP server unavailable at {self._endpoint_url}: {exc.reason}"
            ) from exc
        except (http.client.HTTPException, OSError) as exc:
            raise DeciderUnavailableError(
                f"Strands Decider HTTP transport failure: {exc}"
            ) from exc

        try:
            raw_payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DeciderMalformedResponseError(
                f"Strands Decider returned invalid JSON: {exc}"
            ) from exc

        return _parse_systemone_response(raw_payload, default_model_id=self._model_id)


@dataclass(frozen=True)
class ShadowRoutingObservation:
    """Bounded telemetry/comparison record for one shadow routing evaluation."""

    case_id: str
    input_version: str
    canonical_input_json: str
    authoritative_route: str
    raw_decider_choice: str | None
    confidence: float | None
    threshold: float
    effective_shadow_decision: str | None
    comparison: ShadowComparisonOutcome
    execution_status: DeciderExecutionStatus
    failure_reason: str | None
    latency_ns: int
    probabilities: dict[str, float] | None = None
    server_latency_ms: float | None = None
    model_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "authoritative_route": self.authoritative_route,
            "canonical_input_json": self.canonical_input_json,
            "case_id": self.case_id,
            "comparison": self.comparison,
            "confidence": self.confidence,
            "effective_shadow_decision": self.effective_shadow_decision,
            "execution_status": self.execution_status,
            "failure_reason": self.failure_reason,
            "input_version": self.input_version,
            "latency_ns": self.latency_ns,
            "model_id": self.model_id,
            "probabilities": dict(self.probabilities) if self.probabilities is not None else None,
            "raw_decider_choice": self.raw_decider_choice,
            "server_latency_ms": self.server_latency_ms,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class ShadowCorpusReport:
    """Aggregated comparison metrics across a shadow-evaluation corpus."""

    threshold: float
    input_version: str
    total_cases: int
    agreed_count: int
    disagreed_count: int
    deliberate_none_count: int
    below_threshold_count: int
    execution_failed_count: int
    disabled_count: int
    observations: tuple[ShadowRoutingObservation, ...]


def build_canonical_shadow_input(
    case_id: str,
    prompt: str,
    *,
    has_supplied_artifact_context: bool = False,
) -> CanonicalShadowInput:
    """Construct a validated, deterministically serializable shadow input."""
    return CanonicalShadowInput(
        case_id=case_id,
        prompt=prompt,
        has_supplied_artifact_context=has_supplied_artifact_context,
    )


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a float in [0.0, 1.0]")
    value = float(threshold)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"threshold must be finite and in [0.0, 1.0], got {value!r}")
    return value


def _validate_raw_result(raw_result: DeciderRawResult) -> tuple[str, float]:
    if not isinstance(raw_result, DeciderRawResult):
        raise DeciderMalformedResponseError(
            f"Expected DeciderRawResult, got {type(raw_result).__name__}"
        )
    if not isinstance(raw_result.choice, str) or not raw_result.choice.strip():
        raise DeciderMalformedResponseError("DeciderRawResult.choice must be a non-empty string")
    if isinstance(raw_result.confidence, bool) or not isinstance(
        raw_result.confidence, (int, float)
    ):
        raise DeciderMalformedResponseError("DeciderRawResult.confidence must be numeric")
    conf = float(raw_result.confidence)
    if not math.isfinite(conf) or conf < 0.0 or conf > 1.0:
        raise DeciderMalformedResponseError(
            f"DeciderRawResult.confidence must be in [0.0, 1.0], got {conf!r}"
        )
    return raw_result.choice.strip(), conf


def evaluate_shadow_case(
    case_id: str,
    prompt: str,
    *,
    router: DeterministicCapabilityRouter | None = None,
    adapter: ShadowDeciderAdapter | None = None,
    enabled: bool = True,
    threshold: float = DEFAULT_SHADOW_CONFIDENCE_THRESHOLD,
    has_supplied_artifact_context: bool = False,
) -> tuple[AgentCapability, ShadowRoutingObservation]:
    """Run authoritative deterministic routing and separate shadow Decider observation.

    Returns (authoritative_route, ShadowRoutingObservation). The shadow path
    never alters authoritative_route.
    """
    validated_threshold = _validate_threshold(threshold)
    active_router = router or DeterministicCapabilityRouter()
    authoritative_route = active_router.route(prompt)

    canonical_input = build_canonical_shadow_input(
        case_id=case_id,
        prompt=prompt,
        has_supplied_artifact_context=has_supplied_artifact_context,
    )
    canonical_json = canonical_input.to_canonical_json()

    def _obs(
        *,
        raw_choice: str | None = None,
        confidence: float | None = None,
        effective: str | None = None,
        comparison: ShadowComparisonOutcome,
        status: DeciderExecutionStatus,
        failure_reason: str | None = None,
        latency_ns: int = 0,
        probabilities: dict[str, float] | None = None,
        server_latency_ms: float | None = None,
        model_id: str | None = None,
    ) -> ShadowRoutingObservation:
        return ShadowRoutingObservation(
            case_id=canonical_input.case_id,
            input_version=canonical_input.version,
            canonical_input_json=canonical_json,
            authoritative_route=authoritative_route.value,
            raw_decider_choice=raw_choice,
            confidence=confidence,
            threshold=validated_threshold,
            effective_shadow_decision=effective,
            comparison=comparison,
            execution_status=status,
            failure_reason=failure_reason,
            latency_ns=latency_ns,
            probabilities=probabilities,
            server_latency_ms=server_latency_ms,
            model_id=model_id,
        )

    if not enabled:
        return authoritative_route, _obs(comparison="disabled", status="disabled")

    if adapter is None:
        return authoritative_route, _obs(
            comparison="execution_failed",
            status="unavailable",
            failure_reason="No Strands Decider adapter configured",
        )

    start_ns = time.perf_counter_ns()
    try:
        raw_result = adapter.decide(canonical_input)
        elapsed_ns = time.perf_counter_ns() - start_ns
        raw_choice, conf = _validate_raw_result(raw_result)
    except DeciderUnavailableError as exc:
        elapsed_ns = time.perf_counter_ns() - start_ns
        return authoritative_route, _obs(
            comparison="execution_failed",
            status="unavailable",
            failure_reason=str(exc),
            latency_ns=elapsed_ns,
            model_id=adapter.model_id,
        )
    except (DeciderTimeoutError, TimeoutError) as exc:
        elapsed_ns = time.perf_counter_ns() - start_ns
        return authoritative_route, _obs(
            comparison="execution_failed",
            status="timeout",
            failure_reason=str(exc),
            latency_ns=elapsed_ns,
            model_id=adapter.model_id,
        )
    except DeciderMalformedResponseError as exc:
        elapsed_ns = time.perf_counter_ns() - start_ns
        return authoritative_route, _obs(
            comparison="execution_failed",
            status="malformed",
            failure_reason=str(exc),
            latency_ns=elapsed_ns,
            model_id=adapter.model_id,
        )

    resolved_model_id = raw_result.model_id or adapter.model_id

    if raw_choice not in CANDIDATE_CHOICES:
        return authoritative_route, _obs(
            raw_choice=raw_choice,
            confidence=conf,
            comparison="execution_failed",
            status="unknown_choice",
            failure_reason=f"Unknown candidate choice {raw_choice!r}; expected one of {CANDIDATE_CHOICES}",
            latency_ns=elapsed_ns,
            probabilities=raw_result.probabilities,
            server_latency_ms=raw_result.server_latency_ms,
            model_id=resolved_model_id,
        )

    if raw_choice == SHADOW_CHOICE_NONE:
        return authoritative_route, _obs(
            raw_choice=raw_choice,
            confidence=conf,
            effective=SHADOW_CHOICE_NONE,
            comparison="deliberate_none",
            status="succeeded",
            latency_ns=elapsed_ns,
            probabilities=raw_result.probabilities,
            server_latency_ms=raw_result.server_latency_ms,
            model_id=resolved_model_id,
        )

    if conf < validated_threshold:
        return authoritative_route, _obs(
            raw_choice=raw_choice,
            confidence=conf,
            effective=SHADOW_CHOICE_NONE,
            comparison="below_threshold",
            status="succeeded",
            latency_ns=elapsed_ns,
            probabilities=raw_result.probabilities,
            server_latency_ms=raw_result.server_latency_ms,
            model_id=resolved_model_id,
        )

    comparison: ShadowComparisonOutcome = (
        "agreed" if raw_choice == authoritative_route.value else "disagreed"
    )
    return authoritative_route, _obs(
        raw_choice=raw_choice,
        confidence=conf,
        effective=raw_choice,
        comparison=comparison,
        status="succeeded",
        latency_ns=elapsed_ns,
        probabilities=raw_result.probabilities,
        server_latency_ms=raw_result.server_latency_ms,
        model_id=resolved_model_id,
    )


def _load_cases_by_id(json_path: Path, selected_ids: tuple[str, ...]) -> list[EvaluationCase]:
    raw_items = json.loads(json_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in raw_items}
    cases: list[EvaluationCase] = []
    for case_id in selected_ids:
        item = by_id[case_id]
        cases.append(
            EvaluationCase(
                id=item["id"],
                prompt=item["prompt"],
                expected_capability=item["expected_capability"],
                clarity=item["clarity"],
                rationale=item["rationale"],
                category=item["category"],
                tags=list(item.get("tags", [])),
            )
        )
    return cases


def load_shadow_evaluation_corpus(
    routing_dir: Path | None = None,
) -> list[EvaluationCase]:
    """Load the repository-grounded shadow evaluation corpus plus deliberate-none cases."""
    base_dir = routing_dir or Path(__file__).resolve().parent
    cases: list[EvaluationCase] = []
    cases.extend(_load_cases_by_id(base_dir / "dataset.json", DEFAULT_DATASET_CASE_IDS))
    cases.extend(
        _load_cases_by_id(base_dir / "validation_dataset.json", DEFAULT_VALIDATION_CASE_IDS)
    )
    cases.extend([
        EvaluationCase(
            id="shadow-none-001",
            prompt="Handle this request appropriately.",
            expected_capability="general",
            clarity="ambiguous",
            rationale="Underspecified request lacking task domain or external retrieval intent; deterministic router defaults to general while shadow classifier may abstain with none.",
            category="deliberate_none_underspecified",
            tags=["shadow", "abstention", "underspecified"],
        ),
        EvaluationCase(
            id="shadow-none-002",
            prompt="Proceed with the next step.",
            expected_capability="general",
            clarity="ambiguous",
            rationale="Deictic prompt without context; deterministic router defaults to general while shadow classifier may abstain with none.",
            category="deliberate_none_underspecified",
            tags=["shadow", "abstention", "underspecified"],
        ),
    ])
    return cases


def run_shadow_corpus_evaluation(
    *,
    case_ids: tuple[str, ...] | list[str] | None = None,
    router: DeterministicCapabilityRouter | None = None,
    adapter: ShadowDeciderAdapter | None = None,
    enabled: bool = True,
    threshold: float = DEFAULT_SHADOW_CONFIDENCE_THRESHOLD,
) -> ShadowCorpusReport:
    """Evaluate only the checked-in repository offline corpus with shadow observation.

    Arbitrary external/live prompt lists are intentionally not accepted by the
    corpus runner; callers may optionally filter to a subset of checked-in
    corpus `case_ids`.
    """
    corpus_cases = load_shadow_evaluation_corpus()
    if case_ids is not None:
        by_id = {c.id: c for c in corpus_cases}
        unknown_ids = [cid for cid in case_ids if cid not in by_id]
        if unknown_ids:
            raise ValueError(
                f"Unknown corpus case_ids {unknown_ids!r}; only repository-controlled offline corpus cases are supported"
            )
        active_cases = [by_id[cid] for cid in case_ids]
    else:
        active_cases = corpus_cases

    active_router = router or DeterministicCapabilityRouter()
    observations: list[ShadowRoutingObservation] = []

    for case in active_cases:
        has_artifact = "supplied_artifact" in case.tags or "supplied_code" in case.tags
        _, obs = evaluate_shadow_case(
            case_id=case.id,
            prompt=case.prompt,
            router=active_router,
            adapter=adapter,
            enabled=enabled,
            threshold=threshold,
            has_supplied_artifact_context=has_artifact,
        )
        observations.append(obs)

    return ShadowCorpusReport(
        threshold=float(threshold),
        input_version=CANONICAL_SHADOW_INPUT_VERSION,
        total_cases=len(observations),
        agreed_count=sum(1 for o in observations if o.comparison == "agreed"),
        disagreed_count=sum(1 for o in observations if o.comparison == "disagreed"),
        deliberate_none_count=sum(1 for o in observations if o.comparison == "deliberate_none"),
        below_threshold_count=sum(1 for o in observations if o.comparison == "below_threshold"),
        execution_failed_count=sum(1 for o in observations if o.comparison == "execution_failed"),
        disabled_count=sum(1 for o in observations if o.comparison == "disabled"),
        observations=tuple(observations),
    )
