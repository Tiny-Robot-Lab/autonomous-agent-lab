"""Minimal typed worker boundary and deterministic Increment 1 stub worker."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from evaluation.routing.run_002.errors import PerRecordErrorCode

PROBABILITY_ROW_SUM_TOLERANCE = 0.0001
ALLOWED_SHADOW_DECISIONS: frozenset[str] = frozenset({"general", "research"})


class WorkerStartupError(RuntimeError):
    """Privacy-safe failure initializing or validating the external worker environment."""

    def __init__(self, detail: str) -> None:
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("detail must be a non-empty string")
        self.detail = detail.strip()
        super().__init__(self.detail)


@dataclass(frozen=True)
class WorkerPrediction:
    """Privacy-safe typed prediction returned by a shadow worker."""

    general_probability: float
    research_probability: float
    latency_ns: int
    shadow_decision: str | None = None


@dataclass(frozen=True)
class WorkerFailure:
    """Typed per-record shadow execution failure returned by a worker."""

    error_code: PerRecordErrorCode
    latency_ns: int | None = None


WorkerResult = WorkerPrediction | WorkerFailure


@dataclass(frozen=True)
class ValidatedWorkerPrediction:
    """Validated, post-normalization worker prediction ready for observation recording."""

    shadow_decision: str
    general_probability: float
    research_probability: float
    latency_ns: int


class ShadowWorker(Protocol):
    """Worker boundary isolating Run 002 model inference from the coordinator.

    Accepts only the canonical prompt string and returns a typed privacy-safe
    prediction or per-record failure. Increment 2 replaces the stub with a
    subprocess worker.
    """

    def predict(self, canonical_prompt: str) -> WorkerResult: ...


def _is_valid_non_bool_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def validate_and_resolve_prediction(
    result: Any,
) -> ValidatedWorkerPrediction | WorkerFailure:
    """Validate untrusted worker output against Run 002 probability and schema rules."""
    if isinstance(result, WorkerFailure):
        if not isinstance(result.error_code, PerRecordErrorCode):
            return WorkerFailure(error_code=PerRecordErrorCode.MALFORMED_OUTPUT, latency_ns=None)
        latency = result.latency_ns
        if latency is not None and (
            isinstance(latency, bool) or not isinstance(latency, int) or latency < 0
        ):
            return WorkerFailure(error_code=PerRecordErrorCode.MALFORMED_OUTPUT, latency_ns=None)
        return result

    if not isinstance(result, WorkerPrediction):
        return WorkerFailure(error_code=PerRecordErrorCode.MALFORMED_OUTPUT, latency_ns=None)

    latency = result.latency_ns
    if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
        return WorkerFailure(error_code=PerRecordErrorCode.MALFORMED_OUTPUT, latency_ns=None)

    raw_gen = result.general_probability
    raw_res = result.research_probability

    if isinstance(raw_gen, bool) or isinstance(raw_res, bool):
        return WorkerFailure(
            error_code=PerRecordErrorCode.INVALID_PROBABILITIES,
            latency_ns=latency,
        )
    if not _is_valid_non_bool_number(raw_gen) or not _is_valid_non_bool_number(raw_res):
        return WorkerFailure(
            error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
            latency_ns=latency,
        )

    p_gen = float(raw_gen)
    p_res = float(raw_res)

    if (
        not math.isfinite(p_gen)
        or not math.isfinite(p_res)
        or p_gen < 0.0
        or p_gen > 1.0
        or p_res < 0.0
        or p_res > 1.0
    ):
        return WorkerFailure(
            error_code=PerRecordErrorCode.INVALID_PROBABILITIES,
            latency_ns=latency,
        )

    row_sum = p_gen + p_res
    if (
        not math.isfinite(row_sum)
        or row_sum <= 0.0
        or abs(row_sum - 1.0) > PROBABILITY_ROW_SUM_TOLERANCE
    ):
        return WorkerFailure(
            error_code=PerRecordErrorCode.INVALID_PROBABILITIES,
            latency_ns=latency,
        )

    # Normalize then revalidate per contract.
    norm_gen = p_gen / row_sum
    norm_res = p_res / row_sum
    if (
        not math.isfinite(norm_gen)
        or not math.isfinite(norm_res)
        or norm_gen < 0.0
        or norm_gen > 1.0
        or norm_res < 0.0
        or norm_res > 1.0
    ):
        return WorkerFailure(
            error_code=PerRecordErrorCode.INVALID_PROBABILITIES,
            latency_ns=latency,
        )

    if norm_gen == norm_res:
        return WorkerFailure(
            error_code=PerRecordErrorCode.INVALID_PROBABILITIES,
            latency_ns=latency,
        )

    derived_decision = "general" if norm_gen > norm_res else "research"
    if result.shadow_decision is not None and (
        not isinstance(result.shadow_decision, str)
        or result.shadow_decision not in ALLOWED_SHADOW_DECISIONS
        or result.shadow_decision != derived_decision
    ):
        return WorkerFailure(
            error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
            latency_ns=latency,
        )

    return ValidatedWorkerPrediction(
        shadow_decision=derived_decision,
        general_probability=norm_gen,
        research_probability=norm_res,
        latency_ns=latency,
    )


def validate_predict_proba_output(
    raw_proba: Any,
    *,
    latency_ns: int,
) -> ValidatedWorkerPrediction | WorkerFailure:
    """Validate raw model predict_proba output using the single canonical validator.

    Enforces shape (1, 2) and array-like structure (MALFORMED_OUTPUT on wrong
    type/shape), then delegates scalar probability validation, row-sum tolerance,
    normalization, post-normalization revalidation, tie rejection, and argmax
    resolution to `validate_and_resolve_prediction`.
    """
    if isinstance(latency_ns, bool) or not isinstance(latency_ns, int) or latency_ns < 0:
        return WorkerFailure(error_code=PerRecordErrorCode.MALFORMED_OUTPUT, latency_ns=None)

    if raw_proba is None or isinstance(raw_proba, (str, bytes, bytearray, dict)):
        return WorkerFailure(error_code=PerRecordErrorCode.MALFORMED_OUTPUT, latency_ns=latency_ns)

    matrix: Any = raw_proba
    tolist_fn = getattr(raw_proba, "tolist", None)
    if callable(tolist_fn):
        try:
            matrix = tolist_fn()
        except Exception:  # noqa: BLE001
            return WorkerFailure(
                error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
                latency_ns=latency_ns,
            )

    if (
        not isinstance(matrix, Sequence)
        or isinstance(matrix, (str, bytes, bytearray))
        or len(matrix) != 1
    ):
        return WorkerFailure(error_code=PerRecordErrorCode.MALFORMED_OUTPUT, latency_ns=latency_ns)

    row = matrix[0]
    if (
        not isinstance(row, Sequence)
        or isinstance(row, (str, bytes, bytearray))
        or len(row) != 2
    ):
        return WorkerFailure(error_code=PerRecordErrorCode.MALFORMED_OUTPUT, latency_ns=latency_ns)

    return validate_and_resolve_prediction(
        WorkerPrediction(
            general_probability=row[0],
            research_probability=row[1],
            latency_ns=latency_ns,
        )
    )


class StubShadowWorker:
    """Deterministic stub worker for Increment 1 testing without ML dependencies."""

    def __init__(self, handler: Callable[[str], WorkerResult]) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handler = handler

    def predict(self, canonical_prompt: str) -> WorkerResult:
        if not isinstance(canonical_prompt, str):
            raise TypeError("canonical_prompt must be str")
        return self._handler(canonical_prompt)
