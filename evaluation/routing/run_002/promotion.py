"""Closed promotion-gate evaluation for Run 002 offline shadow routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from evaluation.routing.run_002.corpus import (
    MINIMUM_COMPLETED_PER_AUTHORITATIVE_CLASS,
    MINIMUM_COMPLETED_UNIQUE_ELIGIBLE,
)
from evaluation.routing.run_002.eligibility import ReportStratum
from evaluation.routing.run_002.telemetry import (
    DisagreementCategory,
    ObservationState,
    ShadowObservation,
)


class PromotionGate(StrEnum):
    """Closed Run 002 promotion verdicts. None authorizes automatic promotion."""

    STOP_AND_INVESTIGATE = "STOP_AND_INVESTIGATE"
    INVALID_RUN_EXECUTION_ERRORS = "INVALID_RUN_EXECUTION_ERRORS"
    INVALID_RUN_INSUFFICIENT_EVIDENCE = "INVALID_RUN_INSUFFICIENT_EVIDENCE"
    ELIGIBLE_FOR_HUMAN_REVIEW = "ELIGIBLE_FOR_HUMAN_REVIEW"


@dataclass(frozen=True)
class PromotionEvaluation:
    """Aggregated counts and exact promotion-gate verdict for a bounded Run 002 replay."""

    gate: PromotionGate
    total_records: int
    ineligible_count: int
    eligible_count: int
    agreed_count: int
    disagreed_count: int
    dangerous_disagreement_count: int
    execution_error_count: int
    completed_eligible_count: int
    completed_general_count: int
    completed_research_count: int
    stratum_counts: Mapping[str, int]


def evaluate_promotion_gate(
    observations: Sequence[ShadowObservation],
) -> PromotionEvaluation:
    """Evaluate the Run 002 promotion gate with exact contract precedence.

    Precedence:
    1. Any authoritative RESEARCH / shadow GENERAL disagreement -> STOP_AND_INVESTIGATE
    2. Otherwise, any per-record execution error -> INVALID_RUN_EXECUTION_ERRORS
    3. Otherwise, <200 completed eligible observations or <50 in either
       authoritative class -> INVALID_RUN_INSUFFICIENT_EVIDENCE
    4. Otherwise -> ELIGIBLE_FOR_HUMAN_REVIEW (never automatic promotion)
    """
    stratum_counts: dict[str, int] = {stratum.value: 0 for stratum in ReportStratum}
    ineligible_count = 0
    eligible_count = 0
    agreed_count = 0
    disagreed_count = 0
    dangerous_disagreement_count = 0
    execution_error_count = 0
    completed_general_count = 0
    completed_research_count = 0

    for obs in observations:
        if not isinstance(obs, ShadowObservation):
            raise TypeError("observations must contain ShadowObservation instances")

        if obs.stratum is not None:
            stratum_counts[obs.stratum.value] += 1

        if obs.state is ObservationState.INELIGIBLE:
            ineligible_count += 1
            continue

        eligible_count += 1
        if obs.state is ObservationState.ELIGIBLE_SHADOW_EXECUTION_ERROR:
            execution_error_count += 1
            continue

        if obs.authoritative_capability == "general":
            completed_general_count += 1
        elif obs.authoritative_capability == "research":
            completed_research_count += 1

        if obs.state is ObservationState.ELIGIBLE_AGREED:
            agreed_count += 1
        elif obs.state is ObservationState.ELIGIBLE_DISAGREED:
            disagreed_count += 1
            if (
                obs.disagreement_category
                is DisagreementCategory.AUTHORITATIVE_RESEARCH_SHADOW_GENERAL
            ):
                dangerous_disagreement_count += 1

    completed_eligible_count = agreed_count + disagreed_count

    if dangerous_disagreement_count > 0:
        gate = PromotionGate.STOP_AND_INVESTIGATE
    elif execution_error_count > 0:
        gate = PromotionGate.INVALID_RUN_EXECUTION_ERRORS
    elif (
        completed_eligible_count < MINIMUM_COMPLETED_UNIQUE_ELIGIBLE
        or completed_general_count < MINIMUM_COMPLETED_PER_AUTHORITATIVE_CLASS
        or completed_research_count < MINIMUM_COMPLETED_PER_AUTHORITATIVE_CLASS
    ):
        gate = PromotionGate.INVALID_RUN_INSUFFICIENT_EVIDENCE
    else:
        gate = PromotionGate.ELIGIBLE_FOR_HUMAN_REVIEW

    return PromotionEvaluation(
        gate=gate,
        total_records=len(observations),
        ineligible_count=ineligible_count,
        eligible_count=eligible_count,
        agreed_count=agreed_count,
        disagreed_count=disagreed_count,
        dangerous_disagreement_count=dangerous_disagreement_count,
        execution_error_count=execution_error_count,
        completed_eligible_count=completed_eligible_count,
        completed_general_count=completed_general_count,
        completed_research_count=completed_research_count,
        stratum_counts=MappingProxyType(stratum_counts),
    )
