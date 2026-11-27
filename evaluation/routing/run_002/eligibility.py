"""Pure eligibility policy and run-level fatal preflight for Run 002 shadow routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from orchestrator.domain.assessment import (
    RoutingAssessment,
    RoutingClarity,
    RoutingSignalType,
)
from orchestrator.domain.capability import AgentCapability

from evaluation.routing.run_002.canonicalization import CanonicalPrompt
from evaluation.routing.run_002.errors import FatalRunError, FatalRunErrorCode

SUPPORTED_AUTHORITATIVE_CAPABILITIES: frozenset[AgentCapability] = frozenset({
    AgentCapability.GENERAL,
    AgentCapability.RESEARCH,
})


class RouterMode(StrEnum):
    """Router construction mode permitted for Run 002 offline shadow evaluation."""

    DEFAULT_COORDINATOR_ROUTER = "default_coordinator_router"
    UNSUPPORTED = "unsupported"


class IneligibilityReason(StrEnum):
    """Closed per-record ineligibility reasons per the Run 002 shadow contract."""

    CORPUS_RECORD_NOT_ELIGIBLE = "corpus_record_not_eligible"
    INVALID_ASSESSMENT = "invalid_assessment"
    AMBIGUOUS_ASSESSMENT = "ambiguous_assessment"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    COMPOUND_MIXED_TASK = "compound_mixed_task"


class ReportStratum(StrEnum):
    """Closed reporting strata for eligible Run 002 observations."""

    CLEAR_RESEARCH = "clear_research"
    CLEAR_GENERAL_SIGNAL_BACKED = "clear_general_signal_backed"
    CLEAR_GENERAL_ZERO_SIGNAL_DEFAULT = "clear_general_zero_signal_default"


@dataclass(frozen=True)
class EligibilityDecision:
    """Typed output of the pure Run 002 eligibility policy."""

    eligible: bool
    ineligibility_reason: IneligibilityReason | None
    stratum: ReportStratum | None
    authoritative_capability: AgentCapability | None

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise TypeError("eligible must be a bool")
        if self.eligible:
            if self.ineligibility_reason is not None:
                raise ValueError("ineligibility_reason must be None when eligible=True")
            if not isinstance(self.stratum, ReportStratum):
                raise ValueError("stratum must be ReportStratum when eligible=True")
            if self.authoritative_capability not in SUPPORTED_AUTHORITATIVE_CAPABILITIES:
                raise ValueError(
                    "authoritative_capability must be general or research when eligible=True"
                )
        else:
            if not isinstance(self.ineligibility_reason, IneligibilityReason):
                raise ValueError(
                    "ineligibility_reason must be IneligibilityReason when eligible=False"
                )
            if self.stratum is not None:
                raise ValueError("stratum must be None when eligible=False")
            if self.authoritative_capability is not None:
                raise ValueError("authoritative_capability must be None when eligible=False")


def validate_run_preflight(
    *,
    corpus_approved: bool,
    router_mode: RouterMode,
) -> None:
    """Enforce run-level fatal preflight checks before per-record eligibility.

    Ensures a record-level condition (such as corpus_eligible=False) cannot
    mask a fatal run-level condition.
    """
    if corpus_approved is not True:
        raise FatalRunError(
            FatalRunErrorCode.CORPUS_NOT_APPROVED,
            "Corpus manifest is not approved",
        )
    if router_mode is not RouterMode.DEFAULT_COORDINATOR_ROUTER:
        raise FatalRunError(
            FatalRunErrorCode.UNSUPPORTED_ROUTER_MODE,
            "Coordinator requires default DeterministicCapabilityRouter()",
        )


def _derive_stratum(assessment: RoutingAssessment) -> ReportStratum:
    if assessment.provisional_capability == AgentCapability.RESEARCH:
        return ReportStratum.CLEAR_RESEARCH
    if len(assessment.signals) > 0:
        return ReportStratum.CLEAR_GENERAL_SIGNAL_BACKED
    return ReportStratum.CLEAR_GENERAL_ZERO_SIGNAL_DEFAULT


def evaluate_eligibility(
    *,
    canonical_prompt: CanonicalPrompt,
    corpus_approved: bool,
    corpus_eligible: bool,
    router_mode: RouterMode,
    assessment: Any,
) -> EligibilityDecision:
    """Single pure eligibility function for Run 002 shadow routing.

    Runs after canonicalization and deterministic assessment and before worker
    invocation. Fatal run-level conditions are evaluated before per-record
    ineligibility conditions so record-level flags never mask fatal errors.
    """
    if not isinstance(canonical_prompt, CanonicalPrompt):
        raise TypeError("canonical_prompt must be CanonicalPrompt")

    if canonical_prompt.is_empty:
        raise FatalRunError(
            FatalRunErrorCode.EMPTY_NORMALIZED_PROMPT,
            "Empty canonical prompt is forbidden",
        )

    validate_run_preflight(corpus_approved=corpus_approved, router_mode=router_mode)

    if corpus_eligible is not True:
        return EligibilityDecision(
            eligible=False,
            ineligibility_reason=IneligibilityReason.CORPUS_RECORD_NOT_ELIGIBLE,
            stratum=None,
            authoritative_capability=None,
        )

    if not isinstance(assessment, RoutingAssessment):
        return EligibilityDecision(
            eligible=False,
            ineligibility_reason=IneligibilityReason.INVALID_ASSESSMENT,
            stratum=None,
            authoritative_capability=None,
        )

    if assessment.clarity != RoutingClarity.CLEAR:
        return EligibilityDecision(
            eligible=False,
            ineligibility_reason=IneligibilityReason.AMBIGUOUS_ASSESSMENT,
            stratum=None,
            authoritative_capability=None,
        )

    if assessment.provisional_capability not in SUPPORTED_AUTHORITATIVE_CAPABILITIES:
        return EligibilityDecision(
            eligible=False,
            ineligibility_reason=IneligibilityReason.UNSUPPORTED_CAPABILITY,
            stratum=None,
            authoritative_capability=None,
        )

    if any(
        sig.signal_type == RoutingSignalType.COMPOUND_MIXED_TASK
        for sig in assessment.signals
    ):
        return EligibilityDecision(
            eligible=False,
            ineligibility_reason=IneligibilityReason.COMPOUND_MIXED_TASK,
            stratum=None,
            authoritative_capability=None,
        )

    return EligibilityDecision(
        eligible=True,
        ineligibility_reason=None,
        stratum=_derive_stratum(assessment),
        authoritative_capability=assessment.provisional_capability,
    )
