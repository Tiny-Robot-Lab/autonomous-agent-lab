"""Privacy-safe observation schemas, environment fingerprint, summary, and JSONL output."""

from __future__ import annotations

import math
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agent_contracts.serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
)

from evaluation.routing.run_002.eligibility import IneligibilityReason, ReportStratum
from evaluation.routing.run_002.errors import (
    FatalRunError,
    FatalRunErrorCode,
    PerRecordErrorCode,
)

PROTOCOL_SUPERSESSION_NOTES: Mapping[str, str] = MappingProxyType({
    "disagreement_category_when_agreed": (
        "Contract run-002-offline-shadow-integration-003 supersedes the older Run 002 "
        "protocol by representing agreed observations with disagreement_category=null "
        "instead of 'none'."
    ),
    "run_level_identities_location": (
        "Run-level identities and environment metadata belong in the separate RunSummary "
        "artifact rather than being duplicated in every JSONL observation row."
    ),
    "ineligible_authoritative_capability": (
        "Ineligible observations record authoritative_capability=null per the accepted "
        "contract state matrix."
    ),
    "duplicate_handling": (
        "Duplicate canonical_input_sha256 records are rejected during manifest "
        "validation rather than represented via duplicate_count."
    ),
})

FORBIDDEN_TELEMETRY_KEYS: frozenset[str] = frozenset({
    "raw_prompt",
    "normalized_prompt",
    "canonical_prompt",
    "prompt",
    "reason",
    "evidence",
    "exception",
    "exception_text",
    "traceback",
    "hostname",
    "username",
    "user",
    "absolute_path",
    "env",
    "environment_variables",
})

ALLOWED_ENVIRONMENT_FINGERPRINT_FIELDS: frozenset[str] = frozenset({
    "python_version",
    "platform_system",
    "platform_release_major",
    "machine_architecture",
    "package_versions",
    "device_class",
    "repository_commit",
})

ALLOWED_CAPABILITY_STRINGS: frozenset[str] = frozenset({"general", "research"})


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_valid_probability(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, float)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _is_valid_latency_ns(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


class ObservationState(StrEnum):
    """Closed Run 002 observation states."""

    INELIGIBLE = "ineligible"
    ELIGIBLE_AGREED = "eligible_agreed"
    ELIGIBLE_DISAGREED = "eligible_disagreed"
    ELIGIBLE_SHADOW_EXECUTION_ERROR = "eligible_shadow_execution_error"


class DisagreementCategory(StrEnum):
    """Closed disagreement categories for eligible observations."""

    AUTHORITATIVE_RESEARCH_SHADOW_GENERAL = "authoritative_research_shadow_general"
    AUTHORITATIVE_GENERAL_SHADOW_RESEARCH = "authoritative_general_shadow_research"
    SHADOW_EXECUTION_ERROR = "shadow_execution_error"


@dataclass(frozen=True)
class ShadowObservation:
    """Typed, privacy-safe Run 002 observation enforcing the contract state matrix."""

    observation_id: str
    canonical_input_sha256: str
    state: ObservationState
    eligible: bool
    ineligibility_reason: IneligibilityReason | None
    stratum: ReportStratum | None
    authoritative_capability: str | None
    shadow_decision: str | None
    general_probability: float | None
    research_probability: float | None
    agreement: bool | None
    disagreement_category: DisagreementCategory | None
    error_code: PerRecordErrorCode | None
    latency_ns: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id.strip():
            raise ValueError("observation_id must be a non-empty string")
        if not _is_sha256_hex(self.canonical_input_sha256):
            raise ValueError("canonical_input_sha256 must be a 64-character lowercase hex digest")
        if not isinstance(self.state, ObservationState):
            raise TypeError(f"state must be ObservationState, got {type(self.state).__name__}")
        if type(self.eligible) is not bool:
            raise TypeError("eligible must be a bool")

        if self.state is ObservationState.INELIGIBLE:
            self._validate_ineligible()
        elif self.state is ObservationState.ELIGIBLE_AGREED:
            self._validate_eligible_agreed()
        elif self.state is ObservationState.ELIGIBLE_DISAGREED:
            self._validate_eligible_disagreed()
        elif self.state is ObservationState.ELIGIBLE_SHADOW_EXECUTION_ERROR:
            self._validate_eligible_execution_error()

        payload = self.to_dict()
        if FORBIDDEN_TELEMETRY_KEYS & payload.keys():
            raise ValueError("ShadowObservation contains forbidden telemetry keys")
        check_no_credential_bearing_keys(payload, context="ShadowObservation")

    def _validate_ineligible(self) -> None:
        if self.eligible is not False:
            raise ValueError("ineligible observation must have eligible=False")
        if self.ineligibility_reason is None:
            raise ValueError("ineligible observation requires closed ineligibility_reason")
        if not isinstance(self.ineligibility_reason, IneligibilityReason):
            raise TypeError("ineligibility_reason must be IneligibilityReason")
        if any(
            v is not None
            for v in (
                self.stratum,
                self.authoritative_capability,
                self.shadow_decision,
                self.general_probability,
                self.research_probability,
                self.agreement,
                self.disagreement_category,
                self.error_code,
                self.latency_ns,
            )
        ):
            raise ValueError(
                "ineligible observation must have null stratum, route, model, result, error, and latency fields"
            )

    def _validate_common_eligible_route(self) -> None:
        if self.eligible is not True:
            raise ValueError("eligible observation must have eligible=True")
        if self.ineligibility_reason is not None:
            raise ValueError("eligible observation must have ineligibility_reason=None")
        if self.stratum is None:
            raise ValueError("eligible observation requires non-null ReportStratum")
        if not isinstance(self.stratum, ReportStratum):
            raise TypeError("stratum must be ReportStratum")
        if self.authoritative_capability not in ALLOWED_CAPABILITY_STRINGS:
            raise ValueError("eligible observation requires authoritative_capability general/research")

    def _validate_probabilities_and_latency(self) -> None:
        if self.shadow_decision not in ALLOWED_CAPABILITY_STRINGS:
            raise ValueError("completed eligible observation requires shadow_decision general/research")
        if not _is_valid_probability(self.general_probability):
            raise ValueError("general_probability must be finite float in [0.0, 1.0]")
        if not _is_valid_probability(self.research_probability):
            raise ValueError("research_probability must be finite float in [0.0, 1.0]")
        if not _is_valid_latency_ns(self.latency_ns):
            raise ValueError("completed eligible observation requires non-negative integer latency_ns")

    def _validate_eligible_agreed(self) -> None:
        self._validate_common_eligible_route()
        self._validate_probabilities_and_latency()
        if self.shadow_decision != self.authoritative_capability:
            raise ValueError("eligible_agreed requires shadow_decision == authoritative_capability")
        if self.agreement is not True:
            raise ValueError("eligible_agreed requires agreement=True")
        if self.disagreement_category is not None:
            raise ValueError("eligible_agreed requires disagreement_category=None")
        if self.error_code is not None:
            raise ValueError("eligible_agreed requires error_code=None")

    def _validate_eligible_disagreed(self) -> None:
        self._validate_common_eligible_route()
        self._validate_probabilities_and_latency()
        if self.shadow_decision == self.authoritative_capability:
            raise ValueError("eligible_disagreed requires shadow_decision != authoritative_capability")
        if self.agreement is not False:
            raise ValueError("eligible_disagreed requires agreement=False")
        expected_cat = (
            DisagreementCategory.AUTHORITATIVE_RESEARCH_SHADOW_GENERAL
            if self.authoritative_capability == "research"
            else DisagreementCategory.AUTHORITATIVE_GENERAL_SHADOW_RESEARCH
        )
        if self.disagreement_category is not expected_cat:
            raise ValueError(
                f"eligible_disagreed requires directional disagreement_category {expected_cat.value!r}"
            )
        if self.error_code is not None:
            raise ValueError("eligible_disagreed requires error_code=None")

    def _validate_eligible_execution_error(self) -> None:
        self._validate_common_eligible_route()
        if (
            self.shadow_decision is not None
            or self.general_probability is not None
            or self.research_probability is not None
            or self.agreement is not None
        ):
            raise ValueError(
                "eligible_shadow_execution_error must have null shadow_decision, probabilities, and agreement"
            )
        if self.disagreement_category is not DisagreementCategory.SHADOW_EXECUTION_ERROR:
            raise ValueError(
                "eligible_shadow_execution_error requires disagreement_category='shadow_execution_error'"
            )
        if self.error_code is None:
            raise ValueError("eligible_shadow_execution_error requires non-null PerRecordErrorCode")
        if not isinstance(self.error_code, PerRecordErrorCode):
            raise TypeError("error_code must be PerRecordErrorCode")
        if self.latency_ns is not None and not _is_valid_latency_ns(self.latency_ns):
            raise ValueError("latency_ns must be None or non-negative integer")

    @classmethod
    def ineligible(
        cls,
        *,
        observation_id: str,
        canonical_input_sha256: str,
        reason: IneligibilityReason,
    ) -> ShadowObservation:
        return cls(
            observation_id=observation_id,
            canonical_input_sha256=canonical_input_sha256,
            state=ObservationState.INELIGIBLE,
            eligible=False,
            ineligibility_reason=reason,
            stratum=None,
            authoritative_capability=None,
            shadow_decision=None,
            general_probability=None,
            research_probability=None,
            agreement=None,
            disagreement_category=None,
            error_code=None,
            latency_ns=None,
        )

    @classmethod
    def completed(
        cls,
        *,
        observation_id: str,
        canonical_input_sha256: str,
        stratum: ReportStratum,
        authoritative_capability: str,
        shadow_decision: str,
        general_probability: float,
        research_probability: float,
        latency_ns: int,
    ) -> ShadowObservation:
        agreed = shadow_decision == authoritative_capability
        if agreed:
            return cls(
                observation_id=observation_id,
                canonical_input_sha256=canonical_input_sha256,
                state=ObservationState.ELIGIBLE_AGREED,
                eligible=True,
                ineligibility_reason=None,
                stratum=stratum,
                authoritative_capability=authoritative_capability,
                shadow_decision=shadow_decision,
                general_probability=general_probability,
                research_probability=research_probability,
                agreement=True,
                disagreement_category=None,
                error_code=None,
                latency_ns=latency_ns,
            )
        category = (
            DisagreementCategory.AUTHORITATIVE_RESEARCH_SHADOW_GENERAL
            if authoritative_capability == "research"
            else DisagreementCategory.AUTHORITATIVE_GENERAL_SHADOW_RESEARCH
        )
        return cls(
            observation_id=observation_id,
            canonical_input_sha256=canonical_input_sha256,
            state=ObservationState.ELIGIBLE_DISAGREED,
            eligible=True,
            ineligibility_reason=None,
            stratum=stratum,
            authoritative_capability=authoritative_capability,
            shadow_decision=shadow_decision,
            general_probability=general_probability,
            research_probability=research_probability,
            agreement=False,
            disagreement_category=category,
            error_code=None,
            latency_ns=latency_ns,
        )

    @classmethod
    def execution_error(
        cls,
        *,
        observation_id: str,
        canonical_input_sha256: str,
        stratum: ReportStratum,
        authoritative_capability: str,
        error_code: PerRecordErrorCode,
        latency_ns: int | None,
    ) -> ShadowObservation:
        return cls(
            observation_id=observation_id,
            canonical_input_sha256=canonical_input_sha256,
            state=ObservationState.ELIGIBLE_SHADOW_EXECUTION_ERROR,
            eligible=True,
            ineligibility_reason=None,
            stratum=stratum,
            authoritative_capability=authoritative_capability,
            shadow_decision=None,
            general_probability=None,
            research_probability=None,
            agreement=None,
            disagreement_category=DisagreementCategory.SHADOW_EXECUTION_ERROR,
            error_code=error_code,
            latency_ns=latency_ns,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreement": self.agreement,
            "authoritative_capability": self.authoritative_capability,
            "canonical_input_sha256": self.canonical_input_sha256,
            "disagreement_category": (
                self.disagreement_category.value
                if self.disagreement_category is not None
                else None
            ),
            "eligible": self.eligible,
            "error_code": self.error_code.value if self.error_code is not None else None,
            "general_probability": self.general_probability,
            "ineligibility_reason": (
                self.ineligibility_reason.value
                if self.ineligibility_reason is not None
                else None
            ),
            "latency_ns": self.latency_ns,
            "observation_id": self.observation_id,
            "research_probability": self.research_probability,
            "shadow_decision": self.shadow_decision,
            "state": self.state.value,
            "stratum": self.stratum.value if self.stratum is not None else None,
        }


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Strictly allowlisted environment fingerprint excluding host/user/path/secret data."""

    python_version: str
    platform_system: str
    platform_release_major: str
    machine_architecture: str
    package_versions: Mapping[str, str]
    device_class: str
    repository_commit: str

    def __post_init__(self) -> None:
        for field_name in (
            "python_version",
            "platform_system",
            "platform_release_major",
            "machine_architecture",
            "device_class",
            "repository_commit",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.package_versions, Mapping):
            raise TypeError("package_versions must be a mapping")
        cleaned_versions: dict[str, str] = {}
        for pkg, ver in self.package_versions.items():
            if not isinstance(pkg, str) or not pkg.strip():
                raise ValueError("package_versions keys must be non-empty strings")
            if not isinstance(ver, str) or not ver.strip():
                raise ValueError("package_versions values must be non-empty strings")
            cleaned_versions[pkg.strip()] = ver.strip()
        object.__setattr__(self, "package_versions", MappingProxyType(cleaned_versions))
        payload = self.to_dict()
        if set(payload.keys()) != ALLOWED_ENVIRONMENT_FINGERPRINT_FIELDS:
            raise ValueError("EnvironmentFingerprint keys must match contract allowlist")
        check_no_credential_bearing_keys(payload, context="EnvironmentFingerprint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_class": self.device_class,
            "machine_architecture": self.machine_architecture,
            "package_versions": dict(self.package_versions),
            "platform_release_major": self.platform_release_major,
            "platform_system": self.platform_system,
            "python_version": self.python_version,
            "repository_commit": self.repository_commit,
        }


def collect_safe_environment_fingerprint(
    *,
    repository_commit: str = "unknown",
    device_class: str = "cpu",
    package_versions: Mapping[str, str] | None = None,
) -> EnvironmentFingerprint:
    """Build an allowlisted EnvironmentFingerprint with no host, user, path, or env leakage."""
    raw_release = platform.release()
    release_major = raw_release.split(".", maxsplit=1)[0].split("-", maxsplit=1)[0] or "unknown"
    return EnvironmentFingerprint(
        python_version=platform.python_version(),
        platform_system=platform.system() or sys.platform,
        platform_release_major=release_major,
        machine_architecture=platform.machine() or "unknown",
        package_versions=dict(package_versions or {}),
        device_class=device_class,
        repository_commit=repository_commit,
    )


@dataclass(frozen=True)
class RunSummary:
    """Run-level summary artifact holding identities, counts, strata, and promotion gate."""

    contract_id: str
    schema_version: int
    corpus_id: str
    corpus_sha256: str
    corpus_manifest_sha256: str
    overlap_manifest_hashes: tuple[str, ...]
    expected_model_artifact_identity: str
    expected_protocol_identity: str
    environment: EnvironmentFingerprint
    total_records: int
    ineligible_count: int
    eligible_count: int
    agreed_count: int
    disagreed_count: int
    dangerous_disagreement_count: int
    execution_error_count: int
    completed_general_count: int
    completed_research_count: int
    stratum_counts: Mapping[str, int]
    promotion_gate: str
    completed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.environment, EnvironmentFingerprint):
            raise TypeError("environment must be EnvironmentFingerprint")
        object.__setattr__(self, "stratum_counts", MappingProxyType(dict(self.stratum_counts)))
        payload = self.to_dict()
        if FORBIDDEN_TELEMETRY_KEYS & payload.keys():
            raise ValueError("RunSummary contains forbidden telemetry keys")
        check_no_credential_bearing_keys(payload, context="RunSummary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreed_count": self.agreed_count,
            "completed": self.completed,
            "completed_general_count": self.completed_general_count,
            "completed_research_count": self.completed_research_count,
            "contract_id": self.contract_id,
            "corpus_id": self.corpus_id,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "corpus_sha256": self.corpus_sha256,
            "dangerous_disagreement_count": self.dangerous_disagreement_count,
            "disagreed_count": self.disagreed_count,
            "eligible_count": self.eligible_count,
            "environment": self.environment.to_dict(),
            "execution_error_count": self.execution_error_count,
            "expected_model_artifact_identity": self.expected_model_artifact_identity,
            "expected_protocol_identity": self.expected_protocol_identity,
            "ineligible_count": self.ineligible_count,
            "overlap_manifest_hashes": list(self.overlap_manifest_hashes),
            "promotion_gate": self.promotion_gate,
            "schema_version": self.schema_version,
            "stratum_counts": dict(self.stratum_counts),
            "total_records": self.total_records,
        }

    def to_canonical_json(self) -> str:
        return canonical_json_dumps(self.to_dict())


def serialize_observation_jsonl(observation: ShadowObservation) -> str:
    """Serialize one ShadowObservation to deterministic canonical JSONL line."""
    if not isinstance(observation, ShadowObservation):
        raise TypeError("observation must be ShadowObservation")
    return canonical_json_dumps(observation.to_dict())


def write_observations_jsonl(
    observations: Sequence[ShadowObservation],
    output_path: Path,
) -> None:
    """Write observations as UTF-8 canonical JSONL; map any failure to TELEMETRY_WRITE_FAILED."""
    try:
        lines = [serialize_observation_jsonl(obs) for obs in observations]
        content = "\n".join(lines) + ("\n" if lines else "")
        encoded = content.encode("utf-8", errors="strict")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(encoded)
    except (OSError, TypeError, ValueError) as exc:
        raise FatalRunError(
            FatalRunErrorCode.TELEMETRY_WRITE_FAILED,
            "Failed to write Run 002 telemetry JSONL output",
        ) from exc


def write_run_summary(summary: RunSummary, output_path: Path) -> None:
    """Write RunSummary as strict UTF-8 canonical JSON; map failures to TELEMETRY_WRITE_FAILED."""
    try:
        content = summary.to_canonical_json() + "\n"
        encoded = content.encode("utf-8", errors="strict")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(encoded)
    except (OSError, TypeError, ValueError) as exc:
        raise FatalRunError(
            FatalRunErrorCode.TELEMETRY_WRITE_FAILED,
            "Failed to write Run 002 run summary output",
        ) from exc
