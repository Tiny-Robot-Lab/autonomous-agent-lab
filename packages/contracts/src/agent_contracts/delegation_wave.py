import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any

from .artifact_reference import ArtifactReference
from .delegation_proposal import (
    ALLOWED_CONTROL_FIELDS,
    ALLOWED_TARGET_CAPABILITIES,
    FORBIDDEN_AUTHORITY_FIELDS,
)
from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    freeze_value,
    normalize_canonical_utc_timestamp,
    to_plain_json_structure,
)

DELEGATION_WAVE_CONTRACT_VERSION: int = 1
DELEGATION_WAVE_PROPOSAL_CONTRACT_VERSION: int = 2
DELEGATION_WAVE_RESPONSE_CONTRACT_VERSION: int = 2
WAVE_YIELD_CONTRACT_VERSION: int = 2
WAVE_DEADLINE_POLICY_CONTRACT_VERSION: int = 1

MIN_WAVE_CHILDREN: int = 2
MAX_WAVE_CHILDREN: int = 4
MAX_WAVE_COUNT: int = 1
MAX_WAVE_DEPTH: int = 1
MAX_ACCEPTED_WAVE_BYTES: int = 16 * 1024
MAX_CONTROL_RECORD_BYTES: int = 16 * 1024
MAX_WAVE_IDENTIFIER_BYTES: int = 256
MAX_ITEM_ID_BYTES: int = 128
MAX_CHILD_INTENT_BYTES: int = 2000
MAX_ITEM_CONTROL_FIELDS_BYTES: int = 4096
MAX_FRACTION_VALUE: int = 10_000
MIN_WAVE_DURATION_SECONDS: int = 30
MAX_WAVE_DURATION_SECONDS: int = 3_600

FORBIDDEN_WAVE_AUTHORITY_FIELDS: frozenset[str] = FORBIDDEN_AUTHORITY_FIELDS | frozenset({
    "wave_id",
    "wave_deadline",
    "accepted_at",
    "resolved_at",
    "required_child_task_ids",
    "required_success_count",
    "state",
    "resolution",
    "lineage",
    "claim",
    "continuation",
    "continuation_version",
    "fence",
    "execution_kind",
    "decision",
})

WAVE_DEADLINE_POLICY_ID: str = "deterministic_bounded_baseline_v1"
WAVE_DEADLINE_POLICY_VERSION: int = 1

RESIDUAL_DISPOSITION_ENDED_BARRIER_RESOLVED: str = "ended / barrier_resolved"
ALLOWED_RESIDUAL_DISPOSITIONS: frozenset[str] = frozenset({
    RESIDUAL_DISPOSITION_ENDED_BARRIER_RESOLVED,
})

# Deterministic Increment 1 baseline constants (integer-only, not empirically calibrated).
WORK_COMPLEXITY_SECONDS: MappingProxyType[str, int] = MappingProxyType({
    "low": 30,
    "medium": 60,
    "high": 120,
})

UNCERTAINTY_OVERHEAD_PERCENT: MappingProxyType[str, int] = MappingProxyType({
    "low": 0,
    "medium": 25,
    "high": 50,
})

ALLOWED_WORK_COMPLEXITY_LEVELS: frozenset[str] = frozenset(WORK_COMPLEXITY_SECONDS.keys())
ALLOWED_UNCERTAINTY_LEVELS: frozenset[str] = frozenset(UNCERTAINTY_OVERHEAD_PERCENT.keys())


class BarrierPolicy(str, Enum):
    WAIT_ALL = "WAIT_ALL"
    PARTIAL = "PARTIAL"


ALLOWED_BARRIER_POLICIES: frozenset[str] = frozenset(p.value for p in BarrierPolicy)


class WaveResolution(str, Enum):
    SATISFIED = "SATISFIED"
    UNSATISFIABLE = "UNSATISFIABLE"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


ALLOWED_WAVE_RESOLUTIONS: frozenset[str] = frozenset(r.value for r in WaveResolution)
RESUMABLE_WAVE_RESOLUTIONS: frozenset[str] = frozenset({
    WaveResolution.SATISFIED.value,
    WaveResolution.UNSATISFIABLE.value,
    WaveResolution.TIMED_OUT.value,
})


def _require_int(value: Any, field_name: str, minimum: int = 1) -> int:
    if type(value) is bool or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _require_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_utf8_bound(value: str, field_name: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"{field_name} exceeds maximum UTF-8 byte bound of {max_bytes} bytes")
    return value


def _require_json_byte_bound(json_text: str, record_name: str, max_bytes: int = MAX_CONTROL_RECORD_BYTES) -> None:
    byte_len = len(json_text.encode("utf-8"))
    if byte_len > max_bytes:
        raise ValueError(
            f"{record_name} byte size ({byte_len} bytes) exceeds maximum bound of {max_bytes} bytes"
        )


def _normalize_wave_child_ids(raw_ids: Any, parent_task_id: str) -> tuple[str, ...]:
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
        raise TypeError(
            f"required_child_task_ids must be a sequence of strings, got {type(raw_ids).__name__}"
        )
    if len(raw_ids) == 0:
        raise ValueError("required_child_task_ids must not be empty")

    seen: set[str] = set()
    cleaned: list[str] = []
    for idx, child_id in enumerate(raw_ids):
        if not isinstance(child_id, str) or not child_id.strip():
            raise ValueError(f"required_child_task_ids[{idx}] must be a non-empty string")
        clean_child = child_id.strip()
        if clean_child == parent_task_id:
            raise ValueError(
                f"required_child_task_ids must not contain parent_task_id: '{parent_task_id}'"
            )
        if clean_child in seen:
            raise ValueError(f"Duplicate child task ID in required_child_task_ids: '{clean_child}'")
        seen.add(clean_child)
        cleaned.append(clean_child)
    return tuple(sorted(cleaned))


@dataclass(frozen=True)
class RequiredSuccessFraction:
    """Exact canonical rational representation for PARTIAL barrier thresholds.

    Normalized via GCD reduction so equivalent rationals (e.g. 2/4 and 1/2)
    produce identical canonical representation and semantic identity.
    Strictly enforces 0 < numerator / denominator <= 1 with integer-only values.
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        num = _require_int(self.numerator, "numerator", minimum=1)
        den = _require_int(self.denominator, "denominator", minimum=1)
        if num > MAX_FRACTION_VALUE or den > MAX_FRACTION_VALUE:
            raise ValueError(
                f"required_success_fraction numerator and denominator must be <= {MAX_FRACTION_VALUE}, "
                f"got {num}/{den}"
            )
        if num > den:
            raise ValueError(
                f"required_success_fraction must be <= 1, got {num}/{den}"
            )
        divisor = math.gcd(num, den)
        object.__setattr__(self, "numerator", num // divisor)
        object.__setattr__(self, "denominator", den // divisor)

    def required_success_count(self, required_child_count: int) -> int:
        count = _require_int(required_child_count, "required_child_count", minimum=1)
        derived = (count * self.numerator + self.denominator - 1) // self.denominator
        if derived < 1 or derived > count:
            raise ValueError(
                f"Derived required_success_count {derived} out of bounds [1, {count}]"
            )
        return derived

    def to_dict(self) -> dict[str, int]:
        return {
            "denominator": self.denominator,
            "numerator": self.numerator,
        }

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RequiredSuccessFraction":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {"numerator", "denominator"}
        unexpected = data.keys() - allowed
        if unexpected:
            raise TypeError(f"unexpected RequiredSuccessFraction fields: {sorted(unexpected)}")
        for req in ("numerator", "denominator"):
            if req not in data:
                raise ValueError(f"{req} is required")
        return cls(numerator=data["numerator"], denominator=data["denominator"])


def compute_expected_wave_duration_seconds(
    *,
    work_complexity: str,
    uncertainty_level: str,
    required_child_count: int,
    planned_parallelism: int,
    min_duration_seconds: int,
    max_duration_seconds: int,
) -> int:
    """Pure integer-only Increment 1 baseline duration derivation.

    Applies min_duration_seconds floor when derived duration is below minimum,
    and rejects without clamping when derived duration exceeds max_duration_seconds.
    """
    clean_complexity = _require_nonempty_str(work_complexity, "work_complexity")
    if clean_complexity not in ALLOWED_WORK_COMPLEXITY_LEVELS:
        raise ValueError(
            f"Unsupported work_complexity '{clean_complexity}'. "
            f"Allowed: {sorted(ALLOWED_WORK_COMPLEXITY_LEVELS)}"
        )

    clean_uncertainty = _require_nonempty_str(uncertainty_level, "uncertainty_level")
    if clean_uncertainty not in ALLOWED_UNCERTAINTY_LEVELS:
        raise ValueError(
            f"Unsupported uncertainty_level '{clean_uncertainty}'. "
            f"Allowed: {sorted(ALLOWED_UNCERTAINTY_LEVELS)}"
        )

    child_count = _require_int(required_child_count, "required_child_count", minimum=1)
    parallelism = _require_int(planned_parallelism, "planned_parallelism", minimum=1)
    if parallelism > child_count:
        raise ValueError(
            f"planned_parallelism ({parallelism}) must be <= required_child_count ({child_count})"
        )

    min_sec = _require_int(min_duration_seconds, "min_duration_seconds", minimum=1)
    max_sec = _require_int(max_duration_seconds, "max_duration_seconds", minimum=1)
    if max_sec < min_sec:
        raise ValueError(
            f"max_duration_seconds ({max_sec}) must be >= min_duration_seconds ({min_sec})"
        )

    batches = (child_count + parallelism - 1) // parallelism
    base_seconds = batches * WORK_COMPLEXITY_SECONDS[clean_complexity]
    overhead_pct = UNCERTAINTY_OVERHEAD_PERCENT[clean_uncertainty]
    raw_seconds = base_seconds + (base_seconds * overhead_pct + 99) // 100

    if raw_seconds > max_sec:
        raise ValueError(
            f"Derived wave duration {raw_seconds}s exceeds maximum bound {max_sec}s; "
            "rejected without clamping"
        )
    if raw_seconds < min_sec:
        return min_sec
    return raw_seconds


@dataclass(frozen=True)
class WaveDeadlineDerivation:
    """Immutable evidence record of deterministic bounded wave-deadline derivation.

    Governed by ADR-0011 Decision 6:
    - Records policy ID/version, canonical validated inputs, derived duration, and frozen deadline;
    - Integer-only derivation; no floating-point weights;
    - Re-validates arithmetic consistency on construction and deserialization.
    """

    policy_id: str
    policy_version: int
    work_complexity: str
    uncertainty_level: str
    required_child_count: int
    planned_parallelism: int
    min_duration_seconds: int
    max_duration_seconds: int
    accepted_at: str
    derived_duration_seconds: int
    wave_deadline: str
    version: int = WAVE_DEADLINE_POLICY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        ver = _require_int(self.version, "version", minimum=1)
        if ver != WAVE_DEADLINE_POLICY_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {ver}")

        pid = _require_nonempty_str(self.policy_id, "policy_id")
        if pid != WAVE_DEADLINE_POLICY_ID:
            raise ValueError(f"Unsupported deadline policy_id: '{pid}'")
        object.__setattr__(self, "policy_id", pid)

        pver = _require_int(self.policy_version, "policy_version", minimum=1)
        if pver != WAVE_DEADLINE_POLICY_VERSION:
            raise ValueError(f"Unsupported deadline policy_version: {pver}")

        clean_complexity = _require_nonempty_str(self.work_complexity, "work_complexity")
        clean_uncertainty = _require_nonempty_str(self.uncertainty_level, "uncertainty_level")
        object.__setattr__(self, "work_complexity", clean_complexity)
        object.__setattr__(self, "uncertainty_level", clean_uncertainty)

        expected_duration = compute_expected_wave_duration_seconds(
            work_complexity=clean_complexity,
            uncertainty_level=clean_uncertainty,
            required_child_count=self.required_child_count,
            planned_parallelism=self.planned_parallelism,
            min_duration_seconds=self.min_duration_seconds,
            max_duration_seconds=self.max_duration_seconds,
        )

        supplied_duration = _require_int(
            self.derived_duration_seconds, "derived_duration_seconds", minimum=1
        )
        if supplied_duration != expected_duration:
            raise ValueError(
                f"derived_duration_seconds ({supplied_duration}) does not match "
                f"deterministic policy derivation ({expected_duration})"
            )

        clean_accepted = normalize_canonical_utc_timestamp(self.accepted_at)
        object.__setattr__(self, "accepted_at", clean_accepted)

        try:
            expected_deadline = normalize_canonical_utc_timestamp(
                datetime.fromisoformat(clean_accepted) + timedelta(seconds=expected_duration)
            )
        except OverflowError as exc:
            raise ValueError("Wave deadline timestamp overflows supported datetime range") from exc
        clean_deadline = normalize_canonical_utc_timestamp(self.wave_deadline)
        if clean_deadline != expected_deadline:
            raise ValueError(
                f"wave_deadline ({clean_deadline}) does not match "
                f"accepted_at + derived_duration_seconds ({expected_deadline})"
            )
        object.__setattr__(self, "wave_deadline", clean_deadline)
        _require_json_byte_bound(self.to_json(), "WaveDeadlineDerivation")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "accepted_at": self.accepted_at,
            "derived_duration_seconds": self.derived_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "min_duration_seconds": self.min_duration_seconds,
            "planned_parallelism": self.planned_parallelism,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "required_child_count": self.required_child_count,
            "uncertainty_level": self.uncertainty_level,
            "version": self.version,
            "wave_deadline": self.wave_deadline,
            "work_complexity": self.work_complexity,
        }
        check_no_credential_bearing_keys(data, "WaveDeadlineDerivation.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    def semantic_fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WaveDeadlineDerivation":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {
            "accepted_at",
            "derived_duration_seconds",
            "max_duration_seconds",
            "min_duration_seconds",
            "planned_parallelism",
            "policy_id",
            "policy_version",
            "required_child_count",
            "uncertainty_level",
            "version",
            "wave_deadline",
            "work_complexity",
        }
        unexpected = data.keys() - allowed
        if unexpected:
            raise TypeError(f"unexpected WaveDeadlineDerivation fields: {sorted(unexpected)}")
        for req in allowed - {"version"}:
            if req not in data:
                raise ValueError(f"{req} is required")

        version = data.get("version", WAVE_DEADLINE_POLICY_CONTRACT_VERSION)
        return cls(
            policy_id=data["policy_id"],
            policy_version=data["policy_version"],
            work_complexity=data["work_complexity"],
            uncertainty_level=data["uncertainty_level"],
            required_child_count=data["required_child_count"],
            planned_parallelism=data["planned_parallelism"],
            min_duration_seconds=data["min_duration_seconds"],
            max_duration_seconds=data["max_duration_seconds"],
            accepted_at=data["accepted_at"],
            derived_duration_seconds=data["derived_duration_seconds"],
            wave_deadline=data["wave_deadline"],
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "WaveDeadlineDerivation":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


def _validate_barrier_and_threshold(
    barrier_policy: Any,
    required_child_count: int,
    required_success_fraction: RequiredSuccessFraction | dict[str, Any] | None,
    required_success_count: int | None,
) -> tuple[str, RequiredSuccessFraction | None, int | None]:
    clean_policy = _require_nonempty_str(barrier_policy, "barrier_policy")
    if clean_policy not in ALLOWED_BARRIER_POLICIES:
        raise ValueError(
            f"Unsupported barrier_policy '{clean_policy}'. Allowed: {sorted(ALLOWED_BARRIER_POLICIES)}"
        )

    parsed_fraction: RequiredSuccessFraction | None
    if required_success_fraction is None:
        parsed_fraction = None
    elif isinstance(required_success_fraction, RequiredSuccessFraction):
        parsed_fraction = required_success_fraction
    elif isinstance(required_success_fraction, dict):
        parsed_fraction = RequiredSuccessFraction.from_dict(required_success_fraction)
    else:
        raise TypeError(
            "required_success_fraction must be RequiredSuccessFraction, dict, or None, "
            f"got {type(required_success_fraction).__name__}"
        )

    if clean_policy == BarrierPolicy.WAIT_ALL.value:
        if parsed_fraction is not None:
            raise ValueError("WAIT_ALL barrier must not carry required_success_fraction")
        if required_success_count is not None:
            raise ValueError("WAIT_ALL barrier must not carry required_success_count")
        return clean_policy, None, None

    # PARTIAL
    if parsed_fraction is None:
        raise ValueError("PARTIAL barrier requires required_success_fraction")
    derived_count = parsed_fraction.required_success_count(required_child_count)
    if required_success_count is not None:
        supplied_count = _require_int(required_success_count, "required_success_count", minimum=1)
        if supplied_count != derived_count:
            raise ValueError(
                f"required_success_count ({supplied_count}) does not match derived "
                f"threshold ({derived_count})"
            )
    return clean_policy, parsed_fraction, derived_count


@dataclass(frozen=True)
class AcceptedDelegationWave:
    """Immutable contract for an accepted, strictly unresolved multi-child delegation wave.

    Governed by ADR-0011:
    - Always enters in strictly unresolved state ('unresolved');
    - Immutable required-child set with at least one child;
    - Exact rational threshold for PARTIAL; WAIT_ALL carries no partial threshold;
    - Frozen deterministic deadline derivation.
    """

    wave_id: str
    parent_task_id: str
    required_child_task_ids: tuple[str, ...]
    barrier_policy: str
    deadline_derivation: WaveDeadlineDerivation
    accepted_at: str
    required_success_fraction: RequiredSuccessFraction | None = None
    required_success_count: int | None = None
    wave_deadline: str | None = None
    state: str = "unresolved"
    version: int = DELEGATION_WAVE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        ver = _require_int(self.version, "version", minimum=1)
        if ver != DELEGATION_WAVE_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {ver}")

        if self.state != "unresolved":
            raise ValueError(
                f"AcceptedDelegationWave must be strictly 'unresolved', got '{self.state}'"
            )

        clean_wave_id = _require_nonempty_str(self.wave_id, "wave_id")
        clean_parent_id = _require_nonempty_str(self.parent_task_id, "parent_task_id")
        object.__setattr__(self, "wave_id", clean_wave_id)
        object.__setattr__(self, "parent_task_id", clean_parent_id)

        clean_children = _normalize_wave_child_ids(self.required_child_task_ids, clean_parent_id)
        object.__setattr__(self, "required_child_task_ids", clean_children)

        clean_policy, parsed_fraction, derived_count = _validate_barrier_and_threshold(
            self.barrier_policy,
            len(clean_children),
            self.required_success_fraction,
            self.required_success_count,
        )
        object.__setattr__(self, "barrier_policy", clean_policy)
        object.__setattr__(self, "required_success_fraction", parsed_fraction)
        object.__setattr__(self, "required_success_count", derived_count)

        if not isinstance(self.deadline_derivation, WaveDeadlineDerivation):
            raise TypeError(
                "deadline_derivation must be a WaveDeadlineDerivation, "
                f"got {type(self.deadline_derivation).__name__}"
            )
        if self.deadline_derivation.required_child_count != len(clean_children):
            raise ValueError(
                f"deadline_derivation.required_child_count ({self.deadline_derivation.required_child_count}) "
                f"must equal len(required_child_task_ids) ({len(clean_children)})"
            )

        clean_accepted = normalize_canonical_utc_timestamp(self.accepted_at)
        if clean_accepted != self.deadline_derivation.accepted_at:
            raise ValueError(
                f"accepted_at ({clean_accepted}) must match "
                f"deadline_derivation.accepted_at ({self.deadline_derivation.accepted_at})"
            )
        object.__setattr__(self, "accepted_at", clean_accepted)

        if self.wave_deadline is not None:
            clean_deadline = normalize_canonical_utc_timestamp(self.wave_deadline)
            if clean_deadline != self.deadline_derivation.wave_deadline:
                raise ValueError(
                    f"wave_deadline ({clean_deadline}) does not match "
                    f"deadline_derivation.wave_deadline ({self.deadline_derivation.wave_deadline})"
                )
        object.__setattr__(self, "wave_deadline", self.deadline_derivation.wave_deadline)

        byte_len = len(self.to_json().encode("utf-8"))
        if byte_len > MAX_ACCEPTED_WAVE_BYTES:
            raise ValueError(
                f"AcceptedDelegationWave byte size ({byte_len} bytes) exceeds maximum bound "
                f"of {MAX_ACCEPTED_WAVE_BYTES} bytes"
            )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "accepted_at": self.accepted_at,
            "barrier_policy": self.barrier_policy,
            "deadline_derivation": self.deadline_derivation.to_dict(),
            "parent_task_id": self.parent_task_id,
            "required_child_task_ids": list(self.required_child_task_ids),
            "required_success_count": self.required_success_count,
            "required_success_fraction": (
                self.required_success_fraction.to_dict()
                if self.required_success_fraction is not None
                else None
            ),
            "state": self.state,
            "version": self.version,
            "wave_deadline": self.wave_deadline,
            "wave_id": self.wave_id,
        }
        check_no_credential_bearing_keys(data, "AcceptedDelegationWave.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    def semantic_fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AcceptedDelegationWave":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {
            "accepted_at",
            "barrier_policy",
            "deadline_derivation",
            "parent_task_id",
            "required_child_task_ids",
            "required_success_count",
            "required_success_fraction",
            "state",
            "version",
            "wave_deadline",
            "wave_id",
        }
        unexpected = data.keys() - allowed
        if unexpected:
            raise TypeError(f"unexpected AcceptedDelegationWave fields: {sorted(unexpected)}")
        for req in (
            "wave_id",
            "parent_task_id",
            "required_child_task_ids",
            "barrier_policy",
            "deadline_derivation",
            "accepted_at",
        ):
            if req not in data:
                raise ValueError(f"{req} is required")

        raw_derivation = data["deadline_derivation"]
        derivation = (
            raw_derivation
            if isinstance(raw_derivation, WaveDeadlineDerivation)
            else WaveDeadlineDerivation.from_dict(raw_derivation)
        )
        version = data.get("version", DELEGATION_WAVE_CONTRACT_VERSION)
        return cls(
            wave_id=data["wave_id"],
            parent_task_id=data["parent_task_id"],
            required_child_task_ids=data["required_child_task_ids"],
            barrier_policy=data["barrier_policy"],
            deadline_derivation=derivation,
            accepted_at=data["accepted_at"],
            required_success_fraction=data.get("required_success_fraction"),
            required_success_count=data.get("required_success_count"),
            wave_deadline=data.get("wave_deadline"),
            state=data.get("state", "unresolved"),
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "AcceptedDelegationWave":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class ResolvedDelegationWave:
    """Immutable contract representing an authoritatively resolved delegation wave.

    Explicitly separated from AcceptedDelegationWave so an accepted wave
    remains strictly unresolved while resolved state carries canonical resolution
    and residual child disposition ('ended / barrier_resolved').
    """

    wave_id: str
    parent_task_id: str
    required_child_task_ids: tuple[str, ...]
    barrier_policy: str
    resolution: str
    resolved_at: str
    wave_deadline: str
    residual_disposition: str = RESIDUAL_DISPOSITION_ENDED_BARRIER_RESOLVED
    required_success_fraction: RequiredSuccessFraction | None = None
    required_success_count: int | None = None
    version: int = DELEGATION_WAVE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        ver = _require_int(self.version, "version", minimum=1)
        if ver != DELEGATION_WAVE_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {ver}")

        clean_wave_id = _require_nonempty_str(self.wave_id, "wave_id")
        clean_parent_id = _require_nonempty_str(self.parent_task_id, "parent_task_id")
        object.__setattr__(self, "wave_id", clean_wave_id)
        object.__setattr__(self, "parent_task_id", clean_parent_id)

        clean_children = _normalize_wave_child_ids(self.required_child_task_ids, clean_parent_id)
        object.__setattr__(self, "required_child_task_ids", clean_children)

        clean_policy, parsed_fraction, derived_count = _validate_barrier_and_threshold(
            self.barrier_policy,
            len(clean_children),
            self.required_success_fraction,
            self.required_success_count,
        )
        object.__setattr__(self, "barrier_policy", clean_policy)
        object.__setattr__(self, "required_success_fraction", parsed_fraction)
        object.__setattr__(self, "required_success_count", derived_count)

        clean_resolution = _require_nonempty_str(self.resolution, "resolution")
        if clean_resolution not in ALLOWED_WAVE_RESOLUTIONS:
            raise ValueError(
                f"Unsupported wave resolution '{clean_resolution}'. "
                f"Allowed: {sorted(ALLOWED_WAVE_RESOLUTIONS)}"
            )
        object.__setattr__(self, "resolution", clean_resolution)

        clean_disposition = _require_nonempty_str(
            self.residual_disposition, "residual_disposition"
        )
        if clean_disposition not in ALLOWED_RESIDUAL_DISPOSITIONS:
            raise ValueError(
                f"Unsupported residual_disposition '{clean_disposition}'. "
                f"Allowed: {sorted(ALLOWED_RESIDUAL_DISPOSITIONS)}"
            )
        object.__setattr__(self, "residual_disposition", clean_disposition)

        object.__setattr__(
            self, "resolved_at", normalize_canonical_utc_timestamp(self.resolved_at)
        )
        object.__setattr__(
            self, "wave_deadline", normalize_canonical_utc_timestamp(self.wave_deadline)
        )
        _require_json_byte_bound(self.to_json(), "ResolvedDelegationWave")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "barrier_policy": self.barrier_policy,
            "parent_task_id": self.parent_task_id,
            "required_child_task_ids": list(self.required_child_task_ids),
            "required_success_count": self.required_success_count,
            "required_success_fraction": (
                self.required_success_fraction.to_dict()
                if self.required_success_fraction is not None
                else None
            ),
            "residual_disposition": self.residual_disposition,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at,
            "version": self.version,
            "wave_deadline": self.wave_deadline,
            "wave_id": self.wave_id,
        }
        check_no_credential_bearing_keys(data, "ResolvedDelegationWave.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    def semantic_fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedDelegationWave":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        allowed = {
            "barrier_policy",
            "parent_task_id",
            "required_child_task_ids",
            "required_success_count",
            "required_success_fraction",
            "residual_disposition",
            "resolution",
            "resolved_at",
            "version",
            "wave_deadline",
            "wave_id",
        }
        unexpected = data.keys() - allowed
        if unexpected:
            raise TypeError(f"unexpected ResolvedDelegationWave fields: {sorted(unexpected)}")
        for req in (
            "wave_id",
            "parent_task_id",
            "required_child_task_ids",
            "barrier_policy",
            "resolution",
            "resolved_at",
            "wave_deadline",
        ):
            if req not in data:
                raise ValueError(f"{req} is required")

        version = data.get("version", DELEGATION_WAVE_CONTRACT_VERSION)
        return cls(
            wave_id=data["wave_id"],
            parent_task_id=data["parent_task_id"],
            required_child_task_ids=data["required_child_task_ids"],
            barrier_policy=data["barrier_policy"],
            resolution=data["resolution"],
            resolved_at=data["resolved_at"],
            wave_deadline=data["wave_deadline"],
            residual_disposition=data.get(
                "residual_disposition", RESIDUAL_DISPOSITION_ENDED_BARRIER_RESOLVED
            ),
            required_success_fraction=data.get("required_success_fraction"),
            required_success_count=data.get("required_success_count"),
            version=version,
        )

    @classmethod
    def from_json(cls, value: str) -> "ResolvedDelegationWave":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class DelegationWaveProposalItem:
    """Single caller-supplied child intent within a bounded multi-child wave proposal."""

    item_id: str
    child_intent: str
    artifact_reference: ArtifactReference
    target_capability: str = "analysis"
    control_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        clean_item_id = _require_nonempty_str(self.item_id, "item_id")
        _require_utf8_bound(clean_item_id, "item_id", MAX_ITEM_ID_BYTES)
        object.__setattr__(self, "item_id", clean_item_id)

        clean_cap = _require_nonempty_str(self.target_capability, "target_capability")
        if clean_cap not in ALLOWED_TARGET_CAPABILITIES:
            raise ValueError(
                f"Unsupported target_capability '{clean_cap}'. Allowed: {sorted(ALLOWED_TARGET_CAPABILITIES)}"
            )
        object.__setattr__(self, "target_capability", clean_cap)

        if not isinstance(self.child_intent, str):
            raise TypeError(f"child_intent must be a str, got {type(self.child_intent).__name__}")
        clean_intent = self.child_intent.strip()
        if not clean_intent:
            raise ValueError("child_intent must not be empty")
        _require_utf8_bound(self.child_intent, "child_intent", MAX_CHILD_INTENT_BYTES)
        _require_utf8_bound(clean_intent, "child_intent", MAX_CHILD_INTENT_BYTES)
        object.__setattr__(self, "child_intent", clean_intent)

        if not isinstance(self.artifact_reference, ArtifactReference):
            raise TypeError(
                f"artifact_reference must be an ArtifactReference, got {type(self.artifact_reference).__name__}"
            )

        if not isinstance(self.control_fields, Mapping):
            raise TypeError(
                f"control_fields must be a Mapping, got {type(self.control_fields).__name__}"
            )
        unexpected_controls = set(self.control_fields.keys()) - ALLOWED_CONTROL_FIELDS
        if unexpected_controls:
            raise ValueError(
                f"Unexpected control_fields for delegation wave item: {sorted(unexpected_controls)}"
            )
        check_no_credential_bearing_keys(
            dict(self.control_fields), "DelegationWaveProposalItem.control_fields"
        )
        frozen_controls = freeze_value(dict(self.control_fields))
        _require_json_byte_bound(
            canonical_json_dumps(to_plain_json_structure(frozen_controls)),
            "DelegationWaveProposalItem.control_fields",
            MAX_ITEM_CONTROL_FIELDS_BYTES,
        )
        object.__setattr__(self, "control_fields", frozen_controls)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "artifact_reference": self.artifact_reference.to_dict(),
            "child_intent": self.child_intent,
            "control_fields": to_plain_json_structure(self.control_fields),
            "item_id": self.item_id,
            "target_capability": self.target_capability,
        }
        check_no_credential_bearing_keys(data, "DelegationWaveProposalItem.to_dict")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegationWaveProposalItem":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        check_no_credential_bearing_keys(data, "DelegationWaveProposalItem payload")

        forbidden = set(data.keys()) & FORBIDDEN_WAVE_AUTHORITY_FIELDS
        if forbidden:
            raise ValueError(f"Forbidden authority fields supplied by caller: {sorted(forbidden)}")

        allowed = {
            "artifact_reference",
            "child_intent",
            "control_fields",
            "item_id",
            "target_capability",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise ValueError(f"Unexpected fields in DelegationWaveProposalItem: {sorted(unexpected)}")

        for req in ("item_id", "child_intent", "artifact_reference"):
            if req not in data:
                raise ValueError(f"{req} is required")

        raw_ref = data["artifact_reference"]
        if isinstance(raw_ref, ArtifactReference):
            ref = raw_ref
        elif isinstance(raw_ref, dict):
            ref = ArtifactReference.from_dict(raw_ref)
        else:
            raise TypeError("artifact_reference must be a dict or ArtifactReference")

        return cls(
            item_id=data["item_id"],
            child_intent=data["child_intent"],
            artifact_reference=ref,
            target_capability=data.get("target_capability", "analysis"),
            control_fields=data.get("control_fields", {}),
        )


@dataclass(frozen=True)
class DelegationWaveProposalRequest:
    """Caller-supplied bounded multi-child delegation wave proposal (version 2)."""

    proposal_id: str
    items: tuple[DelegationWaveProposalItem, ...]
    barrier_policy: str
    required_success_fraction: RequiredSuccessFraction | None = None
    work_complexity: str = "medium"
    uncertainty_level: str = "low"
    planned_parallelism: int | None = None
    parent_input_version: int = 1
    version: int = DELEGATION_WAVE_PROPOSAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        ver = _require_int(self.version, "version", minimum=1)
        if ver != DELEGATION_WAVE_PROPOSAL_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {ver}")

        clean_proposal_id = _require_nonempty_str(self.proposal_id, "proposal_id")
        _require_utf8_bound(clean_proposal_id, "proposal_id", MAX_WAVE_IDENTIFIER_BYTES)
        object.__setattr__(self, "proposal_id", clean_proposal_id)

        if isinstance(self.items, (str, bytes)) or not isinstance(self.items, Sequence):
            raise TypeError(f"items must be a sequence, got {type(self.items).__name__}")
        if not (MIN_WAVE_CHILDREN <= len(self.items) <= MAX_WAVE_CHILDREN):
            raise ValueError(
                f"DelegationWaveProposalRequest items must contain {MIN_WAVE_CHILDREN} to "
                f"{MAX_WAVE_CHILDREN} children, got {len(self.items)}"
            )
        parsed_items: list[DelegationWaveProposalItem] = []
        seen_item_ids: set[str] = set()
        for item in self.items:
            parsed_item = (
                item
                if isinstance(item, DelegationWaveProposalItem)
                else DelegationWaveProposalItem.from_dict(item)
            )
            if parsed_item.item_id in seen_item_ids:
                raise ValueError(f"Duplicate item_id in wave proposal: '{parsed_item.item_id}'")
            seen_item_ids.add(parsed_item.item_id)
            parsed_items.append(parsed_item)
        object.__setattr__(self, "items", tuple(parsed_items))

        clean_policy, parsed_fraction, _ = _validate_barrier_and_threshold(
            self.barrier_policy,
            len(parsed_items),
            self.required_success_fraction,
            None,
        )
        object.__setattr__(self, "barrier_policy", clean_policy)
        object.__setattr__(self, "required_success_fraction", parsed_fraction)

        clean_complexity = _require_nonempty_str(self.work_complexity, "work_complexity")
        if clean_complexity not in ALLOWED_WORK_COMPLEXITY_LEVELS:
            raise ValueError(f"Unsupported work_complexity '{clean_complexity}'")
        object.__setattr__(self, "work_complexity", clean_complexity)

        clean_uncertainty = _require_nonempty_str(self.uncertainty_level, "uncertainty_level")
        if clean_uncertainty not in ALLOWED_UNCERTAINTY_LEVELS:
            raise ValueError(f"Unsupported uncertainty_level '{clean_uncertainty}'")
        object.__setattr__(self, "uncertainty_level", clean_uncertainty)

        parallelism = (
            len(parsed_items)
            if self.planned_parallelism is None
            else _require_int(self.planned_parallelism, "planned_parallelism", minimum=1)
        )
        if parallelism > len(parsed_items):
            raise ValueError(
                f"planned_parallelism ({parallelism}) must be <= len(items) ({len(parsed_items)})"
            )
        object.__setattr__(self, "planned_parallelism", parallelism)

        _require_int(self.parent_input_version, "parent_input_version", minimum=1)
        _require_json_byte_bound(self.to_json(), "DelegationWaveProposalRequest")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "barrier_policy": self.barrier_policy,
            "items": [item.to_dict() for item in self.items],
            "parent_input_version": self.parent_input_version,
            "planned_parallelism": self.planned_parallelism,
            "proposal_id": self.proposal_id,
            "required_success_fraction": (
                self.required_success_fraction.to_dict()
                if self.required_success_fraction is not None
                else None
            ),
            "uncertainty_level": self.uncertainty_level,
            "version": self.version,
            "work_complexity": self.work_complexity,
        }
        check_no_credential_bearing_keys(data, "DelegationWaveProposalRequest.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegationWaveProposalRequest":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        check_no_credential_bearing_keys(data, "DelegationWaveProposalRequest payload")

        forbidden = set(data.keys()) & FORBIDDEN_WAVE_AUTHORITY_FIELDS
        if forbidden:
            raise ValueError(f"Forbidden authority fields supplied by caller: {sorted(forbidden)}")

        allowed = {
            "barrier_policy",
            "items",
            "parent_input_version",
            "planned_parallelism",
            "proposal_id",
            "required_success_fraction",
            "uncertainty_level",
            "version",
            "work_complexity",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise ValueError(f"Unexpected fields in DelegationWaveProposalRequest: {sorted(unexpected)}")

        for req in ("proposal_id", "items", "barrier_policy"):
            if req not in data:
                raise ValueError(f"{req} is required")

        return cls(
            proposal_id=data["proposal_id"],
            items=data["items"],
            barrier_policy=data["barrier_policy"],
            required_success_fraction=data.get("required_success_fraction"),
            work_complexity=data.get("work_complexity", "medium"),
            uncertainty_level=data.get("uncertainty_level", "low"),
            planned_parallelism=data.get("planned_parallelism"),
            parent_input_version=data.get("parent_input_version", 1),
            version=data.get("version", DELEGATION_WAVE_PROPOSAL_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> "DelegationWaveProposalRequest":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


def compute_wave_proposal_fingerprint(
    *,
    parent_task_id: str,
    assignment_id: str,
    request: DelegationWaveProposalRequest,
) -> str:
    """Compute deterministic canonical SHA-256 fingerprint over semantic wave proposal fields.

    Preserves caller item order so reordering items changes the semantic fingerprint.
    """
    payload = {
        "assignment_id": str(assignment_id).strip(),
        "barrier_policy": request.barrier_policy,
        "items": [item.to_dict() for item in request.items],
        "parent_input_version": int(request.parent_input_version),
        "parent_task_id": str(parent_task_id).strip(),
        "planned_parallelism": int(request.planned_parallelism or len(request.items)),
        "proposal_id": request.proposal_id,
        "required_success_fraction": (
            request.required_success_fraction.to_dict()
            if request.required_success_fraction is not None
            else None
        ),
        "uncertainty_level": request.uncertainty_level,
        "version": request.version,
        "work_complexity": request.work_complexity,
    }
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DelegationWaveProposalRecord:
    """Authoritative Redis-persisted replay record for an accepted multi-child wave proposal."""

    parent_task_id: str
    proposal_id: str
    wave_id: str
    delegation_decision_id: str
    assignment_id: str
    runtime_principal_id: str
    bound_agent_id: str
    fencing_epoch: int
    proof_digest: str
    proposal_fingerprint: str
    delegation_grant_id: str
    required_child_task_ids: tuple[str, ...]
    barrier_policy: str
    accepted_at: str
    wave_deadline: str
    child_claim_deadline: str
    required_success_fraction: RequiredSuccessFraction | None = None
    required_success_count: int | None = None
    continuation_version: int = 2
    version: int = DELEGATION_WAVE_PROPOSAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        ver = _require_int(self.version, "version", minimum=1)
        if ver != DELEGATION_WAVE_PROPOSAL_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {ver}")

        _require_int(self.fencing_epoch, "fencing_epoch", minimum=1)

        for str_field in (
            "parent_task_id",
            "proposal_id",
            "wave_id",
            "delegation_decision_id",
            "assignment_id",
            "runtime_principal_id",
            "bound_agent_id",
            "proof_digest",
            "proposal_fingerprint",
            "delegation_grant_id",
            "accepted_at",
            "wave_deadline",
            "child_claim_deadline",
        ):
            clean_val = _require_nonempty_str(getattr(self, str_field), str_field)
            _require_utf8_bound(clean_val, str_field, MAX_WAVE_IDENTIFIER_BYTES)
            object.__setattr__(self, str_field, clean_val)

        clean_children = _normalize_wave_child_ids(
            self.required_child_task_ids, self.parent_task_id
        )
        if not (MIN_WAVE_CHILDREN <= len(clean_children) <= MAX_WAVE_CHILDREN):
            raise ValueError(
                f"required_child_task_ids must contain {MIN_WAVE_CHILDREN} to {MAX_WAVE_CHILDREN} children"
            )
        object.__setattr__(self, "required_child_task_ids", clean_children)

        clean_policy, parsed_fraction, derived_count = _validate_barrier_and_threshold(
            self.barrier_policy,
            len(clean_children),
            self.required_success_fraction,
            self.required_success_count,
        )
        object.__setattr__(self, "barrier_policy", clean_policy)
        object.__setattr__(self, "required_success_fraction", parsed_fraction)
        object.__setattr__(self, "required_success_count", derived_count)

        cver = _require_int(self.continuation_version, "continuation_version", minimum=2)
        if cver != 2:
            raise ValueError(f"DelegationWaveProposalRecord continuation_version must be 2, got {cver}")

        object.__setattr__(self, "accepted_at", normalize_canonical_utc_timestamp(self.accepted_at))
        object.__setattr__(self, "wave_deadline", normalize_canonical_utc_timestamp(self.wave_deadline))
        object.__setattr__(
            self, "child_claim_deadline", normalize_canonical_utc_timestamp(self.child_claim_deadline)
        )
        _require_json_byte_bound(self.to_json(), "DelegationWaveProposalRecord")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "accepted_at": self.accepted_at,
            "assignment_id": self.assignment_id,
            "barrier_policy": self.barrier_policy,
            "bound_agent_id": self.bound_agent_id,
            "child_claim_deadline": self.child_claim_deadline,
            "continuation_version": self.continuation_version,
            "delegation_decision_id": self.delegation_decision_id,
            "delegation_grant_id": self.delegation_grant_id,
            "fencing_epoch": self.fencing_epoch,
            "parent_task_id": self.parent_task_id,
            "proof_digest": self.proof_digest,
            "proposal_fingerprint": self.proposal_fingerprint,
            "proposal_id": self.proposal_id,
            "required_child_task_ids": list(self.required_child_task_ids),
            "required_success_count": self.required_success_count,
            "required_success_fraction": (
                self.required_success_fraction.to_dict()
                if self.required_success_fraction is not None
                else None
            ),
            "runtime_principal_id": self.runtime_principal_id,
            "version": self.version,
            "wave_deadline": self.wave_deadline,
            "wave_id": self.wave_id,
        }
        check_no_credential_bearing_keys(data, "DelegationWaveProposalRecord.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegationWaveProposalRecord":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        check_no_credential_bearing_keys(data, "DelegationWaveProposalRecord payload")
        allowed = {
            "accepted_at",
            "assignment_id",
            "barrier_policy",
            "bound_agent_id",
            "child_claim_deadline",
            "continuation_version",
            "delegation_decision_id",
            "delegation_grant_id",
            "fencing_epoch",
            "parent_task_id",
            "proof_digest",
            "proposal_fingerprint",
            "proposal_id",
            "required_child_task_ids",
            "required_success_count",
            "required_success_fraction",
            "runtime_principal_id",
            "version",
            "wave_deadline",
            "wave_id",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise TypeError(f"Unexpected fields in DelegationWaveProposalRecord: {sorted(unexpected)}")
        for req in allowed - {
            "continuation_version",
            "required_success_count",
            "required_success_fraction",
            "version",
        }:
            if req not in data:
                raise ValueError(f"{req} is required")
        return cls(
            parent_task_id=data["parent_task_id"],
            proposal_id=data["proposal_id"],
            wave_id=data["wave_id"],
            delegation_decision_id=data["delegation_decision_id"],
            assignment_id=data["assignment_id"],
            runtime_principal_id=data["runtime_principal_id"],
            bound_agent_id=data["bound_agent_id"],
            fencing_epoch=data["fencing_epoch"],
            proof_digest=data["proof_digest"],
            proposal_fingerprint=data["proposal_fingerprint"],
            delegation_grant_id=data["delegation_grant_id"],
            required_child_task_ids=data["required_child_task_ids"],
            barrier_policy=data["barrier_policy"],
            accepted_at=data["accepted_at"],
            wave_deadline=data["wave_deadline"],
            child_claim_deadline=data["child_claim_deadline"],
            required_success_fraction=data.get("required_success_fraction"),
            required_success_count=data.get("required_success_count"),
            continuation_version=data.get("continuation_version", 2),
            version=data.get("version", DELEGATION_WAVE_PROPOSAL_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> "DelegationWaveProposalRecord":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class WaveYieldRecord:
    """Authoritative audit record created when an active assignment yields turn for a delegation wave."""

    assignment_id: str
    parent_task_id: str
    principal_id: str
    proposal_id: str
    wave_id: str
    required_child_task_ids: tuple[str, ...]
    delegation_decision_id: str
    yielded_at: str
    yield_reason: str = "delegation_wave"
    continuation_version: int = 2
    version: int = WAVE_YIELD_CONTRACT_VERSION

    def __post_init__(self) -> None:
        ver = _require_int(self.version, "version", minimum=1)
        if ver != WAVE_YIELD_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {ver}")

        for str_field in (
            "assignment_id",
            "parent_task_id",
            "principal_id",
            "proposal_id",
            "wave_id",
            "delegation_decision_id",
            "yield_reason",
            "yielded_at",
        ):
            object.__setattr__(self, str_field, _require_nonempty_str(getattr(self, str_field), str_field))

        if self.yield_reason != "delegation_wave":
            raise ValueError(f"Unsupported yield_reason: '{self.yield_reason}'")

        clean_children = _normalize_wave_child_ids(
            self.required_child_task_ids, self.parent_task_id
        )
        object.__setattr__(self, "required_child_task_ids", clean_children)

        cver = _require_int(self.continuation_version, "continuation_version", minimum=2)
        if cver != 2:
            raise ValueError(f"WaveYieldRecord continuation_version must be 2, got {cver}")

        object.__setattr__(self, "yielded_at", normalize_canonical_utc_timestamp(self.yielded_at))
        _require_json_byte_bound(self.to_json(), "WaveYieldRecord")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "assignment_id": self.assignment_id,
            "continuation_version": self.continuation_version,
            "delegation_decision_id": self.delegation_decision_id,
            "parent_task_id": self.parent_task_id,
            "principal_id": self.principal_id,
            "proposal_id": self.proposal_id,
            "required_child_task_ids": list(self.required_child_task_ids),
            "version": self.version,
            "wave_id": self.wave_id,
            "yield_reason": self.yield_reason,
            "yielded_at": self.yielded_at,
        }
        check_no_credential_bearing_keys(data, "WaveYieldRecord.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WaveYieldRecord":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        check_no_credential_bearing_keys(data, "WaveYieldRecord payload")
        allowed = {
            "assignment_id",
            "continuation_version",
            "delegation_decision_id",
            "parent_task_id",
            "principal_id",
            "proposal_id",
            "required_child_task_ids",
            "version",
            "wave_id",
            "yield_reason",
            "yielded_at",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise TypeError(f"Unexpected fields in WaveYieldRecord: {sorted(unexpected)}")
        for req in allowed - {"yield_reason", "continuation_version", "version"}:
            if req not in data:
                raise ValueError(f"{req} is required")
        return cls(
            assignment_id=data["assignment_id"],
            parent_task_id=data["parent_task_id"],
            principal_id=data["principal_id"],
            proposal_id=data["proposal_id"],
            wave_id=data["wave_id"],
            required_child_task_ids=data["required_child_task_ids"],
            delegation_decision_id=data["delegation_decision_id"],
            yielded_at=data["yielded_at"],
            yield_reason=data.get("yield_reason", "delegation_wave"),
            continuation_version=data.get("continuation_version", 2),
            version=data.get("version", WAVE_YIELD_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> "WaveYieldRecord":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class DelegationWaveResponse:
    """Platform response returned on successful multi-child wave acceptance or replay."""

    parent_task_id: str
    proposal_id: str
    wave_id: str
    delegation_decision_id: str
    required_child_task_ids: tuple[str, ...]
    barrier_policy: str
    accepted_at: str
    wave_deadline: str
    child_claim_deadline: str
    required_success_fraction: RequiredSuccessFraction | None = None
    required_success_count: int | None = None
    continuation_version: int = 2
    status: str = "accepted"
    version: int = DELEGATION_WAVE_RESPONSE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        ver = _require_int(self.version, "version", minimum=1)
        if ver != DELEGATION_WAVE_RESPONSE_CONTRACT_VERSION:
            raise ValueError(f"Unsupported contract version: {ver}")

        for str_field in (
            "parent_task_id",
            "proposal_id",
            "wave_id",
            "delegation_decision_id",
            "accepted_at",
            "wave_deadline",
            "child_claim_deadline",
            "status",
        ):
            object.__setattr__(self, str_field, _require_nonempty_str(getattr(self, str_field), str_field))

        clean_children = _normalize_wave_child_ids(
            self.required_child_task_ids, self.parent_task_id
        )
        object.__setattr__(self, "required_child_task_ids", clean_children)

        clean_policy, parsed_fraction, derived_count = _validate_barrier_and_threshold(
            self.barrier_policy,
            len(clean_children),
            self.required_success_fraction,
            self.required_success_count,
        )
        object.__setattr__(self, "barrier_policy", clean_policy)
        object.__setattr__(self, "required_success_fraction", parsed_fraction)
        object.__setattr__(self, "required_success_count", derived_count)

        cver = _require_int(self.continuation_version, "continuation_version", minimum=2)
        if cver != 2:
            raise ValueError(f"DelegationWaveResponse continuation_version must be 2, got {cver}")
        object.__setattr__(self, "accepted_at", normalize_canonical_utc_timestamp(self.accepted_at))
        object.__setattr__(self, "wave_deadline", normalize_canonical_utc_timestamp(self.wave_deadline))
        object.__setattr__(
            self, "child_claim_deadline", normalize_canonical_utc_timestamp(self.child_claim_deadline)
        )
        _require_json_byte_bound(self.to_json(), "DelegationWaveResponse")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "accepted_at": self.accepted_at,
            "barrier_policy": self.barrier_policy,
            "child_claim_deadline": self.child_claim_deadline,
            "continuation_version": self.continuation_version,
            "delegation_decision_id": self.delegation_decision_id,
            "parent_task_id": self.parent_task_id,
            "proposal_id": self.proposal_id,
            "required_child_task_ids": list(self.required_child_task_ids),
            "required_success_count": self.required_success_count,
            "required_success_fraction": (
                self.required_success_fraction.to_dict()
                if self.required_success_fraction is not None
                else None
            ),
            "status": self.status,
            "version": self.version,
            "wave_deadline": self.wave_deadline,
            "wave_id": self.wave_id,
        }
        check_no_credential_bearing_keys(data, "DelegationWaveResponse.to_dict")
        return data

    def to_json(self) -> str:
        return canonical_json_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegationWaveResponse":
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        check_no_credential_bearing_keys(data, "DelegationWaveResponse payload")
        allowed = {
            "accepted_at",
            "barrier_policy",
            "child_claim_deadline",
            "continuation_version",
            "delegation_decision_id",
            "parent_task_id",
            "proposal_id",
            "required_child_task_ids",
            "required_success_count",
            "required_success_fraction",
            "status",
            "version",
            "wave_deadline",
            "wave_id",
        }
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise TypeError(f"Unexpected fields in DelegationWaveResponse: {sorted(unexpected)}")
        for req in allowed - {
            "continuation_version",
            "required_success_count",
            "required_success_fraction",
            "status",
            "version",
        }:
            if req not in data:
                raise ValueError(f"{req} is required")
        return cls(
            parent_task_id=data["parent_task_id"],
            proposal_id=data["proposal_id"],
            wave_id=data["wave_id"],
            delegation_decision_id=data["delegation_decision_id"],
            required_child_task_ids=data["required_child_task_ids"],
            barrier_policy=data["barrier_policy"],
            accepted_at=data["accepted_at"],
            wave_deadline=data["wave_deadline"],
            child_claim_deadline=data["child_claim_deadline"],
            required_success_fraction=data.get("required_success_fraction"),
            required_success_count=data.get("required_success_count"),
            continuation_version=data.get("continuation_version", 2),
            status=data.get("status", "accepted"),
            version=data.get("version", DELEGATION_WAVE_RESPONSE_CONTRACT_VERSION),
        )

    @classmethod
    def from_json(cls, value: str) -> "DelegationWaveResponse":
        if not isinstance(value, str):
            raise TypeError(f"value must be a string, got {type(value).__name__}")
        return cls.from_dict(json.loads(value))
