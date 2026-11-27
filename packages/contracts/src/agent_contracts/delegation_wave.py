import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any

from .serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
    normalize_canonical_utc_timestamp,
)

DELEGATION_WAVE_CONTRACT_VERSION: int = 1
WAVE_DEADLINE_POLICY_CONTRACT_VERSION: int = 1

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

        expected_deadline = normalize_canonical_utc_timestamp(
            datetime.fromisoformat(clean_accepted) + timedelta(seconds=expected_duration)
        )
        clean_deadline = normalize_canonical_utc_timestamp(self.wave_deadline)
        if clean_deadline != expected_deadline:
            raise ValueError(
                f"wave_deadline ({clean_deadline}) does not match "
                f"accepted_at + derived_duration_seconds ({expected_deadline})"
            )
        object.__setattr__(self, "wave_deadline", clean_deadline)

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
