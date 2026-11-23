import json
import logging
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .domain.principal import parse_master_secret, validate_principal_config_registry

logger = logging.getLogger(__name__)

ALLOWED_DELIVERY_MODES = frozenset({"legacy_push", "authenticated_claim"})


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    app_env: str = "development"
    task_delivery_mode: str = "legacy_push"
    claim_delivery_lease_seconds: float = 30.0
    child_claim_deadline_seconds: float = 120.0
    execution_claim_timeout_seconds: float = 900.0
    runtime_credential_pepper: str = ""
    assignment_proof_pepper: str = ""
    runtime_principals: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_delivery_mode not in ALLOWED_DELIVERY_MODES:
            raise ValueError(
                f"Invalid task_delivery_mode: '{self.task_delivery_mode}'. Allowed: {sorted(ALLOWED_DELIVERY_MODES)}"
            )

        if type(self.claim_delivery_lease_seconds) not in (int, float) or self.claim_delivery_lease_seconds <= 0:
            raise ValueError(
                f"claim_delivery_lease_seconds must be a positive number, got {self.claim_delivery_lease_seconds}"
            )

        if type(self.child_claim_deadline_seconds) not in (int, float) or not (
            10.0 <= float(self.child_claim_deadline_seconds) <= 3600.0
        ):
            raise ValueError(
                f"child_claim_deadline_seconds must be a number between 10.0 and 3600.0, got {self.child_claim_deadline_seconds}"
            )

        if type(self.execution_claim_timeout_seconds) not in (int, float) or not (
            60.0 <= float(self.execution_claim_timeout_seconds) <= 7200.0
        ):
            raise ValueError(
                f"execution_claim_timeout_seconds must be a number between 60.0 and 7200.0, got {self.execution_claim_timeout_seconds}"
            )

        if not isinstance(self.runtime_credential_pepper, str):
            raise TypeError("runtime_credential_pepper must be a string")
        if not isinstance(self.assignment_proof_pepper, str):
            raise TypeError("assignment_proof_pepper must be a string")
        if not isinstance(self.runtime_principals, (dict, MappingProxyType)):
            raise TypeError("runtime_principals must be a dictionary of principal configurations")

        if self.runtime_credential_pepper != self.runtime_credential_pepper.strip():
            raise ValueError(
                "Invalid runtime_credential_pepper configuration: value cannot contain leading or trailing whitespace"
            )
        if self.assignment_proof_pepper != self.assignment_proof_pepper.strip():
            raise ValueError(
                "Invalid assignment_proof_pepper configuration: value cannot contain leading or trailing whitespace"
            )

        has_runtime_pepper = bool(self.runtime_credential_pepper)
        has_proof_pepper = bool(self.assignment_proof_pepper)
        has_principals = bool(self.runtime_principals)

        lifecycle_fields_absent = (
            not has_runtime_pepper
            and not has_proof_pepper
            and not has_principals
        )
        lifecycle_fields_complete = (
            has_runtime_pepper
            and has_proof_pepper
            and has_principals
        )

        if not lifecycle_fields_absent and not lifecycle_fields_complete:
            # Partial lifecycle configuration in either mode fails closed
            if not has_runtime_pepper:
                raise ValueError(
                    "runtime_credential_pepper must be configured when lifecycle configuration is present"
                )
            if not has_proof_pepper:
                raise ValueError(
                    "assignment_proof_pepper must be configured when lifecycle configuration is present"
                )
            if not has_principals:
                raise ValueError(
                    "runtime_principals must contain at least one principal when lifecycle configuration is present"
                )

        if self.task_delivery_mode == "authenticated_claim" and lifecycle_fields_absent:
            # authenticated_claim requires full lifecycle configuration
            if not has_runtime_pepper:
                raise ValueError(
                    "runtime_credential_pepper must be configured when task_delivery_mode is 'authenticated_claim'"
                )
            if not has_proof_pepper:
                raise ValueError(
                    "assignment_proof_pepper must be configured when task_delivery_mode is 'authenticated_claim'"
                )
            if not has_principals:
                raise ValueError(
                    "runtime_principals must contain at least one principal when task_delivery_mode is 'authenticated_claim'"
                )

        if lifecycle_fields_complete:
            parse_master_secret(self.runtime_credential_pepper)
            parse_master_secret(self.assignment_proof_pepper)
            validated = validate_principal_config_registry(self.runtime_principals)
            object.__setattr__(
                self,
                "runtime_principals",
                MappingProxyType(validated),
            )
        else:
            object.__setattr__(
                self,
                "runtime_principals",
                MappingProxyType({}),
            )

    @property
    def assignment_lifecycle_configured(self) -> bool:
        return bool(
            self.runtime_credential_pepper
            and self.assignment_proof_pepper
            and self.runtime_principals
        )

    @property
    def executable_delegation_enabled(self) -> bool:
        return (
            self.task_delivery_mode == "authenticated_claim"
            and self.assignment_lifecycle_configured
        )

    @classmethod
    def from_env(cls) -> "Settings":
        raw_mode = os.getenv("TASK_DELIVERY_MODE", "legacy_push").strip()

        raw_lease = os.getenv("CLAIM_DELIVERY_LEASE_SECONDS")
        if raw_lease is not None:
            clean_lease = raw_lease.strip()
            try:
                lease_seconds = float(clean_lease)
            except (ValueError, TypeError):
                raise ValueError("Invalid CLAIM_DELIVERY_LEASE_SECONDS configuration") from None
            if lease_seconds <= 0:
                raise ValueError("Invalid CLAIM_DELIVERY_LEASE_SECONDS configuration")
        else:
            lease_seconds = 30.0

        raw_deadline = os.getenv("CHILD_CLAIM_DEADLINE_SECONDS")
        if raw_deadline is not None:
            clean_deadline = raw_deadline.strip()
            try:
                deadline_seconds = float(clean_deadline)
            except (ValueError, TypeError):
                raise ValueError("Invalid CHILD_CLAIM_DEADLINE_SECONDS configuration") from None
            if not (10.0 <= deadline_seconds <= 3600.0):
                raise ValueError("Invalid CHILD_CLAIM_DEADLINE_SECONDS configuration")
        else:
            deadline_seconds = 120.0

        raw_timeout = os.getenv("EXECUTION_CLAIM_TIMEOUT_SECONDS")
        if raw_timeout is not None:
            clean_timeout = raw_timeout.strip()
            try:
                timeout_seconds = float(clean_timeout)
            except (ValueError, TypeError):
                raise ValueError("Invalid EXECUTION_CLAIM_TIMEOUT_SECONDS configuration") from None
            if not (60.0 <= timeout_seconds <= 7200.0):
                raise ValueError("Invalid EXECUTION_CLAIM_TIMEOUT_SECONDS configuration")
        else:
            timeout_seconds = 900.0

        raw_runtime_pepper = os.getenv("RUNTIME_CREDENTIAL_PEPPER")
        if raw_runtime_pepper is not None:
            if not raw_runtime_pepper:
                raise ValueError(
                    "Invalid RUNTIME_CREDENTIAL_PEPPER configuration: value cannot be empty"
                )
            if raw_runtime_pepper != raw_runtime_pepper.strip():
                raise ValueError(
                    "Invalid RUNTIME_CREDENTIAL_PEPPER configuration: value cannot contain leading or trailing whitespace"
                )

        raw_proof_pepper = os.getenv("ASSIGNMENT_PROOF_PEPPER")
        if raw_proof_pepper is not None:
            if not raw_proof_pepper:
                raise ValueError(
                    "Invalid ASSIGNMENT_PROOF_PEPPER configuration: value cannot be empty"
                )
            if raw_proof_pepper != raw_proof_pepper.strip():
                raise ValueError(
                    "Invalid ASSIGNMENT_PROOF_PEPPER configuration: value cannot contain leading or trailing whitespace"
                )

        raw_principals = os.getenv("RUNTIME_PRINCIPALS_JSON")
        if raw_principals is not None and not raw_principals.strip():
            raise ValueError(
                "Invalid RUNTIME_PRINCIPALS_JSON configuration: value cannot be empty or whitespace-only"
            )

        principals: dict[str, dict[str, Any]] = {}
        if raw_principals is not None:
            clean_principals = raw_principals.strip()
            try:
                parsed = json.loads(clean_principals)
            except (json.JSONDecodeError, TypeError, ValueError):
                raise ValueError("Invalid RUNTIME_PRINCIPALS_JSON configuration") from None

            if isinstance(parsed, dict):
                if not parsed:
                    raise ValueError("Invalid RUNTIME_PRINCIPALS_JSON configuration: empty registry")
                for k, v in parsed.items():
                    if not isinstance(k, str) or not k.strip() or not isinstance(v, dict):
                        raise ValueError("Unsupported RUNTIME_PRINCIPALS_JSON structure")
                principals = parsed
            elif isinstance(parsed, list):
                if not parsed:
                    raise ValueError("Invalid RUNTIME_PRINCIPALS_JSON configuration: empty registry")
                for item in parsed:
                    if (
                        not isinstance(item, dict)
                        or "principal_id" not in item
                        or not isinstance(item["principal_id"], str)
                        or not item["principal_id"].strip()
                    ):
                        raise ValueError("Unsupported RUNTIME_PRINCIPALS_JSON structure")
                    principals[item["principal_id"]] = item
            else:
                raise ValueError("Unsupported RUNTIME_PRINCIPALS_JSON structure")

        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            app_env=os.getenv("APP_ENV", "development"),
            task_delivery_mode=raw_mode,
            claim_delivery_lease_seconds=lease_seconds,
            child_claim_deadline_seconds=deadline_seconds,
            execution_claim_timeout_seconds=timeout_seconds,
            runtime_credential_pepper=raw_runtime_pepper if raw_runtime_pepper is not None else "",
            assignment_proof_pepper=raw_proof_pepper if raw_proof_pepper is not None else "",
            runtime_principals=principals,
        )
