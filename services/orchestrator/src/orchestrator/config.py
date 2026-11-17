import json
import logging
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .domain.principal import parse_master_secret

logger = logging.getLogger(__name__)

ALLOWED_DELIVERY_MODES = frozenset({"legacy_push", "authenticated_claim"})


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    app_env: str = "development"
    task_delivery_mode: str = "legacy_push"
    claim_delivery_lease_seconds: float = 30.0
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

        if self.task_delivery_mode == "authenticated_claim":
            if not isinstance(self.runtime_credential_pepper, str) or not self.runtime_credential_pepper.strip():
                raise ValueError(
                    "runtime_credential_pepper must be configured when task_delivery_mode is 'authenticated_claim'"
                )
            parse_master_secret(self.runtime_credential_pepper)

            if not isinstance(self.assignment_proof_pepper, str) or not self.assignment_proof_pepper.strip():
                raise ValueError(
                    "assignment_proof_pepper must be configured when task_delivery_mode is 'authenticated_claim'"
                )
            parse_master_secret(self.assignment_proof_pepper)

            if not isinstance(self.runtime_principals, (dict, MappingProxyType)) or not self.runtime_principals:
                raise ValueError(
                    "runtime_principals must contain at least one principal when task_delivery_mode is 'authenticated_claim'"
                )
            has_active = any(
                isinstance(p, dict) and p.get("active", True) is True
                for p in self.runtime_principals.values()
            )
            if not has_active:
                raise ValueError(
                    "runtime_principals must contain at least one active principal when task_delivery_mode is 'authenticated_claim'"
                )
        else:
            if isinstance(self.runtime_credential_pepper, str) and self.runtime_credential_pepper.strip():
                parse_master_secret(self.runtime_credential_pepper)
            if isinstance(self.assignment_proof_pepper, str) and self.assignment_proof_pepper.strip():
                parse_master_secret(self.assignment_proof_pepper)

        if isinstance(self.runtime_principals, dict):
            object.__setattr__(
                self,
                "runtime_principals",
                MappingProxyType(dict(self.runtime_principals)),
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

        raw_principals = os.getenv("RUNTIME_PRINCIPALS_JSON")
        principals: dict[str, dict[str, Any]] = {}
        if raw_principals is not None:
            clean_principals = raw_principals.strip()
            if clean_principals:
                try:
                    parsed = json.loads(clean_principals)
                except (json.JSONDecodeError, TypeError, ValueError):
                    raise ValueError("Invalid RUNTIME_PRINCIPALS_JSON configuration") from None

                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if not isinstance(k, str) or not k.strip() or not isinstance(v, dict):
                            raise ValueError("Unsupported RUNTIME_PRINCIPALS_JSON structure")
                    principals = parsed
                elif isinstance(parsed, list):
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
            runtime_credential_pepper=os.getenv("RUNTIME_CREDENTIAL_PEPPER", ""),
            assignment_proof_pepper=os.getenv("ASSIGNMENT_PROOF_PEPPER", ""),
            runtime_principals=principals,
        )
