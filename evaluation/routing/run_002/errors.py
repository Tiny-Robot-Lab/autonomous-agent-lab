"""Closed fatal and per-record error types for Run 002 offline shadow evaluation."""

from __future__ import annotations

from enum import StrEnum


class FatalRunErrorCode(StrEnum):
    """Closed fatal run-aborting error codes per the Run 002 shadow contract."""

    EMPTY_NORMALIZED_PROMPT = "empty_normalized_prompt"
    CORPUS_NOT_APPROVED = "corpus_not_approved"
    UNSUPPORTED_ROUTER_MODE = "unsupported_router_mode"
    ARTIFACT_MANIFEST_MISSING = "artifact_manifest_missing"
    ARTIFACT_IDENTITY_MISMATCH = "artifact_identity_mismatch"
    PROTOCOL_IDENTITY_MISMATCH = "protocol_identity_mismatch"
    LABEL_MAP_MISMATCH = "label_map_mismatch"
    NETWORK_ACCESS_ATTEMPTED = "network_access_attempted"
    TELEMETRY_WRITE_FAILED = "telemetry_write_failed"


class PerRecordErrorCode(StrEnum):
    """Closed per-record continuation error codes per the Run 002 shadow contract."""

    INFERENCE_TIMEOUT = "inference_timeout"
    MALFORMED_OUTPUT = "malformed_output"
    INVALID_PROBABILITIES = "invalid_probabilities"
    UNEXPECTED_OBSERVER_ERROR = "unexpected_observer_error"


class FatalRunError(Exception):
    """Run-aborting integrity, preflight, or telemetry failure.

    Stores a closed error code and sanitized diagnostic message with no raw
    exception text, prompts, or secrets.
    """

    def __init__(self, code: FatalRunErrorCode, detail: str) -> None:
        if not isinstance(code, FatalRunErrorCode):
            raise TypeError(f"code must be FatalRunErrorCode, got {type(code).__name__}")
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError("detail must be a non-empty string")
        self.code = code
        self.detail = detail.strip()
        super().__init__(f"{self.code.value}: {self.detail}")
