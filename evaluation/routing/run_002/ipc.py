"""Typed, bounded, privacy-safe IPC schemas for the Run 002 subprocess worker."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from agent_contracts.serialization import (
    canonical_json_dumps,
    check_no_credential_bearing_keys,
)

from evaluation.routing.run_002.errors import FatalRunErrorCode, PerRecordErrorCode

MAX_IPC_REQUEST_BYTES = 256 * 1024
MAX_IPC_RESPONSE_BYTES = 32 * 1024

ALLOWED_WORKER_ENV_FIELDS: frozenset[str] = frozenset({
    "device_class",
    "machine_architecture",
    "package_versions",
    "platform_release_major",
    "platform_system",
    "python_version",
})

_STARTUP_REQ_KEYS: frozenset[str] = frozenset({
    "artifact_dir",
    "artifact_manifest_path",
    "contract_path",
    "expected_model_artifact_identity",
    "expected_protocol_identity",
    "protocol_path",
    "request_id",
    "type",
})

_STARTUP_OK_KEYS: frozenset[str] = frozenset({
    "artifact_identity",
    "label_index_map",
    "protocol_identity",
    "request_id",
    "type",
    "worker_environment",
})

_FATAL_ERROR_KEYS: frozenset[str] = frozenset({
    "detail_code",
    "error_code",
    "request_id",
    "type",
})

_INFER_REQ_KEYS: frozenset[str] = frozenset({
    "canonical_prompt",
    "request_id",
    "type",
})

_INFER_OK_KEYS: frozenset[str] = frozenset({
    "general_probability",
    "latency_ns",
    "request_id",
    "research_probability",
    "shadow_decision",
    "type",
})

_INFER_ERROR_KEYS: frozenset[str] = frozenset({
    "error_code",
    "latency_ns",
    "request_id",
    "type",
})

_SHUTDOWN_REQ_KEYS: frozenset[str] = frozenset({
    "request_id",
    "type",
})

_SHUTDOWN_OK_KEYS: frozenset[str] = frozenset({
    "request_id",
    "type",
})


class IPCProtocolError(ValueError):
    """Raised when an IPC frame violates bounds, JSON rules, or closed schema."""


def _reject_non_finite_json(token: str) -> Any:
    raise IPCProtocolError(f"Non-finite JSON constant prohibited in IPC: {token!r}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise IPCProtocolError("Duplicate JSON key prohibited in IPC")
        obj[key] = value
    return obj


def _require_non_bool_int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IPCProtocolError(f"{field_name} must be integer >= {minimum}")
    return value


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IPCProtocolError(f"{field_name} must be non-empty str")
    return value


def _require_finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IPCProtocolError(f"{field_name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise IPCProtocolError(f"{field_name} must be finite")
    return parsed


def _decode_json_frame(raw_line: bytes, *, max_bytes: int) -> dict[str, Any]:
    if not isinstance(raw_line, (bytes, bytearray)):
        raise IPCProtocolError("IPC frame must be bytes")
    if len(raw_line) == 0:
        raise IPCProtocolError("IPC frame is empty")
    if len(raw_line) > max_bytes:
        raise IPCProtocolError("IPC frame exceeds maximum byte bound")
    try:
        text = raw_line.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise IPCProtocolError("IPC frame contains invalid UTF-8") from exc
    stripped = text.rstrip("\n")
    if not stripped or "\n" in stripped or "\r" in stripped:
        raise IPCProtocolError("IPC frame must be a single line JSON object")
    try:
        parsed = json.loads(
            stripped,
            parse_constant=_reject_non_finite_json,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except IPCProtocolError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise IPCProtocolError("IPC frame JSON is malformed") from exc
    if not isinstance(parsed, dict):
        raise IPCProtocolError("IPC frame must decode to JSON object")
    return parsed


def encode_ipc_frame(payload: Mapping[str, Any], *, max_bytes: int) -> bytes:
    """Serialize an IPC mapping into a bounded, newline-delimited canonical JSON frame."""
    try:
        serialized = canonical_json_dumps(dict(payload))
    except (TypeError, ValueError) as exc:
        raise IPCProtocolError("IPC payload failed canonical JSON serialization") from exc
    if "\n" in serialized or "\r" in serialized:
        raise IPCProtocolError("Serialized IPC frame must not contain raw newlines")
    encoded = (serialized + "\n").encode("utf-8", errors="strict")
    if len(encoded) > max_bytes:
        raise IPCProtocolError("Encoded IPC frame exceeds maximum byte bound")
    return encoded


def validate_worker_environment_payload(raw_env: object) -> Mapping[str, Any]:
    """Validate strictly allowlisted worker environment metadata returned at startup."""
    if not isinstance(raw_env, Mapping):
        raise IPCProtocolError("worker_environment must be a mapping")
    if set(raw_env.keys()) != ALLOWED_WORKER_ENV_FIELDS:
        raise IPCProtocolError("worker_environment keys must match allowlist")
    cleaned: dict[str, Any] = {}
    for field_name in (
        "python_version",
        "platform_system",
        "platform_release_major",
        "machine_architecture",
        "device_class",
    ):
        cleaned[field_name] = _require_non_empty_str(
            raw_env.get(field_name),
            field_name=field_name,
        ).strip()
    raw_pkgs = raw_env.get("package_versions")
    if not isinstance(raw_pkgs, Mapping):
        raise IPCProtocolError("package_versions must be a mapping")
    cleaned_pkgs: dict[str, str] = {}
    for pkg, ver in raw_pkgs.items():
        pkg_str = _require_non_empty_str(pkg, field_name="package_name").strip()
        ver_str = _require_non_empty_str(ver, field_name="package_version").strip()
        cleaned_pkgs[pkg_str] = ver_str
    cleaned["package_versions"] = MappingProxyType(cleaned_pkgs)
    try:
        check_no_credential_bearing_keys(cleaned, context="WorkerEnvironment")
    except (TypeError, ValueError) as exc:
        raise IPCProtocolError("worker_environment failed safety validation") from exc
    return MappingProxyType(cleaned)


@dataclass(frozen=True)
class StartupRequest:
    """Parent -> worker initialization request with verified artifact configuration."""

    request_id: int
    artifact_dir: str
    artifact_manifest_path: str
    protocol_path: str
    contract_path: str
    expected_model_artifact_identity: str
    expected_protocol_identity: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_dir": self.artifact_dir,
            "artifact_manifest_path": self.artifact_manifest_path,
            "contract_path": self.contract_path,
            "expected_model_artifact_identity": self.expected_model_artifact_identity,
            "expected_protocol_identity": self.expected_protocol_identity,
            "protocol_path": self.protocol_path,
            "request_id": self.request_id,
            "type": "startup",
        }


@dataclass(frozen=True)
class StartupOkResponse:
    """Worker -> parent initialization success response."""

    request_id: int
    artifact_identity: str
    protocol_identity: str
    label_index_map: Mapping[str, str]
    worker_environment: Mapping[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_identity": self.artifact_identity,
            "label_index_map": dict(self.label_index_map),
            "protocol_identity": self.protocol_identity,
            "request_id": self.request_id,
            "type": "startup_ok",
            "worker_environment": dict(self.worker_environment),
        }


@dataclass(frozen=True)
class FatalErrorResponse:
    """Worker -> parent privacy-safe fatal error response."""

    request_id: int
    error_code: FatalRunErrorCode
    detail_code: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "detail_code": self.detail_code,
            "error_code": self.error_code.value,
            "request_id": self.request_id,
            "type": "fatal_error",
        }


@dataclass(frozen=True)
class InferRequest:
    """Parent -> worker single-prompt inference request."""

    request_id: int
    canonical_prompt: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "canonical_prompt": self.canonical_prompt,
            "request_id": self.request_id,
            "type": "infer",
        }


@dataclass(frozen=True)
class InferOkResponse:
    """Worker -> parent validated inference output."""

    request_id: int
    general_probability: float
    research_probability: float
    shadow_decision: str
    latency_ns: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "general_probability": self.general_probability,
            "latency_ns": self.latency_ns,
            "request_id": self.request_id,
            "research_probability": self.research_probability,
            "shadow_decision": self.shadow_decision,
            "type": "infer_ok",
        }


@dataclass(frozen=True)
class InferErrorResponse:
    """Worker -> parent privacy-safe per-record inference error response."""

    request_id: int
    error_code: PerRecordErrorCode
    latency_ns: int | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code.value,
            "latency_ns": self.latency_ns,
            "request_id": self.request_id,
            "type": "infer_error",
        }


@dataclass(frozen=True)
class ShutdownRequest:
    """Parent -> worker clean shutdown request."""

    request_id: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "type": "shutdown",
        }


@dataclass(frozen=True)
class ShutdownOkResponse:
    """Worker -> parent shutdown acknowledgment."""

    request_id: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "type": "shutdown_ok",
        }


ParentRequest = StartupRequest | InferRequest | ShutdownRequest
WorkerIpcResponse = (
    StartupOkResponse
    | FatalErrorResponse
    | InferOkResponse
    | InferErrorResponse
    | ShutdownOkResponse
)


def decode_parent_request(raw_line: bytes) -> ParentRequest:
    """Decode and validate one bounded parent -> worker request frame."""
    data = _decode_json_frame(raw_line, max_bytes=MAX_IPC_REQUEST_BYTES)
    msg_type = data.get("type")
    if msg_type == "startup":
        if set(data.keys()) != _STARTUP_REQ_KEYS:
            raise IPCProtocolError("Startup request keys mismatch")
        return StartupRequest(
            request_id=_require_non_bool_int(data["request_id"], field_name="request_id", minimum=0),
            artifact_dir=_require_non_empty_str(data["artifact_dir"], field_name="artifact_dir"),
            artifact_manifest_path=_require_non_empty_str(
                data["artifact_manifest_path"], field_name="artifact_manifest_path"
            ),
            protocol_path=_require_non_empty_str(data["protocol_path"], field_name="protocol_path"),
            contract_path=_require_non_empty_str(data["contract_path"], field_name="contract_path"),
            expected_model_artifact_identity=_require_non_empty_str(
                data["expected_model_artifact_identity"],
                field_name="expected_model_artifact_identity",
            ),
            expected_protocol_identity=_require_non_empty_str(
                data["expected_protocol_identity"],
                field_name="expected_protocol_identity",
            ),
        )
    if msg_type == "infer":
        if set(data.keys()) != _INFER_REQ_KEYS:
            raise IPCProtocolError("Infer request keys mismatch")
        prompt = data["canonical_prompt"]
        if not isinstance(prompt, str) or prompt == "":
            raise IPCProtocolError("canonical_prompt must be a non-empty str")
        return InferRequest(
            request_id=_require_non_bool_int(data["request_id"], field_name="request_id", minimum=1),
            canonical_prompt=prompt,
        )
    if msg_type == "shutdown":
        if set(data.keys()) != _SHUTDOWN_REQ_KEYS:
            raise IPCProtocolError("Shutdown request keys mismatch")
        return ShutdownRequest(
            request_id=_require_non_bool_int(data["request_id"], field_name="request_id", minimum=0),
        )
    raise IPCProtocolError("Unknown parent IPC request type")


def decode_worker_response(raw_line: bytes) -> WorkerIpcResponse:
    """Decode and validate one bounded worker -> parent response frame."""
    data = _decode_json_frame(raw_line, max_bytes=MAX_IPC_RESPONSE_BYTES)
    msg_type = data.get("type")
    if msg_type == "startup_ok":
        if set(data.keys()) != _STARTUP_OK_KEYS:
            raise IPCProtocolError("startup_ok keys mismatch")
        raw_label_map = data["label_index_map"]
        if not isinstance(raw_label_map, Mapping):
            raise IPCProtocolError("label_index_map must be mapping")
        cleaned_label_map: dict[str, str] = {}
        for k, v in raw_label_map.items():
            cleaned_label_map[_require_non_empty_str(k, field_name="label_index")] = (
                _require_non_empty_str(v, field_name="label_name")
            )
        return StartupOkResponse(
            request_id=_require_non_bool_int(data["request_id"], field_name="request_id", minimum=0),
            artifact_identity=_require_non_empty_str(
                data["artifact_identity"], field_name="artifact_identity"
            ),
            protocol_identity=_require_non_empty_str(
                data["protocol_identity"], field_name="protocol_identity"
            ),
            label_index_map=MappingProxyType(cleaned_label_map),
            worker_environment=validate_worker_environment_payload(data["worker_environment"]),
        )
    if msg_type == "fatal_error":
        if set(data.keys()) != _FATAL_ERROR_KEYS:
            raise IPCProtocolError("fatal_error keys mismatch")
        try:
            code = FatalRunErrorCode(_require_non_empty_str(data["error_code"], field_name="error_code"))
        except ValueError as exc:
            raise IPCProtocolError("Unknown fatal error_code in IPC") from exc
        return FatalErrorResponse(
            request_id=_require_non_bool_int(data["request_id"], field_name="request_id", minimum=0),
            error_code=code,
            detail_code=_require_non_empty_str(data["detail_code"], field_name="detail_code"),
        )
    if msg_type == "infer_ok":
        if set(data.keys()) != _INFER_OK_KEYS:
            raise IPCProtocolError("infer_ok keys mismatch")
        decision = _require_non_empty_str(data["shadow_decision"], field_name="shadow_decision")
        if decision not in ("general", "research"):
            raise IPCProtocolError("shadow_decision must be general or research")
        return InferOkResponse(
            request_id=_require_non_bool_int(data["request_id"], field_name="request_id", minimum=1),
            general_probability=_require_finite_float(
                data["general_probability"], field_name="general_probability"
            ),
            research_probability=_require_finite_float(
                data["research_probability"], field_name="research_probability"
            ),
            shadow_decision=decision,
            latency_ns=_require_non_bool_int(data["latency_ns"], field_name="latency_ns", minimum=0),
        )
    if msg_type == "infer_error":
        if set(data.keys()) != _INFER_ERROR_KEYS:
            raise IPCProtocolError("infer_error keys mismatch")
        try:
            code = PerRecordErrorCode(_require_non_empty_str(data["error_code"], field_name="error_code"))
        except ValueError as exc:
            raise IPCProtocolError("Unknown per-record error_code in IPC") from exc
        raw_latency = data["latency_ns"]
        latency = (
            None
            if raw_latency is None
            else _require_non_bool_int(raw_latency, field_name="latency_ns", minimum=0)
        )
        return InferErrorResponse(
            request_id=_require_non_bool_int(data["request_id"], field_name="request_id", minimum=1),
            error_code=code,
            latency_ns=latency,
        )
    if msg_type == "shutdown_ok":
        if set(data.keys()) != _SHUTDOWN_OK_KEYS:
            raise IPCProtocolError("shutdown_ok keys mismatch")
        return ShutdownOkResponse(
            request_id=_require_non_bool_int(data["request_id"], field_name="request_id", minimum=0),
        )
    raise IPCProtocolError("Unknown worker IPC response type")
