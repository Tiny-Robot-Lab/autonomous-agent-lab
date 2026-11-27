"""Isolated Run 002 ML subprocess worker entrypoint.

Security invariants:
- No ML libraries (setfit, torch, transformers, sentence_transformers, sklearn,
  joblib, pickle) are imported at module top level.
- Artifact trust verification (contract, protocol, manifest, exact bidirectional
  file set, SHA-256 hashes, and persisted label order) executes BEFORE any
  ML import or deserialization.
- Offline environment settings and in-process socket/DNS/HTTP egress guards are
  installed BEFORE importing ML libraries.
- Model loading passes local_files_only=True and trust_remote_code=False.
- Runtime model label ordering is verified before accepting inference requests.
- Raw exceptions, paths, hostnames, usernames, environment variables, and
  prompts are never emitted in IPC error responses.
"""

from __future__ import annotations

import ctypes
import http.client
import importlib.metadata
import inspect
import os
import platform
import socket
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

from evaluation.routing.run_002.contract import EXPECTED_LABEL_INDEX_MAP, load_verified_contract
from evaluation.routing.run_002.errors import FatalRunError, FatalRunErrorCode, PerRecordErrorCode
from evaluation.routing.run_002.ipc import (
    MAX_IPC_REQUEST_BYTES,
    MAX_IPC_RESPONSE_BYTES,
    FatalErrorResponse,
    InferErrorResponse,
    InferOkResponse,
    InferRequest,
    IPCProtocolError,
    ShutdownOkResponse,
    ShutdownRequest,
    StartupOkResponse,
    StartupRequest,
    decode_parent_request,
    encode_ipc_frame,
)
from evaluation.routing.run_002.trust import ArtifactTrustAnchors, verify_artifact_trust
from evaluation.routing.run_002.worker import (
    ValidatedWorkerPrediction,
    WorkerFailure,
    validate_predict_proba_output,
)

_AF_UNIX = getattr(socket, "AF_UNIX", -1)
_SAFE_FATAL_DETAIL_CODES: Mapping[FatalRunErrorCode, str] = {
    FatalRunErrorCode.ARTIFACT_MANIFEST_MISSING: "artifact_manifest_missing",
    FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH: "artifact_identity_mismatch",
    FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH: "protocol_identity_mismatch",
    FatalRunErrorCode.LABEL_MAP_MISMATCH: "label_map_mismatch",
    FatalRunErrorCode.NETWORK_ACCESS_ATTEMPTED: "network_access_attempted",
    FatalRunErrorCode.CORPUS_NOT_APPROVED: "corpus_not_approved",
    FatalRunErrorCode.UNSUPPORTED_ROUTER_MODE: "unsupported_router_mode",
    FatalRunErrorCode.EMPTY_NORMALIZED_PROMPT: "empty_normalized_prompt",
    FatalRunErrorCode.TELEMETRY_WRITE_FAILED: "telemetry_write_failed",
}


class NetworkAccessBlockedError(RuntimeError):
    """Raised when any outbound network or DNS attempt occurs in the offline worker."""

    def __init__(self) -> None:
        super().__init__("Outbound network access is prohibited during offline Run 002 evaluation")


def disable_core_dumps() -> None:
    """Disable POSIX core dump generation for the isolated worker process.

    Prevents native crashes (such as SIGSEGV) from writing core files or
    invoking OS core-dump handlers that can leak process memory or delay
    subprocess termination past parent inference deadlines.
    """
    if resource is not None:
        rlimit_core = getattr(resource, "RLIMIT_CORE", None)
        if rlimit_core is not None:
            try:
                resource.setrlimit(rlimit_core, (0, 0))
            except (OSError, ValueError):
                pass

    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None)
            prctl = getattr(libc, "prctl", None)
            if prctl is not None:
                # PR_SET_DUMPABLE = 4; disabling dumpability also bypasses
                # Linux pipe core_pattern handlers (e.g., apport/systemd-coredump).
                prctl(
                    ctypes.c_int(4),
                    ctypes.c_ulong(0),
                    ctypes.c_ulong(0),
                    ctypes.c_ulong(0),
                    ctypes.c_ulong(0),
                )
        except OSError:
            pass


def install_offline_network_guard() -> None:
    """Enforce strict in-process offline environment flags and block socket/HTTP egress.

    Note: this is an in-process Python socket/DNS/HTTP guard rather than OS-level
    kernel network namespace isolation.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _blocked_network(*_args: Any, **_kwargs: Any) -> Any:
        raise NetworkAccessBlockedError()

    orig_socket_cls = socket.socket

    class _OfflineGuardedSocket(orig_socket_cls):
        def __init__(
            self,
            family: int = socket.AF_INET,
            type: int = socket.SOCK_STREAM,
            proto: int = 0,
            fileno: int | None = None,
        ) -> None:
            if family != _AF_UNIX:
                raise NetworkAccessBlockedError()
            super().__init__(family=family, type=type, proto=proto, fileno=fileno)

        def connect(self, address: Any) -> None:
            if self.family != _AF_UNIX:
                raise NetworkAccessBlockedError()
            return super().connect(address)

        def connect_ex(self, address: Any) -> int:
            if self.family != _AF_UNIX:
                raise NetworkAccessBlockedError()
            return super().connect_ex(address)

        def sendto(self, data: Any, *args: Any) -> int:
            if self.family != _AF_UNIX:
                raise NetworkAccessBlockedError()
            return super().sendto(data, *args)

        def sendmsg(self, buffers: Any, *args: Any) -> int:
            if self.family != _AF_UNIX:
                raise NetworkAccessBlockedError()
            return super().sendmsg(buffers, *args)

    socket.socket = _OfflineGuardedSocket  # type: ignore[misc]
    socket.create_connection = _blocked_network  # type: ignore[assignment]
    socket.getaddrinfo = _blocked_network  # type: ignore[assignment]
    socket.gethostbyname = _blocked_network  # type: ignore[assignment]
    socket.gethostbyname_ex = _blocked_network  # type: ignore[assignment]
    socket.getnameinfo = _blocked_network  # type: ignore[assignment]
    urllib.request.urlopen = _blocked_network  # type: ignore[assignment]

    class _BlockedHttpConnection:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise NetworkAccessBlockedError()

    http.client.HTTPConnection = _BlockedHttpConnection  # type: ignore[misc]
    http.client.HTTPSConnection = _BlockedHttpConnection  # type: ignore[misc]


def _is_network_exception(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, NetworkAccessBlockedError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _normalize_runtime_label_sequence(raw: object) -> list[str]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Runtime label sequence has invalid type",
        )
    items = list(raw)
    if len(items) != 2 or any(not isinstance(item, str) or not item for item in items):
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Runtime label sequence must contain exactly two non-empty strings",
        )
    if len(set(items)) != 2 or items != ["general", "research"]:
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Runtime label sequence does not match [general, research]",
        )
    return items


def _normalize_runtime_id2label(raw_map: object) -> dict[str, str]:
    if not isinstance(raw_map, Mapping) or len(raw_map) == 0:
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Runtime id2label mapping must be a non-empty mapping",
        )
    normalized: dict[str, str] = {}
    for k, v in raw_map.items():
        if isinstance(k, bool) or not isinstance(k, (int, str)):
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Runtime id2label key has invalid type",
            )
        if not isinstance(v, str) or not v:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Runtime id2label value must be non-empty string",
            )
        key_str = str(k)
        if key_str in normalized:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Duplicate key in runtime id2label mapping",
            )
        normalized[key_str] = v
    if len(set(normalized.values())) != 2 or normalized != dict(EXPECTED_LABEL_INDEX_MAP):
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Runtime id2label mapping does not match 0->general, 1->research",
        )
    return normalized


def _normalize_runtime_label2id(raw_map: object) -> dict[str, int]:
    if not isinstance(raw_map, Mapping) or len(raw_map) == 0:
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Runtime label2id mapping must be a non-empty mapping",
        )
    normalized: dict[str, int] = {}
    for k, v in raw_map.items():
        if not isinstance(k, str) or not k:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Runtime label2id key must be non-empty string",
            )
        if isinstance(v, bool) or not isinstance(v, int):
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Runtime label2id index must be integer",
            )
        normalized[k] = v
    if normalized != {"general": 0, "research": 1}:
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Runtime label2id mapping does not match general->0, research->1",
        )
    return normalized


def verify_runtime_label_map(model: Any) -> Mapping[str, str]:
    """Verify loaded model runtime class-column ordering matches 0->general, 1->research.

    Does not infer ordering from configuration; inspects runtime model labels,
    id2label/label2id mappings, and classifier head classes_/out_features.
    """
    verified_signals = 0

    labels_attr = getattr(model, "labels", None)
    if labels_attr is not None:
        _normalize_runtime_label_sequence(labels_attr)
        verified_signals += 1

    id2label_attr = getattr(model, "id2label", None)
    if isinstance(id2label_attr, Mapping) and len(id2label_attr) > 0:
        _normalize_runtime_id2label(id2label_attr)
        verified_signals += 1

    label2id_attr = getattr(model, "label2id", None)
    if isinstance(label2id_attr, Mapping) and len(label2id_attr) > 0:
        _normalize_runtime_label2id(label2id_attr)
        verified_signals += 1

    head = getattr(model, "model_head", None)
    classes_attr = getattr(head, "classes_", None) if head is not None else None
    if classes_attr is None:
        classes_attr = getattr(model, "classes_", None)

    if classes_attr is not None:
        raw_classes = classes_attr.tolist() if hasattr(classes_attr, "tolist") else list(classes_attr)
        if not isinstance(raw_classes, list) or len(raw_classes) != 2:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Runtime classifier classes_ must contain exactly two classes",
            )
        if all(isinstance(c, str) for c in raw_classes):
            _normalize_runtime_label_sequence(raw_classes)
            verified_signals += 1
        elif all(not isinstance(c, bool) and isinstance(c, int) for c in raw_classes):
            if raw_classes != [0, 1]:
                raise FatalRunError(
                    FatalRunErrorCode.LABEL_MAP_MISMATCH,
                    "Runtime integer classifier classes_ order does not match [0, 1]",
                )
            if labels_attr is None and not (
                isinstance(id2label_attr, Mapping) and len(id2label_attr) > 0
            ):
                raise FatalRunError(
                    FatalRunErrorCode.LABEL_MAP_MISMATCH,
                    "Runtime integer classifier classes_ lacks explicit label names",
                )
            verified_signals += 1
        else:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Runtime classifier classes_ has invalid element types",
            )
    elif head is not None:
        out_features = getattr(head, "out_features", None)
        if out_features is None:
            linear = getattr(head, "linear", None)
            out_features = getattr(linear, "out_features", None) if linear is not None else None
        if out_features is not None:
            if isinstance(out_features, bool) or out_features != 2:
                raise FatalRunError(
                    FatalRunErrorCode.LABEL_MAP_MISMATCH,
                    "Runtime differentiable head out_features must equal 2",
                )
            if labels_attr is None and not (
                isinstance(id2label_attr, Mapping) and len(id2label_attr) > 0
            ):
                raise FatalRunError(
                    FatalRunErrorCode.LABEL_MAP_MISMATCH,
                    "Runtime head out_features lacks explicit label names",
                )
            verified_signals += 1

    if verified_signals == 0:
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Loaded runtime model exposes no verifiable class-label ordering",
        )

    return EXPECTED_LABEL_INDEX_MAP


def _load_offline_setfit_model(artifact_dir: Path) -> Any:
    """Import SetFit lazily after verification and load model strictly offline."""
    # SetFit lives only in the operator-supplied external worker environment.
    from setfit import SetFitModel  # type: ignore[import-not-found]

    from_pretrained = SetFitModel.from_pretrained
    sig = inspect.signature(from_pretrained)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not accepts_kwargs and "local_files_only" not in sig.parameters:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Model loader does not support required local_files_only flag",
        )

    load_kwargs: dict[str, Any] = {"local_files_only": True}
    if accepts_kwargs or "trust_remote_code" in sig.parameters:
        load_kwargs["trust_remote_code"] = False

    return from_pretrained(str(artifact_dir), **load_kwargs)


def _collect_worker_environment(model: Any) -> dict[str, Any]:
    raw_release = platform.release()
    release_major = raw_release.split(".", maxsplit=1)[0].split("-", maxsplit=1)[0] or "unknown"

    package_versions: dict[str, str] = {}
    for pkg in ("setfit", "torch", "transformers", "sentence-transformers", "scikit-learn", "numpy"):
        try:
            ver = importlib.metadata.version(pkg)
            if isinstance(ver, str) and ver.strip():
                package_versions[pkg] = ver.strip()
        except importlib.metadata.PackageNotFoundError:
            continue

    raw_device = str(getattr(model, "device", "cpu")).lower()
    if "cuda" in raw_device:
        device_class = "cuda"
    elif "mps" in raw_device:
        device_class = "mps"
    else:
        device_class = "cpu"

    return {
        "device_class": device_class,
        "machine_architecture": platform.machine() or "unknown",
        "package_versions": package_versions,
        "platform_release_major": release_major,
        "platform_system": platform.system() or sys.platform,
        "python_version": platform.python_version(),
    }


def _write_response(payload: Mapping[str, Any]) -> None:
    encoded = encode_ipc_frame(payload, max_bytes=MAX_IPC_RESPONSE_BYTES)
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def _send_fatal(request_id: int, code: FatalRunErrorCode) -> None:
    detail_code = _SAFE_FATAL_DETAIL_CODES.get(code, "fatal_worker_error")
    _write_response(
        FatalErrorResponse(
            request_id=request_id,
            error_code=code,
            detail_code=detail_code,
        ).to_payload()
    )


def main() -> int:
    """Run the isolated Run 002 worker IPC loop."""
    disable_core_dumps()

    raw_startup = sys.stdin.buffer.readline(MAX_IPC_REQUEST_BYTES + 1)
    if not raw_startup:
        return 2

    try:
        startup_msg = decode_parent_request(raw_startup)
    except IPCProtocolError:
        return 2

    if not isinstance(startup_msg, StartupRequest):
        return 2

    # 1. Verify contract, protocol, manifest, files, and persisted labels BEFORE any ML import.
    try:
        load_verified_contract(Path(startup_msg.contract_path))
        anchors = ArtifactTrustAnchors(
            expected_model_artifact_identity=startup_msg.expected_model_artifact_identity,
            expected_protocol_identity=startup_msg.expected_protocol_identity,
            expected_label_index_map=EXPECTED_LABEL_INDEX_MAP,
        )
        verified = verify_artifact_trust(
            artifact_dir=Path(startup_msg.artifact_dir),
            artifact_manifest_path=Path(startup_msg.artifact_manifest_path),
            protocol_path=Path(startup_msg.protocol_path),
            contract_path=Path(startup_msg.contract_path),
            trust_anchors=anchors,
        )
    except FatalRunError as exc:
        _send_fatal(startup_msg.request_id, exc.code)
        return 2

    # 2. Enforce offline flags and network-egress guard BEFORE importing ML modules.
    install_offline_network_guard()

    # 3. Lazily import SetFit, load model strictly offline, and verify runtime label map.
    try:
        model = _load_offline_setfit_model(verified.artifact_dir)
        runtime_label_map = verify_runtime_label_map(model)
    except FatalRunError as exc:
        _send_fatal(startup_msg.request_id, exc.code)
        return 2
    except Exception as exc:  # noqa: BLE001
        if _is_network_exception(exc):
            _send_fatal(startup_msg.request_id, FatalRunErrorCode.NETWORK_ACCESS_ATTEMPTED)
        else:
            _send_fatal(startup_msg.request_id, FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH)
        return 2

    worker_env = _collect_worker_environment(model)
    _write_response(
        StartupOkResponse(
            request_id=startup_msg.request_id,
            artifact_identity=verified.artifact_identity,
            protocol_identity=verified.protocol_identity,
            label_index_map=runtime_label_map,
            worker_environment=worker_env,
        ).to_payload()
    )

    next_expected_request_id = startup_msg.request_id + 1
    while True:
        raw_line = sys.stdin.buffer.readline(MAX_IPC_REQUEST_BYTES + 1)
        if not raw_line:
            break

        try:
            req = decode_parent_request(raw_line)
        except IPCProtocolError:
            _write_response(
                InferErrorResponse(
                    request_id=max(1, next_expected_request_id),
                    error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
                    latency_ns=None,
                ).to_payload()
            )
            return 2

        if isinstance(req, ShutdownRequest):
            _write_response(ShutdownOkResponse(request_id=req.request_id).to_payload())
            return 0

        if not isinstance(req, InferRequest):
            return 2

        if req.request_id < next_expected_request_id:
            _write_response(
                InferErrorResponse(
                    request_id=req.request_id,
                    error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
                    latency_ns=None,
                ).to_payload()
            )
            return 2
        next_expected_request_id = req.request_id + 1

        t0 = time.perf_counter_ns()
        try:
            raw_proba = model.predict_proba([req.canonical_prompt])
            latency_ns = max(0, time.perf_counter_ns() - t0)
        except Exception as exc:  # noqa: BLE001
            latency_ns = max(0, time.perf_counter_ns() - t0)
            if _is_network_exception(exc):
                _send_fatal(req.request_id, FatalRunErrorCode.NETWORK_ACCESS_ATTEMPTED)
                return 2
            _write_response(
                InferErrorResponse(
                    request_id=req.request_id,
                    error_code=PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR,
                    latency_ns=latency_ns,
                ).to_payload()
            )
            continue

        validated = validate_predict_proba_output(raw_proba, latency_ns=latency_ns)
        if isinstance(validated, WorkerFailure):
            _write_response(
                InferErrorResponse(
                    request_id=req.request_id,
                    error_code=validated.error_code,
                    latency_ns=validated.latency_ns,
                ).to_payload()
            )
            continue

        assert isinstance(validated, ValidatedWorkerPrediction)
        _write_response(
            InferOkResponse(
                request_id=req.request_id,
                general_probability=validated.general_probability,
                research_probability=validated.research_probability,
                shadow_decision=validated.shadow_decision,
                latency_ns=validated.latency_ns,
            ).to_payload()
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
