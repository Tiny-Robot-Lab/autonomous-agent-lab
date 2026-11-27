"""Parent-side subprocess worker adapter for isolated Run 002 shadow inference."""

from __future__ import annotations

import math
import os
import selectors
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from evaluation.routing.run_002.contract import DEFAULT_CONTRACT_PATH
from evaluation.routing.run_002.errors import FatalRunError, PerRecordErrorCode
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
    decode_worker_response,
    encode_ipc_frame,
)
from evaluation.routing.run_002.telemetry import EnvironmentFingerprint
from evaluation.routing.run_002.trust import (
    ArtifactTrustAnchors,
    VerifiedArtifactBundle,
    resolve_contract_trust_anchors,
    validate_worker_executable,
    verify_artifact_trust,
)
from evaluation.routing.run_002.worker import (
    WorkerFailure,
    WorkerPrediction,
    WorkerResult,
    WorkerStartupError,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACTS_SRC = _REPO_ROOT / "packages" / "contracts" / "src"
_WORKER_ENTRYPOINT_MODULE = "evaluation.routing.run_002.worker_entrypoint"


@dataclass(frozen=True)
class SubprocessWorkerConfig:
    """Operator-supplied configuration for the isolated Run 002 worker subprocess."""

    worker_python: Path
    artifact_dir: Path
    artifact_manifest_path: Path
    protocol_path: Path
    contract_path: Path | None = None
    timeout_seconds: float = 30.0
    startup_timeout_seconds: float = 60.0
    trust_anchors: ArtifactTrustAnchors | None = None
    repository_commit: str = "unknown"

    def __post_init__(self) -> None:
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise TypeError("timeout_seconds must be a positive finite number")
        if not math.isfinite(float(self.timeout_seconds)) or float(self.timeout_seconds) <= 0.0:
            raise ValueError("timeout_seconds must be > 0")

        if isinstance(self.startup_timeout_seconds, bool) or not isinstance(
            self.startup_timeout_seconds, (int, float)
        ):
            raise TypeError("startup_timeout_seconds must be a positive finite number")
        if (
            not math.isfinite(float(self.startup_timeout_seconds))
            or float(self.startup_timeout_seconds) <= 0.0
        ):
            raise ValueError("startup_timeout_seconds must be > 0")

        if not isinstance(self.repository_commit, str) or not self.repository_commit.strip():
            raise ValueError("repository_commit must be a non-empty string")


def _build_sanitized_worker_env(isolated_cache_root: Path) -> dict[str, str]:
    """Construct a minimal child environment stripped of host secrets, proxies, and caches."""
    python_path = f"{_REPO_ROOT}{os.pathsep}{_CONTRACTS_SRC}"
    env: dict[str, str] = {
        "DO_NOT_TRACK": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(isolated_cache_root / "hf"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": python_path,
        "PYTHONUTF8": "1",
        "SENTENCE_TRANSFORMERS_HOME": str(isolated_cache_root / "st"),
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_HOME": str(isolated_cache_root / "torch"),
        "TRANSFORMERS_OFFLINE": "1",
    }
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    if ld_library_path:
        env["LD_LIBRARY_PATH"] = ld_library_path
    return env


def _read_bounded_line_with_timeout(
    proc: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
    max_bytes: int,
) -> tuple[bytes | None, bool, bool]:
    """Read one newline-terminated IPC frame before deadline.

    Returns (frame_bytes, timed_out, oversized).
    """
    stdout = proc.stdout
    if stdout is None:
        return None, False, False

    fd = stdout.fileno()
    buf = bytearray()
    deadline = time.monotonic() + timeout_seconds
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None, True, False
            events = sel.select(timeout=remaining)
            if not events:
                return None, True, False
            chunk = os.read(fd, 4096)
            if not chunk:
                if not buf:
                    return None, False, False
                return bytes(buf), False, False
            buf.extend(chunk)
            if len(buf) > max_bytes:
                return None, False, True
            nl_idx = buf.find(b"\n")
            if nl_idx != -1:
                if nl_idx != len(buf) - 1:
                    # Unexpected trailing bytes after single IPC frame.
                    return bytes(buf), False, False
                return bytes(buf), False, False
    finally:
        sel.close()


def _terminate_and_reap(proc: subprocess.Popen[bytes] | None) -> int | None:
    """Ensure the worker process group is terminated, killed if needed, and reaped."""
    if proc is None:
        return None
    try:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
            try:
                proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
                proc.wait(timeout=2.0)
        rc = proc.wait()
    finally:
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
    return rc


class SubprocessShadowWorker:
    """Offline subprocess-isolated Run 002 shadow worker with verify-before-launch."""

    def __init__(self, config: SubprocessWorkerConfig) -> None:
        if not isinstance(config, SubprocessWorkerConfig):
            raise TypeError("config must be SubprocessWorkerConfig")
        self._config = config
        self._proc: subprocess.Popen[bytes] | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._next_request_id = 1
        self._verified_bundle: VerifiedArtifactBundle | None = None
        self._worker_environment: EnvironmentFingerprint | None = None

    @property
    def worker_environment(self) -> EnvironmentFingerprint | None:
        return self._worker_environment

    @property
    def verified_bundle(self) -> VerifiedArtifactBundle | None:
        return self._verified_bundle

    def _effective_trust_anchors(self) -> ArtifactTrustAnchors:
        if self._config.trust_anchors is not None:
            return self._config.trust_anchors
        return resolve_contract_trust_anchors(self._config.contract_path)

    def _verify_before_launch(self) -> VerifiedArtifactBundle:
        validate_worker_executable(self._config.worker_python)
        anchors = self._effective_trust_anchors()
        bundle = verify_artifact_trust(
            artifact_dir=self._config.artifact_dir,
            artifact_manifest_path=self._config.artifact_manifest_path,
            protocol_path=self._config.protocol_path,
            contract_path=self._config.contract_path,
            trust_anchors=anchors,
        )
        self._verified_bundle = bundle
        return bundle

    def _build_environment_fingerprint(
        self, raw_worker_env: Mapping[str, Any]
    ) -> EnvironmentFingerprint:
        return EnvironmentFingerprint(
            python_version=str(raw_worker_env["python_version"]),
            platform_system=str(raw_worker_env["platform_system"]),
            platform_release_major=str(raw_worker_env["platform_release_major"]),
            machine_architecture=str(raw_worker_env["machine_architecture"]),
            package_versions=dict(raw_worker_env["package_versions"]),
            device_class=str(raw_worker_env["device_class"]),
            repository_commit=self._config.repository_commit.strip(),
        )

    def _ensure_worker_started(self) -> subprocess.Popen[bytes]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self._proc is not None:
            _terminate_and_reap(self._proc)
            self._proc = None

        bundle = self._verify_before_launch()
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="run002_worker_")
        env = _build_sanitized_worker_env(Path(self._temp_dir.name))

        try:
            proc = subprocess.Popen(
                [str(self._config.worker_python), "-u", "-m", _WORKER_ENTRYPOINT_MODULE],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(_REPO_ROOT),
                env=env,
                start_new_session=True,
                bufsize=0,
            )
        except OSError as exc:
            raise WorkerStartupError("Failed to launch isolated worker subprocess") from exc

        contract_path = self._config.contract_path or DEFAULT_CONTRACT_PATH
        anchors = self._effective_trust_anchors()
        startup_req = StartupRequest(
            request_id=0,
            artifact_dir=str(bundle.artifact_dir),
            artifact_manifest_path=str(bundle.artifact_manifest_path),
            protocol_path=str(bundle.protocol_path),
            contract_path=str(contract_path),
            expected_model_artifact_identity=anchors.expected_model_artifact_identity,
            expected_protocol_identity=anchors.expected_protocol_identity,
        )
        try:
            assert proc.stdin is not None
            proc.stdin.write(
                encode_ipc_frame(startup_req.to_payload(), max_bytes=MAX_IPC_REQUEST_BYTES)
            )
            proc.stdin.flush()
        except (OSError, IPCProtocolError) as exc:
            _terminate_and_reap(proc)
            raise WorkerStartupError("Failed to deliver startup request to worker") from exc

        frame, timed_out, oversized = _read_bounded_line_with_timeout(
            proc,
            timeout_seconds=float(self._config.startup_timeout_seconds),
            max_bytes=MAX_IPC_RESPONSE_BYTES,
        )
        if timed_out:
            _terminate_and_reap(proc)
            raise WorkerStartupError("Worker startup timed out")
        if oversized or frame is None:
            _terminate_and_reap(proc)
            raise WorkerStartupError("Worker exited or produced invalid startup output")

        try:
            resp = decode_worker_response(frame)
        except IPCProtocolError as exc:
            _terminate_and_reap(proc)
            raise WorkerStartupError("Worker returned malformed startup IPC response") from exc

        if isinstance(resp, FatalErrorResponse):
            _terminate_and_reap(proc)
            raise FatalRunError(resp.error_code, f"Worker initialization failed ({resp.detail_code})")

        if not isinstance(resp, StartupOkResponse) or resp.request_id != 0:
            _terminate_and_reap(proc)
            raise WorkerStartupError("Worker returned unexpected startup response")

        if (
            resp.artifact_identity != bundle.artifact_identity
            or resp.protocol_identity != bundle.protocol_identity
            or dict(resp.label_index_map) != dict(bundle.label_index_map)
        ):
            _terminate_and_reap(proc)
            raise FatalRunError(
                FatalRunErrorCode_from_mismatch(resp, bundle),
                "Worker verified identities diverged from parent trust bundle",
            )

        self._worker_environment = self._build_environment_fingerprint(resp.worker_environment)
        self._proc = proc
        return proc

    def preflight(self) -> None:
        """Verify artifacts and perform offline worker startup and runtime label checks."""
        self._ensure_worker_started()

    def predict(self, canonical_prompt: str) -> WorkerResult:
        """Execute one bounded inference request in the isolated subprocess worker."""
        if not isinstance(canonical_prompt, str):
            raise TypeError("canonical_prompt must be str")
        if canonical_prompt == "":
            return WorkerFailure(error_code=PerRecordErrorCode.MALFORMED_OUTPUT, latency_ns=None)

        proc = self._ensure_worker_started()
        req_id = self._next_request_id
        self._next_request_id += 1

        req = InferRequest(request_id=req_id, canonical_prompt=canonical_prompt)
        try:
            frame_bytes = encode_ipc_frame(req.to_payload(), max_bytes=MAX_IPC_REQUEST_BYTES)
        except IPCProtocolError:
            return WorkerFailure(error_code=PerRecordErrorCode.MALFORMED_OUTPUT, latency_ns=None)

        t0 = time.perf_counter_ns()
        try:
            assert proc.stdin is not None
            proc.stdin.write(frame_bytes)
            proc.stdin.flush()
        except OSError:
            elapsed_ns = max(0, time.perf_counter_ns() - t0)
            _terminate_and_reap(proc)
            self._proc = None
            return WorkerFailure(
                error_code=PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR,
                latency_ns=elapsed_ns,
            )

        frame, timed_out, oversized = _read_bounded_line_with_timeout(
            proc,
            timeout_seconds=float(self._config.timeout_seconds),
            max_bytes=MAX_IPC_RESPONSE_BYTES,
        )
        elapsed_ns = max(0, time.perf_counter_ns() - t0)

        if timed_out:
            _terminate_and_reap(proc)
            self._proc = None
            return WorkerFailure(
                error_code=PerRecordErrorCode.INFERENCE_TIMEOUT,
                latency_ns=elapsed_ns,
            )

        if oversized:
            _terminate_and_reap(proc)
            self._proc = None
            return WorkerFailure(
                error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
                latency_ns=elapsed_ns,
            )

        if frame is None:
            rc = _terminate_and_reap(proc)
            self._proc = None
            if rc == 0:
                return WorkerFailure(
                    error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
                    latency_ns=elapsed_ns,
                )
            return WorkerFailure(
                error_code=PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR,
                latency_ns=elapsed_ns,
            )

        try:
            resp = decode_worker_response(frame)
        except IPCProtocolError:
            _terminate_and_reap(proc)
            self._proc = None
            return WorkerFailure(
                error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
                latency_ns=elapsed_ns,
            )

        if isinstance(resp, FatalErrorResponse):
            _terminate_and_reap(proc)
            self._proc = None
            raise FatalRunError(resp.error_code, f"Worker aborted run ({resp.detail_code})")

        if isinstance(resp, InferOkResponse):
            if resp.request_id != req_id:
                _terminate_and_reap(proc)
                self._proc = None
                return WorkerFailure(
                    error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
                    latency_ns=elapsed_ns,
                )
            return WorkerPrediction(
                general_probability=resp.general_probability,
                research_probability=resp.research_probability,
                latency_ns=resp.latency_ns,
                shadow_decision=resp.shadow_decision,
            )

        if isinstance(resp, InferErrorResponse):
            if resp.request_id != req_id:
                _terminate_and_reap(proc)
                self._proc = None
                return WorkerFailure(
                    error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
                    latency_ns=elapsed_ns,
                )
            return WorkerFailure(
                error_code=resp.error_code,
                latency_ns=resp.latency_ns,
            )

        _terminate_and_reap(proc)
        self._proc = None
        return WorkerFailure(
            error_code=PerRecordErrorCode.MALFORMED_OUTPUT,
            latency_ns=elapsed_ns,
        )

    def close(self) -> None:
        """Gracefully shut down and reap the worker subprocess if running."""
        proc = self._proc
        self._proc = None
        if proc is not None:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    req_id = self._next_request_id
                    self._next_request_id += 1
                    shutdown_frame = encode_ipc_frame(
                        ShutdownRequest(request_id=req_id).to_payload(),
                        max_bytes=MAX_IPC_REQUEST_BYTES,
                    )
                    proc.stdin.write(shutdown_frame)
                    proc.stdin.flush()
                    frame, timed_out, _ = _read_bounded_line_with_timeout(
                        proc,
                        timeout_seconds=1.0,
                        max_bytes=MAX_IPC_RESPONSE_BYTES,
                    )
                    if not timed_out and frame is not None:
                        resp = decode_worker_response(frame)
                        if isinstance(resp, ShutdownOkResponse):
                            proc.wait(timeout=1.0)
                except (OSError, IPCProtocolError, subprocess.TimeoutExpired):
                    pass
            _terminate_and_reap(proc)

        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except OSError:
                pass
            self._temp_dir = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def FatalRunErrorCode_from_mismatch(
    resp: StartupOkResponse,
    bundle: VerifiedArtifactBundle,
) -> Any:
    from evaluation.routing.run_002.errors import FatalRunErrorCode

    if resp.protocol_identity != bundle.protocol_identity:
        return FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH
    if dict(resp.label_index_map) != dict(bundle.label_index_map):
        return FatalRunErrorCode.LABEL_MAP_MISMATCH
    return FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH
