"""Externally anchored artifact trust and persisted label-map verification for Run 002.

Verifies the pinned repository contract, physical protocol identity, canonical
artifact manifest self-hash and external trust anchor, exact bidirectional
regular-file tree inside the artifact directory, and persisted label-map
declarations before any ML-runtime import or pickle/joblib deserialization.

Local-filesystem TOCTOU boundary:
All artifact files are verified immediately before launching the worker
subprocess, and the worker re-verifies the artifact tree before importing ML
libraries or loading weights. Symlinks and non-regular filesystem objects are
rejected, and verified files are opened with O_NOFOLLOW where supported. This
protects against stale, missing, extra, symlinked, or tampered files under a
maintainer-controlled local filesystem; it does not claim protection against a
hostile concurrent local administrator modifying files between verification and
third-party library file reads.
"""

from __future__ import annotations

import hashlib
import json
import ntpath
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeGuard

from evaluation.routing.run_002.contract import (
    EXPECTED_LABEL_INDEX_MAP,
    MANIFEST_SELF_HASH_FIELD,
    canonical_json_sha256,
    load_verified_contract,
)
from evaluation.routing.run_002.errors import FatalRunError, FatalRunErrorCode
from evaluation.routing.run_002.worker import WorkerStartupError

_READ_CHUNK_BYTES = 65536
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _is_sha256_hex(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _reject_non_finite_json(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is prohibited: {token!r}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON object key is prohibited")
        obj[key] = value
    return obj


def _parse_strict_json_object(raw_bytes: bytes, *, error_code: FatalRunErrorCode, label: str) -> dict[str, Any]:
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise FatalRunError(error_code, f"{label} contains invalid UTF-8") from exc
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_non_finite_json,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise FatalRunError(error_code, f"{label} JSON is malformed") from exc
    if not isinstance(parsed, dict):
        raise FatalRunError(error_code, f"{label} must be a JSON object")
    return parsed


def _read_regular_file_bytes(
    path: Path,
    *,
    missing_code: FatalRunErrorCode,
    unsafe_code: FatalRunErrorCode,
    label: str,
) -> bytes:
    if not isinstance(path, Path) or not str(path).strip():
        raise FatalRunError(missing_code, f"{label} path is missing or invalid")
    try:
        st = os.lstat(path)
    except FileNotFoundError as exc:
        raise FatalRunError(missing_code, f"{label} is missing") from exc
    except OSError as exc:
        raise FatalRunError(unsafe_code, f"{label} could not be inspected") from exc

    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise FatalRunError(unsafe_code, f"{label} must be a regular non-symlink file")

    flags = os.O_RDONLY | _O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError as exc:
        raise FatalRunError(missing_code, f"{label} is missing") from exc
    except OSError as exc:
        raise FatalRunError(unsafe_code, f"{label} could not be opened safely") from exc

    try:
        fd_stat = os.fstat(fd)
        if not stat.S_ISREG(fd_stat.st_mode):
            raise FatalRunError(unsafe_code, f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except FatalRunError:
        raise
    except OSError as exc:
        raise FatalRunError(unsafe_code, f"{label} could not be read") from exc
    finally:
        os.close(fd)


def _hash_verified_artifact_file(
    artifact_root: Path,
    rel_posix_path: str,
) -> tuple[str, int]:
    """Hash one regular non-symlink file inside artifact_root without following symlinks."""
    segments = rel_posix_path.split("/")
    current = artifact_root
    for segment in segments:
        current = current / segment
        try:
            st = os.lstat(current)
        except OSError as exc:
            raise FatalRunError(
                FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                "Artifact file path component missing or unsafe",
            ) from exc
        if stat.S_ISLNK(st.st_mode):
            raise FatalRunError(
                FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                "Symlink detected inside artifact directory",
            )

    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact file could not be resolved",
        ) from exc

    if not resolved.is_relative_to(artifact_root):
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact file resolves outside artifact directory",
        )

    flags = os.O_RDONLY | _O_NOFOLLOW
    try:
        fd = os.open(current, flags)
    except OSError as exc:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact file could not be opened safely",
        ) from exc

    hasher = hashlib.sha256()
    total_bytes = 0
    try:
        fd_stat = os.fstat(fd)
        if not stat.S_ISREG(fd_stat.st_mode):
            raise FatalRunError(
                FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                "Non-regular file detected inside artifact directory",
            )
        while True:
            chunk = os.read(fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
            total_bytes += len(chunk)
    except FatalRunError:
        raise
    except OSError as exc:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact file read failed during integrity verification",
        ) from exc
    finally:
        os.close(fd)

    return hasher.hexdigest(), total_bytes


def _validate_manifest_relative_path(raw_path: object) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact manifest contains empty or non-string file path",
        )
    if "\x00" in raw_path or "\\" in raw_path:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact manifest contains invalid path characters",
        )
    if raw_path.startswith("/") or Path(raw_path).is_absolute():
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact manifest contains prohibited absolute path",
        )
    drive, _ = ntpath.splitdrive(raw_path)
    if drive or (len(raw_path) >= 2 and raw_path[1] == ":"):
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact manifest contains prohibited drive-qualified path",
        )

    segments = raw_path.split("/")
    for seg in segments:
        if seg in ("", ".", ".."):
            raise FatalRunError(
                FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                "Artifact manifest contains unsafe relative path segment",
            )
    return raw_path


@dataclass(frozen=True)
class ArtifactFileEntry:
    """Verified manifest entry for one regular file inside the artifact directory."""

    relative_path: str
    sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class ArtifactTrustAnchors:
    """External trust anchors pinned by the Run 002 contract."""

    expected_model_artifact_identity: str
    expected_protocol_identity: str
    expected_label_index_map: Mapping[str, str]

    def __post_init__(self) -> None:
        if not _is_sha256_hex(self.expected_model_artifact_identity):
            raise FatalRunError(
                FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                "Expected model artifact identity must be a 64-character hex SHA-256",
            )
        if not _is_sha256_hex(self.expected_protocol_identity):
            raise FatalRunError(
                FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
                "Expected protocol identity must be a 64-character hex SHA-256",
            )
        if dict(self.expected_label_index_map) != dict(EXPECTED_LABEL_INDEX_MAP):
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Expected label index map does not match contract",
            )


@dataclass(frozen=True)
class VerifiedArtifactBundle:
    """Verified artifact metadata returned by pre-import trust verification."""

    artifact_dir: Path
    artifact_manifest_path: Path
    protocol_path: Path
    artifact_identity: str
    protocol_identity: str
    label_index_map: Mapping[str, str]
    verified_files: tuple[ArtifactFileEntry, ...]


def resolve_contract_trust_anchors(contract_path: Path | None = None) -> ArtifactTrustAnchors:
    """Load trust anchors from the verified Run 002 contract."""
    contract = load_verified_contract(contract_path)
    anchors = contract.get("trust_anchors")
    fs_rules = contract.get("artifact_filesystem")
    if not isinstance(anchors, Mapping) or not isinstance(fs_rules, Mapping):
        raise FatalRunError(
            FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
            "Contract missing trust_anchors or artifact_filesystem section",
        )
    raw_label_map = fs_rules.get("expected_label_index_map")
    if not isinstance(raw_label_map, Mapping):
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Contract missing expected_label_index_map",
        )
    return ArtifactTrustAnchors(
        expected_model_artifact_identity=str(anchors.get("expected_model_artifact_identity", "")),
        expected_protocol_identity=str(anchors.get("expected_protocol_identity", "")),
        expected_label_index_map=MappingProxyType(dict(raw_label_map)),
    )


def _extract_manifest_files(manifest_data: Mapping[str, Any]) -> dict[str, ArtifactFileEntry]:
    raw_files = manifest_data.get("files")
    entries: dict[str, ArtifactFileEntry] = {}

    if isinstance(raw_files, Mapping):
        items_iter: list[tuple[object, object]] = list(raw_files.items())
        if not items_iter:
            raise FatalRunError(
                FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                "Artifact manifest files mapping must not be empty",
            )
        for raw_path, spec in items_iter:
            rel_path = _validate_manifest_relative_path(raw_path)
            size_bytes: int | None = None
            if isinstance(spec, str):
                sha = spec
            elif isinstance(spec, Mapping):
                sha = spec.get("sha256")
                raw_size = spec.get("size_bytes", spec.get("size"))
                if raw_size is not None:
                    if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
                        raise FatalRunError(
                            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                            "Artifact manifest file size must be a non-negative integer",
                        )
                    size_bytes = raw_size
            else:
                raise FatalRunError(
                    FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                    "Artifact manifest file entry has invalid type",
                )
            if not _is_sha256_hex(sha):
                raise FatalRunError(
                    FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                    "Artifact manifest file entry missing valid sha256",
                )
            if rel_path in entries:
                raise FatalRunError(
                    FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                    "Artifact manifest contains duplicate file path",
                )
            entries[rel_path] = ArtifactFileEntry(
                relative_path=rel_path,
                sha256=sha,
                size_bytes=size_bytes,
            )
        return entries

    if isinstance(raw_files, Sequence) and not isinstance(raw_files, (str, bytes)):
        if len(raw_files) == 0:
            raise FatalRunError(
                FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                "Artifact manifest files list must not be empty",
            )
        for item in raw_files:
            if not isinstance(item, Mapping):
                raise FatalRunError(
                    FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                    "Artifact manifest file entry must be a mapping",
                )
            raw_path = item.get("path", item.get("relative_path"))
            rel_path = _validate_manifest_relative_path(raw_path)
            sha = item.get("sha256")
            if not _is_sha256_hex(sha):
                raise FatalRunError(
                    FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                    "Artifact manifest file entry missing valid sha256",
                )
            raw_size = item.get("size_bytes", item.get("size"))
            size_bytes = None
            if raw_size is not None:
                if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
                    raise FatalRunError(
                        FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                        "Artifact manifest file size must be a non-negative integer",
                    )
                size_bytes = raw_size
            if rel_path in entries:
                raise FatalRunError(
                    FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                    "Artifact manifest contains duplicate file path",
                )
            entries[rel_path] = ArtifactFileEntry(
                relative_path=rel_path,
                sha256=sha,
                size_bytes=size_bytes,
            )
        return entries

    raise FatalRunError(
        FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
        "Artifact manifest missing valid files collection",
    )


def _scan_artifact_regular_files(artifact_dir: Path) -> tuple[Path, set[str]]:
    try:
        dir_stat = os.lstat(artifact_dir)
    except OSError as exc:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact directory missing or inaccessible",
        ) from exc

    if stat.S_ISLNK(dir_stat.st_mode) or not stat.S_ISDIR(dir_stat.st_mode):
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact directory must be a real non-symlink directory",
        )

    try:
        artifact_root = artifact_dir.resolve(strict=True)
    except OSError as exc:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact directory could not be resolved",
        ) from exc

    discovered: set[str] = set()
    pending: list[tuple[Path, str]] = [(artifact_root, "")]
    while pending:
        current_dir, rel_prefix = pending.pop()
        try:
            with os.scandir(current_dir) as it:
                entries = sorted(it, key=lambda entry: entry.name)
        except OSError as exc:
            raise FatalRunError(
                FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                "Artifact directory traversal failed",
            ) from exc

        for entry in entries:
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise FatalRunError(
                    FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                    "Artifact entry inspection failed",
                ) from exc

            if stat.S_ISLNK(st.st_mode):
                raise FatalRunError(
                    FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                    "Symlink detected inside artifact directory",
                )

            rel_path = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
            if stat.S_ISDIR(st.st_mode):
                pending.append((current_dir / entry.name, rel_path))
            elif stat.S_ISREG(st.st_mode):
                discovered.add(rel_path)
            else:
                raise FatalRunError(
                    FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                    "Non-regular filesystem object detected inside artifact directory",
                )

    return artifact_root, discovered


def _normalize_id2label_mapping(raw_map: object) -> dict[str, str]:
    if not isinstance(raw_map, Mapping):
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Persisted id2label mapping must be a JSON object",
        )
    normalized: dict[str, str] = {}
    for k, v in raw_map.items():
        if isinstance(k, bool) or not isinstance(k, (str, int)):
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Persisted label index key has invalid type",
            )
        key_str = str(k)
        if not isinstance(v, str) or not v:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Persisted label value must be a non-empty string",
            )
        if key_str in normalized:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Duplicate persisted label index",
            )
        normalized[key_str] = v
    return normalized


def _normalize_label2id_mapping(raw_map: object) -> dict[str, str]:
    if not isinstance(raw_map, Mapping):
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Persisted label2id mapping must be a JSON object",
        )
    inverted: dict[str, str] = {}
    for label_name, idx in raw_map.items():
        if not isinstance(label_name, str) or not label_name:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Persisted label2id key must be a non-empty string",
            )
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Persisted label2id index must be an integer",
            )
        idx_str = str(idx)
        if idx_str in inverted:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Duplicate index in persisted label2id mapping",
            )
        inverted[idx_str] = label_name
    return inverted


def _normalize_labels_sequence(raw_labels: object) -> dict[str, str]:
    if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes)):
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Persisted labels entry must be a sequence",
        )
    if len(raw_labels) == 0:
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Persisted labels sequence must not be empty",
        )
    normalized: dict[str, str] = {}
    for idx, label in enumerate(raw_labels):
        if not isinstance(label, str) or not label:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Persisted label entry must be a non-empty string",
            )
        normalized[str(idx)] = label
    return normalized


def _validate_normalized_label_map(
    candidate: Mapping[str, str],
    expected_label_map: Mapping[str, str],
) -> None:
    expected = dict(expected_label_map)
    if len(set(candidate.values())) != len(candidate):
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Duplicate label names in persisted label mapping",
        )
    if dict(candidate) != expected:
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Persisted label mapping does not match expected 0->general, 1->research",
        )


def _is_generic_transformer_placeholder_map(candidate: Mapping[str, str]) -> bool:
    """Return True when candidate only contains HuggingFace default LABEL_<n> placeholders."""
    if not candidate:
        return False
    for idx_str, label in candidate.items():
        if not idx_str.isdigit() or str(int(idx_str)) != idx_str:
            return False
        if label != f"LABEL_{idx_str}":
            return False
    return True


def _extract_label_declarations(
    payload: Mapping[str, Any],
    expected_label_map: Mapping[str, str],
    *,
    allow_generic_transformer_placeholders: bool = False,
) -> list[dict[str, str]]:
    declarations: list[dict[str, str]] = []
    for key in ("label_index_map", "expected_label_index_map"):
        if key in payload and payload[key] is not None:
            parsed = _normalize_id2label_mapping(payload[key])
            _validate_normalized_label_map(parsed, expected_label_map)
            declarations.append(parsed)
    if "id2label" in payload and payload["id2label"] is not None:
        parsed = _normalize_id2label_mapping(payload["id2label"])
        if not (
            allow_generic_transformer_placeholders
            and _is_generic_transformer_placeholder_map(parsed)
        ):
            _validate_normalized_label_map(parsed, expected_label_map)
            declarations.append(parsed)
    if "label2id" in payload and payload["label2id"] is not None:
        parsed = _normalize_label2id_mapping(payload["label2id"])
        if not (
            allow_generic_transformer_placeholders
            and _is_generic_transformer_placeholder_map(parsed)
        ):
            _validate_normalized_label_map(parsed, expected_label_map)
            declarations.append(parsed)
    if "labels" in payload and payload["labels"] is not None:
        parsed = _normalize_labels_sequence(payload["labels"])
        _validate_normalized_label_map(parsed, expected_label_map)
        declarations.append(parsed)
    return declarations


def verify_persisted_label_map(
    *,
    manifest_data: Mapping[str, Any],
    artifact_root: Path,
    verified_paths: set[str],
    expected_label_map: Mapping[str, str] = EXPECTED_LABEL_INDEX_MAP,
) -> Mapping[str, str]:
    """Verify persisted label-map declarations across manifest and JSON configs.

    Requires at least one authoritative persisted declaration matching
    expected_label_map (treating config_setfit.json `labels` as authoritative
    when config_setfit.json is present) and rejects missing, duplicate,
    unexpected, inverted, or conflicting declarations across manifest,
    config_setfit.json, and config.json. Generic HuggingFace `LABEL_<n>`
    placeholders in underlying transformer `config.json` `id2label`/`label2id`
    are ignored only as non-authoritative backbone defaults and never satisfy
    the requirement on their own.
    """
    declarations = _extract_label_declarations(manifest_data, expected_label_map)

    config_candidates = sorted(
        rel
        for rel in verified_paths
        if rel in ("config_setfit.json", "config.json")
        or rel.endswith(("/config_setfit.json", "/config.json"))
    )
    for rel_path in config_candidates:
        raw_bytes = _read_regular_file_bytes(
            artifact_root / rel_path,
            missing_code=FatalRunErrorCode.LABEL_MAP_MISMATCH,
            unsafe_code=FatalRunErrorCode.LABEL_MAP_MISMATCH,
            label="Artifact config metadata",
        )
        config_obj = _parse_strict_json_object(
            raw_bytes,
            error_code=FatalRunErrorCode.LABEL_MAP_MISMATCH,
            label="Artifact config metadata",
        )
        is_setfit_config = rel_path == "config_setfit.json" or rel_path.endswith("/config_setfit.json")
        is_transformer_config = rel_path == "config.json" or rel_path.endswith("/config.json")
        if is_setfit_config and ("labels" not in config_obj or config_obj["labels"] is None):
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "SetFit config missing authoritative labels declaration",
            )
        declarations.extend(
            _extract_label_declarations(
                config_obj,
                expected_label_map,
                allow_generic_transformer_placeholders=is_transformer_config,
            )
        )

    if not declarations:
        raise FatalRunError(
            FatalRunErrorCode.LABEL_MAP_MISMATCH,
            "Persisted label mapping missing from artifact metadata",
        )

    expected = dict(expected_label_map)
    for decl in declarations:
        if decl != expected:
            raise FatalRunError(
                FatalRunErrorCode.LABEL_MAP_MISMATCH,
                "Inconsistent persisted label mapping across artifact metadata",
            )

    return MappingProxyType(expected)


def verify_artifact_trust(
    *,
    artifact_dir: Path,
    artifact_manifest_path: Path,
    protocol_path: Path,
    contract_path: Path | None = None,
    trust_anchors: ArtifactTrustAnchors | None = None,
) -> VerifiedArtifactBundle:
    """Verify contract, protocol, manifest, exact artifact file set, and persisted labels.

    Must execute before any ML library import or model/pickle/joblib
    deserialization. Raises privacy-safe FatalRunError on any discrepancy.
    """
    anchors = trust_anchors or resolve_contract_trust_anchors(contract_path)

    # 1. Verify frozen shadow comparison protocol bytes against external anchor.
    protocol_bytes = _read_regular_file_bytes(
        protocol_path,
        missing_code=FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
        unsafe_code=FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
        label="Shadow protocol artifact",
    )
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if protocol_sha256 != anchors.expected_protocol_identity:
        raise FatalRunError(
            FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
            "Protocol artifact physical SHA-256 mismatch",
        )

    # 2. Load and verify artifact manifest JSON and canonical self-hash.
    manifest_bytes = _read_regular_file_bytes(
        artifact_manifest_path,
        missing_code=FatalRunErrorCode.ARTIFACT_MANIFEST_MISSING,
        unsafe_code=FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
        label="Artifact manifest",
    )
    manifest_data = _parse_strict_json_object(
        manifest_bytes,
        error_code=FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
        label="Artifact manifest",
    )

    declared_manifest_sha = manifest_data.get(MANIFEST_SELF_HASH_FIELD)
    if not _is_sha256_hex(declared_manifest_sha):
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact manifest missing valid canonical_manifest_sha256",
        )

    try:
        computed_manifest_sha = canonical_json_sha256(
            manifest_data,
            exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD,
        )
    except (TypeError, ValueError) as exc:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact manifest canonical serialization failed",
        ) from exc

    if computed_manifest_sha != declared_manifest_sha:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact manifest canonical self-hash mismatch",
        )

    if computed_manifest_sha != anchors.expected_model_artifact_identity:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact manifest identity does not match trusted contract anchor",
        )

    optional_identity = manifest_data.get("artifact_identity", manifest_data.get("model_artifact_identity"))
    if optional_identity is not None and optional_identity != anchors.expected_model_artifact_identity:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact manifest explicit identity field mismatch",
        )

    # 3. Validate manifest paths and verify exact bidirectional regular file set.
    manifest_entries = _extract_manifest_files(manifest_data)
    artifact_root, actual_files = _scan_artifact_regular_files(artifact_dir)

    expected_paths = set(manifest_entries.keys())
    if actual_files != expected_paths:
        raise FatalRunError(
            FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
            "Artifact directory regular file set does not match manifest",
        )

    # 4. Hash every verified regular file inside artifact_root without following symlinks.
    verified_entries: list[ArtifactFileEntry] = []
    for rel_path in sorted(expected_paths):
        entry = manifest_entries[rel_path]
        digest, byte_count = _hash_verified_artifact_file(artifact_root, rel_path)
        if digest != entry.sha256:
            raise FatalRunError(
                FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                "Artifact file SHA-256 mismatch",
            )
        if entry.size_bytes is not None and byte_count != entry.size_bytes:
            raise FatalRunError(
                FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH,
                "Artifact file byte size mismatch",
            )
        verified_entries.append(
            ArtifactFileEntry(
                relative_path=rel_path,
                sha256=digest,
                size_bytes=byte_count,
            )
        )

    # 5. Verify persisted label ordering metadata before any ML import/deserialization.
    verified_label_map = verify_persisted_label_map(
        manifest_data=manifest_data,
        artifact_root=artifact_root,
        verified_paths=expected_paths,
        expected_label_map=anchors.expected_label_index_map,
    )

    return VerifiedArtifactBundle(
        artifact_dir=artifact_root,
        artifact_manifest_path=artifact_manifest_path,
        protocol_path=protocol_path,
        artifact_identity=computed_manifest_sha,
        protocol_identity=protocol_sha256,
        label_index_map=verified_label_map,
        verified_files=tuple(verified_entries),
    )


def validate_worker_executable(worker_python: Path) -> Path:
    """Validate that the operator-supplied worker Python executable exists and is runnable.

    Raises WorkerStartupError with a privacy-safe message (never embedding raw
    paths or usernames).
    """
    if not isinstance(worker_python, Path) or not str(worker_python).strip():
        raise WorkerStartupError("Worker Python executable path must be a non-empty Path")
    if not worker_python.exists():
        raise WorkerStartupError("Configured worker Python executable does not exist")
    if not worker_python.is_file():
        raise WorkerStartupError("Configured worker Python executable is not a file")
    if not os.access(worker_python, os.X_OK):
        raise WorkerStartupError("Configured worker Python executable is not executable")
    return worker_python
