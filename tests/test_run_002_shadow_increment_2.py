"""Deterministic security, lifecycle, IPC, and crash-containment tests for Run 002 Increment 2."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from evaluation.routing.run_002.contract import (
    EXPECTED_LABEL_INDEX_MAP,
    MANIFEST_SELF_HASH_FIELD,
    canonical_json_sha256,
)
from evaluation.routing.run_002.coordinator import (
    run_offline_shadow_evaluation,
    run_synthetic_shadow_evaluation_for_test,
)
from evaluation.routing.run_002.corpus import CorpusManifest, ForbiddenOverlapManifest
from evaluation.routing.run_002.errors import FatalRunError, FatalRunErrorCode, PerRecordErrorCode
from evaluation.routing.run_002.ipc import (
    MAX_IPC_REQUEST_BYTES,
    MAX_IPC_RESPONSE_BYTES,
    InferOkResponse,
    InferRequest,
    IPCProtocolError,
    ShutdownRequest,
    StartupRequest,
    decode_parent_request,
    decode_worker_response,
    encode_ipc_frame,
)
from evaluation.routing.run_002.promotion import PromotionGate
from evaluation.routing.run_002.subprocess_worker import (
    SubprocessShadowWorker,
    SubprocessWorkerConfig,
    _build_sanitized_worker_env,
)
from evaluation.routing.run_002.telemetry import (
    DisagreementCategory,
    ObservationState,
    serialize_observation_jsonl,
)
from evaluation.routing.run_002.trust import (
    ArtifactTrustAnchors,
    validate_worker_executable,
    verify_artifact_trust,
)
from evaluation.routing.run_002.worker import (
    WorkerFailure,
    WorkerPrediction,
    WorkerStartupError,
    validate_predict_proba_output,
)
from evaluation.routing.run_002.worker_entrypoint import (
    NetworkAccessBlockedError,
    install_offline_network_guard,
    verify_runtime_label_map,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_synthetic_artifact_bundle(
    base_dir: Path,
    *,
    manifest_format: str = "list",
    config_setfit_payload: dict[str, Any] | None = None,
    config_json_payload: dict[str, Any] | None = None,
    manifest_extra: dict[str, Any] | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> tuple[Path, Path, Path, ArtifactTrustAnchors]:
    """Create a synthetic artifact directory, manifest, protocol, and trust anchors."""
    artifact_dir = base_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if config_setfit_payload is None:
        config_setfit_payload = {"labels": ["general", "research"], "model_type": "setfit"}
    if config_json_payload is None:
        config_json_payload = {"architectures": ["MPNetModel"], "hidden_size": 8}

    (artifact_dir / "config_setfit.json").write_text(
        json.dumps(config_setfit_payload),
        encoding="utf-8",
    )
    (artifact_dir / "config.json").write_text(
        json.dumps(config_json_payload),
        encoding="utf-8",
    )
    (artifact_dir / "model_head.bin").write_bytes(b"synthetic-head-weights-v1")

    for rel, content in (extra_files or {}).items():
        target = artifact_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    all_rel_files = sorted(
        p.relative_to(artifact_dir).as_posix()
        for p in artifact_dir.rglob("*")
        if p.is_file()
    )

    if manifest_format == "list":
        files_field: Any = [
            {
                "path": rel,
                "sha256": _sha256_file(artifact_dir / rel),
                "size_bytes": (artifact_dir / rel).stat().st_size,
            }
            for rel in all_rel_files
        ]
    else:
        files_field = {
            rel: {
                "sha256": _sha256_file(artifact_dir / rel),
                "size_bytes": (artifact_dir / rel).stat().st_size,
            }
            for rel in all_rel_files
        }

    manifest_obj: dict[str, Any] = {
        "files": files_field,
        "manifest_version": 1,
    }
    if manifest_extra:
        manifest_obj.update(manifest_extra)
    manifest_obj[MANIFEST_SELF_HASH_FIELD] = canonical_json_sha256(
        manifest_obj,
        exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD,
    )

    manifest_path = base_dir / "artifact-manifest.json"
    manifest_path.write_text(json.dumps(manifest_obj), encoding="utf-8")

    protocol_path = base_dir / "protocol.json"
    protocol_path.write_bytes(b'{"protocol":"run-002-shadow-comparison-v1"}\n')

    anchors = ArtifactTrustAnchors(
        expected_model_artifact_identity=manifest_obj[MANIFEST_SELF_HASH_FIELD],
        expected_protocol_identity=_sha256_file(protocol_path),
        expected_label_index_map=EXPECTED_LABEL_INDEX_MAP,
    )
    return artifact_dir, manifest_path, protocol_path, anchors


def _resign_manifest(manifest_path: Path, artifact_dir: Path) -> str:
    """Recompute file hashes and manifest self-hash to simulate coordinated tampering."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = data["files"]
    if isinstance(files, list):
        for item in files:
            p = artifact_dir / item["path"]
            item["sha256"] = _sha256_file(p)
            item["size_bytes"] = p.stat().st_size
    else:
        for rel, spec in files.items():
            p = artifact_dir / rel
            spec["sha256"] = _sha256_file(p)
            spec["size_bytes"] = p.stat().st_size
    data[MANIFEST_SELF_HASH_FIELD] = canonical_json_sha256(
        data,
        exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD,
    )
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    return data[MANIFEST_SELF_HASH_FIELD]


def _make_worker_python_shim(
    base_dir: Path,
    *,
    fake_setfit_code: str,
    probe_modules: dict[str, str] | None = None,
) -> Path:
    """Create an executable shim using the repo venv python and a temporary fake ML package."""
    fake_site = base_dir / "fake_ml_site"
    fake_site.mkdir(parents=True, exist_ok=True)
    (fake_site / "setfit.py").write_text(textwrap.dedent(fake_setfit_code), encoding="utf-8")
    for mod_name, mod_code in (probe_modules or {}).items():
        (fake_site / f"{mod_name}.py").write_text(textwrap.dedent(mod_code), encoding="utf-8")

    shim = base_dir / "worker_python_shim.sh"
    shim.write_text(
        f"#!/bin/sh\n"
        f'export PYTHONPATH="{fake_site}:$PYTHONPATH"\n'
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return shim


def _make_overlap(prompts: list[str]) -> ForbiddenOverlapManifest:
    raw: dict[str, Any] = {
        "approved": True,
        "manifest_id": "inc2-overlap-v1",
        "prompts": prompts,
        "prompts_sha256": canonical_json_sha256(prompts),
    }
    raw[MANIFEST_SELF_HASH_FIELD] = canonical_json_sha256(
        raw, exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD
    )
    return ForbiddenOverlapManifest.from_dict(raw)


def _make_corpus(records: list[dict[str, Any]]) -> CorpusManifest:
    raw: dict[str, Any] = {
        "approved": True,
        "corpus_id": "inc2-synthetic-corpus",
        "corpus_sha256": canonical_json_sha256(records),
        "manifest_version": "1.0",
        "records": records,
    }
    raw[MANIFEST_SELF_HASH_FIELD] = canonical_json_sha256(
        raw, exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD
    )
    return CorpusManifest.from_dict(raw)


_STANDARD_FAKE_SETFIT = """
class _Head:
    def __init__(self, classes=None):
        self.classes_ = classes if classes is not None else ["general", "research"]

class _BodyConfig:
    id2label = {0: "LABEL_0", 1: "LABEL_1"}

class _Body:
    config = _BodyConfig()

class _Proba:
    def __init__(self, rows):
        self._rows = rows
    def tolist(self):
        return self._rows

class SetFitModel:
    load_calls = []

    def __init__(self, *, labels=None, classes=None):
        self.labels = labels if labels is not None else ["general", "research"]
        self.model_head = _Head(classes=classes)
        self.model_body = _Body()
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, model_path, local_files_only=False, trust_remote_code=True, **kwargs):
        cls.load_calls.append({
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        })
        if local_files_only is not True or trust_remote_code is not False:
            raise RuntimeError("Unsafe offline loading flags")
        return cls()

    def predict_proba(self, prompts):
        p = prompts[0]
        if "consensus" in p or "studies" in p or "CVEs" in p:
            return _Proba([[0.12, 0.88]])
        return _Proba([[0.91, 0.09]])
"""


# ---------------------------------------------------------------------------
# 1. Artifact trust verification & filesystem boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("manifest_format", ["list", "dict"])
def test_verify_artifact_trust_accepts_exact_trusted_manifest(
    tmp_path: Path,
    manifest_format: str,
) -> None:
    artifact_dir, manifest_path, protocol_path, anchors = _build_synthetic_artifact_bundle(
        tmp_path,
        manifest_format=manifest_format,
        extra_files={"nested/tokenizer.json": b"{}"},
    )
    verified = verify_artifact_trust(
        artifact_dir=artifact_dir,
        artifact_manifest_path=manifest_path,
        protocol_path=protocol_path,
        trust_anchors=anchors,
    )
    assert verified.artifact_identity == anchors.expected_model_artifact_identity
    assert verified.protocol_identity == anchors.expected_protocol_identity
    assert dict(verified.label_index_map) == {"0": "general", "1": "research"}
    assert [f.relative_path for f in verified.verified_files] == [
        "config.json",
        "config_setfit.json",
        "model_head.bin",
        "nested/tokenizer.json",
    ]


def test_verify_artifact_trust_rejects_coordinated_tampering_against_pinned_contract(
    tmp_path: Path,
) -> None:
    artifact_dir, manifest_path, protocol_path, anchors = _build_synthetic_artifact_bundle(tmp_path)

    # Without test trust_anchors override, contract trust anchor rejects synthetic artifact.
    with pytest.raises(FatalRunError) as exc_default_anchor:
        verify_artifact_trust(
            artifact_dir=artifact_dir,
            artifact_manifest_path=manifest_path,
            protocol_path=protocol_path,
        )
    assert exc_default_anchor.value.code is FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH

    # Coordinated tampering with matching protocol anchor: modify file + update file hash + self-hash.
    (artifact_dir / "model_head.bin").write_bytes(b"tampered-weights")
    new_self_hash = _resign_manifest(manifest_path, artifact_dir)
    assert new_self_hash != anchors.expected_model_artifact_identity

    with pytest.raises(FatalRunError) as exc_coord:
        verify_artifact_trust(
            artifact_dir=artifact_dir,
            artifact_manifest_path=manifest_path,
            protocol_path=protocol_path,
            trust_anchors=anchors,
        )
    assert exc_coord.value.code is FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH
    assert str(tmp_path) not in str(exc_coord.value)


def test_verify_artifact_trust_rejects_missing_extra_changed_symlink_nonregular_and_bad_paths(
    tmp_path: Path,
) -> None:
    # 1. Missing manifest
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "case_missing_man")
    man.unlink()
    with pytest.raises(FatalRunError) as e_missing_man:
        verify_artifact_trust(
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
        )
    assert e_missing_man.value.code is FatalRunErrorCode.ARTIFACT_MANIFEST_MISSING

    # 2. Changed file without manifest update
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "case_changed")
    (art / "model_head.bin").write_bytes(b"altered")
    with pytest.raises(FatalRunError) as e_changed:
        verify_artifact_trust(
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
        )
    assert e_changed.value.code is FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH

    # 3. Missing artifact file
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "case_missing_file")
    (art / "model_head.bin").unlink()
    with pytest.raises(FatalRunError) as e_missing_file:
        verify_artifact_trust(
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
        )
    assert e_missing_file.value.code is FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH

    # 4. Extra unlisted artifact file
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "case_extra_file")
    (art / "unlisted.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(FatalRunError) as e_extra:
        verify_artifact_trust(
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
        )
    assert e_extra.value.code is FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH

    # 5. Symlink inside artifact directory
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "case_symlink")
    outside_target = tmp_path / "outside.txt"
    outside_target.write_text("outside", encoding="utf-8")
    (art / "evil_link").symlink_to(outside_target)
    with pytest.raises(FatalRunError) as e_sym:
        verify_artifact_trust(
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
        )
    assert e_sym.value.code is FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH

    # 6. Non-regular filesystem object (FIFO) inside artifact directory
    if hasattr(os, "mkfifo"):
        art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "case_fifo")
        os.mkfifo(art / "named_pipe")
        with pytest.raises(FatalRunError) as e_fifo:
            verify_artifact_trust(
                artifact_dir=art,
                artifact_manifest_path=man,
                protocol_path=proto,
                trust_anchors=anchors,
            )
        assert e_fifo.value.code is FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH

    # 7. Incorrect protocol identity
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "case_proto")
    proto.write_bytes(b'{"protocol":"tampered"}\n')
    with pytest.raises(FatalRunError) as e_proto:
        verify_artifact_trust(
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
        )
    assert e_proto.value.code is FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH

    # 8. Unsafe manifest paths (absolute, empty, dot, parent traversal, backslash)
    for idx, bad_path in enumerate(["/etc/passwd", "", "./config.json", "../secret.bin", "a\\b.bin"]):
        case_dir = tmp_path / f"case_badpath_{idx}"
        art, man, proto, _ = _build_synthetic_artifact_bundle(case_dir)
        data = json.loads(man.read_text(encoding="utf-8"))
        data["files"].append({"path": bad_path, "sha256": "0" * 64})
        data[MANIFEST_SELF_HASH_FIELD] = canonical_json_sha256(
            data, exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD
        )
        man.write_text(json.dumps(data), encoding="utf-8")
        pinned = ArtifactTrustAnchors(
            expected_model_artifact_identity=data[MANIFEST_SELF_HASH_FIELD],
            expected_protocol_identity=_sha256_file(proto),
            expected_label_index_map=EXPECTED_LABEL_INDEX_MAP,
        )
        with pytest.raises(FatalRunError) as e_badpath:
            verify_artifact_trust(
                artifact_dir=art,
                artifact_manifest_path=man,
                protocol_path=proto,
                trust_anchors=pinned,
            )
        assert e_badpath.value.code is FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH


# ---------------------------------------------------------------------------
# 2. Verify-before-import / verify-before-deserialize probes
# ---------------------------------------------------------------------------


def test_parent_and_worker_never_import_ml_or_deserialize_when_trust_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, manifest_path, protocol_path, anchors = _build_synthetic_artifact_bundle(
        tmp_path / "probe_case"
    )
    # Tamper an artifact file after creating trusted anchors
    (artifact_dir / "model_head.bin").write_bytes(b"tampered")

    probe_marker = tmp_path / "ml_probe_triggered.txt"
    probe_setfit = f"""
    from pathlib import Path
    Path({str(probe_marker)!r}).write_text("IMPORTED", encoding="utf-8")
    class SetFitModel:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            Path({str(probe_marker)!r}).write_text("LOADED", encoding="utf-8")
            return cls()
    """
    shim = _make_worker_python_shim(tmp_path, fake_setfit_code=probe_setfit)

    popen_calls = 0

    def forbid_popen(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("subprocess.Popen must not be called when parent trust fails")

    monkeypatch.setattr(subprocess, "Popen", forbid_popen)
    worker = SubprocessShadowWorker(
        SubprocessWorkerConfig(
            worker_python=shim,
            artifact_dir=artifact_dir,
            artifact_manifest_path=manifest_path,
            protocol_path=protocol_path,
            trust_anchors=anchors,
        )
    )
    with pytest.raises(FatalRunError) as exc_parent:
        worker.preflight()
    assert exc_parent.value.code is FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH
    assert popen_calls == 0
    assert not probe_marker.exists()
    for ml_mod in ("setfit", "torch", "transformers", "sentence_transformers", "sklearn"):
        assert ml_mod not in sys.modules

    monkeypatch.undo()

    # Also prove inside the REAL worker_entrypoint subprocess that verify-before-import holds:
    startup = StartupRequest(
        request_id=0,
        artifact_dir=str(artifact_dir),
        artifact_manifest_path=str(manifest_path),
        protocol_path=str(protocol_path),
        contract_path=str(
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "routing"
            / "contracts"
            / "run-002-offline-shadow-integration-003.json"
        ),
        expected_model_artifact_identity=anchors.expected_model_artifact_identity,
        expected_protocol_identity=anchors.expected_protocol_identity,
    )
    env = _build_sanitized_worker_env(tmp_path / "cache")
    proc = subprocess.run(
        [str(shim), "-u", "-m", "evaluation.routing.run_002.worker_entrypoint"],
        input=encode_ipc_frame(startup.to_payload(), max_bytes=MAX_IPC_REQUEST_BYTES),
        capture_output=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
        check=False,
        timeout=10.0,
    )
    assert proc.returncode != 0
    resp = decode_worker_response(proc.stdout)
    assert resp.error_code is FatalRunErrorCode.ARTIFACT_IDENTITY_MISMATCH  # type: ignore[union-attr]
    assert not probe_marker.exists()


# ---------------------------------------------------------------------------
# 3. Persisted & runtime label ordering verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setfit_cfg", "body_cfg"),
    [
        ({"labels": ["research", "general"]}, {"architectures": ["MPNet"]}),
        ({"id2label": {"0": "research", "1": "general"}}, {"architectures": ["MPNet"]}),
        ({"model_type": "setfit"}, {"architectures": ["MPNet"]}),  # missing labels
        ({"labels": ["general", "general"]}, {"architectures": ["MPNet"]}),  # duplicate
        ({"labels": ["general", "analysis"]}, {"architectures": ["MPNet"]}),  # unexpected
        ({"labels": ["general", "research", "extra"]}, {"architectures": ["MPNet"]}),  # extra
        (
            {"labels": ["general", "research"]},
            {"id2label": {"0": "research", "1": "general"}},  # inconsistent body/head
        ),
        (
            {"id2label": {"0": "general", "1": "research"}, "label2id": {"general": 1, "research": 0}},
            {"architectures": ["MPNet"]},
        ),
        (
            {"labels": ["research", "general"]},
            {"id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 0}},
        ),
        (
            {"model_type": "setfit"},
            {"id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 0}},
        ),
        (
            {"labels": None, "model_type": "setfit"},
            {"id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 0}},
        ),
        (
            {"labels": [], "model_type": "setfit"},
            {"id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 0}},
        ),
        (
            {"labels": "general", "model_type": "setfit"},
            {"id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 0}},
        ),
        (
            {"labels": ["general"], "model_type": "setfit"},
            {"id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 0}},
        ),
        (
            {"labels": ["general", ""], "model_type": "setfit"},
            {"id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 0}},
        ),
        (
            {"labels": ["general", "research"]},
            {"id2label": {"0": "LABEL_1"}, "label2id": {"LABEL_0": 0}},
        ),
        (
            {"labels": ["general", "research"]},
            {"id2label": {"0": "LABEL_0", "1": "research"}},
        ),
        (
            {"labels": ["general", "research"]},
            {"id2label": {"0": "LABEL_0"}, "label2id": {"LABEL_0": 1}},
        ),
    ],
)
def test_persisted_label_map_mismatch_variants_fail_closed(
    tmp_path: Path,
    setfit_cfg: dict[str, Any],
    body_cfg: dict[str, Any],
) -> None:
    art, man, proto, anchors = _build_synthetic_artifact_bundle(
        tmp_path,
        config_setfit_payload=setfit_cfg,
        config_json_payload=body_cfg,
    )
    with pytest.raises(FatalRunError) as exc_info:
        verify_artifact_trust(
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
        )
    assert exc_info.value.code is FatalRunErrorCode.LABEL_MAP_MISMATCH


def test_verify_artifact_trust_accepts_generic_transformer_placeholders_with_setfit_labels(
    tmp_path: Path,
) -> None:
    art, man, proto, anchors = _build_synthetic_artifact_bundle(
        tmp_path,
        config_setfit_payload={"labels": ["general", "research"]},
        config_json_payload={
            "id2label": {"0": "LABEL_0"},
            "label2id": {"LABEL_0": 0},
        },
    )
    verified = verify_artifact_trust(
        artifact_dir=art,
        artifact_manifest_path=man,
        protocol_path=proto,
        trust_anchors=anchors,
    )
    assert dict(verified.label_index_map) == {"0": "general", "1": "research"}


def test_runtime_label_map_verification_in_unit_and_subprocess_worker(tmp_path: Path) -> None:
    class _Model:
        def __init__(self, *, labels=None, id2label=None, classes=None, out_features=None) -> None:
            self.labels = labels
            self.id2label = id2label
            self.model_head = type(
                "_Head",
                (),
                {"classes_": classes, "out_features": out_features},
            )()

    # Valid runtime order passes
    assert dict(verify_runtime_label_map(_Model(labels=["general", "research"], classes=["general", "research"]))) == {
        "0": "general",
        "1": "research",
    }
    # Invalid runtime orders fail closed
    for bad_model in (
        _Model(labels=["research", "general"], classes=["research", "general"]),
        _Model(labels=["general", "research"], classes=["research", "general"]),
        _Model(labels=["general", "research"], classes=[1, 0]),
        _Model(labels=None, classes=[0, 1]),
        _Model(labels=["general", "general"], classes=["general", "general"]),
        _Model(labels=["general", "unexpected"], classes=["general", "unexpected"]),
        _Model(labels=None, id2label=None, classes=None, out_features=None),
        _Model(labels=["general", "research"], out_features=3),
    ):
        with pytest.raises(FatalRunError) as exc_rt:
            verify_runtime_label_map(bad_model)
        assert exc_rt.value.code is FatalRunErrorCode.LABEL_MAP_MISMATCH

    # End-to-end subprocess with inverted runtime classes_ aborts preflight before any record runs
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "rt_bad")
    inverted_setfit = _STANDARD_FAKE_SETFIT.replace(
        'classes if classes is not None else ["general", "research"]',
        'classes if classes is not None else ["research", "general"]',
    )
    shim = _make_worker_python_shim(tmp_path / "rt_bad", fake_setfit_code=inverted_setfit)
    worker = SubprocessShadowWorker(
        SubprocessWorkerConfig(
            worker_python=shim,
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
        )
    )
    corpus = _make_corpus([{"record_id": "r1", "prompt": "What is 15% of 850?", "corpus_eligible": True}])
    overlap = _make_overlap(["Unrelated prompt"])
    with pytest.raises(FatalRunError) as exc_run:
        run_synthetic_shadow_evaluation_for_test(
            corpus_manifest=corpus,
            overlap_manifests=[overlap],
            worker=worker,
        )
    assert exc_run.value.code is FatalRunErrorCode.LABEL_MAP_MISMATCH


# ---------------------------------------------------------------------------
# 4. Strict offline loading & network-egress prohibition
# ---------------------------------------------------------------------------


def test_offline_network_guard_blocks_socket_and_http_without_leaking_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unit test of guard (restored automatically by monkeypatch on exit)
    import http.client
    import socket
    import urllib.request

    monkeypatch.setattr(socket, "socket", socket.socket)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", socket.getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", socket.gethostbyname)
    monkeypatch.setattr(socket, "gethostbyname_ex", socket.gethostbyname_ex)
    monkeypatch.setattr(socket, "getnameinfo", socket.getnameinfo)
    monkeypatch.setattr(urllib.request, "urlopen", urllib.request.urlopen)
    monkeypatch.setattr(http.client, "HTTPConnection", http.client.HTTPConnection)
    monkeypatch.setattr(http.client, "HTTPSConnection", http.client.HTTPSConnection)

    install_offline_network_guard()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    with pytest.raises(NetworkAccessBlockedError):
        socket.create_connection(("secret.example.invalid", 443), timeout=1)
    with pytest.raises(NetworkAccessBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(NetworkAccessBlockedError):
        urllib.request.urlopen("https://secret.example.invalid/token")
    with pytest.raises(NetworkAccessBlockedError):
        http.client.HTTPSConnection("secret.example.invalid")

    # End-to-end subprocess attempt to open HTTP/socket maps to NETWORK_ACCESS_ATTEMPTED without URL leak
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "net_sub")
    net_setfit = """
    import urllib.request
    class SetFitModel:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            urllib.request.urlopen("https://huggingface.co/secret-leak-url/model.bin")
            return cls()
    """
    shim = _make_worker_python_shim(tmp_path / "net_sub", fake_setfit_code=net_setfit)
    worker = SubprocessShadowWorker(
        SubprocessWorkerConfig(
            worker_python=shim,
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
        )
    )
    with pytest.raises(FatalRunError) as exc_net:
        worker.preflight()
    assert exc_net.value.code is FatalRunErrorCode.NETWORK_ACCESS_ATTEMPTED
    assert "huggingface.co" not in str(exc_net.value)
    assert "secret-leak-url" not in str(exc_net.value)

    # Also prove network attempt during inference (not only startup) aborts run safely
    art2, man2, proto2, anchors2 = _build_synthetic_artifact_bundle(tmp_path / "net_infer")
    net_infer_setfit = _STANDARD_FAKE_SETFIT.replace(
        "def predict_proba(self, prompts):",
        'def predict_proba(self, prompts):\n'
        '        if "TRIGGER_NET" in prompts[0]:\n'
        '            import socket\n'
        '            socket.create_connection(("exfil.example.invalid", 8443), timeout=1)\n',
    )
    shim2 = _make_worker_python_shim(tmp_path / "net_infer", fake_setfit_code=net_infer_setfit)
    worker2 = SubprocessShadowWorker(
        SubprocessWorkerConfig(
            worker_python=shim2,
            artifact_dir=art2,
            artifact_manifest_path=man2,
            protocol_path=proto2,
            trust_anchors=anchors2,
        )
    )
    with worker2:
        worker2.preflight()
        with pytest.raises(FatalRunError) as exc_infer_net:
            worker2.predict("TRIGGER_NET prompt")
        assert exc_infer_net.value.code is FatalRunErrorCode.NETWORK_ACCESS_ATTEMPTED
        assert "exfil.example.invalid" not in str(exc_infer_net.value)


def test_sanitized_worker_env_strips_parent_secrets_and_proxies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")

    env = _build_sanitized_worker_env(tmp_path)
    assert "HF_TOKEN" not in env
    assert "HTTP_PROXY" not in env
    assert "HTTPS_PROXY" not in env
    assert "TAVILY_API_KEY" not in env
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"


# ---------------------------------------------------------------------------
# 5. IPC validation, bounds, and probability validation reuse
# ---------------------------------------------------------------------------


def test_ipc_rejects_unknown_fields_nan_oversized_and_non_utf8() -> None:
    req = InferRequest(request_id=1, canonical_prompt="What is 15% of 850?")
    encoded = encode_ipc_frame(req.to_payload(), max_bytes=MAX_IPC_REQUEST_BYTES)
    decoded = decode_parent_request(encoded)
    assert decoded == req

    # Unknown field rejected
    bad_unknown = json.dumps({"type": "infer", "request_id": 1, "canonical_prompt": "hi", "extra": 1}).encode() + b"\n"
    with pytest.raises(IPCProtocolError):
        decode_parent_request(bad_unknown)

    # NaN rejected on encode and decode
    with pytest.raises(IPCProtocolError):
        encode_ipc_frame(
            InferOkResponse(
                request_id=1,
                general_probability=float("nan"),
                research_probability=0.5,
                shadow_decision="general",
                latency_ns=10,
            ).to_payload(),
            max_bytes=MAX_IPC_RESPONSE_BYTES,
        )
    with pytest.raises(IPCProtocolError):
        decode_worker_response(
            b'{"general_probability":NaN,"latency_ns":10,"request_id":1,"research_probability":0.5,"shadow_decision":"general","type":"infer_ok"}\n'
        )

    # Oversized rejected
    with pytest.raises(IPCProtocolError):
        decode_worker_response(b"x" * (MAX_IPC_RESPONSE_BYTES + 1))

    # Invalid UTF-8 rejected
    with pytest.raises(IPCProtocolError):
        decode_worker_response(b"\xff\xfe\n")

    # Clean shutdown roundtrip
    shut = ShutdownRequest(request_id=2)
    assert decode_parent_request(encode_ipc_frame(shut.to_payload(), max_bytes=MAX_IPC_REQUEST_BYTES)) == shut


def test_validate_predict_proba_output_distinguishes_malformed_and_invalid_probabilities() -> None:
    # Malformed shapes / types
    for bad_shape in (None, "general", {"a": 1}, [0.8, 0.2], [[0.8, 0.2, 0.0]], [[0.8]], [["0.8", 0.2]]):
        res = validate_predict_proba_output(bad_shape, latency_ns=42)
        assert isinstance(res, WorkerFailure)
        assert res.error_code is PerRecordErrorCode.MALFORMED_OUTPUT
        assert res.latency_ns == 42

    # Invalid probabilities (bool, NaN, Inf, out-of-range, zero sum, tolerance, exact tie)
    for bad_probs in (
        [[True, False]],
        [[float("nan"), 0.5]],
        [[float("inf"), 0.0]],
        [[1.2, -0.2]],
        [[0.0, 0.0]],
        [[0.6, 0.2]],
        [[0.5, 0.5]],
    ):
        res = validate_predict_proba_output(bad_probs, latency_ns=55)
        assert isinstance(res, WorkerFailure)
        assert res.error_code is PerRecordErrorCode.INVALID_PROBABILITIES
        assert res.latency_ns == 55

    # Valid within tolerance normalizes and argmaxes on normalized values
    ok = validate_predict_proba_output([[0.20, 0.80004]], latency_ns=77)
    assert not isinstance(ok, WorkerFailure)
    assert ok.shadow_decision == "research"
    assert abs((ok.general_probability + ok.research_probability) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 6. Worker executable validation, timeout, reaping, crash containment, E2E
# ---------------------------------------------------------------------------


def test_worker_executable_validation_rejects_missing_dir_and_non_executable_without_path_leak(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(WorkerStartupError) as e_miss:
        validate_worker_executable(missing)
    assert str(tmp_path) not in str(e_miss.value)

    not_file = tmp_path / "subdir"
    not_file.mkdir()
    with pytest.raises(WorkerStartupError) as e_dir:
        validate_worker_executable(not_file)
    assert str(tmp_path) not in str(e_dir.value)

    non_exec = tmp_path / "python_no_exec"
    non_exec.write_text("#!/bin/sh\n", encoding="utf-8")
    non_exec.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(WorkerStartupError) as e_noexec:
        validate_worker_executable(non_exec)
    assert str(tmp_path) not in str(e_noexec.value)


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_subprocess_worker_timeout_crash_malformed_and_recovery_with_reaping(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "child_pids.txt"
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "lifecycle")

    lifecycle_setfit = f"""
    import os, signal, sys, time
    from pathlib import Path

    class _Head:
        classes_ = ["general", "research"]

    class SetFitModel:
        def __init__(self):
            self.labels = ["general", "research"]
            self.model_head = _Head()
            self.device = "cpu"
            with open({str(pid_file)!r}, "a", encoding="utf-8") as fh:
                fh.write(f"{{os.getpid()}}\\n")

        @classmethod
        def from_pretrained(cls, model_path, local_files_only=False, trust_remote_code=True, **kwargs):
            return cls()

        def predict_proba(self, prompts):
            p = prompts[0]
            if "SLEEP_FOREVER" in p:
                time.sleep(5.0)
                return [[0.9, 0.1]]
            if "NATIVE_CRASH" in p:
                os.kill(os.getpid(), signal.SIGKILL)
            if "SIGSEGV_CRASH" in p:
                os.kill(os.getpid(), signal.SIGSEGV)
            if "NONZERO_EXIT" in p:
                os._exit(42)
            if "SILENT_EOF" in p:
                os._exit(0)
            if "MALFORMED_PROB" in p:
                return "not-an-array"
            if "TIE_PROB" in p:
                return [[0.5, 0.5]]
            if "OVERSIZED_IPC" in p:
                sys.stdout.buffer.write(b'{{"x":"' + b"a" * 40000 + b'"}}\\n')
                sys.stdout.buffer.flush()
                return [[0.9, 0.1]]
            if "WRONG_REQ_ID" in p:
                sys.stdout.buffer.write(
                    b'{{"general_probability":0.9,"latency_ns":10,"request_id":999999,"research_probability":0.1,"shadow_decision":"general","type":"infer_ok"}}\\n'
                )
                sys.stdout.buffer.flush()
                return [[0.9, 0.1]]
            if "RAW_MALFORMED_IPC" in p:
                sys.stdout.buffer.write(b"{{corrupt-json\\n")
                sys.stdout.buffer.flush()
                os._exit(0)
            return [[0.90, 0.10]]
    """
    shim = _make_worker_python_shim(tmp_path / "lifecycle", fake_setfit_code=lifecycle_setfit)
    worker = SubprocessShadowWorker(
        SubprocessWorkerConfig(
            worker_python=shim,
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
            timeout_seconds=0.25,
            startup_timeout_seconds=5.0,
        )
    )
    try:
        worker.preflight()

        # 1. Timeout terminates & reaps child
        res_to = worker.predict("SLEEP_FOREVER prompt")
        assert res_to.error_code is PerRecordErrorCode.INFERENCE_TIMEOUT  # type: ignore[union-attr]

        # 2. Parent recovers cleanly on next record
        res_ok1 = worker.predict("What is 15% of 850?")
        assert isinstance(res_ok1, WorkerPrediction)
        assert res_ok1.shadow_decision == "general"

        # 3. Native crash (SIGKILL & SIGSEGV) -> UNEXPECTED_OBSERVER_ERROR & reaped
        res_crash = worker.predict("NATIVE_CRASH prompt")
        assert isinstance(res_crash, WorkerFailure)
        assert res_crash.error_code is PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR

        res_segv = worker.predict("SIGSEGV_CRASH prompt")
        assert isinstance(res_segv, WorkerFailure)
        assert res_segv.error_code is PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR

        # 4. Nonzero exit -> UNEXPECTED_OBSERVER_ERROR
        res_nz = worker.predict("NONZERO_EXIT prompt")
        assert isinstance(res_nz, WorkerFailure)
        assert res_nz.error_code is PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR

        # 5. Clean EOF with no response -> MALFORMED_OUTPUT
        res_eof = worker.predict("SILENT_EOF prompt")
        assert isinstance(res_eof, WorkerFailure)
        assert res_eof.error_code is PerRecordErrorCode.MALFORMED_OUTPUT

        # 6. Malformed predict_proba -> MALFORMED_OUTPUT
        res_mal = worker.predict("MALFORMED_PROB prompt")
        assert isinstance(res_mal, WorkerFailure)
        assert res_mal.error_code is PerRecordErrorCode.MALFORMED_OUTPUT

        # 7. Exact tie -> INVALID_PROBABILITIES
        res_tie = worker.predict("TIE_PROB prompt")
        assert isinstance(res_tie, WorkerFailure)
        assert res_tie.error_code is PerRecordErrorCode.INVALID_PROBABILITIES

        # 8. Malformed / oversized / mismatched-request-id raw IPC -> MALFORMED_OUTPUT & worker torn down
        res_ipc = worker.predict("RAW_MALFORMED_IPC prompt")
        assert isinstance(res_ipc, WorkerFailure)
        assert res_ipc.error_code is PerRecordErrorCode.MALFORMED_OUTPUT

        res_oversized = worker.predict("OVERSIZED_IPC prompt")
        assert isinstance(res_oversized, WorkerFailure)
        assert res_oversized.error_code is PerRecordErrorCode.MALFORMED_OUTPUT

        res_wrong_id = worker.predict("WRONG_REQ_ID prompt")
        assert isinstance(res_wrong_id, WorkerFailure)
        assert res_wrong_id.error_code is PerRecordErrorCode.MALFORMED_OUTPUT

        # 9. Parent remains usable afterwards
        res_ok2 = worker.predict("Summarize item #9.")
        assert isinstance(res_ok2, WorkerPrediction)
        assert res_ok2.shadow_decision == "general"
    finally:
        worker.close()

    pids = [int(line) for line in pid_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(pids) >= 5
    for pid in pids:
        assert not _is_pid_alive(pid), f"Orphan worker PID {pid} remained alive"


def test_coordinator_end_to_end_with_subprocess_worker_preserves_authority_and_privacy(
    tmp_path: Path,
) -> None:
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "e2e")
    seen_log = tmp_path / "seen_prompts.txt"
    e2e_setfit = _STANDARD_FAKE_SETFIT.replace(
        "def predict_proba(self, prompts):",
        f"def predict_proba(self, prompts):\n"
        f"        with open({str(seen_log)!r}, 'a', encoding='utf-8') as fh:\n"
        f"            fh.write(prompts[0] + '\\n')\n",
    )
    shim = _make_worker_python_shim(tmp_path / "e2e", fake_setfit_code=e2e_setfit)
    worker = SubprocessShadowWorker(
        SubprocessWorkerConfig(
            worker_python=shim,
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
            timeout_seconds=5.0,
            repository_commit="9232c7f",
        )
    )

    # Include 220 eligible + 1 ineligible + 1 dangerous disagreement? Let's test clean 220+ first.
    records: list[dict[str, Any]] = []
    for i in range(110):
        records.append({
            "record_id": f"gen-{i:03d}",
            "prompt": f"What is {i + 1} percent of 850?",
            "corpus_eligible": True,
        })
    for i in range(110):
        records.append({
            "record_id": f"res-{i:03d}",
            "prompt": f"Find peer-reviewed studies on topic #{i + 1}.",
            "corpus_eligible": True,
        })
    records.append({
        "record_id": "inelig-001",
        "prompt": "EXCLUDED_SECRET_PROMPT_DO_NOT_REACH_WORKER",
        "corpus_eligible": False,
    })

    corpus = _make_corpus(records)
    overlap = _make_overlap(["Known overlap prompt"])
    jsonl_path = tmp_path / "obs.jsonl"
    summary_path = tmp_path / "summary.json"

    result = run_offline_shadow_evaluation(
        corpus_manifest=corpus,
        overlap_manifests=[overlap],
        worker=worker,
        telemetry_jsonl_path=jsonl_path,
        summary_path=summary_path,
    )

    assert result.promotion.gate is PromotionGate.ELIGIBLE_FOR_HUMAN_REVIEW
    assert result.promotion.completed_eligible_count == 220
    assert result.promotion.ineligible_count == 1
    assert result.summary.environment.python_version
    # Worker executable path must never leak into summary or JSONL
    jsonl_text = jsonl_path.read_text(encoding="utf-8")
    summary_text = summary_path.read_text(encoding="utf-8")
    assert str(shim) not in summary_text
    assert str(tmp_path) not in summary_text
    assert "EXCLUDED_SECRET_PROMPT" not in seen_log.read_text(encoding="utf-8")
    assert "percent of 850" not in jsonl_text
    assert "peer-reviewed studies" not in jsonl_text

    # Also test dangerous disagreement + crash continuation via subprocess worker
    crash_setfit = _STANDARD_FAKE_SETFIT.replace(
        'if "consensus" in p or "studies" in p or "CVEs" in p:\n            return _Proba([[0.12, 0.88]])',
        'if "OpenSSL" in p:\n            return _Proba([[0.90, 0.10]])\n        if "CRASH" in p:\n            import os\n            os._exit(139)',
    )
    shim2 = _make_worker_python_shim(tmp_path / "e2e_err", fake_setfit_code=crash_setfit)
    worker2 = SubprocessShadowWorker(
        SubprocessWorkerConfig(
            worker_python=shim2,
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
        )
    )
    small_corpus = _make_corpus([
        {"record_id": "d1", "prompt": "Research latest CVEs in OpenSSL.", "corpus_eligible": True},
        {"record_id": "c1", "prompt": "CRASH secret prompt text.", "corpus_eligible": True},
        {"record_id": "g1", "prompt": "What is 25 percent of 850?", "corpus_eligible": True},
    ])
    small_res = run_synthetic_shadow_evaluation_for_test(
        corpus_manifest=small_corpus,
        overlap_manifests=[overlap],
        worker=worker2,
    )
    assert small_res.observations[0].state is ObservationState.ELIGIBLE_DISAGREED
    assert small_res.observations[0].authoritative_capability == "research"
    assert (
        small_res.observations[0].disagreement_category
        is DisagreementCategory.AUTHORITATIVE_RESEARCH_SHADOW_GENERAL
    )
    assert small_res.observations[1].state is ObservationState.ELIGIBLE_SHADOW_EXECUTION_ERROR
    assert small_res.observations[1].error_code is PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR
    assert small_res.observations[2].state is ObservationState.ELIGIBLE_AGREED
    assert small_res.promotion.gate is PromotionGate.STOP_AND_INVESTIGATE
    assert "CRASH secret" not in serialize_observation_jsonl(small_res.observations[1])


def test_worker_disables_core_dumps_before_ml_import_and_sigsegv_does_not_race_deadline(
    tmp_path: Path,
) -> None:
    rlimit_log = tmp_path / "rlimit.json"
    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "coredump")
    fake_setfit = f"""
    import json, os, resource, signal, sys
    from pathlib import Path

    class _Head:
        classes_ = ["general", "research"]

    class SetFitModel:
        def __init__(self):
            self.labels = ["general", "research"]
            self.model_head = _Head()
            self.device = "cpu"

        @classmethod
        def from_pretrained(cls, model_path, local_files_only=False, trust_remote_code=True, **kwargs):
            soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
            dumpable = None
            status_path = Path("/proc/self/status")
            if status_path.exists():
                for line in status_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("Dumpable:"):
                        dumpable = line.split(":", 1)[1].strip()
                        break
            Path({str(rlimit_log)!r}).write_text(
                json.dumps({{"soft": soft, "hard": hard, "dumpable": dumpable}}),
                encoding="utf-8",
            )
            return cls()

        def predict_proba(self, prompts):
            if "SIGSEGV_CRASH" in prompts[0]:
                os.kill(os.getpid(), signal.SIGSEGV)
            return [[0.9, 0.1]]
    """
    shim = _make_worker_python_shim(tmp_path / "coredump", fake_setfit_code=fake_setfit)
    worker = SubprocessShadowWorker(
        SubprocessWorkerConfig(
            worker_python=shim,
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
            timeout_seconds=0.25,
            startup_timeout_seconds=5.0,
        )
    )
    with worker:
        worker.preflight()
        recorded = json.loads(rlimit_log.read_text(encoding="utf-8"))
        assert recorded["soft"] == 0
        assert recorded["hard"] == 0
        if sys.platform.startswith("linux") and recorded["dumpable"] is not None:
            assert recorded["dumpable"] == "0"

        res_segv = worker.predict("SIGSEGV_CRASH prompt")
        assert isinstance(res_segv, WorkerFailure)
        assert res_segv.error_code is PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR

        res_ok = worker.predict("What is 2 + 2?")
        assert isinstance(res_ok, WorkerPrediction)
        assert res_ok.shadow_decision == "general"


def test_subprocess_worker_predict_classifies_already_exited_process_on_timeout_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import signal

    import evaluation.routing.run_002.subprocess_worker as sw_mod

    art, man, proto, anchors = _build_synthetic_artifact_bundle(tmp_path / "boundary")
    fake_setfit = """
    class _Head:
        classes_ = ["general", "research"]

    class SetFitModel:
        def __init__(self):
            self.labels = ["general", "research"]
            self.model_head = _Head()
            self.device = "cpu"

        @classmethod
        def from_pretrained(cls, model_path, local_files_only=False, trust_remote_code=True, **kwargs):
            return cls()

        def predict_proba(self, prompts):
            return [[0.9, 0.1]]
    """
    shim = _make_worker_python_shim(tmp_path / "boundary", fake_setfit_code=fake_setfit)
    worker = SubprocessShadowWorker(
        SubprocessWorkerConfig(
            worker_python=shim,
            artifact_dir=art,
            artifact_manifest_path=man,
            protocol_path=proto,
            trust_anchors=anchors,
            timeout_seconds=0.25,
            startup_timeout_seconds=5.0,
        )
    )
    try:
        worker.preflight()
        proc = worker._ensure_worker_started()

        def _fake_timed_out_after_exit(
            p: subprocess.Popen[bytes],
            *,
            timeout_seconds: float,
            max_bytes: int,
        ) -> tuple[bytes | None, bool, bool]:
            os.kill(p.pid, signal.SIGKILL)
            p.wait(timeout=2.0)
            return None, True, False

        monkeypatch.setattr(sw_mod, "_read_bounded_line_with_timeout", _fake_timed_out_after_exit)
        res = worker.predict("boundary prompt")
        assert isinstance(res, WorkerFailure)
        assert res.error_code is PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR
        assert not _is_pid_alive(proc.pid)
    finally:
        worker.close()


# ---------------------------------------------------------------------------
# 7. Opt-in real-model integration test (skipped unless external env/artifacts exist)
# ---------------------------------------------------------------------------

_REAL_MODEL_ENV_VARS = (
    "RUN_002_WORKER_PYTHON",
    "RUN_002_ARTIFACT_DIR",
    "RUN_002_ARTIFACT_MANIFEST",
    "RUN_002_PROTOCOL_PATH",
)


def _real_model_prerequisites_present() -> bool:
    values = [os.environ.get(var, "").strip() for var in _REAL_MODEL_ENV_VARS]
    if not all(values):
        return False
    return all(Path(val).exists() for val in values)


@pytest.mark.run_002_real_model
@pytest.mark.skipif(
    not _real_model_prerequisites_present(),
    reason="External Run 002 worker Python environment or frozen artifacts are unavailable",
)
def test_real_run_002_model_subprocess_smoke() -> None:
    worker = SubprocessShadowWorker(
        SubprocessWorkerConfig(
            worker_python=Path(os.environ["RUN_002_WORKER_PYTHON"]),
            artifact_dir=Path(os.environ["RUN_002_ARTIFACT_DIR"]),
            artifact_manifest_path=Path(os.environ["RUN_002_ARTIFACT_MANIFEST"]),
            protocol_path=Path(os.environ["RUN_002_PROTOCOL_PATH"]),
        )
    )
    with worker:
        worker.preflight()
        res = worker.predict("What is 15% of 850?")
        assert isinstance(res, WorkerPrediction)
        assert res.shadow_decision in ("general", "research")
        assert 0.0 <= res.general_probability <= 1.0
        assert 0.0 <= res.research_probability <= 1.0
