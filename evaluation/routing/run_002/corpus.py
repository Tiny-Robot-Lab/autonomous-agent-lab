"""Corpus-manifest and forbidden-overlap validation for Run 002 shadow routing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeGuard

from agent_contracts.serialization import check_no_credential_bearing_keys

from evaluation.routing.run_002.canonicalization import (
    CanonicalPrompt,
    canonicalize_prompt,
)
from evaluation.routing.run_002.contract import (
    MANIFEST_SELF_HASH_FIELD,
    canonical_json_sha256,
)
from evaluation.routing.run_002.errors import FatalRunError, FatalRunErrorCode

MINIMUM_COMPLETED_UNIQUE_ELIGIBLE = 200
MINIMUM_COMPLETED_PER_AUTHORITATIVE_CLASS = 50


def _is_sha256_hex(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _load_json_object(path: Path, *, context_label: str) -> dict[str, Any]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise FatalRunError(
            FatalRunErrorCode.CORPUS_NOT_APPROVED,
            f"{context_label} could not be read",
        ) from exc

    try:
        decoded = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise FatalRunError(
            FatalRunErrorCode.CORPUS_NOT_APPROVED,
            f"{context_label} contains invalid UTF-8",
        ) from exc

    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise FatalRunError(
            FatalRunErrorCode.CORPUS_NOT_APPROVED,
            f"{context_label} contains malformed JSON",
        ) from exc

    if not isinstance(parsed, dict):
        raise FatalRunError(
            FatalRunErrorCode.CORPUS_NOT_APPROVED,
            f"{context_label} must be a JSON object",
        )
    return parsed


@dataclass(frozen=True)
class CorpusRecord:
    """Single record inside an approved Run 002 replay corpus manifest."""

    record_id: str
    prompt: str = field(repr=False)
    corpus_eligible: bool
    expected_capability: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CorpusRecord:
        if not isinstance(raw, Mapping):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus record must be a mapping",
            )
        record_id = raw.get("record_id")
        prompt = raw.get("prompt")
        corpus_eligible = raw.get("corpus_eligible")
        expected_capability = raw.get("expected_capability")
        raw_metadata = raw.get("metadata", {})

        if not isinstance(record_id, str) or not record_id.strip():
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus record missing valid record_id",
            )
        if not isinstance(prompt, str):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus record prompt must be a string",
            )
        if type(corpus_eligible) is not bool:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus record corpus_eligible must be a boolean",
            )
        if expected_capability is not None and expected_capability not in ("general", "research"):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus record expected_capability must be general, research, or null",
            )
        if not isinstance(raw_metadata, Mapping):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus record metadata must be a mapping",
            )
        try:
            check_no_credential_bearing_keys(dict(raw_metadata), context="CorpusRecord.metadata")
        except (TypeError, ValueError) as exc:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus record metadata failed safety validation",
            ) from exc

        return cls(
            record_id=record_id.strip(),
            prompt=prompt,
            corpus_eligible=corpus_eligible,
            expected_capability=expected_capability,
            metadata=MappingProxyType(dict(raw_metadata)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "corpus_eligible": self.corpus_eligible,
            "expected_capability": self.expected_capability,
            "metadata": dict(self.metadata),
            "prompt": self.prompt,
            "record_id": self.record_id,
        }
        return payload

    def metadata_fingerprint_tuple(self) -> tuple[Any, ...]:
        return (
            self.record_id,
            self.corpus_eligible,
            self.expected_capability,
            canonical_json_sha256(dict(self.metadata)),
        )


@dataclass(frozen=True)
class CorpusManifest:
    """Hash-pinned Run 002 replay corpus manifest."""

    manifest_version: str
    corpus_id: str
    approved: bool
    records: tuple[CorpusRecord, ...]
    corpus_sha256: str
    canonical_manifest_sha256: str
    _verified_from_raw: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self._verified_from_raw:
            self._verify_declared_hashes(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CorpusManifest:
        if not isinstance(raw, Mapping):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest must be a mapping",
            )
        manifest_version = raw.get("manifest_version")
        corpus_id = raw.get("corpus_id")
        approved = raw.get("approved")
        raw_records = raw.get("records")
        corpus_sha256 = raw.get("corpus_sha256")
        canonical_manifest_sha256 = raw.get(MANIFEST_SELF_HASH_FIELD)

        if not isinstance(manifest_version, str) or not manifest_version.strip():
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest missing manifest_version",
            )
        if not isinstance(corpus_id, str) or not corpus_id.strip():
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest missing corpus_id",
            )
        if type(approved) is not bool:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest approved flag must be a boolean",
            )
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest records must be a sequence",
            )
        if not _is_sha256_hex(corpus_sha256):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest missing valid corpus_sha256",
            )
        if not _is_sha256_hex(canonical_manifest_sha256):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest missing valid canonical_manifest_sha256",
            )

        try:
            check_no_credential_bearing_keys(dict(raw), context="CorpusManifest")
        except (TypeError, ValueError) as exc:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest failed credential-key validation",
            ) from exc

        records = tuple(CorpusRecord.from_dict(item) for item in raw_records)
        # Validate declared hashes against the exact raw payload before constructing
        # (direct Python instantiation verifies via __post_init__ against to_dict()).
        raw_records_list = list(raw_records)
        try:
            computed_corpus_sha = canonical_json_sha256(raw_records_list)
            computed_manifest_sha = canonical_json_sha256(
                raw,
                exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD,
            )
        except (TypeError, ValueError) as exc:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest canonical serialization failed",
            ) from exc

        if computed_corpus_sha != corpus_sha256:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "corpus_sha256 does not match canonical records payload",
            )
        if computed_manifest_sha != canonical_manifest_sha256:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "canonical_manifest_sha256 does not match canonical manifest payload",
            )

        return cls(
            manifest_version=manifest_version.strip(),
            corpus_id=corpus_id.strip(),
            approved=approved,
            records=records,
            corpus_sha256=corpus_sha256,
            canonical_manifest_sha256=canonical_manifest_sha256,
            _verified_from_raw=True,
        )

    @classmethod
    def from_json_file(cls, path: Path) -> CorpusManifest:
        return cls.from_dict(_load_json_object(path, context_label="Corpus manifest"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "canonical_manifest_sha256": self.canonical_manifest_sha256,
            "corpus_id": self.corpus_id,
            "corpus_sha256": self.corpus_sha256,
            "manifest_version": self.manifest_version,
            "records": [r.to_dict() for r in self.records],
        }

    def _verify_declared_hashes(self, raw_payload: Mapping[str, Any]) -> None:
        raw_records = raw_payload.get("records")
        if not isinstance(raw_records, Sequence):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest records must be a sequence",
            )
        try:
            computed_corpus_sha = canonical_json_sha256(list(raw_records))
            computed_manifest_sha = canonical_json_sha256(
                raw_payload,
                exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD,
            )
        except (TypeError, ValueError) as exc:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus manifest canonical serialization failed",
            ) from exc

        if computed_corpus_sha != self.corpus_sha256:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "corpus_sha256 does not match canonical records payload",
            )
        if computed_manifest_sha != self.canonical_manifest_sha256:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "canonical_manifest_sha256 does not match canonical manifest payload",
            )


@dataclass(frozen=True)
class ForbiddenOverlapManifest:
    """Hash-pinned manifest of forbidden training/validation prompts for contamination checks."""

    manifest_id: str
    approved: bool
    prompts: tuple[str, ...] = field(repr=False)
    prompts_sha256: str
    canonical_manifest_sha256: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ForbiddenOverlapManifest:
        if not isinstance(raw, Mapping):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Overlap manifest must be a mapping",
            )
        manifest_id = raw.get("manifest_id")
        approved = raw.get("approved")
        raw_prompts = raw.get("prompts")
        prompts_sha256 = raw.get("prompts_sha256")
        canonical_manifest_sha256 = raw.get(MANIFEST_SELF_HASH_FIELD)

        if not isinstance(manifest_id, str) or not manifest_id.strip():
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Overlap manifest missing manifest_id",
            )
        if type(approved) is not bool or not approved:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Overlap manifest must be explicitly approved",
            )
        if (
            not isinstance(raw_prompts, Sequence)
            or isinstance(raw_prompts, (str, bytes))
            or len(raw_prompts) == 0
        ):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Overlap manifest must contain a non-empty prompt sequence",
            )
        for item in raw_prompts:
            if not isinstance(item, str):
                raise FatalRunError(
                    FatalRunErrorCode.CORPUS_NOT_APPROVED,
                    "Overlap manifest prompts must be strings",
                )
        if not _is_sha256_hex(prompts_sha256):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Overlap manifest missing valid prompts_sha256",
            )
        if not _is_sha256_hex(canonical_manifest_sha256):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Overlap manifest missing valid canonical_manifest_sha256",
            )

        try:
            computed_prompts_sha = canonical_json_sha256(list(raw_prompts))
            computed_manifest_sha = canonical_json_sha256(
                raw,
                exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD,
            )
        except (TypeError, ValueError) as exc:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Overlap manifest canonical serialization failed",
            ) from exc

        if computed_prompts_sha != prompts_sha256:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Overlap manifest prompts_sha256 mismatch",
            )
        if computed_manifest_sha != canonical_manifest_sha256:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Overlap manifest canonical_manifest_sha256 mismatch",
            )

        return cls(
            manifest_id=manifest_id.strip(),
            approved=approved,
            prompts=tuple(raw_prompts),
            prompts_sha256=prompts_sha256,
            canonical_manifest_sha256=canonical_manifest_sha256,
        )

    @classmethod
    def from_json_file(cls, path: Path) -> ForbiddenOverlapManifest:
        return cls.from_dict(_load_json_object(path, context_label="Overlap manifest"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "canonical_manifest_sha256": self.canonical_manifest_sha256,
            "manifest_id": self.manifest_id,
            "prompts": list(self.prompts),
            "prompts_sha256": self.prompts_sha256,
        }


@dataclass(frozen=True)
class ValidatedCorpusRecord:
    """Corpus record paired with its single canonicalized prompt."""

    record: CorpusRecord
    canonical: CanonicalPrompt


@dataclass(frozen=True)
class ValidatedCorpus:
    """Validated corpus manifest ready for offline coordinator execution."""

    manifest: CorpusManifest
    records: tuple[ValidatedCorpusRecord, ...]
    overlap_manifest_hashes: tuple[str, ...]
    corpus_eligible_count: int


def _validate_corpus_manifest_impl(
    manifest: CorpusManifest,
    overlap_manifests: Sequence[ForbiddenOverlapManifest],
    *,
    enforce_evidence_floors: bool,
) -> ValidatedCorpus:
    if not isinstance(manifest, CorpusManifest):
        raise FatalRunError(
            FatalRunErrorCode.CORPUS_NOT_APPROVED,
            "Invalid corpus manifest object",
        )
    if manifest.approved is not True:
        raise FatalRunError(
            FatalRunErrorCode.CORPUS_NOT_APPROVED,
            "Corpus manifest is not approved",
        )
    if not manifest.records:
        raise FatalRunError(
            FatalRunErrorCode.CORPUS_NOT_APPROVED,
            "Corpus manifest must not be empty",
        )

    if not manifest._verified_from_raw:
        manifest._verify_declared_hashes(manifest.to_dict())

    if (
        not isinstance(overlap_manifests, Sequence)
        or isinstance(overlap_manifests, (str, bytes))
        or len(overlap_manifests) == 0
    ):
        raise FatalRunError(
            FatalRunErrorCode.CORPUS_NOT_APPROVED,
            "At least one non-empty hash-pinned overlap manifest is required",
        )

    forbidden_hashes: set[str] = set()
    overlap_hashes: list[str] = []
    for overlap in overlap_manifests:
        if not isinstance(overlap, ForbiddenOverlapManifest):
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Invalid forbidden overlap manifest",
            )
        if not overlap.approved or not overlap.prompts:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Forbidden overlap manifest must be approved and non-empty",
            )
        # Re-verify overlap self-hashes against current object fields.
        ForbiddenOverlapManifest.from_dict(overlap.to_dict())
        overlap_hashes.append(overlap.canonical_manifest_sha256)

        for raw_overlap_prompt in overlap.prompts:
            try:
                canon_overlap = canonicalize_prompt(raw_overlap_prompt)
            except (TypeError, UnicodeEncodeError) as exc:
                raise FatalRunError(
                    FatalRunErrorCode.CORPUS_NOT_APPROVED,
                    "Overlap manifest contains non-UTF-8 or invalid prompt",
                ) from exc
            if canon_overlap.is_empty:
                raise FatalRunError(
                    FatalRunErrorCode.EMPTY_NORMALIZED_PROMPT,
                    "Overlap manifest contains empty canonical prompt",
                )
            forbidden_hashes.add(canon_overlap.canonical_input_sha256)

    seen_by_sha: dict[str, CorpusRecord] = {}
    validated_records: list[ValidatedCorpusRecord] = []
    corpus_eligible_count = 0

    for rec in manifest.records:
        try:
            canon = canonicalize_prompt(rec.prompt)
        except (TypeError, UnicodeEncodeError) as exc:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus record prompt failed strict UTF-8 encoding",
            ) from exc

        if canon.is_empty:
            raise FatalRunError(
                FatalRunErrorCode.EMPTY_NORMALIZED_PROMPT,
                "Corpus manifest contains empty canonical prompt",
            )

        existing = seen_by_sha.get(canon.canonical_input_sha256)
        if existing is not None:
            if existing.metadata_fingerprint_tuple() != rec.metadata_fingerprint_tuple():
                raise FatalRunError(
                    FatalRunErrorCode.CORPUS_NOT_APPROVED,
                    "Duplicate canonical_input_sha256 with conflicting metadata",
                )
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Duplicate canonical_input_sha256 in corpus manifest",
            )

        if canon.canonical_input_sha256 in forbidden_hashes:
            raise FatalRunError(
                FatalRunErrorCode.CORPUS_NOT_APPROVED,
                "Corpus contamination detected against forbidden overlap manifest",
            )

        seen_by_sha[canon.canonical_input_sha256] = rec
        validated_records.append(ValidatedCorpusRecord(record=rec, canonical=canon))
        if rec.corpus_eligible:
            corpus_eligible_count += 1

    if enforce_evidence_floors and corpus_eligible_count < MINIMUM_COMPLETED_UNIQUE_ELIGIBLE:
        raise FatalRunError(
            FatalRunErrorCode.CORPUS_NOT_APPROVED,
            "Approved replay corpus requires at least 200 unique corpus-eligible records",
        )

    return ValidatedCorpus(
        manifest=manifest,
        records=tuple(validated_records),
        overlap_manifest_hashes=tuple(overlap_hashes),
        corpus_eligible_count=corpus_eligible_count,
    )


def validate_corpus_manifest(
    manifest: CorpusManifest,
    overlap_manifests: Sequence[ForbiddenOverlapManifest],
) -> ValidatedCorpus:
    """Validate an approved replay corpus manifest with full production evidence floors."""
    return _validate_corpus_manifest_impl(
        manifest,
        overlap_manifests,
        enforce_evidence_floors=True,
    )


def validate_synthetic_corpus_manifest_for_test(
    manifest: CorpusManifest,
    overlap_manifests: Sequence[ForbiddenOverlapManifest],
) -> ValidatedCorpus:
    """Explicit test-only corpus validator allowing <200 eligible records in Increment 1 tests.

    Must never be exposed to the production replay entry point or future CLI.
    """
    return _validate_corpus_manifest_impl(
        manifest,
        overlap_manifests,
        enforce_evidence_floors=False,
    )
