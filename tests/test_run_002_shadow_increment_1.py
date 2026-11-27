"""Deterministic unit and coordinator tests for Run 002 Increment 1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from agent_contracts.serialization import canonical_json_dumps
from orchestrator.domain.assessment import (
    RoutingClarity,
    RoutingSignal,
    RoutingSignalType,
    SignalStrength,
)
from orchestrator.domain.capability import AgentCapability
from orchestrator.domain.router import DeterministicCapabilityRouter

from evaluation.routing.run_002.canonicalization import (
    NORMATIVE_CANONICALIZATION_VECTORS,
    CanonicalPrompt,
    canonicalize_prompt,
)
from evaluation.routing.run_002.contract import (
    CONTRACT_ID,
    CONTRACT_SELF_HASH_FIELD,
    DEFAULT_CONTRACT_PATH,
    EXPECTED_CONTRACT_CANONICAL_SHA256,
    EXPECTED_CONTRACT_PHYSICAL_SHA256,
    MANIFEST_SELF_HASH_FIELD,
    canonical_json_sha256,
    load_verified_contract,
)
from evaluation.routing.run_002.coordinator import (
    run_offline_shadow_evaluation,
    run_synthetic_shadow_evaluation_for_test,
)
from evaluation.routing.run_002.corpus import (
    CorpusManifest,
    ForbiddenOverlapManifest,
    validate_corpus_manifest,
    validate_synthetic_corpus_manifest_for_test,
)
from evaluation.routing.run_002.eligibility import (
    IneligibilityReason,
    ReportStratum,
    RouterMode,
    evaluate_eligibility,
)
from evaluation.routing.run_002.errors import (
    FatalRunError,
    FatalRunErrorCode,
    PerRecordErrorCode,
)
from evaluation.routing.run_002.promotion import (
    PromotionGate,
    evaluate_promotion_gate,
)
from evaluation.routing.run_002.telemetry import (
    ALLOWED_ENVIRONMENT_FINGERPRINT_FIELDS,
    PROTOCOL_SUPERSESSION_NOTES,
    DisagreementCategory,
    EnvironmentFingerprint,
    ObservationState,
    ShadowObservation,
    collect_safe_environment_fingerprint,
    serialize_observation_jsonl,
)
from evaluation.routing.run_002.worker import (
    StubShadowWorker,
    WorkerFailure,
    WorkerPrediction,
    validate_and_resolve_prediction,
)


def _make_overlap_manifest(
    prompts: list[str],
    *,
    manifest_id: str = "overlap-test-v1",
    approved: bool = True,
) -> ForbiddenOverlapManifest:
    raw: dict[str, Any] = {
        "approved": approved,
        "manifest_id": manifest_id,
        "prompts": prompts,
        "prompts_sha256": canonical_json_sha256(prompts),
    }
    raw[MANIFEST_SELF_HASH_FIELD] = canonical_json_sha256(
        raw, exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD
    )
    return ForbiddenOverlapManifest.from_dict(raw)


def _make_corpus_manifest(
    records: list[dict[str, Any]],
    *,
    corpus_id: str = "synthetic-corpus-v1",
    manifest_version: str = "1.0",
    approved: bool = True,
) -> CorpusManifest:
    raw: dict[str, Any] = {
        "approved": approved,
        "corpus_id": corpus_id,
        "corpus_sha256": canonical_json_sha256(records),
        "manifest_version": manifest_version,
        "records": records,
    }
    raw[MANIFEST_SELF_HASH_FIELD] = canonical_json_sha256(
        raw, exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD
    )
    return CorpusManifest.from_dict(raw)


def _build_220_record_corpus() -> CorpusManifest:
    records: list[dict[str, Any]] = []
    # 110 clear general (mix of signal-backed and zero-signal default)
    for idx in range(55):
        records.append({
            "record_id": f"gen-sig-{idx:03d}",
            "prompt": f"What is {idx + 10} percent of 850?",
            "corpus_eligible": True,
            "expected_capability": "general",
            "metadata": {"group": "general_signal"},
        })
    for idx in range(55):
        records.append({
            "record_id": f"gen-def-{idx:03d}",
            "prompt": f"Summarize item number {idx + 1} concisely.",
            "corpus_eligible": True,
            "expected_capability": "general",
            "metadata": {"group": "general_default"},
        })
    # 110 clear research
    for idx in range(110):
        records.append({
            "record_id": f"res-{idx:03d}",
            "prompt": f"Research the latest developments in consensus protocol #{idx + 1}.",
            "corpus_eligible": True,
            "expected_capability": "research",
            "metadata": {"group": "research"},
        })
    # Plus 2 ineligible records (one corpus_eligible=False, one compound_mixed_task)
    records.append({
        "record_id": "inelig-001",
        "prompt": "Excluded corpus record prompt alpha.",
        "corpus_eligible": False,
        "expected_capability": None,
        "metadata": {"group": "excluded"},
    })
    records.append({
        "record_id": "inelig-002",
        "prompt": "Search for the latest OpenSSL CVEs and write a checking script #999.",
        "corpus_eligible": True,
        "expected_capability": None,
        "metadata": {"group": "compound"},
    })
    return _make_corpus_manifest(records, corpus_id="synthetic-220-v1")


def _build_observation(
    idx: int,
    *,
    auth_cap: str = "general",
    shadow: str | None = None,
    state: ObservationState | None = None,
    error_code: PerRecordErrorCode | None = None,
) -> ShadowObservation:
    obs_id = f"obs-{idx:04d}"
    sha = hashlib.sha256(obs_id.encode("utf-8")).hexdigest()
    stratum = (
        ReportStratum.CLEAR_RESEARCH
        if auth_cap == "research"
        else ReportStratum.CLEAR_GENERAL_SIGNAL_BACKED
    )
    if state is ObservationState.ELIGIBLE_SHADOW_EXECUTION_ERROR or error_code is not None:
        return ShadowObservation.execution_error(
            observation_id=obs_id,
            canonical_input_sha256=sha,
            stratum=stratum,
            authoritative_capability=auth_cap,
            error_code=error_code or PerRecordErrorCode.INFERENCE_TIMEOUT,
            latency_ns=None,
        )
    chosen = shadow or auth_cap
    p_gen, p_res = (0.85, 0.15) if chosen == "general" else (0.15, 0.85)
    return ShadowObservation.completed(
        observation_id=obs_id,
        canonical_input_sha256=sha,
        stratum=stratum,
        authoritative_capability=auth_cap,
        shadow_decision=chosen,
        general_probability=p_gen,
        research_probability=p_res,
        latency_ns=1500,
    )


# ---------------------------------------------------------------------------
# 1. Governing contract verification
# ---------------------------------------------------------------------------


def test_load_verified_contract_matches_pinned_hashes(tmp_path: Path) -> None:
    contract = load_verified_contract()
    assert contract["contract_id"] == CONTRACT_ID
    assert contract[CONTRACT_SELF_HASH_FIELD] == EXPECTED_CONTRACT_CANONICAL_SHA256
    assert (
        hashlib.sha256(DEFAULT_CONTRACT_PATH.read_bytes()).hexdigest()
        == EXPECTED_CONTRACT_PHYSICAL_SHA256
    )

    tampered_path = tmp_path / "tampered.json"
    tampered_data = dict(contract)
    tampered_data["status"] = "tampered"
    tampered_path.write_text(json.dumps(tampered_data), encoding="utf-8")
    with pytest.raises(FatalRunError) as exc_info:
        load_verified_contract(tampered_path)
    assert exc_info.value.code is FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH


# ---------------------------------------------------------------------------
# 2. Canonicalization & normative test vectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector",
    NORMATIVE_CANONICALIZATION_VECTORS,
    ids=[v.vector_id for v in NORMATIVE_CANONICALIZATION_VECTORS],
)
def test_normative_canonicalization_vectors(vector) -> None:
    if vector.expect_unicode_error:
        with pytest.raises(UnicodeEncodeError):
            canonicalize_prompt(vector.raw_prompt)
        return

    result = canonicalize_prompt(vector.raw_prompt)
    assert isinstance(result, CanonicalPrompt)
    assert result.canonical_prompt == vector.expected_canonical_prompt
    assert result.canonical_input_sha256 == vector.expected_sha256
    # Ensure repr never exposes the canonical prompt text
    if result.canonical_prompt:
        assert result.canonical_prompt not in repr(result)

    if vector.expect_empty_manifest_rejection:
        assert result.is_empty
        manifest = _make_corpus_manifest([
            {
                "record_id": "empty-1",
                "prompt": vector.raw_prompt,
                "corpus_eligible": True,
            }
        ])
        with pytest.raises(FatalRunError) as exc_info:
            validate_synthetic_corpus_manifest_for_test(
                manifest, [_make_overlap_manifest(["unrelated overlap prompt"])]
            )
        assert exc_info.value.code is FatalRunErrorCode.EMPTY_NORMALIZED_PROMPT


@pytest.mark.parametrize("bad_input", [b"hello", None, 123, ["hello"]])
def test_canonicalize_prompt_requires_python_str(bad_input: object) -> None:
    with pytest.raises(TypeError):
        canonicalize_prompt(bad_input)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Pure eligibility policy & strata
# ---------------------------------------------------------------------------


def test_eligibility_fatal_preflight_never_masked_by_record_level_flags() -> None:
    canon = canonicalize_prompt("What is 15% of 850?")
    assessment = DeterministicCapabilityRouter().assess(canon.canonical_prompt)

    with pytest.raises(FatalRunError) as exc_unapproved:
        evaluate_eligibility(
            canonical_prompt=canon,
            corpus_approved=False,
            corpus_eligible=False,
            router_mode=RouterMode.UNSUPPORTED,
            assessment=assessment,
        )
    assert exc_unapproved.value.code is FatalRunErrorCode.CORPUS_NOT_APPROVED

    with pytest.raises(FatalRunError) as exc_router:
        evaluate_eligibility(
            canonical_prompt=canon,
            corpus_approved=True,
            corpus_eligible=False,
            router_mode=RouterMode.UNSUPPORTED,
            assessment=assessment,
        )
    assert exc_router.value.code is FatalRunErrorCode.UNSUPPORTED_ROUTER_MODE

    empty_canon = canonicalize_prompt("   \t\n")
    with pytest.raises(FatalRunError) as exc_empty:
        evaluate_eligibility(
            canonical_prompt=empty_canon,
            corpus_approved=False,
            corpus_eligible=False,
            router_mode=RouterMode.UNSUPPORTED,
            assessment=None,
        )
    assert exc_empty.value.code is FatalRunErrorCode.EMPTY_NORMALIZED_PROMPT


def test_eligibility_per_record_precedence_and_strata() -> None:
    router = DeterministicCapabilityRouter()

    # 1. corpus_record_not_eligible beats invalid_assessment
    canon_gen = canonicalize_prompt("What is 15 percent of 850?")
    d1 = evaluate_eligibility(
        canonical_prompt=canon_gen,
        corpus_approved=True,
        corpus_eligible=False,
        router_mode=RouterMode.DEFAULT_COORDINATOR_ROUTER,
        assessment="not-an-assessment",
    )
    assert not d1.eligible
    assert d1.ineligibility_reason is IneligibilityReason.CORPUS_RECORD_NOT_ELIGIBLE

    # 2. invalid_assessment
    d2 = evaluate_eligibility(
        canonical_prompt=canon_gen,
        corpus_approved=True,
        corpus_eligible=True,
        router_mode=RouterMode.DEFAULT_COORDINATOR_ROUTER,
        assessment={"provisional_capability": "general"},
    )
    assert not d2.eligible
    assert d2.ineligibility_reason is IneligibilityReason.INVALID_ASSESSMENT

    # 3. ambiguous_assessment
    ambig_canon = canonicalize_prompt("Compare PostgreSQL and MongoDB.")
    ambig_assessment = router.assess(ambig_canon.canonical_prompt)
    assert ambig_assessment.clarity is RoutingClarity.AMBIGUOUS
    d3 = evaluate_eligibility(
        canonical_prompt=ambig_canon,
        corpus_approved=True,
        corpus_eligible=True,
        router_mode=RouterMode.DEFAULT_COORDINATOR_ROUTER,
        assessment=ambig_assessment,
    )
    assert not d3.eligible
    assert d3.ineligibility_reason is IneligibilityReason.AMBIGUOUS_ASSESSMENT

    # 4. compound_mixed_task (Clear + Research, caught after clarity & capability checks)
    comp_canon = canonicalize_prompt("Search for the latest OpenSSL CVEs and write a checking script.")
    comp_assessment = router.assess(comp_canon.canonical_prompt)
    assert comp_assessment.clarity is RoutingClarity.CLEAR
    assert any(s.signal_type is RoutingSignalType.COMPOUND_MIXED_TASK for s in comp_assessment.signals)
    d4 = evaluate_eligibility(
        canonical_prompt=comp_canon,
        corpus_approved=True,
        corpus_eligible=True,
        router_mode=RouterMode.DEFAULT_COORDINATOR_ROUTER,
        assessment=comp_assessment,
    )
    assert not d4.eligible
    assert d4.ineligibility_reason is IneligibilityReason.COMPOUND_MIXED_TASK

    # 5. Exact strata for eligible prompts
    res_canon = canonicalize_prompt("Find peer-reviewed studies and cite sources.")
    res_dec = evaluate_eligibility(
        canonical_prompt=res_canon,
        corpus_approved=True,
        corpus_eligible=True,
        router_mode=RouterMode.DEFAULT_COORDINATOR_ROUTER,
        assessment=router.assess(res_canon.canonical_prompt),
    )
    assert res_dec.eligible
    assert res_dec.stratum is ReportStratum.CLEAR_RESEARCH
    assert res_dec.authoritative_capability is AgentCapability.RESEARCH

    gen_sig_dec = evaluate_eligibility(
        canonical_prompt=canon_gen,
        corpus_approved=True,
        corpus_eligible=True,
        router_mode=RouterMode.DEFAULT_COORDINATOR_ROUTER,
        assessment=router.assess(canon_gen.canonical_prompt),
    )
    assert gen_sig_dec.eligible
    assert gen_sig_dec.stratum is ReportStratum.CLEAR_GENERAL_SIGNAL_BACKED
    assert gen_sig_dec.authoritative_capability is AgentCapability.GENERAL

    zero_sig_canon = canonicalize_prompt("Hello agent")
    zero_sig_assessment = router.assess(zero_sig_canon.canonical_prompt)
    assert len(zero_sig_assessment.signals) == 0
    gen_def_dec = evaluate_eligibility(
        canonical_prompt=zero_sig_canon,
        corpus_approved=True,
        corpus_eligible=True,
        router_mode=RouterMode.DEFAULT_COORDINATOR_ROUTER,
        assessment=zero_sig_assessment,
    )
    assert gen_def_dec.eligible
    assert gen_def_dec.stratum is ReportStratum.CLEAR_GENERAL_ZERO_SIGNAL_DEFAULT
    assert gen_def_dec.authoritative_capability is AgentCapability.GENERAL

    # Verify unsupported capability handling on synthetic assessment if non-general/research
    synthetic_signal = RoutingSignal(
        signal_type=RoutingSignalType.LOCAL_CODE_OR_DEBUGGING,
        target_capability=AgentCapability.GENERAL,
        strength=SignalStrength.STRONG,
        evidence="local",
    )
    assert synthetic_signal.strength is SignalStrength.STRONG


# ---------------------------------------------------------------------------
# 4. Typed observation state / nullability matrix & privacy
# ---------------------------------------------------------------------------


def test_observation_state_matrix_valid_and_invalid_combinations() -> None:
    sha = hashlib.sha256(b"test-prompt").hexdigest()

    inelig = ShadowObservation.ineligible(
        observation_id="rec-1",
        canonical_input_sha256=sha,
        reason=IneligibilityReason.AMBIGUOUS_ASSESSMENT,
    )
    assert inelig.state is ObservationState.INELIGIBLE
    assert inelig.authoritative_capability is None
    assert inelig.latency_ns is None

    with pytest.raises(FrozenInstanceError):
        inelig.eligible = True  # type: ignore[misc]

    # Ineligible must reject non-null route/model/result/error/latency fields
    with pytest.raises(ValueError):
        ShadowObservation(
            observation_id="rec-1",
            canonical_input_sha256=sha,
            state=ObservationState.INELIGIBLE,
            eligible=False,
            ineligibility_reason=IneligibilityReason.AMBIGUOUS_ASSESSMENT,
            stratum=None,
            authoritative_capability="general",
            shadow_decision=None,
            general_probability=None,
            research_probability=None,
            agreement=None,
            disagreement_category=None,
            error_code=None,
            latency_ns=None,
        )

    agreed = ShadowObservation.completed(
        observation_id="rec-2",
        canonical_input_sha256=sha,
        stratum=ReportStratum.CLEAR_GENERAL_SIGNAL_BACKED,
        authoritative_capability="general",
        shadow_decision="general",
        general_probability=0.91,
        research_probability=0.09,
        latency_ns=420,
    )
    assert agreed.state is ObservationState.ELIGIBLE_AGREED
    assert agreed.agreement is True
    assert agreed.disagreement_category is None

    with pytest.raises(ValueError):
        ShadowObservation(
            observation_id="rec-2",
            canonical_input_sha256=sha,
            state=ObservationState.ELIGIBLE_AGREED,
            eligible=True,
            ineligibility_reason=None,
            stratum=ReportStratum.CLEAR_GENERAL_SIGNAL_BACKED,
            authoritative_capability="general",
            shadow_decision="general",
            general_probability=0.91,
            research_probability=0.09,
            agreement=True,
            disagreement_category=DisagreementCategory.AUTHORITATIVE_GENERAL_SHADOW_RESEARCH,
            error_code=None,
            latency_ns=420,
        )

    disagreed = ShadowObservation.completed(
        observation_id="rec-3",
        canonical_input_sha256=sha,
        stratum=ReportStratum.CLEAR_RESEARCH,
        authoritative_capability="research",
        shadow_decision="general",
        general_probability=0.80,
        research_probability=0.20,
        latency_ns=500,
    )
    assert disagreed.state is ObservationState.ELIGIBLE_DISAGREED
    assert disagreed.agreement is False
    assert (
        disagreed.disagreement_category
        is DisagreementCategory.AUTHORITATIVE_RESEARCH_SHADOW_GENERAL
    )

    # Wrong directional category rejected
    with pytest.raises(ValueError):
        ShadowObservation(
            observation_id="rec-3",
            canonical_input_sha256=sha,
            state=ObservationState.ELIGIBLE_DISAGREED,
            eligible=True,
            ineligibility_reason=None,
            stratum=ReportStratum.CLEAR_RESEARCH,
            authoritative_capability="research",
            shadow_decision="general",
            general_probability=0.80,
            research_probability=0.20,
            agreement=False,
            disagreement_category=DisagreementCategory.AUTHORITATIVE_GENERAL_SHADOW_RESEARCH,
            error_code=None,
            latency_ns=500,
        )

    err_before = ShadowObservation.execution_error(
        observation_id="rec-4",
        canonical_input_sha256=sha,
        stratum=ReportStratum.CLEAR_GENERAL_ZERO_SIGNAL_DEFAULT,
        authoritative_capability="general",
        error_code=PerRecordErrorCode.INFERENCE_TIMEOUT,
        latency_ns=None,
    )
    assert err_before.state is ObservationState.ELIGIBLE_SHADOW_EXECUTION_ERROR
    assert err_before.disagreement_category is DisagreementCategory.SHADOW_EXECUTION_ERROR
    assert err_before.latency_ns is None

    err_measured = ShadowObservation.execution_error(
        observation_id="rec-5",
        canonical_input_sha256=sha,
        stratum=ReportStratum.CLEAR_GENERAL_ZERO_SIGNAL_DEFAULT,
        authoritative_capability="general",
        error_code=PerRecordErrorCode.INVALID_PROBABILITIES,
        latency_ns=900,
    )
    assert err_measured.latency_ns == 900

    with pytest.raises(ValueError):
        ShadowObservation(
            observation_id="rec-5",
            canonical_input_sha256=sha,
            state=ObservationState.ELIGIBLE_SHADOW_EXECUTION_ERROR,
            eligible=True,
            ineligibility_reason=None,
            stratum=ReportStratum.CLEAR_GENERAL_ZERO_SIGNAL_DEFAULT,
            authoritative_capability="general",
            shadow_decision="general",
            general_probability=None,
            research_probability=None,
            agreement=None,
            disagreement_category=DisagreementCategory.SHADOW_EXECUTION_ERROR,
            error_code=PerRecordErrorCode.INVALID_PROBABILITIES,
            latency_ns=900,
        )


# ---------------------------------------------------------------------------
# 5. Corpus-manifest & overlap validation
# ---------------------------------------------------------------------------


def test_corpus_manifest_distinguishes_hashes_and_rejects_duplicates_and_contamination(
    tmp_path: Path,
) -> None:
    overlap = _make_overlap_manifest(["\ufeffKnown training prompt  \r\n"])

    # 1. Distinguish corpus_sha256 vs canonical_manifest_sha256
    rec = [{"record_id": "r1", "prompt": "What is 15% of 850?", "corpus_eligible": True}]
    manifest = _make_corpus_manifest(rec)
    assert manifest.corpus_sha256 != manifest.canonical_manifest_sha256

    tampered_corpus_sha = manifest.to_dict()
    tampered_corpus_sha["corpus_sha256"] = "0" * 64
    tampered_corpus_sha[MANIFEST_SELF_HASH_FIELD] = canonical_json_sha256(
        tampered_corpus_sha, exclude_self_hash_field=MANIFEST_SELF_HASH_FIELD
    )
    with pytest.raises(FatalRunError) as exc_sha:
        CorpusManifest.from_dict(tampered_corpus_sha)
    assert exc_sha.value.code is FatalRunErrorCode.CORPUS_NOT_APPROVED

    # 2. Duplicate canonical_input_sha256 (identical metadata and conflicting metadata)
    dup_identical = _make_corpus_manifest([
        {"record_id": "r1", "prompt": "What is 15% of 850?", "corpus_eligible": True},
        {"record_id": "r1", "prompt": "\ufeffWhat is 15% of 850?  \r\n", "corpus_eligible": True},
    ])
    with pytest.raises(FatalRunError) as exc_dup1:
        validate_synthetic_corpus_manifest_for_test(dup_identical, [overlap])
    assert exc_dup1.value.code is FatalRunErrorCode.CORPUS_NOT_APPROVED

    dup_conflict = _make_corpus_manifest([
        {"record_id": "r1", "prompt": "What is 15% of 850?", "corpus_eligible": True},
        {"record_id": "r2", "prompt": "What is 15% of 850?", "corpus_eligible": False},
    ])
    with pytest.raises(FatalRunError) as exc_dup2:
        validate_synthetic_corpus_manifest_for_test(dup_conflict, [overlap])
    assert exc_dup2.value.code is FatalRunErrorCode.CORPUS_NOT_APPROVED

    # 3. Contamination detected across corpus_eligible=False via same canonicalization
    contaminated = _make_corpus_manifest([
        {"record_id": "r1", "prompt": "What is 15% of 850?", "corpus_eligible": True},
        {"record_id": "r2", "prompt": "Known training prompt", "corpus_eligible": False},
    ])
    with pytest.raises(FatalRunError) as exc_contam:
        validate_synthetic_corpus_manifest_for_test(contaminated, [overlap])
    assert exc_contam.value.code is FatalRunErrorCode.CORPUS_NOT_APPROVED

    # 4. Empty overlap list rejected
    with pytest.raises(FatalRunError) as exc_no_overlap:
        validate_synthetic_corpus_manifest_for_test(manifest, [])
    assert exc_no_overlap.value.code is FatalRunErrorCode.CORPUS_NOT_APPROVED

    # 5. Real replay validation enforces >=200 unique eligible floor
    with pytest.raises(FatalRunError) as exc_floor:
        validate_corpus_manifest(manifest, [overlap])
    assert exc_floor.value.code is FatalRunErrorCode.CORPUS_NOT_APPROVED

    # 6. Malformed JSON and invalid UTF-8 mapped deterministically without raw exception text
    bad_utf8_path = tmp_path / "bad_utf8.json"
    bad_utf8_path.write_bytes(b"\xff\xfe{}")
    with pytest.raises(FatalRunError) as exc_utf8:
        CorpusManifest.from_json_file(bad_utf8_path)
    assert exc_utf8.value.code is FatalRunErrorCode.CORPUS_NOT_APPROVED
    assert "utf-8" not in exc_utf8.value.detail.lower() or "invalid utf-8" in exc_utf8.value.detail.lower()

    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(FatalRunError) as exc_json:
        CorpusManifest.from_json_file(bad_json_path)
    assert exc_json.value.code is FatalRunErrorCode.CORPUS_NOT_APPROVED
    assert "Expecting property name" not in str(exc_json.value)


# ---------------------------------------------------------------------------
# 6. Promotion gate precedence
# ---------------------------------------------------------------------------


def test_promotion_gate_exact_precedence() -> None:
    base_200 = [
        _build_observation(i, auth_cap="general" if i < 100 else "research")
        for i in range(200)
    ]

    # 4. Clean >=200 total and >=50 per class -> ELIGIBLE_FOR_HUMAN_REVIEW
    res_clean = evaluate_promotion_gate(base_200)
    assert res_clean.gate is PromotionGate.ELIGIBLE_FOR_HUMAN_REVIEW

    # 3. Insufficient total (<200) or <50 per class -> INVALID_RUN_INSUFFICIENT_EVIDENCE
    res_small = evaluate_promotion_gate(base_200[:199])
    assert res_small.gate is PromotionGate.INVALID_RUN_INSUFFICIENT_EVIDENCE

    skewed = [
        _build_observation(i, auth_cap="general" if i < 160 else "research")
        for i in range(200)
    ]
    res_skewed = evaluate_promotion_gate(skewed)
    assert res_skewed.gate is PromotionGate.INVALID_RUN_INSUFFICIENT_EVIDENCE

    # 2. Execution errors beat >=200 clean evidence
    with_err = list(base_200)
    with_err[0] = _build_observation(
        0,
        auth_cap="general",
        state=ObservationState.ELIGIBLE_SHADOW_EXECUTION_ERROR,
        error_code=PerRecordErrorCode.INFERENCE_TIMEOUT,
    )
    res_err = evaluate_promotion_gate(with_err)
    assert res_err.gate is PromotionGate.INVALID_RUN_EXECUTION_ERRORS

    # 1. Dangerous disagreement (auth=research, shadow=general) beats execution errors and >=200
    with_danger_and_err = list(with_err)
    with_danger_and_err[150] = _build_observation(
        150, auth_cap="research", shadow="general"
    )
    res_danger = evaluate_promotion_gate(with_danger_and_err)
    assert res_danger.gate is PromotionGate.STOP_AND_INVESTIGATE
    assert res_danger.dangerous_disagreement_count == 1


# ---------------------------------------------------------------------------
# 7. Worker boundary, probability validation, and privacy-safe serialization
# ---------------------------------------------------------------------------


def test_worker_probability_and_output_validation() -> None:
    # Valid within tolerance normalizes cleanly
    ok = validate_and_resolve_prediction(
        WorkerPrediction(general_probability=0.80004, research_probability=0.20, latency_ns=100)
    )
    assert not isinstance(ok, WorkerFailure)
    assert ok.shadow_decision == "general"
    assert abs((ok.general_probability + ok.research_probability) - 1.0) < 1e-12

    # Exact tie rejected
    tie = validate_and_resolve_prediction(
        WorkerPrediction(general_probability=0.5, research_probability=0.5, latency_ns=100)
    )
    assert isinstance(tie, WorkerFailure)
    assert tie.error_code is PerRecordErrorCode.INVALID_PROBABILITIES

    # Boolean / NaN / Inf / out-of-range / bad row sum rejected
    for bad_probs in (
        (True, False),
        (float("nan"), 0.5),
        (float("inf"), 0.0),
        (1.2, -0.2),
        (0.6, 0.2),
    ):
        res = validate_and_resolve_prediction(
            WorkerPrediction(
                general_probability=bad_probs[0],  # type: ignore[arg-type]
                research_probability=bad_probs[1],  # type: ignore[arg-type]
                latency_ns=100,
            )
        )
        assert isinstance(res, WorkerFailure)
        assert res.error_code is PerRecordErrorCode.INVALID_PROBABILITIES

    # Non-prediction object or mismatched explicit shadow_decision -> MALFORMED_OUTPUT
    malformed = validate_and_resolve_prediction({"choice": "general"})
    assert isinstance(malformed, WorkerFailure)
    assert malformed.error_code is PerRecordErrorCode.MALFORMED_OUTPUT

    mismatch = validate_and_resolve_prediction(
        WorkerPrediction(
            general_probability=0.9,
            research_probability=0.1,
            latency_ns=100,
            shadow_decision="research",
        )
    )
    assert isinstance(mismatch, WorkerFailure)
    assert mismatch.error_code is PerRecordErrorCode.MALFORMED_OUTPUT


def test_environment_fingerprint_allowlist_and_telemetry_write_failure(tmp_path: Path) -> None:
    fp = collect_safe_environment_fingerprint(
        repository_commit="aca2962353c1bc6d112b40bc7404c2d8954ed43c",
        package_versions={"pytest": "9.2.0"},
    )
    assert isinstance(fp, EnvironmentFingerprint)
    assert set(fp.to_dict().keys()) == ALLOWED_ENVIRONMENT_FINGERPRINT_FIELDS
    assert "agreed_count" not in PROTOCOL_SUPERSESSION_NOTES
    assert "disagreement_category_when_agreed" in PROTOCOL_SUPERSESSION_NOTES

    # Telemetry write failure raises TELEMETRY_WRITE_FAILED, never a record error
    obs = _build_observation(1)
    assert canonical_json_dumps(obs.to_dict()) == serialize_observation_jsonl(obs)

    corpus = _build_220_record_corpus()
    overlap = _make_overlap_manifest(["Do not include this training prompt."])
    worker = StubShadowWorker(
        lambda p: WorkerPrediction(
            general_probability=0.15 if "consensus protocol" in p else 0.90,
            research_probability=0.85 if "consensus protocol" in p else 0.10,
            latency_ns=250,
        )
    )
    # Passing an existing directory as telemetry_jsonl_path forces OSError on write
    bad_jsonl_dir = tmp_path / "existing_dir"
    bad_jsonl_dir.mkdir()
    with pytest.raises(FatalRunError) as exc_write:
        run_offline_shadow_evaluation(
            corpus_manifest=corpus,
            overlap_manifests=[overlap],
            worker=worker,
            telemetry_jsonl_path=bad_jsonl_dir,
        )
    assert exc_write.value.code is FatalRunErrorCode.TELEMETRY_WRITE_FAILED


# ---------------------------------------------------------------------------
# 8. End-to-end Increment 1 offline coordinator
# ---------------------------------------------------------------------------


def test_offline_coordinator_real_replay_reaches_human_review_and_is_privacy_safe(
    tmp_path: Path,
) -> None:
    corpus = _build_220_record_corpus()
    overlap = _make_overlap_manifest(["Known training set prompt"])
    seen_prompts: list[str] = []

    def stub_predict(canonical_prompt: str) -> WorkerPrediction:
        seen_prompts.append(canonical_prompt)
        if "consensus protocol" in canonical_prompt:
            return WorkerPrediction(
                general_probability=0.10,
                research_probability=0.90,
                latency_ns=300,
            )
        return WorkerPrediction(
            general_probability=0.92,
            research_probability=0.08,
            latency_ns=200,
        )

    jsonl_path = tmp_path / "observations.jsonl"
    summary_path = tmp_path / "summary.json"
    result = run_offline_shadow_evaluation(
        corpus_manifest=corpus,
        overlap_manifests=[overlap],
        worker=StubShadowWorker(stub_predict),
        environment=collect_safe_environment_fingerprint(
            repository_commit="aca2962353c1bc6d112b40bc7404c2d8954ed43c"
        ),
        telemetry_jsonl_path=jsonl_path,
        summary_path=summary_path,
    )

    assert result.promotion.gate is PromotionGate.ELIGIBLE_FOR_HUMAN_REVIEW
    assert result.promotion.total_records == 222
    assert result.promotion.ineligible_count == 2
    assert result.promotion.completed_eligible_count == 220
    assert result.promotion.completed_general_count == 110
    assert result.promotion.completed_research_count == 110
    assert result.promotion.stratum_counts == {
        "clear_general_signal_backed": 55,
        "clear_general_zero_signal_default": 55,
        "clear_research": 110,
    }
    # Ineligible records never reached the worker
    assert len(seen_prompts) == 220
    assert all("Excluded corpus record" not in p for p in seen_prompts)
    assert all("checking script #999" not in p for p in seen_prompts)

    jsonl_text = jsonl_path.read_text(encoding="utf-8")
    summary_text = summary_path.read_text(encoding="utf-8")
    assert len(jsonl_text.splitlines()) == 222
    # Verify no prompts or routing reasons leaked into serialized telemetry
    for sensitive in ("What is 10 percent of 850?", "consensus protocol", "Strong general signals", "Strong research signals"):
        assert sensitive not in jsonl_text
        assert sensitive not in summary_text


def test_coordinator_continues_after_dangerous_disagreement_and_executes_error_lifecycle() -> None:
    corpus = _make_corpus_manifest([
        {"record_id": "r1", "prompt": "Research latest CVEs in OpenSSL.", "corpus_eligible": True},
        {"record_id": "r2", "prompt": "What is 15% of 850?", "corpus_eligible": True},
        {"record_id": "r3", "prompt": "Summarize item #3.", "corpus_eligible": True},
    ])
    overlap = _make_overlap_manifest(["Other prompt"])

    calls: list[str] = []

    def mixed_worker(prompt: str) -> WorkerPrediction | WorkerFailure:
        calls.append(prompt)
        if "OpenSSL" in prompt:
            # Dangerous disagreement: authoritative research vs shadow general
            return WorkerPrediction(general_probability=0.88, research_probability=0.12, latency_ns=100)
        if "850" in prompt:
            raise TimeoutError("worker timed out with secret /home/user/path")
        raise RuntimeError("unexpected crash with secret token")

    result = run_synthetic_shadow_evaluation_for_test(
        corpus_manifest=corpus,
        overlap_manifests=[overlap],
        worker=StubShadowWorker(mixed_worker),
    )

    # Bounded run continued after dangerous disagreement through all 3 records
    assert len(calls) == 3
    assert result.observations[0].state is ObservationState.ELIGIBLE_DISAGREED
    assert (
        result.observations[0].disagreement_category
        is DisagreementCategory.AUTHORITATIVE_RESEARCH_SHADOW_GENERAL
    )
    assert result.observations[1].state is ObservationState.ELIGIBLE_SHADOW_EXECUTION_ERROR
    assert result.observations[1].error_code is PerRecordErrorCode.INFERENCE_TIMEOUT
    assert result.observations[2].state is ObservationState.ELIGIBLE_SHADOW_EXECUTION_ERROR
    assert result.observations[2].error_code is PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR
    assert result.promotion.gate is PromotionGate.STOP_AND_INVESTIGATE

    # Ensure raw exception messages did not leak into JSONL
    serialized = "\n".join(serialize_observation_jsonl(o) for o in result.observations)
    assert "/home/user/path" not in serialized
    assert "unexpected crash" not in serialized


def test_synthetic_coordinator_preflight_blocks_unsupported_router_mode_before_records() -> None:
    corpus = _make_corpus_manifest([
        {"record_id": "inelig-only", "prompt": "Some prompt.", "corpus_eligible": False}
    ])
    overlap = _make_overlap_manifest(["Other prompt"])

    with pytest.raises(FatalRunError) as exc_info:
        run_synthetic_shadow_evaluation_for_test(
            corpus_manifest=corpus,
            overlap_manifests=[overlap],
            worker=StubShadowWorker(
                lambda _p: WorkerPrediction(general_probability=0.9, research_probability=0.1, latency_ns=10)
            ),
            router_mode=RouterMode.UNSUPPORTED,
        )
    assert exc_info.value.code is FatalRunErrorCode.UNSUPPORTED_ROUTER_MODE
