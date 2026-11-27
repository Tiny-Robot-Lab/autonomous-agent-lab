"""Offline-only Run 002 shadow-routing coordinator (Increment 1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from orchestrator.domain.router import DeterministicCapabilityRouter

from evaluation.routing.run_002.contract import (
    CONTRACT_ID,
    CONTRACT_SCHEMA_VERSION,
    EXPECTED_MODEL_ARTIFACT_IDENTITY,
    EXPECTED_PROTOCOL_IDENTITY,
    load_verified_contract,
)
from evaluation.routing.run_002.corpus import (
    CorpusManifest,
    ForbiddenOverlapManifest,
    ValidatedCorpus,
    validate_corpus_manifest,
    validate_synthetic_corpus_manifest_for_test,
)
from evaluation.routing.run_002.eligibility import (
    RouterMode,
    evaluate_eligibility,
    validate_run_preflight,
)
from evaluation.routing.run_002.errors import (
    FatalRunError,
    PerRecordErrorCode,
)
from evaluation.routing.run_002.promotion import (
    PromotionEvaluation,
    evaluate_promotion_gate,
)
from evaluation.routing.run_002.telemetry import (
    EnvironmentFingerprint,
    RunSummary,
    ShadowObservation,
    collect_safe_environment_fingerprint,
    write_observations_jsonl,
    write_run_summary,
)
from evaluation.routing.run_002.worker import (
    ShadowWorker,
    ValidatedWorkerPrediction,
    WorkerFailure,
    validate_and_resolve_prediction,
)


@dataclass(frozen=True)
class OfflineShadowRunResult:
    """Bounded result returned by a completed Run 002 offline shadow evaluation."""

    observations: tuple[ShadowObservation, ...]
    promotion: PromotionEvaluation
    summary: RunSummary


def _execute_coordinator_loop(
    *,
    validated_corpus: ValidatedCorpus,
    worker: ShadowWorker,
    router_mode: RouterMode,
    environment: EnvironmentFingerprint | None,
    telemetry_jsonl_path: Path | None,
    summary_path: Path | None,
    contract_path: Path | None,
) -> OfflineShadowRunResult:
    load_verified_contract(contract_path)
    validate_run_preflight(
        corpus_approved=validated_corpus.manifest.approved,
        router_mode=router_mode,
    )

    router = DeterministicCapabilityRouter()
    observations: list[ShadowObservation] = []

    for validated_rec in validated_corpus.records:
        rec = validated_rec.record
        canon = validated_rec.canonical
        assessment = router.assess(canon.canonical_prompt)

        decision = evaluate_eligibility(
            canonical_prompt=canon,
            corpus_approved=validated_corpus.manifest.approved,
            corpus_eligible=rec.corpus_eligible,
            router_mode=router_mode,
            assessment=assessment,
        )

        if not decision.eligible:
            assert decision.ineligibility_reason is not None
            observations.append(
                ShadowObservation.ineligible(
                    observation_id=rec.record_id,
                    canonical_input_sha256=canon.canonical_input_sha256,
                    reason=decision.ineligibility_reason,
                )
            )
            continue

        assert decision.stratum is not None
        assert decision.authoritative_capability is not None
        auth_cap = decision.authoritative_capability.value

        try:
            raw_result = worker.predict(canon.canonical_prompt)
            resolved = validate_and_resolve_prediction(raw_result)
        except FatalRunError:
            raise
        except TimeoutError:
            resolved = WorkerFailure(
                error_code=PerRecordErrorCode.INFERENCE_TIMEOUT,
                latency_ns=None,
            )
        # Contract requires non-fatal worker crashes to continue the bounded run
        # as per-record unexpected_observer_error without leaking exception text.
        except Exception:  # noqa: BLE001
            resolved = WorkerFailure(
                error_code=PerRecordErrorCode.UNEXPECTED_OBSERVER_ERROR,
                latency_ns=None,
            )

        if isinstance(resolved, WorkerFailure):
            observations.append(
                ShadowObservation.execution_error(
                    observation_id=rec.record_id,
                    canonical_input_sha256=canon.canonical_input_sha256,
                    stratum=decision.stratum,
                    authoritative_capability=auth_cap,
                    error_code=resolved.error_code,
                    latency_ns=resolved.latency_ns,
                )
            )
            continue

        assert isinstance(resolved, ValidatedWorkerPrediction)
        observations.append(
            ShadowObservation.completed(
                observation_id=rec.record_id,
                canonical_input_sha256=canon.canonical_input_sha256,
                stratum=decision.stratum,
                authoritative_capability=auth_cap,
                shadow_decision=resolved.shadow_decision,
                general_probability=resolved.general_probability,
                research_probability=resolved.research_probability,
                latency_ns=resolved.latency_ns,
            )
        )

    promotion = evaluate_promotion_gate(observations)
    env = environment or collect_safe_environment_fingerprint()
    summary = RunSummary(
        contract_id=CONTRACT_ID,
        schema_version=CONTRACT_SCHEMA_VERSION,
        corpus_id=validated_corpus.manifest.corpus_id,
        corpus_sha256=validated_corpus.manifest.corpus_sha256,
        corpus_manifest_sha256=validated_corpus.manifest.canonical_manifest_sha256,
        overlap_manifest_hashes=validated_corpus.overlap_manifest_hashes,
        expected_model_artifact_identity=EXPECTED_MODEL_ARTIFACT_IDENTITY,
        expected_protocol_identity=EXPECTED_PROTOCOL_IDENTITY,
        environment=env,
        total_records=promotion.total_records,
        ineligible_count=promotion.ineligible_count,
        eligible_count=promotion.eligible_count,
        agreed_count=promotion.agreed_count,
        disagreed_count=promotion.disagreed_count,
        dangerous_disagreement_count=promotion.dangerous_disagreement_count,
        execution_error_count=promotion.execution_error_count,
        completed_general_count=promotion.completed_general_count,
        completed_research_count=promotion.completed_research_count,
        stratum_counts=promotion.stratum_counts,
        promotion_gate=promotion.gate.value,
        completed=True,
    )

    if telemetry_jsonl_path is not None:
        write_observations_jsonl(observations, telemetry_jsonl_path)
    if summary_path is not None:
        write_run_summary(summary, summary_path)

    return OfflineShadowRunResult(
        observations=tuple(observations),
        promotion=promotion,
        summary=summary,
    )


def run_offline_shadow_evaluation(
    *,
    corpus_manifest: CorpusManifest,
    overlap_manifests: Sequence[ForbiddenOverlapManifest],
    worker: ShadowWorker,
    environment: EnvironmentFingerprint | None = None,
    telemetry_jsonl_path: Path | None = None,
    summary_path: Path | None = None,
    contract_path: Path | None = None,
) -> OfflineShadowRunResult:
    """Execute a bounded Run 002 offline shadow evaluation on an approved corpus.

    Constructs DeterministicCapabilityRouter() directly with no injected
    router instances, keywords, or patterns, and enforces full corpus-manifest
    validation (including >=200 eligible records) before invoking the worker.
    """
    load_verified_contract(contract_path)
    validated_corpus = validate_corpus_manifest(corpus_manifest, overlap_manifests)
    return _execute_coordinator_loop(
        validated_corpus=validated_corpus,
        worker=worker,
        router_mode=RouterMode.DEFAULT_COORDINATOR_ROUTER,
        environment=environment,
        telemetry_jsonl_path=telemetry_jsonl_path,
        summary_path=summary_path,
        contract_path=contract_path,
    )


def run_synthetic_shadow_evaluation_for_test(
    *,
    corpus_manifest: CorpusManifest,
    overlap_manifests: Sequence[ForbiddenOverlapManifest],
    worker: ShadowWorker,
    router_mode: RouterMode = RouterMode.DEFAULT_COORDINATOR_ROUTER,
    environment: EnvironmentFingerprint | None = None,
    telemetry_jsonl_path: Path | None = None,
    summary_path: Path | None = None,
    contract_path: Path | None = None,
) -> OfflineShadowRunResult:
    """Explicit test-only coordinator entry point allowing small synthetic manifests.

    Must never be exposed to production replay or future CLI entry points.
    """
    load_verified_contract(contract_path)
    validate_run_preflight(
        corpus_approved=corpus_manifest.approved,
        router_mode=router_mode,
    )
    validated_corpus = validate_synthetic_corpus_manifest_for_test(
        corpus_manifest,
        overlap_manifests,
    )
    return _execute_coordinator_loop(
        validated_corpus=validated_corpus,
        worker=worker,
        router_mode=router_mode,
        environment=environment,
        telemetry_jsonl_path=telemetry_jsonl_path,
        summary_path=summary_path,
        contract_path=contract_path,
    )
