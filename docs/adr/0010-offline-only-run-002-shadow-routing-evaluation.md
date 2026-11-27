# ADR-0010: Offline-Only Run 002 Shadow Routing Evaluation

- **Status:** Accepted
- **Date:** 2026-11-25
- **Repository basis:** `aca2962353c1bc6d112b40bc7404c2d8954ed43c`
- **Pinned contract:** `run-002-offline-shadow-integration-003`
- **Contract canonical SHA-256:** `40ab49738b4f45ea5fc13b9fc7f498622fab0af313b81c0698e4a46746a13fed`
- **Contract physical SHA-256:** `e9f63fd1e2c2686166588161c0b67eb2631c1f9288f9c2b05eccd6df24c98cfe`

## Decision

Run 002 is evaluated only through an offline coordinator in `evaluation/routing/`. `SubmitTask`, production ports, and production composition roots remain unchanged. `DeterministicCapabilityRouter` remains solely authoritative and comparison uses `RoutingAssessment.provisional_capability`. Live shadowing and assisted routing require separate ADRs.

The coordinator constructs `DeterministicCapabilityRouter()` directly, canonicalizes each approved corpus prompt once, and sends the identical canonical string to the router and a subprocess-isolated model worker. One pure eligibility function runs before model loading or inference. Zero-signal default-general observations are reported separately from signal-backed general observations.

The accepted contract above is normative for canonicalization, eligibility precedence, corpus hashing, duplicate rejection, telemetry state/nullability, promotion-gate precedence, and closed error taxonomy. Expected model and protocol identities are externally pinned in that contract. The exact artifact file set is verified bidirectionally before any loader import or deserialization; symlinks, extra files, unsafe paths, identity mismatch, network attempts, and label-map mismatch abort the run. Loading is local-only with `trust_remote_code=False`.

Per-record inference failures continue the bounded run but prevent a clean review gate. Fatal integrity, preflight, or telemetry failures abort nonzero without a valid completed report. Promotion gates are closed to `STOP_AND_INVESTIGATE`, `INVALID_RUN_EXECUTION_ERRORS`, `INVALID_RUN_INSUFFICIENT_EVIDENCE`, and `ELIGIBLE_FOR_HUMAN_REVIEW`; none authorizes automatic promotion.

## Delivery sequence

1. Offline coordinator, canonicalization, eligibility, schemas, corpus validation, promotion logic, and stub worker, with no ML runtime.
2. Subprocess Run 002 worker with externally anchored verify-before-deserialize, offline/network enforcement, runtime label verification, probability validation, and timeout/native-crash containment.
3. Approved 200-plus-case replay with contamination checks, JSONL and summary evidence, strata, and promotion verdict.

## Dependency boundary

`setfit`, `torch`, `transformers`, `sentence_transformers`, `sklearn`, and `evaluation.routing` imports are prohibited throughout `services/orchestrator/`. Architecture tests enforce the boundary.

## Consequences

The design produces bounded, non-authoritative evidence without introducing live prompts or ML dependencies into production services. It requires an approved corpus, a maintained external trust anchor, subprocess IPC, and a separate future ADR for any live or assisted-routing use. Offline success remains necessary evidence only and does not establish production fitness.

## Acceptance

ADR-0010 was accepted by the human maintainer on 2026-11-25 following independent Gemini 4 Neon architecture review.

Final review disposition: **accept as proposed**.

The pinned `run-002-offline-shadow-integration-003` contract remains byte-for-byte unchanged:

- Canonical SHA-256: `40ab49738b4f45ea5fc13b9fc7f498622fab0af313b81c0698e4a46746a13fed`
- Physical SHA-256: `e9f63fd1e2c2686166588161c0b67eb2631c1f9288f9c2b05eccd6df24c98cfe`

The contract identities, authority boundary, trust anchors, failure lifecycle, telemetry rules, corpus governance, promotion gates, dependency isolation, and three-increment delivery sequence are approved.

Implementation is authorized to proceed with Increment 1 only.
