# Strands Decider Bounded Shadow Routing Evaluation (Issue #237)

This report documents the bounded shadow-routing evaluation harness implemented for GitHub issue #237.

## 1. Purpose

Evaluate whether stock Strands Decider (`strands-decider`) can operate as a bounded, non-authoritative shadow classifier alongside Tiny Agent Lab's deterministic router over a fixed candidate set, with explicit abstention (`none`), deterministic canonical input serialization, and fail-safe isolation.

## 2. Authority Boundary

- **Authoritative router**: `DeterministicCapabilityRouter` (`services/orchestrator/src/orchestrator/domain/router.py`) remains strictly authoritative for all routing decisions.
- **Shadow-only constraint**: The Strands Decider harness (`evaluation/routing/strands_shadow.py`) runs in shadow mode only. No Decider output, confidence score, abstention, timeout, or failure may alter authoritative routing, task state, task lineage, specialist selection, permissions, capability grants, delegation, recovery, provider selection, or terminal outcomes.
- **Non-interference**: Disabling or removing the shadow evaluator leaves Tiny Agent Lab's deterministic routing behavior unchanged.

## 3. Decider Environment, Version, and Transport Boundary

- **Package & model**: `strands-decider` version `0.4.0`, configured model `StrandsAgents/strands-decider-E2B-gemma4-v2-2611` on `mps` (earlier `hobson-v21` and `Qwen3.5-v1` candidates are superseded and not configured).
- **Localhost HTTP boundary**: `LocalStrandsDeciderHttpAdapter` (`evaluation/routing/strands_shadow.py`) uses Python standard-library `urllib.request` to call `POST http://127.0.0.1:8099/v1/systemone`. Only loopback hosts (`127.0.0.1`, `localhost`, `::1`) are permitted; non-loopback hosts, non-HTTP(S) schemes, and credential-bearing URLs are rejected deterministically. Tiny Agent Lab does not start, stop, download, or manage the external server lifecycle.
- **Direct Python API finding**: Package inspection of `strands-decider 0.4.0` established that `strands_decider.Decider` is not exported, so the earlier speculative direct Python adapter was removed and replaced by the bounded localhost HTTP adapter.
- **Request & response contract**: Sends `state` (the bounded repository-controlled prompt) and one `route` choice question with `instructions` (`"Which Tiny Agent Lab route should handle this request?"`) and `general`, `research`, and `none` criteria (no private platform state or extra prompt history). Parses and strictly validates `answers.route.choice`, `answers.route.probabilities`, `answers.route.confidence`, and optional `usage`, `latency_ms` (server-reported), and `model`, while recording client-observed elapsed time (`latency_ns`) separately.
- **Protocol-integration note**: An initial live probe omitted `questions.route.instructions`, and the server returned `HTTP 422 Unprocessable Content` (`questions.route.choice.instructions: Field required`). That failed attempt is recorded solely as protocol-integration evidence (not a routing-quality result); adding `STRANDS_ROUTE_INSTRUCTIONS` resolved the schema mismatch and yielded 17/17 succeeded live evaluations across both passes.

## 4. Canonical Shadow Input Contract

`CanonicalShadowInput` (`version = "shadow_routing_input.v1"`) defines the bounded input passed to the shadow adapter:

- `version`: explicit contract version (`"shadow_routing_input.v1"`).
- `case_id`: non-empty case identifier.
- `prompt`: trimmed routing-relevant task prompt.
- `candidate_choices`: fixed tuple `("general", "research", "none")`.
- `has_supplied_artifact_context`: boolean indicator for supplied local artifact/code context.

Serialization uses `agent_contracts.serialization.canonical_json_dumps` (sorted keys, compact separators, `ensure_ascii=False`, `allow_nan=False`) and `check_no_credential_bearing_keys`. Note that `check_no_credential_bearing_keys` validates structural dictionary keys only and does **not** inspect free-text `prompt` content for secrets.

**Offline-only and telemetry boundaries**:

- Arbitrary live user prompts are not accepted by this experiment; `run_shadow_corpus_evaluation` is restricted to repository-controlled offline corpus cases.
- Canonical input JSON is an offline experiment input format and is not production telemetry.
- This experiment must not persist arbitrary runtime prompts.
- Any later runtime shadow path requires a separately reviewed redaction or bounded-projection contract.

## 5. Candidate Set and Confidence Threshold

- **Candidate set**: `("general", "research", "none")`, derived from `AgentCapability` (`general`, `research`) plus explicit experimental abstention (`none`).
- **Provisional threshold**: `0.70` (`DEFAULT_SHADOW_CONFIDENCE_THRESHOLD`), frozen prior to corpus evaluation as an experimental default for 3-way classification (where uniform chance is `0.333`), not a validated optimal value.
- **Distinct outcome categories**:
  - `deliberate_none`: Decider explicitly returns `"none"`.
  - `below_threshold`: Decider returns `"general"` or `"research"` with `confidence < 0.70` (effective shadow decision becomes `"none"`).
  - Execution failures (`unavailable`, `timeout`, `malformed`, `unknown_choice`): `effective_shadow_decision` remains `None` and is never silently converted into `"none"`.

## 6. Evaluation Corpus and Observed Baseline

`load_shadow_evaluation_corpus()` constructs a 17-case corpus:

- **7 cases** from `evaluation/routing/dataset.json`: clear general (`eval-001`, `eval-002`), clear research (`eval-006`, `eval-007`), and ambiguous borderline (`eval-011`, `eval-013`, `eval-014`).
- **8 cases** from `evaluation/routing/validation_dataset.json`: weak-verb supplied artifacts (`val-001`, `val-003`), freshness/availability (`val-005`, `val-012`), historical/negative guards (`val-015`, `val-020`), and mixed/weak-verb alone (`val-025`, `val-028`).
- **2 deliberate-`none` cases**: `shadow-none-001` and `shadow-none-002` (underspecified prompts where the deterministic router defaults to `general` while shadow abstention is evaluated).

### Live Corpus Execution Summary (`StrandsAgents/strands-decider-E2B-gemma4-v2-2611`, `mps`, `threshold = 0.70`)

Two consecutive passes (Run 1 and Run 2) over the 17-case corpus succeeded (`execution_failed = 0`) with identical aggregate counts: `agreed = 6`, `disagreed = 0`, `deliberate_none = 1`, `below_threshold = 10`, and zero raw-choice, effective-decision, or comparison-classification changes between runs.

| Case ID | Auth Route | Raw Choice | P(general) | P(research) | P(none) | Confidence | Effective Decision | Comparison | Run 1 Client / Server (ms) | Run 2 Client / Server (ms) |
|---|---|---|---:|---:|---:|---:|---|---|---:|---:|
| `eval-001` | `general` | `general` | 0.9390 | 0.0054 | 0.0557 | 0.9084 | `general` | `agreed` | 2008.822 / 1987.59 | 104.606 / 103.22 |
| `eval-002` | `general` | `general` | 0.7912 | 0.1612 | 0.0477 | 0.6868 | `none` | `below_threshold` | 130.668 / 128.65 | 149.088 / 147.82 |
| `eval-006` | `research` | `research` | 0.0344 | 0.9323 | 0.0333 | 0.8984 | `research` | `agreed` | 604.680 / 602.97 | 120.747 / 119.25 |
| `eval-007` | `research` | `research` | 0.0108 | 0.9809 | 0.0083 | 0.9714 | `research` | `agreed` | 347.344 / 346.01 | 120.298 / 118.91 |
| `eval-011` | `general` | `research` | 0.2799 | 0.6280 | 0.0921 | 0.4421 | `none` | `below_threshold` | 141.846 / 139.74 | 119.141 / 117.85 |
| `eval-013` | `research` | `research` | 0.0957 | 0.8884 | 0.0159 | 0.8326 | `research` | `agreed` | 362.881 / 361.00 | 118.849 / 117.41 |
| `eval-014` | `research` | `research` | 0.2025 | 0.7725 | 0.0251 | 0.6587 | `none` | `below_threshold` | 90.446 / 89.05 | 112.827 / 111.20 |
| `val-001` | `general` | `none` | 0.4204 | 0.0160 | 0.5636 | 0.3454 | `none` | `deliberate_none` | 386.771 / 385.50 | 169.269 / 167.71 |
| `val-003` | `general` | `general` | 0.5563 | 0.2280 | 0.2156 | 0.3345 | `none` | `below_threshold` | 179.042 / 177.33 | 112.875 / 111.31 |
| `val-005` | `research` | `research` | 0.0278 | 0.9288 | 0.0435 | 0.8932 | `research` | `agreed` | 110.542 / 109.07 | 124.754 / 123.01 |
| `val-012` | `research` | `research` | 0.0132 | 0.9566 | 0.0302 | 0.9349 | `research` | `agreed` | 116.186 / 114.77 | 115.375 / 113.92 |
| `val-015` | `general` | `research` | 0.2555 | 0.7144 | 0.0302 | 0.5715 | `none` | `below_threshold` | 148.402 / 146.56 | 122.725 / 121.29 |
| `val-020` | `general` | `general` | 0.7707 | 0.2103 | 0.0190 | 0.6560 | `none` | `below_threshold` | 375.222 / 373.78 | 118.799 / 117.79 |
| `val-025` | `research` | `research` | 0.1157 | 0.6782 | 0.2061 | 0.5173 | `none` | `below_threshold` | 451.964 / 450.84 | 201.641 / 200.29 |
| `val-028` | `general` | `general` | 0.4934 | 0.2410 | 0.2656 | 0.2401 | `none` | `below_threshold` | 101.431 / 100.20 | 110.445 / 108.78 |
| `shadow-none-001` | `general` | `general` | 0.5960 | 0.0652 | 0.3389 | 0.3940 | `none` | `below_threshold` | 101.624 / 100.16 | 121.788 / 120.56 |
| `shadow-none-002` | `general` | `general` | 0.7305 | 0.0504 | 0.2192 | 0.5957 | `none` | `below_threshold` | 749.289 / 747.66 | 114.302 / 112.78 |

## 7. Failures, Uncertainty, and Reproducibility

- **Failure-path coverage**: Deterministic tests in `tests/test_strands_decider_shadow.py` verify loopback-only URL validation, `/v1/systemone` request/response parsing, `disabled`, `unavailable` (`URLError` / `ConnectionRefusedError` / `HTTPError`), `timeout`, `malformed` (invalid JSON / non-dict / NaN / out-of-range / bool), `unknown_choice`, `below_threshold`, `deliberate_none`, and `disagreed` outcomes without altering the authoritative route, while letting unrelated programming bugs propagate rather than masking them as classifier failures.
- **Latency & reproducibility**: Client elapsed latency (`latency_ns` via `time.perf_counter_ns()`) and server-reported `latency_ms` are recorded separately. In live Runs 1 and 2, all 17 cases produced identical raw choices, effective decisions, and comparison classifications (`0` diffs), with Run 1 client latency `90.446–2008.822 ms` (`89.05–1987.59 ms` server) and Run 2 client latency `104.606–201.641 ms` (`103.22–200.29 ms` server).
- **Confidence & abstention observations**: Under the frozen `0.70` threshold, Decider agreed with the deterministic router on 5 clear `research` cases (`eval-006`, `eval-007`, `val-005`, `val-012`, plus ambiguous `eval-013`) and 1 clear `general` case (`eval-001`), while 10 cases fell below `0.70` (including `eval-011` and `val-015` where raw choice was `research` below threshold, and `shadow-none-001`/`shadow-none-002` where raw choice was `general` at `0.3940`/`0.5957` confidence) and `val-001` returned deliberate `none` (`0.3454` confidence).

## 8. Post-Evaluation `home-core` CPU Feasibility Test

- **Environment**: `home-core` (AMD Ryzen 5 3500U, 4 cores / 8 threads, 14 GiB physical RAM, 4 GiB swap), with Tiny Agent Lab running (`strands-decider 0.4.0`, checkpoint `StrandsAgents/strands-decider-E2B-gemma4-v2-2611`, device `cpu`).
- **Observed result**: Model download completed (~10.3 GB reconstructed) and weight loading began, at which point the Linux kernel invoked the global OOM killer and terminated `strands-decider` before weight loading completed (kernel evidence reported approximately 14,239,748 kB anonymous RSS at termination; `systemd-oomd` had no corresponding event). Tiny Agent Lab services remained running, Orchestrator and Redis remained healthy, and both local health and public `https://agents.tinyrobotlab.com/health` returned `{"status":"ok"}`.
- **Coexistence & test boundaries**: No Tiny Agent Lab services were stopped to make the experiment pass, and no retry with additional swap or reduced workload was attempted. Because the process was killed before weight loading completed, the model was not fully loaded and no CPU inference latency is claimed.
- **Feasibility summary**:
  - Mac MPS evaluation: **viable**;
  - Current `home-core` CPU feasibility: **no-go**.
- **Bounded conclusion**: The current E2B Gemma 4 v2 checkpoint is not viable on the existing 14 GiB CPU-only `home-core` under the tested coexistence conditions. This does not imply that Strands Decider is generally unsuitable, that additional RAM would necessarily make CPU inference practical, or that any specific RAM upgrade would be sufficient. Any future `home-core` feasibility work requires materially different hardware or a materially smaller suitable current-generation decision model.

## 9. Relationship to Prior Routing v2/v3 Evidence

Historical Routing v2 and Routing v3/Ollama benchmarks in [docs/routing_evaluation.md](routing_evaluation.md) remain unchanged. This harness complements that baseline by isolating a typed, thresholded 3-way shadow interface (`general`, `research`, `none`) that does not require modifying production routing or `RoutingDecision` lifecycle contracts.

## 10. Advisory Recommendation

- **Disposition (advisory)**: **Continue shadow** — keep `LocalStrandsDeciderHttpAdapter` strictly shadow-only; while Mac MPS live inference succeeded cleanly and reproducibly across both runs with zero failures, 10/17 cases fell below the provisional `0.70` threshold (and `val-001` abstained with `none`), and `home-core` CPU execution is a no-go on current hardware, warranting further shadow-only evaluation before any maintainer decision on assisted routing.
- The human maintainer retains sole acceptance and decision authority (`no-go`, `continue shadow`, or `candidate for later assisted-routing issue`).
