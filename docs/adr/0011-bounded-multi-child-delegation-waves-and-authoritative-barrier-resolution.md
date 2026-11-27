# ADR 0011: Bounded Multi-Child Delegation Waves and Authoritative Barrier Resolution

Status: Accepted
Date accepted: 2026-11-27
Related issue: https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/34
Extends: ADR 0006, ADR 0007
Supersedes: N/A

## Review status

Accepted by explicit human decision on 2026-11-27 after repository-grounded
architecture development and validation against live `main` at `5ad8a7e`,
independent Gemini 4 Neon High architecture and concurrency review, validation
of the review findings against the live repository, one consolidated revision,
and a final whole-document acceptance review.

The independent review returned `REVISE BEFORE HUMAN ACCEPTANCE REVIEW` with
six findings. Live-repository validation confirmed five material missing
invariants or contract clarifications and one non-blocking continuation-contract
clarification. The consolidated revision:

1. requires every accepted wave to contain at least one authorized child and
   enter `waiting_for_children` strictly unresolved;
2. constrains `PARTIAL` to exact rational
   `0 < required_success_fraction <= 1`, ensuring the derived required-success
   count is between one and the required-child count;
3. makes recovery eligibility conditional on an active unresolved wave and
   closes residual `recovery_required` children as
   `ended / barrier_resolved`;
4. prevents late child claim-timeout, execution-claim-expiry, recovery, or
   stale-fence operations from mutating children after wave closure;
5. requires all Issue #34 authoritative state to remain within one single-node
   Redis atomic consistency boundary and fail closed on missing, corrupt,
   incompatible, or inconsistent authoritative records;
6. binds exact wave-acceptance replay to the originating parent task,
   assignment, fencing epoch, proposal ID, and canonical proposal fingerprint;
   and
7. explicitly binds the versioned continuation to the wave and closes it as
   non-claimable when cancellation terminates the parent.

The final acceptance review found no remaining blocking or high-priority
architectural gaps, no contradiction with ADR 0006 or ADR 0007, and no
unintended expansion into result aggregation, observational tracing, arbitrary
dependency graphs, retries, replenishment, multiple waves, grandchildren,
dynamic provisioning, or physical process-termination guarantees.

Gemini 4 Neon High informed the revision but did not make the acceptance
decision. Final authority remained with the explicit human architecture
decision. This ADR is now the governing architecture for GitHub Issue #34.
Implementation increments may be planned only within its accepted boundaries.

## Context

ADR 0006 establishes platform-governed task lineage, bounded delegation
authority, depth-1 execution, one delegation wave per root, and a structural
parent-waits-for-children barrier distinct from arbitrary dependency graphs. A
single wave may contain multiple authorized children, but ADR 0006 intentionally
does not define multi-child synchronization, lifecycle closure, or failure
propagation.

ADR 0007 establishes task-owned continuation, authenticated pull-claim,
fenced assignments, canonical terminal outcomes, idempotent atomic completion
projection, parent resumability, claim expiry and recovery, and non-authoritative
wake-up hints. Its Issue #33 specialization implements one required child. ADR
0007 explicitly assigns multiple children, concurrent execution, and wait-all,
partial, timeout, and cancellation synchronization to Issue #34 without creating
a second continuation model.

The live implementation at `5ad8a7e` is intentionally single-child:

- `TaskContinuation` has one `child_task_id`;
- `DelegationGrant` enforces one child, one wave, and depth 1;
- canonical child terminal outcomes are `completed`, `failed`, and
  `claim_timed_out`;
- an acknowledged expired execution claim becomes `recovery_required`, with
  its old fence invalidated and no automatic retry;
- child terminal projection atomically makes the single-child parent
  `resumable` and emits at most one non-secret wake-up hint;
- concurrent claim/timeout and completion/expiry tests establish an atomic
  first-valid-transition-wins precedent;
- waiting parents hold no executor slot, and eligible replacement runtimes may
  later claim their continuation.

Issue #34 extends that substrate to multiple sibling children in one immutable
bounded delegation wave at depth 1. It does not introduce child-to-child
ordering, arbitrary execution DAGs, new continuation ownership, process-affine
callbacks, a completion stream, or a projector worker.

Three time-related mechanisms must remain distinct:

1. A **child claim deadline** applies to a pending child that has never been
   claimed and may produce canonical `claim_timed_out`.
2. An **acknowledged execution-claim expiry** applies to a claimed child whose
   executor authority expires and produces `recovery_required` pending an
   explicit trusted recovery decision.
3. A **delegation-wave deadline** bounds the unresolved collaborative barrier
   as a whole and may resolve the wave `TIMED_OUT`.

## Decision

### 1. One immutable bounded delegation wave

Issue #34 retains one delegation wave and depth 1. At atomic wave acceptance,
the trusted platform:

- validates the authenticated parent assignment, current fence, grant,
  authority intersection, cumulative fan-out bound, depth, proposal
  idempotency, immutable inputs, and sealed artifact references;
- mints and persists every authorized child with authoritative lineage,
  parent membership, required capability, immutable input, initial lifecycle,
  and applicable child claim deadline;
- records an immutable required-child set for the wave;
- records the authoritative barrier policy and its validated parameters;
- derives and freezes the absolute wave deadline under Decision 6;
- writes the task-owned continuation checkpoint and active wave state;
- consumes the root's single delegation wave;
- transitions the parent to `waiting_for_children`, retires the parent turn,
  and makes children independently pull-claimable; and
- emits only non-secret, non-authoritative wake-up hints.

The wave becomes visible atomically or through an equivalently safe consistency
boundary. Partial acceptance is forbidden. Acceptance idempotency is bound to
`(parent_task_id, parent_assignment_id, parent_fencing_epoch, proposal_id)` and
the canonical proposal fingerprint. On retry, this replay record is evaluated
before rejecting the now-retired originating fence: an exact fingerprint match
returns the same accepted wave and children without mutation, while a changed
payload, mismatched origin tuple, or unrecognized retired fence fails closed.

The required-child set must contain at least one child and remain within the
authorized cumulative fan-out bound. Acceptance rejects an empty child set and
rejects any policy configuration that would make the barrier satisfied or
unsatisfiable at acceptance. Every accepted wave therefore enters
`waiting_for_children` strictly unresolved.

The versioned `TaskContinuation` references the authoritative `wave_id` and
immutable required-child set rather than embedding mutable barrier state. Upon
parent release it exposes the immutable canonical wave resolution needed by the
resumed turn. If cancellation terminates the parent, the continuation is
atomically closed and non-claimable.

The required-child set, barrier policy, accepted policy parameters, and frozen
wave deadline cannot be changed after acceptance. No child replenishment,
replacement, dynamic fan-out, second wave, or grandchild creation is permitted.

### 2. Authoritative barrier policies

The supported synchronization policies are `WAIT_ALL` and `PARTIAL`.
Barrier evaluation uses authoritative canonical child lifecycle and outcome
state only. It never interprets, ranks, combines, or evaluates child result
content. Structured aggregation and contributor attribution belong to Issue
#35.

#### WAIT_ALL

`WAIT_ALL` is satisfied when every required child has a canonical terminal
outcome. Success is not required for every child. Canonical `completed`,
`failed`, and `claim_timed_out` outcomes all count as terminal.

`recovery_required` is not terminal. It records an unresolved execution whose
previous authority has expired. While the wave remains active, its recovery is
governed by the existing explicit trusted recovery boundary. If the wave first
resolves through PARTIAL satisfaction or unsatisfiability, timeout, or
authorized cancellation, residual closure revokes recovery eligibility and
ends that child as `ended / barrier_resolved`.

#### PARTIAL

`PARTIAL` uses a platform-approved `required_success_fraction`. It is not a
hard-coded child count. The required number of successful children is:

```text
ceil(required_child_count * required_success_fraction)
```

The platform represents and evaluates the fraction with exact rational
semantics rather than floating-point percentage behavior. Acceptance requires
`0 < required_success_fraction <= 1` and `required_child_count >= 1`. Therefore
`1 <= required_success_count <= required_child_count`. The accepted value and
derived required-success count are validated and frozen with the wave. A policy
configuration that violates these bounds is rejected before wave acceptance.

Only canonical `completed` child outcomes count as successes for this barrier.
A `PARTIAL` wave is:

- `SATISFIED` when canonical successful completions reach the required count;
- `UNSATISFIABLE` when the required count can no longer possibly be reached,
  considering authoritative canonical child state and all still-unresolved
  children; or
- unresolved otherwise.

Failure, `claim_timed_out`, and other non-success terminal outcomes reduce the
remaining possible successes but do not themselves satisfy the success count.
`recovery_required` remains unresolved and potentially recoverable while the
wave remains active.

### 3. Immutable wave resolution

An accepted wave begins unresolved and has exactly one authoritative,
irreversible resolution transition. Its resolution is one of:

- `SATISFIED`;
- `UNSATISFIABLE`;
- `TIMED_OUT`; or
- `CANCELLED`.

Resolution is separate from individual child outcomes and does not rewrite
canonical child history. The operation that resolves the active wave atomically
records the immutable resolution, closes residual child authority as required,
and performs the corresponding parent lifecycle transition.

The parent transition mapping is:

| Wave resolution | Parent transition |
| --- | --- |
| `SATISFIED` | `resumable` |
| `UNSATISFIABLE` | `resumable` |
| `TIMED_OUT` | `resumable` |
| `CANCELLED` | terminal; no resumption |

A wave can cause at most one authoritative parent release. For `SATISFIED`,
`UNSATISFIABLE`, and `TIMED_OUT`, the released continuation exposes the
immutable canonical wave resolution. For `CANCELLED`, the parent terminates and
the continuation is atomically closed and non-claimable. Replayed or concurrent
operations that observe an already-resolved wave cannot resolve it again,
transition the parent again, create another continuation checkpoint, emit
another authoritative release, or independently authorize resumed execution. A
wake-up hint remains non-authoritative; resumed execution still requires
authenticated pull-claim and the current fencing rules.

### 4. Residual child closure

When `PARTIAL` resolves `SATISFIED` or `UNSATISFIABLE`, when the wave resolves
`TIMED_OUT`, or when authorized cancellation resolves it `CANCELLED`, the
platform authoritatively closes residual child contribution authority in the
same logical consistency decision as wave resolution and parent transition.

For a residual child whose contribution is no longer needed:

- pending or unclaimed work becomes non-claimable;
- an active claim loses its fencing/commit authority;
- a `recovery_required` child loses recovery eligibility; a concurrent or later
  recovery command must fail closed rather than restore claimability;
- the durable child record, lineage, assignment history, expiry/recovery
  history, and any already canonical outcome remain preserved; and
- a child without an existing canonical terminal outcome, including one in
  `recovery_required`, ends with the non-error lifecycle disposition
  `ended / barrier_resolved`.

`ended / barrier_resolved` is distinct from `completed`, `failed`,
`claim_timed_out`, and `recovery_required`. It states that platform-owned wave
closure made further contribution unnecessary or unauthorized; it does not
assert execution failure.

Fence invalidation removes future authoritative commit power. It does not
promise physical process termination, interruption of already-running code, or
reversal of external effects. Executors released from the bounded turn may
return to the existing pull-claim work model.

A late completion carrying an invalidated or stale fence cannot create or
alter a child outcome, alter the resolved wave, or release the parent. Once a
residual child is `ended / barrier_resolved`, later child claim-deadline or
execution-claim-expiry evaluation cannot change it to `claim_timed_out` or
`recovery_required`. Those timer operations must atomically verify that both
the child contribution authority and parent wave remain active.

Trusted recovery must likewise atomically verify that the parent wave remains
active and unresolved. It fails closed after wave resolution. If a valid
recovery operation while the wave is active produces or enables a canonical
terminal transition, that transition re-evaluates the barrier under the same
first-authoritative-resolution-wins rule.

### 5. First authoritative resolution wins

There is no precedence hierarchy such as cancellation over timeout over
completion. Every resolving operation conditionally acts only while the wave
is active and unresolved. The first valid operation to commit the authoritative
resolution wins. The resolution is then immutable.

Resolving operations include:

- child completion or terminal projection causing `SATISFIED`;
- child terminal projection causing `UNSATISFIABLE`;
- authoritative wave-deadline evaluation causing `TIMED_OUT`; and
- authorized cancellation causing `CANCELLED`.

External wall-clock arrival times, HTTP timestamps, agent-reported timestamps,
or queue ordering do not establish precedence. The authoritative atomic
transition does.

Examples:

- cancellation commits while active, so the wave is `CANCELLED`; a later child
  completion is rejected by the invalidated fence;
- a qualifying completion commits `SATISFIED`; a later wave-cancellation
  request cannot rewrite it;
- deadline evaluation commits `TIMED_OUT`; a late completion cannot alter it;
- a qualifying completion commits first, so the closed wave cannot later time
  out; and
- concurrent final sibling completions may each establish their own canonical
  child outcome when still authorized, but at most one operation resolves and
  releases the parent.

If a wave is already `SATISFIED` and an authorized action later terminates the
resumable parent, that is a parent-task lifecycle action. It does not rewrite
the historical wave resolution.

### 6. Deterministic bounded wave-deadline policy

The platform derives the proposed wave duration through a trusted,
deterministic, versioned policy. Agents may supply bounded work-requirement
descriptors, but they do not choose authoritative deadlines.

Policy inputs may include:

- bounded work-complexity descriptors;
- the authorized immutable child set;
- planned or effective parallelism; and
- bounded uncertainty descriptors.

This ADR establishes the policy architecture, not specific weights, constants,
or durations. Calibration is deferred until evidence exists.

The policy is subject to configurable lower and upper lifecycle guardrails:

- a derived value below the minimum uses the minimum;
- a derived value within bounds is accepted; and
- a derived value above the maximum rejects the proposed wave before
  acceptance.

The platform does not clamp an over-limit proposal to the maximum and then
knowingly accept work it predicts cannot fit within the collaboration bound.

Upon acceptance, the platform records the policy identifier/version, validated
bounded inputs needed for reproducibility, derived duration, and absolute wave
deadline. The absolute deadline is frozen. It cannot be automatically
recomputed, extended by an agent, or continually moved in response to runtime
conditions.

### 7. Wave timeout

A wave may resolve `TIMED_OUT` only while its barrier is unresolved. When the
frozen authoritative deadline is reached while the wave remains active, the
trusted platform atomically resolves it `TIMED_OUT`, closes residual child
authority under Decision 4, and makes the parent `resumable` at most once.

There is no grace period for an almost-finished child. If the barrier resolves
first, the closed wave can never later time out. No combined state such as
`SATISFIED and TIMED_OUT` exists.

Wave timeout neither fabricates child outcomes nor rewrites already canonical
ones. It is distinct from child `claim_timed_out` and acknowledged execution
claim expiry.

Deadline evaluation stays within the established trusted Orchestrator state
boundary. This ADR does not introduce a completion stream or projector worker.
Whether evaluation is invoked through existing task-scoped claim,
status/recovery, or an equivalently bounded trusted operation is an
implementation choice, provided the frozen deadline and first-resolution-wins
semantics hold.

### 8. Authorized cancellation

Cancellation requires an authorized human action or an approved trusted
platform/orchestrator policy. An agent request alone cannot establish
cancellation authority.

While the wave is active, authorized cancellation atomically:

- resolves the wave `CANCELLED`;
- closes pending and active residual children under Decision 4;
- preserves already canonical child outcomes as durable history;
- terminates the collaborative parent task; and
- creates no parent resumption or subsequent release.

Cancellation is terminal for the collaborative task. The wave/task cannot be
reopened or restarted. Cancellation creates no new deadline, replacement
children, second delegation wave, retry, or replanning authority.

A future trusted orchestrator policy may decide to stop an activity to improve
overall success, but any subsequent retry, replanning, or new-wave behavior
requires separate future architecture. It is not part of Issue #34.

### 9. Canonical state and content separation

Barrier policy evaluates only canonical state:

- immutable required-child membership;
- canonical child lifecycle and terminal outcomes;
- active claim/fence state where closure is required;
- frozen policy parameters and deadline; and
- immutable wave resolution.

It must not inspect child `TaskResult` content, operational events, trace data,
model confidence, inferred quality, or contributor ranking. Issue #35 may later
load the durable authoritative child outcomes and results for structured
aggregation after a resumable parent is validly claimed.

Operational events and Issue #36 tracing remain observational and cannot
resolve a wave or release a parent.

### 10. Required concurrency and idempotency invariants

Implementation must preserve these invariants:

- every accepted wave has `required_child_count >= 1` within its authorized
  cumulative fan-out bound and enters `waiting_for_children` unresolved;
- PARTIAL requires exact rational `0 < required_success_fraction <= 1`, so its
  derived required-success count is between 1 and the required-child count;
- one accepted wave has one immutable required-child set;
- one active wave has at most one immutable resolution;
- one wave causes at most one authoritative parent transition/release;
- every child has at most one valid active claim;
- a stale or invalidated child fence cannot commit;
- canonical child outcomes are immutable and idempotently replayable;
- wave resolution, residual closure, and the corresponding parent transition
  are one atomic or equivalently safe consistency decision;
- concurrent sibling completions cannot release the parent twice;
- completion versus timeout and completion versus cancellation have exactly
  one authoritative winner;
- wave-acceptance replay validates its originating assignment/fence tuple and
  canonical proposal fingerprint before treating a retired fence as an exact
  accepted replay;
- a replay observes and returns the existing canonical decision without
  duplicating children, outcomes, hints, closure, checkpoints, or releases;
- pending children cannot be claimed after wave closure;
- active residual executors cannot commit after fence invalidation;
- closed residual children cannot later transition to `claim_timed_out` or
  `recovery_required`;
- recovery of a child in `recovery_required` atomically verifies that its wave
  remains active, fails closed after resolution, and re-evaluates the barrier
  if recovery produces or enables a canonical terminal transition;
- a waiting parent holds no executor slot;
- a resumable parent requires authenticated pull-claim and a current fence;
- no agent directly writes continuation, barrier, claim, outcome, resolution,
  or parent lifecycle authority;
- all authoritative keys for a root task, its continuation, active wave, and
  required children reside within one single-node atomic Redis consistency
  boundary for Issue #34;
- acceptance, child projection, barrier evaluation, recovery, timeout, or
  cancellation encountering missing required metadata, unsupported schema or
  policy versions, corrupt records, or inconsistent required-child membership
  fails closed without resolving the wave, mutating child outcomes, or
  releasing the parent; and
- external effects remain outside exactly-once guarantees.

### 11. Issue boundaries and exclusions

Issue #34 owns:

- atomic bounded multi-child wave acceptance;
- immutable required-child membership;
- concurrent independent sibling pull-claim and execution;
- `WAIT_ALL` and fraction-based `PARTIAL` barrier semantics;
- immutable wave resolution;
- dynamic deterministic bounded deadline derivation and frozen wave timeout;
- authorized cancellation;
- residual child closure and active-fence invalidation;
- first-authoritative-resolution-wins concurrency; and
- at-most-once parent release.

Issue #34 does not own:

- structured result aggregation or contributor attribution (Issue #35);
- observational cross-agent tracing (Issue #36);
- sibling-to-sibling `depends_on` edges;
- arbitrary execution DAGs;
- dynamic fan-out after wave acceptance;
- replacement or replenishment children;
- multiple delegation waves;
- retries or replanning;
- restart of a resolved, cancelled, or timed-out wave;
- grandchildren or depth greater than 1;
- learned deadline prediction;
- automatic rolling deadline extension;
- physical process termination guarantees; or
- Run 002 Increment 3.

## Required state/decision matrix

| Situation | WAIT_ALL | PARTIAL | Child/authority effect | Parent/wave effect |
| --- | --- | --- | --- | --- |
| Pending/unclaimed child | Unresolved | Unresolved unless threshold already reached or impossible | Claimable only while wave active and child deadline permits | None |
| Actively claimed child | Unresolved | Unresolved unless threshold already reached or impossible | Current fence remains authoritative only while wave active | None |
| Completed child | Terminal contribution | Successful contribution | Canonical outcome/result retained | Re-evaluate barrier |
| Failed child | Terminal contribution | Non-success; reduces possible successes | Canonical failure retained | WAIT_ALL progresses; PARTIAL may become unsatisfiable |
| `claim_timed_out` child | Terminal contribution | Non-success; reduces possible successes | Canonical outcome retained | Same barrier treatment as failed |
| `recovery_required` child | Not terminal while wave active | Unresolved and potentially recoverable | Old fence invalid; trusted recovery must atomically verify active wave; wave closure ends child and revokes recovery eligibility | No release solely from this state |
| All required children terminal | `SATISFIED` | Depends on success threshold | Retain outcomes | Resolve/release once for WAIT_ALL |
| Required successes reached | N/A | `SATISFIED` | Close residual authority | Parent `resumable` once |
| Required successes impossible | N/A | `UNSATISFIABLE` | Close residual authority | Parent `resumable` once |
| Frozen wave deadline wins | `TIMED_OUT` | `TIMED_OUT` | Close residual authority | Parent `resumable` once |
| Authorized cancellation wins | `CANCELLED` | `CANCELLED` | Close residual authority | Parent terminates; no resumption |
| Stale completion after resolution | No change | No change | Reject invalid/stale fence | Immutable resolution |
| Concurrent final siblings | First resolving commit wins | First resolving commit wins | Authorized canonical child outcomes may persist | At most one parent release |
| Completion versus timeout | First authoritative resolution wins | Same | Losing operation cannot rewrite | Exactly one resolution |
| Completion versus cancellation | First authoritative resolution wins | Same | Losing operation cannot rewrite | Exactly one resolution |
| Replay after resolution | Return/observe existing decision | Same | No writes beyond safe replay handling | No duplicate transition or hint |

## Consequences and tradeoffs

Benefits:

- Extends the accepted task-owned continuation model without parallel lifecycle
  authority or process-local coordination.
- Allows bounded sibling concurrency while retaining one wave, depth 1, and
  authoritative platform ownership.
- Exact fraction semantics avoid hard-coded child counts and floating-point
  ambiguity.
- First-resolution-wins provides one uniform concurrency rule for completion,
  timeout, and cancellation.
- Residual closure prevents unnecessary or stale sibling work from committing
  after the barrier no longer needs it.
- Frozen policy-derived deadlines bound collaboration without allowing agents
  to continually extend their own lifecycle.
- Durable child records and outcomes remain available for Issue #35 without
  forcing aggregation into barrier logic.

Costs and limitations:

- The authoritative store gains wave identity, immutable membership, barrier
  policy, policy parameters, deadline derivation evidence, resolution, and
  residual closure state.
- Existing single-child contracts and atomic scripts require versioned
  evolution, including continuation-to-wave binding and canonical resolution
  exposure on parent release.
- Multi-key resolution and closure increase the size and complexity of the
  single-node Redis consistency boundary.
- Active work may continue physically after its fence is invalidated, and
  external effects cannot be rolled back by barrier closure.
- `recovery_required` may leave a `WAIT_ALL` or still-viable `PARTIAL` wave
  unresolved until explicit recovery, wave timeout, or authorized
  cancellation.
- Deadline policy calibration remains evidence-driven future work; this ADR
  defines only deterministic bounded architecture.

## Alternatives considered

### A. Reuse single-child immediate parent release for every sibling

Each child terminal outcome independently makes the parent resumable.

Rejected. It would release the parent before the multi-child barrier is
satisfied and permit duplicate releases.

### B. Block the original parent executor until all children finish

Rejected. It violates ADR 0006 turn-boundary delegation and ADR 0007 task-owned
continuation, holds an executor slot, and is process-affine.

### C. Add a completion stream and projector worker

Rejected. ADR 0007 selected direct authenticated idempotent projection and
explicitly rejected the extra stream/projector boundary for this workflow.
Multi-child barrier projection can extend the existing trusted atomic boundary.

### D. Fixed required child count for PARTIAL

Rejected. A fixed number does not scale with authorized wave size and silently
changes meaning as fan-out changes. A platform-approved exact fraction provides
stable policy semantics.

### E. Clamp every derived deadline to a maximum and accept the wave

Rejected. If trusted policy predicts the proposal cannot fit inside the maximum
bounded lifecycle, acceptance would knowingly admit work outside policy. Such a
proposal is rejected before acceptance.

### F. Cancellation or timeout always overrides completion

Rejected. Timestamp-derived or type-based precedence introduces ambiguous race
rules. One atomic first-authoritative-resolution-wins rule is simpler, auditable,
and consistent with existing fenced transition precedent.

### G. Leave residual claimed children authoritative after PARTIAL resolution

Rejected. Their late completion could mutate a resolved wave or create unclear
contribution semantics. Closure must invalidate residual commit authority while
preserving durable history.

## Acceptance record

The architecture acceptance gate was completed on 2026-11-27:

1. independent Gemini 4 Neon High architecture/concurrency review completed;
2. review findings validated against live repository behavior;
3. one consolidated revision completed;
4. final whole-document consistency review completed; and
5. explicit human architecture acceptance recorded.

ADR 0011 is the governing architecture for Issue #34. Implementation
increments may now be planned within its accepted boundaries.
