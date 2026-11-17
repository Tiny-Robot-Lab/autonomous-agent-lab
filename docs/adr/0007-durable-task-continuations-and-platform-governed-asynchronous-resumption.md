# ADR 0007: Durable Task Continuations and Platform-Governed Asynchronous Resumption

Status: Accepted
Date accepted: 2026-11-16
Related issue: https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/33, https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/34, https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/35, https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/36
Supersedes: N/A

## Review status

Accepted by explicit human decision on 2026-11-16 after repository-grounded
prospective drafting, human architecture review, independent Gemini 4 Argon
High review, validation of Argon's findings against live main at `7e8bfc7`,
revision of the architecture, and final full-document consistency review.
ADR 0006 remains accepted and governing. ADR 0007 fills in ADR 0006's
deferred continuation, completion, and resumption details; it does not reopen
or replace ADR 0006's lineage, delegation-authority, turn-boundary, depth,
wave, or specialist-registration decisions.

### Independent review record

Gemini 4 Argon High independently reviewed the prospective ADR and
recommended revision before acceptance, identifying seven blocking findings.
Those findings were validated against live main at `7e8bfc7`; confirmed
findings were incorporated, and overstated findings were narrowed or rejected:

1. **B-1, proposer identity:** Partially confirmed. The authenticated API
   boundary is selected here; its deployed credential mechanism remains a
   mandatory pre-enablement gate.
2. **B-2, initial root assignment:** Confirmed as a design gap. This revision
   requires initial Research turns to use authenticated pull-claim and bind
   any delegation grant to that assignment.
3. **B-3, result writes and fencing:** Partially confirmed. The previous
   decision prohibited direct writes only for delegated children. This
   revision applies trusted fenced completion to all assignment-backed turns.
4. **B-4, API plus completion stream:** The redundancy concern is valid; the
   claim that the two boundaries are contradictory is not. This revision
   removes the stream and projector in favor of direct idempotent projection.
5. **B-5, pre-bound push recovery:** Partially confirmed. Pull-claim removes
   pre-bound delivery; acknowledged expired execution claims still require
   explicit recovery and are not automatically retried.
6. **B-6, child input and artifact validation:** Confirmed as a design gap.
   This revision adds immutable input/artifact verification to atomic
   acceptance.
7. **B-7, missing workers:** The absence of Orchestrator work processors was
   confirmed, but a requirement for four distinct workers was not. This
   revision assigns claim-deadline evaluation to Orchestrator claim and
   recovery/status operations.

The high-priority validation found that candidate availability before
acceptance is unnecessary under pull-claim. An accepted pending child with no
eligible claim before its deadline reaches `claim_timed_out`. Immutable,
versioned TaskInput wording was already sound.
Explicit continuation transitions, initial root authority issuance, and
issue-#33 authoritative sources for issue-#36 identifiers needed
clarification. The 2–4 KiB recommendation was unsupported by measurements;
16 KiB UTF-8 remains a provisional, test-enforced default. TaskInput and
artifact retention, plus turn identity for fallback history, are now
specified.

The final architecture selects authenticated pull-claim, non-secret wake-up
hints, and direct idempotent atomic platform projection. Argon informed the
revision but did not make the acceptance decision. Acceptance does not mean
the implementation already exists. Executable delegation remains blocked
until the accepted pre-enablement requirements are implemented and
validated, including unique runtime authentication, protected assignment
delivery, least-privilege Redis permissions, fenced canonical completion,
exact schemas, lease/deadline parameters, and deployment persistence/recovery
configuration.

## Context

[ADR 0006](0006-platform-governed-task-lineage-and-bounded-delegation-authority.md)
decides that agents propose delegation, the platform authorizes and dispatches
it, agents yield at a turn boundary, children complete asynchronously, and a
parent task—not a particular agent process—owns later continuation. It leaves
the concrete continuation and resumption mechanism open and blocks executable
delegation until proposer binding and authority composition are resolved.

The current implementation provides useful foundations but not that
lifecycle:

- `TaskRequested` is a versioned task-request identity/input envelope;
  `TaskResult` is a successful-output envelope. Neither is an authenticated
  assignment, continuation, claim, or authority proof; neither represents a
  resumable execution continuation or canonical failed terminal outcome.
  The authoritative store does not retain the original prompt/task input
  independently of the queue.
- `AuthoritativeTaskStore` and `RedisGateway` persist task status, lineage, and
  parent membership. `RegisterChildTask` validates parent/depth and delegates
  child existence to one Redis transaction, but does not atomically update a
  parent continuation, dependency, wait state, or delegation-wave allowance.
- Orchestrator dispatch records routing and writes the advertised dispatch
  target. Redis list dispatch uses `RPUSH`; agents consume with destructive
  `BLPOP`. There is no authenticated claim API for initial, child, or resumed
  work.
- Research Agent and Analysis Agent Redis adapters save successful results
  directly to `task.result:{task_id}`. Their processors publish lifecycle
  events, including an error event on ordinary execution failure, but no
  canonical failed task outcome is persisted. `agent.events` is bounded and
  best-effort operational evidence under
  [ADR 0005](0005-canonical-task-scoped-inference-fallback-history.md), not
  authoritative completion state.
- Candidate discovery filters registered agent metadata by capability and
  liveness; selection and dispatch-target use are platform responsibilities.
  Analysis Agent advertises `analysis` and uses a deterministic analyzer on
  supplied material. It does not need retrieval, provider credentials, tools,
  or external network access for the initial collaboration edge.
- Runtime `agent_id` and advertised metadata do not authenticate the process
  emitting a proposal. Redis queue delivery and self-asserted event identity
  are not proposer authorization.
- The shared Redis task queues do not prove which runtime consumed a queue
  item. A `BLPOP` result is delivery evidence, not authenticated runtime
  identity or a task-assignment claim.

Issue #33 currently describes agents awaiting or subscribing to child
completion and asks that failures be reported to a parent agent. This ADR
proposes the task-owned, non-blocking interpretation required by ADR 0006:
the parent task becomes resumable and a later eligible executor reads the
canonical child outcome. The issue wording and acceptance criteria must be
aligned after this ADR is accepted and before implementation; this ADR does
not modify GitHub.

## Decision

### 1. Task-owned execution and continuation

The durable task is the owner of its goal, lifecycle, dependencies, and
continuation. An agent process is a replaceable executor of a bounded task
turn, not the owner of the collaborative workflow. A waiting parent occupies
no agent execution slot. After the child reaches a canonical terminal outcome,
the platform makes the parent task resumable; any eligible Research Agent may
claim and continue it.

Continuation is capability-targeted (`research` for this scenario), never
bound to an agent ID. The identity of the executor that proposed, executed, or
resumed a turn may be retained as sanitized audit/telemetry metadata, but it
does not determine ownership or completion routing.

### 2. Separate, bounded continuation record

Introduce a provider-neutral, versioned, JSON-compatible `TaskContinuation`
record in authoritative task state. It is distinct from `TaskRequested` and
`TaskResult`. `TaskRequested` retains its task-request identity/input
envelope purpose, and `TaskResult` retains its successful-output envelope
purpose. Neither is an authenticated assignment, continuation, claim, or
authority proof. `TaskAssignment` controls execution; the trusted fenced
platform completion boundary canonically accepts `TaskResult`. Agents no
longer canonically establish results through direct Redis writes. The
continuation record contains:

- `parent_task_id`;
- a reference to the immutable task input and its version;
- `continuation_capability` (initially `research`);
- a bounded, schema-defined `stage` identifying the application resumption
  entry point (initially `incorporate_analysis_child`);
- optional small schema-defined control context and references to
  stage-required artifacts;
- required child task IDs (exactly one for issue #33);
- a monotonically increasing `continuation_version`; and
- platform creation/update timestamps.

Task lifecycle status separately represents `waiting_for_children`,
`resumable`, and `working`; these are not continuation stages.

The record is a small checkpoint, not a transcript or serialized runtime.
Persist only the application fields needed to resume the named stage. Do not
serialize Python frames, arbitrary objects, provider clients, credentials,
full prompts, unrestricted model context, or child output into the
continuation. It references the immutable task input and any stage-required
artifacts rather than duplicating their contents. Large results and material
remain in their task-scoped artifact/result storage and are loaded by
reference. The size limit is configured and enforced by tests. A provisional
initial implementation default is 16 KiB UTF-8, not a permanent
interoperability guarantee; larger material must use artifact references.
Each stage also has a schema-defined allowlist of fields.

### Durable task input and stage artifacts

Every resumable task retains durable access to an immutable, versioned
`TaskInput` record or artifact independently of queue payloads and agent
processes. It includes the original task input needed for execution, such as
the submitted prompt, and may reference other immutable task-scoped artifacts.
The record is keyed by task ID and input version; updates create a new version
rather than mutating an existing input. A distinct version can be selected
only through an explicit authoritative operation; versions are write-once.

The queue message is a delivery mechanism only; consuming it does not consume
the task's authoritative input.

Intermediate Research evidence or material needed across turns is stored as
an immutable, versioned, task-scoped artifact with provenance identifying its
producing task/turn and artifact/schema version. The parent continuation
references it, and the child `TaskInput` references the same artifact rather
than copying its contents into lineage or control state. Artifact access is
scoped by task and the authorized delegation grant. Lineage records task
identity and relationships only; it is not payload storage.

### 3. Parent and child lifecycle

Task lifecycle is authoritative task state and is separate from agent
execution state. The minimum lifecycle for this increment is:

| Parent task state | Meaning |
| --- | --- |
| `pending` | Accepted for initial execution and claimable. |
| `working` | An authenticated executor holds the current fenced task-turn claim. |
| `waiting_for_children` | Delegation was accepted; the continuation and dependency are durable. No executor is held. |
| `resumable` | Every required child has a canonical terminal outcome; an eligible executor may claim the continuation. |
| `recovery_required` | An acknowledged execution claim expired or was abandoned. Its fence is invalidated; a new execution claim requires an explicit recovery decision. |
| `completed` | The parent goal completed. |
| `failed` | The parent goal failed with a canonical safe terminal outcome. |

| Child task state | Meaning |
| --- | --- |
| `pending` | Child exists authoritatively and is claimable; it need not have a reserved executor. |
| `working` | An authenticated specialist holds the current fenced task claim. |
| `recovery_required` | An acknowledged child execution claim expired or was abandoned; its fence is invalidated. |
| `completed` | Canonical successful terminal outcome and normal result are durable. |
| `failed` | Canonical safe execution-failure outcome is durable. |
| `claim_timed_out` | No eligible Analysis runtime claimed the pending child before its bounded claim deadline; this is terminal for the single-child dependency and distinct from execution failure, assignment-delivery failure, or temporary executor unavailability before the deadline. |

The platform transitions task state; agent lifecycle events do not. A
child's canonical terminal completion or failure satisfies its parent
dependency. Under the issue-#33 one-child policy, that makes the parent
`resumable`, including when the child failed or its claim timed out. Root and
resumed work remains claimable when no eligible executor is currently
available. No synchronous wait or process-local callback is used.

### 4. Continuation stages and versions

The issue-#33 continuation has one stage, `incorporate_analysis_child`.
Continuation versions identify immutable checkpoints; ordinary scheduling
state transitions do not create a new checkpoint.

| Operation | Task transition | Continuation effect |
| --- | --- | --- |
| Delegation acceptance | Parent becomes `waiting_for_children`. | Write checkpoint version 1 with stage `incorporate_analysis_child` and the required child ID. |
| Child terminal projection | Parent becomes `resumable`. | Keep version 1; do not increment it. |
| Parent claim | Parent becomes `working` only if the expected version is 1. | Keep version 1. |
| Parent completion | Parent becomes `completed` or `failed`. | Retain version 1 as immutable historical state; task terminal state closes its use. Do not increment it. |

A future change that writes a distinct checkpoint creates a new immutable
version through an explicit authoritative operation. It must not mutate
version 1 in place. The claim and each state transition use the expected
continuation version and fencing epoch where applicable.

### 5. Canonical child terminal outcomes

Successful completion is authoritative task state plus the normal
`TaskResult` (or an authoritative reference to it). Failure is an equally
durable terminal record containing the child task ID, a stable non-sensitive
failure code, and terminal timestamp. The initial failure codes distinguish
at least execution failure and `claim_timed_out`. `claim_timed_out` means
that an accepted pending child received no eligible Analysis claim before its
deadline. It is not an execution failure, an assignment-delivery failure,
or an ordinary temporarily unavailable executor before that deadline. A
fixed safe summary may be included where needed; raw exceptions,
provider/network responses, prompts, credentials, and arbitrary agent text
are excluded.

Neither a missing result nor a bounded `agent.events` entry proves success,
failure, or completion. The parent dependency is satisfied only by the
platform's canonical terminal projection. A parent loads the child's
authoritative terminal outcome and, on success, its normal result.

### 6. Non-secret wake-up hints

A claimable initial root, delegated child, or resumable parent may cause a
lightweight hint to be written to the matching capability queue. A hint
contains only identifiers such as task ID, capability, execution kind, and
expected continuation version where applicable. It contains no assignment
proof, credential, prompt, TaskInput, artifact payload, or authority grant.

A hint grants no authority and is not evidence that a runtime owns a task. It
may be duplicated, lost, or stale without changing authoritative task state.
An agent acts on a hint only by calling the authenticated Orchestrator claim
operation; it may instead poll that operation without a hint. A stale or
failed hint does not move a task to `recovery_required`.

### 7. Authenticated pull-claim and TaskAssignment

All assignment-backed execution kinds use one platform claim model:
initial root, delegated child, and resumed continuation. A live agent
authenticates to the Orchestrator as a unique runtime principal and requests
claimable work for an authorized capability. Agent registration, advertised
capability, `agent_id`, and a wake-up hint aid discovery only; they do not
authenticate the caller.

The claim operation atomically validates that:

- the task is `pending` or `resumable`;
- the authenticated principal is authorized and capability-compatible;
- the expected continuation version matches when the task has a continuation;
  and
- no valid active claim exists.

Each claim request carries a stable request ID scoped to the authenticated
principal. Retrying an unacknowledged request with the same ID returns the
same assignment rather than claiming another task.

The platform creates an assignment ID, claim ID, fencing epoch, validity
window, and applicable delegation grant; transitions the task to `working`;
records the assignment/claim and routing decision IDs with the authenticated
runtime identity, including the registered executor `agent_id` mapped from
that principal rather than asserted by the caller; and returns a private,
versioned `TaskAssignment` over the authenticated response. The assignment
semantically binds:

- assignment ID, task ID, and authenticated runtime principal;
- claim ID and execution kind (`initial`, `delegated_child`, or
  `resumed_continuation`);
- continuation version, when applicable, and fencing epoch;
- assigned capability and bounded delegation grant, when authorized;
- validity; and
- immutable TaskInput, continuation, and required-artifact references and
  versions.

Proof or credential material is detached secret authorization data, not an
ordinary JSON field likely to be logged. The platform-minted assignment proof
is delivered separately and bound to the assignment, principal, task,
continuation version, fencing epoch, capability, and grant.
Every proposal, artifact access, and completion validates the assignment,
principal, fence, task, version, and command identity. The assignment is
returned only over the authenticated channel, never through a shared Redis
queue, `TaskRequested`, public metadata, events, routing decisions, or logs.

The runtime acknowledges an assignment before starting execution. If
delivery is unacknowledged, the same principal may retrieve the same
assignment; on delivery-lease expiry the platform invalidates its fence and
returns the task to its prior claimable state. This is assignment-delivery
recovery, not task re-execution. After acknowledgement, claim expiration
follows Decision 11.

### 8. Trust boundary and initial root authority

The selected authoritative claim, proposal, artifact-access, and completion
boundary is an authenticated Orchestrator API/channel. Every runtime has a
unique principal and credential; a credential shared by every instance of a
service is insufficient for runtime identity. Credentials are provisioned
outside ordinary task payloads, protected in transit, revocable and
rotatable, and excluded from logs, telemetry, and events. Their exact
cryptographic format is implementation-specific, but the deployed mechanism
must be selected, threat-reviewed, and validated before executable
delegation is enabled.

The current shared Redis connection/topology, registration records,
self-asserted `agent_id`, and queue access do not authenticate a runtime.
Agent Redis permissions must not allow bypass writes to authoritative task,
result, continuation, artifact, assignment, claim, or terminal-outcome state.
Telemetry and non-authoritative registration access do not grant those
permissions.

The assigned runtime stages and seals Research artifacts through this
authenticated API. The platform validates the assignment and stamps
task/turn, schema, version, and provenance metadata; service-local Redis
writes cannot create an authoritative artifact.

At root ingestion the platform persists a default root-authority record
bound to the root task. It records the baseline authority and delegation
policy context but contains no secret proof. The default does not itself
authorize delegation. Before an initial Research turn can propose, policy
may issue the narrow one-turn `research -> analysis`, one-child,
one-wave grant only when the root and selected Research execution are
eligible. The grant is bound to that claimed assignment and derived from
the root authority, explicit platform policy, and specialist configuration.
The authority record tracks issuance and consumption; secret proof material
is returned only through the authenticated assignment response. If the
grant cannot be established, the assignment has no delegation authority and
the proposal is rejected. `TaskRequested` never carries the grant or proof.

Resolve ADR 0006 item 19 narrowly for issue #33:

- the only initial delegation edge is `research -> analysis`;
- exactly one child may be accepted for the root's one delegation wave;
- the Analysis Agent performs deterministic processing of supplied material
  only; and
- the child's effective authority is the intersection of the originating
  task's authorized grant, explicit platform delegation policy, and
  specialist configuration. It is never the union of those scopes.

Unknown authority fails closed. Registration/capability advertisement is
discovery evidence, not authorization. The child receives no retrieval,
provider credentials, tool access, or external network authority through
delegation.

### 9. Atomic delegation acceptance and child claimability

Acceptance does not require reserving or selecting an eligible Analysis
runtime. A proposal contains bounded child intent, proposal ID, and
references to the sealed Research artifact and required input; it contains
no child ID, runtime principal, queue name, or dispatch target.

The platform verifies the authenticated parent assignment, principal,
fencing epoch, expected continuation version, proposal ID, one-turn grant,
single-child/depth-one policy, and unconsumed wave. It verifies that the
referenced Research artifact exists, is sealed and immutable, has the
expected schema, version, and provenance, and is authorized for the child.
The artifact must remain pinned while the child or parent continuation can
reference it. The platform creates an immutable child `TaskInput` referencing
that artifact or verifies the exact immutable input within the same
consistency boundary.

One atomic or equivalently safe compare-and-set boundary then:

1. verifies the parent assignment/fence and the expected continuation
   version;
2. validates and consumes the one-turn delegation grant and proposal
   idempotency key, then records the proposal and delegation decision IDs;
3. validates and pins the immutable artifact and child `TaskInput`
   references;
4. mints one child ID and persists child state, forward lineage, and parent
   membership;
5. writes parent continuation version 1 at stage
   `incorporate_analysis_child` and its required-child dependency;
6. consumes the root's one delegation wave and moves the parent to
   `waiting_for_children`; and
7. makes the child `pending` and claimable and writes a non-secret Analysis
   wake-up hint or equivalent marker.

Child existence and the parent wait state become visible together. An exact
retry of the same proposal under the same valid assignment returns the same
accepted child; it cannot consume a second grant or create another child.
The hint is not authoritative. If it is lost, the child remains claimable.

### 10. Candidate availability and claim deadline

Candidate availability is not a prerequisite for delegation acceptance.
The accepted child remains `pending` until an eligible authenticated
Analysis runtime claims it. Root and resumed-parent work may likewise remain
claimable when no eligible executor is currently available.

Acceptance records a bounded child claim deadline. The Orchestrator's
authenticated capability-claim operation evaluates due deadlines for that
capability before returning work. A task-status/recovery operation may invoke
the same bounded recovery use case. No separate dispatcher, resumable-work,
or lease-expiry worker is required. If no eligible child runtime claims
before the deadline, the trusted platform atomically records canonical
`claim_timed_out`, satisfies the parent dependency, makes the parent
`resumable`, and writes one non-secret parent hint. An expired deadline is
materialized on the next claim or recovery/status operation; it does not
authorize task execution. A child claimed before the deadline is governed by
its execution claim and Decision 11.

`claim_timed_out` is distinct from execution failure because no runtime
executed the child; distinct from assignment-delivery failure because no
assignment was claimed; and distinct from temporary executor unavailability
before the deadline, during which the child remains pending and claimable.

### 11. Universal fenced completion and canonical projection

Every assignment-backed execution turn uses trusted platform completion:
initial root, general worker, Research root, delegated Analysis child, and
resumed parent. Agent adapters may not establish canonical completion by
directly writing `task.result:{task_id}`. Existing direct-result behavior
must be migrated incrementally and cannot remain a permanent bypass.

The runtime submits completion or safe failure through the authenticated
Orchestrator API with a stable command ID and assignment/fence proof. For
issue #33, the request carries one bounded canonical `TaskResult` on success,
or a stable safe failure code on failure. The Analysis and parent results
are inline; issue #33 does not support a second staged-result-reference
completion mode. The request size and result bounds are configured and
test-enforced.

Before replying, the API validates the authenticated principal, assignment,
fence, task, continuation version, and command ID and directly performs one
idempotent atomic Redis projection. As applicable, the projection establishes:

- canonical `TaskResult` on success;
- canonical safe terminal outcome and task terminal status;
- child dependency satisfaction;
- the parent transition to `resumable`; and
- a canonical outcome ID associated with the stable completion command ID;
- exactly one non-secret parent wake-up hint or equivalent claimable-work
  marker.

A retry after a lost response uses the same command ID and returns the
recorded canonical outcome. A conflicting terminal command fails closed and
cannot overwrite the first outcome. The API replies only after projection
commits. This ADR does not use a task-completion stream, consumer group, or
projector worker.

Completion-command durability is bounded by configured Redis persistence
and recovery semantics; it is not an unconditional guarantee against Redis
data loss. In-memory queues, agent events, and uncommitted API requests are
not canonical completion. `agent.events` remains bounded, best-effort
observability under ADR 0005.

### 12. Claim expiration and recovery

An unacknowledged assignment delivery may be safely released to its prior
claimable state after its delivery lease expires; the old fence is
invalidated. This is assignment-delivery recovery, not task re-execution.
Loss or staleness of a non-secret wake-up hint does not create
`recovery_required`.

An acknowledged execution claim that expires or is abandoned becomes
observable as `recovery_required`; its proof and fence are invalidated, so a
stale executor cannot commit. The Orchestrator claim/status/recovery
operations detect and record the expired state. A separate explicit trusted
`ResolveExpiredClaim` operation verifies the task, claim, and expired epoch
and records a recovery-decision ID. It may clear `recovery_required` only
under an explicit recovery disposition; any subsequent assignment uses a
new fence. Issue #33 does not define that disposition, automatically
re-execute a task, retry a provider, or redispatch an acknowledged execution
claim. Retry policy remains outside this issue.

### 13. Duplicate semantics and Redis topology

- **Delegation proposal:** A unique proposal ID is scoped to the parent
  assignment/turn. Replaying the same accepted proposal returns its original
  child and acceptance; it never mints another child. A different proposal
  after the wave is consumed is rejected.
- **Child claim:** A task can have at most one valid active claim. Repeated
  claim requests by the same principal can retrieve the same unacknowledged
  assignment; a stale hint cannot claim or execute a task.
- **Completion:** A stable completion command ID and assignment identify a
  completion. Identical retries return the recorded outcome. A conflicting
  terminal outcome cannot overwrite it.
- **Parent unblock:** Child terminal projection, dependency satisfaction,
  parent `resumable` state, and the single claimable-work marker are one
  conditional atomic projection. Retries do not create duplicate work or
  continuation versions.
- **Fencing:** A stale assignment, continuation version, claim, or epoch
  cannot propose, access artifacts, or commit completion.

Issue #33 atomic scripts/CAS are scoped to the bundled single-node Redis
deployment. Do not introduce Redis Cluster hash-tag design without a
deployment requirement. Any alternative Redis topology requires separate
validation of multi-key atomicity. AOF and a named volume do not provide an
unconditional no-loss guarantee; the accepted deployment must state its
persistence and recovery configuration.

### 14. Issue boundaries and exclusions

**Issue #33** is limited to one Research parent, one Analysis child, one
delegation wave, depth one, non-blocking suspend/yield, durable input and
continuation, atomic delegation acceptance, canonical completion/failure,
parent resumability, authenticated pull-claim, and parent resumption with
child outcome incorporation. The parent remains owner of the overall goal.

**Issue #34** owns multiple children, concurrent execution, wait-all/partial/
timeout/cancellation policies, and their synchronization. It uses this
task-owned continuation and completion architecture; it does not create a
second continuation model.

**Issue #35** owns structured aggregation of authoritative child outcomes,
task/contributor attribution, evidence/source preservation, and
partial-failure representation. It must not reconstruct results from
best-effort operational events.

**Issue #36** owns observational tracing through identifiers produced by
authoritative issue-#33 records and safe operational emitters: root task ID;
parent/child task IDs; proposal ID; delegation decision ID; routing decision
ID; assignment ID; claim ID; fencing epoch; continuation version;
completion command/outcome ID; and executor agent ID plus authenticated
runtime principal per turn. Secret proof and credential data must never be
traced. Tracing may correlate and display these values but can never become
authoritative lineage, task state, completion, dependency, continuation, or
claim state.

Inference fallback history remains task-scoped and is additionally grouped
by assignment/turn ID and continuation version where applicable. Attempt
indices may restart within a turn and must not be deduplicated across turns.
The task-level projection may aggregate the per-turn histories.

This ADR does not introduce grandchildren, multiple delegation waves,
arbitrary execution DAGs, sequential plan steps, conditional planning,
retry/replanning policy, dynamic specialist creation, arbitrary
capability-to-capability delegation, process-local synchronous waits,
continuation tied to one agent ID, `agent.events` as authoritative
completion, serialization of Python runtime state, cross-task memory or
v0.8 memory semantics, general v0.7 planning abstractions, automatic public
user routing to Analysis, or parallel execution/aggregation inside issue
#33.

Implementation remains within functional issue #33 but must be delivered
through bounded sub-issues and PR gates, without assigning issue numbers
here:

1. TaskInput, continuation, artifact, and root-authority records/storage.
2. Authenticated runtime principal, pull-claim, TaskAssignment, fencing, and
   narrow root grant.
3. Atomic Research -> Analysis delegation acceptance and child claimability.
4. Universal fenced completion API/projection and canonical outcomes.
5. Resumed-parent claim, Research continuation, expiry/recovery handling,
   and turn-aware telemetry.
6. End-to-end Research -> Analysis -> Research validation.

### Required invariants

- A waiting parent occupies no agent execution slot.
- A resumable task recovers its original immutable TaskInput and all
  stage-required artifacts without relying on consumed messages or process
  memory.
- TaskInput versions and sealed artifacts are immutable; updates create new
  versions through explicit authoritative operations.
- Referenced TaskInput and artifacts cannot expire while any nonterminal
  task, continuation, dependency, assignment, canonical outcome, or required
  result can reference them. Post-terminal deletion is configurable and
  reference-safe.
- Intermediate Research material is immutable, versioned, task-scoped, and
  provenance-bearing; continuation and child input reference it rather than
  copying it into lineage or control state.
- Continuation control state is schema-defined and bounded; larger material
  uses artifact references. The 16 KiB UTF-8 limit is a provisional initial
  default, not an interoperability guarantee.
- Non-secret hints grant no authority and do not change authoritative task
  state.
- Every assignment is bound to a unique authenticated runtime principal and
  current fencing epoch.
- No agent can directly write canonical task result, continuation, claim,
  artifact, or terminal-outcome state.
- At most one valid executor holds an active task-turn claim.
- A stale executor whose assignment proof or fencing epoch is invalidated
  can never commit task state or completion.
- Canonical completion and parent unblocking are idempotent and atomic.
- A terminal child outcome, including `claim_timed_out`, makes the parent
  resumable with that outcome.
- Issue #33 permits one child and one delegation wave per root.
- Operational events and issue-#36 traces are observational, not
  authoritative.
- Task-scoped inference history carries assignment/turn identity and
  continuation version where applicable, so attempt indices from separate
  turns do not collide.

## Deferred questions

These are implementation details, not permission to weaken the decisions
above. They must be resolved and validated before executable delegation is
enabled:

- Which unique per-runtime credential and protected API/channel mechanism
  will be deployed, and how will it be threat-reviewed, rotated, and revoked?
- What delivery-acknowledgement lease, acknowledged execution-claim lease,
  and bounded child claim deadline are appropriate?
- What exact per-stage continuation field schemas, safe failure-code set, and
  bounded inline TaskResult size does issue #33 require?
- What Redis persistence and recovery configuration is required for the
  selected deployment?
- What post-terminal retention period is appropriate, subject to the
  reference-safety invariant?
- Which precise inference telemetry contract changes carry assignment/turn
  identity without exposing secrets or sensitive event detail?

## Consequences and tradeoffs

Benefits:

- Durable task state, rather than a live process, owns work across
  suspension, completion, and resumption.
- Authenticated pull-claim unifies initial, child, and resumed turns while
  keeping assignment authority out of shared queues.
- Non-secret hints reduce polling but are safe to lose, duplicate, or replay.
- Direct idempotent projection centralizes canonical results, outcomes,
  dependency satisfaction, and parent resumption without a completion
  consumer group or projector worker.
- Versioned compare-and-set claims and fencing make stale execution explicit;
  external effects are not exactly-once.
- The initial Research-to-Analysis authority edge is narrow, deterministic,
  and compatible with the platform-owned routing and authorization boundary
  in ADR 0006.
- The same task-owned model supports later bounded fan-in and aggregation
  without pulling arbitrary dependency graphs or general planning into v0.6.

Costs and limitations:

- The Orchestrator becomes the authenticated claim, proposal, artifact, and
  completion boundary and must be available to runtimes at those points.
- Unique runtime credentials, claim acknowledgement, API idempotency, and
  least-privilege Redis permissions require implementation and operational
  validation.
- The platform gains authoritative continuation, dependency, terminal
  outcome, claim/lease, and root-authority state plus conditional atomic
  operations.
- Pull-claim removes push-delivery workers but does not eliminate claim
  deadline evaluation or explicit recovery for acknowledged expired claims.
- Direct API completion depends on the caller retrying the same command ID
  after a lost response; Redis persistence remains deployment-bounded.
- Redis queue consumption currently uses destructive `BLPOP`; hints are
  non-authoritative and do not make current queues reliable task delivery.
- A lease and fencing token prevent concurrent authoritative owners but
  cannot guarantee exactly-once external effects if an executor crashes
  after performing them. This ADR adds no general execution retry policy.
- Continuation schemas and their size bounds must be explicit, versioned, and
  reviewed as they evolve; arbitrary model context is not a safe default.
- Candidate availability and Redis behavior at collaboration scale remain
  unmeasured.

## Alternatives considered

### 1. Blocking in-process wait by the original parent agent

The original Research Agent waits for the child and then continues the parent
in the same call stack.

- **Availability/correctness:** Loses progress on process restart and can
  deadlock a serial agent that must execute the child itself. It contradicts
  ADR 0006's turn-boundary yield.
- **Authority/duplicates:** Keeps continuation in an unversioned process and
  gives no platform claim boundary; timeout/reconnect can duplicate work.
- **Crash recovery/complexity/Redis:** Simple initially, but no durable
  recovery; it misuses the task queue and holds an execution slot while
  waiting.
- **Issues and v0.7:** Couples child completion to the parent runtime, scales
  poorly to #34/#35, and encourages synchronous or sequential planning.
- **Disposition:** Rejected.

### 2. Process-affine callback to the original parent agent instance

The child completion is routed to the Research Agent instance that proposed
it.

- **Availability/correctness:** The callback target may be stopped, restarted,
  or processing other work. It is not the authoritative parent task.
- **Authority/duplicates:** Requires trustworthy process addressing and
  callback authentication not present today; callback retries need their own
  idempotency and stale-instance rules.
- **Crash recovery/complexity/Redis:** Requires a process registry and durable
  callback mailbox in addition to parent state; an in-memory callback is
  lost on restart.
- **Issues and v0.7:** Prevents any eligible Research Agent from resuming,
  complicates #34/#35, and preserves process-local orchestration.
- **Disposition:** Rejected.

### 3. Durable task-owned suspension and resumption

The parent stores a versioned continuation and dependency, yields, and later
is atomically claimed by an authenticated capability-matched executor. This
is the selected workflow model; assignment delivery and completion use the
claim and direct-projection decisions above.

- **Availability/correctness:** Survives process replacement; conditional
  writes, fencing, canonical outcomes, and idempotency protect authoritative
  state. External effects are not exactly-once.
- **Authority/duplicates:** Platform owns acceptance, task identity, routing,
  claim, and completion projection; requires a unique authenticated runtime
  principal, fencing, and explicit duplicate semantics.
- **Crash recovery/complexity/Redis:** Best fit with Redis authoritative
  status/lineage and transaction precedent, but adds state transitions,
  pull-claims, leases, and atomic projection.
- **Issues and v0.7:** Supports #34/#35 without a process-specific callback
  or general planning graph; remains a bounded v0.6 barrier.
- **Disposition:** Selected.

### 4. Service-local agent Redis adapters directly update parent state

Research and Analysis adapters write child results and directly mutate parent
status/dependencies.

- **Availability/correctness:** Uses existing direct Redis access but creates
  partial-write windows across result, child terminal state, dependency, and
  parent continuation.
- **Authority/duplicates:** Gives service-local adapters parent lifecycle
  authority and requires them to reimplement policy, authorization, and
  idempotency. Retries can overwrite outcomes or unblock twice.
- **Crash recovery/complexity/Redis:** Avoids a worker at first, but spreads
  Redis key semantics and transaction logic across services. Redis
  transactions can help only if the service owns all affected keys and
  validates the authoritative transition.
- **Issues and v0.7:** Makes #34/#35 cross-service fan-in harder and risks
  silently introducing graph policy outside the platform.
- **Disposition:** Rejected.

### 5. Authenticated Orchestrator API with direct atomic projection (selected)

Each agent sends a completion request to a trusted platform API, which
validates the assignment and atomically projects the task outcome and any
parent transition before replying. The same API provides pull-claim,
proposal, and authorized artifact access.

- **Availability/correctness:** Centralizes authority and makes completion
  visibility depend on the API being available when the agent reports
  completion. Callers retry with the same stable command ID after a lost
  response.
- **Authority/duplicates:** Strong platform control if the API authenticates
  the assignment; clients still need stable command IDs and safe response
  retry behavior.
- **Crash recovery/complexity/Redis:** Avoids a separate completion stream,
  consumer group, and projector worker. A lost response is resolved by
  replaying the same command ID and reading the canonical outcome.
- **Issues and v0.7:** Supports later bounded multi-child projection without
  changing the task-owned model; Redis durability remains deployment-bounded.
- **Disposition:** Selected.

### 6. Durable completion command stream and trusted projector

Agents submit task-completion commands to a Redis stream; a platform-owned
consumer validates and projects them idempotently.

- **Availability/correctness:** Durable buffering can decouple agent
  completion from immediate projector availability, subject to configured
  Redis persistence and recovery semantics.
- **Authority/duplicates:** The command is untrusted input until assignment
  validation; command identity, terminal-state checks, and idempotent
  projection handle redelivery and conflicting commands.
- **Crash recovery/complexity/Redis:** Fits Redis Streams and transactional
  state storage, but requires a consumer group, pending-entry recovery,
  safe acknowledgement, monitoring, and retention. None exists today.
- **Issues and v0.7:** Supports independent #34 children and gives #35
  authoritative inputs without prescribing aggregation. It adds no general
  planner or retry policy.
- **Disposition:** Rejected for issue #33 in favor of the authenticated API
  with direct idempotent atomic projection. The additional stream/projector
  boundary is not required for the selected bounded workflow.
