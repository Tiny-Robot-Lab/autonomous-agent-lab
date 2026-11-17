# ADR 0007: Durable Task Continuations and Platform-Governed Asynchronous Resumption

Status: Proposed
Date accepted:
Related issue: https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/33, https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/34, https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/35, https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/36
Supersedes: N/A

## Review status

This revision incorporates human architecture review and remains
**Proposed**; it must not be implemented as accepted architecture. ADR 0006
remains accepted and governing. This ADR fills in the continuation,
completion, and resumption details that ADR 0006 deliberately deferred; it
does not reopen or replace its lineage, delegation-authority, turn-boundary,
depth, wave, or specialist-registration decisions.

An independent Gemini 4 Argon review is required. Its findings must be
validated against live `main` after review, and explicit human acceptance is
required before implementation. The repository evidence below was inspected
at `7e8bfc7`, which is also the local `main` baseline for this proposal. No
review, acceptance, issue update, or implementation is implied.

## Context

[ADR 0006](0006-platform-governed-task-lineage-and-bounded-delegation-authority.md)
decides that agents propose delegation, the platform authorizes and dispatches
it, agents yield at a turn boundary, children complete asynchronously, and a
parent task—not a particular agent process—owns later continuation. It leaves
the concrete continuation and resumption mechanism open and blocks executable
delegation until proposer binding and authority composition are resolved.

The current implementation provides useful foundations but not that
lifecycle:

- `TaskRequested` is a versioned task-dispatch contract; `TaskResult` is a
  result envelope. Neither represents a resumable execution continuation or
  canonical failed terminal outcome. The authoritative store does not retain
  the original prompt/task input independently of the queue.
- `AuthoritativeTaskStore` and `RedisGateway` persist task status, lineage, and
  parent membership. `RegisterChildTask` validates parent/depth and delegates
  child existence to one Redis transaction, but does not atomically update a
  parent continuation, dependency, wait state, or delegation-wave allowance.
- Orchestrator dispatch records routing and writes the advertised dispatch
  target. Redis list dispatch uses `RPUSH`; agents consume with destructive
  `BLPOP`. There is no consumer group, completion projector, or resumable-task
  dispatcher.
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
`TaskResult`; those contracts keep their current dispatch and result
semantics. The record contains:

- `parent_task_id`;
- a reference to the immutable task input and its version;
- `continuation_capability` (initially `research`);
- a bounded, schema-defined `stage` (initially waiting for and then
  incorporating the required child outcome);
- optional small schema-defined control context and references to
  stage-required artifacts;
- required child task IDs (exactly one for issue #33);
- a monotonically increasing `continuation_version`; and
- platform creation/update timestamps.

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
rather than mutating an existing input.

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
| `pending` | Accepted for initial execution, not yet claimed. |
| `working` | An executor holds the current fenced task-turn claim. |
| `waiting_for_children` | Delegation was accepted; the continuation and dependency are durable. No executor is held. |
| `resumable` | Every required child has a canonical terminal outcome; an eligible executor may claim the continuation. |
| `recovery_required` | A continuation claim expired or was abandoned; the continuation remains durable, but a new execution claim requires an explicit recovery decision. |
| `completed` | The parent goal completed. |
| `failed` | The parent goal failed with a canonical safe terminal outcome. |

| Child task state | Meaning |
| --- | --- |
| `pending_dispatch` | Child exists authoritatively and awaits platform dispatch. |
| `pending` | Dispatched and available for execution. |
| `working` | A specialist holds the current fenced task claim. |
| `completed` | Canonical successful terminal outcome and normal result are durable. |
| `failed` | Canonical safe execution-failure outcome is durable. |
| `dispatch_failed` | The platform could not dispatch the existing child; this is terminal for the single-child dependency and is distinct from execution failure. |

The platform transitions parent state; agent lifecycle events do not. A
child's canonical terminal completion or failure satisfies its parent
dependency. Under the issue-#33 one-child policy, that makes the parent
`resumable`, including when the child failed or dispatch failed. The resumed
agent loads the bounded continuation and T101's canonical outcome/result,
incorporates it, and may complete or fail the parent according to the task
contract. No synchronous wait or process-local callback is used.

### 4. Canonical child terminal outcomes

Successful completion is authoritative task state plus the normal
`TaskResult` (or an authoritative reference to it). Failure is an equally
durable terminal record containing the child task ID, a stable non-sensitive
failure code, and terminal timestamp. The initial failure codes distinguish
at least execution failure and dispatch failure. A fixed safe summary may be
included where needed; raw exceptions, provider/network responses, prompts,
credentials, and arbitrary agent text are excluded.

Neither a missing result nor a bounded `agent.events` entry proves success,
failure, or completion. The parent dependency is satisfied only by the
platform's canonical terminal projection. A parent loads the child's
authoritative terminal outcome and, on success, its normal result.

### 5. Atomic delegation acceptance and dispatch

Before accepting a bounded proposal, the platform verifies the assignment
proof, applies the explicit policy in Decision 8, and confirms that at least
one eligible `analysis` candidate currently exists. If none exists, it
rejects the proposal without creating a child, consuming the delegation wave,
or suspending the parent. A proposal contains the requested capability,
bounded child intent, and proposal ID; it does not contain a child ID, target
agent ID, queue name, or dispatch target. Acceptance uses one atomic or
equivalently safe compare-and-set boundary to:

1. verify the expected parent execution/continuation version and unconsumed
   one-wave allowance;
2. validate and consume the one-turn delegation proof and proposal
   idempotency key;
3. mint one child ID and persist child status, forward lineage, and parent
   membership;
4. persist the versioned parent continuation and its required-child
   dependency;
5. set the parent to `waiting_for_children` and consume the root's one
   delegation wave; and
6. record a durable pending-dispatch work item for that child.

Child existence and the parent wait state therefore become visible together.
The dispatch work item makes the post-commit interval recoverable; routing
does not become child existence or authorization. After acceptance, the
platform applies selection to the eligible candidate set and dispatches
through the selected candidate's advertised target. If eligibility or
delivery is lost after acceptance, a terminal dispatch failure is recorded
as a canonical child outcome and resumes the parent with that failure. This
post-acceptance failure is distinct from a pre-acceptance unavailable
capability rejection. Assignment-delivery recovery may retry the same child
ID and dispatch identity; it does not authorize task re-execution.

### 6. Platform-controlled completion command path

Use a Redis task-completion command stream, separate from `agent.events`,
consumed by a trusted Orchestrator completion projector. This is a new
authoritative command boundary, not a conversion of operational events into
commands. For delegated children, agent-side adapters must not establish
canonical completion by directly writing `task.result:{task_id}`. A command
may carry the normal result or an immutable staged-result reference; the
projector validates and owns canonical acceptance.

The agent-side result adapter submits a completion command through the
authenticated Orchestrator channel for its assigned task. The trusted
platform accepts that command onto the Redis stream with a stable
command/assignment identity; it carries either the normal task result, an
immutable staged-result reference, or a safe failure code. The adapter does
not directly write canonical `task.result`, parent state, lineage,
dependencies, or resumability. The projector validates the authenticated
assignment and command against authoritative task state. In one atomic or
equivalently safe projection it establishes:

- the canonical `TaskResult` on success;
- the canonical safe terminal outcome;
- the child terminal status;
- parent dependency satisfaction;
- parent `resumable` state when all required outcomes are terminal; and
- exactly one resumable-work record when the parent becomes resumable.

The projector acknowledges the stream entry only after this projection
commits. Redelivery after a crash is expected and idempotent. Entries remain
pending until projection commits; acknowledgement happens afterward. This
command path's durability is bounded by the configured Redis persistence and
recovery semantics. It is not an unconditional guarantee against Redis data
loss. The canonical terminal outcome remains authoritative after command
acknowledgement. A projector outage delays completion visibility but does not
cause the child runtime to wait for a particular parent process.
Consumer-group recovery, monitoring, and bounded post-ack retention are
implementation obligations; the existing `agent.events` stream is not
reused for them.

This choice keeps platform authority centralized and lets the platform
enforce atomic completion-to-parent projection without requiring every
specialist to call an Orchestrator HTTP service successfully at the moment of
completion. It adds a trusted worker and Redis stream-consumer-group
operations that do not exist today.

### 7. Parent continuation claim and delivery

The platform selects an eligible Research candidate using current
registration, liveness, and its advertised dispatch target. It then performs
one atomic claim against the expected `continuation_version` and task state:
only a `resumable` parent with the expected version can transition to
`working`; the claim records a new claim ID, fencing epoch, and assignment ID.
The platform delivers a private, versioned `TaskAssignment` envelope
(initial envelope version 1), distinct from `TaskRequested` and public
metadata. It semantically binds:

- assignment ID and task ID;
- authenticated assigned-runtime principal;
- claim ID;
- execution kind (`initial`, `delegated_child`, or `resumed_continuation`);
- continuation version (null only when the execution has no continuation);
- fencing epoch and assigned capability;
- the bounded delegation grant, if any;
- expiration/validity;
- references and versions for task input, continuation, and required
  artifacts; and
- the platform-minted assignment proof.

The envelope contains references, not copies of full task input or artifacts.
The proof is secret, scoped to this task turn and assignment, replay-protected,
and invalidated on completion, expiration, or fencing. It is absent from
logs, telemetry, events, routing decisions, and public metadata. Only the
trusted assignment boundary may deliver it to the authenticated assigned
runtime.

An assignment delivery that is not yet acknowledged may be retried only as
the same assignment ID, proof, and fencing epoch to the same authenticated
runtime; the runtime must deduplicate by assignment ID. A stale version,
claim, or fencing epoch cannot write task state, submit another delegation,
or complete the task. There is at most one active valid claimant; the system
does not promise exactly-once execution across process crashes or exactly-once
external provider effects.

If no eligible Research candidate is available, the parent stays
`resumable`; it is not failed or silently discarded. When an eligible agent
becomes available, the platform may dispatch it. If an acknowledged execution
claim expires or is abandoned, that state is observable and the continuation
remains durable in `recovery_required`; the old proof and fence are
invalidated, so stale executors can never commit. Issue #33 does not
automatically re-execute the parent or redispatch a new execution assignment
after an ambiguous claim. Any new execution claim after an expired/abandoned
claim requires an explicit recovery decision outside the automatic issue-#33
path. This is not a general task retry policy; only unacknowledged
assignment-delivery recovery may redeliver the same assignment as described
above.

### 8. Trustworthy proposer binding and narrow authority

Resolve ADR 0006 item 18 with a platform-minted, one-turn task-assignment
proof. It is high-entropy and unforgeable, bound to the authoritative parent
task, authenticated runtime identity, assignment ID and fencing epoch, the
`research -> analysis` edge, a one-child limit, and bounded validity. The
platform validates and consumes its delegation authority atomically;
proposal ID plus assignment epoch prevents replay. A consumed, expired,
superseded, or mismatched proof cannot authorize a new proposal. An exact
authenticated replay of an already accepted proposal returns its recorded
acceptance without creating state or consuming new authority. The proof
grants only the ability to submit the bounded proposal; it does not mint a
child ID, select an agent, name a queue, or grant the child permissions.

**The current shared Redis topology does not prove the identity of a queue
consumer.** A self-asserted `agent_id`, advertised capability, Redis
registration record, or possession of a shared Redis credential is not
runtime authentication.

For executable issue #33, the minimum selected trust boundary is an
authenticated platform assignment/delegation channel that authenticates a
runtime instance, binds that principal to the platform-issued `TaskAssignment`,
and validates the proof for proposal and completion commands. The initial
choice is an authenticated Orchestrator API/channel with per-runtime
credentials and a protected assignment delivery path. The `TaskAssignment`
envelope and proof must not be delivered through the current shared Redis
task queue. Redis ACLs may add defense in depth but do not replace this
binding when credentials are shared by service instances.

Service-specific Redis ACL credentials can restrict a service to its
capability queue and command stream, but credentials shared by all instances
of a service do not distinguish which instance consumed a queue item. Unique
per-runtime Redis credentials plus queue ACLs could establish that identity,
but would require per-instance provisioning, queue isolation, rotation, and
validation. An authenticated platform API/channel centralizes principal
binding and proof checks, at the cost of a new runtime dependency and
availability boundary. This ADR selects the latter as the minimum semantic
boundary; its concrete authentication and delivery mechanism must be selected
and tested before delegation is enabled. Until that is done, executable
delegation is blocked. Shared Redis delivery alone is insufficient.

Capability advertisement is not a parent authority grant: the platform must
supply an explicit narrow grant for the parent execution, and must reject
delegation if that grant cannot be established.

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

### 9. Duplicate, retry, and consistency semantics

- **Delegation proposal:** A unique proposal ID is scoped to the parent
  assignment/turn. Replaying the same accepted proposal returns its original
  child and acceptance; it never mints another child. A different proposal
  after the wave is consumed is rejected.
- **Child dispatch:** The child ID and dispatch generation are the
  idempotency key. Retrying platform delivery cannot create another child or
  authorize a second execution claim. A terminal dispatch failure is
  explicit; this ADR does not add general task retry or replanning.
- **Child completion:** A stable completion command ID and child assignment
  identify a completion. Identical redelivery is acknowledged as a no-op
  after projection. A conflicting second terminal outcome cannot overwrite
  the first canonical outcome or result.
- **Parent unblock:** Dependency satisfaction and the transition to
  `resumable` are a single conditional projection. Repeated commands or
  projector delivery do not create a second continuation version or duplicate
  resumable work.
- **Continuation claim/delivery:** An expected continuation version and
  claim idempotency key are checked atomically. Duplicate delivery of the
  same active assignment to the same runtime is safe; a stale claim cannot
  acquire or use the active fence. A new task execution claim after expiration
  requires explicit recovery and a higher fence, not automatic retry.

Redis transactions already establish child-registration and dispatch
boundaries, but a transaction alone does not express every conditional
version/claim check. Implementation must use a Redis atomic script or an
equivalently safe compare-and-set mechanism for multi-key state transitions.
No key layout is mandated here.

### 10. Issue boundaries and exclusions

**Issue #33** is limited to one Research parent, one Analysis child, one
delegation wave, non-blocking suspend/yield, durable continuation, canonical
child completion or failure, parent resumability, one active atomic
continuation claim, and parent resumption with child outcome incorporation.
The parent remains owner of the overall goal.

**Issue #34** owns multiple children, concurrent dispatch/execution,
wait-all/partial/timeout/cancellation policies, and their synchronization.
It uses this task-owned continuation and completion architecture; it does not
create a second continuation model.

**Issue #35** owns structured aggregation of authoritative child outcomes,
task/contributor attribution, evidence/source preservation, and
partial-failure representation. It must not reconstruct results from
best-effort operational events.

**Issue #36** owns observational tracing through authoritative identifiers
and execution turns: root task ID; parent/child task IDs; delegation
proposal/decision ID; routing decision; assignment/claim ID; continuation
version; canonical terminal outcome; and executor agent ID per turn. Tracing
may correlate and display these values but can never become authoritative
lineage, task state, completion, dependency, continuation, or claim state.

This ADR does not introduce grandchildren, multiple delegation waves,
arbitrary execution DAGs, sequential plan steps, conditional planning,
retry/replanning policy, dynamic specialist creation, arbitrary
capability-to-capability delegation, process-local synchronous waits,
continuation tied to one agent ID, `agent.events` as authoritative
completion, serialization of Python runtime state, cross-task memory or
v0.8 memory semantics, general v0.7 planning abstractions, automatic public
user routing to Analysis, or parallel execution/aggregation inside issue
#33.

Implementation remains within functional issue #33 but should be delivered
through bounded increments, without assigning issue numbers here:

1. Task-input, continuation, and artifact contracts and authoritative
   storage.
2. The `TaskAssignment` envelope, authenticated proposer binding, and narrow
   authorization.
3. Atomic delegation acceptance and child dispatch.
4. Completion commands, the trusted projector, and canonical outcomes.
5. Resumable-parent claim and Research continuation.
6. End-to-end Research -> Analysis -> Research validation.

### Required invariants

- A waiting parent occupies no agent execution slot.
- A resumable task can recover its original immutable task input and all
  stage-required artifacts without relying on consumed messages or process
  memory.
- Parent continuation, dependency, and terminal child outcome survive agent
  and process restart, subject to configured Redis persistence and recovery
  semantics.
- Intermediate Research material is immutable, versioned, task-scoped, and
  provenance-bearing; continuation and child input reference it rather than
  copying it into lineage or control state.
- Continuation targets a capability, not an agent ID.
- Child completion, execution failure, and dispatch failure are canonical and
  durable; operational events are not authoritative.
- At most one valid executor holds a continuation's active claim/fence.
- A stale executor whose assignment proof or fencing epoch is invalidated
  can never commit task state or completion.
- Duplicate commands do not duplicate child existence, terminal projection,
  parent unblocking, or active parent resumption.
- A delegated child's canonical `TaskResult` is accepted only by trusted
  platform projection, never by direct agent writes to `task.result`.
- A terminal outcome for the single child eventually makes the parent
  resumable, including child failure and dispatch failure.
- `research -> analysis` is the only initially authorized delegation edge.
- Delegation does not increase effective authority.
- The parent task remains owner of the overall goal.
- Issue #33 permits one child and one delegation wave per root.

## Deferred questions

These are implementation details, not permission to weaken the decisions
above. They must be resolved and validated before executable delegation is
enabled:

- Which concrete per-runtime credential and delivery mechanism implements
  the authenticated Orchestrator assignment/delegation channel, binds queue
  consumption to that runtime, and protects proof issuance and rotation?
- What bounded assignment/claim lease duration and heartbeat policy provide
  useful recovery without allowing stale executors to commit?
- Which Redis persistence and recovery configuration is required for the
  completion-command durability needed by the deployment, and what post-ack
  retention policy reclaims entries without deleting unprojected or pending
  commands?
- What exact per-stage continuation field schemas and stable safe failure
  code set does the issue-#33 contract require?

## Consequences and tradeoffs

Benefits:

- Durable task state, rather than a live process, owns the work across
  suspension, completion, and resumption.
- A separate completion command path preserves the distinction between
  authoritative lifecycle commands and bounded, best-effort observability.
- Canonical failed outcomes allow a parent to resume even when a child fails
  before producing a normal result.
- Versioned compare-and-set claims and idempotent projections make duplicate
  delivery and process restart explicit rather than relying on event timing.
- The initial Research-to-Analysis authority edge is narrow, deterministic,
  and compatible with the platform-owned routing and authorization boundary
  in ADR 0006.
- The same task-owned model supports later bounded fan-in and aggregation
  without pulling arbitrary dependency graphs or general planning into v0.6.

Costs and limitations:

- The platform gains authoritative continuation, dependency, terminal
  outcome, claim/lease, and resumable-work state plus new conditional atomic
  operations.
- A durable completion stream and trusted projector introduce a worker,
  consumer-group recovery, monitoring, retention, and operational failure
  modes absent from the current architecture.
- Redis queue consumption currently uses destructive `BLPOP`; reliable
  delegated-task and continuation delivery must be addressed at the
  assignment boundary without claiming that the current task queues already
  provide at-least-once processing.
- Assignment-proof protection requires a private authenticated runtime
  boundary. The current self-asserted identity and shared Redis behavior are
  insufficient evidence of such a boundary.
- A lease and fencing token prevent concurrent authoritative owners but
  cannot guarantee exactly-once external effects if an executor crashes after
  performing them. This ADR adds no general execution retry policy.
- Continuation schemas and their size bounds must be explicit, versioned, and
  reviewed as they evolve; arbitrary model context is not a safe default.
- Candidate availability and Redis behavior at collaboration scale remain
  unmeasured.

## Alternatives considered

### 1. Blocking in-process wait by the original parent agent

The original Research Agent waits for T101 and then continues T100 in the same
call stack.

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

### 3. Durable task-owned suspension and resumption (selected)

The parent stores a versioned continuation and dependency, yields, and later
is atomically claimed and delivered to an eligible capability-matched
executor.

- **Availability/correctness:** Survives process replacement; conditional
  writes, fencing, canonical outcomes, and idempotency protect authoritative
  state. External effects are not exactly-once.
- **Authority/duplicates:** Platform owns acceptance, task identity, routing,
  claim, and completion projection; requires a protected assignment proof and
  explicit duplicate semantics.
- **Crash recovery/complexity/Redis:** Best fit with Redis authoritative
  status/lineage and transaction precedent, but adds state transitions,
  durable outbox/command handling, claims, leases, and stream consumers.
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

### 5. Synchronous Orchestrator completion API

Each agent sends a completion request to a trusted platform API, which
validates and atomically projects the child outcome and parent transition
before replying.

- **Availability/correctness:** Centralizes authority and can make one
  synchronous request idempotent, but completion visibility depends on the
  API being available at the time the agent reports completion.
- **Authority/duplicates:** Strong platform control if the API authenticates
  the assignment; clients still need stable command IDs and safe response
  retry behavior.
- **Crash recovery/complexity/Redis:** Fewer components than a projector, but
  couples specialist completion latency and availability to the Orchestrator
  API. A lost response leaves the client uncertain whether the transaction
  committed.
- **Issues and v0.7:** The API can project multiple children later, but does
  not itself provide durable buffering while the platform is unavailable.
  It is compatible with the task-owned model but less decoupled for #34.
- **Disposition:** Not selected for the primary completion path. A narrow
  authenticated API may be used as transport only if it durably accepts a
  command before acknowledging it; authority and projection remain in the
  trusted projector.

### 6. Durable completion command queue and trusted projector (selected)

Agents submit durable task-completion commands; a platform-owned consumer
validates and projects them idempotently.

- **Availability/correctness:** Durable buffering decouples agent completion
  from immediate projector availability. The projector is the one authority
  that commits terminal outcome and parent unblocking.
- **Authority/duplicates:** The command is untrusted input until assignment
  validation; command identity, terminal-state checks, and idempotent
  projection handle redelivery and conflicting commands.
- **Crash recovery/complexity/Redis:** Fits Redis Streams and transactional
  state storage, but requires a consumer group, pending-entry recovery,
  safe acknowledgement, monitoring, and retention. None exists today.
- **Issues and v0.7:** Supports independent #34 children and gives #35
  authoritative inputs without prescribing aggregation. It adds no general
  planner or retry policy.
- **Disposition:** Selected over direct Redis writes and synchronous-only
  completion because the repository currently has no durable completion
  projection and agents must not own parent lifecycle authority.
