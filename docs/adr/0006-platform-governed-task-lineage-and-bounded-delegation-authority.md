# ADR 0006: Platform-Governed Task Lineage and Bounded Delegation Authority

Status: Accepted
Date accepted: 2026-11-14
Related issue: https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/31
Supersedes: N/A

## Review status

**Accepted** — this record was accepted by explicit human decision. The
independent review below informed, but did not make, that acceptance.

An independent architecture review was performed by Gemini 4 Argon High.
Overall verdict: "Sound with revisions; blocked from acceptance and
implementation until foundational gaps are resolved." The review raised
material findings concerning durable lineage, referential integrity, atomic
child existence semantics, serial-agent deadlock/turn boundaries, proposer
identity and delegation authority, multi-wave delegation, contract
compatibility, rejection identifiers, and the v0.6 fan-in/v0.7 dependency
boundary.

Those findings were subsequently validated against live `main` at commit
`79168af` by repository inspection. The live validation confirmed most
findings, corrected Argon's overstatement of `READY_FOR_CONTINUATION` as
reusable runtime precedent, clarified that `task.routing` proves selection
but not proposer identity or task existence, verified v1 -> v2 upcasting is
factually safe but must be explicitly implemented, and clarified fan-in as
a later v0.6 milestone concern rather than an intrinsic issue-#31
requirement. This ADR was revised based on those validated findings and
subsequent explicit human architecture decisions made after reviewing them
(authority containment, asynchronous child completion, and incremental
v0.5 -> v0.6 transition). ADR 0006 has now received explicit human
acceptance. Gemini 4 Argon High performed the independent review recorded
above; Argon did not itself accept or approve this ADR.

## Context

Tiny Agent Lab is preparing v0.6.0 — Multi-Agent Collaboration. Issue #31
proposes `parent_task_id`, `root_task_id`, dependency metadata, collaborative
work as a DAG, and querying child tasks by parent. A read-only investigation
of the v0.5.0 implementation established the following.

**Identity and authority today.** Task IDs are minted only through the
orchestrator submission path: `TaskRequested.create()` in
`packages/contracts/src/agent_contracts/messages.py` generates
`task_id=str(uuid4())`, called only from `SubmitTask.execute()` in
`services/orchestrator/src/orchestrator/application/use_cases.py`. Worker and
Research Agent consume already-identified tasks; their `MessageBus` ports
expose `next_task`, `publish`, `register`, and `heartbeat` (plus result
saving) with no enqueue/dispatch authority, and they reuse the inbound
`task_id` for results and events. Separately, the orchestrator mints a
decision-only identifier for the no-eligible-candidate rejection path,
attached to a routing decision, error event, and error response; no task is
enqueued for it (Decision item 17 resolves its status).

**Persistence and queries today.** Tasks form a flat namespace keyed by
`task_id` (`task.status:{task_id}`, `task.result:{task_id}`,
`task.routing:{task_id}`), with per-capability Redis list queues. Task
messages are transient queue payloads consumed by `BLPOP`; routing, status,
and dispatch are separate sequential writes, so partial states are
reachable today. All reads are exact-ID lookups or bounded linear scans of
`agent.events` filtered by `task_id`. There is no task enumeration,
parent/child index, relationship query, dependency graph, cross-task
fan-out/fan-in, or result aggregation; routing selects exactly one eligible
registered agent and each agent executes serially, one task at a time. Per
[ADR 0005](0005-canonical-task-scoped-inference-fallback-history.md),
`agent.events` (`maxlen=500, approximate=True`) is best-effort bounded
operational evidence, not a durable store or audit ledger, and must not
become the authoritative relationship store.

**Proposer identity today.** `task.routing:{task_id}` is durable evidence
of which agent the platform selected for a task. It does not authenticate
which runtime actually emitted anything: `agent_id` is self-asserted
through Redis-facing runtime behaviour (registration, state, heartbeat,
and event publication carry no verified process identity). This is a
current gap, addressed as a requirement in Decision item 18.

**Authority precedent.** Accepted v0.5 architecture separates discovery and
selection from execution authority:
[ADR 0002](0002-bounded-provider-neutral-tool-use-coordination.md) keeps tool
discovery, selection, validation, authorization, execution, and continuation
distinct, and [ADR 0003](0003-exact-tool-authorization-and-security-auditing.md)
requires exact authorization and auditing before tool execution;
authorization comes from trusted runtime composition, not advertised
capabilities. `BoundedToolCoordinator` returning `READY_FOR_CONTINUATION`
after at most one invocation is only loose conceptual vocabulary for
bounded continuation: live validation confirmed it is not wired into the
running Worker or Research Agent services, and it must not be claimed as
reusable runtime infrastructure for delegation.

**Roadmap boundary.** [ROADMAP.md](../../ROADMAP.md) places parent/child
relationships, subtask execution, delegation, aggregation, and cross-agent
tracing in v0.6, and general dependencies, sequential/conditional execution,
retry, and replanning in v0.7. Issue #31's dependency/DAG wording must not
be read as pulling v0.7 planning into v0.6; milestones are planning
direction, not promises. For v0.6, collaboration assumes already registered
specialists; dynamic provisioning is out of scope (Decision item 12).

## Decision

Platform-governed task lineage with bounded, hybrid delegation authority
for v0.6:

1. **Task identity.** Every task has a globally unique `task_id` — "Which
   task is this?" Identity semantics do not change when depth policy does.
2. **Root lineage.** Every task has a required, non-null `root_task_id` —
   "Which overall collaborative execution does this task belong to?" For a
   root, `root_task_id == task_id`; for descendants it is inherited
   unchanged from the originating root.
3. **Immediate parentage.** `parent_task_id` is only the immediate causal
   parent — "Which task directly caused this child task to exist?" For
   roots it is absent/null, with exactly one meaning: no parent exists
   because this task is a root. Null must not mean unknown, unresolved,
   failed lookup, legacy value, or missing data. Roots never self-parent.
4. **Distinct semantics.** The three identifiers must not be overloaded.
   Root grouping uses `root_task_id`, never a null `parent_task_id`.
5. **Lineage is not dependency ordering; bounded fan-in is distinct.**
   Lineage records causation and membership; it does not encode arbitrary
   task dependency ordering. v0.6 may use the bounded structural
   parent-waits-for-children completion barrier required for fan-in/result
   aggregation. v0.6 does not
   introduce sibling-to-sibling or arbitrary cross-task `depends_on`
   edges. General dependency graphs, conditional sequencing, cycle
   detection, retries across dependency graphs, and replanning remain
   v0.7 territory. This fan-in position is milestone-level architecture
   for later v0.6 increments; issue #31 itself remains focused on lineage
   and child lookup.
6. **Representational depth.** The model must represent deeper ancestry
   consistently — representation does not imply execution permission:

   ```text
   A:  task_id=A,  root_task_id=A, parent_task_id=null
   A2: task_id=A2, root_task_id=A, parent_task_id=A
   A3: task_id=A3, root_task_id=A, parent_task_id=A2
   ```

7. **Bounded initial execution depth.** Initial v0.6 execution is one
   level: root -> children. Children may not create grandchildren. Under
   this depth-1 policy, the parent of every child must itself be a root
   (`parent.parent_task_id` is null), so for every child:
   `child.parent_task_id == child.root_task_id`. A later explicit v0.6
   increment may relax depth only after separate review of authorization,
   failure propagation, observability, lifecycle, fan-out, and stopping
   conditions. Relaxing depth changes policy, not identity.
8. **Hybrid delegation authority and trust boundary.** An agent runtime,
   or any collaboration helper operating outside the trusted orchestrator
   policy authority, may only **propose** delegation; it is untrusted for
   authorization purposes and does not directly mint or enqueue arbitrary
   child tasks. The proposing agent does not delegate permissions,
   credentials, tool grants, network access, or other authority to a child
   or specialist; it expresses the required capability and bounded subtask
   intent only. It does not need to know which specialist will execute the
   child or inspect that specialist's security configuration. Only trusted
   platform/orchestrator policy may validate the proposal, authorize
   delegation, enforce depth/fan-out and other bounds, mint the child
   identity, establish lineage, discover eligible registered specialists,
   apply liveness/selection policy, record routing/state/lineage evidence,
   dispatch the child, determine lifecycle/termination behaviour, and
   establish the effective authority available to the child. A proposal is
   not authorization, creation, routing, or dispatch — extending the ADR
   0002/0003 precedent (discovery/selection grants no execution authority)
   from tool invocation to task delegation. The hybrid authority model does
   not by itself define a synchronous interaction topology; the initial
   interaction model is item 9.
9. **Turn-boundary delegation.** Initial v0.6 delegation proposals are
   non-blocking with respect to agent execution slots. An agent proposes
   delegation as the terminal outcome of its current execution turn and
   does not synchronously wait for child completion while retaining that
   turn:

   ```text
   parent execution turn
   -> structured delegation proposal
   -> parent execution turn ends/yields
   -> platform validates/authorizes/bounds proposal
   -> platform establishes and dispatches children
   -> children execute
   -> later bounded aggregation/continuation phase
   ```

   Implementation must prevent same-capability self-deadlock under today's
   serial single-instance agent execution model.

   Child completion is asynchronous with respect to agent runtimes. A child
   does not return directly to, synchronously call, or wait for a specific
   parent agent process or runtime. The logical parent is the authoritative
   parent **task**, identified by `parent_task_id`, not whichever agent
   process previously executed that task. Child completion is recorded
   through platform-controlled task/result/state mechanisms; the platform
   correlates child completion to parent/root lineage, and any later bounded
   fan-in/aggregation/continuation is coordinated from platform-controlled
   parent execution state. This avoids coupling child completion to whether
   a parent agent runtime is currently busy processing unrelated work. The
   exact continuation queue or state machine is not designed in this ADR.
10. **Single delegation wave initially.** Initial v0.6 permits one bounded
    delegation wave per root collaborative execution. A single wave may
    contain multiple authorized children.
    `delegate -> inspect result -> delegate again` sequences within the
    initial policy are not permitted; that is sequential/conditional
    multi-step planning belonging to v0.7. The platform must enforce
    cumulative child/fan-out bounds against authoritative relationship
    state, not merely per individual proposal. No numeric limit is set by
    this ADR.
11. **Registered specialists only.** v0.6 delegates only to existing
    registered eligible specialist agents. If none provides the requested
    capability, the outcome is an explicit observable unavailable-capability
    outcome, consistent with current no-eligible-candidate behaviour. No
    novel specialist is dynamically provisioned or created.
12. **Dynamic provisioning out of scope.** Recorded only as a future
    architectural/security consideration, not designed here. Any later
    consideration requires separate review of ephemeral lifetime, data
    retention, tool/network/communication confinement, credential
    isolation, authorization, auditability, recursion/delegation bounds,
    cleanup, and provenance.
13. **Durable forward lineage, both directions.** Every created task must
    have authoritative durable forward lineage — `task_id`,
    `root_task_id`, `parent_task_id` — queryable by `task_id`
    independently of the transient `TaskRequested` queue payload and of
    bounded `agent.events` history. Both directions must be durably
    supportable: `task_id` -> root/parent lineage, and `parent_task_id`
    -> children. `TaskResult` is too late to serve as the authoritative
    lineage record. No Redis key shape is mandated.
14. **Authoritative child existence.** A child task exists only when its
    durable task state, authoritative forward lineage, and
    parent-to-child membership have been committed as one consistency
    boundary. Dispatchability occurs after this existence boundary.
    Implementation must not expose phantom child-index entries with no
    authoritative child task, or authoritative child tasks omitted from
    the parent lookup, and must provide atomic or equivalently safe
    failure semantics for this boundary. No concrete persistence mechanism
    is prescribed. A child may exist yet subsequently fail before dispatch;
    that state must be explicit rather than indistinguishable from a
    normally queued/pending child.
15. **No root-wide index yet.** `root_task_id` travels with every task so
    later tracing/grouping can use the invariant without redesigning
    identity, but no root-wide index is introduced unless a later
    acceptance criterion, tracing requirement, or implementation evidence
    requires it.
16. **Contract evolution: version-gated v1 -> v2 upcasting (decided).**
    New lineage-aware `TaskRequested` messages use contract version 2.
    v2 requires a valid `root_task_id`; `root_task_id` must never be
    silently defaulted for malformed v2 payloads, and any malformed v2
    lineage payload is rejected rather than repaired by unconditional
    defaults. Explicit v1 payloads may be upcast during deserialization,
    because live-repository validation proved child creation was impossible
    in v1: upcast semantics are `root_task_id = task_id`,
    `parent_task_id = None`. Unknown/unsupported versions must not be
    silently interpreted as v1. Only explicit version-aware deserialization
    semantics are required; the exact implementation location of the
    version gate is not prescribed.
17. **Rejection identifiers are non-tasks (decided).** A
    no-eligible-candidate rejection identifier represents
    routing/delegation-attempt evidence, not a Task. No `TaskRequested`
    exists for it, and it is excluded from authoritative parent-to-child
    task membership. It may carry or root-resolve enough parent/root
    correlation for observability and failure tracing. A routing record
    alone must never be treated as proof that a task exists. Existing
    unavailable-capability observability semantics are preserved where
    possible.
18. **Proposer identity is a required security boundary and a current
    gap.** Before delegation execution is enabled, the platform must
    establish a trustworthy binding between the delegation proposal, the
    authoritative parent task, and the runtime authorized to act for that
    parent. Today `task.routing:{parent_task_id}` proves which agent the
    platform selected; it does not authenticate which runtime emitted a
    proposal, and `agent_id` is self-asserted through Redis-facing runtime
    behaviour (Context). No mechanism (tokens, signed events, connection
    identity, credentials, or other) is prescribed. The current security
    posture must not be overstated: this binding does not exist today.
19. **Delegation must not increase effective authority.** Delegation must
    not increase the effective authority available to the originating
    collaborative execution merely because the selected specialist has
    broader tool, network, credential, or execution privileges. Effective
    child authority must never exceed the authorized scope of the
    originating collaborative execution, the selected specialist's
    independently configured/authorized scope, or the additional
    restrictions imposed by delegation policy: it is constrained by all
    applicable authority boundaries, never expanded by taking their union.
    The platform must apply an explicit delegation-authorization policy
    before child creation. No caller-capability -> target-capability
    matrix, concrete permission-intersection data structure, or policy
    algorithm is mandated. The detailed authority-composition mechanism is a
    separately scoped security/policy design and must be resolved before
    executable delegation is enabled. This is currently a latent design
    threat, not a presently exploitable delegation vulnerability: live
    delegation does not yet exist, and current tool authorization is not
    wired into the running agents.
20. **Lineage propagation scope.** Lineage is not duplicated into every
    contract by default:
    - *Authoritative:* `TaskRequested` must carry lineage to the executing
      specialist; durable authoritative task/lineage state must preserve
      lineage after queue consumption.
    - *Observability:* `RoutingDecision` and `AgentEvent` may carry
      lineage where later tracing requirements justify denormalization;
      such copies are observational, not authoritative relationship state.
    - *Not inherently required:* `TaskResult`, inference telemetry, and
      tool telemetry do not require lineage duplication unless a later
      acceptance criterion demonstrates a need.
21. **Security/privacy.** Lineage metadata is identifiers and relationships
    only — never duplicated prompts, child outputs, credentials, raw
    provider errors, or arbitrary agent working context. Existing
    sanitization/authorization boundaries remain applicable, including
    excluding prompt/output-bearing detail from telemetry projections.
    Relationship metadata must not become an excuse to persist sensitive
    payloads.
22. **Issue #31 scope alignment.** Accepting this architecture requires
    issue #31 to be narrowed separately so its "dependency metadata / DAG"
    wording does not imply arbitrary dependency scheduling in v0.6. When
    this ADR is accepted, #31 should focus on `task_id`/`root_task_id`/
    `parent_task_id` lineage, durable authoritative lineage, child lookup
    by parent, and invariants/compatibility for those relationships.
    General dependency/DAG scheduling belongs in v0.7 tracking. This ADR
    does not itself modify issue #31.
23. **Incremental v0.5 -> v0.6 operational transition.** v0.6 lineage
    foundations must be introduced incrementally rather than relying on an
    untested hard operational cutover from v0.5 semantics. Existing v0.5
    root-task behaviour must remain functionally compatible while lineage
    foundations are introduced, and version-gated v1 -> v2 upcasting
    (item 16) remains the selected wire-contract compatibility strategy.
    Before executable delegation is enabled, validation must cover at
    minimum:
    - explicit v1 root payload upcasts to `root_task_id=task_id` and
      `parent_task_id=None`;
    - valid v2 root accepted;
    - valid v2 child accepted when authoritative parent/root invariants
      hold;
    - malformed v2 missing/invalid root rejected;
    - self-parent rejected;
    - cross-root parentage rejected;
    - grandchild/depth>1 execution rejected under the initial policy; and
    - existing v0.5-style root submission/routing/result behaviour remains
      regression-safe.

    Intermediate public releases or prereleases may be considered later if
    they communicate a genuinely useful/stable checkpoint, but release
    numbering is not decided by this ADR. No intermediate public release
    number is chosen or promised here.

### Required invariants

- `task_id` is globally unique per task.
- `root_task_id` is required and non-null.
- Root: `task_id == root_task_id` and `parent_task_id` is null/absent.
- Child: `task_id != root_task_id` and `parent_task_id` is present.
- `parent_task_id` must never equal `task_id`.
- `parent_task_id` is null if and only if the task is a root.
- Root grouping uses `root_task_id`, never null `parent_task_id`.
- `parent_task_id` on a child must reference an already-existing
  authoritative task.
- `child.root_task_id == parent.root_task_id`.
- Child `root_task_id` is platform-derived from authoritative parent
  state, never trusted from agent/proposer input.
- Under the initial depth-1 policy, every child's parent is itself a root
  (`parent.parent_task_id` is null), therefore
  `child.parent_task_id == child.root_task_id`.
- Bounded v0.6 execution initially allows root -> children only and one
  delegation wave per root, even though the representation can express
  deeper ancestry and multiple waves could be modelled.
- A child exists only at the item-14 consistency boundary, and is
  dispatchable only after it.
- A proposing agent never delegates or transfers its own permissions,
  credentials, tool grants, network access, or other authority; it
  expresses required capability and bounded subtask intent only.
- Effective child authority never exceeds the authorized scope of the
  originating collaborative execution, the selected specialist's
  independently configured/authorized scope, or the additional
  restrictions imposed by delegation policy; it is constrained by all
  applicable boundaries, never expanded by their union.
- Child completion is correlated by the platform to the authoritative
  parent task (`parent_task_id`), not delivered to a specific parent agent
  process or runtime.

### Resolved tensions and remaining deferred questions

Resolved by this revision: contract compatibility is decided as
version-gated v1 -> v2 upcasting (item 16), and rejection identifiers are
decided as non-task attempt evidence excluded from task membership
(item 17).

Explicitly deferred, not solved here: the trustworthy proposer-binding
mechanism (item 18); the delegation authority-composition mechanism
(item 19); the concrete persistence mechanism for the item-14 consistency
boundary and the item-13 lookup; the design of the later bounded
aggregation/continuation phase (item 9); any relaxation of depth or wave
bounds; and the separately scoped narrowing of issue #31 (item 22).
Executable delegation must not be enabled before items 18 and 19 are
resolved.

### What this decision does not do

It implements nothing: no delegation, parallelism, aggregation, tracing UI,
general dependencies, or dynamic agent creation. It defines lineage,
existence, authority, and interaction bounds so later v0.6 work (issues
#33–#38, per their own criteria) builds on stable semantics. It does not
design the delegation-request contract, child-index mechanism,
lifecycle/termination mechanics, numeric limits, or failure propagation.

## Consequences and tradeoffs

Benefits: one platform authority path for creation and routing, consistent
with orchestrator-only minting and the authorization precedent; durable
lineage and a defined child-existence boundary precede delegation
execution, so later v0.6 work (#33–#38) builds on identity rather than
retrofitting it; turn-boundary, single-wave, and depth bounds keep v0.6
separate from v0.7 planning while still permitting bounded fan-in;
version-gated upcasting gives pre-lineage payloads a decided, factually
safe migration path.

Costs: durable lineage plus a parent-to-child lookup with atomic-or-
equivalent consistency (items 13–14) is new authoritative state beyond
today's flat exact-ID writes; contract version 2 and explicit version-aware
deserialization are required; recursive collaboration and multi-wave
delegation are deferred; root grouping stays a future capability without a
root-wide index; the central policy path's performance at collaboration
scale is unmeasured; and executable delegation remains blocked on the
deferred proposer-binding (item 18) and authority-composition (item 19)
designs.

## Alternatives considered

### A. Orchestrator-owned hierarchy and delegation decisions

Orchestrator alone decides what children are needed and creates them; agents
never propose delegation.

- *Benefits/drawbacks:* Simplest authority model, reusing discovery,
  liveness, selection, and dispatch as-is, with the easiest lifecycle
  ownership — but the platform must know at submission time what subtasks a
  goal needs, and agents discovering a need mid-execution have no channel
  to surface it, limiting collaboration to pre-decomposable work.
- *Authority/security:* Strongest central control; no new agent authority.
- *Observability/tracing:* Complete platform-side evidence from creation.
- *Compatibility:* Highest; extends `SubmitTask` without changing agent ports.
- *v0.6→v0.7 coupling risk:* Moderate — platform goal decomposition is
  planning behaviour and would pull v0.7 forward.
- *Disposition:* Not selected whole; its creation/routing authority is
  retained in the hybrid, which adds the missing proposal channel.

### B. Agent-owned direct child creation/delegation

Agents directly mint and enqueue child tasks on other agents' queues.

- *Benefits/drawbacks:* Shortest path from need to action, no bottleneck —
  but children get no routing decision, liveness check, initial status, or
  depth/fan-out bound; lineage is agent-asserted and termination, failure,
  and cleanup have no owner.
- *Authority/security:* Weakest. Grants arbitrary task-creation and
  queue-write authority, bypassing ADR 0003 authorization/audit and ADR
  0002's discovery-is-not-authorization boundary; a mistaken or compromised
  agent could create unbounded work on any queue it can name. Also worsens
  the item-18 proposer-identity gap by making self-asserted identity the
  creation authority.
- *Observability/tracing:* Poor; children could exist off the recorded
  platform path, leaving the ageing-out event stream (ADR 0005) as the only
  cross-reference.
- *Compatibility:* Low; agent `MessageBus` ports deliberately expose no
  dispatch authority, and queues are written only by orchestrator dispatch.
- *v0.6→v0.7 coupling risk:* High — unbounded recursive subtasks reproduce
  planning/multi-step semantics inside agents without v0.7 guardrails.
- *Disposition:* Rejected; it reverses established authority boundaries.

### C. Hybrid agent-proposal / platform-authority model (proposed)

Untrusted agent runtimes/helpers propose bounded delegation; only trusted
platform policy validates, authorizes, bounds, mints, routes, records, and
dispatches (items 8–10).

- *Benefits/drawbacks:* Agents surface needs discovered in execution while
  every creation passes one authorizing, bounding, recording path, and
  lineage is platform-established hence trustworthy — at the cost of a
  proposal channel, an existence-consistency boundary, and platform policy
  that do not yet exist, plus contract versioning and durable lookup (more
  than D, slower than B).
- *Authority/security:* Preserves ADR 0002/0003 precedent: proposing grants
  nothing; authorization and audit stay platform-side via trusted
  composition, not inferred agent metadata. Items 18–19 record the
  proposer-binding and authority-composition work still required.
- *Observability/tracing:* Strong; every child is created on the recorded
  routing path, enabling durable lookup and later cross-agent tracing
  without making the bounded event stream authoritative.
- *Compatibility:* Good; keeps orchestrator-only minting, discovery,
  liveness, and selection, adding a proposal input rather than a second
  creation path. Agents gain at most a request channel, not dispatch power.
- *Interaction note:* The hybrid authority model does not by itself define
  a synchronous topology; item 9 chooses turn-boundary proposal/yield with
  platform-controlled continuation, avoiding same-capability self-deadlock.
- *v0.6→v0.7 coupling risk:* Low–moderate and bounded — lineage cannot
  express arbitrary ordering, the single-wave bound blocks sequential
  re-delegation, and the depth bound blocks recursion; residual risk is
  proposal policy drifting into goal decomposition.
- *Disposition:* Selected and accepted after independent architecture review,
  live-repository validation, and explicit human architecture decision.

### D. Relationship metadata only, no delegation semantics yet

Add the lineage fields and a child lookup, but define no delegation in v0.6.

- *Benefits/drawbacks:* Smallest change, stabilizes identity and lookup
  early with no new authority surface — but no production path would ever
  create a child, so the fields are exercised only by roots while "who may
  cause a child, under what bounds" stays undecided as the model implies it.
- *Authority/security:* Neutral; nothing granted or defined.
- *Observability/tracing:* Root grouping data exists but there are no
  children to trace; the durable lookup would be empty in practice.
- *Compatibility:* Highest overall; contract-and-index change only.
- *v0.6→v0.7 coupling risk:* None directly, but #33–#38 could proceed
  without agreed authority and improvise it per increment.
- *Disposition:* Not selected whole; its lineage semantics are adopted in
  full inside the hybrid, which adds the authority, existence, and
  interaction model.

### E. General dependency/DAG model in v0.6

Implement #31's dependency metadata/DAG wording now, with scheduling.

- *Benefits/drawbacks:* One general model for lineage, ordering, and future
  planning — at the largest cost: edge storage/querying, cycle detection,
  scheduling, conditional execution, graph failure/retry propagation, and
  visualization, none with substrate today (no graph store, scheduler, or
  multi-task lifecycle). Bounded fan-in (item 5) is not this alternative:
  a parent-waits barrier over its own children introduces no arbitrary edges.
- *Authority/security:* Adds undecided graph authority — who adds edges,
  authorizes a graph, bounds its size/cost — harder to bound than a tree.
- *Observability/tracing:* Rich eventually, but graph tracing and
  partial-failure semantics do not exist; edges would ship without
  trustworthy execution evidence.
- *Compatibility:* Low; assumes coordination, aggregation, and lifecycle
  roll-up beyond single-selection routing and serial agents.
- *v0.6→v0.7 coupling risk:* Total — v0.7 roadmap content under a v0.6 label.
- *Disposition:* Rejected for v0.6. Deferral is safe because lineage and
  dependency edges are distinct: adding edges later changes neither task
  identity nor root semantics.
