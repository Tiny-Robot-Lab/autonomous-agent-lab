# ADR 0009: Disabled Default User Write-Only AOF Replay Authority

Status: Accepted
Date accepted: 2026-11-24
Related issue: https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/229
Supersedes: ADR 0008 Decisions 1, 5, and 12 only where they specify or test
a fully permissionless disabled `default` user; ADR 0008 Decision 6 is
qualified by the replay-integrity health gate in this ADR

## Review status

Accepted by explicit human decision on 2026-11-24 after a bounded
repository-external Redis 7.4.12 experiment, independent Gemini 4 Neon High
review, validation of Neon's Redis-specific findings against upstream Redis
behavior, and incorporation of all five required revisions.

The experiment reproduced silent data loss under ADR 0008's fully
permissionless disabled `default` user, validated a disabled write-only replay
rule against a representative fixture with subtraction tests, verified
persistence across coordinated credential rotation, and confirmed that
ordinary clients still could not authenticate or use `default`. Full
subtraction coverage across every command and namespace in the bounded
production replay envelope remains a mandatory implementation gate.

Acceptance does not mean the implementation already exists. Issue #229
implementation remains blocked until the replay rule, static completeness
gate, fail-closed startup replay-integrity gate, AOF lifecycle tests, and ADR
0008 partial-supersession annotation are implemented and validated.

## Context

ADR 0008 selects per-service Redis identities, deny-by-default ACLs, a disabled
`default` user, an ACL rendered on tmpfs before Redis accepts connections,
and AOF data persisted separately under `/data`.

Its accepted canonical default-user rule is:

```text
user default off resetpass resetkeys resetchannels -@all
```

Issue #229 Increments 1 through 3 established the fail-closed Redis client
boundary, bootstrap infrastructure, and a live Redis 7.4.12 ACL verification
matrix. All production identities, the Orchestrator direct-command and
transaction contract, and all production Lua scripts succeeded without broad
permissions.

Increment 4 then tested coordinated credential rotation and preservation of
representative authoritative state across graceful Redis recreation. Redis
persisted the state to AOF, received normal `SIGTERM` as PID 1, and restarted
with the same disposable data volume and newly rendered credentials. During
AOF replay, however, Redis emitted critical `NOPERM` errors from its internal
AOF-loading client. Persisted task state was missing after startup even though
Redis became ready.

The failure was not caused by the live Orchestrator identity. In
`redis:7.4.12-alpine`, `loadSingleAppendOnlyFile()` executes plain,
non-transactional AOF commands through the internal `CLIENT_ID_AOF` loading
client without ordinary per-command ACL checks. However, `execCommand()`
revalidates every command queued inside a `MULTI`/`EXEC` block through
`ACLCheckAllPerm()` without exempting `CLIENT_ID_AOF`. The loading client
inherits `DefaultUser` authority. Redis also propagates explicit Orchestrator
transactions and Lua script side effects to AOF as write commands wrapped in
`MULTI`/`EXEC`. A permissionless disabled `default` user therefore causes the
queued transactional and Lua-produced writes to be rejected with `NOPERM` and
the entire affected transactions to be discarded while Redis can still reach
readiness.

This mechanism explains why the replay rule does not need `+multi`, `+exec`,
or `+eval`, and why plain non-transactional writes such as runtime
`HSET agent.state:*`, `SADD agent.ids`, and `LPOP` propagated from `BLPOP` are
not part of the replay rule unless a future implementation moves them inside a
transaction or Lua script.

A bounded experiment using `redis:7.4.12-alpine` compared ADR 0008's
permissionless rule with a disabled `default` user carrying narrowly scoped
write-only replay selectors. The permissionless rule produced `NOPERM` drops
and incomplete state. The write-only candidate replayed the representative
fixture with zero errors, survived further authoritative work and a second
coordinated rotation, rejected old credentials, accepted new credentials, and
rejected every tested unauthenticated and `default` authentication attempt.

This ADR is a version-pinned local-Compose workaround for Redis 7.4.12. It may
be retired in favor of ADR 0008's fully permissionless default user only after
a future Redis release is verified to exempt the AOF-loading client from
`EXEC` ACL revalidation and passes all replay, isolation, and persistence
gates in this ADR.

This decision addresses only the single-node local Compose AOF replay boundary.
It does not authorize ordinary clients, create a general persistence identity,
or replace the named production users selected by ADR 0008.

## Decision

### 1. Keep the default user externally disabled

The Redis `default` user remains:

- `off`;
- without a password through `resetpass`;
- without channel access through `resetchannels`;
- denied every command by the root rule through `-@all`;
- without any root key pattern; and
- unavailable to ordinary unauthenticated clients or `AUTH`.

The default user is not a production application identity. No application
receives a credential for it, because no credential exists.

Ordinary unauthenticated connections must continue to receive `NOAUTH`.
One-argument `AUTH`, two-argument `AUTH default ...`, and `HELLO ... AUTH
default ...` must fail for empty, arbitrary, and production-identity
passwords.

### 2. Grant only bounded transactional and Lua write-only AOF replay authority

The disabled `default` user receives a bounded production replay envelope of
write-only command and key selectors projected from the complete audited
transactional (`MULTI`/`EXEC`) and Lua-propagated write inventory. This is
distinct from the smaller fixture-minimal rule proven by the original
subtraction experiment.

Plain non-transactional persisted writes bypass `execCommand()` ACL
revalidation in Redis 7.4.12 and are intentionally excluded. `+multi`, `+exec`,
and `+eval` are also omitted because Redis replays Lua effects as
`MULTI`/`EXEC`-wrapped write commands and rechecks the queued inner commands.

The initial bounded production-envelope rule is:

```text
user default off resetpass resetkeys resetchannels -@all +select \
  (+set \
    %W~task.registration:* \
    %W~task.status:* \
    %W~task.lineage:* \
    %W~task.input:* \
    %W~task.authority:* \
    %W~task.authority_grant:* \
    %W~task.required_capability:* \
    %W~task.artifact:* \
    %W~task.continuation:* \
    %W~task.continuation_latest:* \
    %W~task.claim:* \
    %W~task.claim_deadline:* \
    %W~task.claim_expiry:* \
    %W~task.fence:* \
    %W~task.delegation_proposal:* \
    %W~task.assignment_yield:* \
    %W~task.completion:* \
    %W~task.outcome:* \
    %W~task.result:* \
    %W~task.assignment_completion:* \
    %W~task.child_completion:* \
    %W~task.recovery_decision:* \
    %W~task.routing:*) \
  (+del %W~task.claim:*) \
  (+sadd \
    %W~task.children:* \
    %W~task.artifacts:* \
    %W~task.consumed_waves:* \
    %W~task.consumed_grants:* \
    %W~task.delegation_decisions:*) \
  (+rpush \
    %W~agent.hints.general \
    %W~agent.hints.research \
    %W~agent.hints.analysis \
    %W~agent.tasks.general \
    %W~agent.tasks.research \
    %W~agent.tasks.analysis) \
  (+xadd %W~agent.events)
```

The rendered ACL must express an equivalent single Redis ACL line. The line
breaks above are documentation formatting only.

The authority contains no read command and no read-capable key pattern. It
does not grant `GET`, `EXISTS`, `SMEMBERS`, `HGETALL`, `XRANGE`, `XREVRANGE`,
`EVAL`, or another application read path.

Write-only `%W~` patterns are valid only for commands whose Redis 7 key
specifications do not require read access. The accepted envelope is limited to
plain `SET` without `GET`, `DEL`, `SADD`, `RPUSH`, and `XADD`. The audited
transactional and Lua paths currently delete only `task.claim:*` and do not
emit `UNLINK`, `SREM`, `HSET`, `LPOP`, `XTRIM`, `SET ... GET`, or separate
expiration commands such as `EXPIRE`, `PEXPIRE`, `EXPIREAT`, or `PEXPIREAT`.
Any future introduction of those command shapes inside a transaction or Lua
script requires an audited replay-rule update before release. Commands such as
`LPOP` that both read and modify a key would require read-write, not
write-only, key permission.

### 3. Permit `SELECT` only for replayed database selection

Redis AOF may contain `SELECT 0`, including when a transaction or Lua-backed
transition is the first persisted write after startup. The experiment proved
that replay remained incomplete until `+select` was present.

The grant does not change the selected application database. Production
clients remain configured for database 0 and are not granted `SELECT` through
their named identities. Tests must continue to reject nonzero application
databases.

Redis 7.4 provides no per-database ACL selector, so `+select` permits internal
`SELECT` for any database index during replay rather than enforcing database 0
at the ACL layer. Database-0 confinement remains a named-client configuration
and validation invariant.

### 4. Enforce replay completeness and fail-closed startup integrity

The write-only replay rule is an explicit, version-controlled projection of
the complete audited transactional and Lua-propagated write inventory. It must
not be copied from the broader live Orchestrator ACL without review.

Exact-PONG health proves only Redis liveness and healthcheck authentication. It
does not detect `NOPERM` transaction drops during AOF loading. Such drops can
silently remove security-governing state including fences, consumed grants,
claim expiry, and recovery decisions while Redis still reports ready.

Implementation therefore requires both mandatory gates:

1. **Static inventory completeness.** Automated tests must verify that 100% of
   audited transactional and Lua-propagated write commands and key families
   are covered by the disabled-default replay rule. They must also prove the
   rendered default rule retains `off`, `resetpass`, `resetkeys`,
   `resetchannels`, and `-@all`, and contains no `on`, `nopass`, plaintext
   password, or password-hash directive.
2. **Bounded fail-closed startup replay integrity.** Local Compose must not
   treat Redis as healthy if AOF loading emits `CRITICAL` or `NOPERM` replay
   errors. The implementation must capture AOF-load output in a bounded,
   container-local, ephemeral location and make the health decision inspect
   that evidence while preserving the healthcheck identity's `PING`-only ACL,
   operator-visible Redis logs, PID 1 signal handling, and clean AOF flush on
   `SIGTERM`.

Any implementation change that can add or move a persisted write into a
transaction or Lua script, including deletion or TTL behavior, must update in
the same bounded change:

1. the Orchestrator command/key/Lua audit;
2. the named Orchestrator ACL when required;
3. the disabled-default replay rule;
4. the static completeness gate;
5. the live ACL matrix; and
6. live AOF replay and subtraction tests.

A missing replay permission must never be resolved through `+@all`, `~*`,
broad command categories, read permissions, or runtime ACL mutation.

### 5. Pin the validated Redis behavior

This decision is validated against `redis:7.4.12-alpine`. The local Compose
image remains pinned to that exact patch release until a later change repeats:

- AOF replay failure reproduction under the permissionless rule;
- zero-error replay under the write-only rule;
- command and key subtraction tests;
- unauthenticated and `default` authentication isolation;
- named-production-identity regression tests;
- graceful restart and AOF preservation; and
- coordinated credential rotation with old-credential rejection.

A Redis upgrade is not routine dependency maintenance while this replay
boundary depends on version-specific behavior.

### 6. Preserve ACL and AOF separation

The rendered ACL remains on verified `/run/redis` tmpfs as `root:redis` mode
`0440`. The replay authority is not persisted in `/data` or the AOF volume.
AOF data remains under `/data` and is limited by the accepted single-node local
Compose durability claim.

No password, password hash, ACL template, or rendered ACL may appear in AOF,
logs, diagnostics, command arguments, or committed fixtures.

Redis retains its validated multi-part AOF configuration with
`aof-use-rdb-preamble yes`: an RDB base plus an incremental AOF tail. Truncated
or incomplete tail handling remains bounded by the configured
`aof-load-truncated` behavior. This ADR makes no unconditional crash-durability,
disaster-recovery, or loss-free unclean-shutdown claim.

### 7. Preserve named production identities and platform authority

The disabled-default replay rule does not replace or broaden the named
production identities:

- Orchestrator continues to use its audited named ACL identity;
- worker and agent runtimes continue to use per-instance selector-bound
  identities;
- healthcheck continues to use its `PING`-only identity; and
- fixture administration remains isolated to disposable test configuration.

ACL replay authority reconstructs persisted Redis state only. It grants no
Tiny Agent Lab task authority, identity, assignment, proof, fencing,
delegation, completion, expiry, recovery, or routing decision.

Replayed state is not automatically valid or authorized merely because Redis
accepted its commands. Post-restart reads, claims, continuations, fencing
checks, and recovery transitions remain subject to Orchestrator HTTP
authentication, detached assignment proofs, current fencing epochs, and
platform recovery governance.

### 8. Required verification

Implementation is incomplete until deterministic tests prove:

- the previously accepted permissionless default rule reproduces AOF replay
  `NOPERM` and missing/incomplete state;
- the write-only rule replays representative direct, transactional, queue,
  set, stream, and Lua-produced state with zero `CRITICAL` or `NOPERM` replay
  errors;
- all expected fixture data is identical before and after restart;
- continued authoritative work after replay persists through a second restart;
- every command and every key family in the bounded production replay
  envelope is exercised inside a `MULTI`/`EXEC` or Lua replay fixture and fails
  individually when removed from the replay rule;
- no unneeded or untested envelope permission remains;
- static tests prove the default rule retains `off`, `resetpass`, `resetkeys`,
  `resetchannels`, and `-@all`, contains no `on`, `nopass`, plaintext password,
  or password hash, and covers 100% of audited transactional and Lua writes;
- `BGREWRITEAOF` with the multi-part RDB base plus a subsequent transactional
  incremental AOF tail replays cleanly with exact state preservation;
- truncated or incomplete incremental AOF-tail behavior is explicitly tested
  and documented;
- subtracting a required selector causes the startup replay-integrity gate to
  refuse healthy status rather than accepting silent state loss;
- static and audit tests prove that no unaudited expiration command, `UNLINK`,
  or non-claim `DEL` target appears inside transactional or Lua write paths;
- unauthenticated `PING`, reads, writes, transactions, scripting, connection
  metadata, and ACL introspection fail;
- all tested attempts to authenticate as `default` fail;
- named production identities retain their accepted allow/deny behavior;
- production identities cannot perform ACL, configuration, persistence,
  replication, module, restore, migration, or dangerous administration;
- old credentials fail and new credentials succeed after coordinated
  rotation;
- ACL material remains on tmpfs and AOF data remains on the disposable data
  volume;
- no credential or rendered hash appears in AOF or output; and
- all disposable containers, networks, images, volumes, and temporary files
  are removed without touching existing project data.

The full write inventory and replay tests must be version-controlled and
reproducible, not retained only as a one-time experiment transcript.

### 9. ADR 0008 supersession boundary

This ADR supersedes ADR 0008 Decisions 1, 5, and 12 only where ADR 0008
specifies, validates, or tests a fully permissionless disabled `default` user.
It qualifies ADR 0008 Decision 6 by requiring exact-PONG health to be paired
with the bounded AOF-load replay-integrity gate in Decision 4 above. ADR 0008
must be annotated as partially superseded by ADR 0009.

ADR 0008 remains governing for:

- per-service and per-instance identities;
- credential-free `REDIS_URL`;
- file-based secret delivery;
- fail-closed client configuration;
- ACL selectors for runtime command/key isolation;
- the audited Orchestrator ACL;
- tmpfs ACL rendering;
- `root:redis` mode `0440`;
- authenticated health checking;
- coordinated local force-recreation;
- fixture-admin exclusion from production;
- local Compose scope and non-claims; and
- every other accepted invariant not explicitly changed here.

## Consequences and tradeoffs

Benefits:

- AOF reconstruction succeeds without creating an externally authenticatable
  default identity.
- Replay authority is write-only and limited to audited persisted commands and
  namespaces.
- Production identities remain least-privilege and unchanged.
- Credential rotation can preserve authoritative state while rejecting old
  credentials.
- The fixture-minimal rule is subtraction-proven; the broader bounded
  production envelope is governed by complete static inventory and live
  subtraction gates.

Costs and limitations:

- The ACL policy now contains an internal replay authority that is distinct
  from application authority and must be maintained explicitly.
- Redis upgrades require replay and isolation revalidation.
- The replay selector inventory is coupled to persisted-write evolution.
- A missed permission can allow Redis to become ready after incomplete AOF
  reconstruction unless startup logs and state verification fail release
  validation.
- The mechanism relies on Redis 7.4.12 behavior in which `execCommand()`
  rechecks ACLs for queued AOF-loading-client commands while an `off` default
  user remains unavailable to external clients. A future Redis release may
  change that behavior.
- This remains a single-node local Compose persistence design, not an
  unconditional durability guarantee.

## Alternatives considered

### 1. Keep the default user fully permissionless

This preserves the original ADR 0008 wording but causes Redis 7.4.12 AOF
replay `NOPERM` errors and incomplete authoritative state.

Disposition: Rejected.

### 2. Enable the default user

An enabled default user could make replay permissive but risks granting
unauthenticated clients its authority and contradicts the selected external
authentication boundary.

Disposition: Rejected.

### 3. Give the default user a password

A password would create a usable authentication principal and a new credential
lifecycle without being required by the internal AOF-loading client.

Disposition: Rejected.

### 4. Grant the default user the full Orchestrator ACL

The live Orchestrator rule includes read operations and wider authority than
AOF reconstruction requires. The experiment's subtraction tests demonstrated
that replay can use a narrower write-only projection.

Disposition: Rejected.

### 5. Grant broad categories or unrestricted keys

Rules such as `+@all`, `+@write`, `+@scripting`, or `~*` would reduce
maintenance but create authority unrelated to the observed persisted command
surface.

Disposition: Rejected.

### 6. Mutate ACLs during or after startup

Using `ACL SETUSER`, `ACL LOAD`, a privileged bootstrap client, or a temporary
replay ACL creates a live administrative path and startup-state transition that
ADR 0008 deliberately avoids.

Disposition: Rejected.

### 7. Disable AOF persistence

Removing persistence avoids replay but abandons the accepted durable
single-node local deployment behavior and loses state on recreation.

Disposition: Rejected.

### 8. Use a separate replay user

Redis 7.4.12 AOF loading uses the effective default-user authority in the
validated design. A separate named user would not solve replay without an
additional mechanism to select that user internally.

Disposition: Rejected for the validated boundary.

## Deferred questions

These questions do not permit weakening the selected rule:

- Should the replay inventory be generated mechanically from a reviewed policy
  manifest, or remain a manually reviewed explicit ACL line?
- Which future Redis version should be evaluated before upgrading from
  7.4.12?
