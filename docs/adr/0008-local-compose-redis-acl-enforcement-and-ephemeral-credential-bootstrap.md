# ADR 0008: Local Compose Redis ACL Enforcement and Ephemeral Credential Bootstrap

Status: Accepted
Date accepted: 2026-11-24
Related issue: https://github.com/Tiny-Robot-Lab/autonomous-agent-lab/issues/229
Supersedes: N/A

## Review status

Proposed after repository-grounded reconnaissance, human architecture review,
and an independent Gemini 4 Neon High review of the Issue #229 design. Neon
recommended approval with required revisions. The maintainer approved the
reconciled direction on 2026-11-24, including selector-based Redis ACLs,
per-service secret isolation, an ephemeral hashed ACL file on tmpfs, an exact
Redis image pin, a dedicated health-check identity, and mandatory client and
Orchestrator command audits.

After the draft was reviewed independently by Gemini 4 Neon High, the
maintainer accepted its blocking recommendations with two repository-grounded
corrections: source secret files remain host-restricted rather than becoming
world-readable, and the minimum Redis version is 7.2 because the selected
selector and key-permission syntax requires Redis 7.2 or later. The maintainer
explicitly accepted this revised ADR on 2026-11-24. Acceptance does not mean
the implementation already exists; implementation remains gated on the
redis-py handshake audit and the Orchestrator command/key/Lua audit defined
below.

## Context

Tiny Agent Lab uses one single-node Redis service in the local Docker Compose
deployment for authoritative task state, queues, wake-up hints, claims,
fencing epochs, continuations, artifacts, results, outcomes, expiry/recovery
records, routing records, and bounded operational events.

Application-level controls already govern authoritative task operations:

- runtime principals authenticate to the Orchestrator API;
- assignment proofs are detached from ordinary JSON payloads;
- fencing prevents stale assignments from committing authoritative state;
- claim, delegation, continuation, completion, expiry, and recovery
  transitions are platform-governed; and
- multi-key authoritative transitions execute through Orchestrator-owned
  single-node Redis Lua scripts.

The local Compose deployment nevertheless starts Redis without authentication
and gives every service the same unauthenticated URL. Agent runtimes are
therefore not technically prevented by Redis from bypassing the application
boundary and attempting direct writes to authoritative keys.

Issue #229 adds Redis-layer enforcement. The ACL boundary is additive defense
in depth. It does not replace HTTP authentication, detached assignment proof,
fencing, recovery governance, or Lua atomicity.

### Current direct runtime surface

Worker, Research Agent, and Analysis Agent Redis buses currently require only:

- `BLPOP` on their role task queue and role hint queue;
- `SADD` on shared `agent.ids`;
- `HSET` on their own `agent.state:{agent_id}` hash; and
- `XADD` on shared `agent.events`.

They do not require direct access to `task.*` keys or Redis scripting.
Canonical task operations cross the authenticated Orchestrator boundary.

### Security problem with command and key allowlists alone

Redis ACL command grants and key-pattern grants must be bound together. If a
runtime receives `BLPOP`, `SADD`, `HSET`, and `XADD` plus access to all of its
allowed keys in one undifferentiated rule, every granted command may be usable
against every granted key.

That creates a type-poisoning denial-of-service path. Redis removes an empty
list key. A compromised runtime could then recreate its drained queue using
`HSET`, `SADD`, or `XADD`, causing later Orchestrator `RPUSH` and runtime
`BLPOP` operations to fail with `WRONGTYPE`.

Redis ACL selectors are therefore required so each command is usable only on
its intended key or keys.

### Credential-bootstrap requirements

The local deployment needs distinct Redis users without committing plaintext
credentials or password hashes. It must also:

- start fail-closed;
- expose no unauthenticated Redis interval;
- keep each application container isolated to its own Redis credential;
- keep credentials out of `REDIS_URL`, command-line arguments, logs, and
  diagnostics;
- disable the Redis `default` user before accepting normal connections;
- authenticate the Compose health check without giving it key access;
- preserve AOF data independently of regenerated ACL material;
- support deterministic local testing and CI-created ephemeral identities;
  and
- provide a bounded local rotation and recovery procedure.

This ADR is limited to the verified local Docker Compose boundary. It does not
select a production secret manager, TLS/PKI design, multi-host orchestrator,
Redis Cluster design, or zero-downtime production rotation mechanism.

## Decision

### 1. Use distinct Redis identities

The selected Compose profile uses distinct ACL users for:

- `orchestrator`;
- `worker-1`;
- `research-agent-1`;
- `research-agent-2`;
- `analysis-agent-1`; and
- `healthcheck`.

Agent runtime credentials are per deployed instance, not shared by role. This
allows Redis to restrict each runtime to its own
`agent.state:{agent_id}` key while allowing the two Research instances to
consume the same Research queues.

The Redis `default` user is explicitly disabled:

```text
user default off resetpass resetkeys resetchannels -@all
```

All named users begin from a deny-by-default rule using at least:

```text
resetkeys resetchannels -@all
```

Normal application identities receive no ACL-management, server-
administration, persistence-administration, replication, module,
debugging, database-flush, or unrestricted scripting authority.

### 2. Use Redis ACL selectors for runtime command/key isolation

Every agent runtime ACL uses selectors that bind each allowed command to only
its intended keys. The architecture intent is:

```text
(+blpop ~<role-task-queue> ~<role-hint-queue>)
(+sadd ~agent.ids)
(+hset ~agent.state:<exact-agent-id>)
(+xadd ~agent.events)
```

A command must match one complete rule or selector. The `BLPOP` selector
contains exactly the two role queues so either queue order used by the runtime
is allowed, while a multi-key `BLPOP` mixing an allowed queue with any
unauthorized queue or `task.*` key fails atomically.

Separate selectors prevent a runtime from using `HSET`, `SADD`, or `XADD` to
recreate an emptied queue with the wrong Redis type, or from using queue
commands against state and observability keys.

Minimal connection-lifecycle commands confirmed by the redis-py handshake
audit in Decision 7, such as `PING` and `CLIENT SETINFO` only when client
metadata commands cannot be disabled, reside in the runtime user's root rule
alongside `resetkeys resetchannels -@all`. The root rule grants zero key or
channel patterns, so connection commands cannot interact with keys or weaken
selector isolation.

Identity names, agent IDs, and queue names used to render ACL patterns are
validated against whitespace, control characters, and Redis glob metacharacters
before rendering. They are trusted deployment configuration, not caller-
controlled task data.

### 3. Preserve shared observational keys as non-authoritative

`agent.ids` and `agent.events` remain shared for this increment. A compromised
runtime with its valid narrow grants can add spoofed presence identifiers and
append, flood, or aggressively trim observational events through the permitted
commands.

This residual risk is accepted for Issue #229 because these structures are
observational only. They must never be treated as lifecycle, permission,
authorization, routing-authority, assignment, claim, fencing, result, outcome,
expiry, recovery, or evidence-authority truth.

Redesigning presence or event ingestion is outside this ADR.

### 4. Select per-service Compose secrets and ephemeral ACL rendering

The local credential source is a git-ignored host directory:

```text
.secrets/redis/
```

The directory is mode `0700` to restrict host-level access to its owning user.
It contains one high-entropy secret file per identity generated by a version-
controlled helper using an allowlisted character set of at least 32 characters
with no whitespace, newlines, or control characters. Source secret files
remain mode `0600`; this ADR does not permit weakening them to world-readable
host modes.

Docker Compose top-level `secrets` provide per-container read-only single-file
bind mounts. Each application container receives only its own Redis password
file at a stable path such as:

```text
/run/secrets/redis_password
```

Non-Swarm Compose file secrets preserve host UID/GID/mode and do not provide
portable UID, GID, or mode remapping for file-backed secrets. Each application
and health-check container therefore uses a minimal root entrypoint solely to
read its mounted source secret, validate it, copy it into a container-private
ephemeral runtime path owned by the actual non-root service UID with mode
`0400` or `0600`, and then `exec` the service as that non-root user. The
application process itself does not run as root. The copied runtime secret is
never logged and is removed with the container. Container-packaging tests must
prove that the real non-root runtime UID can read only its own copied secret.

Each application service receives:

```text
REDIS_URL=redis://redis:6379/0
REDIS_USERNAME=<exact-identity>
REDIS_PASSWORD_FILE=<container-private-runtime-secret-path>
```

`REDIS_URL` remains credential-free. Redis clients load the password from
`REDIS_PASSWORD_FILE` and fail closed if the file is missing, unreadable,
empty, malformed, contains whitespace or newlines, or if `REDIS_USERNAME` and
`REDIS_PASSWORD_FILE` are partially configured. They pass username and
password as explicit client-construction arguments. Embedded credentials in a
Compose-mode `REDIS_URL` are rejected fail-closed and are also unconditionally
redacted before any validation error, log, traceback, or diagnostic output.
There is no silent fallback to unauthenticated Redis or to credentials embedded
in the URL.

Non-Swarm Compose file secrets are local file mounts, not an encrypted
production secret vault. That limitation is accepted for the local Compose
boundary because the mechanism provides per-service source-secret scoping and
keeps credentials out of environment blocks and command lines.

### 5. Render a hashed ACL file on container-local tmpfs

The Redis service receives the secret files required by the selected Compose
profile and mounts a dedicated memory-backed filesystem at:

```text
/run/redis
```

The tmpfs is bounded and configured with restrictive options including
`noexec`, `nosuid`, and `nodev` where supported by the selected Compose/runtime
combination.

A version-controlled Redis entrypoint performs these steps before Redis binds
its service port:

1. verify every identity declared by the selected profile has a secret file;
2. reject missing, unreadable, empty, short, malformed, whitespace-containing,
   newline-containing, or control-character-containing secrets;
3. compute each secret's SHA-256 digest;
4. combine the non-secret ACL policy/template with `#<64-hex-digest>` password
   rules, never plaintext `><password>` rules;
5. verify there is exactly one canonical
   `user default off resetpass resetkeys resetchannels -@all` directive and no
   competing `user default` line;
6. ensure `/run/redis` is traversable by the unprivileged Redis user, using
   `root:redis` mode `0750` or an equivalently restrictive verified layout;
7. render to a temporary file under `/run/redis`;
8. set the ACL file to `root:redis` mode `0440`, keeping it readable by Redis
   but not writable by the Redis process or world-readable;
9. atomically rename the temporary file to `/run/redis/users.acl`; and
10. replace the entrypoint shell with `redis-server` via `exec`, running Redis
    as the unprivileged Redis user and PID 1 with AOF data under `/data` and the
    ACL file under `/run/redis`.

The entrypoint never enables shell tracing and never prints credentials or
hashes. The ACL file never resides in `/data`, the AOF volume, or another
persistent mount. Redis starts with the ACL already loaded, eliminating a
post-start unauthenticated bootstrap interval. Direct PID 1 signal handling is
required so Compose shutdown and force-recreation deliver `SIGTERM` to Redis
for orderly AOF shutdown rather than leaving a shell wrapper to absorb it.

The exact Redis container image is pinned to an explicit reviewed Redis 7.2 or
later patch release that supports the selected ACL selector and key-permission
syntax. A moving tag such as `redis:7-alpine` is not accepted architecture.
Pinning by immutable image digest is recommended hardening but is not an ADR
acceptance gate.

### 6. Use a PING-only authenticated health-check identity

The `healthcheck` user has no key or channel patterns and receives only
`PING`:

```text
user healthcheck on #<sha256-digest> resetkeys resetchannels -@all +ping
```

A version-controlled health-check script uses the secret-readability and
container-private copy contract in Decision 4. It reads the copied secret into
a transient `REDISCLI_AUTH` variable scoped only to the `redis-cli` invocation,
sets `REDISCLI_HISTFILE=/dev/null`, invokes
`redis-cli --user healthcheck ping`, and succeeds only on exact `PONG` output.
It never exports the password into the static container environment, uses
`-a`, embeds the secret in a URL, or places the secret in command-line
arguments. Exact `PONG` matching prevents protocol or authentication errors
from being treated as healthy.

Redis becomes healthy only after it has started with the complete ACL. Compose
dependents continue to use `service_healthy`, so application services do not
race a privileged post-start bootstrap phase.

### 7. Audit redis-py connection behavior before freezing permissions

The repository pins `redis-py`, and client construction currently uses
`redis.from_url`. Before the runtime ACL text is finalized, implementation
must observe and test the actual initial connection, reconnect, pool health,
and database-selection commands emitted by the pinned client version.

Optional client metadata commands are disabled when practical. Otherwise,
only exact required subcommands, such as `CLIENT SETINFO`, are granted. The
broad `@connection` category is prohibited. `SELECT` is granted only if a
reviewed non-zero database requirement exists; the selected Compose design
uses database 0.

The audit records whether clients use RESP2 or RESP3. Authentication commands
needed to establish the connection are tested explicitly; any post-
authentication `HELLO` use under RESP3 must be included only if the observed
client contract requires it.

The minimum handshake permission is derived from evidence, not guessed from
client-library conventions.

### 8. Audit the complete Orchestrator Redis contract

Before final Orchestrator ACL syntax is written, record an exact audit of:

- every direct `RedisGateway` command;
- every pipeline/transaction command;
- every key namespace read or written;
- every Lua script and every `KEYS[]` argument supplied to it; and
- every `redis.call` or `redis.pcall` command used inside those scripts.

Every Lua-accessed key must be declared through `KEYS[]`. Redis ACLs are
validated at `EVAL` entry against the declared keys and again for every
`redis.call` or `redis.pcall` executed inside the script. Standard writable
scripts require read-write key permission for their declared keys unless a
script explicitly uses a verified no-write mode. Because `EVAL` and all
provided keys must satisfy one complete rule or selector, the Orchestrator's
audited Lua commands and explicit touched namespaces remain in its root
allowlist rule, never `~*`. Read-only key patterns may be used only for
namespaces that are never supplied as declared keys to a writable script.
The final ACL must allow every accepted transition and deny unrelated
namespaces and dangerous commands.

The Orchestrator starts from `-@all` and receives only audited commands and
key patterns. It never receives `~*`, `+@scripting`, `+@admin`, or
`+@dangerous`.

The current implementation uses direct `EVAL`. Therefore, `EVAL` is granted
only if the audit confirms that path. `EVALSHA` and `SCRIPT LOAD` are not
pre-granted. If implementation deliberately migrates to registered scripts,
the additional commands require explicit justification, validation, and ADR
consistency review. `SCRIPT FLUSH`, `SCRIPT KILL`, `SCRIPT DEBUG`, Redis
Functions, and module loading remain denied unless a later accepted decision
changes the architecture.

The Orchestrator does not receive ACL user-management authority during normal
runtime.

### 9. Keep test administration out of normal deployment

Default Compose, the normal ACL template, and normal application secret sets
define no test or administrative Redis user.

Live-Redis integration tests create isolated ephemeral identities and may add
a bounded fixture administrator only through test-specific configuration or a
test-only ACL fragment. No normal application container receives that secret,
and the test identity cannot appear in the default deployment policy.

Static tests verify this separation.

### 10. Use coordinated local rotation and force-recreation

Local credential replacement uses a documented coordinated workflow:

1. atomically generate replacement local source secret file(s);
2. force-recreate Redis with
   `docker compose up -d --force-recreate redis` so Docker binds the new secret
   inodes and regenerates the tmpfs ACL from the selected generation;
3. force-recreate affected application services with
   `docker compose up -d --force-recreate <services>` so they bind the new
   source-secret inodes and recreate their container-private runtime copies;
4. verify Redis health, application authentication, negative ACL boundaries,
   and non-secret credential-generation fingerprints; and
5. recover mismatched services by restoring the prior local generation or
   force-recreating the stale service with the selected generation.

A simple restart is not relied upon to refresh single-file secret bind mounts.
Compose file secrets bind individual host inodes, so atomic host-file
replacement does not alter a running container's existing mount until the
container is recreated.

Redis force-recreation drops live connections; clients reconnect and
reauthenticate. Zero-downtime dual-password rotation is not required for this
local boundary.

The Redis entrypoint and health check normally read the same mounted
health-check source secret generation. Atomic host replacement alone therefore
does not create live health-check drift inside an already-running container.
Health-check failure detects a deliberately mismatched or misconfigured mount,
an in-place same-inode mutation used only by a bounded negative test, or a
stale/mismatched credential introduced through explicit fault injection. A
stale application credential causes only the un-recreated or mismatched
application identity to fail authentication. All failures and diagnostics are
redacted.

AOF data remains on `/data` and is independent of regenerated ACL material.
Clean PID 1 `SIGTERM` handling, restart, and rotation tests must prove that
authoritative data survives while old credentials are rejected.

### 11. Fail closed and preserve existing authority boundaries

Redis startup fails before serving when required secret or ACL material is
missing or invalid. Application startup/reconnect fails when authentication or
required command permissions are absent. The system never falls back to an
unauthenticated or shared-credential mode.

Redis ACLs do not weaken or replace:

- runtime-principal authentication;
- detached assignment-proof validation;
- fencing and stale-turn rejection;
- authoritative Orchestrator APIs;
- recovery-operator authorization;
- atomic Lua transition validation; or
- task-scoped authoritative state ownership.

Agent runtimes have no `task.*` access and no Redis scripting capability.
They cannot directly evaluate expiry, restore claimability, write recovery
decisions, mutate canonical results or outcomes, or advance/bypass fencing.

### 12. Required verification gates

Implementation is not complete until deterministic tests prove:

- every selected runtime identity authenticates with only its own secret;
- no application container receives another application's secret;
- unauthenticated clients and the disabled `default` user cannot use Redis;
- each runtime can perform its required positive queue, state, presence, and
  event operations;
- cross-instance state writes and cross-role queue access fail;
- all runtime access to `task.*` fails;
- runtime scripting and server-administration attempts fail;
- cross-key type-poisoning attempts fail, including wrong command types on
  drained queues and shared/state keys;
- the two allowed `BLPOP` keys work in either order;
- a multi-key `BLPOP` mixing allowed and unauthorized keys fails atomically
  with no consumption;
- connection, reconnect, health-check, and pool behavior succeed with the
  exact pinned client and Redis versions;
- the health-check identity can only perform `PING`;
- the Orchestrator's complete audited positive contract succeeds;
- the Orchestrator is denied database flush, ACL/config/module/function
  administration, script flush, and out-of-namespace access;
- missing or malformed bootstrap secrets fail closed;
- `/run/redis/users.acl` is on tmpfs, mode `0600`, and contains hashes without
  plaintext credentials;
- default Compose contains an exact Redis image pin, credential-free URLs,
  explicit per-service secret isolation, and no test/admin identity;
- AOF data survives Redis restart and coordinated secret rotation;
- old and mismatched credentials fail safely;
- stale health-check credentials make the Redis service unhealthy; and
- real non-root application processes and the Redis health check can read only
  their own container-private copied secret under the verified UID/GID setup;
- embedded credentials in `REDIS_URL` and missing, empty, malformed, or partial
  username/password-file configuration fail closed with redacted output;
- secrets containing whitespace, newlines, control characters, or injected ACL
  directives fail before rendering, and malformed templates or missing or
  duplicate default-user directives fail before Redis accepts connections;
- `redis-server` replaces the wrapper with `exec`, runs as PID 1, and receives
  graceful shutdown signals during force-recreation;
- attempts to authenticate as another identity using the caller's own secret
  fail;
- health-check mismatch tests inject an actual mismatch rather than assuming an
  atomic host rename changes a running bind mount; and
- logs, tracebacks, settings, health output, and diagnostics reveal no Redis
  credential.

Existing claim, delegation, continuation, completion, expiry/recovery,
routing, container-packaging, and end-to-end Redis suites must remain green
under authenticated identities.

### 13. Required implementation artifacts

Implementation will include, with exact paths selected during implementation:

- a non-secret version-controlled ACL policy/template;
- a fail-closed Redis entrypoint;
- an authenticated health-check script;
- a local secret-generation and rotation helper;
- ignored local secret paths and safe examples containing no live secrets;
- client configuration that reads username and password file separately from
  the credential-free URL;
- the recorded Orchestrator command/key/Lua audit;
- static Compose/ACL invariant tests;
- live allow/deny and failure/recovery tests; and
- local setup, rotation, diagnosis, and recovery documentation.

No application implementation begins until this ADR is independently reviewed
and explicitly accepted, and both the Orchestrator command/key/Lua audit and
the redis-py handshake audit are complete enough to freeze the ACL allowlists.

## Consequences and tradeoffs

Benefits:

- Redis technically enforces the application architecture's existing
  Orchestrator-only authority over `task.*` state.
- Per-instance credentials limit credential compromise to one runtime's narrow
  queue, own-state, presence, and observational-event permissions.
- ACL selectors prevent cross-key type poisoning that a simple
  command-plus-key allowlist would permit.
- Loading the ACL before Redis accepts connections removes the insecure
  post-start bootstrap window.
- Per-service secret mounts keep the Orchestrator credential out of runtime
  containers and out of environment inspection.
- Hashed ACL rules on tmpfs keep plaintext and rendered ACL material out of Git,
  AOF persistence, and normal host backups.
- A dedicated health-check identity verifies authentication without granting
  data access.
- The design is reproducible in local Compose and isolated integration tests.

Costs and limitations:

- Local setup requires secret generation before Compose startup.
- The Redis container must receive all identities selected by the Compose
  profile so it can render the complete ACL; compromise of the Redis container
  remains outside the isolation provided to application containers.
- Coordinated rotation recreates Redis and affected services and is not
  zero-downtime.
- Non-Swarm Compose secrets are local file mounts, not encrypted secret-vault
  delivery.
- The entrypoint, ACL template, health-check script, and static policy tests add
  operational code that must be maintained.
- Exact ACLs are coupled to audited Redis commands, client handshake behavior,
  key namespaces, and Lua scripts. Changes to those surfaces require tests and
  policy updates.
- Shared `agent.ids` and `agent.events` remain forgeable or trimmable by a
  compromised runtime within its allowed commands. They remain explicitly
  non-authoritative.
- TLS on the local bridge network is not provided by this decision.
- This design makes no production secret-management, multi-host, orchestrator,
  or Redis Cluster claim.

## Alternatives considered

### 1. Post-start `ACL SETUSER` bootstrap

Redis starts in a temporary bootstrap state and a privileged process creates
users, applies permissions, disables the default/bootstrap identity, and then
signals readiness.

- **Advantages:** ACL changes can be expressed through Redis commands and may
  support dynamic updates.
- **Costs and risks:** Creates an insecure or operationally complex bootstrap
  phase, introduces startup races, requires a privileged client holding all
  credentials, complicates partial-failure recovery, and risks retaining
  bootstrap authority.
- **Disposition:** Rejected for the local Compose baseline.

### 2. Host-generated complete ignored ACL file

A local command generates a complete `users.acl` file on the host and mounts it
read-only into Redis.

- **Advantages:** Redis can start with authentication already active; the
  container entrypoint is simpler.
- **Costs and risks:** The combined artifact persists on the host, may become
  stale or manually broadened, can leak through backups or support bundles,
  and creates drift between service credentials and the mounted ACL.
- **Disposition:** Rejected in favor of rendering hashes on container-local
  tmpfs from isolated source secret files.

### 3. Shared authenticated Redis credential

Every service authenticates with one password, optionally with `requirepass`.

- **Advantages:** Minimal configuration and easy migration from the current
  shared URL.
- **Costs and risks:** Provides authentication but no least-privilege separation.
  Any compromised runtime can impersonate the Orchestrator and bypass the
  intended task-state boundary.
- **Disposition:** Rejected.

### 4. Role-level runtime users

All instances of a role share one Redis ACL identity, for example both
Research Agents use `research-agent`.

- **Advantages:** Fewer secrets and simpler Compose configuration.
- **Costs and risks:** Prevents exact per-instance state-key isolation and makes
  revocation, compromise attribution, and rotation coarser.
- **Disposition:** Rejected for the current fixed local topology.

### 5. Broad command categories with deny overrides

Users receive broad categories such as `+@connection`, `+@write`,
`+@scripting`, or `+@all` followed by selected denials.

- **Advantages:** Less likely to break when clients or code add commands.
- **Costs and risks:** Least privilege becomes difficult to reason about;
  category membership changes and overlooked commands silently expand
  authority. Broad categories expose commands unrelated to the accepted
  runtime behavior.
- **Disposition:** Rejected. Use explicit commands and selectors.

### 6. Plaintext passwords in a tracked or persistent ACL file

The repository or persistent Redis volume contains `><password>` rules.

- **Advantages:** Straightforward Redis configuration.
- **Costs and risks:** Commits or persists reusable credentials and increases
  leakage through history, backups, diagnostics, and support artifacts.
- **Disposition:** Rejected.

### 7. TLS for local Redis traffic in Issue #229

Redis and all clients use locally managed certificates and `rediss://` on the
Compose bridge.

- **Advantages:** Protects local container traffic from same-host network
  observation.
- **Costs and risks:** Adds CA, certificate issuance, trust-store, expiry,
  rotation, and debugging complexity unrelated to the immediate Redis
  authorization gap.
- **Disposition:** Deferred. This ADR neither requires nor prohibits a later
  separately reviewed local TLS decision.

### 8. Zero-downtime dual-password rotation

Each user temporarily accepts old and new password hashes while services
rotate independently.

- **Advantages:** Avoids coordinated downtime and stale-client failures.
- **Costs and risks:** Requires `ACL LOAD` or another privileged live update
  path, dual-generation lifecycle management, rollback rules, and additional
  operational authority.
- **Disposition:** Deferred. Coordinated local recreation is sufficient for
  the Issue #229 boundary.
