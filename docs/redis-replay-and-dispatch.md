# Redis Replay and Dispatch Operations

The governed local Compose deployment uses Redis 7.4.12 only. It disables the
external `default` user and grants a narrow write-only replay projection derived
from production transaction and Lua framing. Application health requires an
exact authenticated `PONG`, a positive current-startup replay marker, a clean
bounded replay log, and a generation binding — PING alone is insufficient.

## Replay audit

Run:

```sh
.venv/bin/python scripts/audit_redis_replay.py
```

The audit independently derives every qualifying production command/family
pair from the governed gateway: transactional pipeline writes are associated
with their actual key arguments, and every Lua `redis.call` is resolved
through `KEYS[n]` aliases and each production invocation to canonical key
families. Lua framing is classified by path-sensitive worst-case analysis:
sequential statements accumulate, each `if` contributes the maximum of its
reachable branches (including the implicit zero branch without `else`), early
`return` terminates its path, read-only calls and read-only loops contribute
nothing, and mutating loops, dynamic command construction, or unclassifiable
control flow fail closed toward multi-effect or gate failure. A script is
single-effect only when every reachable path propagates at most one mutation
with no mutating loop, no dynamic construction, and no outer-transaction
invocation. The union of command/family pairs covers every reachable branch,
so a script can remain single-effect per path while justifying several pairs.
The derived pairs must match the disabled-default ACL selectors exactly in
both directions: no missing selectors, no extra selectors, exact command
association, exact write-only `%W~` mode. Template, entrypoint canonical
rule, and artifact must agree; agreement alone is not correctness.

Redis ACL commands and categories are case-insensitive, so the semantic
rule parser folds every structural token, command token, category token,
and parenthesized command-selector group to lowercase before comparison.
Uppercase and mixed-case variants of forbidden authority (`+MULTI`,
`+EXEC`, `+EVAL`, `+UNLINK`, `+@ALL`, `+@WRITE`, `+@CONNECTION`,
`+@SCRIPTING`, `+@ADMIN`, `+@DANGEROUS`, `USER DEFAULT ON`, `NOPASS`,
`~*`) are rejected exactly like their lowercase forms. Supported replay
groups (`SET`, `DEL`, `SADD`, `RPUSH`, `XADD`) parse case-insensitively
while the `%W~` selector mode stays exact. Unknown parenthesized groups,
malformed groups, unsupported selector modes, and any trailing or embedded
token the parser cannot semantically classify fail closed; an unknown
group is never silently skipped in the hope that a later missing-selector
error catches it. Read-only (`%R~`) selectors, ordinary default-user key
access, unjustified TTL/expiry authority, non-claim `DEL`, runtime ACL
mutation, and administrative/server-management authority remain rejected,
and `task.routing:*` in any capitalization stays out of default replay
authority.

`infrastructure/redis/replay_projection.json` is reviewable evidence of that
derivation (execution sites, per-script path-sensitive classification,
pair union, qualifying and excluded pairs, the separately justified
`+select`). Artifact equality only detects code/policy drift; the semantic
proof is re-derived from production code on every run.

## Governed hint destinations

`RedisGateway.dispatch_hint()` is bound to the same finite-contract
principle as legacy dispatch: a governed capability-to-hint mapping
(`general` -> `agent.hints.general`, `research` -> `agent.hints.research`,
`analysis` -> `agent.hints.analysis`). Authorization binds to the
authoritative Orchestrator-supplied capability, never to a
runtime-self-advertised capability or the untrusted target text. The target
is validated before normalization by exact raw equality against the
canonical approved destination — padded, normalized-to-valid,
control-character, glob, prefix/suffix-spoof, cross-capability, and
unrelated destinations (`agent.tasks.*`, `task.*`, `agent.events`,
`agent.ids`, `agent.state:*`) all fail closed before any Redis write. An
omitted (`None`) target resolves from the authoritative capability alone.
Only the canonical approved constant reaches `RPUSH`, so invalid calls
execute zero Redis writes. Custom deployments must inject an explicit
finite contract; there is no production validation-disable mode, and the
contract is never sourced from runtime-writable Redis state.

The audit records the governed ordinary hint `RPUSH` pairs as excluded
ordinary evidence. They do not justify default replay authority by
themselves; transactional or multi-effect Lua paths may independently
justify the same destination selectors. Any future unresolved dynamic
production ordinary write fails the completeness gate instead of sitting
in an unresolved bucket.

## Current-startup replay integrity

At startup the entrypoint truncates `/run/redis/replay-integrity.log` on the
verified tmpfs, records `/run/redis/startup.json` as exactly
`{"generation_sha256":"<digest>"}` (the active credential-generation
digest only, byte-identical to the entrypoint `printf` shape — no random
identity, no legacy `start_id`), removes any stale sealed verdict, and
always appends the bounded `--logfile … --loglevel notice` options to the
final Redis command, regardless of caller-supplied Redis arguments. A
conflicting caller log path or level fails startup closed. Replay-test
harnesses must write the same exact marker shape on the primary path.
Compatibility tolerates extra obsolete fields only; `generation_sha256`
remains mandatory and missing, malformed, or mismatched generations fail
closed. Ownership model:
root owns every runtime file so the redis user can never replace evidence,
but the replay log itself is `root:redis 0660` — group-writable so the redis
process can append, never world-accessible. The ACL file, startup marker,
and sealed verdict stay read-only. Disposable harness startups and replay
containers apply the same preparation in a root bootstrap before dropping to
the redis user via `setpriv`; Redis never runs as root.

This is an exec-bound fresh-log proof, not a process-ID binding: no random
startup identity is recorded, because health only checks non-emptiness for
such an identifier and that would attest nothing. The proven invariant is
narrower and truthful — verified tmpfs, truncated log, recorded generation
digest, `exec redis-server`, Redis's own readiness statement in that same
fresh log, and authenticated PING against the matching generation
credentials. A same-generation marker from another start cannot satisfy a
new startup: the new start truncates the log, refreshes the marker, and
drops the seal, so only the new startup's replay evidence can pass.

Health requires all of the following for the current startup:

1. a well-formed startup marker whose generation digest matches the active
   `credential-generation.json`;
2. a present, regular replay log newer than the marker;
3. a sealed verdict: `replay-sealed.json` binding the active generation to
   a `window_sha256` that matches the retained startup evidence
   (`replay-startup-evidence.log`), where that evidence is bounded (1 MiB,
   sized for startup output which scales with file count, not dataset
   size), carries the readiness boundary, and contains no `CRITICAL`,
   `NOPERM`, or AOF load-failure evidence. When the live log still carries
   a startup banner the recomputed window must match the seal, so a newer
   startup always re-proves itself and only then is a fresh seal written;
   when no banner remains (truncated log after a sealed startup) the
   sealed artifact stands alone;
4. an exact authenticated `PING` from the keyless healthcheck identity,
   verified on every run including sealed steady states.

The startup window (after the last `is starting` banner through the first
subsequent `Ready to accept connections`) is extracted once and retained as
the bounded evidence artifact; it is never re-derived from a continuing log
that truncation or rotation may have emptied. When the live log carries a
banner, the recomputed window must match the seal, so a newer startup always
re-proves itself; a stale seal can never satisfy a new window. When no
banner remains (truncated log after a sealed startup), the sealed artifact
stands alone — truncation never invalidates a valid seal. The sealed
verdict covers only the pre-readiness prefix: it is immutable for that
start, while later runtime lines keep appending to the same log. A runtime
`NOPERM` after readiness is proven by the client response (Redis 7.4.12
does not emit post-readiness ACL denials at notice level); it stays out of
the sealed verdict by construction. Continuing log growth past the
operational bound (8 MiB, protecting the shared 16 MB tmpfs) is truncated,
never reported as a replay failure, and never invalidates a valid seal; the
bounded startup evidence is preserved separately. A stale log can never
poison a later start (truncation at startup), and a missing marker, stale
log, mismatched generation, absent boundary, oversize or tampered evidence,
failed ownership, or denial marker fails closed. The banner, denial
strings, and readiness statement are pinned Redis 7.4.12 behavior; the live
subtraction test pins the actually observed strings and must be revisited
if Redis emits different ones.

## Legacy dispatch

Legacy dispatch uses a finite capability-to-task-queue contract:

- `general` -> `agent.tasks.general`
- `research` -> `agent.tasks.research`
- `analysis` -> `agent.tasks.analysis`

The target is authorized only by the authoritative `required_capability`
binding. Runtime-advertised destinations are untrusted exact strings. Custom
deployments must inject an explicit finite contract; there is no production
validation-disable mode.

## Restart and replay diagnosis

For pinned Redis 7.4.12:

1. Check `docker compose ps redis` for health.
2. Inspect the current startup window — after the last `is starting` banner
   through the readiness statement — for `CRITICAL`, `NOPERM`, or AOF load
   failures; the window covers only the current startup, while later runtime
   lines remain observable in the continuing log. A runtime `NOPERM` is
   proven by the client response, not by the notice log.
3. Verify `startup.json` generation matches the active generation, and that
   `replay-sealed.json` matches the retained `replay-startup-evidence.log`.
4. Recreate Redis only after credentials, ACL material, and the active
   generation are verified.
5. Verify authoritative state through the normal application APIs, not through
   the replay identity.

Health PING alone is insufficient. A stale, partial, or replay-incomplete
startup must leave Redis unhealthy and dependent services stopped.

## Live proofs: entrypoint control versus subtraction test

Two live proofs stay distinct. The production-entrypoint control
(`tests/test_redis_entrypoint_live.py`) boots Redis through the actual
`redis-acl-entrypoint.sh` and proves real bootstrap, permissions,
executable path, marker shape, logging enforcement, privilege drop, and
health fidelity — including a pre-readiness health denial, a sealed
healthy startup, and fail-closed conflicting log options. It never varies
the ACL. The harness-assembled selector-subtraction test
(`tests/test_redis_replay_health_live.py`) replays the same persisted AOF
under full versus narrowed rules to prove exact replay-policy subtraction
behavior; it must not pretend to be a full production-entrypoint test. A
static parity test fails visibly if harness-required Redis options drift
from production-enforced ones.

## Explicit audit fail-closed behavior

Beyond the bidirectional selector equivalence, the audit fails closed on:
unresolved dynamic production ordinary writes; pipelines acquired outside
the supported `async with ... as pipe` shape; any raw `execute_command`
(static `MULTI`/`EXEC` in any capitalization, dynamic commands, or any
other command without an audit model); `break` outside a loop;
unterminated long-bracket Lua comments; long-bracket Lua strings; unknown
or malformed parenthesized ACL groups; unclassifiable ACL tokens; and a
missing `replay_projection.json` artifact, which only an explicit
`--regenerate-projection` run may recreate.

## What this pass does and does not yet verify

Correction Pass 1 proves the semantic audit equivalence, the enforced replay
log, the current-startup health protocol (static plus one live subtraction
scenario), and the narrowed-selector rule. Correction Pass 1B adds
path-sensitive Lua worst-case analysis, the exec-bound startup binding, the
sealed startup evidence window with size bounds, and a strengthened live
subtraction scenario (denial-before-readiness ordering, runtime-NOPERM
non-retroactivity). Correction Pass 1C adds case-insensitive fail-closed
ACL semantic parsing, the governed finite hint-destination contract with a
closed dynamic-write gate, the exact production marker shape in live replay
tests, and the separate live production-entrypoint control. The full AOF
lifecycle matrix — `BGREWRITEAOF`, incremental tail, truncation bounds,
rollback, dispatch normalization cleanup, stale double migration, and
complete fixture-admin work — remains for Correction Pass 2.
