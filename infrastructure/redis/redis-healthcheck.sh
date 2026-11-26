#!/bin/sh
# Bounded current-startup replay-integrity health gate.
#
# Healthy requires ALL of the following for the current startup:
#   1. a startup marker bound to the active credential generation;
#   2. a present, regular replay log newer than the marker;
#   3. a sealed startup verdict covering the exact startup window, or a
#      fresh seal written only after the window below is proven clean;
#   4. an exact authenticated PING from the keyless healthcheck identity,
#      verified on every run.
# PING alone is insufficient: silent AOF transaction drops leave Redis
# reachable while authoritative state is missing.
#
# The startup window (after the last Redis startup banner through the first
# subsequent readiness statement) is extracted once, sealed into a
# root-owned bounded evidence artifact, and never re-derived from the
# continuing log while the seal stands. Later runtime authorization failures
# stay visible in the continuing log but cannot change the sealed verdict;
# continuing log growth past the operational bound is truncated, never
# reported as a replay failure, and never invalidates a valid seal.

set -eu

password_file=${REDIS_HEALTHCHECK_PASSWORD_FILE:-/run/redis/healthcheck.password}
replay_log=${REDIS_REPLAY_LOG:-/run/redis/replay-integrity.log}
startup_marker=${REDIS_STARTUP_MARKER:-/run/redis/startup.json}
generation_file=${REDIS_CREDENTIAL_GENERATION_FILE:-/run/redis/credential-generation.json}
seal_file=${REDIS_REPLAY_SEAL:-/run/redis/replay-sealed.json}
evidence_file=${REDIS_REPLAY_EVIDENCE:-/run/redis/replay-startup.evidence}
max_window_bytes=1048576
max_log_bytes=8388608
password=
response_file=
window_file=
seal_temporary=
evidence_temporary=

cleanup() {
    unset REDISCLI_AUTH password
    if [ -n "$response_file" ]; then
        rm -f "$response_file"
    fi
    if [ -n "$window_file" ]; then
        rm -f "$window_file"
    fi
    if [ -n "$seal_temporary" ]; then
        rm -f "$seal_temporary"
    fi
    if [ -n "$evidence_temporary" ]; then
        rm -f "$evidence_temporary"
    fi
}

trap cleanup EXIT
trap 'exit 1' HUP INT TERM

unset REDISCLI_AUTH

# 1. Startup marker bound to the active credential generation. This is an
# exec-bound proof, not a process-ID binding: the entrypoint verified the
# tmpfs, truncated the same log, recorded the marker, and immediately exec'd
# redis-server, so readiness evidence sealed from that same fresh window
# belongs to the current startup. No random identity is recorded: health
# only checks non-emptiness for such an identifier, which attests nothing.
if [ ! -f "$startup_marker" ] || [ -L "$startup_marker" ] || [ ! -r "$startup_marker" ]; then
    exit 1
fi
marker_generation=$(sed -n 's/.*"generation_sha256"[ ]*:[ ]*"\([0-9a-f]*\)".*/\1/p' "$startup_marker" | head -n 1)
if [ -z "$marker_generation" ]; then
    exit 1
fi
if [ "${#marker_generation}" -ne 64 ]; then
    exit 1
fi
if [ ! -f "$generation_file" ] || [ -L "$generation_file" ] || [ ! -r "$generation_file" ]; then
    exit 1
fi
active_generation=$(sha256sum < "$generation_file" 2>/dev/null || shasum -a 256 < "$generation_file") || exit 1
active_generation=${active_generation%% *}
if [ "$active_generation" != "$marker_generation" ]; then
    exit 1
fi

# 2. Continuing runtime log: present, regular, and newer than the marker.
# The sealed verdict below is validated against the separate startup
# evidence artifact, never recomputed from this file while the seal stands,
# so truncation or rotation of this log cannot invalidate a valid seal.
if [ ! -f "$replay_log" ] || [ -L "$replay_log" ] || [ ! -r "$replay_log" ]; then
    exit 1
fi
if [ "$replay_log" -ot "$startup_marker" ]; then
    exit 1
fi

# 3. Current startup window, extracted only when the live log still carries
# a startup banner. With no banner the log was truncated after a sealed
# startup (or nothing has started): the sealed artifact below then stands
# alone, and fresh evaluation below fails closed for lack of evidence.
# Verdicts derive from file content only, never from awk exit codes, which
# vary across awk implementations.
window_digest=""
if grep -q 'Redis is starting' "$replay_log" 2>/dev/null; then
    window_file=$(mktemp) || exit 1
    LC_ALL=C awk -v max="$max_window_bytes" '
      /Redis is starting/ { buf=""; bytes=0; in_window=1 }
      in_window {
        buf = buf $0 "\n"; bytes += length($0)+1
        if (bytes > max) { print "__REPLAY_WINDOW_OVER_LIMIT__"; exit }
        if ($0 ~ /Ready to accept connections/) { last = buf; in_window=0 }
      }
      END { if (last != "") printf "%s", last }
    ' "$replay_log" > "$window_file" 2>/dev/null
    if grep -q '__REPLAY_WINDOW_OVER_LIMIT__' "$window_file" 2>/dev/null; then
        rm -f "$seal_file"
        exit 1
    fi
    if ! grep -q 'Ready to accept connections' "$window_file" 2>/dev/null; then
        rm -f "$seal_file"
        exit 1
    fi
    if grep -E 'CRITICAL|NOPERM|Failed to load|Loading AOF failed|Fatal error|Can'"'"'t load|Wrong signature|Unexpected end|Short read' "$window_file" >/dev/null 2>&1
    then
        rm -f "$seal_file"
        exit 1
    fi
    window_digest=$(sha256sum < "$window_file" 2>/dev/null || shasum -a 256 < "$window_file") || exit 1
    window_digest=${window_digest%% *}
fi

# 4. Sealed verdict: a seal counts only when it binds the active generation
# to the exact bytes of the retained startup evidence, and that evidence is
# bounded, carries the readiness boundary, and is clean. When the live log
# carries a startup banner, the recomputed window must additionally match
# the seal, so a newer startup always re-proves itself; a stale seal can
# never satisfy a new window. When no banner remains (truncated log after a
# sealed startup), the sealed artifact stands alone.
sealed_ok=0
if [ -f "$seal_file" ] && [ ! -L "$seal_file" ] && [ -r "$seal_file" ] \
    && [ -f "$evidence_file" ] && [ ! -L "$evidence_file" ] && [ -r "$evidence_file" ]; then
    seal_generation=$(sed -n 's/.*"generation_sha256"[ ]*:[ ]*"\([0-9a-f]*\)".*/\1/p' "$seal_file" | head -n 1)
    seal_window=$(sed -n 's/.*"window_sha256"[ ]*:[ ]*"\([0-9a-f]*\)".*/\1/p' "$seal_file" | head -n 1)
    if [ -n "$seal_window" ] && [ "$seal_generation" = "$active_generation" ]; then
        evidence_bytes=$(wc -c < "$evidence_file" | tr -d '[:space:]')
        case "$evidence_bytes" in
            *[!0-9]*|'') evidence_bytes=-1 ;;
        esac
        if [ "$evidence_bytes" -gt 0 ] && [ "$evidence_bytes" -le "$max_window_bytes" ]; then
            evidence_digest=$(sha256sum < "$evidence_file" 2>/dev/null || shasum -a 256 < "$evidence_file") || exit 1
            evidence_digest=${evidence_digest%% *}
            if [ "$evidence_digest" = "$seal_window" ] \
                && grep -q 'Ready to accept connections' "$evidence_file" 2>/dev/null \
                && ! grep -E 'CRITICAL|NOPERM|Failed to load|Loading AOF failed|Fatal error|Can'"'"'t load|Wrong signature|Unexpected end|Short read' "$evidence_file" >/dev/null 2>&1; then
                if [ -z "$window_digest" ] || [ "$window_digest" = "$seal_window" ]; then
                    sealed_ok=1
                fi
            fi
        fi
    fi
fi

# 5. Without a sealed verdict there must be a freshly proven window to seal.
# A missing banner, a failed window, or a failed PING all fail closed, and a
# failed evaluation removes any stale seal so a bad startup can never
# inherit a previous verdict.
if [ "$sealed_ok" -eq 0 ] && [ -z "$window_digest" ]; then
    rm -f "$seal_file"
    exit 1
fi

# 6. Exact authenticated PING from the keyless healthcheck identity, verified
# on every run including sealed steady states.
if [ ! -f "$password_file" ] || [ -L "$password_file" ] || [ ! -r "$password_file" ]; then
    exit 1
fi

byte_count=$(wc -c < "$password_file" | tr -d '[:space:]')
allowed_count=$(LC_ALL=C tr -cd 'A-Za-z0-9_-' < "$password_file" | wc -c | tr -d '[:space:]')
if [ "$byte_count" -lt 32 ] || [ "$byte_count" -ne "$allowed_count" ]; then
    exit 1
fi

password=$(cat "$password_file") || exit 1
response_file=$(mktemp) || exit 1

if ! REDISCLI_AUTH="$password" REDISCLI_HISTFILE=/dev/null \
    redis-cli --raw --user healthcheck ping > "$response_file" 2>/dev/null
then
    exit 1
fi

if ! printf 'PONG\n' | cmp -s - "$response_file"; then
    exit 1
fi

# 7. Seal a freshly proven startup: retain the exact window bytes as bounded
# startup evidence, then bind generation plus window digest atomically.
# Sealing requires root-owned files; the health identity runs as root
# wherever this gate executes (it already requires the root-only password
# file), while the redis user can read but never replace sealed evidence.
# Ownership or permission failures fail closed rather than silently
# degrading the seal.
if [ "$sealed_ok" -eq 0 ]; then
    seal_dir=$(dirname "$seal_file")
    evidence_temporary=$(mktemp "$seal_dir/.evidence.XXXXXX") || exit 1
    if ! cat "$window_file" > "$evidence_temporary"; then
        rm -f "$evidence_temporary"
        exit 1
    fi
    chown root:redis "$evidence_temporary" || { rm -f "$evidence_temporary"; exit 1; }
    chmod 0640 "$evidence_temporary" || { rm -f "$evidence_temporary"; exit 1; }
    mv -f "$evidence_temporary" "$evidence_file" || exit 1
    evidence_temporary=
    seal_temporary=$(mktemp "$seal_dir/.sealed.XXXXXX") || exit 1
    if ! printf '{"generation_sha256":"%s","window_sha256":"%s"}\n' \
        "$active_generation" "$window_digest" > "$seal_temporary"; then
        rm -f "$seal_temporary"
        exit 1
    fi
    chown root:redis "$seal_temporary" || { rm -f "$seal_temporary"; exit 1; }
    chmod 0640 "$seal_temporary" || { rm -f "$seal_temporary"; exit 1; }
    mv -f "$seal_temporary" "$seal_file" || exit 1
    seal_temporary=
fi

# 8. Bound the continuing runtime log without touching the sealed verdict.
# The startup evidence above is retained separately, so truncating new
# runtime output preserves tmpfs headroom while sealed health stands.
log_bytes=$(wc -c < "$replay_log" | tr -d '[:space:]')
case "$log_bytes" in
    *[!0-9]*|'') exit 1 ;;
esac
if [ "$log_bytes" -gt "$max_log_bytes" ]; then
    : > "$replay_log" || exit 1
fi
exit 0
