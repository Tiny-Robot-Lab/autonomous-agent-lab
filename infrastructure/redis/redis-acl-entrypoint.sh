#!/bin/sh
set -eu

secret_dir=${REDIS_ACL_SECRET_DIR:-/run/secrets}
generation_file=${REDIS_ACL_GENERATION_FILE:-$secret_dir/redis_generation}
template=${REDIS_ACL_TEMPLATE:-/usr/local/bin/users.acl.template}
runtime_dir=${REDIS_ACL_RUNTIME_DIR:-/run/redis}
replay_log=${REDIS_REPLAY_LOG:-$runtime_dir/replay-integrity.log}
startup_marker=${REDIS_STARTUP_MARKER:-$runtime_dir/startup.json}
seal_file=${REDIS_REPLAY_SEAL:-$runtime_dir/replay-sealed.json}
startup_evidence=${REDIS_REPLAY_EVIDENCE:-$runtime_dir/replay-startup.evidence}
replay_log_level=${REDIS_REPLAY_LOG_LEVEL:-notice}
canonical_default='user default off resetpass resetkeys resetchannels -@all +select (+set %W~task.registration:* %W~task.status:* %W~task.lineage:* %W~task.input:* %W~task.authority:* %W~task.authority_grant:* %W~task.required_capability:* %W~task.artifact:* %W~task.continuation:* %W~task.continuation_latest:* %W~task.claim:* %W~task.claim_deadline:* %W~task.claim_expiry:* %W~task.fence:* %W~task.delegation_proposal:* %W~task.delegation_wave_proposal:* %W~task.delegation_wave:* %W~task.delegation_wave_resolution:* %W~task.assignment_yield:* %W~task.completion:* %W~task.outcome:* %W~task.result:* %W~task.assignment_completion:* %W~task.child_completion:* %W~task.recovery_decision:*) (+del %W~task.claim:*) (+sadd %W~task.children:* %W~task.artifacts:* %W~task.consumed_waves:* %W~task.consumed_grants:* %W~task.delegation_decisions:*) (+rpush %W~agent.hints.general %W~agent.hints.research %W~agent.hints.analysis %W~agent.tasks.general %W~agent.tasks.research %W~agent.tasks.analysis) (+xadd %W~agent.events)'
acl_temporary=
health_temporary=
generation_temporary=
marker_temporary=
startup_ready=0
runtime_tmpfs_verified=0

fail() {
    printf 'redis-acl-entrypoint: %s\n' "$1" >&2
    exit 1
}

cleanup() {
    if [ -n "$acl_temporary" ]; then
        rm -f "$acl_temporary"
    fi
    if [ -n "$health_temporary" ]; then
        rm -f "$health_temporary"
    fi
    if [ -n "$generation_temporary" ]; then
        rm -f "$generation_temporary"
    fi
    if [ -n "$marker_temporary" ]; then
        rm -f "$marker_temporary"
    fi
    if [ "$startup_ready" -ne 1 ] && [ "$runtime_tmpfs_verified" -eq 1 ]; then
        rm -f "$runtime_dir/healthcheck.password"
        rm -f "$runtime_dir/credential-generation.json"
        rm -f "$startup_marker"
        rm -f "$seal_file"
        rm -f "$startup_evidence"
    fi
}

trap cleanup EXIT
trap 'exit 1' HUP INT TERM

entrypoint_dir=$(dirname "$0")
if [ ! -r "$entrypoint_dir/redis-mountinfo.sh" ]; then
    fail "Redis tmpfs mount checker is missing"
fi
. "$entrypoint_dir/redis-mountinfo.sh"
if ! redis_mountinfo_is_tmpfs /proc/self/mountinfo "$runtime_dir"; then
    fail "Redis runtime directory is not an exact tmpfs mount"
fi
runtime_tmpfs_verified=1

if [ ! -f "$generation_file" ] || [ -L "$generation_file" ] || [ ! -r "$generation_file" ]; then
    fail "missing or unreadable Redis credential generation metadata"
fi

validate_secret() {
    identity=$1
    secret_file="$secret_dir/$identity"
    if [ ! -f "$secret_file" ] || [ -L "$secret_file" ] || [ ! -r "$secret_file" ]; then
        fail "missing or unreadable Redis secret file: $identity"
    fi

    byte_count=$(wc -c < "$secret_file" | tr -d '[:space:]')
    allowed_count=$(LC_ALL=C tr -cd 'A-Za-z0-9_-' < "$secret_file" | wc -c | tr -d '[:space:]')
    if [ "$byte_count" -eq 0 ] || [ "$byte_count" -ne "$allowed_count" ] || [ "$byte_count" -lt 32 ]; then
        fail "malformed Redis secret file: $identity"
    fi
}

secret_digest() {
    secret_file="$secret_dir/$1"
    if command -v sha256sum >/dev/null 2>&1; then
        digest_line=$(sha256sum < "$secret_file") || fail "cannot hash Redis secret file: $1"
    elif command -v shasum >/dev/null 2>&1; then
        digest_line=$(shasum -a 256 < "$secret_file") || fail "cannot hash Redis secret file: $1"
    else
        fail "no SHA-256 utility is available"
    fi
    digest=${digest_line%% *}
    case "$digest" in
        *[!0-9a-f]*|'') fail "invalid SHA-256 result for Redis secret file: $1" ;;
    esac
    if [ "${#digest}" -ne 64 ]; then
        fail "invalid SHA-256 result for Redis secret file: $1"
    fi
    printf '%s' "$digest"
}

validate_template() {
    if [ ! -f "$template" ] || [ -L "$template" ] || [ ! -r "$template" ]; then
        fail "missing or unreadable Redis ACL template"
    fi

    default_count=$(grep -Ec '^[[:space:]]*user[[:space:]]+default([[:space:]]|$)' "$template" || true)
    canonical_count=$(grep -Fxc "$canonical_default" "$template" || true)
    if [ "$default_count" -ne 1 ] || [ "$canonical_count" -ne 1 ]; then
        fail "Redis ACL template must contain exactly one canonical disabled default user"
    fi
    if grep -Eq '(^|[[:space:]])>[^[:space:]]*([[:space:]]|$)' "$template"; then
        fail "Redis ACL template contains a plaintext password rule"
    fi
    if grep -Eiq '(^|[[:space:]])#[[:xdigit:]]{64}([[:space:]]|$)' "$template"; then
        fail "Redis ACL template contains a committed password hash"
    fi
    if grep -Eiq '(^|[[:space:]])(~\*|\+@all|\+@connection|\+@scripting|\+@admin|\+@dangerous)([[:space:]]|$)' "$template"; then
        fail "Redis ACL template contains a prohibited broad selector or command category"
    fi
    if grep -Eiq '^[[:space:]]*user[[:space:]]+[^[:space:]]*(test|admin)[^[:space:]]*([[:space:]]|$)' "$template"; then
        fail "Redis ACL template contains a test or administrative identity"
    fi

    for placeholder in \
        '@@ORCHESTRATOR_SHA256@@' \
        '@@WORKER_1_SHA256@@' \
        '@@RESEARCH_AGENT_1_SHA256@@' \
        '@@RESEARCH_AGENT_2_SHA256@@' \
        '@@ANALYSIS_AGENT_1_SHA256@@' \
        '@@HEALTHCHECK_SHA256@@'
    do
        placeholder_count=$(grep -F -o "$placeholder" "$template" | wc -l | tr -d '[:space:]')
        if [ "$placeholder_count" -ne 1 ]; then
            fail "Redis ACL template must contain each required identity placeholder once"
        fi
    done
}

render_acl_line() {
    template_line=$1
    case "$template_line" in
        'user orchestrator on #@@ORCHESTRATOR_SHA256@@ '*)
            template_suffix=${template_line#user orchestrator on #@@ORCHESTRATOR_SHA256@@ }
            printf 'user orchestrator on #%s %s\n' "$orchestrator_digest" "$template_suffix"
            ;;
        'user worker-1 on #@@WORKER_1_SHA256@@ '*)
            template_suffix=${template_line#user worker-1 on #@@WORKER_1_SHA256@@ }
            printf 'user worker-1 on #%s %s\n' "$worker_1_digest" "$template_suffix"
            ;;
        'user research-agent-1 on #@@RESEARCH_AGENT_1_SHA256@@ '*)
            template_suffix=${template_line#user research-agent-1 on #@@RESEARCH_AGENT_1_SHA256@@ }
            printf 'user research-agent-1 on #%s %s\n' "$research_agent_1_digest" "$template_suffix"
            ;;
        'user research-agent-2 on #@@RESEARCH_AGENT_2_SHA256@@ '*)
            template_suffix=${template_line#user research-agent-2 on #@@RESEARCH_AGENT_2_SHA256@@ }
            printf 'user research-agent-2 on #%s %s\n' "$research_agent_2_digest" "$template_suffix"
            ;;
        'user analysis-agent-1 on #@@ANALYSIS_AGENT_1_SHA256@@ '*)
            template_suffix=${template_line#user analysis-agent-1 on #@@ANALYSIS_AGENT_1_SHA256@@ }
            printf 'user analysis-agent-1 on #%s %s\n' "$analysis_agent_1_digest" "$template_suffix"
            ;;
        'user healthcheck on #@@HEALTHCHECK_SHA256@@ '*)
            template_suffix=${template_line#user healthcheck on #@@HEALTHCHECK_SHA256@@ }
            printf 'user healthcheck on #%s %s\n' "$healthcheck_digest" "$template_suffix"
            ;;
        *)
            printf '%s\n' "$template_line"
            ;;
    esac
}

for identity in \
    orchestrator \
    worker-1 \
    research-agent-1 \
    research-agent-2 \
    analysis-agent-1 \
    healthcheck
do
    validate_secret "$identity"
done
validate_template

orchestrator_digest=$(secret_digest orchestrator)
worker_1_digest=$(secret_digest worker-1)
research_agent_1_digest=$(secret_digest research-agent-1)
research_agent_2_digest=$(secret_digest research-agent-2)
analysis_agent_1_digest=$(secret_digest analysis-agent-1)
healthcheck_digest=$(secret_digest healthcheck)

mkdir -p "$runtime_dir"
chown root:redis "$runtime_dir"
chmod 0750 "$runtime_dir"

acl_temporary=$(mktemp "$runtime_dir/.users.acl.XXXXXX") || fail "cannot create temporary Redis ACL file"
while IFS= read -r line || [ -n "$line" ]; do
    render_acl_line "$line"
done < "$template" > "$acl_temporary" || fail "cannot render Redis ACL template"
if grep -Fq '@@' "$acl_temporary"; then
    fail "Redis ACL template contains an unresolved placeholder"
fi
chown root:redis "$acl_temporary"
chmod 0440 "$acl_temporary"

health_temporary=$(mktemp "$runtime_dir/.healthcheck.password.XXXXXX") || fail "cannot create health-check secret copy"
cp "$secret_dir/healthcheck" "$health_temporary" || fail "cannot copy health-check secret"
chown root:root "$health_temporary"
chmod 0400 "$health_temporary"
mv -f "$health_temporary" "$runtime_dir/healthcheck.password" || fail "cannot install health-check secret copy"
health_temporary=

generation_temporary=$(mktemp "$runtime_dir/.credential-generation.XXXXXX") || fail "cannot create Redis credential generation copy"
cp "$generation_file" "$generation_temporary" || fail "cannot copy Redis credential generation metadata"
chown root:redis "$generation_temporary"
chmod 0440 "$generation_temporary"
mv -f "$generation_temporary" "$runtime_dir/credential-generation.json" || fail "cannot install Redis credential generation metadata"
generation_temporary=

mv -f "$acl_temporary" "$runtime_dir/users.acl" || fail "cannot install rendered Redis ACL"
acl_temporary=

# Initialize bounded current-startup replay evidence on the verified tmpfs.
# The log is truncated here so it can only contain the current startup's
# output; a stale denial can never poison a later healthy start, and a
# missing marker fails the health gate closed.
case "$replay_log" in
    "$runtime_dir"/*) ;;
    *) fail "Redis replay log must reside in the runtime directory" ;;
esac
case "$startup_marker" in
    "$runtime_dir"/*) ;;
    *) fail "Redis startup marker must reside in the runtime directory" ;;
esac
case "$seal_file" in
    "$runtime_dir"/*) ;;
    *) fail "Redis sealed evidence must reside in the runtime directory" ;;
esac
case "$startup_evidence" in
    "$runtime_dir"/*) ;;
    *) fail "Redis startup evidence must reside in the runtime directory" ;;
esac
if [ -L "$replay_log" ] || [ -L "$startup_marker" ]; then
    fail "Redis replay evidence path must not be a symlink"
fi
# Every entrypoint execution starts unsealed: a sealed verdict or retained
# startup evidence from a previous start (visible across `docker restart`,
# which preserves the tmpfs) must never satisfy the new startup. Only the
# health gate seals, and only after observing readiness in the freshly
# truncated log.
rm -f "$seal_file" "$startup_evidence" || fail "cannot clear stale sealed replay verdict"
: > "$replay_log" || fail "cannot truncate Redis replay log"
# Ownership model: root owns every runtime file so the redis user can never
# replace evidence, but the replay log itself must be appendable by the redis
# process, so the redis group holds write permission. The ACL file and the
# startup marker stay read-only (0440/0640): only the log is group-writable.
chown root:redis "$replay_log"
chmod 0660 "$replay_log"

# The startup marker carries no identity beyond the active generation digest:
# this is an exec-bound fresh-log proof, not a process-ID binding. The tmpfs
# is verified, the log was just truncated, and this marker is recorded
# immediately before exec redis-server, so a passing health check proves the
# readiness evidence in that same fresh log belongs to the current startup.
# A random start_id would be ornamental here: health only checks generation
# binding, log freshness, and Redis's own readiness statement.
generation_digest=$(sha256sum < "$runtime_dir/credential-generation.json" 2>/dev/null \
    || shasum -a 256 < "$runtime_dir/credential-generation.json") \
    || fail "cannot digest Redis credential generation metadata"
generation_digest=${generation_digest%% *}
case "$generation_digest" in
    *[!0-9a-f]*|'') fail "invalid generation digest" ;;
esac
if [ "${#generation_digest}" -ne 64 ]; then
    fail "invalid generation digest"
fi
marker_temporary=$(mktemp "$runtime_dir/.startup.XXXXXX") || fail "cannot create startup marker"
printf '{"generation_sha256":"%s"}\n' "$generation_digest" \
    > "$marker_temporary" || fail "cannot write startup marker"
chown root:redis "$marker_temporary"
chmod 0640 "$marker_temporary"
mv -f "$marker_temporary" "$startup_marker" || fail "cannot install startup marker"
marker_temporary=
startup_ready=1

if [ "$#" -eq 0 ]; then
    set -- redis-server --appendonly yes
fi
if [ "$1" != "redis-server" ]; then
    fail "Redis entrypoint command must be redis-server"
fi
shift
# Enforce the bounded replay-integrity log regardless of caller arguments.
# Caller options are preserved; a conflicting replay-log path or level fails
# startup, keeping replay evidence on the enforced log.
have_logfile=0
have_loglevel=0
expect_value=""
for redis_arg in "$@"; do
    if [ -n "$expect_value" ]; then
        if [ "$expect_value" = "--logfile" ]; then
            [ "$redis_arg" = "$replay_log" ] || fail "Redis logfile must be the replay-integrity log"
            have_logfile=1
        else
            [ "$redis_arg" = "$replay_log_level" ] || fail "Redis loglevel must stay bounded"
            have_loglevel=1
        fi
        expect_value=""
        continue
    fi
    case "$redis_arg" in
        --logfile)
            expect_value="--logfile"
            ;;
        --loglevel)
            expect_value="--loglevel"
            ;;
        --logfile=*)
            [ "${redis_arg#--logfile=}" = "$replay_log" ] || fail "Redis logfile must be the replay-integrity log"
            have_logfile=1
            ;;
        --loglevel=*)
            [ "${redis_arg#--loglevel=}" = "$replay_log_level" ] || fail "Redis loglevel must stay bounded"
            have_loglevel=1
            ;;
    esac
done
if [ -n "$expect_value" ]; then
    fail "Redis replay logging option is missing its value"
fi
if [ "$have_logfile" -eq 0 ]; then
    set -- "$@" --logfile "$replay_log"
fi
if [ "$have_loglevel" -eq 0 ]; then
    set -- "$@" --loglevel "$replay_log_level"
fi
set -- redis-server "$@"
exec setpriv --reuid=redis --regid=redis --clear-groups -- "$@" --aclfile "$runtime_dir/users.acl"
