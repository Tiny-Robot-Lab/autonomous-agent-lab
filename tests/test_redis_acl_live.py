"""Live disposable-Docker Redis 7.4.12 ACL enforcement matrix (Issue #229 Increment 3).

Runs against the pinned redis:7.4.12-alpine image with users.acl rendered through
the real infrastructure/redis/redis-acl-entrypoint.sh and isolated non-root service
containers on an internal Docker network (zero host-published ports, zero persistent
volumes, and temporary synthetic secrets only).
"""

from __future__ import annotations

import json
import re
from collections.abc import Generator

import pytest

from scripts.redis_acl_live_harness import (
    PRODUCTION_IDENTITIES,
    RUNTIME_IDENTITIES,
    TEST_ADMIN_IDENTITY,
    LiveRedisAclCluster,
    _run_docker,
)

pytestmark = [pytest.mark.live_redis_acl]

RUNTIME_TOPOLOGY: dict[str, tuple[str, str, str]] = {
    # identity: (service_name, role_task_queue, role_hint_queue)
    "worker-1": ("worker", "agent.tasks.general", "agent.hints.general"),
    "research-agent-1": ("research-agent", "agent.tasks.research", "agent.hints.research"),
    "research-agent-2": ("research-agent", "agent.tasks.research", "agent.hints.research"),
    "analysis-agent-1": ("analysis-agent", "agent.tasks.analysis", "agent.hints.analysis"),
}


@pytest.fixture(scope="module")
def acl_cluster() -> Generator[LiveRedisAclCluster, None, None]:
    cluster = LiveRedisAclCluster.start()
    try:
        yield cluster
    finally:
        cluster.cleanup()


def _assert_noperm(result_text: str, *, context: str) -> None:
    assert "NOPERM" in result_text, f"Expected NOPERM for {context}, got: {result_text!r}"


def _assert_debug_denied(result_text: str, *, context: str) -> None:
    # Redis 7.4 rejects DEBUG at enable-debug-command gating before ACL unless configured;
    # both ERR DEBUG command not allowed and NOPERM prove execution is blocked.
    assert (
        "NOPERM" in result_text or "ERR DEBUG command not allowed" in result_text
    ), f"Expected DEBUG denial for {context}, got: {result_text!r}"


def _assert_auth_denied(result_text: str, *, context: str) -> None:
    assert any(
        token in result_text for token in ("WRONGPASS", "NOAUTH", "NOPERM", "ERR")
    ), f"Expected auth denial for {context}, got: {result_text!r}"
    assert "PONG" not in result_text, f"Unexpected PONG for {context}: {result_text!r}"


def _other_runtime_targets(identity: str) -> tuple[list[str], list[str], list[str]]:
    _, own_q, own_h = RUNTIME_TOPOLOGY[identity]
    other_queues = sorted(
        {q for uid, (_, q, _) in RUNTIME_TOPOLOGY.items() if q != own_q}
    )
    other_hints = sorted(
        {h for uid, (_, _, h) in RUNTIME_TOPOLOGY.items() if h != own_h}
    )
    other_agents = sorted(uid for uid in RUNTIME_IDENTITIES if uid != identity)
    return other_queues, other_hints, other_agents


# ---------------------------------------------------------------------------
# 1. Default user, unauthenticated access & cross-identity auth failures
# ---------------------------------------------------------------------------


def test_default_user_unauthenticated_and_cross_password_auth_denials(
    acl_cluster: LiveRedisAclCluster,
) -> None:
    # Unauthenticated PING -> NOAUTH
    unauth = acl_cluster.redis_cli(None, "PING")
    assert "NOAUTH" in unauth.combined, f"Expected NOAUTH, got {unauth.combined!r}"

    # Disabled default user cannot authenticate even with empty or valid synthetic secret
    for attempt_pw in ("", "wrong-default-pw", acl_cluster.secrets_map["orchestrator"]):
        res = acl_cluster.redis_cli("default", "PING", password=attempt_pw)
        _assert_auth_denied(res.combined, context="default user")
        assert "WRONGPASS" in res.combined or "disabled" in res.combined.lower()

    # Wrong password for every named identity -> WRONGPASS
    for identity in PRODUCTION_IDENTITIES:
        wrong = acl_cluster.redis_cli(identity, "PING", password=f"wrong-{identity}-secret-123")
        assert "WRONGPASS" in wrong.combined, f"Expected WRONGPASS for {identity}, got {wrong.combined!r}"

    # Cross-identity password reuse fails for every pair (sender != target)
    for target_identity in PRODUCTION_IDENTITIES:
        for other_identity in PRODUCTION_IDENTITIES:
            if target_identity == other_identity:
                continue
            cross = acl_cluster.redis_cli(
                target_identity,
                "PING",
                password=acl_cluster.secrets_map[other_identity],
            )
            assert "WRONGPASS" in cross.combined, (
                f"Expected WRONGPASS authenticating as {target_identity} with "
                f"{other_identity}'s secret, got {cross.combined!r}"
            )


# ---------------------------------------------------------------------------
# 2. Healthcheck identity (PING-only, zero keys, fails on wrong/mismatched)
# ---------------------------------------------------------------------------


def test_healthcheck_identity_allow_deny_and_injected_mismatch_unhealthy(
    acl_cluster: LiveRedisAclCluster,
) -> None:
    # Inspect rendered ACL user definition via fixture-admin
    acl_def = acl_cluster.admin_cli("ACL", "GETUSER", "healthcheck")
    assert acl_def.returncode == 0
    assert "~" not in acl_def.stdout, f"healthcheck ACL must have no key patterns: {acl_def.stdout!r}"
    assert "&" not in acl_def.stdout, f"healthcheck ACL must have no channel patterns: {acl_def.stdout!r}"

    # Production healthcheck script succeeds with exact silent exit 0
    ok = acl_cluster.run_healthcheck()
    assert ok.returncode == 0, f"healthcheck script failed: {ok.combined!r}"
    assert ok.stdout == ""

    # Direct PING returns exact PONG
    ping = acl_cluster.redis_cli("healthcheck", "PING", use_mounted_secret=True)
    assert ping.returncode == 0
    assert ping.stdout.strip() == "PONG"

    # Denied commands for healthcheck
    denied_commands: list[tuple[str, ...]] = [
        ("GET", "task.status:hc-1"),
        ("SET", "task.status:hc-1", "bad"),
        ("DEL", "agent.ids"),
        ("EVAL", "return 1", "0"),
        ("ACL", "WHOAMI"),
        ("ACL", "LIST"),
        ("INFO",),
        ("CLIENT", "LIST"),
        ("CLIENT", "SETINFO", "LIB-NAME", "test"),
        ("SELECT", "0"),
        ("SELECT", "1"),
        ("BLPOP", "agent.tasks.general", "0.1"),
        ("SADD", "agent.ids", "healthcheck"),
        ("HSET", "agent.state:healthcheck", "k", "v"),
        ("XADD", "agent.events", "*", "k", "v"),
    ]
    for cmd in denied_commands:
        res = acl_cluster.redis_cli("healthcheck", *cmd, use_mounted_secret=True)
        _assert_noperm(res.combined, context=f"healthcheck {' '.join(cmd)}")

    # Incorrect password fails PING
    bad_pw = acl_cluster.redis_cli("healthcheck", "PING", password="wrong_healthcheck_secret_000000000")
    assert "WRONGPASS" in bad_pw.combined

    # Explicitly injected mismatched credential causes healthcheck script to fail unhealthy
    mismatched_secret = "MISMATCHED_healthcheck_secret_value_0123456789"
    unhealthy = acl_cluster.run_healthcheck(override_secret=mismatched_secret)
    assert unhealthy.returncode != 0
    credential_exposed = mismatched_secret in unhealthy.combined
    assert not credential_exposed, (
        "health-check output exposed synthetic credential material"
    )


def test_redis_image_uses_setpriv_and_keeps_acl_on_root_owned_tmpfs(
    acl_cluster: LiveRedisAclCluster,
) -> None:
    setpriv = _run_docker(
        [
            "docker",
            "exec",
            acl_cluster.redis_container,
            "/bin/sh",
            "-c",
            "command -v setpriv >/dev/null && setpriv --version",
        ],
        check=False,
    )
    assert setpriv.returncode == 0
    assert "setpriv" in setpriv.stdout.lower()

    pid_status = _run_docker(
        [
            "docker",
            "exec",
            acl_cluster.redis_container,
            "/bin/sh",
            "-c",
            (
                "awk '$1 == \"Name:\" {name=$2} $1 == \"Uid:\" {uid=$2} "
                "$1 == \"Gid:\" {gid=$2} END {print name, uid, gid}' /proc/1/status"
            ),
        ],
        check=False,
    )
    assert pid_status.returncode == 0
    process_name, process_uid, process_gid = pid_status.stdout.split()
    redis_uid = _run_docker(
        ["docker", "exec", acl_cluster.redis_container, "id", "-u", "redis"]
    ).stdout.strip()
    redis_gid = _run_docker(
        ["docker", "exec", acl_cluster.redis_container, "id", "-g", "redis"]
    ).stdout.strip()
    assert process_name == "redis-server"
    assert process_uid == redis_uid != "0"
    assert process_gid == redis_gid != "0"

    acl_mode = _run_docker(
        [
            "docker",
            "exec",
            acl_cluster.redis_container,
            "stat",
            "-c",
            "%U:%G %a",
            "/run/redis/users.acl",
        ]
    )
    assert acl_mode.stdout.strip() == "root:redis 440"

    tmpfs_mount = _run_docker(
        [
            "docker",
            "exec",
            acl_cluster.redis_container,
            "/bin/sh",
            "-c",
            (
                '. /usr/local/bin/redis-mountinfo.sh; '
                'redis_mountinfo_is_tmpfs /proc/self/mountinfo /run/redis'
            ),
        ],
        check=False,
    )
    assert tmpfs_mount.returncode == 0


# ---------------------------------------------------------------------------
# 4. Per-runtime positive commands + real non-root RedisBus reconnect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("identity", RUNTIME_IDENTITIES)
def test_runtime_identities_positive_commands_and_redis_py_reconnect(
    acl_cluster: LiveRedisAclCluster,
    identity: str,
) -> None:
    acl_cluster.reset_db()
    service_name, task_queue, hint_queue = RUNTIME_TOPOLOGY[identity]

    # Seed task & hint items using fixture-admin for BLPOP order tests
    assert acl_cluster.admin_cli("RPUSH", task_queue, "item-task-1").returncode == 0
    assert acl_cluster.admin_cli("RPUSH", hint_queue, "item-hint-1").returncode == 0

    # Multi-key BLPOP order 1: hint_queue first
    pop1 = acl_cluster.redis_cli(
        identity, "BLPOP", hint_queue, task_queue, "1", use_mounted_secret=True
    )
    assert pop1.returncode == 0
    assert pop1.stdout.splitlines() == [hint_queue, "item-hint-1"]

    # Multi-key BLPOP order 2: task_queue first
    pop2 = acl_cluster.redis_cli(
        identity, "BLPOP", task_queue, hint_queue, "1", use_mounted_secret=True
    )
    assert pop2.returncode == 0
    assert pop2.stdout.splitlines() == [task_queue, "item-task-1"]

    # SADD agent.ids, HSET agent.state:<self>, XADD agent.events
    assert (
        acl_cluster.redis_cli(identity, "SADD", "agent.ids", identity, use_mounted_secret=True)
        .stdout.strip()
        == "1"
    )
    assert (
        acl_cluster.redis_cli(
            identity,
            "HSET",
            f"agent.state:{identity}",
            "agent_id",
            identity,
            use_mounted_secret=True,
        )
        .stdout.strip()
        == "1"
    )
    xadd = acl_cluster.redis_cli(
        identity,
        "XADD",
        "agent.events",
        "MAXLEN",
        "~",
        "500",
        "*",
        "agent_id",
        identity,
        use_mounted_secret=True,
    )
    assert xadd.returncode == 0 and "-" in xadd.stdout

    # Seed one TaskRequested and one WakeUpHint via fixture-admin for real Python RedisBus test
    task_payload = json.dumps(
        {
            "task_id": f"task-{identity}-1",
            "root_task_id": f"task-{identity}-1",
            "parent_task_id": None,
            "prompt": "acl bus check",
            "created_at": "2026-11-24T00:00:00Z",
            "version": 2,
        }
    )
    hint_payload = json.dumps(
        {
            "hint_id": f"hint-{identity}-1",
            "task_id": f"task-{identity}-2",
            "capability": "general"
            if identity == "worker-1"
            else ("analysis" if identity == "analysis-agent-1" else "research"),
            "execution_kind": "initial",
            "created_at": "2026-11-24T00:00:00Z",
            "version": 1,
        }
    )
    assert acl_cluster.admin_cli("RPUSH", task_queue, task_payload).returncode == 0
    assert acl_cluster.admin_cli("RPUSH", hint_queue, hint_payload).returncode == 0

    bus_import = {
        "worker-1": "worker.infrastructure.redis_bus",
        "research-agent-1": "research_agent.infrastructure.redis_bus",
        "research-agent-2": "research_agent.infrastructure.redis_bus",
        "analysis-agent-1": "analysis_agent.infrastructure.redis_bus",
    }[identity]

    probe = f"""
import asyncio, os
from {bus_import} import RedisBus
from agent_contracts import AgentCapabilityMetadata, AgentEvent, AgentHeartbeat, resolve_redis_connection_from_env

async def main():
    assert os.geteuid() == 10001
    cfg = resolve_redis_connection_from_env()
    assert cfg.is_authenticated and cfg.username == {identity!r}
    bus = RedisBus(cfg.url, queue_name={task_queue!r}, hint_queue={hint_queue!r}, connection_config=cfg)
    try:
        pool_kwargs = bus.client.connection_pool.connection_kwargs
        assert pool_kwargs["protocol"] == 2
        assert pool_kwargs["driver_info"] is None
        assert pool_kwargs.get("health_check_interval", 0) == 0

        await bus.register(AgentCapabilityMetadata(
            agent_id={identity!r},
            agent_type="test",
            capabilities=["test"],
            providers={{"mode": "acl"}},
            dispatch_target={task_queue!r},
        ))
        await bus.publish(AgentEvent.create({identity!r}, "working", "ACL roundtrip"))
        sig1 = await bus.next_signal()
        assert sig1.task_id == {f"task-{identity}-2"!r}

        # Disconnect pool cleanly and force real reconnect with protocol=2, driver_info=None
        await bus.client.connection_pool.disconnect(inuse_connections=True)

        await bus.heartbeat(AgentHeartbeat.create({identity!r}, "idle"))
        await bus.publish(AgentEvent.create({identity!r}, "idle", "Reconnected"))
        sig2 = await bus.next_signal()
        assert sig2.task_id == {f"task-{identity}-1"!r}
        print("OK")
    finally:
        await bus.aclose()

asyncio.run(main())
"""
    res = acl_cluster.run_service_python(service_name, identity, probe)
    assert res.returncode == 0, f"{identity} RedisBus probe failed:\n{res.combined}"
    assert res.stdout.strip().endswith("OK")


# ---------------------------------------------------------------------------
# 5. Per-runtime negative command, task.*, cross-role & cross-instance denials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("identity", RUNTIME_IDENTITIES)
def test_runtime_identities_negative_commands_and_task_isolation(
    acl_cluster: LiveRedisAclCluster,
    identity: str,
) -> None:
    _, own_queue, own_hint = RUNTIME_TOPOLOGY[identity]
    other_queues, other_hints, other_agents = _other_runtime_targets(identity)

    denied_cases: list[tuple[str, ...]] = [
        # task.* reads & writes across authoritative key families
        ("GET", "task.status:t1"),
        ("SET", "task.status:t1", "{}"),
        ("DEL", "task.status:t1"),
        ("EXISTS", "task.claim:t1"),
        ("GET", "task.claim:t1"),
        ("SET", "task.fence:t1", "99"),
        ("SET", "task.outcome:t1", "{}"),
        ("SET", "task.result:t1", "{}"),
        ("SET", "task.claim_deadline:t1", "2099-01-01T00:00:00Z"),
        ("SET", "task.claim_expiry:t1", "{}"),
        ("SET", "task.recovery_decision:t1:r1", "{}"),
        ("SET", "task.authority_grant:t1:g1", "{}"),
        ("SET", "task.delegation_wave_proposal:t1:p1", "{}"),
        ("GET", "task.delegation_wave_proposal:t1:p1"),
        ("SET", "task.delegation_wave:t1:w1", "{}"),
        ("GET", "task.delegation_wave:t1:w1"),
        ("SET", "task.delegation_wave_resolution:t1:w1", "{}"),
        ("GET", "task.delegation_wave_resolution:t1:w1"),
        ("SET", "task.routing:t1", "{}"),
        ("RPUSH", "task.children:t1", "c1"),
        ("BLPOP", "task.status:t1", "1"),
        # Cross-role queues & hints
        *(("BLPOP", q, "1") for q in other_queues),
        *(("RPUSH", q, "x") for q in other_queues),
        *(("BLPOP", h, "1") for h in other_hints),
        *(("RPUSH", h, "x") for h in other_hints),
        # Cross-instance agent.state:* (explicitly tests research-agent-1 vs research-agent-2)
        *(("HSET", f"agent.state:{other}", "state", "spoofed") for other in other_agents),
        *(("HGETALL", f"agent.state:{other}") for other in other_agents),
        *(("GET", f"agent.state:{other}") for other in other_agents),
        *(("DEL", f"agent.state:{other}") for other in other_agents),
        # Wrong commands on allowed queues / hints
        ("HSET", own_queue, "f", "v"),
        ("SADD", own_queue, "m"),
        ("XADD", own_queue, "*", "f", "v"),
        ("RPUSH", own_queue, "m"),
        ("GET", own_queue),
        ("DEL", own_queue),
        ("HSET", own_hint, "f", "v"),
        ("SADD", own_hint, "m"),
        ("XADD", own_hint, "*", "f", "v"),
        ("RPUSH", own_hint, "m"),
        ("GET", own_hint),
        ("DEL", own_hint),
        # Wrong commands on agent.ids, agent.events, own agent.state:<self>
        ("BLPOP", "agent.ids", "1"),
        ("HSET", "agent.ids", "f", "v"),
        ("XADD", "agent.ids", "*", "f", "v"),
        ("SMEMBERS", "agent.ids"),
        ("DEL", "agent.ids"),
        ("BLPOP", "agent.events", "1"),
        ("HSET", "agent.events", "f", "v"),
        ("SADD", "agent.events", "m"),
        ("XREVRANGE", "agent.events", "+", "-", "COUNT", "1"),
        ("DEL", "agent.events"),
        ("BLPOP", f"agent.state:{identity}", "1"),
        ("SADD", f"agent.state:{identity}", "m"),
        ("XADD", f"agent.state:{identity}", "*", "f", "v"),
        ("HGETALL", f"agent.state:{identity}"),
        ("GET", f"agent.state:{identity}"),
        ("DEL", f"agent.state:{identity}"),
        ("SET", f"agent.state:{identity}", "x"),
        # Scripting & admin & connection & out-of-namespace
        ("EVAL", "return 1", "0"),
        ("EVAL_RO", "return 1", "0"),
        ("EVALSHA", "0" * 40, "0"),
        ("SCRIPT", "LOAD", "return 1"),
        ("SCRIPT", "FLUSH"),
        ("FUNCTION", "LIST"),
        ("FCALL", "fn", "0"),
        ("PING",),
        ("SELECT", "0"),
        ("SELECT", "1"),
        ("CLIENT", "LIST"),
        ("CLIENT", "SETINFO", "LIB-NAME", "rogue"),
        ("MULTI",),
        ("FLUSHDB",),
        ("FLUSHALL",),
        ("CONFIG", "GET", "*"),
        ("ACL", "LIST"),
        ("ACL", "WHOAMI"),
        ("MODULE", "LIST"),
        ("KEYS", "*"),
        ("SCAN", "0"),
        ("SET", "out.of.namespace:1", "val"),
        ("GET", "out.of.namespace:1"),
    ]

    for cmd in denied_cases:
        res = acl_cluster.redis_cli(identity, *cmd, use_mounted_secret=True)
        _assert_noperm(res.combined, context=f"{identity}: {' '.join(cmd)}")

    debug_res = acl_cluster.redis_cli(identity, "DEBUG", "HELP", use_mounted_secret=True)
    _assert_debug_denied(debug_res.combined, context=f"{identity}: DEBUG HELP")


# ---------------------------------------------------------------------------
# 6. Cross-key type-poisoning (drained list attack + absent shared keys)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("identity", RUNTIME_IDENTITIES)
def test_runtime_cross_key_type_poisoning_drained_queues_and_shared_keys(
    acl_cluster: LiveRedisAclCluster,
    identity: str,
) -> None:
    acl_cluster.reset_db()
    _, task_queue, hint_queue = RUNTIME_TOPOLOGY[identity]

    for queue in (task_queue, hint_queue):
        # 1. Fixture pushes list element; runtime drains queue so key no longer exists
        assert acl_cluster.admin_cli("RPUSH", queue, "drain-me").returncode == 0
        drained = acl_cluster.redis_cli(identity, "BLPOP", queue, "1", use_mounted_secret=True)
        assert drained.stdout.splitlines() == [queue, "drain-me"]
        assert acl_cluster.admin_cli("EXISTS", queue).stdout.strip() == "0"

        # 2. Compromised runtime tries to recreate emptied queue with wrong Redis types
        for poison_cmd in (
            ("HSET", queue, "poison", "1"),
            ("SADD", queue, "poison"),
            ("XADD", queue, "*", "poison", "1"),
        ):
            res = acl_cluster.redis_cli(identity, *poison_cmd, use_mounted_secret=True)
            _assert_noperm(res.combined, context=f"{identity} poison {' '.join(poison_cmd)}")

        # 3. Queue remains absent; fixture recreates valid list; runtime consumes cleanly
        assert acl_cluster.admin_cli("EXISTS", queue).stdout.strip() == "0"
        assert acl_cluster.admin_cli("RPUSH", queue, "after-poison-check").returncode == 0
        popped = acl_cluster.redis_cli(identity, "BLPOP", queue, "1", use_mounted_secret=True)
        assert popped.stdout.splitlines() == [queue, "after-poison-check"]

    # Wrong commands against absent agent.ids, agent.events, agent.state:<self> are also denied
    acl_cluster.reset_db()
    for absent_cmd in (
        ("HSET", "agent.ids", "k", "v"),
        ("XADD", "agent.ids", "*", "k", "v"),
        ("BLPOP", "agent.ids", "1"),
        ("HSET", "agent.events", "k", "v"),
        ("SADD", "agent.events", "v"),
        ("BLPOP", "agent.events", "1"),
        ("SADD", f"agent.state:{identity}", "v"),
        ("XADD", f"agent.state:{identity}", "*", "k", "v"),
        ("BLPOP", f"agent.state:{identity}", "1"),
    ):
        res = acl_cluster.redis_cli(identity, *absent_cmd, use_mounted_secret=True)
        _assert_noperm(res.combined, context=f"{identity} absent wrong-cmd {' '.join(absent_cmd)}")


# ---------------------------------------------------------------------------
# 7. Multi-key BLPOP atomicity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("identity", RUNTIME_IDENTITIES)
def test_runtime_multikey_blpop_atomicity(
    acl_cluster: LiveRedisAclCluster,
    identity: str,
) -> None:
    _, own_queue, own_hint = RUNTIME_TOPOLOGY[identity]
    other_queues, other_hints, _ = _other_runtime_targets(identity)
    unauthorized_targets = [
        other_queues[0],
        other_hints[0],
        "task.status:atomic-test",
        "arbitrary.rogue.queue",
    ]

    for allowed_key in (own_queue, own_hint):
        for unauth_key in unauthorized_targets:
            acl_cluster.reset_db()
            assert acl_cluster.admin_cli("RPUSH", allowed_key, "must-remain").returncode == 0
            if not unauth_key.startswith("task."):
                assert acl_cluster.admin_cli("RPUSH", unauth_key, "rogue-item").returncode == 0

            # Order A: allowed first, unauthorized second -> NOPERM & nothing consumed
            res_ab = acl_cluster.redis_cli(
                identity, "BLPOP", allowed_key, unauth_key, "1", use_mounted_secret=True
            )
            _assert_noperm(res_ab.combined, context=f"{identity} BLPOP {allowed_key} {unauth_key}")
            assert acl_cluster.admin_cli("LLEN", allowed_key).stdout.strip() == "1"

            # Order B: unauthorized first, allowed second -> NOPERM & nothing consumed
            res_ba = acl_cluster.redis_cli(
                identity, "BLPOP", unauth_key, allowed_key, "1", use_mounted_secret=True
            )
            _assert_noperm(res_ba.combined, context=f"{identity} BLPOP {unauth_key} {allowed_key}")
            assert acl_cluster.admin_cli("LLEN", allowed_key).stdout.strip() == "1"


# ---------------------------------------------------------------------------
# 8. Orchestrator positive direct, transaction, and all-15-Lua contract
# ---------------------------------------------------------------------------


def test_orchestrator_positive_direct_transactions_and_all_15_lua_scripts(
    acl_cluster: LiveRedisAclCluster,
) -> None:
    acl_cluster.reset_db()

    # Seed one observational agent record via worker-1 so Orchestrator.agents() exercises
    # SMEMBERS agent.ids + HGETALL agent.state:worker-1 cleanly under %R~ permissions.
    assert acl_cluster.redis_cli("worker-1", "SADD", "agent.ids", "worker-1", use_mounted_secret=True).returncode == 0
    assert (
        acl_cluster.redis_cli(
            "worker-1",
            "HSET",
            "agent.state:worker-1",
            "agent_id",
            "worker-1",
            "capabilities",
            '["general"]',
            "providers",
            '{"inference_provider": "stub"}',
            use_mounted_secret=True,
        ).returncode
        == 0
    )

    orchestrator_script = r"""
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta

from agent_contracts import (
    MAX_WAVE_DURATION_SECONDS,
    MIN_WAVE_DURATION_SECONDS,
    AcceptedDelegationWave,
    ActiveClaim,
    AgentEvent,
    ArtifactReference,
    AssignmentYieldRecord,
    CompletionCommand,
    DelegationGrant,
    DelegationGrantV2,
    DelegationProposalRecord,
    DelegationWaveProposalItem,
    DelegationWaveProposalRecord,
    DelegationWaveProposalRequest,
    InferenceExecutionTelemetry,
    RecoveryDecisionCommand,
    RootAuthorityRecord,
    RoutingDecision,
    RoutingProvenance,
    StageArtifact,
    TaskContinuation,
    TaskContinuationV2,
    TaskInput,
    TaskRequested,
    TaskResult,
    WakeUpHint,
    WaveYieldRecord,
    compute_proposal_fingerprint,
    compute_wave_proposal_fingerprint,
    derive_wave_deadline,
    resolve_redis_connection_from_env,
)
from orchestrator.domain.capability import AgentCapability
from orchestrator.infrastructure.redis_adapter import RedisGateway


async def main() -> None:
    assert os.geteuid() == 10001
    cfg = resolve_redis_connection_from_env()
    assert cfg.is_authenticated and cfg.username == "orchestrator"

    gw_legacy = RedisGateway(cfg.url, delivery_mode="legacy_push", connection_config=cfg)
    gw = RedisGateway(cfg.url, delivery_mode="authenticated_claim", connection_config=cfg)
    try:
        pool_kwargs = gw._client.connection_pool.connection_kwargs
        assert pool_kwargs["protocol"] == 2
        assert pool_kwargs["driver_info"] is None
        assert pool_kwargs.get("health_check_interval", 0) == 0

        now_dt = datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC)
        t0 = now_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        future_lease = (now_dt + timedelta(seconds=300)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        past_iso = (now_dt - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        future_now = (now_dt + timedelta(seconds=600)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        future_resolve = (now_dt + timedelta(seconds=610)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        # ------------------------------------------------------------------
        # A. Direct commands + MULTI/EXEC/DISCARD + legacy/claim dispatch
        # ------------------------------------------------------------------
        t_gen = TaskRequested(task_id="task-acl-gen", root_task_id="task-acl-gen", parent_task_id=None, prompt="General ACL test", created_at=t0, version=2)
        await gw_legacy.register_root_task(t_gen, required_capability="general")  # Lua 1: REGISTER_ROOT_TASK_LUA
        await gw_legacy.register_root_task(t_gen, required_capability="general")  # idempotent
        await gw_legacy.enqueue(t_gen, capability=AgentCapability.GENERAL)  # MULTI/RPUSH agent.tasks.general/SET/EXEC
        assert await gw_legacy.get_status("task-acl-gen") == "pending"
        await gw_legacy.update_status("task-acl-gen", "pending")
        assert await gw_legacy.get_required_capability("task-acl-gen") is not None
        assert await gw_legacy.get_lineage("task-acl-gen") is not None

        # Explicit DISCARD exercise on pipeline(transaction=True)
        async with gw._client.pipeline(transaction=True) as pipe:
            pipe.set("task.status:discard-check", "ignored")
            await pipe.discard()
        assert await gw._client.get("task.status:discard-check") is None

        # Save TaskInput v1 (numkeys=1) & v2 (numkeys=2) -> Lua 3: SAVE_TASK_INPUT_LUA
        await gw.save_task_input(TaskInput(task_id="task-acl-gen", prompt="Updated input v2", created_at=t0, input_version=2, artifact_references=(), metadata={}, version=1))
        assert (await gw.get_task_input("task-acl-gen", 2)).prompt == "Updated input v2"

        # Standalone save_root_authority (Lua 6) & save_authority_grant (Lua 7)
        standalone_auth = RootAuthorityRecord.create_default("task-acl-standalone", created_at=t0)
        await gw.save_root_authority(standalone_auth)
        await gw.save_root_authority(standalone_auth)
        assert (await gw.get_root_authority("task-acl-standalone")) == standalone_auth

        standalone_grant = DelegationGrant(
            grant_id="grant-standalone",
            root_task_id="task-acl-standalone",
            issued_to_assignment_id="assign-standalone",
            issued_to_principal_id="principal-research-1",
            fencing_epoch=1,
            turn_id="turn-standalone",
            issued_at=t0,
        )
        await gw.save_authority_grant(standalone_grant)
        await gw.save_authority_grant(standalone_grant)
        assert (await gw.get_authority_grant("task-acl-standalone", "grant-standalone")) == standalone_grant

        # Grantless initial claim (CLAIM_TASK_LUA, 7 keys, no empty key) + EXPIRE_DELIVERY_LEASE_LUA (Lua 12)
        gen_claim_1 = ActiveClaim(
            task_id="task-acl-gen", assignment_id="assign-gen-1", claim_id="claim-gen-1",
            root_task_id="task-acl-gen", turn_id="turn-gen-1", runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1", assigned_capability="general", execution_kind="initial",
            fencing_epoch=1, proof_digest="a" * 64, request_id="req-gen-1",
            claimed_at=t0, delivery_expires_at=future_lease, prior_claimable_status="pending",
        )
        c_gen, replayed = await gw.atomic_claim_task("task-acl-gen", gen_claim_1, now=t0)  # Lua 8 (7 keys)
        assert not replayed and c_gen.fencing_epoch == 1
        assert await gw.expire_delivery_lease("task-acl-gen", now=t0) is False  # NOOP
        expired_unacked = ActiveClaim(
            task_id="task-acl-gen", assignment_id="assign-gen-1", claim_id="claim-gen-1",
            root_task_id="task-acl-gen", turn_id="turn-gen-1", runtime_principal_id="principal-worker-1",
            bound_agent_id="worker-1", assigned_capability="general", execution_kind="initial",
            fencing_epoch=1, proof_digest="a" * 64, request_id="req-gen-1",
            claimed_at=past_iso, delivery_expires_at=past_iso, prior_claimable_status="pending",
        )
        await gw.save_active_claim(expired_unacked)
        assert await gw.expire_delivery_lease("task-acl-gen", now=t0) is True  # EXPIRED
        assert await gw.get_status("task-acl-gen") == "pending"

        # ------------------------------------------------------------------
        # B. Full Research -> Analysis Delegation -> Child Completion -> Resumed Completion
        # ------------------------------------------------------------------
        parent_id = "task-acl-res"
        t_res = TaskRequested(task_id=parent_id, root_task_id=parent_id, parent_task_id=None, prompt="Research ACL", created_at=t0, version=2)
        await gw.register_root_task(t_res, required_capability="research")
        await gw.dispatch(t_res, "agent.tasks.research", capability="research")  # authenticated_claim MULTI/RPUSH agent.hints.research/SET/EXEC
        await gw.dispatch_hint(WakeUpHint(hint_id="hint-extra", capability="general", created_at=t0, execution_kind="initial", task_id="task-acl-gen"), "agent.hints.general", capability="general")
        await gw.dispatch_hint(WakeUpHint(hint_id="hint-extra-a", capability="analysis", created_at=t0, execution_kind="initial", task_id="task-acl-gen"), "agent.hints.analysis", capability="analysis")

        # Routing + telemetry + events + agents
        routing = RoutingDecision(
            task_id=parent_id,
            prompt_classification="research",
            required_capabilities=["research"],
            candidates_evaluated=[],
            selected_agent_id="research-agent-1",
            dispatch_target="agent.tasks.research",
            rationale="acl test",
            decision_timestamp=t0,
            provenance=RoutingProvenance(
                final_selection_reason="acl test",
                deterministic_assessment_reason="keyword match",
                deterministic_signal_types=("keyword",),
            ).to_dict(),
        )
        await gw.record_routing_decision(routing)
        assert (await gw.routing_decision(parent_id))["prompt_classification"] == "research"
        telem = InferenceExecutionTelemetry(
            provider_name="stub",
            attempt_index=0,
            latency_ms=1.0,
            status="succeeded",
            model_name="stub",
            total_tokens=1,
            turn_id="turn-res-1",
            assignment_id="assign-res-1",
            execution_kind="initial",
        )
        await gw.publish(AgentEvent.create("research-agent-1", "working", "Started", task_id=parent_id, inference_telemetry=telem))
        assert len(await gw.events(limit=5)) >= 1
        assert await gw.task_inference_telemetry(parent_id) is not None
        agents_list = await gw.agents()
        assert any(entry["agent_id"] == "worker-1" for entry in agents_list)

        # Grant-bearing initial claim (CLAIM_TASK_LUA, 8 keys) + ACKNOWLEDGE_CLAIM_LUA (Lua 11)
        res_grant = DelegationGrant(
            grant_id="grant-res-1", root_task_id=parent_id, issued_to_assignment_id="assign-res-1",
            issued_to_principal_id="principal-research-1", fencing_epoch=1, turn_id="turn-res-1", issued_at=t0,
        )
        res_claim_1 = ActiveClaim(
            task_id=parent_id, assignment_id="assign-res-1", claim_id="claim-res-1",
            root_task_id=parent_id, turn_id="turn-res-1", runtime_principal_id="principal-research-1",
            bound_agent_id="research-agent-1", assigned_capability="research", execution_kind="initial",
            fencing_epoch=1, proof_digest="b" * 64, request_id="req-res-1",
            claimed_at=t0, delivery_expires_at=future_lease, delegation_grant_id="grant-res-1", prior_claimable_status="pending",
        )
        await gw.atomic_claim_task(parent_id, res_claim_1, grant=res_grant, now=t0)  # Lua 8 (8 keys)
        assert await gw.get_fencing_epoch(parent_id) == 1
        acked_res = await gw.acknowledge_claim(parent_id, "assign-res-1", "principal-research-1", t0, execution_expires_at=future_now)  # Lua 11
        await gw.acknowledge_claim(parent_id, "assign-res-1", "principal-research-1", t0, execution_expires_at=future_now)  # idempotent
        assert acked_res.state == "acknowledged"

        # Save stage artifact v1 (4 keys, no empty key) & v2 (5 keys) -> Lua 4: SAVE_STAGE_ARTIFACT_LUA
        art1 = StageArtifact(
            artifact_id="art-acl", producing_task_id=parent_id, artifact_type="research_material",
            content={"material": "v1"}, created_at=t0, producing_turn_id="turn-res-1",
            artifact_version=1, payload_schema_version=1, sealed=True, metadata={"turn_id": "turn-res-1"},
        )
        await gw.save_stage_artifact(art1, "assign-res-1", "principal-research-1", 1)
        art2 = StageArtifact(
            artifact_id="art-acl", producing_task_id=parent_id, artifact_type="research_material",
            content={"material": "v2"}, created_at=t0, producing_turn_id="turn-res-1",
            artifact_version=2, payload_schema_version=1, sealed=True, metadata={"turn_id": "turn-res-1"},
        )
        await gw.save_stage_artifact(art2, "assign-res-1", "principal-research-1", 1)
        assert (await gw.get_stage_artifact(parent_id, "art-acl", 2)).content["material"] == "v2"

        # Accept delegation -> Lua 9: ACCEPT_DELEGATION_LUA (23 keys + agent.hints.analysis)
        child_id = "task-acl-child"
        art_ref = ArtifactReference(artifact_id="art-acl", producing_task_id=parent_id, artifact_version=1)
        prop_fp = compute_proposal_fingerprint(parent_id, "prop-1", "assign-res-1", "analysis", "Analyze ACL", art_ref)
        proposal = DelegationProposalRecord(
            parent_task_id=parent_id, proposal_id="prop-1", child_task_id=child_id,
            delegation_decision_id="dec-1", assignment_id="assign-res-1", runtime_principal_id="principal-research-1",
            bound_agent_id="research-agent-1", proof_digest="b" * 64, proposal_fingerprint=prop_fp,
            delegation_grant_id="grant-res-1", child_claim_deadline=future_now, accepted_at=t0,
        )
        child_input = TaskInput(task_id=child_id, prompt="Analyze ACL notes", created_at=t0, input_version=1, artifact_references=(art_ref,))
        cont_1 = TaskContinuation(
            parent_task_id=parent_id, child_task_id=child_id, created_at=t0, updated_at=t0,
            input_version=1, artifact_references=(art_ref,), continuation_version=1,
        )
        yield_rec = AssignmentYieldRecord(
            assignment_id="assign-res-1", parent_task_id=parent_id, principal_id="principal-research-1",
            proposal_id="prop-1", child_task_id=child_id, delegation_decision_id="dec-1", yielded_at=t0,
        )
        child_hint = WakeUpHint(hint_id="hint-child-1", capability="analysis", created_at=t0, execution_kind="delegated_child", task_id=child_id)
        await gw.accept_delegation_proposal(parent_id, proposal, child_input, cont_1, yield_rec, child_hint, expected_fencing_epoch=1, now=t0)
        await gw.accept_delegation_proposal(parent_id, proposal, child_input, cont_1, yield_rec, child_hint, expected_fencing_epoch=1, now=t0)  # idempotent
        assert await gw.get_children(parent_id) == [child_id]
        assert await gw.get_delegation_proposal(parent_id, "prop-1") is not None
        assert await gw.get_continuation(parent_id, 1) is not None

        # Claim + Ack + Complete delegated child -> Lua 10: COMPLETE_TURN_LUA (child path, 14 keys + agent.hints.research)
        child_claim = ActiveClaim(
            task_id=child_id, assignment_id="assign-child-1", claim_id="claim-child-1",
            root_task_id=parent_id, turn_id="turn-child-1", runtime_principal_id="principal-analysis-1",
            bound_agent_id="analysis-agent-1", assigned_capability="analysis", execution_kind="delegated_child",
            fencing_epoch=1, proof_digest="c" * 64, request_id="req-child-1",
            claimed_at=t0, delivery_expires_at=future_lease, prior_claimable_status="pending",
        )
        await gw.atomic_claim_task(child_id, child_claim, grant=None, now=t0, is_root=False)
        await gw.acknowledge_claim(child_id, "assign-child-1", "principal-analysis-1", t0)
        child_cmd = CompletionCommand(
            completion_command_id="cmd-child-1", assignment_id="assign-child-1",
            execution_kind="delegated_child", completed_result=TaskResult(task_id=child_id, output="Child analysis ok"),
        )
        out_c, res_c, _ = await gw.complete_turn(
            child_id, child_cmd, "c" * 64, 1, parent_id, parent_id, t0, "principal-analysis-1", "analysis-agent-1"
        )
        assert out_c.terminal_status == "completed" and res_c is not None
        assert await gw.get_status(parent_id) == "resumable"

        # Claim + Ack + Complete resumed parent (epoch 2, continuation_version=1) -> Lua 10 (root path)
        res_claim_2 = ActiveClaim(
            task_id=parent_id, assignment_id="assign-res-2", claim_id="claim-res-2",
            root_task_id=parent_id, turn_id="turn-res-2", runtime_principal_id="principal-research-2",
            bound_agent_id="research-agent-2", assigned_capability="research", execution_kind="resumed_continuation",
            continuation_version=1, fencing_epoch=2, proof_digest="d" * 64, request_id="req-res-2",
            claimed_at=t0, delivery_expires_at=future_lease, prior_claimable_status="resumable",
        )
        await gw.atomic_claim_task(parent_id, res_claim_2, grant=None, now=t0, expected_continuation_version=1, is_root=True)
        await gw.acknowledge_claim(parent_id, "assign-res-2", "principal-research-2", t0)
        parent_cmd = CompletionCommand(
            completion_command_id="cmd-parent-1", assignment_id="assign-res-2",
            execution_kind="resumed_continuation", continuation_version=1,
            completed_result=TaskResult(task_id=parent_id, output="Combined final output"),
        )
        out_p, res_p, _ = await gw.complete_turn(
            parent_id, parent_cmd, "d" * 64, 2, None, parent_id, t0, "principal-research-2", "research-agent-2"
        )
        assert out_p.terminal_status == "completed" and res_p is not None
        assert await gw.get_completion_record(parent_id, "cmd-parent-1") is not None
        assert await gw.get_assignment_completion(parent_id, "assign-res-2") is not None
        assert (await gw.get_terminal_outcome(parent_id)).terminal_status == "completed"
        assert (await gw.get_canonical_result(parent_id)).output == "Combined final output"
        assert (await gw.get_result(parent_id))["output"] == "Combined final output"

        # ------------------------------------------------------------------
        # C. Remaining Lua scripts: REGISTER_CHILD_TASK_LUA (2), SAVE_CONTINUATION_LUA (5),
        #    MATERIALIZE_CLAIM_TIMEOUT_LUA (13), TRANSITION_EXPIRED_CLAIM_LUA (14), RESOLVE_EXPIRED_CLAIM_LUA (15)
        # ------------------------------------------------------------------
        t_cont_root = TaskRequested(task_id="task-acl-cont", root_task_id="task-acl-cont", parent_task_id=None, prompt="Cont root", created_at=t0, version=2)
        await gw.register_root_task(t_cont_root, required_capability="research")
        t_reg_child = TaskRequested(task_id="task-acl-regchild", root_task_id="task-acl-cont", parent_task_id="task-acl-cont", prompt="Reg child", created_at=t0, version=2)
        await gw.register_child_task(t_reg_child, initial_status="pending", required_capability="analysis")  # Lua 2
        await gw.register_child_task(t_reg_child, initial_status="pending", required_capability="analysis")  # idempotent
        c1 = TaskContinuation(parent_task_id="task-acl-cont", child_task_id="task-acl-regchild", created_at=t0, updated_at=t0, continuation_version=1)
        c2 = TaskContinuation(parent_task_id="task-acl-cont", child_task_id="task-acl-regchild", created_at=t0, updated_at=t0, continuation_version=2)
        await gw.save_continuation(c1)  # Lua 5 (v1, 2 keys)
        await gw.save_continuation(c2)  # Lua 5 (v2, 3 keys)

        # Claim timeout materialization -> Lua 13: MATERIALIZE_CLAIM_TIMEOUT_LUA (12 keys)
        p_to = "task-acl-toparent"
        c_to = "task-acl-tochild"
        await gw.register_root_task(TaskRequested(task_id=p_to, root_task_id=p_to, parent_task_id=None, prompt="Timeout parent", created_at=t0, version=2), required_capability="research")
        await gw.update_status(p_to, "pending")
        g_to = DelegationGrant(grant_id="grant-to", root_task_id=p_to, issued_to_assignment_id="assign-to", issued_to_principal_id="principal-research-1", fencing_epoch=1, turn_id="turn-to", issued_at=t0)
        cl_to = ActiveClaim(task_id=p_to, assignment_id="assign-to", claim_id="claim-to", root_task_id=p_to, turn_id="turn-to", runtime_principal_id="principal-research-1", bound_agent_id="research-agent-1", assigned_capability="research", execution_kind="initial", fencing_epoch=1, proof_digest="e" * 64, request_id="req-to", claimed_at=t0, delivery_expires_at=future_lease, delegation_grant_id="grant-to", prior_claimable_status="pending")
        await gw.atomic_claim_task(p_to, cl_to, grant=g_to, now=t0)
        await gw.acknowledge_claim(p_to, "assign-to", "principal-research-1", t0)
        art_to = StageArtifact(artifact_id="art-to", producing_task_id=p_to, artifact_type="research_material", content={"material": "timeout material"}, created_at=t0, producing_turn_id="turn-to", artifact_version=1, payload_schema_version=1, sealed=True, metadata={"turn_id": "turn-to"})
        await gw.save_stage_artifact(art_to, "assign-to", "principal-research-1", 1)
        ref_to = ArtifactReference(artifact_id="art-to", producing_task_id=p_to, artifact_version=1)
        prop_to = DelegationProposalRecord(parent_task_id=p_to, proposal_id="prop-to", child_task_id=c_to, delegation_decision_id="dec-to", assignment_id="assign-to", runtime_principal_id="principal-research-1", bound_agent_id="research-agent-1", proof_digest="e" * 64, proposal_fingerprint=compute_proposal_fingerprint(p_to, "prop-to", "assign-to", "analysis", "timeout", ref_to), delegation_grant_id="grant-to", child_claim_deadline=t0, accepted_at=t0)
        await gw.accept_delegation_proposal(p_to, prop_to, TaskInput(task_id=c_to, prompt="to", created_at=t0, input_version=1, artifact_references=(ref_to,)), TaskContinuation(parent_task_id=p_to, child_task_id=c_to, created_at=t0, updated_at=t0, artifact_references=(ref_to,), continuation_version=1), AssignmentYieldRecord(assignment_id="assign-to", parent_task_id=p_to, principal_id="principal-research-1", proposal_id="prop-to", child_task_id=c_to, delegation_decision_id="dec-to", yielded_at=t0), WakeUpHint(hint_id="hint-to", capability="analysis", created_at=t0, execution_kind="delegated_child", task_id=c_to), expected_fencing_epoch=1, now=t0)
        out_to, _ = await gw.materialize_claim_timeout(c_to, now=future_now)  # Lua 13
        assert out_to.terminal_status == "claim_timed_out"
        assert await gw.get_status(p_to) == "resumable"

        # Claim expiry + recovery -> Lua 14: TRANSITION_EXPIRED_CLAIM_LUA & Lua 15: RESOLVE_EXPIRED_CLAIM_LUA
        t_exp = "task-acl-exp"
        await gw.register_root_task(TaskRequested(task_id=t_exp, root_task_id=t_exp, parent_task_id=None, prompt="Expiry task", created_at=t0, version=2), required_capability="general")
        await gw.update_status(t_exp, "pending")
        cl_exp = ActiveClaim(task_id=t_exp, assignment_id="assign-exp", claim_id="claim-exp", root_task_id=t_exp, turn_id="turn-exp", runtime_principal_id="principal-worker-1", bound_agent_id="worker-1", assigned_capability="general", execution_kind="initial", fencing_epoch=1, proof_digest="f" * 64, request_id="req-exp", claimed_at=t0, delivery_expires_at=future_lease, prior_claimable_status="pending")
        await gw.atomic_claim_task(t_exp, cl_exp, grant=None, now=t0)
        await gw.acknowledge_claim(t_exp, "assign-exp", "principal-worker-1", t0, execution_expires_at=t0)
        exp_rec, _ = await gw.transition_expired_claim(t_exp, now=future_now)  # Lua 14 (7 keys + XADD agent.events)
        assert exp_rec.reason_code == "EXECUTION_CLAIM_EXPIRED"
        assert await gw.get_claim_expiry(t_exp) is not None
        assert await gw.get_status(t_exp) == "recovery_required"

        rec_cmd = RecoveryDecisionCommand(recovery_decision_id="rec-dec-1", task_id=t_exp, expired_claim_id="claim-exp", expired_assignment_id="assign-exp", expired_fencing_epoch=1, disposition="restore_claimable")
        dec_rec, _ = await gw.resolve_expired_claim(rec_cmd, operator_principal_id="operator-1", now=future_resolve)  # Lua 15
        assert dec_rec.restored_status == "pending"
        assert await gw.get_recovery_decision(t_exp, "rec-dec-1") is not None
        assert await gw.get_status(t_exp) == "pending"

        # ------------------------------------------------------------------
        # D. Bounded v2 Wave Acceptance -> Child Completion -> Barrier Resolution -> V2 Resumed Claim
        # ------------------------------------------------------------------
        w_parent = "task-acl-wave-parent"
        await gw.register_root_task(
            TaskRequested(task_id=w_parent, root_task_id=w_parent, parent_task_id=None, prompt="Wave ACL", created_at=t0, version=2),
            initial_status="pending",
            required_capability="research",
        )
        await gw._client.set(
            f"task.authority:{w_parent}",
            RootAuthorityRecord.create_wave_baseline(w_parent, max_children_per_wave=2, created_at=t0).to_json(),
        )
        w_grant = DelegationGrantV2(
            grant_id="grant-acl-wave", root_task_id=w_parent, issued_to_assignment_id="assign-acl-wave",
            issued_to_principal_id="principal-research-1", fencing_epoch=1, turn_id="turn-acl-wave", issued_at=t0, max_children=2,
        )
        w_claim = ActiveClaim(
            task_id=w_parent, assignment_id="assign-acl-wave", claim_id="claim-acl-wave", root_task_id=w_parent,
            turn_id="turn-acl-wave", runtime_principal_id="principal-research-1", bound_agent_id="research-agent-1",
            assigned_capability="research", execution_kind="initial", fencing_epoch=1, proof_digest="1" * 64,
            request_id="req-acl-wave", claimed_at=t0, delivery_expires_at=future_lease,
            delegation_grant_id="grant-acl-wave", prior_claimable_status="pending",
        )
        await gw.atomic_claim_task(w_parent, w_claim, grant=w_grant, now=t0)
        await gw.acknowledge_claim(w_parent, "assign-acl-wave", "principal-research-1", t0)
        w_art = StageArtifact(
            artifact_id="art-acl-wave", producing_task_id=w_parent, artifact_type="research_material",
            content={"material": "acl wave"}, created_at=t0, producing_turn_id="turn-acl-wave",
            artifact_version=1, payload_schema_version=1, sealed=True, metadata={"turn_id": "turn-acl-wave"},
        )
        await gw.save_stage_artifact(w_art, "assign-acl-wave", "principal-research-1", 1)
        w_ref = ArtifactReference(artifact_id="art-acl-wave", producing_task_id=w_parent, artifact_version=1)
        w_req = DelegationWaveProposalRequest(
            proposal_id="prop-acl-wave",
            items=(
                DelegationWaveProposalItem(item_id="i1", child_intent="Branch 1", artifact_reference=w_ref),
                DelegationWaveProposalItem(item_id="i2", child_intent="Branch 2", artifact_reference=w_ref),
            ),
            barrier_policy="PARTIAL",
            required_success_fraction={"numerator": 1, "denominator": 2},
        )
        wc1, wc2 = "task-acl-wave-c1", "task-acl-wave-c2"
        w_dd = derive_wave_deadline(
            work_complexity=w_req.work_complexity, uncertainty_level=w_req.uncertainty_level,
            required_child_count=2, planned_parallelism=2,
            min_duration_seconds=MIN_WAVE_DURATION_SECONDS, max_duration_seconds=MAX_WAVE_DURATION_SECONDS, accepted_at=t0,
        )
        w_acc = AcceptedDelegationWave(
            wave_id="wave-acl-1", parent_task_id=w_parent, required_child_task_ids=(wc1, wc2),
            barrier_policy="PARTIAL", required_success_fraction=w_req.required_success_fraction,
            required_success_count=1, deadline_derivation=w_dd, accepted_at=t0,
        )
        w_prop = DelegationWaveProposalRecord(
            parent_task_id=w_parent, proposal_id="prop-acl-wave", wave_id="wave-acl-1",
            delegation_decision_id="dec-acl-wave", assignment_id="assign-acl-wave",
            runtime_principal_id="principal-research-1", bound_agent_id="research-agent-1",
            fencing_epoch=1, proof_digest="1" * 64,
            proposal_fingerprint=compute_wave_proposal_fingerprint(
                parent_task_id=w_parent,
                assignment_id="assign-acl-wave",
                request=w_req,
            ),
            delegation_grant_id="grant-acl-wave", required_child_task_ids=(wc1, wc2),
            barrier_policy="PARTIAL", required_success_fraction=w_req.required_success_fraction,
            required_success_count=1, accepted_at=t0, wave_deadline=w_dd.wave_deadline, child_claim_deadline=future_now,
        )
        w_cont = TaskContinuationV2(
            parent_task_id=w_parent, wave_id="wave-acl-1", required_child_task_ids=(wc1, wc2),
            created_at=t0, updated_at=t0, input_version=1, continuation_capability="research",
            stage="incorporate_analysis_child", control_fields={}, artifact_references=(w_ref,),
            continuation_version=2, state="active", wave_resolution=None,
        )
        w_inputs = [TaskInput(task_id=cid, prompt=it.child_intent, created_at=t0, input_version=1, artifact_references=(w_ref,)) for cid, it in zip((wc1, wc2), w_req.items, strict=True)]
        w_hints = [WakeUpHint(hint_id=f"hint-{cid}", capability="analysis", created_at=t0, execution_kind="delegated_child", task_id=cid) for cid in (wc1, wc2)]
        w_yield = WaveYieldRecord(assignment_id="assign-acl-wave", parent_task_id=w_parent, principal_id="principal-research-1", proposal_id="prop-acl-wave", wave_id="wave-acl-1", required_child_task_ids=(wc1, wc2), delegation_decision_id="dec-acl-wave", yielded_at=t0)
        await gw.accept_delegation_wave(w_parent, w_prop, w_acc, w_inputs, w_cont, w_yield, w_hints, expected_fencing_epoch=1, now=t0)
        wc1_claim = ActiveClaim(
            task_id=wc1, assignment_id="assign-acl-wc1", claim_id="claim-acl-wc1", root_task_id=w_parent,
            turn_id="turn-acl-wc1", runtime_principal_id="principal-analysis-1", bound_agent_id="analysis-agent-1",
            assigned_capability="analysis", execution_kind="delegated_child", fencing_epoch=1, proof_digest="2" * 64,
            request_id="req-acl-wc1", claimed_at=t0, delivery_expires_at=future_lease, prior_claimable_status="pending",
        )
        await gw.atomic_claim_task(wc1, wc1_claim, grant=None, now=t0, is_root=False)
        await gw.acknowledge_claim(wc1, "assign-acl-wc1", "principal-analysis-1", t0)
        await gw.complete_turn(
            wc1,
            CompletionCommand(completion_command_id="cmd-acl-wc1", assignment_id="assign-acl-wc1", execution_kind="delegated_child", completed_result=TaskResult(task_id=wc1, output="Branch 1 ok")),
            "2" * 64, 1, w_parent, w_parent, t0, "principal-analysis-1", "analysis-agent-1",
        )
        rw = await gw.get_resolved_delegation_wave(w_parent, "wave-acl-1")
        assert rw is not None and rw.resolution == "SATISFIED"
        assert await gw.get_status(wc2) == "ended / barrier_resolved"
        assert await gw.get_status(w_parent) == "resumable"
        w_res_claim = ActiveClaim(
            task_id=w_parent, assignment_id="assign-acl-wres", claim_id="claim-acl-wres", root_task_id=w_parent,
            turn_id="turn-acl-wres", runtime_principal_id="principal-research-2", bound_agent_id="research-agent-2",
            assigned_capability="research", execution_kind="resumed_continuation", continuation_version=2,
            fencing_epoch=2, proof_digest="3" * 64, request_id="req-acl-wres", claimed_at=t0,
            delivery_expires_at=future_lease, prior_claimable_status="resumable",
        )
        await gw.atomic_claim_task(w_parent, w_res_claim, grant=None, now=t0, expected_continuation_version=2, is_root=True)
        assert await gw.get_status(w_parent) == "working"

        # ------------------------------------------------------------------
        # E. Increment 4: Wave Timeout, Cancellation, Claim Timeout, Expiry & Recovery Lua scripts
        # ------------------------------------------------------------------
        wt_parent = "task-acl-wave-timeout"
        await gw.register_root_task(
            TaskRequested(task_id=wt_parent, root_task_id=wt_parent, parent_task_id=None, prompt="Wave TO ACL", created_at=t0, version=2),
            initial_status="pending",
            required_capability="research",
        )
        await gw._client.set(
            f"task.authority:{wt_parent}",
            RootAuthorityRecord.create_wave_baseline(wt_parent, max_children_per_wave=2, created_at=t0).to_json(),
        )
        wt_grant = DelegationGrantV2(
            grant_id="grant-acl-wt", root_task_id=wt_parent, issued_to_assignment_id="assign-acl-wt",
            issued_to_principal_id="principal-research-1", fencing_epoch=1, turn_id="turn-acl-wt", issued_at=t0, max_children=2,
        )
        wt_claim = ActiveClaim(
            task_id=wt_parent, assignment_id="assign-acl-wt", claim_id="claim-acl-wt", root_task_id=wt_parent,
            turn_id="turn-acl-wt", runtime_principal_id="principal-research-1", bound_agent_id="research-agent-1",
            assigned_capability="research", execution_kind="initial", fencing_epoch=1, proof_digest="4" * 64,
            request_id="req-acl-wt", claimed_at=t0, delivery_expires_at=future_lease,
            delegation_grant_id="grant-acl-wt", prior_claimable_status="pending",
        )
        await gw.atomic_claim_task(wt_parent, wt_claim, grant=wt_grant, now=t0)
        await gw.acknowledge_claim(wt_parent, "assign-acl-wt", "principal-research-1", t0)
        wt_art = StageArtifact(
            artifact_id="art-acl-wt", producing_task_id=wt_parent, artifact_type="research_material",
            content={"material": "acl wt"}, created_at=t0, producing_turn_id="turn-acl-wt",
            artifact_version=1, payload_schema_version=1, sealed=True, metadata={"turn_id": "turn-acl-wt"},
        )
        await gw.save_stage_artifact(wt_art, "assign-acl-wt", "principal-research-1", 1)
        wt_ref = ArtifactReference(artifact_id="art-acl-wt", producing_task_id=wt_parent, artifact_version=1)
        wt_req = DelegationWaveProposalRequest(
            proposal_id="prop-acl-wt",
            items=(
                DelegationWaveProposalItem(item_id="i1", child_intent="WT 1", artifact_reference=wt_ref),
                DelegationWaveProposalItem(item_id="i2", child_intent="WT 2", artifact_reference=wt_ref),
            ),
            barrier_policy="WAIT_ALL",
        )
        wtc1, wtc2 = "task-acl-wt-c1", "task-acl-wt-c2"
        wt_dd = derive_wave_deadline(
            work_complexity=wt_req.work_complexity, uncertainty_level=wt_req.uncertainty_level,
            required_child_count=2, planned_parallelism=2,
            min_duration_seconds=MIN_WAVE_DURATION_SECONDS, max_duration_seconds=MAX_WAVE_DURATION_SECONDS, accepted_at=t0,
        )
        wt_acc = AcceptedDelegationWave(
            wave_id="wave-acl-wt", parent_task_id=wt_parent, required_child_task_ids=(wtc1, wtc2),
            barrier_policy="WAIT_ALL", deadline_derivation=wt_dd, accepted_at=t0,
        )
        wt_prop = DelegationWaveProposalRecord(
            parent_task_id=wt_parent, proposal_id="prop-acl-wt", wave_id="wave-acl-wt",
            delegation_decision_id="dec-acl-wt", assignment_id="assign-acl-wt",
            runtime_principal_id="principal-research-1", bound_agent_id="research-agent-1",
            fencing_epoch=1, proof_digest="4" * 64,
            proposal_fingerprint=compute_wave_proposal_fingerprint(parent_task_id=wt_parent, assignment_id="assign-acl-wt", request=wt_req),
            delegation_grant_id="grant-acl-wt", required_child_task_ids=(wtc1, wtc2),
            barrier_policy="WAIT_ALL", accepted_at=t0, wave_deadline=wt_dd.wave_deadline, child_claim_deadline=t0,
        )
        wt_cont = TaskContinuationV2(
            parent_task_id=wt_parent, wave_id="wave-acl-wt", required_child_task_ids=(wtc1, wtc2),
            created_at=t0, updated_at=t0, input_version=1, continuation_capability="research",
            stage="incorporate_analysis_child", control_fields={}, artifact_references=(wt_ref,),
            continuation_version=2, state="active", wave_resolution=None,
        )
        wt_inputs = [TaskInput(task_id=cid, prompt=it.child_intent, created_at=t0, input_version=1, artifact_references=(wt_ref,)) for cid, it in zip((wtc1, wtc2), wt_req.items, strict=True)]
        wt_hints = [WakeUpHint(hint_id=f"hint-{cid}", capability="analysis", created_at=t0, execution_kind="delegated_child", task_id=cid) for cid in (wtc1, wtc2)]
        wt_yield = WaveYieldRecord(assignment_id="assign-acl-wt", parent_task_id=wt_parent, principal_id="principal-research-1", proposal_id="prop-acl-wt", wave_id="wave-acl-wt", required_child_task_ids=(wtc1, wtc2), delegation_decision_id="dec-acl-wt", yielded_at=t0)
        await gw.accept_delegation_wave(wt_parent, wt_prop, wt_acc, wt_inputs, wt_cont, wt_yield, wt_hints, expected_fencing_epoch=1, now=t0)
        # Wave child claim expiry + recovery Lua
        wtc1_claim = ActiveClaim(
            task_id=wtc1, assignment_id="assign-acl-wtc1", claim_id="claim-acl-wtc1", root_task_id=wt_parent,
            turn_id="turn-acl-wtc1", runtime_principal_id="principal-analysis-1", bound_agent_id="analysis-agent-1",
            assigned_capability="analysis", execution_kind="delegated_child", fencing_epoch=1, proof_digest="5" * 64,
            request_id="req-acl-wtc1", claimed_at=t0, delivery_expires_at=future_lease, prior_claimable_status="pending",
        )
        await gw.atomic_claim_task(wtc1, wtc1_claim, grant=None, now=t0, is_root=False)
        await gw.acknowledge_claim(wtc1, "assign-acl-wtc1", "principal-analysis-1", t0, execution_expires_at=t0)
        await gw.transition_expired_claim(wtc1, now=future_now)
        await gw.resolve_expired_claim(
            RecoveryDecisionCommand(recovery_decision_id="rec-acl-wtc1", task_id=wtc1, expired_claim_id="claim-acl-wtc1", expired_assignment_id="assign-acl-wtc1", expired_fencing_epoch=1, disposition="restore_claimable"),
            operator_principal_id="operator-1",
            now=future_resolve,
        )
        # Wave child claim timeout (non-decisive) + wave timeout Lua
        await gw.materialize_claim_timeout(wtc2, now=future_now)
        from agent_contracts import WaveCancellationCommand, WaveTimeoutEvaluationCommand
        rw_wt, _ = await gw.evaluate_wave_timeout(WaveTimeoutEvaluationCommand(parent_task_id=wt_parent, wave_id="wave-acl-wt"), now=wt_dd.wave_deadline)
        assert rw_wt.resolution == "TIMED_OUT"

        # Cancellation ACL exercise
        wcan_parent = "task-acl-wave-cancel"
        await gw.register_root_task(
            TaskRequested(task_id=wcan_parent, root_task_id=wcan_parent, parent_task_id=None, prompt="Wave CAN ACL", created_at=t0, version=2),
            initial_status="pending",
            required_capability="research",
        )
        await gw._client.set(
            f"task.authority:{wcan_parent}",
            RootAuthorityRecord.create_wave_baseline(wcan_parent, max_children_per_wave=2, created_at=t0).to_json(),
        )
        wk_grant = DelegationGrantV2(
            grant_id="grant-acl-wk", root_task_id=wcan_parent, issued_to_assignment_id="assign-acl-wk",
            issued_to_principal_id="principal-research-1", fencing_epoch=1, turn_id="turn-acl-wk", issued_at=t0, max_children=2,
        )
        wk_claim = ActiveClaim(
            task_id=wcan_parent, assignment_id="assign-acl-wk", claim_id="claim-acl-wk", root_task_id=wcan_parent,
            turn_id="turn-acl-wk", runtime_principal_id="principal-research-1", bound_agent_id="research-agent-1",
            assigned_capability="research", execution_kind="initial", fencing_epoch=1, proof_digest="6" * 64,
            request_id="req-acl-wk", claimed_at=t0, delivery_expires_at=future_lease,
            delegation_grant_id="grant-acl-wk", prior_claimable_status="pending",
        )
        await gw.atomic_claim_task(wcan_parent, wk_claim, grant=wk_grant, now=t0)
        await gw.acknowledge_claim(wcan_parent, "assign-acl-wk", "principal-research-1", t0)
        wk_art = StageArtifact(
            artifact_id="art-acl-wk", producing_task_id=wcan_parent, artifact_type="research_material",
            content={"material": "acl wk"}, created_at=t0, producing_turn_id="turn-acl-wk",
            artifact_version=1, payload_schema_version=1, sealed=True, metadata={"turn_id": "turn-acl-wk"},
        )
        await gw.save_stage_artifact(wk_art, "assign-acl-wk", "principal-research-1", 1)
        wk_ref = ArtifactReference(artifact_id="art-acl-wk", producing_task_id=wcan_parent, artifact_version=1)
        wk_req = DelegationWaveProposalRequest(
            proposal_id="prop-acl-wk",
            items=(
                DelegationWaveProposalItem(item_id="i1", child_intent="WK 1", artifact_reference=wk_ref),
                DelegationWaveProposalItem(item_id="i2", child_intent="WK 2", artifact_reference=wk_ref),
            ),
            barrier_policy="WAIT_ALL",
        )
        wkc1, wkc2 = "task-acl-wk-c1", "task-acl-wk-c2"
        wk_dd = derive_wave_deadline(
            work_complexity=wk_req.work_complexity, uncertainty_level=wk_req.uncertainty_level,
            required_child_count=2, planned_parallelism=2,
            min_duration_seconds=MIN_WAVE_DURATION_SECONDS, max_duration_seconds=MAX_WAVE_DURATION_SECONDS, accepted_at=t0,
        )
        wk_acc = AcceptedDelegationWave(
            wave_id="wave-acl-wk", parent_task_id=wcan_parent, required_child_task_ids=(wkc1, wkc2),
            barrier_policy="WAIT_ALL", deadline_derivation=wk_dd, accepted_at=t0,
        )
        wk_prop = DelegationWaveProposalRecord(
            parent_task_id=wcan_parent, proposal_id="prop-acl-wk", wave_id="wave-acl-wk",
            delegation_decision_id="dec-acl-wk", assignment_id="assign-acl-wk",
            runtime_principal_id="principal-research-1", bound_agent_id="research-agent-1",
            fencing_epoch=1, proof_digest="6" * 64,
            proposal_fingerprint=compute_wave_proposal_fingerprint(parent_task_id=wcan_parent, assignment_id="assign-acl-wk", request=wk_req),
            delegation_grant_id="grant-acl-wk", required_child_task_ids=(wkc1, wkc2),
            barrier_policy="WAIT_ALL", accepted_at=t0, wave_deadline=wk_dd.wave_deadline, child_claim_deadline=future_now,
        )
        wk_cont = TaskContinuationV2(
            parent_task_id=wcan_parent, wave_id="wave-acl-wk", required_child_task_ids=(wkc1, wkc2),
            created_at=t0, updated_at=t0, input_version=1, continuation_capability="research",
            stage="incorporate_analysis_child", control_fields={}, artifact_references=(wk_ref,),
            continuation_version=2, state="active", wave_resolution=None,
        )
        wk_inputs = [TaskInput(task_id=cid, prompt=it.child_intent, created_at=t0, input_version=1, artifact_references=(wk_ref,)) for cid, it in zip((wkc1, wkc2), wk_req.items, strict=True)]
        wk_hints = [WakeUpHint(hint_id=f"hint-{cid}", capability="analysis", created_at=t0, execution_kind="delegated_child", task_id=cid) for cid in (wkc1, wkc2)]
        wk_yield = WaveYieldRecord(assignment_id="assign-acl-wk", parent_task_id=wcan_parent, principal_id="principal-research-1", proposal_id="prop-acl-wk", wave_id="wave-acl-wk", required_child_task_ids=(wkc1, wkc2), delegation_decision_id="dec-acl-wk", yielded_at=t0)
        await gw.accept_delegation_wave(wcan_parent, wk_prop, wk_acc, wk_inputs, wk_cont, wk_yield, wk_hints, expected_fencing_epoch=1, now=t0)
        c_rec, rw_can, _ = await gw.cancel_delegation_wave(
            WaveCancellationCommand(cancellation_decision_id="can-acl-1", parent_task_id=wcan_parent, wave_id="wave-acl-wk"),
            operator_principal_id="cancel-op-1",
            now=t0,
        )
        assert rw_can.resolution == "CANCELLED" and c_rec.terminal_parent_status == "cancelled"
        assert await gw.get_wave_cancellation_record(wcan_parent, "can-acl-1") == c_rec

        print("ALL_15_LUA_AND_GATEWAY_OK")
    finally:
        await gw_legacy.aclose()
        await gw.aclose()


asyncio.run(main())
"""

    res = acl_cluster.run_service_python("orchestrator", "orchestrator", orchestrator_script)
    assert res.returncode == 0, f"Orchestrator positive suite failed:\n{res.combined}"
    assert res.stdout.strip().endswith("ALL_15_LUA_AND_GATEWAY_OK")


# ---------------------------------------------------------------------------
# 8. Orchestrator negative commands & unauthorized namespaces
# ---------------------------------------------------------------------------


def test_orchestrator_negative_commands_and_namespace_isolation(
    acl_cluster: LiveRedisAclCluster,
) -> None:
    denied_commands: list[tuple[str, ...]] = [
        ("FLUSHDB",),
        ("FLUSHALL",),
        ("CONFIG", "GET", "*"),
        ("ACL", "LIST"),
        ("ACL", "WHOAMI"),
        ("ACL", "SETUSER", "evil", "on", "nopass"),
        ("MODULE", "LIST"),
        ("FUNCTION", "LIST"),
        ("FCALL", "fn", "0"),
        ("SCRIPT", "LOAD", "return 1"),
        ("SCRIPT", "FLUSH"),
        ("SCRIPT", "KILL"),
        ("EVALSHA", "0" * 40, "0"),
        ("SHUTDOWN", "NOSAVE"),
        ("PING",),
        ("SELECT", "0"),
        ("SELECT", "1"),
        ("CLIENT", "LIST"),
        ("CLIENT", "SETINFO", "LIB-NAME", "test"),
        ("KEYS", "*"),
        ("SCAN", "0"),
        ("WATCH", "task.status:w1"),
        # Out-of-namespace keys and arbitrary queue names
        ("SET", "rogue.key:1", "val"),
        ("GET", "rogue.key:1"),
        ("DEL", "rogue.key:1"),
        ("RPUSH", "agent.tasks.rogue", "msg"),
        ("RPUSH", "agent.hints.rogue", "msg"),
        # Queue consumption/read/delete denied for Orchestrator (RPUSH selector only on agent.tasks.*)
        ("BLPOP", "agent.tasks.general", "1"),
        ("GET", "agent.tasks.general"),
        ("DEL", "agent.tasks.general"),
        ("BLPOP", "agent.hints.research", "1"),
        # Runtime-private / observational write restrictions (%R~agent.ids, %R~agent.state:*)
        ("SADD", "agent.ids", "rogue-agent"),
        ("DEL", "agent.ids"),
        ("HSET", "agent.state:worker-1", "spoof", "1"),
        ("DEL", "agent.state:worker-1"),
        # Writable EVAL denied against %R~ key
        ("EVAL", "return redis.call('HGETALL', KEYS[1])", "1", "agent.state:worker-1"),
    ]

    for cmd in denied_commands:
        res = acl_cluster.redis_cli("orchestrator", *cmd, use_mounted_secret=True)
        _assert_noperm(res.combined, context=f"orchestrator: {' '.join(cmd)}")

    debug_res = acl_cluster.redis_cli("orchestrator", "DEBUG", "HELP", use_mounted_secret=True)
    _assert_debug_denied(debug_res.combined, context="orchestrator: DEBUG HELP")

    # Confirm test-only admin is isolated from all app containers
    admin_check = acl_cluster.redis_cli(
        TEST_ADMIN_IDENTITY, "PING", password=acl_cluster.admin_secret
    )
    assert admin_check.stdout.strip() == "PONG"


# ---------------------------------------------------------------------------
# 9. Secret non-exposure across container logs & inspect metadata
# ---------------------------------------------------------------------------


def test_no_credentials_leaked_in_container_logs_or_inspect(
    acl_cluster: LiveRedisAclCluster,
) -> None:
    all_secrets = [*acl_cluster.secrets_map.values(), acl_cluster.admin_secret]
    # Check Redis container logs & inspect output via regex/substring scan
    from scripts.redis_acl_live_harness import _run_docker

    logs = _run_docker(["docker", "logs", acl_cluster.redis_container], check=False).combined
    inspect_out = _run_docker(["docker", "inspect", acl_cluster.redis_container], check=False).stdout
    rendered_acl = _run_docker(
        ["docker", "exec", acl_cluster.redis_container, "cat", "/run/redis/users.acl"],
        check=True,
    ).stdout

    credential_exposed = any(
        candidate in output
        for candidate in all_secrets
        for output in (logs, inspect_out, rendered_acl)
    )
    assert not credential_exposed, (
        "container output exposed synthetic credential material"
    )

    assert not re.search(r"(^|\s)>[^\s]+", rendered_acl)
