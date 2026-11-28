"""Live Redis 7.4.12 replay-denial to failed-health subtraction scenario.

Writes qualifying transactional state, restarts the same data volume under
a narrowed default-user rule, and proves the replay-integrity health gate
refuses healthy status while PING alone would still succeed. The denial must
be observed before the readiness boundary; a denial appearing only after
readiness fails the test. The full-rule restart control proves the same state
replays cleanly to healthy, seals that verdict, and proves a later runtime
authorization failure — established by the client response and the absence
of the write, not required to appear in the notice log — leaves the sealed
startup verdict and health unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import stat
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.redis_acl_live_harness import (
    REDIS_IMAGE,
    CliResult,
    LiveRedisAclCluster,
    _run_docker,
    redis_replay_log_prep,
)
from scripts.redis_test_support import (
    make_test_gateway,
    register_cluster,
    unregister_cluster,
)

pytestmark = [pytest.mark.live_redis_rotation]

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task-replay-health-subtraction"


def _production_marker_bytes(generation_text: str) -> bytes:
    """Byte-exact production startup-marker shape.

    Mirrors ``redis-acl-entrypoint.sh`` ``printf '{"generation_sha256":"%s"}\\n'``:
    the marker carries only the active generation digest, no random identity.
    Compatibility tolerates extra obsolete fields (such as a legacy
    ``start_id``) but never the omission of the current required fields.
    """
    digest = hashlib.sha256(generation_text.encode()).hexdigest()
    return f'{{"generation_sha256":"{digest}"}}\n'.encode()


def _write_qualifying_state(cluster: LiveRedisAclCluster) -> None:
    async def main() -> None:
        from agent_contracts import TaskRequested

        gateway = make_test_gateway(cluster.redis_url)
        try:
            task = TaskRequested(
                task_id=TASK_ID,
                root_task_id=TASK_ID,
                parent_task_id=None,
                prompt="Replay-integrity subtraction verification",
                created_at="2026-11-24T00:00:00Z",
                version=2,
            )
            await gateway.register_root_task(task, required_capability="general")
            await gateway.dispatch(task, "agent.tasks.general", capability="general")
        finally:
            await gateway.aclose()

    asyncio.run(main())


def _wait_for_log_text(log_file: Path, text: str, timeout: float = 30.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_file.exists():
            content = log_file.read_text(encoding="utf-8", errors="replace")
            if text in content:
                return content
        time.sleep(0.5)
    content = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
    raise AssertionError(f"timed out waiting for {text!r} in replay log; log content:\n{content}")


def _prepare_replay_evidence_dir(
    evidence: Path,
    acl_text: str,
    generation_text: str,
    password_text: str,
) -> None:
    """Create host-side replay evidence with portable least-privilege modes.

    - evidence dir: 0755 (container redis user traverses; host cleans up)
    - users.acl: 0644 (hashed ACL only, no secrets; container redis user reads)
    - generation.json, startup.json, healthcheck.password: 0600 (root-only in container)
    - replay.log: 0664 (pre-created host-side; container root:redis 0664 lets redis write & host read)
    """
    evidence.mkdir(parents=True, exist_ok=True)
    evidence.chmod(0o755)
    acl_path = evidence / "users.acl"
    acl_path.write_text(acl_text, encoding="utf-8")
    acl_path.chmod(0o644)
    gen_path = evidence / "generation.json"
    gen_path.write_text(generation_text, encoding="utf-8")
    gen_path.chmod(0o600)
    pw_path = evidence / "healthcheck.password"
    pw_path.write_text(password_text, encoding="utf-8")
    pw_path.chmod(0o600)
    log_path = evidence / "replay.log"
    log_path.write_text("", encoding="utf-8")
    log_path.chmod(0o664)
    startup_path = evidence / "startup.json"
    startup_path.write_bytes(_production_marker_bytes(generation_text))
    startup_path.chmod(0o600)

    assert stat.S_IMODE(evidence.stat().st_mode) == 0o755
    assert stat.S_IMODE(acl_path.stat().st_mode) == 0o644
    assert stat.S_IMODE(pw_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(gen_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o664


def _start_replay_container(
    cluster: LiveRedisAclCluster,
    name: str,
    evidence: Path,
    acl_text: str,
    generation_text: str,
    password_text: str,
) -> None:
    _prepare_replay_evidence_dir(evidence, acl_text, generation_text, password_text)
    # Root bootstrap inside the container normalizes the bind-mounted logfile
    # (Docker Desktop bind mounts do not inherit the tmpfs ownership the
    # production entrypoint establishes) before dropping to the redis user.
    # Redis never runs as root here.
    bootstrap = (
        "set -eu\n"
        + redis_replay_log_prep(log_path="/run/redis-evd/replay.log", log_mode="0664")
        + "exec setpriv --reuid=redis --regid=redis --clear-groups -- "
        "redis-server --appendonly yes "
        "--aclfile /run/redis-evd/users.acl "
        "--logfile /run/redis-evd/replay.log "
        "--loglevel notice\n"
    )
    _run_docker(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "--mount",
            f"type=bind,src={evidence},dst=/run/redis-evd",
            "--mount",
            (
                f"type=bind,src={REPO_ROOT / 'infrastructure' / 'redis' / 'redis-healthcheck.sh'},"
                "dst=/usr/local/bin/redis-healthcheck.sh,readonly"
            ),
            "--mount",
            f"type=volume,src={cluster.data_volume_name},dst=/data",
            "--entrypoint",
            "/bin/sh",
            REDIS_IMAGE,
            "-c",
            bootstrap,
        ],
        timeout=120,
    )
    cluster.created_containers.append(name)


def _run_evidence_health(name: str) -> CliResult:
    return _run_docker(
        [
            "docker",
            "exec",
            name,
            "/bin/sh",
            "-c",
            (
                "REDIS_HEALTHCHECK_PASSWORD_FILE=/run/redis-evd/healthcheck.password "
                "REDIS_REPLAY_LOG=/run/redis-evd/replay.log "
                "REDIS_STARTUP_MARKER=/run/redis-evd/startup.json "
                "REDIS_CREDENTIAL_GENERATION_FILE=/run/redis-evd/generation.json "
                "REDIS_REPLAY_SEAL=/run/redis-evd/startup-sealed.json "
                "REDIS_REPLAY_EVIDENCE=/run/redis-evd/startup-evidence.log "
                "/bin/sh /usr/local/bin/redis-healthcheck.sh"
            ),
        ],
        check=False,
        timeout=30,
    )


def _read_container_replay_seal(name: str) -> bytes:
    """Read /run/redis-evd/startup-sealed.json via the container without weakening its 0640 mode."""
    res = _run_docker(
        ["docker", "exec", name, "cat", "/run/redis-evd/startup-sealed.json"],
        check=False,
        timeout=30,
    )
    assert res.returncode == 0, res.combined
    return res.stdout.encode("utf-8")


def _assert_redis_runs_unprivileged(name: str) -> None:
    comm = _run_docker(["docker", "exec", name, "cat", "/proc/1/comm"], check=False, timeout=30)
    assert comm.stdout.strip() == "redis-server", comm.combined
    uid = _run_docker(
        ["docker", "exec", name, "/bin/sh", "-c", "awk '/^Uid:/ {print $2}' /proc/1/status"],
        check=False,
        timeout=30,
    )
    assert uid.stdout.strip() not in ("", "0"), uid.combined


def _exec_rejected_transaction(name: str, password: str) -> CliResult:
    """Run an orchestrator MULTI/EXEC whose queued write the ACL must reject.

    Uses the same EXEC-time ACL revalidation path as AOF replay, but after
    readiness, so the denial is a runtime authorization failure rather than
    replay evidence.
    """
    return _run_docker(
        [
            "docker",
            "exec",
            "-i",
            name,
            "/bin/sh",
            "-c",
            (
                'IFS= read -r pw; printf "%s\\n" "MULTI" '
                '"SET agent.tasks.general probe probeval" "EXEC" | '
                'REDISCLI_AUTH="$pw" REDISCLI_HISTFILE=/dev/null '
                "exec redis-cli --raw --user orchestrator"
            ),
        ],
        check=False,
        stdin_text=password + "\n",
        timeout=30,
    )


def _ping_as_healthcheck(name: str, password: str) -> CliResult:
    return _run_docker(
        [
            "docker",
            "exec",
            "-i",
            name,
            "/bin/sh",
            "-c",
            (
                'IFS= read -r pw; REDISCLI_AUTH="$pw" REDISCLI_HISTFILE=/dev/null '
                "exec redis-cli --raw --user healthcheck ping"
            ),
        ],
        check=False,
        stdin_text=password + "\n",
        timeout=30,
    )


def test_narrowed_replay_rule_denial_fails_health() -> None:
    cluster = LiveRedisAclCluster.start(
        build_service_images=False,
        publish_host_port=True,
        appendonly=True,
        persistent_data=True,
    )
    register_cluster(cluster)
    try:
        _write_qualifying_state(cluster)
        sync = cluster.admin_cli("WAITAOF", "1", "0", "5000")
        assert sync.returncode == 0, sync.combined

        rendered = _run_docker(
            ["docker", "exec", cluster.redis_container, "cat", "/run/redis/users.acl"],
            timeout=30,
        )
        assert rendered.returncode == 0, rendered.combined
        full_acl = rendered.stdout
        assert "%W~task.status:*" in full_acl
        narrowed_acl = full_acl.replace(" %W~task.status:*", "")
        assert narrowed_acl != full_acl

        generation_text = (cluster.secret_dir / "current" / "generation.json").read_text(
            encoding="utf-8"
        )
        password_text = cluster.secrets_map["healthcheck"]

        _run_docker(["docker", "stop", cluster.redis_container], timeout=60)

        full_evidence = cluster.temp_dir / "replay-full"
        _start_replay_container(
            cluster,
            f"{cluster.redis_container}-full",
            full_evidence,
            full_acl,
            generation_text,
            password_text,
        )
        full_log = _wait_for_log_text(full_evidence / "replay.log", "Ready to accept connections")
        assert "NOPERM" not in full_log, f"clean replay unexpectedly denied:\n{full_log}"
        assert "CRITICAL" not in full_log, f"clean replay unexpectedly failed:\n{full_log}"
        healthy = _run_evidence_health(f"{cluster.redis_container}-full")
        assert healthy.returncode == 0, healthy.combined
        _assert_redis_runs_unprivileged(f"{cluster.redis_container}-full")
        sealed_path = full_evidence / "startup-sealed.json"
        assert sealed_path.exists(), "clean startup must seal its replay verdict"
        assert stat.S_IMODE(sealed_path.stat().st_mode) == 0o640, (
            "startup seal must remain restrictive (0640) on the host"
        )
        sealed_full = _read_container_replay_seal(f"{cluster.redis_container}-full")

        # The runtime denial is proven by the client response, not by the
        # notice logfile: Redis does not emit post-readiness NOPERM lines at
        # notice level. EXECABORT is atomic: the queued SET never executed,
        # so the rejection itself proves the write did not occur.
        rejected = _exec_rejected_transaction(
            f"{cluster.redis_container}-full", cluster.secrets_map["orchestrator"]
        )
        assert "EXECABORT" in rejected.stdout or "NOPERM" in rejected.stdout, (
            "orchestrator MULTI/EXEC with an unauthorized queued write must be "
            f"rejected; got: {rejected.combined}"
        )
        resealed = _run_evidence_health(f"{cluster.redis_container}-full")
        assert resealed.returncode == 0, (
            "a later runtime authorization failure must not poison the sealed "
            f"startup verdict: {resealed.combined}"
        )
        assert _read_container_replay_seal(f"{cluster.redis_container}-full") == sealed_full, (
            "sealed startup evidence must remain unchanged by later runtime logs"
        )
        _run_docker(["docker", "stop", f"{cluster.redis_container}-full"], timeout=60)

        narrowed_evidence = cluster.temp_dir / "replay-narrowed"
        _start_replay_container(
            cluster,
            f"{cluster.redis_container}-narrowed",
            narrowed_evidence,
            narrowed_acl,
            generation_text,
            password_text,
        )
        narrowed_log = _wait_for_log_text(
            narrowed_evidence / "replay.log", "Ready to accept connections"
        )
        assert "NOPERM" in narrowed_log or "CRITICAL" in narrowed_log, (
            "expected an AOF replay denial for the subtracted selector at "
            f"notice log level; log content:\n{narrowed_log}"
        )
        ping = _ping_as_healthcheck(f"{cluster.redis_container}-narrowed", password_text)
        assert ping.stdout.strip() == "PONG", (
            "PING succeeded while replay dropped transactions; "
            "PING alone must never imply replay integrity"
        )
        unhealthy = _run_evidence_health(f"{cluster.redis_container}-narrowed")
        assert unhealthy.returncode != 0, (
            "health gate accepted a startup with replay denials; log content:\n" + narrowed_log
        )
        _run_docker(["docker", "stop", f"{cluster.redis_container}-narrowed"], timeout=60)
        complete_log = (narrowed_evidence / "replay.log").read_text(
            encoding="utf-8", errors="replace"
        )
        ready_at = complete_log.find("Ready to accept connections")
        assert ready_at != -1, f"readiness boundary missing:\n{complete_log}"
        denial_at = complete_log.find("NOPERM")
        if denial_at == -1:
            denial_at = complete_log.find("CRITICAL")
        assert denial_at != -1, f"no replay denial observed:\n{complete_log}"
        assert denial_at < ready_at, (
            "replay denial must occur before the readiness boundary; a denial "
            f"observed only after readiness does not prove replay failure:\n{complete_log}"
        )
    finally:
        unregister_cluster(cluster)
        cluster.cleanup()


def _write_full_envelope_state(cluster: LiveRedisAclCluster) -> None:
    """Exercise qualifying RedisGateway paths covering every projection selector and +select."""
    from agent_contracts import (
        MAX_WAVE_DURATION_SECONDS,
        MIN_WAVE_DURATION_SECONDS,
        AcceptedDelegationWave,
        ActiveClaim,
        ArtifactReference,
        AssignmentYieldRecord,
        CompletionCommand,
        DelegationGrant,
        DelegationGrantV2,
        DelegationProposalRecord,
        DelegationWaveProposalItem,
        DelegationWaveProposalRecord,
        DelegationWaveProposalRequest,
        RecoveryDecisionCommand,
        RootAuthorityRecord,
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
    )
    from orchestrator.infrastructure.redis_adapter import RedisGateway

    async def main() -> None:
        gw_legacy = RedisGateway(
            cluster.redis_url,
            delivery_mode="legacy_push",
            username="orchestrator",
            password=cluster.secrets_map["orchestrator"],
        )
        gw = RedisGateway(
            cluster.redis_url,
            delivery_mode="authenticated_claim",
            username="orchestrator",
            password=cluster.secrets_map["orchestrator"],
        )
        try:
            now_dt = datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC)
            t0 = now_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            future_lease = (now_dt + timedelta(seconds=300)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            future_now = (now_dt + timedelta(seconds=600)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            future_resolve = (now_dt + timedelta(seconds=610)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            # First write after startup is a multi-effect Lua call -> emits SELECT 0 inside MULTI.
            parent_id = "task-env-parent"
            t_parent = TaskRequested(
                task_id=parent_id,
                root_task_id=parent_id,
                parent_task_id=None,
                prompt="Envelope research parent",
                created_at=t0,
                version=2,
            )
            await gw.register_root_task(t_parent, required_capability="research")
            await gw.dispatch(t_parent, "agent.tasks.research", capability="research")

            # Cover all legacy RPUSH agent.tasks.* and authenticated_claim RPUSH agent.hints.*
            for cap in ("general", "research", "analysis"):
                t_q = TaskRequested(
                    task_id=f"task-env-q-{cap}",
                    root_task_id=f"task-env-q-{cap}",
                    parent_task_id=None,
                    prompt=f"Envelope queue {cap}",
                    created_at=t0,
                    version=2,
                )
                await gw_legacy.register_root_task(t_q, required_capability=cap)
                await gw_legacy.dispatch(t_q, f"agent.tasks.{cap}", capability=cap)
                await gw.dispatch(t_q, f"agent.tasks.{cap}", capability=cap)

            # Register child task on a dedicated root -> SADD task.children:*
            reg_parent_id = "task-env-regparent"
            await gw.register_root_task(
                TaskRequested(
                    task_id=reg_parent_id,
                    root_task_id=reg_parent_id,
                    parent_task_id=None,
                    prompt="Envelope child-reg root",
                    created_at=t0,
                    version=2,
                ),
                required_capability="research",
            )
            t_reg_child = TaskRequested(
                task_id="task-env-regchild",
                root_task_id=reg_parent_id,
                parent_task_id=reg_parent_id,
                prompt="Envelope registered child",
                created_at=t0,
                version=2,
            )
            await gw.register_child_task(
                t_reg_child, initial_status="pending", required_capability="analysis"
            )

            # Claim with grant + acknowledge + save_stage_artifact -> fence, claim, grant, artifact, artifacts
            res_grant = DelegationGrant(
                grant_id="grant-env-1",
                root_task_id=parent_id,
                issued_to_assignment_id="assign-env-1",
                issued_to_principal_id="principal-research-1",
                fencing_epoch=1,
                turn_id="turn-env-1",
                issued_at=t0,
            )
            res_claim = ActiveClaim(
                task_id=parent_id,
                assignment_id="assign-env-1",
                claim_id="claim-env-1",
                root_task_id=parent_id,
                turn_id="turn-env-1",
                runtime_principal_id="principal-research-1",
                bound_agent_id="research-agent-1",
                assigned_capability="research",
                execution_kind="initial",
                fencing_epoch=1,
                proof_digest="a" * 64,
                request_id="req-env-1",
                claimed_at=t0,
                delivery_expires_at=future_lease,
                delegation_grant_id="grant-env-1",
                prior_claimable_status="pending",
            )
            await gw.atomic_claim_task(parent_id, res_claim, grant=res_grant, now=t0)
            await gw.acknowledge_claim(
                parent_id, "assign-env-1", "principal-research-1", t0, execution_expires_at=future_now
            )
            art = StageArtifact(
                artifact_id="art-env-1",
                producing_task_id=parent_id,
                artifact_type="research_material",
                content={"material": "envelope artifact"},
                created_at=t0,
                producing_turn_id="turn-env-1",
                artifact_version=1,
                payload_schema_version=1,
                sealed=True,
                metadata={"turn_id": "turn-env-1"},
            )
            await gw.save_stage_artifact(art, "assign-env-1", "principal-research-1", 1)

            # Accept delegation -> proposal, claim_deadline, continuation, continuation_latest,
            # DEL task.claim:*, assignment_yield, consumed_waves, consumed_grants, delegation_decisions
            child_id = "task-env-child"
            art_ref = ArtifactReference(
                artifact_id="art-env-1", producing_task_id=parent_id, artifact_version=1
            )
            prop_fp = compute_proposal_fingerprint(
                parent_id, "prop-env-1", "assign-env-1", "analysis", "Analyze envelope", art_ref
            )
            proposal = DelegationProposalRecord(
                parent_task_id=parent_id,
                proposal_id="prop-env-1",
                child_task_id=child_id,
                delegation_decision_id="dec-env-1",
                assignment_id="assign-env-1",
                runtime_principal_id="principal-research-1",
                bound_agent_id="research-agent-1",
                proof_digest="a" * 64,
                proposal_fingerprint=prop_fp,
                delegation_grant_id="grant-env-1",
                child_claim_deadline=future_now,
                accepted_at=t0,
            )
            child_input = TaskInput(
                task_id=child_id,
                prompt="Analyze envelope",
                created_at=t0,
                input_version=1,
                artifact_references=(art_ref,),
            )
            cont = TaskContinuation(
                parent_task_id=parent_id,
                child_task_id=child_id,
                created_at=t0,
                updated_at=t0,
                input_version=1,
                artifact_references=(art_ref,),
                continuation_version=1,
            )
            yield_rec = AssignmentYieldRecord(
                assignment_id="assign-env-1",
                parent_task_id=parent_id,
                principal_id="principal-research-1",
                proposal_id="prop-env-1",
                child_task_id=child_id,
                delegation_decision_id="dec-env-1",
                yielded_at=t0,
            )
            child_hint = WakeUpHint(
                hint_id="hint-env-child",
                capability="analysis",
                created_at=t0,
                execution_kind="delegated_child",
                task_id=child_id,
            )
            await gw.accept_delegation_proposal(
                parent_id,
                proposal,
                child_input,
                cont,
                yield_rec,
                child_hint,
                expected_fencing_epoch=1,
                now=t0,
            )

            # Complete delegated child -> completion, outcome, result, assignment_completion, child_completion
            child_claim = ActiveClaim(
                task_id=child_id,
                assignment_id="assign-env-child",
                claim_id="claim-env-child",
                root_task_id=parent_id,
                turn_id="turn-env-child",
                runtime_principal_id="principal-analysis-1",
                bound_agent_id="analysis-agent-1",
                assigned_capability="analysis",
                execution_kind="delegated_child",
                fencing_epoch=1,
                proof_digest="b" * 64,
                request_id="req-env-child",
                claimed_at=t0,
                delivery_expires_at=future_lease,
                prior_claimable_status="pending",
            )
            await gw.atomic_claim_task(child_id, child_claim, grant=None, now=t0, is_root=False)
            await gw.acknowledge_claim(child_id, "assign-env-child", "principal-analysis-1", t0)
            child_cmd = CompletionCommand(
                completion_command_id="cmd-env-child",
                assignment_id="assign-env-child",
                execution_kind="delegated_child",
                completed_result=TaskResult(task_id=child_id, output="Envelope child ok"),
            )
            await gw.complete_turn(
                child_id,
                child_cmd,
                "b" * 64,
                1,
                parent_id,
                parent_id,
                t0,
                "principal-analysis-1",
                "analysis-agent-1",
            )

            # Expiry + recovery -> claim_expiry, XADD agent.events, recovery_decision
            exp_id = "task-env-expiry"
            await gw.register_root_task(
                TaskRequested(
                    task_id=exp_id,
                    root_task_id=exp_id,
                    parent_task_id=None,
                    prompt="Envelope expiry",
                    created_at=t0,
                    version=2,
                ),
                required_capability="general",
            )
            await gw.update_status(exp_id, "pending")
            exp_claim = ActiveClaim(
                task_id=exp_id,
                assignment_id="assign-env-exp",
                claim_id="claim-env-exp",
                root_task_id=exp_id,
                turn_id="turn-env-exp",
                runtime_principal_id="principal-worker-1",
                bound_agent_id="worker-1",
                assigned_capability="general",
                execution_kind="initial",
                fencing_epoch=1,
                proof_digest="c" * 64,
                request_id="req-env-exp",
                claimed_at=t0,
                delivery_expires_at=future_lease,
                prior_claimable_status="pending",
            )
            await gw.atomic_claim_task(exp_id, exp_claim, grant=None, now=t0)
            await gw.acknowledge_claim(
                exp_id, "assign-env-exp", "principal-worker-1", t0, execution_expires_at=t0
            )
            await gw.transition_expired_claim(exp_id, now=future_now)
            rec_cmd = RecoveryDecisionCommand(
                recovery_decision_id="rec-env-1",
                task_id=exp_id,
                expired_claim_id="claim-env-exp",
                expired_assignment_id="assign-env-exp",
                expired_fencing_epoch=1,
                disposition="restore_claimable",
            )
            await gw.resolve_expired_claim(
                rec_cmd, operator_principal_id="operator-1", now=future_resolve
            )

            # Accept bounded multi-child delegation wave -> task.delegation_wave_proposal:*, task.delegation_wave:*
            wave_parent_id = "task-env-wave-parent"
            await gw.register_root_task(
                TaskRequested(
                    task_id=wave_parent_id,
                    root_task_id=wave_parent_id,
                    parent_task_id=None,
                    prompt="Envelope wave parent",
                    created_at=t0,
                    version=2,
                ),
                initial_status="pending",
                required_capability="research",
            )
            auth_set = cluster.admin_cli(
                "SET",
                f"task.authority:{wave_parent_id}",
                RootAuthorityRecord.create_wave_baseline(
                    wave_parent_id, max_children_per_wave=2, created_at=t0
                ).to_json(),
            )
            assert auth_set.returncode == 0, auth_set.combined
            wave_grant = DelegationGrantV2(
                grant_id="grant-env-wave-1",
                root_task_id=wave_parent_id,
                issued_to_assignment_id="assign-env-wave-1",
                issued_to_principal_id="principal-research-1",
                fencing_epoch=1,
                turn_id="turn-env-wave-1",
                issued_at=t0,
                max_children=2,
            )
            wave_claim = ActiveClaim(
                task_id=wave_parent_id,
                assignment_id="assign-env-wave-1",
                claim_id="claim-env-wave-1",
                root_task_id=wave_parent_id,
                turn_id="turn-env-wave-1",
                runtime_principal_id="principal-research-1",
                bound_agent_id="research-agent-1",
                assigned_capability="research",
                execution_kind="initial",
                fencing_epoch=1,
                proof_digest="d" * 64,
                request_id="req-env-wave-1",
                claimed_at=t0,
                delivery_expires_at=future_lease,
                delegation_grant_id="grant-env-wave-1",
                prior_claimable_status="pending",
            )
            await gw.atomic_claim_task(wave_parent_id, wave_claim, grant=wave_grant, now=t0)
            await gw.acknowledge_claim(
                wave_parent_id, "assign-env-wave-1", "principal-research-1", t0
            )
            wave_art = StageArtifact(
                artifact_id="art-env-wave-1",
                producing_task_id=wave_parent_id,
                artifact_type="research_material",
                content={"material": "envelope wave artifact"},
                created_at=t0,
                producing_turn_id="turn-env-wave-1",
                artifact_version=1,
                payload_schema_version=1,
                sealed=True,
                metadata={"turn_id": "turn-env-wave-1"},
            )
            await gw.save_stage_artifact(
                wave_art, "assign-env-wave-1", "principal-research-1", 1
            )
            wave_art_ref = ArtifactReference(
                artifact_id="art-env-wave-1",
                producing_task_id=wave_parent_id,
                artifact_version=1,
            )
            wave_req = DelegationWaveProposalRequest(
                proposal_id="prop-env-wave-1",
                items=(
                    DelegationWaveProposalItem(
                        item_id="item-1",
                        child_intent="Analyze wave branch 1",
                        artifact_reference=wave_art_ref,
                    ),
                    DelegationWaveProposalItem(
                        item_id="item-2",
                        child_intent="Analyze wave branch 2",
                        artifact_reference=wave_art_ref,
                    ),
                ),
                barrier_policy="WAIT_ALL",
            )
            wave_child_ids = ("task-env-wave-child-1", "task-env-wave-child-2")
            wave_deadline_derivation = derive_wave_deadline(
                work_complexity=wave_req.work_complexity,
                uncertainty_level=wave_req.uncertainty_level,
                required_child_count=len(wave_req.items),
                planned_parallelism=wave_req.planned_parallelism or len(wave_req.items),
                min_duration_seconds=MIN_WAVE_DURATION_SECONDS,
                max_duration_seconds=MAX_WAVE_DURATION_SECONDS,
                accepted_at=t0,
            )
            accepted_wave = AcceptedDelegationWave(
                wave_id="wave-env-1",
                parent_task_id=wave_parent_id,
                required_child_task_ids=wave_child_ids,
                barrier_policy=wave_req.barrier_policy,
                deadline_derivation=wave_deadline_derivation,
                accepted_at=t0,
            )
            wave_fp = compute_wave_proposal_fingerprint(
                parent_task_id=wave_parent_id,
                assignment_id="assign-env-wave-1",
                request=wave_req,
            )
            wave_proposal_record = DelegationWaveProposalRecord(
                parent_task_id=wave_parent_id,
                proposal_id=wave_req.proposal_id,
                wave_id=accepted_wave.wave_id,
                delegation_decision_id="dec-env-wave-1",
                assignment_id="assign-env-wave-1",
                runtime_principal_id="principal-research-1",
                bound_agent_id="research-agent-1",
                fencing_epoch=1,
                proof_digest="d" * 64,
                proposal_fingerprint=wave_fp,
                delegation_grant_id="grant-env-wave-1",
                required_child_task_ids=wave_child_ids,
                barrier_policy=accepted_wave.barrier_policy,
                required_success_fraction=accepted_wave.required_success_fraction,
                required_success_count=accepted_wave.required_success_count,
                accepted_at=t0,
                wave_deadline=accepted_wave.wave_deadline or wave_deadline_derivation.wave_deadline,
                child_claim_deadline=future_now,
            )
            wave_cont = TaskContinuationV2(
                parent_task_id=wave_parent_id,
                wave_id=accepted_wave.wave_id,
                required_child_task_ids=wave_child_ids,
                created_at=t0,
                updated_at=t0,
                input_version=1,
                continuation_capability="research",
                stage="incorporate_analysis_child",
                control_fields={},
                artifact_references=(wave_art_ref,),
                continuation_version=2,
                state="active",
                wave_resolution=None,
            )
            wave_child_inputs = [
                TaskInput(
                    task_id=cid,
                    prompt=item.child_intent,
                    created_at=t0,
                    input_version=1,
                    artifact_references=(item.artifact_reference,),
                )
                for cid, item in zip(wave_child_ids, wave_req.items, strict=True)
            ]
            wave_hints = [
                WakeUpHint(
                    hint_id=f"hint-{cid}",
                    capability="analysis",
                    created_at=t0,
                    execution_kind="delegated_child",
                    task_id=cid,
                )
                for cid in wave_child_ids
            ]
            wave_yield_rec = WaveYieldRecord(
                assignment_id="assign-env-wave-1",
                parent_task_id=wave_parent_id,
                principal_id="principal-research-1",
                proposal_id=wave_req.proposal_id,
                wave_id=accepted_wave.wave_id,
                required_child_task_ids=wave_child_ids,
                delegation_decision_id="dec-env-wave-1",
                yielded_at=t0,
            )
            await gw.accept_delegation_wave(
                parent_task_id=wave_parent_id,
                proposal=wave_proposal_record,
                accepted_wave=accepted_wave,
                child_task_inputs=wave_child_inputs,
                parent_continuation=wave_cont,
                yield_record=wave_yield_rec,
                wake_up_hints=wave_hints,
                expected_fencing_epoch=1,
                now=t0,
            )
            # Complete wave children -> task.delegation_wave_resolution:* + barrier resolution
            for idx, w_cid in enumerate(wave_child_ids, start=1):
                w_child_claim = ActiveClaim(
                    task_id=w_cid,
                    assignment_id=f"assign-env-wc-{idx}",
                    claim_id=f"claim-env-wc-{idx}",
                    root_task_id=wave_parent_id,
                    turn_id=f"turn-env-wc-{idx}",
                    runtime_principal_id="principal-analysis-1",
                    bound_agent_id="analysis-agent-1",
                    assigned_capability="analysis",
                    execution_kind="delegated_child",
                    fencing_epoch=1,
                    proof_digest="e" * 64,
                    request_id=f"req-env-wc-{idx}",
                    claimed_at=t0,
                    delivery_expires_at=future_lease,
                    prior_claimable_status="pending",
                )
                await gw.atomic_claim_task(w_cid, w_child_claim, grant=None, now=t0, is_root=False)
                await gw.acknowledge_claim(
                    w_cid, f"assign-env-wc-{idx}", "principal-analysis-1", t0
                )
                w_child_cmd = CompletionCommand(
                    completion_command_id=f"cmd-env-wc-{idx}",
                    assignment_id=f"assign-env-wc-{idx}",
                    execution_kind="delegated_child",
                    completed_result=TaskResult(task_id=w_cid, output=f"Envelope wave child {idx} ok"),
                )
                await gw.complete_turn(
                    w_cid,
                    w_child_cmd,
                    "e" * 64,
                    1,
                    wave_parent_id,
                    wave_parent_id,
                    t0,
                    "principal-analysis-1",
                    "analysis-agent-1",
                )
            # Second wave -> cancel via real orchestrator boundary -> task.delegation_wave_cancellation:*
            from agent_contracts import WaveCancellationCommand

            wcan_parent_id = "task-env-wave-cancel-1"
            await gw.register_root_task(
                TaskRequested(
                    task_id=wcan_parent_id,
                    root_task_id=wcan_parent_id,
                    parent_task_id=None,
                    prompt="Envelope wave cancel parent",
                    created_at=t0,
                    version=2,
                ),
                initial_status="pending",
                required_capability="research",
            )
            cluster.admin_cli(
                "SET",
                f"task.authority:{wcan_parent_id}",
                RootAuthorityRecord.create_wave_baseline(
                    wcan_parent_id, max_children_per_wave=2, created_at=t0
                ).to_json(),
            )
            wk_grant = DelegationGrantV2(
                grant_id="grant-env-wk-1",
                root_task_id=wcan_parent_id,
                issued_to_assignment_id="assign-env-wk-1",
                issued_to_principal_id="principal-research-1",
                fencing_epoch=1,
                turn_id="turn-env-wk-1",
                issued_at=t0,
                max_children=2,
            )
            wk_claim = ActiveClaim(
                task_id=wcan_parent_id,
                assignment_id="assign-env-wk-1",
                claim_id="claim-env-wk-1",
                root_task_id=wcan_parent_id,
                turn_id="turn-env-wk-1",
                runtime_principal_id="principal-research-1",
                bound_agent_id="research-agent-1",
                assigned_capability="research",
                execution_kind="initial",
                fencing_epoch=1,
                proof_digest="f" * 64,
                request_id="req-env-wk-1",
                claimed_at=t0,
                delivery_expires_at=future_lease,
                delegation_grant_id="grant-env-wk-1",
                prior_claimable_status="pending",
            )
            await gw.atomic_claim_task(
                wcan_parent_id, wk_claim, grant=wk_grant, now=t0, is_root=True
            )
            await gw.acknowledge_claim(
                wcan_parent_id, "assign-env-wk-1", "principal-research-1", t0
            )
            wk_art = StageArtifact(
                artifact_id="art-env-wk-1",
                producing_task_id=wcan_parent_id,
                artifact_type="research_material",
                content={"material": "envelope cancel wave artifact"},
                created_at=t0,
                producing_turn_id="turn-env-wk-1",
                artifact_version=1,
                payload_schema_version=1,
                sealed=True,
                metadata={"turn_id": "turn-env-wk-1"},
            )
            await gw.save_stage_artifact(
                wk_art, "assign-env-wk-1", "principal-research-1", 1
            )
            wk_ref = ArtifactReference(
                artifact_id="art-env-wk-1",
                producing_task_id=wcan_parent_id,
                artifact_version=1,
            )
            wk_req = DelegationWaveProposalRequest(
                proposal_id="prop-env-wk-1",
                items=(
                    DelegationWaveProposalItem(
                        item_id="item-1",
                        child_intent="Cancel wave branch 1",
                        artifact_reference=wk_ref,
                    ),
                    DelegationWaveProposalItem(
                        item_id="item-2",
                        child_intent="Cancel wave branch 2",
                        artifact_reference=wk_ref,
                    ),
                ),
                barrier_policy="WAIT_ALL",
            )
            wk_children = ("task-env-wk-c1", "task-env-wk-c2")
            wk_dd = derive_wave_deadline(
                work_complexity=wk_req.work_complexity,
                uncertainty_level=wk_req.uncertainty_level,
                required_child_count=2,
                planned_parallelism=2,
                min_duration_seconds=MIN_WAVE_DURATION_SECONDS,
                max_duration_seconds=MAX_WAVE_DURATION_SECONDS,
                accepted_at=t0,
            )
            wk_acc = AcceptedDelegationWave(
                wave_id="wave-env-wk-1",
                parent_task_id=wcan_parent_id,
                required_child_task_ids=wk_children,
                barrier_policy=wk_req.barrier_policy,
                deadline_derivation=wk_dd,
                accepted_at=t0,
            )
            wk_prop = DelegationWaveProposalRecord(
                parent_task_id=wcan_parent_id,
                proposal_id=wk_req.proposal_id,
                wave_id=wk_acc.wave_id,
                delegation_decision_id="dec-env-wk-1",
                assignment_id="assign-env-wk-1",
                runtime_principal_id="principal-research-1",
                bound_agent_id="research-agent-1",
                fencing_epoch=1,
                proof_digest="f" * 64,
                proposal_fingerprint=compute_wave_proposal_fingerprint(
                    parent_task_id=wcan_parent_id,
                    assignment_id="assign-env-wk-1",
                    request=wk_req,
                ),
                delegation_grant_id="grant-env-wk-1",
                required_child_task_ids=wk_children,
                barrier_policy=wk_acc.barrier_policy,
                accepted_at=t0,
                wave_deadline=wk_dd.wave_deadline,
                child_claim_deadline=future_now,
            )
            wk_cont = TaskContinuationV2(
                parent_task_id=wcan_parent_id,
                wave_id=wk_acc.wave_id,
                required_child_task_ids=wk_children,
                created_at=t0,
                updated_at=t0,
                input_version=1,
                continuation_capability="research",
                stage="incorporate_analysis_child",
                control_fields={},
                artifact_references=(wk_ref,),
                continuation_version=2,
                state="active",
                wave_resolution=None,
            )
            wk_inputs = [
                TaskInput(
                    task_id=cid,
                    prompt=item.child_intent,
                    created_at=t0,
                    input_version=1,
                    artifact_references=(item.artifact_reference,),
                )
                for cid, item in zip(wk_children, wk_req.items, strict=True)
            ]
            wk_hints = [
                WakeUpHint(
                    hint_id=f"hint-{cid}",
                    capability="analysis",
                    created_at=t0,
                    execution_kind="delegated_child",
                    task_id=cid,
                )
                for cid in wk_children
            ]
            wk_yield = WaveYieldRecord(
                assignment_id="assign-env-wk-1",
                parent_task_id=wcan_parent_id,
                principal_id="principal-research-1",
                proposal_id=wk_req.proposal_id,
                wave_id=wk_acc.wave_id,
                required_child_task_ids=wk_children,
                delegation_decision_id="dec-env-wk-1",
                yielded_at=t0,
            )
            await gw.accept_delegation_wave(
                parent_task_id=wcan_parent_id,
                proposal=wk_prop,
                accepted_wave=wk_acc,
                child_task_inputs=wk_inputs,
                parent_continuation=wk_cont,
                yield_record=wk_yield,
                wake_up_hints=wk_hints,
                expected_fencing_epoch=1,
                now=t0,
            )
            await gw.cancel_delegation_wave(
                WaveCancellationCommand(
                    cancellation_decision_id="can-env-1",
                    parent_task_id=wcan_parent_id,
                    wave_id=wk_acc.wave_id,
                ),
                operator_principal_id="operator-cancel-env",
                now=t0,
            )
        finally:
            await gw_legacy.aclose()
            await gw.aclose()

    asyncio.run(main())


def _subtract_from_default_line(
    full_acl: str,
    *,
    drop_select: bool = False,
    drop_command: str | None = None,
    drop_pair: tuple[str, str] | None = None,
) -> str:
    """Return a copy of full_acl with one selector/command/key removed from the default rule."""
    lines = full_acl.splitlines()
    out_lines: list[str] = []
    mutated = False
    for line in lines:
        if not line.startswith("user default "):
            out_lines.append(line)
            continue
        updated = line
        if drop_select:
            assert " +select" in updated
            updated = updated.replace(" +select", "", 1)
        elif drop_command is not None:
            cmd_token = f"(+{drop_command.lower()} "
            pattern = re.compile(re.escape(cmd_token) + r"[^)]+\)")
            assert pattern.search(updated), f"missing command group {drop_command}"
            updated = pattern.sub("", updated, count=1)
            updated = re.sub(r"  +", " ", updated).strip()
        elif drop_pair is not None:
            cmd, family = drop_pair
            pattern = re.compile(
                re.escape(f"(+{cmd.lower()} ") + r"([^)]+)\)"
            )
            match = pattern.search(updated)
            assert match is not None, f"missing command group {cmd}"
            tokens = match.group(1).split()
            target_tok = f"%W~{family}"
            assert target_tok in tokens, f"missing {target_tok} in {match.group(0)}"
            remaining = [tok for tok in tokens if tok != target_tok]
            if remaining:
                replacement = f"(+{cmd.lower()} " + " ".join(remaining) + ")"
            else:
                replacement = ""
            updated = updated[: match.start()] + replacement + updated[match.end() :]
            updated = re.sub(r"  +", " ", updated).strip()
        assert updated != line
        out_lines.append(updated)
        mutated = True
    assert mutated, "user default line not found"
    return "\n".join(out_lines) + "\n"


def test_production_envelope_subtraction_matrix() -> None:
    """Prove +select, every retained command, and every retained (command, family) pair fail when subtracted."""
    projection = json.loads(
        (REPO_ROOT / "infrastructure" / "redis" / "replay_projection.json").read_text(
            encoding="utf-8"
        )
    )
    selectors: dict[str, list[str]] = projection["rule_model"]["selectors"]
    assert projection["rule_model"]["has_select"] is True

    cluster = LiveRedisAclCluster.start(
        build_service_images=False,
        publish_host_port=True,
        appendonly=True,
        persistent_data=True,
    )
    register_cluster(cluster)
    try:
        _write_full_envelope_state(cluster)
        sync = cluster.admin_cli("WAITAOF", "1", "0", "5000")
        assert sync.returncode == 0, sync.combined

        rendered = _run_docker(
            ["docker", "exec", cluster.redis_container, "cat", "/run/redis/users.acl"],
            timeout=30,
        )
        assert rendered.returncode == 0, rendered.combined
        full_acl = rendered.stdout

        generation_text = (cluster.secret_dir / "current" / "generation.json").read_text(
            encoding="utf-8"
        )
        password_text = cluster.secrets_map["healthcheck"]

        _run_docker(["docker", "stop", cluster.redis_container], timeout=60)

        snapshot_dir = cluster.temp_dir / "envelope-snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        _run_docker(
            [
                "docker",
                "run",
                "--rm",
                "--mount",
                f"type=volume,src={cluster.data_volume_name},dst=/data,readonly",
                "--mount",
                f"type=bind,src={snapshot_dir},dst=/out",
                "--entrypoint",
                "/bin/sh",
                REDIS_IMAGE,
                "-c",
                "cp -a /data/. /out/",
            ],
            timeout=60,
        )

        # Build the complete subtraction matrix directly from replay_projection.json
        cases: list[tuple[str, str]] = [
            ("drop_select", _subtract_from_default_line(full_acl, drop_select=True))
        ]
        for cmd in sorted(selectors):
            cases.append(
                (
                    f"drop_cmd_{cmd}",
                    _subtract_from_default_line(full_acl, drop_command=cmd),
                )
            )
            for family in sorted(selectors[cmd]):
                cases.append(
                    (
                        f"drop_pair_{cmd}_{family}",
                        _subtract_from_default_line(full_acl, drop_pair=(cmd, family)),
                    )
                )

        # Verify 100% coverage of projection selectors + commands + select
        assert len(cases) == 1 + len(selectors) + sum(len(v) for v in selectors.values())

        for idx, (case_name, narrowed_acl) in enumerate(cases):
            evidence = cluster.temp_dir / f"sub-{idx}"
            _prepare_replay_evidence_dir(
                evidence, narrowed_acl, generation_text, password_text
            )

            cname = f"{cluster.redis_container}-sub-{idx}"
            bootstrap = (
                "set -eu\n"
                "cp -a /snapshot/. /data/\n"
                "chown -R redis:redis /data\n"
                + redis_replay_log_prep(log_path="/run/redis-evd/replay.log", log_mode="0664")
                + "exec setpriv --reuid=redis --regid=redis --clear-groups -- "
                "redis-server --appendonly yes "
                "--aclfile /run/redis-evd/users.acl "
                "--logfile /run/redis-evd/replay.log "
                "--loglevel notice\n"
            )
            _run_docker(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--name",
                    cname,
                    "--tmpfs",
                    "/data:rw,size=32m",
                    "--mount",
                    f"type=bind,src={snapshot_dir},dst=/snapshot,readonly",
                    "--mount",
                    f"type=bind,src={evidence},dst=/run/redis-evd",
                    "--mount",
                    (
                        f"type=bind,src={REPO_ROOT / 'infrastructure' / 'redis' / 'redis-healthcheck.sh'},"
                        "dst=/usr/local/bin/redis-healthcheck.sh,readonly"
                    ),
                    "--entrypoint",
                    "/bin/sh",
                    REDIS_IMAGE,
                    "-c",
                    bootstrap,
                ],
                timeout=60,
            )
            cluster.created_containers.append(cname)
            try:
                log_text = _wait_for_log_text(
                    evidence / "replay.log", "Ready to accept connections", timeout=20.0
                )
                ready_at = log_text.find("Ready to accept connections")
                denial_at = log_text.find("NOPERM")
                if denial_at == -1:
                    denial_at = log_text.find("CRITICAL")
                assert denial_at != -1 and denial_at < ready_at, (
                    f"{case_name}: expected pre-readiness NOPERM/CRITICAL denial:\n{log_text}"
                )
                hc_res = _run_evidence_health(cname)
                assert hc_res.returncode != 0, (
                    f"{case_name}: healthcheck unexpectedly passed on subtracted rule"
                )
            finally:
                _run_docker(["docker", "container", "rm", "--force", cname], check=False, timeout=30)
                shutil.rmtree(evidence, ignore_errors=True)
    finally:
        unregister_cluster(cluster)
        cluster.cleanup()


def test_bgrewriteaof_and_incremental_tail_replay_cleanly() -> None:
    """Verify BGREWRITEAOF RDB base + subsequent incremental AOF tail replay cleanly."""
    from agent_contracts import TaskRequested

    cluster = LiveRedisAclCluster.start(
        build_service_images=False,
        publish_host_port=True,
        appendonly=True,
        persistent_data=True,
    )
    register_cluster(cluster)
    try:
        async def write_task(task_id: str) -> None:
            from orchestrator.infrastructure.redis_adapter import RedisGateway

            gw = RedisGateway(
                cluster.redis_url,
                username="orchestrator",
                password=cluster.secrets_map["orchestrator"],
            )
            try:
                task = TaskRequested(
                    task_id=task_id,
                    root_task_id=task_id,
                    parent_task_id=None,
                    prompt=f"BGREWRITEAOF {task_id}",
                    created_at="2026-11-24T00:00:00Z",
                    version=2,
                )
                await gw.register_root_task(task, required_capability="general")
                await gw.dispatch(task, "agent.tasks.general", capability="general")
            finally:
                await gw.aclose()

        asyncio.run(write_task("task-bgrewrite-base"))
        rewrite = cluster.admin_cli("BGREWRITEAOF")
        assert rewrite.returncode == 0, rewrite.combined
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            info = cluster.admin_cli("INFO", "persistence")
            if "aof_rewrite_in_progress:0" in info.stdout:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("BGREWRITEAOF did not complete in time")

        asyncio.run(write_task("task-bgrewrite-tail"))
        sync = cluster.admin_cli("WAITAOF", "1", "0", "5000")
        assert sync.returncode == 0, sync.combined

        rendered = _run_docker(
            ["docker", "exec", cluster.redis_container, "cat", "/run/redis/users.acl"],
            timeout=30,
        ).stdout
        generation_text = (cluster.secret_dir / "current" / "generation.json").read_text(
            encoding="utf-8"
        )
        password_text = cluster.secrets_map["healthcheck"]
        _run_docker(["docker", "stop", cluster.redis_container], timeout=60)

        evd = cluster.temp_dir / "bgrewrite-replay"
        cname = f"{cluster.redis_container}-bgrewrite"
        _start_replay_container(cluster, cname, evd, rendered, generation_text, password_text)
        log_text = _wait_for_log_text(evd / "replay.log", "Ready to accept connections")
        assert "CRITICAL" not in log_text and "NOPERM" not in log_text, log_text
        assert _run_evidence_health(cname).returncode == 0
        # Verify both base-RDB task and incremental-tail task survived
        ls_data = _run_docker(
            ["docker", "exec", cname, "find", "/data/appendonlydir", "-type", "f", "-print"],
            timeout=30,
        ).stdout
        assert ".base.rdb" in ls_data and ".incr.aof" in ls_data
        # Check via replay container's admin user
        for tid in ("task-bgrewrite-base", "task-bgrewrite-tail"):
            chk = _run_docker(
                [
                    "docker",
                    "exec",
                    "-i",
                    cname,
                    "/bin/sh",
                    "-c",
                    'IFS= read -r pw; REDISCLI_AUTH="$pw" exec redis-cli --raw --user fixture-admin GET "$1"',
                    "_",
                    f"task.status:{tid}",
                ],
                stdin_text=cluster.admin_secret + "\n",
            )
            assert json.loads(chk.stdout)["status"] == "pending", chk.stdout
    finally:
        unregister_cluster(cluster)
        cluster.cleanup()


def test_truncated_aof_tail_behavior_is_bounded_and_documented() -> None:
    """Verify truncated incremental AOF tail behavior under aof-load-truncated yes."""
    cluster = LiveRedisAclCluster.start(
        build_service_images=False,
        publish_host_port=True,
        appendonly=True,
        persistent_data=True,
    )
    register_cluster(cluster)
    try:
        _write_qualifying_state(cluster)
        sync = cluster.admin_cli("WAITAOF", "1", "0", "5000")
        assert sync.returncode == 0, sync.combined

        rendered = _run_docker(
            ["docker", "exec", cluster.redis_container, "cat", "/run/redis/users.acl"],
            timeout=30,
        ).stdout
        generation_text = (cluster.secret_dir / "current" / "generation.json").read_text(
            encoding="utf-8"
        )
        password_text = cluster.secrets_map["healthcheck"]
        _run_docker(["docker", "stop", cluster.redis_container], timeout=60)

        # Truncate the last 25 bytes of the incremental AOF file mid-command on the volume
        _run_docker(
            [
                "docker",
                "run",
                "--rm",
                "--mount",
                f"type=volume,src={cluster.data_volume_name},dst=/data",
                "--entrypoint",
                "/bin/sh",
                REDIS_IMAGE,
                "-c",
                (
                    'f=$(find /data/appendonlydir -name "*.incr.aof" | head -n 1); '
                    'sz=$(wc -c < "$f"); '
                    'truncate -s $((sz - 25)) "$f"'
                ),
            ],
            timeout=30,
        )

        evd = cluster.temp_dir / "truncated-replay"
        cname = f"{cluster.redis_container}-truncated"
        _start_replay_container(cluster, cname, evd, rendered, generation_text, password_text)
        # Under default aof-load-truncated yes, Redis logs the incomplete tail, truncates it,
        # preserves the complete prior transaction, and reports ready without NOPERM/CRITICAL.
        log_text = _wait_for_log_text(evd / "replay.log", "Ready to accept connections")
        assert "short read while loading the AOF file" in log_text, log_text
        assert "aof-load-truncated is enabled" in log_text, log_text
        assert "NOPERM" not in log_text and "CRITICAL" not in log_text, log_text
        assert _run_evidence_health(cname).returncode == 0, log_text
    finally:
        unregister_cluster(cluster)
        cluster.cleanup()
