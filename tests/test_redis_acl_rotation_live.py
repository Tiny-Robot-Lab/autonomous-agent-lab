"""Bounded Docker Compose Redis credential-rotation and AOF evidence tests."""

from __future__ import annotations

import json

import pytest

from scripts.generate_redis_secrets import get_generation_info
from scripts.redis_acl_live_harness import LiveRedisAclCluster, _run_docker

pytestmark = [pytest.mark.live_redis_rotation]


def _assert_task_status(
    cluster: LiveRedisAclCluster,
    task_id: str,
    password: str,
) -> None:
    result = cluster.redis_cli(
        "orchestrator",
        "GET",
        f"task.status:{task_id}",
        password=password,
    )
    assert result.returncode == 0, result.combined
    assert result.stdout.strip(), result.combined
    assert json.loads(result.stdout) == {
        "status": "pending_dispatch",
        "task_id": task_id,
    }


def test_compose_credential_rotation_recreates_acl_and_preserves_aof() -> None:
    cluster = LiveRedisAclCluster.start(
        appendonly=True,
        persistent_data=True,
        use_compose=True,
    )
    try:
        identity = "orchestrator"
        old_generation = get_generation_info(cluster.secret_dir)
        old_password = cluster.secrets_map[identity]
        old_secret_path = old_generation.path / identity
        old_inode = old_secret_path.stat().st_ino
        task_id = f"task-rotation-{cluster.token}"
        register_task = f"""
import asyncio
from agent_contracts import TaskRequested
from agent_contracts import resolve_redis_connection_from_env
from orchestrator.infrastructure.redis_adapter import RedisGateway

async def main():
    config = resolve_redis_connection_from_env()
    gateway = RedisGateway(config.url, connection_config=config)
    try:
        task = TaskRequested(
            task_id={task_id!r},
            root_task_id={task_id!r},
            parent_task_id=None,
            prompt="AOF rotation verification",
            created_at="2026-11-24T00:00:00Z",
            version=2,
        )
        await gateway.register_root_task(task, required_capability="general")
        assert await gateway.get_status({task_id!r}) == "pending_dispatch"
    finally:
        await gateway.aclose()

asyncio.run(main())
"""
        registered = cluster.run_service_python("orchestrator", identity, register_task)
        assert registered.returncode == 0, registered.combined

        aof_sync = cluster.admin_cli("WAITAOF", "1", "0", "5000")
        assert aof_sync.returncode == 0, aof_sync.combined
        assert aof_sync.stdout.split()[0] == "1"
        aof_files_before_recreate = _run_docker(
            ["docker", "exec", cluster.redis_container, "find", "/data", "-type", "f", "-print"]
        )
        assert any("appendonly" in path for path in aof_files_before_recreate.stdout.splitlines())

        old_values = cluster.rotate_secrets(identity)
        new_password = cluster.secrets_map[identity]
        new_generation = get_generation_info(cluster.secret_dir)
        new_secret_path = new_generation.path / identity
        assert new_generation.generation_id != old_generation.generation_id
        assert new_generation.identity_generations[identity] != (
            old_generation.identity_generations[identity]
        )
        assert old_secret_path.stat().st_ino == old_inode
        assert new_secret_path.stat().st_ino != old_inode
        assert old_values[identity] == old_password
        assert new_password != old_password

        status_key = f"task.status:{task_id}"
        _assert_task_status(cluster, task_id, old_password)
        new_before_recreate = cluster.redis_cli(
            identity, "GET", status_key, password=new_password
        )
        assert "WRONGPASS" in new_before_recreate.combined

        mounted_old_before_recreate = cluster.mounted_secret_matches(identity, old_password)
        mounted_new_before_recreate = cluster.mounted_secret_matches(identity, new_password)
        assert mounted_old_before_recreate and not mounted_new_before_recreate, (
            "credential mount changed before Redis recreation"
        )

        cluster.recreate_redis()
        assert cluster.mounted_secret_matches(identity, new_password)
        assert cluster.run_healthcheck().returncode == 0
        assert cluster.redis_cli(None, "PING").combined == "NOAUTH Authentication required."
        assert "WRONGPASS" in cluster.redis_cli(
            identity, "GET", status_key, password=old_password
        ).combined
        persisted_status = cluster.admin_cli("GET", status_key)
        if not persisted_status.stdout.strip():
            redis_logs = _run_docker(
                ["docker", "logs", cluster.redis_container],
                check=False,
            )
            acl_replay_denials = sum(
                "AOF-loading-client" in line and "NOPERM" in line
                for line in redis_logs.combined.splitlines()
            )
            pytest.fail(
                "Task status missing after force-recreation; Redis reported "
                f"{acl_replay_denials} ACL denials while replaying AOF transactions"
            )
        _assert_task_status(cluster, task_id, new_password)

        unaffected = cluster.redis_cli(
            "worker-1", "SADD", "agent.ids", "worker-1", use_mounted_secret=True
        )
        assert unaffected.returncode == 0
        assert unaffected.stdout.strip() == "1"

        persisted = cluster.run_service_python(
            "orchestrator",
            identity,
            f"""
import asyncio
from agent_contracts import resolve_redis_connection_from_env
from orchestrator.infrastructure.redis_adapter import RedisGateway

async def main():
    config = resolve_redis_connection_from_env()
    gateway = RedisGateway(config.url, connection_config=config)
    try:
        assert await gateway.get_status({task_id!r}) == "pending_dispatch"
        print("PERSISTED")
    finally:
        await gateway.aclose()

asyncio.run(main())
""",
        )
        assert persisted.returncode == 0, persisted.combined
        assert persisted.stdout.strip().endswith("PERSISTED")

        data_files = _run_docker(
            [
                "docker",
                "exec",
                cluster.redis_container,
                "/bin/sh",
                "-c",
                "find /data -type f -print",
            ],
            check=False,
        )
        assert data_files.returncode == 0
        assert any("appendonly" in path for path in data_files.stdout.splitlines())
        assert all(
            "users.acl" not in path and "credential-generation" not in path
            for path in data_files.stdout.splitlines()
        )
    finally:
        cluster.cleanup()
