from __future__ import annotations

import asyncio
import importlib
import logging
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from agent_contracts import (
    ActiveClaim,
    DelegationGrant,
    RedisConnectionConfig,
    StageArtifact,
    read_redis_password_file,
    resolve_client_connection_kwargs,
    resolve_redis_connection_from_env,
    sanitize_redis_url,
    validate_redis_url,
)
from analysis_agent.infrastructure.redis_bus import RedisBus as AnalysisRedisBus
from fastapi import FastAPI
from orchestrator.config import Settings
from orchestrator.infrastructure.redis_adapter import (
    CLAIM_TASK_LUA,
    SAVE_STAGE_ARTIFACT_LUA,
    RedisGateway,
)
from orchestrator.presentation.api import create_app
from research_agent.infrastructure.redis_bus import RedisBus as ResearchRedisBus
from worker.infrastructure.redis_bus import RedisBus as WorkerRedisBus

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_secret_file(path: Path, content: str | bytes) -> Path:
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def test_validate_redis_url_accepts_database_zero_forms() -> None:
    for valid_url in (
        "redis://localhost:6379",
        "redis://localhost:6379/",
        "redis://localhost:6379/0",
        "redis://localhost:6379?db=0",
        "redis://localhost:6379/?db=0",
        "redis://localhost:6379/0?db=0",
        "rediss://redis:6380/0",
        "redis://[::1]:6379/0",
    ):
        assert validate_redis_url(valid_url) == valid_url


def test_validate_redis_url_rejects_embedded_credentials_without_leaking_secret() -> None:
    secret = "SuperSecretRedisPassword123456"
    for cred_url in (
        f"redis://worker-1:{secret}@localhost:6379/0",
        f"redis://:{secret}@redis:6379/0",
        "redis://worker-1@redis:6379/0",
        f"redis://redis:6379/0?password={secret}",
    ):
        with pytest.raises(ValueError) as exc_info:
            validate_redis_url(cred_url)
        msg = str(exc_info.value)
        assert secret not in msg
        assert "worker-1:" not in msg
        sanitized = sanitize_redis_url(cred_url)
        assert secret not in sanitized
        assert "worker-1:" not in sanitized


def test_validate_redis_url_rejects_nonzero_and_conflicting_databases() -> None:
    for bad_url in (
        "redis://localhost:6379/1",
        "redis://localhost:6379/15",
        "redis://localhost:6379?db=1",
        "redis://localhost:6379/?db=15",
        "redis://localhost:6379/0?db=1",
        "redis://localhost:6379/1?db=0",
        "redis://localhost:6379?db=0&db=1",
        "redis://localhost:6379?db=",
        "redis://localhost:6379?db=abc",
        "redis://localhost:6379/abc",
        "redis://localhost:6379/0/extra",
        "redis://localhost:6379/0#frag",
        "redis://localhost:6379/0?ssl=true",
        "http://localhost:6379/0",
        "redis://",
        "",
        "   ",
    ):
        with pytest.raises(ValueError):
            validate_redis_url(bad_url)


def test_paired_username_and_password_file_requirements(tmp_path: Path) -> None:
    secret_path = _write_secret_file(tmp_path / "pw", "abcdefghijklmnopqrstuvwxyz123456")

    # Missing both for Compose non-loopback host fails closed (no unauthenticated fallback)
    with pytest.raises(ValueError, match="REDIS_USERNAME and REDIS_PASSWORD_FILE are required"):
        resolve_redis_connection_from_env({"REDIS_URL": "redis://redis:6379/0"})

    # Explicit required mode on localhost without creds fails closed
    with pytest.raises(ValueError, match="REDIS_USERNAME and REDIS_PASSWORD_FILE are required"):
        resolve_redis_connection_from_env(
            {"REDIS_URL": "redis://localhost:6379/0", "REDIS_AUTH_MODE": "required"}
        )

    # Production environment on localhost without creds fails closed
    with pytest.raises(ValueError, match="REDIS_USERNAME and REDIS_PASSWORD_FILE are required"):
        resolve_redis_connection_from_env(
            {"REDIS_URL": "redis://localhost:6379/0", "APP_ENV": "production"}
        )

    # Partial: username without password file
    with pytest.raises(ValueError, match="REDIS_PASSWORD_FILE is required"):
        resolve_redis_connection_from_env(
            {"REDIS_URL": "redis://localhost:6379/0", "REDIS_USERNAME": "worker-1"}
        )

    # Partial: password file without username
    with pytest.raises(ValueError, match="REDIS_USERNAME is required"):
        resolve_redis_connection_from_env(
            {"REDIS_URL": "redis://localhost:6379/0", "REDIS_PASSWORD_FILE": str(secret_path)}
        )

    # Empty or whitespace username fails closed
    for bad_user in ("", "   ", "worker 1", "worker\n1", "worker\x001"):
        with pytest.raises(ValueError, match="REDIS_USERNAME"):
            resolve_redis_connection_from_env(
                {
                    "REDIS_URL": "redis://redis:6379/0",
                    "REDIS_USERNAME": bad_user,
                    "REDIS_PASSWORD_FILE": str(secret_path),
                }
            )

    # Non-zero DB is rejected in authenticated Compose mode
    with pytest.raises(ValueError, match="database must be 0"):
        resolve_redis_connection_from_env(
            {
                "REDIS_URL": "redis://redis:6379/1",
                "REDIS_USERNAME": "worker-1",
                "REDIS_PASSWORD_FILE": str(secret_path),
            }
        )

    # Compose host cannot explicitly opt into unauthenticated_local_test
    with pytest.raises(ValueError, match="unauthenticated_local_test"):
        resolve_redis_connection_from_env(
            {
                "REDIS_URL": "redis://redis:6379/0",
                "REDIS_AUTH_MODE": "unauthenticated_local_test",
            }
        )


def test_password_file_validation_rejects_bad_files_and_redacts_secrets(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Missing file
    with pytest.raises(ValueError, match="missing or unreadable"):
        read_redis_password_file(str(tmp_path / "does_not_exist"))

    # Directory path instead of file
    with pytest.raises(ValueError, match="missing or unreadable"):
        read_redis_password_file(str(tmp_path))

    # Unreadable file (skip permission test only if running as root where mode 000 is readable)
    if os.geteuid() != 0:
        unreadable = _write_secret_file(tmp_path / "unreadable", "validsecretmaterial1234567890abcd")
        unreadable.chmod(0o000)
        try:
            with pytest.raises(ValueError, match="missing or unreadable"):
                read_redis_password_file(str(unreadable))
        finally:
            unreadable.chmod(0o600)

    # Non-UTF-8 bytes
    bad_utf8 = _write_secret_file(tmp_path / "bad_utf8", b"\xff\xfe\xfd\x00")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_redis_password_file(str(bad_utf8))

    # Empty, whitespace, newline, and control characters
    secret_marker = "SecretBodyDoNotLeak123456"
    bad_contents = [
        "",
        "   ",
        "\t",
        f"{secret_marker}\n",
        f"{secret_marker}\r\n",
        f" {secret_marker}",
        f"{secret_marker} ",
        f"abc {secret_marker}",
        f"{secret_marker}\x00",
        f"{secret_marker}\x1b",
        f"{secret_marker}\x7f",
    ]
    with caplog.at_level(logging.DEBUG):
        for idx, content in enumerate(bad_contents):
            file_path = _write_secret_file(tmp_path / f"bad_{idx}", content)
            with pytest.raises(ValueError) as exc_info:
                read_redis_password_file(str(file_path))
            formatted_tb = "".join(
                traceback.format_exception(type(exc_info.value), exc_info.value, exc_info.tb)
            )
            assert secret_marker not in str(exc_info.value)
            assert secret_marker not in formatted_tb
    assert secret_marker not in caplog.text


def test_connection_config_redacts_repr_str_and_diagnostics(tmp_path: Path) -> None:
    secret_value = "VerySecretRedisAuthToken9876543210"
    secret_file = _write_secret_file(tmp_path / "orch.pw", secret_value)

    cfg = resolve_redis_connection_from_env(
        {
            "REDIS_URL": "redis://redis:6379/0",
            "REDIS_USERNAME": "orchestrator",
            "REDIS_PASSWORD_FILE": str(secret_file),
        }
    )

    assert cfg.is_authenticated is True
    assert cfg.username == "orchestrator"
    assert cfg.password == secret_value
    assert secret_value not in repr(cfg)
    assert secret_value not in str(cfg)
    diag = cfg.to_safe_diagnostics()
    assert diag["url"] == "redis://redis:6379/0"
    assert diag["username"] == "orchestrator"
    assert diag["authenticated"] is True
    assert diag["password_configured"] is True
    assert secret_value not in repr(diag)

    # Direct Settings wrapping cfg also never prints the Redis password
    settings = Settings(redis_url=cfg.url, redis_connection=cfg)
    assert secret_value not in repr(settings)
    assert secret_value not in str(settings)


def test_all_four_clients_construct_with_protocol_2_and_no_driver_info(tmp_path: Path) -> None:
    secret_value = "ClientKwargsSecretToken1234567890"
    secret_file = _write_secret_file(tmp_path / "worker.pw", secret_value)
    cfg = resolve_redis_connection_from_env(
        {
            "REDIS_URL": "redis://redis:6379/0",
            "REDIS_USERNAME": "worker-1",
            "REDIS_PASSWORD_FILE": str(secret_file),
        }
    )

    for bus_cls in (WorkerRedisBus, ResearchRedisBus, AnalysisRedisBus):
        with patch("redis.asyncio.from_url") as mock_from_url:
            bus_cls("redis://redis:6379/0", connection_config=cfg)
            mock_from_url.assert_called_once_with(
                "redis://redis:6379/0",
                decode_responses=True,
                protocol=2,
                driver_info=None,
                username="worker-1",
                password=secret_value,
            )
            assert "health_check_interval" not in mock_from_url.call_args.kwargs

    with patch("redis.asyncio.from_url") as mock_from_url:
        RedisGateway("redis://redis:6379/0", connection_config=cfg)
        mock_from_url.assert_called_once_with(
            "redis://redis:6379/0",
            decode_responses=True,
            protocol=2,
            driver_info=None,
            username="worker-1",
            password=secret_value,
        )
        assert "health_check_interval" not in mock_from_url.call_args.kwargs

    # Compatibility path (unauthenticated local URL) also applies protocol=2, driver_info=None,
    # and still rejects embedded URL credentials.
    with patch("redis.asyncio.from_url") as mock_from_url:
        RedisGateway("redis://localhost:6379/15", allow_unauthenticated_local_test=True)
        mock_from_url.assert_called_once_with(
            "redis://localhost:6379/15",
            decode_responses=True,
            protocol=2,
            driver_info=None,
        )

    for cls_ctor in (RedisGateway, WorkerRedisBus, ResearchRedisBus, AnalysisRedisBus):
        with pytest.raises(ValueError, match="embedded credentials"):
            cls_ctor(
                f"redis://user:{secret_value}@localhost:6379/0",
                allow_unauthenticated_local_test=True,
            )


def test_settings_from_env_wires_authenticated_redis_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret_value = "OrchestratorEnvSecretToken1234567890"
    secret_file = _write_secret_file(tmp_path / "orch.pw", secret_value)

    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("REDIS_USERNAME", "orchestrator")
    monkeypatch.setenv("REDIS_PASSWORD_FILE", str(secret_file))

    settings = Settings.from_env()
    assert settings.redis_connection is not None
    assert settings.redis_connection.username == "orchestrator"
    assert settings.redis_connection.password == secret_value

    with patch("redis.asyncio.from_url") as mock_from_url:
        app = create_app(settings=settings)
        assert app.state.gateway is not None
        mock_from_url.assert_called_once_with(
            "redis://redis:6379/0",
            decode_responses=True,
            protocol=2,
            driver_info=None,
            username="orchestrator",
            password=secret_value,
        )

    # Compose host without credentials fails closed on Settings.from_env()
    monkeypatch.delenv("REDIS_USERNAME", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD_FILE", raising=False)
    with pytest.raises(ValueError, match="REDIS_USERNAME and REDIS_PASSWORD_FILE are required"):
        Settings.from_env()


@pytest.mark.anyio
async def test_service_mains_wire_authenticated_config_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_value = "AgentRuntimeSecretToken1234567890"
    secret_file = _write_secret_file(tmp_path / "agent.pw", secret_value)

    monkeypatch.setenv("RUNTIME_TOKEN", "runtime-http-token")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")

    from analysis_agent.main import run_analysis_agent
    from research_agent.main import run_research_agent
    from worker.main import run_worker

    # Non-local Compose endpoint without credentials fails closed before creating bus
    for runner in (run_worker, run_research_agent, run_analysis_agent):
        with pytest.raises(ValueError, match="REDIS_USERNAME and REDIS_PASSWORD_FILE are required"):
            await runner()

    # Valid paired credentials wire connection_config with protocol=2, driver_info=None
    monkeypatch.setenv("REDIS_USERNAME", "worker-1")
    monkeypatch.setenv("REDIS_PASSWORD_FILE", str(secret_file))
    with patch("worker.main.RedisBus") as mock_bus_cls, patch("worker.main.TaskProcessor"):
        mock_bus = AsyncMock()
        mock_bus.next_signal.side_effect = asyncio.CancelledError()
        mock_bus_cls.return_value = mock_bus
        with pytest.raises(asyncio.CancelledError):
            await run_worker()
        cfg = mock_bus_cls.call_args.kwargs["connection_config"]
        assert isinstance(cfg, RedisConnectionConfig)
        assert cfg.username == "worker-1"
        assert cfg.password == secret_value

    monkeypatch.setenv("REDIS_USERNAME", "research-agent-1")
    with (
        patch("research_agent.main.RedisBus") as mock_bus_cls,
        patch("research_agent.main.ResearchProcessor"),
    ):
        mock_bus = AsyncMock()
        mock_bus.next_signal.side_effect = asyncio.CancelledError()
        mock_bus_cls.return_value = mock_bus
        with pytest.raises(asyncio.CancelledError):
            await run_research_agent()
        cfg = mock_bus_cls.call_args.kwargs["connection_config"]
        assert isinstance(cfg, RedisConnectionConfig)
        assert cfg.username == "research-agent-1"
        assert cfg.password == secret_value

    monkeypatch.setenv("REDIS_USERNAME", "analysis-agent-1")
    with (
        patch("analysis_agent.main.RedisBus") as mock_bus_cls,
        patch("analysis_agent.main.AnalysisProcessor"),
    ):
        mock_bus = AsyncMock()
        mock_bus.next_signal.side_effect = asyncio.CancelledError()
        mock_bus_cls.return_value = mock_bus
        with pytest.raises(asyncio.CancelledError):
            await run_analysis_agent()
        cfg = mock_bus_cls.call_args.kwargs["connection_config"]
        assert isinstance(cfg, RedisConnectionConfig)
        assert cfg.username == "analysis-agent-1"
        assert cfg.password == secret_value


@pytest.mark.anyio
async def test_atomic_claim_task_never_passes_empty_lua_keys() -> None:
    gateway = RedisGateway("redis://localhost:6379/0", allow_unauthenticated_local_test=True)
    now = datetime.now(UTC).isoformat()

    claim_without_grant = ActiveClaim(
        task_id="task-1",
        assignment_id="assign-1",
        claim_id="claim-1",
        root_task_id="task-1",
        turn_id="turn-1",
        runtime_principal_id="principal-worker-1",
        bound_agent_id="worker-1",
        assigned_capability="general",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="a" * 64,
        request_id="req-1",
        claimed_at=now,
        delivery_expires_at="2099-01-01T00:00:00Z",
    )

    gateway._client.eval = AsyncMock(return_value=["OK", "1"])
    claimed, is_replay = await gateway.atomic_claim_task("task-1", claim_without_grant, now=now)
    assert is_replay is False
    assert claimed.task_id == "task-1"

    call_args = gateway._client.eval.call_args.args
    assert call_args[0] == CLAIM_TASK_LUA
    assert call_args[1] == 7
    keys_no_grant = list(call_args[2 : 2 + 7])
    assert keys_no_grant == [
        "task.status:task-1",
        "task.claim:task-1",
        "task.fence:task-1",
        "task.required_capability:task-1",
        "task.continuation_latest:task-1",
        "task.authority:task-1",
        "task.claim_deadline:task-1",
    ]
    assert "" not in keys_no_grant

    grant = DelegationGrant(
        grant_id="grant-1",
        root_task_id="task-1",
        issued_to_assignment_id="assign-1",
        issued_to_principal_id="principal-research-1",
        fencing_epoch=1,
        turn_id="turn-1",
        issued_at=now,
    )
    claim_with_grant = ActiveClaim(
        task_id="task-1",
        assignment_id="assign-1",
        claim_id="claim-1",
        root_task_id="task-1",
        turn_id="turn-1",
        runtime_principal_id="principal-research-1",
        bound_agent_id="research-agent-1",
        assigned_capability="research",
        execution_kind="initial",
        fencing_epoch=1,
        proof_digest="b" * 64,
        request_id="req-2",
        claimed_at=now,
        delivery_expires_at="2099-01-01T00:00:00Z",
        delegation_grant_id="grant-1",
    )

    gateway._client.eval = AsyncMock(return_value=["OK", "1"])
    await gateway.atomic_claim_task("task-1", claim_with_grant, grant=grant, now=now)

    call_args_grant = gateway._client.eval.call_args.args
    assert call_args_grant[1] == 8
    keys_with_grant = list(call_args_grant[2 : 2 + 8])
    assert keys_with_grant == [
        "task.status:task-1",
        "task.claim:task-1",
        "task.fence:task-1",
        "task.required_capability:task-1",
        "task.continuation_latest:task-1",
        "task.authority:task-1",
        "task.claim_deadline:task-1",
        "task.authority_grant:task-1:grant-1",
    ]
    assert "" not in keys_with_grant


@pytest.mark.anyio
async def test_save_stage_artifact_never_passes_empty_lua_keys() -> None:
    gateway = RedisGateway("redis://localhost:6379/0", allow_unauthenticated_local_test=True)
    now = datetime.now(UTC).isoformat()

    art_v1 = StageArtifact(
        artifact_id="art-1",
        producing_task_id="task-1",
        artifact_type="research_material",
        content={"material": "v1"},
        created_at=now,
        artifact_version=1,
    )
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_stage_artifact(art_v1, "assign-1", "principal-research-1", 1)

    v1_args = gateway._client.eval.call_args.args
    assert v1_args[0] == SAVE_STAGE_ARTIFACT_LUA
    assert v1_args[1] == 4
    v1_keys = list(v1_args[2 : 2 + 4])
    assert v1_keys == [
        "task.artifact:task-1:art-1:1",
        "task.artifacts:task-1",
        "task.claim:task-1",
        "task.fence:task-1",
    ]
    assert "" not in v1_keys

    art_v2 = StageArtifact(
        artifact_id="art-1",
        producing_task_id="task-1",
        artifact_type="research_material",
        content={"material": "v2"},
        created_at=now,
        artifact_version=2,
    )
    gateway._client.eval = AsyncMock(return_value="OK")
    await gateway.save_stage_artifact(art_v2, "assign-1", "principal-research-1", 1)

    v2_args = gateway._client.eval.call_args.args
    assert v2_args[1] == 5
    v2_keys = list(v2_args[2 : 2 + 5])
    assert v2_keys == [
        "task.artifact:task-1:art-1:2",
        "task.artifacts:task-1",
        "task.claim:task-1",
        "task.fence:task-1",
        "task.artifact:task-1:art-1:1",
    ]
    assert "" not in v2_keys


def test_claim_task_lua_declares_stable_seven_required_keys_plus_optional_grant() -> None:
    expected_header = (
        "local status_key = KEYS[1]\n"
        "local claim_key = KEYS[2]\n"
        "local fence_key = KEYS[3]\n"
        "local capability_key = KEYS[4]\n"
        "local continuation_latest_key = KEYS[5]\n"
        "local authority_key = KEYS[6]\n"
        "local deadline_key = KEYS[7]\n"
        "local grant_key = KEYS[8]"
    )
    assert expected_header in CLAIM_TASK_LUA


def test_service_requirements_pin_exact_redis_version() -> None:
    for svc in ("orchestrator", "worker", "research_agent", "analysis_agent"):
        req_lines = [
            line.strip()
            for line in (REPO_ROOT / "services" / svc / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert "redis==8.1.0" in req_lines, f"{svc} must pin redis==8.1.0"


# ============================================================================
# Increment 1 review corrections: explicit unauthenticated test mode
# (Correction 1), create_app fail-closed (Correction 2), Lua hardening
# (Correction 3), missing connection coverage (Correction 4), lazy api.app
# binding evidence.
# ============================================================================

_ALL_CLIENT_CLASSES = (RedisGateway, WorkerRedisBus, ResearchRedisBus, AnalysisRedisBus)


def test_bare_constructors_fail_closed_without_credentials_or_test_mode() -> None:
    for cls in _ALL_CLIENT_CLASSES:
        for url in ("redis://localhost:6379/0", "redis://redis:6379/0"):
            with pytest.raises(ValueError, match="allow_unauthenticated_local_test"):
                cls(url)


def test_direct_paired_credentials_accepted_and_partial_rejected() -> None:
    secret = "DirectCtorSecretToken1234567890abcd"
    for cls in _ALL_CLIENT_CLASSES:
        with patch("redis.asyncio.from_url") as mock_from_url:
            cls("redis://redis:6379/0", username="worker-1", password=secret)
            assert mock_from_url.call_args.kwargs["username"] == "worker-1"
            assert mock_from_url.call_args.kwargs["password"] == secret
            assert mock_from_url.call_args.kwargs["protocol"] == 2
            assert mock_from_url.call_args.kwargs["driver_info"] is None
            assert mock_from_url.call_args.kwargs["decode_responses"] is True

    for cls in _ALL_CLIENT_CLASSES:
        with pytest.raises(ValueError, match="Both username and password"):
            cls("redis://redis:6379/0", username="worker-1")
        with pytest.raises(ValueError, match="Both username and password"):
            cls("redis://redis:6379/0", password=secret)

    cfg = RedisConnectionConfig(
        url="redis://redis:6379/0",
        username="worker-1",
        password=secret,
        auth_mode="required",
    )
    with pytest.raises(ValueError, match="connection_config"):
        RedisGateway(
            "redis://redis:6379/0",
            connection_config=cfg,
            allow_unauthenticated_local_test=True,
        )
    with pytest.raises(ValueError, match="Do not combine credentials"):
        RedisGateway(
            "redis://localhost:6379/0",
            username="worker-1",
            password=secret,
            allow_unauthenticated_local_test=True,
        )


def test_invalid_redis_auth_mode_rejected() -> None:
    with pytest.raises(ValueError, match="REDIS_AUTH_MODE"):
        resolve_redis_connection_from_env({"REDIS_AUTH_MODE": "occasionally"})
    with pytest.raises(ValueError, match="REDIS_AUTH_MODE"):
        resolve_redis_connection_from_env(
            {"REDIS_URL": "redis://localhost:6379/0", "REDIS_AUTH_MODE": " sometimes "}
        )
    with pytest.raises(ValueError, match="Redis auth_mode"):
        RedisConnectionConfig(
            url="redis://localhost:6379/0",
            auth_mode="occasionally",  # type: ignore[arg-type]
        )


def test_explicit_test_mode_accepts_loopback_only() -> None:
    for url in (
        "redis://localhost:6379/0",
        "redis://127.0.0.1:6379/0",
        "redis://[::1]:6379/0",
        "redis://localhost:6379/15",
    ):
        for cls in _ALL_CLIENT_CLASSES:
            cls(url, allow_unauthenticated_local_test=True)

    cfg = resolve_redis_connection_from_env(
        {
            "REDIS_URL": "redis://127.0.0.1:6379/0",
            "REDIS_AUTH_MODE": "unauthenticated_local_test",
        }
    )
    assert cfg.is_authenticated is False


def test_explicit_test_mode_rejects_non_loopback_hosts() -> None:
    for url in (
        "redis://redis:6379/0",
        "redis://10.0.0.5:6379/0",
        "redis://0.0.0.0:6379/0",
        "redis://example.com:6379/0",
    ):
        for cls in _ALL_CLIENT_CLASSES:
            with pytest.raises(ValueError, match="loopback"):
                cls(url, allow_unauthenticated_local_test=True)
        with pytest.raises(ValueError, match="loopback"):
            RedisConnectionConfig(url=url, auth_mode="unauthenticated_local_test")
        with pytest.raises(ValueError, match="loopback"):
            resolve_redis_connection_from_env(
                {"REDIS_URL": url, "REDIS_AUTH_MODE": "unauthenticated_local_test"}
            )


def test_explicit_test_mode_prohibited_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(ValueError, match="production"):
        RedisGateway(
            "redis://localhost:6379/0", allow_unauthenticated_local_test=True
        )
    with pytest.raises(ValueError, match="production"):
        resolve_redis_connection_from_env(
            {
                "REDIS_URL": "redis://localhost:6379/0",
                "REDIS_AUTH_MODE": "unauthenticated_local_test",
            }
        )


def test_no_secret_or_credential_url_in_connection_errors() -> None:
    secret = "ConnectionErrorSecretMarker654321"
    bad_urls = (
        f"redis://worker-1:{secret}@localhost:6379/0",
        f"redis://:{secret}@redis:6379/0",
        f"redis://redis:6379/0?password={secret}",
    )
    for url in bad_urls:
        for cls in _ALL_CLIENT_CLASSES:
            with pytest.raises(ValueError) as exc_info:
                cls(url, allow_unauthenticated_local_test=True)
            assert secret not in str(exc_info.value)
            assert secret not in "".join(
                traceback.format_exception(
                    type(exc_info.value), exc_info.value, exc_info.tb
                )
            )
        with pytest.raises(ValueError) as bare_exc_info:
            RedisGateway(url)
        assert secret not in str(bare_exc_info.value)
        with pytest.raises(ValueError) as env_exc_info:
            resolve_redis_connection_from_env(
                {"REDIS_URL": url, "REDIS_AUTH_MODE": "unauthenticated_local_test"}
            )
        assert secret not in str(env_exc_info.value)
        _, kwargs = resolve_client_connection_kwargs(
            "redis://localhost:6379/0",
            username="worker-1",
            password=secret,
        )
        assert kwargs["password"] == secret
        assert secret not in repr(
            RedisConnectionConfig(
                url="redis://redis:6379/0",
                username="worker-1",
                password=secret,
                auth_mode="required",
            )
        )


def test_create_app_fails_closed_without_validated_connection() -> None:
    settings = Settings(redis_url="redis://redis:6379/0")
    assert settings.redis_connection is None
    with (
        patch("orchestrator.presentation.api.RedisGateway") as mock_gateway,
        pytest.raises(
            ValueError,
            match="requires a validated authenticated RedisConnectionConfig",
        ),
    ):
        create_app(settings=settings)
    mock_gateway.assert_not_called()


def test_create_app_fails_closed_for_unauthenticated_test_config() -> None:
    test_cfg = RedisConnectionConfig(
        url="redis://localhost:6379/0",
        auth_mode="unauthenticated_local_test",
    )
    settings = Settings(redis_url=test_cfg.url, redis_connection=test_cfg)
    with (
        patch("orchestrator.presentation.api.RedisGateway") as mock_gateway,
        pytest.raises(
            ValueError,
            match="requires a validated authenticated RedisConnectionConfig",
        ),
    ):
        create_app(settings=settings)
    mock_gateway.assert_not_called()


def test_claim_task_lua_grant_key_uses_explicit_type_check() -> None:
    assert (
        'local has_grant_key = (type(grant_key) == "string" and grant_key ~= "")'
        in CLAIM_TASK_LUA
    )
    assert "grant_key ~= nil" not in CLAIM_TASK_LUA


def test_api_module_import_has_no_redis_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "REDIS_URL",
        "REDIS_USERNAME",
        "REDIS_PASSWORD_FILE",
        "REDIS_AUTH_MODE",
        "APP_ENV",
    ):
        monkeypatch.delenv(var, raising=False)
    import orchestrator.presentation.api as api_module

    reloaded = importlib.reload(api_module)
    assert reloaded._default_app is None


def test_api_lazy_binding_caches_app_and_resolves_module_attribute() -> None:
    import orchestrator.presentation.api as api_module

    previous = api_module._default_app
    api_module._default_app = None
    try:
        sentinel = FastAPI()
        with patch.object(
            api_module, "create_app", return_value=sentinel
        ) as mock_create:
            loaded = importlib.import_module("orchestrator.presentation.api")
            # Plain attribute access exercises the same PEP 562 module
            # __getattr__ hook that Uvicorn's getattr(module, "app") uses.
            assert loaded.app is sentinel
            assert loaded.app is sentinel
            mock_create.assert_called_once_with()
    finally:
        api_module._default_app = previous
