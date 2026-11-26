"""Static tests for the fail-closed replay-integrity startup protocol.

Covers entrypoint replay-log enforcement and the current-startup health
gate without Docker. The live Redis subtraction scenario lives in
``test_redis_replay_health_live.py``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "infrastructure" / "redis" / "redis-acl-entrypoint.sh"
HEALTHCHECK = REPO_ROOT / "infrastructure" / "redis" / "redis-healthcheck.sh"
MOUNTINFO_HELPER = REPO_ROOT / "infrastructure" / "redis" / "redis-mountinfo.sh"

IDENTITIES = (
    "orchestrator",
    "worker-1",
    "research-agent-1",
    "research-agent-2",
    "analysis-agent-1",
    "healthcheck",
)
SYNTHETIC_SECRET = b"REPLAY_health_secret_for_tests_0123456789"


def _write_synthetic_secrets(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for identity in IDENTITIES:
        (directory / identity).write_bytes(SYNTHETIC_SECRET)
    (directory / "redis_generation").write_text(
        json.dumps(
            {
                "generation_id": "ab" * 16,
                "identity_generations": {identity: "cd" * 16 for identity in IDENTITIES},
            }
        )
    )


def _fake_bin(directory: Path, capture: Path) -> Path:
    fake_bin = directory / "fake-bin"
    fake_bin.mkdir()
    (fake_bin / "chown").write_text("#!/bin/sh\nexit 0\n")
    (fake_bin / "chown").chmod(0o755)
    (fake_bin / "setpriv").write_text(
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in --) shift; break;; --*) shift;; *) break;; esac\n'
        "done\n"
        'printf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
    )
    (fake_bin / "setpriv").chmod(0o755)
    capture.write_text("")
    return fake_bin


def _run_entrypoint(
    secret_dir: Path,
    runtime_dir: Path,
    directory: Path,
    args: list[str],
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    entrypoint_directory = directory / "entrypoint"
    entrypoint_directory.mkdir(parents=True, exist_ok=True)
    mountinfo = directory / "mountinfo"
    mountinfo.write_text(f"36 25 0:32 / {runtime_dir} rw,nosuid - tmpfs tmpfs rw\n")
    entrypoint_copy = entrypoint_directory / ENTRYPOINT.name
    entrypoint_copy.write_text(
        ENTRYPOINT.read_text().replace("/proc/self/mountinfo", str(mountinfo))
    )
    shutil.copyfile(MOUNTINFO_HELPER, entrypoint_directory / MOUNTINFO_HELPER.name)
    capture = directory / "argv-capture"
    fake_bin = _fake_bin(directory, capture)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "REDIS_ACL_SECRET_DIR": str(secret_dir),
            "REDIS_ACL_RUNTIME_DIR": str(runtime_dir),
            "REDIS_ACL_TEMPLATE": str(
                REPO_ROOT / "infrastructure" / "redis" / "users.acl.template"
            ),
            "REDIS_REPLAY_SEAL": str(runtime_dir / "replay-sealed.json"),
            "REDIS_REPLAY_EVIDENCE": str(runtime_dir / "startup-evidence.log"),
            "CAPTURE_FILE": str(capture),
            **(extra_env or {}),
        }
    )
    result = subprocess.run(
        ["/bin/sh", str(entrypoint_copy), *args],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    result_args = capture.read_text().split()
    return subprocess.CompletedProcess(
        result.args, result.returncode, result.stdout, result.stderr
    ), result_args


def _entrypoint_ok(tmp_path: Path, args: list[str], extra_env: dict[str, str] | None = None):
    secret_dir = tmp_path / "secrets"
    runtime_dir = tmp_path / "runtime"
    _write_synthetic_secrets(secret_dir)
    return (
        secret_dir,
        runtime_dir,
        *_run_entrypoint(secret_dir, runtime_dir, tmp_path / "work", args, extra_env),
    )


def test_entrypoint_enforces_replay_log_for_compose_args(tmp_path: Path) -> None:
    _secret_dir, runtime_dir, result, argv = _entrypoint_ok(
        tmp_path, ["redis-server", "--appendonly", "yes"]
    )
    assert result.returncode == 0, result.stderr
    log = runtime_dir / "replay-integrity.log"
    assert "--logfile" in argv and "--loglevel" in argv
    assert argv[argv.index("--logfile") + 1] == str(log)
    assert argv[argv.index("--loglevel") + 1] == "notice"
    assert argv[argv.index("--aclfile") + 1] == str(runtime_dir / "users.acl")
    assert "--appendonly" in argv and "yes" in argv
    assert log.exists() and log.read_bytes() == b""
    assert stat.S_IMODE(log.stat().st_mode) == 0o660
    marker = json.loads((runtime_dir / "startup.json").read_text())
    assert set(marker) == {"generation_sha256"}
    installed = runtime_dir / "credential-generation.json"
    assert marker["generation_sha256"] == hashlib.sha256(installed.read_bytes()).hexdigest()
    assert (runtime_dir / "users.acl").exists()
    assert not (runtime_dir / "replay-sealed.json").exists()
    assert not (runtime_dir / "startup-evidence.log").exists()


def test_entrypoint_truncates_stale_replay_log(tmp_path: Path) -> None:
    secret_dir = tmp_path / "secrets"
    runtime_dir = tmp_path / "runtime"
    _write_synthetic_secrets(secret_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stale = runtime_dir / "replay-integrity.log"
    stale.write_text("C 1:M 01 Jan * CRITICAL NOPERM denial from a prior start\n")
    result, _ = _run_entrypoint(
        secret_dir, runtime_dir, tmp_path / "work", ["redis-server", "--appendonly", "yes"]
    )
    assert result.returncode == 0, result.stderr
    assert stale.read_bytes() == b""


def test_entrypoint_restart_creates_fresh_current_start_evidence(tmp_path: Path) -> None:
    secret_dir = tmp_path / "secrets"
    runtime_dir = tmp_path / "runtime"
    _write_synthetic_secrets(secret_dir)
    first, _ = _run_entrypoint(
        secret_dir, runtime_dir, tmp_path / "work", ["redis-server", "--appendonly", "yes"]
    )
    assert first.returncode == 0, first.stderr
    marker_file = runtime_dir / "startup.json"
    first_marker = marker_file.read_bytes()
    log_file = runtime_dir / "replay-integrity.log"
    log_file.write_bytes(b"sentinel from previous start\n")
    (runtime_dir / "replay-sealed.json").write_bytes(b'{"stale": true}\n')
    (runtime_dir / "startup-evidence.log").write_bytes(b"stale evidence\n")
    old_mtime = marker_file.stat().st_mtime_ns
    assert old_mtime > 0
    second, _ = _run_entrypoint(
        secret_dir, runtime_dir, tmp_path / "work2", ["redis-server", "--appendonly", "yes"]
    )
    assert second.returncode == 0, second.stderr
    assert log_file.read_bytes() == b""
    assert marker_file.stat().st_mtime_ns >= old_mtime
    assert (
        json.loads(marker_file.read_bytes())["generation_sha256"]
        == json.loads(first_marker)["generation_sha256"]
    )
    assert not (runtime_dir / "replay-sealed.json").exists()
    assert not (runtime_dir / "startup-evidence.log").exists()


def test_entrypoint_preserves_matching_caller_log_options(tmp_path: Path) -> None:
    secret_dir = tmp_path / "secrets"
    runtime_dir = tmp_path / "runtime"
    _write_synthetic_secrets(secret_dir)
    log = runtime_dir / "replay-integrity.log"
    result, argv = _run_entrypoint(
        secret_dir,
        runtime_dir,
        tmp_path / "work",
        [
            "redis-server",
            "--appendonly",
            "yes",
            "--logfile",
            str(log),
            "--loglevel",
            "notice",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert argv.count("--logfile") == 1
    assert argv.count("--loglevel") == 1


def test_entrypoint_rejects_conflicting_logfile(tmp_path: Path) -> None:
    secret_dir = tmp_path / "secrets"
    runtime_dir = tmp_path / "runtime"
    _write_synthetic_secrets(secret_dir)
    result, _ = _run_entrypoint(
        secret_dir,
        runtime_dir,
        tmp_path / "work",
        ["redis-server", "--appendonly", "yes", "--logfile", "/tmp/elsewhere.log"],
    )
    assert result.returncode != 0
    assert "replay-integrity" in result.stderr


def test_entrypoint_rejects_unbounded_loglevel(tmp_path: Path) -> None:
    secret_dir = tmp_path / "secrets"
    runtime_dir = tmp_path / "runtime"
    _write_synthetic_secrets(secret_dir)
    result, _ = _run_entrypoint(
        secret_dir,
        runtime_dir,
        tmp_path / "work",
        ["redis-server", "--appendonly", "yes", "--loglevel", "verbose"],
    )
    assert result.returncode != 0
    assert "bounded" in result.stderr


def test_entrypoint_rejects_symlinked_replay_log(tmp_path: Path) -> None:
    secret_dir = tmp_path / "secrets"
    runtime_dir = tmp_path / "runtime"
    _write_synthetic_secrets(secret_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "replay-integrity.log").symlink_to("/tmp/replay-evasion.log")
    result, _ = _run_entrypoint(
        secret_dir,
        runtime_dir,
        tmp_path / "work",
        ["redis-server", "--appendonly", "yes"],
        extra_env={"REDIS_REPLAY_LOG": str(runtime_dir / "replay-integrity.log")},
    )
    assert result.returncode != 0
    assert "symlink" in result.stderr


def _fake_redis_cli(directory: Path) -> Path:
    # chown is faked like in the entrypoint tests: success by default so the
    # protocol logic is exercised as the test user, with an explicit failure
    # mode proving seal creation fails closed when ownership cannot be set.
    # Production health checks run as root (they already require the
    # root-only password file), so the real chown succeeds there.
    fake_bin = directory / "cli-bin"
    fake_bin.mkdir()
    (fake_bin / "redis-cli").write_text(
        "#!/bin/sh\n"
        'if [ "${FAKE_REDISCLI_EXIT:-0}" != "0" ]; then exit 1; fi\n'
        'printf "%s\\n" "${FAKE_REDISCLI_OUTPUT:-PONG}"\n',
    )
    (fake_bin / "redis-cli").chmod(0o755)
    (fake_bin / "chown").write_text(
        '#!/bin/sh\nif [ "${FAKE_CHOWN_FAIL:-0}" != "0" ]; then exit 1; fi\nexit 0\n'
    )
    (fake_bin / "chown").chmod(0o755)
    return fake_bin


READY_BANNER = "1:M 01 Jan 00:00:00.000 oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo\n"
READY_LINE = "1:M 01 Jan 00:00:00.001 * Ready to accept connections\n"
CLEAN_STARTUP_LOG = READY_BANNER + READY_LINE


def _health_env(
    tmp_path: Path,
    *,
    log_content: str | None = CLEAN_STARTUP_LOG,
    marker_generation: str | None = "match",
    legacy_marker: bool = False,
    malformed_marker: bool = False,
    password: bytes = SYNTHETIC_SECRET,
    ping_output: str = "PONG",
    log_older_than_marker: bool = False,
    chown_fails: bool = False,
) -> dict[str, str]:
    runtime_dir = tmp_path / "health-runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    generation_file = runtime_dir / "generation.json"
    generation_file.write_bytes(b'{"generation_id":"unit-health-generation"}\n')
    digest = hashlib.sha256(generation_file.read_bytes()).hexdigest()
    if marker_generation == "match":
        marker_generation = digest
    elif marker_generation is None:
        marker_generation = "0" * 64
    marker_file = runtime_dir / "startup.json"
    if malformed_marker:
        marker_file.write_text("not-json{{{", encoding="utf-8")
    elif legacy_marker:
        marker_file.write_text(
            json.dumps({"start_id": "legacy-start", "generation_sha256": marker_generation}),
            encoding="utf-8",
        )
    else:
        marker_file.write_text(
            json.dumps({"generation_sha256": marker_generation}),
            encoding="utf-8",
        )
    log_file = runtime_dir / "replay-integrity.log"
    if log_content is None:
        if log_file.exists():
            log_file.unlink()
    else:
        log_file.write_text(log_content, encoding="utf-8")
    if log_older_than_marker and log_file.exists():
        old = 1_000_000_000
        os.utime(log_file, (old, old))
        new = 2_000_000_000
        os.utime(marker_file, (new, new))
    password_file = runtime_dir / "healthcheck.password"
    password_file.write_bytes(password)
    fake_bin = _fake_redis_cli(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "REDIS_HEALTHCHECK_PASSWORD_FILE": str(password_file),
            "REDIS_REPLAY_LOG": str(log_file),
            "REDIS_STARTUP_MARKER": str(marker_file),
            "REDIS_CREDENTIAL_GENERATION_FILE": str(generation_file),
            "REDIS_REPLAY_SEAL": str(runtime_dir / "replay-sealed.json"),
            "REDIS_REPLAY_EVIDENCE": str(runtime_dir / "startup-evidence.log"),
            "FAKE_REDISCLI_OUTPUT": ping_output,
            "FAKE_REDISCLI_EXIT": "0",
            "FAKE_CHOWN_FAIL": "1" if chown_fails else "0",
        }
    )
    return environment


def _run_health(environment: dict[str, str]) -> int:
    return subprocess.run(
        ["/bin/sh", str(HEALTHCHECK)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    ).returncode


def test_health_passes_with_current_marker_clean_log_and_pong(tmp_path: Path) -> None:
    assert _run_health(_health_env(tmp_path)) == 0


def test_health_fails_without_replay_log(tmp_path: Path) -> None:
    assert _run_health(_health_env(tmp_path, log_content=None)) == 1


def test_health_fails_without_marker(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    Path(environment["REDIS_STARTUP_MARKER"]).unlink()
    assert _run_health(environment) == 1


def test_health_fails_with_stale_log(tmp_path: Path) -> None:
    environment = _health_env(tmp_path, log_older_than_marker=True)
    assert _run_health(environment) == 1


def test_health_fails_with_wrong_generation(tmp_path: Path) -> None:
    assert _run_health(_health_env(tmp_path, marker_generation=None)) == 1


def test_health_fails_with_malformed_marker(tmp_path: Path) -> None:
    assert _run_health(_health_env(tmp_path, malformed_marker=True)) == 1


def test_health_fails_on_current_denial_marker(tmp_path: Path) -> None:
    log = (
        READY_BANNER
        + "1:M 01 Jan 00:00:00.001 # AOF-loading-client ::user default ::cmd SET NOPERM\n"
        + READY_LINE
    )
    environment = _health_env(tmp_path, log_content=log)
    assert _run_health(environment) == 1
    assert not Path(environment["REDIS_REPLAY_SEAL"]).exists()


def test_health_fails_on_aof_load_failure(tmp_path: Path) -> None:
    log = READY_BANNER + "1:M 01 Jan 00:00:01.000 # Failed to load append only file:aoi\n"
    assert _run_health(_health_env(tmp_path, log_content=log)) == 1


def test_health_requires_startup_banner(tmp_path: Path) -> None:
    assert _run_health(_health_env(tmp_path, log_content=READY_LINE)) == 1


def test_health_requires_readiness_boundary(tmp_path: Path) -> None:
    environment = _health_env(tmp_path, log_content=READY_BANNER)
    assert _run_health(environment) == 1
    assert not Path(environment["REDIS_REPLAY_SEAL"]).exists()


def test_legacy_marker_without_random_id_still_passes(tmp_path: Path) -> None:
    assert _run_health(_health_env(tmp_path, legacy_marker=True)) == 0


def test_health_seals_clean_startup(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    assert _run_health(environment) == 0
    seal_file = Path(environment["REDIS_REPLAY_SEAL"])
    assert seal_file.exists()
    seal = json.loads(seal_file.read_text(encoding="utf-8"))
    assert len(seal["generation_sha256"]) == 64
    assert len(seal["window_sha256"]) == 64


def test_sealed_startup_ignores_later_runtime_noperm(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    assert _run_health(environment) == 0
    seal_file = Path(environment["REDIS_REPLAY_SEAL"])
    sealed_bytes = seal_file.read_bytes()
    log_file = Path(environment["REDIS_REPLAY_LOG"])
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("1:M 01 Jan 00:01:00.000 # Runtime client ::user worker-1 ::cmd EVAL NOPERM\n")
    assert _run_health(environment) == 0
    assert seal_file.read_bytes() == sealed_bytes
    assert "NOPERM" in log_file.read_text(encoding="utf-8")


def test_stale_seal_from_previous_start_is_ignored(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    assert _run_health(environment) == 0
    log_file = Path(environment["REDIS_REPLAY_LOG"])
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(READY_BANNER)
        handle.write("1:M 01 Jan 00:02:00.000 # New-start replay denial NOPERM\n")
        handle.write(READY_LINE)
    assert _run_health(environment) == 1


def test_restart_with_clean_replay_replaces_seal(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    assert _run_health(environment) == 0
    first_seal = Path(environment["REDIS_REPLAY_SEAL"]).read_bytes()
    log_file = Path(environment["REDIS_REPLAY_LOG"])
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("1:M 01 Jan 00:05:00.000 oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo\n")
        handle.write("1:M 01 Jan 00:05:00.001 * Ready to accept connections\n")
    assert _run_health(environment) == 0
    second_seal = Path(environment["REDIS_REPLAY_SEAL"]).read_bytes()
    assert second_seal != first_seal


def test_startup_evidence_is_retained_separately_and_bounded(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    assert _run_health(environment) == 0
    evidence_file = Path(environment["REDIS_REPLAY_EVIDENCE"])
    assert evidence_file.exists()
    window = (READY_BANNER + READY_LINE).encode()
    assert evidence_file.read_bytes() == window
    assert len(window) <= 1048576


def test_continuing_log_truncation_does_not_invalidate_seal(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    assert _run_health(environment) == 0
    seal_file = Path(environment["REDIS_REPLAY_SEAL"])
    sealed_bytes = seal_file.read_bytes()
    Path(environment["REDIS_REPLAY_LOG"]).write_bytes(b"")
    assert _run_health(environment) == 0
    assert seal_file.read_bytes() == sealed_bytes


def test_oversize_continuing_log_is_truncated_not_failed(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    assert _run_health(environment) == 0
    log_file = Path(environment["REDIS_REPLAY_LOG"])
    with log_file.open("ab") as handle:
        handle.write(b"R" * (8388608 + 1))
    assert _run_health(environment) == 0
    assert log_file.stat().st_size <= 8388608
    assert _run_health(environment) == 0


def test_seal_ownership_failure_fails_closed(tmp_path: Path) -> None:
    assert _run_health(_health_env(tmp_path, chown_fails=True)) == 1


def test_tampered_startup_evidence_cannot_satisfy_health(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    assert _run_health(environment) == 0
    evidence_file = Path(environment["REDIS_REPLAY_EVIDENCE"])
    evidence_file.write_bytes(b"tampered evidence\n")
    log_file = Path(environment["REDIS_REPLAY_LOG"])
    log_file.write_bytes(b"")
    assert _run_health(environment) == 1
    assert not Path(environment["REDIS_REPLAY_SEAL"]).exists()


def test_window_exactly_at_limit_is_accepted(tmp_path: Path) -> None:
    max_window = 1048576
    filler_length = max_window - len(READY_BANNER.encode()) - len(READY_LINE.encode()) - 1
    assert filler_length > 0
    log = READY_BANNER + "F" * filler_length + "\n" + READY_LINE
    assert len(log.encode()) == max_window
    assert _run_health(_health_env(tmp_path, log_content=log)) == 0


def test_window_over_limit_fails_closed(tmp_path: Path) -> None:
    max_window = 1048576
    filler_length = max_window - len(READY_BANNER.encode()) - len(READY_LINE.encode())
    log = READY_BANNER + "F" * filler_length + "\n" + READY_LINE
    assert len(log.encode()) > max_window
    environment = _health_env(tmp_path, log_content=log)
    assert _run_health(environment) == 1
    assert not Path(environment["REDIS_REPLAY_SEAL"]).exists()


def test_ping_without_positive_evidence_fails(tmp_path: Path) -> None:
    assert _run_health(_health_env(tmp_path, log_content="noise without readiness\n")) == 1


def test_health_fails_on_wrong_pong(tmp_path: Path) -> None:
    assert _run_health(_health_env(tmp_path, ping_output="ERR")) == 1


def test_health_identity_remains_keyless() -> None:
    text = HEALTHCHECK.read_text(encoding="utf-8")
    assert text.count("redis-cli") == 1
    cli_lines = [line for line in text.splitlines() if "redis-cli" in line]
    assert len(cli_lines) == 1
    assert "ping" in cli_lines[0]
    for command in ("get", "set", "keys", "del", "xadd", "rpush", "eval", "xread", "hgetall"):
        assert command not in cli_lines[0].lower().split("ping")[1]


def test_harness_final_command_includes_replay_logging() -> None:
    from scripts.redis_acl_live_harness import redis_server_command

    for appendonly in (True, False):
        command = redis_server_command(appendonly=appendonly)
        assert "--logfile /run/redis/replay-integrity.log" in command
        assert "--loglevel notice" in command
        assert "--aclfile /run/redis/users.acl" in command


def test_compose_uses_governed_entrypoint_without_log_conflicts() -> None:
    compose = (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "/usr/local/bin/redis-acl-entrypoint.sh" in compose
    assert "--logfile" not in compose
    assert "--loglevel" not in compose


def test_replay_log_preparation_is_ordered_and_restrictive() -> None:
    from scripts.redis_acl_live_harness import REDIS_REPLAY_LOG_PATH, redis_replay_log_prep

    prep = redis_replay_log_prep()
    assert REDIS_REPLAY_LOG_PATH in prep
    symlink_reject = prep.index("-L")
    truncate = prep.index(": >")
    chown = prep.index("chown root:redis")
    chmod = prep.index("chmod 0660")
    assert symlink_reject < truncate < chown < chmod
    assert "0666" not in prep
    assert "0777" not in prep
    assert "o+w" not in prep
    assert "a+w" not in prep


def test_replay_log_preparation_supports_evidence_paths() -> None:
    import tests.test_redis_replay_health_live as live
    from scripts.redis_acl_live_harness import redis_replay_log_prep

    assert "/run/redis-evd/replay.log" in redis_replay_log_prep(
        log_path="/run/redis-evd/replay.log"
    )
    source = inspect.getsource(live._start_replay_container)
    assert "redis_replay_log_prep" in source
    assert "setpriv --reuid=redis --regid=redis" in source


def test_harness_startup_paths_prepare_log_before_redis() -> None:
    from scripts.redis_acl_live_harness import LiveRedisAclCluster

    for method in ("_start_redis_container", "_start_redis_with_compose"):
        source = inspect.getsource(getattr(LiveRedisAclCluster, method))
        assert "redis_replay_log_prep" in source
        assert "redis_server_command(" in source
        assert source.index("redis_replay_log_prep") < source.index("redis_server_command(")


def test_final_redis_process_remains_redis_user() -> None:
    from scripts.redis_acl_live_harness import redis_server_command

    command = redis_server_command(appendonly=True)
    assert command.startswith(
        "exec setpriv --reuid=redis --regid=redis --clear-groups -- redis-server"
    )
    entrypoint = (REPO_ROOT / "infrastructure" / "redis" / "redis-acl-entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert "exec setpriv --reuid=redis --regid=redis --clear-groups -- " in entrypoint


def test_replay_evidence_permissions_are_restrictive() -> None:
    entrypoint = (REPO_ROOT / "infrastructure" / "redis" / "redis-acl-entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert 'chown root:redis "$replay_log"' in entrypoint
    assert 'chmod 0660 "$replay_log"' in entrypoint
    assert 'chmod 0640 "$marker_temporary"' in entrypoint
    assert "chmod 0666" not in entrypoint
    assert "chmod 0777" not in entrypoint


def test_live_marker_helper_matches_production_entrypoint_byte_for_byte(
    tmp_path: Path,
) -> None:
    import tests.test_redis_replay_health_live as live

    secret_dir, runtime_dir, result, _argv = _entrypoint_ok(
        tmp_path, ["redis-server", "--appendonly", "yes"]
    )
    assert result.returncode == 0, result.stderr
    generation_text = (secret_dir / "redis_generation").read_text(encoding="utf-8")
    assert (runtime_dir / "startup.json").read_bytes() == live._production_marker_bytes(
        generation_text
    )


def test_legacy_extra_start_id_tolerated_but_generation_remains_mandatory(
    tmp_path: Path,
) -> None:
    # Compatibility tolerates extra obsolete fields only: a legacy marker
    # carrying both start_id and the current generation digest passes, while
    # the same shape without generation_sha256 fails closed.
    assert _run_health(_health_env(tmp_path / "legacy", legacy_marker=True)) == 0
    environment = _health_env(tmp_path / "no-generation", legacy_marker=True)
    Path(environment["REDIS_STARTUP_MARKER"]).write_text(
        json.dumps({"start_id": "legacy-start"}), encoding="utf-8"
    )
    assert _run_health(environment) == 1


def test_health_fails_with_short_generation_digest(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    Path(environment["REDIS_STARTUP_MARKER"]).write_text(
        json.dumps({"generation_sha256": "abc"}), encoding="utf-8"
    )
    assert _run_health(environment) == 1


def test_health_fails_with_non_string_generation_digest(tmp_path: Path) -> None:
    environment = _health_env(tmp_path)
    Path(environment["REDIS_STARTUP_MARKER"]).write_text(
        '{"generation_sha256": 123}', encoding="utf-8"
    )
    assert _run_health(environment) == 1
