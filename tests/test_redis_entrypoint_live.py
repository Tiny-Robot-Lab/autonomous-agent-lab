"""Live control booting Redis through the actual production ACL entrypoint.

This control executes ``infrastructure/redis/redis-acl-entrypoint.sh`` for
real inside a disposable Redis 7.4.12 container and proves the production
bootstrap: tmpfs verification, secret and generation validation, ACL
rendering with the exact canonical default-user rule, runtime file
permissions, the exact production startup-marker shape, enforced replay
logging, unprivileged PID 1, and the production healthcheck sealing a
healthy startup.

Scope distinction: the harness-assembled selector-subtraction test in
``test_redis_replay_health_live.py`` varies the ACL to prove exact
replay-policy subtraction behavior on the same persisted AOF. This file
proves the real bootstrap path instead and must not be mistaken for a
subtraction test: it always uses the real template and the real entrypoint.

Determinism note: the pre-readiness proof uses a test-only ``redis-server``
PATH wrapper that blocks at the final exec boundary inside the disposable
container. The real entrypoint still performs the full bootstrap (tmpfs
verification, secret validation, ACL rendering, marker and replay-log
preparation, governed logging enforcement, setpriv drop) before reaching the
wrapper, and the wrapper records the exact argument vector, waits for an
explicit host-side release signal, then execs the real Redis binary with the
original arguments. Correctness comes from that explicit barrier, never from
winning a host-side polling race against Redis readiness.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

import pytest

from scripts.generate_redis_secrets import generate_secrets, read_generation_secret
from scripts.redis_acl_live_harness import (
    ACL_TEMPLATE_PATH,
    PRODUCTION_IDENTITIES,
    REDIS_ENTRYPOINT_PATH,
    REDIS_HEALTHCHECK_PATH,
    REDIS_IMAGE,
    REDIS_MOUNTINFO_PATH,
    REDIS_REPLAY_LOG_LEVEL,
    REDIS_REPLAY_LOG_PATH,
    CliResult,
    _run_docker,
    redis_replay_log_prep,
    redis_server_command,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = "/run/redis"
TMPFS_OPTIONS = "rw,noexec,nosuid,nodev,size=16m,mode=0750"


def _secret_mounts(secret_dir: Path) -> list[str]:
    current = secret_dir / "current"
    mounts: list[str] = []
    for identity in PRODUCTION_IDENTITIES:
        mounts.extend(
            [
                "--mount",
                f"type=bind,src={current / identity},dst=/run/secrets/{identity},readonly",
            ]
        )
    mounts.extend(
        [
            "--mount",
            f"type=bind,src={current / 'generation.json'},dst=/run/secrets/redis_generation,readonly",
        ]
    )
    for src_path, dst_path in (
        (REDIS_ENTRYPOINT_PATH, "/usr/local/bin/redis-acl-entrypoint.sh"),
        (REDIS_HEALTHCHECK_PATH, "/usr/local/bin/redis-healthcheck.sh"),
        (REDIS_MOUNTINFO_PATH, "/usr/local/bin/redis-mountinfo.sh"),
        (ACL_TEMPLATE_PATH, "/usr/local/bin/users.acl.template"),
    ):
        mounts.extend(["--mount", f"type=bind,src={src_path},dst={dst_path},readonly"])
    return mounts


def _run_production_entrypoint(
    name: str, secret_dir: Path, *redis_args: str
) -> CliResult:
    return _run_docker(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "--tmpfs",
            f"{RUNTIME_DIR}:{TMPFS_OPTIONS}",
            *_secret_mounts(secret_dir),
            "--entrypoint",
            "/bin/sh",
            REDIS_IMAGE,
            "/usr/local/bin/redis-acl-entrypoint.sh",
            *redis_args,
        ],
        timeout=120,
    )


GATE_MOUNT_DST = "/opt/tiny-agent-lab-entrypoint-gate"
GATE_SIGNAL_DIR = "/tmp/tiny-agent-lab-entrypoint-gate"
GATE_REACHED_TIMEOUT = 60.0

# Test-only pre-readiness barrier. Installed through the disposable
# container's PATH (same technique the ACL harness already uses for its
# render stub), so the real entrypoint resolves this wrapper at its final
# `exec setpriv ... -- redis-server ...` step after completing bootstrap and
# the privilege drop. The wrapper records the exact argument vector, signals
# arrival, blocks on a host-controlled release file, then execs the real
# Redis binary (resolved from the PATH with the wrapper directory removed)
# with the original arguments. It never touches the ACL, the marker, the
# replay log, or any seal/evidence artifact.
GATE_WRAPPER_SCRIPT = """#!/bin/sh
set -eu
gate_dir="GATE_SIGNAL_DIR_PLACEHOLDER"
reached="$gate_dir/reached"
argv_file="$gate_dir/argv"
release="$gate_dir/release"
mkdir -p "$gate_dir"
{
    printf '%s\\n' "$#"
    printf '%s\\n' "$@"
} > "$argv_file"
: > "$reached"
while [ ! -f "$release" ]; do
    sleep 0.2
done
wrapper_dir=$(dirname "$0")
filtered=""
old_ifs=$IFS
IFS=:
# shellcheck disable=SC2162
for entry in $PATH; do
    if [ -n "$entry" ] && [ "$entry" != "$wrapper_dir" ]; then
        if [ -z "$filtered" ]; then
            filtered="$entry"
        else
            filtered="$filtered:$entry"
        fi
    fi
done
IFS=$old_ifs
real_binary=$(PATH="$filtered" command -v redis-server) || exit 127
exec "$real_binary" "$@"
""".replace("GATE_SIGNAL_DIR_PLACEHOLDER", GATE_SIGNAL_DIR)


def _write_gate_wrapper(host_dir: Path) -> Path:
    """Install the test-only redis-server wrapper into a host mount dir."""
    host_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    host_dir.chmod(0o755)
    wrapper = host_dir / "redis-server"
    wrapper.write_text(GATE_WRAPPER_SCRIPT, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def _run_gated_production_entrypoint(
    name: str, secret_dir: Path, gate_host_dir: Path
) -> CliResult:
    """Boot the real entrypoint with the PATH gate shadowing redis-server.

    A minimal ``/bin/sh -c`` shim prepares the container-local signal
    directory and prepends the read-only gate mount to PATH before execing
    the unmodified production entrypoint with the production arguments.
    Every bootstrap step still runs inside the real entrypoint.
    """
    mounts = _secret_mounts(secret_dir)
    mounts.extend(
        [
            "--mount",
            f"type=bind,src={gate_host_dir},dst={GATE_MOUNT_DST},readonly",
        ]
    )
    shim = (
        f"mkdir -p {GATE_SIGNAL_DIR} && "
        f"chown redis:redis {GATE_SIGNAL_DIR} && "
        f"chmod 0700 {GATE_SIGNAL_DIR} && "
        f'PATH="{GATE_MOUNT_DST}:$PATH" '
        "exec /bin/sh /usr/local/bin/redis-acl-entrypoint.sh "
        "redis-server --appendonly yes"
    )
    return _run_docker(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "--tmpfs",
            f"{RUNTIME_DIR}:{TMPFS_OPTIONS}",
            *mounts,
            "--entrypoint",
            "/bin/sh",
            REDIS_IMAGE,
            "-c",
            shim,
        ],
        timeout=120,
    )


def _exec(name: str, *argv: str, stdin_text: str | None = None) -> CliResult:
    # Forward stdin only when the caller provides it: without -i the
    # Docker CLI drops piped input and an in-container `read` sees EOF
    # (observed live as an empty healthcheck password). This matches the
    # disposable harness, which always pairs stdin with `docker exec -i`.
    interactive = ["-i"] if stdin_text is not None else []
    return _run_docker(
        ["docker", "exec", *interactive, name, *argv],
        check=False,
        stdin_text=stdin_text,
        timeout=30,
    )


def _exec_sh(name: str, script: str, stdin_text: str | None = None) -> CliResult:
    return _exec(name, "/bin/sh", "-c", script, stdin_text=stdin_text)


def _read_log(name: str) -> str:
    result = _exec(name, "cat", f"{RUNTIME_DIR}/replay-integrity.log")
    assert result.returncode == 0, result.combined
    return result.stdout


def _wait_for_readiness(name: str, timeout: float = 60.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        content = _read_log(name)
        if "Ready to accept connections" in content:
            return content
        time.sleep(0.5)
    raise AssertionError(
        f"Redis never reached readiness in {name}; log content:\n{content}"
    )


def _run_production_healthcheck(name: str, extra_env: str = "") -> CliResult:
    return _exec_sh(
        name,
        f"{extra_env} /bin/sh /usr/local/bin/redis-healthcheck.sh",
    )


def _wait_for_gate_reached(
    name: str, timeout: float = GATE_REACHED_TIMEOUT
) -> None:
    """Wait for the test-only wrapper barrier signal, guarded by a timeout.

    The signal proves the real production entrypoint completed bootstrap and
    reached the final redis-server exec boundary while Redis itself is still
    blocked. A timeout guards a hung boot but never substitutes for the
    explicit barrier signal.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reached = _exec_sh(name, f"test -f {GATE_SIGNAL_DIR}/reached")
        if reached.returncode == 0:
            return
        state = _run_docker(
            ["docker", "inspect", "--format", "{{.State.Status}}", name],
            check=False,
            timeout=30,
        )
        if state.returncode != 0 or state.stdout.strip() != "running":
            logs = _run_docker(["docker", "logs", name], check=False, timeout=30)
            raise AssertionError(
                f"gated entrypoint container {name} exited before reaching "
                f"the redis-server barrier:\n{logs.combined}"
            )
        time.sleep(0.2)
    logs = _run_docker(["docker", "logs", name], check=False, timeout=30)
    raise AssertionError(
        f"entrypoint gate never reached redis-server exec in {name}:\n{logs.combined}"
    )


def _read_gate_argv(name: str) -> list[str]:
    """Read the exact entrypoint-produced Redis argument vector."""
    result = _exec(name, "cat", f"{GATE_SIGNAL_DIR}/argv")
    assert result.returncode == 0, result.combined
    lines = result.stdout.splitlines()
    assert lines, f"gate argv file is empty in {name}"
    argc = int(lines[0])
    argv = lines[1:]
    assert len(argv) == argc, (
        f"gate argv count mismatch in {name}: declared {argc}, "
        f"observed {len(argv)}: {argv!r}"
    )
    return argv


def _assert_gate_argv_governed(argv: list[str], name: str) -> None:
    """Prove the wrapper saw the exact governed production arguments."""
    for flag, expected in (
        ("--logfile", f"{RUNTIME_DIR}/replay-integrity.log"),
        ("--loglevel", "notice"),
        ("--aclfile", f"{RUNTIME_DIR}/users.acl"),
    ):
        assert flag in argv, f"{flag} missing from gate argv in {name}: {argv!r}"
        assert argv[argv.index(flag) + 1] == expected, (
            f"{flag} value mismatch in gate argv in {name}: {argv!r}"
        )


def _boot_with_barrier_proof(
    base: str,
    secret_dir: Path,
    created: list[str],
    gate_host_dir: Path,
    expected_generation: bytes,
) -> str:
    """Boot once through the real entrypoint, proving pre-readiness denial.

    The PATH wrapper blocks Redis before readiness, so the host observes the
    denial deterministically: no retry loop, no readiness race. The wrapper
    is released only after the blocked-phase proofs complete.
    """
    name = f"{base}-gated-0"
    result = _run_gated_production_entrypoint(name, secret_dir, gate_host_dir)
    assert result.returncode == 0, result.combined
    created.append(name)
    try:
        _wait_for_gate_reached(name)
        argv = _read_gate_argv(name)
        _assert_gate_argv_governed(argv, name)

        # Bootstrap artifacts already exist while Redis is still blocked.
        marker = _exec(name, "cat", f"{RUNTIME_DIR}/startup.json")
        assert marker.returncode == 0, marker.combined
        expected_digest = hashlib.sha256(expected_generation).hexdigest()
        assert marker.stdout.encode() == (
            f'{{"generation_sha256":"{expected_digest}"}}\n'
        ).encode(), marker.combined
        content = _read_log(name)
        assert "Ready to accept connections" not in content, (
            f"Redis reached readiness before the barrier release in {name}"
        )

        # No valid replay seal exists while blocked.
        absent = _exec_sh(
            name,
            f"test ! -e {RUNTIME_DIR}/replay-sealed.json && "
            f"test ! -e {RUNTIME_DIR}/replay-startup.evidence",
        )
        assert absent.returncode == 0, absent.combined

        # The actual production healthcheck fails deterministically here:
        # no readiness window and no seal can satisfy the gate.
        blocked = _run_production_healthcheck(name)
        assert blocked.returncode != 0, blocked.combined

        released = _exec_sh(name, f": > {GATE_SIGNAL_DIR}/release")
        assert released.returncode == 0, released.combined
        _wait_for_readiness(name)
        return name
    except BaseException:
        _run_docker(["docker", "rm", "-f", name], check=False, timeout=60)
        created.pop()
        raise


@pytest.mark.live_redis_rotation
def test_production_entrypoint_boot_seals_healthy_startup() -> None:
    """Boot Redis through the real entrypoint and prove the sealed startup."""
    temp_dir = Path(tempfile.mkdtemp(prefix="tiny-agent-lab-entrypoint-"))
    temp_dir.chmod(0o700)
    secret_dir = temp_dir / "synthetic-secrets"
    secret_dir.mkdir(mode=0o700)
    generate_secrets(secret_dir)
    health_password = read_generation_secret(secret_dir, "healthcheck").decode("ascii")
    generation_bytes = (secret_dir / "current" / "generation.json").read_bytes()
    gate_host_dir = temp_dir / "entrypoint-gate"
    _write_gate_wrapper(gate_host_dir)
    created: list[str] = []
    try:
        name = _boot_with_barrier_proof(
            "tiny-agent-lab-entrypoint-redis",
            secret_dir,
            created,
            gate_host_dir,
            generation_bytes,
        )

        # The runtime directory is a real tmpfs, as the entrypoint verified.
        # mountinfo fields: $5 is the mount point, the filesystem type
        # follows the "-" separator (same layout the production
        # redis-mountinfo.sh helper parses).
        mount_type = _exec_sh(
            name,
            "awk '$5 == \"/run/redis\" "
            "{for (i = 6; i <= NF; i++) if ($i == \"-\") print $(i + 1)}' "
            "/proc/self/mountinfo",
        )
        assert mount_type.returncode == 0, mount_type.combined
        assert mount_type.stdout.strip() == "tmpfs", mount_type.combined

        # Runtime file ownership and modes installed by the real entrypoint.
        ownership = _exec(
            name,
            "stat",
            "-c",
            "%n %U %G %a",
            f"{RUNTIME_DIR}/users.acl",
            f"{RUNTIME_DIR}/healthcheck.password",
            f"{RUNTIME_DIR}/credential-generation.json",
            f"{RUNTIME_DIR}/startup.json",
            f"{RUNTIME_DIR}/replay-integrity.log",
        )
        assert ownership.returncode == 0, ownership.combined
        assert ownership.stdout.splitlines() == [
            f"{RUNTIME_DIR}/users.acl root redis 440",
            f"{RUNTIME_DIR}/healthcheck.password root root 400",
            f"{RUNTIME_DIR}/credential-generation.json root redis 440",
            f"{RUNTIME_DIR}/startup.json root redis 640",
            f"{RUNTIME_DIR}/replay-integrity.log root redis 660",
        ], ownership.combined

        # No unsafe symlinks anywhere under the runtime directory.
        symlinks = _exec_sh(name, f"find {RUNTIME_DIR} -type l | wc -l")
        assert symlinks.returncode == 0, symlinks.combined
        assert symlinks.stdout.strip() == "0", symlinks.combined

        # The rendered ACL carries the exact canonical default-user rule from
        # the real template, and nothing else for the default user.
        template_default = next(
            line
            for line in ACL_TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
            if line.startswith("user default ")
        )
        rendered_default = _exec_sh(
            name, f"grep '^user default ' {RUNTIME_DIR}/users.acl"
        )
        assert rendered_default.returncode == 0, rendered_default.combined
        assert rendered_default.stdout.strip() == template_default

        # The startup marker is the exact production shape: generation only.
        expected_digest = hashlib.sha256(generation_bytes).hexdigest()
        marker = _exec(name, "cat", f"{RUNTIME_DIR}/startup.json")
        assert marker.returncode == 0, marker.combined
        assert marker.stdout.encode() == (
            f'{{"generation_sha256":"{expected_digest}"}}\n'
        ).encode()

        # Redis started through the entrypoint: PID 1 is unprivileged
        # redis-server from the pinned image, with enforced logging options.
        comm = _exec(name, "cat", "/proc/1/comm")
        assert comm.returncode == 0, comm.combined
        assert comm.stdout.strip() == "redis-server", comm.combined
        uid = _exec_sh(name, "awk '/^Uid:/ {print $2}' /proc/1/status")
        assert uid.returncode == 0, uid.combined
        assert uid.stdout.strip() not in ("", "0"), uid.combined
        version = _exec(name, "redis-server", "--version")
        assert version.returncode == 0, version.combined
        assert "7.4.12" in version.stdout, version.combined
        # Redis rewrites its own process title after startup (live
        # observation: "/usr/local/bin/redis-server *:6379"), so the
        # enforced options cannot be read back from /proc/1/cmdline. The
        # exact option vector was captured at the exec boundary before the
        # rewrite instead: re-read it post-release (same PID 1 lineage; the
        # wrapper execs exactly once) and prove the running server honors
        # it through the governed log and the installed ACL below.
        cmdline = _exec_sh(name, "tr '\\0' ' ' < /proc/1/cmdline")
        assert cmdline.returncode == 0, cmdline.combined
        assert cmdline.stdout.startswith("/usr/local/bin/redis-server"), (
            cmdline.combined
        )
        assert GATE_MOUNT_DST not in cmdline.stdout, cmdline.combined
        released_argv = _read_gate_argv(name)
        _assert_gate_argv_governed(released_argv, name)

        # The production healthcheck passes with production defaults and
        # seals the startup evidence with restrictive ownership.
        healthy = _run_production_healthcheck(name)
        assert healthy.returncode == 0, healthy.combined
        sealed = _exec(
            name,
            "stat",
            "-c",
            "%n %U %G %a",
            f"{RUNTIME_DIR}/replay-sealed.json",
            f"{RUNTIME_DIR}/replay-startup.evidence",
        )
        assert sealed.returncode == 0, sealed.combined
        assert sealed.stdout.splitlines() == [
            f"{RUNTIME_DIR}/replay-sealed.json root redis 640",
            f"{RUNTIME_DIR}/replay-startup.evidence root redis 640",
        ], sealed.combined

        # The keyless health identity authenticates for PING but cannot read
        # or write an application key.
        ping = _exec_sh(
            name,
            'IFS= read -r pw; REDISCLI_AUTH="$pw" REDISCLI_HISTFILE=/dev/null '
            "exec redis-cli --raw --user healthcheck ping",
            stdin_text=health_password + "\n",
        )
        assert ping.stdout.strip() == "PONG", ping.combined
        denied_write = _exec_sh(
            name,
            'IFS= read -r pw; REDISCLI_AUTH="$pw" REDISCLI_HISTFILE=/dev/null '
            "exec redis-cli --raw --user healthcheck SET app.key probe",
            stdin_text=health_password + "\n",
        )
        assert "NOPERM" in denied_write.combined, denied_write.combined
        denied_read = _exec_sh(
            name,
            'IFS= read -r pw; REDISCLI_AUTH="$pw" REDISCLI_HISTFILE=/dev/null '
            "exec redis-cli --raw --user healthcheck GET app.key",
            stdin_text=health_password + "\n",
        )
        assert "NOPERM" in denied_read.combined, denied_read.combined
        # redis-cli exits 0 when the server answers, even with an error
        # reply (observed live), so the denial evidence is the NOAUTH
        # reply itself: unauthenticated access is refused.
        unauthenticated = _exec(name, "redis-cli", "--raw", "ping")
        assert "NOAUTH" in unauthenticated.combined, unauthenticated.combined

        # Bounded companions through the same production healthcheck: a
        # mismatched generation and a missing marker both fail closed.
        tampered = _exec_sh(
            name,
            f"cp {RUNTIME_DIR}/startup.json /tmp/tampered-startup.json && "
            "sed -i 's/[0-9a-f]/0/g' /tmp/tampered-startup.json",
        )
        assert tampered.returncode == 0, tampered.combined
        mismatched = _run_production_healthcheck(
            name, extra_env="REDIS_STARTUP_MARKER=/tmp/tampered-startup.json"
        )
        assert mismatched.returncode != 0, mismatched.combined
        missing = _run_production_healthcheck(
            name, extra_env="REDIS_STARTUP_MARKER=/run/redis/does-not-exist.json"
        )
        assert missing.returncode != 0, missing.combined
    finally:
        for container in created:
            _run_docker(["docker", "rm", "-f", container], check=False, timeout=60)
        _run_docker(["docker", "volume", "prune", "--force"], check=False, timeout=60)
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.live_redis_rotation
def test_production_entrypoint_rejects_conflicting_log_options() -> None:
    """Conflicting caller log options fail closed with no Redis running."""
    temp_dir = Path(tempfile.mkdtemp(prefix="tiny-agent-lab-entrypoint-conflict-"))
    temp_dir.chmod(0o700)
    secret_dir = temp_dir / "synthetic-secrets"
    secret_dir.mkdir(mode=0o700)
    generate_secrets(secret_dir)
    created: list[str] = []
    try:
        for index, extra in enumerate(
            (("--logfile", "/tmp/other.log"), ("--loglevel", "debug"))
        ):
            name = f"tiny-agent-lab-entrypoint-conflict-{index}"
            result = _run_production_entrypoint(
                name, secret_dir, "redis-server", "--appendonly", "yes", *extra
            )
            assert result.returncode == 0, result.combined
            created.append(name)
            exited = _run_docker(["docker", "wait", name], timeout=60)
            assert exited.returncode == 0, exited.combined
            assert exited.stdout.strip() != "0", extra
            running = _run_docker(
                ["docker", "inspect", "-f", "{{.State.Running}}", name], timeout=30
            )
            assert running.returncode == 0, running.combined
            assert running.stdout.strip() == "false", running.combined
    finally:
        for container in created:
            _run_docker(["docker", "rm", "-f", container], check=False, timeout=60)
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_entrypoint_gate_wrapper_is_test_only_and_bounded(tmp_path: Path) -> None:
    """The barrier wrapper stays out of production wiring and forges nothing.

    Static proof: the wrapper records the exact argv, blocks on an explicit
    release signal, and execs the real binary with the original arguments,
    while production Compose, the production entrypoint, and the production
    healthcheck carry no reference to the gate. The synchronization files
    live only under the container-local signal directory and the disposable
    host mount, both removed with the test's temp dir and container.
    """
    assert GATE_SIGNAL_DIR.startswith("/tmp/")
    assert GATE_MOUNT_DST not in (
        REDIS_ENTRYPOINT_PATH.read_text(encoding="utf-8")
    )
    assert GATE_MOUNT_DST not in (
        REDIS_HEALTHCHECK_PATH.read_text(encoding="utf-8")
    )
    compose_text = (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert GATE_MOUNT_DST not in compose_text
    assert "entrypoint-gate" not in compose_text

    # The wrapper preserves the exact vector and execs the real binary.
    assert 'printf \'%s\\n\' "$@"' in GATE_WRAPPER_SCRIPT
    assert 'exec "$real_binary" "$@"' in GATE_WRAPPER_SCRIPT
    assert "command -v redis-server" in GATE_WRAPPER_SCRIPT

    # The wrapper forges no readiness, log, ACL, marker, seal, or evidence.
    for forged in (
        "Ready to accept connections",
        "replay-sealed.json",
        "replay-startup.evidence",
        "startup.json",
        "users.acl",
        "replay-integrity.log",
    ):
        assert forged not in GATE_WRAPPER_SCRIPT

    # The gate mount dir and wrapper file are 0755 (non-secret) so the
    # container's dropped unprivileged redis user can traverse and execute
    # them on native Linux bind mounts, while secret_dir remains 0700.
    wrapper = _write_gate_wrapper(tmp_path / "gate")
    assert wrapper.name == "redis-server"
    assert wrapper.read_text(encoding="utf-8") == GATE_WRAPPER_SCRIPT
    import stat as stat_module

    assert stat_module.S_IMODE(wrapper.parent.stat().st_mode) == 0o755
    assert stat_module.S_IMODE(wrapper.stat().st_mode) == 0o755
    assert ": > " in GATE_WRAPPER_SCRIPT or ': > "$reached"' in GATE_WRAPPER_SCRIPT


def test_harness_options_match_production_enforcement() -> None:
    """Harness-to-production drift in Redis options fails visibly here."""
    entrypoint = REDIS_ENTRYPOINT_PATH.read_text(encoding="utf-8")
    assert REDIS_IMAGE == "redis:7.4.12-alpine"
    assert REDIS_REPLAY_LOG_PATH == "/run/redis/replay-integrity.log"
    assert "replay_log=${REDIS_REPLAY_LOG:-$runtime_dir/replay-integrity.log}" in entrypoint
    assert REDIS_REPLAY_LOG_LEVEL == "notice"
    assert "replay_log_level=${REDIS_REPLAY_LOG_LEVEL:-notice}" in entrypoint
    assert (
        "Redis logfile must be the replay-integrity log" in entrypoint
        and "Redis loglevel must stay bounded" in entrypoint
    )
    for appendonly in (True, False):
        command = redis_server_command(appendonly=appendonly)
        assert f"--logfile {REDIS_REPLAY_LOG_PATH}" in command
        assert f"--loglevel {REDIS_REPLAY_LOG_LEVEL}" in command
        assert f"--aclfile {RUNTIME_DIR}/users.acl" in command
        assert "setpriv --reuid=redis --regid=redis --clear-groups" in command
    assert 'exec setpriv --reuid=redis --regid=redis --clear-groups -- "$@"' in entrypoint
    assert REDIS_REPLAY_LOG_PATH in redis_replay_log_prep()
    assert "root:redis" in redis_replay_log_prep()
