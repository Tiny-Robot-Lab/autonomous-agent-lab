"""Synthetic-secret tests for local Redis credential bootstrap."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.generate_redis_secrets import IDENTITIES, generate_secrets, main

REPO_ROOT = Path(__file__).resolve().parents[1]
ACL_TEMPLATE = REPO_ROOT / "infrastructure" / "redis" / "users.acl.template"
REDIS_ENTRYPOINT = REPO_ROOT / "infrastructure" / "redis" / "redis-acl-entrypoint.sh"
MOUNTINFO_HELPER = REPO_ROOT / "infrastructure" / "redis" / "redis-mountinfo.sh"
SYNTHETIC_SECRET = b"SYNTHETIC_redis_secret_for_tests_0123456789"


def _write_synthetic_secrets(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for identity in IDENTITIES:
        (directory / identity).write_bytes(SYNTHETIC_SECRET)


def _fake_runtime_commands(directory: Path) -> Path:
    fake_bin = directory / "fake-bin"
    fake_bin.mkdir()
    for executable in ("chown", "setpriv"):
        command = fake_bin / executable
        command.write_text("#!/bin/sh\nexit 0\n")
        command.chmod(0o755)
    return fake_bin


def _run_redis_entrypoint(
    secret_dir: Path,
    runtime_dir: Path,
    template: Path = ACL_TEMPLATE,
    *,
    fake_bin: Path,
    mountinfo_entry: str | None = None,
) -> subprocess.CompletedProcess[str]:
    entrypoint_directory = runtime_dir.parent / f"{runtime_dir.name}-entrypoint"
    entrypoint_directory.mkdir(parents=True, exist_ok=True)
    mountinfo = runtime_dir.parent / f"{runtime_dir.name}-mountinfo"
    if mountinfo_entry is None:
        mountinfo_entry = _mountinfo_entry(runtime_dir)
    mountinfo.write_text(mountinfo_entry)
    entrypoint_copy = entrypoint_directory / REDIS_ENTRYPOINT.name
    entrypoint_copy.write_text(
        REDIS_ENTRYPOINT.read_text().replace("/proc/self/mountinfo", str(mountinfo))
    )
    shutil.copyfile(MOUNTINFO_HELPER, entrypoint_directory / MOUNTINFO_HELPER.name)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "REDIS_ACL_SECRET_DIR": str(secret_dir),
            "REDIS_ACL_RUNTIME_DIR": str(runtime_dir),
            "REDIS_ACL_TEMPLATE": str(template),
        }
    )
    return subprocess.run(
        ["/bin/sh", str(entrypoint_copy), "redis-server", "--appendonly", "yes"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _mountinfo_entry(
    mountpoint: Path | str,
    *,
    filesystem: str = "tmpfs",
    root: str = "/",
) -> str:
    return f"36 25 0:32 {root} {mountpoint} rw,nosuid - {filesystem} {filesystem} rw\n"


def _mountinfo_check(mountinfo: Path, mountpoint: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/sh",
            "-c",
            '. "$1"; redis_mountinfo_is_tmpfs "$2" "$3"',
            "mountinfo-check",
            str(MOUNTINFO_HELPER),
            str(mountinfo),
            mountpoint,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_secret_generator_creates_private_allowlisted_files_without_newlines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_dir = tmp_path / "synthetic-redis-secrets"
    generated = generate_secrets(secret_dir)

    assert generated == IDENTITIES
    assert stat.S_IMODE(secret_dir.stat().st_mode) == 0o700
    for identity in IDENTITIES:
        path = secret_dir / identity
        value = path.read_bytes()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert len(value) >= 32
        assert all(character in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for character in value)
        assert b"\n" not in value and b"\r" not in value
    assert not list(secret_dir.glob(".*"))
    output = capsys.readouterr().out
    assert all((secret_dir / identity).read_text() not in output for identity in IDENTITIES)


def test_secret_generator_reports_identity_names_without_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_dir = tmp_path / "synthetic-redis-secrets"

    assert main([], secret_directory=secret_dir) == 0

    output = capsys.readouterr().out
    assert all(identity in output for identity in IDENTITIES)
    assert all((secret_dir / identity).read_text() not in output for identity in IDENTITIES)


def test_secret_generator_refuses_default_overwrite_and_bounds_explicit_rotation(
    tmp_path: Path,
) -> None:
    secret_dir = tmp_path / "synthetic-redis-secrets"
    generate_secrets(secret_dir)
    previous = {identity: (secret_dir / identity).read_bytes() for identity in IDENTITIES}

    with pytest.raises(FileExistsError):
        generate_secrets(secret_dir)
    assert {identity: (secret_dir / identity).read_bytes() for identity in IDENTITIES} == previous
    assert not (secret_dir / ".generation.lock").exists()

    assert generate_secrets(secret_dir, rotate="research-agent-2") == ("research-agent-2",)
    assert (secret_dir / "research-agent-2").read_bytes() != previous["research-agent-2"]
    for identity in set(IDENTITIES) - {"research-agent-2"}:
        assert (secret_dir / identity).read_bytes() == previous[identity]

    before_all_rotation = {
        identity: (secret_dir / identity).read_bytes() for identity in IDENTITIES
    }
    assert generate_secrets(secret_dir, rotate="all") == IDENTITIES
    assert all(
        (secret_dir / identity).read_bytes() != before_all_rotation[identity]
        for identity in IDENTITIES
    )
    assert sorted(path.name for path in secret_dir.iterdir()) == sorted(IDENTITIES)

    with pytest.raises(ValueError, match="Unknown Redis identity"):
        generate_secrets(secret_dir, rotate="unknown")


@pytest.mark.parametrize(
    "bad_value",
    [
        b"",
        b"short",
        SYNTHETIC_SECRET[:-1] + b"!",
        SYNTHETIC_SECRET + b" ",
        SYNTHETIC_SECRET + b"\n",
        SYNTHETIC_SECRET + b"\r",
        SYNTHETIC_SECRET + b"\nuser default on >injected",
    ],
)
def test_redis_entrypoint_rejects_malformed_secrets_without_rendering(
    tmp_path: Path,
    bad_value: bytes,
) -> None:
    secret_dir = tmp_path / "synthetic-source-secrets"
    _write_synthetic_secrets(secret_dir)
    (secret_dir / "worker-1").write_bytes(bad_value)
    runtime_dir = tmp_path / "runtime"
    fake_bin = _fake_runtime_commands(tmp_path)

    result = _run_redis_entrypoint(
        secret_dir,
        runtime_dir,
        fake_bin=fake_bin,
    )

    assert result.returncode != 0
    assert "worker-1" in result.stderr
    assert SYNTHETIC_SECRET.decode() not in result.stderr
    assert not (runtime_dir / "users.acl").exists()
    assert not list(runtime_dir.glob(".*")) if runtime_dir.exists() else True


def test_redis_entrypoint_rejects_missing_secrets_and_malformed_default_rules(
    tmp_path: Path,
) -> None:
    secret_dir = tmp_path / "synthetic-source-secrets"
    _write_synthetic_secrets(secret_dir)
    (secret_dir / "healthcheck").unlink()
    fake_bin = _fake_runtime_commands(tmp_path)

    missing_result = _run_redis_entrypoint(
        secret_dir,
        tmp_path / "missing-runtime",
        fake_bin=fake_bin,
    )
    assert missing_result.returncode != 0
    assert "healthcheck" in missing_result.stderr
    assert SYNTHETIC_SECRET.decode() not in missing_result.stderr

    _write_synthetic_secrets(secret_dir)
    malformed_template = tmp_path / "malformed.acl"
    malformed_template.write_text(
        ACL_TEMPLATE.read_text().replace(
            "user default off resetpass resetkeys resetchannels -@all",
            "user default on resetpass resetkeys resetchannels -@all",
        )
    )
    malformed_result = _run_redis_entrypoint(
        secret_dir,
        tmp_path / "malformed-runtime",
        malformed_template,
        fake_bin=fake_bin,
    )
    assert malformed_result.returncode != 0
    assert "canonical disabled default user" in malformed_result.stderr

    competing_template = tmp_path / "competing.acl"
    competing_template.write_text(
        ACL_TEMPLATE.read_text() + "user default on resetpass resetkeys resetchannels -@all\n"
    )
    competing_result = _run_redis_entrypoint(
        secret_dir,
        tmp_path / "competing-runtime",
        competing_template,
        fake_bin=fake_bin,
    )
    assert competing_result.returncode != 0
    assert "canonical disabled default user" in competing_result.stderr


def test_redis_entrypoint_rejects_non_tmpfs_and_missing_mount_before_writing(
    tmp_path: Path,
) -> None:
    secret_dir = tmp_path / "synthetic-source-secrets"
    _write_synthetic_secrets(secret_dir)
    fake_bin = _fake_runtime_commands(tmp_path)
    runtime_dir = tmp_path / "overlay-runtime"

    overlay_result = _run_redis_entrypoint(
        secret_dir,
        runtime_dir,
        fake_bin=fake_bin,
        mountinfo_entry=_mountinfo_entry(runtime_dir, filesystem="overlay"),
    )
    assert overlay_result.returncode != 0
    assert "not an exact tmpfs mount" in overlay_result.stderr
    assert not runtime_dir.exists()

    missing_result = _run_redis_entrypoint(
        secret_dir,
        tmp_path / "missing-runtime",
        fake_bin=fake_bin,
        mountinfo_entry="",
    )
    assert missing_result.returncode != 0
    assert "not an exact tmpfs mount" in missing_result.stderr
    assert not (tmp_path / "missing-runtime").exists()


def test_mountinfo_checker_accepts_only_exact_tmpfs_mounts(tmp_path: Path) -> None:
    mountinfo = tmp_path / "mountinfo"
    exact_tmpfs = _mountinfo_entry("/run/redis")
    mountinfo.write_text(exact_tmpfs)
    assert _mountinfo_check(mountinfo, "/run/redis").returncode == 0

    mountinfo.write_text(_mountinfo_entry("/run/redis", filesystem="overlay"))
    assert _mountinfo_check(mountinfo, "/run/redis").returncode != 0

    mountinfo.write_text(_mountinfo_entry("/run/redis", root="/backing"))
    assert _mountinfo_check(mountinfo, "/run/redis").returncode != 0

    mountinfo.write_text(
        _mountinfo_entry("/backing-source") + _mountinfo_entry("/run/redis")
    )
    assert _mountinfo_check(mountinfo, "/run/redis").returncode != 0

    mountinfo.write_text(_mountinfo_entry("/run/redis-shadow"))
    assert _mountinfo_check(mountinfo, "/run/redis").returncode != 0

    mountinfo.write_text("")
    assert _mountinfo_check(mountinfo, "/run/redis").returncode != 0

    assert _mountinfo_check(tmp_path / "missing-mountinfo", "/run/redis").returncode != 0


@pytest.mark.parametrize(
    ("template_text", "error"),
    [
        (
            ACL_TEMPLATE.read_text()
            + "user extra on >synthetic-plaintext resetkeys resetchannels -@all\n",
            "plaintext password rule",
        ),
        (
            ACL_TEMPLATE.read_text()
            + f"user extra on #{'0' * 64} resetkeys resetchannels -@all\n",
            "committed password hash",
        ),
        (
            ACL_TEMPLATE.read_text()
            + f"user extra on #{'F' * 64} resetkeys resetchannels -@all\n",
            "committed password hash",
        ),
        (
            ACL_TEMPLATE.read_text().replace("@@WORKER_1_SHA256@@", "@@UNKNOWN_SHA256@@"),
            "required identity placeholder once",
        ),
        (
            ACL_TEMPLATE.read_text() + "# @@WORKER_1_SHA256@@\n",
            "required identity placeholder once",
        ),
    ],
)
def test_redis_entrypoint_rejects_untrusted_acl_template_rules(
    tmp_path: Path,
    template_text: str,
    error: str,
) -> None:
    secret_dir = tmp_path / "synthetic-source-secrets"
    _write_synthetic_secrets(secret_dir)
    template = tmp_path / "invalid.acl"
    template.write_text(template_text)
    fake_bin = _fake_runtime_commands(tmp_path)

    result = _run_redis_entrypoint(
        secret_dir,
        tmp_path / "runtime",
        template,
        fake_bin=fake_bin,
    )

    assert result.returncode != 0
    assert error in result.stderr
    assert SYNTHETIC_SECRET.decode() not in result.stderr


@pytest.mark.parametrize(
    "forbidden_rule",
    [
        "~*",
        "+@all",
        "+@connection",
        "+@scripting",
        "+@admin",
        "+@dangerous",
    ],
)
def test_redis_entrypoint_rejects_broad_acl_rules(
    tmp_path: Path,
    forbidden_rule: str,
) -> None:
    secret_dir = tmp_path / "synthetic-source-secrets"
    _write_synthetic_secrets(secret_dir)
    template = tmp_path / "invalid.acl"
    template.write_text(
        ACL_TEMPLATE.read_text()
        + f"user extra on #@@EXTRA_SHA256@@ resetkeys resetchannels -@all {forbidden_rule}\n"
    )
    fake_bin = _fake_runtime_commands(tmp_path)

    result = _run_redis_entrypoint(
        secret_dir,
        tmp_path / "runtime",
        template,
        fake_bin=fake_bin,
    )

    assert result.returncode != 0
    assert "prohibited broad selector" in result.stderr
    assert SYNTHETIC_SECRET.decode() not in result.stdout + result.stderr


@pytest.mark.parametrize("identity", ["test-fixture", "fixture-admin", "TEST-ADMIN"])
def test_redis_entrypoint_rejects_test_or_admin_identities(
    tmp_path: Path,
    identity: str,
) -> None:
    secret_dir = tmp_path / "synthetic-source-secrets"
    _write_synthetic_secrets(secret_dir)
    template = tmp_path / "invalid.acl"
    template.write_text(
        ACL_TEMPLATE.read_text()
        + f"user {identity} on #@@EXTRA_SHA256@@ resetkeys resetchannels -@all +ping\n"
    )
    fake_bin = _fake_runtime_commands(tmp_path)

    result = _run_redis_entrypoint(
        secret_dir,
        tmp_path / "runtime",
        template,
        fake_bin=fake_bin,
    )

    assert result.returncode != 0
    assert "test or administrative identity" in result.stderr
    assert SYNTHETIC_SECRET.decode() not in result.stdout + result.stderr


def test_redis_entrypoint_rejects_duplicate_default_user_rule(tmp_path: Path) -> None:
    secret_dir = tmp_path / "synthetic-source-secrets"
    _write_synthetic_secrets(secret_dir)
    template = tmp_path / "duplicate-default.acl"
    template.write_text(ACL_TEMPLATE.read_text() * 2)
    fake_bin = _fake_runtime_commands(tmp_path)

    result = _run_redis_entrypoint(
        secret_dir,
        tmp_path / "runtime",
        template,
        fake_bin=fake_bin,
    )

    assert result.returncode != 0
    assert "canonical disabled default user" in result.stderr


def test_redis_entrypoint_renders_hashes_atomically_on_private_runtime_path(
    tmp_path: Path,
) -> None:
    secret_dir = tmp_path / "synthetic-source-secrets"
    _write_synthetic_secrets(secret_dir)
    runtime_dir = tmp_path / "runtime"
    fake_bin = _fake_runtime_commands(tmp_path)

    result = _run_redis_entrypoint(
        secret_dir,
        runtime_dir,
        fake_bin=fake_bin,
    )

    assert result.returncode == 0, result.stderr
    rendered = (runtime_dir / "users.acl").read_text()
    assert "user default off resetpass resetkeys resetchannels -@all" in rendered
    for identity in IDENTITIES:
        secret = (secret_dir / identity).read_bytes()
        assert secret.decode() not in rendered
        assert f"#{hashlib.sha256(secret).hexdigest()}" in rendered
    assert stat.S_IMODE((runtime_dir / "users.acl").stat().st_mode) == 0o440
    assert stat.S_IMODE((runtime_dir / "healthcheck.password").stat().st_mode) == 0o400
    assert (runtime_dir / "healthcheck.password").read_bytes() == SYNTHETIC_SECRET
    assert sorted(path.name for path in runtime_dir.iterdir()) == [
        "healthcheck.password",
        "users.acl",
    ]
    assert SYNTHETIC_SECRET.decode() not in result.stdout + result.stderr
