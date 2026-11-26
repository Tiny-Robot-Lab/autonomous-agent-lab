"""Synthetic-secret tests for local Redis credential bootstrap."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.generate_redis_secrets import (
    IDENTITIES,
    discard_rollback_generation,
    generate_secrets,
    get_generation_info,
    main,
    read_generation_secret,
    rollback_generation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ACL_TEMPLATE = REPO_ROOT / "infrastructure" / "redis" / "users.acl.template"
REDIS_ENTRYPOINT = REPO_ROOT / "infrastructure" / "redis" / "redis-acl-entrypoint.sh"
MOUNTINFO_HELPER = REPO_ROOT / "infrastructure" / "redis" / "redis-mountinfo.sh"
SYNTHETIC_SECRET = b"SYNTHETIC_redis_secret_for_tests_0123456789"


def _write_synthetic_secrets(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for identity in IDENTITIES:
        (directory / identity).write_bytes(SYNTHETIC_SECRET)
    (directory / "redis_generation").write_text(
        json.dumps(
            {
                "generation_id": "1" * 32,
                "identity_generations": {
                    identity: str(index) * 32
                    for index, identity in enumerate(IDENTITIES, start=1)
                },
            }
        )
    )


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
    assert (secret_dir / "current").is_symlink()
    assert stat.S_IMODE((secret_dir / "generations").stat().st_mode) == 0o700
    generation = get_generation_info(secret_dir)
    assert set(generation.identity_generations) == set(IDENTITIES)
    for identity in IDENTITIES:
        path = secret_dir / "current" / identity
        value = path.read_bytes()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert len(value) >= 32
        assert all(character in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for character in value)
        assert b"\n" not in value and b"\r" not in value
    assert stat.S_IMODE((secret_dir / ".generation.lock").stat().st_mode) == 0o600
    output = capsys.readouterr().out
    assert all((secret_dir / "current" / identity).read_text() not in output for identity in IDENTITIES)


def test_secret_generator_reports_identity_names_without_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_dir = tmp_path / "synthetic-redis-secrets"

    assert main([], secret_directory=secret_dir) == 0

    created = capsys.readouterr()
    created_output = created.out
    assert all(f"Created Redis credential for {identity}" in created_output for identity in IDENTITIES)
    assert get_generation_info(secret_dir).generation_id in created_output
    initial_secrets = {
        identity: (secret_dir / "current" / identity).read_text() for identity in IDENTITIES
    }
    assert all(secret not in created_output and secret not in created.err for secret in initial_secrets.values())

    assert main(["--rotate", "research-agent-2"], secret_directory=secret_dir) == 0

    rotated_one = capsys.readouterr()
    assert "Rotated Redis credential for research-agent-2" in rotated_one.out
    assert all(
        identity not in rotated_one.out
        for identity in IDENTITIES
        if identity != "research-agent-2"
    )
    assert get_generation_info(secret_dir).generation_id in rotated_one.out
    after_one_secrets = {
        identity: (secret_dir / "current" / identity).read_text() for identity in IDENTITIES
    }
    all_seen_secrets = set(initial_secrets.values()) | set(after_one_secrets.values())
    assert all(secret not in rotated_one.out and secret not in rotated_one.err for secret in all_seen_secrets)

    assert main(["--rotate", "all"], secret_directory=secret_dir) == 0

    rotated_all = capsys.readouterr()
    assert all(f"Rotated Redis credential for {identity}" in rotated_all.out for identity in IDENTITIES)
    assert get_generation_info(secret_dir).generation_id in rotated_all.out
    after_all_secrets = {
        identity: (secret_dir / "current" / identity).read_text() for identity in IDENTITIES
    }
    all_seen_secrets |= set(after_all_secrets.values())
    assert all(secret not in rotated_all.out and secret not in rotated_all.err for secret in all_seen_secrets)


def test_secret_generator_refuses_default_overwrite_and_bounds_explicit_rotation(
    tmp_path: Path,
) -> None:
    secret_dir = tmp_path / "synthetic-redis-secrets"
    generate_secrets(secret_dir)
    previous_info = get_generation_info(secret_dir)
    previous = {
        identity: read_generation_secret(secret_dir, identity) for identity in IDENTITIES
    }

    with pytest.raises(FileExistsError):
        generate_secrets(secret_dir)
    assert {
        identity: read_generation_secret(secret_dir, identity) for identity in IDENTITIES
    } == previous

    assert generate_secrets(secret_dir, rotate="research-agent-2") == ("research-agent-2",)
    after_one = get_generation_info(secret_dir)
    assert read_generation_secret(secret_dir, "research-agent-2") != previous["research-agent-2"]
    for identity in set(IDENTITIES) - {"research-agent-2"}:
        assert read_generation_secret(secret_dir, identity) == previous[identity]
        assert (
            after_one.identity_generations[identity]
            == previous_info.identity_generations[identity]
        )
    assert (secret_dir / "previous").readlink().as_posix() == (
        f"generations/{previous_info.generation_id}"
    )

    before_all_rotation = {
        identity: read_generation_secret(secret_dir, identity) for identity in IDENTITIES
    }
    assert generate_secrets(secret_dir, rotate="all") == IDENTITIES
    assert all(
        read_generation_secret(secret_dir, identity) != before_all_rotation[identity]
        for identity in IDENTITIES
    )
    assert len(list((secret_dir / "generations").iterdir())) == 2

    with pytest.raises(ValueError, match="Unknown Redis identity"):
        generate_secrets(secret_dir, rotate="unknown")


def test_secret_generation_can_rollback_and_discard_without_secret_backup_files(
    tmp_path: Path,
) -> None:
    secret_dir = tmp_path / "synthetic-redis-secrets"
    generate_secrets(secret_dir)
    initial = get_generation_info(secret_dir)
    initial_values = {
        identity: read_generation_secret(secret_dir, identity) for identity in IDENTITIES
    }

    generate_secrets(secret_dir, rotate="worker-1")
    rotated = get_generation_info(secret_dir)
    assert read_generation_secret(secret_dir, "worker-1") != initial_values["worker-1"]

    assert rollback_generation(secret_dir) == initial.generation_id
    restored = get_generation_info(secret_dir)
    assert restored.generation_id == initial.generation_id
    assert {
        identity: read_generation_secret(secret_dir, identity) for identity in IDENTITIES
    } == initial_values
    assert (secret_dir / "previous").readlink().as_posix() == (
        f"generations/{rotated.generation_id}"
    )

    assert discard_rollback_generation(secret_dir) == rotated.generation_id
    assert not (secret_dir / "previous").exists()
    assert sorted(path.name for path in (secret_dir / "generations").iterdir()) == [
        initial.generation_id
    ]


def test_secret_generator_recovers_an_interrupted_pointer_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.generate_redis_secrets as generator

    secret_dir = tmp_path / "synthetic-redis-secrets"
    generate_secrets(secret_dir)
    initial = get_generation_info(secret_dir)
    original_set_pointer = generator._atomic_set_pointer

    def fail_after_current_switch(directory: Path, name: str, generation_id: str | None) -> None:
        if name == "previous":
            raise OSError("synthetic pointer-switch interruption")
        original_set_pointer(directory, name, generation_id)

    monkeypatch.setattr(generator, "_atomic_set_pointer", fail_after_current_switch)
    with pytest.raises(OSError, match="synthetic pointer-switch interruption"):
        generate_secrets(secret_dir, rotate="worker-1")

    monkeypatch.setattr(generator, "_atomic_set_pointer", original_set_pointer)
    assert (secret_dir / ".generation-transaction.json").exists()
    assert rollback_generation(secret_dir) == initial.generation_id
    assert not (secret_dir / ".generation-transaction.json").exists()
    assert get_generation_info(secret_dir).generation_id == initial.generation_id


def test_secret_generator_adopts_legacy_files_without_overwriting_them_first(
    tmp_path: Path,
) -> None:
    secret_dir = tmp_path / "synthetic-redis-secrets"
    secret_dir.mkdir()
    for identity in IDENTITIES:
        (secret_dir / identity).write_bytes(SYNTHETIC_SECRET)

    assert generate_secrets(secret_dir) == IDENTITIES
    assert all(
        read_generation_secret(secret_dir, identity) == SYNTHETIC_SECRET
        for identity in IDENTITIES
    )
    assert all(not (secret_dir / identity).exists() for identity in IDENTITIES)


def test_secret_generator_rejects_partial_legacy_layout(tmp_path: Path) -> None:
    secret_dir = tmp_path / "synthetic-redis-secrets"
    secret_dir.mkdir()
    (secret_dir / "orchestrator").write_bytes(SYNTHETIC_SECRET)

    with pytest.raises(ValueError, match="Incomplete legacy Redis secrets"):
        generate_secrets(secret_dir)
    assert (secret_dir / "orchestrator").read_bytes() == SYNTHETIC_SECRET
    assert not (secret_dir / "current").exists()


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
    assert stat.S_IMODE(
        (runtime_dir / "credential-generation.json").stat().st_mode
    ) == 0o440
    assert json.loads(
        (runtime_dir / "credential-generation.json").read_text()
    )["generation_id"] == "1" * 32
    assert sorted(path.name for path in runtime_dir.iterdir()) == [
        "credential-generation.json",
        "healthcheck.password",
        "replay-integrity.log",
        "startup.json",
        "users.acl",
    ]
    assert (runtime_dir / "replay-integrity.log").read_bytes() == b""
    marker = json.loads((runtime_dir / "startup.json").read_text())
    assert marker["generation_sha256"] == hashlib.sha256(
        (runtime_dir / "credential-generation.json").read_bytes()
    ).hexdigest()
    assert SYNTHETIC_SECRET.decode() not in result.stdout + result.stderr
