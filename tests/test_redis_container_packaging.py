"""Application image and authenticated-healthcheck packaging tests."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.container.redis_secret_entrypoint import (
    prepare_private_generation_metadata,
    prepare_private_secret,
    restrict_secret_source_directory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HEALTHCHECK = REPO_ROOT / "infrastructure" / "redis" / "redis-healthcheck.sh"
SYNTHETIC_SECRET = b"SYNTHETIC_redis_secret_for_tests_0123456789"
SYNTHETIC_GENERATION_ID = "1" * 32
SYNTHETIC_IDENTITY_GENERATIONS = {
    "orchestrator": "2" * 32,
    "worker-1": "3" * 32,
    "research-agent-1": "4" * 32,
    "research-agent-2": "5" * 32,
    "analysis-agent-1": "6" * 32,
    "healthcheck": "7" * 32,
}


def _run_healthcheck(
    password_file: Path,
    fake_bin: Path,
    *,
    response: bytes,
    exit_code: int = 0,
    log_content: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    evidence_dir = password_file.parent
    generation_file = evidence_dir / "packaging-generation.json"
    generation_file.write_bytes(b'{"generation_id":"packaging-health"}\n')
    digest = hashlib.sha256(generation_file.read_bytes()).hexdigest()
    marker_file = evidence_dir / "packaging-startup.json"
    marker_file.write_text(json.dumps({"generation_sha256": digest}))
    log_file = evidence_dir / "packaging-replay.log"
    if log_content is None:
        log_content = (
            b"1:M 01 Jan 00:00:00.000 oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo\n"
            b"1:M 01 Jan 00:00:01.000 * Ready to accept connections\n"
        )
    log_file.write_bytes(log_content)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "REDIS_HEALTHCHECK_PASSWORD_FILE": str(password_file),
            "REDIS_REPLAY_LOG": str(log_file),
            "REDIS_STARTUP_MARKER": str(marker_file),
            "REDIS_CREDENTIAL_GENERATION_FILE": str(generation_file),
            "REDIS_REPLAY_SEAL": str(evidence_dir / "packaging-sealed.json"),
            "REDIS_REPLAY_EVIDENCE": str(evidence_dir / "packaging-evidence.log"),
            "FAKE_REDIS_OUTPUT": response.decode("ascii"),
            "FAKE_REDIS_STATUS": str(exit_code),
        }
    )
    return subprocess.run(
        ["/bin/sh", str(HEALTHCHECK)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_app_entrypoint_secret_copy_is_private_and_rejects_bad_values(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-source"
    destination = tmp_path / "runtime" / "password"
    source.write_bytes(SYNTHETIC_SECRET)

    prepare_private_secret(source, destination, os.geteuid(), os.getegid())

    assert destination.read_bytes() == SYNTHETIC_SECRET
    assert stat.S_IMODE(destination.stat().st_mode) == 0o400
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700
    for invalid in (
        b"",
        b"short",
        SYNTHETIC_SECRET + b"\n",
        SYNTHETIC_SECRET[:-1] + b"!",
        SYNTHETIC_SECRET + b" injected",
    ):
        source.write_bytes(invalid)
        with pytest.raises(ValueError, match="malformed"):
            prepare_private_secret(source, destination, os.geteuid(), os.getegid())

    restrict_secret_source_directory(source.parent, os.geteuid(), os.getegid())
    assert stat.S_IMODE(source.parent.stat().st_mode) == 0o700


def test_app_entrypoint_generation_metadata_is_private_and_identity_scoped(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synthetic-generation.json"
    destination = tmp_path / "runtime" / "generation.json"
    source.write_text(
        json.dumps(
            {
                "generation_id": SYNTHETIC_GENERATION_ID,
                "identity_generations": SYNTHETIC_IDENTITY_GENERATIONS,
            }
        )
    )

    generation_id, identity_generation = prepare_private_generation_metadata(
        source,
        destination,
        "worker-1",
        os.geteuid(),
        os.getegid(),
    )

    assert generation_id == SYNTHETIC_GENERATION_ID
    assert identity_generation == SYNTHETIC_IDENTITY_GENERATIONS["worker-1"]
    assert destination.read_text() == source.read_text()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o400
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700

    for identity, invalid_metadata in (
        ("unknown", json.loads(source.read_text())),
        (
            "worker-1",
            {
                "generation_id": SYNTHETIC_GENERATION_ID,
                "identity_generations": {"worker-1": "3" * 32},
            },
        ),
        (
            "worker-1",
            {
                "generation_id": "not-a-generation",
                "identity_generations": SYNTHETIC_IDENTITY_GENERATIONS,
            },
        ),
        (
            "worker-1",
            {
                "generation_id": SYNTHETIC_GENERATION_ID,
                "identity_generations": SYNTHETIC_IDENTITY_GENERATIONS,
                "password": SYNTHETIC_SECRET.decode(),
            },
        ),
    ):
        source.write_text(json.dumps(invalid_metadata))
        with pytest.raises(ValueError, match="generation metadata is invalid"):
            prepare_private_generation_metadata(
                source,
                destination,
                identity,
                os.geteuid(),
                os.getegid(),
            )

    source.unlink()
    with pytest.raises(ValueError, match="missing or invalid"):
        prepare_private_generation_metadata(
            source,
            destination,
            "worker-1",
            os.geteuid(),
            os.getegid(),
        )


def test_application_images_drop_privileges_and_keep_original_commands() -> None:
    dockerfiles = {
        "services/orchestrator/Dockerfile": (
            'CMD ["uvicorn", "orchestrator.presentation.api:app", "--host", "0.0.0.0", "--port", "8080"]'
        ),
        "services/worker/Dockerfile": 'CMD ["python", "-m", "worker.main"]',
        "services/research_agent/Dockerfile": 'CMD ["python", "-m", "research_agent.main"]',
        "services/analysis_agent/Dockerfile": 'CMD ["python", "-m", "analysis_agent.main"]',
    }
    entrypoint_source = (
        REPO_ROOT / "scripts" / "container" / "redis_secret_entrypoint.py"
    ).read_text()

    assert "os.setgid(APP_GID)" in entrypoint_source
    assert "os.setuid(APP_UID)" in entrypoint_source
    assert "os.execvpe(command[0], command, environment)" in entrypoint_source
    assert "os.fchown(descriptor, uid, gid)" in entrypoint_source
    assert "os.fchmod(descriptor, 0o400)" in entrypoint_source
    assert "restrict_secret_source_directory(SOURCE_FILE.parent)" in entrypoint_source
    assert "REDIS_CREDENTIAL_GENERATION" in entrypoint_source

    for relative_path, original_cmd in dockerfiles.items():
        dockerfile = (REPO_ROOT / relative_path).read_text()
        assert "groupadd --system --gid 10001 app" in dockerfile
        assert "useradd --system --uid 10001 --gid app" in dockerfile
        assert (
            "COPY scripts/container/redis_secret_entrypoint.py "
            "/usr/local/bin/redis-secret-entrypoint.py"
        ) in dockerfile
        assert (
            'ENTRYPOINT ["python", "/usr/local/bin/redis-secret-entrypoint.py"]'
            in dockerfile
        )
        assert original_cmd in dockerfile


def test_redis_healthcheck_requires_exact_pong_and_never_uses_password_arguments(
    tmp_path: Path,
) -> None:
    password_file = tmp_path / "synthetic-healthcheck-password"
    password_file.write_bytes(SYNTHETIC_SECRET)
    fake_bin = tmp_path / "redis-cli-bin"
    fake_bin.mkdir()
    fake_cli = fake_bin / "redis-cli"
    fake_cli.write_text(
        "#!/bin/sh\n"
        '[ "$#" -eq 4 ] || exit 20\n'
        '[ "$1" = "--raw" ] && [ "$2" = "--user" ] && [ "$3" = "healthcheck" ] && [ "$4" = "ping" ] || exit 21\n'
        '[ -n "${REDISCLI_AUTH:-}" ] || exit 22\n'
        '[ "$REDISCLI_HISTFILE" = "/dev/null" ] || exit 23\n'
        'printf "%s" "$FAKE_REDIS_OUTPUT"\n'
        'exit "$FAKE_REDIS_STATUS"\n'
    )
    fake_cli.chmod(0o755)
    # chown is faked like in the replay-health tests: success by default so
    # the protocol logic is exercised, with production running as root where
    # the real root:redis chown succeeds. Ownership failure is covered by
    # test_seal_ownership_failure_fails_closed.
    fake_chown = fake_bin / "chown"
    fake_chown.write_text("#!/bin/sh\nexit 0\n")
    fake_chown.chmod(0o755)

    valid = _run_healthcheck(password_file, fake_bin, response=b"PONG\n")
    assert valid.returncode == 0
    assert valid.stdout == ""
    assert SYNTHETIC_SECRET.decode() not in valid.stderr

    for response, exit_code in (
        (b"WRONGPASS\n", 0),
        (b"NOAUTH\n", 0),
        (b"", 0),
        (b"PONG\n\n", 0),
        (b"PONG\n", 1),
    ):
        invalid = _run_healthcheck(
            password_file,
            fake_bin,
            response=response,
            exit_code=exit_code,
        )
        assert invalid.returncode != 0
        assert SYNTHETIC_SECRET.decode() not in invalid.stdout + invalid.stderr
