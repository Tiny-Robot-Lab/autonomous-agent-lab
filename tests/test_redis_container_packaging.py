"""Application image and authenticated-healthcheck packaging tests."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.container.redis_secret_entrypoint import (
    prepare_private_secret,
    restrict_secret_source_directory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HEALTHCHECK = REPO_ROOT / "infrastructure" / "redis" / "redis-healthcheck.sh"
SYNTHETIC_SECRET = b"SYNTHETIC_redis_secret_for_tests_0123456789"


def _run_healthcheck(
    password_file: Path,
    fake_bin: Path,
    *,
    response: bytes,
    exit_code: int = 0,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "REDIS_HEALTHCHECK_PASSWORD_FILE": str(password_file),
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
