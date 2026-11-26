"""Run bounded Docker integration probes for the Issue #229 Redis boundary."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REDIS_IMAGE = "redis:7.4.12-alpine"
APPLICATION_DOCKERFILES = (
    ("orchestrator", "services/orchestrator/Dockerfile"),
    ("worker", "services/worker/Dockerfile"),
    ("research-agent", "services/research_agent/Dockerfile"),
    ("analysis-agent", "services/analysis_agent/Dockerfile"),
)
REDIS_IDENTITIES = (
    "orchestrator",
    "worker-1",
    "research-agent-1",
    "research-agent-2",
    "analysis-agent-1",
    "healthcheck",
)
APPLICATION_PROBE = """
import hashlib
import os
import stat
from pathlib import Path

assert os.geteuid() == 10001, os.geteuid()
assert os.getegid() == 10001, os.getegid()
private_file = Path(os.environ["REDIS_PASSWORD_FILE"])
metadata = private_file.stat()
assert (metadata.st_uid, metadata.st_gid, stat.S_IMODE(metadata.st_mode)) == (
    10001,
    10001,
    0o400,
)
assert hashlib.sha256(private_file.read_bytes()).hexdigest() == os.environ[
    "EXPECTED_SECRET_SHA256"
]
try:
    Path("/run/secrets/redis_password").read_bytes()
except PermissionError:
    pass
else:
    raise SystemExit("source secret remained readable after privilege drop")
print("uid/gid=10001:10001 private-copy=0400 source=unreadable")
"""


class ValidationError(RuntimeError):
    """A bounded Docker validation command failed."""


def _run(
    arguments: list[str],
    *,
    timeout: int = 60,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(f"Timed out after {timeout}s: {' '.join(arguments[:4])}") from exc
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValidationError(
            f"Command exited {result.returncode}: {' '.join(arguments[:4])}"
            + (f"\n{detail}" if detail else "")
        )
    return result


def _write_synthetic_secrets(directory: Path) -> dict[str, bytes]:
    directory.mkdir(mode=0o700)
    secrets: dict[str, bytes] = {}
    for identity in REDIS_IDENTITIES:
        value = (
            f"SYNTHETIC_redis_secret_{identity.replace('-', '_')}_container_probe_0123456789"
        ).encode("ascii")
        path = directory / identity
        path.write_bytes(value)
        path.chmod(0o600)
        secrets[identity] = value
    return secrets


def _mount(source: Path, destination: str) -> list[str]:
    return [
        "--mount",
        f"type=bind,src={source},dst={destination},readonly",
    ]


def _assert_exact(result: subprocess.CompletedProcess[str], expected: str, label: str) -> None:
    if result.stdout != expected:
        raise ValidationError(f"{label} returned unexpected output: {result.stdout!r}")


def _build_application_images(token: str, images: list[str]) -> dict[str, str]:
    built: dict[str, str] = {}
    for service, dockerfile in APPLICATION_DOCKERFILES:
        image = f"tiny-agent-lab-issue229-{token}-{service}:smoke"
        _run(
            [
                "docker",
                "build",
                "--quiet",
                "--tag",
                image,
                "--file",
                dockerfile,
                ".",
            ],
            timeout=600,
        )
        images.append(image)
        built[service] = image
        print(f"Built {service} image.")
    return built


def _run_redis_probe(
    container_name: str,
    secrets: dict[str, bytes],
    secret_directory: Path,
) -> None:
    mounts: list[str] = []
    for identity in REDIS_IDENTITIES:
        mounts.extend(_mount(secret_directory / identity, f"/run/secrets/{identity}"))
    for relative_path, destination in (
        (
            "infrastructure/redis/redis-acl-entrypoint.sh",
            "/usr/local/bin/redis-acl-entrypoint.sh",
        ),
        (
            "infrastructure/redis/redis-healthcheck.sh",
            "/usr/local/bin/redis-healthcheck.sh",
        ),
        (
            "infrastructure/redis/redis-mountinfo.sh",
            "/usr/local/bin/redis-mountinfo.sh",
        ),
        (
            "infrastructure/redis/users.acl.template",
            "/usr/local/bin/users.acl.template",
        ),
    ):
        mounts.extend(_mount(REPO_ROOT / relative_path, destination))

    _run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            container_name,
            "--network",
            "none",
            "--tmpfs",
            "/run/redis:rw,noexec,nosuid,nodev,size=16m,mode=0750",
            *mounts,
            "--entrypoint",
            "/bin/sh",
            REDIS_IMAGE,
            "/usr/local/bin/redis-acl-entrypoint.sh",
            "redis-server",
            "--appendonly",
            "yes",
        ],
        timeout=120,
    )

    ready = False
    for _ in range(60):
        state = _run(
            ["docker", "inspect", "--format", "{{.State.Status}}", container_name],
            check=False,
        )
        if state.returncode or state.stdout.strip() != "running":
            break
        health = _run(
            [
                "docker",
                "exec",
                container_name,
                "/bin/sh",
                "/usr/local/bin/redis-healthcheck.sh",
            ],
            check=False,
        )
        if health.returncode == 0:
            ready = True
            break
        time.sleep(0.5)
    if not ready:
        logs = _run(["docker", "logs", container_name], check=False)
        raise ValidationError(
            "Redis did not become healthy. Container output:\n"
            + (logs.stdout + logs.stderr).strip()
        )

    _run(
        [
            "docker",
            "exec",
            container_name,
            "/bin/sh",
            "-c",
            "command -v setpriv >/dev/null && setpriv --version >/dev/null",
        ],
    )
    mount_check = _run(
        [
            "docker",
            "exec",
            container_name,
            "/bin/sh",
            "-c",
            (
                ". /usr/local/bin/redis-mountinfo.sh && "
                "redis_mountinfo_is_tmpfs /proc/self/mountinfo /run/redis"
            ),
        ],
        check=False,
    )
    if mount_check.returncode:
        raise ValidationError("Redis runtime directory is not verified as tmpfs.")

    pid_uid = _run(
        [
            "docker",
            "exec",
            container_name,
            "/bin/sh",
            "-c",
            (
                'test "$(cat /proc/1/comm)" = redis-server && '
                "awk '/^Uid:/ {print $2; exit}' /proc/1/status"
            ),
        ]
    ).stdout.strip()
    if not pid_uid.isdigit() or int(pid_uid) == 0:
        raise ValidationError(f"Redis PID 1 is not non-root: uid={pid_uid!r}")

    acl_metadata = _run(
        ["docker", "exec", container_name, "stat", "-c", "%U:%G %a", "/run/redis/users.acl"]
    ).stdout.strip()
    if acl_metadata != "root:redis 440":
        raise ValidationError(f"Unexpected rendered ACL ownership/mode: {acl_metadata!r}")

    direct_health_ping = _run(
        [
            "docker",
            "exec",
            container_name,
            "/bin/sh",
            "-c",
            (
                'REDISCLI_AUTH="$(cat /run/secrets/healthcheck)" '
                "redis-cli --raw --user healthcheck ping"
            ),
        ]
    )
    _assert_exact(direct_health_ping, "PONG\n", "Authenticated health PING")

    unauthenticated_ping = _run(
        ["docker", "exec", container_name, "redis-cli", "--raw", "ping"],
        check=False,
    )
    if "NOAUTH" not in unauthenticated_ping.stdout:
        raise ValidationError(
            "Unauthenticated PING was not denied with NOAUTH: "
            f"{unauthenticated_ping.stdout!r}"
        )

    write = _run(
        [
            "docker",
            "exec",
            container_name,
            "/bin/sh",
            "-c",
            (
                'REDISCLI_AUTH="$(cat /run/secrets/orchestrator)" '
                "redis-cli --raw --user orchestrator SET "
                "task.status:container-smoke synthetic-aof-write"
            ),
        ]
    )
    _assert_exact(write, "OK\n", "Authenticated AOF write")
    aof = _run(
        [
            "docker",
            "exec",
            container_name,
            "/bin/sh",
            "-c",
            (
                "for attempt in 1 2 3 4 5 6 7 8 9 10; do "
                "if find /data -type f -name '*.aof' -size +0c | grep -q .; then exit 0; fi; "
                "sleep 0.2; done; exit 1"
            ),
        ],
        check=False,
    )
    if aof.returncode:
        raise ValidationError(
            "Pinned Redis failed to create non-empty AOF data under /data "
            "without ownership repair."
        )

    print(
        "Redis runtime passed: setpriv executed; "
        f"PID 1 UID={pid_uid}; ACL={acl_metadata}; tmpfs verified; "
        "health PONG/unauthenticated NOAUTH; AOF write persisted."
    )
    del secrets


def _run_application_probes(
    images: dict[str, str],
    container_names: list[str],
    secret_file: Path,
    secret: bytes,
    token: str,
) -> None:
    expected_hash = hashlib.sha256(secret).hexdigest()
    for service, image in images.items():
        container_name = f"tiny-agent-lab-app-{token}-{service}"
        container_names.append(container_name)
        result = _run(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "--network",
                "none",
                *_mount(secret_file, "/run/secrets/redis_password"),
                "--env",
                f"EXPECTED_SECRET_SHA256={expected_hash}",
                image,
                "python",
                "-c",
                APPLICATION_PROBE,
            ],
            timeout=45,
        )
        if result.stdout.strip() != "uid/gid=10001:10001 private-copy=0400 source=unreadable":
            raise ValidationError(f"{service} image probe returned unexpected output.")
        print(f"{service} runtime passed: {result.stdout.strip()}")


def _cleanup(container_names: list[str], images: list[str]) -> None:
    errors: list[str] = []
    for name in container_names:
        inspected = _run(
            ["docker", "container", "inspect", name],
            check=False,
        )
        if inspected.returncode == 0:
            removed = _run(
                ["docker", "container", "rm", "--force", "--volumes", name],
                check=False,
            )
            if removed.returncode:
                errors.append(f"Could not remove container {name}.")
    for image in images:
        removed = _run(["docker", "image", "rm", image], check=False)
        if removed.returncode:
            errors.append(f"Could not remove temporary image {image}.")
    if errors:
        raise ValidationError("Docker cleanup failed: " + " ".join(errors))
    print("Docker cleanup passed: disposable containers and temporary app images removed.")


def main() -> int:
    token = uuid.uuid4().hex[:10]
    container_names = [
        f"tiny-agent-lab-redis-acl-{token}",
        *(f"tiny-agent-lab-app-{token}-{service}" for service, _ in APPLICATION_DOCKERFILES),
    ]
    images: list[str] = []
    try:
        _run(["docker", "info"], timeout=30)
        with tempfile.TemporaryDirectory(prefix="synthetic-redis-acl-container-probe-") as directory:
            temporary = Path(directory)
            secret_directory = temporary / "secrets"
            secrets = _write_synthetic_secrets(secret_directory)
            application_secret = b"SYNTHETIC_application_secret_for_container_probe_0123456789"
            application_secret_file = temporary / "application-secret"
            application_secret_file.write_bytes(application_secret)
            application_secret_file.chmod(0o600)

            app_images = _build_application_images(token, images)
            _run_redis_probe(container_names[0], secrets, secret_directory)
            _run_application_probes(
                app_images,
                container_names[1:],
                application_secret_file,
                application_secret,
                token,
            )
        _cleanup(container_names, images)
    except (OSError, ValidationError) as exc:
        try:
            _cleanup(container_names, images)
        except ValidationError as cleanup_error:
            print(cleanup_error)
        print(f"Docker container validation failed: {exc}")
        return 1

    print("Bounded Docker Redis ACL and application secret probes passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
