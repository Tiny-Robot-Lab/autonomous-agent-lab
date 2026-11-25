"""Disposable Docker harness for live Issue #229 Redis ACL verification."""

from __future__ import annotations

import hashlib
import secrets
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REDIS_IMAGE: Final[str] = "redis:7.4.12-alpine"
ACL_TEMPLATE_PATH: Final[Path] = REPO_ROOT / "infrastructure" / "redis" / "users.acl.template"
REDIS_ENTRYPOINT_PATH: Final[Path] = (
    REPO_ROOT / "infrastructure" / "redis" / "redis-acl-entrypoint.sh"
)
REDIS_HEALTHCHECK_PATH: Final[Path] = (
    REPO_ROOT / "infrastructure" / "redis" / "redis-healthcheck.sh"
)
REDIS_MOUNTINFO_PATH: Final[Path] = (
    REPO_ROOT / "infrastructure" / "redis" / "redis-mountinfo.sh"
)
PRODUCTION_IDENTITIES: Final[tuple[str, ...]] = (
    "orchestrator",
    "worker-1",
    "research-agent-1",
    "research-agent-2",
    "analysis-agent-1",
    "healthcheck",
)
RUNTIME_IDENTITIES: Final[tuple[str, ...]] = (
    "worker-1",
    "research-agent-1",
    "research-agent-2",
    "analysis-agent-1",
)
SERVICE_DOCKERFILES: Final[dict[str, str]] = {
    "orchestrator": "services/orchestrator/Dockerfile",
    "worker": "services/worker/Dockerfile",
    "research-agent": "services/research_agent/Dockerfile",
    "analysis-agent": "services/analysis_agent/Dockerfile",
}
TEST_ADMIN_IDENTITY: Final[str] = "fixture-admin"


class HarnessError(RuntimeError):
    """Raised when the disposable Docker Redis ACL harness fails."""


@dataclass(frozen=True)
class CliResult:
    """Captured output and exit code from a Redis command execution."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        parts = [p for p in (self.stdout.strip(), self.stderr.strip()) if p]
        return "\n".join(parts)


@dataclass
class LiveRedisAclCluster:
    """Isolated, disposable Docker Redis + built service-image test environment.

    - Uses pinned redis:7.4.12-alpine with /run/redis tmpfs (no persistent volumes).
    - Renders the real users.acl.template via the real redis-acl-entrypoint.sh.
    - Adds an ephemeral fixture-admin user strictly inside the disposable container ACL.
    - Publishes zero host ports; all traffic stays on a disposable --internal bridge network.
    """

    token: str
    temp_dir: Path
    secret_dir: Path
    network_name: str
    redis_container: str
    secrets_map: dict[str, str] = field(repr=False)
    admin_secret: str = field(repr=False)
    service_images: dict[str, str] = field(default_factory=dict)
    created_containers: list[str] = field(default_factory=list)
    _closed: bool = False

    def __repr__(self) -> str:
        return (
            "LiveRedisAclCluster("
            f"token={self.token!r}, "
            f"network_name={self.network_name!r}, "
            f"redis_container={self.redis_container!r}, "
            f"services={sorted(self.service_images)!r})"
        )

    @classmethod
    def start(cls) -> LiveRedisAclCluster:
        token = uuid.uuid4().hex[:10]
        temp_dir = Path(tempfile.mkdtemp(prefix=f"tiny-agent-lab-live-acl-{token}-"))
        temp_dir.chmod(0o700)
        secret_dir = temp_dir / "synthetic-secrets"
        secret_dir.mkdir(mode=0o700)

        secrets_map: dict[str, str] = {}
        for identity in PRODUCTION_IDENTITIES:
            secret_value = (
                f"SYNTH_{identity.replace('-', '_')}_{secrets.token_urlsafe(28)}"
            )
            secret_path = secret_dir / identity
            secret_path.write_text(secret_value, encoding="ascii")
            secret_path.chmod(0o600)
            secrets_map[identity] = secret_value

        admin_secret = f"FIXTURE_ADMIN_{secrets.token_urlsafe(32)}"
        admin_sha256 = hashlib.sha256(admin_secret.encode("ascii")).hexdigest()
        admin_hash_file = temp_dir / "fixture-admin.sha256"
        admin_hash_file.write_text(admin_sha256, encoding="ascii")
        admin_hash_file.chmod(0o600)

        cluster = cls(
            token=token,
            temp_dir=temp_dir,
            secret_dir=secret_dir,
            network_name=f"tiny-agent-lab-acl-net-{token}",
            redis_container=f"tiny-agent-lab-acl-redis-{token}",
            secrets_map=secrets_map,
            admin_secret=admin_secret,
        )

        try:
            _run_docker(["docker", "info"], timeout=30)
            _run_docker(
                ["docker", "network", "create", "--internal", cluster.network_name],
                timeout=30,
            )

            for service, dockerfile in SERVICE_DOCKERFILES.items():
                image_tag = f"tiny-agent-lab-acl-{token}-{service}:test"
                _run_docker(
                    [
                        "docker",
                        "build",
                        "--quiet",
                        "--tag",
                        image_tag,
                        "--file",
                        dockerfile,
                        ".",
                    ],
                    timeout=600,
                )
                cluster.service_images[service] = image_tag

            cluster._start_redis_container(admin_hash_file)
            return cluster
        except BaseException:
            cluster.cleanup()
            raise

    def _start_redis_container(self, admin_hash_file: Path) -> None:
        mounts: list[str] = []
        for identity in PRODUCTION_IDENTITIES:
            mounts.extend(
                [
                    "--mount",
                    f"type=bind,src={self.secret_dir / identity},dst=/run/secrets/{identity},readonly",
                ]
            )
        for src_path, dst_path in (
            (REDIS_ENTRYPOINT_PATH, "/usr/local/bin/redis-acl-entrypoint.sh"),
            (REDIS_HEALTHCHECK_PATH, "/usr/local/bin/redis-healthcheck.sh"),
            (REDIS_MOUNTINFO_PATH, "/usr/local/bin/redis-mountinfo.sh"),
            (ACL_TEMPLATE_PATH, "/usr/local/bin/users.acl.template"),
            (admin_hash_file, "/run/fixture-admin.sha256"),
        ):
            mounts.extend(["--mount", f"type=bind,src={src_path},dst={dst_path},readonly"])

        # Disposable harness only:
        # 1. Uses a temporary redis-server stub in PATH so the real redis-acl-entrypoint.sh
        #    performs full tmpfs, template, default-user, and SHA-256 rendering cleanly.
        # 2. Removes the stub, appends only the ephemeral fixture-admin ACL rule,
        #    re-applies root:redis 0440 on /run/redis/users.acl, and execs the real
        #    non-root redis-server via setpriv.
        bootstrap_script = (
            "set -eu\n"
            "mkdir -p /tmp/render-stub\n"
            "printf '#!/bin/sh\\nexit 0\\n' > /tmp/render-stub/redis-server\n"
            "chmod 0755 /tmp/render-stub/redis-server\n"
            'PATH="/tmp/render-stub:$PATH" /bin/sh /usr/local/bin/redis-acl-entrypoint.sh '
            "redis-server --appendonly no --save ''\n"
            "rm -rf /tmp/render-stub\n"
            "admin_hash=$(cat /run/fixture-admin.sha256)\n"
            "chmod 0640 /run/redis/users.acl\n"
            'printf "user %s on #%s resetkeys resetchannels +@all ~* &*\\n" '
            f'"{TEST_ADMIN_IDENTITY}" "$admin_hash" >> /run/redis/users.acl\n'
            "chown root:redis /run/redis/users.acl\n"
            "chmod 0440 /run/redis/users.acl\n"
            "exec setpriv --reuid=redis --regid=redis --clear-groups -- "
            'redis-server --appendonly no --save "" --aclfile /run/redis/users.acl\n'
        )

        self.created_containers.append(self.redis_container)
        _run_docker(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                self.redis_container,
                "--network",
                self.network_name,
                "--network-alias",
                "redis",
                "--tmpfs",
                "/run/redis:rw,noexec,nosuid,nodev,size=16m,mode=0750",
                *mounts,
                "--entrypoint",
                "/bin/sh",
                REDIS_IMAGE,
                "-c",
                bootstrap_script,
            ],
            timeout=120,
        )

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            state = _run_docker(
                ["docker", "inspect", "--format", "{{.State.Status}}", self.redis_container],
                check=False,
            )
            if state.returncode or state.stdout.strip() != "running":
                logs = _run_docker(["docker", "logs", self.redis_container], check=False)
                raise HarnessError(
                    "Disposable Redis ACL container exited early:\n" + logs.combined
                )
            hc = _run_docker(
                [
                    "docker",
                    "exec",
                    self.redis_container,
                    "/bin/sh",
                    "/usr/local/bin/redis-healthcheck.sh",
                ],
                check=False,
            )
            if hc.returncode == 0:
                return
            time.sleep(0.25)

        logs = _run_docker(["docker", "logs", self.redis_container], check=False)
        raise HarnessError(
            "Disposable Redis ACL container failed to become healthy:\n" + logs.combined
        )

    def redis_cli(
        self,
        username: str | None,
        *args: str,
        password: str | None = None,
        use_mounted_secret: bool = False,
    ) -> CliResult:
        """Execute redis-cli inside the disposable Redis container without placing secrets in argv."""
        if username is None and password is None and not use_mounted_secret:
            proc = _run_docker(
                [
                    "docker",
                    "exec",
                    self.redis_container,
                    "redis-cli",
                    "--raw",
                    *args,
                ],
                check=False,
            )
            return CliResult(proc.returncode, proc.stdout, proc.stderr)

        if use_mounted_secret:
            assert username is not None
            return self._exec_cli_from_mounted_secret(username, *args)

        resolved_pw = password if password is not None else self._lookup_secret(username or "")
        user_flags = ["--user", username] if username is not None else []
        proc = _run_docker(
            [
                "docker",
                "exec",
                "-i",
                self.redis_container,
                "/bin/sh",
                "-c",
                (
                    'IFS= read -r pw; REDISCLI_AUTH="$pw" REDISCLI_HISTFILE=/dev/null '
                    'exec redis-cli --raw "$@"'
                ),
                "_",
                *user_flags,
                *args,
            ],
            stdin_text=resolved_pw + "\n",
            check=False,
        )
        return CliResult(proc.returncode, proc.stdout, proc.stderr)

    def _exec_cli_from_mounted_secret(self, username: str, *args: str) -> CliResult:
        proc = _run_docker(
            [
                "docker",
                "exec",
                self.redis_container,
                "/bin/sh",
                "-c",
                (
                    'REDISCLI_AUTH=$(cat "/run/secrets/$1"); export REDISCLI_AUTH; '
                    'REDISCLI_HISTFILE=/dev/null; user=$1; shift; '
                    'exec redis-cli --raw --user "$user" "$@"'
                ),
                "_",
                username,
                *args,
            ],
            check=False,
        )
        return CliResult(proc.returncode, proc.stdout, proc.stderr)

    def admin_cli(self, *args: str) -> CliResult:
        """Run a fixture setup/inspection command as the ephemeral fixture-admin identity."""
        return self.redis_cli(TEST_ADMIN_IDENTITY, *args, password=self.admin_secret)

    def reset_db(self) -> None:
        """Flush database 0 in the disposable container via fixture-admin."""
        res = self.admin_cli("FLUSHDB")
        if res.returncode != 0 or res.stdout.strip() != "OK":
            raise HarnessError(f"fixture-admin FLUSHDB failed: {res.combined}")

    def run_healthcheck(self, *, override_secret: str | None = None) -> CliResult:
        """Run infrastructure/redis/redis-healthcheck.sh inside the disposable Redis container."""
        if override_secret is None:
            proc = _run_docker(
                [
                    "docker",
                    "exec",
                    self.redis_container,
                    "/bin/sh",
                    "/usr/local/bin/redis-healthcheck.sh",
                ],
                check=False,
            )
            return CliResult(proc.returncode, proc.stdout, proc.stderr)

        proc = _run_docker(
            [
                "docker",
                "exec",
                "-i",
                self.redis_container,
                "/bin/sh",
                "-c",
                (
                    'tmp=$(mktemp); trap "rm -f \\"$tmp\\"" EXIT; '
                    'IFS= read -r val; printf "%s" "$val" > "$tmp"; chmod 0600 "$tmp"; '
                    'REDIS_HEALTHCHECK_PASSWORD_FILE="$tmp" /bin/sh /usr/local/bin/redis-healthcheck.sh'
                ),
            ],
            stdin_text=override_secret + "\n",
            check=False,
        )
        return CliResult(proc.returncode, proc.stdout, proc.stderr)

    def run_service_python(
        self,
        service: str,
        identity: str,
        script: str,
        *,
        override_secret: str | None = None,
    ) -> CliResult:
        """Run a disposable non-root service container executing Python against live Redis."""
        if service not in self.service_images:
            raise KeyError(f"Unknown service image: {service}")

        secret_file = self.secret_dir / identity
        if override_secret is not None:
            secret_file = self.temp_dir / f"override-{uuid.uuid4().hex[:8]}"
            secret_file.write_text(override_secret, encoding="ascii")
            secret_file.chmod(0o600)

        proc = _run_docker(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--network",
                self.network_name,
                "--mount",
                f"type=bind,src={secret_file},dst=/run/secrets/redis_password,readonly",
                "--env",
                "REDIS_URL=redis://redis:6379/0",
                "--env",
                "REDIS_AUTH_MODE=required",
                "--env",
                f"REDIS_USERNAME={identity}",
                "--env",
                "REDIS_PASSWORD_FILE=/run/redis-auth/password",
                self.service_images[service],
                "python",
                "-",
            ],
            stdin_text=script,
            check=False,
            timeout=90,
        )
        return CliResult(proc.returncode, proc.stdout, proc.stderr)

    def _lookup_secret(self, identity: str) -> str:
        if identity == TEST_ADMIN_IDENTITY:
            return self.admin_secret
        if identity not in self.secrets_map:
            raise KeyError(f"Unknown identity: {identity}")
        return self.secrets_map[identity]

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []

        for container_name in reversed(self.created_containers):
            inspected = _run_docker(["docker", "container", "inspect", container_name], check=False)
            if inspected.returncode == 0:
                removed = _run_docker(
                    ["docker", "container", "rm", "--force", "--volumes", container_name],
                    check=False,
                )
                if removed.returncode != 0:
                    errors.append(f"container {container_name}: {removed.stderr.strip()}")

        net_inspect = _run_docker(["docker", "network", "inspect", self.network_name], check=False)
        if net_inspect.returncode == 0:
            net_rm = _run_docker(["docker", "network", "rm", self.network_name], check=False)
            if net_rm.returncode != 0:
                errors.append(f"network {self.network_name}: {net_rm.stderr.strip()}")

        for image in self.service_images.values():
            img_inspect = _run_docker(["docker", "image", "inspect", image], check=False)
            if img_inspect.returncode == 0:
                img_rm = _run_docker(["docker", "image", "rm", image], check=False)
                if img_rm.returncode != 0:
                    errors.append(f"image {image}: {img_rm.stderr.strip()}")

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.temp_dir.exists():
            errors.append(f"temp_dir {self.temp_dir} still exists")

        if errors:
            raise HarnessError("Disposable cleanup failed: " + "; ".join(errors))


def _run_docker(
    arguments: list[str],
    *,
    timeout: int = 60,
    check: bool = True,
    stdin_text: str | None = None,
) -> CliResult:
    try:
        completed = subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(
            f"Docker command timed out after {timeout}s: {' '.join(arguments[:4])}"
        ) from exc

    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise HarnessError(
            f"Docker command failed ({completed.returncode}): {' '.join(arguments[:4])}"
            + (f"\n{detail}" if detail else "")
        )
    return CliResult(completed.returncode, completed.stdout, completed.stderr)
