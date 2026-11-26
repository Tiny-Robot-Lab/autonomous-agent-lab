"""Disposable Docker harness for live Issue #229 Redis ACL verification."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from scripts.generate_redis_secrets import (
    generate_secrets,
    get_generation_info,
    read_generation_secret,
)

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
REDIS_REPLAY_LOG_PATH: Final[str] = "/run/redis/replay-integrity.log"
REDIS_REPLAY_LOG_LEVEL: Final[str] = "notice"


def redis_replay_log_prep(*, log_path: str = REDIS_REPLAY_LOG_PATH) -> str:
    """Root bootstrap fragment preparing the current-start replay log.

    Rejects symlinked log paths, truncates the current-start log so stale
    denials cannot poison a later start, and assigns root:redis ownership
    with group-write permission so the redis process can append while no
    other identity can replace the evidence. Must run as root before the
    final setpriv drop to the redis user.
    """
    return (
        f'if [ -L "{log_path}" ]; then echo "symlinked replay log" >&2; exit 1; fi\n'
        f': > "{log_path}"\n'
        f'chown root:redis "{log_path}"\n'
        f'chmod 0660 "{log_path}"\n'
    )


def redis_server_command(*, appendonly: bool) -> str:
    """Final redis-server invocation with enforced bounded replay logging.

    Used by every disposable-harness Redis startup path so the
    replay-integrity health gate always observes the current startup,
    regardless of caller-supplied Redis options.
    """
    aof = "yes" if appendonly else "no"
    return (
        "exec setpriv --reuid=redis --regid=redis --clear-groups -- "
        f'redis-server --appendonly {aof} --save "" '
        f"--logfile {REDIS_REPLAY_LOG_PATH} --loglevel {REDIS_REPLAY_LOG_LEVEL} "
        "--aclfile /run/redis/users.acl\n"
    )
APPLICATION_SECRET_SECURITY_PRELUDE: Final[str] = """
import json
import os
import stat
from pathlib import Path

assert os.geteuid() == 10001 and os.getegid() == 10001
private_password = Path("/run/redis-auth/password")
password_stat = private_password.stat()
assert password_stat.st_uid == 10001 and password_stat.st_gid == 10001
assert stat.S_IMODE(password_stat.st_mode) == 0o400
generation_file = Path(os.environ["REDIS_CREDENTIAL_GENERATION_FILE"])
generation_stat = generation_file.stat()
assert generation_stat.st_uid == 10001 and generation_stat.st_gid == 10001
assert stat.S_IMODE(generation_stat.st_mode) == 0o400
generation = json.loads(generation_file.read_text())
identity = os.environ["REDIS_USERNAME"]
assert generation["identity_generations"][identity] == os.environ["REDIS_CREDENTIAL_GENERATION"]
try:
    Path("/run/secrets/redis_password").read_bytes()
except PermissionError:
    pass
else:
    raise AssertionError("source Redis secret remained readable after privilege drop")
"""


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
    admin_password_file: Path = field(repr=False)
    admin_hash_file: Path = field(repr=False)
    service_images: dict[str, str] = field(default_factory=dict)
    created_containers: list[str] = field(default_factory=list)
    created_volumes: list[str] = field(default_factory=list)
    host_port: int | None = None
    publish_host_port: bool = False
    data_volume_name: str | None = None
    appendonly: bool = False
    compose_file: Path | None = None
    compose_project: str | None = None
    compose_generation_id: str | None = None
    _closed: bool = False

    def __repr__(self) -> str:
        return (
            "LiveRedisAclCluster("
            f"token={self.token!r}, "
            f"network_name={self.network_name!r}, "
            f"redis_container={self.redis_container!r}, "
            f"services={sorted(self.service_images)!r}, "
            f"host_port={self.host_port!r})"
        )

    @classmethod
    def start(
        cls,
        *,
        build_service_images: bool = True,
        publish_host_port: bool = False,
        appendonly: bool = False,
        persistent_data: bool = False,
        use_compose: bool = False,
    ) -> LiveRedisAclCluster:
        if use_compose and publish_host_port:
            raise ValueError("Compose-backed Redis rotation tests do not publish host ports")
        token = uuid.uuid4().hex[:10]
        temp_dir = Path(tempfile.mkdtemp(prefix=f"tiny-agent-lab-live-acl-{token}-"))
        temp_dir.chmod(0o700)
        secret_dir = temp_dir / "synthetic-secrets"
        secret_dir.mkdir(mode=0o700)

        generate_secrets(secret_dir)
        secrets_map: dict[str, str] = {}
        for identity in PRODUCTION_IDENTITIES:
            secrets_map[identity] = read_generation_secret(
                secret_dir, identity
            ).decode("ascii")

        admin_secret = f"FIXTURE_ADMIN_{secrets.token_urlsafe(32)}"
        admin_sha256 = hashlib.sha256(admin_secret.encode("ascii")).hexdigest()
        admin_hash_file = temp_dir / "fixture-admin.sha256"
        admin_hash_file.write_text(admin_sha256, encoding="ascii")
        admin_hash_file.chmod(0o600)
        admin_password_file = temp_dir / "fixture-admin.password"
        admin_password_file.write_text(admin_secret, encoding="ascii")
        admin_password_file.chmod(0o600)
        data_volume_name = (
            f"tiny-agent-lab-acl-data-{token}"
            if persistent_data or use_compose
            else None
        )

        cluster = cls(
            token=token,
            temp_dir=temp_dir,
            secret_dir=secret_dir,
            network_name=f"tiny-agent-lab-acl-net-{token}",
            redis_container=f"tiny-agent-lab-acl-redis-{token}",
            secrets_map=secrets_map,
            admin_secret=admin_secret,
            admin_password_file=admin_password_file,
            admin_hash_file=admin_hash_file,
            publish_host_port=publish_host_port,
            data_volume_name=data_volume_name,
            appendonly=appendonly,
            compose_file=temp_dir / "compose.yaml" if use_compose else None,
            compose_project=f"tiny-acl-{token}" if use_compose else None,
        )

        try:
            _run_docker(["docker", "info"], timeout=30)
            network_command = ["docker", "network", "create"]
            if not publish_host_port:
                network_command.append("--internal")
            network_command.append(cluster.network_name)
            _run_docker(network_command, timeout=30)

            if data_volume_name is not None:
                _run_docker(["docker", "volume", "create", data_volume_name])
                cluster.created_volumes.append(data_volume_name)

            if build_service_images:
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

            if use_compose:
                cluster._start_redis_with_compose()
            else:
                cluster._start_redis_container(admin_hash_file)
            if publish_host_port:
                port_result = _run_docker(
                    ["docker", "port", cluster.redis_container, "6379/tcp"]
                )
                match = re.fullmatch(r"127\.0\.0\.1:(\d+)\s*", port_result.stdout)
                if match is None:
                    raise HarnessError("Disposable Redis host port was not loopback-bound")
                cluster.host_port = int(match.group(1))
            return cluster
        except BaseException:
            cluster.cleanup()
            raise

    @property
    def redis_url(self) -> str:
        if self.host_port is None:
            raise HarnessError("Disposable Redis container has no host-published port")
        return f"redis://127.0.0.1:{self.host_port}/0"

    @property
    def current_generation_id(self) -> str:
        return get_generation_info(self.secret_dir).generation_id

    def rotate_secrets(self, rotate: str = "all") -> dict[str, str]:
        old_values = dict(self.secrets_map)
        generate_secrets(self.secret_dir, rotate=rotate)
        self.secrets_map = {
            identity: read_generation_secret(self.secret_dir, identity).decode("ascii")
            for identity in PRODUCTION_IDENTITIES
        }
        return old_values

    def _start_redis_container(self, admin_hash_file: Path) -> None:
        mounts: list[str] = []
        for identity in PRODUCTION_IDENTITIES:
            mounts.extend(
                [
                    "--mount",
                    f"type=bind,src={self.secret_dir / 'current' / identity},dst=/run/secrets/{identity},readonly",
                ]
            )
        mounts.extend(
            [
                "--mount",
                f"type=bind,src={self.secret_dir / 'current' / 'generation.json'},dst=/run/secrets/redis_generation,readonly",
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
            + redis_replay_log_prep()
            + redis_server_command(appendonly=self.appendonly)
        )

        if self.redis_container not in self.created_containers:
            self.created_containers.append(self.redis_container)
        if self.data_volume_name is not None:
            mounts.extend(
                [
                    "--mount",
                    f"type=volume,src={self.data_volume_name},dst=/data",
                ]
            )
        published_port = (
            ["--publish", "127.0.0.1::6379"] if self.publish_host_port else []
        )
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
                *published_port,
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

        self._wait_until_healthy()

    def _start_redis_with_compose(self) -> None:
        if (
            self.compose_file is None
            or self.compose_project is None
            or self.data_volume_name is None
        ):
            raise HarnessError("Compose-backed Redis harness is not configured")
        command = (
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
            + redis_replay_log_prep()
            + redis_server_command(appendonly=self.appendonly)
        )
        bootstrap_path = self.temp_dir / "redis-bootstrap.sh"
        bootstrap_path.write_text(command, encoding="utf-8")
        bootstrap_path.chmod(0o700)
        volume_sources = (
            (bootstrap_path, "/run/redis-bootstrap.sh"),
            (REDIS_ENTRYPOINT_PATH, "/usr/local/bin/redis-acl-entrypoint.sh"),
            (REDIS_HEALTHCHECK_PATH, "/usr/local/bin/redis-healthcheck.sh"),
            (REDIS_MOUNTINFO_PATH, "/usr/local/bin/redis-mountinfo.sh"),
            (ACL_TEMPLATE_PATH, "/usr/local/bin/users.acl.template"),
            (self.temp_dir / "fixture-admin.sha256", "/run/fixture-admin.sha256"),
        )
        volumes = [
            f"      - {json.dumps(f'{source}:{target}:ro')}"
            for source, target in volume_sources
        ]
        volume_names = [f"redis_{identity.replace('-', '_')}" for identity in PRODUCTION_IDENTITIES]
        secret_mounts = [
            f"      - source: {name}\n        target: {identity}"
            for name, identity in zip(volume_names, PRODUCTION_IDENTITIES, strict=True)
        ]
        secret_mounts.append("      - source: redis_generation\n        target: redis_generation")
        generation = get_generation_info(self.secret_dir)
        self.compose_generation_id = generation.generation_id
        secret_files = [
            f"  {name}:\n    file: {json.dumps(str(generation.path / identity))}"
            for name, identity in zip(volume_names, PRODUCTION_IDENTITIES, strict=True)
        ]
        secret_files.append(
            "  redis_generation:\n"
            f"    file: {json.dumps(str(generation.path / 'generation.json'))}"
        )
        service_volumes = "\n".join(volumes)
        service_secrets = "\n".join(secret_mounts)
        secrets_section = "\n".join(secret_files)
        compose_text = (
            "services:\n"
            "  redis:\n"
            f"    image: {REDIS_IMAGE}\n"
            f"    container_name: {self.redis_container}\n"
            '    entrypoint: ["/bin/sh", "/run/redis-bootstrap.sh"]\n'
            "    volumes:\n"
            f"{service_volumes}\n"
            f"      - redis_data:/data\n"
            "    tmpfs:\n"
            "      - /run/redis:rw,noexec,nosuid,nodev,size=16m,mode=0750\n"
            "    secrets:\n"
            f"{service_secrets}\n"
            "    healthcheck:\n"
            '      test: ["CMD", "/bin/sh", "/usr/local/bin/redis-healthcheck.sh"]\n'
            "      interval: 1s\n"
            "      timeout: 3s\n"
            "      retries: 30\n"
            "    networks: [default]\n"
            "volumes:\n"
            "  redis_data:\n"
            f"    name: {self.data_volume_name}\n"
            "    external: true\n"
            "networks:\n"
            "  default:\n"
            "    name: "
            f"{self.network_name}\n"
            "    external: true\n"
            "secrets:\n"
            f"{secrets_section}\n"
        )
        self.compose_file.write_text(compose_text, encoding="utf-8")
        self.compose_file.chmod(0o600)
        if self.redis_container not in self.created_containers:
            self.created_containers.append(self.redis_container)
        self._run_compose(["up", "-d", "redis"])
        self._wait_until_healthy()

    def _refresh_compose_secret_sources(self) -> None:
        if self.compose_file is None or self.compose_generation_id is None:
            raise HarnessError("Compose-backed Redis harness is not configured")
        previous = get_generation_info(self.secret_dir, self.compose_generation_id)
        current = get_generation_info(self.secret_dir)
        if current.generation_id == previous.generation_id:
            return

        compose_text = self.compose_file.read_text(encoding="utf-8")
        for identity in PRODUCTION_IDENTITIES:
            old_path = json.dumps(str(previous.path / identity))
            new_path = json.dumps(str(current.path / identity))
            if old_path not in compose_text:
                raise HarnessError("Compose Redis secret source could not be updated")
            compose_text = compose_text.replace(old_path, new_path)
        old_generation_path = json.dumps(str(previous.path / "generation.json"))
        new_generation_path = json.dumps(str(current.path / "generation.json"))
        if old_generation_path not in compose_text:
            raise HarnessError("Compose Redis generation source could not be updated")
        compose_text = compose_text.replace(old_generation_path, new_generation_path)
        self.compose_file.write_text(compose_text, encoding="utf-8")
        self.compose_file.chmod(0o600)
        self.compose_generation_id = current.generation_id

    def _run_compose(self, arguments: list[str], *, check: bool = True) -> CliResult:
        if self.compose_file is None or self.compose_project is None:
            raise HarnessError("Compose-backed Redis harness is not configured")
        return _run_docker(
            [
                "docker",
                "compose",
                "--project-name",
                self.compose_project,
                "--file",
                str(self.compose_file),
                *arguments,
            ],
            timeout=120,
            check=check,
        )

    def _wait_until_healthy(self) -> None:
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

    def restart_redis(self) -> None:
        if self.compose_file is not None:
            self._run_compose(["restart", "redis"])
        else:
            _run_docker(["docker", "restart", self.redis_container], timeout=60)
        self._wait_until_healthy()

    def recreate_redis(self) -> None:
        if self.compose_file is not None:
            self._refresh_compose_secret_sources()
            self._run_compose(["up", "-d", "--force-recreate", "redis"])
            self._wait_until_healthy()
            return
        _run_docker(
            [
                "docker",
                "container",
                "rm",
                "--force",
                "--volumes",
                self.redis_container,
            ],
            timeout=60,
        )
        self._start_redis_container(self.admin_hash_file)

    def mounted_secret_matches(self, identity: str, expected_secret: str) -> bool:
        result = _run_docker(
            [
                "docker",
                "exec",
                "-i",
                self.redis_container,
                "/bin/sh",
                "-c",
                (
                    'IFS= read -r expected; mounted=$(cat "/run/secrets/$1"); '
                    '[ "$mounted" = "$expected" ]'
                ),
                "_",
                identity,
            ],
            check=False,
            stdin_text=expected_secret + "\n",
        )
        return result.returncode == 0

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

        secret_file = self.secret_dir / "current" / identity
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
                "--mount",
                f"type=bind,src={self.secret_dir / 'current' / 'generation.json'},dst=/run/secrets/redis_generation,readonly",
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
                stdin_text=APPLICATION_SECRET_SECURITY_PRELUDE + "\n" + script,
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

        if self.compose_file is not None:
            compose_down = self._run_compose(
                ["down", "--volumes", "--remove-orphans"],
                check=False,
            )
            if compose_down.returncode != 0:
                errors.append(f"Compose cleanup: {compose_down.stderr.strip()}")

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

        for volume in reversed(self.created_volumes):
            inspected = _run_docker(["docker", "volume", "inspect", volume], check=False)
            if inspected.returncode == 0:
                removed = _run_docker(["docker", "volume", "rm", volume], check=False)
                if removed.returncode != 0:
                    errors.append(f"volume {volume}: {removed.stderr.strip()}")

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
