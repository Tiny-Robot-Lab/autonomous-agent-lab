"""Copy the mounted Redis secret into private runtime storage, then drop privileges."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

APP_UID = 10001
APP_GID = 10001
SOURCE_FILE = Path("/run/secrets/redis_password")
DESTINATION_FILE = Path("/run/redis-auth/password")
GENERATION_SOURCE_FILE = Path("/run/secrets/redis_generation")
GENERATION_DESTINATION_FILE = Path("/run/redis-auth/generation.json")
SECRET_PATTERN = re.compile(rb"[A-Za-z0-9_-]{32,}\Z")
GENERATION_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
REDIS_IDENTITIES = {
    "orchestrator",
    "worker-1",
    "research-agent-1",
    "research-agent-2",
    "analysis-agent-1",
    "healthcheck",
}


def prepare_private_secret(
    source_file: Path,
    destination_file: Path,
    uid: int,
    gid: int,
) -> None:
    try:
        source_stat = source_file.lstat()
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("mounted Redis secret is not a regular file")
        with source_file.open("rb") as source:
            secret = source.read()
    except OSError as exc:
        raise ValueError("mounted Redis secret is missing or unreadable") from exc

    if not SECRET_PATTERN.fullmatch(secret):
        raise ValueError("mounted Redis secret is malformed")

    runtime_directory = destination_file.parent
    try:
        runtime_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory_stat = runtime_directory.lstat()
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError("private Redis secret directory is not a directory")
        os.chown(runtime_directory, uid, gid)
        os.chmod(runtime_directory, 0o700)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".redis-password.",
            dir=runtime_directory,
        )
        temporary_file = Path(temporary_name)
        try:
            os.fchown(descriptor, uid, gid)
            os.fchmod(descriptor, 0o400)
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(secret)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_file, destination_file)
        finally:
            temporary_file.unlink(missing_ok=True)
    except OSError as exc:
        raise ValueError("cannot create private Redis secret copy") from exc


def restrict_secret_source_directory(
    directory: Path,
    uid: int = 0,
    gid: int = 0,
) -> None:
    try:
        directory_stat = directory.lstat()
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError("mounted Redis secret directory is not a directory")
        os.chown(directory, uid, gid)
        os.chmod(directory, 0o700)
    except OSError as exc:
        raise ValueError("cannot restrict mounted Redis secret directory") from exc


def prepare_private_generation_metadata(
    source_file: Path,
    destination_file: Path,
    identity: str,
    uid: int,
    gid: int,
) -> tuple[str, str]:
    try:
        source_stat = source_file.lstat()
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("mounted Redis generation metadata is not a regular file")
        raw_metadata = source_file.read_bytes()
        metadata = json.loads(raw_metadata)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("mounted Redis generation metadata is missing or invalid") from exc

    if (
        len(raw_metadata) > 4096
        or not isinstance(metadata, dict)
        or set(metadata) != {"generation_id", "identity_generations"}
        or not isinstance(metadata["identity_generations"], dict)
        or set(metadata["identity_generations"]) != REDIS_IDENTITIES
        or identity not in REDIS_IDENTITIES
    ):
        raise ValueError("mounted Redis generation metadata is invalid")

    generation_id = metadata["generation_id"]
    identity_generation = metadata["identity_generations"][identity]
    if (
        not isinstance(generation_id, str)
        or not GENERATION_ID_PATTERN.fullmatch(generation_id)
        or not isinstance(identity_generation, str)
        or not GENERATION_ID_PATTERN.fullmatch(identity_generation)
        or any(
            not isinstance(value, str) or not GENERATION_ID_PATTERN.fullmatch(value)
            for value in metadata["identity_generations"].values()
        )
    ):
        raise ValueError("mounted Redis generation metadata is invalid")

    runtime_directory = destination_file.parent
    try:
        runtime_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chown(runtime_directory, uid, gid)
        os.chmod(runtime_directory, 0o700)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".redis-generation.",
            dir=runtime_directory,
        )
        temporary_file = Path(temporary_name)
        try:
            os.fchown(descriptor, uid, gid)
            os.fchmod(descriptor, 0o400)
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(raw_metadata)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_file, destination_file)
        finally:
            temporary_file.unlink(missing_ok=True)
    except OSError as exc:
        raise ValueError("cannot create private Redis generation metadata copy") from exc

    return generation_id, identity_generation


def main(arguments: Sequence[str] | None = None) -> int:
    command = list(sys.argv[1:] if arguments is None else arguments)
    if not command:
        print("redis-secret-entrypoint: missing application command", file=sys.stderr)
        return 2
    if os.geteuid() != 0:
        print("redis-secret-entrypoint: entrypoint must start as root", file=sys.stderr)
        return 1

    try:
        prepare_private_secret(SOURCE_FILE, DESTINATION_FILE, APP_UID, APP_GID)
        username = os.environ.get("REDIS_USERNAME", "")
        generation_id, identity_generation = prepare_private_generation_metadata(
            GENERATION_SOURCE_FILE,
            GENERATION_DESTINATION_FILE,
            username,
            APP_UID,
            APP_GID,
        )
        restrict_secret_source_directory(SOURCE_FILE.parent)
    except ValueError as exc:
        print(f"redis-secret-entrypoint: {exc}", file=sys.stderr)
        return 1

    environment = os.environ.copy()
    environment["REDIS_PASSWORD_FILE"] = str(DESTINATION_FILE)
    environment["REDIS_CREDENTIAL_GENERATION_FILE"] = str(
        GENERATION_DESTINATION_FILE
    )
    environment["REDIS_CREDENTIAL_GENERATION_ID"] = generation_id
    environment["REDIS_CREDENTIAL_GENERATION"] = identity_generation
    try:
        os.setgroups([])
        os.setgid(APP_GID)
        os.setuid(APP_UID)
        os.execvpe(command[0], command, environment)
    except OSError as exc:
        print(f"redis-secret-entrypoint: cannot exec application: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
