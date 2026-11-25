"""Copy the mounted Redis secret into private runtime storage, then drop privileges."""

from __future__ import annotations

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
SECRET_PATTERN = re.compile(rb"[A-Za-z0-9_-]{32,}\Z")


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
        restrict_secret_source_directory(SOURCE_FILE.parent)
    except ValueError as exc:
        print(f"redis-secret-entrypoint: {exc}", file=sys.stderr)
        return 1

    environment = os.environ.copy()
    environment["REDIS_PASSWORD_FILE"] = str(DESTINATION_FILE)
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
