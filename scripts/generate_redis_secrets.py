#!/usr/bin/env python3
"""Generate local, per-identity Redis secrets without printing their values."""

from __future__ import annotations

import argparse
import os
import secrets
import string
import tempfile
from collections.abc import Sequence
from pathlib import Path

IDENTITIES = (
    "orchestrator",
    "worker-1",
    "research-agent-1",
    "research-agent-2",
    "analysis-agent-1",
    "healthcheck",
)
SECRET_LENGTH = 48
SECRET_ALPHABET = string.ascii_letters + string.digits + "_-"
DEFAULT_SECRET_DIRECTORY = (
    Path(__file__).resolve().parents[1] / ".secrets" / "redis"
)


def _write_secret_atomically(directory: Path, identity: str, secret: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{identity}.",
        dir=directory,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(secret)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, directory / identity)
    finally:
        temporary_path.unlink(missing_ok=True)


def generate_secrets(
    directory: Path,
    *,
    rotate: str | None = None,
) -> tuple[str, ...]:
    """Create all secrets, or explicitly rotate one identity or all identities."""
    if rotate is not None and rotate != "all" and rotate not in IDENTITIES:
        raise ValueError(f"Unknown Redis identity: {rotate}")

    identities = IDENTITIES if rotate in (None, "all") else (rotate,)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)

    lock_path = directory / ".generation.lock"
    try:
        lock_path.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RuntimeError("Redis secret generation is already in progress") from exc

    try:
        targets = tuple(directory / identity for identity in identities)
        if rotate is None and any(target.exists() or target.is_symlink() for target in targets):
            raise FileExistsError(
                "Redis secrets already exist; use --rotate IDENTITY or --rotate all"
            )

        for identity, target in zip(identities, targets, strict=True):
            if rotate is not None and target.exists() and not target.is_file():
                raise ValueError(f"Cannot rotate non-regular Redis secret file: {identity}")
            value = "".join(
                secrets.choice(SECRET_ALPHABET) for _ in range(SECRET_LENGTH)
            ).encode("ascii")
            _write_secret_atomically(directory, identity, value)
    finally:
        lock_path.rmdir()

    return identities


def main(
    argv: Sequence[str] | None = None,
    *,
    secret_directory: Path = DEFAULT_SECRET_DIRECTORY,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rotate",
        metavar="IDENTITY_OR_ALL",
        help="explicitly rotate one identity or all selected identities",
    )
    args = parser.parse_args(argv)
    try:
        generated = generate_secrets(secret_directory, rotate=args.rotate)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    action = "Rotated" if args.rotate is not None else "Created"
    for identity in generated:
        print(f"{action} Redis secret for {identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
