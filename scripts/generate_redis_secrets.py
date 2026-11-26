#!/usr/bin/env python3
"""Generate local, per-identity Redis secrets without printing their values."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import secrets
import shutil
import string
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
_GENERATION_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_SECRET_PATTERN = re.compile(rb"[A-Za-z0-9_-]{32,}\Z")
_MANIFEST_NAME = "generation.json"


@dataclass(frozen=True)
class GenerationInfo:
    generation_id: str
    identity_generations: Mapping[str, str]
    path: Path


def _validate_generation_id(value: object) -> str:
    if not isinstance(value, str) or not _GENERATION_ID_PATTERN.fullmatch(value):
        raise ValueError("Invalid Redis credential generation metadata")
    return value


def _read_pointer(directory: Path, name: str) -> str | None:
    pointer = directory / name
    try:
        target = os.readlink(pointer)
    except FileNotFoundError:
        if pointer.exists() or pointer.is_symlink():
            raise ValueError(f"Invalid Redis {name} generation pointer")
        return None
    except OSError as exc:
        raise ValueError(f"Invalid Redis {name} generation pointer") from exc

    match = re.fullmatch(r"generations/([0-9a-f]{32})", target)
    if match is None:
        raise ValueError(f"Invalid Redis {name} generation pointer")
    generation_id = _validate_generation_id(match.group(1))
    if not (directory / "generations" / generation_id).is_dir():
        raise ValueError(f"Redis {name} generation does not exist")
    return generation_id


def _atomic_write(path: Path, content: bytes, mode: int = 0o600) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_set_pointer(directory: Path, name: str, generation_id: str | None) -> None:
    pointer = directory / name
    if generation_id is None:
        pointer.unlink(missing_ok=True)
        return

    temporary_pointer = directory / f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        os.symlink(f"generations/{generation_id}", temporary_pointer)
        os.replace(temporary_pointer, pointer)
    finally:
        temporary_pointer.unlink(missing_ok=True)


def _load_generation(directory: Path, generation_id: str) -> GenerationInfo:
    generation_id = _validate_generation_id(generation_id)
    generation_path = directory / "generations" / generation_id
    manifest_path = generation_path / _MANIFEST_NAME
    try:
        manifest_stat = manifest_path.lstat()
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("Invalid Redis credential generation metadata")
        if manifest_stat.st_mode & 0o077:
            raise ValueError("Redis credential generation metadata is not private")
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read Redis credential generation metadata") from exc

    if (
        not isinstance(manifest, dict)
        or manifest.get("generation_id") != generation_id
        or not isinstance(manifest.get("identity_generations"), dict)
        or set(manifest["identity_generations"]) != set(IDENTITIES)
    ):
        raise ValueError("Invalid Redis credential generation metadata")

    identity_generations = {
        identity: _validate_generation_id(manifest["identity_generations"][identity])
        for identity in IDENTITIES
    }
    for identity in IDENTITIES:
        secret_path = generation_path / identity
        try:
            secret_stat = secret_path.lstat()
            secret = secret_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Redis credential generation is missing {identity}") from exc
        if (
            not secret_path.is_file()
            or secret_path.is_symlink()
            or secret_stat.st_mode & 0o077
            or not _SECRET_PATTERN.fullmatch(secret)
        ):
            raise ValueError(f"Redis credential generation contains invalid {identity}")

    return GenerationInfo(generation_id, identity_generations, generation_path)


def get_generation_info(
    directory: Path,
    generation_id: str | None = None,
) -> GenerationInfo:
    """Return non-secret metadata for the selected or named generation."""
    selected_id = generation_id or _read_pointer(directory, "current")
    if selected_id is None:
        raise ValueError("No Redis credential generation is selected")
    return _load_generation(directory, selected_id)


def read_generation_secret(
    directory: Path,
    identity: str,
    generation_id: str | None = None,
) -> bytes:
    """Read one local credential for controlled runtime tooling and tests."""
    if identity not in IDENTITIES:
        raise ValueError(f"Unknown Redis identity: {identity}")
    info = get_generation_info(directory, generation_id)
    return (info.path / identity).read_bytes()


def _write_generation(
    directory: Path,
    secret_values: Mapping[str, bytes],
    identity_generations: Mapping[str, str],
) -> str:
    generation_id = uuid.uuid4().hex
    generations_directory = directory / "generations"
    generations_directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(generations_directory, 0o700)

    staging_directory = generations_directory / f".staging-{generation_id}"
    final_directory = generations_directory / generation_id
    staging_directory.mkdir(mode=0o700)
    try:
        for identity in IDENTITIES:
            value = secret_values[identity]
            if not _SECRET_PATTERN.fullmatch(value):
                raise ValueError(f"Invalid Redis secret for {identity}")
            _atomic_write(staging_directory / identity, value)

        manifest = {
            "generation_id": generation_id,
            "identity_generations": {
                identity: identity_generations[identity] for identity in IDENTITIES
            },
        }
        _atomic_write(
            staging_directory / _MANIFEST_NAME,
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii"),
        )
        os.replace(staging_directory, final_directory)
    except BaseException:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise
    return generation_id


def _write_transaction(directory: Path, transaction: Mapping[str, str | None]) -> None:
    _atomic_write(
        directory / ".generation-transaction.json",
        json.dumps(transaction, sort_keys=True, separators=(",", ":")).encode("ascii"),
    )


def _recover_transaction(directory: Path) -> None:
    transaction_path = directory / ".generation-transaction.json"
    if not transaction_path.exists():
        return
    try:
        transaction = json.loads(transaction_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cannot recover Redis credential generation transaction") from exc

    if not isinstance(transaction, dict) or set(transaction) != {
        "old_current",
        "old_previous",
        "new_current",
    }:
        raise RuntimeError("Invalid Redis credential generation transaction")
    old_current = (
        None
        if transaction["old_current"] is None
        else _validate_generation_id(transaction["old_current"])
    )
    old_previous = (
        None
        if transaction["old_previous"] is None
        else _validate_generation_id(transaction["old_previous"])
    )
    new_current = _validate_generation_id(transaction["new_current"])
    current = _read_pointer(directory, "current")

    if current == new_current:
        _atomic_set_pointer(directory, "previous", old_current)
    elif current == old_current:
        _atomic_set_pointer(directory, "previous", old_previous)
        shutil.rmtree(directory / "generations" / new_current, ignore_errors=True)
    else:
        raise RuntimeError("Redis credential generation pointers cannot be recovered")
    transaction_path.unlink(missing_ok=True)


def _switch_generation(directory: Path, generation_id: str) -> None:
    old_current = _read_pointer(directory, "current")
    old_previous = _read_pointer(directory, "previous")
    transaction = {
        "old_current": old_current,
        "old_previous": old_previous,
        "new_current": generation_id,
    }
    _write_transaction(directory, transaction)
    _atomic_set_pointer(directory, "current", generation_id)
    _atomic_set_pointer(directory, "previous", old_current)
    (directory / ".generation-transaction.json").unlink(missing_ok=True)


def _prune_generations(directory: Path) -> None:
    current = _read_pointer(directory, "current")
    previous = _read_pointer(directory, "previous")
    generations_directory = directory / "generations"
    if not generations_directory.exists():
        return
    for path in generations_directory.iterdir():
        if path.name.startswith(".staging-") or path.name not in {current, previous}:
            shutil.rmtree(path, ignore_errors=True)


def _legacy_secret_values(directory: Path) -> dict[str, bytes] | None:
    paths = {identity: directory / identity for identity in IDENTITIES}
    present = {
        identity
        for identity, path in paths.items()
        if path.exists() or path.is_symlink()
    }
    if not present:
        return None
    if present != set(IDENTITIES):
        missing = ", ".join(sorted(set(IDENTITIES) - present))
        raise ValueError(f"Incomplete legacy Redis secrets; missing identities: {missing}")

    values: dict[str, bytes] = {}
    for identity, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Legacy Redis secret is not a regular file: {identity}")
        value = path.read_bytes()
        if not _SECRET_PATTERN.fullmatch(value):
            raise ValueError(f"Legacy Redis secret is malformed: {identity}")
        values[identity] = value
    return values


def _remove_legacy_files(directory: Path) -> None:
    for identity in IDENTITIES:
        path = directory / identity
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Legacy Redis secret is not a regular file: {identity}")
        path.unlink(missing_ok=True)


def _create_initial_generation(
    directory: Path,
    *,
    legacy_values: Mapping[str, bytes] | None = None,
) -> str:
    values = dict(legacy_values or {})
    identity_generations: dict[str, str] = {}
    for identity in IDENTITIES:
        if identity not in values:
            values[identity] = "".join(
                secrets.choice(SECRET_ALPHABET) for _ in range(SECRET_LENGTH)
            ).encode("ascii")
        identity_generations[identity] = uuid.uuid4().hex
    generation_id = _write_generation(directory, values, identity_generations)
    _switch_generation(directory, generation_id)
    if legacy_values is not None:
        _remove_legacy_files(directory)
    return generation_id


def _generate_replacement(
    directory: Path,
    identities: tuple[str, ...],
) -> None:
    current_info = get_generation_info(directory)
    current_values = {
        identity: (current_info.path / identity).read_bytes()
        for identity in IDENTITIES
    }
    identity_generations = dict(current_info.identity_generations)
    for identity in identities:
        current_values[identity] = "".join(
            secrets.choice(SECRET_ALPHABET) for _ in range(SECRET_LENGTH)
        ).encode("ascii")
        identity_generations[identity] = uuid.uuid4().hex

    generation_id = _write_generation(
        directory,
        current_values,
        identity_generations,
    )
    _switch_generation(directory, generation_id)


def _generation_lock(directory: Path):
    lock_path = directory / ".generation.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.fchmod(descriptor, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def generate_secrets(
    directory: Path,
    *,
    rotate: str | None = None,
) -> tuple[str, ...]:
    """Create a generation or rotate exactly one identity or all identities."""
    if rotate is not None and rotate != "all" and rotate not in IDENTITIES:
        raise ValueError(f"Unknown Redis identity: {rotate}")

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Redis secret directory must be a real directory")
    os.chmod(directory, 0o700)
    lock_descriptor = _generation_lock(directory)

    try:
        _recover_transaction(directory)
        _prune_generations(directory)
        current = _read_pointer(directory, "current")
        legacy_values = None if current is not None else _legacy_secret_values(directory)

        if current is None:
            if rotate is not None and legacy_values is None:
                raise RuntimeError(
                    "Create the initial Redis credential generation before rotating"
                )
            _create_initial_generation(directory, legacy_values=legacy_values)
            current = _read_pointer(directory, "current")
            if rotate is None:
                return IDENTITIES
        elif rotate is None:
            raise FileExistsError(
                "Redis credential generation already exists; use --rotate IDENTITY or --rotate all"
            )

        selected = IDENTITIES if rotate == "all" else (rotate,)
        _generate_replacement(directory, selected)
        _prune_generations(directory)
        return selected
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def rollback_generation(directory: Path) -> str:
    """Select the previous complete generation and retain the current one for recovery."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    lock_descriptor = _generation_lock(directory)
    try:
        _recover_transaction(directory)
        _prune_generations(directory)
        previous = _read_pointer(directory, "previous")
        if previous is None:
            raise RuntimeError("No previous Redis credential generation is available")
        _switch_generation(directory, previous)
        _prune_generations(directory)
        return previous
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def discard_rollback_generation(directory: Path) -> str | None:
    """Remove the retained rollback generation without changing the selected one."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    lock_descriptor = _generation_lock(directory)
    try:
        _recover_transaction(directory)
        _prune_generations(directory)
        previous = _read_pointer(directory, "previous")
        _atomic_set_pointer(directory, "previous", None)
        _prune_generations(directory)
        return previous
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def main(
    argv: Sequence[str] | None = None,
    *,
    secret_directory: Path = DEFAULT_SECRET_DIRECTORY,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--rotate",
        metavar="IDENTITY_OR_ALL",
        help="rotate one identity or all identities into a new generation",
    )
    actions.add_argument(
        "--rollback",
        action="store_true",
        help="select the retained previous complete generation",
    )
    actions.add_argument(
        "--discard-rollback",
        action="store_true",
        help="discard the retained previous generation",
    )
    actions.add_argument(
        "--status",
        action="store_true",
        help="print non-secret active and rollback generation metadata",
    )
    args = parser.parse_args(argv)

    try:
        if args.status:
            current = get_generation_info(secret_directory)
            previous_id = _read_pointer(secret_directory, "previous")
            print(f"Current Redis credential generation: {current.generation_id}")
            print(f"Previous Redis credential generation: {previous_id or 'none'}")
            for identity in IDENTITIES:
                print(
                    f"{identity}: "
                    f"{current.identity_generations[identity]}"
                )
            return 0
        if args.rollback:
            generation_id = rollback_generation(secret_directory)
            print(f"Selected previous Redis credential generation: {generation_id}")
            return 0
        if args.discard_rollback:
            discarded_id = discard_rollback_generation(secret_directory)
            print(f"Discarded Redis rollback generation: {discarded_id or 'none'}")
            return 0

        generated = generate_secrets(secret_directory, rotate=args.rotate)
        generation = get_generation_info(secret_directory)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    action = "Rotated" if args.rotate is not None else "Created"
    print(f"{action} Redis credential generation: {generation.generation_id}")
    for identity in generated:
        print(f"{action} Redis credential for {identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
