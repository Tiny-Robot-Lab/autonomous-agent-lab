"""Canonical JSON helpers and pinned Run 002 contract verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agent_contracts.serialization import canonical_json_dumps

from evaluation.routing.run_002.errors import FatalRunError, FatalRunErrorCode

CONTRACT_ID = "run-002-offline-shadow-integration-003"
CONTRACT_SCHEMA_VERSION = 3
EXPECTED_CONTRACT_CANONICAL_SHA256 = (
    "40ab49738b4f45ea5fc13b9fc7f498622fab0af313b81c0698e4a46746a13fed"
)
EXPECTED_CONTRACT_PHYSICAL_SHA256 = (
    "e9f63fd1e2c2686166588161c0b67eb2631c1f9288f9c2b05eccd6df24c98cfe"
)
EXPECTED_MODEL_ARTIFACT_IDENTITY = (
    "2d449b48722dfe6b7fc6a520b0beffd59472b8458c8c65560febd58adcae5b05"
)
EXPECTED_PROTOCOL_IDENTITY = (
    "1e03a65057e3dd2938975dbe28f0e0ddd92417f336598de89ae270963b748beb"
)
CONTRACT_SELF_HASH_FIELD = "canonical_contract_sha256"
MANIFEST_SELF_HASH_FIELD = "canonical_manifest_sha256"
EXPECTED_LABEL_INDEX_MAP: Mapping[str, str] = MappingProxyType({
    "0": "general",
    "1": "research",
})
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / f"{CONTRACT_ID}.json"
)


def canonical_json_sha256(
    payload: Mapping[str, Any] | list[Any] | tuple[Any, ...],
    *,
    exclude_self_hash_field: str | None = None,
) -> str:
    """Compute SHA-256 of strict UTF-8 canonical JSON, optionally removing one self-hash field."""
    target: Any = payload
    if exclude_self_hash_field is not None:
        if not isinstance(payload, Mapping):
            raise TypeError("exclude_self_hash_field requires a mapping payload")
        target = {k: v for k, v in payload.items() if k != exclude_self_hash_field}

    serialized = canonical_json_dumps(target)
    return hashlib.sha256(serialized.encode("utf-8", errors="strict")).hexdigest()


def load_verified_contract(contract_path: Path | None = None) -> dict[str, Any]:
    """Load the pinned Run 002 contract and verify physical and canonical SHA-256."""
    path = contract_path or DEFAULT_CONTRACT_PATH
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise FatalRunError(
            FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
            "Contract artifact could not be read",
        ) from exc

    physical_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if physical_sha256 != EXPECTED_CONTRACT_PHYSICAL_SHA256:
        raise FatalRunError(
            FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
            "Contract physical SHA-256 mismatch",
        )

    try:
        data = json.loads(raw_bytes.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FatalRunError(
            FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
            "Contract JSON is malformed",
        ) from exc

    if not isinstance(data, dict):
        raise FatalRunError(
            FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
            "Contract payload must be a JSON object",
        )

    if data.get("contract_id") != CONTRACT_ID or data.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise FatalRunError(
            FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
            "Contract identifier or schema version mismatch",
        )

    declared_canonical = data.get(CONTRACT_SELF_HASH_FIELD)
    computed_canonical = canonical_json_sha256(
        data, exclude_self_hash_field=CONTRACT_SELF_HASH_FIELD
    )
    if (
        declared_canonical != EXPECTED_CONTRACT_CANONICAL_SHA256
        or computed_canonical != EXPECTED_CONTRACT_CANONICAL_SHA256
    ):
        raise FatalRunError(
            FatalRunErrorCode.PROTOCOL_IDENTITY_MISMATCH,
            "Contract canonical SHA-256 mismatch",
        )

    return data
