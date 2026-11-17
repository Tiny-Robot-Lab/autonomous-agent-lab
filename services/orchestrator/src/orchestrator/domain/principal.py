import base64
import binascii
import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any

from agent_contracts import canonical_json_dumps

TOKEN_KEY_LABEL = b"autonomous-agent-lab.assignment-proof.token-key.v1"
VERIFIER_KEY_LABEL = b"autonomous-agent-lab.assignment-proof.verifier-key.v1"
HEX_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RuntimePrincipal:
    """Authenticated runtime identity resolved from a verified credential.

    Governed by ADR-0006 (Item 18), ADR-0007 (Decisions 7, 8), and Issue #220:
    - Unique per-runtime principal;
    - Mapped to authorized capabilities and bound executor agent_id;
    - Caller-asserted agent_id or capability overrides are strictly rejected.
    """

    principal_id: str
    bound_agent_id: str
    authorized_capabilities: tuple[str, ...]
    active: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.principal_id, str) or not self.principal_id.strip():
            raise ValueError("principal_id must be a non-empty string")
        object.__setattr__(self, "principal_id", self.principal_id.strip())

        if not isinstance(self.bound_agent_id, str) or not self.bound_agent_id.strip():
            raise ValueError("bound_agent_id must be a non-empty string")
        object.__setattr__(self, "bound_agent_id", self.bound_agent_id.strip())

        if not isinstance(self.authorized_capabilities, (list, tuple)):
            raise TypeError("authorized_capabilities must be a tuple or list")
        clean_caps: list[str] = []
        for cap in self.authorized_capabilities:
            if not isinstance(cap, str) or not cap.strip():
                raise ValueError("capability in authorized_capabilities must be a non-empty string")
            clean_caps.append(cap.strip())
        object.__setattr__(self, "authorized_capabilities", tuple(clean_caps))

    def has_capability(self, capability: str) -> bool:
        return capability.strip() in self.authorized_capabilities


def compute_token_digest(token: str, pepper: str) -> str:
    """Compute HMAC-SHA256 digest of a runtime bearer token using the configured pepper."""
    return hmac.new(
        pepper.encode("utf-8"),
        token.strip().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_token(token: str, pepper: str, expected_digest: str) -> bool:
    """Compare a runtime bearer token against expected digest in constant time."""
    actual_digest = compute_token_digest(token, pepper)
    return hmac.compare_digest(actual_digest, expected_digest.strip())


def parse_master_secret(raw: str | bytes) -> bytes:
    """Parse and validate master key material, ensuring at least 32 bytes of secret material.

    Accepts base64-encoded strings (preferred) or raw strings/bytes of at least 32 bytes.
    """
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            raise ValueError("Secret pepper must be a non-empty string")
        try:
            decoded = base64.b64decode(cleaned.encode("ascii"), validate=True)
            if len(decoded) >= 32:
                return decoded
        except (ValueError, binascii.Error):
            pass
        utf8_bytes = cleaned.encode("utf-8")
        if len(utf8_bytes) >= 32:
            return utf8_bytes
        raise ValueError("Secret pepper must provide at least 32 bytes of secret material")
    elif isinstance(raw, bytes):
        if len(raw) >= 32:
            return raw
        raise ValueError("Secret pepper must provide at least 32 bytes of secret material")
    else:
        raise TypeError("Secret pepper must be a string or bytes")


def derive_proof_subkeys(master_secret: bytes) -> tuple[bytes, bytes]:
    """Derive versioned token_key and verifier_key from master key material."""
    token_key = hmac.new(master_secret, TOKEN_KEY_LABEL, hashlib.sha256).digest()
    verifier_key = hmac.new(master_secret, VERIFIER_KEY_LABEL, hashlib.sha256).digest()
    return token_key, verifier_key


def encode_canonical_proof_identity(
    *,
    assignment_id: str,
    claim_id: str,
    task_id: str,
    root_task_id: str,
    turn_id: str,
    runtime_principal_id: str,
    bound_agent_id: str,
    assigned_capability: str,
    execution_kind: str,
    continuation_version: int | None,
    fencing_epoch: int,
    request_id: str,
    delivery_expires_at: str,
    delegation_grant_id: str | None,
    proof_contract_version: int = 1,
) -> bytes:
    """Encode complete canonical proof identity dictionary into canonical UTF-8 JSON bytes."""
    payload: dict[str, Any] = {
        "proof_contract_version": proof_contract_version,
        "assignment_id": assignment_id.strip(),
        "claim_id": claim_id.strip(),
        "task_id": task_id.strip(),
        "root_task_id": root_task_id.strip(),
        "turn_id": turn_id.strip(),
        "runtime_principal_id": runtime_principal_id.strip(),
        "bound_agent_id": bound_agent_id.strip(),
        "assigned_capability": assigned_capability.strip(),
        "execution_kind": execution_kind.strip(),
        "continuation_version": continuation_version,
        "fencing_epoch": int(fencing_epoch),
        "request_id": request_id.strip(),
        "delivery_expires_at": delivery_expires_at.strip(),
        "delegation_grant_id": delegation_grant_id.strip() if delegation_grant_id is not None else None,
    }
    return canonical_json_dumps(payload).encode("utf-8")


def derive_assignment_proof(token_key: bytes, canonical_identity: bytes) -> str:
    """Derive 64-character lowercase hex assignment proof using token_key."""
    proof = hmac.new(token_key, canonical_identity, hashlib.sha256).hexdigest()
    if not HEX_64_PATTERN.match(proof):
        raise ValueError("Invalid proof format derived")
    return proof


def compute_assignment_verifier_digest(verifier_key: bytes, proof: str) -> str:
    """Compute 64-character lowercase hex verifier digest from assignment proof using verifier_key."""
    clean_proof = proof.strip().lower()
    if not HEX_64_PATTERN.match(clean_proof):
        raise ValueError("Invalid proof format for verifier digest computation")
    digest = hmac.new(verifier_key, clean_proof.encode("utf-8"), hashlib.sha256).hexdigest()
    if not HEX_64_PATTERN.match(digest):
        raise ValueError("Invalid verifier digest format")
    return digest


def verify_assignment_proof(
    verifier_key: bytes, presented_proof: str, expected_verifier_digest: str
) -> bool:
    """Verify presented assignment proof in constant time against expected verifier digest."""
    clean_proof = presented_proof.strip().lower()
    if not HEX_64_PATTERN.match(clean_proof):
        return False
    clean_expected = expected_verifier_digest.strip().lower()
    if not HEX_64_PATTERN.match(clean_expected):
        return False
    computed_digest = compute_assignment_verifier_digest(verifier_key, clean_proof)
    return hmac.compare_digest(computed_digest, clean_expected)
