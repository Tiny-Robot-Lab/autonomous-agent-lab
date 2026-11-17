import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

CANONICAL_JSON_SEPARATORS = (",", ":")

# Forbidden credential segment tokens when normalized
FORBIDDEN_CREDENTIAL_SEGMENTS: frozenset[str] = frozenset({
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "credentials",
    "apikey",
    "bearer",
    "authorization",
})

# Forbidden exact compound key names (normalized with underscores)
FORBIDDEN_EXACT_KEY_NAMES: frozenset[str] = frozenset({
    "api_key",
    "apikey",
    "private_key",
    "secret_key",
    "access_token",
    "auth_token",
    "client_secret",
    "refresh_token",
    "id_token",
    "bearer_token",
})

CAMEL_CASE_SPLIT_REGEX = re.compile(r"([a-z0-9])([A-Z])")
DELIMITER_SPLIT_REGEX = re.compile(r"[_\-.\s]+")


def canonical_json_dumps(obj: Any) -> str:
    """Serialize data to deterministic canonical UTF-8 JSON.

    Guarantees:
    - Sorted dictionary keys (sort_keys=True)
    - Compact standard separators (",", ":") with no extra whitespace
    - Direct UTF-8 representation without escaping non-ASCII (ensure_ascii=False)
    """
    plain = to_plain_json_structure(obj)
    return json.dumps(
        plain,
        sort_keys=True,
        separators=CANONICAL_JSON_SEPARATORS,
        ensure_ascii=False,
    )


def freeze_value(val: Any) -> Any:
    """Recursively convert nested mutable mappings/sequences to immutable structures."""
    if isinstance(val, (Mapping, MappingProxyType)):
        return MappingProxyType({str(k): freeze_value(v) for k, v in val.items()})
    if isinstance(val, (list, tuple)):
        return tuple(freeze_value(item) for item in val)
    return val


def to_plain_json_structure(val: Any) -> Any:
    """Recursively convert frozen/immutable structures back to plain dicts and lists for JSON."""
    if isinstance(val, (Mapping, MappingProxyType)):
        return {str(k): to_plain_json_structure(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_plain_json_structure(item) for item in val]
    if hasattr(val, "to_dict"):
        return to_plain_json_structure(val.to_dict())
    return val


def tokenize_key_name(key: str) -> list[str]:
    """Tokenize key name across camelCase, snake_case, kebab-case, and dot notation."""
    expanded = CAMEL_CASE_SPLIT_REGEX.sub(r"\1 \2", key)
    segments = [s.lower() for s in DELIMITER_SPLIT_REGEX.split(expanded) if s]
    return segments


def is_credential_bearing_key(key: str) -> bool:
    """Determine if a key name represents a credential or secret."""
    normalized_full = key.strip().lower().replace("-", "_").replace(".", "_")
    if normalized_full in FORBIDDEN_EXACT_KEY_NAMES:
        return True
    segments = tokenize_key_name(key)
    return any(seg in FORBIDDEN_CREDENTIAL_SEGMENTS for seg in segments)


def check_no_credential_bearing_keys(
    data: Any,
    context: str,
    max_depth: int = 5,
    current_depth: int = 0,
) -> None:
    """Recursively validate that no credential or secret keys exist in bounded JSON structures."""
    if current_depth > max_depth:
        raise ValueError(f"{context} exceeds maximum structural nesting depth of {max_depth}")

    if isinstance(data, (Mapping, MappingProxyType)):
        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(f"{context} key must be a string, got {type(k).__name__}")
            clean_k = k.strip()
            if is_credential_bearing_key(clean_k):
                raise ValueError(
                    f"Credential-bearing key '{clean_k}' is forbidden in {context}"
                )
            check_no_credential_bearing_keys(v, context, max_depth, current_depth + 1)
    elif isinstance(data, (list, tuple)):
        for item in data:
            check_no_credential_bearing_keys(item, context, max_depth, current_depth + 1)
