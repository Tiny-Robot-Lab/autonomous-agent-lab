from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

RedisAuthMode = Literal["required", "unauthenticated_local_test"]

ALLOWED_REDIS_SCHEMES: frozenset[str] = frozenset({"redis", "rediss"})
ALLOWED_REDIS_AUTH_MODES: frozenset[str] = frozenset(
    {"required", "unauthenticated_local_test"}
)
LOCAL_UNAUTHENTICATED_HOSTS: frozenset[str] = frozenset(
    {"localhost", "127.0.0.1", "::1"}
)


def _contains_whitespace_or_control(value: str) -> bool:
    return any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)


def _is_production_environment() -> bool:
    """Return True only when APP_ENV explicitly selects production.

    This is the single production signal for the unauthenticated test-mode
    boundary: test mode must never be reachable in production, no matter which
    construction path (direct config, client-kwarg resolver, or environment
    resolver) requests it.
    """
    return os.environ.get("APP_ENV", "").strip().lower() == "production"


def sanitize_redis_url(url: object) -> str:
    """Return a credential-free, query-redacted Redis endpoint string safe for logs and errors."""
    if not isinstance(url, str):
        return "<non-string-redis-url>"
    if not url or _contains_whitespace_or_control(url):
        return "<invalid-redis-url>"
    try:
        parsed = urlparse(url)
    except ValueError:
        return "<invalid-redis-url>"

    scheme = parsed.scheme
    if not scheme or not parsed.netloc:
        return "<invalid-redis-url>"

    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return f"{scheme}://<invalid-endpoint>"

    if not hostname:
        return f"{scheme}://<invalid-endpoint>"

    formatted_host = f"[{hostname}]" if ":" in hostname else hostname
    port_suffix = f":{port}" if port is not None else ""
    has_userinfo = (
        parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
    )
    userinfo_prefix = "***@" if has_userinfo else ""
    path = parsed.path or ""
    query_suffix = "?<redacted>" if parsed.query else ""
    return f"{scheme}://{userinfo_prefix}{formatted_host}{port_suffix}{path}{query_suffix}"


def _parse_and_validate_redis_url(
    url: object,
    *,
    allow_nonzero_db: bool = False,
) -> tuple[str, str, int]:
    """Validate Redis URL structure, absence of embedded credentials, and database index."""
    if not isinstance(url, str):
        raise TypeError("REDIS_URL must be a string")
    if not url or not url.strip():
        raise ValueError("Invalid REDIS_URL configuration: value cannot be empty")
    if url != url.strip() or _contains_whitespace_or_control(url):
        raise ValueError(
            "Invalid REDIS_URL configuration: URL cannot contain whitespace or control characters"
        )

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValueError("Invalid REDIS_URL format") from exc

    sanitized = sanitize_redis_url(url)
    if parsed.scheme not in ALLOWED_REDIS_SCHEMES:
        raise ValueError(
            f"Invalid REDIS_URL configuration: unsupported scheme in {sanitized}"
        )

    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
    ):
        raise ValueError(
            f"Invalid REDIS_URL configuration: embedded credentials are not allowed in {sanitized}"
        )

    try:
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(
            f"Invalid REDIS_URL configuration: malformed host or port in {sanitized}"
        ) from exc

    if not hostname:
        raise ValueError(
            f"Invalid REDIS_URL configuration: missing host in {sanitized}"
        )

    if parsed.params or parsed.fragment:
        raise ValueError(
            f"Invalid REDIS_URL configuration: unsupported URL parameters or fragment in {sanitized}"
        )

    path_db: int | None = None
    if parsed.path not in ("", "/"):
        db_segment = parsed.path.removeprefix("/")
        if "/" in db_segment or not db_segment.isdigit():
            raise ValueError(
                f"Invalid REDIS_URL configuration: malformed database path in {sanitized}"
            )
        path_db = int(db_segment)

    query_db: int | None = None
    if parsed.query:
        try:
            query_params = parse_qs(
                parsed.query,
                keep_blank_values=True,
                strict_parsing=True,
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid REDIS_URL configuration: malformed query string in {sanitized}"
            ) from exc

        extra_keys = set(query_params.keys()) - {"db"}
        if extra_keys:
            raise ValueError(
                f"Invalid REDIS_URL configuration: unsupported query parameters in {sanitized}"
            )

        db_values = query_params.get("db", [])
        if len(db_values) != 1 or not db_values[0] or not db_values[0].isdigit():
            raise ValueError(
                f"Invalid REDIS_URL configuration: invalid or conflicting 'db' query parameter in {sanitized}"
            )
        query_db = int(db_values[0])

    if path_db is not None and query_db is not None and path_db != query_db:
        raise ValueError(
            f"Invalid REDIS_URL configuration: conflicting path and query database values in {sanitized}"
        )

    effective_db = (
        query_db
        if query_db is not None
        else (path_db if path_db is not None else 0)
    )
    if not allow_nonzero_db and effective_db != 0:
        raise ValueError(
            f"Invalid REDIS_URL configuration: database must be 0 in {sanitized}"
        )

    return url, hostname.lower(), effective_db


def validate_redis_url(url: object, *, allow_nonzero_db: bool = False) -> str:
    """Validate that a Redis URL contains host, port, and database only, without credentials."""
    validated_url, _, _ = _parse_and_validate_redis_url(
        url,
        allow_nonzero_db=allow_nonzero_db,
    )
    return validated_url


def validate_redis_username(username: object) -> str:
    """Validate a Redis ACL username fail-closed."""
    if not isinstance(username, str):
        raise TypeError("REDIS_USERNAME must be a string")
    if not username or not username.strip():
        raise ValueError("Invalid REDIS_USERNAME configuration: value cannot be empty")
    if username != username.strip() or _contains_whitespace_or_control(username):
        raise ValueError(
            "Invalid REDIS_USERNAME configuration: value cannot contain whitespace or control characters"
        )
    return username


def validate_redis_password(password: object) -> str:
    """Validate a Redis ACL password fail-closed without logging or normalizing bytes."""
    if not isinstance(password, str):
        raise TypeError("Redis password must be a string")
    if not password:
        raise ValueError("Invalid Redis password configuration: secret cannot be empty")
    if _contains_whitespace_or_control(password):
        raise ValueError(
            "Invalid Redis password configuration: secret cannot contain whitespace, newlines, or control characters"
        )
    return password


def read_redis_password_file(password_file: object) -> str:
    """Read and validate a Redis password file fail-closed without exposing secret contents."""
    if not isinstance(password_file, str):
        raise TypeError("REDIS_PASSWORD_FILE must be a string")
    if not password_file or not password_file.strip():
        raise ValueError(
            "Invalid REDIS_PASSWORD_FILE configuration: path cannot be empty"
        )
    if password_file != password_file.strip() or any(
        ord(ch) < 0x20 or ord(ch) == 0x7F for ch in password_file
    ):
        raise ValueError(
            "Invalid REDIS_PASSWORD_FILE configuration: path cannot contain surrounding whitespace or control characters"
        )

    path = Path(password_file)
    try:
        with path.open("rb") as handle:
            raw_bytes = handle.read()
    except OSError as exc:
        raise ValueError(
            "Invalid REDIS_PASSWORD_FILE configuration: password file is missing or unreadable"
        ) from exc

    try:
        password = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Invalid REDIS_PASSWORD_FILE configuration: password file is not valid UTF-8"
        ) from exc

    return validate_redis_password(password)


@dataclass(frozen=True)
class RedisConnectionConfig:
    """Validated Redis connection configuration with redacted representation."""

    url: str
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    auth_mode: RedisAuthMode = "required"
    allow_nonzero_db: bool = False

    def __post_init__(self) -> None:
        if self.auth_mode not in ALLOWED_REDIS_AUTH_MODES:
            raise ValueError(
                f"Invalid Redis auth_mode: '{self.auth_mode}'. Allowed: {sorted(ALLOWED_REDIS_AUTH_MODES)}"
            )

        if self.auth_mode == "required":
            if self.allow_nonzero_db:
                raise ValueError(
                    "allow_nonzero_db is not permitted when Redis authentication is required"
                )
            validate_redis_url(self.url, allow_nonzero_db=False)
            if self.username is None or self.password is None:
                raise ValueError(
                    "Both username and password are required for authenticated Redis connections"
                )
            validate_redis_username(self.username)
            validate_redis_password(self.password)
        else:
            if self.username is not None or self.password is not None:
                raise ValueError(
                    "Credentials cannot be supplied in unauthenticated_local_test mode"
                )
            if _is_production_environment():
                raise ValueError(
                    "Invalid Redis configuration: unauthenticated_local_test mode "
                    "is prohibited in production"
                )
            _, hostname, _ = _parse_and_validate_redis_url(
                self.url, allow_nonzero_db=self.allow_nonzero_db
            )
            if hostname not in LOCAL_UNAUTHENTICATED_HOSTS:
                raise ValueError(
                    "Invalid Redis configuration: unauthenticated_local_test mode "
                    "permits loopback hosts only "
                    f"(localhost, 127.0.0.1, ::1) in {sanitize_redis_url(self.url)}"
                )

    @property
    def is_authenticated(self) -> bool:
        return self.auth_mode == "required"

    @property
    def sanitized_url(self) -> str:
        return sanitize_redis_url(self.url)

    def to_from_url_kwargs(self) -> dict[str, Any]:
        """Build minimal audited redis-py 8.1.0 client kwargs (RESP2, no CLIENT SETINFO)."""
        kwargs: dict[str, Any] = {
            "decode_responses": True,
            "protocol": 2,
            "driver_info": None,
        }
        if self.auth_mode == "required":
            kwargs["username"] = self.username
            kwargs["password"] = self.password
        return kwargs

    def to_safe_diagnostics(self) -> dict[str, Any]:
        """Return diagnostic metadata free of credentials and raw secret URLs."""
        return {
            "url": self.sanitized_url,
            "username": self.username,
            "auth_mode": self.auth_mode,
            "authenticated": self.is_authenticated,
            "password_configured": self.password is not None,
        }

    def __repr__(self) -> str:
        return (
            "RedisConnectionConfig("
            f"url={self.sanitized_url!r}, "
            f"username={self.username!r}, "
            f"password={'<redacted>' if self.password is not None else None!r}, "
            f"auth_mode={self.auth_mode!r}, "
            f"allow_nonzero_db={self.allow_nonzero_db!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


def resolve_redis_connection_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    default_url: str = "redis://localhost:6379/0",
    allow_nonzero_db_in_local_test: bool = False,
) -> RedisConnectionConfig:
    """Resolve and validate Redis client configuration from environment variables.

    Boundary rule (fail closed, no inference):
    - Authenticated mode is the default: without an explicit
      REDIS_AUTH_MODE='unauthenticated_local_test', paired REDIS_USERNAME +
      REDIS_PASSWORD_FILE and database 0 are always required, on every host and
      in every environment. Absent credentials never imply test mode, and an
      absent APP_ENV never permits unauthenticated access.
    - The explicit unauthenticated test mode applies only when
      REDIS_AUTH_MODE='unauthenticated_local_test' is set, no credential
      variables are present, the environment is not production, and REDIS_URL
      targets a loopback host ('localhost', '127.0.0.1', or '::1'). The
      RedisConnectionConfig constructor enforces the loopback, production, and
      credential-free constraints.
    """
    env = os.environ if environ is None else environ

    raw_url = env.get("REDIS_URL")
    candidate_url = default_url if raw_url is None else raw_url

    raw_username = env.get("REDIS_USERNAME")
    raw_password_file = env.get("REDIS_PASSWORD_FILE")
    raw_auth_mode = env.get("REDIS_AUTH_MODE")
    app_env = env.get("APP_ENV", "development").strip().lower()

    if raw_auth_mode is not None:
        clean_auth_mode = raw_auth_mode.strip()
        if clean_auth_mode not in ALLOWED_REDIS_AUTH_MODES:
            raise ValueError(
                "Invalid REDIS_AUTH_MODE configuration: must be 'required' or 'unauthenticated_local_test'"
            )
    else:
        clean_auth_mode = None

    if clean_auth_mode == "unauthenticated_local_test":
        if raw_username is not None or raw_password_file is not None:
            raise ValueError(
                "Invalid Redis configuration: unauthenticated_local_test mode cannot "
                "be combined with REDIS_USERNAME or REDIS_PASSWORD_FILE"
            )
        if app_env == "production":
            raise ValueError(
                "Invalid Redis configuration: unauthenticated_local_test mode is not "
                "permitted in production"
            )
        return RedisConnectionConfig(
            url=candidate_url,
            username=None,
            password=None,
            auth_mode="unauthenticated_local_test",
            allow_nonzero_db=allow_nonzero_db_in_local_test,
        )

    validate_redis_url(candidate_url, allow_nonzero_db=False)
    if raw_username is None and raw_password_file is None:
        raise ValueError(
            "Invalid Redis authentication configuration: both REDIS_USERNAME and REDIS_PASSWORD_FILE are required; "
            "for explicit local-test use only, set REDIS_AUTH_MODE='unauthenticated_local_test' "
            "with a loopback REDIS_URL"
        )
    if raw_username is None:
        raise ValueError(
            "Invalid Redis authentication configuration: REDIS_USERNAME is required when REDIS_PASSWORD_FILE is set"
        )
    if raw_password_file is None:
        raise ValueError(
            "Invalid Redis authentication configuration: REDIS_PASSWORD_FILE is required when REDIS_USERNAME is set"
        )
    username = validate_redis_username(raw_username)
    password = read_redis_password_file(raw_password_file)
    return RedisConnectionConfig(
        url=candidate_url,
        username=username,
        password=password,
        auth_mode="required",
        allow_nonzero_db=False,
    )


def resolve_client_connection_kwargs(
    url: str,
    *,
    connection_config: RedisConnectionConfig | None = None,
    username: str | None = None,
    password: str | None = None,
    allow_unauthenticated_local_test: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Resolve validated URL and redis-py 8.1.0 from_url kwargs for service Redis clients.

    Fail-closed boundary: bare URL construction without credentials never falls back
    to unauthenticated Redis. Callers must either pass a validated
    RedisConnectionConfig, paired direct username/password credentials, or
    explicitly opt into the test-only mode with
    allow_unauthenticated_local_test=True. The explicit test mode permits
    loopback hosts only (localhost, 127.0.0.1, ::1), rejects embedded URL
    credentials and non-loopback hosts, is prohibited in production, and
    preserves nonzero-database compatibility only for the existing isolated
    local Redis tests. All paths reject embedded URL credentials and apply
    protocol=2, driver_info=None, and decode_responses=True.
    """
    if connection_config is not None:
        if (
            username is not None
            or password is not None
            or allow_unauthenticated_local_test
        ):
            raise ValueError(
                "Do not pass username/password/test-mode alongside connection_config"
            )
        return connection_config.url, connection_config.to_from_url_kwargs()

    if username is not None or password is not None:
        if allow_unauthenticated_local_test:
            raise ValueError(
                "Do not combine credentials with allow_unauthenticated_local_test"
            )
        if username is None or password is None:
            raise ValueError(
                "Both username and password must be provided together when configuring Redis credentials"
            )
        cfg = RedisConnectionConfig(
            url=url,
            username=username,
            password=password,
            auth_mode="required",
            allow_nonzero_db=False,
        )
        return cfg.url, cfg.to_from_url_kwargs()

    if not allow_unauthenticated_local_test:
        raise ValueError(
            "Invalid Redis configuration: unauthenticated Redis construction requires "
            "explicit opt-in via allow_unauthenticated_local_test=True "
            f"(loopback hosts only) in {sanitize_redis_url(url)}; production and Compose "
            "paths must configure REDIS_USERNAME/REDIS_PASSWORD_FILE"
        )

    cfg = RedisConnectionConfig(
        url=url,
        username=None,
        password=None,
        auth_mode="unauthenticated_local_test",
        allow_nonzero_db=True,
    )
    return cfg.url, cfg.to_from_url_kwargs()
