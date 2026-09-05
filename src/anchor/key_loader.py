"""Anchor gateway auth token source.

Security contract (v0.10+, 2026-07-21; admin scope v0.9.51):
  Anchor's internal gateway tokens (the ``sk-anchor-*`` family) are loaded
  EXCLUSIVELY from environment variables. There is intentionally **no file
  or JSON fallback** -- if no client keys are configured, the gateway enters
  *fail-closed* mode (zero client tokens, requests rejected with HTTP 503
  unless ``ANCHOR_AUTH_DISABLED=1`` is set together with ``ANCHOR_PROFILE=debug``).

Client keys:
  ``ANCHOR_API_KEYS`` — comma- or newline-separated raw tokens.
  Each becomes scope=client (defaults rpm=60, daily_tokens=0 unlimited;
  override with ANCHOR_CLIENT_RPM / ANCHOR_CLIENT_DAILY_TOKENS).

Admin keys (v0.9.51):
  ``ANCHOR_API_KEY_ADMIN`` — comma- or newline-separated admin tokens.
  Each becomes scope=admin. Required for /admin/* and /debug/*.

Why fail-closed is mandatory:
  The previous implementation loaded tokens from ``data/api_keys.json``,
  which was accidentally tracked in git history (commit 527306d).

Rotation procedure: see ``README.md`` -> *Security -> Token rotation*.
"""
from __future__ import annotations

import logging
import os
from typing import Final, TypedDict

_log = logging.getLogger("anchor.key_loader")

_KEY_ENV_VAR: Final[str] = "ANCHOR_API_KEYS"
_ADMIN_KEY_ENV_VAR: Final[str] = "ANCHOR_API_KEY_ADMIN"

_DEFAULT_RPM: Final[int] = 60
_DEFAULT_DAILY_TOKENS: Final[int] = 0  # pure mode: unlimited daily (0 = off)
_DEFAULT_SCOPE: Final[str] = "client"
_ADMIN_SCOPE: Final[str] = "admin"


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        _log.warning("invalid %s=%r; using default %s", name, raw, default)
        return default


def _client_rpm() -> int:
    return max(1, _env_int("ANCHOR_CLIENT_RPM", _DEFAULT_RPM))


def _client_daily_tokens() -> int:
    # 0 or negative = unlimited (personal / local gateway)
    return _env_int("ANCHOR_CLIENT_DAILY_TOKENS", _DEFAULT_DAILY_TOKENS)


def _admin_rpm() -> int:
    return max(1, _env_int("ANCHOR_ADMIN_RPM", _DEFAULT_RPM))


def _admin_daily_tokens() -> int:
    return _env_int("ANCHOR_ADMIN_DAILY_TOKENS", _DEFAULT_DAILY_TOKENS)


class ApiKeyRecord(TypedDict):
    """Normalized auth record consumed by ``anchor.server``."""
    key: str
    name: str
    rpm: int
    daily_tokens: int
    scope: str


def _parse_tokens(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return []
    return [tok.strip() for tok in raw.replace("\n", ",").split(",") if tok.strip()]


def _records_from_tokens(
    tokens: list[str],
    *,
    scope: str,
    name_prefix: str,
    rpm: int | None = None,
    daily_tokens: int | None = None,
) -> list[ApiKeyRecord]:
    rpm_v = _DEFAULT_RPM if rpm is None else rpm
    daily_v = _DEFAULT_DAILY_TOKENS if daily_tokens is None else daily_tokens
    return [
        {
            "key": tok,
            "name": f"{name_prefix}-{i}" if len(tokens) > 1 else name_prefix,
            "rpm": rpm_v,
            "daily_tokens": daily_v,
            "scope": scope,
        }
        for i, tok in enumerate(tokens, start=1)
    ]


def load_anchor_api_keys() -> list[ApiKeyRecord]:
    """Load client + admin gateway API keys from env vars.

    Behavior:
      * ``ANCHOR_API_KEYS`` missing/empty and no admin keys → log error, return ``[]``
      * Admin keys from ``ANCHOR_API_KEY_ADMIN`` always scope=admin
      * Client keys always scope=client
      * No file or JSON fallback by design
    """
    client_tokens = _parse_tokens(os.environ.get(_KEY_ENV_VAR))
    admin_tokens = _parse_tokens(os.environ.get(_ADMIN_KEY_ENV_VAR))

    if not client_tokens and not admin_tokens:
        _log.error(
            "ANCHOR_API_KEYS missing or empty -- gateway in fail-closed mode "
            "(no auth tokens; client requests will be rejected). "
            "Set %s (and optionally %s for admin) and restart the server.",
            _KEY_ENV_VAR, _ADMIN_KEY_ENV_VAR,
        )
        return []

    if not client_tokens:
        # Admin-only config is valid for ops, but client traffic will 401.
        _log.warning(
            "ANCHOR_API_KEYS empty but %s set — admin-only mode "
            "(client chat requests will fail auth)",
            _ADMIN_KEY_ENV_VAR,
        )

    records: list[ApiKeyRecord] = []
    # Client keys keep name="unnamed" for backward compatibility with tests/ops.
    client_rpm = _client_rpm()
    client_daily = _client_daily_tokens()
    records.extend(
        {
            "key": tok,
            "name": "unnamed",
            "rpm": client_rpm,
            "daily_tokens": client_daily,
            "scope": _DEFAULT_SCOPE,
        }
        for tok in client_tokens
    )
    records.extend(
        _records_from_tokens(
            admin_tokens,
            scope=_ADMIN_SCOPE,
            name_prefix="admin",
            rpm=_admin_rpm(),
            daily_tokens=_admin_daily_tokens(),
        )
    )

    # De-dupe by key: if same token appears in both, admin wins
    by_key: dict[str, ApiKeyRecord] = {}
    for rec in records:
        prev = by_key.get(rec["key"])
        if prev is None or rec["scope"] == _ADMIN_SCOPE:
            by_key[rec["key"]] = rec
    return list(by_key.values())


def is_anchor_key_env_configured() -> bool:
    """True iff client or admin key env is non-empty."""
    return bool(
        _parse_tokens(os.environ.get(_KEY_ENV_VAR))
        or _parse_tokens(os.environ.get(_ADMIN_KEY_ENV_VAR))
    )


__all__ = [
    "ApiKeyRecord",
    "load_anchor_api_keys",
    "is_anchor_key_env_configured",
]
