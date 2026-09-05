"""API-key auth + per-key quota middleware (extracted from server.py)."""
from __future__ import annotations

import json
import logging
import os
import secrets
from typing import Callable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from anchor.key_quota import check_and_consume as _kq_consume, usage_snapshot as _kq_usage
from anchor.key_loader import load_anchor_api_keys as _load_anchor_api_keys

_logger = logging.getLogger("anchor.auth")

AUTH_EXEMPT_PATHS = {"/healthz"}
ADMIN_PATH_PREFIXES = ("/admin/", "/debug/")
QUOTA_EXEMPT_PATHS = {"/healthz", "/v1/keys/me"}


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


# audit 2026-08-16 (LOW): the L2 warning below used to fire on EVERY request
# whenever ANCHOR_LOOPBACK_TRUST=1 (the middleware calls loopback_trust_enabled()
# per request at auth.py:205), flooding logs and masking real errors. Emit once.
_LOOPBACK_WARNED = False


def loopback_trust_enabled() -> bool:
    """Daily self-use: trust loopback clients without a real Anchor key.

    Safer than ANCHOR_AUTH_DISABLED (which is global): only 127.0.0.1/::1.
    Codex ignores provider api_key and sends auth.json OPENAI_API_KEY; with this
    flag, local Codex/pi can hit :8088 directly (no auth-rewrite proxy).
    """
    global _LOOPBACK_WARNED
    enabled = truthy(os.environ.get("ANCHOR_LOOPBACK_TRUST"))
    if enabled and not _LOOPBACK_WARNED:
        # L2 (security LOW): surface once so operators notice when
        # loopback-trust is on (it grants unlimited quota to loopback).
        _LOOPBACK_WARNED = True
        _logger.warning(
            "ANCHOR_LOOPBACK_TRUST=1 — loopback clients (127.0.0.1/::1) "
            "bypass API-key auth with unlimited quota. Intended for local "
            "dev only; do NOT enable in multi-tenant deployments."
        )
    return enabled


def is_loopback_client(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    # Do NOT honor X-Forwarded-For here — that would allow remote spoofing.
    return host in {"127.0.0.1", "::1", "localhost"}


def loopback_key_record(provided: str = "") -> dict:
    """Synthetic unlimited client record for trusted loopback traffic."""
    return {
        "key": provided or "loopback-local",
        "name": "loopback-local",
        "rpm": int(os.environ.get("ANCHOR_CLIENT_RPM") or 60),
        "daily_tokens": int(os.environ.get("ANCHOR_CLIENT_DAILY_TOKENS") or 0),
        "scope": "client",
    }



def configured_key_records() -> list[dict]:
    return list(_load_anchor_api_keys())


def configured_api_keys() -> list[str]:
    return [record["key"] for record in configured_key_records()]


def match_key_record(provided: str) -> dict | None:
    for record in configured_key_records():
        if secrets.compare_digest(provided, record["key"]):
            return record
    return None


def estimate_request_tokens(request: Request, body: bytes | None = None) -> int:
    """Rough pre-charge for key daily quota.

    Uses message size + a *clamped* completion budget. Unclamped max_tokens
    (e.g. pi max_completion_tokens=65536) would burn the daily cap in one call.
    """
    try:
        raw = body if body is not None else getattr(request, "_body", b"")
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(payload, dict):
            return 1
        text = json.dumps(payload.get("messages", payload.get("input", "")), ensure_ascii=False)
        requested = (
            payload.get("max_tokens")
            or payload.get("max_output_tokens")
            or payload.get("max_completion_tokens")
            or 0
        )
        # Cap prepaid completion estimate; actual generation is usually much smaller.
        completion = min(int(requested or 0), 4096)
        return max(1, int(len(text) / 4) + completion)
    except Exception:
        content_length = request.headers.get("content-length")
        try:
            return max(1, int(int(content_length or "0") / 4))
        except ValueError:
            return 1


async def enforce_key_quota(request: Request, key_record: dict) -> JSONResponse | None:
    if request.url.path in QUOTA_EXEMPT_PATHS:
        return None
    # L2 (security/DoS MEDIUM): pre-charge on Content-Length header only,
    # without reading the body. The body-size middleware already enforces
    # the hard cap (8 MB by default) before this middleware runs, but that
    # means a near-cap body is fully allocated before we reject on quota.
    # Using CL alone is sufficient for token estimation (len(text)/4 is
    # close enough to the post-parse count) and avoids the O(N) body read.
    body = getattr(request, "_body", b"")
    tokens = estimate_request_tokens(request, body)
    rpm = int(key_record["rpm"] if key_record.get("rpm") is not None else 60)
    daily_tokens = int(key_record["daily_tokens"] if key_record.get("daily_tokens") is not None else 0)
    err = _kq_consume(key_record["key"], tokens, rpm=rpm, daily_tokens=daily_tokens)
    if err is None:
        return None
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(err.get("retry_after", 1))},
        content={
            "error": {
                "message": err.get("message", "rate limit"),
                "type": err.get("type", "rate_limit_error"),
            }
        },
    )


def key_usage(record: dict) -> dict:
    rpm = int(record["rpm"] if record.get("rpm") is not None else 60)
    daily_tokens = int(record["daily_tokens"] if record.get("daily_tokens") is not None else 0)
    return _kq_usage(
        record["key"], rpm=rpm, daily_tokens=daily_tokens, name=record.get("name"),
    )


def extract_api_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return (
        request.headers.get("x-api-key")
        or request.headers.get("api-key")
        or ""
    ).strip()


def auth_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "authentication_error"}},
    )


def is_admin_path(path: str) -> bool:
    # L2 (security MEDIUM): strict prefix match — only "/admin/..." and
    # "/debug/..." with trailing "/". Avoids accidentally matching future
    # routes like "/admin-public-stats" which would bypass the admin
    # scope requirement under loopback-trust bypass.
    return any(
        path == p.rstrip("/") or path.startswith(p)
        for p in ADMIN_PATH_PREFIXES
    )


def require_admin_scope(request: Request) -> JSONResponse | None:
    record = getattr(request.state, "anchor_api_key_record", None)
    if record is None:
        return auth_error(403, "Admin access requires a valid API key with admin scope")
    if record.get("scope") != "admin":
        return auth_error(403, "This endpoint requires an admin-scoped API key")
    return None


def register_auth_middleware(app, *, metrics_auth_failure: Optional[Callable] = None) -> None:
    """Install the API-key + quota middleware on a FastAPI app."""

    @app.middleware("http")
    async def _api_key_auth(request: Request, call_next):
        if request.url.path in AUTH_EXEMPT_PATHS:
            return await call_next(request)

        is_admin = is_admin_path(request.url.path)

        if truthy(os.environ.get("ANCHOR_AUTH_DISABLED")):
            if not is_admin and os.environ.get("ANCHOR_PROFILE") == "debug":
                return await call_next(request)

        records = configured_key_records()
        if not records:
            if os.environ.get("ANCHOR_PROFILE") == "debug" and truthy(os.environ.get("ANCHOR_AUTH_DISABLED")):
                return await call_next(request)
            return auth_error(503, "Anchor auth is required but no API key is configured")

        provided = extract_api_key(request)
        key_record = match_key_record(provided) if provided else None
        if key_record is None and loopback_trust_enabled() and is_loopback_client(request) and not is_admin:
            # Local Codex/pi: accept any/missing bearer; no auth-rewrite proxy needed.
            key_record = loopback_key_record(provided)
            _logger.debug("loopback trust path=%s key_present=%s", request.url.path, bool(provided))
        if key_record is None:
            try:
                if metrics_auth_failure is not None:
                    reason = "missing_key" if not provided else "invalid_key"
                    metrics_auth_failure(request.url.path, reason=reason)
            except Exception as exc:
                _logger.warning("auth failure metric update failed: %s", exc)
            return auth_error(401, "Missing or invalid API key")
        request.state.anchor_api_key = provided or key_record.get("key", "")
        request.state.anchor_api_key_record = key_record

        if is_admin:
            scope_err = require_admin_scope(request)
            if scope_err is not None:
                return scope_err

        quota_error = await enforce_key_quota(request, key_record)
        if quota_error is not None:
            return quota_error
        return await call_next(request)
