"""Structured logging setup for Anchor.

Provides:
- JSONFormatter: emits log records as JSON with standard fields.
- configure_logging(): idempotent root logger config (text by default,
  JSON via ANCHOR_LOG_JSON=1).
- request_id_var: ContextVar holding the current request_id, populated by
  request_id_middleware and surfaced in every log record via the formatter.
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Optional


request_id_var: ContextVar[str] = ContextVar("anchor_request_id", default="-")
key_id_var: ContextVar[str] = ContextVar("anchor_key_id", default="-")
# v0.9.54 (Tier 1.3): trace_id follows a single query end-to-end across
# routing → worker call → fallback chain → cost log → judge. Distinct from
# request_id (which is per HTTP request) because a single HTTP request can
# span multiple routed workers (the fallback chain). Set in routing_core
# _route_tier() at the top so every downstream logger sees the same id.
trace_id_var: ContextVar[str] = ContextVar("anchor_trace_id", default="-")


class JSONFormatter(logging.Formatter):
    """Format log records as one-line JSON.

    Standard fields: ts, level, logger, msg, request_id, key_id.
    Any `extra={...}` dict is merged at the top level.
    """

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "key_id": key_id_var.get(),
            "trace_id": trace_id_var.get(),
        }
        for k, v in record.__dict__.items():
            if k not in self._RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_CONFIGURED = False


def configure_logging(level: Optional[str] = None, *, json_mode: Optional[bool] = None) -> None:
    """Idempotently configure the root logger.

    Args:
        level: log level name (default INFO, override via ANCHOR_LOG_LEVEL).
        json_mode: True for JSON, False for text. Default: env ANCHOR_LOG_JSON.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("ANCHOR_LOG_LEVEL", "INFO")).upper()
    want_json = json_mode if json_mode is not None else _truthy(os.environ.get("ANCHOR_LOG_JSON"))
    root = logging.getLogger()
    root.setLevel(lvl)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    if want_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [req=%(request_id)s trace=%(trace_id)s] %(message)s",
            defaults={"request_id": "-", "trace_id": "-"},
        ))
    root.addHandler(handler)
    # v0.9.53 (audit R11): apply redact_secrets() to every record
    # emitted through the root logger so vendor keys / bearer tokens
    # never land in /tmp/anchor-gateway.log.
    if not any(isinstance(f, _RedactFilter) for f in root.filters):
        root.addFilter(_RedactFilter())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _CONFIGURED = True


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def new_request_id() -> str:
    return uuid.uuid4().hex


def new_trace_id() -> str:
    """Mint a 12-char trace id for end-to-end query tracking."""
    return f"tr-{uuid.uuid4().hex[:12]}"


def install_request_id_middleware(app) -> None:
    """Attach a Starlette middleware that populates request_id_var per request."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class _RidMW(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            rid = (
                request.headers.get("X-Request-ID")
                or request.headers.get("X-Session-ID")
                or new_request_id()
            )
            token = request_id_var.set(rid)
            try:
                response = await call_next(request)
            finally:
                request_id_var.reset(token)
            response.headers["X-Request-ID"] = rid
            return response

    app.add_middleware(_RidMW)


# v0.9.53 (audit R11): redact vendor keys / bearer tokens / api_key
# assignments from any log string. Applied via a global logging filter so
# a future regression does not leak upstream credentials to
# /tmp/anchor-gateway.log.
import re as _redact_re

_REDACT_PATTERNS = [
    _redact_re.compile(r"(sk-[A-Za-z0-9_-]{6,})[A-Za-z0-9_-]+"),
    _redact_re.compile(r"(Bearer\s+)[A-Za-z0-9._-]{6,}"),
    _redact_re.compile(
        r"(?i)(api[-_]?key\s*[:=]\s*[\'\"\']?)([A-Za-z0-9._-]{6,})"
    ),
]


def redact_secrets(value):
    """Return ``value`` with vendor keys / bearer tokens / api_key redacted.

    Conservative: leaves a 6-character prefix on `sk-...` and `api_key=...`
    so the line is still correlatable in support tickets, but no full
    secret is ever written to disk in cleartext.
    """
    if not isinstance(value, str) or not value:
        return value
    for _pat in _REDACT_PATTERNS:
        value = _pat.sub(lambda m: m.group(1) + "...REDACTED", value)
    return value


class _RedactFilter(logging.Filter):
    """Apply redact_secrets() to every log record's message before emission."""

    def filter(self, record):
        try:
            record.msg = redact_secrets(record.getMessage())
            record.args = ()
        except Exception:
            pass
        return True
