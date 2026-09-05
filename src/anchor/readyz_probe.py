"""Readiness worker probe (v0.9.51+).

Modes via ANCHOR_READYZ_PROBE:
  - chat: real min chat completion against an enabled worker (true readiness)
  - key: at least one enabled worker has a non-empty API key
  - http: key present AND base_url accepts TCP/HTTP within timeout
            (HTTP 401/403 still counts as "up")
  - off: presence of any enabled worker (legacy; not for production)

Default when unset:
  - production → chat
  - debug/test or ANCHOR_AUTH_DISABLED → key (fast tests)
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import Any
from urllib.parse import urlparse

from anchor.config import Worker, _read_api_key

_logger = logging.getLogger("anchor.readyz")


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


def worker_has_api_key(worker: Worker) -> bool:
    if worker.kind == "opencode-cli":
        # Local CLI — no cloud key required for "configured"
        return True
    envs = worker.api_key_envs or ((worker.api_key_env,) if worker.api_key_env else ())
    for envn in envs:
        if envn and (_read_api_key(envn) or "").strip():
            return True
    return False


def _tcp_reachable(base_url: str, timeout: float = 2.0) -> bool:
    if not base_url:
        return False
    try:
        parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _http_reachable(base_url: str, timeout: float = 2.0) -> bool:
    if not base_url:
        return False
    try:
        import httpx
    except ImportError:
        return await asyncio.to_thread(_tcp_reachable, base_url, timeout)
    url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            # Any HTTP response means the endpoint is reachable (incl 401/404)
            r = await client.get(url)
            return r.status_code < 500 or r.status_code in (401, 403, 404)
    except Exception:
        return await asyncio.to_thread(_tcp_reachable, base_url, timeout)


def probe_mode() -> str:
    raw = (os.environ.get("ANCHOR_READYZ_PROBE") or "").strip().lower()
    if raw in {"key", "http", "off", "presence", "chat"}:
        return "off" if raw == "presence" else raw
    # unset: production wants real chat probe; debug/tests keep fast key mode
    profile = (os.environ.get("ANCHOR_PROFILE") or "").strip().lower()
    if profile in {"debug", "test"} or _truthy(os.environ.get("ANCHOR_AUTH_DISABLED")):
        return "key"
    return "chat"



async def _chat_reachable(worker: Worker, timeout: float = 4.0) -> bool:
    """True if worker answers a 1-token chat (or tool-compatible empty with 2xx)."""
    if not worker_has_api_key(worker):
        return False
    try:
        from anchor.clients.factory import build_client
        client = build_client(worker)
        resp = await asyncio.wait_for(
            client.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            ),
            timeout=timeout,
        )
        if not isinstance(resp, dict):
            return False
        # content may be empty on some reasoning models; presence of id/usage counts
        if resp.get("content") or resp.get("id") or resp.get("usage") is not None:
            return True
        if resp.get("tool_calls"):
            return True
        return False
    except Exception as exc:
        _logger.warning("READY_PROBE_CHAT_ERR worker=%s err=%s", worker.name, exc)
        return False


async def probe_enabled_workers(workers: list[Worker], *, timeout: float = 2.5) -> dict[str, Any]:
    """Probe cheapest-first enabled workers. Returns status dict."""
    mode = probe_mode()
    if not workers:
        return {
            "ok": False,
            "mode": mode,
            "n_workers_pinged": 0,
            "n_ponged": 0,
            "reason": "no_enabled_workers",
            "worker": None,
        }
    if mode == "off":
        return {
            "ok": True,
            "mode": mode,
            "n_workers_pinged": 1,
            "n_ponged": 1,
            "reason": "presence_only",
            "worker": workers[0].name,
        }

    # Prefer workers that already have keys
    def _probe_rank(w: Worker) -> tuple:
        # Prefer key-ready cheap/fast workers for chat probe (flash → m3 → others)
        prefer = {"deepseek-v4-flash": 0, "minimax-m3": 1, "gpt-5.6-sol": 2, "grok-4-6-reasoning": 3}
        return (0 if worker_has_api_key(w) else 1, prefer.get(w.name, 10), w.slot)
    ordered = sorted(workers, key=_probe_rank)
    pinged = 0
    for w in ordered[:3]:
        pinged += 1
        has_key = worker_has_api_key(w)
        if mode == "key":
            if has_key:
                return {
                    "ok": True,
                    "mode": mode,
                    "n_workers_pinged": pinged,
                    "n_ponged": 1,
                    "reason": "key_present",
                    "worker": w.name,
                }
            continue
        if not has_key:
            continue
        if mode == "chat":
            try:
                up = await _chat_reachable(w, timeout=max(timeout, 4.0))
            except Exception as exc:
                _logger.warning("READY_PROBE_CHAT_ERR worker=%s err=%s", w.name, exc)
                up = False
            if up:
                return {
                    "ok": True,
                    "mode": mode,
                    "n_workers_pinged": pinged,
                    "n_ponged": 1,
                    "reason": "chat_pong",
                    "worker": w.name,
                }
            continue
        # http mode
        try:
            up = await _http_reachable(w.base_url, timeout=timeout)
        except Exception as exc:
            _logger.warning("READY_PROBE_HTTP_ERR worker=%s err=%s", w.name, exc)
            up = False
        if up:
            return {
                "ok": True,
                "mode": mode,
                "n_workers_pinged": pinged,
                "n_ponged": 1,
                "reason": "http_up",
                "worker": w.name,
            }

    # Soft mode for local/unit tests: auth-disabled (debug) or ANCHOR_READYZ_SOFT=1.
    # Production must leave ANCHOR_AUTH_DISABLED unset so this path is skipped.
    soft_env = os.environ.get("ANCHOR_READYZ_SOFT")
    if soft_env is not None:
        soft = _truthy(soft_env)
    else:
        soft = (
            _truthy(os.environ.get("ANCHOR_AUTH_DISABLED"))
            and os.environ.get("ANCHOR_PROFILE", "debug") in {"debug", "test", ""}
        )
    if soft:
        return {
            "ok": True,
            "mode": mode,
            "n_workers_pinged": pinged,
            "n_ponged": 0,
            "reason": "debug_soft",
            "worker": workers[0].name if workers else None,
        }

    return {
        "ok": False,
        "mode": mode,
        "n_workers_pinged": pinged,
        "n_ponged": 0,
        "reason": "no_key_or_upstream",
        "worker": ordered[0].name if ordered else None,
    }
