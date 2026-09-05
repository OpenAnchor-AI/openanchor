"""Singleton pool for httpx.AsyncClient and openai.AsyncOpenAI clients.

Prevents connection/FD leaks by reusing clients across requests.
Per-process singleton keyed by (base_url, api_key) so each worker+key
gets exactly one httpx.AsyncClient + openai.AsyncOpenAI.

Usage:
    from anchor.clients.pool import get_openai_client, shutdown
    client = get_openai_client(base_url="...", api_key="...")
    resp = await client.chat.completions.create(...)

Shutdown:
    await shutdown()  # closes all pooled clients (idempotent)
"""
from __future__ import annotations
import logging
from typing import Mapping, Optional

import httpx
import openai

_logger = logging.getLogger(__name__)

# (base_url, api_key) -> openai.AsyncOpenAI
_pool: dict[tuple[str, str], openai.AsyncOpenAI] = {}

_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)
_DEFAULT_TIMEOUT = httpx.Timeout(60.0, pool=None)


def _make_http_client(
    timeout: float = 60.0,
    headers: Optional[Mapping[str, str]] = None,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, pool=None),
        limits=_LIMITS,
        trust_env=False,
        headers=dict(headers) if headers else None,
    )


def get_openai_client(
    base_url: str,
    api_key: str,
    timeout: float = 60.0,
    max_retries: int = 1,
    headers: Optional[Mapping[str, str]] = None,
) -> openai.AsyncOpenAI:
    key = (base_url, api_key)
    if key not in _pool:
        # Local / no-auth endpoints (e.g. ollama) pass api_key=""; the SDK
        # rejects an empty api_key at construction, so substitute a placeholder
        # only for building the client (requests still carry no auth for local).
        _sdk_key = api_key or "local"
        # P29-UA: do NOT pass a custom http_client. When an httpx.AsyncClient
        # is injected, the openai SDK merges headers differently and the
        # User-Agent from default_headers gets overridden by the httpx
        # python-httpx UA -> OpenCode Zen free models return 429. Let the
        # SDK build its own transport so default_headers survives.
        _pool[key] = openai.AsyncOpenAI(
            api_key=_sdk_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=dict(headers) if headers else None,
        )
    return _pool[key]


def get_pool_size() -> int:
    return len(_pool)


async def shutdown() -> None:
    for key, client in list(_pool.items()):
        try:
            await client.close()
        except Exception:
            _logger.exception("pool close error: %s", key)
    _pool.clear()
