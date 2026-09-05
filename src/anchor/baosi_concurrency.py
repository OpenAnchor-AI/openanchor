"""Process-wide concurrency gate for BaosiAPI workers.

All BaosiAPI channels share one semaphore: the upstream quota is shared across
model families, so per-worker gates are insufficient.  The gate is deliberately
small and dependency-free so it can be used by both routing and clients.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

BAOSIAPI_MAX_CONCURRENCY = 30

_gate = asyncio.Semaphore(BAOSIAPI_MAX_CONCURRENCY)
_inflight = 0
_inflight_lock = asyncio.Lock()


def is_baosi_channel(channel: str | None) -> bool:
    return bool(channel and channel.startswith("baosiapi"))


def should_shunt() -> bool:
    """Return whether new BaosiAPI work should be routed to Tier 1."""
    try:
        threshold = max(1, min(BAOSIAPI_MAX_CONCURRENCY, int(
            os.environ.get("ANCHOR_BAOSI_SHUNT_THRESHOLD", "28")
        )))
    except ValueError:
        threshold = 28
    return _inflight >= threshold


async def inflight() -> int:
    async with _inflight_lock:
        return _inflight


@asynccontextmanager
async def slot():
    """Acquire one global BaosiAPI slot and release it on every exit path."""
    global _inflight
    await _gate.acquire()
    async with _inflight_lock:
        _inflight += 1
    try:
        yield
    finally:
        async with _inflight_lock:
            _inflight -= 1
        _gate.release()


def reset_for_tests() -> None:
    """Reset the process-local gate after isolated unit tests."""
    global _gate, _inflight
    _gate = asyncio.Semaphore(BAOSIAPI_MAX_CONCURRENCY)
    _inflight = 0
