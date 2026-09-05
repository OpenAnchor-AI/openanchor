"""In-process LRU cache for oracle responses.

Keyed by hash(prompt + model + ts_bucket(3600)) so repeated judge
comparisons for the same prompt within 1 hour reuse the cached oracle
response instead of re-sampling. v0.9.53 (P0-1 fix): cache uses
ORACLE_MODEL (default minimax-m3) instead of hardcoded Opus-4-8 to
match the new oracle/judge decoupling.

Usage:
    from anchor.judge_cache import get_cached_oracle_response, cache_oracle_response
    cached = get_cached_oracle_response(query)
    if cached is None:
        response = await call_oracle(query)
        cache_oracle_response(query, response)
    else:
        response = cached
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict

# v0.9.53: cache key includes ORACLE_MODEL so swapping the oracle does
# not poison the cache with old model responses.
try:
    from anchor.config import ORACLE_MODEL as _DEFAULT_ORACLE
except Exception:  # pragma: no cover - defensive for early-import cycles
    _DEFAULT_ORACLE = "minimax-m3"

_LRU_MAX = 4096
_lru: OrderedDict[str, str] = OrderedDict()
_lru_hits = 0
_lru_misses = 0


def _ts_bucket(ts: float | None = None, bucket_s: int = 3600) -> int:
    return int((ts if ts is not None else time.time()) // bucket_s)


def _oracle_cache_key(prompt: str, model: str, ts_bucket: int) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(ts_bucket).encode("utf-8"))
    return h.hexdigest()


def get_cached_oracle_response(prompt: str, model: str | None = None) -> str | None:
    global _lru_hits, _lru_misses
    model = model or _DEFAULT_ORACLE
    bucket = _ts_bucket()
    k = _oracle_cache_key(prompt, model, bucket)
    if k in _lru:
        _lru.move_to_end(k)
        _lru_hits += 1
        return _lru[k]
    _lru_misses += 1
    return None


def cache_oracle_response(prompt: str, response: str, model: str | None = None) -> None:
    model = model or _DEFAULT_ORACLE
    bucket = _ts_bucket()
    k = _oracle_cache_key(prompt, model, bucket)
    _lru[k] = response
    _lru.move_to_end(k)
    while len(_lru) > _LRU_MAX:
        _lru.popitem(last=False)


def oracle_cache_stats() -> dict:
    return {"hits": _lru_hits, "misses": _lru_misses, "size": len(_lru), "maxsize": _LRU_MAX}


def clear_oracle_cache() -> None:
    _lru.clear()
    _lru_hits = 0
    _lru_misses = 0


__all__ = [
    "get_cached_oracle_response",
    "cache_oracle_response",
    "oracle_cache_stats",
    "clear_oracle_cache",
]
