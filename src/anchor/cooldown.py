from anchor.workers import FABLE5, OPUS, SONNET
"""Day 17 (Fable 5 decision #5, 2026-07-06): per-worker cooldown.

When a worker returns a 429 / 5xx / rate-limit error, mark it
'cooling down' for COOLDOWN_SECS. During cooldown, the router
skips the worker and tries the next-best fallback in the same tier.

State: in-memory dict {worker_name: cooldown_until_epoch}.
State is process-local (no need to persist across restarts).
"""
import threading
import time
from typing import Optional

COOLDOWN_SECS = 300  # 5 min — short enough to recover fast, long enough to dodge cascading 429


_state: dict[str, float] = {}
_lock = threading.Lock()


def trip(worker_name: str, reason: str = "error", *, seconds: Optional[int] = None) -> None:
    """Mark worker as cooling down until now + duration.

    Transient timeouts/connection errors use a short cool (default 45s) so a
    single slow primary does not free-lane hard traffic for 5 minutes.
    Rate-limit / 429 / persistent issues keep the longer worker-specific cool.

    v0.9.70: optional ``seconds=`` keyword allows callers (e.g. AKRL cooldown
    at routing_core.py:_route_tier) to override the per-reason cooldown with
    an explicit retry_after_s value derived from upstream.
    """
    if seconds is not None:
        cooldown_secs = max(1, int(seconds))
    else:
        reason_l = (reason or "").lower()
        transient = (
            "timeout" in reason_l
            or "connection" in reason_l
            or reason_l in {"timeoutError".lower(), "apitimeouterror", "apiconnectionerror"}
        )
        if transient:
            cooldown_secs = WORKER_TIMEOUT_COOLDOWN.get(worker_name, TIMEOUT_COOLDOWN_SECS)
        else:
            cooldown_secs = WORKER_SPECIFIC_COOLDOWN.get(worker_name, COOLDOWN_SECS)
    with _lock:
        _state[worker_name] = time.time() + cooldown_secs


def is_cooling(worker_name: str) -> bool:
    with _lock:
        until = _state.get(worker_name)
        if until is None:
            return False
        now = time.time()
        if now >= until:
            _state.pop(worker_name, None)
            return False
        return True


def remaining(worker_name: str) -> int:
    with _lock:
        until = _state.get(worker_name)
    if until is None:
        return 0
    return max(0, int(until - time.time()))


def all_cooling() -> dict[str, int]:
    now = time.time()
    with _lock:
        snapshot = list(_state.items())
        expired = [worker_name for worker_name, until in snapshot if now >= until]
        for worker_name in expired:
            _state.pop(worker_name, None)
    return {
        worker_name: max(0, int(until - now))
        for worker_name, until in snapshot
        if worker_name not in expired
    }


def reset() -> None:
    """Clear all cooldowns (for testing)."""
    with _lock:
        _state.clear()


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect 429 / rate-limit / quota / overflow / all-keys-failed style errors."""
    msg = (str(exc) or "").lower()
    name = type(exc).__name__.lower()
    triggers = (
        "429", "rate", "limit", "quota", "throttl", "overload",
        "ratelimiterror", "apierror", "serviceunavailable",
        "all keys failed", "all_keys_failed",  # KeyPool multi-key failover exhaustion
    )
    return any(t in msg for t in triggers) or any(t in name for t in triggers)

# v0.9.20 (Fable 5 directive): worker-specific long cooldown to prevent repeat hits
# on workers with persistent upstream issues. fable-5 on baosiapi returns vendor
# placeholder for ~100% of requests as of 2026-07-09; skip it for 10min after first hit.
# v0.9.33: deepseek-v4-flash timeout cooldown 60s — transient upstream timeouts
# observed 2026-07-11 (5 occurrences in /tmp/uvicorn_v0.9.23.log). Without cooldown
# trip, M3 absorbs the fallback traffic and inflates cost. 60s balances recovery
# vs cascading 429 from sustained upstream issues.
TIMEOUT_COOLDOWN_SECS = 20  # transient timeout/connection — keep steel in pool
WORKER_TIMEOUT_COOLDOWN = {
    "gpt-5.6-sol": 15,
    "grok-4-6-reasoning": 15,
    "claude-fable-5": 20,
    "minimax-m3": 15,
    "deepseek-v4-flash": 30,
    "deepseek-v4-pro": 15,
}
WORKER_SPECIFIC_COOLDOWN = {
    FABLE5: 900,
    OPUS: 600,
    SONNET: 600,
    "deepseek-v4-flash": 120,
}


# ---------------------------------------------------------------------------
# v0.9.70: empty-rate cooldown (defense against transient upstream storms)
# ---------------------------------------------------------------------------
#
# Background (2026-08-08): deepseek-v4-flash (OpenCode Zen proxy) returned
# empty content for 132/213 calls (62%) during a 5-minute upstream outage.
# The per-call cooldown + same-worker retry could not recover: each empty
# call cost 3-4s on flash + the fallback chain on M3/Grok/Sol. Solution:
# detect an elevated empty-rate from the recent session log and trip a
# longer cooldown (default 10min) so subsequent requests skip flash
# entirely instead of burning fallback quota.
#
# The query is intentionally cheap (one pass over the recent JSONL rows)
# and gated on a caller flag (we only invoke when an empty was just
# observed), so cost is bounded by empty-event frequency, not by request
# volume.

EMPTY_RATE_DEFAULT_THRESHOLD = 0.40  # 40% empty in window trips cooldown
EMPTY_RATE_DEFAULT_MIN_N = 10        # need at least 10 samples
EMPTY_RATE_DEFAULT_WINDOW_S = 120    # last 2 minutes
EMPTY_RATE_DEFAULT_COOLDOWN_S = 600  # 10 minutes once tripped


def _count_recent_empty_rate(
    worker_name: str,
    *,
    window_s: float = EMPTY_RATE_DEFAULT_WINDOW_S,
    sessions: Optional[list[dict]] = None,
) -> tuple[int, int]:
    """Return (empty_count, total_count) for ``worker_name`` in the last
    ``window_s`` seconds.

    Reads from JSONL session log (cheap; one pass over recent rows).
    When ``sessions`` is provided (test injection), skip I/O and filter
    in-memory. Returns (0, 0) when no data is available.    """
    import time as _t

    cutoff = _t.time() - float(window_s)
    if sessions is None:
        try:
            from anchor.session_log import SESSIONS_DIR as _SD
            from datetime import datetime, timedelta, timezone as _tz
            import json as _json
            today = datetime.now(_tz.utc).date()
            cutoff_day = today - timedelta(days=2)
            rows: list[dict] = []
            if not _SD.exists():
                return 0, 0
            for fp in sorted(_SD.glob("*.jsonl")):
                try:
                    d = datetime.strptime(fp.stem, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if d < cutoff_day:
                    continue
                with open(fp) as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            rows.append(_json.loads(ln))
                        except Exception:
                            continue
        except Exception as _e:
            return 0, 0
    else:
        rows = sessions

    total = 0
    empty = 0
    for rec in rows:
        try:
            if rec.get("model_used") != worker_name:
                continue
            ts = float(rec.get("ts") or 0.0)
            if ts < cutoff:
                continue
            total += 1
            if rec.get("error_kind") == "empty":
                empty += 1
        except Exception:
            continue
    return empty, total


def trip_empty_rate(
    worker_name: str,
    *,
    threshold: float = EMPTY_RATE_DEFAULT_THRESHOLD,
    min_n: int = EMPTY_RATE_DEFAULT_MIN_N,
    window_s: float = EMPTY_RATE_DEFAULT_WINDOW_S,
    cooldown_s: int = EMPTY_RATE_DEFAULT_COOLDOWN_S,
    sessions: Optional[list[dict]] = None,
) -> bool:
    """Inspect the recent empty-rate for ``worker_name``; if it exceeds
    ``threshold`` over at least ``min_n`` samples in the last ``window_s``
    seconds, trip a cooldown of ``cooldown_s`` seconds.

    Returns True iff a cooldown was tripped. Idempotent: re-calling within
    the cooldown window does NOT extend the cooldown (caller-side concern;
    the underlying ``trip()`` always sets an absolute deadline).

    Sessions may be injected for tests via ``sessions=``; production callers
    rely on the JSONL session log. Any I/O or parsing failure returns False
    (fail-open: we would rather burn one fallback call than miss a real
    empty-rate spike).
    """
    empty, total = _count_recent_empty_rate(
        worker_name, window_s=window_s, sessions=sessions
    )
    if total < int(min_n):
        return False
    rate = (empty / total) if total else 0.0
    if rate < float(threshold):
        return False
    trip(worker_name, reason="empty_rate_spike", seconds=int(cooldown_s))
    return True
