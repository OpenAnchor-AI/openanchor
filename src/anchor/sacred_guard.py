"""v0.9.51+: sacred cost guard with multi-process backend.

- Per-API-key soft cap ¥5 → warn
- Per-API-key hard cap ¥10 → fallback away from sacred workers
- Spend buckets bind to API key fingerprint (not client-supplied X-Session-ID)
- Backends:
    memory (default single-process)
    sqlite (shared across local uvicorn workers; auto when WEB_CONCURRENCY>1)
    redis  (multi-host; ANCHOR_REDIS_URL / REDIS_URL; optional FakeRedis for tests)
    set ANCHOR_SACRED_BACKEND=memory|sqlite|redis explicitly to override
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Optional, Protocol

from anchor import _ROOT as _ANCHOR_ROOT
from anchor.workers import SACRED

_logger = logging.getLogger("anchor.sacred_guard")

SOFT_CAP_YUAN = 5.0
HARD_CAP_YUAN = 10.0
_DEFAULT_DB = Path(str(_ANCHOR_ROOT)) / "data" / "sacred_spend.db"


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def resolve_sacred_bucket(
    api_key: str | None = None,
    client_session_id: str | None = None,
) -> str:
    """Derive spend bucket id.

    Prefer API-key fingerprint so clients cannot rotate X-Session-ID to bypass
    the hard cap. Client session id is ignored for cap accounting when a key
    is present (still accepted for logging elsewhere).
    """
    if api_key:
        fp = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"key:{fp}"
    if client_session_id:
        return f"sess:{client_session_id}"
    return "anon"


class _Store(Protocol):
    def add(self, bucket: str, cost_yuan: float) -> float: ...
    def get(self, bucket: str) -> float: ...
    def reset(self, bucket: str) -> None: ...
    def snapshot(self) -> tuple[dict[str, float], dict[str, float]]: ...


class _MemoryStore:
    def __init__(self) -> None:
        self._spend: dict[str, float] = defaultdict(float)
        self._daily: dict[str, float] = defaultdict(float)
        self._lock = Lock()

    def add(self, bucket: str, cost_yuan: float) -> float:
        with self._lock:
            self._spend[bucket] = self._spend.get(bucket, 0.0) + cost_yuan
            day = _today()
            self._daily[day] = self._daily.get(day, 0.0) + cost_yuan
            return self._spend[bucket]

    def get(self, bucket: str) -> float:
        with self._lock:
            return float(self._spend.get(bucket, 0.0))

    def reset(self, bucket: str) -> None:
        with self._lock:
            self._spend.pop(bucket, None)

    def snapshot(self) -> tuple[dict[str, float], dict[str, float]]:
        with self._lock:
            return (
                {k: round(v, 4) for k, v in self._spend.items()},
                dict(self._daily),
            )

    def clear_all(self) -> None:
        with self._lock:
            self._spend.clear()
            self._daily.clear()


class _SqliteStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = Path(
            db_path or os.environ.get("ANCHOR_SACRED_DB", str(_DEFAULT_DB))
        )
        self._lock = Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sacred_bucket (
                        bucket TEXT PRIMARY KEY,
                        yuan REAL NOT NULL DEFAULT 0.0
                    );
                    CREATE TABLE IF NOT EXISTS sacred_daily (
                        day TEXT PRIMARY KEY,
                        yuan REAL NOT NULL DEFAULT 0.0
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def add(self, bucket: str, cost_yuan: float) -> float:
        day = _today()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO sacred_bucket(bucket, yuan) VALUES (?, ?)
                    ON CONFLICT(bucket) DO UPDATE SET yuan = yuan + excluded.yuan
                    """,
                    (bucket, float(cost_yuan)),
                )
                conn.execute(
                    """
                    INSERT INTO sacred_daily(day, yuan) VALUES (?, ?)
                    ON CONFLICT(day) DO UPDATE SET yuan = yuan + excluded.yuan
                    """,
                    (day, float(cost_yuan)),
                )
                cur = conn.execute(
                    "SELECT yuan FROM sacred_bucket WHERE bucket=?", (bucket,)
                )
                row = cur.fetchone()
                total = float(row[0]) if row else float(cost_yuan)
                conn.commit()
                return total
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def get(self, bucket: str) -> float:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT yuan FROM sacred_bucket WHERE bucket=?", (bucket,)
                )
                row = cur.fetchone()
                return float(row[0]) if row else 0.0
            finally:
                conn.close()

    def reset(self, bucket: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM sacred_bucket WHERE bucket=?", (bucket,))
                conn.commit()
            finally:
                conn.close()

    def snapshot(self) -> tuple[dict[str, float], dict[str, float]]:
        with self._lock:
            conn = self._connect()
            try:
                buckets = {
                    str(r[0]): round(float(r[1]), 4)
                    for r in conn.execute("SELECT bucket, yuan FROM sacred_bucket")
                }
                daily = {
                    str(r[0]): float(r[1])
                    for r in conn.execute("SELECT day, yuan FROM sacred_daily")
                }
                return buckets, daily
            finally:
                conn.close()

    def clear_all(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM sacred_bucket")
                conn.execute("DELETE FROM sacred_daily")
                conn.commit()
            finally:
                conn.close()


class _RedisStore:
    """Multi-host sacred spend store via Redis.

    Keys:
      anchor:sacred:bucket:{id}  float yuan
      anchor:sacred:daily:{day}  float yuan
    """

    def __init__(self, url: str = "", client=None) -> None:
        if client is not None:
            self._r = client
        else:
            import redis  # type: ignore
            self._r = redis.Redis.from_url(url, decode_responses=True)
        self._r.ping()

    def add(self, bucket: str, cost_yuan: float) -> float:
        bkey = f"anchor:sacred:bucket:{bucket}"
        dkey = f"anchor:sacred:daily:{_today()}"
        pipe = self._r.pipeline()
        pipe.incrbyfloat(bkey, float(cost_yuan))
        pipe.incrbyfloat(dkey, float(cost_yuan))
        pipe.expire(dkey, 86400 * 14)
        total, _, _ = pipe.execute()
        return float(total)

    def get(self, bucket: str) -> float:
        raw = self._r.get(f"anchor:sacred:bucket:{bucket}")
        return float(raw or 0.0)

    def reset(self, bucket: str) -> None:
        self._r.delete(f"anchor:sacred:bucket:{bucket}")

    def snapshot(self) -> tuple[dict[str, float], dict[str, float]]:
        buckets: dict[str, float] = {}
        daily: dict[str, float] = {}
        for key in self._r.keys("anchor:sacred:bucket:*") or []:
            ks = str(key)
            bid = ks.split("anchor:sacred:bucket:", 1)[-1]
            buckets[bid] = round(float(self._r.get(ks) or 0.0), 4)
        for key in self._r.keys("anchor:sacred:daily:*") or []:
            ks = str(key)
            day = ks.split("anchor:sacred:daily:", 1)[-1]
            daily[day] = float(self._r.get(ks) or 0.0)
        return buckets, daily

    def clear_all(self) -> None:
        keys = list(self._r.keys("anchor:sacred:bucket:*") or [])
        keys += list(self._r.keys("anchor:sacred:daily:*") or [])
        if keys:
            self._r.delete(*keys)



_store: Optional[_Store] = None
_store_lock = Lock()

# Back-compat aliases for tests that poke module internals
_SACRED_SPEND: dict[str, float] = {}
_SACRED_DAILY: dict[str, float] = {}
_SACRED_SPEND_LOCK = Lock()


def _select_backend_kind() -> str:
    explicit = (os.environ.get("ANCHOR_SACRED_BACKEND") or "").strip().lower()
    if explicit in {"memory", "sqlite", "redis"}:
        return explicit
    workers = 1
    for envn in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "ANCHOR_WORKERS"):
        try:
            workers = max(workers, int(os.environ.get(envn) or "1"))
        except ValueError:
            pass
    # audit 2026-08-16 (E6): default was "memory" for single-worker deploys —
    # the ¥10 sacred cap evaporated on every restart. Default to sqlite so the
    # spend ledger (and the cap) survives restarts; opt out with
    # ANCHOR_SACRED_BACKEND=memory if process-local semantics are wanted.
    return "sqlite"


def get_store() -> _Store:
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        kind = _select_backend_kind()
        if kind == "redis":
            url = os.environ.get("ANCHOR_REDIS_URL") or os.environ.get("REDIS_URL") or ""
            fake = os.environ.get("ANCHOR_SACRED_REDIS_FAKE", "").strip()
            if fake:
                try:
                    from anchor._fake_redis import FakeRedis
                    _store = _RedisStore(url or "redis://fake", client=FakeRedis())
                    _logger.info("SACRED_BACKEND redis (fake)")
                    return _store
                except Exception as exc:
                    _logger.warning("SACRED_REDIS_FAKE_FAIL fallback=sqlite err=%s", exc)
                    kind = "sqlite"
            elif url:
                try:
                    _store = _RedisStore(url)
                    _logger.info("SACRED_BACKEND redis")
                    return _store
                except Exception as exc:
                    _logger.warning("SACRED_REDIS_FAIL fallback=sqlite err=%s", exc)
                    kind = "sqlite"
            else:
                _logger.warning("SACRED_REDIS_NO_URL fallback=sqlite")
                kind = "sqlite"
        if kind == "sqlite":
            try:
                _store = _SqliteStore()
                _logger.info("SACRED_BACKEND sqlite path=%s", _DEFAULT_DB)
                return _store
            except Exception as exc:
                _logger.warning("SACRED_SQLITE_FAIL fallback=memory err=%s", exc)
        _store = _MemoryStore()
        _logger.info("SACRED_BACKEND memory")
        return _store


def reset_backend_for_tests(*, clear: bool = True) -> None:
    """Drop singleton so next call re-reads env.

    clear=True (default): wipe store contents (unit test isolation).
    clear=False: only drop singleton — used to re-open same sqlite path
    as another "process".
    """
    global _store
    with _store_lock:
        if clear and _store is not None and hasattr(_store, "clear_all"):
            try:
                _store.clear_all()  # type: ignore[attr-defined]
            except Exception:
                pass
        _store = None
    with _SACRED_SPEND_LOCK:
        _SACRED_SPEND.clear()
        _SACRED_DAILY.clear()


def record_sacred_spend(session_id: str, worker: str, cost_yuan: float) -> dict:
    """Record a sacred (fable-5 or opus-4-8) spend, return guard status.

    ``session_id`` should be a bucket from ``resolve_sacred_bucket`` (callers
    that pass raw client ids still work, but production paths should bind).
    """
    if worker not in SACRED:
        return {"guarded": False}
    sid = session_id or f"anon-{int(time.time())}"
    total = get_store().add(sid, float(cost_yuan))
    # keep legacy dict roughly in sync for any external introspection
    with _SACRED_SPEND_LOCK:
        _SACRED_SPEND[sid] = total
    return {
        "guarded": total >= SOFT_CAP_YUAN,
        "soft_cap_hit": total >= SOFT_CAP_YUAN,
        "hard_cap_hit": total >= HARD_CAP_YUAN,
        "session_sacred_yuan": round(total, 4),
        "bucket": sid,
    }


def check_sacred_guard(session_id: str) -> dict:
    """Check if bucket is over cap. Returns {allowed, fallback_required, total}."""
    sid = session_id or "anon"
    total = get_store().get(sid)
    if total >= HARD_CAP_YUAN:
        return {
            "allowed": False,
            "fallback_required": True,
            "session_sacred_yuan": round(total, 4),
            "bucket": sid,
        }
    if total >= SOFT_CAP_YUAN:
        return {
            "allowed": True,
            "warn": True,
            "fallback_required": False,
            "session_sacred_yuan": round(total, 4),
            "bucket": sid,
        }
    return {
        "allowed": True,
        "warn": False,
        "fallback_required": False,
        "session_sacred_yuan": round(total, 4),
        "bucket": sid,
    }


def get_sacred_stats(days: int = 7) -> dict:
    sessions, daily = get_store().snapshot()
    total_yuan = sum(sessions.values())
    over_soft = sum(1 for v in sessions.values() if v >= SOFT_CAP_YUAN)
    over_hard = sum(1 for v in sessions.values() if v >= HARD_CAP_YUAN)
    return {
        "total_sacred_yuan": round(total_yuan, 4),
        "n_sessions": len(sessions),
        "over_soft_cap": over_soft,
        "over_hard_cap": over_hard,
        "soft_cap_yuan": SOFT_CAP_YUAN,
        "hard_cap_yuan": HARD_CAP_YUAN,
        "daily": daily,
        "top_sessions": sorted(sessions.items(), key=lambda x: -x[1])[:10],
        "backend": type(get_store()).__name__,
    }


def reset_session(session_id: str) -> None:
    get_store().reset(session_id)
    with _SACRED_SPEND_LOCK:
        _SACRED_SPEND.pop(session_id, None)


# V2 A2.2 audit 2026-08-15: Layer 3 opt-in autopromote (B5 audit 决议保留 default OFF)
import os as _os_a22


def maybe_autopromote(
    role: str,
    primary: str,
    daily_cost: float | None = None,
    daily_cap: float | None = None,
) -> str | None:
    """Opt-in autopromote: when daily budget 充足, upgrade hard role to Fable.

    B5 audit 2026-08-15 否决决议保留: 默认 OFF.
    启用: ANCHOR_AUTOPROMOTE=1

    触发条件 (全部满足):
      1. ANCHOR_AUTOPROMOTE=1
      2. role == "hard"
      3. daily_cost < daily_cap * 0.8 (剩余 20% budget)
      4. Fable 不在 cooldown

    Returns:
      "claude-fable-5" if all conditions met, else None
    """
    # 1. opt-in env gate
    enabled = _os_a22.environ.get("ANCHOR_AUTOPROMOTE", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None

    # 2. role check
    if role != "hard":
        return None

    # 3. budget check (if caller provided)
    if daily_cost is not None or daily_cap is not None:
        # caller 给了 budget 信息但 cap 无效 (0 / 负) -> 拒绝升级
        if daily_cap is None or daily_cap <= 0:
            return None
        if daily_cost >= daily_cap * 0.8:
            return None  # 已用 80%, 不升级

    # 4. cooldown check
    try:
        from anchor.cooldown import is_cooling
        if is_cooling("claude-fable-5"):
            return None
    except Exception:
        pass

    # 5. import FABLE5
    from anchor.workers import FABLE5
    return FABLE5
