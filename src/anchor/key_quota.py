"""Per-API-key RPM + daily token quota (v0.9.51).

Backends (selected by env, fail-open to memory):
  - auto    (default unset): memory if single worker; sqlite when
            WEB_CONCURRENCY / UVICORN_WORKERS / ANCHOR_WORKERS > 1
  - memory  : process-local, locked
  - sqlite  (ANCHOR_KEY_QUOTA_BACKEND=sqlite): shared across local workers
  - redis   (ANCHOR_KEY_QUOTA_BACKEND=redis + ANCHOR_REDIS_URL): multi-host
            requires optional `redis` package; falls back to sqlite/memory

Public API used by server middleware:
  check_and_consume(key, tokens, rpm, daily_tokens) -> None | error_dict
  usage_snapshot(key, rpm, daily_tokens) -> dict
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from anchor import _ROOT as _ANCHOR_ROOT

_logger = logging.getLogger("anchor.key_quota")

_DEFAULT_DB = Path(str(_ANCHOR_ROOT)) / "data" / "key_quota.db"


def _utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


class _MemoryBackend:
    _MAX_KEYS = 10000  # soft cap: beyond this, LRU-evict oldest key

    def __init__(self) -> None:
        self._state: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._key_order: list[str] = []  # insertion order for LRU eviction

    def check_and_consume(
        self, key: str, tokens: int, rpm: int, daily_tokens: int, now: float | None = None,
    ) -> Optional[dict]:
        now = now if now is not None else time.time()
        day = _utc_day()
        with self._lock:
            # Soft cap: if attacker floods with distinct keys, evict oldest
            # entries so the dict stays bounded. Production keys are usually
            # <100; _MAX_KEYS = 10000 leaves generous headroom for legitimate
            # clients while preventing unbounded RAM growth.
            if key not in self._state and len(self._state) >= self._MAX_KEYS:
                while self._key_order and len(self._state) >= self._MAX_KEYS:
                    evict = self._key_order.pop(0)
                    self._state.pop(evict, None)
            if key not in self._state:
                self._key_order.append(key)
            st = self._state.setdefault(key, {"window": [], "day": (day, 0)})
            window = [(ts, tok) for ts, tok in st.get("window", []) if now - ts < 60]
            if len(window) >= rpm:
                retry = max(1, int(60 - (now - window[0][0]))) if window else 1
                return {"type": "rate_limit_error", "message": "rate limit exceeded",
                        "retry_after": retry}
            state_day, used = st.get("day", (day, 0))
            if state_day != day:
                state_day, used = day, 0
            if daily_tokens > 0 and used + tokens > daily_tokens:
                return {"type": "rate_limit_error", "message": "daily token quota exceeded",
                        "retry_after": 3600}
            window.append((now, tokens))
            st["window"] = window
            st["day"] = (state_day, used + tokens)
        return None

    def usage(self, key: str, rpm: int, daily_tokens: int) -> dict:
        now = time.time()
        day = _utc_day()
        with self._lock:
            st = self._state.setdefault(key, {"window": [], "day": (day, 0)})
            window = [(ts, tok) for ts, tok in st.get("window", []) if now - ts < 60]
            st["window"] = window
            d, used = st.get("day", (day, 0))
            if d != day:
                d, used = day, 0
                st["day"] = (d, used)
            return {
                "rpm": rpm,
                "rpm_used": len(window),
                "rpm_remaining": max(0, rpm - len(window)),
                "daily_tokens": daily_tokens,
                "daily_tokens_used": used,
                "daily_tokens_remaining": None if daily_tokens <= 0 else max(0, daily_tokens - used),
                "day_utc": d,
                "backend": "memory",
            }


class _SqliteBackend:
    """Process-shared quota via SQLite WAL (multi-uvicorn workers, same host)."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = Path(db_path or os.environ.get("ANCHOR_KEY_QUOTA_DB", str(_DEFAULT_DB)))
        self._lock = threading.Lock()
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
                    CREATE TABLE IF NOT EXISTS key_rpm (
                        key_hash TEXT NOT NULL,
                        ts REAL NOT NULL,
                        tokens INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_key_rpm_key_ts ON key_rpm(key_hash, ts);
                    CREATE TABLE IF NOT EXISTS key_daily (
                        key_hash TEXT NOT NULL,
                        day TEXT NOT NULL,
                        tokens INTEGER NOT NULL,
                        PRIMARY KEY (key_hash, day)
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _hash(key: str) -> str:
        import hashlib
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

    def check_and_consume(
        self, key: str, tokens: int, rpm: int, daily_tokens: int, now: float | None = None,
    ) -> Optional[dict]:
        now = now if now is not None else time.time()
        day = _utc_day()
        kh = self._hash(key)
        cutoff = now - 60.0
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM key_rpm WHERE key_hash=? AND ts < ?", (kh, cutoff))
                cur = conn.execute(
                    "SELECT ts FROM key_rpm WHERE key_hash=? ORDER BY ts ASC", (kh,)
                )
                rows = cur.fetchall()
                if len(rows) >= rpm:
                    oldest = rows[0][0]
                    retry = max(1, int(60 - (now - oldest)))
                    conn.rollback()
                    return {"type": "rate_limit_error", "message": "rate limit exceeded",
                            "retry_after": retry}
                cur = conn.execute(
                    "SELECT tokens FROM key_daily WHERE key_hash=? AND day=?", (kh, day)
                )
                row = cur.fetchone()
                used = int(row[0]) if row else 0
                if daily_tokens > 0 and used + tokens > daily_tokens:
                    conn.rollback()
                    return {"type": "rate_limit_error", "message": "daily token quota exceeded",
                            "retry_after": 3600}
                conn.execute(
                    "INSERT INTO key_rpm(key_hash, ts, tokens) VALUES (?,?,?)",
                    (kh, now, tokens),
                )
                conn.execute(
                    """
                    INSERT INTO key_daily(key_hash, day, tokens) VALUES (?,?,?)
                    ON CONFLICT(key_hash, day) DO UPDATE SET tokens = tokens + excluded.tokens
                    """,
                    (kh, day, tokens),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return None

    def usage(self, key: str, rpm: int, daily_tokens: int) -> dict:
        now = time.time()
        day = _utc_day()
        kh = self._hash(key)
        cutoff = now - 60.0
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM key_rpm WHERE key_hash=? AND ts < ?", (kh, cutoff))
                cur = conn.execute(
                    "SELECT COUNT(*) FROM key_rpm WHERE key_hash=?", (kh,)
                )
                rpm_used = int(cur.fetchone()[0])
                cur = conn.execute(
                    "SELECT tokens FROM key_daily WHERE key_hash=? AND day=?", (kh, day)
                )
                row = cur.fetchone()
                used = int(row[0]) if row else 0
                conn.commit()
            finally:
                conn.close()
        return {
            "rpm": rpm,
            "rpm_used": rpm_used,
            "rpm_remaining": max(0, rpm - rpm_used),
            "daily_tokens": daily_tokens,
            "daily_tokens_used": used,
            "daily_tokens_remaining": None if daily_tokens <= 0 else max(0, daily_tokens - used),
            "day_utc": day,
            "backend": "sqlite",
        }


class _RedisBackend:
    """Optional multi-host backend. Requires `redis` package (or test fake)."""

    def __init__(self, url: str, client=None) -> None:
        if client is not None:
            self._r = client
        else:
            import redis  # type: ignore
            self._r = redis.Redis.from_url(url, decode_responses=True)
        self._r.ping()

    def check_and_consume(
        self, key: str, tokens: int, rpm: int, daily_tokens: int, now: float | None = None,
    ) -> Optional[dict]:
        now = now if now is not None else time.time()
        day = _utc_day()
        import hashlib
        kh = hashlib.sha256(key.encode()).hexdigest()[:32]
        rpm_key = f"anchor:kq:rpm:{kh}"
        day_key = f"anchor:kq:day:{kh}:{day}"
        pipe = self._r.pipeline()
        pipe.zremrangebyscore(rpm_key, 0, now - 60)
        pipe.zcard(rpm_key)
        pipe.get(day_key)
        _, count, used_raw = pipe.execute()
        count = int(count or 0)
        used = int(used_raw or 0)
        if count >= rpm:
            return {"type": "rate_limit_error", "message": "rate limit exceeded",
                    "retry_after": 1}
        if daily_tokens > 0 and used + tokens > daily_tokens:
            return {"type": "rate_limit_error", "message": "daily token quota exceeded",
                    "retry_after": 3600}
        member = f"{now}:{tokens}:{os.getpid()}"
        pipe = self._r.pipeline()
        pipe.zadd(rpm_key, {member: now})
        pipe.expire(rpm_key, 120)
        pipe.incrby(day_key, tokens)
        pipe.expire(day_key, 86400 * 2)
        pipe.execute()
        return None

    def usage(self, key: str, rpm: int, daily_tokens: int) -> dict:
        now = time.time()
        day = _utc_day()
        import hashlib
        kh = hashlib.sha256(key.encode()).hexdigest()[:32]
        rpm_key = f"anchor:kq:rpm:{kh}"
        day_key = f"anchor:kq:day:{kh}:{day}"
        self._r.zremrangebyscore(rpm_key, 0, now - 60)
        rpm_used = int(self._r.zcard(rpm_key) or 0)
        used = int(self._r.get(day_key) or 0)
        return {
            "rpm": rpm,
            "rpm_used": rpm_used,
            "rpm_remaining": max(0, rpm - rpm_used),
            "daily_tokens": daily_tokens,
            "daily_tokens_used": used,
            "daily_tokens_remaining": None if daily_tokens <= 0 else max(0, daily_tokens - used),
            "day_utc": day,
            "backend": "redis",
        }


_backend: Any = None
_backend_lock = threading.Lock()


def get_backend():
    """Lazy singleton backend selected by ANCHOR_KEY_QUOTA_BACKEND."""
    global _backend
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is not None:
            return _backend
        # Default: memory. Auto-upgrade to sqlite when multi-worker is likely
        # (WEB_CONCURRENCY / UVICORN_WORKERS > 1) so RPM limits are process-shared.
        explicit = (os.environ.get("ANCHOR_KEY_QUOTA_BACKEND") or "").strip().lower()
        if explicit:
            kind = explicit
        else:
            workers = 1
            for envn in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "ANCHOR_WORKERS"):
                try:
                    workers = max(workers, int(os.environ.get(envn) or "1"))
                except ValueError:
                    pass
            kind = "sqlite" if workers > 1 else "memory"
        if kind == "redis":
            url = os.environ.get("ANCHOR_REDIS_URL") or os.environ.get("REDIS_URL") or ""
            fake = os.environ.get("ANCHOR_KEY_QUOTA_REDIS_FAKE", "").strip()
            if fake:
                try:
                    from anchor._fake_redis import FakeRedis
                    _backend = _RedisBackend(url or "redis://fake", client=FakeRedis())
                    _logger.info("KEY_QUOTA_BACKEND redis (fake)")
                    return _backend
                except Exception as exc:
                    _logger.warning("KEY_QUOTA_REDIS_FAKE_FAIL fallback=sqlite err=%s", exc)
                    kind = "sqlite"
            elif url:
                try:
                    _backend = _RedisBackend(url)
                    _logger.info("KEY_QUOTA_BACKEND redis")
                    return _backend
                except Exception as exc:
                    _logger.warning("KEY_QUOTA_REDIS_FAIL fallback=sqlite err=%s", exc)
                    kind = "sqlite"
            else:
                _logger.warning("KEY_QUOTA_REDIS_NO_URL fallback=sqlite")
                kind = "sqlite"
        if kind == "sqlite":
            try:
                _backend = _SqliteBackend()
                _logger.info("KEY_QUOTA_BACKEND sqlite path=%s", _DEFAULT_DB)
                return _backend
            except Exception as exc:
                _logger.warning("KEY_QUOTA_SQLITE_FAIL fallback=memory err=%s", exc)
        _backend = _MemoryBackend()
        _logger.info("KEY_QUOTA_BACKEND memory")
        return _backend


def reset_backend_for_tests() -> None:
    """Test helper: drop singleton so next call re-reads env."""
    global _backend
    with _backend_lock:
        _backend = None


def check_and_consume(key: str, tokens: int, *, rpm: int = 60, daily_tokens: int = 0) -> Optional[dict]:
    return get_backend().check_and_consume(key, tokens, rpm, daily_tokens)


def usage_snapshot(key: str, *, rpm: int = 60, daily_tokens: int = 0, name: str | None = None) -> dict:
    snap = get_backend().usage(key, rpm, daily_tokens)
    if name is not None:
        snap["name"] = name
    return snap
