"""SQLite v2 schema + writer (Day 3, per PLAN-v0.2.md)."""
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path.home() / "anchor" / "data" / "anchor.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    tier TEXT NOT NULL,
    query TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    true_worker_idx INTEGER,
    workers_called TEXT,
    confidences TEXT,
    latency_ms INTEGER,
    cost_yuan REAL,
    -- reward column semantics (v0.9.7X-P4): OPT-IN feedback signal, populated
    -- only by explicit feedback.record_outcome() calls (e.g. user thumbs-up/down).
    -- NOT auto-quality monitoring. Quality flows through judge_cache.verdict
    -- (96% coverage, see docs/audit-2026-08-07-routing-quality.md Section 2).
    -- Therefore NULL reward on successful calls is BY DESIGN, not a "blind spot".
    reward REAL
);
CREATE INDEX IF NOT EXISTS idx_ts ON queries(ts);
CREATE INDEX IF NOT EXISTS idx_tier ON queries(tier);
CREATE INDEX IF NOT EXISTS idx_query_hash ON queries(query_hash);

-- v0.9.7X-A1 (audit 2026-08-07): error_class column is added via ALTER TABLE
-- migration in _connect() AFTER SCHEMA runs (so existing dbs upgrade in-place).
-- For fresh dbs, _connect() runs the CREATE TABLE + ALTER TABLE in sequence;
-- the CREATE TABLE above intentionally OMITS error_class to keep CREATE TABLE
-- IF NOT EXISTS idempotent (no schema mismatch on existing tables). The
-- ALTER TABLE adds the column unconditionally. The idx_error_class index is
-- created in _connect() AFTER the ALTER TABLE succeeds.
--
-- Values: 'timeout' | 'auth' | 'quota' | 'connection' | 'server_5xx' |
--         'placeholder' | 'silent_fail' | 'empty_zero_cost' | 'other' | NULL.
-- NULL means no failure detected (covers success + free-quota paths).

CREATE TABLE IF NOT EXISTS retrain_proposed_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    mode TEXT NOT NULL,                  -- 'shadow' or 'applied'
    prior_before TEXT NOT NULL,          -- JSON
    prior_after TEXT NOT NULL,           -- JSON
    W_before TEXT NOT NULL,              -- JSON
    W_after TEXT NOT NULL,               -- JSON
    n_priors INTEGER,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_retrain_ts ON retrain_proposed_changes(ts);

CREATE TABLE IF NOT EXISTS judge_cache (
    cache_key TEXT PRIMARY KEY,   -- SHA1(query + "|" + response_a + "|" + response_b), 24 hex
    query TEXT NOT NULL,
    response_a TEXT NOT NULL,
    response_b TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    verdict TEXT NOT NULL,        -- 'A' | 'B' | 'TIE'
    raw_judge TEXT,               -- raw Opus output, for audit
    ts REAL NOT NULL,
    latency_ms INTEGER,
    cost_yuan REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_judge_cache_ts ON judge_cache(ts);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    last_turn_at REAL NOT NULL,
    n_turns INTEGER NOT NULL DEFAULT 0,
    total_cost_yuan REAL NOT NULL DEFAULT 0.0,
    last_tier TEXT,
    last_worker TEXT,
    avg_quality_score REAL
);
CREATE INDEX IF NOT EXISTS idx_sessions_last_turn_at ON sessions(last_turn_at);
"""



def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL + busy_timeout for safe concurrent writes.

    WAL allows concurrent readers + 1 writer without blocking; busy_timeout
    makes a contended writer wait instead of raising immediately. Both are
    required for multi-uvicorn-worker safety (defense-in-depth — Anchor runs
    single-worker by default in the team-internal phase).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    # v0.9.7X-A1: idempotent migration for existing dbs (adds error_class if missing).
    # audit 2026-08-16 (E5): CREATE INDEX was inside the same try as the ALTER,
    # so on any existing DB the ALTER raised "duplicate column", the except:pass
    # swallowed it, and the index was never created — cascade_error_monitor's
    # WHERE error_class=... full-scanned the 175MB queries table.
    try:
        conn.execute("ALTER TABLE queries ADD COLUMN error_class TEXT")
    except Exception:
        pass  # column already exists or fresh db path; safe to ignore
    conn.execute("CREATE INDEX IF NOT EXISTS idx_error_class ON queries(error_class)")
    conn.commit()
    return conn


def initdb(db_path: Optional[Path] = None) -> Path:
    db_path = Path(db_path) if db_path else DEFAULT_PATH
    conn = _connect(db_path)
    conn.close()
    return db_path


def _hash_query(q: str) -> str:
    return hashlib.sha1(q.encode("utf-8")).hexdigest()[:16]


def insert_query(
    query: str,
    tier: str,
    *,
    ts: Optional[float] = None,
    true_worker_idx: Optional[int] = None,
    workers_called: Optional[str] = None,
    confidences: Optional[str] = None,
    latency_ms: Optional[int] = None,
    cost_yuan: Optional[float] = None,
    reward: Optional[float] = None,
    error_class: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    db_path = Path(db_path) if db_path else DEFAULT_PATH
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """INSERT INTO queries
               (ts, tier, query, query_hash, true_worker_idx,
                workers_called, confidences, latency_ms, cost_yuan, reward, error_class)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts or time.time(),
                tier,
                query,
                _hash_query(query),
                true_worker_idx,
                workers_called,
                confidences,
                latency_ms,
                cost_yuan,
                reward,
                error_class,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_recent(n: int = 100, db_path: Optional[Path] = None) -> list[dict]:
    db_path = Path(db_path) if db_path else DEFAULT_PATH
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT id, ts, tier, query, query_hash, true_worker_idx, "
            "workers_called, confidences, latency_ms, cost_yuan, reward, error_class "
            "FROM queries ORDER BY id DESC LIMIT ?",
            (n,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def export_pairs(min_n: int = 200, db_path: Optional[Path] = None) -> list[tuple]:
    """Return list of (query, tier, query_hash) tuples for ≥min_n rows."""
    db_path = Path(db_path) if db_path else DEFAULT_PATH
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT query, tier, query_hash FROM queries "
            "WHERE reward IS NOT NULL LIMIT ?",
            (max(min_n, 1),),
        )
        return cur.fetchall()
    finally:
        conn.close()



def _judge_cache_key(query: str, response_a: str, response_b: str) -> str:
    """Stable SHA1 over the triple, 24 hex chars.

    24 chars = 96 bits = collision-safe for typical eval sizes (<10M pairs).
    """
    h = hashlib.sha1()
    h.update((query or "").encode("utf-8"))
    h.update(b"|")
    h.update((response_a or "").encode("utf-8"))
    h.update(b"|")
    h.update((response_b or "").encode("utf-8"))
    return h.hexdigest()[:24]


def lookup_judge_cache(keys: list[str], judge_model: str, db_path: Optional[Path] = None) -> dict[str, str]:
    """Batch lookup: returns {cache_key: verdict} for hits. Empty if no keys or no hits."""
    db_path = Path(db_path) if db_path else DEFAULT_PATH
    if not keys:
        return {}
    out: dict[str, str] = {}
    conn = _connect(db_path)
    try:
        # SQLite default SQLITE_MAX_VARIABLE_NUMBER is 999; chunk to be safe.
        CHUNK = 500
        for i in range(0, len(keys), CHUNK):
            chunk = keys[i:i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = conn.execute(
                f"SELECT cache_key, verdict FROM judge_cache "
                f"WHERE cache_key IN ({placeholders}) AND judge_model = ?",
                chunk + [judge_model],
            )
            for row in cur.fetchall():
                out[row[0]] = row[1]
    finally:
        conn.close()
    return out


def insert_judge_cache(
    key: str,
    query: str,
    response_a: str,
    response_b: str,
    judge_model: str,
    verdict: str,
    raw_judge: str,
    latency_ms: int,
    cost_yuan: float = 0.0,
    db_path: Optional[Path] = None,
) -> int:
    """Insert (or replace) one cache row. Returns affected rowid (or 0 if duplicate)."""
    db_path = Path(db_path) if db_path else DEFAULT_PATH
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """INSERT OR REPLACE INTO judge_cache
               (cache_key, query, response_a, response_b, judge_model,
                verdict, raw_judge, ts, latency_ms, cost_yuan)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                query,
                response_a,
                response_b,
                judge_model,
                verdict,
                raw_judge,
                time.time(),
                int(latency_ms),
                float(cost_yuan),
            ),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def insert_judge_cache_many(rows: list[dict], db_path: Optional[Path] = None) -> int:
    """Batch insert via executemany. Each row dict needs the fields of insert_judge_cache.

    Returns number of rows written. Use for batch writes (50+).
    """
    if not rows:
        return 0
    db_path = Path(db_path) if db_path else DEFAULT_PATH
    conn = _connect(db_path)
    try:
        ts_now = time.time()
        payload = [
            (
                r["cache_key"],
                r["query"],
                r["response_a"],
                r["response_b"],
                r["judge_model"],
                r["verdict"],
                r.get("raw_judge", ""),
                r.get("ts", ts_now),
                int(r.get("latency_ms", 0)),
                float(r.get("cost_yuan", 0.0)),
            )
            for r in rows
        ]
        cur = conn.executemany(
            """INSERT OR REPLACE INTO judge_cache
               (cache_key, query, response_a, response_b, judge_model,
                verdict, raw_judge, ts, latency_ms, cost_yuan)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            payload,
        )
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Sessions (v0.9.36 — multi-turn session cost aggregation)
# ---------------------------------------------------------------------------

_QT_SCORE = {
    "excellent": 1.0,
    "good": 0.8,
    "fair": 0.6,
    "degraded": 0.4,
    "error": 0.2,
    None: None,
}


def _now() -> float:
    return time.time()


def upsert_session(
    session_id: str,
    *,
    ts: Optional[float] = None,
    tier: Optional[str] = None,
    worker: Optional[str] = None,
    cost_yuan: float = 0.0,
    quality_tier: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert new session row, or accumulate on existing one.

    For existing sessions: n_turns++, total_cost_yuan += cost_yuan,
    avg_quality_score is running mean of the qt_score mapping,
    last_tier / last_worker / last_turn_at updated.

    Returns affected rowid (1 for new, 0 for update w/ same rowid in sqlite).
    """
    db_path = Path(db_path) if db_path else DEFAULT_PATH
    ts = ts if ts is not None else _now()
    qt_score = _QT_SCORE.get(quality_tier)
    conn = _connect(db_path)
    try:
        # audit 2026-08-16 (E4): was a lock-free SELECT-then-INSERT/UPDATE —
        # concurrent turns on one session_id either raced into a PRIMARY KEY
        # IntegrityError (second INSERT) or lost updates on n_turns/cost.
        # Single atomic UPSERT (SQLite >= 3.24).
        if qt_score is None:
            _avg_expr = "avg_quality_score"  # keep existing running mean
        else:
            _avg_expr = (
                "(avg_quality_score * n_turns + excluded.avg_quality_score)"
                " / (n_turns + 1)"
            )
        conn.execute(
            f"""INSERT INTO sessions
               (session_id, started_at, last_turn_at, n_turns,
                total_cost_yuan, last_tier, last_worker, avg_quality_score)
               VALUES (?, ?, ?, 1, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
               last_turn_at = excluded.last_turn_at,
               n_turns = sessions.n_turns + 1,
               total_cost_yuan = sessions.total_cost_yuan + excluded.total_cost_yuan,
               last_tier = excluded.last_tier,
               last_worker = excluded.last_worker,
               avg_quality_score = {_avg_expr}""",
            (
                session_id,
                ts,
                ts,
                float(cost_yuan),
                tier,
                worker,
                qt_score,
            ),
        )
        conn.commit()
        return 1  # 1 row affected (insert or update)
    finally:
        conn.close()


def get_session(session_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    db_path = Path(db_path) if db_path else DEFAULT_PATH
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT session_id, started_at, last_turn_at, n_turns, "
            "total_cost_yuan, last_tier, last_worker, avg_quality_score "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        return dict(zip(cols, row)) if row else None
    finally:
        conn.close()


def list_active_sessions(since_hours: float = 24.0, db_path: Optional[Path] = None) -> list[dict]:
    db_path = Path(db_path) if db_path else DEFAULT_PATH
    cutoff = _now() - since_hours * 3600.0
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT session_id, started_at, last_turn_at, n_turns, "
            "total_cost_yuan, last_tier, last_worker, avg_quality_score "
            "FROM sessions WHERE last_turn_at >= ? ORDER BY last_turn_at DESC",
            (cutoff,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Async wrappers (v0.9.51): keep sync public API intact; wrap with to_thread
# so FastAPI hot paths can avoid blocking the event loop when desired.
# ---------------------------------------------------------------------------
import asyncio as _asyncio


async def ainitdb(db_path: Optional[Path] = None) -> Path:
    return await _asyncio.to_thread(initdb, db_path)


async def ainsert_query(*args, **kwargs):
    return await _asyncio.to_thread(insert_query, *args, **kwargs)


async def aget_recent(n: int = 100, db_path: Optional[Path] = None) -> list[dict]:
    return await _asyncio.to_thread(get_recent, n, db_path)


async def aexport_pairs(min_n: int = 200, db_path: Optional[Path] = None) -> list[tuple]:
    return await _asyncio.to_thread(export_pairs, min_n, db_path)


async def alookup_judge_cache(keys: list[str], judge_model: str, db_path: Optional[Path] = None) -> dict[str, str]:
    return await _asyncio.to_thread(lookup_judge_cache, keys, judge_model, db_path)


async def ainsert_judge_cache_many(rows: list[dict], db_path: Optional[Path] = None) -> int:
    return await _asyncio.to_thread(insert_judge_cache_many, rows, db_path)


async def aupsert_session(*args, **kwargs):
    return await _asyncio.to_thread(upsert_session, *args, **kwargs)


async def aget_session(session_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    return await _asyncio.to_thread(get_session, session_id, db_path)


async def alist_active_sessions(since_hours: float = 24.0, db_path: Optional[Path] = None) -> list[dict]:
    return await _asyncio.to_thread(list_active_sessions, since_hours, db_path)
