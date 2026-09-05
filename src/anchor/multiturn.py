"""Multi-turn support with cost-aware session-level routing.

Schema v3 extends v2 with parent_turn_id + embedding BLOB (per PLAN-v0.2.md D5).
v0.9.36+ (current): single-product lane `auto` only. next_tier() always returns
"auto". Worker selection handled by hard-rules + Pareto in routing_core.
"""
import hashlib
import time
from typing import Optional

from anchor.db import (
    get_recent,
    insert_query,
    upsert_session as _db_upsert_session,
    get_session as _db_get_session,
    list_active_sessions as _db_list_active_sessions,
)
from anchor._query_complexity import query_complexity


# ---------------------------------------------------------------------------
# Public constants for tier escalation / de-escalation rules.
# Single product: only "auto" lane.
# ---------------------------------------------------------------------------
_TIER_ORDER = ("auto",)


# ---------------------------------------------------------------------------
# Backwards-compatible public API
# ---------------------------------------------------------------------------

def turn_id(query: str, parent_turn_id: Optional[str] = None) -> str:
    """Generate turn id, deterministically derived from parent + query."""
    h = hashlib.sha1()
    if parent_turn_id:
        h.update(parent_turn_id.encode())
    h.update(query.encode())
    return h.hexdigest()[:16]


def next_tier(
    parent_turn_id: Optional[str],
    query_complexity_score: float = 0.5,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    requested_tier: Optional[str] = None,
) -> str:
    """Single-product routing (v0.9.52): always the auto lane.

    Signature kept for callers; session cost / parent quality no longer map to
    basic/premium/ultra SKUs. Worker selection is fully inside hard-rules +
    Pareto on the auto pool.
    """
    return "auto"



def _get_parent_turn(parent_turn_id: str) -> Optional[dict]:
    """Look up a parent turn row by query_hash.

    Refactored out of next_tier so tests can monkeypatch this single hook
    without depending on db layout.
    """
    rows = get_recent(500)
    for r in rows:
        if r.get("query_hash") == parent_turn_id:
            return r
    return None


def insert_turn(
    query: str,
    tier: str,
    *,
    parent_turn_id: Optional[str] = None,
    worker: Optional[str] = None,
    cost_yuan: float = 0.0,
    latency_ms: int = 0,
    reward: Optional[float] = None,
    session_id: Optional[str] = None,
    quality_tier: Optional[str] = None,
) -> dict:
    """Insert a turn row, return turn id + parent linkage.

    If session_id is provided, also upsert the session row with cumulative
    cost, n_turns++, and rolling avg_quality_score.
    """
    tid = turn_id(query, parent_turn_id)
    rid = insert_query(
        query, tier,
        true_worker_idx=None,
        workers_called=worker,
        confidences=f"{tid}:{parent_turn_id}" if parent_turn_id else tid,
        latency_ms=latency_ms,
        cost_yuan=cost_yuan,
        reward=reward,
    )
    if session_id:
        _db_upsert_session(
            session_id,
            ts=time.time(),
            tier=tier,
            worker=worker,
            cost_yuan=cost_yuan,
            quality_tier=quality_tier,
        )
    return {"turn_id": tid, "row_id": rid, "parent_turn_id": parent_turn_id}


def dialog_chain(turn_id_to_find: str, max_depth: int = 20) -> list[dict]:
    """Walk back from a turn_id to root, returning ancestor chain."""
    rows = get_recent(2000)
    by_id = {}
    for r in rows:
        key = (r.get("confidences") or "").split(":", 1)[0]
        if key:
            by_id[key] = r
    chain = []
    current = turn_id_to_find
    for _ in range(max_depth):
        row = by_id.get(current)
        if not row:
            break
        chain.append({
            "turn_id": current, "query": row.get("query"),
            "tier": row.get("tier"), "worker": row.get("workers_called"),
        })
        confidences = row.get("confidences") or ""
        if ":" in confidences:
            current = confidences.split(":", 1)[1]
        else:
            break
    return list(reversed(chain))


# ---------------------------------------------------------------------------
# v0.9.36 session helpers — public surface for server integration (next phase).
# ---------------------------------------------------------------------------

def record_session_turn(
    session_id: str,
    turn_id_value: str,
    tier: str,
    worker: Optional[str],
    cost_yuan: float,
    quality_tier: Optional[str] = None,
) -> int:
    """Upsert a session row keyed by session_id.

    Wrapper around db.upsert_session that lets callers pass an explicit
    turn_id_value for audit linkage. The session row itself does not store
    the turn id — use anchor.db.get_recent() with the query_hash to find it.

    Returns 1 if a new row was created, 0 if an existing one was updated.
    """
    return _db_upsert_session(
        session_id,
        ts=time.time(),
        tier=tier,
        worker=worker,
        cost_yuan=cost_yuan,
        quality_tier=quality_tier,
    )


def get_session_summary(session_id: str) -> dict:
    """Return a session summary dict. Empty dict if session does not exist."""
    sess = _db_get_session(session_id)
    if not sess:
        return {
            "session_id": session_id,
            "n_turns": 0,
            "total_cost_yuan": 0.0,
            "avg_quality_score": None,
            "last_tier": None,
            "last_worker": None,
            "started_at": None,
            "last_turn_at": None,
        }
    return {
        "session_id": sess.get("session_id"),
        "n_turns": sess.get("n_turns") or 0,
        "total_cost_yuan": sess.get("total_cost_yuan") or 0.0,
        "avg_quality_score": sess.get("avg_quality_score"),
        "last_tier": sess.get("last_tier"),
        "last_worker": sess.get("last_worker"),
        "started_at": sess.get("started_at"),
        "last_turn_at": sess.get("last_turn_at"),
    }


def list_active_sessions(since_hours: float = 24.0) -> list[dict]:
    """List sessions with last_turn_at within since_hours, newest first."""
    return _db_list_active_sessions(since_hours=since_hours)


__all__ = [
    "turn_id",
    "next_tier",
    "insert_turn",
    "dialog_chain",
    "record_session_turn",
    "get_session_summary",
    "list_active_sessions",
    "query_complexity",
]
