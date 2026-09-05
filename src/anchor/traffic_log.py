"""Day 15: traffic log writer. Captures every query through Anchor into SQLite."""
from typing import Optional

from anchor.db import insert_query


def log_query(
    query: str,
    tier: str,
    worker: str,
    *,
    cost_yuan: float = 0.0,
    latency_ms: int = 0,
    query_type: str = "en",
    success: Optional[float] = None,
    source: str = "unknown",
) -> int:
    """Log a query from any source (Codex/OpenCode/Claude Code/agy/curl) to SQLite."""
    return insert_query(
        query, tier,
        true_worker_idx=None,
        workers_called=worker,
        confidences=source,
        latency_ms=latency_ms,
        cost_yuan=cost_yuan,
        reward=success,
    )


def log_batch(entries: list[dict]) -> int:
    """Bulk insert. Returns count of rows written."""
    n = 0
    for e in entries:
        log_query(
            e["query"], e.get("tier", "premium"), e.get("worker", "unknown"),
            cost_yuan=e.get("cost_yuan", 0.0),
            latency_ms=e.get("latency_ms", 0),
            query_type=e.get("query_type", "en"),
            success=e.get("success"),
            source=e.get("source", "batch"),
        )
        n += 1
    return n
