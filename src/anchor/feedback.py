"""Day 9: Online learning loop. Server logs each query; feedback endpoint updates head."""

from anchor.head import update as head_update
from anchor.db import insert_query, get_recent


WORKER_ALIASES = {
    "sonnet-5": "claude-sonnet-5",
    "dpsk-v4-flash": "deepseek-v4-flash",
}


def normalize_worker_name(worker: str) -> str:
    return WORKER_ALIASES.get(worker, worker)


def record_outcome(
    query: str,
    tier: str,
    worker: str,
    *,
    success: float,
    prompt_len: int = 500,
    budget: float = 0.5,
    cost_yuan: float = 0.0,
    latency_ms: int = 0,
    query_type: str = "en",
) -> int:
    """Record query outcome + update head prior + log to DB. Returns row id."""
    worker = normalize_worker_name(worker)
    head_update(query_type, tier, prompt_len, budget, worker, success)
    row_id = insert_query(
        query, tier,
        true_worker_idx=None,
        workers_called=worker,
        confidences="",
        latency_ms=latency_ms,
        cost_yuan=cost_yuan,
        reward=success,
    )
    # v1.0.x STAGE1 (second quality signal): task_success is the human-labelled
    # success from /feedback — same value as reward, named explicitly so
    # calibration consumers can distinguish it from the LLM judge_score.
    # DB column is `reward` (no schema change needed); this alias documents intent.
    return row_id


def get_recent_outcomes(n: int = 100) -> list[dict]:
    """Get last n queries with non-null reward."""
    rows = get_recent(n)
    return [r for r in rows if r.get("reward") is not None]
