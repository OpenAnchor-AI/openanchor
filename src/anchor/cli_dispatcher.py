"""Day 16: CLI dispatcher.

Routes a query through the right worker based on (cli, query_type, tier).
Each CLI is mapped to a default tier + worker.
"""
from dataclasses import dataclass
from typing import Optional

from anchor.head import predict
from anchor.classifier import classify
from anchor.traffic_log import log_query


CLI_DEFAULTS = {
    "codex":   {"tier": "ultra",   "rationale": "codex default quality"},
    "opencode": {"tier": "premium", "rationale": "opencode default coding"},
    "claude":  {"tier": "premium", "rationale": "claude code default"},
    "agy":     {"tier": "ultra",   "rationale": "antigravity default quality"},
}


@dataclass
class DispatchResult:
    cli: str
    query: str
    query_type: str
    tier: str
    worker: str
    score: float
    logged_row_id: int


def dispatch(
    cli: str,
    query: str,
    *,
    query_type: Optional[str] = None,
    override_tier: Optional[str] = None,
    cost_yuan: float = 0.0,
    latency_ms: int = 0,
) -> DispatchResult:
    """Route a query from `cli` through Anchor and log it."""
    qt = query_type or classify(query)
    tier = override_tier or CLI_DEFAULTS.get(cli, {}).get("tier", "premium")
    worker, score = predict(qt, tier, prompt_len=500, budget=0.5)
    rid = log_query(
        query, tier, worker,
        cost_yuan=cost_yuan, latency_ms=latency_ms,
        query_type=qt, source=cli,
    )
    return DispatchResult(cli, query, qt, tier, worker, score, rid)
