"""Day 17 (per Fable 5 decision #1, 2026-07-06): append-only JSONL session log.

Why separate from SQLite:
- SQLite keeps structured aggregates (per-query tier/worker/cost).
- JSONL is the canonical stream used by v6 retrain / judge sampling /
  session_id grouping. Human-readable, no schema migration cost.

Path: data/anchor_sessions/YYYY-MM-DD.jsonl
One line per /v1/chat/completions request (regardless of tier/worker/stream).
"""
import json
import hashlib
import os
import time
from pathlib import Path
from anchor import _ROOT as _A_ROOT
from typing import Optional

SESSIONS_DIR = Path(
    os.environ.get("ANCHOR_SESSIONS_DIR", str(_A_ROOT / "data/anchor_sessions"))
)


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _file_for(date_str: str) -> Path:
    return SESSIONS_DIR / f"{date_str}.jsonl"


def hash_query(q: str) -> str:
    return hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]


def append_session(
    *,
    query: str,
    routed_tier: str,
    model_used: str,
    latency_ms: int,
    cost_yuan: float,
    judge_score=None,
    session_id=None,
    extra=None,
    source: str = "unknown",
    error_kind: Optional[str] = None,
    cache_hit_tokens: int = 0,
    task_success: Optional[float] = None,
) -> Path:
    """Append one JSONL line. Returns path of file written to.

    Blocking I/O is intentional: we want durability for the session log.
    Call sites are in the request hot path but FastAPI runs handlers in
    a threadpool via starlette; a sub-millisecond append is acceptable.

    error_kind (v0.9.55): when set, tags the row with one of
    {"transport", "quota", "empty", "malformed", "bad_answer"} so the
    quarantine sweep can distinguish transport/quota failures from real
    quality failures. Backwards compatible: omitted rows default to
    "unknown" and the sweep applies a heuristic (see admin_ops).

    cache_hit_tokens (v0.9.60): prompt-cache hit tokens reported by
    OpenAI usage.prompt_tokens_details.cached_tokens. Backwards
    compatible: omitted defaults to 0.

    task_success (v1.0.x STAGE1): second quality signal, human feedback
    (0-1), stored alongside judge_score for calibrating judge bias.
    """
    _ensure_dir()
    rec = {
        "ts": time.time(),
        "date": _today(),
        "query_hash": hash_query(query),
        "query_text": query,
        "routed_tier": routed_tier,
        "model_used": model_used,
        "latency_ms": int(latency_ms),
        "cost_yuan": float(cost_yuan),
        "judge_score": judge_score,
        "session_id": session_id,
        "source": source,
        "cache_hit_tokens": cache_hit_tokens,
        "task_success": task_success,
    }
    if extra:
        rec.update(extra)
    if error_kind is not None:
        rec["error_kind"] = str(error_kind)
    fp = _file_for(_today())
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return fp
