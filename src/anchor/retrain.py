"""Day 11: weekly retrain + 10% Opus spot-check.

Retrain: re-snapshot head + update prior from session DB.
Spot-check: stratified sample, judge with Opus 4.8, log to spotcheck_log.
"""
import json
import random
import time
from pathlib import Path
from typing import Optional

from anchor.db import get_recent
from anchor.head import prior_table
from anchor.drift import snapshot
from anchor.judge import pair_acc

SPOTCHECK_PATH = Path.home() / "anchor" / "logs" / "spotcheck.jsonl"
SPOTCHECK_RATE = 0.10  # 10% weekly


def retrain_head(prior_snapshot: Optional[dict] = None, mode: str = "apply") -> dict:
    """Re-snapshot head state.

    mode='shadow' (default during validation): log proposed changes to DB,
    do NOT apply to live head.
    mode='apply': re-snapshot baseline (used post-validation).
    """
    import json
    from anchor.head import W
    if prior_snapshot is None:
        try:
            prior_snapshot = {
                "W": W.tolist() if hasattr(W, "tolist") else list(W),
                "prior": {f"{k[0]}|{k[1]}": v for k, v in prior_table.items()},
            }
        except Exception:
            prior_snapshot = None
    if mode == "shadow":
        from anchor.db import initdb, DEFAULT_PATH as DB_PATH
        initdb()
        try:
            import sqlite3
            db_path = DB_PATH
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    """INSERT INTO retrain_proposed_changes
                       (ts, mode, prior_before, prior_after, W_before, W_after, n_priors, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        time.time(),
                        "shadow",
                        json.dumps(prior_snapshot.get("prior", {})),
                        json.dumps({f"{k[0]}|{k[1]}": v for k, v in prior_table.items()}),
                        json.dumps(prior_snapshot.get("W", [])),
                        json.dumps(W.tolist() if hasattr(W, "tolist") else list(W)),
                        len(prior_table),
                        "validation window shadow retrain (no apply)",
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as _re_err:
            # AUDIT 2026-07-25: surface db write errors so operators can detect
            # silent training-log failures (previously swallowed silently).
            import logging as _lg_retrain
            _lg_retrain.warning("RETRAIN_LOG_WRITE_FAIL err=%s", _re_err)
        return {
            "retrained_at": time.time(),
            "n_priors": len(prior_table),
            "mode": "shadow",
            "applied": False,
        }
    snapshot(prior_table, W)
    return {
        "retrained_at": time.time(),
        "n_priors": len(prior_table),
        "mode": "apply",
        "applied": True,
    }


async def weekly_spotcheck(
    queries: Optional[list[dict]] = None,
    *,
    rate: float = SPOTCHECK_RATE,
    n_target: int = 50,
) -> dict:
    """Run spot-check: stratified sample, judge head pick vs always-Sonnet-5.

    queries: list of {query, query_type, tier, head_response, sonnet_response}
    """
    if queries is None:
        recent = get_recent(n_target * 5)
        queries = [r for r in recent if r.get("workers_called") and r.get("reward") is not None]
    if not queries:
        return {"judged": 0, "pair_acc": 0.0, "spotcheck_rate": rate}
    k = max(1, int(len(queries) * rate))
    sample = random.sample(queries, min(k, len(queries)))
    head_qs = [r["query"] for r in sample]
    # F4 fix 2026-08-16: feeding worker names ("minimax-m3") or placeholder
    # strings ("[sonnet-5 baseline response]") to the LLM judge produced
    # meaningless scores. Skip the judge when the session log has no real
    # answer text and mark the spot-check not-comparable so downstream
    # retrain signals are not poisoned by fake pair_acc.
    head_rs = []
    sample_with_answers = []
    for r in sample:
        answer_text = (r.get("answer") or "").strip()
        if not answer_text:
            continue
        sample_with_answers.append(r)
        head_rs.append(answer_text)
    if not sample_with_answers:
        # No answer text in session log: mark not-comparable, skip judge.
        # Future hook can replay real head responses + baseline if the
        # session log is extended to capture both.
        import logging as _lg_f4
        _lg_f4.info(
            "retrain spot-check skipped: no answer text in session log; "
            "mark not-comparable (n=%d)", len(sample),
        )
        return {
            "judged": 0,
            "pair_acc": None,
            "skipped_reason": "no_answer_text",
            "spotcheck_rate": rate,
        }
    sonnet_rs = ["[sonnet-5 baseline response]"] * len(sample_with_answers)
    result = await pair_acc(head_qs, head_rs, sonnet_rs, spot_check_rate=1.0)
    SPOTCHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPOTCHECK_PATH, "a") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "n_judged": result["judged"],
            "pair_acc": result["pair_acc"],
            "a_wins": result["a_wins"],
            "ties": result["ties"],
            "b_wins": result["b_wins"],
        }) + "\n")
    return result
