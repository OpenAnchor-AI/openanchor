"""Day 18 (Fable 5, 2026-07-06): /debug/dashboard ops view.

Aggregates session JSONL + cooldown state into a single ops snapshot.
Used to monitor:
- Routing distribution: which workers are picking what
- Cost: total + per-worker breakdown
- Latency: avg, p50 (approximated)
- Cooldown: who's down and for how long
- Tier usage: basic/premium/ultra split
"""
import json
import time
from collections import Counter
from anchor import _ROOT as _A_ROOT
from pathlib import Path
from typing import Optional

SESSIONS_DIR = Path(str(_A_ROOT / "data/anchor_sessions"))


def _iter_today():
    fp = SESSIONS_DIR / f"{time.strftime('%Y-%m-%d', time.localtime())}.jsonl"
    if not fp.exists():
        return
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _drift_snapshot() -> dict:
    try:
        from anchor.drift import compute_drift
        from anchor.head import prior_table
        return compute_drift(prior_table)
    except Exception:
        return {"mean_kl": 0.0, "max_kl": 0.0, "n": 0, "has_baseline": False}


def _head_snapshot() -> dict:
    try:
        from anchor.head import prior_table, WORKERS, W
        return {
            "n_priors": len(prior_table),
            "n_workers": len(WORKERS),
            "weights": W.tolist() if hasattr(W, "tolist") else list(W),
        }
    except Exception:
        return {"n_priors": 0, "n_workers": 0, "weights": []}


def _legacy_reward_metrics(limit: int) -> dict:
    try:
        from anchor.db import get_recent
        rows = get_recent(limit)
    except Exception:
        rows = []
    rewards = [r.get("reward") for r in rows if r.get("reward") is not None]
    return {
        "n_with_reward": len(rewards),
        "mean_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
    }


def _worker_pool_snapshot() -> list:
    """List of {name, enabled, auth_gated} per anchor.config.WORKERS worker.

    v0.9.65: surface the baosiapi Claude auth gate to the dashboard so ops can
    tell apart "env-disabled" from "auth-gated". auth_gated is True ONLY for
    the four claude-* workers whose source-level enabled was True but were
    flipped to False by _apply_claude_auth_gate().
    """
    try:
        from anchor.config import WORKERS, SOURCE_ENABLED_BY_NAME
    except Exception:
        return []
    out = []
    for w in WORKERS:
        source_enabled = SOURCE_ENABLED_BY_NAME.get(w.name, w.enabled)
        auth_gated = (
            w.name in ("claude-fable-5", "claude-opus-5",
                       "claude-haiku-4-5", "claude-sonnet-5")
            and source_enabled is True
            and w.enabled is False
        )
        out.append({"name": w.name, "enabled": w.enabled, "auth_gated": auth_gated})
    return out


def _base_snapshot(date: Optional[str], last_n: int) -> dict:
    return {
        "ts": time.time(),
        "date": date or time.strftime("%Y-%m-%d"),
        "n_queries": 0,
        "n_unique_sessions": 0,
        "total_cost_yuan": 0.0,
        "avg_cost_yuan": 0.0,
        "latency_ms": {"avg": 0, "p50": 0, "p95": 0, "min": 0, "max": 0},
        "tier_distribution": {},
        "worker_distribution": {},
        "cost_by_worker": {},
        "cooling_now": {},
        "avg_quality_score": None,
        "quality_by_worker": {},
        "quality_tier_distribution": {},
        "worker_pool": _worker_pool_snapshot(),
        **_legacy_reward_metrics(last_n),
        "drift": _drift_snapshot(),
        "head": _head_snapshot(),
    }


def metrics_snapshot(date: Optional[str] = None, last_n: int = 1000, n_recent: Optional[int] = None) -> dict:
    """Compute ops snapshot. If date is None, use today."""
    if n_recent is not None:
        last_n = n_recent
    if date:
        fp = SESSIONS_DIR / f"{date}.jsonl"
        rows = []
        if fp.exists():
            with open(fp) as f:
                rows = [json.loads(line) for line in f if line.strip()]
    else:
        rows = list(_iter_today())

    # Apply last_n
    rows = rows[-last_n:]

    base = _base_snapshot(date, last_n)
    if not rows:
        return {**base, "message": "no queries yet today"}

    # Worker / tier / session
    worker_counts = Counter(r.get("model_used", "?") for r in rows)
    tier_counts = Counter(r.get("routed_tier", "?") for r in rows)
    n_unique_sessions = len(set(r.get("session_id") for r in rows if r.get("session_id")))

    # Cost
    total_cost = sum(r.get("cost_yuan", 0) for r in rows)
    cost_by_worker = {}
    for r in rows:
        w = r.get("model_used", "?")
        cost_by_worker[w] = cost_by_worker.get(w, 0.0) + r.get("cost_yuan", 0)

    # Latency
    latencies = sorted(r.get("latency_ms", 0) for r in rows)
    n = len(latencies)
    p50 = latencies[n // 2] if n else 0
    p95 = latencies[min(n - 1, int(n * 0.95))] if n else 0
    avg = sum(latencies) // n if n else 0

    # Cooldown (live)
    try:
        from anchor.cooldown import all_cooling
        cooling = all_cooling()
    except Exception:
        cooling = {}

    # Quality (Day 18: heuristic judge_score per row)
    quality_scores = [r.get("judge_score") for r in rows if r.get("judge_score") is not None]
    avg_quality = round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None
    quality_by_worker = {}
    for r in rows:
        if r.get("judge_score") is None:
            continue
        w = r.get("model_used", "?")
        quality_by_worker.setdefault(w, []).append(r["judge_score"])
    quality_by_worker = {k: round(sum(v)/len(v), 4) for k, v in quality_by_worker.items()}

    # Quality tier distribution
    tier_counts_q = Counter()
    for r in rows:
        qt = r.get("quality_tier")
        if qt:
            tier_counts_q[qt] += 1

    return {
        **base,
        "date": date or time.strftime("%Y-%m-%d"),
        "n_queries": len(rows),
        "n_unique_sessions": n_unique_sessions,
        "total_cost_yuan": round(total_cost, 6),
        "avg_cost_yuan": round(total_cost / n, 6) if n else 0,
        "latency_ms": {"avg": avg, "p50": p50, "p95": p95, "min": latencies[0] if n else 0, "max": latencies[-1] if n else 0},
        "tier_distribution": dict(tier_counts.most_common()),
        "worker_distribution": dict(worker_counts.most_common()),
        "cost_by_worker": {k: round(v, 6) for k, v in sorted(cost_by_worker.items(), key=lambda x: -x[1])},
        "cooling_now": cooling,
        "avg_quality_score": avg_quality,
        "quality_by_worker": quality_by_worker,
        "quality_tier_distribution": dict(tier_counts_q.most_common()),
    }


def metrics_snapshot_for_range(days: int = 7) -> dict:
    """Aggregate over the last N days."""
    from datetime import datetime, timedelta
    today = datetime.now()
    rows = []
    for i in range(days):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        fp = SESSIONS_DIR / f"{date}.jsonl"
        if fp.exists():
            with open(fp) as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
    if not rows:
        return {"n_queries": 0, "window_days": days, "message": "no data in window"}
    return {
        "window_days": days,
        "n_queries": len(rows),
        "total_cost_yuan": round(sum(r.get("cost_yuan", 0) for r in rows), 6),
        **metrics_snapshot(),
    }


# === v0.9.46j: cost burn-rate computation ===

def _burn_rate_per_tier(window_hours: float = 1.0) -> dict:
    """Compute yuan/hour per tier over the last N hours from cost_log.jsonl."""
    from anchor.cost import LOG_PATH
    if not LOG_PATH.exists():
        return {}
    cutoff = time.time() - window_hours * 3600.0
    by_tier = {}
    with LOG_PATH.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("ts", 0) < cutoff:
                continue
            tier = e.get("tier", "unknown")
            by_tier[tier] = by_tier.get(tier, 0.0) + float(e.get("yuan", 0.0))
    return {t: round(v / window_hours, 4) for t, v in by_tier.items()}


def burn_rate_alerts(window_hours: float = 24.0) -> list:
    """Return alerts for tiers whose burn rate would breach cap within 7 days."""
    from anchor.cost import HARD_CAP_YUAN, WARN_THRESHOLD_YUAN
    burn = _burn_rate_per_tier(window_hours=window_hours)
    alerts = []
    horizon_days = 7
    for tier, yuan_per_hour in burn.items():
        projected_7d = yuan_per_hour * 24 * horizon_days
        if HARD_CAP_YUAN > 0 and projected_7d >= HARD_CAP_YUAN:
            alerts.append({
                "tier": tier,
                "burn_yuan_per_hour": yuan_per_hour,
                "projected_7d_yuan": round(projected_7d, 2),
                "severity": "critical",
                "msg": f"{tier}: {yuan_per_hour:.2f} Y/h x 7d = {projected_7d:.0f} Y (>= hard cap {HARD_CAP_YUAN} Y)",
            })
        elif projected_7d >= WARN_THRESHOLD_YUAN:
            alerts.append({
                "tier": tier,
                "burn_yuan_per_hour": yuan_per_hour,
                "projected_7d_yuan": round(projected_7d, 2),
                "severity": "warning",
                "msg": f"{tier}: {yuan_per_hour:.2f} Y/h x 7d = {projected_7d:.0f} Y (>= warn {WARN_THRESHOLD_YUAN} Y)",
            })
    return alerts
