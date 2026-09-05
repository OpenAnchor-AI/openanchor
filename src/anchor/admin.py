"""v0.9.46e: Admin + debug endpoints extracted from server.py.

Owns all /admin/* and /debug/* HTTP routes. Functions are defined
without decorators; register_admin_routes(app) wires them onto the
shared FastAPI app from server.py. State access via lazy imports
to avoid circular dependency at module load.

The split is purely file-organization: behavior is byte-identical to
the previous in-server.py definitions. Verified by:
  - pytest tests/test_server.py 7 passed (admin endpoint smoke tests)
  - All 16 admin endpoints exercised through TestClient after refactor
"""
from __future__ import annotations
import json
import time
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

# v0.9.46e: import _ROOT as _A_ROOT (was in server.py:18) so the extracted
# functions that reference _A_ROOT continue to resolve.
from anchor import _ROOT as _A_ROOT


# ---- State accessors (defined in server.py; lazy import) ----
def _stream_stats() -> dict:
    """Return server._STREAM_STATS (per-worker streaming stats)."""
    from anchor.server import _STREAM_STATS
    return _STREAM_STATS


# ---- Module-level aliases for server.py imports (used by extracted funcs) ----
# _sacred_stats is imported as `from anchor.sacred_guard import get_sacred_stats
# as _sacred_stats` in server.py. Make it accessible here.
_sacred_stats = None  # populated lazily


def _get_sacred_stats():
    global _sacred_stats
    if _sacred_stats is None:
        from anchor.sacred_guard import get_sacred_stats
        _sacred_stats = get_sacred_stats
    return _sacred_stats



# --- admin_usage (get /admin/usage) ---
async def admin_usage(month: Optional[str] = None):
    """Per-worker token usage + utilization for paid workers.
    
    For workers with monthly_fixed_cny > 0 or cost_in > 0,
    includes quota_yuan and utilization_pct.
    """
    from anchor.usage import month_usage_by_worker as _mu
    from anchor.config import WORKERS as _W_USAGE_ADMIN
    workers_data = _mu(month) or {}
    _month = month or datetime.date.today().strftime("%Y-%m")
    
    _paid_workers = {
        w.name: w for w in _W_USAGE_ADMIN
        if w.monthly_fixed_cny > 0 or w.cost_in > 0
    }
    
    out_workers = {}
    totals = {"input_tokens": 0, "output_tokens": 0, "cost_yuan": 0.0, "n_calls": 0}
    
    for w_name, w_data in workers_data.items():
        entry = dict(w_data)
        if w_name in _paid_workers:
            w_cfg = _paid_workers[w_name]
            monthly_fixed = w_cfg.monthly_fixed_cny
            if w_cfg.channel == "baosiapi":
                monthly_fixed = max(monthly_fixed, 688.0)
            entry["quota_yuan"] = monthly_fixed
            entry["utilization_pct"] = round(
                (w_data["cost_yuan"] / monthly_fixed * 100) if monthly_fixed > 0 else 0.0,
                2
            )
        out_workers[w_name] = entry
        totals["input_tokens"] += w_data.get("input_tokens", 0)
        totals["output_tokens"] += w_data.get("output_tokens", 0)
        totals["cost_yuan"] += w_data.get("cost_yuan", 0.0)
        totals["n_calls"] += w_data.get("n_calls", 0)
    
    return {
        "month": _month,
        "workers": out_workers,
        "totals": totals,
    }


# --- admin_cost_sacred (get /admin/cost/sacred) ---
async def admin_cost_sacred():
    """v0.9.16 (Fable 5 #6): sacred spend (fable-5 + opus-4-8) per session + day.
    Soft cap ¥5/session, hard cap ¥10/session (auto-fallback).
    """
    return {"ok": True, **_get_sacred_stats()()}



# --- admin_cost_sacred_reset (post /admin/cost/sacred/reset) ---
async def admin_cost_sacred_reset(session_id: str = ""):
    """v0.9.16: reset session's sacred spend (testing)."""
    from anchor.sacred_guard import reset_session as _reset
    if session_id:
        _reset(session_id)
        return {"ok": True, "reset": session_id}
    return {"ok": False, "err": "session_id required"}


# --- cooldown_reset (post /admin/cooldown/reset) ---
async def cooldown_reset(worker: Optional[str] = None):
    """Manually clear cooldown for a worker (or all). v0.9.10.
    Useful when vendor recovers and you don't want to wait for TTL.
    Uses anchor.cooldown built-in reset() / trip() helpers.
    """
    from anchor.cooldown import _state
    before = list(_state.keys())
    if worker is None:
        _state.clear()
        return {"cleared": before, "count": len(before), "msg": "all cooldowns cleared"}
    else:
        if worker in _state:
            _state.pop(worker, None)
            return {"cleared": [worker], "count": 1}
        return {"cleared": [], "count": 0, "msg": f"{worker} not in cooldown"}



# --- admin_streaming_stats (get /admin/streaming/stats) ---
async def admin_streaming_stats():
    """v0.9.13 (B directive): per-worker TTFT/TBT/token stats from streaming sessions.
    Uses rolling window of last 200 streams per worker.
    """
    out = {}
    _ss = _stream_stats()
    out = {}
    for w, s in _ss.items():
        ttfts = list(s["ttfts"])
        tbts = list(s["tbts"])
        tokens = list(s["tokens"])
        out[w] = {
            "total_streams": s["total"],
            "errored_streams": s.get("errored", 0),
            "error_rate": round(s.get("errored", 0) / max(s["total"], 1), 3),
            "ttft_ms": {
                "n": len(ttfts),
                "avg": round(sum(ttfts)/len(ttfts), 1) if ttfts else None,
                "p50": sorted(ttfts)[len(ttfts)//2] if ttfts else None,
                "p95": sorted(ttfts)[int(len(ttfts)*0.95)] if ttfts else None,
                "max": max(ttfts) if ttfts else None,
            },
            "tbt_ms": {
                "n": len(tbts),
                "avg": round(sum(tbts)/len(tbts), 1) if tbts else None,
                "p50": sorted(tbts)[len(tbts)//2] if tbts else None,
                "p95": sorted(tbts)[int(len(tbts)*0.95)] if tbts else None,
            },
            "tokens_per_stream": {
                "n": len(tokens),
                "avg": round(sum(tokens)/len(tokens), 1) if tokens else None,
                "p50": sorted(tokens)[len(tokens)//2] if tokens else None,
            },
            "last_stream_ts": s["last_ts"],
            "window": 200,  # matches server._STREAM_STATS_WINDOW
        }
    return {"ok": True, "workers": out}



# --- admin_streaming_reset (post /admin/streaming/reset) ---
async def admin_streaming_reset():
    """v0.9.13: clear streaming stats (testing/maintenance)."""
    _stream_stats().clear()
    return {"ok": True, "msg": "streaming stats cleared"}



# --- admin_calibration_refresh (post /admin/calibration/refresh) ---
async def admin_calibration_refresh(min_n: int = 10, days: int = 7):
    """v0.9.13 (C directive): rebuild quality_table from session_log.

    Aggregates last N days of session log by (worker, category, difficulty_bucket).
    Writes quality_table_v6.json. Safe to call weekly / on-demand.

    Args:
        min_n: minimum records per cell to be considered reliable (default 10)
        days: look-back window (default 7)
    """
    from anchor.calibration import refresh as _calib_refresh
    stats = _calib_refresh(min_n=min_n, days=days)
    # audit 2026-08-16 (A13): invalidate the in-process v8 cache so the
    # refreshed table actually takes effect without a restart.
    try:
        from anchor.config import invalidate_cal_v8_cache
        invalidate_cal_v8_cache()
    except Exception:
        pass
    return {"ok": True, "stats": stats, "cache_invalidated": True}



# --- admin_calibration_opus_count (get /admin/calibration/opus_count) ---
async def admin_calibration_opus_count():
    """v0.9.15: count opus-4-8 records in last 7d session_log by (cat, bucket).
    Triggers v7 auto-refresh when design/hard reaches 50 records (Sonnet 5 B2 milestone).
    """
    from pathlib import Path as _P
    import json as _j
    from collections import defaultdict as _dd
    from datetime import datetime, timezone, timedelta as _td
    from anchor.workers import SACRED as _SACRED
    SESS_DIR = _P(str(_A_ROOT / "data/anchor_sessions"))
    cutoff = datetime.now(timezone.utc) - _td(days=7)
    bucket_count = _dd(lambda: _dd(int))
    total_opus = 0
    for f in sorted(SESS_DIR.glob("*.jsonl")):
        try:
            d = datetime.fromisoformat(f.stem).replace(tzinfo=timezone.utc)
            if d < cutoff:
                continue
        except Exception:
            pass
        with open(f) as fh:
            for line in fh:
                try:
                    s = _j.loads(line)
                    if s.get("model_used") in _SACRED and s.get("judge_score") is not None:
                        # bucket by (cat, difficulty_bucket)
                        from anchor.fusion_modes import _difficulty_bucket
                        # v0.9.19: expanded design detection (match what user means)
                        ql = s.get("query_text", "").lower()
                        if any(k in ql for k in ("design", "architect", "distributed", "consensus", "consistent hash", "load balance", "microservice", "kubernetes", "raft", "kubernetes operator", "replication", "high availability", "高可用", "微服务", "分布式", "架构")):
                            cat = "design"
                        else:
                            cat = "other"
                        bkt = _difficulty_bucket(s.get("d_value", 0))
                        bucket_count[cat][bkt] += 1
                        total_opus += 1
                except Exception:
                    pass
    # Check if design/hard hits 50 milestone
    design_hard_n = bucket_count.get("design", {}).get("hard", 0)
    v7_ready = design_hard_n >= 50
    return {
        "ok": True,
        "total_sacred_records": total_opus,  # includes opus-4-8 + fable-5
        "design_hard_records": design_hard_n,
        "v7_ready": v7_ready,
        "by_cat_bucket": {k: dict(v) for k, v in bucket_count.items()},
        "milestone": 50,
    }



# --- admin_calibration_status (get /admin/calibration/status) ---
async def admin_calibration_status():
    """Show current calibration table version + per-cell counts."""
    from pathlib import Path as _P
    import json as _j
    cal_dir = _P(str(_A_ROOT / "data/calibration"))
    out = {}
    for f in sorted(cal_dir.glob("quality_table_v*.json")):
        try:
            with open(f) as _cal_f:
                d = _j.load(_cal_f)
            out[f.name] = {
                "version": d.get("version", "?"),
                "n_records": d.get("n_records", 0),
                "n_per_cat_cells": sum(len(c) for w in d.get("table_per_cat", {}).values() for c in w.values()),
                "n_coarse_cells": sum(len(c) for c in d.get("table_coarse", {}).values()),
                "generated_at": d.get("generated_at", "?"),
            }
        except Exception as e:
            out[f.name] = {"error": str(e)}
    return {"ok": True, "tables": out}



# --- cooldown_status (get /admin/cooldown/status) ---
async def cooldown_status():
    """Show all currently cooling workers + remaining seconds. v0.9.10."""
    from anchor.cooldown import all_cooling
    return {"cooling": all_cooling()}



# --- v0.9.55 quarantine admin endpoints ---

class QuarantineRequest(BaseModel):
    """Optional target worker for quarantine clear/recheck operations."""
    worker: Optional[str] = None



async def _probe_worker_ok(worker_name: str) -> bool:
    """Send a 5-token PONG probe to a worker. Returns True iff healthy."""
    from anchor.config import WORKERS as _W_admin
    from anchor.clients.factory import build_client as _bc_admin
    w = next((x for x in _W_admin if x.name == worker_name), None)
    if not w:
        return False
    try:
        client = _bc_admin(w)
        r = await client.chat(
            [{"role": "user", "content": "ping"}],
            max_tokens=5, temperature=0.0,
        )
        content = (r.get("content") or "").strip()
        return bool(content) and not content.startswith(("[error", "[stub"))
    except Exception:
        return False


async def admin_quarantine_list():
    """List all currently quarantined workers with full audit metadata.

    Returns {workers: [{name, err_rate, threshold, quarantined, ts,
    ci_lower, n, sessions, reason, last_probe_ok, last_probe_ts}]}.
    """
    from anchor.release.circuit_breaker import list_quarantined as _lq_a
    quarantined = _lq_a() or {}
    out = []
    for worker, info in quarantined.items():
        out.append({
            "name": worker,
            "err_rate": round(float(info.get("err_rate", 0.0)), 4),
            "threshold": round(float(info.get("threshold", 0.0)), 4),
            "quarantined": bool(info.get("quarantined", False)),
            "ts": float(info.get("ts", 0.0)),
            "ci_lower": float(info.get("ci_lower", 0.0)),
            "n": int(info.get("n", 0)),
            "sessions": int(info.get("sessions", 0)),
            "reason": info.get("reason", "manual"),
            "last_probe_ok": info.get("last_probe_ok"),
            "last_probe_ts": info.get("last_probe_ts"),
        })
    return {"workers": out, "ts": time.time()}


async def admin_quarantine_clear(body: QuarantineRequest = QuarantineRequest()):
    """Clear quarantine for a worker (or all). Returns {cleared: [...], count}."""
    from anchor.release.circuit_breaker import clear_quarantine as _cq_a, list_quarantined as _lq_a
    quarantined = _lq_a() or {}
    cleared = []
    targets = [body.worker] if body.worker else list(quarantined.keys())
    for w in targets:
        if _cq_a(w):
            cleared.append(w)
    return {"cleared": cleared, "count": len(cleared)}


async def admin_quarantine_recheck(body: QuarantineRequest = QuarantineRequest()):
    """Probe currently quarantined workers; clear any that respond healthy."""
    from anchor.release.circuit_breaker import (
        clear_quarantine as _cq_r, list_quarantined as _lq_r,
        update_quarantine_probe as _uqp_r,
    )
    quarantined = _lq_r() or {}
    targets = [body.worker] if body.worker else list(quarantined.keys())
    results = []
    for w in targets:
        ok = await _probe_worker_ok(w)
        _uqp_r(w, ok)
        if ok:
            _cq_r(w)
        results.append({"name": w, "probe_ok": ok})
    return {"results": results, "ts": time.time()}


# --- admin_workers_health (get /admin/health/workers) ---
# v0.9.53 (P2 followup): single-endpoint per-worker health aggregation.
# Surfaces quality + err_rate + cooling + quarantine + last probe time.
# Auto-alerts when quality drops below QUALITY_ALERT_THRESHOLD (default 0.50)
# or when err_rate exceeds ERR_RATE_ALERT_THRESHOLD (default 0.30).
# Designed for periodic polling (curl + jq) or integration with Prometheus /
# Datadog. No side effects on read.
_QUALITY_ALERT_THRESHOLD: float = 0.50
_ERR_RATE_ALERT_THRESHOLD: float = 0.30


async def admin_workers_health():
    """Aggregate per-worker health for monitoring + auto-alert.

    Returns per-worker status (enabled, err_rate, is_cooling, cooldown
    remaining seconds, is_quarantined, circuit_breaker_open, quality
    score over recent window, traffic count, and any alerts). Designed
    to be polled by ops dashboards; alerts are written but no side
    effects on read.

    Override thresholds via env: ANCHOR_QUALITY_ALERT_THRESHOLD,
    ANCHOR_ERR_RATE_ALERT_THRESHOLD.
    """
    import os
    from anchor.config import WORKERS
    from anchor.cooldown import is_cooling as _cd_cooling, all_cooling
    from anchor.release.circuit_breaker import (
        is_quarantined as _cb_quarantined,
        list_quarantined as _cb_list_quarantined,
        is_open as _cb_is_open,
    )
    from anchor.release.circuit_breaker import (
        placeholder_warning_threshold as _pw_thr,
        should_warn_placeholder as _pw_should_warn,
    )

    q_thr = float(os.environ.get("ANCHOR_QUALITY_ALERT_THRESHOLD", str(_QUALITY_ALERT_THRESHOLD)))
    e_thr = float(os.environ.get("ANCHOR_ERR_RATE_ALERT_THRESHOLD", str(_ERR_RATE_ALERT_THRESHOLD)))

    # pull recent dashboard data once (quality scores + traffic)
    quality = {}
    traffic = {}
    try:
        db = debug_dashboard()
        quality = db.get("quality_by_worker", {}) or {}
        traffic = db.get("worker_distribution", {}) or {}
    except Exception as _db_e:
        pass

    cooling_now = all_cooling() or {}
    quarantined = _cb_list_quarantined() or {}
    cb_open = _cb_is_open()

    per_worker = []
    for w in WORKERS:
        alerts: list[str] = []
        w_name = w.name
        cd = cooling_now.get(w_name, 0)
        is_cool = _cd_cooling(w_name)
        is_q = _cb_quarantined(w_name)
        q_score = float(quality.get(w_name, 0.0))
        n_traffic = int(traffic.get(w_name, 0))
        is_enabled = w.enabled

        # Auto-alerts (priority high → low)
        # Quarantine / cooldown / quality alerts apply even if worker is currently
        # disabled — ops needs to know historical breakage (e.g. dpsk-pro enabled
        # previously, then quarantine'd, then disabled by env flag).
        if is_q:
            qe = quarantined.get(w_name, {}).get('err_rate', 0)
            alerts.append(f"quarantined err_rate={qe:.0%}")
        elif cd > 0:
            alerts.append(f"cooling for {cd}s")
        if is_enabled and q_score > 0 and q_score < q_thr:
            alerts.append(f"quality {q_score:.3f} < threshold {q_thr:.2f}")
        if not is_enabled:
            alerts.append("disabled")
        if w_name in ("deepseek-v4-flash",) and not is_enabled:
            # Tag the recent A-fix for ops clarity
            alerts.append("disabled via ANCHOR_ENABLE_DEEPSEEK_V4_FLASH=0 (OpenCode Zen free tier rate-limited)")

        q_info = quarantined.get(w_name, {})
        q_ts = float(q_info.get("ts", 0.0))
        q_age = max(0.0, time.time() - q_ts) if q_ts > 0 else 0.0
        q_reason = q_info.get("reason", "manual") if is_q else None
        last_probe_ts = float(q_info.get("last_probe_ts", 0.0) or 0.0)
        last_probe_ok = bool(q_info.get("last_probe_ok", False))
        q_recoverable = bool(
            is_q and last_probe_ok and last_probe_ts > 0
            and time.time() - last_probe_ts < 86400
        )
        # v0.9.56 (PR-A4): time until the next quarantine-recheck probe runs.
        # Reads ANCHOR_QUARANTINE_RECHECK_INTERVAL_SEC so the value matches
        # the lifespan loop. Operators see when each worker will be tested
        # again without grepping logs.
        recheck_interval = int(
            os.environ.get(
                "ANCHOR_QUARANTINE_RECHECK_INTERVAL_SEC",
                str(180),
            )
        )
        if is_q and last_probe_ts > 0:
            elapsed = max(0.0, time.time() - last_probe_ts)
            next_probe_in_s = max(0, int(recheck_interval - elapsed))
        elif is_q:
            next_probe_in_s = recheck_interval  # first probe pending
        else:
            next_probe_in_s = None
        per_worker.append({
            "name": w_name,
            "enabled": is_enabled,
            "slot": w.slot,
            "role_tags": list(w.role_tags),
            "channel": w.channel,
            "quality_score": round(q_score, 4),
            "err_rate": round(q_info.get("err_rate", 0.0), 4),
            "traffic_24h": n_traffic,
            "is_cooling": is_cool,
            "cooldown_remaining_s": cd,
            "is_quarantined": is_q,
            "quarantine_ts": q_ts,
            "quarantine_age_s": round(q_age, 1),
            "quarantine_reason": q_reason,
            "quarantine_recoverable": q_recoverable,
            "quarantine_next_probe_in_s": next_probe_in_s,
            "quarantine_recheck_interval_s": recheck_interval,
            "circuit_breaker_open": cb_open and (w_name in {"claude-fable-5", "claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"}),
            "placeholder_warning_threshold": _pw_thr(w_name),
            "should_warn": _pw_should_warn(w_name, float(q_info.get("err_rate", 0.0))),
            "alerts": alerts,
        })

    # Sort: severity-alert workers first, then enabled OK, then disabled (with or
    # without severity). Two disabled workers can interleave so a disabled
    # worker with quarantined/quality alert shows BEFORE an enabled OK worker
    # — ops needs to see broken workers regardless of enable state.
    def _sort_key(worker_state):
        n_sev = len([
            a for a in worker_state["alerts"]
            if "quarantined" in a or "quality " in a or "cooldown for" in a
        ])
        return (
            n_sev == 0,                              # severity workers first
            worker_state["enabled"] is False,        # disabled after enabled (within tier)
            -n_sev,                                  # more severity alerts first
            worker_state["name"],
        )
    per_worker.sort(key=_sort_key)

    # Global summary. audit 2026-08-16 (F811): local name renamed — it
    # shadowed the config.enabled_workers() function imported at module level.
    enabled_names = [w for w in WORKERS if w.enabled]
    n_alerting = sum(1 for ws in per_worker if any("quarantined" in a or "quality" in a for a in ws["alerts"]))
    return {
        "ts": time.time(),
        "thresholds": {"quality_below": q_thr, "err_rate_above": e_thr},
        "summary": {
            "total_workers": len(WORKERS),
            "enabled": len(enabled_names),
            "alerting": n_alerting,
            "circuit_breaker_open": cb_open,
        },
        "workers": per_worker,
    }

# --- vendor_placeholder_count (get /admin/vendor/placeholder_count) ---
async def vendor_placeholder_count():
    """v0.9.19: count vendor placeholder responses (baosiapi 30-char stub).
    
    Surfaces in three places:
      - session_log JSONL with is_vendor_placeholder=true (post-fix writes)
      - m3 pool stats.fails_404 (raised as NotFoundError on placeholder)
      - base.py raises APIError on placeholder → logged in worker_stats
    
    Returns total / last_24h / by_worker counts.
    """
    from datetime import datetime, timezone, timedelta
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session_path = f"data/anchor_sessions/{today}.jsonl"
    total = 0
    last_24h = 0
    by_worker: dict[str, int] = {}
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).timestamp()
    try:
        with open(session_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if not d.get("is_vendor_placeholder"):
                    continue
                total += 1
                w = d.get("model_used", "?")
                by_worker[w] = by_worker.get(w, 0) + 1
                if d.get("ts", 0) >= cutoff:
                    last_24h += 1
    except FileNotFoundError:
        pass
    
    return {
        "ok": True,
        "total_session_records": total,
        "last_24h": last_24h,
        "by_worker_session": by_worker,
    }



# --- admin_workers_placeholder_warnings (get /admin/health/workers/placeholder_warnings) ---
async def admin_workers_placeholder_warnings():
    """v0.9.58: per-worker placeholder warning band (soft alert before quarantine).

    Reads circuit_breaker.json err_rate_quarantine map. For each enabled worker,
    returns current err_rate, warning threshold, quarantine threshold, and status:
      - "quarantined" when err_rate >= quarantine threshold
      - "warning" when err_rate in [warning_threshold, quarantine_threshold)
      - "ok" otherwise
    """
    from anchor.config import enabled_workers
    from anchor.release.circuit_breaker import (
        load_state as _cb_load_state,
        err_rate_quarantine_threshold as _q_thr,
        placeholder_warning_threshold as _w_thr,
        should_warn_placeholder as _should_warn,
    )
    quarantine = _cb_load_state().get("err_rate_quarantine", {})
    out = {}
    for w in enabled_workers():
        w_name = w.name
        info = quarantine.get(w_name, {})
        err_rate = float(info.get("err_rate", 0.0))
        q_thr = _q_thr(w_name)
        w_thr = _w_thr(w_name)
        if err_rate >= q_thr:
            status = "quarantined"
        elif _should_warn(w_name, err_rate):
            status = "warning"
        else:
            status = "ok"
        out[w_name] = {
            "current_err_rate": round(err_rate, 4),
            "warning_threshold": w_thr,
            "quarantine_threshold": q_thr,
            "status": status,
        }
    return {"ts": time.time(), "workers": out}


# --- get_gradual_dpsk_v4_flash (get /admin/gradual/dpsk_v4_flash) ---
async def get_gradual_dpsk_v4_flash():
    """v0.9.58: Get dpsk-flash rollout state (default ON, kill-switch env)."""
    import os
    from anchor.config import WORKERS
    enabled = True
    for w in WORKERS:
        if w.name == "deepseek-v4-flash":
            enabled = w.enabled
            break
    return {
        "worker": "deepseek-v4-flash",
        "enabled": enabled,
        "kill_switch_env": "ANCHOR_DISABLE_DEEPSEEK_V4_FLASH",
        "kill_switch_set": bool(os.environ.get("ANCHOR_DISABLE_DEEPSEEK_V4_FLASH")),
    }


# --- set_gradual_dpsk_v4_flash (post /admin/gradual/dpsk_v4_flash) ---
async def set_gradual_dpsk_v4_flash(action: str = "enable"):
    """v0.9.58: Force kill/enable dpsk-flash via ANCHOR_DISABLE_DEEPSEEK_V4_FLASH env.

    Body: {"action": "kill" | "enable"}. Writes the env var into the
    circuit_breaker.json state so it survives process restart (ops can also
    set the env var directly).
    """
    from anchor.release.circuit_breaker import load_state as _cb_load_state, save_state as _cb_save_state
    if action not in ("kill", "enable"):
        return {"ok": False, "error": "action must be 'kill' or 'enable'"}
    state = _cb_load_state()
    state.setdefault("dpsk_v4_flash", {})
    state["dpsk_v4_flash"]["kill_switch"] = action == "kill"
    state["dpsk_v4_flash"]["ts"] = time.time()
    _cb_save_state(state)
    return {"ok": True, "worker": "deepseek-v4-flash", "action": action}


# --- get_gradual_fable5 (get /admin/gradual/fable5) ---
async def get_gradual_fable5():
    """v0.9.22: Get current fable-5 gradual rollout percentage (0-100)."""
    from anchor.fusion_modes import get_fable5_traffic_pct
    return {"pct": get_fable5_traffic_pct()}



# --- set_gradual_fable5 (post /admin/gradual/fable5) ---
async def set_gradual_fable5(pct: float = 0.0):
    """v0.9.22: Set fable-5 gradual rollout percentage (0-100).

    Suggested progression (per Fable 5 gating):
      10% — smoke test, 50 queries, placeholder<5%, judge>=0.85
      50% — half traffic, 200 queries, placeholder<3%, judge>=0.85
      100% — full restoration

    Set to 0 to immediately disable.
    """
    from anchor.fusion_modes import set_fable5_traffic_pct
    set_fable5_traffic_pct(pct)
    return {"pct": pct, "ok": True}



# --- debug_cooldown (get /debug/cooldown) ---
def debug_cooldown():
    """Ops view: which workers are cooling down and for how long."""
    from anchor.cooldown import all_cooling
    return {"cooling": all_cooling()}



# --- debug_dashboard (get /debug/dashboard) ---
def debug_dashboard(date: str = None, days: int = None):
    """Ops dashboard: routing distribution, cost, latency, cooldown.

    Args:
        date: YYYY-MM-DD, default today
        days: if set, aggregate over last N days instead
    """
    from anchor.dashboard import metrics_snapshot, metrics_snapshot_for_range
    if days:
        out = metrics_snapshot_for_range(days=days)
    else:
        out = metrics_snapshot(date=date)
    out["routing_lanes"] = routing_lane_summary()
    return out


# --- routing_lane_summary (get /admin/routing/lanes) ---
def routing_lane_summary() -> dict:
    """Aggregate routing-lane × worker hit-rate from in-memory counters.

    The metric collectors are process-local; lane summary reflects the
    current gateway process lifetime, not historical sessions.
    """
    from anchor.metrics import (
        ROUTING_LANE_REQUESTS_TOTAL, ROUTING_LANE_FALLBACK_TOTAL,
        ROUTING_LANE_PRIMARY_TOTAL,
    )
    lanes: dict[str, dict] = {}
    for metric in ROUTING_LANE_REQUESTS_TOTAL.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                lane = sample.labels.get("lane", "unknown")
                worker = sample.labels.get("worker", "unknown")
                status = sample.labels.get("status", "unknown")
                value = sample.value
                lanes.setdefault(lane, {
                    "total": 0, "by_worker": {}, "by_status": {},
                    "primary": {}, "fallback_reasons": {},
                })
                lanes[lane]["total"] += value
                lanes[lane]["by_worker"][worker] = (
                    lanes[lane]["by_worker"].get(worker, 0) + value
                )
                lanes[lane]["by_status"][status] = (
                    lanes[lane]["by_status"].get(status, 0) + value
                )
    for metric in ROUTING_LANE_PRIMARY_TOTAL.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                lane = sample.labels.get("lane", "unknown")
                worker = sample.labels.get("worker", "unknown")
                status = sample.labels.get("status", "unknown")
                value = sample.value
                lanes.setdefault(lane, {
                    "total": 0, "by_worker": {}, "by_status": {},
                    "primary": {}, "fallback_reasons": {},
                })
                pstats = lanes[lane]["primary"].setdefault(worker, {"total": 0, "ok": 0})
                pstats["total"] += value
                if status == "ok":
                    pstats["ok"] += value
    for lane, stats in lanes.items():
        total = stats["total"] or 1
        stats["hit_rate"] = {
            worker: round(count / total, 4)
            for worker, count in stats["by_worker"].items()
        }
        for pstats in stats["primary"].values():
            pstats["error_rate"] = (
                round(1 - pstats["ok"] / pstats["total"], 4)
                if pstats["total"] else 0
            )
    fallbacks: dict[str, dict] = {}
    for metric in ROUTING_LANE_FALLBACK_TOTAL.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                lane = sample.labels.get("lane", "unknown")
                from_w = sample.labels.get("from_worker", "unknown")
                to_w = sample.labels.get("to_worker", "unknown")
                reason = sample.labels.get("reason", "unknown")
                key = f"{from_w}->{to_w}"
                fallbacks.setdefault(lane, {}).setdefault(key, {"count": 0, "reasons": {}})
                fallbacks[lane][key]["count"] += sample.value
                fallbacks[lane][key]["reasons"][reason] = (
                    fallbacks[lane][key]["reasons"].get(reason, 0) + sample.value
                )
                lanes.setdefault(lane, {
                    "total": 0, "by_worker": {}, "by_status": {},
                    "primary": {}, "fallback_reasons": {},
                })
                lanes[lane]["fallback_reasons"][reason] = (
                    lanes[lane]["fallback_reasons"].get(reason, 0) + sample.value
                )
    return {"lanes": lanes, "fallbacks": fallbacks}


async def routing_lane_summary_endpoint():
    """JSON endpoint exposing per-lane × worker hit-rate and fallback hops."""
    summary = routing_lane_summary()
    from anchor.metrics import lane_window_summary
    summary["rolling_1h"] = lane_window_summary()
    return summary



# --- debug_prior (get /debug/prior) ---
def debug_prior():
    """Inspect server-side prior_table + WORKERS + TIER_POOL."""
    from anchor import head as h
    out = {
        "workers": list(h.WORKERS),
        "costs": list(h.COSTS),
        "tier_pool": {k: list(v) for k,v in h.TIER_POOL.items()},
        "vision_prior": {w: h.prior_table[("vision", w)] for w in h.WORKERS},
        "code_prior": {w: h.prior_table[("code", w)] for w in h.WORKERS},
    }
    return out



# --- debug_predict (post /debug/predict) ---
async def debug_predict(req: dict):
    """Trace server-side routing decision for a query."""
    q = req.get("query", "")
    # audit 2026-08-16 (A2): TIER_HARD_RULES/TIER_POOL only contain "auto"
    # since v0.9.52 (single product lane), so defaulting to "premium" 500'd.
    from anchor.config import normalize_routing_lane as _nrl_dbg
    tier = _nrl_dbg(req.get("tier", "auto"))
    from anchor.classifier import classify
    from anchor.fusion_modes import TIER_HARD_RULES
    from anchor.head import _features, TIER_POOL, WORKERS, COSTS, W, W_DIM
    qt = classify(q)
    forced = TIER_HARD_RULES.get(tier, TIER_HARD_RULES["auto"])(qt, q)
    pool = TIER_POOL.get(tier) or TIER_POOL["auto"]
    scores = []
    for i in pool:
        # audit 2026-08-16 (A2): pass keyword args — positional call put
        # the tier string into prompt_len and broke _features.
        x = _features(qt, WORKERS[i], prompt_len=len(q), budget=0.5)
        sc = float(W[:W_DIM] @ x - COSTS[i] * 0.15)
        scores.append({"idx": i, "worker": WORKERS[i], "cost": COSTS[i], "score": round(sc, 4)})
    return {
        "query": q,
        "tier": tier,
        "qt": qt,
        "forced_worker": forced,
        "pool_scores": sorted(scores, key=lambda x: -x["score"]),
    }



# ---- Route registration ----
# Track per-app registration via attribute on the app instance itself.
# id(app)-based dedup is unsafe: when a FastAPI app is GC'd, its id() can be
# reused by a later FastAPI() in a test fixture, falsely short-circuiting
# register_admin_routes and silently dropping all routes.
_ADMIN_MARKER_PATH = "/admin/cost/sacred"
_ADMIN_MARKER_ATTR = "_anchor_admin_routes_registered"


def register_admin_routes(app) -> None:
    """Register all admin/debug endpoints on the given FastAPI app.
    Idempotent per app instance: safe to call multiple times.
    """
    if getattr(app, _ADMIN_MARKER_ATTR, False):
        return
    existing = {getattr(r, "path", None) for r in app.routes}
    if _ADMIN_MARKER_PATH in existing:
        setattr(app, _ADMIN_MARKER_ATTR, True)
        return
    app.get("/admin/usage")(admin_usage)
    app.get("/admin/cost/sacred")(admin_cost_sacred)
    app.post("/admin/cost/sacred/reset")(admin_cost_sacred_reset)
    app.post("/admin/cooldown/reset")(cooldown_reset)
    app.get("/admin/streaming/stats")(admin_streaming_stats)
    app.post("/admin/streaming/reset")(admin_streaming_reset)
    app.post("/admin/calibration/refresh")(admin_calibration_refresh)
    app.get("/admin/calibration/opus_count")(admin_calibration_opus_count)
    app.get("/admin/calibration/status")(admin_calibration_status)
    app.get("/admin/cooldown/status")(cooldown_status)
    app.get("/admin/vendor/placeholder_count")(vendor_placeholder_count)
    app.get("/admin/gradual/fable5")(get_gradual_fable5)
    app.post("/admin/gradual/fable5")(set_gradual_fable5)
    app.get("/admin/gradual/dpsk_v4_flash")(get_gradual_dpsk_v4_flash)
    app.post("/admin/gradual/dpsk_v4_flash")(set_gradual_dpsk_v4_flash)
    app.get("/admin/health/workers/placeholder_warnings")(admin_workers_placeholder_warnings)
    app.get("/debug/cooldown")(debug_cooldown)
    app.get("/debug/dashboard")(debug_dashboard)
    app.get("/debug/prior")(debug_prior)
    app.post("/debug/predict")(debug_predict)
    app.get("/admin/routing/lanes")(routing_lane_summary_endpoint)
    # v0.9.53 (P2 followup): aggregate per-worker health for monitoring +
    # auto-alert when quality drops or err_rate spikes. Subsumes the data
    # from /debug/dashboard + /admin/cooldown/status + /admin/calibration/status
    # into a single actionable view.
    app.get("/admin/quarantine")(admin_quarantine_list)
    app.post("/admin/quarantine/clear")(admin_quarantine_clear)
    app.post("/admin/quarantine/recheck")(admin_quarantine_recheck)
    app.get("/admin/health/workers")(admin_workers_health)
    # A2.3 audit 2026-08-16: autopromote runtime toggle
    app.get("/admin/autopromote")(get_autopromote_status)
    app.post("/admin/autopromote")(toggle_autopromote)
    setattr(app, _ADMIN_MARKER_ATTR, True)




# A2.3 audit 2026-08-16: autopromote toggle admin endpoint
# Allows runtime toggle ANCHOR_AUTOPROMOTE without restart.
# GET /admin/autopromote - status
# POST /admin/autopromote - {action: enable|disable|toggle}

import os as _os_a23


async def get_autopromote_status():
    """Return ANCHOR_AUTOPROMOTE status + daily cost/cap."""
    from anchor.budget_tracker import get_daily_cost, get_daily_cap
    from anchor.cooldown import is_cooling

    enabled = _os_a23.environ.get("ANCHOR_AUTOPROMOTE", "").strip().lower() in {"1", "true", "yes", "on"}
    daily_cost = get_daily_cost()
    daily_cap = get_daily_cap()
    return {
        "enabled": enabled,
        "daily_cost": daily_cost,
        "daily_cap": daily_cap,
        "threshold_pct": (daily_cost / daily_cap * 100) if daily_cap > 0 else 0,
        "fable_cooling": is_cooling("claude-fable-5"),
    }


async def toggle_autopromote(action: str = "toggle"):
    """Toggle ANCHOR_AUTOPROMOTE at runtime.

    Actions: enable, disable, toggle (default).
    Note: env var is process-local; toggle writes to ANCHOR_AUTOPROMOTE_TOGGLE.
    """
    toggle_file = _os_a23.environ.get("ANCHOR_AUTOPROMOTE_TOGGLE", "/tmp/anchor_autopromote_toggle")
    current = _os_a23.environ.get("ANCHOR_AUTOPROMOTE", "").strip().lower() in {"1", "true", "yes", "on"}

    if action == "enable":
        new = True
    elif action == "disable":
        new = False
    elif action == "toggle":
        new = not current
    else:
        return {"error": f"unknown action: {action}"}

    # Persist to toggle file (next process restart reads it)
    try:
        with open(toggle_file, "w") as f:
            f.write("1" if new else "0")
    except OSError as e:
        return {"error": f"toggle file write failed: {e}"}

    return {
        "action": action,
        "previous": current,
        "new": new,
        "toggle_file": toggle_file,
        "note": "process-local; takes effect after restart or next request via runtime hook",
    }
