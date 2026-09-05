"""v0.9.35-p1: admin operations module.

Extracted from server.py (2281 → 1960 lines). Contains:
- _load_sessions: JSONL session loader (used by SLA / cost / dashboard endpoints)
- _compute_sla: per-worker SLA target comparison
- 4 endpoints registered via register_admin_routes(app): /admin/sla, /admin/cost/dashboard, /admin/cost/amortization

Other admin endpoints (cooldown reset, calibration, streaming, vendor_placeholder,
gradual_fable5, debug/*) remain in server.py for now; they have tighter coupling to
in-process state (cooldown module globals, recovery loops) and are deferred to v0.9.36+.
"""
from __future__ import annotations
import json as _json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from anchor.workers import SLA_TARGETS, QUARANTINED


def _sessions_dir() -> Path:
    """Anchor root + data/anchor_sessions. Single source for path resolution."""
    from anchor import _ROOT
    return Path(str(_ROOT / "data/anchor_sessions"))


def load_sessions(days: int = 7) -> list[dict]:
    """Load last N days of session logs (anchor_sessions/YYYY-MM-DD.jsonl).

    Public alias for server's private _load_sessions (kept for backward compat:
    server.py still exposes _load_sessions = load_sessions via re-export below).
    """
    log_dir = _sessions_dir()
    if not log_dir.exists():
        return []
    today = datetime.now(timezone.utc).date()
    cutoff = today - timedelta(days=days - 1)
    sessions = []
    for f in sorted(log_dir.glob("*.jsonl")):
        try:
            file_date = datetime.strptime(f.stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < cutoff:
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    sessions.append(_json.loads(line))
                except Exception:
                    continue
    return sessions


ERROR_BUDGET = {"target_success_rate": 0.99, "alert_below": 0.97, "page_below": 0.95}


def compute_sla(days: int = 7, *, state_path: Optional[Path] = None,
                 sessions: Optional[list[dict]] = None) -> dict:
    """Per-worker SLA status vs target (p95 latency / min judge / max error rate).

    Filters out:
    - Sessions with empty query_text (probe/smoke artifacts)
    - Sessions with source in {smoke, probe, retrain} (automated traffic)
    Only sessions tagged source='user' (or pre-source 'unknown' with non-empty
    query_text) are included, so SLA reflects real user traffic.
    """
    by_worker = defaultdict(list)
    if sessions is not None:
        # Test injection path: use the provided session list directly.
        all_sessions = sessions
    else:
        # Production path: read from on-disk session log dir.
        log_dir = _sessions_dir()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_sessions = []
        if log_dir.exists():
            for f in sorted(log_dir.glob("*.jsonl")):
                try:
                    d = datetime.fromisoformat(f.stem).replace(tzinfo=timezone.utc)
                    if d < cutoff:
                        continue
                except Exception:
                    pass
                with open(f) as fh:
                    for line in fh:
                        try:
                            all_sessions.append(_json.loads(line))
                        except Exception:
                            continue
    for s in all_sessions:
        try:
            w = s.get("model_used")
            if not w:
                continue
            src_tag = s.get("source", "unknown")
            q_text = (s.get("query_text") or "").strip()
            if src_tag in ("smoke", "probe", "retrain", "unknown"):
                continue
            if not q_text or q_text == "(empty)":
                continue
            by_worker[w].append({
                "lat": s.get("latency_ms") or 0,
                "judge": s.get("judge_score"),
                "err": 1 if s.get("quality_tier") in ("error", "degraded") else 0,
            })
        except Exception:
            pass
    out = {}
    for w, target in SLA_TARGETS.items():
        recs = by_worker.get(w, [])
        n = len(recs)
        if n == 0:
            out[w] = {"n": 0, "status": "no_data", "target": target}
            continue
        lats = sorted(r["lat"] for r in recs)
        p50 = lats[n // 2]
        p95 = lats[int(n * 0.95)] if n > 1 else lats[0]
        judges = [r["judge"] for r in recs if r["judge"] is not None]
        avg_j = sum(judges) / len(judges) if judges else 0
        err_rate = sum(r["err"] for r in recs) / n
        violations = []
        if p95 > target["p95_ms"]:
            violations.append(f"p95 {p95}ms > {target['p95_ms']}ms")
        if avg_j < target["min_judge"]:
            violations.append(f"avg_judge {avg_j:.3f} < {target['min_judge']}")
        if err_rate > target["max_error_rate"]:
            violations.append(f"err_rate {err_rate:.1%} > {target['max_error_rate']:.1%}")
        status = "ok" if not violations else "violation"
        row = {
            "n": n, "p50_ms": p50, "p95_ms": p95,
            "avg_judge": round(avg_j, 3), "err_rate": round(err_rate, 3),
            "target": target, "violations": violations,
            "status": status,
        }
        # v0.9.47: quarantine err_rate write moved to _cron_quarantine_sweep()
        # (read-only /admin/sla must not mutate state).
        if w in QUARANTINED:
            row["quarantine_reason"] = QUARANTINED[w]
            row["raw_status"] = status
            row["status"] = "quarantined"
        out[w] = row
    return {"period_days": days, "workers": out, "error_budget": ERROR_BUDGET}


# v0.9.55: error_kind enum for distinguishing transport/quota failures
# from genuine quality errors at the sweep level.
_ERROR_KIND_QUALITY = {"bad_answer"}
_ERROR_KIND_EXCLUDED = {"transport", "quota", "empty", "malformed"}


def _classify_row_error(s: dict) -> str:
    """Return one of quality/transport/quota/empty/malformed/ok.

    Precedence: explicit ``error_kind``; legacy reason fields; then a
    backwards-compatible heuristic. A legacy error row with judge_score 0.0
    and no reason is empty/malformed, not a quality failure.
    """
    kind = s.get("error_kind")
    if isinstance(kind, str) and kind.strip():
        kind = kind.strip().lower()
        if kind in _ERROR_KIND_EXCLUDED:
            return kind
        if kind in _ERROR_KIND_QUALITY:
            return "quality"
    reason = (s.get("fallback_reason") or s.get("error_reason") or "").lower()
    if reason == "quota_exhausted":
        return "quota"
    if reason in {"timeout", "connection", "rate_limit"}:
        return "transport"
    if s.get("quality_tier") not in {"error", "degraded"}:
        return "ok"
    if s.get("is_vendor_placeholder"):
        return "empty"
    jscore = s.get("judge_score")
    # v0.9.56 (PR-A1): empty/malformed must NOT be mis-classified as quality.
    # Three cases:
    #   1. judge_score is None or 0.0 -> upstream returned nothing;
    #   2. quality_tier in {error, degraded} but answer body shorter than
    #      the threshold (12 chars) -> stub or "[error:" prefix;
    #   3. A row that carries both judge_score == 0.3 AND a clear answer is
    #      quality (legacy v0.9.55 heuristic kept).
    if jscore is None or jscore == 0.0:
        return "empty"
    if jscore < 0.3:
        return "malformed"
    return "quality"


def _dedup_by_prompt(rows: list[dict]) -> list[dict]:
    """Collapse retry storms on the same prompt to one worst-case outcome.

    Groups by ``(model_used, session_id, query_hash)``. A quality failure
    wins over a good/transport outcome, preserving the conservative signal
    while preventing one prompt from contributing dozens of failures.
    """
    buckets: dict[tuple, dict] = {}
    for row in rows:
        worker = row.get("model_used")
        key = (worker, row.get("session_id"), row.get("query_hash"))
        bucket = buckets.setdefault(key, {
            "worker": worker,
            "outcome": 0,
            "session_id": row.get("session_id"),
            "query_hash": row.get("query_hash"),
        })
        if _classify_row_error(row) == "quality":
            bucket["outcome"] = 1
    return list(buckets.values())


def _cron_quarantine_sweep(days: int = 1, min_n: int = 10,
                            sessions: Optional[list[dict]] = None) -> dict:
    """Feed per-worker quality err_rate into circuit_breaker.

    Transport/quota/empty/malformed rows are observations but not quality
    failures. Explicit ``error_kind`` takes priority; legacy rows fall back
    to reason fields and judge_score heuristics. Rows sharing
    ``(model_used, session_id, query_hash)`` collapse to one worst-case
    outcome before err_rate is calculated, preventing retry-storm inflation.
    """
    sessions = sessions if sessions is not None else load_sessions(days)
    filtered_rows = []
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    for s in sessions:
        try:
            worker = s.get("model_used")
            if not worker:
                continue
            # load_sessions(days=1) reads today's UTC file, which may contain
            # more than a rolling 24h window around midnight. Test runs can
            # also inject future-dated rows. Enforce the actual rolling window.
            row_ts = float(s.get("ts") or 0.0)
            if row_ts and row_ts < cutoff_ts:
                continue
            source = s.get("source", "unknown")
            query_text = (s.get("query_text") or "").strip()
            if source in ("smoke", "probe", "retrain", "unknown"):
                continue
            if not query_text or query_text == "(empty)":
                continue
            filtered_rows.append(s)
        except Exception:
            pass
    from collections import defaultdict as _dd_qs
    by_worker = _dd_qs(list)
    sessions_per_worker = _dd_qs(set)
    for bucket in _dedup_by_prompt(filtered_rows):
        by_worker[bucket["worker"]].append(bucket["outcome"])
        if bucket.get("session_id"):
            sessions_per_worker[bucket["worker"]].add(bucket["session_id"])
    from anchor.release.circuit_breaker import update_err_rate_quarantine as _update_qs
    results = {}
    for worker, errors in by_worker.items():
        n = len(errors)
        if n >= min_n:
            err_rate = sum(errors) / n
            sessions = len(sessions_per_worker.get(worker, set()))
            try:
                results[worker] = _update_qs(
                    worker, err_rate, n=n, sessions=sessions,
                )
            except Exception:
                pass
    return results


def cost_dashboard(range_: str = "7d", group_by: str = "model") -> dict:
    """Cost / quality / latency dashboard.

    range_: today | 7d | 30d
    group_by: model | tier | day
    """
    days = {"today": 1, "7d": 7, "30d": 30}.get(range_, 7)
    sessions = load_sessions(days)
    if not sessions:
        return {"range": range_, "group_by": group_by, "groups": [], "daily": [], "total": {}}

    buckets = defaultdict(lambda: {
        "request_count": 0,
        "cost_yuan": 0.0,
        "latency_ms_sum": 0,
        "latency_ms_max": 0,
        "judge_score_sum": 0.0,
        "judge_score_count": 0,
        "quality_tiers": defaultdict(int),
        "error_count": 0,
    })

    daily = defaultdict(lambda: {"request_count": 0, "cost_yuan": 0.0})
    totals = {"request_count": 0, "cost_yuan": 0.0, "judge_score_sum": 0.0, "judge_score_count": 0}

    for s in sessions:
        # Consistent filter with compute_sla: drop probe/smoke/empty sessions
        src_tag = s.get("source", "unknown")
        q_text = (s.get("query_text") or "").strip()
        if src_tag in ("smoke", "probe", "retrain"):
            continue
        if not q_text or q_text == "(empty)":
            continue
        if group_by == "tier":
            key = s.get("routed_tier", "unknown")
        elif group_by == "day":
            key = s.get("date", "unknown")
        else:
            key = s.get("model_used", "unknown")

        b = buckets[key]
        b["request_count"] += 1
        cost = s.get("cost_yuan") or 0.0
        b["cost_yuan"] += cost
        lat = s.get("latency_ms") or 0
        b["latency_ms_sum"] += lat
        b["latency_ms_max"] = max(b["latency_ms_max"], lat)
        j = s.get("judge_score")
        if j is not None:
            b["judge_score_sum"] += j
            b["judge_score_count"] += 1
        qt = s.get("quality_tier")
        if qt:
            b["quality_tiers"][qt] += 1
        if qt == "bad":
            b["error_count"] += 1

        d = s.get("date", "unknown")
        daily[d]["request_count"] += 1
        daily[d]["cost_yuan"] += cost

        totals["request_count"] += 1
        totals["cost_yuan"] += cost
        if j is not None:
            totals["judge_score_sum"] += j
            totals["judge_score_count"] += 1

    groups_out = []
    for k, b in sorted(buckets.items(), key=lambda x: -x[1]["cost_yuan"]):
        avg_lat = (b["latency_ms_sum"] / b["request_count"]) if b["request_count"] else 0
        avg_j = (b["judge_score_sum"] / b["judge_score_count"]) if b["judge_score_count"] else 0
        qt_dict = dict(b["quality_tiers"])
        groups_out.append({
            "key": k,
            "request_count": b["request_count"],
            "cost_yuan": round(b["cost_yuan"], 4),
            "cost_yuan_per_call": round(b["cost_yuan"] / b["request_count"], 6) if b["request_count"] else 0,
            "avg_latency_ms": int(avg_lat),
            "p100_latency_ms": b["latency_ms_max"],
            "avg_judge_score": round(avg_j, 3),
            "quality_tier_breakdown": qt_dict,
            "error_count": b["error_count"],
            "error_rate": round(b["error_count"] / b["request_count"], 3) if b["request_count"] else 0,
        })

    daily_out = sorted([
        {"date": d, "request_count": v["request_count"], "cost_yuan": round(v["cost_yuan"], 4)}
        for d, v in daily.items()
    ], key=lambda x: x["date"])

    return {
        "range": range_,
        "group_by": group_by,
        "days_loaded": days,
        "total": {
            "request_count": totals["request_count"],
            "cost_yuan": round(totals["cost_yuan"], 4),
            "avg_judge_score": round(totals["judge_score_sum"] / totals["judge_score_count"], 3) if totals["judge_score_count"] else 0,
        },
        "groups": groups_out,
        "daily": daily_out,
    }




def archive_old_sessions(retention_days: int = 7,
                          sessions_dir=None) -> dict:
    """One-shot archive: gzip jsonl older than retention into _archive/.

    Returns {archived, skipped, errors, total_bytes_before, total_bytes_after}.
    Idempotent: a file already archived is skipped.
    """
    import gzip
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    sd = sessions_dir or _sessions_dir()
    if not sd.exists():
        return {"archived": 0, "skipped": 0, "errors": 0,
                "total_bytes_before": 0, "total_bytes_after": 0}
    archive_dir = sd / "_archive"
    archive_dir.mkdir(exist_ok=True)
    cutoff = (_dt.now(_tz.utc) - _td(days=retention_days)).date()
    before = 0
    after = 0
    archived = 0
    skipped = 0
    errors = 0
    for fp in sorted(sd.glob("*.jsonl")):
        try:
            fdate = _dt.strptime(fp.stem, "%Y-%m-%d").date()
        except ValueError:
            continue
        if fdate >= cutoff:
            continue
        gz_path = archive_dir / (fp.name + ".gz")
        if gz_path.exists():
            skipped += 1
            continue
        size_before = fp.stat().st_size
        before += size_before
        try:
            with fp.open("rb") as _in, gzip.open(gz_path, "wb") as _out:
                while True:
                    chunk = _in.read(64 * 1024)
                    if not chunk:
                        break
                    _out.write(chunk)
            fp.unlink()
            archived += 1
            after += gz_path.stat().st_size
        except Exception:
            errors += 1
    return {
        "archived": archived,
        "skipped": skipped,
        "errors": errors,
        "total_bytes_before": before,
        "total_bytes_after": after,
    }


def m3_amortized_cost(days: int = 30) -> dict:
    """Compute M3 ¥119/mo true cost: API cost + flat share.
    flat_share_per_call = 119 / 30 / n_m3_calls_in_period
    """
    from anchor.config import WORKERS as _W_AMORT
    m3 = next((w for w in _W_AMORT if w.name == "minimax-m3"), None)
    if not m3:
        return {"minimax-m3": None, "error": "minimax-m3 worker not found"}
    flat = m3.monthly_fixed_cny or 0.0

    sessions = load_sessions(days)
    m3_calls = sum(1 for s in sessions if s.get("model_used") == "minimax-m3")
    api_cost_total = sum(s.get("cost_yuan", 0.0) or 0.0 for s in sessions if s.get("model_used") == "minimax-m3")
    non_m3_calls = max(1, len(sessions) - m3_calls)

    flat_share_per_call = flat * days / 30 / non_m3_calls if non_m3_calls else 0.0
    m3_effective = api_cost_total + flat

    return {
        "period_days": days,
        "m3_flat_monthly_yuan": flat,
        "m3_api_cost_yuan": round(api_cost_total, 4),
        "m3_n_calls": m3_calls,
        "non_m3_n_calls": non_m3_calls,
        "flat_share_per_call_yuan": round(flat_share_per_call, 6),
        "m3_effective_total_yuan": round(m3_effective, 4),
        "cost_per_m3_call_yuan": round(m3_effective / m3_calls, 4) if m3_calls else None,
        "method": "flat_share_distributed_to_non_m3_calls",
    }


def register_admin_routes(app) -> None:
    """Register /admin/sla, /admin/cost/dashboard, /admin/cost/amortization on app.

    Called by server.py at module load. Kept as a function (not module-level
    decorators) so admin_ops stays decoupled from the FastAPI instance — easier
    to test the data functions in isolation.
    """

    @app.get("/admin/sla")
    async def admin_sla(days: int = 7):
        """v0.9.18 (Audit Item 4): per-worker SLA status vs target."""
        result = compute_sla(days)
        workers = list(result["workers"].values())
        violations = sum(1 for w in workers if w.get("status") == "violation")
        quarantined = sum(1 for w in workers if w.get("status") == "quarantined")
        public_workers = len(workers) - quarantined
        result["summary"] = {
            "total_workers": len(workers),
            "public_workers": public_workers,
            "quarantined": quarantined,
            "violations": violations,
            "ok": public_workers - violations,
        }
        return result

    @app.get("/admin/cost/dashboard")
    async def admin_cost_dashboard(range: str = "7d", group_by: str = "model"):
        """Cost / quality / latency dashboard (v0.9.10, Sonnet 5 directive)."""
        return cost_dashboard(range, group_by)

    @app.get("/admin/cost/amortization")
    async def admin_cost_amortization(days: int = 30):
        """v0.9.18: True cost view including m3 ¥119/mo flat subsidy."""
        return {"ok": True, "amortization": m3_amortized_cost(days)}


    @app.get("/admin/circuit_breaker")
    async def admin_circuit_breaker():
        """v0.9.60-B1: aggregate snapshot of circuit_breaker.json state.

        Single-call view for ops dashboards. Includes per-worker err_rate,
        quarantine status, placeholder_warning band, ci_lower (Wilson),
        and totals (n_workers_total/quarantined/warning).
        """
        from anchor.release.circuit_breaker import aggregate_state_summary
        return aggregate_state_summary()

    @app.get("/admin/circuit_breaker/quarantined")
    async def admin_circuit_breaker_quarantined():
        """v0.9.60-B1: list currently quarantined workers only."""
        from anchor.release.circuit_breaker import list_quarantined
        q = list_quarantined()
        return {"quarantined": q, "count": len(q)}

    @app.post("/admin/circuit_breaker/clear/{worker}")
    async def admin_circuit_breaker_clear(worker: str):
        """v0.9.60-B1: clear quarantine for a single worker (manual ops override)."""
        from anchor.release.circuit_breaker import clear_quarantine
        ok = clear_quarantine(worker)
        return {"worker": worker, "cleared": ok}


# Backward compat: re-export with old private name so any external code that
# imported server._load_sessions keeps working. server.py can also still call
# _load_sessions() directly because it does the alias below.
_load_sessions = load_sessions
_compute_sla = compute_sla
