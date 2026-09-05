"""v0.9.13 (C directive) + v0.9.19: auto-calibration from session_log.

Reads all session jsonl files in data/anchor_sessions/, aggregates
worker × category × difficulty_bucket from real judge_score,
writes quality_table_v7.json (live_routed, 7d rolling, includes sacred data).

Endpoint: /admin/calibration/refresh

v0.9.53 (P0-3 fix): NO LONGER filters `quality_tier == "degraded"` records
by default. The previous filter caused survivorship bias: real failures
(empty responses, placeholders, vendor errors) are exactly the records
that should INFLUENCE the mean and TCO retry/escalation estimates. Opt-
in to the old behavior with `ANCHOR_CALIBRATION_FILTER_DEGRADED=1`.
"""
import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import sys
from anchor import _ROOT as _A_ROOT

sys.path.insert(0, str(Path(__file__).parent))
from anchor.fusion_modes import _detect_category, _difficulty_bucket

SESS_DIR = Path(str(_A_ROOT / "data/anchor_sessions"))
CAL_DIR = Path(str(_A_ROOT / "data/calibration"))
WINDOW_DAYS = 7
MIN_RECORDS = 10  # require ≥10 records per (worker, cat, bucket) to be reliable

# v0.9.53: opt-in to legacy degraded-filter behavior. Default OFF.
_FILTER_DEGRADED: bool = os.environ.get("ANCHOR_CALIBRATION_FILTER_DEGRADED", "0") == "1"

# v1.0.x STAGE1 (survivorship bias): when set, exclude records that were
# served by a FALLBACK worker (fallback_from present) from the per-worker
# mean. Those judge_scores reflect the fallback path, not the primary
# worker's real quality — mixing them inflates primary estimates.
# Default OFF (backwards compatible); the share is always reported.
_EXCLUDE_FALLBACK: bool = os.environ.get("ANCHOR_CALIBRATION_EXCLUDE_FALLBACK", "0") == "1"


def _collect_sessions(days: int = WINDOW_DAYS):
    """Load last N days of session logs."""
    sessions = []
    now = datetime.now(timezone.utc)
    for f in sorted(SESS_DIR.glob("*.jsonl")):
        try:
            d = datetime.fromisoformat(f.stem).replace(tzinfo=timezone.utc)
            age_days = (now - d).days
            if age_days > days:
                continue
        except Exception:
            sessions.extend(_read_jsonl(f))
            continue
        sessions.extend(_read_jsonl(f))
    return sessions


def _read_jsonl(f):
    out = []
    with open(f) as fh:
        for line in fh:
            try:
                s = json.loads(line)
                if s.get("judge_score") is not None and s.get("model_used"):
                    out.append(s)
            except Exception:
                pass
    return out


def _aggregate(sessions, min_n=MIN_RECORDS):
    """Build table_per_cat and table_coarse from sessions.

    Returns: (per_cat, coarse, n_records, n_by_worker)

    v0.9.53 (P0-3): by default, INCLUDES degraded-tier records (real
    failures) so the mean reflects actual availability. This is the fix
    for survivorship bias that inflated worker quality scores. Real
    failures should INCREASE estimated retry/escalation cost, not be
    dropped. Set ANCHOR_CALIBRATION_FILTER_DEGRADED=1 to restore the
    legacy behavior (only if downstream tooling depends on the old mean).

    v0.9.53 (P1 fix): also filters records whose `model_used` is NOT in
    the canonical WORKERS list. Defends calibration against external
    tools (load tests, monitoring) that send legacy short aliases like
    "m3" via `worker_override` (accepted when ANCHOR_ALLOW_WORKER_OVERRIDE
    is set). Those records trigger the `[stub:<name>]` fallback path
    and would otherwise pollute the quality table with non-canonical
    worker names.
    """
    # Canonical worker registry (rebuilt per call to pick up enabled
    # changes; cheap — 12 workers max).
    try:
        from anchor.config import WORKERS as _CANONICAL
        _canonical_names = {w.name for w in _CANONICAL}
    except Exception:
        _canonical_names = None  # defensive: don't filter if config fails to load
    # data: worker -> cat -> bucket -> list of (judge_score, quality_tier)
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    skipped_unknown = 0
    for s in sessions:
        w = s["model_used"]
        # P1 fix: drop records whose worker is not a canonical name.
        # This is the post-hoc counterpart to the server-side whitelist
        # check in worker_override handling — protects calibration even
        # if a misconfigured client bypasses the API validation.
        if _canonical_names is not None and w not in _canonical_names:
            skipped_unknown += 1
            continue
        cat = _detect_category(s.get("query_text", ""))
        bkt = _difficulty_bucket(s.get("d_value", 0))
        sc = s["judge_score"]
        qt = s.get("quality_tier", "")
        # v0.9.53: only filter degraded when explicitly opted in. Default
        # includes real failures (judge_score = 0 for placeholders/errors)
        # so Pareto TCO retry/escalation estimates reflect actual reliability.
        if _FILTER_DEGRADED and qt == "degraded":
            continue
        # v1.0.x STAGE1: mark records served by a fallback worker. Their
        # judge_score measures the fallback path, not the primary worker.
        is_fallback = bool(s.get("fallback_from"))
        agg[w][cat][bkt].append((sc, qt, is_fallback))
    if skipped_unknown:
        import logging as _lg
        _lg.info(
            "CALIBRATION_SKIP_UNKNOWN_WORKER skipped=%d (legacy short aliases like "
            "'m3' from worker_override when ANCHOR_ALLOW_WORKER_OVERRIDE=1)",
            skipped_unknown,
        )

    per_cat = {}
    coarse = {}
    n_records = 0
    n_by_worker = defaultdict(int)
    for w in agg:
        per_cat[w] = {}
        coarse[w] = {}
        for cat in agg[w]:
            per_cat[w][cat] = {}
            for bkt in ["easy", "medium", "hard"]:
                arr = agg[w][cat].get(bkt, [])
                if not arr:
                    continue
                # require min_n
                if len(arr) < min_n:
                    continue
                # v1.0.x STAGE1: optionally exclude fallback-served records
                # from the mean (survivorship bias). Default includes them.
                if _EXCLUDE_FALLBACK:
                    arr = [t for t in arr if not t[2]]
                    if not arr:
                        continue
                avg_score = sum(s for s, _, _ in arr) / len(arr)
                per_cat[w][cat][bkt] = round(avg_score, 3)
                n_records += len(arr)
                n_by_worker[w] += len(arr)
        # coarse = avg across cats
        all_buckets = {}
        for cat in agg[w]:
            for bkt in ["easy", "medium", "hard"]:
                arr = agg[w][cat].get(bkt, [])
                if not arr:
                    continue
                if len(arr) < min_n:
                    continue
                if _EXCLUDE_FALLBACK:
                    arr = [t for t in arr if not t[2]]
                    if not arr:
                        continue
                all_buckets.setdefault(bkt, []).extend(s for s, _, _ in arr)
        for bkt, scores in all_buckets.items():
            coarse[w][bkt] = round(sum(scores) / len(scores), 3)
    return per_cat, coarse, n_records, dict(n_by_worker)


def compute_fallback_share(sessions, min_n=MIN_RECORDS):
    """v1.0.x STAGE1: per-worker share of records served via fallback.

    Reports how often each worker's judge_score comes from a fallback
    path (fallback_from present). High share = the quality table for that
    worker is dominated by rescues, not direct routing.
    """
    try:
        from anchor.config import WORKERS as _CANONICAL
        _canonical_names = {w.name for w in _CANONICAL}
    except Exception:
        _canonical_names = None
    fallback = defaultdict(int)
    total = defaultdict(int)
    for s in sessions:
        w = s.get("model_used")
        if _canonical_names is not None and w not in _canonical_names:
            continue
        total[w] += 1
        if s.get("fallback_from"):
            fallback[w] += 1
    return {
        w: {
            "total": total[w],
            "fallback": fallback[w],
            "share": round(fallback[w] / total[w], 4) if total[w] else 0.0,
        }
        for w in total
    }


def refresh(min_n: int = MIN_RECORDS, days: int = WINDOW_DAYS, write: bool = True):
    """Refresh quality_table from session log. Returns stats dict."""
    sessions = _collect_sessions(days)
    per_cat, coarse, n_records, n_by_worker = _aggregate(sessions, min_n=min_n)
    fallback_share = compute_fallback_share(sessions, min_n=min_n)
    if write:
        out = {
            # audit 2026-08-16 (A13/F1): was v7_live_routed_* while the B6
            # data-driven ladder/head read quality_table_v8.json — a manual
            # refresh had zero routing effect.
            "version": f"v8_live_routed_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            "n_records": n_records,
            "window_days": days,
            "min_records_per_cell": min_n,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "table_per_cat": per_cat,
            "table_coarse": coarse,
            "n_by_worker": n_by_worker,
            "fallback_share": fallback_share,
        }
        out_file = CAL_DIR / "quality_table_v8.json"
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    return {
        "n_sessions": len(sessions),
        "n_records": n_records,
        "n_by_worker": n_by_worker,
        "per_cat_cells": sum(len(c) for w in per_cat.values() for c in w.values()),
        "coarse_cells": sum(len(c) for c in coarse.values()),
    }


if __name__ == "__main__":
    stats = refresh()
    print(json.dumps(stats, indent=2))
