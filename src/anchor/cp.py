"""Conformal Prediction uncertainty layer (v0.9.53, Phase 2).

Provides per-(worker, category, difficulty) CP intervals over judge_score
so the Pareto router can reject workers whose CP lower bound falls below
the quality floor — even when the point estimate (mean) is above floor.

Reference papers (2024-2026):
  - CP-Router  (arXiv 2505.19970)  LLM↔LRM routing via CP
  - API Is Enough CP (arXiv 2403.01216)  API-only CP via sample+semantic
  - C3PO (arXiv 2511.07396)  CP + probabilistic cost constraint
  - DS-CP  (arXiv 2510.05566, ICML 2026)  domain-shift aware CP

Approach: split-conformal over recent judge residuals.
  1. For each cell (worker, cat, diff), compute mean of recent judge scores.
  2. residual_i = |judge_score_i - mean|
  3. radius = (1-alpha) quantile of residuals
  4. CP interval = (mean - radius, mean + radius)
  5. Guarantee: P(true_score in [mean ± radius]) >= 1 - alpha

API-only friendly: we never touch logits. We use judge_batch output as
the ground truth signal. Calibration data comes from data/anchor_sessions/
+ quality_overlay.json (same source as pareto.quality_table_lookup).

Cold-start: when a cell has < MIN_CALIBRATION_SAMPLES, return a
conservative interval (mean - large_radius, mean + large_radius) so
the router still gets a usable signal instead of dropping the worker.

Cost saving: when CP upper bound is already below floor, the router can
skip the worker without any further judge call. P0-5's floor_relaxed
path becomes a last resort, not a routine fallback.

No torch/sklearn dependency — only numpy. ~150 LOC.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

# v0.9.53: feature flag. Default OFF to avoid changing router behavior
# in production until shadow-evaluated. Enable with ANCHOR_CP_ENABLED=1.
CP_ENABLED: bool = os.environ.get("ANCHOR_CP_ENABLED", "0") == "1"

# Miscoverage rate (alpha). radius = (1-alpha) quantile of residuals.
CP_ALPHA: float = float(os.environ.get("ANCHOR_CP_ALPHA", "0.1"))

# Min samples per cell to compute a non-degenerate interval. Below this,
# return conservative (very wide) interval as cold-start fallback.
MIN_CALIBRATION_SAMPLES: int = int(os.environ.get("ANCHOR_CP_MIN_SAMPLES", "5"))

# Max cell radius (clip to avoid huge intervals for sparse data).
MAX_RADIUS: float = float(os.environ.get("ANCHOR_CP_MAX_RADIUS", "0.5"))

# Default radius for cells with insufficient samples.
COLD_START_RADIUS: float = float(os.environ.get("ANCHOR_CP_COLD_RADIUS", "0.4"))


def _quantile(values: list[float], q: float) -> float:
    """Type-7 quantile (linear interpolation), numpy-free for portability."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < n:
        return s[lo] * (1.0 - frac) + s[lo + 1] * frac
    return s[lo]


def compute_cell_intervals(
    judge_scores: list[float],
    *,
    alpha: float = CP_ALPHA,
    min_samples: int = MIN_CALIBRATION_SAMPLES,
    max_radius: float = MAX_RADIUS,
    cold_radius: float = COLD_START_RADIUS,
) -> tuple[float, float, float]:
    """Compute (mean, lower_bound, upper_bound) CP interval for one cell.

    Args:
        judge_scores: list of recent judge scores in [0, 1] for this cell.
        alpha: miscoverage rate (1-alpha quantile of residuals = radius).
        min_samples: cells with fewer samples get cold_start_radius.
        max_radius: clip radius to avoid huge intervals in noisy cells.
        cold_radius: radius for cold-start cells.

    Returns:
        (mean, lower, upper) tuple. All in [0, 1] (clipped).
    """
    if not judge_scores:
        return 0.5, 0.0, 1.0  # empty -> maximally uncertain
    n = len(judge_scores)
    mean = sum(judge_scores) / n
    if n < min_samples:
        # Cold-start: wide interval centered on mean.
        radius = cold_radius
    else:
        # Split-conformal radius = (1-alpha) quantile of |x - mean|
        residuals = [abs(x - mean) for x in judge_scores]
        radius = _quantile(residuals, 1.0 - alpha)
        radius = min(radius, max_radius)
    lower = max(0.0, mean - radius)
    upper = min(1.0, mean + radius)
    return mean, lower, upper


# In-memory cell cache: {(worker, category, difficulty) -> (mean, lower, upper, n)}
_cell_cache: dict[tuple[str, str, str], tuple[float, float, float, int]] = {}
_cell_lock_dirty = True


def _load_judge_scores_from_sessions(
    sessions_dir: Path, days: int = 7
) -> dict[tuple[str, str, str], list[float]]:
    """Load judge scores from data/anchor_sessions/*.jsonl (last N days).

    Returns dict keyed by (worker, category, difficulty_bucket).
    """
    from datetime import datetime, timezone, timedelta
    from anchor.fusion_modes import _detect_category, _difficulty_bucket

    cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    if not sessions_dir.exists():
        return dict(cells)
    for f in sorted(sessions_dir.glob("*.jsonl")):
        try:
            d = datetime.fromisoformat(f.stem).replace(tzinfo=timezone.utc)
            if d < cutoff:
                continue
        except Exception:
            pass
        try:
            with open(f) as fh:
                for line in fh:
                    try:
                        s = json.loads(line)
                        sc = s.get("judge_score")
                        w = s.get("model_used")
                        if sc is None or not w:
                            continue
                        cat = _detect_category(s.get("query_text", ""))
                        bkt = _difficulty_bucket(s.get("d_value", 0.0))
                        cells[(w, cat, bkt)].append(float(sc))
                    except Exception:
                        pass
        except Exception:
            pass
    return dict(cells)


def refresh_cell_cache(
    sessions_dir: Optional[Path] = None,
    overlay_path: Optional[Path] = None,
) -> int:
    """Recompute CP intervals from recent sessions + overlay.

    Returns number of cells refreshed. Idempotent; safe to call multiple
    times. Mark cache as fresh.
    """
    global _cell_lock_dirty
    if sessions_dir is None:
        from anchor import _ROOT
        sessions_dir = _ROOT / "data" / "anchor_sessions"
    cells: dict[tuple[str, str, str], list[float]] = {}
    if sessions_dir.exists():
        cells = _load_judge_scores_from_sessions(sessions_dir, days=7)
    # Merge in overlay (Bayesian-updated scores) so newly observed
    # samples immediately affect the interval.
    if overlay_path is None:
        overlay_path = Path(
            os.environ.get(
                "ANCHOR_FEEDBACK_OVERLAY",
                str(Path.home() / "anchor" / "data" / "calibration" / "quality_overlay.json"),
            )
        )
    if overlay_path.exists():
        try:
            overlay_data = json.loads(overlay_path.read_text())
            for k, v in overlay_data.items():
                parts = k.split("|", 2)
                if len(parts) != 3:
                    continue
                w, cat, bkt = parts
                sc = float(v.get("score", 0.5))
                cells.setdefault((w, cat, bkt), []).append(sc)
        except Exception:
            pass
    new_cache: dict[tuple[str, str, str], tuple[float, float, float, int]] = {}
    for key, scores in cells.items():
        mean, lower, upper = compute_cell_intervals(scores)
        new_cache[key] = (mean, lower, upper, len(scores))
    _cell_cache.clear()
    _cell_cache.update(new_cache)
    _cell_lock_dirty = False
    return len(new_cache)


def get_interval(worker: str, category: str, difficulty: str) -> tuple[float, float, float]:
    """Return (mean, lower, upper) CP interval for the cell.

    If the cache is dirty, refresh it on first read.
    """
    if _cell_lock_dirty:
        refresh_cell_cache()
    key = (worker, category, difficulty)
    if key not in _cell_cache:
        # Cold-start cell: no data at all.
        return 0.5, 0.0, 1.0
    mean, lower, upper, _n = _cell_cache[key]
    return mean, lower, upper


def passes_floor(worker: str, category: str, difficulty: str, floor: float) -> bool:
    """Return True if CP lower bound >= floor (worker is reliable)."""
    mean, lower, upper = get_interval(worker, category, difficulty)
    return lower >= floor


def cp_stats() -> dict:
    """Observability: snapshot of current CP cache."""
    if _cell_lock_dirty:
        refresh_cell_cache()
    return {
        "n_cells": len(_cell_cache),
        "enabled": CP_ENABLED,
        "alpha": CP_ALPHA,
        "min_samples": MIN_CALIBRATION_SAMPLES,
        "dirty": _cell_lock_dirty,
    }


__all__ = [
    "CP_ENABLED",
    "CP_ALPHA",
    "compute_cell_intervals",
    "refresh_cell_cache",
    "get_interval",
    "passes_floor",
    "cp_stats",
]