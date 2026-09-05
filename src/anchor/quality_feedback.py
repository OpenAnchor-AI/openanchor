"""First-principles quality feedback loop (v0.9.37).

In-memory Bayesian overlay on quality_table_v7.json. Production judge
verdicts from judge_batch update quality scores (alpha-weighted running
mean) without rewriting the file. quality_table_lookup reads the overlay
first, then falls back to file.

  score_updated = (1 - alpha) * score_file + alpha * score_observed

Default alpha=0.1 keeps updates conservative (10 observations to dominate
prior). When ANCHOR_FEEDBACK_PERSIST=1, overlay is also written back to a
sidecar file (data/calibration/quality_overlay.json) so updates survive
process restarts.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

# pareto is imported lazily to avoid circular import

# Default alpha for Bayesian update; 0.1 -> ~10 observations to dominate prior.
DEFAULT_ALPHA: float = float(os.environ.get("ANCHOR_FEEDBACK_ALPHA", "0.1"))

# Overlay sidecar file (if persist enabled).
OVERLAY_PATH = Path(
    os.environ.get(
        "ANCHOR_FEEDBACK_OVERLAY",
        str(Path.home() / "anchor" / "data" / "calibration" / "quality_overlay.json"),
    )
)

# In-memory overlay: {(worker, category, difficulty) -> {"score": float, "n": int, "ts": float}}
_overlay: dict[tuple[str, str, str], dict] = {}
_overlay_lock = threading.Lock()


def _key(worker: str, category: str, difficulty: str) -> tuple[str, str, str]:
    return (worker, category, difficulty)


def update_quality_cell(
    worker: str,
    category: str,
    difficulty: str,
    observed_score: float,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Bayesian update of (worker, category, difficulty) cell.

    observed_score in [0, 1]. Returns the new posterior score.
    Locks overlay for thread safety. Persists to overlay sidecar if
    ANCHOR_FEEDBACK_PERSIST=1.
    """
    observed_score = max(0.0, min(1.0, float(observed_score)))
    key = _key(worker, category, difficulty)
    with _overlay_lock:
        prior = _overlay.get(key)
        if prior is None:
            # First observation: pure prior = observed (alpha=1 effectively)
            new_score = observed_score
            n = 1
        else:
            old_score = prior["score"]
            n = prior["n"] + 1
            new_score = (1.0 - alpha) * old_score + alpha * observed_score
        _overlay[key] = {"score": new_score, "n": n, "ts": time.time()}
    # v0.9.46: persist by default (1 disk write per sampled query, cheap)
    if os.environ.get("ANCHOR_FEEDBACK_PERSIST", "1") != "0":
        _persist_overlay()
    return new_score


def lookup_with_overlay(
    worker: str,
    category: str,
    difficulty: str,
) -> Optional[float]:
    """Read overlay first, fall back to file-based quality_table_lookup."""
    key = _key(worker, category, difficulty)
    with _overlay_lock:
        if key in _overlay:
            return _overlay[key]["score"]
    # Fall through to file-based lookup (avoids circular import)
    from anchor.pareto import quality_table_lookup as _file_lookup
    return _file_lookup(worker, category, difficulty)


def overlay_stats() -> dict:
    """Return summary of overlay state for dashboards / admin."""
    with _overlay_lock:
        cells = {
            f"{w}/{c}/{d}": {"score": round(v["score"], 4), "n": v["n"]}
            for (w, c, d), v in _overlay.items()
        }
    return {"n_cells": len(cells), "cells": cells}


def _persist_overlay() -> None:
    """Snapshot overlay to disk. Called on every update if persist enabled."""
    with _overlay_lock:
        snapshot = {
            f"{w}|{c}|{d}": v for (w, c, d), v in _overlay.items()
        }
    OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERLAY_PATH.write_text(json.dumps(snapshot, indent=2))


def load_persisted_overlay() -> int:
    """Load overlay from disk at startup. Returns n cells loaded."""
    if not OVERLAY_PATH.exists():
        return 0
    try:
        data = json.loads(OVERLAY_PATH.read_text())
    except Exception:
        return 0
    with _overlay_lock:
        for k, v in data.items():
            parts = k.split("|", 2)
            if len(parts) != 3:
                continue
            _overlay[(parts[0], parts[1], parts[2])] = v
    return len(_overlay)


# Auto-load persisted overlay on import (server startup picks this up).
load_persisted_overlay()


def submit_judgment(
    worker: str,
    category: str,
    difficulty: str,
    verdict: str,
) -> Optional[float]:
    """Convert judge verdict ('A' / 'B' / 'TIE') to a [0,1] quality signal.

    'A' or 'B' (whichever side the worker is on) -> 1.0
    TIE -> 0.5
    Unknown side (judge compared against another worker) -> 0.7 (default win)
    Returns new posterior score.
    """
    mapping = {"A": 1.0, "B": 1.0, "TIE": 0.5}
    score = mapping.get(verdict, 0.7)
    return update_quality_cell(worker, category, difficulty, score)


__all__ = [
    "update_quality_cell",
    "lookup_with_overlay",
    "overlay_stats",
    "submit_judgment",
    "load_persisted_overlay",
    "DEFAULT_ALPHA",
    "OVERLAY_PATH",
]
