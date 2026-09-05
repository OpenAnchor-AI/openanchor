"""Day 10: KL-divergence drift detector (alert if rolling 1k-query KL > 0.15).

Compares current prior_table against a snapshot baseline.
"""
import json
import time
from pathlib import Path
from typing import Optional

BASELINE_PATH = Path.home() / "anchor" / "data" / "head_baseline.json"
DRIFT_THRESHOLD = 0.15
ROLLING_WINDOW = 1000


def snapshot(prior_table: dict, W, out_path: Optional[Path] = None) -> Path:
    p = out_path or BASELINE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ts": time.time(),
        "W": W.tolist() if hasattr(W, "tolist") else list(W),
        "prior": {f"{k[0]}|{k[1]}": v for k, v in prior_table.items()},
    }
    p.write_text(json.dumps(data))
    return p


def _load_baseline(in_path: Optional[Path] = None) -> Optional[dict]:
    p = in_path or BASELINE_PATH
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _kl(p: float, q: float) -> float:
    """Symmetric scalar KL: p*log(p/q) + (1-p)*log((1-p)/(1-q))."""
    import math
    p = max(min(p, 0.999), 0.001)
    q = max(min(q, 0.999), 0.001)
    return float(p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q)))


def compute_drift(prior_table: dict, baseline_path: Optional[Path] = None) -> dict:
    """Return mean KL between current prior and baseline. None if no baseline."""
    base = _load_baseline(baseline_path)
    if base is None:
        return {"mean_kl": 0.0, "max_kl": 0.0, "n": 0, "has_baseline": False}
    base_prior = {tuple(k.split("|", 1)): float(v) for k, v in base["prior"].items()}
    cells = []
    for k, v in prior_table.items():
        bp = base_prior.get(k, 0.5)
        cells.append(_kl(float(v), bp))
    return {
        "mean_kl": sum(cells) / max(len(cells), 1),
        "max_kl": max(cells) if cells else 0.0,
        "n": len(cells),
        "has_baseline": True,
    }


class DriftAlert(Exception):
    pass


def check_drift(prior_table: dict, threshold: float = DRIFT_THRESHOLD,
                baseline_path: Optional[Path] = None) -> dict:
    """Compute drift + raise DriftAlert if mean_kl > threshold."""
    result = compute_drift(prior_table, baseline_path)
    if result["has_baseline"] and result["mean_kl"] > threshold:
        raise DriftAlert(
            f"drift detected: mean_kl={result['mean_kl']:.3f} > {threshold}"
        )
    return result
