"""Pareto curve evaluation for ship_gate (v0.9.37).

Replaces the single-point ship gate with a multi-lambda scan over the
quality-cost tradeoff curve. Sweeps ``LAMBDA_GRID`` (8 points spanning
quality-only to cost-only) and reports the dominance frontier + knee.

API: ``LAMBDA_GRID``, ``score_lambda``, ``ParetoCurveReport``,
``evaluate_pareto_curve``, ``ship_verdict``, ``save_report``,
``format_curve_table``.

Invariants: ``LAMBDA_GRID`` is exactly 8 points; ``pareto.py`` is only
*consumed*; ``TIER_FLOORS`` reused from ``pareto.py``.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable, Awaitable

from anchor.pareto import TIER_FLOORS, _get_worker

LAMBDA_GRID: list[float] = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
COST_TOLERANCE_RATIO: float = 1.10
BASELINE_WORKER: str = "claude-sonnet-5"
MIN_EVAL_CASES: int = 30

# Portable fallback: anchor package is anchor.parent.parent from this file.
_ROOT = Path(os.environ.get("ANCHOR_ROOT", Path(__file__).resolve().parents[2]))


def load_eval_set(path: Optional[Path] = None) -> list[dict]:
    """Load eval cases from data/eval_set/*.json.

    Returns list of dicts with keys: prompt, category, expected_tier,
    expected_worker.
    """
    if path is None:
        path = _ROOT / "data" / "eval_set"
    cases: list[dict] = []
    if path.is_dir():
        for f in sorted(path.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    cases.extend(data)
            except (json.JSONDecodeError, OSError):
                continue
    elif path.is_file():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                cases.extend(data)
        except (json.JSONDecodeError, OSError):
            pass
    return cases


def count_real_eval_cases(
    responses_by_worker: dict[str, list[str]],
    mock: bool = False,
) -> int:
    """Count eval cases that have real (non-placeholder) worker output.

    A response is considered real if it is non-empty and does not start
    with the synthetic prefix "[{worker}] response for:".
    """
    if mock:
        return 0
    count = 0
    for worker, responses in responses_by_worker.items():
        for r in responses:
            if r and not r.startswith(f"[{worker}] response for:"):
                count += 1
    return count

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_lambda(
    lam: float,
    *,
    quality_scores: dict[str, float],
    costs: dict[str, float],
) -> dict[str, float]:
    """Per-worker score = quality - lam * cost. Higher is better."""
    return {w: float(q) - float(lam) * float(costs.get(w, 0.0))
            for w, q in quality_scores.items()}

def aggregate_quality(per_query_scores: list[float]) -> float:
    """Mean of per-query quality scores in [0, 1]."""
    if not per_query_scores:
        return 0.0
    return float(sum(per_query_scores)) / len(per_query_scores)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ParetoCurveReport:
    points: list[dict] = field(default_factory=list)
    dominance_frontier: list[int] = field(default_factory=list)
    recommended_lambda: int = 0
    baseline_comparison: dict = field(default_factory=dict)
    n_queries: int = 0
    n_workers: int = 0
    tier_floors: dict[str, float] = field(default_factory=dict)
    lambdas: list[float] = field(default_factory=list)
    mock: bool = False
    ts: float = 0.0
    def to_dict(self) -> dict: return asdict(self)
    def to_json(self) -> str: return json.dumps(self.to_dict(), indent=2, default=str)

# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------

def _dominates(a: dict, b: dict) -> bool:
    """a dominates b: no worse on both, strictly better on at least one."""
    return (a["cost"] <= b["cost"] and a["quality"] >= b["quality"]
            and (a["cost"] < b["cost"] or a["quality"] > b["quality"]))

def _extract_dominance_frontier(points: list[dict]) -> list[int]:
    """Indices of non-dominated points. O(n^2), n<=8 fine."""
    return [i for i, pi in enumerate(points)
            if not any(_dominates(pj, pi) for j, pj in enumerate(points) if i != j)]

def _select_knee(points: list[dict], frontier: list[int]) -> int:
    """Knee = max of min(1 - cost_norm, quality_norm).

    Mid-curve pick on a quality-rising / cost-decreasing frontier.
    """
    if len(points) <= 1:
        return 0
    c_min, c_max = min(p["cost"] for p in points), max(p["cost"] for p in points)
    q_min, q_max = min(p["quality"] for p in points), max(p["quality"] for p in points)
    sc = (c_max - c_min) or 1.0
    sq = (q_max - q_min) or 1.0
    def _m(p: dict) -> float:
        return min(1.0 - (p["cost"] - c_min) / sc, (p["quality"] - q_min) / sq)
    pool = frontier or list(range(len(points)))
    return max(pool, key=lambda i: _m(points[i]))

# ---------------------------------------------------------------------------
# Mock helpers (used when live judge API is unreachable)
# ---------------------------------------------------------------------------

_WORKER_QUALITY_BIAS = {
    "claude-opus-5": 0.85,
    "claude-fable-5": 0.85,
    BASELINE_WORKER: 0.78,
    "deepseek-v4-flash": 0.80,
    "minimax-m3": 0.80,
}

def _mock_quality_per_worker(queries: list[str], workers: list[str],
                              *, seed: int = 0) -> dict[str, list[float]]:
    """Deterministic mock per-query quality in [0.4, 1.0]."""
    out: dict[str, list[float]] = {w: [] for w in workers}
    for q in queries:
        for w in workers:
            rng = hash((q, w, seed)) & 0xFFFFFFFF
            base = _WORKER_QUALITY_BIAS.get(w, 0.72)
            jitter = ((rng % 1000) / 1000.0 - 0.5) * 0.30
            out[w].append(max(0.40, min(1.0, base + jitter)))
    return out

def _cost_per_worker(workers: list[str]) -> dict[str, float]:
    """Per-query cost (yuan); prompt=200, out=500 from anchor config."""
    out: dict[str, float] = {}
    for name in workers:
        w = _get_worker(name)
        if w is None:
            out[name] = 0.0
            continue
        out[name] = float((200 / 1e6) * w.cost_in + (500 / 1e6) * w.cost_out)
    return out

# ---------------------------------------------------------------------------
# Live evaluation
# ---------------------------------------------------------------------------

async def _gather_live_quality(
    queries: list[str],
    responses_by_worker: dict[str, list[str]],
) -> dict[str, list[float]]:
    """Run judge ensemble (worker vs baseline) -> per-query win-rates."""
    from anchor.judge_calibration import judge_ensemble

    baseline = BASELINE_WORKER if BASELINE_WORKER in responses_by_worker \
        else next(iter(responses_by_worker))
    baseline_rs = responses_by_worker[baseline]
    out: dict[str, list[float]] = {}
    for w, rs in responses_by_worker.items():
        if w == baseline:
            out[w] = [0.5] * len(queries)
            continue
        scores: list[float] = []
        for q, a, b in zip(queries, rs, baseline_rs):
            ens, *_ = await judge_ensemble(q, a, b, mock=False)
            scores.append(1.0 if ens == "A" else (0.5 if ens == "TIE" else 0.0))
        out[w] = scores
    return out

def _mock_quality(queries: list[str],
                  responses_by_worker: dict[str, list[str]]) -> dict[str, list[float]]:
    """Mock quality with baseline self-compare pinned to 0.5."""
    base = _mock_quality_per_worker(queries, list(responses_by_worker), seed=0)
    if BASELINE_WORKER in base:
        base[BASELINE_WORKER] = [0.5] * len(queries)
    return base

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def evaluate_pareto_curve(
    *,
    queries: list[str],
    responses_by_worker: dict[str, list[str]],
    tier_floors: dict[str, float] = TIER_FLOORS,
    judge_ensemble_fn: Optional[Callable[..., Awaitable]] = None,
    lambdas: list[float] = LAMBDA_GRID,
    mock: Optional[bool] = None,
    worker_fn: Optional[Callable[..., Awaitable]] = None,
) -> ParetoCurveReport:
    """Evaluate the quality-cost Pareto curve.

    Args: queries list, responses_by_worker dict, tier_floors (default
    TIER_FLOORS), lambdas (default LAMBDA_GRID), mock (None = try live
    then fall back), worker_fn (optional async callable(worker_name, query)
    -> str for collecting real worker output).
    """
    n_q = len(queries)
    workers = list(responses_by_worker.keys())

    # Collect real responses via worker_fn when provided
    if worker_fn is not None:
        for w in workers:
            real_responses: list[str] = []
            for q in queries:
                try:
                    resp = await worker_fn(w, q)
                    real_responses.append(resp if resp else "")
                except Exception:
                    real_responses.append("")
            responses_by_worker[w] = real_responses

    if mock is None:
        mock = False
        try:
            ens_fn = judge_ensemble_fn
            if ens_fn is None:
                from anchor.judge_calibration import judge_ensemble as _je
                ens_fn = _je
            if n_q == 0 or len(workers) < 2:
                mock = True
            else:
                await ens_fn(queries[0], responses_by_worker[workers[0]][0],
                             responses_by_worker[workers[1]][0])
        except Exception:
            mock = True
    if mock and not judge_ensemble_fn and workers:
        print("WARNING: ensemble not available, single-judge bias applies "
              "(mock mode for pareto_curve evaluation)")

    # Eval guard: require >= MIN_EVAL_CASES real responses for promote
    n_real = count_real_eval_cases(responses_by_worker, mock=mock)
    if not mock and n_real < MIN_EVAL_CASES:
        print(f"SHIP_GATE: n_eval_cases={n_real} mock={mock} "
              f"(need >= {MIN_EVAL_CASES} real eval cases to promote)")
        return ParetoCurveReport(
            points=[], dominance_frontier=[], recommended_lambda=0,
            baseline_comparison={}, n_queries=n_q,
            n_workers=len(workers), tier_floors=dict(tier_floors),
            lambdas=list(lambdas), mock=bool(mock), ts=time.time(),
        )
    print(f"SHIP_GATE: n_eval_cases={n_real} mock={mock}")

    if not workers or not queries:
        return ParetoCurveReport(
            points=[], dominance_frontier=[], recommended_lambda=0,
            baseline_comparison={}, n_queries=len(queries),
            n_workers=len(workers), tier_floors=dict(tier_floors),
            lambdas=list(lambdas), mock=bool(mock), ts=time.time(),
        )
    per_query = (_mock_quality(queries, responses_by_worker) if mock
                 else await _gather_live_quality(queries, responses_by_worker))

    costs = _cost_per_worker(workers)
    qualities = {w: aggregate_quality(per_query[w]) for w in workers}

    bq = qualities.get(BASELINE_WORKER)
    bc = costs.get(BASELINE_WORKER, 0.0) if bq is not None else 0.0
    points: list[dict] = []
    baseline_comparison: dict[str, dict] = {}
    for i, lam in enumerate(lambdas):
        scores = score_lambda(lam, quality_scores=qualities, costs=costs)
        selected = max(scores, key=scores.get)
        q, c = qualities[selected], costs[selected]
        points.append({
            "lambda": float(lam), "index": i, "selected_worker": selected,
            "score": float(scores[selected]), "cost": float(c), "quality": float(q),
            "tier_satisfied": bool(q >= tier_floors["basic"]),
            "all_tiers_satisfied": bool(q >= max(tier_floors.values())),
        })
        if bq is not None:
            baseline_comparison[str(i)] = {
                "lambda": float(lam), "baseline_worker": BASELINE_WORKER,
                "baseline_quality": float(bq), "baseline_cost": float(bc),
                "baseline_score": float(bq - lam * bc),
                "head_quality": float(q), "head_cost": float(c),
                "head_beats_baseline_quality": bool(q >= bq),
                "head_cost_within_tolerance": bool(c <= bc * COST_TOLERANCE_RATIO),
            }

    frontier = _extract_dominance_frontier(points)
    knee = _select_knee(points, frontier)

    return ParetoCurveReport(
        points=points,
        dominance_frontier=frontier,
        recommended_lambda=knee,
        baseline_comparison=baseline_comparison,
        n_queries=n_q,
        n_workers=len(workers),
        tier_floors=dict(tier_floors),
        lambdas=list(lambdas),
        mock=bool(mock),
        ts=time.time(),
    )

# ---------------------------------------------------------------------------
# SHIP verdict
# ---------------------------------------------------------------------------

def ship_verdict(report: ParetoCurveReport) -> dict:
    """SHIP iff knee head quality >= baseline AND head cost within tolerance."""
    if not report.points or not report.baseline_comparison:
        return {"ship": False, "reason": "empty report"}
    cmp = report.baseline_comparison.get(str(report.recommended_lambda))
    if cmp is None:
        return {"ship": False, "reason": "no baseline_comparison at knee"}
    ok_q = cmp["head_beats_baseline_quality"]
    ok_c = cmp["head_cost_within_tolerance"]
    kp = report.points[report.recommended_lambda]
    return {
        "ship": bool(ok_q and ok_c),
        "knee_lambda": kp["lambda"], "knee_worker": kp["selected_worker"],
        "head_quality": cmp["head_quality"], "baseline_quality": cmp["baseline_quality"],
        "head_cost": cmp["head_cost"], "baseline_cost": cmp["baseline_cost"],
        "cost_tolerance_ratio": COST_TOLERANCE_RATIO,
        "ok_quality": bool(ok_q), "ok_cost": bool(ok_c), "mock": bool(report.mock),
    }

# ---------------------------------------------------------------------------
# File persistence + table
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path(os.environ.get(
    "ANCHOR_CALIBRATION_DIR", str(_ROOT / "data/calibration")
))

def save_report(report: ParetoCurveReport, *, out_path: Optional[Path] = None) -> Path:
    """Persist to ``data/calibration/pareto_curve_<date>.json``.

    Also writes ``recommended_lambda.json`` (single-key file) for runtime
    routing to read the current knee-point lambda. v0.9.45.
    """
    if out_path is None:
        d = time.strftime("%Y-%m-%d", time.gmtime(report.ts))
        out_path = DEFAULT_OUTPUT_DIR / f"pareto_curve_{d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_json())
    # Write runtime-readable knee-point config
    kp = report.points[report.recommended_lambda]
    runtime_cfg = {
        "lambda": kp["lambda"],
        "selected_worker": kp.get("selected_worker"),
        "cost": kp.get("cost"),
        "quality": kp.get("quality"),
        "index": report.recommended_lambda,
        "ts": report.ts,
        "mock": report.mock,
    }
    rec_path = DEFAULT_OUTPUT_DIR / "recommended_lambda.json"
    rec_path.write_text(json.dumps(runtime_cfg, indent=2))
    return out_path

def format_curve_table(report: ParetoCurveReport) -> str:
    """Human-readable table of the 8-row curve."""
    hdr = "lam   selected_worker       cost(yuan)   quality   tier_sat   frontier  beats_sonnet5"
    lines = [hdr, "-" * len(hdr)]
    fs = set(report.dominance_frontier)
    for p in report.points:
        cmp = report.baseline_comparison.get(str(p["index"]), {})
        beats = "?"
        if cmp:
            beats = "YES" if (cmp["head_beats_baseline_quality"]
                              and cmp["head_cost_within_tolerance"]) else "no"
        marker = "*" if p["index"] in fs else " "
        lines.append(
            f"{p['lambda']:<5.2f} {p['selected_worker']:<21} "
            f"{p['cost']:<11.5f} {p['quality']:<9.4f} "
            f"{str(p['tier_satisfied']):<10} {marker:<9} {beats}"
        )
    kp = report.points[report.recommended_lambda]
    lines += [
        "",
        f"Frontier indices: {report.dominance_frontier}",
        f"Knee point: index={report.recommended_lambda} "
        f"lambda={kp['lambda']} worker={kp['selected_worker']}",
        f"Mock mode: {report.mock}",
    ]
    return "\n".join(lines)

__all__ = [
    "LAMBDA_GRID",
    "COST_TOLERANCE_RATIO",
    "BASELINE_WORKER",
    "MIN_EVAL_CASES",
    "ParetoCurveReport",
    "score_lambda",
    "aggregate_quality",
    "evaluate_pareto_curve",
    "ship_verdict",
    "save_report",
    "format_curve_table",
    "load_eval_set",
    "count_real_eval_cases",
]
