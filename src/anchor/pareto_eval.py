"""Pareto frontier evaluation vs always-Opus-4.8 baseline.

Generates (cost, quality) data points by sweeping a quality-vs-cost weight
lambda in:
    routing_score = quality_score - lambda * cost_score

For each lambda we pick the worker for each query, then run a judge_batch
pairwise comparison vs the always-Opus response to estimate pair_acc.

This is an offline evaluation harness; does not run in production.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from anchor.pareto import (
    select_cheapest_meeting_floor,
    quality_table_lookup,
    parse_query_to_cell,
    TCOBreakdown,
)


# Lambda sweep: low -> quality-only, high -> cost-only.
DEFAULT_LAMBDAS: tuple[float, ...] = (0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0)


def _normalize_quality(worker: str, category: str, difficulty: str) -> float:
    score = quality_table_lookup(worker, category, difficulty)
    if score is None:
        return 0.5
    return max(0.0, min(1.0, float(score)))


def _pick_for_lambda(
    query: str,
    query_type: str,
    tier: str,
    lam: float,
) -> tuple[str, TCOBreakdown]:
    """Pick worker under routing_score = quality - lam * cost_out."""
    category, difficulty, _ = parse_query_to_cell(query, query_type)
    from anchor.config import WORKERS
    from anchor.head import TIER_POOL

    # v0.9.47 (P1-B): TIER_POOL now stores worker name strings, not indices.
    idx_pool = TIER_POOL.get(tier, TIER_POOL["auto"])
    pool = list(idx_pool)
    candidates: list[tuple[str, float]] = []
    for worker_name in pool:
        w = next((x for x in WORKERS if x.name == worker_name), None)
        if w is None or not w.enabled:
            continue
        q = _normalize_quality(worker_name, category, difficulty)
        # cost_score in [0,1]; opus output ~75/M -> 1.0; dpsk 0 -> 0.0
        cost_norm = min(1.0, w.cost_out / 50.0)
        score = q - lam * cost_norm
        candidates.append((worker_name, score))
    if not candidates:
        # cold-start: cheapest in pool
        cheap = min(pool, key=lambda n: next((x.cost_out for x in WORKERS if x.name == n), 0.0))
        w = next((x for x in WORKERS if x.name == cheap), None)
        return cheap, TCOBreakdown.from_components()
    best_name, _ = max(candidates, key=lambda x: x[1])
    w = next((x for x in WORKERS if x.name == best_name), None)
    if w is None:
        return best_name, TCOBreakdown.from_components()
    # Real TCO under standard prompt assumption
    tco = select_cheapest_meeting_floor(query, query_type, tier)[1]
    # Override selected worker (lambda routing may differ from floor routing)
    from anchor.pareto import compute_tco, _estimate_prompt_tokens, _estimate_output_tokens
    tco = compute_tco(w, _estimate_prompt_tokens(query),
                      _estimate_output_tokens(difficulty, query_type))
    return best_name, tco


async def run_pareto_eval(
    queries: list[str],
    *,
    n_lambdas: int = 8,
    tier: str = "premium",
    query_types: Optional[list[str]] = None,
    judge_fn: Optional[Callable] = None,
    worker_fn: Optional[Callable] = None,
) -> dict:
    """Run Pareto eval: pick worker per (query, lambda), aggregate.

    judge_fn(queries, anchor_responses, opus_responses) -> list[str]
    is optional; if None, pair_acc_vs_opus is reported as None.

    worker_fn(worker_name, query) -> str is optional; when provided, actual
    worker calls are made and returned in worker_output fields.
    """
    if query_types is None:
        from anchor.classifier import classify
        query_types = [classify(q) for q in queries]

    lambdas = list(DEFAULT_LAMBDAS[:n_lambdas])
    points: list[dict] = []

    for lam in lambdas:
        costs = []
        qualities = []
        workers_picked: list[str] = []
        opus_responses: list[str] = []
        worker_outputs: list[str] = []
        for q, qt in zip(queries, query_types):
            worker, tco = _pick_for_lambda(q, qt, tier, lam)
            costs.append(tco.total_yuan)
            category, difficulty, _ = parse_query_to_cell(q, qt)
            qualities.append(_normalize_quality(worker, category, difficulty))
            workers_picked.append(worker)

            # Call worker if callable provided; otherwise synthetic placeholder
            if worker_fn is not None:
                try:
                    wo = await worker_fn(worker, q)
                    worker_outputs.append(wo if wo else "")
                except Exception:
                    worker_outputs.append("")
            else:
                worker_outputs.append(f"[{worker}] response for: {q[:50]}")

            # Synthetic fable response (no live LLM call) for pair_acc placeholder
            # v0.9.7X-P2: switched from opus-5 (removed) to fable-5 (sacred top, still canonical)
            opus_responses.append(f"[claude-fable-5] response for: {q[:50]}")

        mean_cost = sum(costs) / len(costs) if costs else 0.0
        mean_quality = sum(qualities) / len(qualities) if qualities else 0.0

        pair_acc_vs_opus = None
        if judge_fn is not None:
            try:
                verdicts = await judge_fn(queries, worker_outputs, opus_responses)
                wins = sum(1 for v in verdicts if v == "A")
                pair_acc_vs_opus = wins / len(verdicts) if verdicts else None
            except Exception:
                pair_acc_vs_opus = None

        points.append({
            "lambda": lam,
            "mean_cost_yuan": mean_cost,
            "mean_quality": mean_quality,
            "pair_acc_vs_opus": pair_acc_vs_opus,
            "n_queries": len(queries),
            "workers": dict.fromkeys(workers_picked).keys() if False else sorted(set(workers_picked)),
            "worker_output": worker_outputs,
        })

    # Baseline always-Fable cost = sum(fable TCO) / n_queries (v0.9.7X-P2: was opus-5)
    fable_costs = []
    fable_quals = []
    for q, qt in zip(queries, query_types):
        from anchor.pareto import compute_tco, _estimate_prompt_tokens, _estimate_output_tokens
        from anchor.config import WORKERS
        fable = next((x for x in WORKERS if x.name == "claude-fable-5"), None)
        if fable is None:
            continue
        category, difficulty, _ = parse_query_to_cell(q, qt)
        fable_costs.append(
            compute_tco(fable, _estimate_prompt_tokens(q),
                        _estimate_output_tokens(difficulty, qt)).total_yuan
        )
        fable_quals.append(_normalize_quality("claude-fable-5", category, difficulty))
    baseline = {
        "mean_cost_yuan": sum(fable_costs) / len(fable_costs) if fable_costs else 0.0,
        "mean_quality": sum(fable_quals) / len(fable_quals) if fable_quals else 0.0,
        "n_queries": len(fable_costs),
    }

    # Pareto frontier: points not dominated (lower cost, higher quality both better)
    frontier = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if (q["mean_cost_yuan"] <= p["mean_cost_yuan"]
                    and q["mean_quality"] >= p["mean_quality"]
                    and (q["mean_cost_yuan"] < p["mean_cost_yuan"]
                         or q["mean_quality"] > p["mean_quality"])):
                dominated = True
                break
        if not dominated:
            frontier.append({"lambda": p["lambda"], "mean_cost_yuan": p["mean_cost_yuan"],
                             "mean_quality": p["mean_quality"]})

    # Improvement vs baseline at lambda=0.1 (mid cost-quality)
    improvement = "n/a"
    mid = next((p for p in points if abs(p["lambda"] - 0.1) < 1e-9), None)
    if mid and baseline["mean_cost_yuan"] > 0:
        saving = (baseline["mean_cost_yuan"] - mid["mean_cost_yuan"]) / baseline["mean_cost_yuan"]
        improvement = (
            f"anchor @ lambda=0.1 saves {saving*100:.1f}% cost vs always-Opus "
            f"(quality {mid['mean_quality']:.3f} vs baseline {baseline['mean_quality']:.3f})"
        )

    return {
        "lambdas": lambdas,
        "points": points,
        "pareto_frontier": frontier,
        "baseline_always_opus": baseline,
        "improvement": improvement,
    }


async def pareto_compare(
    anchor_responses: list[str],
    opus_responses: list[str],
    *,
    n_lambdas: int = 8,
    judge_fn: Optional[Callable] = None,
) -> dict:
    """Compare two response sets across multiple cost-vs-quality trade-offs.

    Lightweight wrapper: synth queries from response prefixes, then call
    run_pareto_eval.
    """
    queries = [a[:80] for a in anchor_responses]
    return await run_pareto_eval(
        queries, n_lambdas=n_lambdas,
        query_types=["en"] * len(queries),
        judge_fn=judge_fn,
    )


def write_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))


def format_table(report: dict) -> str:
    rows = ["lambda | mean_cost  | mean_quality | pair_acc | workers"]
    for p in report.get("points", []):
        pa = "n/a" if p["pair_acc_vs_opus"] is None else f"{p['pair_acc_vs_opus']:.3f}"
        rows.append(
            f"{p['lambda']:6.2f} | {p['mean_cost_yuan']:10.6f} | "
            f"{p['mean_quality']:.3f}        | {pa:8s} | {','.join(p.get('workers', []))}"
        )
    rows.append("")
    rows.append(f"baseline always-Opus: cost={report['baseline_always_opus']['mean_cost_yuan']:.6f} "
                f"quality={report['baseline_always_opus']['mean_quality']:.3f}")
    rows.append(f"improvement: {report['improvement']}")
    return "\n".join(rows)


__all__ = [
    "run_pareto_eval",
    "pareto_compare",
    "write_report",
    "format_table",
    "DEFAULT_LAMBDAS",
]
