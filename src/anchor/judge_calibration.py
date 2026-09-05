"""3-judge ensemble + calibration curve (v0.9.36).

Establishes a reference baseline free of judge-worker tier overlap.

Background - Unsolvability Ceiling (arXiv 2605.07395): a single LLM
judge that also appears as a candidate worker tier inflates pair_acc
by 13-17pp (truncation 65% MMLU / 57% MedQA). Anchor ship_gate uses
claude-opus-5 as the *only* judge, and Opus 4.8 IS a tier - direct
confound. This module provides a 3-judge panel that does NOT overlap
with the tier pool, so the ensemble can be the new ship_gate baseline.

API:
    - THREE_JUDGE_PANEL: default 3 judges, none in tier selection.
    - judge_ensemble(query, A, B, panel=...): returns (verdict,
      agreement, per_judge_dict) using the existing judge.py cache +
      concurrency infrastructure.
    - judge_calibration_curve(queries, responses_per_worker, ...):
      runs the panel on n_sample pairs and reports per-judge win
      rates + ensemble agreement + Opus-vs-ensemble delta.

Style: mirrors judge.py helpers - async, cache-aware, tolerant of
judge failures (failed judges treated as TIE; panel of 3 supports
1 failure and still returns a majority; 2 failures -> ensemble TIE).
"""
from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from pathlib import Path
from typing import Optional

# Default 3-judge panel. Selection criteria (2026-07-12, updated 2026-07-22):
#   - claude-haiku-4-5: Anthropic family but lightweight tier (no overlap
#     with premium/ultra basic chains; Sonnet-5 and Opus-4-8 do).
#   - minimax-m3: in-house oracle; not in any worker tier selection pool
#     (M3 is reserved as retrain oracle, not a routing target).
#   - gpt-5.6-sol: cross-family (Baosi GPT group), NOT in TIER_POOL, and
#     NOT the ship_gate baseline (claude-opus-5). Avoids Claude-family
#     bias. v0.9.50-p1: replaced claude-sonnet-5 (which is the
#     ship_gate mid-tier baseline; using it as panel would self-bias).
THREE_JUDGE_PANEL: list[str] = [
    "gpt-5.6-sol",
    "grok-4-6-reasoning",
    "claude-fable-5",
]


def _mock_ensemble_vote(
    query: str,
    response_a: str,
    response_b: str,
    seed: int = 0,
) -> dict[str, str]:
    """Deterministic mock ensemble for offline testing.

    Three independent heuristic votes:
      - judge-A: rewards length (mirrors judge._mock_judge)
      - judge-B: rewards second response if it has more question marks
      - judge-C: rewards lexicographically earlier non-empty response
    Deterministic given (query, A, B, seed) so test fixtures are stable.
    """
    rng = random.Random(hash((query, response_a, response_b, seed)) & 0xFFFFFFFF)
    la, lb = len(response_a or ""), len(response_b or "")
    a_vote = "A" if la > lb * 1.05 else ("B" if lb > la * 1.05 else rng.choice(["A", "B", "TIE"]))
    qa = (response_a or "").count("?")
    qb = (response_b or "").count("?")
    b_vote = "B" if qb > qa else ("A" if qa > qb else rng.choice(["A", "B", "TIE"]))
    a_s = (response_a or "").strip()
    b_s = (response_b or "").strip()
    if a_s and not b_s:
        c_vote = "A"
    elif b_s and not a_s:
        c_vote = "B"
    elif a_s and b_s:
        c_vote = "A" if a_s < b_s else ("B" if b_s < a_s else "TIE")
    else:
        c_vote = "TIE"
    return {
        "mock-judge-A": a_vote,
        "mock-judge-B": b_vote,
        "mock-judge-C": c_vote,
    }

def _majority_vote(verdicts: list[str]) -> tuple[str, float]:
    """Aggregate verdicts into (majority, agreement_score).

    agreement_score = share of verdicts matching the majority (1.0 if
    unanimous, 2/3 if one dissenter, 1/3 if split 3-way -> fall back
    to lex smallest verdict with agreement=1/3).
    """
    if not verdicts:
        return "TIE", 0.0
    counts = Counter(verdicts)
    n = len(verdicts)
    top_count = max(counts.values())
    winners = sorted(v for v, c in counts.items() if c == top_count)
    return winners[0], top_count / n


async def _judge_one(
    query: str,
    response_a: str,
    response_b: str,
    judge_model: str,
    *,
    concurrency: int = 5,
    db_path: Optional[Path] = None,
) -> tuple[str, str | None]:
    """Run a single judge_model on a single pair.

    Returns (judge_model, verdict). On failure returns None so the
    ensemble can count successful vs failed judges and surface
    "missing observation" rather than forcing TIE (P0-1 fix).
    """
    from anchor.judge import judge_batch
    try:
        results = await judge_batch(
            [query], [response_a], [response_b],
            judge_model=judge_model,
            concurrency=concurrency,
            db_path=db_path,
        )
        return judge_model, (results[0] if results else None)
    except Exception:
        return judge_model, None


async def judge_ensemble(
    query: str,
    response_a: str,
    response_b: str,
    *,
    panel: Optional[list[str]] = None,
    concurrency: int = 5,
    db_path: Optional[Path] = None,
    mock: bool = False,
    mock_seed: int = 0,
) -> tuple[str | None, float, dict[str, str | None]]:
    """Run a 3-judge ensemble on a single (q, A, B) triple.

    Returns (majority_vote, agreement_score, per_judge_verdicts).
    per_judge_verdicts is {judge_name: A/B/TIE/None} with one entry per
    panel member; failures show up as None (distinct from a judge that
    actually returned TIE). majority_vote is None if no judge succeeded.

    v0.9.53 (P0-1): callers should check `majority is None` or count
    non-None verdicts in per_judge to detect insufficient signal.
    """
    if panel is None:
        try:
            from anchor.config import resolve_judge_panel
            panel = resolve_judge_panel()
        except Exception:
            panel = list(THREE_JUDGE_PANEL)

    if mock:
        per_judge = _mock_ensemble_vote(query, response_a, response_b, seed=mock_seed)
        if len(panel) != 3:
            la = len(response_a or "")
            lb = len(response_b or "")
            base = "A" if la > lb else ("B" if lb > la else "TIE")
            per_judge = {p: base for p in panel}
        majority, agreement = _majority_vote(list(per_judge.values()))
        return majority, agreement, per_judge

    tasks = [
        _judge_one(query, response_a, response_b, model,
                   concurrency=concurrency, db_path=db_path)
        for model in panel
    ]
    rows = await asyncio.gather(*tasks)
    per_judge = {name: verdict for name, verdict in rows}
    valid = [v for v in per_judge.values() if v is not None]
    if not valid:
        return None, 0.0, per_judge
    majority, agreement = _majority_vote(valid)
    return majority, agreement, per_judge

def _win_rate(vs: list[str]) -> float:
    if not vs:
        return 0.0
    scores = [1.0 if v == "A" else (0.5 if v == "TIE" else 0.0) for v in vs]
    return sum(scores) / len(vs)


async def judge_calibration_curve(
    queries: list[str],
    responses_per_worker: dict[str, list[str]],
    *,
    n_sample: int = 100,
    panel: Optional[list[str]] = None,
    baseline_judge: str = "claude-opus-5",
    seed: int = 42,
    mock: bool = False,
) -> dict:
    """Run the ensemble on n_sample pairs and compare to a single baseline.

    Returns dict with per_judge_win_rate, ensemble_pair_acc,
    baseline_pair_acc, ensemble_agreement, opus_vs_ensemble_disagreement,
    opus_vs_ensemble_delta_pp, n_sample, n_total, ts.
    """
    from anchor.judge import judge_batch

    rng = random.Random(seed)
    worker_names = list(responses_per_worker.keys())
    if len(worker_names) < 2:
        raise ValueError("responses_per_worker must have >= 2 workers")
    if not queries:
        raise ValueError("queries must not be empty")
    for w in worker_names:
        if len(responses_per_worker[w]) != len(queries):
            raise ValueError(
                f"worker {w!r} responses length {len(responses_per_worker[w])} "
                f"!= queries length {len(queries)}"
            )

    sample_n = min(n_sample, len(queries))
    sampled_idx = rng.sample(range(len(queries)), sample_n)

    pairs_a_resp: list[str] = []
    pairs_b_resp: list[str] = []
    pair_queries: list[str] = []
    for idx in sampled_idx:
        wa, wb = rng.sample(worker_names, 2)
        pair_queries.append(queries[idx])
        pairs_a_resp.append(responses_per_worker[wa][idx])
        pairs_b_resp.append(responses_per_worker[wb][idx])

    panel = panel or THREE_JUDGE_PANEL
    per_judge_verdicts: dict[str, list[str]] = {m: [] for m in panel}
    ensemble_verdicts: list[str] = []
    agreement_scores: list[float] = []

    if mock:
        for q, a, b in zip(pair_queries, pairs_a_resp, pairs_b_resp):
            votes = _mock_ensemble_vote(q, a, b, seed=seed)
            key_map = {
                panel[0]: "mock-judge-A",
                panel[1]: "mock-judge-B",
                panel[2]: "mock-judge-C",
            }
            for m in panel:
                per_judge_verdicts[m].append(votes.get(key_map.get(m, panel[0]), "TIE"))
            ens, agr, _ = await judge_ensemble(q, a, b, panel=panel, mock=True, mock_seed=seed)
            ensemble_verdicts.append(ens)
            agreement_scores.append(agr)
    else:
        async def _one_pair(q, a, b):
            ens, agr, per = await judge_ensemble(q, a, b, panel=panel)
            return per, ens, agr

        pair_results = await asyncio.gather(*[
            _one_pair(q, a, b) for q, a, b in zip(pair_queries, pairs_a_resp, pairs_b_resp)
        ])
        for per, ens, agr in pair_results:
            for m in panel:
                per_judge_verdicts[m].append(per.get(m, "TIE"))
            ensemble_verdicts.append(ens)
            agreement_scores.append(agr)

    if mock:
        # Mock baseline = first mock judge vote (proxy for current Opus judge).
        baseline_verdicts = list(per_judge_verdicts[panel[0]])
    else:
        baseline_verdicts = await judge_batch(
            pair_queries, pairs_a_resp, pairs_b_resp,
            judge_model=baseline_judge,
        )

    per_judge_win_rate = {m: _win_rate(per_judge_verdicts[m]) for m in panel}
    per_judge_win_rate[baseline_judge] = _win_rate(baseline_verdicts)

    ensemble_pair_acc = _win_rate(ensemble_verdicts)
    baseline_pair_acc = _win_rate(baseline_verdicts)
    ensemble_agreement = (
        sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0
    )

    disagreement = sum(
        1 for ev, bv in zip(ensemble_verdicts, baseline_verdicts) if ev != bv
    )
    delta_pp = abs(baseline_pair_acc - ensemble_pair_acc) * 100.0

    return {
        "per_judge_win_rate": per_judge_win_rate,
        "ensemble_pair_acc": ensemble_pair_acc,
        "baseline_pair_acc": baseline_pair_acc,
        "baseline_judge": baseline_judge,
        "ensemble_agreement": ensemble_agreement,
        "opus_vs_ensemble_disagreement": disagreement,
        "opus_vs_ensemble_delta_pp": delta_pp,
        "n_sample": sample_n,
        "n_total": len(queries),
        "ts": time.time(),
    }

def curve_to_jsonable(curve: dict) -> dict:
    """Convert curve result to JSON-safe dict (floats, no Path objects)."""
    out = dict(curve)
    out["per_judge_win_rate"] = {k: float(v) for k, v in curve["per_judge_win_rate"].items()}
    for k in [
        "ensemble_pair_acc", "baseline_pair_acc",
        "ensemble_agreement", "opus_vs_ensemble_delta_pp",
    ]:
        out[k] = float(curve.get(k, 0.0))
    out["opus_vs_ensemble_disagreement"] = int(curve.get("opus_vs_ensemble_disagreement", 0))
    out["n_sample"] = int(curve.get("n_sample", 0))
    out["n_total"] = int(curve.get("n_total", 0))
    out["ts"] = float(curve.get("ts", 0.0))
    return out

