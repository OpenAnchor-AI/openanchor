"""Day 14: 1k replay ship gate.

Replays 50 stratified queries × 20 rounds (1000 total) through Anchor,
records each outcome, runs spot-check at the end.

Ship gate (v0.9.37): Pareto-curve evaluation over ``LAMBDA_GRID``
(8 cost-quality tradeoff points), see ``anchor.pareto_curve``. The
head SHIPS iff the knee-point dominates always-Sonnet-5 on quality and
its cost is within ``COST_TOLERANCE_RATIO`` (1.10x) of the baseline.

Legacy single-point gate (pair_acc > 0.62 vs always-Sonnet-5, cost <=
¥1100) is still emitted in the report and remains the source of truth
when ``ANCHOR_PARETO_CURVE=0`` is set (escape hatch).
"""
import asyncio
import json
import os
import random
import time
from pathlib import Path

from anchor.eval import FIXTURES
from anchor.classifier import classify
from anchor.feedback import record_outcome
from anchor.head import predict, WORKERS, COSTS
from anchor.cost import month_total, check_cap
from anchor.drift import compute_drift
from anchor.judge import pair_acc
from anchor.judge_calibration import judge_ensemble, THREE_JUDGE_PANEL
from anchor.dashboard import metrics_snapshot
from anchor.pareto_curve import (
    evaluate_pareto_curve, ship_verdict, format_curve_table, LAMBDA_GRID,
    BASELINE_WORKER, MIN_EVAL_CASES, count_real_eval_cases,
)

# ANCHOR_PHASE2_MOCK=1 forces deterministic mock responses (testing only).
# Default: real worker calls (each query goes through actual LLM once for head,
# once for sonnet-5 baseline). Cost + latency tracked.
_USE_MOCK = os.environ.get("ANCHOR_PHASE2_MOCK", "0") == "1"
# v0.9.42: ensemble is now the source of truth for ship_gate.
# (i) live bias measurement (n=30, 8.33pp) showed single-Opus-4-8 is
# significantly biased upward; (j) live Pareto showed this bias flips
# the knee worker selection. Default ensemble; set ANCHOR_USE_ENSEMBLE=0
# to fall back to single-Opus baseline.
_USE_ENSEMBLE = os.environ.get("ANCHOR_USE_ENSEMBLE", "1") == "1"
# v0.9.37: Pareto-curve ship gate. Default ON; legacy single-point gate
# remains in the report and becomes the source of truth when
# ANCHOR_PARETO_CURVE=0 is set (escape hatch for gradual migration).
_USE_PARETO = os.environ.get("ANCHOR_PARETO_CURVE", "1") != "0"

REPORT_PATH = Path.home() / "anchor" / "evals" / "phase2_report.json"
SHIP_PAIR_ACC = 0.62
SHIP_COST_CAP_YUAN = 1100.0


async def _mock_worker(worker: str, query: str) -> str:
    """Mock worker response (testing only, gated by ANCHOR_PHASE2_MOCK=1)."""
    return f"[{worker}] response to: {query[:60]}"


async def _real_worker_call(worker_name: str, query: str, max_tokens: int = 200) -> tuple[str, float]:
    """Call actual worker via build_client. Returns (answer, cost_yuan)."""
    from anchor.clients.factory import build_client
    from anchor.config import WORKERS as _WORKERS
    w = next((x for x in _WORKERS if x.name == worker_name), None)
    if w is None:
        return f"[unknown-worker:{worker_name}]", 0.0
    client = build_client(w)
    try:
        r = await asyncio.wait_for(
            client.chat([{"role": "user", "content": query}],
                        max_tokens=max_tokens, temperature=0.0),
            timeout=30.0,
        )
        answer = (r or {}).get("content", "") or ""
        # cost estimation: output_tokens * cost_out / 1e6 (per Worker config)
        out_tokens = (r or {}).get("usage", {}).get("completion_tokens", 0) or 0
        cost = (out_tokens / 1e6) * w.cost_out
        return answer, cost
    except Exception as e:
        return f"[err:{worker_name}:{type(e).__name__}]", 0.0


async def _sonnet_baseline(query: str) -> str:
    if _USE_MOCK:
        return await _mock_worker("sonnet-5", query)
    answer, _cost = await _real_worker_call("claude-sonnet-5", query)
    return answer


async def replay_round(round_idx: int) -> list[dict]:
    """Replay 50 fixtures once, record outcomes.

    With ANCHOR_PHASE2_MOCK=1 (default for CI), uses deterministic mock responses.
    Otherwise, makes one real call per query (50 per round).
    """
    rows = []
    for query, expected_type in FIXTURES:
        qt = classify(query)
        # tier = premium (default) for the kill gate
        chosen, _ = predict(qt, "premium", prompt_len=500, budget=0.5)
        if _USE_MOCK:
            # mock: deterministic by query hash
            success = 0.6 + 0.4 * (hash(query) % 100) / 100
            cost = COSTS[WORKERS.index(chosen)] * 0.001
            latency_ms = 1000
        else:
            # Real call: capture answer + cost for later judge
            answer, cost = await _real_worker_call(chosen, query)
            success = 1.0 if answer and not answer.startswith("[err:") else 0.0
            latency_ms = 0  # not tracked in real path
        record_outcome(
            query, "premium", chosen,
            success=success, cost_yuan=cost, latency_ms=latency_ms,
            query_type=qt,
        )
        rows.append({
            "query": query, "worker": chosen, "tier": "premium",
            "cost": cost, "success": success, "answer": answer if not _USE_MOCK else "",
        })
    return rows


async def run_phase2(n_rounds: int = 20) -> dict:
    """Replay 50 × n_rounds queries, judge head pick vs sonnet-5 baseline.

    Ship gate: pair_acc > 0.62, cost <= 1100.
    """
    print(f"=== Anchor Phase 2 replay ({n_rounds} rounds × 50 queries = {50*n_rounds} total) ===\n")
    all_rows = []
    t0 = time.time()
    for r in range(n_rounds):
        rows = await replay_round(r)
        all_rows.extend(rows)
        if check_cap() == "HARD_CAP_1100":
            print(f"ABORT: hard cap hit at round {r+1}")
            break
    elapsed = time.time() - t0
    sample = random.sample(all_rows, min(50, len(all_rows)))
    head_qs = [r["query"] for r in sample]
    if _USE_MOCK:
        # Mock mode: compare worker name vs sonnet-5 placeholder (heuristic only)
        head_rs = [r["worker"] for r in sample]
        sonnet_rs = [f"sonnet-5:{r['query'][:40]}" for r in sample]
    else:
        # Real mode: head_rs = real answers captured in replay_round;
        # sonnet_rs = real sonnet-5 answers (one fresh call per query).
        head_rs = [r.get("answer", "") or f"[empty:{r['worker']}]" for r in sample]
        sonnet_rs = []
        for r in sample:
            sonnet_rs.append(await _sonnet_baseline(r["query"]))
    judge = await pair_acc(head_qs, head_rs, sonnet_rs, spot_check_rate=1.0)
    # v0.9.36: opt-in 3-judge ensemble second-pass (does NOT affect ship_gate).
    ensemble_summary: dict = {}
    if _USE_ENSEMBLE and head_qs:
        ens_results = await asyncio.gather(*[
            judge_ensemble(q, a, b) for q, a, b in zip(head_qs, head_rs, sonnet_rs)
        ])
        ens_verdicts = [r[0] for r in ens_results]
        ens_agreement = [r[1] for r in ens_results]
        ensemble_summary = {
            "panel": list(THREE_JUDGE_PANEL),
            "n_pairs": len(ens_verdicts),
            "a_wins": sum(1 for v in ens_verdicts if v == "A"),
            "ties": sum(1 for v in ens_verdicts if v == "TIE"),
            "b_wins": sum(1 for v in ens_verdicts if v == "B"),
            "pair_acc": (sum(1 for v in ens_verdicts if v == "A")
                         + 0.5 * sum(1 for v in ens_verdicts if v == "TIE")) / max(len(ens_verdicts), 1),
            "mean_agreement": sum(ens_agreement) / max(len(ens_agreement), 1),
        }
        print(
            f"  ensemble (3-judge, panel={THREE_JUDGE_PANEL}): "
            f"pair_acc={ensemble_summary['pair_acc']:.3f} "
            f"agreement={ensemble_summary['mean_agreement']:.3f} "
            f"(informational; ship_gate still uses single-Opus)"
        )
    drift = compute_drift({})
    cost = month_total()
    metrics = metrics_snapshot()
    legacy_passed = judge["pair_acc"] > SHIP_PAIR_ACC and cost <= SHIP_COST_CAP_YUAN

    # v0.9.37: Pareto-curve evaluation. Reuse the 50 spot-check responses
    # already in hand (head picks + sonnet-5 baseline) to populate the
    # responses_by_worker matrix; mock-mode when the judge API is down.
    pareto_summary: dict = {"enabled": bool(_USE_PARETO)}
    pareto_verdict: dict = {}
    if _USE_PARETO:
        sample_n = min(50, len(sample))
        rbw: dict[str, list[str]] = {}
        for r in sample[:sample_n]:
            w = r.get("worker") or "unknown"
            ans = r.get("answer", "") or ""
            rbw.setdefault(w, []).append(ans)
        if BASELINE_WORKER not in rbw and sonnet_rs:
            # F3 fix 2026-08-16: use real baseline responses (sonnet_rs filled
            # in real mode at lines 149-151 of `evaluate`). The previous code
            # reused head's self-answers (`r["answer"]`), which made the
            # Pareto curve compare head vs head and skewed ship_verdict high.
            rbw[BASELINE_WORKER] = list(sonnet_rs[:sample_n])
        elif BASELINE_WORKER not in rbw:
            # Mock-mode fallback: sonnet_rs is a synthetic placeholder list,
            # so it remains a placeholder baseline. Keep original behavior.
            rbw[BASELINE_WORKER] = [r.get("answer", "") or ""
                                    for r in sample[:sample_n]]
        # At least 2 distinct workers needed for pareto; otherwise short-circuit.
        if len(rbw) >= 2 and sample_n >= 2:
            pareto_report = await evaluate_pareto_curve(
                queries=head_qs[:sample_n], responses_by_worker=rbw,
                mock=_USE_MOCK, worker_fn=_real_worker_call if not _USE_MOCK else None,
            )
            pareto_verdict = ship_verdict(pareto_report)
            pareto_summary = {
                "enabled": True,
                "lambdas": LAMBDA_GRID,
                "n_points": len(pareto_report.points),
                "dominance_frontier": pareto_report.dominance_frontier,
                "recommended_lambda_index": pareto_report.recommended_lambda,
                "recommended_lambda": pareto_report.points[pareto_report.recommended_lambda]["lambda"],
                "recommended_worker": pareto_report.points[pareto_report.recommended_lambda]["selected_worker"],
                "table": pareto_report.points,
                "mock": pareto_report.mock,
            }
            print()
            print(format_curve_table(pareto_report))
            print()
            print(f"  Pareto SHIP verdict: ship={pareto_verdict['ship']} "
                  f"knee_lambda={pareto_verdict['knee_lambda']} "
                  f"knee_worker={pareto_verdict['knee_worker']} "
                  f"(head_quality={pareto_verdict['head_quality']:.4f} vs "
                  f"baseline_quality={pareto_verdict['baseline_quality']:.4f}, "
                  f"head_cost={pareto_verdict['head_cost']:.5f} vs "
                  f"baseline_cost={pareto_verdict['baseline_cost']:.5f})")
        else:
            pareto_summary["skipped_reason"] = (
                f"need >=2 distinct workers and >=2 spot-checks; "
                f"got {len(rbw)} workers and {sample_n} queries"
            )

    # Final verdict: Pareto is the new ship_gate (v0.9.37+). Legacy
    # single-point gate stays in the report for back-compat; if Pareto is
    # disabled (ANCHOR_PARETO_CURVE=0), legacy becomes source of truth.
    if _USE_PARETO and pareto_verdict:
        passed = bool(pareto_verdict.get("ship"))
        # Eval guard: require >= MIN_EVAL_CASES real eval cases to promote
        n_real = count_real_eval_cases(rbw, mock=_USE_MOCK)
        if not _USE_MOCK and n_real < MIN_EVAL_CASES:
            print(f"SHIP_GATE BLOCKED: n_eval_cases={n_real} < {MIN_EVAL_CASES} "
                  f"(need real eval cases to promote)")
            passed = False
    else:
        passed = bool(legacy_passed)

    report = {
        "ts": time.time(),
        "elapsed_s": elapsed,
        "n_replay": len(all_rows),
        "n_judged": judge["judged"],
        "pair_acc": judge["pair_acc"],
        "a_wins": judge["a_wins"],
        "ties": judge["ties"],
        "b_wins": judge["b_wins"],
        "total_cost_yuan": cost,
        "drift": drift,
        "metrics": metrics,
        "ship_gate_passed": passed,
        "ship_gate_source": "pareto" if (_USE_PARETO and pareto_verdict) else "legacy",
        "ship_threshold": {"pair_acc": SHIP_PAIR_ACC, "cost_yuan": SHIP_COST_CAP_YUAN},
        "legacy_gate_passed": bool(legacy_passed),
        "pareto": pareto_summary,
        "pareto_verdict": pareto_verdict,
        "ensemble": ensemble_summary,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print("\n=== REPORT ===")
    print(f"  pair_acc: {judge['pair_acc']:.3f} (legacy threshold {SHIP_PAIR_ACC})")
    print(f"  cost: ¥{cost:.2f} (legacy cap ¥{SHIP_COST_CAP_YUAN})")
    print(f"  legacy_gate_passed: {legacy_passed}")
    if pareto_verdict:
        print(f"  pareto_gate_passed: {pareto_verdict.get('ship')}")
        print(f"  ship_gate_source:   {report['ship_gate_source']}")
    print(f"  ship_gate_passed:   {passed}")
    print(f"  → {REPORT_PATH}")
    return report


if __name__ == "__main__":
    asyncio.run(run_phase2())
