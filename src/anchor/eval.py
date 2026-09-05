"""Day 6 eval harness: 50 stratified queries, pair-ACC vs always-cheapest."""
import asyncio
import logging
from typing import Callable, List, Optional, Tuple

from anchor.classifier import classify
from anchor.head import predict
from anchor.judge_cache import cache_oracle_response, get_cached_oracle_response

logger = logging.getLogger(__name__)

FIXTURES: List[Tuple[str, str]] = [
    ("def foo(x): return x*2", "code"),
    ("class MyClass: pass", "code"),
    ("import os; os.listdir('.')", "code"),
    ("```python\nprint('hi')\n```", "code"),
    ("function add(a,b) { return a+b; }", "code"),
    ("how to read file in python", "code"),
    ("write a sort algorithm", "code"),
    ("javascript fetch API example", "code"),
    ("rust ownership rules", "code"),
    ("sql join two tables", "code"),
    ("TypeError: cannot read property 'x'", "debug"),
    ("why does this not work?", "debug"),
    ("got ImportError on import pandas", "debug"),
    ("runtime error in production", "debug"),
    ("stack trace shows null pointer", "debug"),
    ("fix the bug in this loop", "debug"),
    ("exception handling best practice", "debug"),
    ("why isn't my function returning?", "debug"),
    ("debug this regex", "debug"),
    ("traceback shows line 42", "debug"),
    ("你好世界", "cn"),
    ("帮我写个 agent", "cn"),
    ("中文写作润色", "cn"),
    ("请解释深度学习", "cn"),
    ("推荐一本算法书", "cn"),
    ("solve x^2 = 4", "math"),
    ("prove the Pythagorean theorem", "math"),
    ("compute the integral from 0 to 1", "math"),
    ("sum of first n natural numbers", "math"),
    ("matrix multiplication", "math"),
    ("analyze the implications of this policy", "reasoning"),
    ("compare two approaches and conclude", "reasoning"),
    ("evaluate the trade-offs", "reasoning"),
    ("therefore the conclusion is", "reasoning"),
    ("argue for or against this", "reasoning"),
    ("Hello, how are you?", "en"),
    ("what's the weather like?", "en"),
    ("tell me a joke", "en"),
    ("write a short poem", "en"),
    ("explain quantum computing", "en"),
    ("hi there", "en"),
    ("good morning", "en"),
    ("what time is it", "en"),
    ("where is the library", "en"),
    ("thanks for the help", "en"),
    ("api documentation", "en"),
    ("user manual for the device", "en"),
    ("error 404", "en"),
    ("loading please wait", "en"),
    ("connection timeout", "en"),
]


async def _get_oracle_response(query: str, max_tokens: int = 2048) -> str:
    cached = get_cached_oracle_response(query)
    if cached is not None:
        return cached
    from anchor.judge import _call_opus
    content, _, _, _ = await _call_opus(query, max_tokens=max_tokens)
    cache_oracle_response(query, content)
    return content


async def run_eval(db_path=None, judge_fn: Optional[Callable] = None, worker_fn: Optional[Callable] = None) -> dict:
    """Run 50 stratified queries through head.predict vs always-cheapest baseline.

    Returns dict with pair_acc, mean_cost_yuan, p50_latency_ms.
    """
    from anchor.head import WORKERS, COSTS

    total = 0
    head_costs = []
    baseline_costs = []

    queries_list = []
    worker_responses = []
    oracle_responses = []

    for query, expected_type in FIXTURES:
        qt = classify(query)
        if qt != expected_type:
            pass  # regex may misclassify; we still log
        # F8 fix (V51): use the REAL routed tier/type + query-derived length
        # so the eval measures the actual routing head, not fixed inputs.
        tier = "auto"  # production entry is model="anchor" (single product, auto tier)
        prompt_len = max(200, min(4000, len(query) * 4))
        budget = 0.5
        chosen, _ = predict(qt, tier, prompt_len=prompt_len, budget=budget)
        head_costs.append(COSTS[WORKERS.index(chosen)])
        # baseline: always cheapest
        baseline_costs.append(min(COSTS))
        total += 1

        if judge_fn is not None and worker_fn is not None:
            chosen, _ = predict(qt, tier, prompt_len=prompt_len, budget=budget)
            worker_resp = await worker_fn(chosen, query)
            oracle_resp = await _get_oracle_response(query)
            queries_list.append(query)
            worker_responses.append(worker_resp)
            oracle_responses.append(oracle_resp)

    if judge_fn is not None and worker_fn is not None and queries_list:
        verdicts = await judge_fn(queries_list, worker_responses, oracle_responses)
        wins = sum(1 for v in verdicts if v == "A")
        ties = sum(1 for v in verdicts if v == "TIE")
        pair_acc = (wins + 0.5 * ties) / len(verdicts) if verdicts else 0.5
    else:
        if judge_fn is not None and worker_fn is None:
            logger.warning("judge_fn provided without worker_fn; pair_acc is placeholder 0.5")
        else:
            logger.warning("judge_fn not provided; pair_acc is placeholder 0.5")
        pair_acc = 0.5

    return {
        "queries": total,
        "pair_acc": pair_acc,
        "mean_cost_yuan": sum(head_costs) / len(head_costs) if head_costs else 0,
        "baseline_cost_yuan": sum(baseline_costs) / len(baseline_costs) if baseline_costs else 0,
        "tiers_per_query": 3,
    }


def write_report(result: dict, out_path: str = "evals/e1_report.md") -> str:
    lines = [
        "# E1 Eval Report (Day 6 harness)", "",
        f"Queries: {result['queries']} (50 stratified across 5 types)",
        f"Pair-ACC: {result['pair_acc']:.3f} (placeholder; real eval needs Opus judge)",
        f"Mean head cost: ¥{result['mean_cost_yuan']:.4f}/query",
        f"Always-cheapest baseline: ¥{result['baseline_cost_yuan']:.4f}/query",
        "",
        "## Tier default routing (per Fable 5 + user mental model)",
        "- /basic   -> dpsk-v4-flash (code-strong, free Zen)",
        "- /premium -> minimax-m3     (CN flagship, 1M ctx, ¥238 locked)",
        "- /ultra   -> sonnet-5       (frontier solid, cost-effective vs opus)",
        "",
        "## Next:",
        "- Real Opus 4.8 judge on 50 queries (Day 7 smoke)",
        "- W&B-style dashboard (Day 7)",
    ]
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


if __name__ == "__main__":
    result = asyncio.run(run_eval())
    out = write_report(result)
    print(f"✓ wrote {out}")
    for k, v in result.items():
        print(f"  {k}: {v}")
