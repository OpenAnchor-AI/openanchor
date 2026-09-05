"""Opus 5 judge: returns pair_acc vs always-Sonnet-5 baseline.

Day 8 per Fable 5 Phase 2 plan. Used by eval harness + retrain + spot-check.
"""
import asyncio
import random
import re
import time
from pathlib import Path
from typing import Optional

import logging

# Module-level logger for cache write diagnostics (v0.9.42).
judge_logger = logging.getLogger("anchor.judge")

# Serialize concurrent cache writes so 20+ callers don't collide on the same
# SQLite file or leave partial rows behind (v0.9.42). Asyncio lock is
# sufficient because all callers share a single event loop (the cache
# writers themselves are sync sqlite3 operations).
# v0.9.46j: in-process LRU front-end for hot pairs. Avoids DB lookup
# during tight scoring loops (retrain, eval). Bounded at 4096 entries.
# Cache key: hash(query, response_a, response_b, judge_model). Verdict stored
# as plain string ("A"/"B"/"TIE"). On overflow, evict oldest (FIFO).
from collections import OrderedDict
import hashlib as _hl

_LRU_MAX = 4096
_lru_cache: "OrderedDict[str, str]" = OrderedDict()
_lru_hits = 0
_lru_misses = 0


def _lru_key(query: str, response_a: str, response_b: str, judge_model: str) -> str:
    h = _hl.sha256()
    h.update(query.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(response_a.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(response_b.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(judge_model.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _lru_get(query: str, response_a: str, response_b: str, judge_model: str) -> "str | None":
    """Return cached verdict or None. Updates LRU position on hit."""
    global _lru_hits, _lru_misses
    k = _lru_key(query, response_a, response_b, judge_model)
    if k in _lru_cache:
        _lru_cache.move_to_end(k)
        _lru_hits += 1
        return _lru_cache[k]
    _lru_misses += 1
    return None


def _lru_put(query: str, response_a: str, response_b: str, judge_model: str, verdict: str) -> None:
    """Populate LRU after a judge call (cache miss -> real verdict)."""
    k = _lru_key(query, response_a, response_b, judge_model)
    _lru_cache[k] = verdict
    _lru_cache.move_to_end(k)
    while len(_lru_cache) > _LRU_MAX:
        _lru_cache.popitem(last=False)


def lru_stats() -> dict:
    """Observability: hits/misses/size for the in-process LRU."""
    return {
        "hits": _lru_hits,
        "misses": _lru_misses,
        "size": len(_lru_cache),
        "maxsize": _LRU_MAX,
    }


def lru_clear() -> None:
    """Reset the in-process LRU (for tests + admin endpoint)."""
    global _lru_hits, _lru_misses
    _lru_cache.clear()
    _lru_hits = 0
    _lru_misses = 0


_judge_cache_write_lock = asyncio.Lock()

# USD -> CNY rate (2026-07 baseline; refresh from anchor cost guard periodically)
_USD_TO_CNY = 7.2

# Default Opus 5 pricing (yuan per million tokens, USD-denominated).
# Used for cost estimation when real usage.token info is missing.
_OPUS_INPUT_USD_PER_M = 15.0   # baosiapi shared tier
_OPUS_OUTPUT_USD_PER_M = 75.0

JUDGE_PROMPT = """\
You are an impartial LLM quality judge. Compare two responses to the same query.

Query: {query}

Response A: {response_a}
Response B: {response_b}

Reply with EXACTLY ONE of:
- "A" if A is clearly better
- "B" if B is clearly better
- "TIE" if roughly equal

Do not explain. Just the letter."""


async def _call_opus(prompt: str, *, max_tokens: int = 10, judge_model: str = "claude-opus-5") -> tuple[str, dict, int, float]:
    """Call judge model via existing anchor clients.

    Returns (content, usage_dict, latency_ms, cost_yuan). On any failure returns
    ("", {}, 0, 0.0) so the caller can default to TIE.
    """
    from anchor.clients.factory import build_client
    from anchor.config import WORKERS
    try:
        w = next(x for x in WORKERS if x.name == judge_model)
    except StopIteration:
        return "", {}, 0, 0.0
    try:
        client = build_client(w)
        t0 = time.time()
        r = await client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0,
        )
        latency_ms = int((time.time() - t0) * 1000)
        content = (r.get("content") or "").strip()
        usage = r.get("usage") or {}
        # Estimate cost from real usage if available; fall back to None
        pt = usage.get("prompt_tokens") or 0
        ct = usage.get("completion_tokens") or 0
        cost_yuan = (pt / 1e6 * _OPUS_INPUT_USD_PER_M + ct / 1e6 * _OPUS_OUTPUT_USD_PER_M) * _USD_TO_CNY
        return content, usage, latency_ms, cost_yuan
    except Exception:
        return "", {}, 0, 0.0


async def judge_pair(query: str, response_a: str, response_b: str) -> str:
    """Return 'A' / 'B' / 'TIE' for one pair.

    Thin wrapper over judge_batch (n=1) so single callers share the same cache +
    concurrency code path. Kept for backward compatibility with retrain / phase2.
    """
    verdicts = await judge_batch([query], [response_a], [response_b])
    return verdicts[0] if verdicts else "TIE"



def _mock_judge(query: str, a: str, b: str) -> str:
    """Deterministic mock: longer response wins (testing only)."""
    if len(a) > len(b) * 1.1:
        return "A"
    if len(b) > len(a) * 1.1:
        return "B"
    return "TIE"


def mock_pair_acc(queries, a, b) -> dict:
    """Synchronous mock for testing without API calls."""
    n = len(queries)
    aw = sum(1 for q, x, y in zip(queries, a, b) if _mock_judge(q, x, y) == "A")
    bw = sum(1 for q, x, y in zip(queries, a, b) if _mock_judge(q, x, y) == "B")
    t = n - aw - bw
    return {
        "judged": n, "a_wins": aw, "b_wins": bw, "ties": t,
        "pair_acc": (aw + 0.5 * t) / max(n, 1),
    }



# ---------- batch judge (v0.9.36) ----------

# Regex used by every parser path; shared so tests can assert against it.
_VERDICT_RE = re.compile(r"\b(A|B|TIE)\b", re.I)


def _parse_verdict(raw: str) -> str:
    """Extract A / B / TIE from raw judge output; default to TIE on miss."""
    m = _VERDICT_RE.search(raw or "")
    return m.group(1).upper() if m else "TIE"


async def _call_opus_batch(
    prompts: list[str],
    *,
    judge_model: str = "claude-opus-5",
    concurrency: int = 5,
    timeout_s: float = 30.0,
    max_tokens: int = 10,
) -> list[tuple[str, dict, int, float]]:
    """Call judge model for each prompt with bounded concurrency.

    Reuses the same client factory / chat path as _call_opus. On per-call failure
    (timeout, network, judge_model missing), returns ("", {}, 0, 0.0) so the
    caller treats the verdict as TIE rather than raising.

    Returns list of (content, usage, latency_ms, cost_yuan) in input order.
    """
    if not prompts:
        return []

    from anchor.clients.factory import build_client
    from anchor.config import WORKERS
    try:
        w = next(x for x in WORKERS if x.name == judge_model)
    except StopIteration:
        return [("", {}, 0, 0.0) for _ in prompts]
    try:
        client = build_client(w)
    except Exception:
        return [("", {}, 0, 0.0) for _ in prompts]

    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[tuple[str, dict, int, float] | None] = [None] * len(prompts)

    async def _one(i: int, prompt: str) -> None:
        async with sem:
            t0 = time.time()
            try:
                r = await asyncio.wait_for(
                    client.chat(
                        [{"role": "user", "content": prompt}],
                        max_tokens=max_tokens, temperature=0.0,
                    ),
                    timeout=timeout_s,
                )
                latency_ms = int((time.time() - t0) * 1000)
                content = (r.get("content") or "").strip()
                usage = r.get("usage") or {}
                pt = usage.get("prompt_tokens") or 0
                ct = usage.get("completion_tokens") or 0
                cost_yuan = (pt / 1e6 * _OPUS_INPUT_USD_PER_M + ct / 1e6 * _OPUS_OUTPUT_USD_PER_M) * _USD_TO_CNY
                results[i] = (content, usage, latency_ms, cost_yuan)
            except Exception:
                results[i] = ("", {}, 0, 0.0)

    await asyncio.gather(*[_one(i, p) for i, p in enumerate(prompts)])
    # type: ignore[list-item] -- every slot was filled above
    return results  # type: ignore[return-value]


async def judge_batch(
    queries: list[str],
    response_a: list[str],
    response_b: list[str],
    *,
    judge_model: str = "claude-opus-5",
    concurrency: int = 5,
    db_path: Optional[Path] = None,
) -> list[str]:
    """Cache-aware batch judge. Returns verdicts in input order.

    Cache lookup is keyed by (query, A, B) + judge_model so swapping judges
    (e.g. Opus -> Gemini) does not poison the cache.

    Only the misses hit the judge API; hits return immediately.
    """
    from anchor.db import _judge_cache_key, lookup_judge_cache, insert_judge_cache_many

    n = len(queries)
    if not (n == len(response_a) == len(response_b)):
        raise ValueError("queries, response_a, response_b must have same length")
    if n == 0:
        return []

    keys = [_judge_cache_key(q, a, b) for q, a, b in zip(queries, response_a, response_b)]
    cached = lookup_judge_cache(keys, judge_model, db_path=db_path)

    # Find miss indices + their prompts
    miss_positions: list[int] = []
    miss_prompts: list[str] = []
    for i, k in enumerate(keys):
        if k not in cached:
            miss_positions.append(i)
            miss_prompts.append(JUDGE_PROMPT.format(
                query=queries[i], response_a=response_a[i], response_b=response_b[i],
            ))

    if miss_prompts:
        # Single batch call (concurrent internally); cheapest fast-path
        raw_results = await _call_opus_batch(
            miss_prompts, judge_model=judge_model, concurrency=concurrency,
        )
        rows_to_write: list[dict] = []
        ts_now = time.time()
        for pos, (content, usage, lat_ms, cost_yuan) in zip(miss_positions, raw_results):
            # audit 2026-08-16 (F10): a failed judge call returns ("", {}, 0, 0.0)
            # and _parse_verdict("") defaults to TIE — which was then PERSISTED to
            # the judge cache, biasing pair_acc toward 0.5 forever. Use a
            # MISSING sentinel, do NOT write the row, and let downstream score
            # computation exclude it.
            if not (content or "").strip():
                cached[keys[pos]] = "MISSING"
                continue
            verdict = _parse_verdict(content)
            cached[keys[pos]] = verdict
            rows_to_write.append({
                "cache_key": keys[pos],
                "query": queries[pos],
                "response_a": response_a[pos],
                "response_b": response_b[pos],
                "judge_model": judge_model,
                "verdict": verdict,
                "raw_judge": content,
                "ts": ts_now,
                "latency_ms": lat_ms,
                "cost_yuan": cost_yuan,
            })
        # v0.9.42: serialize concurrent cache writes (lock module-level
        # singleton) and emit full traceback on failure instead of
        # silently swallowing. In-memory `cached` dict already reflects
        # the verdict, so the caller still gets the right result — but
        # on-disk cache may be stale until the next successful write.
        try:
            async with _judge_cache_write_lock:
                insert_judge_cache_many(rows_to_write, db_path=db_path)
        except Exception:
            judge_logger.exception(
                "insert_judge_cache_many failed (%d rows, db=%s); "
                "caller result is intact but on-disk cache stale",
                len(rows_to_write), db_path,
            )

    return [cached[k] for k in keys]


def _bootstrap_ci(
    verdicts: list[str],
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Non-parametric bootstrap 95% CI for pair_acc.

    pair_acc = (a_wins + 0.5 * ties) / n.
    Returns (mean, ci_low, ci_high). Deterministic given `seed`.
    """
    # audit 2026-08-16 (F10): MISSING verdicts (failed judge calls) are
    # excluded from the CI — scoring them as B would bias pair_acc downward.
    verdicts = [v for v in verdicts if v != "MISSING"]
    n = len(verdicts)
    if n == 0:
        return 0.0, 0.0, 0.0
    # score per verdict: A=1.0, TIE=0.5, B=0.0
    scores = [1.0 if v == "A" else (0.5 if v == "TIE" else 0.0) for v in verdicts]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1] if n_boot >= 40 else means[-1]
    return sum(scores) / n, lo, hi


async def pair_acc(
    queries: list[str],
    response_a: list[str],
    response_b: list[str],
    *,
    spot_check_rate: float = 1.0,
    judge_model: str = "claude-opus-5",
    concurrency: int = 5,
    db_path: Optional[Path] = None,
    bootstrap_n: int = 1000,
) -> dict:
    """Judge pairs and return pair_acc plus 95% bootstrap CI and judge cost.

    Returns dict with keys:
      judged, a_wins, b_wins, ties, pair_acc,
      pair_acc_ci_low, pair_acc_ci_high, cost_yuan_total, judge_model
    """
    if not (len(queries) == len(response_a) == len(response_b)):
        raise ValueError("queries, response_a, response_b must have same length")
    n = len(queries)
    if n == 0:
        return {
            "judged": 0, "a_wins": 0, "b_wins": 0, "ties": 0, "pair_acc": 0.0,
            "pair_acc_ci_low": 0.0, "pair_acc_ci_high": 0.0,
            "cost_yuan_total": 0.0, "judge_model": judge_model,
        }

    indices = list(range(n))
    if spot_check_rate < 1.0:
        k = max(1, int(n * spot_check_rate))
        indices = sorted(random.sample(indices, k))

    sub_q = [queries[i] for i in indices]
    sub_a = [response_a[i] for i in indices]
    sub_b = [response_b[i] for i in indices]

    verdicts = await judge_batch(
        sub_q, sub_a, sub_b,
        judge_model=judge_model,
        concurrency=concurrency,
        db_path=db_path,
    )

    a_wins = sum(1 for v in verdicts if v == "A")
    b_wins = sum(1 for v in verdicts if v == "B")
    ties = sum(1 for v in verdicts if v == "TIE")
    judged = len(verdicts)
    pair_acc_val = (a_wins + 0.5 * ties) / max(judged, 1)

    # Bootstrap CI for the judged sample
    _, ci_low, ci_high = _bootstrap_ci(verdicts, n_boot=bootstrap_n)

    # Cost: read what we wrote to the cache this run (rough aggregate).
    # We don't re-query the DB here; cost is also recoverable from
    # /admin/cost/dashboard via the judge_cache table.
    cost_yuan_total = 0.0
    try:
        from anchor.db import DEFAULT_PATH
        p = Path(db_path) if db_path else DEFAULT_PATH
        import sqlite3
        conn = sqlite3.connect(str(p))
        try:
            cur = conn.execute(
                "SELECT COALESCE(SUM(cost_yuan), 0.0) FROM judge_cache "
                "WHERE ts >= ? AND judge_model = ?",
                (time.time() - 600, judge_model),  # last 10 min
            )
            cost_yuan_total = float(cur.fetchone()[0] or 0.0)
        finally:
            conn.close()
    except Exception:
        cost_yuan_total = 0.0

    return {
        "judged": judged,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "pair_acc": pair_acc_val,
        "pair_acc_ci_low": ci_low,
        "pair_acc_ci_high": ci_high,
        "cost_yuan_total": cost_yuan_total,
        "judge_model": judge_model,
    }
