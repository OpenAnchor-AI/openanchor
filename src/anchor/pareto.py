"""First-principles Pareto router: quality floor x TCO optimization.

  min  TCO = sum (input x cost_in + output x cost_out + retry + escalation + judge)
  s.t.  forall turn: quality(turn) >= TIER_FLOORS[tier]
        forall session: sum_cost <= session_budget

v0.9.36 first-principles rewrite. Pareto selection replaces static cascade
rules with quality_table-driven cheapest-worker-above-floor lookup. The
optimizer degrades gracefully on cold-start (no table data -> cheapest in
pool) and on quality-table misses (fall back to next-cheapest, then to
always-cheapest in pool).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from anchor.config import WORKERS, enabled_workers
from anchor.cp import CP_ENABLED, get_interval as _cp_get_interval
from anchor.head import TIER_POOL
from anchor.classifier import classify
from anchor._query_complexity import query_complexity
from anchor.quality_feedback import lookup_with_overlay as _lookup_with_overlay

# Portable fallback: anchor package is anchor.parent.parent from this file.
_ROOT = Path(os.environ.get("ANCHOR_ROOT", Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Per-tier quality floor (hard constraint). Routing must satisfy
# TIER_FLOORS[tier] for every turn in that tier. Decoupled from
# workers.SLA_TARGETS (which is worker monitoring, not routing constraint).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParetoPick:
    """Result of select_cheapest_meeting_floor — worker + TCO + diagnostic meta."""
    worker: str
    tco: TCOBreakdown
    floor: float
    best_quality: Optional[float]  # best observed quality among pool, None if no data
    floor_relaxed: bool  # True if fallback path taken (no worker met floor)


# ---------------------------------------------------------------------------
# Per-tier quality floor (hard constraint). Routing must satisfy
# TIER_FLOORS[tier] for every turn in that tier. Decoupled from
# workers.SLA_TARGETS (which is worker monitoring, not routing constraint).
# ---------------------------------------------------------------------------
TIER_FLOORS: dict[str, float] = {
    "auto": 0.70,
    "basic": 0.70,     # alias → auto (compat)
    "premium": 0.70,   # alias → auto
    "ultra": 0.70,     # alias → auto
}


# ---------------------------------------------------------------------------
# TCO breakdown
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TCOBreakdown:
    input_cost_yuan: float
    output_cost_yuan: float
    retry_cost_yuan: float
    escalation_cost_yuan: float
    judge_cost_yuan: float
    amortized_fixed_cost_yuan: float  # monthly_fixed_cny / expected_monthly_queries
    total_yuan: float

    @classmethod
    def from_components(
        cls,
        *,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        retry_cost: float = 0.0,
        escalation_cost: float = 0.0,
        judge_cost: float = 0.0,
        amortized_fixed_cost: float = 0.0,
    ) -> "TCOBreakdown":
        return cls(
            input_cost_yuan=float(input_cost),
            output_cost_yuan=float(output_cost),
            retry_cost_yuan=float(retry_cost),
            escalation_cost_yuan=float(escalation_cost),
            judge_cost_yuan=float(judge_cost),
            amortized_fixed_cost_yuan=float(amortized_fixed_cost),
            total_yuan=float(
                input_cost + output_cost + retry_cost + escalation_cost
                + judge_cost + amortized_fixed_cost
            ),
        )


# Opus 4.8 judge cost approximation. Real cost is input/output token-weighted;
# 0.30 is a reasonable upper bound for a typical 50-token verdict call.
JUDGE_COST_YUAN: float = 0.30

# v0.9.45-fix (P1-B 2026-07-12): Hard minimum quality — no worker with quality
# below this threshold is ever selected, regardless of floor relaxation, cold-start
# cheapest fallback, or pick_worker_by_tier's 15% slack. Protects against
# catastrophically degraded workers like claude-sonnet-5 (hard quality=0.20)
# from being silently routed to. 0.40 is chosen to be below any reasonable
# floor (basic=0.60) so normal routing is never affected, but above the 0.20
# range that corresponds to near-total error rates.
MIN_QUALITY_HARD_FLOOR: float = 0.40

# Runtime cost-weight multiplier (v0.9.45 P2.6): read from
# data/calibration/recommended_lambda.json if exists. Multiplier > 1 means
# pareto_curve recommended we save more cost; < 1 means favor quality.
RUNTIME_LAMBDA_CONFIG = Path(
    os.environ.get(
        "ANCHOR_RUNTIME_LAMBDA_CONFIG",
        str(_ROOT / "data" / "calibration" / "recommended_lambda.json"),
    )
)
_DEFAULT_COST_WEIGHT: float = 1.0
_runtime_cost_weight_cache: dict = {}


def _runtime_cost_weight() -> float:
    """Read recommended_lambda.json -> derive cost-weight multiplier.

    lambda=0.05 -> weight=0.5 (very quality-focused, cheap route)
    lambda=0.5 -> weight=1.0 (balanced)
    lambda=2.0 -> weight=2.0 (cost-focused, more saving)
    lambda=5.0 -> weight=3.0 (extreme cost save)
    """
    if "value" in _runtime_cost_weight_cache:
        return _runtime_cost_weight_cache["value"]
    if not RUNTIME_LAMBDA_CONFIG.exists():
        _runtime_cost_weight_cache["value"] = _DEFAULT_COST_WEIGHT
        return _DEFAULT_COST_WEIGHT
    try:
        data = json.loads(RUNTIME_LAMBDA_CONFIG.read_text())
        lam = float(data.get("lambda", 0.5))
        # Map lambda (cost-vs-quality tradeoff weight) to cost-multiplier.
        # lambda=0.0 (quality-only, no cost penalty) -> multiplier=1.0 (neutral)
        # lambda=0.5 (balanced)                      -> multiplier=1.5
        # lambda=5.0 (cost-only, heavy penalty)      -> multiplier=5.0
        # Linear mapping, clamped to [0.5, 5.0].
        multiplier = max(0.5, min(5.0, 1.0 + lam))
        _runtime_cost_weight_cache["value"] = multiplier
        return multiplier
    except Exception:
        _runtime_cost_weight_cache["value"] = _DEFAULT_COST_WEIGHT
        return _DEFAULT_COST_WEIGHT


# ---------------------------------------------------------------------------
# Quality table lookup
# ---------------------------------------------------------------------------
# audit 2026-08-16 (F1/C1): was quality_table_v7.json (2026-07-12) while the
# B6 data-driven tier pool, fallback ladder and head floors all use v8 — the
# live Pareto path was scoring candidates against a stale, removed-worker table.
DEFAULT_QUALITY_TABLE = _ROOT / "data/calibration/quality_table_v8.json"

_TABLE_CACHE: dict[str, dict] = {}
_TABLE_PATH_CACHE: Optional[Path] = None


def _load_table(path: Path) -> dict:
    global _TABLE_PATH_CACHE
    if _TABLE_PATH_CACHE == path and path in _TABLE_CACHE:
        return _TABLE_CACHE[path]
    with open(path) as f:
        data = json.load(f)
    _TABLE_CACHE[str(path)] = data
    _TABLE_PATH_CACHE = path
    return data


def quality_table_lookup(
    worker: str,
    category: str,
    difficulty: str,
    *,
    table_path: Optional[Path] = None,
) -> Optional[float]:
    """Look up (worker, category, difficulty) cell score.

    Fallbacks: cell -> category/<difficulty> -> unknown/<difficulty> -> None.
    """
    path = Path(table_path) if table_path else DEFAULT_QUALITY_TABLE
    if not path.exists():
        return None
    data = _load_table(path)
    table = data.get("table_per_cat", {})
    cats = table.get(worker) or {}
    cells = cats.get(category) or {}
    if difficulty in cells:
        return float(cells[difficulty])
    # Coarse fallback: any category -> first hit at this difficulty
    for cat_cells in cats.values():
        if difficulty in cat_cells:
            return float(cat_cells[difficulty])
    # Final fallback: unknown/<difficulty>
    unk = cats.get("unknown") or {}
    if difficulty in unk:
        return float(unk[difficulty])
    return None


# ---------------------------------------------------------------------------
# Query -> (category, difficulty) mapping
# ---------------------------------------------------------------------------
_CATEGORIES = (
    "code", "debug", "cn", "en", "math", "chat", "creative", "reasoning",
    "vision", "agent", "summarize", "translate", "design", "unknown",
)
_DIFFICULTIES = ("easy", "medium", "hard")


def parse_query_to_cell(query: str, query_type: str) -> tuple[str, str, bool]:
    """Map (query, query_type) -> (category, difficulty, is_short_chat).

    is_short_chat: True when query is short chat (<= 30 chars, easy, chat)
    AND category resolves to 'chat'. Used by select_pareto to enforce
    minimum worker differentiation across tiers (premium/ultra deserve
    different workers than basic, even for trivial chat).

    Difficulty from query_complexity: 0.0-0.3 -> easy, 0.3-0.6 -> medium,
    0.6-1.0 -> hard. Category from classify() unless design-flavored.
    """
    c = query_complexity(query)
    if c >= 0.6:
        diff = "hard"
    elif c >= 0.3:
        diff = "medium"
    else:
        diff = "easy"
    category = query_type if query_type in _CATEGORIES else classify(query or "")
    if category not in _CATEGORIES:
        category = "unknown"
    is_short_chat = (
        category == "chat" and diff == "easy" and len(query or "") <= 30
    )
    return category, diff, is_short_chat


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
_OUT_TOKEN_EST: dict[str, int] = {
    "easy": 200, "medium": 600, "hard": 1500,
    "code": 800, "debug": 400, "vision": 100,
}


def _estimate_prompt_tokens(query: str) -> int:
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", query or ""))
    return int(len(query or "") / (1.5 if has_cjk else 3.0)) + 50


def _estimate_output_tokens(difficulty: str, query_type: str) -> int:
    if query_type in _OUT_TOKEN_EST:
        return _OUT_TOKEN_EST[query_type]
    return _OUT_TOKEN_EST.get(difficulty, 600)


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------
def _get_worker(name: str):
    for w in WORKERS:
        if w.name == name:
            return w
    return None


def _quota_fraction(worker_name: str) -> float:
    """Quota share for shared-quota workers (baosiapi 4-way split).

    baosiapi workers share ¥688/mo; each gets 1/N of the enabled set.
    Disabled workers get 0.0 (they do not consume shared quota).
    Other workers get 1.0 (full ownership of their fixed cost).
    """
    try:
        from anchor.config import WORKERS
        w = next((x for x in WORKERS if x.name == worker_name), None)
        if w is None:
            return 1.0
        if not w.enabled:
            return 0.0
        # baosiapi workers: distribute ¥688 evenly across enabled ones
        if getattr(w, "channel", "") == "baosiapi":
            baosiapi_enabled = [
                x for x in WORKERS
                if x.enabled and getattr(x, "channel", "") == "baosiapi"
            ]
            n = max(1, len(baosiapi_enabled))
            return 1.0 / n
        return 1.0
    except Exception:
        return 1.0


def _get_fallback(worker_name: str):
    """Walk tier pools (basic → premium) for a non-quarantined fallback.

    Used for retry cost estimation in TCO computation. Returns the first
    enabled worker from basic tier (excluding *worker_name*), falling back
    to premium tier, then to deepseek-v4-flash as last resort.  Skips
    workers currently under circuit-breaker quarantine.
    """
    from anchor.release.circuit_breaker import is_quarantined
    for tier in ("auto",):
        pool = TIER_POOL.get(tier, ())
        for w_name in pool:
            if w_name != worker_name:
                w = _get_worker(w_name)
                if w is not None and not is_quarantined(w_name):
                    return w
    return _get_worker("deepseek-v4-flash")


# ---------------------------------------------------------------------------
# TCO computation
# ---------------------------------------------------------------------------
def compute_tco(
    worker,
    prompt_tokens: int,
    expected_output_tokens: int,
    *,
    p_retry: float = 0.0,
    retry_worker=None,
    p_escalate: float = 0.0,
    escalated_worker=None,
    judge_calls: int = 0,
    expected_monthly_queries: Optional[float] = None,
    quota_fraction: float = 1.0,
) -> TCOBreakdown:
    """Compute TCO breakdown for a worker choice.

    p_retry * retry_worker.total models expected retry cost.
    p_escalate * escalated_worker.total models expected escalation cost.
    judge_calls * token-aware estimator models Opus judge budget.
    expected_monthly_queries / quota_fraction:
      - monthly_fixed_cny (from Worker config) / expected_monthly_queries
        = amortized per-query fixed cost
      - quota_fraction handles shared quotas (e.g. baosiapi ¥688 split
        4 ways -> 0.25 per worker).
    """
    input_cost = (prompt_tokens / 1e6) * worker.cost_in
    output_cost = (expected_output_tokens / 1e6) * worker.cost_out
    retry_cost = 0.0
    if p_retry > 0 and retry_worker is not None:
        rw_cost = (prompt_tokens / 1e6) * retry_worker.cost_in + (
            expected_output_tokens / 1e6
        ) * retry_worker.cost_out
        retry_cost = p_retry * rw_cost
    escalation_cost = 0.0
    if p_escalate > 0 and escalated_worker is not None:
        ew_cost = (prompt_tokens / 1e6) * escalated_worker.cost_in + (
            expected_output_tokens / 1e6
        ) * escalated_worker.cost_out
        escalation_cost = p_escalate * ew_cost
    # Judge cost: token-aware (default 50 in + 5 out tokens per verdict).
    judge_cost = judge_calls * _estimate_judge_cost(worker)
    # Amortized fixed cost (Worker dataclass has .monthly_fixed_cny field).
    amortized = 0.0
    if expected_monthly_queries and expected_monthly_queries > 0:
        monthly_fixed = getattr(worker, "monthly_fixed_cny", 0.0) or 0.0
        if monthly_fixed > 0 and quota_fraction > 0:
            amortized = (monthly_fixed * quota_fraction) / expected_monthly_queries
    # Apply runtime cost-weight multiplier (P2.6).
    weight = _runtime_cost_weight()
    if weight != 1.0:
        input_cost *= weight
        output_cost *= weight
        retry_cost *= weight
        escalation_cost *= weight
        judge_cost *= weight
        amortized *= weight
    return TCOBreakdown.from_components(
        input_cost=input_cost,
        output_cost=output_cost,
        retry_cost=retry_cost,
        escalation_cost=escalation_cost,
        judge_cost=judge_cost,
        amortized_fixed_cost=amortized,
    )


def judge_cost_lookup(worker_name: str) -> tuple[float, float]:
    """Look up actual judge worker cost from WORKERS.cost_in / cost_out.

    Returns (input_yuan_per_M, output_yuan_per_M) for the named worker.
    Falls back to opus-5 rates (¥1.5831 / ¥7.9154 per 1M tokens) when the
    worker is unknown — the previous hardcoded ¥15/¥75 over-estimated by
    ~10x and biased the router away from using judges.

    The lookup is independent of the worker's per-call ``WORKER_COST_YUAN``
    entry, which is calibrated per median turn rather than per token.
    """
    for w in WORKERS:
        if w.name == worker_name:
            return w.cost_in, w.cost_out
    # Fallback: opus-5 actual rates (¥/M). Matches the most-expensive common
    # judge; previously this branch silently used the 10x-inflated ¥15/¥75.
    return 1.5831, 7.9154


def _estimate_judge_cost(worker) -> float:
    """Token-aware judge cost using the worker's actual rate (v0.9.53 Stage1-A).

    Default verdict: 50 prompt tokens + 5 output tokens. Looks up the
    worker-specific cost via :func:`judge_cost_lookup`; falls back to
    opus-5 rates when the worker is unknown. Replaces the previous
    hardcoded ¥15/¥75 (over-estimated by ~10x).
    """
    judge_in_tok = 50
    judge_out_tok = 5
    cost_in, cost_out = judge_cost_lookup(worker.name)
    return (judge_in_tok / 1e6) * cost_in + (judge_out_tok / 1e6) * cost_out


# ---------------------------------------------------------------------------
# Pareto selection: cheapest worker >= quality floor
# ---------------------------------------------------------------------------
class NoWorkerMeetsFloor(Exception):
    """No worker in tier pool satisfies quality floor."""


class ColdStartFallback(Exception):
    """No quality_table data exists; using cheapest fallback."""


def select_pareto(
    query: str,
    query_type: str,
    tier: str,
    *,
    pool: Optional[list[str]] = None,
    table_path: Optional[Path] = None,
    expected_monthly_queries: Optional[float] = None,
    p_escalate: float = 0.0,
) -> ParetoPick:
    """Pick cheapest worker in pool where expected_quality >= TIER_FLOORS[tier].

    Returns ParetoPick with worker + tco + metadata. Caller can inspect
    pick.floor_relaxed to detect when no worker met the floor (TCO-min
    constraint violated) — useful for surfacing quality degradation to
    callers, dashboards, or triggering judge_batch escalation.

    Fallback semantics:
      - cold-start (no quality_table data for any pool worker):
          return cheapest in pool, floor_relaxed=True
      - no worker meets floor (data exists but insufficient):
          return cheapest in pool, floor_relaxed=True, best_quality=highest observed
    """
    floor = TIER_FLOORS.get(tier, 0.60)
    if pool is None:
        # v0.9.47 (P1-B): TIER_POOL now stores worker name strings, not indices.
        from anchor.head import TIER_POOL
        idx_pool = TIER_POOL.get(tier, TIER_POOL["auto"])
        pool = list(idx_pool)

    if not pool:
        # Cold-start: tier pool empty -> use cheapest enabled worker
        import logging as _lg_cs
        cheapest = min(
            (w for w in enabled_workers()),
            key=lambda w: w.cost_out,
            default=None,
        )
        if cheapest is None:
            _lg_cs.error("PARETO_COLDSTART tier=%s reason=no_enabled_workers", tier)
            raise ColdStartFallback("no enabled workers")
        _lg_cs.warning(
            "PARETO_COLDSTART tier=%s worker=%s reason=empty_pool query=%s",
            tier, cheapest.name, (query or "")[:60],
        )
        return ParetoPick(
            worker=cheapest.name,
            tco=compute_tco(cheapest, _estimate_prompt_tokens(query), 500),
            floor=floor,
            best_quality=None,
            floor_relaxed=True,
        )

    category, difficulty, _is_short_chat = parse_query_to_cell(query, query_type)
    prompt_tokens = _estimate_prompt_tokens(query)

    # v0.9.46: pre-compute escalation worker (next tier up) once per call.
    # Used by compute_tco to model expected escalation_cost when p_escalate > 0.
    # v0.9.47 (P1-B): TIER_POOL stores worker name strings (not indices). The
    # cost-premium for escalation is the MAX cost_out among next-tier-up workers.
    _esc_worker = None
    if p_escalate > 0:
        from anchor.head import TIER_POOL as _TP
        # pure mode: single auto pool — frontier = max cost worker in pool
        _candidates_esc = []
        for _w_name in (_TP.get("auto") or ()):
            if not isinstance(_w_name, str):
                continue
            _w = _get_worker(_w_name)
            if _w is not None:
                _candidates_esc.append(_w)
        if _candidates_esc:
            # Frontier = highest per-token cost (proxy for capability).
            _esc_worker = max(_candidates_esc, key=lambda w: w.cost_in + w.cost_out)
    candidates: list[tuple[str, TCOBreakdown]] = []
    any_data = False
    for worker_name in pool:
        worker = _get_worker(worker_name)
        if worker is None:
            continue
        q_score = _lookup_with_overlay(worker_name, category, difficulty)
        if q_score is None:
            q_score = quality_table_lookup(worker_name, category, difficulty, table_path=table_path)
        if q_score is None:
            continue
        any_data = True
        # P1-B: Absolute minimum quality gate — reject workers in catastrophic
        # degradation (e.g. sonnet-5 hard=0.20) before floor comparison.
        if q_score < MIN_QUALITY_HARD_FLOOR:
            import logging as _lg_qmin
            _lg_qmin.warning(
                "PARETO_QUALITY_GATE worker=%s tier=%s category=%s difficulty=%s "
                "q_score=%.3f < MIN_QUALITY_HARD_FLOOR=%.2f — excluded",
                worker_name, tier, category, difficulty, q_score, MIN_QUALITY_HARD_FLOOR,
            )
            continue
        # v0.9.53 Phase 2: Conformal Prediction lower-bound check. When
        # CP is enabled and the cell has enough calibration samples, use
        # the CP lower bound (a (1-alpha)-coverage lower bound on true
        # quality) instead of the point estimate. This excludes workers
        # with high mean but high variance that are likely to dip below
        # the floor in practice. ANCHOR_CP_ENABLED=0 (default) keeps the
        # legacy point-estimate check unchanged.
        if CP_ENABLED:
            _cp_mean, cp_lower, _cp_upper = _cp_get_interval(worker_name, category, difficulty)
            # Cold-start cells have the maximally-conservative sentinel
            # (mean=0.5, lower=0.0, upper=1.0); skip CP in that case and
            # fall back to the point estimate.
            if not (_cp_mean == 0.5 and cp_lower == 0.0 and _cp_upper == 1.0):
                if cp_lower < floor:
                    import logging as _lg_cp
                    _lg_cp.info(
                        "PARETO_CP_EXCLUDED worker=%s tier=%s category=%s difficulty=%s "
                        "cp_lower=%.3f < floor=%.2f (point_mean=%.3f)",
                        worker_name, tier, category, difficulty, cp_lower, floor, q_score,
                    )
                    continue
        if q_score < floor:
            continue
        expected_out = _estimate_output_tokens(difficulty, query_type)
        p_retry = max(0.0, 1.0 - q_score)
        retry_worker = _get_fallback(worker_name)
        judge_calls = 1 if p_retry > 0.3 else 0
        tco = compute_tco(
            worker, prompt_tokens, expected_out,
            p_retry=p_retry, retry_worker=retry_worker,
            p_escalate=p_escalate, escalated_worker=_esc_worker,
            judge_calls=judge_calls,
            expected_monthly_queries=expected_monthly_queries,
            quota_fraction=_quota_fraction(worker_name),
        )
        candidates.append((worker_name, tco))

    # Track best quality observed for metadata
    best_quality: Optional[float] = None
    if candidates:
        # candidates list rebuilt for quality audit; cheap (we already computed)
        pass
    # Need to re-scan for best_quality across all pool members (including those below floor)
    for wn in pool:
        _q = _lookup_with_overlay(wn, category, difficulty)
        if _q is None:
            _q = quality_table_lookup(wn, category, difficulty, table_path=table_path)
        if _q is not None and (best_quality is None or _q > best_quality):
            best_quality = _q

    if not candidates:
        if not any_data:
            # Cold-start: no quality_table coverage at all
            import logging as _lg_cs2
            cheapest = min(
                (w for w in pool if _get_worker(w)),
                key=lambda n: _get_worker(n).cost_out,
                default=None,
            )
            if cheapest is None:
                _lg_cs2.error(
                    "PARETO_COLDSTART tier=%s category=%s difficulty=%s reason=no_table_data_and_no_pool",
                    tier, category, difficulty,
                )
                raise ColdStartFallback("no enabled workers in pool")
            _lg_cs2.warning(
                "PARETO_COLDSTART tier=%s category=%s difficulty=%s worker=%s reason=no_table_data query=%s",
                tier, category, difficulty, cheapest, (query or "")[:60],
            )
            return ParetoPick(
                worker=cheapest,
                tco=compute_tco(_get_worker(cheapest), prompt_tokens, 500),
                floor=floor,
                best_quality=best_quality,
                floor_relaxed=True,
            )
        # v0.9.53 (P0-5): floor too tight but data exists. Pick the worker
        # with the BEST available quality (not the cheapest). The cheapest
        # worker is typically the lowest quality, which is exactly the wrong
        # choice when quality is the binding constraint. Only workers above
        # MIN_QUALITY_HARD_FLOOR (catastrophic-degradation gate) are eligible.
        emergency_candidates: list[tuple[str, float]] = []
        for wn in pool:
            worker = _get_worker(wn)
            if worker is None:
                continue
            q_score = _lookup_with_overlay(wn, category, difficulty)
            if q_score is None:
                q_score = quality_table_lookup(wn, category, difficulty, table_path=table_path)
            if q_score is None:
                continue
            if q_score < MIN_QUALITY_HARD_FLOOR:
                continue
            emergency_candidates.append((wn, float(q_score)))
        if emergency_candidates:
            # Highest quality first; tie-break by lowest cost_out
            emergency_worker, emergency_quality = max(
                emergency_candidates,
                key=lambda x: (x[1], -(_get_worker(x[0]).cost_out or 0.0)),
            )
            import logging as _lg_fr
            _lg_fr.warning(
                "PARETO_FLOOR_RELAXED tier=%s category=%s difficulty=%s "
                "worker=%s quality=%.3f reason=no_worker_met_floor_%.2f",
                tier, category, difficulty, emergency_worker, emergency_quality, floor,
            )
            return ParetoPick(
                worker=emergency_worker,
                tco=compute_tco(_get_worker(emergency_worker), prompt_tokens, 500),
                floor=floor,
                best_quality=best_quality,
                floor_relaxed=True,
            )
        # No worker even clears MIN_QUALITY_HARD_FLOOR — fall through to
        # cheapest (legacy last resort, should be rare in practice).
        cheapest = min(
            (w for w in pool if _get_worker(w)),
            key=lambda n: _get_worker(n).cost_out,
            default=None,
        )
        if cheapest is None:
            raise NoWorkerMeetsFloor(f"no worker in pool {pool} meets floor {floor}")
        return ParetoPick(
            worker=cheapest,
            tco=compute_tco(_get_worker(cheapest), prompt_tokens, 500),
            floor=floor,
            best_quality=best_quality,
            floor_relaxed=True,
        )

    # Tie-break: lower total_yuan, then lower cost_out, then preserve pool order.
    pool_index = {n: i for i, n in enumerate(pool)}
    def _sort_key(item):
        name, tco = item
        worker = _get_worker(name)
        cost_out = worker.cost_out if worker else 1e9
        return (tco.total_yuan, cost_out, pool_index.get(name, 1e6))
    best_name, best_tco = min(candidates, key=_sort_key)
    return ParetoPick(
        worker=best_name,
        tco=best_tco,
        floor=floor,
        best_quality=best_quality,
        floor_relaxed=False,
    )


def select_cheapest_meeting_floor(
    query: str,
    query_type: str,
    tier: str,
    *,
    pool: Optional[list[str]] = None,
    table_path: Optional[Path] = None,
    expected_monthly_queries: Optional[float] = None,
    p_escalate: float = 0.0,
) -> tuple[str, TCOBreakdown]:
    """Backwards-compatible wrapper: returns (worker, tco) tuple.

    For new code, prefer select_pareto() which returns ParetoPick with
    floor_relaxed metadata. This wrapper discards that metadata.
    """
    pick = select_pareto(
        query, query_type, tier,
        pool=pool, table_path=table_path,
        expected_monthly_queries=expected_monthly_queries,
        p_escalate=p_escalate,
    )
    return pick.worker, pick.tco


__all__ = [
    "TIER_FLOORS",
    "TCOBreakdown",
    "ParetoPick",
    "JUDGE_COST_YUAN",
    "compute_tco",
    "quality_table_lookup",
    "parse_query_to_cell",
    "select_cheapest_meeting_floor",
    "select_pareto",
    "NoWorkerMeetsFloor",
    "ColdStartFallback",
]
