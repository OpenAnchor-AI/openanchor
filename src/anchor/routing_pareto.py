"""Pareto selection + pool filters + outcome annotation (from routing_core)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastapi import Request

# --- _filter_quarantined_workers ---
def _filter_quarantined_workers(pool: list) -> list:
    """v0.9.45 (P0.2 SLA): remove workers quarantined by circuit_breaker.

    Returns original pool if quarantine check fails (preserves ship).
    """
    try:
        from anchor.release.circuit_breaker import is_quarantined as _is_q
        return [w for w in pool if not _is_q(w)]
    except Exception as _q_e:
        import logging as _lg_q
        _lg_q.warning("QUARANTINE_FILTER_SKIP %s", _q_e)
        return pool



# --- _is_tool_query + _filter_chat_only_workers (v0.9.46g) ---
_TOOL_QUERY_MARKERS = (
    # Explicit tool/function-call hints
    "use tool", "call tool", "use the function", "call the function",
    "function_call", "tool_call", "tool_use",
    # Agent step-by-step / chained execution
    " and then ", "step by step", "step 1", "step 2",
    "first do", "then do", "next do", "finally do",
    # Code execution context
    "def ", "class ", "import ", "function ", "traceback",
    "```python", "```javascript", "```typescript",
    # CN equivalents
    "工具", "调用", "函数", "执行", "然后", "接着", "第一步", "步骤",
    "设计", "架构", "实现", "推导", "证明", "分析", "比较", "评估",
)


def _is_tool_query(query: str) -> bool:
    """v0.9.46g: detect queries that need tool/function-call support.

    Agnes-2.0-flash and similar chat-only workers cannot do function
    calling. When the query is tool-shaped, we exclude them from the pool
    to avoid them being picked and failing silently.

    Heuristic: any tool/agent/code-marker substring present -> True.
    Conservative: short greetings ("hi", "thanks") won't match.
    """
    ql = (query or "").lower()
    return any(m in ql for m in _TOOL_QUERY_MARKERS)


def _filter_chat_only_workers(pool: list, query: str) -> list:
    """v0.9.46g: exclude chat-only workers when query needs tool use.

    Returns original pool unchanged if filter fails (preserves ship).
    """
    if not _is_tool_query(query):
        return pool
    try:
        from anchor.config import WORKERS
        _chat_only_names = {w.name for w in WORKERS if getattr(w, "chat_only", False)}
        _filtered = [w for w in pool
                     if (getattr(w, "name", w) if isinstance(w, str) is False
                         else w) not in _chat_only_names]
        # Handle the case where pool elements are strings (not Worker objects)
        if pool and isinstance(pool[0], str):
            _filtered = [w for w in pool if w not in _chat_only_names]
        if len(_filtered) < len(pool):
            import logging as _lg_co
            _lg_co.info(
                "CHAT_ONLY_FILTER removed=%s query=%s",
                sorted(set(pool) - set(_filtered)) if pool and isinstance(pool[0], str) else "see logs",
                (query or "")[:50],
            )
        return _filtered if _filtered else pool  # never empty (preserves ship)
    except Exception as _co_e:
        import logging as _lg_co2
        _lg_co2.warning("CHAT_ONLY_FILTER_SKIP %s", _co_e)
        return pool



# --- _resolve_pareto_worker ---
async def _resolve_pareto_worker(tier: str, q, request: Optional["Request"] = None) -> dict:
    """v0.9.36 (first-principles): Pareto selector wrapper for legacy tier endpoints.

    Returns dict {worker, tco, floor, best_quality, floor_relaxed} on success,
    or {"worker": None, "floor_relaxed": True} on failure (caller falls back
    to TIER_HARD_RULES + head.predict).

    Tier override policy: only override caller's tier when:
      (a) session_id is provided AND session cost warrants cap (>= ¥2 mid / >= ¥5 ultra)
      (b) parent_turn_id is provided (multi-turn reuse)
    Pure mode: normalize to single auto lane.
    """
    try:
        from anchor._query_complexity import query_complexity as _qcomp
        from anchor.fusion_modes import _auto_detect_query_type as _adt
        from anchor.pareto import select_pareto as _select_pareto
        _complexity = _qcomp(q.query or "")
        _parent_pid = None
        _session_id = None
        if request is not None:
            _parent_pid = request.headers.get("X-Parent-Turn-ID") or None
            _session_id = request.headers.get("X-Session-ID") or None
        # v0.9.52: single product lane. Session/parent no longer map to SKUs.
        from anchor.config import normalize_routing_lane as _nrl_eff
        effective_tier = _nrl_eff(tier)
        # Parent quality still informs p_escalate below (TCO), not product tier.
        if _parent_pid:
            try:
                from anchor.multiturn import next_tier as _mt_next_tier
                _tier_cw = _mt_next_tier(
                    _parent_pid, query_complexity_score=_complexity,
                    session_id=None, query=q.query or None,
                )
                effective_tier = _nrl_eff(_tier_cw)
            except Exception as _pte_e:
                import logging as _lg_pte
                _lg_pte.warning("PARENT_TURN_LOOKUP_SKIP pid=%s err=%s", _parent_pid, _pte_e)
        # v0.9.46 (p_escalate wire): estimate next-turn escalation probability
        # from parent quality_tier. error/degraded -> p_escalate=1.0
        # (next_tier will force ultra); fair -> 0.4; good/excellent -> 0.05.
        # Fed to compute_tco as expected cost of escalation.
        _p_escalate = 0.0
        if _parent_pid:
            try:
                from anchor.multiturn import _get_parent_turn as _gtbh
                _p_turn = _gtbh(_parent_pid) or {}
                _p_qt = (_p_turn or {}).get("quality_tier") or ""
                if _p_qt == "error":
                    _p_escalate = 1.0
                elif _p_qt == "degraded":
                    _p_escalate = 0.8
                elif _p_qt == "fair":
                    _p_escalate = 0.4
                elif _p_qt in ("good", "excellent"):
                    _p_escalate = 0.05
            except Exception as _gtbh_e:
                import logging as _lg_pe
                _lg_pe.getLogger("anchor.routing_core").warning(
                    "p_escalate lookup failed, defaulting to 0.0: %s", _gtbh_e)
        _qt_for_pareto = _adt(q.query or "")
        # v0.9.45 (P0.2): filter SLA-quarantined workers (sonnet-5 41.2% err_rate
        # -> auto-quarantine via circuit_breaker). The Pareto head pool is
        # reduced to non-quarantined workers; floor_relaxed still surfaces
        # if the remaining pool can't satisfy the tier floor.
        # v0.9.47 (P1-B): TIER_POOL now stores worker name strings, not indices.
        from anchor.head import TIER_POOL as _TIER_POOL_HELPER
        _idx_pool = _TIER_POOL_HELPER.get(effective_tier) or _TIER_POOL_HELPER.get("auto") or ()
        _raw_pool = list(_idx_pool)
        _filtered_pool = _filter_quarantined_workers(_raw_pool)
        # v0.9.46g: exclude chat-only workers (e.g. agnes-2.0-flash) when
        # the query needs tool/function-call support. They have no
        # function_calling and would silently fail. Free stable fallback
        # for the query remains dpsk / m3 / kilo.
        _filtered_pool = _filter_chat_only_workers(_filtered_pool, q.query or "")
        # v0.9.46 (P1.4 wire): plumb expected_monthly_queries so M3/baosiapi
        # monthly_fixed_cny is amortized into per-query TCO. Without this
        # the router systematically under-costs subscription workers and
        # over-prefers free ones.
        from anchor.config import ROUTING as _ROUTING_CFG
        _pick = _select_pareto(
            q.query or "", _qt_for_pareto, effective_tier,
            pool=_filtered_pool,
            expected_monthly_queries=_ROUTING_CFG.expected_monthly_queries,
            p_escalate=_p_escalate,
        )
        return {
            "worker": _pick.worker,
            "tco": _pick.tco,
            "floor": _pick.floor,
            "best_quality": _pick.best_quality,
            "floor_relaxed": _pick.floor_relaxed,
        }
    except Exception as _pareto_e:
        import logging as _lg_pp2
        _lg_pp2.warning("PARETO_LEGACY_SKIP %s", _pareto_e)
        return {"worker": None, "floor_relaxed": True, "best_quality": None}




# --- _annotate_pareto_meta ---
def _annotate_pareto_meta(resp, pareto_meta: dict) -> None:
    """Surface floor_relaxed + best_quality to callers via duck-type field.

    Anchored in _anchor_meta dict when available; otherwise attaches via
    setattr on the duck-typed Response object.
    """
    try:
        if pareto_meta.get("floor_relaxed"):
            _meta = getattr(resp, "_anchor_meta", None) or {}
            _meta["floor_relaxed"] = True
            _meta["best_quality"] = pareto_meta.get("best_quality")
            _meta["floor"] = pareto_meta.get("floor")
            try:
                resp._anchor_meta = _meta
            except Exception as _am_e:
                # S11 (SRE audit): duck-type setattr failure (Pydantic v2
                # doesn't allow arbitrary attrs); log and continue.
                import logging as _lg_am
                _lg_am.warning("ANCHOR_META_SETATTR_SKIP err=%s", _am_e)
            import logging as _lg_pf
            _lg_pf.warning(
                "FLOOR_RELAXED tier-floor=%.2f best_quality=%s -> fallback",
                pareto_meta.get("floor") or 0.0,
                pareto_meta.get("best_quality"),
            )
    except Exception as _ann_e:
        import logging as _lg_ann
        _lg_ann.warning("PARETO_ANNOTATE_SKIP %s", _ann_e)



