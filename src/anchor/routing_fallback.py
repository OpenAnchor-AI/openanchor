"""Fallback ladder helpers (extracted from routing_core).

Tables + candidate walk. `_fallback_candidates` reads
`routing_core._FALLBACK_FAST_PATH` at call time so unit tests can
monkeypatch the path on the routing_core module.

v0.9.53 (P0-4 fix): the `_FALLBACK_FAST_PATH` is now derived from
`config.WORKER_FALLBACK_CHAIN` (canonical) so all three fallback graphs
(worker chain, name next-hop, tier fast-path) share one source of truth.
"""
from __future__ import annotations

import os

from anchor.worker_gate import _truthy as _env_truthy


def _max_fallback_attempts() -> int:
    """Max retry attempts in fallback chain before giving up.

    Configurable via ANCHOR_MAX_FALLBACK_ATTEMPTS env var (default 2).
    """
    try:
        return max(0, int(os.environ.get("ANCHOR_MAX_FALLBACK_ATTEMPTS", "2")))
    except ValueError:
        return 2


_DEEPSEEK_PRO_OPT_IN = _env_truthy(os.environ.get("ANCHOR_ENABLE_DEEPSEEK_PRO"))


def _maybe_pro(pool: tuple) -> tuple:
    """Append deepseek-v4-pro as last-resort insurance (v0.9.54).

    deepseek-v4-pro is opt-in (ANCHOR_ENABLE_DEEPSEEK_PRO=1, default OFF).
    When enabled, it sits at the END of the fallback chain — only reached
    when all baosiapi workers are exhausted. This replaces the old behavior
    that placed it immediately after minimax-m3.
    """
    pool = list(pool)
    if _DEEPSEEK_PRO_OPT_IN:
        if "deepseek-v4-pro" not in pool:
            pool.append("deepseek-v4-pro")
    if (os.environ.get("BAOSIAPI_GPT_API_KEY") or os.environ.get("BAOSI_GPT_API_KEY")) and "gpt-5.6-sol" not in pool:
        pool.append("gpt-5.6-sol")
    return tuple(pool)


# Vision-capable worker set used to constrain vision queries.
_VISION_CAPABLE: frozenset[str] = frozenset({
    "minimax-m3",
    "claude-fable-5",
    "claude-haiku-4-5",
    "claude-sonnet-5",
})


def _eligible_workers(bucket, enabled_names):
    """Phase 3 FIX-B eligibility: v8 table_coarse[bucket] >= 0.7 AND err% < 10%."""
    from anchor.config import _load_cal_v8
    cal = _load_cal_v8()
    if not cal:
        return []
    coarse = cal.get("table_coarse", {}) or {}
    err_pct = cal.get("err_pct") or {}
    eligible = []
    for name, cell in coarse.items():
        if name not in enabled_names:
            continue
        score = cell.get(bucket)
        if score is None or score < 0.7:
            continue
        e = err_pct.get(name)
        if e is not None and e >= 10.0:
            continue
        eligible.append((name, score))
    return eligible


def build_fallback_fast_path():
    """Lean single-product fallback ladder; legacy tier keys alias to auto.

    Phase 3 FIX-B: data-driven. Within each tier, workers are ordered
    quality-aware (higher calibrated quality first, ties broken by
    worker_cost ascending). Vision queries get the vision subset only.
    Unknown workers are EXCLUDED.
    """
    from anchor.config import WORKERS, WORKER_FALLBACK_CHAIN, worker_cost
    enabled_names = {w.name for w in WORKERS if w.enabled}
    bucket = "medium"
    eligible = _eligible_workers(bucket, enabled_names)
    eligible.sort(key=lambda ns: (-ns[1], worker_cost(ns[0]), ns[0]))
    ladder = tuple(name for name, _score in eligible)
    if not ladder:
        import logging as _lg
        _lg.warning("FIX-B: no eligible workers from v8; using legacy ladder.")
        # V6.3 audit 2026-08-15: legacy fallback ladder — uses grok-4-6 (replaced grok-4-5).
        # gpt-5.6-sol also excluded (V4.2 opt-out, Phase 3 FIX-E). Both removed from ladder;
        # auto pool already excludes them via enabled flags.
        ladder = (
            "deepseek-v4-flash",
            "minimax-m3",
            "grok-4-6-reasoning",     # V6.3: replaces grok-4-5 (slot 15 default ON)
            "claude-fable-5",
        )
    missing = [w for w in ladder if w not in WORKER_FALLBACK_CHAIN]
    if missing:
        raise RuntimeError(
            f"fallback fast-path references unknown workers: {missing}. "
            "Update WORKER_FALLBACK_CHAIN or the calibration table."
        )
    reachable = set(ladder)
    frontier = list(ladder)
    while frontier:
        w = frontier.pop()
        for nxt in WORKER_FALLBACK_CHAIN.get(w, []):
            if nxt not in enabled_names:
                continue
            if nxt not in reachable:
                reachable.add(nxt)
                frontier.append(nxt)
    unreachable = enabled_names - reachable
    if unreachable:
        import logging as _lg
        _lg.warning(
            "FIX-B: enabled worker(s) %s not reachable from data-driven ladder.",
            sorted(unreachable),
        )
    return {"auto": ladder}


# Default table (also re-bound on routing_core for monkeypatch).
_FALLBACK_FAST_PATH = build_fallback_fast_path()



def _fallback_candidates(tier: str, chosen: str, pool: list[str], vision_only: bool = False) -> list[str]:
    """Return fallback workers when `chosen` fails.

    Quality-first order: escalate up the ladder first (grok→sol→fable),
    then degrade (M3→flash). Hard misses must not free-lane first.
    """
    from anchor.fusion_modes import _VISION_CAPABLE as _VC
    try:
        from anchor import routing_core as _rc
        fast_path = getattr(_rc, "_FALLBACK_FAST_PATH", _FALLBACK_FAST_PATH)
    except Exception:
        fast_path = _FALLBACK_FAST_PATH
    ladder = list(fast_path.get(tier, ()) or fast_path.get("auto", ()))
    ordered: list[str] = []
    if chosen in ladder:
        idx = ladder.index(chosen)
        # audit 2026-08-16 (B1): the ladder is now quality-DESCENDING
        # (build_fallback_fast_path sorts by -calibrated_quality, ties by
        # cost asc), so ladder[0] = best. The old formula
        # `ladder[idx+1:] + reversed(ladder[:idx])` was written for the
        # legacy ascending ladder and tried the WORST worker first (a grok
        # failure burned both attempts on the two weakest paid workers).
        # Quality-first semantics: escalate to better workers first
        # (idx-1 → 0), then degrade to worse (idx+1 → end).
        ordered.extend(reversed(ladder[:idx]))   # escalate: better-quality peers
        ordered.extend(ladder[idx + 1 :])        # degrade: worse-quality peers
    else:
        # Hard miss (chosen not in ladder): full ladder in quality order.
        ordered.extend(ladder)
    ordered.extend(pool)
    seen = {chosen}
    candidates: list[str] = []
    for worker in ordered:
        if worker in seen:
            continue
        seen.add(worker)
        if vision_only and worker not in _VC:
            continue
        candidates.append(worker)
    return candidates


def _recent_sla_violations(worker_name: str, days: int = 1) -> list[str]:
    try:
        from anchor.admin_ops import compute_sla as _compute_sla
        worker = _compute_sla(days).get("workers", {}).get(worker_name, {})
    except Exception as _sla_e:
        # S11 (SRE audit): SLA lookup failure must not silently disable
        # quarantine; log so operators can see if compute_sla is broken.
        import logging as _lg_sla
        _lg_sla.warning("SLA_LOOKUP_SKIP worker=%s days=%s err=%s", worker_name, days, _sla_e)
        return []
    if worker.get("status") in ("violation", "quarantined"):
        return list(worker.get("violations") or [])
    return []
