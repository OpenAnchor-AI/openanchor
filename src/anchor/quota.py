"""v1.1: Multi-pool monthly quota tracker with real resource boundaries.

Pool A: M3 (¥119/mo fixed, ~18B tokens) — primary T1
Pool B-Claude: baosiapi Claude group ($13k shared) — sonnet/fable/opus/haiku
Pool B-GPT: baosiapi-gpt independent quota — gpt-5.6-*
Pool B-Grok: baosiapi-grok independent quota
Pool B-Kimi: baosiapi-kimi independent quota
Pool C: DeepSeek V4 Pro (pay-per-token) — independent insurance
Pool D: deepseek-v4-flash (free) — retry/batch

Usage:
    from anchor.quota import get_quota_tracker
    tracker = get_quota_tracker()
    if tracker.should_fallback_to_c():
        # fallback to Pool C when enabled
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from anchor import usage as _usage
from anchor.config import WORKERS as _CFG_WORKERS

_logger = logging.getLogger("anchor.quota")

# Claude group shared USD budget (baosiapi channel exact match only)
_POOL_B_CLAUDE_BUDGET_USD = float(os.environ.get("ANCHOR_POOL_B_CLAUDE_BUDGET_USD", "13000"))
_POOL_B_GPT_BUDGET_USD = float(os.environ.get("ANCHOR_POOL_B_GPT_BUDGET_USD", "13000"))
_POOL_B_GROK_BUDGET_USD = float(os.environ.get("ANCHOR_POOL_B_GROK_BUDGET_USD", "5000"))
_POOL_B_KIMI_BUDGET_USD = float(os.environ.get("ANCHOR_POOL_B_KIMI_BUDGET_USD", "5000"))

# Exact channel -> budget (do NOT use startswith("baosiapi"))
_BAOSI_POOLS: dict[str, float] = {
    "baosiapi": _POOL_B_CLAUDE_BUDGET_USD,
    "baosiapi-gpt": _POOL_B_GPT_BUDGET_USD,
    "baosiapi-grok": _POOL_B_GROK_BUDGET_USD,
    "baosiapi-kimi": _POOL_B_KIMI_BUDGET_USD,
}

_POOL_C_WORKER = "deepseek-v4-pro"
_POOL_A_WORKER = "minimax-m3"


class QuotaTracker:
    """Track monthly usage against independent financial pools."""

    def __init__(self):
        # Default 18B tokens/month (¥119/mo subscription, v0.9.50-p0)
        self._m3_monthly_token_quota = int(os.environ.get(
            "ANCHOR_M3_MONTHLY_TOKEN_QUOTA", "18000000000",
        ))
        self._fallback_threshold = float(os.environ.get(
            "ANCHOR_A_POOL_FALLBACK_THRESHOLD", "0.80",
        ))
        self._usd_cny_rate = float(os.environ.get(
            "ANCHOR_USD_CNY_RATE", "7.2",
        ))
        self._warn_threshold = float(os.environ.get(
            "ANCHOR_POOL_WARN_THRESHOLD", "0.80",
        ))

    def _channel_map(self) -> dict[str, str]:
        return {w.name: w.channel for w in _CFG_WORKERS}

    def _enabled_names(self) -> set[str]:
        return {w.name for w in _CFG_WORKERS if w.enabled}

    def pool_a_m3_used_pct(self) -> float:
        """Pool A (M3) usage as fraction of monthly token quota."""
        usage = _usage.month_usage_by_worker()
        m3 = usage.get(_POOL_A_WORKER, {})
        total_tokens = m3.get("input_tokens", 0) + m3.get("output_tokens", 0)
        if self._m3_monthly_token_quota <= 0:
            return 0.0
        return total_tokens / self._m3_monthly_token_quota

    def pool_usage_usd(self, channel: str) -> float:
        """USD used for an exact channel resource pool."""
        usage = _usage.month_usage_by_worker()
        channel_map = self._channel_map()
        total_yuan = sum(
            stats.get("cost_yuan", 0.0)
            for worker_name, stats in usage.items()
            if channel_map.get(worker_name) == channel
        )
        if self._usd_cny_rate <= 0:
            return 0.0
        return total_yuan / self._usd_cny_rate

    def pool_b_baosi_used_usd(self) -> float:
        """Backward-compat: Claude group (channel == 'baosiapi') only.

        Does NOT include baosiapi-gpt / grok / kimi (independent quotas).
        """
        return self.pool_usage_usd("baosiapi")

    def pool_b_gpt_used_usd(self) -> float:
        return self.pool_usage_usd("baosiapi-gpt")

    def pool_b_grok_used_usd(self) -> float:
        return self.pool_usage_usd("baosiapi-grok")

    def pool_b_kimi_used_usd(self) -> float:
        return self.pool_usage_usd("baosiapi-kimi")

    def baosi_pool_status(self) -> list[tuple[str, float, float]]:
        """List of (channel, used_usd, budget_usd) for each baosi pool."""
        return [
            (ch, self.pool_usage_usd(ch), budget)
            for ch, budget in _BAOSI_POOLS.items()
        ]

    def pool_c_dpsk_pro_total_yuan(self) -> float:
        usage = _usage.month_usage_by_worker()
        dpsk_pro = usage.get(_POOL_C_WORKER, {})
        return float(dpsk_pro.get("cost_yuan", 0.0))

    def should_fallback_to_c(self) -> bool:
        """True if Pool A usage exceeds threshold (default 80%)."""
        return self.pool_a_m3_used_pct() > self._fallback_threshold

    def should_warn_b_pool(self) -> bool:
        """True if Claude Pool B exceeds warn threshold of its budget."""
        budget = _BAOSI_POOLS["baosiapi"]
        return self.pool_b_baosi_used_usd() > self._warn_threshold * budget

    def warned_baosi_pools(self) -> list[str]:
        """Channels that exceed the warn threshold."""
        out = []
        for ch, used, budget in self.baosi_pool_status():
            if budget > 0 and used > self._warn_threshold * budget:
                out.append(ch)
        return out

    def pool_c_available(self, tier_pool: Iterable[str] | None = None) -> bool:
        """Pool C is usable only when worker is enabled and (optionally) in pool."""
        if _POOL_C_WORKER not in self._enabled_names():
            return False
        if tier_pool is None:
            return True
        return _POOL_C_WORKER in set(tier_pool)

    def record_quota_check(self) -> None:
        pct = self.pool_a_m3_used_pct()
        claude = self.pool_b_baosi_used_usd()
        gpt = self.pool_b_gpt_used_usd()
        _logger.info(
            "QUOTA_CHECK pool_A=%.1f%% pool_B_claude=$%.0f pool_B_gpt=$%.0f pool_C=¥%.2f",
            pct * 100, claude, gpt, self.pool_c_dpsk_pro_total_yuan(),
        )


_tracker: QuotaTracker | None = None


def get_quota_tracker() -> QuotaTracker:
    global _tracker
    if _tracker is None:
        _tracker = QuotaTracker()
    return _tracker


def maybe_quota_fallback(chosen: str, tier: str, tier_pool: list[str]) -> str:
    """Pool A exhaustion hook: only fall back to real Pool C when available.

    When Pool A (M3) exceeds threshold and tier in (basic, premium):
      - If deepseek-v4-pro is enabled and present in tier_pool → switch to it
      - Otherwise keep M3 (do NOT silently dump to Pool D free flash)

    Also logs per-channel B-pool warnings when any baosi* budget exceeds threshold.

    Returns the (possibly modified) chosen worker name.
    """
    tracker = get_quota_tracker()

    for ch in tracker.warned_baosi_pools():
        used = tracker.pool_usage_usd(ch)
        budget = _BAOSI_POOLS.get(ch, 0.0)
        _logger.warning(
            "B_POOL_WARNING channel=%s usage=$%.0f / $%.0f",
            ch, used, budget,
        )

    # v0.9.52: single product lane `auto` (legacy names still accepted)
    if tier not in ("auto", "basic", "premium", "ultra"):
        return chosen
    if chosen != _POOL_A_WORKER:
        return chosen
    if not tracker.should_fallback_to_c():
        return chosen

    if tracker.pool_c_available(tier_pool):
        pct = tracker.pool_a_m3_used_pct()
        _logger.warning(
            "QUOTA_FALLBACK pool=A->C usage=%.1f%% tier=%s from=%s to=%s",
            pct * 100, tier, chosen, _POOL_C_WORKER,
        )
        return _POOL_C_WORKER

    pct = tracker.pool_a_m3_used_pct()
    _logger.warning(
        "QUOTA_FALLBACK_SKIPPED pool_C_unavailable usage=%.1f%% tier=%s kept=%s",
        pct * 100, tier, chosen,
    )
    return chosen
