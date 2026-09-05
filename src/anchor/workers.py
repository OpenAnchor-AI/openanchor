"""Worker name constants and feature flags.

Centralizes worker name strings so future re-enable / disable / rename
is a one-file change. Description strings and historical comments are
left in their original modules.

v0.9.21 history: fable-5 was permanently disabled after Gate 1 fail
(60% vendor placeholder from baosiapi). See docs/AUDIT_VERIFY_2026-07-12.md
and fusion_modes._FABLE5_PCT for gradual rollout mechanism.

v0.9.52 lean default pool: flash/M3/sol/fable enabled; sonnet/haiku/opus env-gated.

v0.9.50-p0 (2026-07-22 resource-boundary cleanup):
  - nemotron-3-ultra-free removed (probe inconclusive; resource boundary eliminated)
  - deepseek-v4-pro added (DeepSeek official API, Gate-1 probe pending)
  - gpt-5.6-sol added (Baosi GPT/Codex group, baosiapi.com host)
  - fable-5 default rollout changed 0% -> 10% (gradual re-enable)
  - Baosi Claude group (sonnet-5 / fable-5 / opus-4-8 / haiku-4-5): base_url=https://baosiapi.com/v1
    (api.baosiapi.com retired). Prefer BAOSIAPI_GPT_API_KEY; BAOSIAPI_API_KEY is legacy.
  - Baosi GPT/Grok group also on https://baosiapi.com/v1 (same host, often same key)
"""
from __future__ import annotations
import os as _os

# Worker name constants (single source of truth)
FABLE5: str = "claude-fable-5"
OPUS: str = "claude-opus-5"
SONNET: str = "claude-sonnet-5"
HAIKU: str = "claude-haiku-4-5"
M3: str = "minimax-m3"
# Note: worker.name "deepseek-v4-flash" is OpenCode-Zen free (model=deepseek-v4-flash-free).
#       Distinct from official deepseek-v4-pro (DeepSeek direct API, different channel).
DPSK: str = "deepseek-v4-flash"
DPSK_PRO: str = "deepseek-v4-pro"
GPT_SOL: str = "gpt-5.6-sol"
GPT_LUNA: str = "gpt-5.6-luna"
GPT_TERRA: str = "gpt-5.6-terra"
# Local Ollama env-gated worker (v0.9.57).
# Slotted between Flash and M3 in _AUTO_COST_ORDER. Disabled by default;
# enable via ANCHOR_ENABLE_OLLAMA_ORNITH. Zero marginal cost.
ORNITH: str = "ollama-ornith-35b"

# Sacred = workers with hard ¥10/session cost cap (v0.9.16 Fable 5 #6)
SACRED: frozenset[str] = frozenset({FABLE5, OPUS})

# Per-worker specific cooldown (seconds). Fable 5 long cooldown
# to prevent repeat hits on baosiapi's placeholder bug.
# v1.0.1 + B5 audit 2026-08-15: 900s -> 3600s (4x stricter).
WORKER_SPECIFIC_COOLDOWN_S: dict[str, int] = {
    FABLE5: 3600,
}

# SLA targets: per-worker p95 latency, min judge score, max error rate
SLA_TARGETS: dict[str, dict[str, float]] = {
    DPSK:     {"p95_ms": 30000, "min_judge": 0.55, "max_error_rate": 0.20},
    DPSK_PRO: {"p95_ms": 30000, "min_judge": 0.65, "max_error_rate": 0.15},
    M3:       {"p95_ms": 60000, "min_judge": 0.55, "max_error_rate": 0.25},
    HAIKU:    {"p95_ms": 60000, "min_judge": 0.70, "max_error_rate": 0.15},
    SONNET:   {"p95_ms": 60000, "min_judge": 0.55, "max_error_rate": 0.30},
    FABLE5:   {"p95_ms": 60000, "min_judge": 0.85, "max_error_rate": 0.10},
    OPUS:     {"p95_ms": 60000, "min_judge": 0.80, "max_error_rate": 0.10},
    GPT_SOL:  {"p95_ms": 30000, "min_judge": 0.70, "max_error_rate": 0.15},
    GPT_LUNA: {"p95_ms": 30000, "min_judge": 0.65, "max_error_rate": 0.20},
    GPT_TERRA: {"p95_ms": 30000, "min_judge": 0.70, "max_error_rate": 0.15},
}

# Manual quarantines (kept in routing for diagnostics, removed from public).
# Empty since v0.9.50-p0 — Agnes/Kilo/Gemini removed entirely (no manual fallback).
QUARANTINED: dict[str, str] = {}
QUARANTINED_WORKERS = QUARANTINED  # alias for backward compat

# v0.9.50-p1: Fable-5 gradual rollout — Gate-2 probe 2026-07-22 PASSED (placeholder_rate=0.0%
# on 50 stratified samples vs opus-4-8, see data/probe_fable5_results.json). Per v0.9.50-p0
# spec, <5% placeholder rate → "enable" (push to 100%). Default rolled from 10% → 100%.
# Set FABLE5_TRAFFIC_PCT env (0-100) to override at runtime. Use /admin/gradual/fable5
# endpoint for in-process tuning. Rollback to 0 set FABLE5_TRAFFIC_PCT=0.
FABLE5_TRAFFIC_PCT: float = float(_os.environ.get("FABLE5_TRAFFIC_PCT", "100"))

# Legacy alias kept for callers that still read the boolean flag. True iff
# rollout percentage > 0. Note: gradual rollout (10%) is NOT the same as
# fully enabled — callers should branch on FABLE5_TRAFFIC_PCT for proportion.
FABLE5_ENABLED: bool = FABLE5_TRAFFIC_PCT > 0
