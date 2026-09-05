from __future__ import annotations
from anchor.workers import FABLE5
"""Anchor configuration — 15 workers configured (see WORKERS below).

Cost units (audit 2026-08-16 F9/C17): cost_in / cost_out are YUAN per 1M
TOKENS (¥/M) for every worker — baosiapi values are stored directly in
RMB (e.g. opus 1.5831), deepseek-official too. The old header claiming
"USD per 1M ×0.0452" was wrong (the FX direction was also inverted) and
is removed. Per-call routing cost lives in WORKER_COST_YUAN (v8-derived;
see _build_worker_cost_yuan). Baosi unified ¥588/mo covers all baosiapi
workers (fable/opus/sol) from shared $13,000 credit pool.


Anchor = the fusion model router.
Each query gets routed through the optimal combination of frontier models,
learning from every interaction. Cheaper than the strongest single, smarter
than the sum of its parts.

Worker pool (canonical 2026-07-22 lean single-product pool):
  slot 2   DeepSeek V4 Flash     FREE coding              OpenCode Zen         ENABLED
  slot 3   MiniMax M3            ¥119/mo, ~4.83B/30d real use      direct (multi-key)   ENABLED (main)
  slot 9   DeepSeek V4 Pro       ¥3/MTok in / ¥6/MTok     DeepSeek official    OFF (Gate-1 gradual; ANCHOR_ENABLE_DEEPSEEK_PRO=1)
  slot 10  GPT-5.6 Sol           baosiapi GPT group       baosiapi.com/v1      ENABLED
  slot 51  Claude Fable 5        baosiapi unified        baosiapi             ENABLED (top quality; Gate-2 PASSED)
  slot 5   Claude Sonnet 5       baosiapi unified        baosiapi             OFF (redundant vs fable; ANCHOR_ENABLE_SONNET=1)
  slot 52  Claude Opus 5       baosiapi unified        baosiapi             ENABLED (hard lane, $5/M; v0.9.54)
  slot 53  Claude Haiku 4.5      baosiapi unified        baosiapi             OFF (redundant; ANCHOR_ENABLE_HAIKU=1)
  slot 11  Grok 4.5 Reasoning    T2 速度                  baosiapi-grok        probe/env (Gate PASSED 2026-07-23)
  slot 12  GPT-5.6 Luna          T2.5 balanced            baosiapi-gpt         OFF (ANCHOR_ENABLE_LUNA=1)
  slot 13  GPT-5.6 Terra         T3 economy               baosiapi-gpt         OFF (ANCHOR_ENABLE_TERRA=1)
  slot 14  Kimi K3               T3 中文 multimodal       baosiapi-kimi        OFF (ANCHOR_ENABLE_K3=1)

Resource boundaries (Baosi Claude group, slots 5/51/52/53):
  - Host: https://baosiapi.com/v1 (api.baosiapi.com retired / no quota)
  - Key: BAOSIAPI_GPT_API_KEY preferred; BAOSIAPI_API_KEY legacy fallback
  - Fable default ON (sacred); Opus 5 default ON (hard); haiku env-gated; sonnet REMOVED

Resource boundaries (Baosi GPT group, slots 10/12/13):
  - Independent quota (separate key BAOSIAPI_GPT_API_KEY)
  - base_url: https://baosiapi.com/v1

Resource boundaries (Baosi Grok group, slot 11):
  - Independent quota (shared BAOSIAPI_GPT_API_KEY)
  - base_url: https://baosiapi.com/v1

Resource boundaries (Baosi Kimi group, slot 14):
  - Independent quota (shared BAOSIAPI_GPT_API_KEY)
  - base_url: https://baosiapi.com/v1

Resource boundaries (DeepSeek, slots 2 + 9):
  - slot 2: OpenCode Zen free (worker.name=deepseek-v4-flash, model=deepseek-v4-flash-free)
  - slot 9: DeepSeek official API (worker.name=deepseek-v4-pro) — Gate-1 enable_gradual
            3.3% after thinking-disable fix 2026-07-23 (was false HOLD from reasoning
            eating max_tokens). Opt-in: ANCHOR_ENABLE_DEEPSEEK_PRO=1.

Removed (v0.9.50-p0 / v0.9.50-p1):
  - slot 1  Agnes 2.0 Flash   (cleanup — image-gen `agnes-image-2.1-flash` retained as separate worker)
  - slot 4  GLM-5.2 Pro       (Zhipu subscription not active)
  - slot 6  Gemini 2.5 Flash  (quota exhausted 2026-07-11; image-gen `gemini-2.5-flash-image` also removed)
  - slot 7  Kilo auto-free    (upstream logs prompts; opt-in fallback removed)
  - slot 8  Nemotron 3 Ultra  (probe inconclusive; resource boundary eliminated)

Architecture: 100% API. No local models in worker pool. Only ollama
qwen3-embedding is used internally for KNN routing (not user-visible).
"""
# ── Pricing unit ──────────────────────────────────────────────
# cost_in / cost_out:  USD per million tokens (¥/M after FX).
# FX rate (baosiapi):  ¥588/mo → $13,000 credit → ¥0.0452/USD.
#   Example: fable $10/M → actual ¥0.452/M.
# DeepSeek official:    ¥3/M direct (no baosiapi FX).
# MiniMax M3:           ¥119/mo subscription, ~¥0.025/M at 4.83B/30d.
# ─────────────────────────────────────────────────────────────

import os
from dataclasses import dataclass, replace
from anchor.worker_gate import (
    _truthy as _env_truthy,
)
from typing import Literal

# V6.3 audit 2026-08-15: grok-4-5 opt-out by default (cost 39.7%, q=0.5255, B7 HOLD).
# V6.2 推荐: 替代为 grok-4-6 (slot 15 default ON, q=0.7777 vs 0.5255, 50x cheaper).
# Set ANCHOR_DISABLE_GROK4_5=0 to enable grok-4-5 calibration opt-in (legacy probe env).
ANCHOR_DISABLE_GROK4_5_DEFAULT: bool = os.environ.get("ANCHOR_DISABLE_GROK4_5", "1") == "1"

Kind = Literal["openai-compat", "opencode-cli", "direct", "multi-key", "anthropic"]
Channel = Literal["baosiapi", "opencode-zen",
                  "minimax-direct",
                  "deepseek-official", "baosiapi-gpt",
                  "baosiapi-grok", "baosiapi-kimi", "kilo",
                  "ollama-local"]


@dataclass(frozen=True)
class Worker:
    slot: int
    name: str
    kind: Kind
    channel: Channel
    model: str
    base_url: str
    role_tags: tuple[str, ...]
    cost_in: float = 0.0
    cost_out: float = 0.0
    # v1.0.x E9 audit 2026-08-17: cached-input discount per 1M tokens
    # (0 = default 10% of cost_in at usage-time).
    cost_in_cached: float = 0.0
    rpm: int = 0
    enabled: bool = True
    description: str = ""
    # v0.9.46g: chat-only workers (no tool/function-call support) are excluded
    # from the routing pool when the query requires tool use. Set per-worker.
    chat_only: bool = False
    api_key_env: str = ""
    monthly_fixed_cny: float = 0.0
    notes: str = ""
    # Multi-key pool (preferred when non-empty). Round-robin + per-key cooldown.
    # If empty, falls back to single api_key_env.
    api_key_envs: tuple[str, ...] = ()
    # v0.9.76+: how the multi-key pool distributes requests.
    #   "round-robin" — atomic rotate through all keys (use when keys are on
    #     INDEPENDENT accounts with INDEPENDENT quotas; spreads load so both
    #     hit limits around the same time — best for parallel capacity).
    #   "failover"   — keys[0] is primary; only use keys[1..] when keys[0]
    #     is in per-key cooldown (use when keys share a single upstream
    #     quota bucket — round-robin would burn key_2's quota every time
    #     key_1 hits the limit, which is wasteful when the quota is shared).
    key_strategy: str = "round-robin"
    # Optional system prompt injected at the start of every request.
    # Use for thinking-first models (e.g. dpsk free) to force direct answer.
    system_prompt: str = ""
    # v0.9.27: maximum input context window (tokens). 0 = unknown / use upstream default.
    # Documented per worker; not enforced (no runtime truncation).
    context_window: int = 0
    # v0.9.59-A1: worker-provided max_tokens override. None = use existing clamp behavior.
    max_tokens: int | None = None
    # v0.9.60 (L2): thinking-first model — inject extra_body thinking=disabled
    # by default. True for minimax/m3/deepseek families; set per-worker so
    # adding a new thinking-capable worker doesn't require editing the
    # hard-coded prefix tuple in clients/key_pool.py.
    thinking_first: bool = False


# V4.2 audit 2026-08-15: Sol opt-out gate (cost 40%, q=0.7785, B7 HOLD).
# Set ANCHOR_DISABLE_SOL=0 to opt back in for calibration runs.
ANCHOR_DISABLE_SOL_DEFAULT: bool = os.environ.get("ANCHOR_DISABLE_SOL", "1") == "1"

# === 12-worker registry (4 lean + Grok via probe/env) ==================
# baosiapi unified ¥588/mo → $13,000 credit; only Fable enabled by default
WORKERS: list[Worker] = [
    # ----- DeepSeek V4 Flash (coding, free via OpenCode Zen) -----
    Worker(
        slot=2, name="deepseek-v4-flash", kind="multi-key", channel="opencode-zen",
        model="deepseek-v4-flash-free",
        context_window=65536,
        base_url="https://opencode.ai/zen/v1",
        role_tags=("coding", "light"),
        cost_in=0.0, cost_out=0.0, rpm=60,
        thinking_first=True,  # v0.9.60 (L2): disable thinking by default
        # v0.9.57-p1: Default ON. OpenCode Zen free tier stabilized (probe 2026-08-02, n=5, placeholder_rate=0.000%).
        # Set ANCHOR_ENABLE_DEEPSEEK_V4_FLASH=0 to disable per-machine (kill switch).
        # Logic: unset/None → ON; "0" → OFF; "1" → ON.
        enabled=not os.environ.get("ANCHOR_ENABLE_DEEPSEEK_V4_FLASH")
        or _env_truthy(os.environ.get("ANCHOR_ENABLE_DEEPSEEK_V4_FLASH")),
        api_key_env="OPENCODE_ZEN_API_KEY",
        api_key_envs=("OPENCODE_ZEN_API_KEY", "OPENCODE_ZEN_API_KEY_2", "OPENCODE_API_KEY"),
        # v0.9.76+: failover mode — Zen's two keys appear to share one
        # upstream free-tier quota (probed 2026-08-19: both 429'd at the
        # same instant when the bucket emptied). Round-robin here would
        # burn key_2 every time key_1 hit the cap, so we pin keys[0] as
        # primary and only advance to keys[1..] when keys[0] is in
        # per-key cooldown.
        key_strategy="failover",
        # dpsk is a thinking-first model that spends all max_tokens on reasoning_content,
        # returning content="". Force direct answer with no reasoning blocks.
        system_prompt="Answer directly without reasoning or thinking blocks. Be concise.",
        # v0.9.59-A1: 4096 caps output to prevent length truncation
        # (n=20 probe 2026-08-02: 1/20 sample had finish_reason=length + response="",
        # root cause = dpsk-flash default limit too low for code prompts).
        # FREE tier so no cost concern.
        max_tokens=4096,
        description="DeepSeek V4 Flash via OpenCode Zen. FREE 2-key pool (failover). Best for: cheap coding, light agent.",
    ),

    # ----- Kilo free pool (Boss WS#1 decision, 2026-08-29) -----
    # Zero-marginal-cost peer for easy/medium work and overflow.
    Worker(
        slot=7, name="kilo-auto-free", kind="openai-compat", channel="kilo",
        model="kilo-auto/free",
        context_window=32768,
        base_url="https://api.kilo.ai/api/gateway/v1",
        role_tags=("light", "free", "en"),
        cost_in=0.0, cost_out=0.0, rpm=20,
        enabled=not os.environ.get("ANCHOR_ENABLE_KILO")
        or _env_truthy(os.environ.get("ANCHOR_ENABLE_KILO")),
        api_key_env="KILOCODE_API_KEY",
        system_prompt="Answer directly without preamble. Use concise, plain language.",
        description="Kilo Auto Free via api.kilo.ai gateway. Free peer for easy/medium work and overflow; disable with ANCHOR_ENABLE_KILO=0.",
    ),

    # ----- CN agent / 1M context -----
    Worker(
        slot=3, name="minimax-m3", kind="multi-key", channel="minimax-direct",
        model="MiniMax-M3",
        context_window=1048576, base_url="https://api.minimaxi.com/v1",
        role_tags=("cn-agent", "multi", "mid-en", "hard"),
        cost_in=0.0, cost_out=0.0, rpm=300,
        api_key_env="MINIMAX_API_KEY_1",
        api_key_envs=("MINIMAX_API_KEY_1", "MINIMAX_API_KEY_2"),
        thinking_first=True,  # v0.9.60 (L2): disable thinking by default
        # Note: MINIMAX_API_KEY (legacy alias = _2) intentionally removed
        # here; key_pool.KeyPool._avail_keys() dedups duplicate values,
        # so removing it doesn't change runtime behavior, but the list
        # otherwise misleads anyone reading WORKERS about key count.
        # Round-robin across _1/_2 is meaningful: verified 2026-07-26
        # that the two keys belong to different minimax accounts with
        # independent 5h Token Plan quota (under burst, _1 hit 429 while
        # _2 stayed 200).
        # M3 leaks <think> reasoning tags by default. Strip in KeyPool._call().
        system_prompt="Answer directly. Do not include <think> or </think> tags.",
        # v0.9.54: ¥119/mo subscription. Dashboard: 4.83B tokens/30d real usage (~¥0.025/M). Supports M3/M2.7/image/voice/music, 4-5 concurrent agents, 1M context, multimodal.
        # Actual: ¥119 / 4830M = ¥0.025/M tokens. Set to 0.0 for routing (sunk cost).
        # (≈¥0.015/query); use 0.0 in cost_in/out to avoid double-counting.
        monthly_fixed_cny=119.0,
        description="MiniMax M3 1M context, 2-key pool (round-robin). CN, multimodal, agent. ¥119/mo / 18B tokens subscription.",
    ),

    # ----- Local Ollama Ornith 35B (v0.9.57, env-gated, zero marginal cost) -----
    # OpenAI-compat endpoint exposed by an Ollama daemon.
    # Disabled by default; opt in with ANCHOR_ENABLE_OLLAMA_ORNITH=1.
    # Base URL configurable via ANCHOR_OLLAMA_BASE_URL.
    # Default after t_7b06ca78 (Advisor Directive · 2026-08-29):
    # aimax farm 10.0.0.4 (amd ai max+ 395, wg utun11) with ollama 0.32.x serving
    # ornith-35b-q8-agent:latest (35.5B Q4_K_M) + qwen3-embed-8b:latest (4.7GB embed).
    # Override to a same-host loopback (e.g. http://127.0.0.1:11435/v1) by exporting
    # ANCHOR_OLLAMA_BASE_URL when running anchor on the same box as the ollama daemon.
    # Zero marginal cost -> cost_in/out = 0, monthly_fixed_cny = 0.
    # chat_only=False (tool-capable; smolagents/Ollama qwen-style agent tags).
    Worker(
        slot=1, name="ollama-ornith-35b", kind="openai-compat", channel="ollama-local",
        model="ornith-35b-q8-agent:latest",
        context_window=65536,
        base_url=os.environ.get("ANCHOR_OLLAMA_BASE_URL", "http://10.0.0.4:11434/v1"),
        role_tags=("coding", "agent", "reasoning", "cn-agent", "mid-en", "local"),
        cost_in=0.0, cost_out=0.0, rpm=8,
        enabled=False,  # v1.0.1: local ollama off by default
        api_key_env="",  # local; factory passes empty key (Ollama ignores Bearer)
        chat_only=False,
        monthly_fixed_cny=0.0,
        notes="Local Ollama. ANCHOR_ENABLE_OLLAMA_ORNITH=1 to enable. ANCHOR_OLLAMA_BASE_URL overrides base URL. Zero marginal cost. Default endpoint: aimax 10.0.0.4:11434 (t_7b06ca78).",
        description="Ollama Ornith 35B (q8-agent) on aimax farm (10.0.0.4). Tool-capable, 65k context. Zero marginal cost. Env-gated; ANCHOR_OLLAMA_BASE_URL for same-host override.",
    ),

    # ----- EN mid / Sonnet 5 -----
    Worker(
        slot=5, name="claude-sonnet-5", kind="openai-compat", channel="baosiapi",
        model="claude-sonnet-5",
        context_window=200000, base_url="https://baosiapi.com/v1",
        role_tags=("mid-en", "hard", "coding"),
        cost_in=0.9498, cost_out=4.7492, rpm=20,
        enabled=True,  # v1.0.1 calibration prep: forced ON for re-bench; Phase 3 决定去留
        api_key_env="BAOSIAPI_GPT_API_KEY",
        api_key_envs=("BAOSIAPI_GPT_API_KEY", "BAOSI_GPT_API_KEY", "BAOSIAPI_API_KEY", "BAOSI_API_KEY"),
        monthly_fixed_cny=0.0,  # shared Claude group fixed cost lives on fable
        description="Claude Sonnet 5 via baosiapi.com. v1.0.1 calibration prep forced ON; Phase 3 决定去留. ANCHOR_ENABLE_SONNET legacy env不再门控.",
    ),

    # ----- Claude Fable 5 (v0.9.50-p0: gradual rollout; v0.9.50-p1 full rollout) -----
    # v0.9.21: baosiapi serves a 30-char Chinese stub for fable-5 ~60% of the time
    # even though HTTP returns 200 OK. cooldown.py cannot catch successful empty
    # responses. v0.9.50-p1 Gate-2 probe PASSED 2026-07-22 (placeholder_rate=0.0%
    # on 50 stratified samples vs opus-4-8); rolled to 100% per spec rule "rate<5%
    # → enable". Override via FABLE5_TRAFFIC_PCT env or /admin/gradual/fable5.
    Worker(
        slot=51, name=FABLE5, kind="openai-compat", channel="baosiapi",
        model=FABLE5,
        context_window=200000, base_url="https://baosiapi.com/v1",
        role_tags=("code", "hard", "sacred"),
        cost_in=3.1662, cost_out=15.8308, rpm=20, enabled=False,  # Phase 3 FIX-E: 12% err + expensive + hard underperforms flash/m3 → HOLD (opt-in)
        api_key_env="BAOSIAPI_GPT_API_KEY",
        api_key_envs=("BAOSIAPI_GPT_API_KEY", "BAOSI_GPT_API_KEY", "BAOSIAPI_API_KEY", "BAOSI_API_KEY"),
        monthly_fixed_cny=588.0,  # baosiapi unified ¥588/mo → $13,000 credit (sole default Claude worker)
        description="Claude Fable 5 via baosiapi.com (sacred top). Host migrated from retired api.baosiapi.com. Gate-2 PASSED. FABLE5_TRAFFIC_PCT=100 default. Sacred ¥10/session hard cap. baosi架构价 input $70/output $350/cache $7 per 1M.",
    ),

    # v0.9.7X-P2 + v1.0.1 calibration prep: claude-opus-5 WorkerSpec re-added for Phase 1
    # re-calibration (audit 2026-08-07 结论受 fallback 路径污染, 87.7% 是兜底, 不是 opus-5 真质量).
    # enabled=True 供校准覆盖; Phase 3 校准数据出来后决定 enabled/fallback (本任务不动 fallback chain).
    # _CLAUDE_BAOSI_WORKERS 不含 opus-5 (auth gate 不需要; BAOSIAPI_GPT_API_KEY 已能通过 gate).
    Worker(
        slot=52, name="claude-opus-5", kind="openai-compat", channel="baosiapi",
        model="claude-opus-5",
        context_window=200000, base_url="https://baosiapi.com/v1",
        role_tags=("hard",),
        cost_in=1.5831, cost_out=7.9154, rpm=20,
        # audit 2026-08-16 (B8/C2): B6 flipped this to default ON, but opus-5 is
        # excluded from the v8-eligibility pool (err 10.75% >= 10% gate), absent
        # from hard rules / vision set / fallback reachability, and the auth gate
        # never disables it keyless — "default ON" was an unreachable false state.
        # Boss WS#1: paid quality floor uses Opus-5; auth gate disables it
        # automatically when no usable baosiapi key is present.
        enabled=True,
        api_key_env="BAOSIAPI_GPT_API_KEY",
        api_key_envs=("BAOSIAPI_GPT_API_KEY", "BAOSI_GPT_API_KEY", "BAOSIAPI_API_KEY", "BAOSI_API_KEY"),
        monthly_fixed_cny=0.0,  # shared Claude group (fable ¥588/mo covers claude-* all)
        description="Claude Opus 5 via baosiapi.com (calibration prep, Phase 1). 2026-08: ¥1.5831/¥7.9154 per 1M. Default ON for calibration; Phase 3 decides routing role. See p2_opus5_eval.md for re-eval context.",
    ),

    # ----- Claude Haiku 4.5 (cheap mid) -----
    Worker(
        slot=53, name="claude-haiku-4-5", kind="openai-compat", channel="baosiapi",
        model="claude-haiku-4-5-20251001",
        context_window=200000,
        base_url="https://baosiapi.com/v1",
        role_tags=("light", "mid-en"),
        cost_in=0.2262, cost_out=1.1308, rpm=20,
        enabled=_env_truthy(os.environ.get("ANCHOR_ENABLE_HAIKU")),
        api_key_env="BAOSIAPI_GPT_API_KEY",
        api_key_envs=("BAOSIAPI_GPT_API_KEY", "BAOSI_GPT_API_KEY", "BAOSIAPI_API_KEY", "BAOSI_API_KEY"),
        monthly_fixed_cny=0.0,  # shared Claude group
        description="Claude Haiku 4.5 via baosiapi.com. Default OFF (flash/M3 cover light). ANCHOR_ENABLE_HAIKU=1 to enable.",
    ),

    # ----- DeepSeek V4 Pro (DeepSeek official API) -----
    # v0.9.50-p0: separate worker from OpenCode-Zen free (slot 2).
    # Gate-1 probe 2026-07-22 detected vendor placeholder echo on this channel
    # (same symptom as baosiapi fable-5). Default disabled pending Gate-1
    # re-validation. Enable with ANCHOR_ENABLE_DEEPSEEK_PRO=1.
    # Cost: ¥3/MTok input, ¥6/MTok output (peak hours ×2 multiplier applies).
    Worker(
        slot=9, name="deepseek-v4-pro", kind="openai-compat", channel="deepseek-official",
        model="deepseek-v4-pro",
        context_window=65536, base_url="https://api.deepseek.com/v1",
        role_tags=("coding", "mid-en", "hard"),
        cost_in=9.0, cost_out=27.0, rpm=60,  # deepseek 官方人民币 2026-08-17 新价: 输入 9 元/输出 27 元 per 1M
        enabled=_env_truthy(os.environ.get("ANCHOR_ENABLE_DEEPSEEK_PRO")),  # B6 audit 2026-08-15: 默认 opt-in (¥9/M, opus-5 ¥1.58/M 更便宜); ANCHOR_ENABLE_DEEPSEEK_PRO=1 启用
        api_key_env="DEEPSEEK_API_KEY",
        thinking_first=True,  # v0.9.60 (L2): disable thinking by default
        notes="DeepSeek official API (NOT OpenCode-Zen). baosi架构价 input $4.35/output $8.70/cache $1.0875 per 1M (flat ¥0.585/M, 全场最便宜 paid). B6 audit 2026-08-15 default opt-in (opus-5 更便宜). ANCHOR_ENABLE_DEEPSEEK_PRO=1 启用.",
    ),


    # ----- DeepSeek V4 Flash Official (DeepSeek paid API) -----
    # v0.9.58: distinct from OpenCode-Zen free (slot 2).
    # DeepSeek官方 paid API, 比 dpsk-pro 还便宜 (input $0.27/output $1.10 per 1M, ¥0.039/M).
    # v0.9.62: probe n=10 live placeholder_rate=0.0%. Default ON. Kill switch ANCHOR_DISABLE_DEEPSEEK_V4_FLASH_OFFICIAL=1.
    # v0.9.76+: ANCHOR_ENABLE_DEEPSEEK_V4_FLASH_OFFICIAL defaults to 0 — Pareto
    # gate has been excluding this worker on q_score=0.000 < 0.40 (no recent
    # judge data), so the previous default "ON" just made every fallback
    # pay a wasted probe round-trip before the gate threw it out. Opt back
    # in with ANCHOR_ENABLE_DEEPSEEK_V4_FLASH_OFFICIAL=1 once the worker has
    # enough judge history to clear the floor.
    Worker(
        slot=16, name="deepseek-v4-flash-official", kind="openai-compat", channel="deepseek-official",
        model="deepseek-v4-flash",
        context_window=65536, base_url="https://api.deepseek.com/v1",
        role_tags=("coding", "mid-en", "fast"),
        cost_in=3.0, cost_out=9.0, rpm=60,  # deepseek 官方人民币 2026-08-17 新价: 输入 3 元/输出 9 元 per 1M
        enabled=_env_truthy(os.environ.get("ANCHOR_ENABLE_DEEPSEEK_V4_FLASH_OFFICIAL", "0")),
        api_key_env="DEEPSEEK_API_KEY",
        thinking_first=True,  # v0.9.60 (L2): disable thinking by default
        notes="DeepSeek official paid API (NOT OpenCode-Zen free, slot 2). baosi架构 input $0.27/output $1.10 per 1M (flat ¥0.039/M, 比 dpsk-pro ¥0.585/M 更便宜). v0.9.76+ default OFF (Pareto gate has been excluding on q_score=0); set ANCHOR_ENABLE_DEEPSEEK_V4_FLASH_OFFICIAL=1 to opt back in once judge history clears the floor.",
    ),

    # ----- GPT-5.6 Sol (Baosi GPT/Codex group) -----
    # v0.9.50-p0: independent quota (separate key from baosiapi Claude group).
    # base_url=https://baosiapi.com/v1 (unified Baosi host for GPT + Claude). Cross-family judge candidate.
    # Phase 3 FIX-E hold: 11% err + cost ¥2.26/M vs Luna ¥0.09/M (25x), no quality
    # benefit on T2 hard (Luna covers T1 code, Sol covers T2 but error rate 11%
    # exceeds 5% gate). Re-enable requires quality probe n>=50 placeholder_rate<5%.
    # Per B7 audit 2026-08-15: docstring added to prevent future re-enable without
    # proper probe.
    Worker(
        slot=10, name="gpt-5.6-sol", kind="openai-compat", channel="baosiapi-gpt",
        model="gpt-5.6-sol",
        context_window=128000, base_url="https://baosiapi.com/v1",
        role_tags=("coding", "mid-en", "hard", "agent"),
        cost_in=2.2615, cost_out=13.5692, rpm=20,
        enabled=False,  # Phase 3 FIX-E: 11% err + expensive + hard underperforms flash/m3 → HOLD (opt-in)
        api_key_env="BAOSIAPI_GPT_API_KEY",
        api_key_envs=("BAOSIAPI_GPT_API_KEY", "BAOSI_GPT_API_KEY"),
        notes="Baosi GPT group on baosiapi.com/v1 (same host as Claude/Fable after api.baosiapi.com retirement). baosi架构价 input $50/output $300/cache $5 per 1M (OpenAI 未降价). V4.2 audit 2026-08-15: opt-out by default (ANCHOR_DISABLE_SOL=1); calibration opt-in via ANCHOR_DISABLE_SOL=0.",
    ),

    # ----- Grok 4.5 Reasoning (T2 速度, baosiapi-grok / GPT host) -----
    # Gate PASSED 2026-07-23 (scripts/probe_grok.py → enable).
    # ANCHOR_ENABLE_GROK4=1 force on; =0 force off; unset → probe JSON.
    Worker(
        slot=11, name="grok-4-5-reasoning", kind="openai-compat", channel="baosiapi-grok",
        context_window=128000, role_tags=("reasoning", "hard", "mid-en"),
        cost_in=0.9046, cost_out=2.7138, rpm=60,  # baosi架构 2026-08: ¥0.9046/¥2.7138 per 1M (13000 USD=588 CNY)
        enabled=False,  # B7 HOLD + V6.3 opt-out: superseded by grok-4-6 (q=0.7777 vs 0.5255, 50x cheaper); calibration opt-in via ANCHOR_DISABLE_GROK4_5=0
        base_url="https://baosiapi.com/v1",
        model="grok-4.5",
        api_key_env="BAOSIAPI_GPT_API_KEY",
        api_key_envs=("BAOSIAPI_GPT_API_KEY", "BAOSI_GPT_API_KEY"),
        notes="T2 speed worker (legacy). V6.3 superseded by grok-4-6 (slot 15 default ON, q=0.7777 vs 0.5255, 50x cheaper); calibration opt-in via ANCHOR_DISABLE_GROK4_5=0. model=grok-4.5 (real xAI API name; worker.name kept as anchor internal id). baosi架构价 input $20/output $60/cache $5 per 1M.",
    ),


    # ----- Grok 4.6 Reasoning (calibration prep, baosiapi-grok) -----
    # v1.0.1 calibration: baosiapi实测可调 (返回 GROK46_OK).
    # model=grok-4.6 (real xAI API name). Phase 3 校准数据出来后定 4.5 vs 4.6 去留.
    Worker(
        slot=15, name="grok-4-6-reasoning", kind="openai-compat", channel="baosiapi-grok",
        context_window=128000, role_tags=("reasoning", "hard", "mid-en"),
        cost_in=0.9046, cost_out=2.7138, rpm=60,
        enabled=True,  # Phase 1 calibration: enabled for benchmark coverage; Phase 3 决定
        base_url="https://baosiapi.com/v1",
        model="grok-4.6",
        api_key_env="BAOSIAPI_GPT_API_KEY",
        api_key_envs=("BAOSIAPI_GPT_API_KEY", "BAOSI_GPT_API_KEY"),
        notes="Grok 4.6 baosiapi实测 GROK46_OK (calibration prep, Phase 1). model=grok-4.6. Default ON for calibration; Phase 3 decides 4.5 vs 4.6.",
    ),

    # ----- GPT-5.6 Luna (T2.5 balanced, baosiapi new channel) -----
    # v1.0: T2.5 balanced probe candidate. Enable with ANCHOR_ENABLE_LUNA=1.
    Worker(
        slot=12, name="gpt-5.6-luna", kind="openai-compat", channel="baosiapi-gpt",
        context_window=128000, role_tags=("code", "cn", "mid-en"),
        cost_in=0.0905, cost_out=0.5428, rpm=60,
        enabled=False,  # v1.0.1 calibration prep: forced ON for Phase 1 re-bench; Phase 3 校准数据出来后定去留 (原 env-gated via ANCHOR_ENABLE_LUNA)
        base_url="https://baosiapi.com/v1",
        model="gpt-5.6-luna",
        api_key_env="BAOSIAPI_GPT_API_KEY",
        api_key_envs=("BAOSIAPI_GPT_API_KEY", "BAOSI_GPT_API_KEY"),
        notes="gpt-5.6 series variant. OpenAI -80% 降价后 baosi input $2/output $12/cache $0.2 per 1M (flat ¥0.633/M). Default OFF (env-gated, ANCHOR_ENABLE_LUNA=1 to opt in; v0.9.57 'default ON' intent rolled back; audit 2026-08-04: 0 calls in 30d).",
    ),

    # ----- GPT-5.6 Terra (T3 economy, baosiapi new channel) -----
    # v1.0: T3 economy with long-context/reasoning strength. Enable with ANCHOR_ENABLE_TERRA=1.
    Worker(
        slot=13, name="gpt-5.6-terra", kind="openai-compat", channel="baosiapi-gpt",
        context_window=256000, role_tags=("reasoning", "hard", "long-context"),
        cost_in=0.9046, cost_out=5.4277, rpm=60,
        enabled=True,  # v1.0.1 calibration prep: forced ON for Phase 1 re-bench; Phase 3 校准数据出来后定去留 (原 env-gated via ANCHOR_ENABLE_TERRA)
        base_url="https://baosiapi.com/v1",
        model="gpt-5.6-terra",
        api_key_env="BAOSIAPI_GPT_API_KEY",
        api_key_envs=("BAOSIAPI_GPT_API_KEY", "BAOSI_GPT_API_KEY"),
        notes="Default OFF (audit 2026-08-04: 0 calls in 30 days). ANCHOR_ENABLE_TERRA=1 to opt in.",
    ),

    # ----- Kimi K3 (T3 中文 multimodal, baosiapi new channel, deferred) -----
    # v1.0: T3 Chinese multimodal. env-gated default-disabled.
    # Cost: baosi架构 input $30/output $150/cache $30 per 1M (flat ¥8.14/M). Vs M3 ¥0.025/M (1M ctx) — too expensive.
    Worker(
        slot=14, name="kimi-k3", kind="openai-compat", channel="baosiapi-kimi",
        context_window=128000, role_tags=("cn", "multi", "vision", "long-context"),
        cost_in=1.3569, cost_out=6.7846, rpm=60,
        # audit 2026-08-16 (B8/C3): was default ON (env default "1") while its
        # own notes/README/AGENTS.md said "Default OFF" — and it is excluded from
        # the pool (absent from v8 table) yet costed as free. Default OFF now;
        # opt-in via ANCHOR_ENABLE_K3=1.
        enabled=_env_truthy(os.environ.get("ANCHOR_ENABLE_K3", "0")),
        base_url="https://baosiapi.com/v1",
        model="k3",
        api_key_env="BAOSIAPI_GPT_API_KEY",
        api_key_envs=("BAOSIAPI_GPT_API_KEY", "BAOSI_GPT_API_KEY"),
        notes="Kimi K3 multimodal. baosi架构 input $30/output $150/cache $30 per 1M (flat ¥8.14/M). Default OFF (vs M3 ¥0.025/M, 1M ctx 没性价比). ANCHOR_ENABLE_K3=1 to enable.",
    ),
]

# audit 2026-08-16 (C9): the slot-based FALLBACK_CHAIN was removed — it was
# only consumed by workers_for_role's dead fallback branch and diverged from
# WORKER_FALLBACK_CHAIN (the single source of truth used by routing_fallback).
# See WORKER_FALLBACK_CHAIN below.

# v0.9.53 (P0-4 fix): CANONICAL worker fallback graph.
# Replaces the conflicting triple-source definitions (FALLBACK_CHAIN slot-based
# role expansion, WORKER_NAME_FALLBACK name-based single next-hop with loops,
# _FALLBACK_FAST_PATH tier-based ladder). All three are now derived from
# WORKER_FALLBACK_CHAIN.
#
# Format: worker_name -> ordered list of fallback workers (chain, not single
# next hop). First element is the preferred next worker; chain continues until
# exhausted. Empty list = terminal (e.g. top-quality Fable).
#
# Validation at import time: no loops, all referenced workers exist.
WORKER_FALLBACK_CHAIN: dict[str, list[str]] = {
    # v0.9.63 (Design B): cost-effective tiered. STRICT tier ordering —
    # each node falls back to STRICTLY HIGHER tier only (DAG, no cycles).
    # Tiers (lower = cheaper, higher = more capable):
    #   T4 cheapest: deepseek-v4-flash (Zen free), deepseek-v4-flash-official
    #   T3 cheap mid: gpt-5.6-luna, gpt-5.6-terra
    #   T2 specialty strong: minimax-m3 (1M ctx, multimodal), grok-4-5-reasoning
    #   T1 premium: claude-opus-5
    #   T0 terminal (sacred top): claude-fable-5
    # Probe-driven defaults:
    #   - dpsk-flash-official: n=10 0% placeholder, DEFAULT ON, ¥0.04/M flat
    #   - dpsk-flash Zen: n=30 0% placeholder, $0 marginal, ANCHOR_ENABLE_DEEPSEEK_V4_FLASH=1
    #   - gpt-5.6-sol: n=30 20% placeholder = HOLD (kept in graph for opt-in)
    #   - ollama-ornith: aimax-local only, env-gated, opt-in
    #
    # Default-ON workers (set by feature flags):
    "claude-fable-5":              [],
    # v0.9.7X-P2: claude-opus-5 removed from WORKER_FALLBACK_CHAIN (see p2_opus5_eval.md).
    # WorkerSpec above (lines 228-239) is dormant until re-added to chains.
    "deepseek-v4-flash":           ["kilo-auto-free", "minimax-m3", "claude-opus-5"],
    "kilo-auto-free":              ["minimax-m3", "claude-opus-5"],
    "minimax-m3":                  ["claude-opus-5"],
    "grok-4-6-reasoning":          ["minimax-m3", "claude-opus-5"],
    "deepseek-v4-pro":             ["minimax-m3", "claude-opus-5"],
    "gpt-5.6-luna":                ["minimax-m3", "claude-opus-5"],
    "gpt-5.6-terra":               ["minimax-m3", "claude-opus-5"],
    "gpt-5.6-sol":                 ["minimax-m3", "claude-opus-5"],
    "deepseek-v4-flash-official":  ["minimax-m3", "claude-opus-5"],
    "claude-haiku-4-5":            ["minimax-m3", "claude-opus-5"],
    "claude-sonnet-5":             ["minimax-m3", "claude-opus-5"],
    "kimi-k3":                     ["minimax-m3", "claude-opus-5"],
    "ollama-ornith-35b":           ["minimax-m3", "claude-opus-5"],
    # Phase 3 FIX-E/B: claude-opus-5 chain (12% err + expensive; HOLD per
    # calibration team 2026-08-15). audit 2026-08-16 (C9): this key was
    # duplicated with identical value — the duplicate is removed.
    "claude-opus-5":               [],
}


_CLAUDE_BAOSI_WORKERS = ("claude-fable-5", "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5")  # audit 2026-08-16 (C6): opus-5 added back — it shares BAOSIAPI_GPT_API_KEY and was the only Claude worker the gate never disabled keyless


_CLAUDE_KEY_ENVS = ("ANCHOR_BAOSIAPI_CLAUDE_API_KEY", "BAOSIAPI_CLAUDE_API_KEY")


def _apply_claude_auth_gate() -> None:
    """Disable baosiapi Claude workers only when NO usable key exists.

    v0.9.63 assumed claude-* needs a separate Claude-group key (GPT key -> 401).
    Verified 2026-08-15: BAOSIAPI_GPT_API_KEY successfully calls claude-fable-5 /
    claude-opus-5. So a worker stays enabled when any of its api_key_envs is set,
    OR a legacy Claude-group key env is set (kept for backward-compat with tests).
    Only when no key at all is available do we disable (fail-closed).
    """
    has_claude_key = any(os.environ.get(k) for k in _CLAUDE_KEY_ENVS)
    for i, worker in enumerate(WORKERS):
        if worker.name in _CLAUDE_BAOSI_WORKERS:
            usable = has_claude_key or any(os.environ.get(k) for k in worker.api_key_envs)
            if not usable:
                WORKERS[i] = replace(
                    worker,
                    enabled=False,
                    notes="No usable baosiapi key; disabled by auth gate (set BAOSIAPI_GPT_API_KEY or BAOSIAPI_CLAUDE_API_KEY)",
                )


_apply_claude_auth_gate()

# audit 2026-08-16 (C6): computed AFTER the auth gate so dashboard reports
# the post-gate (fail-closed) enabled state instead of a stale pre-gate snapshot.
SOURCE_ENABLED_BY_NAME = {w.name: w.enabled for w in WORKERS}


def _validate_fallback_graph() -> None:
    """Validate WORKER_FALLBACK_CHAIN at import: no loops, all workers exist.

    v0.9.53 (P0-4): fails loudly if any chain has a cycle or references an
    unknown worker. Previously the Fable ↔ Sol loop silently caused failover
    to bounce between two workers without making progress on real failures.
    """
    worker_names = {w.name for w in WORKERS}
    errors: list[str] = []
    for src, chain in WORKER_FALLBACK_CHAIN.items():
        if src not in worker_names:
            errors.append(f"unknown source worker {src!r}")
        for dst in chain:
            if dst not in worker_names:
                errors.append(f"{src!r} -> unknown worker {dst!r}")
    # Full DFS, not a one-hop check: Flash→Ornith→M3→Flash must fail too.
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            start = stack.index(node) if node in stack else 0
            errors.append("loop detected: " + " -> ".join(stack[start:] + [node]))
            return
        state[node] = 1
        stack.append(node)
        for dst in WORKER_FALLBACK_CHAIN.get(node, []):
            if dst in worker_names:
                visit(dst)
        stack.pop()
        state[node] = 2

    for src in WORKER_FALLBACK_CHAIN:
        visit(src)
    if errors:
        raise RuntimeError(
            "WORKER_FALLBACK_CHAIN validation failed: " + "; ".join(errors)
        )


_validate_fallback_graph()


# Derived: name -> next-hop (first element of chain) for legacy callers.
# Keep this for back-compat with anchor.failover and any third-party code.
WORKER_NAME_FALLBACK: dict[str, str | None] = {
    name: (chain[0] if chain else None)
    for name, chain in WORKER_FALLBACK_CHAIN.items()
}



def _read_api_key(env_name: str) -> str:
    """Read API key from os.environ. No file fallback — fail-closed."""
    return os.environ.get(env_name, "")


def enabled_workers() -> list[Worker]:
    return [w for w in WORKERS if w.enabled]


def active_n_agents() -> int:
    """TRINITY head N_AGENTS — must equal len(enabled_workers)."""
    return len(enabled_workers())


def workers_for_role(role: str) -> list[Worker]:
    """Return enabled workers that have this role tag, or the full pool.

    audit 2026-08-16 (C9): the old "fallbacks applied" branch walked the
    slot-based FALLBACK_CHAIN — but it iterated workers matching the role,
    and that condition is exactly what made `direct` empty, so the branch was
    dead by construction and always returned enabled_workers(). Role-based
    fallback expansion lives in routing_fallback (WORKER_FALLBACK_CHAIN), not
    here.
    """
    direct = [w for w in enabled_workers() if role in w.role_tags]
    if direct:
        return direct
    return enabled_workers()


# ---------------------------------------------------------------------------
# Oracle + judge panel (v0.9.53 - decoupled to fix self-bias, P0-1).
#
# Background: previously judge_quality.py used Claude Opus 5 to BOTH
# generate the oracle reference answer AND judge the worker response.
# This creates a 13-17pp self-preference confound (Unsolvability Ceiling,
# arXiv 2605.07395). Anchor spec says M3 is oracle; judge panel must be
# cross-family for bias cancellation.
#
# Override:
#   ANCHOR_ORACLE_MODEL     -> str (default: minimax-m3)
#   ANCHOR_JUDGE_PANEL      -> comma-separated model list (default below)
# ---------------------------------------------------------------------------
# Enable prompt caching (cache_control on system messages, 5min TTL).
# Supported by OpenAI v1.1+, Anthropic, and MiniMax M3.
# Saves ~50% on input tokens for shared system prompts.
PROMPT_CACHE_ENABLED: bool = os.environ.get("ANCHOR_PROMPT_CACHE_ENABLED", "0") == "1"

# Semantic cache (qwen3-embedding + vector store).
# Enable with ANCHOR_SEMANTIC_CACHE_ENABLED=1.
# Set ANCHOR_SEMANTIC_CACHE_THRESHOLD (default 0.92) for similarity threshold.
# Set ANCHOR_SEMANTIC_CACHE_DB_URL for pgvector persistence (in-memory default).
SEMANTIC_CACHE_ENABLED: bool = os.environ.get("ANCHOR_SEMANTIC_CACHE_ENABLED", "0") == "1"

ORACLE_MODEL: str = os.environ.get("ANCHOR_ORACLE_MODEL", "minimax-m3")
# Preferred judges. Runtime resolution filters disabled workers, the oracle,
# and the worker being evaluated, then fills from the enabled pool. This keeps
# background judging aligned with the actual production pool when a worker is
# disabled via env (for example DeepSeek Flash during upstream rate limits).
JUDGE_PANEL: list[str] = [
    m.strip() for m in os.environ.get(
        "ANCHOR_JUDGE_PANEL",
        "gpt-5.6-sol,grok-4-6-reasoning,claude-fable-5",
    ).split(",") if m.strip()
]
# Min panel size to record a quality observation. 2 tolerates one judge failure.
MIN_JUDGE_PANEL_SIZE: int = int(os.environ.get("ANCHOR_MIN_JUDGE_PANEL_SIZE", "2"))


def resolve_judge_panel(*, evaluated_worker: str | None = None, limit: int = 3) -> list[str]:
    """Return enabled judges disjoint from oracle and evaluated worker.

    Preferred names come from ``JUDGE_PANEL``. Missing/disabled entries are
    replaced by other enabled workers, ordered to maximize provider-family
    diversity before adding same-family fallbacks.
    """
    enabled = [w for w in WORKERS if w.enabled]
    by_name = {w.name: w for w in enabled}
    excluded = {ORACLE_MODEL}
    if evaluated_worker:
        excluded.add(evaluated_worker)

    preferred = [name for name in JUDGE_PANEL if name in by_name and name not in excluded]
    remainder = [w.name for w in enabled if w.name not in excluded and w.name not in preferred]
    ordered = preferred + remainder

    def family(name: str) -> str:
        worker = by_name[name]
        channel = (worker.channel or "").lower()
        if "claude" in name or "baosiapi" in channel and name.startswith("claude-"):
            return "claude"
        if name.startswith("gpt-"):
            return "gpt"
        if name.startswith("grok-"):
            return "grok"
        if "deepseek" in name:
            return "deepseek"
        if "minimax" in name:
            return "minimax"
        return channel or name.split("-", 1)[0]

    selected: list[str] = []
    seen_families: set[str] = set()
    for name in ordered:
        fam = family(name)
        if fam in seen_families:
            continue
        selected.append(name)
        seen_families.add(fam)
        if len(selected) >= limit:
            return selected
    for name in ordered:
        if name not in selected:
            selected.append(name)
            if len(selected) >= limit:
                break
    return selected

# ---------------------------------------------------------------------------
# Single product worker pool (names, cost-ordered). Pure mode = one AUTO pool.
# Legacy basic/premium/ultra names normalize to auto; not product SKUs.
# ---------------------------------------------------------------------------
_AUTO_COST_ORDER: tuple[str, ...] = (
    "deepseek-v4-flash",         # ¥0 free (Zen)
    "ollama-ornith-35b",         # ¥0 local (env-gated; v0.9.57)
    # v0.9.7X-B1: minimax-m3 removed from default auto pool
    # (audit 2026-08-07: 97.5% wasted routing attempts in auto tier;
    # only wins on <100ms cache hits; 26,284 wasted × 16s avg =
    # 117h compute wasted / 30d). Worker remains usable via forced_worker.
    # Escalation path: dpsk-flash → grok-4-6 → sol → fable (v0.9.7X-P2: opus-5 removed).
    "deepseek-v4-pro",           # ¥0.585/M (baosi架构; default OFF, opt-in)
    "gpt-5.6-luna",              # ¥0.633/M (OpenAI -80% 后, default ON)
    "grok-4-6-reasoning",        # ¥3.62/M
    "gpt-5.6-terra",             # ¥6.34/M (OpenAI -20% 后, default ON)
    "kimi-k3",                   # ¥8.14/M (default OFF, opt-in)
    "gpt-5.6-sol",               # ¥15.47/M
    "claude-fable-5",            # ¥18.55/M (sacred top)
    # opt-in / residual (only when enabled)
    "claude-haiku-4-5",
    "claude-sonnet-5",
)

# Back-compat aliases: all map to the same order.
_TIER_COST_ORDER: dict[str, tuple[str, ...]] = {
    "auto": _AUTO_COST_ORDER,
    "basic": _AUTO_COST_ORDER,
    "premium": _AUTO_COST_ORDER,
    "ultra": _AUTO_COST_ORDER,
}

# Phase 3 FIX-A: unified per-worker cost table (yuan per call, median bucket).
# Single source of truth for cost-aware routing. All callers must use
# `worker_cost(name)` instead of reading WORKER_COST or _AUTO_COST_ORDER directly.
#
# audit 2026-08-16 (F9): the previous hardcoded values claimed to be v8-derived
# but were 22-67x off (opus 0.001583 vs measured 0.107) and mis-ordered workers
# (flash-official priced above opus). The dict below is now a FALLBACK only;
# after `_load_cal_v8` is defined, WORKER_COST_YUAN is rebuilt from the v8
# cost_yuan_avg[worker].medium values (see _build_worker_cost_yuan).
_WORKER_COST_FALLBACK_YUAN: dict[str, float] = {
    "deepseek-v4-flash":            0.0,
    "deepseek-v4-flash-official":   0.0024,  # deepseek 官方人民币 2026-08-17 新价
    "deepseek-v4-pro":              0.0072,  # deepseek 官方人民币 2026-08-17 新价
    "minimax-m3":                   0.0,
    "ollama-ornith-35b":            0.0,
    "claude-haiku-4-5":             0.000226,
    "claude-sonnet-5":              0.000950,
    "claude-opus-5":                0.001583,
    "claude-fable-5":               0.003166,
    "gpt-5.6-luna":                 0.000100,
    "gpt-5.6-terra":                0.000995,
    "gpt-5.6-sol":                  0.002488,
    "grok-4-6-reasoning":           0.000724,
    "grok-4-5-reasoning":          0.000724,  # V6.3: superseded by grok-4-6 but cost retained for calibration
    "kimi-k3":                      0.0,  # not in v8 table; keep 0 fallback (worker default-ON decision in P1-5)
}

# Populated from v8 calibration at the end of module init (see _build_worker_cost_yuan).
WORKER_COST_YUAN: dict[str, float] = dict(_WORKER_COST_FALLBACK_YUAN)


def worker_cost(name: str) -> float:
    """Per-call cost (yuan) for `name`. 0.0 for free / unknown workers.

    Single canonical cost lookup (Phase 3 FIX-A). All routing/scheduling code
    should call this rather than reading WORKER_COST or _AUTO_COST_ORDER
    directly. Unknown workers return 0.0 (not a crash) so the router can
    still surface them in telemetry while skipping them from cost-first
    ordering.
    """
    return float(WORKER_COST_YUAN.get(name, 0.0))

# Absolute max_tokens ceilings (client cannot exceed).
TIER_MAX_TOKENS: dict[str, int] = {
    "auto": 16384,
    "basic": 16384,      # alias → auto
    "premium": 16384,    # alias → auto
    "ultra": 16384,      # alias → auto
    "image-gen": 1024,
}
ABSOLUTE_MAX_TOKENS: int = 32000


def normalize_routing_lane(tier: str | None) -> str:
    """Collapse legacy basic/premium/ultra names to the single `auto` lane."""
    if not tier or tier in {"basic", "premium", "ultra", "auto"}:
        return "auto"
    return tier


def tier_pool_from_config(tier: str) -> tuple[str, ...]:
    """Return enabled worker names for the single auto pool (cost-ordered).

    Phase 3 FIX-B: data-driven pool. Eligibility filter from the v8 calibration:
    for the requested difficulty bucket, a worker is ELIGIBLE if its calibrated
    table_coarse[bucket] >= 0.7 AND the calibration err% < 10 percent.
    Eligible workers are sorted by worker_cost() ascending. Unknown
    workers (not present in the v8 table) are EXCLUDED rather than admitted
    via a 0.5 fallback.
    tier is accepted for back-compat but ignored for chat lanes
    (basic/premium/ultra/auto same pool). image-gen is not handled here.
    Vision queries are constrained to the VISION_CAPABLE set by the caller.
    """
    tier = normalize_routing_lane(tier)
    enabled_names = {w.name for w in WORKERS if w.enabled}
    bucket = _default_difficulty_bucket_for_pool()
    eligible = _eligible_workers_for_bucket(bucket, enabled_names)
    # Boss WS#1 routing policy: keep the zero-marginal-cost pool and M3 in the
    # normal auto lane even when calibration has no row for a newly restored
    # provider. Paid baosiapi workers are not admitted by this policy pool;
    # Opus-5 is selected only by hard-quality rules or overflow fallback.
    free_policy = [
        n for n in ("deepseek-v4-flash", "kilo-auto-free", "minimax-m3")
        if n in enabled_names
    ]
    if eligible:
        eligible = free_policy + [n for n in eligible if n not in free_policy]
    if not eligible:
        order = _TIER_COST_ORDER.get(tier) or _AUTO_COST_ORDER
        order = free_policy + [n for n in order if n not in free_policy]
        return tuple(n for n in order if n in enabled_names)
    return tuple(eligible)


def _default_difficulty_bucket_for_pool() -> str:
    """Bucket for the auto pool eligibility check. Default = medium."""
    return "medium"


def _eligible_workers_for_bucket(bucket: str, enabled_names):
    """Return enabled worker names eligible for bucket, cost-ascending."""
    cal = _load_cal_v8()
    if not cal:
        return []
    coarse = cal.get("table_coarse", {}) or {}
    err_pct = cal.get("err_pct") or cal.get("error_rate_pct") or {}
    pool = []
    seen = set()
    for name, cell in coarse.items():
        if name not in enabled_names or name in seen:
            continue
        score = cell.get(bucket)
        if score is None or score < 0.7:
            continue
        e = err_pct.get(name)
        if e is not None and e >= 10.0:
            continue
        pool.append((worker_cost(name), name))
        seen.add(name)
    pool.sort(key=lambda x: (x[0], x[1]))
    return [n for _, n in pool]


_CAL_V8_CACHE: dict | None = None
_CAL_V8_SIG: tuple | None = None


def invalidate_cal_v8_cache() -> None:
    """Drop the cached v8 calibration so the next call reloads from disk.

    audit 2026-08-16 (A13/F1): the cache previously lived forever — after a
    weekly re-calibration rewrote the JSON, routing kept scoring against the
    stale copy until restart. /admin/calibration/refresh calls this.
    """
    global _CAL_V8_CACHE, _CAL_V8_SIG
    _CAL_V8_CACHE = None
    _CAL_V8_SIG = None


def _cal_v8_signature() -> tuple | None:
    """(mtime_ns, size) of the v8 JSON + raw.jsonl; None if either is missing."""
    from pathlib import Path as _P
    import os as _os
    _base = _P(__file__).resolve().parent.parent.parent / "data" / "calibration"
    _sig = []
    for _name in ("quality_table_v8.json", "raw.jsonl"):
        _p = _base / _name
        try:
            _st = _os.stat(_p)
        except OSError:
            return None
        _sig.append((_st.st_mtime_ns, _st.st_size))
    return tuple(_sig)


def _load_cal_v8() -> dict:
    """Lazy-load the v8 calibration JSON; returns {} on miss.

    Augments the v8 file with an err_pct dict derived from raw.jsonl
    (Phase 3 FIX-B eligibility filter needs per-worker error rate). The
    augmentation is done at load time only; the underlying JSON file is
    unchanged.

    audit 2026-08-16: cache invalidates when the JSON/raw.jsonl mtime or
    size changes, so a re-calibration is picked up without a restart.
    """
    global _CAL_V8_CACHE, _CAL_V8_SIG
    _sig = _cal_v8_signature()
    if _CAL_V8_CACHE is not None and _sig == _CAL_V8_SIG:
        return _CAL_V8_CACHE
    try:
        from pathlib import Path as _P
        import json as _json
        _cal_v8_path = _P(__file__).resolve().parent.parent.parent / "data" / "calibration" / "quality_table_v8.json"
        with open(_cal_v8_path) as _f:
            _CAL_V8_CACHE = _json.load(_f)
        # Derive err_pct from raw.jsonl (single source of truth for v8 cohort).
        _raw_path = _P(__file__).resolve().parent.parent.parent / "data" / "calibration" / "raw.jsonl"
        _err_n: dict = {}
        _err_total: dict = {}
        try:
            with open(_raw_path) as _rf:
                for _line in _rf:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _row = _json.loads(_line)
                    except Exception:
                        continue
                    _w = _row.get("worker")
                    if not _w:
                        continue
                    _err_total[_w] = _err_total.get(_w, 0) + 1
                    if _row.get("error"):
                        _err_n[_w] = _err_n.get(_w, 0) + 1
            _err_pct: dict = {
                _w: round(100.0 * _err_n.get(_w, 0) / _err_total[_w], 2)
                for _w in _err_total
            }
            _CAL_V8_CACHE["err_pct"] = _err_pct
        except Exception:
            _CAL_V8_CACHE["err_pct"] = {}
        _CAL_V8_SIG = _sig  # remember signature only on a successful load
    except Exception:
        _CAL_V8_CACHE = {}
        _CAL_V8_SIG = None
    return _CAL_V8_CACHE


def _build_worker_cost_yuan() -> dict[str, float]:
    """Rebuild WORKER_COST_YUAN from the v8 calibration (audit 2026-08-16 F9).

    v8 \"cost_yuan_avg[worker].medium\" is the authoritative per-call cost
    (yuan, medium bucket). Workers absent from the v8 table keep the
    documented fallback value (kimi-k3/orchid etc.).
    """
    cal = _load_cal_v8()
    costs = dict(_WORKER_COST_FALLBACK_YUAN)
    for _name, _cell in (cal.get("cost_yuan_avg") or {}).items():
        _med = (_cell or {}).get("medium")
        if _med is not None:
            costs[_name] = float(_med)
    return costs


# audit 2026-08-16 (F9): rebuilt from v8 at import time (falls back to the
# documented per-worker values if the calibration file is missing).
WORKER_COST_YUAN = _build_worker_cost_yuan()


def clamp_max_tokens(max_tokens: int | None, tier: str = "auto", worker_max_tokens: int | None = None) -> int:
    """Clamp client/request max_tokens to absolute ceilings (single product).

    If ``worker_max_tokens`` is set, it overrides the client request value
    (still clamped to the absolute ceiling).
    """
    if worker_max_tokens is not None:
        max_tokens = worker_max_tokens
    if max_tokens is None or max_tokens <= 0:
        max_tokens = 1024
    lane = normalize_routing_lane(tier)
    cap = min(
        TIER_MAX_TOKENS.get(lane, TIER_MAX_TOKENS["auto"]),
        ABSOLUTE_MAX_TOKENS,
    )
    return max(1, min(int(max_tokens), cap))


@dataclass(frozen=True)
class RoutingConfig:
    pre_router_classes: tuple[str, ...] = ("code", "math", "chat", "vision", "agent")
    pre_router_confidence_threshold: float = 0.85
    knn_top_k: int = 3
    head_input_dim: int = 1024
    head_n_roles: int = 3
    adapter_latent_dim: int = 32
    cost_budget_cny_per_query: float = 0.05
    latency_budget_ms: int = 5000
    cost_baseline_per_query_usd: float = 0.005
    alpha_pro: float = 0.0
    alpha_balanced: float = 0.01
    alpha_flash: float = 0.5
    max_turns: int = 5
    # v0.9.46 (P1.4 wire): expected monthly query volume for amortizing
    # monthly_fixed_cny of subscription workers (M3 ¥119/mo, baosiapi
    # ¥588/mo shared). 0 = disabled (no amortization).
    # Calibrated from 2026-07-06..2026-07-12 actuals: ~268/day × 30 = 8000/mo.
    expected_monthly_queries: int = 8000


ROUTING = RoutingConfig()


# === 5 fusion modes (the public API) ===================================
FUSION_MODES = {
    "best":     {"workers": "all",        "multi_turn": True,  "alpha": 0.0,
                 "desc": "all enabled + multi-turn, near-Fable-5 quality"},
    "pro":      {"workers": "all",        "multi_turn": False, "alpha": 0.0,
                 "desc": "all enabled, 1-step, quality first"},
    "balanced": {"workers": "mid+hard",   "multi_turn": False, "alpha": 0.01,
                 "desc": "subset of mid+hard tier, 1-step, quality-cost balance"},
    "flash":    {"workers": "free",       "multi_turn": False, "alpha": 0.5,
                 "desc": "free workers only, 0 fixed cost"},
    "cn":       {"workers": "cn",         "multi_turn": False, "alpha": 0.01,
                 "desc": "M3 only, CN optimized"},
}


def verify_runtime() -> dict:
    """Probe each enabled worker's /ping. Returns {name: 'ok' | 'error:...'}."""
    import urllib.request
    import json
    results: dict[str, str] = {}
    for w in enabled_workers():
        try:
            if w.kind == "openai-compat":
                key = _read_api_key(w.api_key_env)
                req = urllib.request.Request(
                    f"{w.base_url}/chat/completions",
                    data=json.dumps({"model": w.model,
                                     "messages": [{"role": "user", "content": "ping"}],
                                     "max_tokens": 1}).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {key}"},
                )
                t = 5
            elif w.kind == "direct":
                # M3 direct
                key = _read_api_key(w.api_key_env) or _read_api_key("MINIMAX_API_KEY_2")
                req = urllib.request.Request(
                    f"{w.base_url}/chat/completions",
                    data=json.dumps({"model": w.model,
                                     "messages": [{"role": "user", "content": "ping"}],
                                     "max_tokens": 1}).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {key}"},
                )
                t = 5
            else:  # opencode-cli
                results[w.name] = "ok (cli, not probed via REST)"
                continue
            with urllib.request.urlopen(req, timeout=t) as resp:
                body = resp.read().decode()
                if '"error"' in body or resp.status >= 400:
                    results[w.name] = f"error: {body[:200]}"
                else:
                    results[w.name] = "ok"
        except Exception as e:
            results[w.name] = f"error: {type(e).__name__}: {str(e)[:200]}"
    return results


if __name__ == "__main__":
    from anchor import __version__ as _ver
    print(f"=== Anchor v{_ver} ===")
    print()
    print(f"workers: {len(WORKERS)} configured, {active_n_agents()} enabled")
    for w in WORKERS:
        flag = "OK " if w.enabled else "OFF"
        print(f"  [{flag}] slot {w.slot:<2} {w.name:20s} {w.channel:14s} "
              f"{w.model:35s} roles={w.role_tags}")
    print()
    print("monthly fixed cost (enabled, no double-counting shared quota):")
    baosiapi_workers = [w for w in enabled_workers() if w.channel == "baosiapi"]
    other = [w for w in enabled_workers() if w.channel != "baosiapi"]
    baosiapi_total = sum(w.monthly_fixed_cny for w in baosiapi_workers) if baosiapi_workers else 588.0
    other_total = sum(w.monthly_fixed_cny for w in other)
    for w in other:
        if w.monthly_fixed_cny > 0:
            print(f"  {w.name:20s} ¥{w.monthly_fixed_cny:.0f}/mo")
    if baosiapi_workers:
        print(f"  baosiapi Claude tier (×{len(baosiapi_workers)} workers, 1 shared ¥588/mo)"
              f"  ¥588/mo (shared)")
    print(f"  {'TOTAL':20s} ¥{baosiapi_total + other_total:.0f}/mo")
    print()
    print("fusion modes:")
    for name, spec in FUSION_MODES.items():
        print(f"  anchor/{name:10s} {spec['desc']}")
    print()
    n = active_n_agents()
    r = ROUTING.head_n_roles
    d = ROUTING.head_input_dim
    print(f"N_AGENTS for TRINITY head = {n}")
    print(f"head shape = ({n} + {r}) x {d} = {(n + r) * d}")
    print()
    print("worker-for-role matrix:")
    for role in ("light", "coding", "cn-agent", "mid-en", "hard", "multi"):
        ws = workers_for_role(role)
        names = [w.name for w in ws]
        print(f"  {role:10s} -> {names}")
