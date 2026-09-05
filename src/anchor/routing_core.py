"""Routing pipeline (select worker → call → record).

Submodules (v0.9.52-p5):
  - routing_fallback: fallback ladder + SLA quarantine helper
  - routing_models: PUBLIC_MODEL_IDS / aliases / _pick_tier_from_model
  - routing_decompose: cheap-try / judge / decompose

This module keeps `_call_worker`, `_route_tier`, pareto resolve, and
text helpers; re-exports submodule symbols for server/tests.
"""
from __future__ import annotations
import asyncio
import os
import re
import time
from typing import TYPE_CHECKING, Optional

# Local helpers

# Routing dependencies (extracted from server.py imports)
from fastapi import HTTPException


from anchor.cost import CapHitError, guard, append_log
from anchor.db import ainsert_query

# v0.9.46f: judge_batch is opt-in (graceful fallback if anchor.judge
# import fails — typically because real Opus API not configured).
try:
    from anchor.judge import judge_batch as _judge_batch
except Exception:
    _judge_batch = None
from anchor.sacred_guard import (
    record_sacred_spend as _record_sacred,
    check_sacred_guard as _sacred_check,
    SOFT_CAP_YUAN as _SACRED_SOFT,
)
from anchor.workers import SACRED
from anchor.fusion_modes import TIER_HARD_RULES
from anchor.head import predict as _head_predict

if TYPE_CHECKING:
    from anchor.api_models import Query, Response


# === Module-level lookup tables ===

# === Re-exports: fallback + public models (submodules) ===

# noqa: F401 — intentional re-exports consumed by server.py / tests.
from anchor.routing_fallback import (  # noqa: F401
    _max_fallback_attempts,  # noqa: F401
    _maybe_pro,  # noqa: F401
    _fallback_candidates,  # noqa: F401
    _recent_sla_violations,  # noqa: F401
    build_fallback_fast_path,  # noqa: F401
    _DEEPSEEK_PRO_OPT_IN,  # noqa: F401
)
from anchor.routing_models import (  # noqa: F401
    PUBLIC_MODEL_IDS,  # noqa: F401
    MODEL_ALIASES,  # noqa: F401
    _MODEL_TO_TIER,  # noqa: F401
    _WORKER_TO_TIER,  # noqa: F401
    _pick_tier_from_model,  # noqa: F401
)

# Bound on this module so tests can monkeypatch rc._FALLBACK_FAST_PATH
_FALLBACK_FAST_PATH = build_fallback_fast_path()
_AUTO_FAST = _FALLBACK_FAST_PATH["auto"]

# Kilo is an explicitly enabled, zero-cost last-resort peer.  Keep the gate
# here (rather than baking it into the calibration table) so the fallback
# remains available when the data-driven eligible-worker set is stale.
_KILO_OPT_IN = os.environ.get("ANCHOR_ENABLE_KILO", "1").strip().lower() not in {
    "0", "false", "no", "off",
}


def _kilo_opt_in() -> bool:
    """Return whether the Kilo free fallback is enabled."""
    return _KILO_OPT_IN


def _maybe_kilo(pool: tuple) -> tuple:
    """Append Kilo as the final fallback when its feature gate is enabled."""
    if _kilo_opt_in() and "kilo-auto-free" not in pool:
        return tuple(pool) + ("kilo-auto-free",)
    return tuple(pool)


def _retry_after_seconds(exc: Exception) -> int | None:
    """Extract Retry-After from an OpenAI-compatible 429 exception."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        match = re.search(r"Retry-After\s*[:=]\s*([^,; ]+)", str(exc), re.I)
        value = match.group(1) if match else None
    if value is None:
        return None
    try:
        return max(0, int(float(str(value).strip())))
    except (TypeError, ValueError):
        return None


def _retry_after_from_answer(answer: str | None) -> int | None:
    """Read the stable retry marker emitted by `_call_worker`."""
    match = re.search(r"retry_after_s=(\d+)", answer or "")
    return int(match.group(1)) if match else None




# --- classify_query ---
def _compute_cost_yuan(w, usage: dict) -> float:
    """Pure cost calc from a Worker-like object + usage dict.

    v1.0.x E9 audit 2026-08-17: cached-input discount applied. Mirrors
    `anchor.clients.base._cost_yuan_from_usage` semantics but stays a
    local helper so routing_core stays self-contained and unit-testable
    without spinning up the client factory.

    - `cost_in_cached` <= 0 → falls back to 10% of `cost_in`.
    - cached tokens are subtracted from prompt_tokens; remainder billed
      at full price.
    """
    in_t = usage.get("prompt_tokens", 0) or 0
    out_t = usage.get("completion_tokens", 0) or 0
    _cached_t = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    _in_uncached = max(in_t - _cached_t, 0)
    _cached_cost_in = getattr(w, "cost_in_cached", 0.0) or 0.0
    if _cached_cost_in <= 0:
        _cached_cost_in = w.cost_in * 0.1
    return (
        w.cost_in * _in_uncached / 1_000_000
        + _cached_cost_in * _cached_t / 1_000_000
        + w.cost_out * out_t / 1_000_000
    )


def classify_query(query: str, query_type: str) -> str:
    """Use real classifier (regex-based). query_type is facade hint, may be overridden."""
    from anchor.classifier import classify as _real_classify
    if query_type and query_type != "chat":
        return query_type
    return _real_classify(query)



# --- _call_worker ---
def _worker_call_timeout(worker_name: str) -> float:
    """Per-worker upstream call timeout (seconds).

    Grok reasoning often spends 8–11s even on short answers; cap primary
    attempt so quality-preserving fallback (sol/M3) can win the race for p95.
    Override: ANCHOR_WORKER_TIMEOUT_<NAME> or ANCHOR_WORKER_TIMEOUT_DEFAULT.
    """
    import os as _os
    name = getattr(worker_name, "name", worker_name) or ""
    env_key = "ANCHOR_WORKER_TIMEOUT_" + str(name).upper().replace("-", "_").replace(".", "_")
    raw = _os.environ.get(env_key) or _os.environ.get("ANCHOR_WORKER_TIMEOUT_DEFAULT")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    # Quality-first: give steel room to cut. Failures still escalate.
    defaults = {
        "grok-4-6-reasoning": 25.0,  # reasoning often 8–20s; room before escalate
        "claude-fable-5": 50.0,
        "gpt-5.6-sol": 60.0,  # baosi GPT hard answers often >45s
        "minimax-m3": 25.0,
        "deepseek-v4-flash": 20.0,
        "deepseek-v4-pro": 25.0,
    }
    return float(defaults.get(str(name), 15.0))


def _grok_race_enabled() -> bool:
    """Parallel Grok↔Sol race: latency hedge only.

    Default OFF. Quality-first still holds via sequential path:
    pick Grok (mid-upper) / Sol (hard) / Fable (sacred) without hesitation;
    on failure/timeout, quality-preserving fallback escalates (not free-first).
    Parallel race double-bills when both complete — opt-in via ANCHOR_GROK_RACE=1.
    """
    import os as _os
    raw = _os.environ.get("ANCHOR_GROK_RACE")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    return False


def _grok_race_delay_s() -> float:
    """Seconds to wait on primary before starting race peer (opt-in thrift).

    Default 2.5s: if Grok finishes quickly, skip peer and avoid double-bill.
    Override: ANCHOR_GROK_RACE_DELAY_S (0 = fire peer immediately).
    """
    import os as _os
    raw = _os.environ.get("ANCHOR_GROK_RACE_DELAY_S")
    if raw is None or str(raw).strip() == "":
        return 2.5
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 2.5


def _grok_race_peer() -> str | None:
    """Peer for Grok hedge race: prefer sol, else M3."""
    from anchor.config import WORKERS as _W
    enabled = {w.name for w in _W if w.enabled}
    for cand in ("gpt-5.6-sol", "minimax-m3", "deepseek-v4-flash"):
        if cand in enabled:
            return cand
    return None


async def _call_worker(worker_name: str, query: str, max_tokens: int = 200,
                       messages: list | None = None,
                       tools: list | None = None,
                       tool_choice: str | dict | None = None,
                       extra_kw: dict | None = None) -> tuple[str, float, int, bool, list, dict, Optional[str]]:
    """Call worker via existing clients.

    Returns (answer, cost_yuan, latency_ms, vendor_placeholder_flag, tool_calls,
    usage, finish_reason, cascade_triggered, cascade_reason).

    `finish_reason` is the upstream OpenAI-compat value (stop/length/tool_calls/
    content_filter) when the client reports one, else None. T-AUDIT-05 surfacing
    so the gateway can report truncation honestly (previously hardcoded "stop").

    `extra_kw` is merged into the OpenAI-compat chat() call kwargs. Used for
    stop / top_p / frequency_penalty / presence_penalty / response_format
    passthrough (T-AUDIT-04 compat MEDIUM).
    """
    """Call worker via existing clients. Returns (answer, cost_yuan, latency_ms, vendor_placeholder_flag, tool_calls, usage).

    v0.9.8: accept `messages` (full OpenAI messages) for vision/multimodal support.
    v0.9.19: returns 4th bool = True if upstream returned vendor placeholder
             (baosiapi 30-char "模型未返回可见内容...") — caller should NOT judge_score this.
    v0.9.46h: forwards `tools`/`tool_choice` to OpenAI-compatible clients; the 5th
              tuple element carries the OpenAI-format tool_calls list back
              (empty list when the worker didn't emit any function call).
    """
    _vp = False
    _tool_calls: list = []
    t0 = time.time()
    from anchor.cascade_signals import should_cascade as _sc, cascade_score as _csc
    try:
        from anchor.clients.factory import build_client
        from anchor.config import WORKERS
        # worker_name may be Worker object (from predict()) or str
        if hasattr(worker_name, "name"):
            w = worker_name
            worker_name = w.name
        else:
            w = next((x for x in WORKERS if x.name == worker_name), None)
        if w is None:
            return f"[stub:{worker_name}] echo: {query[:50]}", 0.001, int((time.time()-t0)*1000), False, [], {}, None, False, None
        client = build_client(w)
        # v0.9.8: pass full messages if provided (vision/image_url preserved),
        # otherwise legacy single-user-text fallback.
        call_messages = messages if messages else [{"role": "user", "content": query}]
        # v0.9.46h: pass tools/tool_choice through to OpenAI-compatible clients.
        # Gemini (native) ignores them silently; agnes chat-only ignores them.
        _chat_kw: dict = {"max_tokens": max_tokens, "temperature": 0.0}
        # T-AUDIT-04: max_completion_tokens (o1/o3) supersedes max_tokens when
        # both are present. Some upstream APIs reject max_tokens for o-series.
        if extra_kw and extra_kw.get("max_completion_tokens"):
            _chat_kw["max_completion_tokens"] = extra_kw["max_completion_tokens"]
            _chat_kw.pop("max_tokens", None)
        if tools:
            _chat_kw["tools"] = tools
            if tool_choice is not None:
                _chat_kw["tool_choice"] = tool_choice
            elif tool_choice is None:
                _chat_kw.setdefault("tool_choice", "auto")
        # Forward remaining extras (stop, top_p, frequency_penalty,
        # presence_penalty, response_format, parallel_tool_calls).
        if extra_kw:
            for _k, _v in extra_kw.items():
                if _k in ("messages", "max_completion_tokens"):
                    continue  # already handled
                if _v is not None:
                    _chat_kw[_k] = _v
        r = await asyncio.wait_for(
            client.chat(call_messages, **_chat_kw),
            timeout=_worker_call_timeout(worker_name),
        )
        content = r.get("content", "")
        # v0.9.46h: surface tool_calls when worker emits them.
        _raw_tcs = r.get("tool_calls") or []
        # OpenAI SDK returns objects; clients/base.py normalizes to OpenAI-shape dicts.
        for _tc in _raw_tcs:
            try:
                if hasattr(_tc, "model_dump"):
                    _tool_calls.append(_tc.model_dump())
                elif isinstance(_tc, dict):
                    _tool_calls.append(_tc)
            except Exception as _tc_e:
                # S11 (SRE audit): bad tool_call shape upstream; log so
                # we can detect vendor regressions instead of silently dropping.
                import logging as _lg_tc
                _lg_tc.warning("TOOL_CALL_DROP worker=%s err=%s", worker_name, _tc_e)
        usage = r.get("usage", {})
        cost = _compute_cost_yuan(w, usage)
        # v0.9.19: detect vendor placeholder leaked through (defense in depth)
        if content and any(p in content for p in ("模型未返回可见内容", "I cannot provide", "Sorry, I cannot", "I’m sorry")):
            _vp = True
            try:
                from anchor.cooldown import trip as _trip_vp
                _trip_vp(worker_name, reason="vendor_placeholder")
            except Exception as _vp_trip_e:
                # S11 (SRE audit): cooldown trip failures mean we can't cool
                # the worker — bad vendor responses will keep getting routed here.
                import logging as _lg_vpt
                _lg_vpt.warning("VP_TRIP_SKIP worker=%s err=%s", worker_name, _vp_trip_e)
        _cascade = _sc(content) if content else False
        _cascade_reason: str | None = None
        if _cascade:
            _cs = _csc(content)
            if _cs == 0.0:
                from anchor.cascade_signals import is_truncated_response as _itr, has_vendor_placeholder as _hvp
                if _itr(content):
                    _cascade_reason = "truncated"
                elif _hvp(content):
                    _cascade_reason = "placeholder"
                else:
                    _cascade_reason = "low_quality"
            else:
                _cascade_reason = "low_quality"
        return content, cost, int((time.time() - t0) * 1000), _vp, _tool_calls, dict(usage) if usage else {}, r.get("finish_reason"), _cascade, _cascade_reason
    except Exception as e:
        # Quick-win 2026-07-26: recognize KeyPool.AllKeysRateLimited
        # specifically so we can propagate retry_after_s all the way
        # to the HTTP layer for HTTP 503 + Retry-After semantics.
        try:
            from anchor.clients.key_pool import AllKeysRateLimited as _AKRL_kp
        except Exception:
            _AKRL_kp = None  # type: ignore[assignment]
        try:
            from anchor.cooldown import trip as _trip, is_rate_limit_error as _isrl
            err_type = type(e).__name__
            err_str = str(e).lower()
            # v0.9.33: trip cooldown on timeout/connection errors too (dpsk pattern).
            # Without this, M3 absorbs all fallback traffic when dpsk upstream flaps.
            _is_transient = err_type in ("APITimeoutError", "TimeoutError", "APIConnectionError") or                              "timeout" in err_str or "connection" in err_str
            if _isrl(e) or _is_transient:
                _trip(worker_name, reason=err_type)
        except Exception as _trip_e:
            # S11 (SRE audit): worker cooldown trip failed; subsequent
            # requests will keep getting routed to a failing worker.
            import logging as _lg_trip
            _lg_trip.warning("WORKER_TRIP_SKIP worker=%s err=%s", worker_name, _trip_e)
        em = str(e)
        _retry_after_s = _retry_after_seconds(e)
        # v0.9.19: m3._call raises NotFoundError on vendor placeholder; detect via message.
        if "vendor placeholder" in em or "模型未返回可见内容" in em:
            _vp = True
            try:
                from anchor.cooldown import trip as _trip_vp
                _trip_vp(worker_name, reason="vendor_placeholder")
            except Exception as _vp2_e:
                # S11 (SRE audit): cooldown trip on vendor placeholder failed;
                # log so operators can see why repeated vendor leaks persist.
                import logging as _lg_vp2
                _lg_vp2.warning("VP_ERR_TRIP_SKIP worker=%s err=%s", worker_name, _vp2_e)
        # v0.9.52-p3: log full error message before truncation so operators
        # can diagnose root cause (e.g. "invalid role: function" vs "n > 1").
        # Previously only em[:60] was returned to the user with no log record.
        import logging as _lg_err
        _lg_err.warning("WORKER_ERR worker=%s err_type=%s full_msg=%s",
                        worker_name, type(e).__name__, em)
        # Quick-win 2026-07-26: special-case AllKeysRateLimited so the server
        # can detect it and emit HTTP 503 + Retry-After. We embed the
        # retry_after_s in the answer string with a stable marker that
        # server.py keys off; the standard classifier still classifies it
        # as "quota_exhausted" via the "Token Plan / all keys" path.
        if _AKRL_kp is not None and isinstance(e, _AKRL_kp):
            _answer = f"[error:{worker_name}:AllKeysRateLimited] retry_after_s={e.retry_after_s}; {em[:160]}"
        elif _retry_after_s is not None and _retry_after_s > 0:
            # t_ea9c35dc: surface the upstream Retry-After header so the
            # fallback walker can pace itself instead of free-fall retrying.
            _answer = f"[error:{worker_name}:{type(e).__name__}] retry_after_s={_retry_after_s}; {em[:200]}"
        else:
            _answer = f"[error:{worker_name}:{type(e).__name__}] {em[:200]}"
        # Keep enough of the upstream error for fallback reason classification.
        # 200 chars is enough for "Token Plan 用量上限" / 429 / timeout markers.
        return _answer, 0.0, int((time.time()-t0)*1000), _vp, [], {}, "error", False, None  # T-AUDIT-05: surface error path with finish_reason="error" (not OpenAI-canonical, but distinguishable from "stop")



# === Re-exports: pareto / filters / outcome (routing_pareto) ===

# noqa: F401 — re-exports consumed by server.py / tests.
from anchor.routing_pareto import (  # noqa: F401
    _filter_quarantined_workers,  # noqa: F401
    _is_tool_query,  # noqa: F401
    _filter_chat_only_workers,  # noqa: F401
    _resolve_pareto_worker,  # noqa: F401
    _annotate_pareto_meta,  # noqa: F401
    _TOOL_QUERY_MARKERS,  # noqa: F401
)




# === Re-exports: message normalization (routing_text) ===

# noqa: F401 — re-exports consumed by server.py / tests.
from anchor.routing_text import (  # noqa: F401
    _messages_have_image_url,  # noqa: F401
    _extract_user_text,  # noqa: F401
    _responses_content_to_text,  # noqa: F401
    normalize_openai_messages,  # noqa: F401
)

def _classify_fallback_reason(answer: str | None, vendor_placeholder: bool = False) -> str:
    """Classify a worker failure for metrics / policy decisions."""
    text = (answer or "").lower()
    if vendor_placeholder:
        return "vendor_placeholder"
    # Plan / quota exhaustion must be checked before generic 429 detection.
    if (
        "token plan" in text
        or "用量上限" in text
        or "quota" in text
        or "2056" in text
        or ("all keys failed" in text and "429" in text)
    ):
        return "quota_exhausted"
    if "ratelimiterror" in text or "rate_limit" in text or "429" in text:
        return "rate_limit"
    if "timeouterror" in text or "apitimeouterror" in text or "timeout" in text:
        return "timeout"
    if "apiconnectionerror" in text or "connectionerror" in text or "connection" in text:
        return "connection"
    if "badrequesterror" in text or "bad_request" in text or "error code: 400" in text:
        return "bad_request"
    if not answer:
        return "empty"
    return "unknown"


# --- _route_tier ---
def _cheapest_non_sacred_enabled() -> str:
    """audit 2026-08-16 (E7): cheapest ENABLED, non-sacred worker for the
    sacred-cap fallback. The old hardcoded gpt-5.6-sol target is disabled
    (FIX-E HOLD) — routing to it errored and the fallback intent was lost."""
    try:
        from anchor.config import WORKERS as _W_SACRED, worker_cost as _wc_sacred
        _pool = [w.name for w in _W_SACRED if w.enabled and w.name not in SACRED]
        if _pool:
            return min(_pool, key=lambda n: _wc_sacred(n))
    except Exception:
        pass
    return "deepseek-v4-flash"  # last-resort fallback

async def _route_tier(tier: str, q: Query, max_tokens: int = 200, forced_worker=None, vision_only: bool = False, extra_kw: dict | None = None, session_id: str | None = None, api_key: str | None = None, select_only: bool = False) -> Response:
    t0 = time.time()
    from anchor.config import normalize_routing_lane as _nrl
    tier = _nrl(tier)  # single-product: collapse basic/premium/ultra → auto
    # v0.9.54 (Tier 1.3): mint a per-query trace id and bind it to the
    # logging context so every log record + cost log + judge call lines up.
    # Set BEFORE any code path (worker call, fallback chain) so failure
    # inside one branch still carries the same id through the next one.
    _trace_token = None
    try:
        from anchor.logging_setup import trace_id_var as _tiv, new_trace_id as _nti
        _trace_token = _tiv.set(_nti())
    except Exception:
        pass
    qt = classify_query(q.query, q.query_type)
    # v0.9.19: side-channel for vendor_placeholder flag (Response is pydantic, can't setattr)
    _route_tier_extra: dict = {}
    rule = TIER_HARD_RULES.get(tier)
    if forced_worker is None:
        forced_worker = rule(qt, q.query) if rule else None
    # Lean pool: never force a disabled worker (sonnet/haiku/opus opt-in).
    if forced_worker:
        from anchor.config import WORKERS as _W_EN
        _fw = next((x for x in _W_EN if x.name == forced_worker), None)
        if _fw is None or not _fw.enabled:
            forced_worker = None
    try:
        guard()
    except CapHitError as e:
        raise HTTPException(status_code=503, detail=str(e))
    # v0.9.47: filter quarantined workers from the tier pool BEFORE selection.
    from anchor.release.circuit_breaker import is_quarantined as _is_q_tier
    from anchor.fusion_modes import TIER_POOL_ORDER as _tpo_q
    _tier_pool = [w for w in _tpo_q.get(tier, _tpo_q["auto"]) if not _is_q_tier(w)]
    if not _tier_pool:
        _tier_pool = list(_tpo_q.get(tier, _tpo_q["auto"]))
    if forced_worker:
        if _is_q_tier(forced_worker):
            forced_worker = None
        else:
            chosen = forced_worker
            score = 1.0
    if not forced_worker:
        chosen, score = _head_predict(qt, tier, prompt_len=q.prompt_len, budget=q.budget)
        # v0.9.51: Pool A->C only when Pool C (dpsk-pro) is enabled + in pool
        from anchor.quota import maybe_quota_fallback as _mqf
        _mqf_chosen = _mqf(chosen, tier, _tier_pool)
        if _mqf_chosen != chosen:
            chosen = _mqf_chosen
            score = 1.0
    if _is_q_tier(chosen):
        for _w in _tier_pool:
            if not _is_q_tier(_w):
                chosen = _w
                score = 1.0
                break
    # v0.9.46h-p1: detect tools array presence (not just query text).
    # When tools are present, ONLY override if the chosen worker is chat_only
    # (e.g. agnes) which cannot handle tool calls. Workers like deepseek-v4-flash
    # and minimax-m3 handle tools natively. Do NOT trigger the pool walk purely
    # because tools are present - that would replace a capable worker with the
    # first non-chat_only entry in the pool.
    _has_tools = bool(q.tools)

    # t_2fd1d7f4 (Boss directive): BaosiAPI upstream quota is shared across
    # all baosiapi-* channels. When the global slot is at/near the cap, shunt
    # the request to Tier 1 free pool so we never *cause* a 429 by stacking
    # requests on the upstream. Honor `forced_worker` (hard rule) so explicit
    # design/vision choices still reach the chosen worker.
    if not forced_worker:
        try:
            from anchor.config import WORKERS as _W_SHUNT
            from anchor.baosi_concurrency import should_shunt as _baosi_shunt
            _shunt_on = _baosi_shunt()
            _ch_w = next((x for x in _W_SHUNT if x.name == chosen), None)
            _ch_is_baosi = bool(_ch_w and getattr(_ch_w, "channel", "") and str(_ch_w.channel).startswith("baosiapi"))
            if _shunt_on and _ch_is_baosi:
                _shunt_target = None
                for _alt in _tier_pool:
                    _aw = next((x for x in _W_SHUNT if x.name == _alt), None)
                    if not _aw or not _aw.enabled:
                        continue
                    _aw_ch = getattr(_aw, "channel", "") or ""
                    if _aw_ch.startswith("baosiapi"):
                        continue
                    _shunt_target = _alt
                    break
                if _shunt_target:
                    import logging as _lg_shunt
                    _lg_shunt.warning(
                        "BAOSI_SHUNT chosen=%s -> %s (inflight at cap, baosi deferral)",
                        chosen, _shunt_target,
                    )
                    chosen = _shunt_target
                    score = 1.0
        except Exception as _shunt_e:
            import logging as _lg_shunt2
            _lg_shunt2.getLogger("anchor.routing_core").warning(
                "BAOSI_SHUNT check failed (degrading to head decision): %s", _shunt_e)

    if chosen and (_is_tool_query(q.query or "") or _has_tools):
        from anchor.config import WORKERS as _W_CHAT
        _chosen_worker = next((w for w in _W_CHAT if w.name == chosen), None)
        if _chosen_worker and getattr(_chosen_worker, "chat_only", False):
            import logging as _lg_chat
            _lg_chat.warning(
                "TOOL_AWARE_OVERRIDE chosen=%s tools=%s query=%s -> routing to tool-capable",
                chosen, bool(q.tools), (q.query or "")[:50],
            )
            # Walk TIER_POOL_ORDER for the same tier to find a tool-capable worker
            from anchor.fusion_modes import TIER_POOL_ORDER as _tpo_chat
            for _w_name in _tpo_chat.get(tier, _tpo_chat["auto"]):
                _w = next((x for x in _W_CHAT if x.name == _w_name), None)
                if _w and not getattr(_w, "chat_only", False):
                    chosen = _w_name
                    break
    # Cooldown-aware fallback (Day 17 / Fable 5 decision #5):
    # If chosen worker is cooling, walk down the tier pool (cost order) and pick the
    # next non-cooling one. Cap retries to avoid loops on full-pool outages.
    from anchor.cooldown import is_cooling as _is_cool
    attempted = [chosen]
    if not forced_worker and (_is_cool(chosen) or _is_q_tier(chosen)):
        for w in _tier_pool:
            if w == chosen:
                continue
            if _is_cool(w):
                continue
            chosen = w
            attempted.append(w)
            break
    # v0.9.16 (Fable 5 #6): hard cap ¥10/session → auto-fallback fable→opus
    # v0.9.18: tier-aware fallback (basic has no opus, fall to m3)
    if chosen in SACRED:
        from anchor.sacred_guard import resolve_sacred_bucket as _rsb
        _sid_local = _rsb(api_key=api_key, client_session_id=session_id)
        _guard = _sacred_check(_sid_local)
        if _guard.get("fallback_required"):
            import logging as _lg_guard
            # audit 2026-08-16 (E7): the fallback target was the hardcoded
            # gpt-5.6-sol, which is disabled (FIX-E HOLD) — routing to it errored
            # and the "move off sacred" intent silently degraded to whatever the
            # generic chain picked. Resolve the cheapest enabled NON-sacred
            # worker at runtime instead.
            _alt_sacred = _cheapest_non_sacred_enabled()
            _lg_guard.warning("SACRED_GUARD fallback_required sid=%s total=¥%.4f chosen=%s -> %s",
                              _sid_local, _guard["session_sacred_yuan"], chosen, _alt_sacred)
            chosen = _alt_sacred
    # T-AUDIT-04 (arch C6): per-query budget enforcement. If chosen worker's
    # estimated cost > cost_budget_cny_per_query, walk TIER_POOL_ORDER and pick
    # the cheapest worker that fits the budget. Defers to head decision when
    # nothing fits (preserves ship behavior — better to overspend than to 503).
    #
    # Pure-mode / hard-rules: never budget-downgrade an explicit forced_worker
    # (design→fable, vision→m3, direct alias). Otherwise ¥0.05 default silently
    # demotes fable (est ≈¥0.06) back to flash/M3 and breaks quality floors.
    # budget<=0 disables the cap entirely.
    try:
        if forced_worker is None:
            from anchor.config import ROUTING as _ROUTING
            from anchor.config import WORKERS as _W_BUDGET
            import os as _os_b
            _budget_raw = _os_b.environ.get("ANCHOR_COST_BUDGET_CNY")
            if _budget_raw is not None and str(_budget_raw).strip() != "":
                _budget = float(_budget_raw)
            else:
                _budget = float(getattr(_ROUTING, "cost_budget_cny_per_query", 0.05))
            _chosen_w = next((w for w in _W_BUDGET if w.name == chosen), None)
            if _chosen_w is not None and _budget > 0:
                from anchor.fusion_modes import TIER_POOL_ORDER as _tpo_b
                # audit 2026-08-16 (B6): the old expression mixed USD/1M into
                # USD/1K and compared it to a yuan-per-query budget — never
                # fired at the ¥0.05 default and mis-fired ~22x if tuned.
                # Use the canonical per-call yuan cost (v8-derived).
                from anchor.config import worker_cost as _wc_budget
                _chosen_cost_est = _wc_budget(chosen)
                if _chosen_cost_est > _budget:
                    _lg_b = __import__("logging").getLogger("anchor.routing_core")
                    _lg_b.warning(
                        "BUDGET_OVERRIDE chosen=%s est=¥%.4f > budget=¥%.4f; walking pool for cheaper",
                        chosen, _chosen_cost_est, _budget,
                    )
                    for _alt in _tpo_b.get(tier, _tpo_b.get("auto", ())):
                        if _alt == chosen:
                            continue
                        _alt_w = next((w for w in _W_BUDGET if w.name == _alt), None)
                        if _alt_w is None:
                            continue
                        _alt_cost = _wc_budget(_alt)
                        if _alt_cost <= _budget:
                            chosen = _alt
                            _lg_b.info("BUDGET_OVERRIDE -> %s (est=¥%.4f <= budget)", _alt, _alt_cost)
                            break
    except Exception as _be:
        # Budget enforcement is best-effort; never break the request.
        import logging as _lg_be
        _lg_be.getLogger("anchor.routing_core").warning(
            "budget check failed (degrading to no-cap): %s", _be)

    # Final guard: chosen must be enabled; else cheapest in tier pool.
    from anchor.config import WORKERS as _W_CH, enabled_workers as _en_ch
    _cw = next((x for x in _W_CH if x.name == chosen), None)
    if _cw is None or not _cw.enabled:
        _pool_ok = [w for w in _tier_pool if any(x.name == w and x.enabled for x in _W_CH)]
        chosen = _pool_ok[0] if _pool_ok else ( _en_ch()[0].name if _en_ch() else chosen )

    if select_only:
        # v0.9.51: routing decision only (no worker call). Used by SSE path
        # to avoid double-billing (select then stream once).
        #
        # v0.9.76: also emit the *full* fallback chain so server.py _gen
        # can retry cascade when the primary stream() fails (e.g.
        # dpsk-flash-zen-429 → minimax-m3 → luna). Without this, stream
        # callers got an SSE error event and the client (opencode SDK)
        # reported UnknownError instead of silently cascading.
        from anchor.api_models import Response as _RespSelect
        # Reuse the same candidate order chat mode walks (line 838):
        # skip quarantined / cooling workers, dedup with chosen at head.
        _fc: list[str] = []
        for _w in _fallback_candidates(tier, chosen, _tier_pool, vision_only=vision_only):
            if _w in ("", chosen) or _w in _fc:
                continue
            try:
                from anchor.cooldown import is_cooling as _isc_fc
                if _isc_fc(_w):
                    continue
            except Exception:
                pass
            _fc.append(_w)
        return _RespSelect(
            answer="",
            worker=chosen,
            cost_yuan=0.0,
            latency_ms=0,
            confidence=float(locals().get("score") or 1.0),
            tier=tier,
            primary_worker=chosen,
            fallback_from=None,
            fallback_chain=[chosen] + _fc,
        )
    primary_worker = chosen
    fallback_from = None
    fallback_reason = None
    primary_success = None
    # Grok hedge race: fire peer in parallel to cut p95 when reasoning is slow.
    _race_peer = None
    if (
        chosen == "grok-4-6-reasoning"
        and _grok_race_enabled()
        and not (q.tools or vision_only)
        and not select_only
    ):
        _race_peer = _grok_race_peer()
        if _race_peer == chosen:
            _race_peer = None
    if _race_peer:
        import logging as _lg_race
        _lg_race.info("GROK_RACE primary=%s peer=%s", chosen, _race_peer)

        async def _safe_call(_wname: str):
            try:
                return await _call_worker(
                    _wname, q.query, max_tokens=max_tokens, messages=q.messages,
                    tools=q.tools, tool_choice=q.tool_choice, extra_kw=extra_kw,
                )
            except Exception as _e:
                _lg_race.warning("GROK_RACE call_fail worker=%s err=%s", _wname, _e)
                return (
                    f"[error:{type(_e).__name__}]", 0.0,
                    int((time.time() - t0) * 1000), False, [], {}, None, False, None,
                )

        def _is_good(_res) -> bool:
            if not _res:
                return False
            _ans = _res[0] or ""
            _vp = bool(_res[3]) if len(_res) > 3 else False
            _tcs = _res[4] if len(_res) > 4 else []
            return (bool(_ans) and not str(_ans).startswith("[error:") and not _vp) or bool(_tcs)

        _t_g = asyncio.create_task(_safe_call(chosen), name="race-grok")
        _t_p = None
        _race_delay = _grok_race_delay_s()
        _winner = None
        _winner_name = None
        try:
            # Delayed hedge: primary head-start; skip peer if already good (thrift).
            if _race_delay > 0:
                try:
                    await asyncio.wait_for(asyncio.shield(_t_g), timeout=_race_delay)
                except asyncio.TimeoutError:
                    pass
                if _t_g.done() and not _t_g.cancelled():
                    try:
                        _early = _t_g.result()
                    except Exception:
                        _early = None
                    if _is_good(_early):
                        _winner, _winner_name = _early, chosen
                        _lg_race.info(
                            "GROK_RACE primary_fast worker=%s delay_s=%.2f (peer skipped)",
                            chosen, _race_delay,
                        )
            if _winner is None:
                _t_p = asyncio.create_task(_safe_call(_race_peer), name="race-peer")
                _done, _pending = await asyncio.wait(
                    {_t_g, _t_p}, return_when=asyncio.FIRST_COMPLETED,
                )
                for _task in _done:
                    _name = chosen if _task is _t_g else _race_peer
                    if _task.cancelled():
                        continue
                    try:
                        _res = _task.result()
                    except Exception:
                        continue
                    if _is_good(_res):
                        _winner, _winner_name = _res, _name
                        break
                if _winner is None and _pending:
                    _done2, _ = await asyncio.wait(_pending, return_when=asyncio.ALL_COMPLETED)
                    for _task in list(_done) + list(_done2):
                        _name = chosen if _task is _t_g else _race_peer
                        if _task.cancelled():
                            continue
                        try:
                            _res = _task.result()
                        except Exception:
                            continue
                        if _is_good(_res):
                            _winner, _winner_name = _res, _name
                            break
                        if _winner is None:
                            _winner, _winner_name = _res, _name
                else:
                    for _task in _pending:
                        _task.cancel()
                    if _pending:
                        await asyncio.gather(*_pending, return_exceptions=True)
        except Exception as _race_e:
            _lg_race.warning("GROK_RACE error %s; falling back to primary only", _race_e)
            for _task in (_t_g, _t_p):
                if _task is None:
                    continue
                if not _task.done():
                    _task.cancel()
            await asyncio.gather(
                *([_t_g] + ([_t_p] if _t_p is not None else [])),
                return_exceptions=True,
            )
            if _winner is None:
                _winner = await _safe_call(chosen)
                _winner_name = chosen

        if _winner is None:
            answer, cost, worker_ms, _vp_flag, _chosen_tcs, _usage, _chosen_fr, _cascade_trig, _cascade_reason = (
                "[error:grok_race_failed]", 0.0, int((time.time() - t0) * 1000),
                False, [], {}, None, False, None,
            )
        else:
            answer, cost, worker_ms, _vp_flag, _chosen_tcs, _usage, _chosen_fr, _cascade_trig, _cascade_reason = _winner
            if _winner_name and _winner_name != primary_worker:
                fallback_from = primary_worker
                chosen = _winner_name
            _lg_race.info("GROK_RACE winner=%s primary=%s", chosen, primary_worker)
    else:
        answer, cost, worker_ms, _vp_flag, _chosen_tcs, _usage, _chosen_fr, _cascade_trig, _cascade_reason = await _call_worker(
            chosen, q.query, max_tokens=max_tokens, messages=q.messages,
            tools=q.tools, tool_choice=q.tool_choice, extra_kw=extra_kw,
        )
    _route_tier_extra["cascade_triggered"] = _cascade_trig
    _route_tier_extra["cascade_reason"] = _cascade_reason
    _route_tier_extra["primary_worker"] = primary_worker
    # v0.9.16: record sacred spend after success
    if chosen in SACRED and not (not answer or answer.startswith("[error:")):
        _guard_status = _record_sacred(_sid_local, chosen, cost)
        if _guard_status.get("soft_cap_hit"):
            # audit 2026-08-16 (F811): `import logging as _lg_warn` was then
            # overwritten by logging.warning — worked by accident, but the
            # redefinition was a lint trap. Use a plain logger.
            import logging as _lg_warn
            _lg_warn.getLogger("anchor.routing_core").warning(
                "SACRED_GUARD soft_cap sid=%s now=¥%.4f (cap=¥%.2f)",
                _sid_local, _guard_status["session_sacred_yuan"], _SACRED_SOFT)
    # Day 19 (dual-key): if chosen worker returned error / empty (e.g. both keys 429'd),
    # walk the tier pool for a fallback.
    _is_bad = (not answer or answer.startswith("[error:") or _vp_flag)
    primary_success = not _is_bad
    fallback_reason = _classify_fallback_reason(answer, _vp_flag) if _is_bad else None
    # v0.9.46h: empty `answer` is legal when the worker emitted tool_calls.
    # Tool calls surface a non-empty OpenAI-shape list in _chosen_tcs, so we
    # treat that case as success and skip the fallback chain entirely.
    if not answer and _chosen_tcs:
        _is_bad = False
    # v0.9.70: flash-empty rate cooldown. When the primary came back empty
    # AND it is the first attempt (we have not yet entered the fallback chain),
    # AND the worker is deepseek-v4-flash (the known-flaky Zen free proxy),
    # consult the recent empty-rate from the session log. If it crosses
    # threshold (default 40% over last 2min with >=10 samples), trip a 10min
    # cooldown BEFORE entering the fallback chain so subsequent requests
    # skip flash entirely. This bounds the 2026-08-08 failure mode where
    # 62% of flash calls came back empty and the fallback chain absorbed the
    # damage with no signal that flash was the systemic problem.
    if (
        _is_bad
        and fallback_reason == "empty"
        and chosen == "deepseek-v4-flash"
        and not _vp_flag  # vp case already tripped cooldown; do not double-trip
    ):
        try:
            from anchor.cooldown import trip_empty_rate as _trip_er
            _tripped = _trip_er(chosen)
            if _tripped:
                import logging as _lg_er
                _lg_er.warning(
                    "FLASH_EMPTY_RATE_COOLDOWN worker=%s tripped=10min",
                    chosen,
                )
        except Exception as _er_e:
            import logging as _lg_er2
            _lg_er2.warning("FLASH_EMPTY_RATE_COOLDOWN_SKIP worker=%s err=%s", chosen, _er_e)
    # Day 19 fix: always retry on bad answer, even when chosen came from a hard rule.
    # The hard rule encodes routing preference, not "must succeed even if worker down".
    if _is_bad:
        from anchor.cooldown import is_cooling as _is_cool2
        import logging as _lg2
        _lg2.warning("FALLBACK chosen=%s failed (ans=%r), retrying tier=%s", chosen, answer[:80], tier)
        attempts = 0
        max_attempts = _max_fallback_attempts()

        # v0.9.53-p6: one same-worker retry for transient failures. M3 has
        # multiple keys and near-zero marginal cost, so a timeout/connection
        # failure should consume one retry on M3 before escalating to paid
        # Grok/Sol/Fable. Protocol/client 400s and plan-exhausted 429s are
        # deterministic for the current quota window and must NOT retry the
        # same payload.
        _same_worker_transient = fallback_reason in {
            "timeout", "connection",
        }
        # flash is free牛马: skip retry, escalate directly to M3 on failure.
        if _same_worker_transient and max_attempts > 0 and chosen != "deepseek-v4-flash":
            _lg2.warning("FALLBACK retry_same worker=%s reason=%s", chosen, fallback_reason)
            attempts += 1
            rt_answer, rt_cost, rt_ms, rt_vp, rt_tcs, rt_usage, rt_fr, *__ = await _call_worker(
                chosen, q.query, max_tokens=max_tokens, messages=q.messages,
                tools=q.tools, tool_choice=q.tool_choice, extra_kw=extra_kw,
            )
            _rt_bad = (not rt_answer or rt_answer.startswith("[error:") or rt_vp)
            if not rt_answer and rt_tcs:
                _rt_bad = False
            if not _rt_bad:
                answer, cost, worker_ms = rt_answer, rt_cost, rt_ms
                _vp_flag, _chosen_tcs = rt_vp, rt_tcs
                _usage, _chosen_fr = rt_usage, rt_fr
                _is_bad = False
                _lg2.warning("FALLBACK retry_same %s OK", chosen)
            else:
                answer = rt_answer
                fallback_reason = _classify_fallback_reason(rt_answer, rt_vp)
                _lg2.warning("FALLBACK retry_same %s also failed reason=%s", chosen, fallback_reason)

        # AllKeysRateLimited: the primary worker (e.g. minimax-m3) hit its per-key
        # rate limit and the entire KeyPool is locked for retry_after_s. Mark the
        # primary as cooling so it isn't re-selected, then cascade through the
        # fallback chain — other workers are independent providers and may succeed.
        _primary_akrl = bool(answer and ":AllKeysRateLimited]" in answer)
        if _primary_akrl:
            _lg2.warning(
                "ALL_KEYS_429_COOL primary=%s -> cooling + cascading fallback",
                chosen,
            )
            try:
                from anchor.cooldown import trip as _trip_akrl
                import re as _re_akrl
                _m = _re_akrl.search(r"retry_after_s=(\d+)", answer or "")
                _trip_akrl(chosen, seconds=int(_m.group(1)) if _m else 30,
                           reason="all_keys_429")
            except Exception:
                pass
        # t_ea9c35dc: honor the upstream Retry-After (from header or the
        # answer marker) before walking the fallback chain.  Without this
        # the loop hammers a worker that just told us to back off, turning
        # a 429 into a free-fall retry storm.
        _retry_after_s = _retry_after_from_answer(answer)
        if _retry_after_s and _retry_after_s > 0:
            _cap = int(os.environ.get("ANCHOR_FALLBACK_BACKOFF_MAX_S", "120") or 120)
            _wait = min(_retry_after_s, _cap)
            _lg2.warning(
                "FALLBACK retry-after backoff=%ss (raw=%s cap=%s) worker=%s",
                _wait, _retry_after_s, _cap, chosen,
            )
            try:
                await asyncio.sleep(_wait)
            except Exception as _sleep_e:  # pragma: no cover — never fatal
                _lg2.warning("FALLBACK backoff sleep failed: %s", _sleep_e)
        if _is_bad:
            # t_ea9c35dc: append kilo-auto-free as the final free fallback so
            # we always have a zero-cost peer to chase once paid workers are
            # exhausted or in cooldown.
            _fallback_walk = _maybe_kilo(
                tuple(_fallback_candidates(tier, chosen, _tier_pool, vision_only=vision_only))
            )
            for w in _fallback_walk:
                if attempts >= max_attempts:
                    _lg2.warning("FALLBACK stop after %s attempts tier=%s", attempts, tier)
                    break
                if _is_cool2(w):
                    _lg2.warning("FALLBACK skip %s (cooling)", w)
                    continue
                if _is_q_tier(w):
                    _lg2.warning("FALLBACK skip %s (quarantined)", w)
                    continue
                # AKRL is not a stop-the-world condition — other workers in the
                # chain (deepseek, gpt, fable, etc.) are independent providers.
                _lg2.warning("FALLBACK trying %s", w)
                attempts += 1
                fb_answer, fb_cost, fb_ms, fb_vp, _fb_tcs, _fb_usage, _fb_fr, *__ = await _call_worker(
                    w, q.query, max_tokens=max_tokens, messages=q.messages,
                    tools=q.tools, tool_choice=q.tool_choice, extra_kw=extra_kw,
                )
                # v0.9.36 (first-principles): judge_batch decides whether fallback
                # is actually better than the bad primary answer. If verdict B
                # (fallback wins) -> accept. Else continue trying next candidate.
                # On any judge error: keep legacy fallback (accept the first non-error
                # answer), so ship gate is preserved.
                _fb_bad = (not fb_answer or fb_answer.startswith("[error:") or fb_vp)
                # v0.9.53: only consult judge when fallback answer is non-error.
                # If both primary and fallback are errors, shortcut judge and
                # continue the chain — the next candidate may succeed.
                if _judge_batch is not None and answer and fb_answer and not _fb_bad:
                    try:
                        _jd = await _judge_batch(
                            queries=[q.query], response_a=[answer], response_b=[fb_answer],
                            judge_model="claude-opus-5",
                        )
                        if _jd and _jd[0] == "B":
                            _lg2.warning("JUDGE_FALLBACK %s beats primary, accepting", w)
                            if fallback_from is None and w != primary_worker:
                                fallback_from = primary_worker
                            chosen = w
                            answer = fb_answer
                            cost = fb_cost
                            worker_ms = fb_ms
                            _chosen_tcs = _fb_tcs
                            _usage = _fb_usage
                            _chosen_fr = _fb_fr
                            break
                    except Exception as _je:
                        _lg2.warning("JUDGE_FALLBACK error %s, legacy fallback path", _je)
                if not fb_answer and _fb_tcs:
                    _fb_bad = False
                if not _fb_bad:
                    if fallback_from is None and w != primary_worker:
                        fallback_from = primary_worker
                    chosen = w
                    answer = fb_answer
                    cost = fb_cost
                    worker_ms = fb_ms
                    _chosen_tcs = _fb_tcs
                    _chosen_fr = _fb_fr
                    _usage = _fb_usage
                    _lg2.warning("FALLBACK %s OK primary=%s", w, primary_worker)
                    break
                _lg2.warning("FALLBACK %s also failed", w)
    latency_ms = int((time.time() - t0) * 1000)
    # v0.9.19: re-detect vendor placeholder from final answer (covers primary + fallback paths)
    _final_vp = bool(
        answer
        and any(p in answer for p in ("模型未返回可见内容", "I cannot provide", "Sorry, I cannot", "I’m sorry"))
    )
    # or _vp_flag from primary path (m3 NotFoundError)
    _final_vp = _final_vp or bool(_vp_flag)
    # v0.9.7X-A1: classify failure mode for sqlite logging.
    # Priority: vendor placeholder > caught worker error ([error:...] pattern from
    # _safe_call) > silent zero-cost fail. NULL = success.
    _err_class: str | None = None
    if _final_vp:
        _err_class = "placeholder"
    elif answer and answer.startswith("[error:"):
        _err_class = "silent_fail"  # _safe_call caught the worker exception
    elif not answer and (cost or 0.0) == 0.0:
        _err_class = "empty_zero_cost"  # no answer + no cost = unusual failure
    await ainsert_query(
        q.query, tier,
        true_worker_idx=None,
        workers_called=chosen,
        confidences=f"{score:.3f}",
        latency_ms=latency_ms,
        cost_yuan=cost,
        reward=None,
        error_class=_err_class,
    )
    _pt_log = int((_usage or {}).get("prompt_tokens") or 0)
    _ct_log = int((_usage or {}).get("completion_tokens") or 0)
    append_log({
        "yuan": cost, "tier": tier, "worker": chosen, "ms": latency_ms,
        "vendor_placeholder": _final_vp,
        "input_tokens": _pt_log, "output_tokens": _ct_log,
        "prompt_tokens": _pt_log, "completion_tokens": _ct_log,
    })
    # v0.9.53-p6: emit per-lane metrics so /metrics + dashboards can prove
    # M3 hit-rate shifts from the agent-tool-heavy + retry rewrites.
    try:
        from anchor.metrics import (
            record_routing_lane_request, record_routing_lane_fallback,
        )
        _lane = q.query_type or qt or "unknown"
        _status = "ok" if (answer and not answer.startswith("[error:")) else "error"
        record_routing_lane_request(
            _lane, chosen, _status, latency_ms / 1000.0,
            primary_worker=primary_worker,
            primary_success=bool(primary_success),
            fallback_reason=fallback_reason,
        )
        if fallback_from and fallback_from != chosen:
            record_routing_lane_fallback(
                _lane, fallback_from, chosen, fallback_reason or "unknown",
            )
    except Exception as _lane_m_e:
        import logging as _lg_lane
        _lg_lane.warning("LANE_METRIC_SKIP err=%s", _lane_m_e)
    # v0.9.46f: lazy import Response (defined in server.py) to avoid circular
    from anchor.api_models import Response
    _resp_obj = Response(
        answer=answer, worker=chosen, cost_yuan=cost,
        latency_ms=latency_ms, confidence=score, tier=tier,
        tool_calls=_chosen_tcs or None,  # v0.9.46h: surface OpenAI-format list (None → omitted in pydantic v2)
        usage=_usage or None,  # E-01: upstream-reported token usage
        finish_reason=_chosen_fr,  # T-AUDIT-05: surface upstream finish_reason (stop/length/content_filter/...)
        primary_worker=primary_worker,
        fallback_from=fallback_from,
        degraded=bool(fallback_from),
        # v0.9.55: terminal fallback reason for quarantine telemetry.
        # None on full success; classifier-tagged string otherwise.
        fallback_reason=fallback_reason,
    )
    if _trace_token is not None:
        try:
            from anchor.logging_setup import trace_id_var as _tiv2
            _tiv2.reset(_trace_token)
        except Exception:
            pass
    return _resp_obj



# === Re-exports: decompose (late-binds _call_worker) ===
# noqa: F401 — intentional re-exports; server.py and tests import these
# names from anchor.routing_core (ruff cannot see cross-module consumers).
from anchor.routing_decompose import (  # noqa: F401
    _cheap_try,  # noqa: F401
    _judge_quality,  # noqa: F401
    _do_decompose,  # noqa: F401
    _is_splittable,  # noqa: F401
    _maybe_decompose,  # noqa: F401
)
