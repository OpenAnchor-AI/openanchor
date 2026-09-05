"""Generic OpenAI-compatible client — used for baosiapi, apihub, Zhipu, Google AI."""
from __future__ import annotations
import openai
import re
from typing import AsyncIterator


PLACEHOLDER_MARKERS = (
    "模型未返回可见内容",  # baosiapi vendor placeholder
    "I cannot provide",
    "Sorry, I cannot",
    "I'm sorry",
)


def _classify_error(e: BaseException | str) -> str:
    """v0.9.7X-A1: classify exception/error into one of 8 failure modes.

    Used by routing_core.py to populate queries.error_class column.
    Order matters: more specific patterns (quota 429, auth 401/403) checked first.
    """
    s = (str(e) if not isinstance(e, BaseException) else str(e)) or ""
    s_low = s.lower()
    if "timeout" in s_low or "timed out" in s_low or "asyncio.timeout" in s_low:
        return "timeout"
    if "401" in s or "403" in s or "auth" in s_low or "api key" in s_low or "unauthorized" in s_low:
        return "auth"
    if "429" in s or "rate" in s_low or "quota" in s_low or "too many requests" in s_low:
        return "quota"
    if "connect" in s_low or "connection" in s_low or "dns" in s_low or "name resolution" in s_low:
        return "connection"
    if "500" in s or "502" in s or "503" in s or "504" in s or "internal server" in s_low or "bad gateway" in s_low:
        return "server_5xx"
    if "404" in s or "not found" in s_low or "no such" in s_low:
        return "not_found"
    if "placeholder" in s_low:
        return "placeholder"
    return "other"

# v0.9.52-p3: strip leaked <think> reasoning blocks (some thinking-first
# models ignore extra_body thinking=disabled in stream mode). Compiled once.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class OpenAICompatClient:
    """One-line-per-provider client. Just needs base_url + api_key + model."""

    def __init__(self, base_url: str, api_key: str, model: str, name: str = "",
                 timeout: float = 60.0, max_retries: int = 1,
                 system_prompt: str | None = None):
        from .pool import get_openai_client
        self.client = get_openai_client(
            base_url=base_url, api_key=api_key,
            timeout=timeout, max_retries=max_retries,
        )
        self.model = model
        self.name = name or model
        self.base_url = base_url
        self.timeout = timeout
        self.system_prompt = system_prompt
        # v0.9.55 P0-B: optional channel for prompt_cache_key vendor blacklist.
        self.channel: str | None = None
        # v0.9.7X-A1: tracks last failure mode for sqlite error_class logging.
        # Set by chat() before raising APIError; reset to None on success.
        # routing_core.py reads this after a failed worker call to populate
        # the queries.error_class column (NULL on success).
        self._last_error_class: str | None = None

    async def chat(self, messages, **kw) -> dict:
        # v0.9.7X-A1: reset error_class at start of each call (clears stale state).
        self._last_error_class = None
        # v0.9.27: inject worker-level system_prompt if configured.
        if self.system_prompt:
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages = [{"role": "system", "content": self.system_prompt}] + list(messages)
        # Thinking-first models (Claude via baosiapi, DeepSeek official pro,
        # MiniMax-style) can spend the entire max_tokens budget on
        # reasoning_content and return content="" with finish_reason=length.
        # Disable thinking by default for non-empty answers; callers can
        # override via extra_body. Do NOT inject for providers that reject
        # the field (e.g. Google AI).
        _m = self.model.lower()
        if any(x in _m for x in ("claude", "deepseek", "minimax")):
            kw.setdefault("extra_body", {"thinking": {"type": "disabled"}})
        # v0.9.55 P0-B: inject prompt_cache_key for OpenAI-compat workers.
        try:
            from anchor.prompt_cache_key import build_prompt_cache_kwargs
            kw = build_prompt_cache_kwargs(
                messages, worker_kind="openai-compat",
                worker_channel=getattr(self, "channel", None),
                base_kwargs=kw,
            )
        except Exception:
            pass
        try:
            # BaosiAPI has one shared upstream quota across all baosiapi-* channels.
            # Acquire the process-wide slot immediately before the network call and
            # release it even on timeout/error. Non-Baosi workers are unaffected.
            from anchor.baosi_concurrency import is_baosi_channel, slot as baosi_slot
            _baosi_guard = baosi_slot() if is_baosi_channel(getattr(self, "channel", None)) else None
            if _baosi_guard is not None:
                async with _baosi_guard:
                    resp = await self.client.chat.completions.create(
                        model=self.model, messages=messages, **kw)
            else:
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=messages, **kw)
        except Exception as _ce:
            # v0.9.7X-A1: classify network/auth/quota failures for sqlite logging.
            self._last_error_class = _classify_error(_ce)
            raise
        msg = resp.choices[0].message
        content = msg.content or ""
        # Strip <think>...</think> blocks (e.g. dpsk reasoning leak)
        content = _THINK_RE.sub("", content).strip()
        # v0.9.24: reason→content fallback for non-standard gateways. Some
        # Anthropic-compatible routes return the answer inside reasoning /
        # reasoning_content / model_extra.reasoning instead of content. OpenAI
        # 2.x keeps unknown fields in Pydantic model_extra, so probe that first.
        if not content:
            rc = getattr(msg, "reasoning_content", None)
            if not rc:
                try:
                    extra = msg.model_extra or {}
                    rc = extra.get("reasoning")
                except Exception:
                    rc = None
            if not rc:
                rc = getattr(msg, "reasoning", None) or ""
            if rc:
                rc = re.sub(r"</?think>", "", rc).strip()
                # Reject planning stubs that look like thinking-out-loud rather
                # than a final answer; require >=30 chars.
                _plan_prefixes = (
                    "We are asked", "We need to", "Let me",
                    "I need to", "First, ", "Step 1",
                    "用户要求", "我需要先", "首先",
                )
                _is_planning = any(rc.startswith(p) for p in _plan_prefixes)
                from anchor.shadow_log import record_reasoning_fallback as _shadow_rec
                _shadow_rec(worker=self.name, rc=rc, is_planning=_is_planning)
                if not _is_planning and len(rc) >= 30:
                    content = rc
        # v0.9.19 (refined 2026-07-09): detect vendor placeholder robustly.
        # Issue: substring match "I cannot provide" wrongly flagged valid Chinese replies
        # like "我无法提供完整方案" or English "I cannot provide that specifically".
        # Fix: require either (a) substring at very start of reply OR (b) total length < 60.
        # The actual baosiapi placeholder is exactly 30 chars; real refusals are 100+ chars.
        _is_placeholder = False
        # v0.9.46h: empty content is legal when the worker emits tool_calls.
        # We must NOT treat empty content as a vendor placeholder in that case.
        _raw_tcs = msg.tool_calls or []
        if not content and not _raw_tcs:
            _is_placeholder = True
        elif content:
            for marker in PLACEHOLDER_MARKERS:
                if content.startswith(marker) or (len(content) < 60 and marker in content):
                    _is_placeholder = True
                    break
        if _is_placeholder:
            self._last_error_class = "placeholder"
            raise openai.APIError(
                message=f"vendor placeholder from {self.name}: {content[:60]!r}",
                request=None, body=None
            )
        try:
            _usage_dict = resp.usage.model_dump() if resp.usage else {}
            _in_t = _usage_dict.get("prompt_tokens", 0)
            _out_t = _usage_dict.get("completion_tokens", 0)
            _cached_t = ((_usage_dict.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            _cost_yuan = 0.0
            try:
                from anchor.config import WORKERS as _W_USAGE
                for _wu in _W_USAGE:
                    if _wu.name == self.name:
                        _in_t_uncached = max(_in_t - _cached_t, 0)
                        _cached_cost_in = getattr(_wu, "cost_in_cached", 0.0) or 0.0
                        if _cached_cost_in <= 0:
                            _cached_cost_in = _wu.cost_in * 0.1
                        _cost_yuan = (
                            _wu.cost_in * _in_t_uncached / 1_000_000
                            + _cached_cost_in * _cached_t / 1_000_000
                            + _wu.cost_out * _out_t / 1_000_000
                        )
                        break
            except Exception:
                pass
            from anchor.usage import record_usage as _ru
            _ru(self.name, _in_t, _out_t, self.model, 0, _cost_yuan, cached_input_tokens=_cached_t)
        except Exception as _ue:
            __import__("logging").getLogger("anchor.base").warning(
                "BASE_RECORD_USAGE_FAILED worker=%s err=%s", self.name, _ue)
        return {
            "id": resp.id,
            "model": resp.model,
            "content": content,
            "usage": resp.usage.model_dump() if resp.usage else {},
            "finish_reason": resp.choices[0].finish_reason,
            "_worker": self.name,
            # v0.9.46h: surface OpenAI-format tool_calls (list of dicts).
            # Each entry: {"id": "...", "type": "function",
            #              "function": {"name": "...", "arguments": "..."}}
            "tool_calls": [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in (msg.tool_calls or [])
            ],
        }

    async def stream(self, messages, **kw) -> AsyncIterator[str]:
        # T-AUDIT-04 (compat HIGH): inject thinking-disable for thinking-first
        # models (Claude/DeepSeek/MiniMax). Matches chat() behavior — avoid
        # 0 chunks for 5-30s during reasoning, and content="" finish=length.
        _m = self.model.lower()
        if any(x in _m for x in ("claude", "deepseek", "minimax")):
            kw.setdefault("extra_body", {"thinking": {"type": "disabled"}})
        # v0.9.55 P0-B: inject prompt_cache_key (stream parity with chat()).
        try:
            from anchor.prompt_cache_key import build_prompt_cache_kwargs
            kw = build_prompt_cache_kwargs(
                messages, worker_kind="openai-compat",
                worker_channel=getattr(self, "channel", None),
                base_kwargs=kw,
            )
        except Exception:
            pass
        # Process-wide BaosiAPI gate (shared across all baosiapi-* channels).
        from anchor.baosi_concurrency import is_baosi_channel, slot as baosi_slot
        _baosi_guard = baosi_slot() if is_baosi_channel(getattr(self, "channel", None)) else None
        if _baosi_guard is not None:
            await _baosi_guard.__aenter__()
        try:
            stream = await self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **kw)
            async for chunk in stream:
                # Some chunks (e.g. final usage-only) have no choices.
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    delta = _THINK_RE.sub("", delta)
                    if delta:
                        yield delta
        finally:
            if _baosi_guard is not None:
                await _baosi_guard.__aexit__(None, None, None)

    async def close(self):
        pass
