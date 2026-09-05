"""Generic round-robin multi-key pool for OpenAI-compat APIs.

Replaces bespoke failover in clients/m3.py with a reusable component.
Used by any worker that needs >1 API key (dpsk today, m3 tomorrow).

Features:
- Round-robin across N keys
- Per-key cooldown on rate-limit / auth errors
- Async-safe (asyncio.Lock for stats + rotation)
- Stream() uses first available key (no failover mid-stream)
- stats() returns calls / fails / rotations / n_keys

Cooldown policy:
- RateLimitError → 30s (provider will recover fast)
- AuthenticationError → 60s (longer; might be temporary)
- APIConnectionError → 15s (network blip)
- APITimeoutError → 15s
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import AsyncIterator, Optional

# v0.9.52-p3: strip leaked <think> reasoning blocks (M3 ignores
# extra_body thinking=disabled in stream mode). Compiled once.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

import openai


# v1.0.2-anchor-f1: vendor Retry-After may be int delta-seconds OR an
# RFC 7231 HTTP-date. We support both. Anything past / malformed
# falls back to None so the caller can use the default 90s cooldown.
MAX_RETRY_AFTER_S = 86400  # 24h; prevents vendor 365d cooldowns.


def _retry_after_from_exception(e) -> int | None:
    """Extract vendor-provided Retry-After from an OpenAI SDK error.

    Supports both formats (RFC 7231):
      - delta-seconds: "3600" -> 3600
      - HTTP-date:     "Wed, 21 Oct 2026 07:28:00 GMT" -> seconds-until-then

    Returns None if the header is missing, malformed, or in the past.
    """
    resp = getattr(e, "response", None)
    if resp is None:
        return None
    try:
        ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    except Exception:
        return None
    if not ra:
        return None
    ra = ra.strip()
    if ra.isdigit():
        return int(ra)
    # HTTP-date format (RFC 7231)
    try:
        target = parsedate_to_datetime(ra)
        if target is None:
            return None
        now = datetime.now(timezone.utc)
        if target <= now:
            return None
        return int((target - now).total_seconds())
    except Exception:
        return None


def _clamp_retry_after(raw) -> int:
    """Clamp Retry-After to [1, MAX_RETRY_AFTER_S]. None -> 90 default.

    Rationale: a vendor returning ``Retry-After: 31536000`` (365 days)
    would otherwise produce a permanent cooldown. 24h is a sensible ceiling:
    any vendor cooldown longer than a day is almost certainly a bug or
    a misconfigured rate-limit window we shouldn't honor verbatim.
    """
    if raw is None:
        return 90
    return min(max(int(raw), 1), MAX_RETRY_AFTER_S)


class AllKeysRateLimited(RuntimeError):
    """Raised by KeyPool.chat() when:

    (a) ALL keys just returned 429 simultaneously (active trigger —
        cooldown is then armed on the pool for ~90s), OR
    (b) a call arrives while that cooldown is still hot (short-circuit
        — last_err / cause may be None).

    Carries retry_after_s so the server can emit HTTP 503 + Retry-After.
    Defined at module top (not under KeyPool) so callers can catch it
    without importing KeyPool.
    """

    def __init__(self, worker_name: str, *, retry_after_s: int, cause: Optional[Exception] = None):
        self.worker_name = worker_name
        self.retry_after_s = max(1, int(retry_after_s))
        self.cause = cause
        msg = f"{worker_name}: all keys rate-limited (retry_after={self.retry_after_s}s)"
        if cause is not None:
            msg += f"; cause={type(cause).__name__}: {str(cause)[:120]}"
        super().__init__(msg)


class KeyPool:
    """Round-robin N-key pool with per-key cooldown."""

    def __init__(
        self,
        *,
        env_keys: tuple[str, ...],
        base_url: str,
        model: str,
        worker_name: str,
        system_prompt: str = "",
        key_strategy: str = "round-robin",
    ):
        keys: list[str] = []
        # os.environ first
        for env in env_keys:
            k = os.environ.get(env, "").strip()
            if k and k not in keys:
                keys.append(k)
        if not keys:
            raise RuntimeError(
                f"{worker_name}: no keys found for env_keys={env_keys} (env only)"
            )
        self.keys = keys
        self.system_prompt = system_prompt
        self.base_url = base_url
        self.model = model
        self.worker_name = worker_name
        # v0.9.76+: "round-robin" (default) rotates through every available key
        # under concurrency. "failover" pins keys[0] as the primary and only
        # advances to keys[1..] when keys[0] is in per-key cooldown — preferred
        # when multiple keys share one upstream quota bucket.
        if key_strategy not in ("round-robin", "failover"):
            raise ValueError(f"{worker_name}: unknown key_strategy={key_strategy!r}")
        self.key_strategy = key_strategy
        # v0.9.55 P0-B: optional channel for prompt_cache_key vendor blacklist.
        self.channel: str | None = None
        self.idx = 0
        self._lock = asyncio.Lock()
        self._stats = {"calls": 0, "fails": 0, "rotations": 0, "worker_trips": 0,
                       "all_429_trips": 0, "all_429_short_circuits": 0}
        # Quick-win guardrail (2026-07-26): when ALL keys 429 simultaneously,
        # set a process-local cooldown so subsequent chat() calls short-circuit
        # instead of re-hitting the locked-out upstream. Empirically minimax
        # lockout lasts 30s..2min (probed 2026-07-26 /tmp/m3_reset_probe.sh);
        # 90s default sits inside that window without burning quota.
        self._all_429_until = 0.0
        self._ALL_429_COOLDOWN_SECS = 90.0
        # v0.9.76: per-key cooldown map (key -> monotonic time until retry).
        # A key that 429'd (or auth/5xx'd) backs off so the pool stops
        # re-trying it every call and the healthy key(s) absorb the load.
        # Re-enables the policy in the module docstring (was a no-op since
        # Day 19; the observed failure was key_1 getting probed every call
        # during a Token Plan window, hammering 52x 429s).
        self._key_until: dict[str, float] = {}
        self._PER_KEY_COOLDOWN_SECS = 30.0

    def _avail_keys(self) -> list[str]:
        """v0.9.76: keys whose per-key cooldown has expired. Returns [] when
        every key is cooling (chat() then surfaces the shortest remaining
        window via AllKeysRateLimited)."""
        now = time.monotonic()
        return [k for k in self.keys if self._key_until.get(k, 0.0) <= now]

    def _cooldown(self, key: str, seconds: float) -> None:
        """v0.9.76: back a single key off for `seconds` (no-op guard for
        non-positive). Keeps the earliest-armed deadline so concurrent
        failures don't shrink the window."""
        if seconds <= 0:
            return
        until = max(time.monotonic() + seconds, self._key_until.get(key, 0.0))
        self._key_until[key] = until

    async def _stream_retry_collect(self, messages, **kw) -> str:
        """v0.9.69: dpsk-flash-empty retry. Collect full stream() output into a
        single string. Only called when chat() returned content="" with no
        tool_calls. Caller gates on "deepseek" in self.model.lower()."""
        parts: list[str] = []
        async for chunk in self.stream(messages, **kw):
            parts.append(chunk)
        return "".join(parts)

    async def _call(self, key: str, messages, **kw) -> dict:
        from .pool import get_openai_client
        # P29-UA: OpenCode Zen free models gate on User-Agent; only
        # `User-Agent: opencode/<ver>` is allowed past their 429 abuse filter
        # (vendor issue #42029). Non-free vendors are unaffected; we gate on
        # base_url to avoid touching M3 / baosiapi / OpenRouter.
        _ua_headers = (
            {"User-Agent": "opencode/1.18.16"}
            if "opencode.ai" in self.base_url
            else None
        )
        client = get_openai_client(
            base_url=self.base_url, api_key=key, headers=_ua_headers,
        )
        # Inject system_prompt at the front if configured and not already present.
        if self.system_prompt:
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages = [{"role": "system", "content": self.system_prompt}] + list(messages)
        # v0.9.21+fix3: m3 default thinking mode eats all output tokens for design prompts
        # (199/200 tokens go to <think> block, content comes back truncated/empty).
        # Disable thinking by default for MiniMax-M3 / similar models. Callers can override
        # via extra_body in **kw.
        # v0.9.26 fix: dpsk (deepseek-v4-flash-free via OpenCode Zen) is also a
        # thinking-first model. Without thinking disabled, dpsk spends most of
        # max_tokens on reasoning_content and returns content="" (finish=length).
        # Caller then sees reasoning leak as content, or worse, vendor placeholder.
        # Same fix as M3 above: inject extra_body to disable thinking.
        _thinking_first_names = ("minimax", "m3", "deepseek")
        if any(n in self.model.lower() for n in _thinking_first_names):
            if "extra_body" not in kw:
                kw["extra_body"] = {"thinking": {"type": "disabled"}}
        # v0.9.55 P0-B: inject prompt_cache_key (OpenAI request-level).
        # Worker kind "multi-key" is supported. Wrapped in try/except so a
        # vendor SDK that rejects unknown fields (TypeError) falls back to
        # kw without the key rather than 500ing the user.
        try:
            from anchor.prompt_cache_key import build_prompt_cache_kwargs
            kw = build_prompt_cache_kwargs(
                messages, worker_kind="multi-key",
                worker_channel=getattr(self, "channel", None),
                base_kwargs=kw,
            )
        except Exception:
            pass
        from anchor.baosi_concurrency import is_baosi_channel, slot as baosi_slot
        _baosi_guard = baosi_slot() if is_baosi_channel(getattr(self, "channel", None)) else None
        if _baosi_guard is not None:
            async with _baosi_guard:
                resp = await client.chat.completions.create(
                    model=self.model, messages=messages, **kw
                )
        else:
            resp = await client.chat.completions.create(
                model=self.model, messages=messages, **kw
            )
        msg = resp.choices[0].message
        content = msg.content or ""
        # Strip inline <think>...</think> blocks (some models leak reasoning).
        content = _THINK_RE.sub("", content).strip()
        # v0.8.9 → v0.9.6: fallback to reasoning_content only when content is
        # empty AND reasoning is substantial (>= 100 chars, not "We are asked...").
        # Otherwise dpsk would leak short reasoning stubs as the answer.
        if not content:
            rc = getattr(msg, "reasoning_content", None)
            if not rc:
                # Kilo / some non-OpenAI-standard gateways return reasoning
                # under a custom field rather than reasoning_content. OpenAI 2.x
                # SDK keeps unknown fields in Pydantic model_extra, so check that.
                rc = None
                try:
                    extra = msg.model_extra or {}
                    rc = extra.get("reasoning")
                except Exception:
                    pass
                if not rc:
                    rc = getattr(msg, "reasoning", None) or ""
            if rc:
                rc = re.sub(r"</?think>", "", rc).strip()
                # v0.9.24: drop 100→30 char floor and add planning-phrase
                # blacklist. The old threshold was tuned for M3 <think>
                # stubs that have since been disabled; for Kilo and other
                # thinking-first gateways, reasoning IS the answer and a
                # few-word summary would otherwise look empty. We still
                # defend against dpsk-style planning stubs by rejecting
                # reasoning that starts with plan phrases.
                _plan_prefixes = (
                    "We are asked", "We need to", "Let me",
                    "I need to", "First, ", "Step 1",
                    "用户要求", "我需要先", "首先", "Step 1",
                )
                _is_planning = any(rc.startswith(p) for p in _plan_prefixes)
                # v0.9.25 fix: dpsk reasoning naturally starts with "We are asked to..."
                # but is the actual answer when long. Only treat as planning-stub when
                # BOTH conditions hold: starts with plan phrase AND <80 chars (no answer follows).
                # Real dpsk response: 200+ chars ending with the actual answer.
                # Real stub: "We are asked to..." then truncated, no answer follows.
                if len(rc) >= 30 and not (_is_planning and len(rc) < 80):
                    content = rc
        # v0.9.21+fix3: detect vendor placeholder (baosiapi 30-char "模型未返回可见内容...").
        # Refined: require start-with OR (len<60 AND marker in content) to avoid false positives
        # on legitimate refusals. Raise so the calling chat() walks the key pool.
        # v0.9.46h: empty content is legal when tool_calls are emitted; only
        # flag placeholder when there's no tool_calls to surface either.
        _raw_tcs_kp = resp.choices[0].message.tool_calls or []
        if not content and not _raw_tcs_kp:
            # v0.9.69: dpsk-flash-free proxy returns content="" on the
            # non-stream chat() call ~65% of the time, but emits the real
            # answer via reasoning_content deltas in stream=True SSE.
            # Retry via stream() before declaring empty. Gate: only dpsk
            # (other workers' empty is genuine — don't add a network round-trip).
            if "deepseek" in self.model.lower():
                try:
                    _streamed = await self._stream_retry_collect(messages, **kw)
                except Exception as _stream_e:
                    _streamed = ""
                if _streamed.strip():
                    content = _streamed.strip()
            if not content:
                raise openai.APIError(
                    f"{self.worker_name} vendor placeholder: empty response",
                    request=None, body=None,
                )
        if content:
            content = re.sub(r'\]<\]minimax\[>\[|\[<minimax>\]|\[<tool_calls?>\]', '', content).strip()
        _placeholders = (
            "]<]minimax[>[",
            "[<minimax>]",
            "模型未返回可见内容",  # baosiapi vendor placeholder
            "I cannot provide",
            "Sorry, I cannot",
            "I’m sorry",
        )
        _is_ph = False
        for marker in _placeholders:
            if content.startswith(marker) or (len(content) < 60 and marker in content):
                _is_ph = True
                break
        if _is_ph:
            raise openai.APIError(
                f"{self.worker_name} vendor placeholder: {content[:60]!r}",
                request=None, body=None,
            )
        return {
            "id": resp.id,
            "model": self.worker_name,
            "content": content,
            "usage": resp.usage.model_dump() if resp.usage else {},
            "finish_reason": resp.choices[0].finish_reason,
            "_worker": self.worker_name,
            # v0.9.46h: surface OpenAI-format tool_calls (list of dicts).
            # Round-robin key pool reuses the same parse path as base.py.
            "tool_calls": [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in (resp.choices[0].message.tool_calls or [])
            ],
        }

    async def chat(self, messages, **kw) -> dict:
        """Day 19 (Fable 5 Q1=A): try each key in round-robin order, no per-key
        cooldown. If all keys fail, trip worker-level cooldown so the router
        skips this worker and falls back to the next tier pool worker.

        Quick-win 2026-07-26: short-circuit the loop when self._all_429_until
        is still in the future. Caller receives retry_after_s via the raised
        AllKeysRateLimited so server can emit HTTP 503 + Retry-After header
        instead of running the fallback chain for 2-3 minutes.
        """
        import logging as _lg_kp
        # Quick-win guardrail: skip the call loop if we are still inside the
        # post-burst cooldown. Saves 2-3min of wasted fallback per held call.
        _now = time.monotonic()
        if _now < self._all_429_until:
            _retry = max(1, int(self._all_429_until - _now))
            async with self._lock:
                self._stats["all_429_short_circuits"] += 1
            _lg_kp.warning(
                "KeyPool[%s] all_keys_429_cooldown retry_after=%ds",
                self.worker_name, _retry,
            )
            raise AllKeysRateLimited(self.worker_name, retry_after_s=_retry)
        last_err: Optional[Exception] = None
        # v1.0.1-dpsk-zen: track max Retry-After across the loop so the
        # final cooldown reflects the actual upstream window, not a guess.
        max_retry_after = 0
        avail = self._avail_keys()
        if not avail:
            # v0.9.76: every key is in per-key cooldown — surface the
            # shortest remaining window instead of looping over nothing.
            _min_until = min(self._key_until.get(k, 0.0) for k in self.keys)
            _retry = max(1, int(_min_until - time.monotonic()))
            raise AllKeysRateLimited(self.worker_name, retry_after_s=_retry)
        # v0.9.76: advance the round-robin cursor atomically under the lock so
        # concurrent requests land on different keys. Before, `start` was read
        # outside the lock → N concurrent callers all read the same idx and
        # collapsed onto key[0], defeating the multi-key pool under load.
        async with self._lock:
            if self.key_strategy == "failover" and self.keys[0] in avail:
                # v0.9.76+: failover mode pins keys[0] as primary. Only
                # advance to keys[1..] when keys[0] is in cooldown. Without
                # this, round-robin under "shared upstream quota" would
                # burn key_2's quota every time key_1 hit the limit.
                avail = [self.keys[0]] + [k for k in avail if k != self.keys[0]]
                start = 0
            else:
                start = self.idx % len(avail)
            self.idx = (self.idx + 1) % len(self.keys)
        ordered = avail[start:] + avail[:start]
        for i, key in enumerate(ordered):
            try:
                result = await self._call(key, messages, **kw)
                async with self._lock:
                    self._stats["calls"] += 1
                    if i > 0:
                        self._stats["rotations"] += 1
                try:
                    _usage_kp = result.get("usage", {})
                    _in_t_kp = _usage_kp.get("prompt_tokens", 0)
                    _out_t_kp = _usage_kp.get("completion_tokens", 0)
                    _cost_kp = 0.0
                    try:
                        from anchor.config import WORKERS as _W_USAGE_KP
                        for _wu_kp in _W_USAGE_KP:
                            if _wu_kp.name == self.worker_name:
                                _cost_kp = (_wu_kp.cost_in * _in_t_kp + _wu_kp.cost_out * _out_t_kp) / 1_000_000
                                break
                    except Exception:
                        pass
                    from anchor.usage import record_usage as _ru_kp
                    _ru_kp(self.worker_name, _in_t_kp, _out_t_kp, result.get("model", self.model), 0, _cost_kp)
                except Exception as _ue_kp:
                    __import__("logging").getLogger("anchor.key_pool").warning(
                        "KEYPOOL_RECORD_USAGE_FAILED worker=%s err=%s", self.worker_name, _ue_kp)
                return result
            except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                # audit 2026-08-16 (D1): a revoked/rotated key used to abort the
                # whole call (re-raise) instead of trying the next key — with a
                # 2-3 key pool, one dead key degraded the worker permanently.
                async with self._lock:
                    self._stats["fails"] += 1
                # v0.9.76: back a revoked/rotated key off for 60s so we stop
                # re-probing it every call (was: every rotation started there).
                self._cooldown(key, 60.0)
                _lg_kp.warning(
                    "KeyPool[%s] key=...%s EXC=%s msg=%r (rotating to next key)",
                    self.worker_name, key[-6:], type(e).__name__, str(e)[:120],
                )
                last_err = e
                continue
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as e:
                async with self._lock:
                    self._stats["fails"] += 1
                # v1.0.1-dpsk-zen: capture vendor Retry-After. OpenCode Zen
                # FreeUsageLimitError returns ~12h; ignoring it forces the
                # 90s short-circuit loop to spin for 13h, burning quota.
                # v1.0.2-anchor-f1: clamp to MAX_RETRY_AFTER_S to defend
                # against vendor quirks like Retry-After: 31536000 (365d).
                _ra = _retry_after_from_exception(e)
                if _ra is not None:
                    _clamped = _clamp_retry_after(_ra)
                    max_retry_after = max(max_retry_after, _clamped)
                # v0.9.76: back this key off (vendor Retry-After when present,
                # else the 30s default) so a rate-limited key stops being the
                # first probe of every call until it cools down.
                self._cooldown(key, _clamped if _ra is not None else self._PER_KEY_COOLDOWN_SECS)
                _lg_kp.warning(
                    "KeyPool[%s] key=...%s EXC=%s msg=%r",
                    self.worker_name, key[-6:], type(e).__name__, str(e)[:120],
                )
                last_err = e
                continue
            except openai.APIStatusError as e:
                # audit 2026-08-16 (D1): rotate on 5xx (per-key upstream issue);
                # re-raise 4xx (bad request shape) — rotating won't help there.
                if e.status_code is not None and e.status_code < 500:
                    raise
                async with self._lock:
                    self._stats["fails"] += 1
                self._cooldown(key, self._PER_KEY_COOLDOWN_SECS)
                _lg_kp.warning(
                    "KeyPool[%s] key=...%s HTTP %s (rotating to next key)",
                    self.worker_name, key[-6:], e.status_code,
                )
                last_err = e
                continue
            except openai.APIError as e:
                # audit 2026-08-16 (D1): vendor placeholder / generic APIError
                # (raised by _call for placeholder content) was designed to
                # "walk the key pool" but was never caught — try the next key.
                async with self._lock:
                    self._stats["fails"] += 1
                self._cooldown(key, self._PER_KEY_COOLDOWN_SECS)
                _lg_kp.warning(
                    "KeyPool[%s] key=...%s EXC=%s msg=%r (rotating to next key)",
                    self.worker_name, key[-6:], type(e).__name__, str(e)[:120],
                )
                last_err = e
                continue
        # All keys failed: trip worker-level cooldown so router skips this worker.
        try:
            from anchor.cooldown import trip as _trip
            _trip(self.worker_name, reason="all_keys_failed",
                  seconds=max_retry_after or None)
        except Exception:
            pass
        # Quick-win guardrail: arm a process-local cooldown so the *next*
        # caller within ~90s short-circuits via AllKeysRateLimited below
        # instead of re-paying the round-robin latency.
        cooldown_secs = (
            max(self._ALL_429_COOLDOWN_SECS, max_retry_after)
            if max_retry_after
            else self._ALL_429_COOLDOWN_SECS
        )
        async with self._lock:
            self._all_429_until = time.monotonic() + cooldown_secs
            self._stats["all_429_trips"] += 1
            if max_retry_after:
                self._stats["vendor_retry_after_used"] = self._stats.get("vendor_retry_after_used", 0) + 1
        raise AllKeysRateLimited(
            self.worker_name,
            retry_after_s=int(cooldown_secs),
            cause=last_err,
        )

    async def stream(self, messages, **kw) -> AsyncIterator[str]:
            """Streaming with key failover parity with chat() (v0.9.76+).

            T-AUDIT-04: inject extra_body={"thinking":{"type":"disabled"}} for
            thinking-first models (M3, dpsk) so the first content chunk arrives
            within ~1s instead of waiting 5-30s on internal reasoning. Matches
            _call() so stream/non-stream parity is preserved.

            v0.9.52-p3: strip <think>...</think> blocks from stream chunks.
            MiniMax-M3 ignores extra_body thinking=disabled in stream mode.

            v0.9.76+: key failover loop. Before this, stream() picked one key
            (avail[0]) and stuck with it — if that key was 429-locked but its
            sibling was hot, every stream request still returned
            FreeUsageLimitError. Now mirrors chat()'s cooldown-aware loop,
            so e.g. m3 with key_1 429'd + key_2 healthy retries through to
            key_2 mid-stream without surfacing an SSE error event. Mid-stream
            failover (after the first chunk) is intentionally NOT done: once
            the SSE generator has emitted role+content to the client,
            swapping mid-flight would break OpenAI stream chunk ordering.
            Failover only fires when the upstream `create(...)` or the first
            `__anext__` raises — before any chunk leaves the server.
            """
            import logging as _lg_kp
            # audit 2026-08-16 (D2): chat() short-circuits on _all_429_until, but
            # stream() bypassed it entirely — mirror chat().
            _now_s = time.monotonic()
            if _now_s < self._all_429_until:
                _retry_s = max(1, int(self._all_429_until - _now_s))
                raise AllKeysRateLimited(self.worker_name, retry_after_s=_retry_s)

            # Same thinking-disable + prompt_cache-key injection as chat().
            # Inject once up front so retries reuse the prepared kwargs.
            _thinking_first_names = ("minimax", "m3", "deepseek")
            if any(n in self.model.lower() for n in _thinking_first_names):
                kw.setdefault("extra_body", {"thinking": {"type": "disabled"}})
            try:
                from anchor.prompt_cache_key import build_prompt_cache_kwargs
                kw = build_prompt_cache_kwargs(
                    messages, worker_kind="multi-key",
                    worker_channel=getattr(self, "channel", None),
                    base_kwargs=kw,
                )
            except Exception:
                pass

            # v0.9.76: key failover loop. We pick a key, open the upstream
            # stream, and yield chunks. If the upstream create() or first
            # __anext__ raises RateLimitError/Timeout/Auth/5xx, we cooldown
            # the key and retry with the next key — before any chunk has
            # been emitted to the client, so the SSE stream remains intact.
            _tried_keys: set[str] = set()
            while True:
                avail = self._avail_keys() or self.keys
                if not avail:
                    _min_until = min(self._key_until.get(k, 0.0) for k in self.keys)
                    _retry = max(1, int(_min_until - time.monotonic()))
                    raise AllKeysRateLimited(self.worker_name, retry_after_s=_retry)
                async with self._lock:
                    if self.key_strategy == "failover" and self.keys[0] in avail:
                        avail = [self.keys[0]] + [k for k in avail if k != self.keys[0]]
                        start = 0
                    else:
                        start = self.idx % len(avail)
                    self.idx = (self.idx + 1) % len(self.keys)
                ordered = avail[start:] + avail[:start]
                _picked = False
                for key in ordered:
                    if key in _tried_keys:
                        continue
                    _tried_keys.add(key)
                    from .pool import get_openai_client
                    _ua_headers = (
                        {"User-Agent": "opencode/1.18.16"}
                        if "opencode.ai" in self.base_url
                        else None
                    )
                    client = get_openai_client(
                        base_url=self.base_url, api_key=key, headers=_ua_headers,
                    )
                    _picked = True
                    try:
                        stream = await client.chat.completions.create(
                            model=self.model, messages=messages, stream=True, **kw
                        )
                    except (openai.RateLimitError, openai.APITimeoutError,
                        openai.APIConnectionError) as _e_sf:
                        _ra = _retry_after_from_exception(_e_sf)
                        _sf_secs = (_clamp_retry_after(_ra)
                                    if _ra is not None
                                    else self._PER_KEY_COOLDOWN_SECS)
                        self._cooldown(key, _sf_secs)
                        async with self._lock:
                            self._stats["fails"] += 1
                        _lg_kp.warning(
                            "KeyPool[%s] stream_fail key=...%s EXC=%s (retrying next key)",
                            self.worker_name, key[-6:], type(_e_sf).__name__,
                        )
                        continue
                    except openai.AuthenticationError as _e_sf:
                        self._cooldown(key, 60.0)
                        async with self._lock:
                            self._stats["fails"] += 1
                        _lg_kp.warning(
                            "KeyPool[%s] stream_fail key=...%s EXC=AuthenticationError (retrying next key)",
                            self.worker_name, key[-6:],
                        )
                        continue
                    except openai.APIStatusError as _e_sf:
                        # 5xx: rotate to next key (per-key upstream issue)
                        if _e_sf.status_code is not None and _e_sf.status_code >= 500:
                            self._cooldown(key, self._PER_KEY_COOLDOWN_SECS)
                            async with self._lock:
                                self._stats["fails"] += 1
                            _lg_kp.warning(
                                "KeyPool[%s] stream_fail key=...%s HTTP %d (retrying next key)",
                                self.worker_name, key[-6:], _e_sf.status_code,
                            )
                            continue
                        raise  # 4xx: request is broken, don't retry
                    # Stream opened cleanly. From here on we yield whatever
                    # the upstream sends; mid-stream errors propagate.
                    async for chunk in stream:
                        if not getattr(chunk, "choices", None):
                            continue
                        _d = chunk.choices[0].delta
                        _parts: list[str] = []
                        _dc = getattr(_d, "content", None)
                        if _dc:
                            _parts.append(_THINK_RE.sub("", _dc))
                        _drc = getattr(_d, "reasoning_content", None)
                        if _drc:
                            _parts.append(_drc)
                        if _parts:
                            joined = "".join(_parts)
                            if not joined:
                                continue
                            yield joined
                    return  # stream completed normally
                if not _picked:
                    # All avail keys were already tried this call (single key
                    # pool, or all keys duplicated). Surface the last error.
                    _min_until = min(self._key_until.get(k, 0.0) for k in self.keys)
                    _retry = max(1, int(_min_until - time.monotonic()))
                    raise AllKeysRateLimited(self.worker_name, retry_after_s=_retry)

    def stats(self) -> dict:
        return {
            **self._stats,
            "n_keys": len(self.keys),
            "current_idx": self.idx,
        }

    async def close(self) -> None:
        pass
