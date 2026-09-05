"""v0.9.55+: prompt_cache_key hashing + injection gate (OpenAI chat.completions)."""
from __future__ import annotations

import hashlib
import os as _os

# Lazy env flag - conftest strips ANCHOR_* in tests; tests set explicitly.
_PROMPT_CACHE_ENABLED: bool | None = None

# Worker kinds that go through the OpenAI SDK and accept top-level
# prompt_cache_key. anthropic uses message-level cache_control (different
# mechanism, handled in routing_text.py). opencode-cli / direct use
# non-OpenAI SDK paths and are not compatible with this field.
_SUPPORTED_KINDS = frozenset({"openai-compat", "multi-key"})

# Optional per-channel blacklist (comma-separated). When set, injection is
# skipped for any worker whose channel matches. Use case: vendors known to
# reject prompt_cache_key with TypeError (deepseek-official, opencode-zen).
_PROMPT_CACHE_VENDOR_BLACKLIST: frozenset[str] | None = None


def prompt_cache_enabled() -> bool:
    global _PROMPT_CACHE_ENABLED
    if _PROMPT_CACHE_ENABLED is None:
        _PROMPT_CACHE_ENABLED = _os.environ.get("ANCHOR_PROMPT_CACHE_ENABLED", "0") == "1"
    return _PROMPT_CACHE_ENABLED


def _vendor_blacklist() -> frozenset[str]:
    global _PROMPT_CACHE_VENDOR_BLACKLIST
    if _PROMPT_CACHE_VENDOR_BLACKLIST is None:
        raw = _os.environ.get("ANCHOR_PROMPT_CACHE_VENDOR_BLACKLIST", "") or ""
        _PROMPT_CACHE_VENDOR_BLACKLIST = frozenset(
            s.strip() for s in raw.split(",") if s.strip()
        )
    return _PROMPT_CACHE_VENDOR_BLACKLIST


def hash_prompt_cache_key(messages) -> str:
    """Stable hash from first system + first user text. 16-char hex prefix."""
    sys_text = ""
    user_text = ""
    if messages:
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role == "system" and not sys_text:
                if isinstance(content, str):
                    sys_text = content
            elif role == "user" and not user_text:
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            t = p.get("text")
                            if isinstance(t, str):
                                parts.append(t)
                    user_text = "\n".join(parts)
            if sys_text and user_text:
                break
    raw = sys_text + "\x00" + user_text
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def should_inject_prompt_cache_key(worker_kind, worker_channel=None) -> bool:
    """Return True if the caller should inject prompt_cache_key."""
    if not prompt_cache_enabled():
        return False
    if worker_kind not in _SUPPORTED_KINDS:
        return False
    bl = _vendor_blacklist()
    if worker_channel and worker_channel in bl:
        return False
    return True


def build_prompt_cache_kwargs(messages, worker_kind, worker_channel=None, base_kwargs=None):
    """Return a new kwargs dict with prompt_cache_key set if injection applies.

    Returns base_kwargs unchanged when injection is disabled / not supported.
    Never raises; safe to call from any client path.
    """
    if base_kwargs is None:
        base_kwargs = {}
    if not should_inject_prompt_cache_key(worker_kind, worker_channel):
        return base_kwargs
    if "prompt_cache_key" in base_kwargs:
        return base_kwargs
    key = hash_prompt_cache_key(messages)
    return {**base_kwargs, "prompt_cache_key": key}
