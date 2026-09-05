"""Message/text extraction helpers (from routing_core)."""
from __future__ import annotations
import os as _os

# Lazy cache flag — re-read at first call so gateway restart picks up env
# changes when ANCHOR_PROMPT_CACHE_ENABLED is toggled in tests/runtime.
_PROMPT_CACHE_ENABLED: bool | None = None

def _prompt_cache_enabled() -> bool:
    global _PROMPT_CACHE_ENABLED
    if _PROMPT_CACHE_ENABLED is None:
        _PROMPT_CACHE_ENABLED = _os.environ.get("ANCHOR_PROMPT_CACHE_ENABLED", "0") == "1"
    return _PROMPT_CACHE_ENABLED

# --- _messages_have_image_url ---
def _messages_have_image_url(messages: list | None) -> bool:
    if not messages:
        return False
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        parts = content if isinstance(content, list) else []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
    return False




# --- _extract_user_text ---
def _extract_user_text(content) -> str:
    """Vision-safe extractor: multimodal content may be a list of parts
    (text / image_url), not a plain string. Returns concatenated text only.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return ""




# --- _responses_content_to_text ---
def _responses_content_to_text(content, _depth: int = 0) -> str:
    # L2 (security/DoS LOW): client-controlled JSON can nest arbitrarily
    # deep; cap recursion to avoid RecursionError killing the worker.
    if _depth > 32:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _responses_content_to_text(item, _depth + 1)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return ""


def normalize_openai_messages(
    messages: list | None,
    worker_kind: str | None = None,
) -> list:
    """Normalize and strictly repair OpenAI Chat tool protocol.

    The canonical implementation lives in :mod:`anchor.tool_protocol` and is
    shared by Chat Completions and Responses input conversion. This wrapper is
    retained for backward compatibility with existing imports/tests.

    v0.9.54+: prompt-cache injection (cache_control on system messages) is
    **worker-kind aware**. Only Anthropic-native workers consume the
    ``cache_control`` field — for every other kind (openai-compat,
    multi-key, opencode-cli, direct) the field is silently dropped by the
    OpenAI SDK and provides zero savings while falsely inflating the
    "enabled" dashboard metric. To inject cache_control, both
    ``ANCHOR_PROMPT_CACHE_ENABLED=1`` and ``worker_kind == "anthropic"``
    must hold. Callers that do not pass ``worker_kind`` (backward-compat
    default) get the safe "no injection" behavior.

    WARNING (L2): the chat-completions path in server.py calls this
    WITHOUT ``worker_kind``, so cache_control injection never fires there.
    That is intentional (safe default) — do NOT "fix" it by passing
    ``worker_kind="openai-compat"`` from server.py, or you will silently
    enable cache_control on workers that drop it. Only Anthropic-native
    call sites (e.g. a future direct-Claude client) should pass
    ``worker_kind="anthropic"``.
    """
    from anchor.tool_protocol import normalize_tool_protocol
    out = normalize_tool_protocol(messages) or []
    if _prompt_cache_enabled() and worker_kind == "anthropic":
        for msg in out:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str) and msg["content"]:
                msg.setdefault("cache_control", {"type": "ephemeral"})
    # v0.9.55 P0-B: defense in depth — strip any pre-existing cache_control
    # field on the non-anthropic path. Some clients (and the older anchor
    # gateway) hand us Anthropic-shaped messages; on the OpenAI SDK the field
    # is silently dropped, which hides bugs and falsely suggests cache is
    # active. We make the strip explicit so the behavior is auditable.
    if worker_kind and worker_kind != "anthropic":
        for msg in out:
            if "cache_control" in msg:
                msg.pop("cache_control", None)
    return out

