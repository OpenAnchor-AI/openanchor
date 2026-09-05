"""test_prompt_cache_key.py — prompt_cache_key hashing + injection gate (10 cases).

The injection gate decides whether to add `prompt_cache_key` to the
upstream SDK call. Adding it to the wrong channel (e.g. deepseek-
official which rejects it with TypeError) breaks the request path, so
the gating is heavily locked.
"""
from __future__ import annotations

import pytest

from anchor import prompt_cache_key as pck


@pytest.fixture(autouse=True)
def _isolate_module_state():
    """Every test must start from a clean flag state — globals leak across tests."""
    import importlib

    importlib.reload(pck)
    yield
    importlib.reload(pck)


def test_hash_stable_across_calls():
    """Same messages → same 16-char hex prefix."""
    msgs = [
        {"role": "system", "content": "you are anchor"},
        {"role": "user", "content": "hello world"},
    ]
    a = pck.hash_prompt_cache_key(msgs)
    b = pck.hash_prompt_cache_key(msgs)
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_hash_only_uses_first_system_and_user():
    """Hash must ignore subsequent messages (deterministic cache key)."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "extra"},
        {"role": "assistant", "content": "reply"},
    ]
    a = pck.hash_prompt_cache_key(msgs[:2])
    b = pck.hash_prompt_cache_key(msgs)
    assert a == b


def test_hash_handles_user_content_as_list():
    """User content may be a list of typed parts — only text parts contribute."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": "http://x"},
            ],
        }
    ]
    out = pck.hash_prompt_cache_key(msgs)
    assert len(out) == 16
    # Different text → different hash
    msgs2 = [{"role": "user", "content": [{"type": "text", "text": "describe that"}]}]
    assert pck.hash_prompt_cache_key(msgs2) != out


def test_hash_skips_non_dict_messages():
    """Non-dict items must be silently ignored (defensive)."""
    msgs = ["junk", 42, {"role": "user", "content": "real"}, None]
    out = pck.hash_prompt_cache_key(msgs)
    assert len(out) == 16


def test_injection_disabled_by_default():
    """Without ANCHOR_PROMPT_CACHE_ENABLED=1, the gate must refuse injection."""
    pck._PROMPT_CACHE_ENABLED = None  # force lazy reload
    pck.should_inject_prompt_cache_key.cache_clear() if hasattr(
        pck.should_inject_prompt_cache_key, "cache_clear"
    ) else None
    assert pck.prompt_cache_enabled() is False
    assert pck.should_inject_prompt_cache_key("openai-compat") is False


def test_injection_enabled_for_openai_compat(monkeypatch):
    """With the env flag on, supported kinds pass the gate."""
    monkeypatch.setenv("ANCHOR_PROMPT_CACHE_ENABLED", "1")
    pck._PROMPT_CACHE_ENABLED = None
    assert pck.prompt_cache_enabled() is True
    assert pck.should_inject_prompt_cache_key("openai-compat") is True
    assert pck.should_inject_prompt_cache_key("multi-key") is True


def test_injection_rejects_anthropic_kind():
    """Anthropic uses message-level cache_control, not top-level prompt_cache_key."""
    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setenv("ANCHOR_PROMPT_CACHE_ENABLED", "1")
    pck._PROMPT_CACHE_ENABLED = None
    assert pck.should_inject_prompt_cache_key("anthropic") is False
    monkeypatch.undo()


def test_injection_honors_vendor_blacklist(monkeypatch):
    """Vendors in ANCHOR_PROMPT_CACHE_VENDOR_BLACKLIST must be skipped."""
    monkeypatch.setenv("ANCHOR_PROMPT_CACHE_ENABLED", "1")
    monkeypatch.setenv("ANCHOR_PROMPT_CACHE_VENDOR_BLACKLIST", "deepseek-official,opencode-zen")
    pck._PROMPT_CACHE_ENABLED = None
    pck._PROMPT_CACHE_VENDOR_BLACKLIST = None
    assert pck.should_inject_prompt_cache_key("openai-compat", "deepseek-official") is False
    assert pck.should_inject_prompt_cache_key("openai-compat", "opencode-zen") is False
    assert pck.should_inject_prompt_cache_key("openai-compat", "baosiapi") is True


def test_build_kwargs_returns_unchanged_when_disabled():
    """With injection off, kwargs pass through untouched."""
    base = {"temperature": 0.5}
    out = pck.build_prompt_cache_kwargs([], "openai-compat", base_kwargs=base)
    assert out is base or out == base
    assert "prompt_cache_key" not in out


def test_build_kwargs_injects_when_allowed(monkeypatch):
    """With injection on + supported kind, the dict gains the cache key."""
    monkeypatch.setenv("ANCHOR_PROMPT_CACHE_ENABLED", "1")
    pck._PROMPT_CACHE_ENABLED = None
    msgs = [{"role": "user", "content": "test"}]
    out = pck.build_prompt_cache_kwargs(msgs, "openai-compat", base_kwargs={"x": 1})
    assert "prompt_cache_key" in out
    assert out["x"] == 1
    assert len(out["prompt_cache_key"]) == 16