"""test_classifier.py — 12-label regex classifier (10 cases).

The classifier is the first thing routing sees. Priority order matters:
changing which regex wins changes downstream tier selection and the
shadow log, so we lock the order with explicit fixtures.
"""
from __future__ import annotations

from anchor.classifier import classify, classify_to_tier


def test_classify_chinese_query():
    """Any CJK character → 'cn' label (highest priority after vision)."""
    assert classify("你好，今天天气如何？") == "cn"


def test_classify_debug_query():
    """Stack trace / exception keywords → 'debug'."""
    assert classify("TypeError: cannot read property 'x'") == "debug"
    assert classify("got an exception when importing pandas") == "debug"
    assert classify("how to fix the bug in my python script") == "debug"


def test_classify_code_query():
    """Python/JS keywords + file extensions → 'code'."""
    assert classify("def foo(x): return x * 2") == "code"
    assert classify("implement a linked list in python") == "code"
    assert classify("refactor pyproject.toml setup") == "code"


def test_classify_math_query():
    """Math keywords + LaTeX bracket → 'math'."""
    assert classify("solve the integral of sin(x)") == "math"
    assert classify("compute the limit of $\\frac{1}{n}$") == "math"


def test_classify_agent_query():
    """Tool-use phrases → 'agent'."""
    assert classify("use the tools to read the file") == "agent"
    assert classify("please create file foo.txt") == "agent"


def test_classify_vision_query():
    """Image / photo keywords → 'vision' (highest priority, before cn)."""
    assert classify("describe this image") == "vision"
    assert classify("what is in this picture?") == "vision"


def test_classify_reasoning_query():
    """Reasoning keywords → 'reasoning'."""
    assert classify("analyze the implications of this conclusion") == "reasoning"


def test_classify_default_is_en():
    """No keyword match → 'en' (default English chat)."""
    assert classify("just chatting about life today") == "en"


def test_priority_order_cn_beats_en():
    """When multiple labels could match, priority wins. CJK must beat 'en' default."""
    assert classify("我在写英文") == "cn"  # CJK wins even with 'write' in there


def test_classify_to_tier_returns_auto():
    """Single-product routing: every query_type maps to 'auto' lane."""
    assert classify_to_tier("code") == "auto"
    assert classify_to_tier("cn") == "auto"
    assert classify_to_tier("vision") == "auto"