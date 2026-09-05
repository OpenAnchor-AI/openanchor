"""test_cascade_signals.py — cheap heuristic cascade trigger detector (8 cases).

These functions decide whether the response is so bad it must cascade to
a fallback worker. Anything that triggers a cascade on a healthy
response is a 5xx storm waiting to happen, so the boundary is locked.
"""
from __future__ import annotations

import pytest

from anchor import cascade_signals as cs


def test_empty_response_cascades():
    """Empty / whitespace must always cascade (cascade_score = 0.0)."""
    assert cs.cascade_score("") == 0.0
    assert cs.cascade_score("   ") == 0.0
    assert cs.should_cascade("") is True


@pytest.mark.parametrize(
    "response",
    [
        "I cannot provide that response.",
        "Sorry, I cannot help with this.",
        "I'm sorry, but I cannot.",
        "I apologize, but I cannot.",
        "As an AI, I must decline.",
        "模型未返回可见内容，请重试",
    ],
)
def test_vendor_placeholders_cascade(response):
    """Any known vendor placeholder marker must trigger cascade."""
    assert cs.has_vendor_placeholder(response) is True
    assert cs.cascade_score(response) == 0.0
    assert cs.should_cascade(response) is True


def test_truncated_short_response_cascades():
    """A 19-char response with no ending punctuation is truncated → cascade."""
    short = "ok response body"  # 16 chars, no sentence-end punctuation
    assert cs.is_truncated_response(short, min_length=20) is True
    assert cs.cascade_score(short) == 0.0


def test_truncated_long_response_with_punctuation_passes():
    """A long response ending in '.' is NOT truncated, scores 1.0 when length >= 60."""
    body = "This response is long enough to clear the sixty character length floor and ends here."  # > 60 chars
    assert cs.is_truncated_response(body) is False
    assert cs.cascade_score(body) == 1.0
    assert cs.should_cascade(body) is False


def test_short_but_complete_response_scores_0_5():
    """A response 20-59 chars ending in punctuation is 0.5 (borderline)."""
    body = "Yes, that works fine."  # 22 chars ends '.'
    assert cs.cascade_score(body) == 0.5
    assert cs.should_cascade(body) is False  # 0.5 is NOT < 0.5


def test_no_punctuation_short_response_cascades():
    """Below the length floor with no punctuation → cascade."""
    short = "abcdef"  # 6 chars, no '.!?'
    assert cs.is_truncated_response(short, min_length=20) is True
    assert cs.cascade_score(short) == 0.0


def test_chinese_ending_punctuation_counts_as_end():
    """The detector must recognize CJK sentence-ending punctuation as non-truncated.

    Note: short CJK responses still score < 1.0 because the cascade_score
    function applies a length floor of 60 to reach full credit. The
    truncation detector itself does respect CJK endings — that's the
    contract under test here.
    """
    body = "好的，这个问题我明白了。"  # ends with '。'
    assert cs.is_truncated_response(body) is False
    # cascade_score still weighs length: 14-char CJK response scores 0.0
    # because it's below the 20-char floor for the "complete" branch.
    assert cs.cascade_score(body) == 0.0
    assert cs.should_cascade(body) is True  # 0.0 < 0.5 → cascade