"""test_query_complexity.py — heuristic tier-0 routing scorer (8 cases).

The function lives at src/anchor/_query_complexity.py and is the
cheapest signal routing_core uses to decide whether to skip the head.
The four bands (easy / short / medium / hard) must stay stable across
refactors or the cheap-path short-circuit becomes a footgun.
"""
from __future__ import annotations

import pytest

from anchor import _query_complexity as qc


@pytest.mark.parametrize(
    "query,expected",
    [
        ("hi", 0.1),                # easy keyword
        ("hello there", 0.1),       # easy keyword + short
        ("thanks!", 0.1),           # easy keyword + ends with '!' (still <20)
        ("design distributed raft consensus", 0.9),  # hard keyword
        ("证明 微积分 极限 推导", 0.9),  # cn hard keyword
        ("implement a linked list in python", 0.5),  # medium keyword
        ("analyze the result", 0.5),  # medium keyword
        ("这是一段中等长度的纯闲聊", 0.2),  # >=50 chars Chinese, falls through to short (no medium CJK kw)
    ],
)
def test_query_complexity_bands(query, expected):
    """Each band must return its anchor score (off-by-one refactor breaks routing)."""
    got = qc.query_complexity(query)
    assert got == pytest.approx(expected, abs=1e-9), (
        f"query_complexity({query!r}) returned {got}, expected {expected}"
    )


def test_empty_query_returns_low_score():
    """Empty / whitespace-only input must not raise."""
    assert qc.query_complexity("") == pytest.approx(0.2)
    assert qc.query_complexity("   ") == pytest.approx(0.2)


def test_long_unknown_query_lands_in_medium_band():
    """A long query with no keyword lands in 0.4 (medium-low)."""
    long_q = "x" * 80
    assert qc.query_complexity(long_q) == pytest.approx(0.4)


def test_short_unknown_query_lands_in_low_band():
    """Short query with no keyword falls through to 0.2."""
    assert qc.query_complexity("foobar") == pytest.approx(0.2)