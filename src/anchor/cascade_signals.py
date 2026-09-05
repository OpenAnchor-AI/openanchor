"""Pure functions for cheap heuristic cascade trigger detection.

ZERO I/O — all functions are pure transformations on strings.
Cascade = a signal that the response is likely low-quality and the
upstream router should consider falling back to another worker.
"""
from __future__ import annotations

PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "I cannot provide",
    "Sorry, I cannot",
    "I'm sorry",
    "模型未返回可见内容",
    "I apologize",
    "As an AI",
)

_SENTENCE_ENDINGS: frozenset[str] = frozenset(".!?。！？\"')】」』")


def is_truncated_response(response: str, min_length: int = 20) -> bool:
    if not response or not response.strip():
        return True
    stripped = response.strip()
    if stripped[-1] in _SENTENCE_ENDINGS:
        return False
    return len(stripped) < min_length


def has_vendor_placeholder(response: str, markers: tuple[str, ...] = PLACEHOLDER_MARKERS) -> bool:
    return any(m in response for m in markers)


def cascade_score(response: str) -> float:
    """Return 0.0-1.0 quality hint. 0.0 = definitely cascade, 1.0 = looks fine."""
    if not response or not response.strip():
        return 0.0
    if has_vendor_placeholder(response):
        return 0.0
    if is_truncated_response(response):
        return 0.0
    stripped = response.strip()
    if len(stripped) < 20:
        return 0.0
    if len(stripped) < 60:
        return 0.5
    return 1.0


def should_cascade(response: str) -> bool:
    return cascade_score(response) < 0.5
