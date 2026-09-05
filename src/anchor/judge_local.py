"""DEPRECATED 2026-07-12 — use anchor.judge_calibration.judge_ensemble
for ship_gate. Kept only for legacy session_log population until v1.0.

Day 18 (Fable 5, 2026-07-06): lightweight local heuristic quality scorer.

Replaces (or complements) LLM-based judge. Runs in microseconds,
no API cost. Used to populate judge_score in session_log so v6 retrain
has supervision signal immediately.

Score is in [0.0, 1.0]:
- 0.0   response is empty / error stub
- 0.3   error prefix detected ([error:...], [stub:...])
- 0.5   non-empty but very short (< 5 chars) or very long (> 8k chars)
- 0.6   has substance but may be low-quality (≤ 50 chars)
- 0.75  normal response (50-2000 chars, no error markers)
- 0.85  substantive response (200-2000 chars, looks like prose/code)
- 0.95  long-form answer (> 500 chars) — usually a real answer

These thresholds are deliberately conservative; the LLM judge
(via /feedback endpoint) remains the source of truth when it fires.
"""
import re

_ERROR_PREFIX = re.compile(r"^\[(error|stub):", re.IGNORECASE)
# v0.9.1: tightened — exclude common markdown delimiters (- _ = * ` ~)
# which sonnet-5/opus/gemini use in tables and code fences.
# Only true "stuck" patterns (non-delimiter chars repeated 9+ times) count.
_REPETITION = re.compile(
    r"([^\-_*=~`#\s"
    "─━│┃┌┐└┘├┤┬┴┼╭╮╯╰"
    r"])\1{8,}",
    flags=0,
)  # v0.9.4: + box-drawing chars (U+2500-257F): + box-drawing chars (U+2500-257F)


def heuristic_quality(response: str, latency_ms: int = 0) -> float:
    """Compute heuristic quality score for a response string.

    Args:
        response: the worker's text reply
        latency_ms: optional — used to slightly penalize extreme latency
    Returns:
        float in [0.0, 1.0]
    """
    if response is None:
        return 0.0
    s = response.strip()

    if not s:
        return 0.0

    if _ERROR_PREFIX.match(s):
        return 0.3

    if _REPETITION.search(s):
        # "aaaaaaaaaa..." or "............." — clearly stuck
        return 0.2

    n = len(s)
    # Length-based bucket. Short answers are valid (e.g. "hi" / "ok" / "Paris")
    # so we only mildly penalize very short.
    if n < 3:
        return 0.6
    if n < 20:
        return 0.7
    if n < 100:
        return 0.75
    if n < 500:
        return 0.8
    if n < 2000:
        return 0.85
    if n < 8000:
        return 0.9
    # Very long — could be a code dump or just verbose; mild reward
    return 0.85


def quality_tier(score: float) -> str:
    """Bucket the score into a human-readable tier."""
    if score >= 0.85:
        return "excellent"
    if score >= 0.7:
        return "good"
    if score >= 0.5:
        return "fair"
    if score >= 0.3:
        return "degraded"
    return "error"
