"""v0.9.72: shadow logging for the reasoning_content fallback gate (audit C).

Background: the current fallback at clients/base.py:104-132 promotes
reasoning_content -> content when ``not _is_planning and len(rc) >= 30``.
Audit C is an alternative spec: length floor 10 + narrower 3-prefix planning
list. To decide whether to switch, we want 3-7 days of prod data showing
how many pass/fail decisions would change.

This module records each fallback trigger to a separate JSONL file so the
production session log schema stays untouched. The shadow stream is
append-only, one JSON object per line, with a small fixed schema:

    ts                   UTC epoch seconds (int)
    worker               worker name (str)
    rc_len               length of reasoning_content after think strip (int)
    rc_prefix50          first 50 chars of rc, truncated (str)
    is_planning          current 9-prefix planning decision (bool)
    current_gate_pass    current line-131 expression: not is_planning and rc_len>=30 (bool)
    audit_c_gate_pass    hypothetical spec: rc_len>=10 and not in 3-prefix list (bool)

Output path defaults to ``data/anchor_sessions/_shadow_reasoning.jsonl`` but
can be overridden via the ``ANCHOR_SHADOW_LOG_PATH`` env var (used by tests
to redirect to a tmpdir).

The actual fallback behaviour is NOT changed -- this is a pure observability
hook that runs only when the fallback gate is about to make a decision.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Audit C spec parameters (frozen for the 3-7 day collection window).
_AUDIT_C_LEN_FLOOR = 10
_AUDIT_C_PLAN_PREFIXES = (
    "We are asked to",
    "We need to",
    "用户要求",
)


def audit_c_gate(rc: str) -> bool:
    """Return True iff rc would pass the hypothetical audit C gate."""
    if len(rc) < _AUDIT_C_LEN_FLOOR:
        return False
    return not any(rc.startswith(p) for p in _AUDIT_C_PLAN_PREFIXES)


def current_gate(rc: str, is_planning: bool) -> bool:
    """Mirror the existing line-131 gate expression for shadow parity."""
    return (not is_planning) and (len(rc) >= 30)


def _default_path() -> Path:
    """Resolve the default shadow log path under data/anchor_sessions/."""
    from anchor import _ROOT as _A_ROOT
    return Path(os.environ.get(
        "ANCHOR_SHADOW_LOG_PATH",
        str(_A_ROOT / "data" / "anchor_sessions" / "_shadow_reasoning.jsonl"),
    ))


def record_reasoning_fallback(
    *,
    worker: str,
    rc: str,
    is_planning: bool,
    path: Path | None = None,
) -> None:
    """Append one JSONL line capturing the fallback gate decision.

    Best-effort: any IO / encoding error is swallowed so the production
    chat path is never impacted by observability failures.
    """
    try:
        rc_len = len(rc)
        rc_prefix50 = rc[:50]
        record = {
            "ts": int(time.time()),
            "worker": worker,
            "rc_len": rc_len,
            "rc_prefix50": rc_prefix50,
            "is_planning": bool(is_planning),
            "current_gate_pass": current_gate(rc, is_planning),
            "audit_c_gate_pass": audit_c_gate(rc),
        }
        dest = path or _default_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Observability must never block the production chat path.
        pass
