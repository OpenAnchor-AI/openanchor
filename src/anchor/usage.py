"""Append-only JSONL usage logger for per-worker token tracking."""
from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from typing import Optional

from anchor import _ROOT as _ANCHOR_ROOT

_LOG_DIR = Path(str(_ANCHOR_ROOT)) / "data"
LOG_PATH = _LOG_DIR / "usage_log.jsonl"
_logger = logging.getLogger("anchor.usage")


def record_usage(
    worker: str,
    input_tokens: int,
    output_tokens: int,
    model: str,
    latency_ms: int,
    cost_yuan: float,
    *,
    cached_input_tokens: int = 0,
) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "worker": worker,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "latency_ms": latency_ms,
            "cost_yuan": cost_yuan,
            "cached_input_tokens": cached_input_tokens,
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        _logger.warning("USAGE_RECORD_FAILED worker=%s err=%s", worker, exc)


def month_usage_by_worker(month: Optional[str] = None) -> dict[str, dict]:
    if month is None:
        month = datetime.date.today().strftime("%Y-%m")
    try:
        if not LOG_PATH.exists():
            return {}
        by_worker: dict[str, dict] = {}
        with open(LOG_PATH) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    ts = d.get("ts", 0)
                    dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m")
                    if dt != month:
                        continue
                    w = d.get("worker", "?")
                    if w not in by_worker:
                        by_worker[w] = {"input_tokens": 0, "output_tokens": 0,
                                        "cost_yuan": 0.0, "n_calls": 0}
                    by_worker[w]["input_tokens"] += d.get("input_tokens", 0)
                    by_worker[w]["output_tokens"] += d.get("output_tokens", 0)
                    by_worker[w]["cost_yuan"] += d.get("cost_yuan", 0.0)
                    by_worker[w]["n_calls"] += 1
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        return by_worker
    except Exception:
        _logger.warning("MONTH_USAGE_READ_FAILED", exc_info=True)
        return {}
