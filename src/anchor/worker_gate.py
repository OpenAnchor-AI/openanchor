"""Worker enable / direct-alias production gates (v0.9.52-p3)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from anchor import _ROOT as _A_ROOT


def _truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


def _falsey(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in {"0", "false", "no", "off"}


def allow_direct_worker_alias() -> bool:
    """Whether clients may pin a worker via direct model alias.

    Production lockdown:
      - ANCHOR_ALLOW_DIRECT_WORKER=0|false → deny (public models only)
      - ANCHOR_ALLOW_DIRECT_WORKER=1|true  → allow
      - unset: allow when ANCHOR_PROFILE=debug OR ANCHOR_AUTH_DISABLED=1;
               otherwise deny (safe production default).
    """
    raw = os.environ.get("ANCHOR_ALLOW_DIRECT_WORKER")
    if raw is not None and str(raw).strip() != "":
        return _truthy(raw)
    profile = (os.environ.get("ANCHOR_PROFILE") or "").strip().lower()
    if profile == "debug" or _truthy(os.environ.get("ANCHOR_AUTH_DISABLED")):
        return True
    return False


def probe_result_path(worker: str) -> Path:
    safe = worker.replace("/", "_")
    return Path(str(_A_ROOT)) / "data" / f"probe_{safe}_results.json"


def probe_recommends_enable(worker: str, path: Path | None = None) -> bool:
    """True when a probe results JSON recommends enabling the worker.

    Accepted recommendation values: enable | add_to_pool | full_rollout.
    """
    p = path or probe_result_path(worker)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
    except Exception:
        return False
    rec = (
        data.get("recommendation")
        or data.get("routing_recommendation")
        or (data.get("routing_recommendation") or {}).get(worker)
        or data.get("verdict")
    )
    if isinstance(rec, dict):
        rec = rec.get(worker) or rec.get("recommendation")
    if not isinstance(rec, str):
        return False
    return rec.strip().lower() in {
        "enable", "add_to_pool", "full_rollout", "pass", "passed", "ok",
    }


def env_or_probe_enabled(env_name: str, worker: str) -> bool:
    """Enable if explicit env truthy, or probe recommends enable (unless env falsey)."""
    raw = os.environ.get(env_name)
    if raw is not None and str(raw).strip() != "":
        if _falsey(raw):
            return False
        if _truthy(raw):
            return True
    return probe_recommends_enable(worker)
