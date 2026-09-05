"""Circuit breaker: 24h rolling pair_acc < 0.50 -> force sonnet-5 routing."""
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from anchor import _ROOT as _ANCHOR_ROOT

# Concurrency guard: serialize state mutations across /v1/chat + cron + admin.
_state_lock = threading.Lock()
from anchor.release.gates import ROLLING_24H_FLOOR, CIRCUIT_BREAKER_FORCE

STATE_PATH = Path(str(_ANCHOR_ROOT)) / "data" / "circuit_breaker.json"

# Per-worker err_rate quarantine (added v0.9.45).
# Default: err_rate >= threshold removes the worker from new primary routing.
# M3 is subscription/amortized and should be used aggressively, but protocol
# bugs and transient transport failures must not count the same as bad model
# quality. Its higher threshold is intentionally worker-specific; transient
# failures already get short cooldown + same-worker retry.
ERR_RATE_QUARANTINE_THRESHOLD: float = 0.40
WORKER_ERR_RATE_QUARANTINE_THRESHOLD: dict[str, float] = {
    "minimax-m3": 0.60,
}
ERR_RATE_WINDOW_HOURS: float = 24.0

# v0.9.58: per-worker placeholder warning threshold (soft alert band between
# SLA floor and auto-quarantine). dpsk-flash has historically had vendor
# placeholder storms before timeout → quarantine. Warning at 15% gives ops
# ~25% headroom before the 40% auto-quarantine threshold trips.
PLACEHOLDER_WARNING_THRESHOLD: float = 0.20
WORKER_PLACEHOLDER_WARNING_THRESHOLD: dict[str, float] = {
    "deepseek-v4-flash": 0.15,
}


# v0.9.55: statistical guards before quarantining.
MIN_N_FOR_QUARANTINE: int = int(os.environ.get("ANCHOR_QUARANTINE_MIN_N", "20"))
MIN_SESSIONS_FOR_QUARANTINE: int = int(
    os.environ.get("ANCHOR_QUARANTINE_MIN_SESSIONS", "5")
)
WILSON_Z: float = float(os.environ.get("ANCHOR_QUARANTINE_WILSON", "1.96"))


def _wilson_lower_bound(successes: int, n: int, z=WILSON_Z) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    # spread = z * sqrt(p*(1-p)/n + z^2/(4n^2))
    variance = (p * (1 - p) / n) + (z * z) / (4 * n * n)
    spread = (z / denom) * variance ** 0.5
    return max(0.0, centre - spread)


def _should_quarantine(worker, err_rate, *, n, sessions, ci_lower):
    if n <= 0:
        threshold = err_rate_quarantine_threshold(worker)
        return err_rate >= threshold, 0.0, "legacy"
    if n < MIN_N_FOR_QUARANTINE:
        return False, ci_lower if ci_lower is not None else 0.0, "below_min_n"
    if sessions < MIN_SESSIONS_FOR_QUARANTINE:
        return False, ci_lower if ci_lower is not None else 0.0, "ci_too_wide"
    threshold = err_rate_quarantine_threshold(worker)
    if ci_lower is None:
        ci_lower = _wilson_lower_bound(round(err_rate * n), n)
    if ci_lower <= threshold:
        return False, ci_lower, "below_threshold_with_storm"
    return True, ci_lower, "below_threshold_with_storm"


def err_rate_quarantine_threshold(worker: str) -> float:
    return WORKER_ERR_RATE_QUARANTINE_THRESHOLD.get(
        worker, ERR_RATE_QUARANTINE_THRESHOLD
    )


def placeholder_warning_threshold(worker: str) -> float:
    return WORKER_PLACEHOLDER_WARNING_THRESHOLD.get(
        worker, PLACEHOLDER_WARNING_THRESHOLD
    )


def should_warn_placeholder(worker: str, current_err_rate: float) -> bool:
    return (
        current_err_rate >= placeholder_warning_threshold(worker)
        and current_err_rate < err_rate_quarantine_threshold(worker)
    )


def _default_state() -> dict:
    return {
        "open": False, "opened_at": None, "closed_at": None, "rolling_24h": None,
        "err_rate_quarantine": {},
    }


def load_state(state_path: Optional[Path] = None) -> dict:
    p = state_path or STATE_PATH
    if not p.exists():
        return _default_state()
    try:
        state = json.loads(p.read_text())
    except (OSError, ValueError):
        # audit 2026-08-16 (E1): a torn/truncated state file (crash mid-write
        # or a non-atomic concurrent writer) used to raise JSONDecodeError
        # straight through the per-request is_quarantined() path, 500ing every
        # request until an operator deleted the file. Self-heal instead: log
        # once, return defaults, and let the next save rewrite the file.
        import logging as _lg_cb
        _lg_cb.getLogger("anchor.circuit_breaker").warning(
            "circuit_breaker state unreadable at %s — resetting to defaults", p)
        return _default_state()
    if not isinstance(state, dict):
        return _default_state()
    state.setdefault("err_rate_quarantine", {})
    return state


def _atomic_write_text(path: Path, text: str) -> None:
    """tmp + os.replace + fsync — a crash mid-write never leaves a torn file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(STATE_PATH, json.dumps(state, indent=2))


def save_state_to(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(state_path, json.dumps(state, indent=2))


def aggregate_state_summary() -> dict:
    """v0.9.60-B1: aggregate view of circuit_breaker state for admin/dashboard.

    Returns counts of quarantined + warning-band workers, plus per-worker
    status including placeholder_warning + placeholder_warning_threshold.
    """
    state = load_state()
    workers = state.get("err_rate_quarantine", {})
    n_quarantined = sum(1 for w in workers.values() if w.get("quarantined"))
    n_warning = sum(
        1 for w in workers.values()
        if not w.get("quarantined")
        and w.get("err_rate") is not None
        and w["err_rate"] >= placeholder_warning_threshold(w.get("name", ""))
    )
    summary = {
        "open": state.get("open", False),
        "opened_at": state.get("opened_at"),
        "closed_at": state.get("closed_at"),
        "n_workers_total": len(workers),
        "n_workers_quarantined": n_quarantined,
        "n_workers_warning": n_warning,
        "workers": {},
        "ts": time.time(),
    }
    for w_name, w in workers.items():
        ph_thr = placeholder_warning_threshold(w_name)
        ph_warn = (
            w.get("err_rate") is not None
            and w["err_rate"] >= ph_thr
            and not w.get("quarantined", False)
        )
        summary["workers"][w_name] = {
            "err_rate": w.get("err_rate"),
            "threshold": w.get("threshold"),
            "quarantined": w.get("quarantined", False),
            "ci_lower": w.get("ci_lower"),
            "n": w.get("n"),
            "sessions": w.get("sessions"),
            "reason": w.get("reason"),
            "ts": w.get("ts"),
            "placeholder_warning": ph_warn,
            "placeholder_warning_threshold": ph_thr,
        }
    return summary


def is_open() -> bool:
    return load_state().get("open", False)


def force_worker() -> Optional[str]:
    return CIRCUIT_BREAKER_FORCE if is_open() else None


def update_err_rate_quarantine(worker: str, err_rate: float, state_path=None,
                                    *, n=0, sessions=0, ci_lower=None) -> dict:
    """Track per-worker err_rate. Quarantine when statistical guard passes.

    v0.9.55: quarantine only when n >= MIN_N_FOR_QUARANTINE AND
    sessions >= MIN_SESSIONS_FOR_QUARANTINE AND Wilson 95% lower
    confidence bound > threshold. Legacy callers (n=0) fall back to the
    direct err_rate >= threshold comparison so old tests keep working.
    """
    with _state_lock:
        state = load_state(state_path)
        quarantined_map = state.setdefault("err_rate_quarantine", {})
        threshold = err_rate_quarantine_threshold(worker)
        quarantined, computed_ci, reason = _should_quarantine(
            worker, err_rate, n=n, sessions=sessions, ci_lower=ci_lower,
        )
        quarantined_map[worker] = {
            "err_rate": float(err_rate),
            "threshold": float(threshold),
            "quarantined": bool(quarantined),
            "ci_lower": float(computed_ci),
            "n": int(n),
            "sessions": int(sessions),
            "reason": reason,
            "ts": time.time(),
        }
        state["err_rate_quarantine"] = quarantined_map
        if state_path is not None:
            save_state_to(state_path, state)
        else:
            save_state(state)
    return {
        "worker": worker,
        "err_rate": err_rate,
        "threshold": threshold,
        "quarantined": quarantined,
        "ci_lower": float(computed_ci),
        "n": int(n),
        "sessions": int(sessions),
        "reason": reason,
        "ts": quarantined_map[worker]["ts"],
    }


def is_quarantined(worker: str, state_path: Optional[Path] = None) -> bool:
    """Return True if worker is currently quarantined due to high err_rate."""
    state = load_state(state_path)
    qmap = state.get("err_rate_quarantine", {})
    entry = qmap.get(worker, {})
    return bool(entry.get("quarantined", False))


def list_quarantined(state_path: Optional[Path] = None) -> dict[str, dict]:
    """Return {worker: {err_rate, ts, quarantined}} for currently quarantined workers."""
    state = load_state(state_path)
    qmap = state.get("err_rate_quarantine", {})
    return {w: v for w, v in qmap.items() if v.get("quarantined", False)}


def update_quarantine_probe(worker: str, ok: bool,
                            state_path: Optional[Path] = None) -> bool:
    """Persist last probe result. Failed probes also refresh ts for ops."""
    with _state_lock:
        state = load_state(state_path)
        qmap = state.setdefault("err_rate_quarantine", {})
        entry = qmap.get(worker)
        if not entry:
            return False
        now = time.time()
        entry["last_probe_ok"] = bool(ok)
        entry["last_probe_ts"] = now
        if not ok:
            entry["ts"] = now
        qmap[worker] = entry
        state["err_rate_quarantine"] = qmap
        if state_path is not None:
            save_state_to(state_path, state)
        else:
            save_state(state)
    return True


def clear_quarantine(worker: str, state_path: Optional[Path] = None) -> bool:
    """Release a quarantined worker. Returns True if state changed."""
    with _state_lock:
        state = load_state(state_path)
        qmap = state.setdefault("err_rate_quarantine", {})
        if worker not in qmap or not qmap[worker].get("quarantined"):
            return False
        qmap[worker]["quarantined"] = False
        qmap[worker]["reason"] = "manual"
        qmap[worker]["ts"] = time.time()
        state["err_rate_quarantine"] = qmap
        if state_path is not None:
            save_state_to(state_path, state)
        else:
            save_state(state)
    return True


def check_and_update(rolling_24h: float, state_path: Optional[Path] = None) -> dict:
    """Check 24h rolling pair_acc, open/close breaker accordingly.

    Returns {open, action, rolling_24h, threshold}.
    """
    state = load_state(state_path)
    if rolling_24h is None:
        return {"open": state.get("open", False), "action": state.get("action", "armed_no_data"),
                "rolling_24h": None, "threshold": ROLLING_24H_FLOOR}
    if rolling_24h < ROLLING_24H_FLOOR:
        if not state.get("open"):
            state["open"] = True
            state["opened_at"] = time.time()
            state["action"] = f"force:{CIRCUIT_BREAKER_FORCE}"
            if state_path is not None:
                save_state_to(state_path, state)
            else:
                save_state(state)
        return {
            "open": True,
            "action": f"force:{CIRCUIT_BREAKER_FORCE}",
            "rolling_24h": rolling_24h,
            "threshold": ROLLING_24H_FLOOR,
        }
    if rolling_24h is not None and state.get("open") and rolling_24h >= ROLLING_24H_FLOOR:
        state["open"] = False
        state["closed_at"] = time.time()
        state["action"] = "recovered"
        if state_path is not None:
            save_state_to(state_path, state)
        else:
            save_state(state)
    state["rolling_24h"] = rolling_24h
    if state_path is not None:
        save_state_to(state_path, state)
    else:
        save_state(state)
    return {
        "open": state["open"],
        "action": state.get("action", "armed"),
        "rolling_24h": rolling_24h,
        "threshold": ROLLING_24H_FLOOR,
    }


def reset() -> None:
    """Force-close the breaker (for testing/recovery)."""
    state = {"open": False, "opened_at": None, "closed_at": time.time(), "rolling_24h": None, "action": "armed"}
    save_state(state)
