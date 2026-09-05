"""Cost tracker: ¥800 warn, ¥1100 hard cap (per F8 + Fable 5 refine)."""
import datetime
import json
import threading
import time
from pathlib import Path
from typing import Optional

from anchor import _ROOT as _ANCHOR_ROOT

_LOG_DIR = Path(str(_ANCHOR_ROOT)) / "logs"
LOG_PATH = _LOG_DIR / "cost_log.jsonl"  # legacy unsuffixed path


def _daily_log_path() -> Path:
    """Return date-suffixed log path for daily rotation."""
    return _LOG_DIR / f"cost_log.{datetime.date.today().isoformat()}.jsonl"

# Concurrency guard: serialize cap-check + log-append so two concurrent requests
# can't both pass guard() before either appends (TOCTOU race on monthly total).
_reservation_lock = threading.Lock()
_in_flight_yuan: float = 0.0

# v0.9.51: O(1) hot-path totals (default log only). Custom log_path still scans file.
_counters_lock = threading.Lock()
_counters_loaded = False
_counter_day: str | None = None
_counter_day_yuan: float = 0.0
_counter_month: str | None = None  # YYYY-MM
_counter_month_yuan: float = 0.0


def _month_key(d: datetime.date | None = None) -> str:
    d = d or datetime.date.today()
    return f"{d.year:04d}-{d.month:02d}"


def _load_counters_from_disk() -> None:
    """One-shot hydrate of in-memory day/month totals from LOG_PATH."""
    global _counters_loaded, _counter_day, _counter_day_yuan
    global _counter_month, _counter_month_yuan
    if _counters_loaded:
        return
    today = datetime.date.today()
    day_s = today.isoformat()
    month_s = _month_key(today)
    day_total = 0.0
    month_total_v = 0.0
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    ts = d.get("ts", 0)
                    dt = datetime.datetime.fromtimestamp(ts).date()
                    yuan = float(d.get("yuan", 0.0))
                    if dt.year == today.year and dt.month == today.month:
                        month_total_v += yuan
                    if dt.isoformat() == day_s:
                        day_total += yuan
                except Exception:
                    continue
    _counter_day = day_s
    _counter_day_yuan = day_total
    _counter_month = month_s
    _counter_month_yuan = month_total_v
    _counters_loaded = True


def _bump_counters(yuan: float, ts: float | None = None) -> None:
    global _counter_day, _counter_day_yuan, _counter_month, _counter_month_yuan
    if yuan == 0:
        return
    _load_counters_from_disk()
    dt = datetime.datetime.fromtimestamp(ts or time.time()).date()
    day_s = dt.isoformat()
    month_s = _month_key(dt)
    today = datetime.date.today()
    # Roll day/month if calendar advanced since last update
    if _counter_day != today.isoformat():
        if day_s == today.isoformat():
            _counter_day = day_s
            _counter_day_yuan = 0.0
        else:
            # entry not for today — only update if matches stored day
            pass
    if _counter_month != _month_key(today):
        if month_s == _month_key(today):
            _counter_month = month_s
            _counter_month_yuan = 0.0
    if day_s == (_counter_day or today.isoformat()) or day_s == today.isoformat():
        if _counter_day != day_s:
            _counter_day = day_s
            _counter_day_yuan = 0.0
        _counter_day_yuan += yuan
    if month_s == (_counter_month or _month_key(today)) or month_s == _month_key(today):
        if _counter_month != month_s:
            _counter_month = month_s
            _counter_month_yuan = 0.0
        _counter_month_yuan += yuan
WARN_THRESHOLD_YUAN = 800.0
HARD_CAP_YUAN = float(__import__("os").environ.get("ANCHOR_HARD_CAP_YUAN", "0"))
# Anchor ships with the monthly hard cap disabled by default.
# Operators who want a ceiling opt in by setting ANCHOR_HARD_CAP_YUAN=N.
DAILY_SOFT_CAP_YUAN = 31.0  # beta: per-day throttle (was 1100/mo / 30 days). Soft-only.


class CapHitError(Exception):
    """Raised when monthly hard cap is hit; head must stop new routing."""


def _ensure_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.touch()


def append_log(entry: dict, log_path: Optional[Path] = None) -> None:
    p = log_path or _daily_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)
    e = {"ts": entry.get("ts", time.time()), **entry}
    line = json.dumps(e, ensure_ascii=False) + "\n"
    with open(p, "a") as f:
        f.write(line)
        f.flush()
    if log_path is None and LOG_PATH.exists() and LOG_PATH != p:
        with open(LOG_PATH, "a") as f:
            f.write(line)
            f.flush()
    # Hot-path O(1) counters only for default production log
    if log_path is None:
        try:
            with _counters_lock:
                _bump_counters(float(e.get("yuan", 0.0) or 0.0), e.get("ts"))
        except Exception:
            pass


def today_total(log_path: Optional[Path] = None) -> float:
    if log_path is None:
        with _counters_lock:
            _load_counters_from_disk()
            today = datetime.date.today().isoformat()
            if _counter_day != today:
                return 0.0
            return float(_counter_day_yuan)
    p = log_path or LOG_PATH
    if not p.exists():
        return 0.0
    today = datetime.date.today().isoformat()
    total = 0.0
    with open(p) as f:
        for i, line in enumerate(f, 1):
            try:
                d = json.loads(line)
                ts = d.get("ts", 0)
                if datetime.datetime.fromtimestamp(ts).date().isoformat() == today:
                    total += float(d.get("yuan", 0.0))
            except (json.JSONDecodeError, ValueError, TypeError):
                # S15 (SRE R5): per-line JSON parse failure must log so
                # operators can detect log corruption. Silent continue
                # risks undercounting today's spend (overcount is safe,
                # undercount is not).
                import logging as _lg_tt
                _lg_tt.getLogger("anchor.cost").warning(
                    "TODAY_TOTAL_PARSE_SKIP line=%d err_type=%s",
                    i, "json_or_value")
            except Exception as _tte:
                # Unexpected: log + continue (don't let one bad line
                # poison the whole daily total)
                import logging as _lg_tt
                _lg_tt.getLogger("anchor.cost").warning(
                    "TODAY_TOTAL_UNEXPECTED_SKIP line=%d err=%s", i, _tte)
    return total


def check_daily_throttle(log_path: Optional[Path] = None) -> str:
    """Return 'OK' / 'THROTTLE_DAILY_31'."""
    total = today_total(log_path)
    if total >= DAILY_SOFT_CAP_YUAN:
        return "THROTTLE_DAILY_31"
    return "OK"


def month_total(log_path: Optional[Path] = None) -> float:
    if log_path is None:
        with _counters_lock:
            _load_counters_from_disk()
            mk = _month_key()
            if _counter_month != mk:
                return 0.0
            return float(_counter_month_yuan)
    p = log_path or LOG_PATH
    if not p.exists():
        return 0.0
    now = datetime.date.today()
    total = 0.0
    with open(p) as f:
        for i, line in enumerate(f, 1):
            try:
                d = json.loads(line)
                ts = d.get("ts", 0)
                dt = datetime.datetime.fromtimestamp(ts).date()
                if dt.year == now.year and dt.month == now.month:
                    total += float(d.get("yuan", 0.0))
            except (json.JSONDecodeError, ValueError, TypeError):
                import logging as _lg_mt
                _lg_mt.getLogger("anchor.cost").warning(
                    "MONTH_TOTAL_PARSE_SKIP line=%d err_type=%s",
                    i, "json_or_value")
            except Exception as _mte:
                import logging as _lg_mt
                _lg_mt.getLogger("anchor.cost").warning(
                    "MONTH_TOTAL_UNEXPECTED_SKIP line=%d err=%s", i, _mte)
    return total


def check_cap(log_path: Optional[Path] = None) -> str:
    """Return 'OK' / 'WARN_800' / 'HARD_CAP_1100'. HARD_CAP_YUAN=0 disables."""
    if HARD_CAP_YUAN <= 0:
        return "OK"
    total = month_total(log_path)
    if total >= HARD_CAP_YUAN:
        return "HARD_CAP_1100"
    if total >= WARN_THRESHOLD_YUAN:
        return "WARN_800"
    return "OK"


def guard(log_path: Optional[Path] = None) -> None:
    """Raise CapHitError if hard cap reached. Opt-in via ANCHOR_HARD_CAP_YUAN=N."""
    if HARD_CAP_YUAN <= 0:
        return
    if check_cap(log_path) == "HARD_CAP_1100":
        raise CapHitError(
            f"monthly hard cap {HARD_CAP_YUAN} ¥ hit, "
            f"head routing paused. Run `anchor /resume` after reviewing cost_log."
        )


def reserve_budget(estimated_yuan: float, log_path: Optional[Path] = None) -> bool:
    """Atomically reserve estimated_yuan against the in-flight pool.

    Returns True if reservation succeeded (within cap), False otherwise.
    Pair with commit_budget() after append_log() to release the reservation.

    Closes the TOCTOU race where two concurrent /v1/chat requests both pass
    guard() before either appends, allowing total spend to exceed cap.
    """
    global _in_flight_yuan
    if estimated_yuan <= 0:
        return True
    if HARD_CAP_YUAN <= 0:
        return True
    with _reservation_lock:
        committed = month_total(log_path)
        projected = committed + _in_flight_yuan + estimated_yuan
        if projected >= HARD_CAP_YUAN:
            return False
        _in_flight_yuan += estimated_yuan
        return True


def commit_budget(actual_yuan: float) -> None:
    """Release a reservation after the actual cost is appended."""
    global _in_flight_yuan
    with _reservation_lock:
        _in_flight_yuan = max(0.0, _in_flight_yuan - actual_yuan)


def in_flight_yuan() -> float:
    """Current uncommitted reservation total. For tests + metrics."""
    with _reservation_lock:
        return _in_flight_yuan
