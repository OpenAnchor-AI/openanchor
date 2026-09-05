"""Budget tracker for A2.2 maybe_autopromote (V12.3 design).

Provides:
- get_daily_cost() -> float (sum cost_yuan from queries table for today UTC)
- get_daily_cap() -> float (default ¥100/day or ANCHOR_DAILY_BUDGET_CNY env)
- get_weekly_cost() -> float (7d window)
- get_monthly_cost() -> float (30d window)

Usage:
    from anchor.budget_tracker import get_daily_cost, get_daily_cap
    daily_cost = get_daily_cost()
    daily_cap = get_daily_cap()
    if daily_cost >= daily_cap * 0.8:
        # Budget nearly exhausted, skip autopromote
"""
from __future__ import annotations
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_PATH = Path(os.environ.get("ANCHOR_DB", "data/anchor.db"))


def _connect():
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(str(DB_PATH))


def get_daily_cost() -> float:
    """Sum cost_yuan from queries table for today (UTC).

    Returns 0.0 if DB not found or query fails.
    """
    conn = _connect()
    if conn is None:
        return 0.0
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT COALESCE(SUM(cost_yuan), 0.0)
        FROM queries
        WHERE ts >= strftime('%s', 'start of day', 'now')
          AND cost_yuan IS NOT NULL
        """)
        return float(cur.fetchone()[0])
    except Exception:
        return 0.0
    finally:
        conn.close()


def get_daily_cap() -> float:
    """Daily budget cap from ANCHOR_DAILY_BUDGET_CNY env or default ¥100/day.

    Override via env: ANCHOR_DAILY_BUDGET_CNY=200
    """
    env_val = os.environ.get("ANCHOR_DAILY_BUDGET_CNY", "100")
    try:
        return float(env_val)
    except ValueError:
        return 100.0


def get_weekly_cost(days: int = 7) -> float:
    """Sum cost_yuan for last N days."""
    conn = _connect()
    if conn is None:
        return 0.0
    try:
        since_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
        cur = conn.cursor()
        cur.execute("""
        SELECT COALESCE(SUM(cost_yuan), 0.0)
        FROM queries WHERE ts >= ? AND cost_yuan IS NOT NULL
        """, (since_ts,))
        return float(cur.fetchone()[0])
    except Exception:
        return 0.0
    finally:
        conn.close()


def get_monthly_cost() -> float:
    """Sum cost_yuan for last 30 days."""
    return get_weekly_cost(30)


if __name__ == "__main__":
    # CLI usage
    print(f"daily_cost: ¥{get_daily_cost():.2f}")
    print(f"daily_cap:  ¥{get_daily_cap():.2f}")
    print(f"weekly_cost: ¥{get_weekly_cost():.2f}")
    print(f"monthly_cost: ¥{get_monthly_cost():.2f}")
