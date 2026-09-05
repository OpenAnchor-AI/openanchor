"""Day 17: cron job installer + entry points.

Generates crontab lines for:
- Weekly retrain: Sun 03:00 CST
- Daily drift: 04:00 daily
- 15-min health probe
"""
import subprocess
import sys
from pathlib import Path

ANCHOR_BIN = Path.home() / "anchor" / "bin"
INSTALL_PATH = Path.home() / "anchor" / "install" / "install_cron.sh"

CRON_LINES = [
    "# Anchor weekly retrain (Sun 03:00 CST)",
    "0 3 * * 0 /usr/bin/bash -lc 'python -m anchor.cron retrain-weekly >> $ANCHOR_ROOT/logs/cron.log 2>&1'",
    "# Anchor daily drift check (04:00 daily)",
    "0 4 * * * /usr/bin/bash -lc 'python -m anchor.cron drift-check >> $ANCHOR_ROOT/logs/cron.log 2>&1'",
    "# Anchor health probe (every 15 min)",
    "*/15 * * * * /usr/bin/bash -lc 'python -m anchor.cron health-probe >> $ANCHOR_ROOT/logs/cron.log 2>&1'",
]


def render_crontab() -> str:
    return "\n".join(CRON_LINES) + "\n"


def install_cron(dry_run: bool = False) -> str:
    """Install 3 cron jobs for Anchor. Returns crontab text."""
    text = render_crontab()
    if dry_run:
        return text
    try:
        current = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, check=False,
        ).stdout
    except FileNotFoundError:
        current = ""
    new = current + "\n" + text
    if current.strip():
        new = current + "\n# --- anchor ---\n" + text
    p = subprocess.run(["crontab", "-"], input=new, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"crontab install failed: {p.stderr}")
    return text


def uninstall_cron() -> int:
    """Remove all anchor cron lines. Returns count removed."""
    try:
        current = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, check=False,
        ).stdout
    except FileNotFoundError:
        return 0
    lines = current.splitlines()
    filtered = [line for line in lines if "anchor" not in line.lower()]
    n = len(lines) - len(filtered)
    new = "\n".join(filtered) + "\n"
    subprocess.run(["crontab", "-"], input=new, capture_output=True, text=True)
    return n


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("retrain-weekly")
    sub.add_parser("drift-check")
    sub.add_parser("health-probe")
    sub.add_parser("install")
    sub.add_parser("uninstall")
    sub.add_parser("show")
    args = parser.parse_args()
    if args.cmd == "install":
        print(install_cron(dry_run=False))
    elif args.cmd == "uninstall":
        n = uninstall_cron()
        print(f"removed {n} cron entries")
    elif args.cmd == "show":
        print(render_crontab())
    elif args.cmd == "retrain-weekly":
        from anchor.retrain import retrain_head, weekly_spotcheck
        import asyncio
        r_train = retrain_head(mode="shadow")
        r = asyncio.run(weekly_spotcheck())
        print(f"retrain mode={r_train['mode']} applied={r_train['applied']}, spotcheck pair_acc={r['pair_acc']:.3f}")
    elif args.cmd == "drift-check":
        from anchor.drift import compute_drift
        from anchor.head import prior_table
        d = compute_drift(prior_table)
        print(f"mean_kl={d['mean_kl']:.3f} has_baseline={d['has_baseline']}")
    elif args.cmd == "health-probe":
        from anchor.dashboard import metrics_snapshot
        from anchor.cost import check_daily_throttle, today_total
        from anchor.release.circuit_breaker import check_and_update
        m = metrics_snapshot(n_recent=10)
        daily = today_total()
        throttle = check_daily_throttle()
        # Use real spotcheck pair_acc if available, else fall back to mean_reward
        spotcheck_log = Path.home() / "anchor" / "logs" / "spotcheck.jsonl"
        rolling = None  # only update breaker when we have real pair_acc
        if spotcheck_log.exists():
            try:
                import json
                import time
                cutoff = time.time() - 86400
                accs = []
                with open(spotcheck_log) as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            if d.get("ts", 0) >= cutoff:
                                accs.append(float(d.get("pair_acc", 0)))
                        except (json.JSONDecodeError, ValueError, TypeError):
                            continue
                if accs:
                    rolling = sum(accs) / len(accs)
            except Exception:
                pass
        cb = check_and_update(rolling)
        _rd = "no_data" if rolling is None else rolling
        print(f"n={m['n_queries']} cost=¥{m['total_cost_yuan']:.2f}/daily=¥{daily:.2f} "
              f"drift={m['drift']['mean_kl']:.3f} rolling24h={_rd} "
              f"throttle={throttle} cb={'OPEN' if cb['open'] else 'closed'} cb_action={cb['action']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
