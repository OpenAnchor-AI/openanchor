"""T-04: `anchor doctor` diagnostic command.

Three checks, in order:
    1. workers.yaml + keys.yaml parse cleanly and the keys they declare
       are present (or are already in env).
    2. Every selected worker in workers.yaml is reachable (probe via the
       same machinery as init's Q4).
    3. head baseline age — warn if the prior_table snapshot is older
       than 7 days (per the t_1f4a Head Audit 2026-09-08 ceiling).

Exit code: 0 if all checks pass, 1 if any check fails. The human-readable
report is printed to stdout; nothing is written to disk.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

from anchor.cli_init import (
    HEAD_BASELINE_MAX_AGE_DAYS,
    build_channel_groups,
    head_baseline_age_days,
    parse_keys_yaml,
    parse_workers_yaml,
    probe_worker,
)
from anchor.cli_init import _anchor_home as _anchor_home  # late-bound HOME
from anchor.cli_init import _workers_yaml_path as _workers_yaml_path
from anchor.cli_init import _keys_yaml_path as _keys_yaml_path
from anchor.config import WORKERS, _read_api_key, Worker


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_kv(rows: list[tuple[str, str]]) -> None:
    for k, v in rows:
        print(f"  {k:<22} {v}")


def doctor_command(args: argparse.Namespace) -> int:
    """Entry point for `anchor doctor`."""
    print("=== Anchor doctor ===")
    print()

    # ---- 1. YAML + keys -------------------------------------------------
    yaml_ok = True
    workers_yaml_path = _workers_yaml_path()
    keys_yaml_path = _keys_yaml_path()
    if not os.path.exists(workers_yaml_path):
        print(f"  ✗ workers.yaml missing at {workers_yaml_path}")
        print(f"    run `anchor init` first")
        yaml_ok = False
        selected: dict[str, list[str]] = {}
    else:
        try:
            with open(workers_yaml_path) as f:
                selected = parse_workers_yaml(f.read())
            if not selected:
                print(f"  ✓ workers.yaml parsed (no channels — empty config is valid)")
            else:
                print(f"  ✓ workers.yaml parsed ({len(selected)} channels declared)")
        except Exception as e:
            print(f"  ✗ workers.yaml parse error: {e}")
            selected = {}
            yaml_ok = False

    keys: dict[str, str] = {}
    if os.path.exists(keys_yaml_path):
        try:
            with open(keys_yaml_path) as f:
                keys = parse_keys_yaml(f.read())
            print(f"  ✓ keys.yaml parsed ({len(keys)} env vars)")
        except Exception as e:
            print(f"  ✗ keys.yaml parse error: {e}")
            yaml_ok = False
    else:
        print(f"  ! keys.yaml not found at {keys_yaml_path} (env-only setup is OK)")

    # Validate env coverage: for every selected worker, the channel's
    # primary env-var must resolve (either via keys.yaml or the live env).
    worker_by_name = {w.name: w for w in WORKERS}
    coverage_ok = True
    if selected:
        print()
        print("  Key coverage:")
        groups_by_name = {g.name: g for g in build_channel_groups()}
        for chan, names in selected.items():
            g = groups_by_name.get(chan)
            if g is None:
                print(f"  ! channel {chan!r} not in current registry; ignoring")
                continue
            env_name = g.primary_env
            val = keys.get(env_name) or _read_api_key(env_name)
            mark = "✓" if val else "✗"
            print(f"  {mark} {chan:<18} {env_name:<28} "
                  f"{'set' if val else 'MISSING'}")
            if not val:
                coverage_ok = False

    # ---- 2. Provider connectivity --------------------------------------
    print()
    print("  Provider probes:")
    probes_ok = True
    if not selected:
        print("  (no workers selected — skipping probes)")
    else:
        for chan, names in selected.items():
            for wname in names:
                w = worker_by_name.get(wname)
                if w is None:
                    print(f"  ✗ {wname}: not in current WORKERS registry")
                    probes_ok = False
                    continue
                g = next((g for g in build_channel_groups() if g.name == chan), None)
                env_name = g.primary_env if g else ""
                key = keys.get(env_name, "") or _read_api_key(env_name)
                if w.kind == "opencode-cli":
                    print(f"  ✓ {w.name:<24} opencode-cli (no probe)")
                    continue
                ok, detail = probe_worker(w, key)
                mark = "✓" if ok else "✗"
                short = (detail or "")[:60]
                print(f"  {mark} {w.name:<24} {short}")
                if not ok:
                    probes_ok = False

    # ---- 3. head baseline age ------------------------------------------
    print()
    print("  Head baseline:")
    age = head_baseline_age_days()
    if not age.get("exists"):
        print(f"  ! no baseline at {age.get('path')}")
        print("    run weekly_retrain.sh to create one")
    else:
        ad = age.get("age_days")
        stale = age.get("stale")
        if ad is None:
            print(f"  ✗ baseline unreadable: {age.get('reason')}")
        else:
            mark = "✗" if stale else "✓"
            print(f"  {mark} age {ad} day(s) (limit {HEAD_BASELINE_MAX_AGE_DAYS}) — "
                  f"{'STALE, re-train' if stale else 'fresh'}")
            print(f"    {age.get('path')}")
    baseline_ok = age.get("exists") and not age.get("stale")

    # ---- Summary -------------------------------------------------------
    print()
    # yaml_ok: parsed cleanly (or empty, which is a valid no-op config).
    # key_coverage: only relevant when at least one channel is declared.
    # probes: every selected worker reachable.
    # head_baseline: present and not stale.
    checks = {
        "yaml": yaml_ok,
        "key_coverage": (coverage_ok if selected else yaml_ok),
        "probes": probes_ok,
        "head_baseline": baseline_ok,
    }
    overall = all(checks.values())
    print("  Summary:")
    _print_kv([(k, "ok" if v else "FAIL") for k, v in checks.items()])
    print()
    print("  → " + ("All checks passed." if overall else
                 "Failures above. `anchor init` re-runs cleanly; "
                 "edit the yaml files directly for surgical fixes."))
    return 0 if overall else 1


# ---------------------------------------------------------------------------
# Parser glue
# ---------------------------------------------------------------------------


def add_doctor_subparser(sub) -> None:
    p = sub.add_parser(
        "doctor",
        help="check config validity + provider connectivity + head baseline age",
        description=(
            "Run read-only diagnostics over ~/.anchor/{workers,keys}.yaml "
            "and the head baseline snapshot. Exits 1 on any failure."
        ),
    )
    p.set_defaults(_cli_doctor=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("import as anchor.cli_doctor; wire into anchor.cli")
