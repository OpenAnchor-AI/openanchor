"""test_cli_doctor.py — `anchor doctor` diagnostic command (4 cases).

Coverage:
  1. Missing workers.yaml -> exit code 1 with a clear hint.
  2. Parses a well-formed workers.yaml + keys.yaml + reports each worker's
     probe status, with mocked urllib so no real network calls happen.
  3. Head baseline age is reported: fresh <= 7 days, stale > 7 days.
  4. Empty workers.yaml (no channels selected) still exits 0 with a clear
     "(no workers selected)" note in the report.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# 1. Missing workers.yaml
# ---------------------------------------------------------------------------


def test_doctor_exits_1_when_workers_yaml_missing(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from anchor.cli_doctor import doctor_command
    rc = doctor_command(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 1, "missing workers.yaml must fail the diagnostic"
    assert "workers.yaml missing" in out
    assert "anchor init" in out


# ---------------------------------------------------------------------------
# 2. Happy path: well-formed yaml + mocked probes
# ---------------------------------------------------------------------------


def test_doctor_happy_path_with_mocked_probes(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    # Write a valid workers.yaml + keys.yaml in the tmp HOME.
    # The wizard writes to ~/.anchor (= $HOME/.anchor) — same place.
    anchor_home = tmp_path / ".anchor"
    anchor_home.mkdir(parents=True, exist_ok=True)
    (anchor_home / "workers.yaml").write_text(
        "channels:\n"
        "  - name: opencode-zen\n"
        "    enabled: true\n"
        "    models:\n"
        "      - deepseek-v4-flash\n"
        "  - name: baosiapi\n"
        "    enabled: true\n"
        "    models:\n"
        "      - claude-fable-5\n"
    )
    (anchor_home / "keys.yaml").write_text(
        "keys:\n"
        "  OPENCODE_ZEN_API_KEY: sk-zen-1\n"
        "  BAOSIAPI_GPT_API_KEY: sk-baosi-1\n"
    )

    # Create a fresh head baseline (age = 0 days).
    data_dir = tmp_path / "anchor" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "head_baseline.json").write_text(
        json.dumps({"ts": time.time(), "W": [0.5] * 5, "prior": {}})
    )

    # Mock probe_worker to avoid network access. Return "ok" for first,
    # error for second; the report must surface both.
    from anchor import cli_doctor
    call_log: list[str] = []

    def fake_probe(worker, key):
        call_log.append(worker.name)
        if worker.name == "deepseek-v4-flash":
            return True, "ok"
        return False, "http 401: invalid_key"

    monkeypatch.setattr(cli_doctor, "probe_worker", fake_probe)

    rc = cli_doctor.doctor_command(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 1, "fable-5 probe failed; doctor should report non-ok"

    # The two workers are mentioned in the report.
    assert "deepseek-v4-flash" in out
    assert "claude-fable-5" in out
    # Probe was actually called for both selected workers.
    assert set(call_log) == {"deepseek-v4-flash", "claude-fable-5"}


# ---------------------------------------------------------------------------
# 3. Head baseline age report
# ---------------------------------------------------------------------------


def test_doctor_reports_fresh_head_baseline(tmp_path, capsys, monkeypatch):
    """A baseline < 7 days old is reported as fresh (✓)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / "anchor" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "head_baseline.json").write_text(
        json.dumps({"ts": time.time() - 86400, "W": [0.5] * 5, "prior": {}})  # 1 day old
    )
    from anchor.cli_doctor import doctor_command
    rc = doctor_command(argparse.Namespace())
    out = capsys.readouterr().out
    assert "age 1.0 day" in out
    assert "fresh" in out


def test_doctor_reports_stale_head_baseline(tmp_path, capsys, monkeypatch):
    """A baseline > 7 days old is reported as STALE and fails the check."""
    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / "anchor" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "head_baseline.json").write_text(
        json.dumps({"ts": time.time() - 30 * 86400, "W": [0.5] * 5, "prior": {}})
    )
    from anchor.cli_doctor import doctor_command
    rc = doctor_command(argparse.Namespace())
    out = capsys.readouterr().out
    assert "STALE" in out
    assert "Re-train" in out or "Re-train" in out or "re-train" in out
    assert rc == 1, "stale baseline must fail the overall check"


# ---------------------------------------------------------------------------
# 4. Empty workers.yaml
# ---------------------------------------------------------------------------


def test_doctor_with_empty_workers_yaml_exits_zero(tmp_path, capsys, monkeypatch):
    """An empty (no-channels) workers.yaml is a valid state, not a failure.

    User ran `anchor init` but disabled every provider; the file is still
    a record of intent. Doctor should report 'no workers selected' and
    exit 0 (no probes, no key coverage to check).
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    anchor_home = tmp_path / ".anchor"
    anchor_home.mkdir(parents=True, exist_ok=True)
    (anchor_home / "workers.yaml").write_text("channels: []\n")

    # No head baseline — also acceptable (only warnings, not failures,
    # for the empty-config case because there's nothing to validate).
    # We make a fresh baseline to keep the test focused on the YAML.
    data_dir = tmp_path / "anchor" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "head_baseline.json").write_text(
        json.dumps({"ts": time.time(), "W": [0.5] * 5, "prior": {}})
    )

    from anchor.cli_doctor import doctor_command
    rc = doctor_command(argparse.Namespace())
    out = capsys.readouterr().out
    assert rc == 0, f"empty config is valid; got rc={rc}, output:\n{out}"
    assert "no workers selected" in out
