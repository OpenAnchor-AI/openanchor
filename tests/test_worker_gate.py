"""test_worker_gate.py — worker enable / direct-alias gates (7 cases).

These gates decide whether a client is allowed to pin a worker by
direct model alias (production lockdown vs debug bypass), and whether
a worker's probe JSON recommends enabling it. Bugs here expose
unauthorised workers or block legitimate ones.
"""
from __future__ import annotations

import json

import pytest

from anchor import worker_gate as wg


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Each test starts from a clean env so we exercise the gate branches
    deterministically. (No implicit debug/auth-disabled defaults leaking.)"""
    for k in (
        "ANCHOR_ALLOW_DIRECT_WORKER",
        "ANCHOR_PROFILE",
        "ANCHOR_AUTH_DISABLED",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


def test_default_production_lockdown_denies(monkeypatch):
    """No env vars → safe production default (deny)."""
    monkeypatch.delenv("ANCHOR_ALLOW_DIRECT_WORKER", raising=False)
    monkeypatch.delenv("ANCHOR_PROFILE", raising=False)
    monkeypatch.delenv("ANCHOR_AUTH_DISABLED", raising=False)
    assert wg.allow_direct_worker_alias() is False


def test_debug_profile_allows(monkeypatch):
    monkeypatch.setenv("ANCHOR_PROFILE", "debug")
    assert wg.allow_direct_worker_alias() is True


def test_auth_disabled_allows(monkeypatch):
    monkeypatch.setenv("ANCHOR_AUTH_DISABLED", "1")
    assert wg.allow_direct_worker_alias() is True


def test_explicit_env_falsey_overrides(monkeypatch):
    """ANCHOR_ALLOW_DIRECT_WORKER=0 must deny even when profile=debug."""
    monkeypatch.setenv("ANCHOR_PROFILE", "debug")
    monkeypatch.setenv("ANCHOR_ALLOW_DIRECT_WORKER", "0")
    assert wg.allow_direct_worker_alias() is False


def test_explicit_env_truthy_overrides(monkeypatch):
    """ANCHOR_ALLOW_DIRECT_WORKER=1 must allow even in production."""
    monkeypatch.setenv("ANCHOR_ALLOW_DIRECT_WORKER", "true")
    assert wg.allow_direct_worker_alias() is True


def test_probe_recommends_enable_accepts_known_verdicts(tmp_path):
    """All known recommendation verdicts must be accepted."""
    p = tmp_path / "probe_claude-sonnet-5_results.json"
    for verdict in ("enable", "add_to_pool", "full_rollout", "pass", "passed", "ok"):
        p.write_text(json.dumps({"recommendation": verdict}))
        assert wg.probe_recommends_enable("claude-sonnet-5", path=p) is True


def test_probe_recommends_enable_rejects_unknown_verdict(tmp_path):
    p = tmp_path / "probe_x_results.json"
    p.write_text(json.dumps({"recommendation": "block"}))
    assert wg.probe_recommends_enable("x", path=p) is False


def test_probe_missing_file_returns_false(tmp_path):
    """Missing probe file → no recommendation (off by default)."""
    p = tmp_path / "does-not-exist.json"
    assert wg.probe_recommends_enable("x", path=p) is False


def test_probe_unparseable_file_returns_false(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("not valid json {{{")
    assert wg.probe_recommends_enable("x", path=p) is False


def test_env_or_probe_disabled_env_wins(tmp_path, monkeypatch):
    """Explicit env=0 must beat a positive probe recommendation."""
    p = tmp_path / "probe_x_results.json"
    p.write_text(json.dumps({"recommendation": "enable"}))
    monkeypatch.setenv("ANCHOR_ENABLE_X", "0")
    assert wg.env_or_probe_enabled("ANCHOR_ENABLE_X", "x") is False


def test_env_or_probe_truthy_env_wins(tmp_path, monkeypatch):
    """Explicit env=1 must enable regardless of probe."""
    monkeypatch.setenv("ANCHOR_ENABLE_X", "1")
    assert wg.env_or_probe_enabled("ANCHOR_ENABLE_X", "x") is True


def test_env_or_probe_uses_probe_when_env_unset(tmp_path, monkeypatch):
    """Without env var (raw=None), fall back to probe recommendation.

    Pass the probe path explicitly so the test doesn't depend on ANCHOR_ROOT.
    """
    p = tmp_path / "probe_x_results.json"
    p.write_text(json.dumps({"recommendation": "add_to_pool"}))
    monkeypatch.delenv("ANCHOR_ENABLE_X", raising=False)
    # When the probe is not at the default location we must point at our tmp file
    # by monkey-patching probe_result_path.
    monkeypatch.setattr(wg, "probe_result_path", lambda w: p)
    assert wg.env_or_probe_enabled("ANCHOR_ENABLE_X", "x") is True


def test_env_or_probe_falls_through_when_env_empty_string(tmp_path, monkeypatch):
    """When env var is set to '' (empty string), fall through to probe."""
    p = tmp_path / "probe_x_results.json"
    p.write_text(json.dumps({"recommendation": "add_to_pool"}))
    monkeypatch.setenv("ANCHOR_ENABLE_X", "")
    monkeypatch.setattr(wg, "probe_result_path", lambda w: p)
    assert wg.env_or_probe_enabled("ANCHOR_ENABLE_X", "x") is True


def test_probe_result_path_sanitizes_slashes():
    """Worker names with slashes (path-like) must not corrupt the data dir."""
    p = wg.probe_result_path("foo/bar")
    # The replacement converts / to _ so the filename stays in one component
    assert p.name == "probe_foo_bar_results.json"
    assert "/" not in p.name  # no path separator inside the leaf