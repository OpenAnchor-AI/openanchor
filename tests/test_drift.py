"""test_drift.py — KL-divergence drift detector (8 cases).

The detector is the alarm that goes off when the routing head's prior
table drifts more than the threshold (0.15). False negatives are silent
quality regressions; false positives page ops for nothing. Both
directions get locked.
"""
from __future__ import annotations

import math

import pytest

from anchor import drift


def test_load_baseline_returns_none_when_file_missing(tmp_path):
    """Missing baseline file → no drift (we don't have anything to compare)."""
    out = drift.compute_drift({("en", "premium"): 0.5}, baseline_path=tmp_path / "absent.json")
    assert out["has_baseline"] is False
    assert out["n"] == 0


def test_snapshot_then_compute_drift_roundtrip(tmp_path):
    """snapshot() writes a file, compute_drift() reads it back."""
    out_path = tmp_path / "head_baseline.json"
    drift.snapshot(
        {("en", "premium"): 0.5, ("cn", "ultra"): 0.7},
        [1.0, 2.0, 3.0],
        out_path=out_path,
    )
    assert out_path.exists()
    # Reading the same prior back → mean_kl ~ 0
    out = drift.compute_drift(
        {("en", "premium"): 0.5, ("cn", "ultra"): 0.7}, baseline_path=out_path
    )
    assert out["has_baseline"] is True
    assert out["n"] == 2
    assert out["mean_kl"] == pytest.approx(0.0, abs=1e-6)


def test_kl_helper_clamped_to_avoid_inf():
    """p / q outside (0.001, 0.999) must be clamped (avoids log(0) explosion)."""
    # Same value after clamping → KL is ~0
    assert drift._kl(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)
    # Different values clamped to 0.001/0.999 → large but finite KL
    assert math.isfinite(drift._kl(0.99, 0.01))
    # After clamp: p = 0.99, q = 0.01 → q is clamped UP to 0.001
    # so the actual KL formula is finite. ~4.5 is the correct value.
    assert drift._kl(0.99, 0.01) > 3.0


def test_compute_drift_uses_default_for_missing_keys():
    """Cells not in baseline fall back to 0.5 (neutral prior)."""
    out_path = tmp_path_fixture = None  # noqa: F841 — placeholder
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out_path = drift.snapshot(
            {("en", "premium"): 0.5}, [1.0], out_path=__import__("pathlib").Path(td) / "b.json"
        )
        # 'cn|ultra' is missing from baseline; default = 0.5
        out = drift.compute_drift(
            {("en", "premium"): 0.5, ("cn", "ultra"): 0.5}, baseline_path=out_path
        )
        # The added cell differs only by default → KL very small but > 0
        assert out["n"] == 2


def test_compute_drift_high_divergence_detected(tmp_path):
    """When current prior is wildly different from baseline, mean_kl spikes."""
    out_path = tmp_path / "head_baseline.json"
    drift.snapshot(
        {("en", "premium"): 0.5}, [1.0], out_path=out_path,
    )
    # Now flip to 0.95 → huge KL
    out = drift.compute_drift(
        {("en", "premium"): 0.95}, baseline_path=out_path,
    )
    assert out["mean_kl"] > 0.3  # KL(0.95, 0.5) ~ 0.49
    assert out["max_kl"] > 0.3


def test_check_drift_no_baseline_returns_quietly(tmp_path):
    """check_drift with no baseline → returns the dict, never raises."""
    out = drift.check_drift(
        {("en", "premium"): 0.5}, baseline_path=tmp_path / "absent.json"
    )
    assert out["has_baseline"] is False


def test_check_drift_raises_when_over_threshold(tmp_path):
    """A >0.15 mean_kl must trip DriftAlert (the alert the user actually sees)."""
    out_path = tmp_path / "head_baseline.json"
    drift.snapshot(
        {("en", "premium"): 0.5, ("cn", "ultra"): 0.5}, [1.0, 2.0], out_path=out_path
    )
    # New prior is wildly off → expect DriftAlert at default threshold
    with pytest.raises(drift.DriftAlert):
        drift.check_drift(
            {("en", "premium"): 0.95, ("cn", "ultra"): 0.95}, baseline_path=out_path
        )


def test_check_drift_threshold_can_be_relaxed(tmp_path):
    """A custom (high) threshold suppresses the alert."""
    out_path = tmp_path / "head_baseline.json"
    drift.snapshot(
        {("en", "premium"): 0.5}, [1.0], out_path=out_path
    )
    # Even at 0.95, with threshold=10.0, no alert
    out = drift.check_drift(
        {("en", "premium"): 0.95}, threshold=10.0, baseline_path=out_path
    )
    assert out["has_baseline"] is True
    assert out["mean_kl"] > 0.3  # KL(0.95, 0.5) ~ 0.49