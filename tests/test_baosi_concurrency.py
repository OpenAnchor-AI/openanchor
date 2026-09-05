"""test_baosi_concurrency.py — BaosiAPI process-wide semaphore gate (6 cases).

The gate caps concurrent BaosiAPI work because all channels share one
upstream quota. A regression that lets N>MAX slip through will 429-storm
the upstream within minutes.
"""
from __future__ import annotations

import asyncio

from anchor import baosi_concurrency as bc


def test_baosi_channel_detection():
    """Only channels starting with 'baosiapi' count."""
    assert bc.is_baosi_channel("baosiapi") is True
    assert bc.is_baosi_channel("baosiapi-gpt") is True
    assert bc.is_baosi_channel(None) is False
    assert bc.is_baosi_channel("") is False
    assert bc.is_baosi_channel("openai") is False
    assert bc.is_baosi_channel("anthropic") is False


def test_should_shunt_at_threshold(monkeypatch):
    """When inflight >= threshold, should_shunt returns True."""
    monkeypatch.setenv("ANCHOR_BAOSI_SHUNT_THRESHOLD", "5")
    bc._inflight = 5
    assert bc.should_shunt() is True
    bc._inflight = 4
    assert bc.should_shunt() is False


def test_should_shunt_invalid_env_falls_back(monkeypatch):
    """A non-numeric env value must fall back to the safe default (28)."""
    monkeypatch.setenv("ANCHOR_BAOSI_SHUNT_THRESHOLD", "not-a-number")
    bc._inflight = 28
    assert bc.should_shunt() is True
    bc._inflight = 27
    assert bc.should_shunt() is False


def test_should_shunt_clamps_above_max(monkeypatch):
    """Setting threshold > MAX_CONCURRENCY clamps to MAX (still gates properly)."""
    monkeypatch.setenv("ANCHOR_BAOSI_SHUNT_THRESHOLD", "999")
    bc._inflight = bc.BAOSIAPI_MAX_CONCURRENCY  # already at cap
    assert bc.should_shunt() is True


def test_should_shunt_clamps_below_one(monkeypatch):
    """Setting threshold=0 must clamp to 1 (avoids divide-by-zero / always-true)."""
    monkeypatch.setenv("ANCHOR_BAOSI_SHUNT_THRESHOLD", "0")
    bc._inflight = 0
    # After clamp to 1: 0 >= 1 is False
    assert bc.should_shunt() is False
    bc._inflight = 1
    assert bc.should_shunt() is True


def test_slot_acquire_and_release():
    """The async slot() context manager releases on every exit path (incl. raise)."""

    async def runner():
        before = bc._inflight
        async with bc.slot():
            during = bc._inflight
            assert during == before + 1
            # Simulate a worker raising
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                pass
        after = bc._inflight
        assert after == before  # released even after raise

    asyncio.run(runner())


def test_reset_for_tests_restores_baseline():
    """After reset_for_tests(), the gate is back to a clean state."""
    bc._inflight = 12
    bc.reset_for_tests()
    assert bc._inflight == 0
    # Gate is a fresh semaphore of size BAOSIAPI_MAX_CONCURRENCY
    assert bc._gate._value == bc.BAOSIAPI_MAX_CONCURRENCY