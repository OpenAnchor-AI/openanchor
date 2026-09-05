"""Prometheus metrics for Anchor.

Exposed at /metrics (text format). All counters/labels are populated by
helpers below; the server's hot path calls record_request() and
record_worker_call() to feed them.
"""
from __future__ import annotations
import time

from prometheus_client import (
    CollectorRegistry, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,
)


REGISTRY = CollectorRegistry()


REQUESTS_TOTAL = Counter(
    "anchor_requests_total",
    "Total chat completion requests",
    ["tier", "worker", "status"],
    registry=REGISTRY,
)

REQUEST_DURATION = Histogram(
    "anchor_request_duration_seconds",
    "Chat completion request latency",
    ["tier", "worker"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)

TOKENS_TOTAL = Counter(
    "anchor_tokens_total",
    "Total tokens processed (estimated or upstream-reported)",
    ["tier", "worker", "direction"],  # direction: prompt | completion
    registry=REGISTRY,
)

COST_YUAN_TOTAL = Counter(
    "anchor_cost_yuan_total",
    "Total cost in CNY (yuan)",
    ["tier", "worker"],
    registry=REGISTRY,
)

WORKER_COOLDOWN = Gauge(
    "anchor_worker_cooldown_seconds_remaining",
    "Remaining cooldown per worker (0 if active)",
    ["worker"],
    registry=REGISTRY,
)

WORKER_LAST_SUCCESS = Gauge(
    "anchor_worker_last_success_ts",
    "Unix timestamp of last successful call per worker (0 if never)",
    ["worker"],
    registry=REGISTRY,
)

WORKER_CONSECUTIVE_FAILURES = Gauge(
    "anchor_worker_consecutive_failures",
    "Consecutive failed calls per worker (resets on success)",
    ["worker"],
    registry=REGISTRY,
)

WORKER_QUARANTINED = Gauge(
    "anchor_worker_quarantined",
    "1 if worker is quarantined by circuit breaker, else 0",
    ["worker"],
    registry=REGISTRY,
)


# v0.9.54 (Tier 1.2): counter for bandit vs linear-head disagreement.
# Reads from predict(): each request where BANDIT_SHADOW=1 increments
# per query_type when the two policies pick different workers. Operators
# can correlate this with judge_score from /v1/responses logs to
# decide when ANCHOR_BANDIT_ENABLED=1 is safe to enable.
SHADOW_BANDIT_DISAGREE = Counter(
    "anchor_shadow_bandit_disagree_total",
    "Number of requests where bandit shadow pick != linear head pick",
    ["qt"],
    registry=REGISTRY,
)

ACTIVE_REQUESTS = Gauge(
    "anchor_active_requests",
    "In-flight chat completion requests",
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def record_request(tier: str, worker: str, status: str, duration_s: float,
                   prompt_tokens: int = 0, completion_tokens: int = 0,
                   cost_yuan: float = 0.0) -> None:
    """Update request-level metrics. Cheap enough for the hot path."""
    REQUESTS_TOTAL.labels(tier=tier, worker=worker, status=status).inc()
    REQUEST_DURATION.labels(tier=tier, worker=worker).observe(duration_s)
    if prompt_tokens:
        TOKENS_TOTAL.labels(tier=tier, worker=worker, direction="prompt").inc(prompt_tokens)
    if completion_tokens:
        TOKENS_TOTAL.labels(tier=tier, worker=worker, direction="completion").inc(completion_tokens)
    if cost_yuan:
        COST_YUAN_TOTAL.labels(tier=tier, worker=worker).inc(cost_yuan)


def record_worker_success(worker: str) -> None:
    WORKER_LAST_SUCCESS.labels(worker=worker).set(time.time())
    WORKER_CONSECUTIVE_FAILURES.labels(worker=worker).set(0)


def record_worker_failure(worker: str) -> None:
    WORKER_CONSECUTIVE_FAILURES.labels(worker=worker).inc()


def set_worker_quarantined(worker: str, quarantined: bool) -> None:
    WORKER_QUARANTINED.labels(worker=worker).set(1 if quarantined else 0)


def set_worker_cooldown(worker: str, seconds_remaining: float) -> None:
    WORKER_COOLDOWN.labels(worker=worker).set(max(0.0, seconds_remaining))


def incr_active() -> None:
    ACTIVE_REQUESTS.inc()


def decr_active() -> None:
    ACTIVE_REQUESTS.dec()


# === Rolling 1h lane window ===
# Bounded deque so long-running gateway sessions do not confuse cumulative
# hit rates with 1h rolling hit rates. Lets dashboards separate "M3 timeout
# actually went up" from "M3 accumulated errors over 24h".
import threading as _thr_lane
from collections import deque as _dq_lane
_LANE_WINDOW_SEC = 3600
_lane_buf: _dq_lane = _dq_lane()
_lane_lock = _thr_lane.Lock()


class _LaneWindow:
    def __init__(self) -> None:
        self.buf: _dq_lane = _dq_lane()

    def append(
        self, lane: str, worker: str, status: str,
        *,
        primary_worker: str | None = None,
        primary_success: bool | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        ts = time.time()
        with _lane_lock:
            self.buf.append((
                ts, lane, worker, status, primary_worker,
                primary_success, fallback_reason,
            ))
            cutoff = ts - _LANE_WINDOW_SEC
            while self.buf and self.buf[0][0] < cutoff:
                self.buf.popleft()


_lane_window = _LaneWindow()


def lane_window_summary() -> dict:
    """Per-lane rolling 1h hit / error rate, plus primary worker stats.

    Returns:
        {
          "window_sec": 3600,
          "lanes": {
            <lane>: {
              "window_total": int,
              "window_error": int,
              "window_workers": {worker: count},
              "hit_rate":     {worker: ratio},
              "error_rate":   float,
              "primary": {
                <primary_worker>: {"total": int, "ok": int, "error_rate": float}
              },
              "fallback_reasons": {reason: count},
            }
          }
        }
    """
    cutoff = time.time() - _LANE_WINDOW_SEC
    lanes: dict[str, dict] = {}
    with _lane_lock:
        snapshot = list(_lane_window.buf)
    for entry in snapshot:
        if len(entry) == 7:
            ts, lane, worker, status, primary, success, reason = entry
        else:
            ts, lane, worker, status = entry[:4]
            primary = success = reason = None
        if ts < cutoff:
            continue
        stats = lanes.setdefault(lane, {
            "window_total": 0, "window_error": 0, "window_workers": {},
            "primary": {}, "fallback_reasons": {},
        })
        stats["window_total"] += 1
        if status == "error":
            stats["window_error"] += 1
        bw = stats["window_workers"]
        bw[worker] = bw.get(worker, 0) + 1
        if primary:
            pstats = stats["primary"].setdefault(primary, {"total": 0, "ok": 0})
            pstats["total"] += 1
            if success is True:
                pstats["ok"] += 1
        if reason:
            stats["fallback_reasons"][reason] = (
                stats["fallback_reasons"].get(reason, 0) + 1
            )
    for lane, stats in lanes.items():
        n = stats["window_total"]
        stats["hit_rate"] = {
            worker: round(count / n, 4) if n else 0
            for worker, count in stats["window_workers"].items()
        }
        stats["error_rate"] = round(stats["window_error"] / n, 4) if n else 0
        for pstats in stats["primary"].values():
            pstats["error_rate"] = (
                round(1 - pstats["ok"] / pstats["total"], 4)
                if pstats["total"] else 0
            )
    return {"window_sec": _LANE_WINDOW_SEC, "lanes": lanes}


# === AUDIT-2026-07-15 additions ===

AUTH_FAILURES = Counter(
    "anchor_auth_failures_total",
    "Total API auth failures (401)",
    ["path_prefix", "reason"],
    registry=REGISTRY,
)

STREAMING_CHUNKS_DROPPED = Counter(
    "anchor_streaming_chunks_dropped_total",
    "SSE chunks dropped due to upstream errors or stream break",
    ["worker", "reason"],
    registry=REGISTRY,
)

CIRCUIT_BREAKER_STATE = Gauge(
    "anchor_circuit_breaker_state",
    "Circuit breaker state per worker (0=open, 1=closed, 2=half_open)",
    ["worker"],
    registry=REGISTRY,
)

COST_BURN_RATE = Gauge(
    "anchor_cost_burn_yuan_per_hour",
    "Recent cost burn rate (yuan/hour, last 1h rolling)",
    ["tier"],
    registry=REGISTRY,
)

# v0.9.53-p6: routing lane labels expose per-lane (lane × worker × status)
# counts so dashboards can prove M3 hit rate changes from the
# agent-tool-heavy / dual-main rewrites.
ROUTING_LANE_REQUESTS_TOTAL = Counter(
    "anchor_routing_lane_requests_total",
    "Requests by routing lane (query_type) and chosen worker",
    ["lane", "worker", "status"],
    registry=REGISTRY,
)

ROUTING_LANE_DURATION_SECONDS = Histogram(
    "anchor_routing_lane_duration_seconds",
    "Request latency bucketed by routing lane and chosen worker",
    ["lane", "worker"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)

ROUTING_LANE_FALLBACK_TOTAL = Counter(
    "anchor_routing_lane_fallback_total",
    "Fallback hops by routing lane and target worker",
    ["lane", "from_worker", "to_worker", "reason"],
    registry=REGISTRY,
)

ROUTING_LANE_PRIMARY_TOTAL = Counter(
    "anchor_routing_lane_primary_total",
    "Primary worker attempts by routing lane and primary success status",
    ["lane", "worker", "status"],
    registry=REGISTRY,
)


def record_auth_failure(path: str, reason: str = "missing_key") -> None:
    """Called by auth middleware when a request is rejected."""
    prefix = "/" + path.lstrip("/").split("/", 1)[0] if path.startswith("/") else path
    AUTH_FAILURES.labels(path_prefix=prefix, reason=reason).inc()


def record_streaming_drop(worker: str, reason: str) -> None:
    STREAMING_CHUNKS_DROPPED.labels(worker=worker, reason=reason).inc()


def set_circuit_breaker_state(worker: str, state: str) -> None:
    """state: 'open' | 'closed' | 'half_open'."""
    mapping = {"open": 0, "closed": 1, "half_open": 2}
    CIRCUIT_BREAKER_STATE.labels(worker=worker).set(mapping.get(state, 1))


def set_cost_burn_rate(tier: str, yuan_per_hour: float) -> None:
    COST_BURN_RATE.labels(tier=tier).set(yuan_per_hour)


def record_routing_lane_request(lane: str, worker: str, status: str,
                                duration_s: float = 0.0, *,
                                primary_worker: str | None = None,
                                primary_success: bool | None = None,
                                fallback_reason: str | None = None) -> None:
    """Increment final and primary lane metrics."""
    lane = lane or "unknown"
    ROUTING_LANE_REQUESTS_TOTAL.labels(
        lane=lane, worker=worker, status=status
    ).inc()
    if primary_worker:
        ROUTING_LANE_PRIMARY_TOTAL.labels(
            lane=lane,
            worker=primary_worker,
            status="ok" if primary_success else "error",
        ).inc()
    if duration_s > 0:
        ROUTING_LANE_DURATION_SECONDS.labels(lane=lane, worker=worker).observe(duration_s)
    _lane_window.append(
        lane, worker, status,
        primary_worker=primary_worker,
        primary_success=primary_success,
        fallback_reason=fallback_reason,
    )


def record_routing_lane_fallback(lane: str, from_worker: str, to_worker: str,
                                 reason: str) -> None:
    """Record a fallback hop with its triggering reason."""
    ROUTING_LANE_FALLBACK_TOTAL.labels(
        lane=lane or "unknown",
        from_worker=from_worker or "unknown",
        to_worker=to_worker or "unknown",
        reason=reason or "unknown",
    ).inc()
