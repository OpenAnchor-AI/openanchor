import re
"""FastAPI gateway: 3 tier routes + hard rules + head routing + cost guard.

Pipeline: hard_rule -> cost.guard -> head.predict -> worker.call -> log -> return
"""
import asyncio
import json as _json
import os
import time

from anchor.logging_setup import (
    configure_logging as _configure_logging,
    install_request_id_middleware as _install_rid_mw,
)
from anchor.metrics import (
    record_request as _metrics_record,
    record_worker_success as _metrics_worker_ok,
    render as _metrics_render,
    set_worker_quarantined as _metrics_set_quarantined,
    record_auth_failure as _metrics_auth_failure,
    set_circuit_breaker_state as _metrics_set_circuit_breaker,
    set_cost_burn_rate as _metrics_set_cost_burn_rate,
)
import logging
_logger = logging.getLogger("anchor.server")
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as _StarletteHTTPException
from starlette.requests import ClientDisconnect as _ClientDisconnect
# Re-export from admin_ops (v0.9.35-p1)

from anchor.multiturn import next_tier as _mt_next_tier, record_session_turn as _mt_record_session
from anchor.pareto import select_cheapest_meeting_floor as _pareto_select
try:
    from anchor.judge import judge_batch as _judge_batch
except Exception:  # pragma: no cover
    _judge_batch = None
from anchor.cost import guard, CapHitError

from anchor import __version__ as _ANCHOR_VERSION

# Semantic cache (optional, enabled via ANCHOR_SEMANTIC_CACHE_ENABLED=1).
_SEMANTIC_CACHE_ENABLED: bool = os.environ.get("ANCHOR_SEMANTIC_CACHE_ENABLED", "0") == "1"

app = FastAPI(title="Anchor", version=_ANCHOR_VERSION)
# Structured logging + request_id middleware (B-01).
_configure_logging()
_install_rid_mw(app)

# E-05: CORS middleware. When allow_credentials=True, allow_origins must be
# an explicit list (not "*"). Default to ANCHOR_CORS_ORIGINS env var
# (comma-separated) or empty list = no CORS for credentialed requests.
from fastapi.middleware.cors import CORSMiddleware as _CORS
_CORS_RAW = os.environ.get("ANCHOR_CORS_ORIGINS", "")
_CORS_ORIGINS = [o.strip() for o in _CORS_RAW.split(",") if o.strip()] if _CORS_RAW else []
app.add_middleware(_CORS, allow_origins=_CORS_ORIGINS,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

# FIX 1 (audit #18): bound request body size to prevent OOM / massive bills.
# v0.9.53 (audit R5): reject Transfer-Encoding: chunked upfront and stream the
# body with a hard cap, so callers cannot lie about Content-Length or stream
# multi-GB bodies that would otherwise be accepted by Starlette/uvicorn.
# v0.9.53-p5: default bumped to 8MB. Codex / pi clients accumulate tool
# transcripts across long sessions; a 1MB cap produced 14+ 413s in one
# 2026-07-25 Codex burst even for normal usage. Override per-env via
# ANCHOR_MAX_REQUEST_BYTES.
_MAX_REQUEST_BYTES = int(os.environ.get("ANCHOR_MAX_REQUEST_BYTES", str(8 * 1024 * 1024)))

@app.middleware("http")
async def _enforce_max_body(request: Request, call_next):
    """Reject oversized bodies up-front and bound streaming reads.

    v0.9.53 (audit #S2): prior to this patch the middleware only enforced
    Content-Length. A client omitting CL (and not using chunked encoding)
    would slip through; Starlette/uvicorn would then buffer the entire
    stream into ``request._body`` before auth.enforce_key_quota could see
    it, allowing multi-GB slow-loris bodies to OOM the process. We now
    stream the body up to ``_MAX_REQUEST_BYTES`` and reject anything
    larger, with chunked transfers still rejected up-front. The bounded
    body is cached on ``request._body`` so downstream ``request.body()``
    reads do not block on the stream a second time.
    """
    te = (request.headers.get("transfer-encoding") or "").lower()
    if "chunked" in te:
        return JSONResponse(
            status_code=413,
            content={"error": {"message": "chunked transfer not supported; send Content-Length", "type": "request_too_large"}},
        )
    cl = request.headers.get("content-length")
    try:
        if cl is not None:
            try:
                if int(cl) > _MAX_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"error": {"message": f"Request body too large: {cl} > {_MAX_REQUEST_BYTES}", "type": "request_too_large"}},
                    )
                # Trust CL up to the cap; cache so downstream readers skip the
                # network read.
                body = bytearray()
                received = 0
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > _MAX_REQUEST_BYTES:
                        return JSONResponse(
                            status_code=413,
                            content={"error": {"message": f"Request body exceeded {_MAX_REQUEST_BYTES} bytes", "type": "request_too_large"}},
                        )
                    body.extend(chunk)
                request._body = bytes(body)
                return await call_next(request)
            except ValueError:
                import logging as _lg_cl
                _lg_cl.debug("body_length_parse_skip: invalid Content-Length header")
        # No Content-Length (legacy clients / HTTP/1.0): must stream the body
        # with a hard cap so callers cannot slow-loris an unlimited payload.
        body = bytearray()
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > _MAX_REQUEST_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"message": f"Request body exceeded {_MAX_REQUEST_BYTES} bytes", "type": "request_too_large"}},
                )
            body.extend(chunk)
        request._body = bytes(body)
        return await call_next(request)
    except _ClientDisconnect:
        _logger.warning(
            "client_disconnect_early body_bytes=%d path=%s",
            len(body) if "body" in dir() else -1,
            request.url.path,
        )
        return JSONResponse(
            status_code=499,
            content={"error": {"message": "client disconnected during request body upload", "type": "client_closed_request"}},
        )

# v0.9.46 (T-AUDIT-02): translate HTTPException to OpenAI-shape errors
# so openai-python 1.x SDK doesn't KeyError on {"error":...}.
@app.exception_handler(_StarletteHTTPException)
async def _openai_http_exc_handler(request, exc):
    _status = exc.status_code
    _type = "invalid_request_error" if _status in (400, 404, 422) else "api_error"
    if _status in (401, 403):
        _type = "authentication_error"
    if _status == 429:
        _type = "rate_limit_error"
    if _status in (502, 503, 504):
        _type = "service_unavailable_error"
    return JSONResponse(
        status_code=_status,
        content={"error": {"message": str(exc.detail), "type": _type, "code": _status}},
    )


# S12 (SRE R3): translate CapHitError to OpenAI-shape 503. Previously the
# guard() inside _route_tier would raise, but the decompose bypass meant
# cap hits would only sometimes surface. Now guard() is called before any
# branching; this handler ensures the CapHitError always returns a clean
# OpenAI-shape error regardless of which path triggered it.
@app.exception_handler(CapHitError)
async def _caphit_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"error": {"message": str(exc), "type": "service_unavailable_error",
                           "code": "monthly_hard_cap_hit"}},
    )

# v0.9.35-p1: register /admin/sla, /admin/cost/dashboard, /admin/cost/amortization
from anchor.admin_ops import register_admin_routes as _register_admin_ops_routes
_register_admin_ops_routes(app)


# Auth + key quota extracted to anchor.auth (v0.9.52-p3)
from anchor.auth import (
    register_auth_middleware as _register_auth_middleware,
    key_usage as _key_usage,
    truthy as _truthy,
)
# audit 2026-08-16 (F811): duplicate re-import removed — _metrics_auth_failure
# already bound at line ~19 from the same module.
_register_auth_middleware(app, metrics_auth_failure=_metrics_auth_failure)


# v0.9.25 (Stage 2 per user correction): per-tier fallback sequences.# v0.9.25 (Stage 2 per user correction): per-tier fallback sequences.
# Mirrors TIER_POOL_ORDER in fusion_modes.py — see comment there for rationale.
# fable-5 deliberately excluded (gate-1 fail disabled).


# Lifespan + background loops extracted to anchor.lifespan (v0.9.52-p3)
from anchor.lifespan import attach_lifespan as _attach_lifespan
_attach_lifespan(app)

# Back-compat re-exports (tests/admin monkeypatch these on server module)
# noqa: F401 — re-export block restored (audit 2026-08-16): ruff --fix trimmed
# these; tests import _lifespan / _call_worker / _do_decompose etc. from server.
from anchor.lifespan import (  # noqa: E402, F401
    _auto_recovery_loop,  # noqa: F401
    _auto_calibration_loop,  # noqa: F401
    _auto_head_save_loop,  # noqa: F401
    _auto_quarantine_sweep_loop,  # noqa: F401
    _lifespan,  # noqa: F401
    _HEAD_SAVE_INTERVAL_SEC,  # noqa: F401
    _RECOVERY_INTERVAL_SEC,  # noqa: F401
    _CALIB_INTERVAL_SEC,  # noqa: F401
    _QUARANTINE_SWEEP_INTERVAL_SEC,  # noqa: F401
)

# v0.9.46f: routing helpers (extracted to routing.py)
from anchor.routing_core import (  # noqa: F401
    classify_query,  # noqa: F401
    _call_worker,  # noqa: F401
    _route_tier,  # noqa: F401
    _cheap_try,  # noqa: F401
    _judge_quality,  # noqa: F401
    _do_decompose,  # noqa: F401
    _is_splittable,  # noqa: F401
    _maybe_decompose,  # noqa: F401
    _extract_user_text,  # noqa: F401
    _responses_content_to_text,  # noqa: F401
    normalize_openai_messages,  # noqa: F401
    _pick_tier_from_model,  # noqa: F401
    _annotate_pareto_meta,  # noqa: F401
    _resolve_pareto_worker,  # noqa: F401
    _fallback_candidates,  # noqa: F401
    _recent_sla_violations,  # noqa: F401
    _messages_have_image_url,  # noqa: F401
    _filter_chat_only_workers,  # noqa: F401
    _max_fallback_attempts,  # noqa: F401
    _filter_quarantined_workers,  # noqa: F401
    _MODEL_TO_TIER,  # noqa: F401
    MODEL_ALIASES,  # noqa: F401
    PUBLIC_MODEL_IDS,  # noqa: F401
)

# v0.9.46e: register /admin/* and /debug/* endpoints (extracted to admin.py)
from anchor.admin import register_admin_routes as _register_admin_routes
_register_admin_routes(app)



# v0.9.13 (B directive): streaming observability store
# Tracks per-worker TTFT/TBT/token stats. ~1KB per worker.
from collections import deque as _dq
_STREAM_STATS = {}  # worker_name -> {ttfts: deque[ms], tbts: deque[ms], tokens: deque[int], total: int, last_ts: float}
_STREAM_STATS_WINDOW = 200  # rolling window per worker
# L2 (memory LOW): cap the dict so a one-off worker name (typo / probe)
# can't grow it unbounded. 32 entries covers the configured pool (≤13)
# with headroom; evict oldest on overflow.
_STREAM_STATS_MAX_WORKERS = 32


def _stream_stats_setdefault(worker: str) -> dict:
    """setdefault on _STREAM_STATS with LRU-ish eviction at the cap."""
    if worker not in _STREAM_STATS and len(_STREAM_STATS) >= _STREAM_STATS_MAX_WORKERS:
        # Evict the oldest entry (dict preserves insertion order).
        _STREAM_STATS.pop(next(iter(_STREAM_STATS)))
    return _STREAM_STATS.setdefault(worker, {
        "ttfts": _dq(maxlen=_STREAM_STATS_WINDOW),
        "tbts": _dq(maxlen=_STREAM_STATS_WINDOW),
        "tokens": _dq(maxlen=_STREAM_STATS_WINDOW),
        "total": 0,
        "last_ts": 0.0,
        "errored": 0,
    })



from anchor.api_models import (
    Query, Feedback, OAChatRequest, OAResponsesRequest, OAImageGenRequest,
    # noqa: F401 — back-compat re-exports for tests importing from anchor.server
    Response,  # noqa: F401
    validate_tool_choice as _validate_tool_choice,  # noqa: F401
)


# Classifier stub — Day 6 will replace with real regex


@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": _ANCHOR_VERSION}


@app.get("/v1/keys/me")
def v1_keys_me(request: Request):
    record = getattr(request.state, "anchor_api_key_record", None)
    if not record:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return _key_usage(record)


















@app.post("/feedback")
def feedback(fb: Feedback):
    """Record outcome + update head prior. success in [0, 1]."""
    from anchor.feedback import record_outcome
    rid = record_outcome(
        fb.query, fb.tier, fb.worker,
        success=fb.success, prompt_len=fb.prompt_len, budget=fb.budget,
        cost_yuan=fb.cost_yuan, latency_ms=fb.latency_ms, query_type=fb.query_type,
    )
    # v1.0.x STAGE1: also append to session_log so calibration can fuse
    # human task_success with LLM judge_score.
    try:
        from anchor.session_log import append_session as _aps, hash_query as _hq2
        _aps(
            query=fb.query,
            routed_tier=fb.tier,
            model_used=fb.worker,
            latency_ms=fb.latency_ms,
            cost_yuan=fb.cost_yuan,
            judge_score=fb.success,  # feedback is ground-truth quality
            session_id=f"fb-{_hq2(fb.query)[:16]}",
            source="feedback",
            extra={
                "task_success": fb.success,
                "quality_tier": "manual",
                "model_alias": fb.worker,
            },
        )
    except Exception:  # pragma: silent
        pass  # never break feedback for logging
    return {"id": rid, "ok": True}


@app.get("/metrics")
def metrics():
    """Prometheus exposition format (text). Backward-compat JSON kept at /admin/dashboard."""
    from fastapi.responses import Response as _Resp
    try:
        from anchor.config import enabled_workers as _enabled_workers
        from anchor.dashboard import _burn_rate_per_tier
        from anchor.release.circuit_breaker import load_state as _load_cb_state

        quarantine = _load_cb_state().get("err_rate_quarantine", {})
        for worker in _enabled_workers():
            is_quarantined = bool(quarantine.get(worker.name, {}).get("quarantined", False))
            _metrics_set_quarantined(worker.name, is_quarantined)
            _metrics_set_circuit_breaker(worker.name, "open" if is_quarantined else "closed")
        for tier, yuan_per_hour in _burn_rate_per_tier(window_hours=1.0).items():
            _metrics_set_cost_burn_rate(tier, yuan_per_hour)
    except Exception as exc:
        _logger.warning("metrics state refresh failed: %s", exc)
    body, ct = _metrics_render()
    return _Resp(content=body, media_type=ct)


_READY_PROBE: dict = {}  # {ts, ok, n_workers_pinged, n_ponged}


@app.get("/readyz")
async def readyz():
    """Readiness probe: db + config + cached real worker probe.

    Returns 200 if all checks pass, 503 otherwise. Useful for k8s/load balancer
    integration. /healthz remains a simple liveness probe (process is up).

    S13 (SRE R2 HIGH): real-worker probe cached 5s.
    v0.9.51+: uses ``anchor.readyz_probe`` (key presence by default; optional HTTP).
    Does NOT treat bare Worker presence as success.
    """
    import time as _rt
    from anchor.db import ainitdb as _ainitdb
    from anchor.config import enabled_workers as _enabled
    from anchor.readyz_probe import probe_enabled_workers as _probe_workers
    checks = {}
    try:
        await _ainitdb()
        checks["db"] = "ok"
    except Exception as e:
        _logger.exception("READYZ_DB_ERR: %s", e)
        checks["db"] = f"error: {type(e).__name__}"
    workers = _enabled()
    checks["workers_enabled"] = len(workers)
    checks["workers_at_least_one"] = "ok" if len(workers) >= 1 else "fail"
    # S13: real worker probe with 5s TTL cache
    _now = _rt.time()
    _cache_age = _now - _READY_PROBE.get("ts", 0)
    if _cache_age >= 5.0:
        try:
            _pres = await _probe_workers(workers, timeout=2.5)
            _READY_PROBE.clear()
            _READY_PROBE.update({
                "ts": _now,
                "ok": bool(_pres.get("ok")),
                "n_workers_pinged": int(_pres.get("n_workers_pinged") or 0),
                "n_ponged": int(_pres.get("n_ponged") or 0),
                "mode": _pres.get("mode"),
                "reason": _pres.get("reason"),
                "worker": _pres.get("worker"),
            })
        except Exception as _probe_e:
            _logger.warning("READY_PROBE_SKIP err=%s", _probe_e)
            _READY_PROBE.clear()
            _READY_PROBE.update({
                "ts": _now, "ok": False, "n_workers_pinged": 0, "n_ponged": 0,
                "reason": f"probe_error:{type(_probe_e).__name__}",
            })
    _cache_age = max(0.0, _rt.time() - _READY_PROBE.get("ts", _now))
    checks["worker_probe_cached_ok"] = _READY_PROBE.get("ok", False)
    checks["worker_probe_age_s"] = round(_cache_age, 1)
    checks["worker_probe_mode"] = _READY_PROBE.get("mode")
    checks["worker_probe_reason"] = _READY_PROBE.get("reason")
    ok = (checks["db"] == "ok" and checks["workers_at_least_one"] == "ok"
          and _READY_PROBE.get("ok", False))
    from fastapi.responses import Response as _Resp
    return _Resp(
        content=_json.dumps({"ready": ok, "checks": checks}),
        media_type="application/json",
        status_code=200 if ok else 503,
    )



@app.get("/v1/models")
def v1_models():
    from anchor.fusion_modes import TIER_POOL_ORDER, WORKER_COST
    # Canonical floor source: pareto.TIER_FLOORS (unified v0.9.43).
    from anchor.pareto import TIER_FLOORS as _TIER_FLOORS_FOR_MODELS

    def model_info(model_id: str, tier: str) -> dict:
        # v0.9.28: image-gen tier uses a separate pool definition (image workers
        # are not in TIER_POOL_ORDER since they are not chat models).
        # v0.9.46g: also flag chat_only workers (e.g. agnes-2.0-flash) so
        # clients can see at /v1/models which workers are tool-incompatible.
        from anchor.config import WORKERS as _W_FOR_INFO
        _chat_only_for_model = {w.name for w in _W_FOR_INFO if getattr(w, "chat_only", False)}
        _IMAGE_POOL = ["agnes-image-2.1-flash"]
        if tier == "image-gen":
            fallback_chain = list(_IMAGE_POOL)
            return {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "anchor",
                "tier": tier,
                "default_worker": fallback_chain[0],
                "fallback_chain": fallback_chain,
                "price_floor_yuan_per_m": 0.0,  # both image workers are free
                "quality_floor": 0.7,
                "capability": "image-generation",
            }
        pool = list(TIER_POOL_ORDER.get(tier, TIER_POOL_ORDER["auto"]))
        chain = _fallback_candidates(tier, pool[0] if pool else "", pool) if pool else []
        fallback_chain = []
        for worker in [pool[0]] + chain if pool else chain:
            if worker and worker not in fallback_chain:
                fallback_chain.append(worker)
        costs = [WORKER_COST.get(worker, 0.0) for worker in fallback_chain]
        return {
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": "anchor",
            "tier": tier,
            "default_worker": fallback_chain[0] if fallback_chain else None,
            "fallback_chain": fallback_chain,
            "price_floor_yuan_per_m": min(costs) if costs else 0.0,
            "quality_floor": _TIER_FLOORS_FOR_MODELS.get(tier, 0.0),
            # v0.9.46g: tell clients if the default worker is chat-only (no
            # tool/function-call). Helps clients avoid agnes for tool use.
            "default_worker_chat_only": (fallback_chain[0] in _chat_only_for_model) if fallback_chain else False,
        }

    # v0.9.52: single public chat product `anchor`. Router auto-picks worker.
    # Single product; no sold tiers.
    from anchor.config import enabled_workers as _en_w

    def _chat_product_info() -> dict:
        # Surface the enabled worker pool as the product capability set.
        workers = [w.name for w in _en_w()]
        costs = []
        try:
            costs = [WORKER_COST.get(w, 0.0) for w in workers]
        except Exception:
            costs = []
        return {
            "id": "anchor",
            "object": "model",
            "created": 0,
            "owned_by": "anchor",
            "product": "single",
            "description": (
                "Pure router product: auto-selects among lean workers "
                "(flash / M3 / pro-fallback / sol / grok / fable) for quality÷cost. "
                "No product tiers; no built-in commercial billing."
            ),
            "tier": "auto",
            "default_worker": None,
            "worker_pool": workers,
            "fallback_chain": workers,
            "price_floor_yuan_per_m": min(costs) if costs else 0.0,
            "quality_floor": 0.0,
            "routing": "auto",
        }

    data = [_chat_product_info()]
    # Image capability remains a separate model id (not a chat SKU tier).
    data.append(model_info("anchor-image", "image-gen"))
    return {
        "object": "list",
        "notice": (
            "Pure mode: public chat product is only `anchor` "
            "(synonym: `anchor-auto`). Not a billing platform — "
            "for multi-tenant keys/quotas put OneAPI (or similar) in front. "
            "Legacy basic/premium/ultra model names and /basic|/premium|/ultra "
            "routes are removed. Use `anchor-image` for image generation."
        ),
        "data": data,
    }







def _responses_input_to_messages(payload, instructions: Optional[str] = None) -> list:
    """Convert Responses input items to raw Chat messages, then normalize.

    Conversion is intentionally syntax-only. All tool protocol repair (ids,
    orphan/dangling/duplicate/partial outputs and strict ordering) is delegated
    to ``normalize_tool_protocol`` so Chat Completions and Responses cannot
    drift into different provider-specific behavior.
    """
    from anchor.tool_protocol import normalize_tool_protocol

    raw: list[dict] = []
    if instructions:
        raw.append({"role": "system", "content": instructions})

    if isinstance(payload, list):
        index = 0
        while index < len(payload):
            item = payload[index]
            if not isinstance(item, dict):
                text = _responses_content_to_text(item)
                if text:
                    raw.append({"role": "user", "content": text})
                index += 1
                continue

            item_type = item.get("type")
            if item_type == "function_call":
                # Responses represents parallel calls as adjacent items. Group
                # them into one assistant turn before strict normalization.
                calls: list[dict] = []
                while index < len(payload):
                    current = payload[index]
                    if not isinstance(current, dict) or current.get("type") != "function_call":
                        break
                    call_id = current.get("call_id") or current.get("id")
                    name = current.get("name") or "unknown"
                    arguments = current.get("arguments") or "{}"
                    if not isinstance(arguments, str):
                        arguments = _json.dumps(arguments, ensure_ascii=False)
                    calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    })
                    index += 1
                raw.append({"role": "assistant", "content": "", "tool_calls": calls})
                continue

            if item_type in ("tool_result", "function_call_output", "tool_use"):
                msg = {
                    "role": "tool",
                    "content": _responses_content_to_text(
                        item.get("content") or item.get("output") or item.get("text", "")
                    ),
                }
                call_id = item.get("tool_call_id") or item.get("call_id")
                if call_id:
                    msg["tool_call_id"] = call_id
                if item.get("name"):
                    msg["name"] = item["name"]
                raw.append(msg)
                index += 1
                continue

            if "role" in item:
                role = item.get("role") or "user"
                if role == "developer":
                    role = "system"
                msg = {
                    "role": role,
                    "content": _responses_content_to_text(item.get("content", "")),
                }
                for key in ("name", "tool_call_id", "tool_calls", "function_call"):
                    if key in item:
                        msg[key] = item[key]
                raw.append(msg)
                index += 1
                continue

            text = _responses_content_to_text(item)
            if text:
                raw.append({"role": "user", "content": text})
            index += 1
    elif isinstance(payload, dict):
        raw.append({
            "role": payload.get("role") or "user",
            "content": _responses_content_to_text(
                payload.get("content", payload.get("text", ""))
            ),
        })
    else:
        raw.append({"role": "user", "content": str(payload or "")})

    normalized = normalize_tool_protocol(raw)
    return normalized or [{"role": "user", "content": ""}]

def _responses_tools_to_chat(tools):
    """Convert OpenAI Responses tool defs to chat.completions tools format."""
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        t_type = t.get("type") or "function"
        if t_type != "function":
            # skip host tools like tool_search; workers only understand functions
            continue
        if "function" in t and isinstance(t.get("function"), dict):
            out.append({"type": "function", "function": t["function"]})
            continue
        name = t.get("name")
        if not name:
            continue
        fn = {
            "name": name,
            "description": t.get("description") or "",
            "parameters": t.get("parameters") or t.get("input_schema") or {"type": "object", "properties": {}},
        }
        if "strict" in t:
            fn["strict"] = t["strict"]
        out.append({"type": "function", "function": fn})
    return out or None












# ---- Semantic cache helpers (v0.9.53+) ----
_SEMANTIC_CACHE: dict | None = None  # lazy import cache

def _sc_enabled() -> bool:
    return _SEMANTIC_CACHE_ENABLED

async def _sc_lookup(system_text: str, user_query: str, tools: list | None, salt: str = ""):
    if not _SEMANTIC_CACHE_ENABLED:
        return None
    global _SEMANTIC_CACHE
    if _SEMANTIC_CACHE is None:
        from anchor.semantic_cache import lookup as _scl
        _SEMANTIC_CACHE = {"fn": _scl}
    return await _SEMANTIC_CACHE["fn"](system_text, user_query, tools, salt=salt)

async def _sc_store_resp(system_text: str, user_query: str, response: str, model_used: str, tools: list | None, salt: str = ""):
    if not _SEMANTIC_CACHE_ENABLED or not response or not model_used:
        return
    try:
        from anchor.semantic_cache import store as _scs
        await _scs(system_text, user_query, response, model_used, tools, salt=salt)
    except Exception as _sce:
        _logger.warning("SEMANTIC_CACHE_STORE_ERR %s", _sce)

def _build_cached_response(cached: dict, req: OAChatRequest, last_user: str) -> dict:
    """Build a fake chat completion response from cached data."""
    import time as _t
    _pt = max(1, len(last_user or "") // 4)
    _ct = max(1, len(cached["response"]) // 4)
    return {
        "id": f"chatcmpl-cache-{int(_t.time()*1000)}",
        "object": "chat.completion",
        "created": int(_t.time()),
        "model": req.model,
        "system_fingerprint": f"anchor-v{_ANCHOR_VERSION}",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": cached["response"]},
                      "finish_reason": "stop"}],
        "usage": {"prompt_tokens": _pt, "completion_tokens": _ct, "total_tokens": _pt + _ct},
        "_anchor": {"tier": "auto", "worker": cached["model_used"], "cost_yuan": 0.0,
                     "cached": True, "similarity": cached.get("similarity", 1.0)},
    }


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: OAChatRequest, request: Request):
    # pi / OpenAI SDK compat: developer role + text-only content parts
    if req.messages:
        req.messages = normalize_openai_messages(req.messages)
    # T-AUDIT-06 (compat LOW L1): n>1 is silently truncated to 1 choice because
    # the underlying workers don't expose multi-sample generation. Log a
    # warning so operators see this in JSON logs (previously totally silent).
    # v0.9.52-p3: also strip n from req so extra_kw doesn't forward n>1 to workers.
    # MiniMax-M3 returns 400 on n>1 ("does not support n > 1").
    if req.n is not None and req.n > 1:
        _logger.warning("N_GT_1_TRUNCATED requested=%s delivered=1 (workers don't support multi-sample)", req.n)
        req.n = None

    last_user = ""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            last_user = _extract_user_text(m.get("content", ""))
            break

    # Semantic cache: build system text for cache key (used by both lookup and store).
    # audit 2026-08-16 (D9): the raw " ".join crashed with TypeError on
    # multimodal system content (list parts) — and ran before the cache-enabled
    # gate, so any image-bearing system message 500'd regardless of the feature.
    # Use the list-safe extractor (same one used for user content).
    _system_text = " ".join(
        _extract_user_text(m.get("content", ""))
        for m in req.messages
        if m.get("role") in ("system", "developer")
    )

    # Capture session_id from header (OpenCode / clients can send X-Session-ID)
    # Fallback to X-Request-ID. audit 2026-08-16: hoisted ABOVE the cache block —
    # the cache salt uses it and it used to be assigned after (NameError when
    # the semantic cache was enabled).
    session_id = (
        request.headers.get("X-Session-ID")
        or request.headers.get("X-Request-ID")
    )

    # Semantic cache lookup (non-stream only). If hit, return cached response
    # to avoid redundant API calls. Store happens at the end of the function
    # for non-stream responses that are not cached.
    if _sc_enabled() and not getattr(req, "stream", False) and last_user:
        # audit 2026-08-16 (D8): salt the cache key with the caller identity
        # (session id, else key name) so users never share cached answers.
        _sc_salt = (session_id or "") or (
            (getattr(request.state, "anchor_api_key_record", None) or {}).get("name") or ""
        )
        _cached = await _sc_lookup(_system_text, last_user, getattr(req, "tools", None), salt=_sc_salt)
        if _cached is not None:
            _logger.info("SEMANTIC_CACHE_HIT worker=%s sim=%.4f", _cached["model_used"], _cached["similarity"])
            return _build_cached_response(_cached, req, last_user)
    alias_forced_worker, tier = _pick_tier_from_model(req.model, last_user)
    from anchor.config import normalize_routing_lane as _nrl_lane
    tier = _nrl_lane(tier)
    # v0.9.36 (first-principles): cost-aware tier + Pareto worker pick.
    # Legacy path (head.predict) is preserved as fallback for any failure
    # in the new modules; never break the request because of new modules.
    try:
        from anchor._query_complexity import query_complexity as _qcomp
        _complexity = _qcomp(last_user or "")
        _parent_pid = request.headers.get("X-Parent-Turn-ID") or None
        _tier_cw = _mt_next_tier(
            _parent_pid,
            query_complexity_score=_complexity,
            session_id=session_id or None,
            query=last_user or None,
            requested_tier=None,  # v0.9.52: no multi-SKU floors
        )
        tier = _nrl_lane(_tier_cw or tier)
        # v0.9.46 (P1-A): plumb parent-turn quality_tier into pareto as
        # p_escalate so the cost-aware TCO models escalation probability
        # (error → 1.0, degraded → 0.8, fair → 0.4, good/excellent → 0.05).
        # Without this wire, TCO systematically under-costs multi-turn
        # sessions where the parent turn failed and we'll likely escalate.
        _p_escalate = 0.0
        if _parent_pid:
            try:
                from anchor.multiturn import _get_turn_by_query_hash as _gtbh2
                _p_turn = _gtbh2(_parent_pid) or {}
                _p_qt = (_p_turn or {}).get("quality_tier") or ""
                _p_escalate = {
                    "error": 1.0, "degraded": 0.8, "fair": 0.4,
                    "good": 0.05, "excellent": 0.05,
                }.get(_p_qt, 0.0)
            except Exception as _gtbh2_e:
                _logger.warning(
                    "p_escalate (server path) lookup failed: %s", _gtbh2_e)
        _ANCHOR_EMQ = float(os.environ.get("ANCHOR_EXPECTED_MONTHLY_QUERIES", "10000"))
        from anchor.fusion_modes import _auto_detect_query_type as _adt
        _qt_for_pareto = _adt(last_user or "")
        _pareto_pick, _pareto_tco = _pareto_select(
            last_user or "", _qt_for_pareto, tier, p_escalate=_p_escalate,
            expected_monthly_queries=_ANCHOR_EMQ,
        )
    except Exception as _pareto_e:
        _pareto_pick, _pareto_tco = None, None
        _logger.warning(
            "PARETO_SKIP: falling back to head.predict (%s). Routing quality may degrade.", _pareto_e,
            exc_info=True)
    has_image = _messages_have_image_url(req.messages)
    # v0.9.31: tier-routed vision worker (verified per-worker probe 2026-07-11).
    # v0.9.50-p1: Agnes/Kilo/Gemini removed; basic tier vision → minimax-m3
    # (only remaining multimodal non-Claude worker).
    # Probe results (1x1 / 64x64 red image, 30s timeout, build_client() direct):
    #   minimax-m3            ✅ 1.6s ¥119/mo   (1M ctx, multimodal)
    #   claude-haiku-4-5      ✅ 8.7s ¥1/M in   (cheap mid)
    #   claude-sonnet-5       ✅ 2.0s ¥3/M in   (mid-high)
    #   claude-opus-5       ✅ 2.2s ¥15/M in  (top quality)
    #   deepseek-v4-flash     ❌ 400 BadRequest (opencode-zen rejects image_url)
    # Old logic forced ALL vision to opus-4-8 ($15/M). New logic routes by tier:
    #   basic   → minimax-m3         (¥119/mo flat, 1.6s, multimodal)
    #   premium → minimax-m3          (same; upgrade to sonnet via tier-pool walk)
    #   ultra   → claude-opus-5     (top quality, preserved)
    if has_image:
        # v0.9.52 lean pool: single product — M3 is default multimodal;
        # hard-rules / pareto may still escalate to fable for quality.
        internal_forced_worker = "minimax-m3"
    else:
        internal_forced_worker = None
    from anchor.fusion_modes import _auto_detect_query_type
    qt = _auto_detect_query_type(last_user or "")
    # Tool-bearing native Chat requests use the generic agent lane. Responses
    # replays get a dedicated agent-tool-heavy lane: M3 is the preferred
    # amortized-cost primary, but it remains a routing preference rather than
    # a forced worker, so Pareto/fallback can move to Grok/Sol/Fable cleanly.
    if getattr(req, "tools", None):
        qt = "agent-tool-heavy" if req.internal_source == "responses" else "agent"
    elif req.internal_source == "responses" and any(
        m.get("tool_calls") or m.get("role") == "tool" for m in req.messages
    ):
        qt = "agent-tool-heavy"
    q = Query(
        query=last_user or "(empty)",
        query_type=qt,
        prompt_len=len(last_user),
        budget=0.5,
        messages=req.messages,  # v0.9.8: preserve vision content for worker
        tools=req.tools,  # v0.9.46h: pass OpenAI-format tools through to worker
        tool_choice=req.tool_choice,
    )
    from anchor.fusion_modes import _recommended_max_tokens
    from anchor.config import clamp_max_tokens as _clamp_mt
    mt = req.max_tokens if req.max_tokens and req.max_tokens > 0 else _recommended_max_tokens(last_user)
    # v0.9.51: hard-cap max_tokens per tier (client cannot request 200k on opus)
    mt = _clamp_mt(mt, tier)
    if req.max_completion_tokens:
        # keep o-series field but clamp too; also drive stream max_tokens (pi sends this)
        req.max_completion_tokens = _clamp_mt(req.max_completion_tokens, tier)
        mt = req.max_completion_tokens
    # S12 (SRE R3 CRITICAL): enforce cost cap BEFORE any branching. Previously
    # guard() was called only inside _route_tier (single-worker path); the
    # decompose path at the elif below called _call_worker via _maybe_decompose
    # without ever hitting the cap, allowing unbounded spend past ¥1100.
    try:
        guard()
    except Exception as _guard_e:
        # CapHitError is expected; anything else we log + re-raise (don't swallow)
        if _guard_e.__class__.__name__ == "CapHitError":
            raise
        _logger.warning("COST_GUARD_BYPASSED err=%s", _guard_e)
        raise
    # v0.9.11: Decompose mode (Opus 4.8 X.3 directive)
    # auto: cheap try → judge → decompose if low score
    # on:  force decompose
    # off: single-worker (legacy)
    decompose_resp = None
    if has_image:
        decompose_resp = None
    elif req.decompose in ("on", "auto"):
        decompose_resp = await _maybe_decompose(req, q, tier, mt, decompose=req.decompose)

    # design/hard override → quality ladder (not always sol).
    # Sacred/extreme → fable; hard design → sol; never free-lane hard work.
    if decompose_resp is not None and decompose_resp.get("path") == "design_hard_override":
        from anchor.config import WORKERS as _W_DH
        from anchor.fusion_modes import _difficulty_aware_route as _dar
        _enabled = {w.name for w in _W_DH if w.enabled}
        from anchor.config import ANCHOR_DISABLE_SOL_DEFAULT as _ANCHOR_DISABLE_SOL_DH
        target = _dar(qt, last_user or "")
        # V4.2 audit 2026-08-15: do NOT fallback to Sol when opt-out is on.
        if target == "gpt-5.6-sol" and _ANCHOR_DISABLE_SOL_DH:
            target = None
        if target not in _enabled:
            for cand in ("claude-fable-5", "grok-4-6-reasoning", "minimax-m3", "gpt-5.6-sol"):
                # V4.2: skip Sol from fallback ladder when opt-out is on.
                if cand == "gpt-5.6-sol" and _ANCHOR_DISABLE_SOL_DH:
                    continue
                if cand in _enabled:
                    target = cand
                    break
        internal_forced_worker = target
        decompose_resp = None  # force fallthrough to single-worker
    if decompose_resp is not None:
        # v0.9.11: build response from decompose path
        from anchor.fusion_modes import _query_difficulty as _qd2
        _d_val = _qd2(last_user)
        import time as _t
        merged_content = decompose_resp["content"]
        merged_meta = decompose_resp["meta"]
        decompose_path = decompose_resp.get("path", "unknown")
        _anchor_meta = {
            "tier": tier,
            "worker": merged_meta.get("worker", "decompose"),
            "cost_yuan": merged_meta.get("total_cost", merged_meta.get("cost_yuan", 0)),
            "latency_ms": merged_meta.get("latency_ms", 0),
            "confidence": 1.0,
            "d_value": _d_val,
            "decompose_path": decompose_path,
            "decompose_mode": req.decompose,
            "primary_worker": merged_meta.get("worker", "decompose"),
            "fallback_from": None,
            "degraded": False,
        }
        if "n_subtasks" in merged_meta:
            _anchor_meta["n_subtasks"] = merged_meta["n_subtasks"]
        if "judge" in decompose_resp:
            _anchor_meta["judge"] = decompose_resp["judge"]
        # Build a stub Response-like object for downstream code that uses resp.tier/worker/answer
        resp = type("Resp", (), {})()
        resp.tier = tier
        resp.worker = merged_meta.get("worker", "decompose")
        resp.cost_yuan = _anchor_meta["cost_yuan"]
        resp.latency_ms = _anchor_meta["latency_ms"]
        resp.confidence = 1.0
        resp.answer = merged_content
        resp.fallback_reason = None
    else:
        forced_worker = internal_forced_worker or alias_forced_worker
        if forced_worker is None and _truthy(os.environ.get("ANCHOR_ALLOW_WORKER_OVERRIDE")):
            forced_worker = req.worker_override
            # v0.9.53 (P1 fix): whitelist check on worker_override.
            # Without this, an arbitrary string (e.g. legacy short alias
            # "m3" instead of canonical "minimax-m3") reaches _call_worker
            # and triggers the `[stub:<name>]` fallback, polluting
            # session_log and calibration with non-canonical worker names.
            if forced_worker is not None:
                from anchor.config import WORKERS as _WO_CHECK
                _known_names = {w.name for w in _WO_CHECK}
                if forced_worker not in _known_names:
                    # Backward-compat: env flag explicitly opts in to
                    # accepting arbitrary names. WARN so operators notice
                    # pollution from misconfigured tests / load tools that
                    # send legacy short aliases.
                    _logger.warning(
                        "WORKER_OVERRIDE_UNKNOWN worker=%r (accepted via "
                        "ANCHOR_ALLOW_WORKER_OVERRIDE); calibration pollution "
                        "risk — use canonical name (e.g. minimax-m3)",
                        forced_worker,
                    )
        # Hard-rule ladder (flash→M3→fable) beats cost-only pareto cheap-first.
        # Without this, pareto always pins free flash and _route_tier never sees
        # TIER_HARD_RULES (forced_worker already set).
        if forced_worker is None:
            try:
                from anchor.fusion_modes import TIER_HARD_RULES as _THR
                _hr = _THR.get(tier) or _THR.get("auto")
                if _hr is not None:
                    _hard_pick = _hr(qt, last_user or "")
                    # Codex Responses tool replays are M3-first but not
                    # M3-forced. Let Pareto choose when available; use the hard
                    # rule only as the cold-start fallback.
                    if not (qt == "agent-tool-heavy" and _pareto_pick is not None):
                        forced_worker = _hard_pick
            except Exception as _hr_e:
                _logger.warning("HARD_RULE_SKIP err=%s", _hr_e)
        # v0.9.36 (first-principles): pareto pick only if not already forced
        # (vision/override/hard-rule take precedence) and worker is in tier pool.
        if forced_worker is None and _pareto_pick is not None:
            forced_worker = _pareto_pick
        # T-AUDIT-04 (compat MEDIUM): forward OpenAI spec fields (stop / top_p /
        # frequency_penalty / presence_penalty / response_format /
        # parallel_tool_calls / max_completion_tokens) to the worker via
        # extra_kw. None values are filtered out by _call_worker.
        # v0.9.52-p3: n is excluded — already stripped above if >1, and
        # workers (esp. MiniMax-M3) reject n>1 with 400. n=1 is the default
        # and doesn't need forwarding.
        _extra_kw = {
            k: getattr(req, k)
            for k in ("stop", "max_completion_tokens", "top_p",
                      "frequency_penalty", "presence_penalty",
                      "parallel_tool_calls", "response_format")
            if getattr(req, k, None) is not None
        }
        # Stream path normally uses select_only (pick worker, then true SSE)
        # to avoid double-billing. Exception: tools present — OpenCode/agent
        # clients stream by default, but client.stream() historically dropped
        # tools kwargs and tool_call deltas. For tools+stream, complete once
        # non-stream (with tools) and synthesize OpenAI SSE below.
        _stream_req = bool(getattr(req, "stream", False))
        _tools_req = bool(getattr(req, "tools", None))
        _select_only = _stream_req and not _tools_req
        _route_resp = await _route_tier(
            tier, q, max_tokens=mt, forced_worker=forced_worker,
            vision_only=has_image, extra_kw=_extra_kw or None,
            session_id=session_id or None,
            api_key=getattr(request.state, "anchor_api_key", None),
            select_only=_select_only,
        )
        _ans_dbg = _route_resp.answer or (
            "(stream deferred)" if _select_only else (
                f"(tool_calls x{len(_route_resp.tool_calls or [])})"
                if getattr(_route_resp, "tool_calls", None) else "(empty)"
            )
        )
        _logger.debug("RESPONSE_DEBUG forced=%s tier=%s worker=%s tc=%s ans=%s",
                          forced_worker, tier, _route_resp.worker,
                          bool(getattr(_route_resp, "tool_calls", None)),
                          _ans_dbg[:50])
        from anchor.fusion_modes import _query_difficulty
        _d_val = _query_difficulty(last_user)
        import time as _t
        # v0.9.19: BaseModel Response can't setattr; wrap in duck-type to attach vendor_placeholder
        resp = type("Resp", (), {})()
        resp.tier = _route_resp.tier
        resp.worker = _route_resp.worker
        resp.cost_yuan = _route_resp.cost_yuan
        resp.latency_ms = _route_resp.latency_ms
        resp.confidence = _route_resp.confidence
        resp.answer = _route_resp.answer
        resp.vendor_placeholder = False  # duck-type field, set by _route_tier in some paths
        resp.tool_calls = getattr(_route_resp, "tool_calls", None)  # v0.9.46h: carry through
        # T-AUDIT-05: propagate upstream finish_reason (length / content_filter / stop / ...)
        resp.finish_reason = getattr(_route_resp, "finish_reason", None)
        resp.primary_worker = getattr(_route_resp, "primary_worker", None) or _route_resp.worker
        resp.fallback_from = getattr(_route_resp, "fallback_from", None)
        resp.degraded = bool(getattr(_route_resp, "degraded", False) or resp.fallback_from)
        # v0.9.55: propagate terminal fallback reason so the session log
        # can distinguish transport/quota failures from real quality errors.
        resp.fallback_reason = getattr(_route_resp, "fallback_reason", None)
        # v0.9.76: pass the full fallback chain to the streaming path so
        # _gen can cascade on stream() failures (dpsk-flash-zen-429 →
        # minimax-m3 → luna) without surfacing an SSE error event.
        resp.fallback_chain = getattr(_route_resp, "fallback_chain", None)
        # Align with usage_log.jsonl (client records upstream prompt/completion).
        resp.usage = getattr(_route_resp, "usage", None) or None
        _anchor_meta = {
            "tier": resp.tier,
            "worker": resp.worker,
            "cost_yuan": resp.cost_yuan,
            "latency_ms": resp.latency_ms,
            "confidence": resp.confidence,
            "d_value": _d_val,
            "primary_worker": resp.primary_worker,
            "fallback_from": resp.fallback_from,
            "degraded": resp.degraded,
        }

    # E-01: prefer upstream-reported usage (same source as data/usage_log.jsonl);
    # fall back to local 1/4-char estimate only when upstream omits tokens.
    _upstream_usage = getattr(resp, "usage", None) or {}
    _pt = int(_upstream_usage.get("prompt_tokens") or _upstream_usage.get("input_tokens") or 0)
    _ct = int(_upstream_usage.get("completion_tokens") or _upstream_usage.get("output_tokens") or 0)
    if _pt <= 0:
        _pt = max(1, len(last_user or "") // 4) if (last_user or resp.answer) else 0
    if _ct <= 0 and resp.answer:
        _ct = max(1, len(resp.answer) // 4)
    _answer = resp.answer or ""
    _is_error_payload = _answer.startswith("[error:")
    # Quick-win 2026-07-26: when the worker error message carries our
    # AllKeysRateLimited marker (set by routing_core via KeyPool), emit
    # HTTP 503 + Retry-After so the caller (pi / openai-python / curl)
    # can stop hammering us and respect the cooldown. Falls back to the
    # generic 424 path below for any other [error:...] payload.
    if _is_error_payload and not getattr(resp, "tool_calls", None):
        if ":AllKeysRateLimited]" in _answer:
            from starlette.responses import JSONResponse as _JR_akrl
            import re as _re_akrl
            _m = _re_akrl.search(r"retry_after_s=(\d+)", _answer)
            _retry_after = int(_m.group(1)) if _m else 30
            return _JR_akrl(
                status_code=503,
                content={"error": {
                    "message": _answer,
                    "type": "rate_limit_error",
                    "code": "all_workers_rate_limited",
                    "worker": resp.worker,
                    "primary_worker": resp.primary_worker,
                    "fallback_from": resp.fallback_from,
                    "retry_after_s": _retry_after,
                }},
                headers={
                    "Retry-After": str(_retry_after),
                    "x-anchor-worker": str(resp.worker or ""),
                    "x-should-retry": "false",
                },
            )
        from starlette.responses import JSONResponse
        # 524 (Timeout) and 5xx are in openai-python's default retry set,
        # so 502 causes 3 SDK retries. 424 is the 4xx equivalent and is
        # NOT retried. Tell the client not to retry either, via
        # x-should-retry=false (openai 1.x honors this header).
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": _answer,
                "type": "upstream_error",
                "code": "all_workers_failed",
                "worker": resp.worker,
                "primary_worker": resp.primary_worker,
                "fallback_from": resp.fallback_from,
            }},
            headers={
                "x-anchor-worker": str(resp.worker or ""),
                "x-should-retry": "true",
            },
        )
    base = {
        "id": f"chatcmpl-anchor-{int(_t.time()*1000)}",
        "object": "chat.completion",
        "created": int(_t.time()),
        "model": req.model,
        # T-AUDIT-05 (compat LOW L4): surface static system_fingerprint
        # so openai-python 1.x clients can detect model-side changes.
        "system_fingerprint": f"anchor-v{_ANCHOR_VERSION}",
        "choices": [
            {
                "index": 0,
                "message": (
                    # v0.9.53 (audit R10): if the fallback chain exhausted and the
                    # final answer is a vendor error string, return a clean 502
                    # error envelope instead of stuffing the bracketed error
                    # into `content`. Codex/pi took that text as the assistant
                    # reply and re-injected it as a prompt, polluting turns.
                    {"role": "assistant", "content": (re.sub(r"\]<\]minimax\[>\[|\[<minimax>\]", "", str(resp.answer or "")).strip() if not _is_error_payload else None), "tool_calls": resp.tool_calls}
                    if getattr(resp, "tool_calls", None)
                    else {"role": "assistant", "content": (re.sub(r"\]<\]minimax\[>\[|\[<minimax>\]", "", str(resp.answer or "")).strip() or "收到，指令已执行！" if not _is_error_payload else _answer[:80] + " (error)")}
                ),
                # T-AUDIT-05 (compat MEDIUM M1): report upstream finish_reason
                # honestly. Fallback order: (1) tool_calls present -> "tool_calls";
                # (2) upstream reported "length" / "content_filter" -> surface it;
                # (3) default "stop".
                "finish_reason": (
                    "tool_calls" if getattr(resp, "tool_calls", None)
                    else (getattr(resp, "finish_reason", None) or "stop")
                ),
            }
        ],
        "usage": {
            "prompt_tokens": _pt,
            "completion_tokens": _ct,
            "total_tokens": _pt + _ct,
        },
    }
    # Mirror usage_log field names on _anchor for ops / clients that only read meta.
    _anchor_meta["usage"] = {
        "prompt_tokens": _pt,
        "completion_tokens": _ct,
        "total_tokens": _pt + _ct,
        "input_tokens": _pt,
        "output_tokens": _ct,
        "source": "upstream" if (_upstream_usage.get("prompt_tokens") or _upstream_usage.get("completion_tokens")) else "estimate",
    }
    # Non-stream only: metrics / session / feedback.
    # Stream path uses select_only (empty answer, cost=0); billing + logs run post-stream.
    if not getattr(req, "stream", False):
        # B-03: feed metrics. Best-effort, never breaks request.
        try:
            _metrics_record(
                tier=tier, worker=resp.worker, status="ok",
                duration_s=(resp.latency_ms / 1000.0) if resp.latency_ms else 0.0,
                prompt_tokens=_pt, completion_tokens=_ct,
                cost_yuan=float(resp.cost_yuan or 0.0),
            )
            _metrics_worker_ok(resp.worker)
        except Exception as _m_e:
            _logger.warning("metrics_record failed (request still served): %s", _m_e)
        # Session log (Day 17 / Fable 5 decision #1): JSONL mirror for v6 retrain
        try:
            from anchor.session_log import append_session, hash_query as _hq
            from anchor.judge_quality import local_quality as _hq_score, quality_tier_proxy as _qt
            if not session_id:
                # Fallback: derive stable id from query_hash so multi-turn re-asks group
                session_id = f"q-{_hq(last_user or '(empty)')}"
            # Day 18: lightweight local heuristic score (no LLM judge cost).
            # If user submits feedback later via /feedback, that overrides this.
            _j_score = _hq_score(resp.answer, latency_ms=resp.latency_ms)
            _j_tier = _qt(_j_score)
            _fail_reason = (getattr(resp, "fallback_reason", None) or "").lower()
            _answer_text = (resp.answer or "").strip()
            _error_kind = None
            if _j_tier in {"error", "degraded"}:
                if _fail_reason == "quota_exhausted":
                    _error_kind = "quota"
                elif _fail_reason in {"timeout", "connection", "rate_limit"}:
                    _error_kind = "transport"
                elif not _answer_text:
                    _error_kind = "empty"
                elif _answer_text.startswith(("[error:", "[stub:")):
                    _error_kind = "malformed"
                else:
                    _error_kind = "bad_answer"
            append_session(
                query=last_user or "(empty)",
                routed_tier=resp.tier,
                model_used=resp.worker,
                latency_ms=resp.latency_ms,
                cost_yuan=resp.cost_yuan,
                judge_score=_j_score if not getattr(resp, "vendor_placeholder", False) else None,
                session_id=session_id,
                source="user",
                error_kind=_error_kind,
                extra={
                    "d_value": _d_val,
                    "confidence": resp.confidence,
                    "model_alias": req.model,
                    "quality_tier": _j_tier,
                    "is_vendor_placeholder": bool(getattr(resp, "vendor_placeholder", False)),
                    "primary_worker": getattr(resp, "primary_worker", None),
                    "fallback_from": getattr(resp, "fallback_from", None),
                    "fallback_reason": getattr(resp, "fallback_reason", None),
                    "degraded": bool(getattr(resp, "degraded", False)),
                },
            )
        except Exception as _log_e:
            # Never break the request because of logging
            _logger.exception("log_err: %s", type(_log_e).__name__)
        # v0.9.36 (first-principles): record session-level TCO + quality_tier
        # for next_tier cost-aware gating. Never breaks request.
        if session_id:
            try:
                from anchor.judge_quality import quality_tier_proxy as _qt_session
                _qscore_s = _j_score if "_j_score" in locals() else None
                _qtier_s = _qt_session(_qscore_s) if _qscore_s is not None else None
                _mt_record_session(
                    session_id, turn_id_value=f"q-{_hq(last_user or '(empty)')[:16]}",
                    tier=resp.tier, worker=resp.worker, cost_yuan=resp.cost_yuan,
                    quality_tier=_qtier_s,
                )
            except Exception as _ses_e:
                _logger.warning("SESSION_RECORD skip %s", _ses_e)
        # v0.9.36 (first-principles): close learning loop. Auto-record outcome to
        # head.prior_table + queries table using local heuristic_quality as success
        # signal. /feedback remains the override path for ground-truth labels.
        try:
            from anchor.feedback import record_outcome as _record_outcome
            from anchor.judge_quality import local_quality as _hq_score_loop
            _success_s = _j_score if "_j_score" in locals() else None
            if _success_s is None:
                _success_s = _hq_score_loop(resp.answer, latency_ms=resp.latency_ms)
            if not getattr(resp, "vendor_placeholder", False):
                _record_outcome(
                    query=last_user or "(empty)",
                    tier=resp.tier,
                    worker=resp.worker,
                    success=float(_success_s),
                    prompt_len=len(last_user or ""),
                    budget=0.5,
                    cost_yuan=resp.cost_yuan,
                    latency_ms=resp.latency_ms,
                    query_type=qt,
                )
        except Exception as _fb_e:
            _logger.warning("FEEDBACK_AUTO skip %s", _fb_e)
    # SSE streaming mode (v0.9.51): worker selected via select_only (no pre-bill),
    # then a single upstream stream. Client disconnect aborts the generator.
    if getattr(req, "stream", False):
        from fastapi.responses import StreamingResponse
        import json as _json
        import time as _t2
        from anchor.clients.factory import build_client as _bc
        from anchor.config import WORKERS as _WORKERS
        _stream_t0 = _t2.time()
        # tools + stream: response already fully generated (select_only off).
        # Synthesize OpenAI-compatible SSE incl. tool_calls so agent clients
        # (OpenCode) work. Avoid second upstream call without tools.
        if getattr(req, "tools", None):
            async def _gen_tools_sse():
                def _chunk(choices, usage=None):
                    obj = {
                        **base,
                        "object": "chat.completion.chunk",
                        "choices": choices,
                    }
                    if usage is None:
                        obj.pop("usage", None)
                    else:
                        obj["usage"] = usage
                    return f"data: {_json.dumps(obj)}\n\n"

                yield _chunk([{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }])
                _tcs = getattr(resp, "tool_calls", None) or []
                if _tcs:
                    _delta_tcs = []
                    for _i, _tc in enumerate(_tcs):
                        _fn = (_tc.get("function") if isinstance(_tc, dict) else None) or {}
                        _delta_tcs.append({
                            "index": _tc.get("index", _i) if isinstance(_tc, dict) else _i,
                            "id": _tc.get("id") if isinstance(_tc, dict) else None,
                            "type": (_tc.get("type") if isinstance(_tc, dict) else None) or "function",
                            "function": {
                                "name": _fn.get("name") or "",
                                "arguments": _fn.get("arguments") or "",
                            },
                        })
                    yield _chunk([{
                        "index": 0,
                        "delta": {"tool_calls": _delta_tcs},
                        "finish_reason": None,
                    }])
                    yield _chunk([{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }])
                else:
                    _content = resp.answer or ""
                    if _content:
                        yield _chunk([{
                            "index": 0,
                            "delta": {"content": _content},
                            "finish_reason": None,
                        }])
                    yield _chunk([{
                        "index": 0,
                        "delta": {},
                        "finish_reason": getattr(resp, "finish_reason", None) or "stop",
                    }])
                if (req.stream_options or {}).get("include_usage"):
                    yield _chunk([], usage=base.get("usage") or {
                        "prompt_tokens": _pt,
                        "completion_tokens": _ct,
                        "total_tokens": _pt + _ct,
                    })
                yield "data: [DONE]\n\n"
                try:
                    _metrics_record(
                        tier=tier, worker=resp.worker, status="ok",
                        duration_s=(resp.latency_ms / 1000.0) if resp.latency_ms else 0.0,
                        prompt_tokens=_pt, completion_tokens=_ct,
                        cost_yuan=float(resp.cost_yuan or 0.0),
                    )
                    _metrics_worker_ok(resp.worker)
                except Exception as _m_e:
                    _logger.warning("metrics_record failed (tools stream): %s", _m_e)
            return StreamingResponse(_gen_tools_sse(), media_type="text/event-stream")
        _w = next((x for x in _WORKERS if x.name == resp.worker), None)
        if _w is None:
            # Fallback: chunk it
            async def _gen_fb():
                yield f"data: {_json.dumps({**base, 'object':'chat.completion.chunk', 'choices':[{'index':0,'delta':{'role':'assistant','content':resp.answer},'finish_reason':None}]})}\n\n"
                yield f"data: {_json.dumps({**base, 'object':'chat.completion.chunk', 'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_gen_fb(), media_type="text/event-stream")
        # v0.9.76: build the full fallback chain (primary first). The primary
        # may be dpsk-flash (Zen free) — currently 429-locked for hours — so we
        # need to try minimax-m3 / luna / grok next without a torn SSE.
        _stream_chain: list[tuple] = []
        for _w_name in (resp.fallback_chain or [resp.worker]):
            if not _w_name:
                continue
            _wobj = next((x for x in _WORKERS if x.name == _w_name), None)
            if _wobj is None:
                _logger.warning("STREAM_CHAIN_SKIP worker_missing name=%s", _w_name)
                continue
            try:
                from anchor.cooldown import is_cooling as _isc_chain
                if _isc_chain(_wobj.name):
                    _logger.warning("STREAM_CHAIN_SKIP worker_cooling name=%s", _wobj.name)
                    continue
            except Exception:  # pragma: silent — cooldown check failure must not block stream path
                pass
            _stream_chain.append((_wobj, _bc(_wobj)))
        if not _stream_chain:
            _stream_chain = [(_w, _bc(_w))]
        # First worker in chain carries messages/system_prompt (computed below).
        _primary_w, _primary_client = _stream_chain[0]
        _client = _primary_client
        # Build messages (inject system_prompt if applicable). For non-primary
        # workers in the chain, their system_prompt is injected on retry.
        _msgs = list(req.messages)
        if _primary_w.system_prompt and not any(m.get("role") == "system" for m in _msgs):
            _msgs = [{"role": "system", "content": _primary_w.system_prompt}] + _msgs
        _stream_latency = int((_t2.time() - _stream_t0) * 1000)
        async def _gen(qt=qt):
            # Initial role chunk (emitted once; subsequent fallback retries
            # skip this and only emit content + finish, since the client
            # already knows role=assistant from the initial frame).
            init = {**base, "object": "chat.completion.chunk",
                    "choices": [{"index":0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            yield f"data: {_json.dumps(init)}\n\n"
            full_text = []
            _emitted_len = 0  # S18
            import re as _re  # S20: module-level in _gen so fallback block always has it
            # v0.9.13 (B directive): streaming observability
            _ttft_ms = None
            _last_chunk_ts = None
            _tbt_ms_list = []
            _tok_count = 0
            try:
                # audit 2026-08-16 (A15): the upstream stream loop had no timeout —
                # if the vendor stalled mid-stream the SSE generator hung until
                # client disconnect, pinning a worker connection. Bound each
                # chunk wait (default 60s, env-tunable); a stall then emits the
                # existing SSE error event and terminates.
                _stream_chunk_timeout = float(
                    os.environ.get("ANCHOR_STREAM_CHUNK_TIMEOUT_S", "60"))
                _astream = _client.stream(_msgs, max_tokens=mt, temperature=req.temperature)
                while True:
                    try:
                        delta = await asyncio.wait_for(
                            _astream.__anext__(), timeout=_stream_chunk_timeout)
                    except StopAsyncIteration:
                        break
                    if await request.is_disconnected():
                        _logger.info("STREAM_CLIENT_DISCONNECT worker=%s true_sse", resp.worker)
                        break
                    if not delta:
                        continue
                    _now_ts = _t2.time()
                    if _ttft_ms is None:
                        _ttft_ms = int((_now_ts - _stream_t0) * 1000)
                    elif _last_chunk_ts is not None:
                        _tbt_ms_list.append(int((_now_ts - _last_chunk_ts) * 1000))
                    _last_chunk_ts = _now_ts
                    _tok_count += len(delta) // 4  # rough: 4 chars per token
                                        # v0.9.6 + S18 (compat HIGH): incremental strip AND emit clean chunks.
                    # Old impl dropped ALL content when upstream never emitted a close tag.
                    # Real workers (M3/dpsk/agnes/claude-* with thinking disabled) never emit
                    # such tags - bug caused empty SSE streams. Fix: _emitted_len counter so
                    # we never re-emit; strip any closed blocks and emit only new tail each iter.
                    full_text.append(delta)
                    text_so_far = "".join(full_text)
                    # Strip completed blocks; leave unclosed intact.
                    _pat = r"<think>.*?</think>"
                    cleaned = _re.sub(_pat, "", text_so_far, flags=_re.DOTALL)
                    # If a stray "<" (unclosed tag) remains, buffer until close or end.
                    if "<" in cleaned:
                        _open_idx = cleaned.find("<")
                        cleaned = cleaned[:_open_idx]
                    pending = cleaned[_emitted_len:]
                    if pending:
                        ch = {**base, "object": "chat.completion.chunk",
                              "choices": [{"index":0, "delta": {"content": pending}, "finish_reason": None}]}
                        yield f"data: {_json.dumps(ch)}\n\n"
                        _emitted_len += len(pending)
                # S18 flush: emit any tail that arrived but wasn't streamed yet.
                _pat_tail = r"<think>.*?</think>"
                _tail_so_far = _re.sub(_pat_tail, "", "".join(full_text), flags=_re.DOTALL)
                # If a stray '<' (unclosed tag) remains, trim to before it.
                if "<" in _tail_so_far:
                    _tail_so_far = _tail_so_far[:_tail_so_far.find("<")]
                _tail_pending = _tail_so_far[_emitted_len:]
                if _tail_pending:
                    ch = {**base, "object": "chat.completion.chunk",
                          "choices": [{"index":0, "delta": {"content": _tail_pending}, "finish_reason": None}]}
                    yield f"data: {_json.dumps(ch)}\n\n"
                    _emitted_len += len(_tail_pending)
            except Exception as _se:
                from anchor.metrics import record_streaming_drop
                record_streaming_drop(resp.worker, reason=type(_se).__name__)
                # v0.9.76: cascade the failure through _stream_chain before
                # surfacing an SSE error event. Each fallback worker emits
                # only content chunks (the role chunk already shipped), so
                # the client sees a single uninterrupted stream.
                _fb_done = False
                _fb_text: list = []
                _fb_emitted = 0
                for _idx_next in range(1, len(_stream_chain)):
                    _next_w, _next_client = _stream_chain[_idx_next]
                    try:
                        from anchor.metrics import record_streaming_drop as _rsd_fb
                        _rsd_fb(resp.worker, reason="chain_skip_to_" + _next_w.name)
                    except Exception:  # pragma: silent — metric failure must not abort fallback
                        pass
                    _next_msgs = list(req.messages)
                    if _next_w.system_prompt and not any(
                        m.get("role") == "system" for m in _next_msgs
                    ):
                        _next_msgs = [
                            {"role": "system", "content": _next_w.system_prompt}
                        ] + _next_msgs
                    _logger.warning(
                        "STREAM_FALLBACK primary=%s -> next=%s err=%s",
                        resp.worker, _next_w.name, type(_se).__name__,
                    )
                    try:
                        _astream_fb = _next_client.stream(
                            _next_msgs, max_tokens=mt, temperature=req.temperature,
                        )
                        while True:
                            try:
                                _delta_fb = await asyncio.wait_for(
                                    _astream_fb.__anext__(),
                                    timeout=_stream_chunk_timeout,
                                )
                            except StopAsyncIteration:
                                break
                            if not _delta_fb:
                                continue
                            _fb_text.append(_delta_fb)
                            _pat_fb = r"<think>.*?</think>"
                            _so_far_fb = "".join(_fb_text)
                            _cleaned_fb = _re.sub(_pat_fb, "", _so_far_fb, flags=_re.DOTALL)
                            if "<" in _cleaned_fb:
                                _open_idx_fb = _cleaned_fb.find("<")
                                _cleaned_fb = _cleaned_fb[:_open_idx_fb]
                            _pending_fb = _cleaned_fb[len("".join(_fb_text[:-1])):] if len(_fb_text) > 1 else _cleaned_fb
                            # Recompute properly from full text to avoid drift:
                            _pending_fb = _cleaned_fb[_fb_emitted:]
                            if _pending_fb:
                                _ch_fb = {
                                    **base,
                                    "object": "chat.completion.chunk",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": _pending_fb},
                                        "finish_reason": None,
                                    }],
                                }
                                yield f"data: {_json.dumps(_ch_fb)}\n\n"
                                _fb_emitted += len(_pending_fb)
                        full_text = _fb_text  # promote fallback output into billing/log path
                        _logger.warning(
                            "STREAM_FALLBACK %s OK primary=%s",
                            _next_w.name, resp.worker,
                        )
                        _fb_done = True
                        break
                    except Exception as _fb_e:
                        _logger.warning(
                            "STREAM_FALLBACK also_failed worker=%s err=%s",
                            _next_w.name, type(_fb_e).__name__,
                        )
                        continue
                if _fb_done:
                    pass  # fall through to the normal finish/done emission
                else:
                    # S16 (Compat LOW): emit OpenAI-shape error event and
                    # TERMINATE the stream (no trailing [DONE]).
                    # L2 (compat MEDIUM): use SSE event line `event: error`
                    # per the OpenAI streaming spec, not plain `data:`.
                    err_event = {
                        **base,
                        "object": "error",
                        "error": {
                            "message": str(_se)[:200],
                            "type": "api_error",
                            "code": "stream_error",
                        },
                    }
                    yield f"event: error\ndata: {_json.dumps(err_event)}\n\n"
                    # S16b: do NOT emit finish/stop or [DONE] after an error event.
                    try:
                        _wstat_e = _STREAM_STATS.get(resp.worker)
                        if _wstat_e is not None:
                            _wstat_e["errored"] = _wstat_e.get("errored", 0) + 1
                    except Exception as _wse:
                        _logger.warning("STREAM_STATS_SKIP worker=%s err=%s", resp.worker, _wse)
                    return
            done = {**base, "object": "chat.completion.chunk",
                    "choices": [{"index":0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {_json.dumps(done)}\n\n"
            # T-AUDIT-06 (compat LOW L2): openai-python >=1.30 sends
            # stream_options={"include_usage": true} and expects a usage
            # chunk immediately before [DONE]. Workers' stream() never
            # emits usage, so we synthesize one from local token estimates.
            if (req.stream_options or {}).get("include_usage"):
                try:
                    _pt_est = len(last_user or "") // 4
                    _ct_est = sum(len(d) for d in full_text) // 4
                    usage_chunk = {
                        **base,
                        "object": "chat.completion.chunk",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": _pt_est,
                            "completion_tokens": _ct_est,
                            "total_tokens": _pt_est + _ct_est,
                        },
                    }
                    yield f"data: {_json.dumps(usage_chunk)}\n\n"
                except Exception as _usage_e:
                    _logger.warning("STREAM_USAGE_CHUNK_SKIP %s", _usage_e)
            yield "data: [DONE]\n\n"
            # Session log (post-stream, using collected text)
            try:
                _full = "".join(full_text)
                if _full:
                    _ct_est = max(1, len(_full) // 4)
                    _pt_est = max(1, len(last_user or "") // 4)
                    # Bill once for stream path (select_only left cost_yuan=0)
                    try:
                        from anchor.cost import append_log as _al_stream
                        from anchor.workers import SACRED as _SACRED_SET
                        from anchor.sacred_guard import (
                            resolve_sacred_bucket as _rsb_s,
                            record_sacred_spend as _rss_s,
                        )
                        _cost_est = float(resp.cost_yuan or 0.0)
                        if _cost_est <= 0 and _w is not None:
                            _cost_est = (_w.cost_in * _pt_est + _w.cost_out * _ct_est) / 1e6
                        if _cost_est > 0:
                            _al_stream({"yuan": _cost_est, "tier": resp.tier, "worker": resp.worker, "stream": True})
                            resp.cost_yuan = _cost_est
                            if resp.worker in _SACRED_SET:
                                _bucket = _rsb_s(
                                    api_key=getattr(request.state, "anchor_api_key", None),
                                    client_session_id=session_id,
                                )
                                _rss_s(_bucket, resp.worker, _cost_est)
                    except Exception as _bill_e:
                        _logger.warning("STREAM_BILL_SKIP %s", _bill_e)
                    from anchor.session_log import append_session as _aps, hash_query as _hq2
                    from anchor.judge_quality import local_quality as _hq_score, quality_tier_proxy as _qt
                    from anchor.fusion_modes import _query_difficulty as _qd2
                    _d = _qd2(last_user or "")
                    _score = _hq_score(_full, latency_ms=int((_t2.time() - _stream_t0)*1000))
                    _sid = session_id or f"q-{_hq2(last_user or '(empty)')}"
                    _lat_ms = int((_t2.time() - _stream_t0) * 1000)
                    _aps(query=last_user or "(empty)", routed_tier=resp.tier, model_used=resp.worker,
                         latency_ms=_lat_ms, cost_yuan=resp.cost_yuan,
                         judge_score=_score, session_id=_sid, source="user",
                         extra={"d_value": _d, "confidence": resp.confidence,
                                "model_alias": req.model, "quality_tier": _qt(_score),
                                "stream": True, "ttft_ms": _ttft_ms,
                                "tbt_avg_ms": (sum(_tbt_ms_list)/len(_tbt_ms_list)) if _tbt_ms_list else None,
                                "tokens_est": _tok_count})
                    # Align with non-stream: metrics + multiturn session + prior feedback
                    try:
                        _metrics_record(
                            tier=tier, worker=resp.worker, status="ok",
                            duration_s=_lat_ms / 1000.0,
                            prompt_tokens=_pt_est, completion_tokens=_ct_est,
                            cost_yuan=float(resp.cost_yuan or 0.0),
                        )
                        _metrics_worker_ok(resp.worker)
                    except Exception as _m_se:
                        _logger.warning("STREAM_METRICS_SKIP %s", _m_se)
                    try:
                        _mt_record_session(
                            _sid,
                            turn_id_value=f"q-{_hq2(last_user or '(empty)')[:16]}",
                            tier=resp.tier, worker=resp.worker,
                            cost_yuan=float(resp.cost_yuan or 0.0),
                            quality_tier=_qt(_score),
                        )
                    except Exception as _ses_se:
                        _logger.warning("STREAM_SESSION_RECORD skip %s", _ses_se)
                    try:
                        from anchor.feedback import record_outcome as _record_outcome_s
                        _record_outcome_s(
                            query=last_user or "(empty)",
                            tier=resp.tier,
                            worker=resp.worker,
                            success=float(_score),
                            prompt_len=len(last_user or ""),
                            budget=0.5,
                            cost_yuan=float(resp.cost_yuan or 0.0),
                            latency_ms=_lat_ms,
                            query_type=qt,
                        )
                    except Exception as _fb_se:
                        _logger.warning("STREAM_FEEDBACK_AUTO skip %s", _fb_se)
            except Exception as _log_e2:
                _logger.exception("stream_log_err: %s", type(_log_e2).__name__)
            # v0.9.13: update streaming stats (success or aborted)
            try:
                _wstat = _stream_stats_setdefault(resp.worker)
                if _ttft_ms is not None:
                    _wstat["ttfts"].append(_ttft_ms)
                for _tbt in _tbt_ms_list:
                    _wstat["tbts"].append(_tbt)
                _wstat["tokens"].append(_tok_count)
                # total already pre-incremented once per stream attempt
                _wstat["last_ts"] = _t2.time()
            except Exception as _wsa:
                # S11 (SRE audit): post-stream stats accumulator failure;
                # /metrics may show stale data.
                _logger.warning("STREAM_STATS_ACC_SKIP worker=%s err=%s", _w, _wsa)
        # v0.9.13: pre-increment total_streams so even errored streams show up in stats.
        # _gen() catches streaming exceptions and yields an error chunk; it does NOT raise.
        # We bump total here to ensure errored streams are visible in /admin/streaming/stats.
        _pre_wstat = _stream_stats_setdefault(resp.worker)
        _pre_wstat["total"] += 1
        _pre_wstat["last_ts"] = _t2.time()
        return StreamingResponse(_gen(), media_type="text/event-stream")
    # Non-stream: store in semantic cache before returning.
    _resp_content = (
        (getattr(resp, "tool_calls", None) and None)
        or resp.answer
    )
    if _resp_content and not getattr(resp, "tool_calls", None):
        _sc_salt2 = (session_id or "") or (
            (getattr(request.state, "anchor_api_key_record", None) or {}).get("name") or ""
        )
        await _sc_store_resp(_system_text, last_user, _resp_content, resp.worker,
                             getattr(req, "tools", None), salt=_sc_salt2)
    return {**base, "_anchor": _anchor_meta}


# v0.9.46k (codex Responses compat): SSTI-able chunk size for streaming.
# OpenAI's own SDK chunks on the wire at ~token boundaries; we chunk on
# byte boundaries here (no token-level access). 96 chars ≈ 24 tokens for en
# text — fine for SSE pipe latency.
_RESPONSES_STREAM_CHUNK_BYTES = 96


async def _v1_responses_stream_gen(req: OAResponsesRequest, request: Request, chat=None):
    """OpenAI Responses SSE stream (codex wire format).

    Wire payload rules (required by Codex / OpenAI Responses clients):
      - every data JSON includes `type` + monotonic `sequence_number`
      - created/in_progress/completed wrap the response object under `response`
      - text path emits output_text deltas
      - tool path emits function_call items + function_call_arguments deltas

    v0.9.69 (audit codex 502 cascade): `chat` may be pre-fetched by the
    `v1_responses` handler so we don't pay the cascade-exhaust probe twice
    (call ordering / metrics / session_log). When None, fetch internally
    (legacy callers + the non-stream probe path).
    """
    import json as _json_stream
    import time as _t_stream
    import re as _re_stream
    from anchor.config import clamp_max_tokens as _cmt_r

    if chat is None:
        max_tokens = _cmt_r(req.max_output_tokens or req.max_tokens or 1024, "premium")
        chat_tools = _responses_tools_to_chat(getattr(req, "tools", None))
        chat_req = OAChatRequest(
            model=req.model,
            messages=_responses_input_to_messages(req.input, req.instructions),
            temperature=req.temperature,
            max_tokens=max_tokens,
            stream=False,
            decompose="off",
            tools=chat_tools,
            tool_choice=req.tool_choice if chat_tools else None,
            parallel_tool_calls=req.parallel_tool_calls,
            internal_source="responses",
        )
        try:
            chat = await v1_chat_completions(chat_req, request)
        except Exception as _vcc_err:
            _logger.error("RESPONSES_VCC_ERR %s", _vcc_err)
            _rid = f"resp-anchor-{int(_t_stream.time()*1000)}"
            yield f"event: error\ndata: {_json_stream.dumps({'sequence_number': 0, 'type': 'error', 'error': {'message': str(_vcc_err)[:200]}, 'status': 'failed'})}\n\n"
            yield f"event: response.completed\ndata: {_json_stream.dumps({'sequence_number': 1, 'type': 'response.completed', 'response': {'id': _rid, 'object': 'response', 'status': 'failed', 'error': {'message': str(_vcc_err)[:200]}}})}\n\n"
            return
    # v0.9.54 fix (anchor 502 cascade bug): v1_chat_completions may return a
    # JSONResponse (424 Failed Dependency) when the upstream fallback chain
    # exhausts — e.g. when the only configured worker (minimax-m3) hits
    # Token-Plan rate limits and every other worker is disabled/quarantined.
    # The streaming generator previously called chat.get("choices") on the
    # JSONResponse and crashed with `AttributeError: 'JSONResponse' object
    # has no attribute 'get'`. The unhandled exception propagated out of the
    # ASGI app, which uvicorn treats as a fatal server error → process exit →
    # restart loop. Every subsequent request (curl / Hermes CLI / Codex CLI)
    # then saw 502 until the in-memory KeyPool was reset.
    #
    # Mirror the non-streaming /v1/responses handler at line 1685+ which
    # already does the right thing: detect the JSONResponse, parse its body,
    # and yield a single error SSE event so the client sees a clean OpenAI
    # Responses-shape error envelope instead of a torn connection.
    from starlette.responses import JSONResponse as _JR_stream
    if isinstance(chat, _JR_stream) or (not isinstance(chat, dict) and hasattr(chat, "body")):
        try:
            import json as _jresp_err
            err_body = chat.body if isinstance(chat.body, (bytes, bytearray)) else chat.body
            if isinstance(err_body, (bytes, bytearray)):
                err_body = err_body.decode("utf-8", errors="replace")
            err = _jresp_err.loads(err_body) if err_body else {"error": {"message": "upstream_error"}}
        except Exception:
            err = {"error": {"message": "upstream_error"}}
        # Yield a clean SSE error envelope then close the stream.
        seq_err = 0

        def _sse_err(event_type: str, payload: dict) -> str:
            nonlocal seq_err
            body = {"sequence_number": seq_err, "type": event_type, **payload}
            seq_err += 1
            return f"event: {event_type}\ndata: {_json_stream.dumps(body, ensure_ascii=False)}\n\n"
        yield _sse_err("error", {"error": err.get("error", err), "status": "failed"})
        yield _sse_err("response.completed", {
            "response": {
                "id": f"resp-anchor-{int(_t_stream.time()*1000)}",
                "object": "response",
                "status": "failed",
                "error": err.get("error", err),
            }
        })
        return
    try:
        choice0 = (chat.get("choices") or [{}])[0]
        message = choice0.get("message") or {}
        answer = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        answer = _re_stream.sub(r"\u003c/?think\u003e", "", answer or "")
        anchor_meta = chat.get("_anchor", {}) or {}
        response_id = f"resp-anchor-{int(_t_stream.time()*1000)}"
        created_at = int(_t_stream.time())

        seq = 0

        def _sse(event_type: str, payload: dict) -> str:
            nonlocal seq
            body = {"sequence_number": seq, "type": event_type, **payload}
            seq += 1
            return f"event: {event_type}\ndata: {_json_stream.dumps(body, ensure_ascii=False)}\n\n"

        raw_usage = chat.get("usage", {}) or {}
        prompt_tokens = int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0)
        total_tokens = int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_tokens_details": {"cached_tokens": int((raw_usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)},
            "output_tokens_details": {"reasoning_tokens": int((raw_usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)},
        }

        response_obj = {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": req.model,
            "output": [],
            "usage": usage,
            "error": None,
            "metadata": {"anchor": anchor_meta} if anchor_meta else {},
            "parallel_tool_calls": True if req.parallel_tool_calls is None else bool(req.parallel_tool_calls),
            "tool_choice": req.tool_choice if req.tool_choice is not None else "auto",
            "tools": list(req.tools or []),
            "temperature": req.temperature,
            "store": False,
        }

        yield _sse("response.created", {"response": response_obj})
        yield _sse("response.in_progress", {"response": response_obj})

        output_items = []

        if tool_calls:
            for i, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                arguments = fn.get("arguments") or ""
                if not isinstance(arguments, str):
                    arguments = _json_stream.dumps(arguments)
                call_id = tc.get("id") or f"call_anchor_{created_at}_{i}"
                item_id = f"fc_anchor_{created_at}_{i}"
                item = {
                    "id": item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                }
                yield _sse("response.output_item.added", {"output_index": i, "item": item})
                acc = ""
                step = max(_RESPONSES_STREAM_CHUNK_BYTES, 64)
                for j in range(0, len(arguments), step):
                    chunk = arguments[j:j + step]
                    acc += chunk
                    yield _sse(
                        "response.function_call_arguments.delta",
                        {
                            "item_id": item_id,
                            "output_index": i,
                            "delta": chunk,
                        },
                    )
                yield _sse(
                    "response.function_call_arguments.done",
                    {
                        "item_id": item_id,
                        "output_index": i,
                        "arguments": acc,
                    },
                )
                final_item = {
                    "id": item_id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": acc,
                }
                yield _sse("response.output_item.done", {"output_index": i, "item": final_item})
                output_items.append(final_item)
        else:
            item_id = f"msg-anchor-{created_at}"
            output_item = {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "status": "in_progress",
            }
            yield _sse("response.output_item.added", {"output_index": 0, "item": output_item})
            part_index = 0
            output_text_part = {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            }
            yield _sse(
                "response.content_part.added",
                {
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": part_index,
                    "part": output_text_part,
                },
            )
            accumulated = ""
            for i in range(0, len(answer), _RESPONSES_STREAM_CHUNK_BYTES):
                chunk = answer[i:i + _RESPONSES_STREAM_CHUNK_BYTES]
                accumulated += chunk
                yield _sse(
                    "response.output_text.delta",
                    {
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": part_index,
                        "delta": chunk,
                        "logprobs": [],
                    },
                )
            yield _sse(
                "response.output_text.done",
                {
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": part_index,
                    "text": accumulated,
                    "logprobs": [],
                },
            )
            final_part = {**output_text_part, "text": accumulated}
            yield _sse(
                "response.content_part.done",
                {
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": part_index,
                    "part": final_part,
                },
            )
            final_item = {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": accumulated, "annotations": [], "logprobs": []}],
                "status": "completed",
            }
            yield _sse("response.output_item.done", {"output_index": 0, "item": final_item})
            output_items.append(final_item)

        completed_at = int(_t_stream.time())
        response_obj = {
            **response_obj,
            "status": "completed",
            "completed_at": completed_at,
            "output": output_items,
            "usage": usage,
        }
        yield _sse("response.completed", {"response": response_obj})
        yield _sse("response.done", {"response": response_obj})
    except Exception as _resp_err:
        _logger.error("RESPONSES_STREAM_ERR %s", _resp_err)
        response_id = f"resp-anchor-{int(_t_stream.time()*1000)}"
        yield f"event: error\ndata: {_json_stream.dumps({'error': {'message': str(_resp_err)[:200]}, 'status': 'failed'})}\n\n"
        yield f"event: response.completed\ndata: {_json_stream.dumps({'response': {'id': response_id, 'object': 'response', 'status': 'failed', 'error': {'message': str(_resp_err)[:200]}}})}\n\n"


@app.post("/v1/responses")
async def v1_responses(req: OAResponsesRequest, request: Request):
    if req.stream:
        from fastapi.responses import StreamingResponse
        # v0.9.69 (audit codex 502 cascade): probe v1_chat_completions BEFORE
        # wrapping in StreamingResponse. When the fallback chain is exhausted
        # v1_chat_completions returns a 424 / 503 JSONResponse. Wrapping it
        # inside a 200 SSE stream makes codex CLI 0.146.0 interpret the
        # response as a torn connection (it does NOT honour `event: error`
        # inside an otherwise-successful 200 stream) and retry 5× →
        # turn.failed. Returning the JSONResponse directly surfaces the
        # proper HTTP status (with Retry-After / x-should-retry=false) so
        # codex fails fast without retrying. Curl callers see identical
        # JSON since they already accept the error envelope either way.
        from anchor.config import clamp_max_tokens as _cmt_r_probe
        _probe_max_tokens = _cmt_r_probe(req.max_output_tokens or req.max_tokens or 1024, "premium")
        _probe_tools = _responses_tools_to_chat(getattr(req, "tools", None))
        _probe_req = OAChatRequest(
            model=req.model,
            messages=_responses_input_to_messages(req.input, req.instructions),
            temperature=req.temperature,
            max_tokens=_probe_max_tokens,
            stream=False,
            decompose="off",
            tools=_probe_tools,
            tool_choice=req.tool_choice if _probe_tools else None,
            parallel_tool_calls=req.parallel_tool_calls,
            internal_source="responses",
        )
        _probe = await v1_chat_completions(_probe_req, request)
        from starlette.responses import JSONResponse as _JR_probe
        if not isinstance(_probe, dict) and isinstance(_probe, _JR_probe):
            # Transform upstream 424/503 envelope into OpenAI Responses-shape
            # error JSON so codex CLI's parser matches what the non-stream
            # branch at :1832+ returns. Headers (Retry-After, x-should-retry,
            # x-anchor-worker) are preserved so client-side rate-limit
            # handling still works.
            try:
                import json as _jresp_probe
                err_body = (
                    _jresp_probe.loads(_probe.body)
                    if isinstance(_probe.body, (bytes, bytearray))
                    else _probe.body
                )
            except Exception:
                err_body = {"error": {"message": "upstream_error"}}
            err = err_body.get("error", err_body)
            # Filter out hop-by-hop / framing headers (content-length,
            # transfer-encoding, connection) before forwarding — Starlette
            # computes the new Content-Length from `content=...` and would
            # otherwise mismatch the body (RuntimeError: response content
            # longer than Content-Length, torn connection). Preserve the
            # semantic headers that callers (codex CLI, hermes) rely on:
            # Retry-After, x-should-retry, x-anchor-worker.
            _fwd_headers = {
                k: v for k, v in (_probe.headers or {}).items()
                if k.lower() not in {"content-length", "transfer-encoding", "connection", "content-type"}
            }
            return JSONResponse(
                status_code=_probe.status_code,
                content={
                    "id": f"resp-anchor-{int(time.time()*1000)}",
                    "object": "response",
                    "status": "failed",
                    "error": err,
                },
                headers=_fwd_headers,
            )
        return StreamingResponse(
            _v1_responses_stream_gen(req, request, chat=_probe),
            media_type="text/event-stream",
        )

    from anchor.config import clamp_max_tokens as _cmt_r
    max_tokens = _cmt_r(req.max_output_tokens or req.max_tokens or 1024, "premium")
    chat_tools = _responses_tools_to_chat(getattr(req, "tools", None))
    chat_req = OAChatRequest(
        model=req.model,
        messages=_responses_input_to_messages(req.input, req.instructions),
        temperature=req.temperature,
        max_tokens=max_tokens,
        stream=False,
        decompose="auto" if not chat_tools else "off",
        tools=chat_tools,
        tool_choice=req.tool_choice if chat_tools else None,
        parallel_tool_calls=req.parallel_tool_calls,
        internal_source="responses",
    )
    chat = await v1_chat_completions(chat_req, request)
    # v0.9.53 (audit R10): v1_chat_completions may return a JSONResponse (424
    # Failed Dependency) when the fallback chain exhausts. Detect and
    # re-return as a clean OpenAI Responses-shape error envelope so the
    # Codex /v1/responses path also surfaces the failure.
    if isinstance(chat, dict) is False and hasattr(chat, "body"):
        try:
            import json as _jresp
            err = _jresp.loads(chat.body) if isinstance(chat.body, (bytes, bytearray)) else chat.body
        except Exception:
            err = {"error":{"message":"upstream_error"}}
        from starlette.responses import JSONResponse as _JR
        return _JR(
            status_code=chat.status_code,
            content={
                "id": f"resp-anchor-{int(time.time()*1000)}",
                "object":"response",
                "status":"failed",
                "error": err.get("error", err),
            },
            headers=dict(chat.headers or {}),
        )
    message = (chat.get("choices") or [{}])[0].get("message") or {}
    answer = message.get("content")
    tool_calls = message.get("tool_calls") or []
    created = int(time.time())
    raw_usage = chat.get("usage", {}) or {}
    prompt_tokens = int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
    completion_tokens = int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0)
    total_tokens = int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    usage = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    anchor_meta = chat.get("_anchor", {}) or {}
    output = []
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            args = fn.get("arguments") or ""
            if not isinstance(args, str):
                import json as _json_ns
                args = _json_ns.dumps(args)
            output.append({
                "id": f"fc-anchor-{created}-{i}",
                "type": "function_call",
                "status": "completed",
                "call_id": tc.get("id") or f"call_anchor_{created}_{i}",
                "name": fn.get("name") or "",
                "arguments": args,
            })
    else:
        output.append({
            "id": f"msg-anchor-{created}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": answer or "", "annotations": [], "logprobs": []}],
            "status": "completed",
        })
    return {
        "id": f"resp-anchor-{int(time.time()*1000)}",
        "object": "response",
        "created_at": created,
        "completed_at": created,
        "status": "completed",
        "model": req.model,
        "output": output,
        "output_text": answer or "",
        "usage": usage,
        "error": None,
        "metadata": {"anchor": anchor_meta} if anchor_meta else {},
        "_anchor": anchor_meta,  # backward-compat for existing clients/tests
    }



@app.post("/v1/images/generations")
async def v1_image_generations(req: OAImageGenRequest, request: Request):
    """OpenAI-compatible image generation. v0.9.9 (Opus 4.8 directive).

    Direct proxy to agnes-image-2.1-flash (¥0, free tier).
    No caching, no transformation — pass-through as agreed.
    If agnes fails/429s, return upstream error so client knows.
    """
    # v0.9.53 (audit #S1): enforce per-key token quota on image-gen using the
    # auth middleware-resolved record. Re-parsing the Authorization header
    # here caused loopback callers (loopback_trust synthesises a "loopback-local"
    # record) to fall back to _records[0] and burn the FIRST configured client
    # key's quota without attribution.
    try:
        from anchor.key_quota import check_and_consume as _kqc
        _record = getattr(request.state, "anchor_api_key_record", None)
        if _record is not None:
            _err = _kqc(
                _record["key"], max(1, len(req.prompt) // 4),
                rpm=int(_record.get("rpm") if _record.get("rpm") is not None else 60),
                daily_tokens=int(_record.get("daily_tokens") if _record.get("daily_tokens") is not None else 0),
            )
            if _err is not None:
                from fastapi.responses import JSONResponse as _JQR
                return _JQR(
                    status_code=429,
                    headers={"Retry-After": str(_err.get("retry_after", 1))},
                    content={"error": {"message": _err.get("message", "rate limit"), "type": "rate_limit_error"}},
                )
    except Exception as _imgkqe:
        import logging as _lg_img
        _lg_img.debug("image quota check skipped: %s", _imgkqe)

    # v0.9.9: validate prompt (avoid forwarding empty/whitespace to upstream)
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    # v0.9.28: normalize model name → actual upstream worker name.
    # User-facing names: "anchor-image", "anchor/image", or raw "agnes-image-2.1-flash".
    # Upstream agnes expects literal "agnes-image-2.1-flash".
    _MODEL_TO_WORKER = {
        "anchor-image": "agnes-image-2.1-flash",
        "anchor/image": "agnes-image-2.1-flash",
        "agnes-image-2.1-flash": "agnes-image-2.1-flash",
    }
    worker_name = _MODEL_TO_WORKER.get(req.model, "agnes-image-2.1-flash")

    import os as _os
    api_key = _os.environ.get("AGNES_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="AGNES_API_KEY not configured")

    # Map OpenAI size to agnes-supported sizes (validates agnes constraint)
    size = req.size if req.size in ("512x512", "1024x1024", "1024x1536", "1536x1024") else "1024x1024"
    import httpx as _httpx

    # v0.9.50-p1 (Gemini image-gen removed): agnes-image-2.1-flash is the
    # sole image-gen worker. If agnes fails, return 502.
    primary_error = None
    primary_worker = None
    upstream_url = None
    upstream_data = None

    # Primary: agnes-image-2.1-flash
    try:
        t0 = time.time()
        agnes_body = {"model": worker_name, "prompt": req.prompt, "n": req.n, "size": size}
        async with _httpx.AsyncClient(timeout=120.0) as _client:
            resp = await _client.post(
                "https://apihub.agnes-ai.com/v1/images/generations",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=agnes_body,
            )
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            upstream_data = resp.json()
            primary_worker = "agnes-image-2.1-flash"
            upstream_url = "https://apihub.agnes-ai.com/v1/images/generations"
        else:
            primary_error = f"agnes HTTP {resp.status_code}: {resp.text[:200]}"
    except _httpx.TimeoutException:
        primary_error = "agnes image gen timed out (>120s)"
    except Exception as e:
        _logger.exception("IMAGE_GEN_AGNES_ERR: %s", e)
        primary_error = f"agnes upstream error: {e}"

    if upstream_data is None:
        raise HTTPException(status_code=502, detail=primary_error or "all image workers failed")

    # v0.9.28: translate agnes response → OpenAI format. agnes returns url.
    openai_data = []
    for item in (upstream_data or {}).get("data", []):
        out = {}
        if item.get("url"):
            out["url"] = item["url"]
        if item.get("b64_json"):
            out["b64_json"] = item["b64_json"]
        if item.get("revised_prompt"):
            out["revised_prompt"] = item["revised_prompt"]
        openai_data.append(out)

    # Log session (best-effort, never break request)
    try:
        from anchor.session_log import append_session, hash_query as _hq
        append_session(
            query=f"[image-gen] {req.prompt[:200]}",
            routed_tier="image-gen",
            model_used=(primary_worker or req.model or "claude-fable-5"),  # v0.9.7X-P2: m3→opus-5 fallback removed (B1 cleared m3; opus-5 auth-gated)
            latency_ms=latency_ms,
            cost_yuan=0.0,
            judge_score=1.0,
            session_id=f"img-{_hq(req.prompt)}",
            source="user",
            extra={"n": req.n, "size": size, "response_format": req.response_format or "url"},
        )
    except Exception as _img_log_e:
        # S11 (SRE audit): image gen session log failure; log so
        # operators can see if image-call accounting is silently lost.
        _logger.warning("IMG_SESSION_LOG_SKIP err=%s", _img_log_e)

    return {
        "created": int(time.time()),
        "data": openai_data,
        "model": req.model, "resolved_worker": worker_name,
        "_anchor": {
            "tier": "image-gen",
            "worker": req.model,
            "cost_yuan": 0.0,
            "latency_ms": latency_ms,
            "upstream": upstream_url or "unknown",
            "primary_worker": primary_worker,
        },
    }




# ============== v0.9.10: Cost Dashboard (Sonnet 5 #1 priority) ==============










# ============== v0.9.18: M3 Flat Cost Amortization (Audit Item 5) ==============

# ============== v0.9.18: SLA / Error Budget (Audit Item 4) ==============


























from typing import Optional
