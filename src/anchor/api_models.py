"""Pydantic HTTP models (extracted from server.py for split + circular-import hygiene)."""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator

from anchor import __version__ as _ANCHOR_VERSION


class Query(BaseModel):
    query: str
    query_type: str = "chat"
    prompt_len: int = 500
    budget: float = 0.5
    messages: list = []
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    # v1.0.1 + B5 audit 2026-08-15: Sacred dispatch flag.
    # When True, server.py / _route_sacred() routes the query to Fable
    # bypassing the normal ladder. Default False (no sacred escalation).
    # Triggers Layer 1 of the 3-layer sacred dispatch.
    sacred: bool = False


class Response(BaseModel):
    answer: str
    worker: str
    cost_yuan: float
    latency_ms: int
    confidence: float
    tier: str
    degraded: bool = False
    tool_calls: Optional[list] = None
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None
    system_fingerprint: Optional[str] = f"anchor-v{_ANCHOR_VERSION}"
    # Observability: who was first pick vs who actually answered.
    primary_worker: Optional[str] = None
    fallback_from: Optional[str] = None
    # v0.9.55: quarantine telemetry — propagate the *terminal* fallback
    # reason (timeout/quota_exhausted/rate_limit/bad_request/etc.) so the
    # session log can distinguish transport failures from real quality
    # errors. May be None on fully successful runs.
    fallback_reason: Optional[str] = None
    # v0.9.76: ordered fallback chain for stream retry. Populated by
    # _route_tier in select_only=True mode (server.py _gen iterates it on
    # stream failures so a dead primary like dpsk-flash-zen-429 doesn't
    # surface an SSE error event — it cascades to minimax-m3 / luna
    # before the client sees a torn stream). None in non-streaming paths.
    fallback_chain: Optional[list[str]] = None


class Feedback(BaseModel):
    query: str
    tier: str
    worker: str
    # audit 2026-08-16 (A1): unbounded success let any client key drive
    # head.update() with err ~999, corrupting the shared routing head W.
    success: float = Field(ge=0.0, le=1.0, description="quality in [0,1]; clamped to protect the routing head")
    query_type: str = "en"
    prompt_len: int = 500
    budget: float = 0.5
    cost_yuan: float = 0.0
    latency_ms: int = 0


def validate_tool_choice(v):
    """OpenAI tool_choice boundary validation."""
    if v is None:
        return v
    if isinstance(v, str):
        if v not in ("none", "auto", "required"):
            raise ValueError(
                f"tool_choice str must be one of 'none'|'auto'|'required', got {v!r}"
            )
        return v
    if isinstance(v, dict):
        if v.get("type") == "function":
            fn = v.get("function")
            if not isinstance(fn, dict) or "name" not in fn:
                raise ValueError("tool_choice.function dict must have 'name' field")
            if not isinstance(fn["name"], str) or not fn["name"]:
                raise ValueError("tool_choice.function.name must be non-empty str")
            return v
        return v
    raise ValueError(f"tool_choice must be str or dict, got {type(v).__name__}")


class OAChatRequest(BaseModel):
    model: str = "anchor"
    messages: list = []
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False
    worker_override: Optional[str] = None
    decompose: str = "auto"
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    stop: Optional[Union[str, list]] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[dict] = None
    stream_options: Optional[dict] = None
    # Internal-only routing hint set by /v1/responses. It is not forwarded to
    # workers; it lets routing distinguish a Codex Responses replay from a
    # native Chat Completions request without relying on User-Agent strings.
    internal_source: Optional[str] = None

    _tool_choice_validator = field_validator("tool_choice")(validate_tool_choice)


class OAResponsesRequest(BaseModel):
    model: str = "anchor"
    input: str | list | dict = ""
    instructions: Optional[str] = None
    temperature: float = 0.7
    max_output_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    reasoning: Optional[dict] = None
    previous_response_id: Optional[str] = None
    tools: Optional[list] = None
    tool_choice: Optional[object] = None
    metadata: Optional[dict] = None
    truncation: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None


class OAImageGenRequest(BaseModel):
    model: str = "agnes-image-2.1-flash"
    prompt: str
    # Cap upstream image count to prevent multi-image amplification
    # abuse (Grok 4.5 finding MED #2). Agnes free tier supports at
    # most 4 images per request and rejects higher n with 422.
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    quality: Optional[str] = None
    response_format: Optional[str] = "url"
    user: Optional[str] = None
