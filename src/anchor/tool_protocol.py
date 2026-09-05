"""Strict cross-provider normalization for OpenAI Chat tool messages.

The normalizer enforces the strictest common protocol accepted by MiniMax M3,
OpenAI/GPT and DeepSeek-compatible gateways:

* every assistant tool call has a non-empty, conversation-unique id;
* an assistant tool-call turn is immediately followed by exactly one tool
  result per call, in declaration order;
* missing outputs are synthesized before a later non-tool message or EOF;
* orphan, late and duplicate results are re-anchored as independent calls;
* missing, duplicate and reused ids are rewritten deterministically.

Both Chat Completions and Responses conversion feed this normalizer.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

_MISSING_TOOL_OUTPUT = "[tool output unavailable: conversation history was truncated]"
_UNKNOWN_TOOL_NAME = "unknown"


def _coerce_arguments(arguments: Any) -> str:
    """Return a strict, valid JSON object string for tool-call arguments.

    Strict vendors (MiniMax M3) reject any tool_call whose ``arguments`` is
    not a parseable JSON string. Codex / pi sometimes send non-strings
    (``{}``, ``[]``, ints, ``None``) or invalid JSON; we coerce to a valid
    JSON string. Empty / unparseable / missing values become ``"{}"``.
    """
    import json as _json_args
    if isinstance(arguments, str):
        s = arguments.strip()
        if not s:
            return "{}"
        try:
            _json_args.loads(s)
            return s
        except Exception:
            return _json_args.dumps({"_repaired": s}, ensure_ascii=False)
    if arguments is None:
        return "{}"
    try:
        return _json_args.dumps(arguments, ensure_ascii=False)
    except Exception:
        return "{}"


def _normalize_content(content: Any) -> Any:
    """Flatten text-only parts while preserving multimodal image lists."""
    if not isinstance(content, list):
        return content
    if any(isinstance(p, dict) and p.get("type") == "image_url" for p in content):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(parts)


def normalize_tool_protocol(messages: Iterable[Any] | None) -> list[dict]:
    """Return a strict canonical Chat message sequence.

    The repair is deterministic and idempotent. Valid non-tool fields are
    preserved; only tool protocol fields are canonicalized.
    """
    if not messages:
        return []

    out: list[dict] = []
    used_ids: set[str] = set()
    counter = 0

    pending_calls: list[dict] = []
    # Raw incoming id -> canonical ids for the active assistant turn. A queue
    # handles duplicate/reused raw ids without losing output association.
    pending_aliases: dict[str, deque[str]] = {}
    pending_results: dict[str, dict] = {}
    anonymous_results: deque[dict] = deque()

    def fresh_id(raw: Any = None) -> str:
        nonlocal counter
        candidate = str(raw or "").strip()
        if candidate and candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        while True:
            counter += 1
            candidate = f"call_fallback_{counter}"
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate

    def tool_message(tcid: str, content: Any, name: str | None = None) -> dict:
        result = {
            "role": "tool",
            "content": "" if content is None else _normalize_content(content),
            "tool_call_id": tcid,
        }
        if name:
            result["name"] = name
        return result

    def assistant_anchor(tcid: str, name: str) -> dict:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": tcid,
                "type": "function",
                "function": {"name": name or _UNKNOWN_TOOL_NAME, "arguments": _coerce_arguments("{}")},
            }],
        }

    def flush_pending() -> None:
        nonlocal pending_calls, pending_aliases, pending_results, anonymous_results
        if not pending_calls:
            return
        for call in pending_calls:
            tcid = call["id"]
            result = pending_results.get(tcid)
            if result is None and anonymous_results:
                result = anonymous_results.popleft()
            if result is None:
                result = tool_message(
                    tcid, _MISSING_TOOL_OUTPUT, call["function"]["name"]
                )
            else:
                result = tool_message(
                    tcid,
                    result.get("content", ""),
                    result.get("name") or call["function"]["name"],
                )
            out.append(result)
        pending_calls = []
        pending_aliases = {}
        pending_results = {}
        anonymous_results = deque()

    def emit_orphan(raw: dict) -> None:
        # Preserve an orphan's id when globally unique; late/duplicate ids are
        # rewritten so each result belongs to exactly one assistant call.
        name = raw.get("name") or _UNKNOWN_TOOL_NAME
        tcid = fresh_id(raw.get("tool_call_id"))
        out.append(assistant_anchor(tcid, name))
        out.append(tool_message(tcid, raw.get("content", ""), name))

    for raw in messages:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role") or "user"
        if role == "developer":
            role = "system"
        elif role == "function":
            role = "tool"

        if role == "tool":
            result = dict(raw)
            result["content"] = _normalize_content(result.get("content"))
            raw_id = str(result.get("tool_call_id") or "").strip()
            if pending_calls:
                target: str | None = None
                if raw_id:
                    queue = pending_aliases.get(raw_id)
                    while queue and queue[0] in pending_results:
                        queue.popleft()
                    if queue:
                        target = queue.popleft()
                elif len(pending_results) + len(anonymous_results) < len(pending_calls):
                    anonymous_results.append(result)
                    if len(pending_results) + len(anonymous_results) == len(pending_calls):
                        flush_pending()
                    continue
                if target and target not in pending_results:
                    pending_results[target] = result
                    if len(pending_results) + len(anonymous_results) == len(pending_calls):
                        flush_pending()
                    continue
                # A mismatched result cannot interrupt an active assistant
                # turn. Complete that turn, then re-anchor the real result.
                flush_pending()
            emit_orphan(result)
            continue

        # No non-tool message may interrupt an active tool-call turn.
        flush_pending()

        msg = dict(raw)
        msg["role"] = role
        content = _normalize_content(msg.get("content"))
        raw_calls = msg.get("tool_calls")
        calls: list[dict] = []
        aliases: dict[str, deque[str]] = defaultdict(deque)
        if role == "assistant" and isinstance(raw_calls, list):
            for raw_call in raw_calls:
                if not isinstance(raw_call, dict):
                    continue
                fn = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
                name = fn.get("name") or raw_call.get("name") or _UNKNOWN_TOOL_NAME
                arguments = _coerce_arguments(
                    fn.get("arguments", raw_call.get("arguments"))
                )
                raw_id = str(raw_call.get("id") or "").strip()
                tcid = fresh_id(raw_id)
                calls.append({
                    "id": tcid,
                    "type": "function",
                    "function": {"name": str(name), "arguments": arguments},
                })
                if raw_id:
                    aliases[raw_id].append(tcid)
        if calls:
            msg["content"] = "" if content is None else content
            msg["tool_calls"] = calls
            out.append(msg)
            pending_calls = calls
            pending_aliases = dict(aliases)
            continue

        if role != "assistant":
            msg.pop("tool_calls", None)
        msg["content"] = "" if content is None and role == "assistant" else content
        out.append(msg)

    flush_pending()
    return out


__all__ = ["normalize_tool_protocol"]
