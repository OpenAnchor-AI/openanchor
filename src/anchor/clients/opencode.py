"""OpenCode CLI client — for models that only run via `opencode run --model ...`.

v0.9.46h: forwards OpenAI-format tools/tool_choice via prompt injection
(opencode CLI doesn't accept --tools flags directly). Surfaces tool_calls
when the worker emits them; otherwise returns the textual reply.

Usage:
    client = OpenCodeCLIClient(model="opencode/deepseek-v4-flash-free",
                                name="deepseek-v4-flash")
    r = await client.chat([{"role":"user","content":"weather Tokyo?"}],
                          tools=[{"type":"function","function":{...}}],
                          tool_choice="auto")
"""
from __future__ import annotations
import asyncio
import json
import re
import uuid
from typing import AsyncIterator


class OpenCodeCLIClient:
    """Spawns `opencode run --model <m> --share` and parses the output."""

    def __init__(self, model: str, name: str = ""):
        self.model = model
        self.name = name or model

    async def chat(self, messages, **kw) -> dict:
        # take last user message as the prompt
        user_msg = next((m["content"] for m in reversed(messages)
                         if m.get("role") == "user"), "")
        # v0.9.46h: prepend tool schema when caller asks for tool_calls.
        # opencode CLI has no --tools flag, so we inject a short instruction
        # + JSON schema; downstream parsers detect "tool_calls" JSON blocks.
        tools = kw.get("tools")
        tool_choice = kw.get("tool_choice")
        prompt = user_msg
        if tools:
            schema_lines = []
            for t in tools:
                if not isinstance(t, dict):
                    continue
                fn = t.get("function") or {}
                schema_lines.append(
                    f"- {fn.get('name','fn')}({json.dumps(fn.get('parameters', {}))}): "
                    f"{fn.get('description','')}"
                )
            schema_block = "\n".join(schema_lines) if schema_lines else "(no schema)"
            choice_note = ""
            if tool_choice in ("required", {"type": "function"}):
                choice_note = "\nYou MUST call one of the following tools."
            elif tool_choice == "auto":
                choice_note = "\nIf a tool fits, call it; otherwise answer directly."
            prompt = (
                f"[TOOLS_AVAILABLE]\n{schema_block}{choice_note}\n"
                f"[END_TOOLS]\n\n"
                f"{user_msg}\n\n"
                f"If calling a tool, output exactly one JSON line:\n"
                f' <tool_call>{{"name":"<fn>","arguments":{{...}}}}</tool_call>'
            )

        cmd = ["opencode", "run", "--model", self.model, "--share", "--format", "default"]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode()),
                timeout=kw.get("timeout", 90),
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            proc.kill()
            raise RuntimeError("opencode timeout")
        text = stdout.decode(errors="ignore")
        content = _strip_opencode_banner(text)

        # v0.9.46h: extract tool_call JSON blocks when present.
        tool_calls, leftover = _extract_tool_calls(content)

        return {
            "id": "opencode-" + str(hash(prompt))[:8],
            "model": self.model,
            "content": leftover if tool_calls else content,
            "usage": {},  # opencode doesn't expose usage
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "_worker": self.name,
            "tool_calls": tool_calls,
        }

    async def stream(self, messages, **kw) -> AsyncIterator[str]:
        result = await self.chat(messages, **kw)
        if result.get("content"):
            yield result["content"]

    async def close(self):
        pass


def _strip_opencode_banner(text: str) -> str:
    """OpenCode prints 'Sisyphus - ultraworker · <model>\\n> ' banner before the response."""
    # find the first '> ' prompt marker after the banner
    # Find the assistant's first prompt marker line ('> ...'). The content
    # AFTER '> ' on that same line + subsequent lines is the model output.
    # Multi-line output (e.g. <tool_call>{...}</tool_call> blocks) is preserved.
    # Fallback: full text stripped (no banner found).
    m = re.search(r"(?m)^> (.*?)(?:\n(.*))?$", text, re.S)
    if m:
        first = (m.group(1) or "").strip()
        rest = (m.group(2) or "").strip()
        return (first + "\n" + rest).strip() if rest else first
    return text.strip()


def _extract_tool_calls(text: str) -> tuple[list, str]:
    """Parse <tool_call>...</tool_call> blocks → OpenAI-shape tool_calls list.

    Returns ([{id, type, function:{name, arguments}}, ...], remaining_text).
    """
    # Two accepted output formats:
    #   (a)  上一篇：```json\n{"name":"...","arguments":{...}}\n```   (opencode-cli prompt injection)
    #   (b) <tool_call>{"name":"...","arguments":{...}}</tool_call>
    # Both are converted to OpenAI-shape tool_calls.
    pattern = re.compile(
        r"(?:  上一篇：```json\s*(\{.*?\})\s*```|<tool_call>\s*(\{.*?\})\s*</tool_call>)",
        re.DOTALL,
    )
    calls = []
    for m in pattern.finditer(text):
        raw = (m.group(1) or m.group(2) or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        name = obj.get("name") or obj.get("function") or ""
        args = obj.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    leftover = pattern.sub("", text).strip()
    return calls, leftover
