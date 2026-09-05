"""Lightweight self-check against the aimax farm ollama endpoint.

Implements the §2.2 deliverable from kanban task t_7b06ca78
(Advisor Directive · 2026-08-29). Verifies the macmini → 10.0.0.4 path through
the WG utun11 inner net:

    1. /api/version   — TCP+HTTP handshake, returns the ollama version string
    2. /v1/chat/completions — small ornith-35b message to confirm chat lane
    3. /api/embeddings      — qwen3-embed-8b to confirm embedding lane

Each step is independently timeout-bounded so a slow/failing embed does not
mask chat success. ``run_all`` returns a structured dict with per-step
``{ok, elapsed_s, status, detail}`` blocks so dashboards / cron jobs can
consume it without parsing prose.

Usage from a Python REPL or a lightweight check script:

    from anchor.aimax_self_check import run_all
    report = run_all()
    print(report["summary"]["all_ok"])

Or from the shell (after installing anchor in editable mode):

    python -m anchor.aimax_self_check

The module is intentionally dependency-light: only ``httpx``. It performs
no LLM inference, no caching, and no writes — read-only health probe.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import httpx


# ── Defaults (mirror anchor.config / anchor.semantic_cache) ─────────
DEFAULT_AIMAX_BASE_URL = os.environ.get(
    "ANCHOR_OLLAMA_BASE_URL", "http://10.0.0.4:11434"
)
DEFAULT_CHAT_MODEL = "ornith-35b-q8-agent:latest"
DEFAULT_EMBED_MODEL = "qwen3-embed-8b:latest"

# Per-step timeouts. Embedding is allowed more slack because the first call
# has to load ~4.7 GB of weights into VRAM; on a slow WG link it can take
# ~60-90 s the first time.
VERSION_TIMEOUT = 4.0
CHAT_TIMEOUT = 15.0
EMBED_TIMEOUT = 90.0


def _strip_v1(url: str) -> str:
    """Convert ``http://host:port/v1`` → ``http://host:port``."""
    return url.rstrip("/").removesuffix("/v1")


def check_version(base_url: str | None = None, timeout: float = VERSION_TIMEOUT) -> dict[str, Any]:
    """Probe ``GET /api/version`` on the aimax ollama daemon."""
    base = _strip_v1(base_url or DEFAULT_AIMAX_BASE_URL)
    url = f"{base}/api/version"
    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(url)
            elapsed = time.monotonic() - started
            if r.status_code != 200:
                return {"ok": False, "elapsed_s": round(elapsed, 3), "status": r.status_code,
                        "detail": r.text[:160]}
            data = r.json()
            return {"ok": True, "elapsed_s": round(elapsed, 3), "status": 200,
                    "detail": f"ollama {data.get('version', '?')}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "elapsed_s": round(time.monotonic() - started, 3),
                "status": 0, "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}


def check_chat(base_url: str | None = None, timeout: float = CHAT_TIMEOUT,
               model: str | None = None) -> dict[str, Any]:
    """Probe ``POST /v1/chat/completions`` with a tiny ornith-35b ping."""
    base = (base_url or DEFAULT_AIMAX_BASE_URL).rstrip("/")
    url = f"{base}/v1/chat/completions"
    payload = {
        "model": model or DEFAULT_CHAT_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(url, json=payload)
            elapsed = time.monotonic() - started
            if r.status_code != 200:
                return {"ok": False, "elapsed_s": round(elapsed, 3), "status": r.status_code,
                        "detail": r.text[:160]}
            data = r.json()
            choices = data.get("choices") or []
            content = (choices[0].get("message") or {}).get("content", "") if choices else ""
            return {"ok": True, "elapsed_s": round(elapsed, 3), "status": 200,
                    "detail": f"model={data.get('model','?')} bytes={len(content)}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "elapsed_s": round(time.monotonic() - started, 3),
                "status": 0, "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}


def check_embed(base_url: str | None = None, timeout: float = EMBED_TIMEOUT,
                model: str | None = None,
                prompt: str = "anchor aimax self-check") -> dict[str, Any]:
    """Probe ``POST /api/embeddings`` with a short qwen3-embed-8b ping.

    The first call against a cold ollama takes much longer (model load over
    the WG link). We surface status without raising so chat/version probes
    earlier in the report aren't masked.
    """
    base = _strip_v1(base_url or DEFAULT_AIMAX_BASE_URL)
    url = f"{base}/api/embeddings"
    payload = {"model": model or DEFAULT_EMBED_MODEL, "prompt": prompt}
    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(url, json=payload)
            elapsed = time.monotonic() - started
            if r.status_code != 200:
                return {"ok": False, "elapsed_s": round(elapsed, 3), "status": r.status_code,
                        "detail": r.text[:160]}
            data = r.json()
            vec = data.get("embedding") or []
            dim = len(vec)
            norm = round(sum(x * x for x in vec) ** 0.5, 4) if dim else 0.0
            return {"ok": dim > 0, "elapsed_s": round(elapsed, 3), "status": 200,
                    "detail": f"dim={dim} L2-norm={norm} model={data.get('model','?')}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "elapsed_s": round(time.monotonic() - started, 3),
                "status": 0, "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}


def run_all(base_url: str | None = None,
            *,
            embed_timeout: float = EMBED_TIMEOUT,
            chat_timeout: float = CHAT_TIMEOUT) -> dict[str, Any]:
    """Run version + chat + embed probes and return a structured report.

    The summary block records ``all_ok`` (every step passed), ``embed_ok``
    (version+chat+embed all green), and ``chat_ok`` (version+chat green but
    embed may be slow/down — useful for diagnostics where embed cold-load
    is acceptable).
    """
    base = base_url or DEFAULT_AIMAX_BASE_URL
    version = check_version(base, timeout=VERSION_TIMEOUT)
    chat = check_chat(base, timeout=chat_timeout)
    embed = check_embed(base, timeout=embed_timeout)
    all_ok = version["ok"] and chat["ok"] and embed["ok"]
    chat_ok = version["ok"] and chat["ok"]
    return {
        "base_url": base,
        "version": version,
        "chat": chat,
        "embed": embed,
        "summary": {
            "all_ok": all_ok,
            "chat_ok": chat_ok,
            "all_ok_str": "PASS" if all_ok else "FAIL",
        },
    }


def _print_human(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("=== Anchor aimax self-check · t_7b06ca78 ===")
    print(f"base_url: {report['base_url']}")
    for name in ("version", "chat", "embed"):
        r = report[name]
        flag = "PASS" if r["ok"] else "FAIL"
        print(f"  [{flag}] {name:8s} {r.get('elapsed_s',0):>7.2f}s "
              f"http={r.get('status',0)} {r.get('detail','')[:120]}")
    print(f"summary: all_ok={s['all_ok']} chat_ok={s['chat_ok']} verdict={s['all_ok_str']}")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    import json
    args = dict(a.split("=", 1) for a in argv if "=" in a)
    base = args.get("base")
    embed_to = float(args.get("embed_timeout", EMBED_TIMEOUT))
    chat_to = float(args.get("chat_timeout", CHAT_TIMEOUT))
    if "--json" in argv:
        print(json.dumps(run_all(base_url=base, embed_timeout=embed_to,
                                 chat_timeout=chat_to), indent=2))
    else:
        _print_human(run_all(base_url=base, embed_timeout=embed_to,
                             chat_timeout=chat_to))
    return 0


if __name__ == "__main__":
    sys.exit(main())
