"""Decompose / cheap-try / judge helpers (extracted from routing_core).

Late-imports `_call_worker` from routing_core to avoid circular import at load.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.api_models import OAChatRequest


def _call_worker_ref():
    from anchor.routing_core import _call_worker
    return _call_worker


# --- _cheap_try ---
async def _cheap_try(req: OAChatRequest, q, tier, mt) -> tuple[str, dict]:
    """Cheap single-worker attempt for decompose auto mode.

    Forces cheapest viable worker for the tier to minimize cost on trial.
    Returns (content, anchor_meta).
    """
    import time as _tt
    t0 = _tt.time()
    # Pick cheapest worker per tier
    # Single product: always trial M3 (flat subscription, stable).
    forced = "minimax-m3"
    try:
        answer, cost, worker_ms, _vp_flag, _, _usage_cheap, _, *__ = await _call_worker_ref()(forced, q.query, max_tokens=mt, messages=q.messages)
    except Exception as e:
        return f"[cheap_try_err:{type(e).__name__}] {e}", {"worker": forced, "cost_yuan": 0, "latency_ms": int((_tt.time()-t0)*1000), "tier": tier}
    return answer, {"worker": forced, "cost_yuan": cost, "latency_ms": int((_tt.time()-t0)*1000), "tier": tier}



# --- _judge_quality ---
async def _judge_quality(content: str, query: str) -> dict:
    """Judge essay quality via dpsk (cheap). Returns dict with score fields.

    Output schema: {"structure": 0-3, "tone": 0-3, "coverage": 0-3, "total": 0-9, "should_decompose": bool}
    """
    import re as _re
    prompt = f"""Score this response on three criteria (0-3 each):
- structure (clear sections, logical flow)
- tone (matches request tone)
- coverage (covers all requested points)

Query: {query[:300]}

Response: {content[:2500]}

Output ONLY valid JSON: {{"structure": N, "tone": N, "coverage": N, "total": N, "should_decompose": true/false}}

Use should_decompose=true ONLY if total < 6 OR major missing sections.
"""
    try:
        r_text, _, _, _, _, _, _, *__ = await _call_worker_ref()("deepseek-v4-flash", prompt, max_tokens=200)
        # Strip <think> tags dpsk may add
        r_text = _re.sub(r"<think>.*?</think>\s*", "", r_text, flags=_re.DOTALL).strip()
        m = _re.search(r"\{[^\}]*\}", r_text, _re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as _de_e:
        # S11 (SRE audit): decompose-heuristic JSON parse failed. The
        # default ("keep cheap") is intentional, but log so operators
        # can detect upstream judge regressions.
        import logging as _lg_de
        _lg_de.warning("DECOMPOSE_PARSE_SKIP err=%s", _de_e)
    return {"total": 9, "should_decompose": False}  # default = keep cheap



# --- _do_decompose ---
async def _do_decompose(req: OAChatRequest, q, tier, mt) -> tuple[str, dict]:
    """Sonet-5 decompose → N m3/agnes writes → merge.

    Returns (combined_content, meta).
    """
    import time as _tt
    t0 = _tt.time()

    decompose_prompt = f"""Decompose into 3 atomic subtasks (intro, body, conclusion).
Output ONLY valid JSON array, no markdown:
[{{"id":"intro","task":"Write a 150-word intro to: {q.query[:300]}","type":"intro"}},
 {{"id":"body","task":"Write a 350-word body covering main points: {q.query[:300]}","type":"body"}},
 {{"id":"conclusion","task":"Write a 100-word conclusion: {q.query[:300]}","type":"conclusion"}}]
"""
    # Use sonnet-5 for decomposition (good at planning)
    try:
        # Local call via baosiapi for sonnet
        import os as _os
        import httpx as _httpx
        baosi_key = (_os.environ.get("BAOSIAPI_GPT_API_KEY") or _os.environ.get("BAOSI_GPT_API_KEY") or _os.environ.get("BAOSIAPI_API_KEY") or _os.environ.get("BAOSI_API_KEY") or "")
        if not baosi_key:
            return None, {"decompose": "no_baosi_key"}
        async with _httpx.AsyncClient(timeout=30.0) as _client:
            r = await _client.post(
                "https://baosiapi.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {baosi_key}"},
                json={"model":"claude-opus-5","messages":[{"role":"user","content":decompose_prompt}],"max_tokens":500,"temperature":0.0},
            )
        d = r.json()
        if "choices" not in d:
            return None, {"decompose": "sonnet_failed", "err": str(d)[:200]}
        raw = d["choices"][0]["message"]["content"]
        # Strip markdown fence
        import re as _re
        m = _re.search(r"\[.*\]", raw, _re.DOTALL)
        if m:
            subtasks = json.loads(m.group(0))
        else:
            subtasks = json.loads(raw)
    except Exception as e:
        return None, {"decompose": "sonnet_exception", "err": str(e)[:200]}

    # Write each subtask via cheap worker (m3 if available, else agnes)
    parts = {}
    total_cost = 0.0
    for s in subtasks:
        sid = s.get("id", "?")
        task = s.get("task", "")
        # Public API quality-first: use m3 for generated subtasks on every tier.
        forced = "minimax-m3"
        try:
            ans, cost, _, _, _, _, _, *__ = await _call_worker_ref()(forced, task, max_tokens=mt, messages=q.messages)
            if ans.startswith("[error:"):
                # Stable fallback; do not route public API subtasks through Agnes.
                ans, cost, _, _, _, _, _, *__ = await _call_worker_ref()("claude-haiku-4-5", task, max_tokens=mt, messages=q.messages)
            parts[sid] = ans
            total_cost += cost
        except Exception as e:
            parts[sid] = f"[subtask_err:{type(e).__name__}] {e}"

    # Merge: preserve task order
    ordered_ids = [s.get("id", "?") for s in subtasks]
    combined = "\n\n".join(parts.get(sid, "") for sid in ordered_ids)
    total_latency = int((_tt.time() - t0) * 1000)
    return combined, {
        "decompose": "on",
        "n_subtasks": len(subtasks),
        "total_cost": round(total_cost, 4),
        "latency_ms": total_latency,
        "subtask_workers": {sid: "minimax-m3" for sid in ordered_ids},
    }



# --- _is_splittable ---
def _is_splittable(query: str) -> bool:
    """P2-A (2026-07-12): Detect tasks with genuinely independent subtasks.

    Decompose is only worth triggering when subtasks have NO cross-references:
      - Multi-document summarize: parallel haiku runs → cheap m3 merge
      - Multi-image batch: vision worker per image → merge
      - Parallel translation: same text → N languages independently
      - Numbered list tasks: user explicitly enumerated N items

    EXPLICITLY NOT splittable (returns False):
      - Anything with design+code in the same query (context coupling)
      - Single-document analysis / architecture / reasoning tasks
      - Any query with cross-referencing language ("based on X, do Y")
    """
    import re as _re
    q = query.lower()
    ql = len(query)

    # Negative gates — must check first
    _design_code_coupled = (
        any(dk in q for dk in ("设计", "架构", "design", "architect", "implement", "实现"))
        and any(ck in q for ck in ("def ", "class ", "import ", "代码", "python", "function", "code"))
    )
    if _design_code_coupled:
        return False  # integrated design+code task — splitting breaks coherence

    _has_cross_ref = bool(_re.search(
        r"\b(based on|refer to|given the above|using the result|after that|then use|以上|基于|结合上述)",
        q
    ))
    if _has_cross_ref:
        return False  # explicit cross-subtask references → context dependency

    # Positive gates — any of these suggests genuine parallelism
    # 1. Multi-document: numbered list of items to summarize / analyze
    multi_doc = bool(_re.search(
        r"(总结|summarize|分析|analyze|review).{0,30}(以下|following|these|下[面列])[\s\S]{0,200}"  
        r"(\d+[.)、]|第[一二三四五六七八九十])",
        q
    ))
    if multi_doc:
        return True

    # 2. Explicit N items: user enumerates 3+ parallel items with markers
    numbered_items = len(_re.findall(r"(?:^|\n)\s*(?:\d+[.)、]|[-*])", query))
    if numbered_items >= 3 and ql > 100:
        return True

    # 3. Multi-image: multiple image:/ 图片: markers
    image_count = q.count("image:") + q.count("图片:") + q.count("照片:")
    if image_count >= 2:
        return True

    # 4. Parallel translation: "translate ... into N languages"
    if _re.search(r"translat.{0,20}(into|to).{0,30}(and|,).{0,30}(language|语言)", q):
        return True
    if _re.search(r"(翻译成|译成).{0,20}(和|以及|与).{0,20}(语|文)", q):
        return True

    return False



# --- _maybe_decompose ---
async def _maybe_decompose(req, q, tier, mt, decompose: str):
    """X.3 self-improving router. Returns merged content dict or None (use single-worker).

    Logic:
    - decompose=off → None (legacy path)
    - decompose=on  → force decompose, skip cheap try
    - decompose=auto → cheap try → judge → decompose if should_decompose
    """
    if decompose == "off":
        return None

    if decompose == "on":
        merged, meta = await _do_decompose(req, q, tier, mt)
        if merged is None:
            return None  # decompose failed, fall back to single-worker
        return {"content": merged, "meta": meta, "path": "decompose_forced"}

    # auto
    # Gate 1: skip short/simple queries fast
    if len(q.query) < 80:
        return None
    # Gate 2: skip vision (already routes to agnes/gemini)
    if any(k in q.query.lower() for k in ("image:", "图片:", "describe this image")):
        return None

    # v0.9.14 (Sonnet 5 A1): design/hard override — return marker for caller
    # to force opus-4-8 single-worker. Decompose's cheap mosaic can't match opus.
    ql_dh = q.query.lower()
    is_design_dh = any(k in ql_dh for k in ("设计", "架构", "分布式", "算法", "design", "architect", "distributed", "consensus", "raft", "consistent hash", "load balance"))
    has_hard_dh = (
        len(q.query) >= 100 or
        sum(1 for k in ("设计", "架构", "分布式", "算法", "design", "architect", "distributed", "consensus", "raft", "consistent hash", "load balance", "scalable", "high availability", "高可用", "微服务", "microservice") if k in ql_dh) >= 2
    )
    if is_design_dh and has_hard_dh:
        return {"content": "__FORCE_OPUS__", "meta": {"decompose": "skipped_design_hard"}, "path": "design_hard_override"}

    # Gate 3 (P2-A 2026-07-12): only decompose when subtasks are genuinely independent.
    # Integrated design+code, cross-referencing, or single-document analysis queries
    # must go single-worker — decompose produces incoherent fragments for these.
    if not _is_splittable(q.query):
        return None  # fall through to single-worker routing

    # Cheap try
    cheap_content, cheap_meta = await _cheap_try(req, q, tier, mt)
    if cheap_content.startswith("[cheap_try_err"):
        return None  # cheap failed, let single-worker handle

    # Judge quality
    judge = await _judge_quality(cheap_content, q.query)
    if not judge.get("should_decompose", False):
        # Cheap is good enough → return cheap content (skip single-worker)
        return {"content": cheap_content, "meta": cheap_meta, "path": "cheap_kept", "judge": judge}

    # Decompose triggered
    merged, decomp_meta = await _do_decompose(req, q, tier, mt)
    if merged is None:
        return {"content": cheap_content, "meta": cheap_meta, "path": "decompose_failed_fallback", "judge": judge}
    return {"content": merged, "meta": {**cheap_meta, **decomp_meta}, "path": "decompose_triggered", "judge": judge}
