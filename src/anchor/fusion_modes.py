from anchor.workers import FABLE5, OPUS, SONNET
"""Hard rules + calibration picker (single product auto lane; legacy tier aliases)."""
from typing import Optional
import re as _re
from anchor.config import ANCHOR_DISABLE_SOL_DEFAULT as _ANCHOR_DISABLE_SOL

def _sol_allowed() -> bool:
    """V4.2 audit 2026-08-15: Sol opt-out gate.

    Returns False when ANCHOR_DISABLE_SOL=1 (default) so Sol is excluded
    from all default routing fallbacks. Calibration may flip via env=0.
    """
    return not _ANCHOR_DISABLE_SOL


def _sol_in(en) -> bool:
    """Whether Sol is currently usable (opt-in gate + enabled + known)."""
    return (not _ANCHOR_DISABLE_SOL) and ("gpt-5.6-sol" in en)


# v0.9.52: single product — one rule profile; legacy names alias for callers.
_AUTO_RULE = {
    "alpha": 0.5,
    "default_workers": [
        "deepseek-v4-flash",
        "minimax-m3",
        "gpt-5.6-sol",
        "claude-fable-5",
    ],
}
TIER_RULES = {
    "auto": dict(_AUTO_RULE),
    "basic": dict(_AUTO_RULE),
    "premium": dict(_AUTO_RULE),
    "ultra": dict(_AUTO_RULE),
}



CN_PATTERN = _re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

def _auto_detect_query_type(query) -> str:
    """v0.6.5 (Fable 5 directive): auto-detect query_type from text.
    Returns: 'cn' | 'agent' | 'creative' | 'chat'

    v0.9.8: list-safe (vision/multimodal messages pass list, not str).
    """
    if isinstance(query, list):
        # Concatenate text parts for type detection only.
        parts = []
        for p in query:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        query = "\n".join(parts)
    if CN_PATTERN.search(query):
        return "cn"
    q = query.lower()
    # Agent detection (v0.9.51-p3): avoid over-trigger on casual English
    # ("and then", "first", "next") without real multi-step/tool intent.
    _agent_strong = (
        "step by step", "do these in order", "in the following order",
        "execute the following", "run the following", "multi-step",
        "first do ", "create file", "read file", "write file",
        "use the tool", "use tools", "call the tool",
        "step 1", "step 2", "step 3",
    )
    # Pure reasoning with "step by step" is NOT agent work (no tools/files).
    _pure_reason_hint = (
        "why ", "explain", "prove ", "reason", "theorem", "axiom",
        "diverges", "converges", "tradeoff", "trade-off", "compare ",
        "推导", "证明", "为什么", "解释",
    )
    _toolish_hint = (
        "file", "tool", "deploy", "commit", "install", "run the",
        "execute", "create ", "write a script", "refactor ",
    )
    if any(s in q for s in _agent_strong):
        if not (any(h in q for h in _pure_reason_hint) and not any(h in q for h in _toolish_hint)):
            return "agent"
    # Sequence markers counted without double-counting (" and then " ⊃ " then ").
    _rest = q
    _seq_n = 0
    if " and then " in _rest:
        _seq_n += 1
        _rest = _rest.replace(" and then ", " ")
    if " after that " in f" {_rest} " or "after that," in _rest:
        _seq_n += 1
    if " afterwards " in f" {_rest} ":
        _seq_n += 1
    if " next, " in f" {_rest} ":
        _seq_n += 1
    elif " then " in _rest:
        _seq_n += 1
    _act = (
        "deploy ", "build ", "commit ", "install ", "run ", "execute ",
        "create ", "write a script", "refactor ", "migrate ", "configure ",
        "implement ", "open the ", "edit the ",
    )
    _act_n = sum(1 for m in _act if m in q)
    if _seq_n >= 1 and _act_n >= 1:
        return "agent"
    if _act_n >= 2:
        return "agent"
    if _seq_n >= 2:
        return "agent"
    # Creative
    creative_kw = ("write a story", "compose", "imagine", "tell me a tale",
                   "poem", "haiku", "narrative", "fictional", "creative")
    if any(k in q for k in creative_kw):
        return "creative"
    # Defer to regex classifier for code/debug/math (pi coding prompts)
    try:
        from anchor.classifier import classify as _cls
        _qt = _cls(query)
        if _qt in ("code", "debug", "math", "reasoning", "agent", "vision"):
            return _qt
    except Exception:
        pass
    return "chat"


def _detect_category(query: str) -> str:
    """v0.6.5 (Fable 5 directive): detect query category. CJK + English supported.
    Returns: 'debug' | 'code' | 'reason' | 'design' | 'creative' | 'chat' | 'vision' | 'unknown'
    """
    q = query.lower()
    # CJK keyword expansion (v0.6.5): add Chinese tech terms
    if any(k in q for k in ("image:", "图片:", "照片:", "photo:", "screenshot:")):
        return "vision"
    if any(k in q for k in ("error:", "fix:", "bug:", "traceback", "indentationerror",
                             "typeerror", "nameerror", "syntaxerror", "nullpointerexception",
                             "memory leak", "race condition", "why does my", "how to debug",
                             "why does this", "not updating", "time out", "exits immediately",
                             "报错", "异常", "错误", "无法", "出问题", "调试")):
        return "debug"
    if any(k in q for k in ("def ", "class ", "import ", "function ", "write a", "implement ",
                             "build a", "regex ", "code to", "convert ", "parse ", "reverse ",
                             "stack ", "linked list", "binary search", "lru", "cache",
                             "merge sort", "http server", "trie", "consistent hash",
                             "worker pool", "url parser", "deep-copy", "rest api",
                             "in python", "in javascript", "in rust", "in java", "in go",
                             # Chinese: 实现/写/函数/代码/写一个/用 py/写个
                             "实现", "写", "函数", "代码", "算法", "二分", "递归", "排序",
                             "服务器", "客户端", "接口", "数据库", "索引", "缓存", "爬虫",
                             " 写 ", "用 python", "用 java", "写一个")):
        return "code"
    if any(k in q for k in ("design ", "architect", "system design", "how would you design",
                             "build a system", "design a ", "scale", "rate limit",
                             "pub/sub", "queue", "feature flag",
                             "设计", "架构", "系统", "方案", "分布式")):
        return "design"
    if any(k in q for k in ("story", "poem", "joke", "haiku", "tagline", "describe a",
                             "write a story", "write a poem", "write a product",
                             "continue this story", "robot",
                             "故事", "写诗", "笑话", "诗歌", "叙事", "小说", "续写")):
        return "creative"
    if any(k in q for k in ("explain", "why is", "why does", "what is the difference",
                             "what's the difference", "how does", "how do",
                             "halting", "theorem", "byzantine", "raft ", "consistent hashing",
                             "https", "tcp", "udp", "dns", "acid", "encryption",
                             "eventual consistency", "vector clock", "oauth",
                             "解释", "为什么", "怎么", "如何", "区别", "比较")):
        return "reason"
    return "unknown"


def _query_difficulty(query: str) -> float:
    """Heuristic difficulty 0.0-1.0. Higher = harder.
    ≥0.7 hard; 0.3–0.7 medium; <0.3 easy. Math/word-problem baseline ≥0.3.
    v0.6.2: rewrite — pick MAX of categorical buckets, not additive score, to
    avoid keyword-stacking (e.g. "implement binary search in python" used to hit 1.0).
    """
    q = query.lower()
    has_digit = any(c.isdigit() for c in query)
    is_question = q.rstrip().endswith("?") or q.rstrip().endswith("？")
    
    buckets = []  # list of category scores; take max
    
    # B1: very hard keywords (technical depth)
    very_hard = ("raft", "paxos", "consensus", "quantum", "byzantine", "zero-knowledge",
                 "tla+", "coq", "isabelle")
    if any(k in q for k in very_hard):
        buckets.append(0.95)
    
    # B2: hard keywords (design/architecture/prove)
    hard_kw = ("design ", "architect", "algorithm", "compare ", "evaluate ",
               "trade-off", "tradeoff", "analyze", "derive", "prove ",
               "complexity", "optimize ", "分布式", "架构", "证明", "推导", "设计",
               "实现", "优化")
    if any(k in q for k in hard_kw):
        buckets.append(0.7)
    
    # B3: code-task baseline (write code, debug, fix)
    code_task_kw = ("write a function", "write code", "implement a", "implement an",
                    "implement binary", "implement ", "find duplicates", "reverse ", "sort ",
                    "algorithm", "sql query", "regex", "parse", "write a sql", "find duplicate",
                    "longest", "substring", "subarray", "binary search", "linked list",
                    "in rust", "in python", "in javascript", "in typescript",
                    "in java", "in go", "in c++", "in c#", "in ruby", "in php",
                    "write a hello", "write hello", "docker", "compose", "nginx",
                    "lru", "cache", "kubernetes",
                    "fix:", "bug:", "error:", "traceback",
                    "what does ", "what do ",
                    "写一个", "实现一个", "写代码", "编写", "代码", "函数")
    if any(k in q for k in code_task_kw):
        buckets.append(0.5)  # code task = at least medium, sometimes high
    
    # B4: math word problem baseline
    math_word_kw = ("prime", "factorial", "fibonacci", "modulo", "sqrt", "calculate",
                    "compute ", "how many", "how much", "which is", "what is ",
                       "theorem", "cap theorem", "byzantine",
                    "求", "几", "多少")
    if any(k in q for k in math_word_kw) and (has_digit or is_question):
        buckets.append(0.5)
    
    # B5: reasoning puzzle / logical syllogism
    reason_puzzle_kw = ("bat and ball", "widgets", "lily pads",
                        "probability", "either", "therefore", "hence", "conclude",
                        "if all ", "if some ", "syllogism", "can we conclude")
    if any(k in q for k in reason_puzzle_kw):
        buckets.append(0.7)

    # B5b: pure math/analysis mid-upper (Grok lane)
    math_reason_kw = (
        "series", "diverges", "converges", "integral", "derivative",
        "axiom", "lemma", "corollary", "induction", "harmonic",
        "why the", "prove that", "peano",
    )
    if any(k in q for k in math_reason_kw):
        buckets.append(0.65)
    
    # B6: short reasoning (explain / why / how / difference / define)
    short_reason_kw = ("explain ", "what is the difference", "what's the difference",
                       "why is ", "why do ", "why does ", "how does ",
                       "when do ", "where do ", "who does ",
                       "meet", "leaves", "mph", "miles", "kilometers",
                       "difference between", "compare ",
                       "halting", "tcp", "udp", "protocol", "photosynthesis",
                       "define ", "solve ", "计算", "解释", "比较", "为什么", "定义",
                       # creative
                       "continue ", "continue the", "expand ", "expanding",
                       "modify ", "give a concrete", "give me a",
                       # story
                       "story", "dragon", "tale", "narrative", "讲故事")
    if any(k in q for k in short_reason_kw):
        buckets.append(0.3)  # short reasoning / creative = at least medium
    
    # B7: long questions get a slight bump
    if len(query) > 500:
        buckets.append(0.4)
    elif len(query) > 200:
        buckets.append(0.25)
    elif len(query) > 100:
        buckets.append(0.15)
    
    # B8: code markers (def/class/import) directly
    if any(k in q for k in ("def ", "class ", "function ", "import ", "=>")):
        buckets.append(0.3)
    
    if not buckets:
        return 0.0
    # T-AUDIT-04 (arch C8): multi-evidence bonus. When 3+ independent difficulty
    # signals fire, the query is genuinely hard across dimensions (e.g. long +
    # code markers + hard keywords). Add a small bonus on top of max().
    # Keep keyword-stacking guard (max), but reward cross-bucket agreement.
    base = max(buckets)
    n_signals = len(buckets)
    bonus = 0.0
    if n_signals >= 4:
        bonus = 0.15
    elif n_signals >= 3:
        bonus = 0.08
    return min(1.0, base + bonus)



def _enabled_names() -> set[str]:
    from anchor.config import WORKERS as _W
    return {w.name for w in _W if w.enabled}


def _is_sacred_query(query: str, d: float) -> bool:
    """Fable only for extreme/sacred — not default hard design or pedagogy.

    Bare "prove 2+2" / short homework stays sol/grok. Fable is for formal
    verification, Byzantine/CRDT theory, machine-checked proofs, d≥0.9.
    """
    if d >= 0.9:
        return True
    ql = query.lower()
    strong = (
        "formal verification", "formally verify", "formally prove",
        "byzantine", "byzantine fault", "soundness", "completeness proof",
        "zero-knowledge", "cryptographic proof", "type safety proof",
        "machine-checked", "coq ", "lean4", "isabelle",
        "形式化", "证明正确", "正确性证明", "安全关键", "零知识",
    )
    if any(s in query or s in ql for s in strong):
        return True
    # Systems-theory sacred only with enough difficulty (not casual mention).
    systems = ("CRDT", "crdt", "paxos", "raft vs")
    if d >= 0.7 and any(s in query or s in ql for s in systems):
        return True
    # Bare prove/theorem: only when already hard.
    if d >= 0.85 and any(s in ql for s in ("prove ", "prove the", "theorem")):
        return True
    if d >= 0.85 and any(s in query for s in ("证明",)):
        return True
    return False


def _pick_mid_lower() -> Optional[str]:
    """Dual-main lower: M3."""
    en = _enabled_names()
    if "minimax-m3" in en:
        return "minimax-m3"
    return _pick_mid_upper()


def _pick_mid_upper() -> Optional[str]:
    """Mid-upper reasoning: Grok primary, M3 economic fallback, then Sol.

    Do not send mid-upper reasoning directly from Grok to Sol when Grok is
    cooling/quarantined: M3 has near-zero marginal cost and is the intended
    dual-main backup. Sol remains the fallback only when both are unavailable.
    """
    en = _enabled_names()
    from anchor.cooldown import is_cooling
    from anchor.release.circuit_breaker import is_quarantined

    def healthy(name: str) -> bool:
        return name in en and not is_cooling(name) and not is_quarantined(name)

    if healthy("grok-4-6-reasoning"):
        return "grok-4-6-reasoning"
    if healthy("minimax-m3"):
        return "minimax-m3"
    if _sol_allowed() and healthy("gpt-5.6-sol"):
        return "gpt-5.6-sol"
    # All candidates are unhealthy: keep deterministic legacy ordering so the
    # downstream fallback engine can make the final availability decision.
    for name in ("grok-4-6-reasoning", "minimax-m3", "gpt-5.6-sol"):
        if _sol_allowed() and name in en:
            return name
    return None


def _pick_hard() -> Optional[str]:
    """Hard/high-end default = Opus-5 quality floor, then free M3 fallback."""
    en = _enabled_names()
    if OPUS in en:
        return OPUS
    if "minimax-m3" in en:
        return "minimax-m3"
    if "deepseek-v4-flash" in en:
        return "deepseek-v4-flash"
    return None


def _pick_sacred_worker() -> Optional[str]:
    en = _enabled_names()
    if FABLE5 in en and _should_try_fable5():
        return FABLE5
    # degrade: hard → mid-upper → M3
    return _pick_hard() or _pick_mid_upper() or _pick_mid_lower()


def _difficulty_aware_route(query_type: str, query: str) -> Optional[str]:
    """Single-product dual-main ladder.

    flash (free/light)
      → M3 (mid-lower: CN / mid code-agent / mid design)
      → Grok (mid-upper dual main)
      → Sol (hard / high-end)
      → Fable (sacred / extreme only)
    """
    ql = query.lower()
    # Responses/Codex tool-heavy sessions are M3-first by economics: its
    # monthly subscription is effectively amortized/free. Handle this before
    # generic short-chat routing so tiny tool-loop prompts such as "continue"
    # or "?" do not get diverted to Flash.
    if query_type == "agent-tool-heavy":
        return _pick_mid_lower() or "minimax-m3"
    # v0.6.6: tighter vision trigger. Only "image:" / "图片:" prefix or explicit "what is in this image"
    # Avoid matching "photosynthesis", "photoshopped", "screenshot of code" etc.
    if (ql.startswith("image:") or ql.startswith("图片:") or ql.startswith("照片:")
        or "what is in this image" in ql or "describe this image" in ql
        or "describe the image" in ql or "what's in this picture" in ql
        or "what's in this photo" in ql):
        return "minimax-m3"  # public API avoids Agnes quality drift
    # v0.9.2: CJK queries are NEVER short chat (always need reasoning)
    is_cjk_query = bool(CN_PATTERN.search(query))
    # v0.9.30: relaxed is_chat_short.
    is_chat_short = (
        not is_cjk_query
        and len(query) < 30
        and not any(k in ql for k in ("def ", "class ", "import ", "analyze", "compare", "design", "code", "设计", "分析", "比较", "实现", "架构", "推导", "证明", "复杂", "对比", "评估", "function "))
    )
    if is_chat_short:
        return "deepseek-v4-flash"
    d = _query_difficulty(query)

    # CN first (dual-main mid-lower): never let EN code keywords steal CJK.
    if is_cjk_query or query_type == "cn":
        return _pick_mid_lower() or "minimax-m3"

    # Code/debug: light flash; mid-lower M3; hard code → sol (not grok).
    _code_kw = (
        "def ", "class ", "import ", "function ", "traceback", "python",
        "linked list", "refactor ", "implement ", "asyncio", "race condition",
    )
    if query_type in ("code", "debug") or any(k in ql for k in _code_kw):
        # Hard code only (high d / long); all debug + medium code → M3 mid-lower.
        if d >= 0.75 or len(query) >= 200:
            return _pick_hard() or "minimax-m3"
        if query_type == "debug" or d >= 0.35 or len(query) >= 80:
            return _pick_mid_lower() or "minimax-m3"
        return "deepseek-v4-flash"

    is_design = any(k in ql for k in (
        "设计", "架构", "分布式", "算法", "design", "architect", "distributed",
        "consensus", "raft", "consistent hash", "load balance",
    ))
    _hard_kw = (
        "设计", "架构", "分布式", "算法", "design", "architect", "distributed",
        "consensus", "raft", "consistent hash", "load balance", "scalable",
        "high availability", "高可用", "微服务", "microservice", "CRDT",
        "CQRS", "saga", "idempotent", "exactly-once", "multi-tenant", "sharding",
    )
    _n_kw = sum(1 for k in _hard_kw if k in ql)
    has_hard_signal = (
        len(query) >= 100
        or (_n_kw >= 1 and len(query) >= 60)
        or _n_kw >= 2
    )
    if is_design and has_hard_signal:
        if _is_sacred_query(query, d):
            # audit 2026-08-16: the Sol opt-out gate used to short-circuit sacred
            # to None when ANCHOR_DISABLE_SOL=1, so sacred queries fell through to
            # the auto pool (flash) instead of the intended degrade path. Pick the
            # sacred worker (fable→hard→mid, already sol-aware via _sol_in); only
            # use sol as a last-resort when the gate allows it.
            _sp = _pick_sacred_worker()
            return _sp or ("gpt-5.6-sol" if _sol_allowed() else None)
        # Heavy systems design → sol; otherwise mid-upper grok.
        _arch_heavy = sum(
            1 for k in (
                "multi-tenant", "sharding", "high availability", "failover",
                "微服务", "microservice", "saga", "cqrs", "exactly-once",
                "quota isolation", "pareto",
            ) if k in ql
        )
        if d >= 0.78 or _arch_heavy >= 2 or len(query) >= 140:
            return _pick_hard() or "minimax-m3"
        return _pick_mid_upper() or "minimax-m3"
    if is_design:
        # Medium design → M3 mid-lower.
        return _pick_mid_lower() or "minimax-m3"

    if query_type == "agent":
        _agent_hard = any(s in ql for s in (
            "multi-step", "step by step", "do these in order", "run tests",
            "fix failures", "then commit", "refactor", "architecture",
        ))
        if is_cjk_query or d >= 0.35 or len(query) >= 70 or _agent_hard:
            # Hard agent architecture → sol; else M3 mid-lower (tools).
            if d >= 0.75 or "architecture" in ql:
                return _pick_hard() or "minimax-m3"
            return _pick_mid_lower() or "minimax-m3"
        return "deepseek-v4-flash"
    if query_type == "cn":
        return _pick_mid_lower() or "minimax-m3"

    # General EN / reasoning dual-main bands.
    # mid-lower: d < 0.55 → M3
    # mid-upper: 0.55 <= d < 0.78 → Grok (pure reason extends to <0.9)
    # hard: d >= 0.78 → Sol (unless pure reason still mid-upper)
    # sacred: fable
    _reason_kw = (
        "analyze", "analyse", "compare", "tradeoff", "trade-off", "pros and cons",
        "why ", "explain", "reasoning", "推导", "分析", "比较", "权衡", "对比",
    )
    is_pure_reason = (
        query_type in ("reasoning", "en", "chat")
        and any(k in ql for k in _reason_kw)
        and not any(k in ql for k in ("def ", "class ", "import ", "function ", "implement ", "code "))
    )
    if d >= 0.55:
        if _is_sacred_query(query, d):
            # audit 2026-08-16: same fix as the design branch — never short-circuit
            # sacred to None; degrade via _pick_sacred_worker (sol-aware).
            _sp2 = _pick_sacred_worker()
            return _sp2 or ("gpt-5.6-sol" if _sol_allowed() else None)
        # Grok owns mid-upper reasoning; only extreme pure-reason goes sol/fable.
        if is_pure_reason and d < 0.9:
            return _pick_mid_upper() or "minimax-m3"
        if d >= 0.78:
            return _pick_hard() or "minimax-m3"
        return _pick_mid_upper() or "minimax-m3"

    # General low/mid: calibration picker, then bias dual-main.
    chosen = pick_worker_by_tier("auto", d, query_type)
    if chosen in (FABLE5, OPUS) and not _is_sacred_query(query, d):
        return _pick_mid_upper() or chosen
    return chosen




# Quality floors: single product uses TIER_FLOORS["auto"] (legacy keys alias).
# Unified floor source: anchor.pareto.TIER_FLOORS.
# v0.9.43: removed duplicate definition; canonical source is pareto.py.
# Re-exported under legacy name TIER_FLOOR for backwards compatibility.
from anchor.pareto import TIER_FLOORS as TIER_FLOOR
from anchor.pareto import MIN_QUALITY_HARD_FLOOR  # P1-B: hard exclusion gate

# Per-tier worker pool (in cost order)
# Design/hard may select fable/opus when present in pool.
# target. picker normally prefers cheaper workers, but design/hard override routes
# directly to opus. Pool inclusion lets fallback-walk also reach opus.
# Boss WS#1: free pool is dpsk (OpenCode Zen) + Kilo, with M3 as the
# zero-marginal-cost subscription main. Paid baosiapi is reserved for Opus-5
# quality-floor escalation and overflow.
# v0.9.50-p0: fable-5 gradually re-enabled at 10% rollout; TierPool inclusion
# lets fallback-walk reach it but routing probability gate (_should_try_fable5)
# keeps traffic low.
# Opt-in workers (pro/sonnet/...) join pool only when enabled.
# Vision path forces vision-capable workers.
# v0.9.32: Vision-capable worker set. dpsk does not support image_url input
# (opencode-zen rejects image_url with 400). Used by server.py fallback walk
# to skip non-vision workers.
# Verified per-worker probe 2026-07-11 (1x1 + 64x64 red PNG, 30s timeout).
# Vision-capable set (lean pool + opt-in Claude mid).
_VISION_CAPABLE = {
    "minimax-m3",            # multimodal main
    "claude-fable-5",        # top quality vision (default enabled)
    "claude-haiku-4-5",      # opt-in
    "claude-sonnet-5",       # opt-in
    # v0.9.7X-P2: claude-opus-5 removed from vision worker set (auth-gated since v0.9.63; see p2_opus5_eval.md)
}

# v0.9.51: single source of truth — config.tier_pool_from_config().
# Keep dict shape for backward-compat callers.
import os as _os
from anchor.config import tier_pool_from_config as _tier_pool_from_config

_DEEPSEEK_PRO_ENABLED = bool(_os.environ.get("ANCHOR_ENABLE_DEEPSEEK_PRO"))


def _build_tier_pool_order() -> dict[str, list[str]]:
    # pure mode: single auto pool only (no basic/premium/ultra product keys).
    auto = list(_tier_pool_from_config("auto"))
    return {"auto": list(auto)}


TIER_POOL_ORDER = _build_tier_pool_order()

# Phase 3 FIX-A: legacy WORKER_COST kept as a back-compat alias for any
# third-party callers (server.py /v1/models, etc.). New code should use
# ``config.worker_cost(name)`` which reads the canonical WORKER_COST_YUAN
# table sourced from the v8 calibration `cost_yuan_avg.medium` column.
#
# audit 2026-08-16 (C7): the hardcoded values (sonnet 0.1, opus 10.0,
# fable 15.0) were ~6000x off the v8 per-call costs and leaked into
# /v1/models pricing. Build purely from WORKER_COST_YUAN now.
from anchor.config import WORKER_COST_YUAN as _WCY, worker_cost as _wc
WORKER_COST = {
    SONNET: _WCY.get(SONNET, 0.0),
    OPUS: _WCY.get(OPUS, 0.0),
    FABLE5: _WCY.get(FABLE5, 0.0),
}
for _n, _v in _WCY.items():
    WORKER_COST.setdefault(_n, _v)




# v0.9.22 (re-enabled v0.9.50-p1): fable-5 gradual rollout gate. v0.9.50-p1 Gate-2
# probe PASSED (placeholder_rate=0.0% on 50 samples vs opus-4-8). Default 100% per
# v0.9.50-p0 spec rule "rate<5% → enable". Reduced to 50% in v0.9.66 (P3 audit:
# fable 2.7% of calls but 72.4% of cost; 78% fallback_from). Further reduced to
# 25% in v0.9.67 (continued P3 follow-up; expected ~¥300/mo vs ¥516 at 50%).
# Override via FABLE5_TRAFFIC_PCT env var (0-100) or /admin/gradual/fable5 endpoint.
# Used at sacred decision points to (probabilistically) try fable-5 instead of
# opus-4-8.
import random as _random

_FABLE5_PCT = float(_os.environ.get("FABLE5_TRAFFIC_PCT", "25"))


def set_fable5_traffic_pct(pct: float) -> None:
    """Update the fable-5 gradual rollout percentage (0-100). Persists in process memory."""
    global _FABLE5_PCT
    _FABLE5_PCT = max(0.0, min(100.0, float(pct)))


def get_fable5_traffic_pct() -> float:
    """Current fable-5 rollout percentage (0-100)."""
    return _FABLE5_PCT


def _should_try_fable5() -> bool:
    """Probabilistic gate: return True if fable-5 should be tried for this request."""
    if _FABLE5_PCT <= 0:
        return False
    if _FABLE5_PCT >= 100:
        return True
    return _random.random() * 100 < _FABLE5_PCT


import json as _json
from pathlib import Path as _Path
from anchor import _ROOT as _A_ROOT

_CAL_FILE = _Path(str(_A_ROOT / "data/calibration/quality_table.json"))
_CAL_V2_FILE = _Path(str(_A_ROOT / "data/calibration/quality_table_v2.json"))
_CAL_V3_FILE = _Path(str(_A_ROOT / "data/calibration/quality_table_v3.json"))
_CAL_V6_FILE = _Path(str(_A_ROOT / "data/calibration/quality_table_v6.json"))
# Phase 3 FIX-C: v8 calibration (1023 calls, 11 workers x 95 cases). Authoritative.
_CAL_V8_FILE = _Path(str(_A_ROOT / "data/calibration/quality_table_v8.json"))

_WORKER_QUALITY: dict | None = None
_CAL_CACHE: dict[str, dict] = {}


def _load_cal(path: _Path) -> dict:
    """Lazy-load a calibration JSON file, cached after first access."""
    key = str(path)
    if key not in _CAL_CACHE:
        try:
            with open(path) as _cal_f:
                _CAL_CACHE[key] = _json.load(_cal_f)
        except FileNotFoundError:
            _CAL_CACHE[key] = {}
        except Exception:
            _CAL_CACHE[key] = {}
    return _CAL_CACHE[key]


def _get_cal_v6() -> dict:
    return _load_cal(_CAL_V6_FILE)


def _get_cal_v8() -> dict:
    return _load_cal(_CAL_V8_FILE)


def _get_cal_v3() -> dict:
    return _load_cal(_CAL_V3_FILE)


def _get_cal_v2() -> dict:
    return _load_cal(_CAL_V2_FILE)


def _get_cal_v1() -> dict:
    return _load_cal(_CAL_FILE)


def _get_worker_quality() -> dict:
    global _WORKER_QUALITY
    if _WORKER_QUALITY is not None:
        return _WORKER_QUALITY
    # Phase 3 FIX-C: prefer v8 (1023-call authoritative re-bench) first,
    # then v6, v3, v1. v8 supersedes v6: 11 workers x 95 cases vs v6's
    # smaller sample; calibration team decision 2026-08-15.
    cal_v8 = _get_cal_v8()
    if cal_v8 and "table_coarse" in cal_v8 and cal_v8["table_coarse"]:
        _WORKER_QUALITY = cal_v8["table_coarse"]
        return _WORKER_QUALITY
    cal_v6 = _get_cal_v6()
    cal_v3 = _get_cal_v3()
    if cal_v6 and "table_coarse" in cal_v6 and cal_v6["table_coarse"]:
        _WORKER_QUALITY = cal_v6["table_coarse"]
    elif "table_coarse" in cal_v3 and cal_v3["table_coarse"]:
        _WORKER_QUALITY = cal_v3["table_coarse"]
    else:
        try:
            cal = _get_cal_v1()
            _WORKER_QUALITY = cal.get("table", {})
        except Exception:
            _WORKER_QUALITY = {}
    return _WORKER_QUALITY


def _difficulty_bucket(d: float) -> str:
    """Map d_value (0-1) to bucket name."""
    if d >= 0.7:
        return "hard"
    if d >= 0.3:
        return "medium"
    return "easy"


def _expected_quality(worker: str, d_value: float) -> float:
    """Look up worker quality for the given difficulty bucket. Falls back to 0.5 if unknown."""
    wq = _get_worker_quality()
    if not wq or worker not in wq:
        return 0.5
    bucket = _difficulty_bucket(d_value)
    return wq[worker].get(bucket, 0.5)



def _expected_quality_per_cat(worker: str, category: str, d_bucket: str) -> float:
    """v0.9.12 (F directive): lookup priority is v3 (live_routed, 100 records,
    reflects anchor's actual routing traffic) > v2 > fallback 0.5.
    Prefer latest live_routed calibration tables.
    """
    # Phase 3 FIX-C: prefer v8 (authoritative re-bench) per_cat first,
    # then v6, v3, v2.
    cal_v8 = _get_cal_v8()
    if cal_v8:
        per_cat = cal_v8.get("table_per_cat", {}).get(worker, {}).get(category, {})
        if d_bucket in per_cat:
            return per_cat[d_bucket]
        coarse = cal_v8.get("table_coarse", {}).get(worker, {})
        if d_bucket in coarse:
            return coarse[d_bucket]
    # v0.9.13: prefer v6 > v3 > v2 (latest live_routed)
    cal_v6 = _get_cal_v6()
    if cal_v6:
        per_cat = cal_v6.get("table_per_cat", {}).get(worker, {}).get(category, {})
        if d_bucket in per_cat:
            return per_cat[d_bucket]
        coarse = cal_v6.get("table_coarse", {}).get(worker, {})
        if d_bucket in coarse:
            return coarse[d_bucket]
    cal_v3 = _get_cal_v3()
    if cal_v3:
        coarse = cal_v3.get("table_coarse", {}).get(worker, {})
        if d_bucket in coarse:
            return coarse[d_bucket]
    # Fallback to v2 per_cat
    cal_v2 = _get_cal_v2()
    if cal_v2:
        per_cat = cal_v2.get("table_per_cat", {}).get(worker, {}).get(category, {})
        if d_bucket in per_cat:
            return per_cat[d_bucket]
        coarse = cal_v2.get("table_coarse", {}).get(worker, {})
        if d_bucket in coarse:
            return coarse[d_bucket]
    return 0.5


def pick_worker_by_tier(tier: str, d_value: float, query_type: str = "en", query: str = "") -> str:
    """Category-aware calibration picker for the single auto lane.

    Cheapest worker in pool whose expected quality >= TIER_FLOORS["auto"].
    Hard design may force fable/opus when present in pool.
    """
    # v0.9.14: design + d>=0.85 → opus (real hard design, not just keyword).
    # _query_difficulty may give design keyword 0.7 (high), but trivial design like
    # "design a hello world" only gets 0.7. To distinguish real hard design we
    # require explicit hard signals: long query, multiple concepts, distributed/architect
    # keywords, etc.
    ql = (query or "").lower()
    is_design = any(k in ql for k in ("设计", "架构", "分布式", "算法", "design", "architect", "distributed", "consensus", "raft", "consistent hash", "load balance"))
    # Real hard design signals: long query (>=100 chars) + multiple concept words
    has_hard_signal = (
        len(query or "") >= 100 or
        sum(1 for k in ("设计", "架构", "分布式", "算法", "design", "architect", "distributed", "consensus", "raft", "consistent hash", "load balance", "scalable", "high availability", "高可用", "微服务", "microservice") if k in ql) >= 2
    )
    from anchor.config import normalize_routing_lane as _nrl
    lane = _nrl(tier)
    pool = TIER_POOL_ORDER.get(lane) or TIER_POOL_ORDER.get("auto") or list(TIER_POOL_ORDER.values())[0]
    pool_set = set(pool)
    # Single product: one quality floor.
    floor = TIER_FLOOR.get("auto", 0.70)

    if is_design and has_hard_signal:
        # Dual-main: sacred→fable; heavy hard→sol; else mid-upper grok.
        if _is_sacred_query(query or "", d_value):
            if FABLE5 in pool_set and _should_try_fable5():
                return FABLE5
            if _sol_allowed() and "gpt-5.6-sol" in pool_set:
                return "gpt-5.6-sol"
            if OPUS in pool_set:
                return OPUS
            if FABLE5 in pool_set:
                return FABLE5
        ql_p = (query or "").lower()
        _arch_heavy = sum(
            1 for k in (
                "multi-tenant", "sharding", "high availability", "failover",
                "微服务", "microservice", "saga", "cqrs", "exactly-once",
                "quota isolation", "pareto",
            ) if k in ql_p
        )
        if d_value >= 0.78 or _arch_heavy >= 2 or len(query or "") >= 140:
            if _sol_allowed() and "gpt-5.6-sol" in pool_set:
                return "gpt-5.6-sol"
        if "grok-4-6-reasoning" in pool_set:
            return "grok-4-6-reasoning"
        if _sol_allowed() and "gpt-5.6-sol" in pool_set:
            return "gpt-5.6-sol"

    # Detect category if query provided
    category = _detect_category(query) if query else "unknown"
    d_bucket = _difficulty_bucket(d_value)

    # Hard difficulty: still cost-first among workers meeting floor (full pool).
    # cost-first with category-aware quality lookup
    candidates = []
    for w in pool:
        q = _expected_quality_per_cat(w, category, d_bucket)
        if q >= floor:
            cost = _wc(w)
            candidates.append((cost, -q, w, q))

    if not candidates:
        # No worker meets floor in this category — relax to coarse
        # P1-B: Still enforce MIN_QUALITY_HARD_FLOOR in the relaxed path.
        candidates = []
        for w in pool:
            q = _expected_quality(w, d_value)
            if q < MIN_QUALITY_HARD_FLOOR:
                continue  # never select catastrophically bad workers
            if q >= floor * 0.85:  # 15% slack
                cost = _wc(w)
                candidates.append((cost, -q, w, q))
        if not candidates:
            return max(
                (w for w in pool if _expected_quality(w, d_value) >= MIN_QUALITY_HARD_FLOOR),
                key=lambda w: _expected_quality(w, d_value),
                default=pool[-1],
            )

    candidates.sort()
    return candidates[0][2]


def _auto_route_worker(query_type: str, query: str) -> Optional[str]:
    """Single-product hard rule: difficulty-aware lean-pool ladder."""
    return _difficulty_aware_route(query_type, query)


_ultra_difficulty_aware = _difficulty_aware_route  # back-compat alias

# v0.9.52: one product → one hard-rule function; legacy keys are aliases.
TIER_HARD_RULES = {
    "auto": _auto_route_worker,
}


def _recommended_max_tokens(query: str) -> int:
    """Pick max_tokens: thrift on easy, never starve hard/sacred quality.

    Principle: 能省则省 on light work; 当用则用 on hard/sacred — full budget.
    """
    q = query or ""
    ql = q.lower()
    d = _query_difficulty(q)
    # Explicit short-form asks: modest cap (still enough for quality bullets)
    import re as _re
    m = _re.search(r"(\d+)\s*(bullet|bullets|points|条|点)", ql)
    if m:
        n = max(1, min(20, int(m.group(1))))
        # ~60 tokens/bullet + headroom; hard topics still get more
        base = 80 + n * 60
        return min(1024, max(256, base)) if d >= 0.7 else min(512, max(128, base))
    if any(k in ql for k in ("one word", "one sentence", "一句话", "一个词", "pong", "say only")):
        return 64
    if d >= 0.9 or _is_sacred_query(q, d):
        return 1536  # sacred/extreme: never soft-cap quality
    if d >= 0.7:
        return 1024  # hard design/reason: full power
    if d >= 0.45:
        return 512   # mid dual-main
    if d >= 0.25:
        return 256
    return 128  # trivial chat
