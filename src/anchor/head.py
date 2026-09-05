"""TrinityHead v0.3 — quality-anchor LLM router head.

Design: per-tier worker pool + per (query_type, worker) prior + linear scorer.
Cold-start: prior=0.5, W@x gives constant base, cost breaks ties.
After learning: priors differentiate, head picks real best per (qt, tier).
References: Fable 5 meta-answer 2026-07-05, PLAN-v0.2.md.
"""
import json
import os
import re
import time
from pathlib import Path
import numpy as np
from anchor.config import WORKERS as _CFG_WORKERS, tier_pool_from_config

# Worker names derived from config.WORKERS (enabled only). Single source of
# truth: config.WORKERS defines the pool, this tuple mirrors it for the
# index-based predict/update internals.
WORKERS = tuple(w.name for w in _CFG_WORKERS if w.enabled)

# Costs in ¥/1M tokens, derived from config worker cost_out values.
# Used for routing bias in predict(); real billing is in client.
COSTS = tuple(
    next((w.cost_out for w in _CFG_WORKERS if w.name == name), 0.0)
    for name in WORKERS
)

TIERS = ("auto",)  # v0.9.52 single product lane
QUERIES = ("code", "debug", "cn", "en", "math", "chat", "creative",
           "reasoning", "vision", "agent", "summarize", "translate")

# Per-tier allowed worker pool (names, cost-ordered).
# Canonical source: config.tier_pool_from_config(). This dict is kept for
# backward compatibility; new code should call tier_pool_from_config(tier).
TIER_POOL = {
    tier: tier_pool_from_config(tier) for tier in ("auto",)
}

# v1.0.x lambda audit 2026-08-17: per-query-type cost penalty
# lambda=0.30 (light) -> 0.05 (hard). Default 0.15 preserves legacy.
LAMBDA_BY_ROLE = {
    "light": 0.30,
    "mid-en": 0.15,
    "hard": 0.05,
    "code": 0.20,
    "cn": 0.15,
    "agent": 0.10,
}

W = np.array([0.40, 0.10, 0.08, 0.06, 0.04, 0.03], dtype=float)
# v0.9.54 (Tier 1.1): 5-dim non-redundant features.
# Index 0: prior(qt, worker)        — table lookup
# Index 1: prompt_log_len            — log10(1 + prompt_len/200)
# Index 2: has_tools                 — 0/1 (tool calls prefer Fable/M3)
# Index 3: tool_call_n               — min(n, 5) / 5
# Index 4: json_mode                 — 0/1 (json-mode preference)
# Replaces the 4-dim [prior, prompt/1000, budget, 1-budget] which had
# x[2] + x[3] = 1.0 always (rank-deficient; model could only learn 2.5
# effective signals). For backward compat, _features_4d() is exported.
W_DIM = 6
prior_table: dict = {}  # populated by reset_to_baseline() below

# S14 (SRE R1 CRITICAL): threading lock for module-global W + prior_table.
# Without this lock, concurrent update() calls cause W vector corruption
# (interleaved numpy array writes) and prior table overwrites. Acquired in
# update() and predict(). Lock is module-level (single-process) — multi-worker
# uvicorn deploys already have per-process state per the snapshot comment.
import threading as _thr
# v0.9.53 (audit R6): safe npz loader. Reject any non-ndarray entries so a
# tampered snapshot cannot smuggle pickled objects (which np.load(allow_pickle=True)
# would otherwise happily execute).
def _safe_load_npz(path):
    d = np.load(path, allow_pickle=False)
    return d
_HEAD_LOCK = _thr.Lock()



def apply_role_prior_floors(*, force: bool = False) -> None:
    """Ensure designed role floors survive snapshot/learning overwrite.

    Snapshot load merges learned priors and can flatten agent/debug/code
    boosts back to ~0.5. Call after reset_to_baseline and after
    load_from_snapshot. force=True writes floors unconditionally (baseline);
    force=False only raises values below the floor (preserve higher learned).

    Phase 3 FIX-D: floors are derived from
    data/calibration/quality_table_v8.json where available
    (table_coarse[worker][bucket] and table_per_cat[worker][cat]).
    Hand-tuned values remain as the fallback when calibration data is missing
    for a worker (or the whole table is absent).
    """
    _VISION_WORKERS = {
        "minimax-m3", "claude-fable-5", "claude-sonnet-5",
        "claude-opus-5", "claude-haiku-4-5",
    }
    _FALLBACK_FLOORS = {
        ("code", "deepseek-v4-flash"): 0.90,
        ("code", "minimax-m3"): 0.84,
        ("code", "gpt-5.6-sol"): 0.86,
        ("code", "grok-4-5-reasoning"): 0.74,
        ("code", "claude-fable-5"): 0.70,
        ("agent", "deepseek-v4-flash"): 0.90,
        ("agent", "minimax-m3"): 0.88,
        ("agent", "gpt-5.6-sol"): 0.84,
        ("agent", "grok-4-5-reasoning"): 0.76,
        ("agent", "claude-fable-5"): 0.70,
        ("debug", "minimax-m3"): 0.90,
        ("debug", "deepseek-v4-flash"): 0.76,
        ("debug", "gpt-5.6-sol"): 0.84,
        ("debug", "grok-4-5-reasoning"): 0.76,
        ("debug", "claude-fable-5"): 0.74,
        ("reasoning", "grok-4-5-reasoning"): 0.90,
        ("reasoning", "gpt-5.6-sol"): 0.86,
        ("reasoning", "minimax-m3"): 0.80,
        ("reasoning", "claude-fable-5"): 0.78,
        ("reasoning", "deepseek-v4-flash"): 0.55,
        ("en", "grok-4-5-reasoning"): 0.88,
        ("en", "gpt-5.6-sol"): 0.86,
        ("en", "minimax-m3"): 0.82,
        ("en", "claude-fable-5"): 0.76,
        ("en", "deepseek-v4-flash"): 0.50,
        ("vision", "minimax-m3"): 0.95,
        ("vision", "claude-fable-5"): 0.95,
        ("vision", "claude-sonnet-5"): 0.95,
        ("vision", "claude-opus-5"): 0.95,
        ("vision", "claude-haiku-4-5"): 0.95,
        ("cn", "minimax-m3"): 0.95,
    }
    _ORNITH_FLOORS = {
        ("code", "ollama-ornith-35b"): 0.75,
        ("agent", "ollama-ornith-35b"): 0.78,
        ("reasoning", "ollama-ornith-35b"): 0.72,
        ("debug", "ollama-ornith-35b"): 0.75,
        ("chat", "ollama-ornith-35b"): 0.60,
        ("light", "ollama-ornith-35b"): 0.45,
    }

    # --- Phase 3 FIX-D: derive floors from calibration table v8 ---
    table_coarse = {}
    table_per_cat = {}
    try:
        from anchor import _ROOT as _ANCHOR_ROOT
        _tbl_path = _ANCHOR_ROOT / "data" / "calibration" / "quality_table_v8.json"
        if _tbl_path.exists():
            with open(_tbl_path, "r", encoding="utf-8") as _f:
                _tbl = json.load(_f)
            table_coarse = _tbl.get("table_coarse", {}) or {}
            table_per_cat = _tbl.get("table_per_cat", {}) or {}
    except Exception:
        table_coarse = {}
        table_per_cat = {}


    def _bucket(worker, role):
        wrow = table_coarse.get(worker)
        if not wrow:
            return None
        if role in ("reasoning", "en"):
            bucket = "hard"
        else:
            bucket = "medium"
        v = wrow.get(bucket)
        return float(v) if isinstance(v, (int, float)) else None

    def _cn_value(worker):
        cat_row = table_per_cat.get(worker) or {}
        cn = cat_row.get("cn")
        if cn is None:
            return None
        vals = [v for v in cn.values() if isinstance(v, (int, float))]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    floors = {}
    for role in ("code", "debug", "agent"):
        for w in WORKERS:
            v = _bucket(w, role)
            if v is not None:
                floors[(role, w)] = v
    for role in ("reasoning", "en"):
        for w in WORKERS:
            v = _bucket(w, role)
            if v is not None:
                floors[(role, w)] = v
    for w in _VISION_WORKERS:
        wrow = table_coarse.get(w)
        if wrow:
            vals = [v for v in wrow.values() if isinstance(v, (int, float))]
            if vals:
                floors[("vision", w)] = min(max(vals), 0.95)
    for w in WORKERS:
        v = _cn_value(w)
        if v is not None:
            floors[("cn", w)] = v
        else:
            fb = _bucket(w, "cn")
            if fb is not None:
                floors[("cn", w)] = fb


    for k, v in _FALLBACK_FLOORS.items():
        floors.setdefault(k, v)

    for w in WORKERS:
        if w not in _VISION_WORKERS:
            floors.setdefault(("vision", w), 0.30)
        if w != "minimax-m3":
            floors.setdefault(("cn", w), 0.30)

    for key, floor in _ORNITH_FLOORS.items():
        if key[1] not in WORKERS:
            continue
        cur = prior_table.get(key)
        if cur is None:
            prior_table[key] = floor
        elif force or cur < floor:
            prior_table[key] = floor

    for key, floor in floors.items():
        if key[1] not in WORKERS:
            continue
        cur = prior_table.get(key)
        if cur is None:
            prior_table[key] = floor
        elif force or cur < floor:
            prior_table[key] = floor



# v0.9.54: soft role bias for HydraHead score, applied in predict().
# Small bias nudges hard->Opus, agent->Sol, reasoning->Grok, coding->flash
# without overriding quality differences. Bias is small (<=0.10) so
# quality gap still wins — this is a tie-breaker, not a hard override.
ROLE_WORKER_BIAS: dict[tuple[str, str], float] = {
    ("hard", "claude-opus-5"): 0.05,
    ("agent", "gpt-5.6-sol"): 0.05,
    ("reasoning", "grok-4-6-reasoning"): 0.03,
    ("code", "deepseek-v4-flash"): 0.10,
    ("code", "minimax-m3"): 0.05,
    ("light", "deepseek-v4-flash"): 0.10,
    ("light", "minimax-m3"): 0.05,
    # v0.9.57: nudge local-ornith on code and agent (env-gated, zero marginal
    # cost) so the head prefers it as a cheap local option when ranked.
    ("code", "ollama-ornith-35b"): 0.03,
    ("agent", "ollama-ornith-35b"): 0.03,
}


def reset_to_baseline() -> None:
    """Reset prior_table to baseline (cold-start + role boosts).

    Idempotent. Safe to call from tests or server lifespan; resets any
    in-process learning back to the designed cold-start distribution.
    Production calls this once at startup before any /v1 request lands;
    tests call it in fixtures to isolate from other tests' mutations.
    """
    global prior_table
    # T-AUDIT-04 (SRE C-01 fix): mutate in-place rather than rebind. Rebinding
    # would orphan any reference imported via `from anchor.head import
    # prior_table` (the importer keeps the old dict). In-place mutation
    # keeps a single dict identity across all callers (tests + lifespan).
    prior_table.clear()
    prior_table.update({(q, w): 0.5 for q in QUERIES for w in WORKERS})
    # v0.9.50-p0: vision-capable worker set (gemini/agnes removed). Vision
    # cold-start prior boost for native multimodal workers; non-vision workers
    # (dpsk / dpsk-pro / gpt-5.6-sol) get 0.3 prior so they're skipped for
    # image queries until quality_table picks them up.
    _VISION_WORKERS = {"minimax-m3", "claude-fable-5", "claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"}
    _CODE_WORKERS = {"deepseek-v4-flash"}
    _CN_WORKERS = {"minimax-m3"}
    apply_role_prior_floors(force=True)


def lookup_difficulty_proxy(query_type: str, worker_name: str) -> float | None:
    """v0.9.54 (Tier 2.4): read the offline-computed difficulty proxy for a
    (query_type, worker) cell from data/anchor_sessions/_difficulty_proxy.jsonl.

    Returns None when the file or the cell is missing; callers fall back to
    the heuristic buckets via _features(difficulty=...). This separation lets
    us tune the proxy in scripts/compute_difficulty_proxy.py without touching
    the head's hot path.
    """
    import json as _json_dx
    from pathlib import Path as _P_dx
    try:
        # v0.9.54: anchor/head.py lives 2 levels below the repo root, so the
        # repo-rooted data path is ../../data/... not ../data/...
        path = _P_dx(__file__).resolve().parent.parent.parent / "data" / "anchor_sessions" / "_difficulty_proxy.jsonl"
    except Exception:
        return None
    if not path.exists():
        return None
    try:
        for line in path.open("r"):
            line = line.strip()
            if not line:
                continue
            try:
                row = _json_dx.loads(line)
            except Exception:
                continue
            if row.get("query_type") == query_type and row.get("worker") == worker_name:
                v = row.get("difficulty")
                return float(v) if isinstance(v, (int, float)) else None
    except Exception:
        return None
    return None


def _extract_query_features(query):
    """Lightweight query features for head.predict (A1 retry 2026-08-15)."""
    if not query:
        return {"prompt_len": 0, "has_code": False, "has_math": False,
                "has_image_url": False, "lang": "en", "difficulty_proxy": 0.0}
    q_lower = query.lower()
    prompt_len = len(query) // 4
    has_code = bool(re.search(r"```|\bdef\b|\bclass\b|\bimport\b", query))
    has_math = bool(re.search(r"\$[^$]+\$|\\[a-z]+\{|[\d]+\s*[+\-*/=<>]", query))
    has_image_url = bool(
        re.search(r"!\[[^\]]*\]\([^)]+\)|https?://[^\s]+\.(png|jpg|jpeg|gif|webp)", q_lower)
    )
    cjk_count = sum(1 for c in query if '\u4e00' <= c <= '\u9fff')
    if cjk_count > len(query) * 0.3:
        lang = "cn"
    elif cjk_count > 5:
        lang = "mixed"
    else:
        lang = "en"
    difficulty = min(
        1.0,
        (prompt_len / 1000) * 0.5 + (0.3 if has_code else 0) + (0.2 if has_math else 0),
    )
    return {
        "prompt_len": prompt_len,
        "has_code": has_code,
        "has_math": has_math,
        "has_image_url": has_image_url,
        "lang": lang,
        "difficulty_proxy": round(difficulty, 3),
    }


_WORKER_ROLE_TAGS = {
    "claude-fable-5": ("code", "hard", "sacred"),
    "claude-opus-5": ("hard",),
    "claude-sonnet-5": ("mid-en", "hard", "coding"),
    "claude-haiku-4-5": ("light", "mid-en"),
    "minimax-m3": ("cn-agent", "multi", "mid-en", "hard"),
    "deepseek-v4-flash": ("coding", "light"),
    "deepseek-v4-pro": ("coding", "mid-en", "hard"),
    "deepseek-v4-flash-official": ("coding", "mid-en", "fast"),
    "gpt-5.6-sol": ("coding", "mid-en", "hard", "agent"),
    "gpt-5.6-luna": ("code", "cn", "mid-en"),
    "gpt-5.6-terra": ("reasoning", "hard", "long-context"),
    "grok-4-6-reasoning": ("reasoning", "hard", "mid-en"),
    "grok-4-5-reasoning": ("reasoning", "hard", "mid-en"),
    "kimi-k3": ("cn", "multi", "vision", "long-context"),
    "ollama-ornith-35b": ("coding", "agent", "reasoning", "cn-agent", "mid-en", "local"),
}


def _features_4d(query_type, worker_name, prompt_len, budget):
    """Legacy 4-dim feature vector (kept for snapshot backward compat)."""
    p = prior_table.get((query_type, worker_name), 0.5)
    return np.array([p, prompt_len / 1000.0, budget, 1.0 - budget])


def _features(query_type, worker_name, prompt_len=500, budget=0.5,
              *, prompt_len_actual: int | None = None,
              has_tools: bool = False,
              tool_call_n: int = 0,
              json_mode: bool = False,
              difficulty: float | None = None) -> np.ndarray:
    """5-dim non-redundant feature vector.

    Layout (mirrors W/W_DIM docstring):
      [0] prior(qt, worker)
      [1] prompt_log_len   = log10(1 + prompt_len / 200)
      [2] has_tools        = 0.0 or 1.0
      [3] tool_call_n_norm = min(tool_call_n, 5) / 5
      [4] json_mode        = 0.0 or 1.0
    """
    p = prior_table.get((query_type, worker_name), 0.5)
    actual = prompt_len if prompt_len_actual is None else prompt_len_actual
    log_len = float(np.log10(1.0 + max(actual, 0) / 200.0))
    if difficulty is None:
        try:
            difficulty = lookup_difficulty_proxy(query_type, worker_name)
        except Exception:
            difficulty = None
    if difficulty is None:
        # v0.9.54 fallback: heuristic from _query_complexity (Fable 5 d-value).
        try:
            from anchor._query_complexity import query_complexity as _qc
            # cheap heuristic; worker_name unused so just consume the slot
            _ = worker_name
            _ = query_type
            difficulty = _qc('placeholder')
        except Exception:
            difficulty = 0.5
    difficulty = float(max(0.0, min(1.0, difficulty)))
    return np.array([
        p,
        log_len,
        1.0 if has_tools else 0.0,
        min(int(tool_call_n), 5) / 5.0,
        1.0 if json_mode else 0.0,
        difficulty,
    ], dtype=float)


def _bandit_select(bandit, pool, prior_snapshot, query_type,
                  prompt_len, budget):
    """v0.9.54: shared bandit pick routine for prod + shadow paths."""
    import numpy as _np
    x = _np.zeros(W_DIM, dtype=float)
    x[1] = float(_np.log10(1.0 + max(prompt_len, 0) / 200.0))
    x[2] = budget
    x[3] = 1.0 - budget
    candidates = []
    for name in pool:
        if name not in bandit.worker_idx:
            continue
        x_arm = x.copy()
        x_arm[0] = prior_snapshot.get((query_type, name), 0.5)
        w, ucb = bandit.select(x_arm, [name])
        if w:
            candidates.append((w, ucb))
    if not candidates:
        return None, float("-inf")
    return max(candidates, key=lambda c: c[1])


def predict(query_type, tier, prompt_len=500, budget=0.5,
            *, worker_name=None, query=None, prompt_features=None):
    """Predict best worker.

    v1.0.1 + A1 retry 2026-08-15: query features added.
      - New kwargs: worker_name (reserved), query (str|None),
        prompt_features (dict|None, precomputed).
      - Penalty: long prompt + light worker      -0.10
      - Penalty: CN query   + non-CN worker     -0.15
      - Penalty: math query + non-reasoning    -0.10
      - Bonus:   query_type matches worker role tag  +0.05
    Backward-compatible: existing positional call sites unchanged.
    """
    # v0.9.53 (audit R9): take a consistent snapshot under the same lock used
    # by update(). The previous lock only protected writers; readers could see
    # a new prior_table with an old W (or vice versa) during concurrent updates.
    with _HEAD_LOCK:
        w_snapshot = W.copy()
        prior_snapshot = dict(prior_table)
    # A1: extract query features (skipped when None)
    if prompt_features is None and query is not None:
        prompt_features = _extract_query_features(query)
    pf = prompt_features or {}
    _pf_lang = pf.get("lang", "en")
    _pf_has_math = bool(pf.get("has_math", False))
    _pf_prompt_len = int(pf.get("prompt_len", 0))
    _pf_is_cn = _pf_lang == "cn"
    from anchor.config import normalize_routing_lane as _nrl
    lane = _nrl(tier)
    pool = TIER_POOL.get(lane) or TIER_POOL.get("auto") or ()
    if not pool:
        pool = WORKERS
    _ = worker_name  # reserved for future targeted bias
    # v0.9.53 Phase 3: LinUCB contextual bandit selection. Falls back to
    # linear W if bandit is disabled (ANCHOR_BANDIT_ENABLED=0 default).
    # When enabled, bandit also updates on `update()` calls below so the
    # arms stay in sync with the linear head — useful for shadow eval.
    try:
        from anchor.bandit import BANDIT_ENABLED, get_bandit
    except Exception:
        BANDIT_ENABLED = False  # type: ignore
    if BANDIT_ENABLED:
        bandit = get_bandit(dim=W_DIM)
        pick, _ucb = _bandit_select(bandit, pool, prior_snapshot, query_type,
                                    prompt_len, budget)
        if pick:
            return pick, 0.0  # bandit path production return

    # v0.9.54 (Tier 1.2): shadow bandit mode. With ANCHOR_BANDIT_SHADOW=1
    # we run the bandit alongside the linear head in production and log
    # disagreement (linear != bandit). When ANCHOR_BANDIT_ENABLED flips
    # on later, the bandit has already accumulated context observations
    # and we know it disagrees-vs-wins rate on real traffic.
    _shadow_bandit_pick: str | None = None
    try:
        from anchor.bandit import BANDIT_SHADOW as _BS
    except Exception:
        _BS = False  # type: ignore
    if _BS:
        from anchor.bandit import get_bandit as _gb_sh
        _bandit_sh = _gb_sh(dim=W_DIM)
        _shadow_bandit_pick, _ = _bandit_select(
            _bandit_sh, pool, prior_snapshot, query_type, prompt_len, budget,
        )
    scores: dict[str, float] = {}
    for name in pool:
        x = _features(query_type, name, prompt_len, budget,
                      prompt_len_actual=prompt_len)
        idx = WORKERS.index(name) if name in WORKERS else None
        cost = COSTS[idx] if idx is not None else 0.0
        role_bias = ROLE_WORKER_BIAS.get((query_type, name), 0.0)
        # A1: role-match bonus + query-feature penalties
        _tags = _WORKER_ROLE_TAGS.get(name, ())
        _role_bonus = 0.05 if (query_type and query_type in _tags) else 0.0
        _penalty = 0.0
        if _pf_prompt_len > 500 and "light" in _tags:
            _penalty -= 0.10
        if _pf_is_cn and "cn" not in _tags and "cn-agent" not in _tags:
            _penalty -= 0.15
        if _pf_has_math and "reasoning" not in _tags and "hard" not in _tags:
            _penalty -= 0.10
        scores[name] = (float(w_snapshot[:W_DIM] @ x)
                        - cost * LAMBDA_BY_ROLE.get(query_type, 0.15) + role_bias
                        + _role_bonus + _penalty)
    if not scores:
        return WORKERS[0], 0.0
    best = max(scores, key=scores.get)
    score = scores[best]
    if score == -1e9:
        score = 0.0
    if _shadow_bandit_pick and _shadow_bandit_pick != best:
        try:
            from anchor.metrics import SHADOW_BANDIT_DISAGREE
            SHADOW_BANDIT_DISAGREE.labels(qt=query_type).inc()
        except Exception:
            pass
    return best, score


def update(query_type, tier, prompt_len, budget, worker_name, success):
    """S14 (SRE R1): thread-safe update. Concurrent callers serialized via _HEAD_LOCK."""
    global W
    with _HEAD_LOCK:
        if worker_name not in WORKERS:
            return
        # audit 2026-08-16 (A1): defensive clamp — the API model now bounds
        # success to [0,1], but head.update is also reachable from internal
        # callers; an out-of-range success would drive W with err ~999.
        success = float(min(max(success, 0.0), 1.0))
        key = (query_type, worker_name)
        p = prior_table.get(key, 0.5)
        prior_table[key] = p + 0.05 * (success - p)
        x = _features(query_type, worker_name, prompt_len, budget,
                      prompt_len_actual=prompt_len)
        idx = WORKERS.index(worker_name)
        cost = COSTS[idx] if idx < len(COSTS) else 0.0
        _lambda = LAMBDA_BY_ROLE.get(query_type, 0.15)
        pred = (W[:W_DIM] @ x) - cost * _lambda
        # v0.9.53 (audit R7): numerically-stable sigmoid. Direct exp(-pred) on
        # a large negative pred (e.g. -1000 from a runaway W) raises
        # RuntimeWarning("overflow") and returns inf. Use the softplus form
        # plus a clip so a single bad batch can never poison the head.
        pred = float(np.clip(pred, -60.0, 60.0))
        if pred >= 0:
            z = np.exp(-pred)
            sig = 1.0 / (1.0 + z)
        else:
            z = np.exp(pred)
            sig = z / (1.0 + z)
        err = success - sig
        W = W + 0.01 * err * x
        # v0.9.53 Phase 3: also feed LinUCB when enabled. Reward uses the
        # quality signal directly (success is already in [0,1]) plus a
        # small cost penalty so bandit learns quality/cost tradeoff over time.
        try:
            from anchor.bandit import BANDIT_ENABLED, get_bandit
        except Exception:
            BANDIT_ENABLED = False  # type: ignore
        if BANDIT_ENABLED:
            try:
                bandit = get_bandit(dim=W_DIM)
                if worker_name in bandit.worker_idx:
                    reward = float(success) - 0.1 * float(cost)
                    bandit.update(x, worker_name, reward)
            except Exception:
                pass



def load_from_snapshot(path) -> dict:
    """Load W + prior_table from a snapshot file written by drift.snapshot().

    The snapshot JSON layout (produced by drift.snapshot):
        {"ts": ..., "W": [w0, w1, w2, w3],
         "prior": {"<qt>|<worker>": <float>, ...}}

    Used by the server lifespan to populate in-memory state at startup,
    so multiple uvicorn workers share a consistent cold-start (defense
    against the per-process drift that arises with >1 worker).

    Returns {"loaded": bool, "n_priors": int, "ts": float|None}.
    On any error, leaves in-memory state unchanged and returns
    {"loaded": False, "error": str}.
    """
    import json as _json
    from pathlib import Path as _P
    p = _P(path)
    global W, prior_table
    if not p.exists():
        return {"loaded": False, "error": "snapshot file not found"}
    try:
        data = _json.loads(p.read_text())
    except Exception as e:
        return {"loaded": False, "error": f"parse: {e}"}
    try:
        new_w = [float(x) for x in data.get("W", [])]
        # audit 2026-08-16 (F5): W_DIM is 6 now; the old branch only padded
        # 4-dim snapshots when W_DIM == 5, so 4/5-dim snapshots silently left
        # W at cold-start while reporting loaded=True. Pad short snapshots to
        # W_DIM (defaults = cold-start priors) and reject over-long vectors
        # loudly instead of silently no-op.
        import numpy as _np
        if len(new_w) == W_DIM:
            W = _np.array(new_w, dtype=float)
        elif len(new_w) < W_DIM:
            W = _np.array(
                new_w + [0.04, 0.03, 0.02][: W_DIM - len(new_w)], dtype=float)
        else:
            return {"loaded": False, "error": f"W length {len(new_w)} > W_DIM {W_DIM}"}
        new_prior = {
            tuple(k.split("|", 1)): float(v)
            for k, v in data.get("prior", {}).items()
        }
        # T-AUDIT-04 (SRE C-01 fix): unconditional overwrite. The previous
        # `if k in prior_table` guard silently dropped snapshot priors for
        # any (query_type, worker) key that reset_to_baseline() hadn't
        # pre-populated, which would have lost newly-introduced query
        # types or workers. Lifespan order (reset_to_baseline -> load)
        # is preserved so role boosts from reset seed first, then snapshot
        # overrides with the live measurement values.
        prior_table.update(new_prior)
        apply_role_prior_floors(force=False)
        return {"loaded": True, "n_priors": len(new_prior),
                "ts": data.get("ts")}
    except Exception as e:
        return {"loaded": False, "error": f"apply: {e}"}


def save(path):
    # v0.9.53 (audit R6): write JSON sidecar with prior + npz for W.
    # Prior must NOT be np.array(dtype=object) any more — allow_pickle=False
    # loaders can't read object arrays. Keeping two files matches the
    # existing head_baseline.json + W convention used by lifespan.
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # v0.9.53 (audit R8): atomic write — write to .tmp then os.replace so a
    # crash mid-write cannot leave a half-written snapshot for load() to
    # read. Without this, a SIGKILL during save() corrupts head state.
    # v0.9.54: emit W_DIM-aligned vector (5 floats). Older snapshots
    # stored 4 floats; load_from_snapshot() pads those with the default
    # slot-4 weight. Coerce via np.asarray so a tampered or odd-length
    # vector doesn't silently break the writer.
    _W_serialized = list(np.asarray(W, dtype=float).reshape(-1).tolist())
    np.savez(p, W=W)
    sidecar = p.with_suffix('.json')
    tmp = sidecar.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({
        "schema": "head-v2",  # v2 = 5-dim W
        "ts": time.time(),
        "W": _W_serialized,
        "prior": {f"{qt}|{w}": v for (qt, w), v in prior_table.items()},
    }, indent=2), encoding="utf-8")
    os.replace(tmp, sidecar)
