"""HyDRA-style catalog-agnostic learned router head (v0.9.36).

Independent of anchor.head (trinity head) — kept separate until ship-gate
verifies this approach beats the static cascade. Same per-tier cost pool
semantics as fusion_modes.py, but the per-(query, worker) score is learned
from real calibration data (data/calibration/quality_table_v7.json) instead
of being a hard-coded cascade rule.

Design (per arxiv 2605.17106 spirit):
- Each worker gets a learned embedding (no per-(qt, worker) lookup table).
- A query projection W_q maps 64-dim query features into the embedding space.
- Score(query, worker) = (query_feat @ W_q.T) @ worker_emb[worker] + bias[worker].
- add_worker() appends a row to worker_emb without mutating existing rows.
  This is the catalog-agnostic invariant: old workers' scores stay bit-identical.

Training:
- Online logistic regression on (query_feat, worker, reward ∈ [0,1]) tuples.
- Loss: BCE(prediction, reward) + α * p * cost_yuan_per_query
  where α = ALPHA_PARETO env var (default 0.1). The cost term penalizes
  expensive workers, implementing the Pareto objective J(θ) from AGENTS.md.
- Real data source: quality_table_v7.json, anti-sparse-cell weighting so the
  high-volume worker (dpsk 174) doesn't drown the low-volume (opus 14).

Only numpy. No torch / jax / sklearn. ~280 LOC.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
# v0.9.53 (audit R6): safe npz loader. Reject any non-ndarray entries so a
# tampered snapshot cannot smuggle pickled objects (which np.load(allow_pickle=True)
# would otherwise happily execute).
def _safe_load_npz(path):
    d = np.load(path, allow_pickle=False)
    return d

# Portable fallback: anchor package is anchor.parent.parent from this file.
_ROOT = Path(os.environ.get("ANCHOR_ROOT", Path(__file__).resolve().parents[2]))

# 12 query types in priority order matching anchor.classifier.QUERIES (subset)
_QUERY_TYPES = (
    "code", "debug", "cn", "en", "math", "chat", "creative",
    "reasoning", "vision", "agent", "summarize", "translate", "unknown",
)
_QUERY_TYPE_IDX = {q: i for i, q in enumerate(_QUERY_TYPES)}

# 3 difficulty buckets (matches fusion_modes._query_difficulty output contract)
_DIFF_IDX = {"easy": 0, "medium": 1, "hard": 2}

# Pareto objective coefficient: α in J(θ) = E[quality] - α·cost.
# Read from env so tests can override per-run. Default 0.1.
ALPHA_PARETO: float = float(os.environ.get("ALPHA_PARETO", "0.1"))

# Total query-feature vector length.
# Layout:
#   [0:13]   one-hot query_type (13 slots; 12 categories + 1 unknown bucket)
#   [13:16]  one-hot difficulty (easy/medium/hard)
#   [16:20]  prompt_len bucket (4 bins: <100, 100-500, 500-2k, >2k)
#   [20:24]  binary signals (has_cjk, has_code_kw, has_math_kw, has_qmark)
#   [24:25]  cn_pattern bool (alias of has_cjk for backward compat)
#   [25:64]  zero padding (reserved for future embedding / multilingual)
_QUERY_DIM = 64


def _prompt_len_bucket(prompt_len: int) -> int:
    if prompt_len < 100:
        return 0
    if prompt_len < 500:
        return 1
    if prompt_len < 2000:
        return 2
    return 3


def _query_difficulty(query: str) -> str:
    """Heuristic: design/architect/分布式/Raft → hard; analyze/compare → medium; else easy."""
    ql = (query or "").lower()
    if any(k in ql for k in ("design", "architect", "分布式", "consensus", "raft", "crdt",
                              "consistent hash", "sharding")):
        return "hard"
    if any(k in ql for k in ("analyze", "compare", "evaluate", "explain", "summarize")):
        return "medium"
    return "easy"


def encode_query(query: str, query_type: str, prompt_len: int, difficulty: str) -> np.ndarray:
    """64-dim dense feature vector. Pure function, no RNG.

    The unknown / unknown fields default gracefully (one-hot index 12 / 0).
    """
    f = np.zeros(_QUERY_DIM, dtype=float)
    qt = _QUERY_TYPE_IDX.get(query_type, 12)  # unknown bucket
    f[qt] = 1.0
    df = _DIFF_IDX.get(difficulty, 0)
    f[13 + df] = 1.0
    f[16 + _prompt_len_bucket(prompt_len)] = 1.0
    q = query or ""
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in q)
    has_code = any(kw in q.lower() for kw in ("def ", "class ", "function ", "import ", "=>",
                                                 ".py", ".js", ".ts"))
    has_math = any(kw in q.lower() for kw in ("solve", "equation", "integral", "matrix",
                                                "sqrt", "theorem"))
    has_qmark = "?" in q
    f[20] = float(has_cjk)
    f[21] = float(has_code)
    f[22] = float(has_math)
    f[23] = float(has_qmark)
    f[24] = float(has_cjk)  # cn_pattern alias
    return f


class HydraHead:
    """Learned router head with catalog-agnostic add_worker."""

    def __init__(self, worker_names: list[str], emb_dim: int = 16, seed: int = 42):
        if not worker_names:
            raise ValueError("worker_names must be non-empty")
        self.worker_names: list[str] = list(worker_names)
        self.worker_idx: dict[str, int] = {n: i for i, n in enumerate(self.worker_names)}
        self.emb_dim = int(emb_dim)
        rng = np.random.RandomState(seed)
        self.worker_emb: np.ndarray = rng.randn(len(worker_names), self.emb_dim) * 0.01
        self.W_q: np.ndarray = rng.randn(self.emb_dim, _QUERY_DIM) * 0.01
        self.bias: np.ndarray = np.zeros(len(worker_names), dtype=float)

    # --- core ops ---

    def _score(self, query_feat: np.ndarray) -> np.ndarray:
        """Raw score per worker (no pool mask). Shape: (n_workers,)."""
        q_emb = query_feat @ self.W_q.T  # (emb_dim,)
        return q_emb @ self.worker_emb.T + self.bias

    def predict(self, query_feat: np.ndarray, pool: list[str]) -> tuple[str, float]:
        """Pick best worker from `pool` (subset of worker_names)."""
        if not pool:
            raise ValueError("pool must be non-empty")
        scores = self._score(query_feat)
        mask = np.full(len(self.worker_names), -1e9, dtype=float)
        for w in pool:
            i = self.worker_idx.get(w)
            if i is not None:
                mask[i] = 0.0
        masked = scores + mask
        best = int(np.argmax(masked))
        return self.worker_names[best], float(masked[best])

    def top_k(self, query_feat: np.ndarray, pool: list[str], k: int = 3) -> list[tuple[str, float]]:
        """Top-k workers by score, descending."""
        scores = self._score(query_feat)
        mask = np.full(len(self.worker_names), -1e9, dtype=float)
        for w in pool:
            i = self.worker_idx.get(w)
            if i is not None:
                mask[i] = 0.0
        masked = scores + mask
        idx = np.argsort(-masked)[:k]
        return [(self.worker_names[int(i)], float(masked[int(i)])) for i in idx]

    def fit_step(self, query_feat: np.ndarray, worker: str, reward: float, lr: float = 0.01,
                 cost_yuan_per_query: float = 0.0) -> float:
        """Online BCE + Pareto cost update. reward ∈ [0, 1]. Returns the loss for this step.

        Loss = BCE(y, p) + ALPHA_PARETO * p * cost_yuan_per_query.
        The cost penalty penalizes expensive workers, implementing J(θ).

        v0.9.53 (P0-2 fix): the cost penalty now flows through the gradient,
        not just the reported loss. Previously only the BCE term influenced
        parameter updates, so the head never actually learned to prefer
        cheaper workers (it just reported a higher "loss" while params
        tracked only BCE).

        err_total = (y - p) - ALPHA_PARETO * c * p * (1 - p)
            (since current code uses += with err, sign matches BCE direction)
        """
        idx = self.worker_idx.get(worker)
        if idx is None:
            raise ValueError(f"worker {worker!r} not in head")
        # Clip reward to (0, 1) to avoid log(0)
        reward_c = float(min(max(reward, 1e-6), 1.0 - 1e-6))
        q_emb = query_feat @ self.W_q.T  # (emb_dim,)
        score = float(q_emb @ self.worker_emb[idx] + self.bias[idx])
        score_clip = max(-50.0, min(50.0, score))
        pred = 1.0 / (1.0 + math.exp(-score_clip))
        bce_err = reward_c - pred
        # Cost penalty gradient: d/dp [ALPHA * p * c] = ALPHA * c,
        # and dp/d_raw = p*(1-p), so d/d_raw = ALPHA * c * p * (1-p).
        # Subtract from err (BCE direction) to penalize expensive workers.
        cost_err = ALPHA_PARETO * cost_yuan_per_query * pred * (1.0 - pred)
        err = bce_err - cost_err
        # Gradient of score w.r.t. parameters:
        #   d score / d worker_emb[idx] = q_emb
        #   d score / d W_q             = outer(worker_emb[idx], query_feat)  (emb_dim, _QUERY_DIM)
        #   d score / d bias[idx]       = 1
        # Chain rule: d L / d param = err * d score / d param (err already
        # absorbs the sigmoid derivative term via reward - pred shorthand,
        # which works well for online logistic in practice).
        self.worker_emb[idx] += lr * err * q_emb
        self.W_q += lr * err * np.outer(self.worker_emb[idx], query_feat)
        self.bias[idx] += lr * err
        bce = -(reward_c * math.log(pred + 1e-12) + (1.0 - reward_c) * math.log(1.0 - pred + 1e-12))
        cost_penalty = ALPHA_PARETO * pred * cost_yuan_per_query
        return float(bce + cost_penalty)

    # --- catalog-agnostic mutation ---

    def add_worker(self, name: str, init_emb: Optional[np.ndarray] = None) -> None:
        """Append a new worker. NEVER mutates existing rows.

        Idempotent: re-adding an existing name is a no-op.
        """
        if name in self.worker_idx:
            return
        if init_emb is None:
            new_emb = np.random.RandomState(hash(name) & 0xFFFFFFFF).randn(self.emb_dim) * 0.01
        else:
            new_emb = np.asarray(init_emb, dtype=float).reshape(self.emb_dim)
        self.worker_emb = np.vstack([self.worker_emb, new_emb.reshape(1, -1)])
        self.bias = np.concatenate([self.bias, [0.0]])
        self.worker_idx[name] = len(self.worker_names)
        self.worker_names.append(name)

    # --- persistence ---

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            worker_names=np.array(self.worker_names, dtype='<U128'),
            worker_emb=self.worker_emb,
            W_q=self.W_q,
            bias=self.bias,
        )

    @classmethod
    def load(cls, path: str) -> "HydraHead":
        d = _safe_load_npz(path)
        names = list(d["worker_names"])
        head = cls(names, emb_dim=int(d["worker_emb"].shape[1]))
        head.worker_emb = d["worker_emb"].astype(float)
        head.W_q = d["W_q"].astype(float)
        head.bias = d["bias"].astype(float)
        return head


# ---------- training data loader + trainer ----------

def _synth_features(query_type: str, difficulty: str) -> np.ndarray:
    """Build a 64-dim query feature for a (cat, diff) cell. Query text is synthetic
    (not used by encoder) but we pick a representative snippet so the binary
    signals (has_code_kw etc.) match the cell's category."""
    synth = {
        "code":      "def example(): return 42",
        "debug":     "Why does this TypeError happen?",
        "cn":        "请解释一下红黑树",
        "en":        "Tell me about closures",
        "math":      "Solve the integral of x^2 from 0 to 1",
        "chat":      "hi there",
        "creative":  "Write a poem about autumn",
        "reasoning": "Analyze the implications of X",
        "vision":    "Describe this image",
        "agent":     "Use tool to query database",
        "summarize": "Summarize this article",
        "translate": "Translate to Chinese",
        "unknown":   "general query",
    }
    prompt_len = 200 if difficulty == "easy" else (500 if difficulty == "medium" else 1200)
    return encode_query(synth.get(query_type, "general query"), query_type, prompt_len, difficulty)


def load_training_records(
    quality_table_path: str | None = None,
    *,
    seed: int = 0,
    worker_filter: Optional[set[str]] = None,
) -> list[tuple[np.ndarray, str, float]]:
    """Load (query_feat, worker, reward) tuples from the quality table.

    Anti-sparse-cell weighting: each cell is replicated `ceil(score * log1p(n))`
    times, so high-volume workers still dominate the gradient but rare cells
    (opus code/hard=0.811, n=14) get enough signal to be learned (1 rep).

    Empty cells (None or missing dict) are skipped — they would just dilute
    with reward=0 and zero out the sigmoid.

    worker_filter: optional set; if given, only records for workers in the set
    are returned. This is the catalog-agnostic safety net: a HydraHead trained
    on 8 workers never sees signals for a 9th worker it doesn't know about.
    """
    p = Path(quality_table_path) if quality_table_path is not None else (_ROOT / "data/calibration/quality_table_v7.json")
    if not p.exists():
        raise FileNotFoundError(f"quality table not found: {p}")
    with open(p) as f:
        data = json.load(f)
    table = data.get("table_per_cat", {})
    n_by = data.get("n_by_worker", {})
    records: list[tuple[np.ndarray, str, float]] = []
    for worker, cats in table.items():
        if worker_filter is not None and worker not in worker_filter:
            continue
        for cat, diffs in cats.items():
            if not isinstance(diffs, dict):
                continue
            for diff, score in diffs.items():
                if score is None:
                    continue
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    continue
                if score_f <= 0.0:
                    continue
                n = max(1, int(math.ceil(score_f * math.log1p(n_by.get(worker, 1)))))
                q_feat = _synth_features(cat, diff)
                for _ in range(n):
                    records.append((q_feat, worker, score_f))
    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def train_from_quality_table(
    head: HydraHead,
    *,
    epochs: int = 50,
    lr: float = 0.05,
    quality_table_path: str | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Train head from calibration data; returns final_loss / per-epoch losses / n_records.

    Catalog-agnostic: only loads records for workers already in the head.
    To train on extra workers, add_worker() them first, then train.
    """
    records = load_training_records(
        quality_table_path, seed=seed, worker_filter=set(head.worker_names),
    )
    if not records:
        return {"final_loss": 0.0, "losses": [], "n_records": 0}
    losses: list[float] = []
    rng = random.Random(seed)
    for ep in range(epochs):
        rng.shuffle(records)
        ep_loss = 0.0
        for q_feat, worker, reward in records:
            ep_loss += head.fit_step(q_feat, worker, reward, lr=lr)
        ep_loss /= max(len(records), 1)
        losses.append(ep_loss)
        if verbose and ep % 10 == 0:
            print(f"  epoch {ep:3d}  loss={ep_loss:.4f}  n={len(records)}")
    return {"final_loss": losses[-1], "losses": losses, "n_records": len(records)}


# ---------- drop-in integration entry points ----------

# Tier pool mirrors anchor.head.TIER_POOL but uses string worker names.
# basic:   dpsk first (cheap code), then m3 for CN/1M-ctx, then haiku for light en
# premium: + sonnet-5 for mid-en reasoning
# ultra:   + opus-4-8 for design/hard + fable-5 (sacred tier)
# v0.9.50-p1: gemini/agnes removed (resource-boundary cleanup).
_TIER_POOL: dict[str, tuple[str, ...]] = {
    "basic":   ("deepseek-v4-flash", "minimax-m3", "claude-haiku-4-5"),
    "premium": ("deepseek-v4-flash", "minimax-m3", "claude-haiku-4-5",
                "claude-sonnet-5"),
    "ultra":   ("deepseek-v4-flash", "minimax-m3", "claude-haiku-4-5",
                "claude-sonnet-5",
                "claude-opus-5", "claude-fable-5"),
}


def _tier_pool(tier: str) -> tuple[str, ...]:
    return _TIER_POOL.get(tier, _TIER_POOL["basic"])


def build_default_head() -> HydraHead:
    """Build head with current enabled workers from anchor.config.WORKERS."""
    from anchor.config import WORKERS
    enabled = [w.name for w in WORKERS if w.enabled]
    return HydraHead(enabled, emb_dim=16)


def predict(query: str, tier: str, prompt_len: int = 500) -> tuple[str, float]:
    """Drop-in predict: classify → encode → pick from tier pool.

    Note: builds a fresh head each call (no module-level cache). Production
    use should cache the trained head via load() after train_from_quality_table.
    """
    from anchor.classifier import classify
    head = build_default_head()
    qt = classify(query)
    diff = _query_difficulty(query)
    q_feat = encode_query(query, qt, prompt_len, diff)
    pool = _tier_pool(tier)
    # Filter pool to workers known to head (skip if head missing some)
    pool_have = tuple(w for w in pool if w in head.worker_idx)
    if not pool_have:
        pool_have = tuple(head.worker_names)
    return head.predict(q_feat, list(pool_have))


# ---------- v0.9.38: HydraMF backend (RouteLLM-style MF, opt-in) ----------

def _backend() -> str:
    """Read ANCHOR_HEAD_BACKEND env var. Default = 'logreg' (existing behavior)."""
    import os as _os
    return _os.environ.get("ANCHOR_HEAD_BACKEND", "logreg").lower()


def predict_mf(query: str, tier: str, prompt_len: int = 500,
               head_path: str | None = None) -> tuple[str, float]:
    """Drop-in predict using HydraMF backend (RouteLLM-style MF).

    Mirrors predict() interface: classify → encode → pick from tier pool.
    Loads a trained HydraMF from head_path (default: data/calibration/
    hydra_mf_v1.npz) or builds a fresh one if the file is missing.
    """
    from anchor.classifier import classify
    from anchor.hydra_mf import HydraMF
    from anchor.hydra_head import encode_query, _query_difficulty, _TIER_POOL

    if head_path is None:
        head_path = str(_ROOT / "data/calibration/hydra_mf_v1.npz")
    p = Path(head_path)
    if p.exists():
        head = HydraMF.load(p)
    else:
        # Fresh untrained head — produces random scores; used only when no
        # trained checkpoint exists yet. Caller should ensure train_hydra.py
        # has been run at least once.
        from anchor.config import WORKERS as _WORKERS
        enabled = [w.name for w in _WORKERS if w.enabled]
        head = HydraMF(n_workers=len(enabled), emb_dim=8, seed=42,
                        worker_names=enabled)

    qt = classify(query)
    diff = _query_difficulty(query)
    q_feat = encode_query(query, qt, prompt_len, diff)
    pool = _TIER_POOL.get(tier, _TIER_POOL["basic"])
    pool_have = tuple(w for w in pool if w in head.worker_idx)
    if not pool_have:
        pool_have = tuple(head.worker_names)
    return head.best(q_feat, list(pool_have))


def predict_dispatch(query: str, tier: str, prompt_len: int = 500
                     ) -> tuple[str, float]:
    """Route to logreg or MF backend based on ANCHOR_HEAD_BACKEND env var.

    Default = 'logreg' (existing HydraHead behavior, unchanged).
    Set ANCHOR_HEAD_BACKEND=mf to use the trained HydraMF model.
    """
    if _backend() == "mf":
        return predict_mf(query, tier, prompt_len=prompt_len)
    return predict(query, tier, prompt_len=prompt_len)
