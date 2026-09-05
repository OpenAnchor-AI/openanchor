"""LinUCB Contextual Bandit head (v0.9.53, Phase 3).

Replaces (or augments) the linear W of TrinityHead with a per-arm LinUCB
policy. Each arm = a worker. Context = the same 4 features used by
TrinityHead:
    x = [prior(qtype, worker), prompt_len/1000, budget, 1-budget]

LinUCB algorithm (Li et al. 2010, simplified):
  - per arm w: ridge-regression on (context, reward)
    A_w = sum_i x_i x_i^T + λI
    b_w = sum_i x_i * r_i
    theta_w = A_w^-1 b_w
  - selection: argmax_w (theta_w . x + alpha * sqrt(x^T A_w^-1 x))
  - update: increment A_w, b_w with new (x, r)

Reward = judge_score - ALPHA_PARETO * cost_yuan (the J(θ) objective).

Reference papers (2025):
  - PILOT (arXiv 2508.21141, EMNLP 2025 Findings): LinUCB + knapsack
  - HybridLLM (arXiv 2404.14618, ICLR 2024): quality-aware routing

Scope (v0.9.53 minimal viable):
  - 4-dim context (same as TrinityHead — incremental, no MiniLM yet)
  - Per-arm ridge regression with diagonal A (skip matrix inversion)
  - Exploration bonus alpha (env: ANCHOR_BANDIT_ALPHA, default 1.0)
  - Cold-start: arms with n_arm < 5 use only exploration bonus (no theta)
  - Safety net: role_prior_floors still applied before bandit selection
    (floor = 0.5 by default; bandit can only pick workers above it)

API:
  select(query_features: np.ndarray, pool: list[str]) -> (worker, score)
  update(query_features: np.ndarray, worker: str, reward: float)
  save(path) / load(path)  — JSON-based (audit R6: no pickle surface)
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# v0.9.53: feature flag. Default OFF to allow shadow evaluation.
# Enable with ANCHOR_BANDIT_ENABLED=1.
BANDIT_ENABLED: bool = os.environ.get("ANCHOR_BANDIT_ENABLED", "0") == "1"

# v0.9.54 (Tier 1.2): shadow mode. When ANCHOR_BANDIT_SHADOW=1, head.predict()
# runs the bandit in parallel with the linear head and records disagreement
# (without using the bandit pick for routing). This lets operators accumulate
# bandit context observations on real traffic without disturbing the prod
# pick, and validate that bandit != linear before flipping BANDIT_ENABLED.
BANDIT_SHADOW: bool = os.environ.get("ANCHOR_BANDIT_SHADOW", "0") == "1"

# v0.9.54 (Tier 3 B.3): mutex. Both flags ON simultaneously is operator
# misconfiguration (bandit picks routing AND we still spend CPU on shadow
# counters). When bandit is ENABLED, shadow silently turns off to avoid
# the spurious disagreement count.
if BANDIT_ENABLED and BANDIT_SHADOW:
    import warnings as _w_mut
    _w_mut.warn(
        "ANCHOR_BANDIT_ENABLED=1 with ANCHOR_BANDIT_SHADOW=1: shadow "
        "counters are misleading (bandit always agrees with itself). "
        "Disabling BANDIT_SHADOW for this process.",
        RuntimeWarning,
    )
    BANDIT_SHADOW = False

# Exploration bonus multiplier (sqrt(x^T A^-1 x) * alpha).
BANDIT_ALPHA: float = float(os.environ.get("ANCHOR_BANDIT_ALPHA", "1.0"))

# Ridge regularization (λ in A_w = sum x x^T + λI).
BANDIT_LAMBDA: float = float(os.environ.get("ANCHOR_BANDIT_LAMBDA", "1.0"))

# Cold-start threshold: arms with < N observations get pure exploration.
BANDIT_COLD_N: int = int(os.environ.get("ANCHOR_BANDIT_COLD_N", "5"))

# Default feature dim (matches TrinityHead: 4-dim).
# v0.9.54 (Tier 1.1): align with anchor.head.W_DIM=5. Tests use this as
# the default arm dimension; live wiring always passes head.W_DIM via the
# LinUCB(dim=...) ctor at get_bandit() call site so production code is
# unaffected even if the default moves.
FEATURE_DIM: int = 5


class LinUCBArm:
    """Per-worker ridge regression state."""

    __slots__ = ("A_diag", "b", "n", "theta")

    def __init__(self, dim: int = FEATURE_DIM, lambda_: float = BANDIT_LAMBDA):
        self.A_diag: np.ndarray = np.full(dim, lambda_, dtype=float)
        self.b: np.ndarray = np.zeros(dim, dtype=float)
        self.n: int = 0
        # Cached theta = A^-1 b (recomputed on update).
        self.theta: np.ndarray = np.zeros(dim, dtype=float)

    def update(self, x: np.ndarray, r: float) -> None:
        """Incremental ridge update with observation (x, r)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        self.A_diag += x * x
        self.b += x * float(r)
        self.n += 1
        # Recompute theta = A^-1 b (diagonal A).
        self.theta = self.b / self.A_diag

    def score(self, x: np.ndarray) -> tuple[float, float]:
        """Return (mean, ucb_bonus) for this arm on context x.

        mean = theta . x
        bonus = alpha * sqrt(sum (x_i^2 / A_diag_i))
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        mean = float(self.theta @ x)
        # UCB bonus: sqrt(x^T A^-1 x) ≈ sqrt(sum x_i^2 / A_diag_i) for diagonal A
        bonus = float(np.sqrt(np.sum((x * x) / self.A_diag)))
        return mean, bonus


class LinUCB:
    """Multi-arm LinUCB policy over worker pool."""

    def __init__(
        self,
        worker_names: list[str],
        dim: int = FEATURE_DIM,
        alpha: float = BANDIT_ALPHA,
        lambda_: float = BANDIT_LAMBDA,
        cold_n: int = BANDIT_COLD_N,
    ):
        self.worker_names: list[str] = list(worker_names)
        self.worker_idx: dict[str, int] = {n: i for i, n in enumerate(self.worker_names)}
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.cold_n = int(cold_n)
        self.arms: dict[str, LinUCBArm] = {
            n: LinUCBArm(dim=self.dim, lambda_=lambda_) for n in self.worker_names
        }
        self._lock = threading.Lock()

    def _get_arm(self, worker: str) -> Optional[LinUCBArm]:
        return self.arms.get(worker)

    def select(self, x: np.ndarray, pool: list[str]) -> tuple[Optional[str], float]:
        """Select worker from `pool` maximizing UCB.

        Returns (worker_name, score) or (None, -inf) if pool is empty.
        """
        if not pool:
            return None, float("-inf")
        x = np.asarray(x, dtype=float).reshape(-1)
        best_worker: Optional[str] = None
        best_score: float = float("-inf")
        for w in pool:
            arm = self._get_arm(w)
            if arm is None:
                continue
            mean, bonus = arm.score(x)
            if arm.n < self.cold_n:
                # Cold-start: pure exploration (no exploitation signal yet)
                score = self.alpha * bonus
            else:
                score = mean + self.alpha * bonus
            if score > best_score:
                best_score = score
                best_worker = w
        return best_worker, best_score

    def select_with_breakdown(
        self, x: np.ndarray, pool: list[str]
    ) -> list[tuple[str, float, float, float]]:
        """Return [(worker, mean, bonus, ucb)] sorted by ucb desc.

        Useful for dashboards and shadow evaluation. mean and bonus
        are NaN for cold-start arms (no theta available).
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        rows: list[tuple[str, float, float, float]] = []
        for w in pool:
            arm = self._get_arm(w)
            if arm is None:
                continue
            mean, bonus = arm.score(x)
            if arm.n < self.cold_n:
                rows.append((w, float("nan"), bonus, self.alpha * bonus))
            else:
                rows.append((w, mean, bonus, mean + self.alpha * bonus))
        rows.sort(key=lambda r: r[3], reverse=True)
        return rows

    def update(self, x: np.ndarray, worker: str, reward: float) -> None:
        """Update arm `worker` with observation (x, reward).

        reward = quality - ALPHA * cost (i.e. J(θ) component).
        Silently skips unknown workers.
        """
        arm = self._get_arm(worker)
        if arm is None:
            return
        x = np.asarray(x, dtype=float).reshape(-1)
        with self._lock:
            arm.update(x, float(reward))

    def get_arm_stats(self, worker: str) -> dict:
        arm = self._get_arm(worker)
        if arm is None:
            return {"exists": False}
        return {
            "exists": True,
            "n": arm.n,
            "theta": arm.theta.tolist(),
            "A_diag": arm.A_diag.tolist(),
        }

    def all_stats(self) -> dict:
        return {w: {"n": a.n, "theta": a.theta.tolist()} for w, a in self.arms.items()}

    def save(self, path: str) -> None:
        # v0.9.53 (audit R6): write JSON instead of pickle so the file cannot
        # carry arbitrary deserialization payloads. Each arm is serialized as
        # a flat dict (worker name -> {n, theta, A_diag}). npz/numpy is
        # rebuilt on load; nothing is eval()'d or unpickled.
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "linucb-v1",
            "worker_names": list(self.worker_names),
            "dim": int(self.dim),
            "alpha": float(self.alpha),
            "cold_n": int(self.cold_n),
            "arms": {
                name: {
                    "n": int(arm.n),
                    "theta": np.asarray(arm.theta, dtype=float).tolist(),
                    "A_diag": np.asarray(arm.A_diag, dtype=float).tolist(),
                }
                for name, arm in self.arms.items()
            },
        }
        # v0.9.53 (audit R8): atomic write — write to .tmp then os.replace
        # so a crash mid-write can't leave a half-written snapshot.
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: str) -> "LinUCB":
        # v0.9.53 (audit R6): JSON loader. Validates schema and types so a
        # tampered file can't instantiate arbitrary objects.
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict) or d.get("schema") != "linucb-v1":
            raise ValueError(f"unsupported bandit snapshot schema: {d.get('schema')!r}")
        obj = cls(
            worker_names=list(d.get("worker_names") or []),
            dim=int(d["dim"]),
            alpha=float(d["alpha"]),
            cold_n=int(d["cold_n"]),
        )
        arms_in = d.get("arms") or {}
        for name, arm_data in arms_in.items():
            if not isinstance(arm_data, dict):
                continue
            arm = LinUCBArm(dim=obj.dim, lambda_=1.0)
            arm.n = int(arm_data.get("n", 0))
            arm.theta = np.asarray(arm_data.get("theta", [0.0] * obj.dim), dtype=float)
            arm.A_diag = np.asarray(arm_data.get("A_diag", [1.0] * obj.dim), dtype=float)
            obj.arms[name] = arm
        return obj


# ---------------------------------------------------------------------------
# Module-level singleton + TrinityHead bridge
# ---------------------------------------------------------------------------
_singleton: Optional[LinUCB] = None
_singleton_lock = threading.Lock()


def get_bandit(worker_names: Optional[list[str]] = None, *, dim: Optional[int] = None) -> LinUCB:
    """Lazy-init module-level LinUCB singleton.

    First call initializes arms for the given worker_names (or current
    enabled workers from anchor.config if None). Subsequent calls return
    the same object so updates accumulate.

    Args:
        worker_names: explicit arm list, or None to use enabled workers.
        dim: feature vector dimension (default: FEATURE_DIM=5). Tests can
            pass dim=4 to keep the v0.9.53 unit tests' contract intact.
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        if worker_names is None:
            try:
                from anchor.config import enabled_workers
                worker_names = [w.name for w in enabled_workers()]
            except Exception:
                worker_names = []
        _dim = int(dim) if dim is not None else FEATURE_DIM
        _singleton = LinUCB(worker_names=worker_names, dim=_dim)
        return _singleton


def reset_bandit() -> None:
    """Test helper: drop the singleton so a fresh one is built on next get."""
    global _singleton
    with _singleton_lock:
        _singleton = None


__all__ = [
    "LinUCB",
    "LinUCBArm",
    "BANDIT_ENABLED",
    "BANDIT_SHADOW",
    "BANDIT_ALPHA",
    "BANDIT_LAMBDA",
    "BANDIT_COLD_N",
    "FEATURE_DIM",
    "get_bandit",
    "reset_bandit",
]