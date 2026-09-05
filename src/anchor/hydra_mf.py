"""RouteLLM-style Matrix Factorization router head (v0.9.38).

Independent of anchor.hydra_head (logreg) and anchor.head (trinity) —
HydraMF is the third backend for the routing head, trained on Anchor's
own judged data. Selectable via ANCHOR_HEAD_BACKEND=mf (default keeps
the existing logreg path).

Design (RouteLLM arXiv 2406.18665, simplified):
- Query embedding v_q in R^{query_dim} (64-d, mirrors hydra_head._QUERY_DIM)
- Worker embedding v_m in R^{emb_dim} (8-d, one row per worker)
- Shared query projection W1 in R^{emb_dim x query_dim} + bias b1 in R^{emb_dim}
- Per-worker head w2 in R^{n_workers x emb_dim} + bias b2 in R^{n_workers}
- Score delta(query, worker) = w2[i] . (v_m[i] (x) (W1 v_q + b1)) + b2[i]

Parameter budget (8 workers, emb_dim=8, query_dim=64):
  W1: 8x64 = 512,  b1: 8,  v_m: 8x8 = 64,  w2: 8x8 = 64,  b2: 8
  total = 656 params (well under 5K target).

Training:
- Full-batch gradient descent on BCE(pred, reward), reward in [0, 1].
- L2 regularization against zero on all parameters (keeps the head small
  enough that 200 epochs on a few hundred pairs converges cleanly).
- Catalog-agnostic invariant: add_worker() appends a v_m row and a w2/b2
  row without mutating existing rows. Old workers' predictions stay
  bit-identical (floating-point reordering notwithstanding).

Only numpy. No torch / jax / sklearn. ~280 LOC.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
# v0.9.53 (audit R6): safe npz loader. Reject any non-ndarray entries so a
# tampered snapshot cannot smuggle pickled objects (which np.load(allow_pickle=True)
# would otherwise happily execute).
def _safe_load_npz(path):
    d = np.load(path, allow_pickle=False)
    return d

# Defaults — overridable via ANCHOR_MF_EMB_DIM / ANCHOR_MF_LR / ANCHOR_MF_EPOCHS.
_DEFAULT_EMB_DIM = 8
_DEFAULT_LR = 0.05
_DEFAULT_EPOCHS = 200
_DEFAULT_L2 = 1e-4

# Pareto objective coefficient: α in J(θ) = E[quality] - α·cost.
# Read from env so tests can override per-run. Default 0.1.
ALPHA_PARETO: float = float(os.environ.get("ALPHA_PARETO", "0.1"))



class HydraMF:
    """RouteLLM-style MF router head with catalog-agnostic add_worker."""

    def __init__(self, n_workers: int, emb_dim: int = _DEFAULT_EMB_DIM, *,
                 seed: int = 42, query_dim: int = 64,
                 worker_names: list[str] | None = None):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self.emb_dim = int(emb_dim)
        self.query_dim = int(query_dim)
        self.rng = np.random.default_rng(seed)
        s_w1 = 1.0 / math.sqrt(max(query_dim, 1))
        self.W1: np.ndarray = self.rng.standard_normal((self.emb_dim, self.query_dim)) * s_w1
        self.b1: np.ndarray = np.zeros(self.emb_dim, dtype=float)
        self.v_m: np.ndarray = self.rng.standard_normal((n_workers, self.emb_dim)) * 0.01
        self.w2: np.ndarray = self.rng.standard_normal((n_workers, self.emb_dim)) * 0.01
        self.b2: np.ndarray = np.zeros(n_workers, dtype=float)
        if worker_names is None:
            worker_names = [f"worker_{i}" for i in range(n_workers)]
        elif len(worker_names) != n_workers:
            raise ValueError("worker_names length must match n_workers")
        self.worker_names: list[str] = list(worker_names)
        self.worker_idx: dict[str, int] = {n: i for i, n in enumerate(worker_names)}

    # --- core ops ---

    def _project(self, query_emb: np.ndarray) -> np.ndarray:
        """W1 v_q + b1. Shape: (emb_dim,)."""
        return self.W1 @ np.asarray(query_emb, dtype=float) + self.b1

    def _scores(self, query_emb: np.ndarray) -> np.ndarray:
        """Per-worker delta. Shape: (n_workers,)."""
        h = self._project(query_emb)  # (emb_dim,)
        gated = self.v_m * h  # (n_workers, emb_dim)
        return np.einsum("ij,ij->i", self.w2, gated) + self.b2

    def predict(self, query_emb: np.ndarray) -> np.ndarray:
        """Return per-worker score. Shape: (n_workers,)."""
        return self._scores(query_emb)

    def best(self, query_emb: np.ndarray,
             pool: Optional[list[str]] = None) -> tuple[str, float]:
        """Return (worker_name, score) for the best-scoring worker (in pool or all)."""
        scores = self._scores(query_emb)
        if pool is not None:
            if not pool:
                raise ValueError("pool must be non-empty")
            mask = np.full(len(self.worker_names), -1e9, dtype=float)
            for w in pool:
                i = self.worker_idx.get(w)
                if i is not None:
                    mask[i] = 0.0
            masked = scores + mask
        else:
            masked = scores
        i = int(np.argmax(masked))
        return self.worker_names[i], float(masked[i])

    # --- catalog-agnostic mutation ---

    def add_worker(self, name: str) -> int:
        """Append a new worker. NEVER mutates existing rows.

        Idempotent: re-adding an existing name returns its index unchanged.
        Returns the worker index.
        """
        if name in self.worker_idx:
            return self.worker_idx[name]
        new_v = self.rng.standard_normal(self.emb_dim) * 0.01
        new_w2 = self.rng.standard_normal(self.emb_dim) * 0.01
        self.v_m = np.vstack([self.v_m, new_v.reshape(1, -1)])
        self.w2 = np.vstack([self.w2, new_w2.reshape(1, -1)])
        self.b2 = np.concatenate([self.b2, [0.0]])
        idx = len(self.worker_names)
        self.worker_names.append(name)
        self.worker_idx[name] = idx
        return idx
    # --- training ---

    def fit(self, X, *, lr=_DEFAULT_LR, n_epochs=_DEFAULT_EPOCHS, l2=_DEFAULT_L2):
        """Train via full-batch gradient descent on BCE + Pareto cost penalty.

        X: list of (query_emb, worker_idx, reward in [0,1]) or
           (query_emb, worker_idx, reward, cost_yuan) for Pareto-aware training.
        Default cost_yuan=0.0 when 4th element is missing (backward compat).
        Returns {final_loss, n_pairs, epochs_run}.

        v0.9.53 (P0-2 fix): the cost penalty now flows through the gradient.
        Previously only BCE influenced updates (params -= lr * err where
        err = (pred - y) / n), so the head never actually learned to
        prefer cheaper workers. New err signal:
            err = ((pred - y) + ALPHA_PARETO * c * pred * (1-pred)) / n
        so updates -= lr * dL/d_raw now properly push toward cheaper workers
        when predicted quality is similar.
        """
        if not X:
            return {"final_loss": 0.0, "n_pairs": 0, "epochs_run": 0}
        Q_list, W_list, Y_list, C_list = [], [], [], []
        for item in X:
            q, w, r = item[0], item[1], item[2]
            c = float(item[3]) if len(item) > 3 else 0.0
            Q_list.append(np.asarray(q, dtype=float).reshape(-1))
            W_list.append(w)
            Y_list.append(min(max(float(r), 1e-6), 1.0 - 1e-6))
            C_list.append(c)
        Q = np.stack(Q_list)
        W = np.array(W_list, dtype=int)
        Y = np.array(Y_list)
        C = np.array(C_list, dtype=float)
        n = len(X)
        last_loss = 0.0
        for _ep in range(int(n_epochs)):
            H = Q @ self.W1.T + self.b1
            G = self.v_m[W] * H
            raw = np.einsum("ij,ij->i", self.w2[W], G) + self.b2[W]
            raw_c = np.clip(raw, -50.0, 50.0)
            pred = 1.0 / (1.0 + np.exp(-raw_c))
            eps = 1e-12
            bce = float(-np.mean(Y * np.log(pred + eps) + (1.0 - Y) * np.log(1.0 - pred + eps)))
            cost_penalty = ALPHA_PARETO * float(np.mean(pred * C))
            loss = bce + cost_penalty
            last_loss = loss
            # dL/d_raw = (pred - y) + ALPHA_PARETO * c * pred * (1-pred)
            # err is dL/d_raw (averaged over n); params -= lr * err * d_raw/d_param.
            cost_grad = ALPHA_PARETO * C * pred * (1.0 - pred)
            err = ((pred - Y) + cost_grad) / n
            d_w2 = np.zeros_like(self.w2)
            d_v_m = np.zeros_like(self.v_m)
            d_b2 = np.zeros_like(self.b2)
            np.add.at(d_w2, W, err[:, None] * G)
            np.add.at(d_v_m, W, err[:, None] * self.w2[W] * H)
            np.add.at(d_b2, W, err)
            d_h = err[:, None] * (self.v_m[W] * self.w2[W])
            d_W1 = d_h.T @ Q
            d_b1 = d_h.sum(axis=0)
            d_W1 += l2 * self.W1
            d_b1 += l2 * self.b1
            d_v_m += l2 * self.v_m
            d_w2 += l2 * self.w2
            d_b2 += l2 * self.b2
            self.W1 -= lr * d_W1
            self.b1 -= lr * d_b1
            self.v_m -= lr * d_v_m
            self.w2 -= lr * d_w2
            self.b2 -= lr * d_b2
        return {"final_loss": float(last_loss), "n_pairs": n, "epochs_run": int(n_epochs)}

    # --- persistence ---

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            worker_names=np.array(self.worker_names, dtype='<U128'),
            W1=self.W1, b1=self.b1,
            v_m=self.v_m, w2=self.w2, b2=self.b2,
        )

    @classmethod
    def load(cls, path):
        d = _safe_load_npz(str(path))
        names = list(d["worker_names"])
        emb_dim = int(d["v_m"].shape[1])
        head = cls(n_workers=len(names), emb_dim=emb_dim,
                   query_dim=int(d["W1"].shape[1]))
        head.worker_names = names
        head.worker_idx = {n: i for i, n in enumerate(names)}
        head.W1 = d["W1"].astype(float)
        head.b1 = d["b1"].astype(float)
        head.v_m = d["v_m"].astype(float)
        head.w2 = d["w2"].astype(float)
        head.b2 = d["b2"].astype(float)
        return head
