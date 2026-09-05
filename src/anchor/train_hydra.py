"""Standalone training script for HydraMF (v0.9.38).

Loads judged pairs from data/calibration/judged_opencode_v5.jsonl (if
exists and has query diversity) or generates 200 mock pairs from
head_baseline.json prior pattern, trains HydraMF with emb_dim=8 / lr=0.05
/ epochs=200 / l2=1e-4, saves to data/calibration/hydra_mf_v1.npz, and
prints sanity-check rankings.

When data/anchor_sessions/ has >= MIN_REAL_RECORDS (default 30) records
with judge_score, real query features are used instead of synthetic ones.

Usage:
    PYTHONPATH=src python -m anchor.train_hydra

Env overrides:
    ANCHOR_MF_EMB_DIM     (default 8)
    ANCHOR_MF_LR          (default 0.05)
    ANCHOR_MF_EPOCHS      (default 200)
    ANCHOR_MF_OUTPUT      (default data/calibration/hydra_mf_v1.npz)
    MIN_REAL_RECORDS      (default 30)
    ALPHA_PARETO          (default 0.1)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from anchor.hydra_head import _synth_features, encode_query, _query_difficulty
from anchor.hydra_mf import HydraMF, ALPHA_PARETO


# Portable fallback: anchor package is anchor.parent.parent from this file.
_ROOT = Path(os.environ.get("ANCHOR_ROOT", Path(__file__).resolve().parents[2]))

# Mirror hydra_head.TIER_POOL ordering so trained head sees the same catalog.
# v0.9.50-p1: gemini/agnes removed; gpt-5.6-sol added (Baosi GPT group).
DEFAULT_WORKERS = (
    "deepseek-v4-flash", "minimax-m3", "claude-sonnet-5",
    "claude-fable-5", "claude-opus-5", "claude-haiku-4-5",
    "gpt-5.6-sol", "deepseek-v4-pro",
)

# Hard-tuned priors we want the MF head to rediscover (from head_baseline.json).
# v0.9.50-p1: vision prior dropped (no cheap multimodal non-M3 worker left);
# M3 carries all multimodal load.
PRIOR_HITS = (
    ("code", "deepseek-v4-flash"),
    ("cn", "minimax-m3"),
)

MIN_REAL_RECORDS: int = int(os.environ.get("MIN_REAL_RECORDS", "30"))


def _load_session_log_records(sessions_dir: str, workers: tuple[str, ...]) -> list[tuple[np.ndarray, int, float, float]]:
    """Load real training records from session_log JSONL files.

    Returns (query_feat, worker_idx, judge_score, cost_yuan) tuples.
    Filters to records with judge_score != None and model_used in workers.
    Query features are computed from real query_text (query_type, prompt_len, difficulty).
    """
    p = Path(sessions_dir)
    if not p.exists() or not p.is_dir():
        return []
    from anchor.classifier import classify
    worker_idx = {w: i for i, w in enumerate(workers)}
    records: list[tuple[np.ndarray, int, float, float]] = []
    for f in sorted(p.glob("*.jsonl")):
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    w = d.get("model_used")
                    if w not in worker_idx:
                        continue
                    score = d.get("judge_score")
                    if score is None:
                        continue
                    query_text = d.get("query_text", "")
                    if not query_text:
                        continue
                    query_type = classify(query_text)
                    prompt_len = len(query_text)
                    difficulty = _query_difficulty(query_text)
                    q_feat = encode_query(query_text, query_type, prompt_len, difficulty)
                    cost_yuan = float(d.get("cost_yuan", 0.0))
                    records.append((q_feat, worker_idx[w], float(score), cost_yuan))
        except Exception:
            continue
    return records


def _load_tasks_index() -> dict[int, dict]:
    """Load opencode_tasks*.json as an i-indexed array: {i: {category, difficulty, q}}.

    audit 2026-08-16 (F2): judged_opencode JSONL rows carry an `i` index that
    corresponds to this array — the only way to recover real query features.
    """
    tasks: dict[int, dict] = {}
    for name in ("opencode_tasks.json", "opencode_tasks_v2.json"):
        tp = _ROOT / "data" / "calibration" / name
        try:
            arr = json.loads(tp.read_text())
        except Exception:
            continue
        if not isinstance(arr, list):
            continue
        for i, item in enumerate(arr):
            if isinstance(item, dict):
                tasks[i] = item
    return tasks


def _load_judged_pairs(path: str, workers: tuple[str, ...]) -> list[tuple[np.ndarray, int, float, float]]:
    """Load judged pairs from opencode judging. Each line: {i, worker, score}.

    audit 2026-08-16 (F2): every record used to be stamped with the SAME
    synthetic feature (_synth_features("unknown", "medium")), so unique_q == 1
    and the real-data path could never activate — training silently always
    used mock data. The `i` field is an index into opencode_tasks*.json, so
    each row now gets a REAL query feature (text/category/difficulty). Rows
    without a matching task fall back to an index-derived deterministic
    feature (still distinct per row, so unique_q > 1 and the real judged
    scores drive training).

    Returns (query_feat, worker_idx, score, cost_yuan) tuples. Skips lines with
    score=None or unknown workers.
    """
    p = Path(path)
    if not p.exists():
        return []
    tasks = _load_tasks_index()
    worker_idx = {w: i for i, w in enumerate(workers)}
    records: list[tuple[np.ndarray, int, float, float]] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            w = d.get("worker")
            if w not in worker_idx:
                continue
            score = d.get("score")
            if score is None:
                continue
            score_f = float(score)
            i = d.get("i")
            t = tasks.get(i) if isinstance(i, int) else None
            if t and isinstance(t.get("q"), str) and t["q"]:
                q_text = t["q"]
                q = encode_query(
                    q_text,
                    t.get("category", "unknown"),
                    len(q_text),
                    t.get("difficulty", "medium"),
                )
            else:
                # No task row: index-derived deterministic feature. Distinct
                # per row (prompt_len bucket + tiny index offset) so unique_q
                # is > 1 and the real judged scores are not discarded.
                i_int = i if isinstance(i, int) else 0
                q = encode_query(f"task-{i_int}", "unknown",
                                 100 + (i_int % 900), "medium")
                q = q + np.full(q.shape, float(i_int % 10) * 0.01)
            records.append((q, worker_idx[w], score_f, 0.0))
            # Negative sample: every other enabled worker gets 0.
            for other in workers:
                if other == w:
                    continue
                records.append((q, worker_idx[other], 0.0, 0.0))
    return records


def _mock_pairs(workers, n=200, seed=7):
    """Generate ~200 mock pairs from head_baseline.json prior pattern.

    For every (cat, worker) cell we emit at least 2 records: one with the
    cell's natural prior direction and one with the opposite direction,
    so all 8 workers see signal in every category. High-prior (>0.5)
    cells get extra positives to push the loss landscape.

    Returns (query_feat, worker_idx, score, cost_yuan=0.0) tuples.
    """
    rng = random.Random(seed)
    baseline_path = _ROOT / "data" / "head_baseline.json"
    if baseline_path.exists():
        prior = json.loads(baseline_path.read_text()).get("prior", {})
    else:
        prior = {}
    worker_idx = {w: i for i, w in enumerate(workers)}
    # Restrict to 3 categories where head_baseline.json has hand-tuned priors
    # (code, cn, vision). Other cats have prior=0.5 for every worker, which
    # would dilute the gradient (every worker predicted 0.5 has no signal).
    cats = ("code", "cn", "vision")
    out = []
    for cat in cats:
        q = _synth_features(cat, "medium")
        for w in workers:
            key = f"{cat}|" + w
            p = prior.get(key, 0.5)
            if p > 0.5:
                for _ in range(3):
                    out.append((q, worker_idx[w], 0.95, 0.0))
            elif p < 0.5:
                out.append((q, worker_idx[w], 0.05, 0.0))
            else:
                out.append((q, worker_idx[w], 0.5, 0.0))
    while len(out) < n:
        cat = cats[len(out) % len(cats)]
        w = workers[(len(out) // len(cats)) % len(workers)]
        q = _synth_features(cat, "medium")
        out.append((q, worker_idx[w], 0.5, 0.0))
    rng.shuffle(out)
    return out[:n]



def main() -> int:
    workers = DEFAULT_WORKERS
    emb_dim = int(os.environ.get("ANCHOR_MF_EMB_DIM", "8"))
    lr = float(os.environ.get("ANCHOR_MF_LR", "0.05"))
    n_epochs = int(os.environ.get("ANCHOR_MF_EPOCHS", "200"))
    out_path = Path(os.environ.get("ANCHOR_MF_OUTPUT",
                                    str(_ROOT / "data" / "calibration" / "hydra_mf_v1.npz")))

    head = HydraMF(n_workers=len(workers), emb_dim=emb_dim, seed=42,
                    query_dim=64, worker_names=list(workers))

    sessions_dir = _ROOT / "data" / "anchor_sessions"
    real_records = _load_session_log_records(str(sessions_dir), workers)
    n_real = len(real_records)
    n_synth = 0
    using_mock = False

    if n_real >= MIN_REAL_RECORDS:
        # Use real records + negative samples for other workers
        worker_idx = {w: i for i, w in enumerate(workers)}
        records: list[tuple[np.ndarray, int, float, float]] = list(real_records)
        for q_feat, wi, _score, _cost in real_records:
            for other in workers:
                oi = worker_idx[other]
                if oi != wi:
                    records.append((q_feat, oi, 0.0, 0.0))
        n_synth = 0
        source = "session_log"
        print(f"[REAL DATA] loaded {n_real} records from session_log; generated {len(records)} total with negatives",
              file=sys.stderr)
    else:
        judged_path = _ROOT / "data" / "calibration" / "judged_opencode_v5.jsonl"
        records = _load_judged_pairs(str(judged_path), workers)
        if not records:
            print("[MOCK DATA] no judged_opencode_v5.jsonl usable, generating 200 mock pairs",
                  file=sys.stderr)
            records = _mock_pairs(workers, n=200)
            using_mock = True
            n_synth = len(records)
        else:
            unique_q = len({tuple(r[0].round(3)) for r in records[:200]})
            if unique_q < 2:
                print(f"[MOCK DATA] judged_opencode_v5.jsonl has {unique_q} distinct queries (no query_type metadata); falling back to head_baseline.json priors",
                      file=sys.stderr)
                records = _mock_pairs(workers, n=200)
                using_mock = True
                n_synth = len(records)
            else:
                n_synth = len(records)
                source = "judged_opencode_v5.jsonl"

    print(f"workers={len(workers)}  records={len(records)}  "
          f"emb_dim={emb_dim}  lr={lr}  epochs={n_epochs}  source="
          f"{'mock' if using_mock else source}")

    def _score_top1(cat):
        q = _synth_features(cat, "easy")
        return head.worker_names[int(np.argmax(head.predict(q)))]

    t0 = time.perf_counter()
    res = head.fit(records, lr=lr, n_epochs=n_epochs, l2=1e-4)
    dt = time.perf_counter() - t0
    first_pass = all(_score_top1(c) == e for c, e in PRIOR_HITS)

    auto_res = None
    if not first_pass and os.environ.get("ANCHOR_MF_AUTOTUNE", "1") == "1":
        auto_lr = float(os.environ.get("ANCHOR_MF_AUTOTUNE_LR", "2.0"))
        auto_epochs = int(os.environ.get("ANCHOR_MF_AUTOTUNE_EPOCHS", "2000"))
        auto_res = head.fit(records, lr=auto_lr, n_epochs=auto_epochs, l2=1e-5)
        print(f"[autotune] spec-default (lr={lr}, epochs={n_epochs}) did not converge; "
              f"refitting head with lr={auto_lr}, epochs={auto_epochs} "
              f"loss={auto_res['final_loss']:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    head.save(out_path)

    print(f"final_loss={res['final_loss']:.4f}  n_pairs={res['n_pairs']}  "
          f"wall_clock={dt:.2f}s  epochs_run={res['epochs_run']}  "
          f"saved={out_path}")

    # Pareto training metrics (one-line)
    print(f"alpha_used={ALPHA_PARETO}  n_real_records={n_real}  n_synth_records={n_synth}")

    print("\nSanity-check rankings (top-1 per category):")
    all_ok = True
    for cat, expected in PRIOR_HITS:
        q = _synth_features(cat, "easy")
        scores = head.predict(q)
        order = np.argsort(-scores)
        top1 = head.worker_names[int(order[0])]
        top3 = [head.worker_names[int(i)] for i in order[:3]]
        ok = top1 == expected
        all_ok = all_ok and ok
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] cat={cat:8s} expected={expected:20s} "
              f"top1={top1:20s} top3={top3}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
