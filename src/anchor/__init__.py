"""Anchor v0.9.75 — quality-anchor LLM router (formerly Fusio).

Quality-anchor LLM router, 15 workers configured / single product. Inspired by Sakana Fugu
(arXiv:2606.21228), TRINITY (arXiv:2512.04695), and Conductor (arXiv:2512.04388).
Licensed under Apache-2.0 (see LICENSE).

Quick start::

    from anchor import config
    print(config.active_n_agents())      # 4 enabled workers by default
    for w in config.enabled_workers():
        print(w.name, w.cost_in)

The router head (TrinityHead, 50 LOC NumPy) will live in :mod:`anchor.head` (Day 2).
Worker clients live in :mod:`anchor.clients`.
"""
__version__ = "0.9.75"
__all__ = ["config"]

import os as _os
from pathlib import Path as _Path
_ROOT = _Path(_os.environ.get("ANCHOR_ROOT", _Path(__file__).resolve().parents[2]))
