"""pytest bootstrap for Anchor core tests.

Goals (per pytest.ini addopts):
  * make `import anchor` resolve to src/anchor without a venv install
  * keep ANCHOR_ROOT / ANCHOR_DB / prompt-cache env vars from leaking in

We don't install the package — we just prepend src/ to sys.path so each
test module can `from anchor import ...` like production code does.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Anchor ships its code at src/anchor. The pyproject.toml also pins
# pythonpath = ["src"], but the venv created for this run may not have
# the package installed (tests use a throwaway .venv-test/). Prepend
# explicitly so import resolution is deterministic.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Wipe env-driven state that ships from CI shells so tests are
# reproducible. The Anchor modules re-read these at call time, so the
# wipe must happen before any import.
for k in (
    "ANCHOR_PROMPT_CACHE_ENABLED",
    "ANCHOR_PROMPT_CACHE_VENDOR_BLACKLIST",
    "ANCHOR_BAOSI_SHUNT_THRESHOLD",
    "ANCHOR_ROOT",
    "ANCHOR_DB",
):
    os.environ.pop(k, None)