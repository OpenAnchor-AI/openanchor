"""Public model IDs + aliases + tier pick (extracted from routing_core)."""
from __future__ import annotations

from fastapi import HTTPException

from anchor.fusion_modes import TIER_POOL_ORDER


# === Public product model IDs (listed by /v1/models) ===
PUBLIC_MODEL_IDS: tuple[str, ...] = (
    "anchor",
    "anchor-image",
)

_MODEL_TO_TIER = {
    "anchor": "auto",
    "anchor-auto": "auto",
    "anchor-image": "image-gen",
    "anchor/image": "image-gen",
}

# Maps model names to either:
#   - None: single-product router auto-pick
#   - str: direct worker name (ops/debug; gated)
# Pure mode: no basic/premium/ultra product aliases.
MODEL_ALIASES: dict[str, str | None] = {
    "anchor":           None,
    "anchor-auto":      None,
    "deepseek-v4-flash":   "deepseek-v4-flash",
    "minimax-m3":          "minimax-m3",
    "claude-sonnet-5":     "claude-sonnet-5",
    "claude-haiku-4-5":    "claude-haiku-4-5",
    "claude-opus-5":     "claude-opus-5",
    "deepseek-v4-pro":     "deepseek-v4-pro",
    "gpt-5.6-sol":         "gpt-5.6-sol",
    "claude-fable-5":      "claude-fable-5",
    "grok-4-6-reasoning":  "grok-4-6-reasoning",
    "gpt-5.6-luna":        "gpt-5.6-luna",
    "gpt-5.6-terra":       "gpt-5.6-terra",
    "kimi-k3":             "kimi-k3",
}

_REMOVED_TIER_SKUS = frozenset({
    "anchor-basic", "anchor-premium", "anchor-ultra",
    "anchor/basic", "anchor/premium", "anchor/ultra",
    "basic", "premium", "ultra",
})

_WORKER_TO_TIER: dict[str, str] = {}
for _wt_tier, _wt_workers in TIER_POOL_ORDER.items():
    for _wt_w in _wt_workers:
        if _wt_w not in _WORKER_TO_TIER:
            _WORKER_TO_TIER[_wt_w] = _wt_tier


def _pick_tier_from_model(model: str, last_user_msg: str) -> tuple[str | None, str]:
    if model in _REMOVED_TIER_SKUS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Product tier model '{model}' was removed. "
                f"Use public model 'anchor' (router auto-pick)."
            ),
        )
    direct_worker = MODEL_ALIASES.get(model)
    if direct_worker is not None:
        from anchor.worker_gate import allow_direct_worker_alias
        if not allow_direct_worker_alias():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Direct worker alias '{model}' is disabled in this environment. "
                    f"Use public model 'anchor' (or set ANCHOR_ALLOW_DIRECT_WORKER=1)."
                ),
            )
        from anchor.config import WORKERS as _W_ALIAS
        _w = next((x for x in _W_ALIAS if x.name == direct_worker), None)
        if _w is None or not _w.enabled:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Worker '{direct_worker}' is not enabled. "
                    f"Enable via the corresponding ANCHOR_ENABLE_* env var if available."
                ),
            )
        wt = _WORKER_TO_TIER.get(direct_worker, "auto")
        return (direct_worker, wt)
    if model in MODEL_ALIASES:
        from anchor.classifier import classify_to_tier, classify as _cls
        qt = _cls(last_user_msg) if last_user_msg else "chat"
        return (None, classify_to_tier(qt))
    t = _MODEL_TO_TIER.get(model)
    if t is not None:
        if t == "auto":
            from anchor.classifier import classify_to_tier, classify as _cls
            qt = _cls(last_user_msg) if last_user_msg else "chat"
            return (None, classify_to_tier(qt))
        return (None, t)
    public = ", ".join(PUBLIC_MODEL_IDS)
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unknown model: '{model}'. Public product: {public}. "
            f"Direct worker aliases accepted when enabled."
        ),
    )
