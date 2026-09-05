# Anchor — quality-anchor LLM router

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-100%25%20passing-brightgreen)]()

> Single product `anchor` — auto multi-LLM routing.
> Cheaper than the strongest single. Smarter than the sum of its parts.

Inspired by [Sakana Fugu](https://arxiv.org/abs/2606.21228), [TRINITY](https://arxiv.org/abs/2512.04695),
and [Conductor](https://arxiv.org/abs/2512.04388). Independent open-source reimplementation.

## What it does

A pure **quality / cost / stability router** for LLM calls. One OpenAI-compatible
endpoint exposes `model: "anchor"`; the router picks a worker from a lean pool
(cheap flash model as primary, M3 as oracle, optional pro/sol/fable) based on
query complexity, Pareto quality/cost, and live circuit-breaker state.

## Install

```bash
git clone https://github.com/OpenAnchor-AI/openanchor.git
cd openanchor
pip install -e ".[dev]"
cp .env.example .env   # then edit with your keys
```

## Quick start

```bash
# Run the gateway on :8088
PYTHONPATH=src python -m uvicorn anchor.server:app --host 0.0.0.0 --port 8088

# Call it
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Authorization: Bearer ${ANCHOR_API_KEYS%%,*}" \
  -H 'Content-Type: application/json' \
  -d '{"model":"anchor","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'
```

## Tests

```bash
PYTHONPATH=src pytest -q
```

The suite ships hermetic — no network, no real keys. All upstream calls are
faked behind `anchor.clients.factory` (see `tests/test_*`).

## Configuration

All knobs live in `.env` (see `.env.example`). Workers are declared in
`src/anchor/config.py:WORKERS`. Hard fallback rules in
`src/anchor/fusion_modes.py:TIER_HARD_RULES`. The router never calls a worker
whose required API key env-var is unset — it auto-disables it at boot
(see `_apply_claude_auth_gate()` for the canonical example).

## Architecture

- `src/anchor/server.py` — FastAPI gateway, OpenAI-compatible surface
- `src/anchor/head.py` — Trinity head (50-LOC NumPy Pareto router)
- `src/anchor/routing_core.py` — per-request routing loop with stream fallback
- `src/anchor/config.py` — worker pool, env knobs, fallbacks
- `src/anchor/fusion_modes.py` — hard rules + tier-DAG enforcement
- `src/anchor/release/` — circuit breaker + rollout gates

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

- Sakana AI — Fugu / TRINITY / Conductor papers
- trotsky1997 — [OpenFugu](https://github.com/trotsky1997/OpenFugu)
- RouteWorks — [RouterArena](https://github.com/RouteWorks/RouterArena)
