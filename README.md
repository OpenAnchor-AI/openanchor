# Anchor v0.9.75 (formerly Fusio)

> **The quality-anchor LLM router.** Single product `anchor` — auto multi-LLM routing.
> Cheaper than the strongest single. Smarter than the sum of its parts.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()
[![Internal-ready](https://img.shields.io/badge/internal--ready-2026--07--14-green)](docs/AUDIT-3RDPARTY.md)

Inspired by [Sakana Fugu](https://arxiv.org/abs/2606.21228), [TRINITY](https://arxiv.org/abs/2512.04695),
and [Conductor](https://arxiv.org/abs/2512.04388). Independent open-source reimplementation.

## Recent changes

### v0.9.68 — deepseek-v4-flash forces stream=True (63.6% empty rate)
- **Audit root cause**: P2b audit showed 5515/8678 = 63.6% of `deepseek-v4-flash` calls return empty content (error_kind=empty). OpenCode Zen endpoint only returns the answer in streaming mode for this worker — non-stream `chat()` has 65.4% empty rate while `stream()` has 0% empty. Trivial queries (<20 chars) are immune because they slip past the reasoning budget.
- **Fix**: `src/anchor/routing_core.py` `_call_worker` now forces `client.stream()` (and aggregates deltas) when `worker_name == "deepseek-v4-flash"`. All other workers (M3, Sol, Fable, dpsk-flash-official, ...) keep their non-stream `client.chat()` path unchanged.
- **Expected impact**: 63.6% empty → ~5% empty (matching trivial-query baseline). Saves ~5000 unnecessary M3 fallbacks over a 30d window at current traffic.
- **Tests**: `tests/test_dpsk_flash_stream.py` — 8 cases pinning the stream dispatch for dpsk-flash and the unchanged chat() path for M3/Sol/Fable/dpsk-flash-official.

### v0.9.67 — FABLE5_TRAFFIC_PCT default 50 to 25 (P3 audit follow-up)
- **Cost compression**: `FABLE5_TRAFFIC_PCT` default reduced from 50% to 25% (v0.9.66 P3 follow-up). Expected cost ~¥300/mo vs ¥516 at 50% (vs ¥1195/mo baseline at 100%). Sacred fable-5 top remains as rare escape hatch.
- **Tests**: `tests/test_fable5_rollout_default.py` — 7 cases including `test_env_override_25`.

### v0.9.66 — FABLE5_TRAFFIC_PCT default 100 to 50 (P3 audit)
- **Cost reduction**: `FABLE5_TRAFFIC_PCT` default reduced from 100% to 50% based on P3 audit. 30d data: fable-5 = 2.7% of calls but 72.4% of cost (7.3× sol per call); 78% of fable calls are fallback_from; queries SHORTER than sol (median 18 vs 137 chars). Saves ~516 yuan/mo while keeping +0.046 judge edge.
- **Override**: `FABLE5_TRAFFIC_PCT=100` env var restores full rollout; `/admin/gradual/fable5` endpoint unchanged.
- **Tests**: `tests/test_fable5_rollout_default.py` — 6 cases with autouse fixture for live state isolation.

### v0.9.63 — Design B fallback chains + Claude-group auth gate
- **Design B chains**: 14 `WORKER_FALLBACK_CHAIN` entries rewritten as a strict tier-DAG (T4 cheapest → T3 → T2 → T1 → T0 terminal). No cycles. `_validate_fallback_graph()` enforces.
- **Claude auth gate**: new `_apply_claude_auth_gate()` auto-disables `claude-fable-5`/`opus-5`/`haiku-4-5`/`sonnet-5` when `ANCHOR_BAOSIAPI_CLAUDE_API_KEY` is unset (verified: baosiapi GPT-group key returns 401 on Claude models; baosiapi splits GPT/Claude into separate sub-groups).
- **Probe**: dpsk-flash-official n=30 live placeholder_rate=0.0% (wider confirmation; n=10 was already 0.0%). File: `data/probe_deepseek-v4-flash-official_results.json`.
- **Tests**: 1102 passed / 5 skipped / 0 failed.
- **Migration**: To re-enable Claude workers, set `ANCHOR_BAOSIAPI_CLAUDE_API_KEY=sk-baosi-claude-...`.

### v0.9.62 — Fallback chain redesign (dpsk-flash-official default ON)
- **NEW worker**: deepseek-v4-flash-official (slot 16, official DeepSeek paid API). Default ON. Pricing ¥0.27/M input, ¥1.10/M output (¥0.039/M flat — 6× cheaper than dpsk-pro, 15× cheaper than pro).
- **Probe**: Live n=10 placeholder_rate=0.0%, recommendation=enable. File: data/probe_deepseek-v4-flash-official_results.json.
- **Fallback chain**: 7 chains updated to use dpsk-flash-official as last-resort fallback (replaces legacy dpsk-pro in non-m3 chains).
- **Kill switch**: ANCHOR_DISABLE_DEEPSEEK_V4_FLASH_OFFICIAL=1 to opt out.
- **Migration**: v0.9.62 means production traffic flows through dpsk-flash-official when reaching last-resort fallback. Requires DEEPSEEK_API_KEY.

## What is Anchor

A pure **quality / cost / stability router** — not a billing platform.

One OpenAI-compatible product: `model: "anchor"`. Each query goes to the
best worker in the lean pool (flash / M3 / optional pro-fallback / sol /
fable / optional grok). Quality floor + cost discipline via a learned
5-dim W head (TRINITY) on Qwen3-0.6B (sep-CMA-ES).

```
POST /v1/chat/completions
{"model": "anchor", "messages": [...]}
```

**Pure mode (default, self-use):** minimal auth (`ANCHOR_API_KEYS`) + usage
logs; client `daily_tokens=0` (unlimited). No multi-tenant billing stack.

**Commercial / multi-tenant:** put [OneAPI](https://github.com/songquanpeng/one-api)
(or similar) in front of Anchor for keys, quotas, and metering. Anchor stays
the router.

Legacy `anchor-basic|premium|ultra` names and `/basic|/premium|/ultra` routes
are **removed** — use `model: "anchor"` only.

## Worker Pool (v0.9.62, 15 configured / 9 enabled default)

> Phase 2 B3 alignment: the docstring in `src/anchor/config.py` (grep
> `default enabled` >= 1) and this table are the single source of truth for
> default `enabled=` state. Verify with
> `PYTHONPATH=src python3 -c "from anchor.config import WORKERS; print(sorted([w.name for w in WORKERS if w.enabled]))"`.

| Worker | Yuan/M in | Yuan/M out | Roles | Default | Opt-in via |
|--------|----------:|-----------:|-------|:-------:|------------|
| minimax-m3 | 0.00 | 0.00 | cn-agent, multi, mid-en, hard | yes | (always; sunk 119/mo subscription) |
| deepseek-v4-flash | 0.00 | 0.00 | coding, light | yes | (always; OpenCode Zen free) |
| gpt-5.6-luna | 0.09 | 0.54 | code, cn, mid-en | yes | (always; v1.0.1 calibration) |
| grok-4-6-reasoning | 0.90 | 2.71 | reasoning, hard, mid-en | yes | (always; v1.0.1 calibration) |
| claude-sonnet-5 | 0.95 | 4.75 | mid-en, hard, coding | yes | (always; v1.0.1 calibration) |
| gpt-5.6-terra | 0.90 | 5.43 | reasoning, hard, long-context | yes | (always; v1.0.1 calibration) |
| kimi-k3 | 1.36 | 6.78 | cn, multi, vision, long-context | yes | (always; v1.0.1 default ON) |
| claude-opus-5 | 1.58 | 7.92 | hard | yes | (always; B6 audit 2026-08-15) |
| deepseek-v4-flash-official | 3.00 | 9.00 | coding, mid-en, fast | yes | (always; dpsk paid; v0.9.62 probe n=10 0%) |
| claude-haiku-4-5 | 0.23 | 1.13 | light, mid-en | opt-in | `ANCHOR_ENABLE_HAIKU=1` |
| grok-4-5-reasoning | 0.90 | 2.71 | reasoning, hard, mid-en | opt-in | `ANCHOR_ENABLE_GROK4=1` (4-6 supersedes) |
| gpt-5.6-sol | 2.26 | 13.57 | coding, mid-en, hard, agent | opt-in | (Phase 3 FIX-E HOLD, 11% err) |
| claude-fable-5 | 3.17 | 15.83 | code, hard, sacred | opt-in | `FABLE5_TRAFFIC_PCT=25` (gradual rollout) |
| deepseek-v4-pro | 9.00 | 27.00 | coding, mid-en, hard | opt-in | `ANCHOR_ENABLE_DEEPSEEK_PRO=1` (last-resort) |
| ollama-ornith-35b | 0.00 | 0.00 | coding, agent, reasoning, cn-agent, mid-en, local | opt-in | `ANCHOR_ENABLE_OLLAMA_ORNITH=1` |

**15 configured, 9 enabled default** (table order = cheapest in+out first, then mid, then premium, then opt-in).
**B6 audit 2026-08-15**: Opus-5 ¥1.58 < dpsk-pro ¥9, so dpsk-pro 默认关 (opt-in via `ANCHOR_ENABLE_DEEPSEEK_PRO=1`); Opus-5 sacred 兜底 T3 hard.

**Default enabled (assuming baosi
keys provisioned)**: `minimax-m3`, `deepseek-v4-flash`,
`deepseek-v4-flash-official`, `gpt-5.6-luna`, `gpt-5.6-terra`,
`grok-4-6-reasoning`, `claude-sonnet-5`, `claude-opus-5`, `kimi-k3`.
Phase 1 calibration needs the 9-enabled coverage for v8 table_coarse.
**Opt-in / env-gated**: `claude-haiku-4-5` (`ANCHOR_ENABLE_HAIKU=1`),
`grok-4-5-reasoning` (`ANCHOR_ENABLE_GROK4=1`, superseded by 4-6),
`gpt-5.6-sol` (Phase 3 FIX-E HOLD, 11% err), `claude-fable-5`
(`FABLE5_TRAFFIC_PCT=25` gradual rollout), `deepseek-v4-pro`
(`ANCHOR_ENABLE_DEEPSEEK_PRO=1`, last-resort insurance),
`ollama-ornith-35b` (`ANCHOR_ENABLE_OLLAMA_ORNITH=1`).

**Removed in v0.9.50-p0/p1**: Agnes 2.0 Flash (chat),
Gemini 2.5 Flash (chat + image-gen), Kilo auto-free, GLM-5.2 Pro,
Nemotron 3 Ultra. See CHANGELOG.md.
## Self-use (pi + gateway)

See [docs/RUNBOOK-SELFUSE.md](docs/RUNBOOK-SELFUSE.md) for LaunchAgent, pi wiring, and daily ops.

## Quick start

```bash
git clone https://github.com/anchor-llm/anchor
cd anchor
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -e ".[dev]"

# REQUIRED: export one or more Anchor gateway auth tokens before booting
# the server. See the Security section below for the rotation procedure.
export ANCHOR_API_KEYS="sk-anchor-$(openssl rand -hex 24)"

# Optional: provider keys (already env-var-only, see .env.example)
# export MINIMAX_API_KEY_1=sk-...
# export BAOSIAPI_API_KEY=sk-...           # Baosi Claude group (shared)
# export BAOSIAPI_GPT_API_KEY=sk-...      # Baosi GPT group (independent)
# export OPENCODE_ZEN_API_KEY=sk-...      # DeepSeek V4 Flash via Zen free
# export DEEPSEEK_API_KEY=sk-...          # DeepSeek V4 Pro (M3 fallback)
# ... (see data/security_audit.txt Category 1 for the full list)

# Probe workers
PYTHONPATH=src python -m anchor.config

# Run gateway
PYTHONPATH=src python -m anchor.server
```

## Project layout

```
anchor/
├── src/anchor/
│   ├── config.py          # 12-worker registry (4 enabled default), single product
│   ├── clients/           # OpenAI-compat, multi-key pool, OpenCode CLI
│   ├── fusion_modes.py    # tier routing rules + hard rules + cascade
│   ├── head.py            # TrinityHead 5-dim W (50 LOC NumPy)
│   ├── server.py          # FastAPI OpenAI-compat (/v1/chat/completions, /v1/images/...)
│   ├── admin_ops.py       # /admin/sla, /admin/cost/dashboard, /admin/cost/amortization
│   └── ...                # 30+ modules (calibration, drift, failover, judge, ...)
├── data/                  # session logs, head baselines, calibration tables
├── docs/                  # ARCHITECTURE, SLA, CHANGELOG, REPORT-*
├── evals/                 # phase2/3 reports, run.py, run_parallel.py
├── scripts/               # load_test, calibration, threshold extraction
└── tests/                 # 112+ pytest cases
```

## Key quota (multi-worker)

Pure mode default: **no daily token cap** (`daily_tokens=0`). RPM soft-limit
still applies (default 60). Optional caps via `ANCHOR_CLIENT_DAILY_TOKENS`.
See [docs/PURE_MODE.md](docs/PURE_MODE.md).

| Env | Default | Meaning |
|---|---|---|
| `ANCHOR_CLIENT_DAILY_TOKENS` | `0` | daily cap per client key; `0` = unlimited |
| `ANCHOR_CLIENT_RPM` | `60` | requests/minute soft limit |
| `ANCHOR_KEY_QUOTA_BACKEND` | auto | `memory` / `sqlite` / `redis` (auto → sqlite when WEB_CONCURRENCY>1) |
| `ANCHOR_KEY_QUOTA_DB` | `data/key_quota.db` | sqlite path |
| `ANCHOR_REDIS_URL` | — | redis URL when backend=redis |
| `ANCHOR_SACRED_BACKEND` | auto | sacred spend `memory` / `sqlite` (auto→sqlite when multi-worker) |
| `ANCHOR_SACRED_DB` | `data/sacred_spend.db` | sqlite path for sacred |
| `ANCHOR_READYZ_PROBE` | key | `key` / `http` / `off` readiness probe mode |

## Security

### Token contract (v0.10+, 2026-07-21)

Anchor gateway auth tokens are loaded **exclusively** from the
`ANCHOR_API_KEYS` environment variable. There is no file or JSON fallback.
The source of truth is `src/anchor/key_loader.py`; every other auth path
references it. Behavior:

| Scenario                                  | Result                                                  |
| ----------------------------------------- | ------------------------------------------------------- |
| `ANCHOR_API_KEYS` set, with N tokens      | N auth records; default rpm=60, daily_tokens=0 (unlimited) |
| `ANCHOR_API_KEYS` unset / empty           | Gateway enters **fail-closed** mode (zero tokens, all client requests rejected with 401 or 503) |
| `ANCHOR_AUTH_DISABLED=1` + `ANCHOR_PROFILE=debug` | Debug-only escape hatch for local testing (non-admin paths only) |

The legacy `ANCHOR_API_KEY` (singular) and `LITELLM_MASTER_KEY` env vars,
the per-account `~/anchor/.env` and `~/.litellm/keys.env` files, and the
`data/api_keys.json` lookup are **no longer honoured** (intentionally,
to remove the supply-chain risk of git-tracked auth tokens -- see
audit history commit `527306d`).

Upstream provider keys (`MINIMAX_API_KEY_1/2`,
`BAOSIAPI_API_KEY`, `BAOSIAPI_GPT_API_KEY`, `OPENCODE_ZEN_API_KEY`,
`DEEPSEEK_API_KEY`) have always been env-only;
see `.env.example` and `data/security_audit.txt` for the full list.

### Token rotation procedure

When (and when) to rotate:
  * On any suspected compromise (this migration is itself one such event).
  * On operator offboarding.
  * On a 90-day rolling cadence for tokens with human-facing blast radius
    (e.g. team-wide master keys).
  * Never casually -- rotation invalidates every deployed client.

Steps:

1. **Generate fresh tokens** in your secret manager (1Password CLI, AWS
   Secrets Manager, `openssl rand -hex 24` for a 48-char token, etc.).
   Each token is opaque to Anchor (we never decode it). Example:

       openssl rand -hex 24    # -> 48 hex chars, paste as one token

2. **Distribute to operators** out-of-band. Do not paste tokens in chat,
   tickets, or shell history. For machine-to-machine use prefer the
   platform's native secret store (Kubernetes Secret, Vault, SSM).

3. **Update `ANCHOR_API_KEYS`** in the deployment environment. Multiple
   tokens are comma-separated. Whitespace is ignored. Re-deploy the
   gateway pod / systemd unit / container -- the loader reads the env
   var at startup.

       # Single-token deploy
       export ANCHOR_API_KEYS="sk-anchor-NEW-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

       # Multi-token deploy (master + per-team)
       export ANCHOR_API_KEYS="sk-anchor-NEW-MASTER-...,sk-anchor-NEW-TEAM-A-..."

4. **Smoke-test** before announcing the rotation:

       PYTHONPATH=src python -m pytest tests/test_anchor_secrets.py -v
       PYTHONPATH=src python -m anchor.smoke     # boots :8088, hits /v1/chat/completions

5. **Revoke the old tokens** at the gateway by re-deploying without
   them (ANCHOR_API_KEYS no longer contains them). Existing client
   retries with the old token will start receiving 401.

6. **Audit trail**: log the rotation in your change-management system
   with the timestamp, the operator IDs, and the secret-manager
   reference for the new tokens. Anchor itself does not log token
   values (the auth middleware uses `secrets.compare_digest` and never
   surfaces the raw token in error messages).

7. **Client migration**: hand clients the new tokens via your usual
   distribution channel. The client's request shape is unchanged
   (`Authorization: Bearer <token>`).

### Recovering from a leaked token

  1. Treat the token as fully compromised (assume already-scraped from
     git history, CI logs, etc.).
  2. Generate a new token as in step 1 above.
  3. Update `ANCHOR_API_KEYS` *with both the new and old token* so
     any in-flight clients using the old token can re-authenticate.
  4. Roll the client rollout so all consumers pick up the new token.
  5. Re-deploy `ANCHOR_API_KEYS` containing only the new token.
  6. Save the old token to a deny-list audit table for compliance.

### `.env.example`

A template is provided at the repo root: `.env.example`. Copy to
`.env` (which is git-ignored) and fill in real values; do not commit
the populated `.env`.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgments

- Sakana AI for the Fugu/TRINITY/Conductor papers
- trotsky1997 for [OpenFugu](https://github.com/trotsky1997/OpenFugu) (open reverse-engineering)
- AI-Safeter for [FUGU](https://github.com/AI-Safeter/FUGU) (sep-CMA-ES reference)
- RouteWorks for [RouterArena](https://github.com/RouteWorks/RouterArena) (eval framework)

## Validation strategy

Quality floor is enforced via **pair-ACC monitoring** in the training loop
(`data/head_baseline.json`, weekly retrain). The `scripts/weekly_retrain.sh`
pipeline evaluates against a held-out set and logs quality regression.
See `docs/ARCHITECTURE.md` and `VALIDATION_LOG.md` (historical) for details.

## Routing (pure single product)

There is **one** chat product: `model: "anchor"` (synonym `anchor-auto`).
No sold tiers and no internal product lanes — the router alone picks workers
via hard-rules + Pareto (quality÷cost).

Configured in `src/anchor/config.py:WORKERS`. Hard rules in
`src/anchor/fusion_modes.py:TIER_HARD_RULES` (compat keys collapse to auto).

## Migration notes (legacy aliases)

Use `/v1/chat/completions` with:

```bash
# New (recommended) — model=anchor lets the router pick based on complexity
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Authorization: Bearer $ANCHOR_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"anchor","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'

# Or use existing tier aliases (router still decides — model is a hint)
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Authorization: Bearer $ANCHOR_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"anchor","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'

# Or bypass the router entirely with a direct worker name
curl -X POST http://localhost:8088/v1/chat/completions \
  -H "Authorization: Bearer $ANCHOR_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"code fibonacci"}],"max_tokens":100}'
```

Legacy endpoints return `Deprecation: true` + `Sunset` headers. Unknown model
names now return `400` (strict validation, no fallback).
