# Anchor — Agent Dispatch Rules

## Communication
- 中文优先（除非用户用英文）
- 简洁直接, no fluff
- 代码注释保持英文

## Project Rules

### Naming
- Package name: `anchor` (lowercase, no hyphen)
- CLI: `anchor`
- API path: `POST /v1/chat/completions {"model": "anchor"}` (single product, no tier SKU)
- Python modules: `anchor.config`, `anchor.server`, `anchor.clients`, `anchor.fusion_modes`, `anchor.head`

### File layout
- All code under `src/anchor/`
- Tests under `tests/` (mirror `src/anchor/` layout)

### Worker pool
- Workers declared in `src/anchor/config.py:WORKERS` (single source of truth)
- `enabled=False` means blocked / env-gated / opt-in
- Fallback chains: `WORKER_FALLBACK_CHAIN` (runtime) + `FALLBACK_CHAIN` (role expansion)
- All OpenAI-compat workers share `OpenAICompatClient`
- Public product: only `model: "anchor"` (+ `anchor-image`)
- Add a worker = append to `WORKERS` + update `src/anchor/workers.py` constants

### Quality bar
- Quality **and** cost are tracked Pareto-style; never optimize one without the other
- M3 is the **oracle** for the training head (always)
- head weights ship via git lfs

### Validation (every PR)
```bash
PYTHONPATH=src python -c "import anchor"
PYTHONPATH=src python -m anchor.config
PYTHONPATH=src pytest -q
```

### Anti-patterns
- ❌ Don't break import order to silence linters; fix the import
- ❌ Don't commit `.env` or any populated secret file
- ❌ Don't inline ≥100 LOC — write a file
- ❌ Don't add worker aliases to the public product surface
