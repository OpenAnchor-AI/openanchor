"""Factory: build the right client from a Worker config."""
from __future__ import annotations
from anchor.config import Worker, _read_api_key
from .base import OpenAICompatClient
from .opencode import OpenCodeCLIClient
from .key_pool import KeyPool


def build_client(worker: Worker):
    if worker.kind == "openai-compat":
        env_keys = worker.api_key_envs or ((worker.api_key_env,) if worker.api_key_env else ())
        api_key = ""
        for envn in env_keys:
            api_key = _read_api_key(envn) or ""
            if api_key.strip():
                break
        client = OpenAICompatClient(
            base_url=worker.base_url,
            api_key=api_key,
            model=worker.model,
            name=worker.name,
            system_prompt=getattr(worker, "system_prompt", None),
        )
        client.channel = getattr(worker, "channel", None)  # v0.9.72: for vendor blacklist
        return client
    if worker.kind == "multi-key":
        # Round-robin or failover multi-key pool. Works for any OpenAI-compat endpoint.
        env_keys = worker.api_key_envs or (worker.api_key_env,)
        client = KeyPool(
            env_keys=env_keys,
            base_url=worker.base_url,
            model=worker.model,
            worker_name=worker.name,
            system_prompt=worker.system_prompt,
            key_strategy=getattr(worker, "key_strategy", "round-robin"),
        )
        client.channel = getattr(worker, "channel", None)  # v0.9.72: for vendor blacklist
        return client
    if worker.kind == "opencode-cli":
        return OpenCodeCLIClient(model=worker.model, name=worker.name)
    raise ValueError(f"unknown worker kind: {worker.kind!r}")
