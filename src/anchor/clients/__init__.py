"""Worker clients for Anchor.

Each client implements the same interface::

    async def chat(messages, **kwargs) -> dict   # returns OpenAI-format dict
    async def stream(messages, **kwargs) -> AsyncIterator[str]

Two kinds of clients:
  - OpenAICompatClient (base, ~40 LOC) — for any OpenAI-compatible provider
  - OpenCodeCLIClient (~30 LOC) — for OpenCode Zen models (CLI spawn)

The factory in :mod:`anchor.clients.factory` builds the right client from
a :class:`anchor.config.Worker` definition.
"""
