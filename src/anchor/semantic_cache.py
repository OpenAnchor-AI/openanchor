"""Semantic cache for LLM responses using embedding similarity.

Caches responses for semantically similar queries to avoid redundant API
calls. Uses Ollama qwen3-embedding for text embeddings and an in-memory
numpy-based vector store (always available). Optionally supports pgvector
for persistent storage when psycopg2 is installed.

Enable with:
    ANCHOR_SEMANTIC_CACHE_ENABLED=1
    ANCHOR_SEMANTIC_CACHE_THRESHOLD=0.92   # cosine similarity (0-1)
    ANCHOR_SEMANTIC_CACHE_TTL=3600         # seconds
"""
from __future__ import annotations

import os
import json
import time
import hashlib
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


_ENABLED: bool = os.environ.get("ANCHOR_SEMANTIC_CACHE_ENABLED", "0") == "1"
# After t_7b06ca78 (Advisor Directive · 2026-08-29): defaults point at aimax farm
# 10.0.0.4:11434 (wired via WG utun11). Override to a same-host loopback by exporting
# the env vars when running anchor on the same box as the ollama daemon.
_EMBEDDING_MODEL: str = os.environ.get(
    "ANCHOR_SEMANTIC_CACHE_EMBEDDING_MODEL", "qwen3-embed-8b:latest"
)
_EMBEDDING_URL: str = os.environ.get(
    "ANCHOR_SEMANTIC_CACHE_EMBEDDING_URL",
    "http://10.0.0.4:11434/api/embeddings",
)
_SIMILARITY_THRESHOLD: float = float(os.environ.get("ANCHOR_SEMANTIC_CACHE_THRESHOLD", "0.92"))
_TTL_S: int = int(os.environ.get("ANCHOR_SEMANTIC_CACHE_TTL", "3600"))


def _query_key(system_prompt: str, user_query: str, tools_json: str = "",
               salt: str = "") -> str:
    """Build a stable hash from query components for exact-match lookup.

    audit 2026-08-16 (D8): an optional per-user/per-session salt keeps the
    cache from cross-serving one user's private-data-laden answer to another
    user with a semantically similar query. Defaults to "" for callers that
    explicitly want a shared cache.
    """
    raw = f"{salt}||{system_prompt}||{user_query}||{tools_json}"
    return hashlib.sha256(raw.encode()).hexdigest()


async def get_embedding(text: str) -> list[float] | None:
    """Call Ollama embedding API. Returns embedding vector or None on failure."""
    if not text.strip():
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(_EMBEDDING_URL, json={
                "model": _EMBEDDING_MODEL,
                "prompt": text,
            })
            resp.raise_for_status()
            data = resp.json()
            return data.get("embedding")
    except Exception as exc:
        logger.warning("SEMANTIC_CACHE embedding error: %s", exc)
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import numpy as _np
    a_np = _np.array(a, dtype=_np.float64)
    b_np = _np.array(b, dtype=_np.float64)
    norm_a = _np.linalg.norm(a_np)
    norm_b = _np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(_np.dot(a_np, b_np) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# In-memory vector store (always available)
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """Numpy-backed in-memory vector store. Entries expire after TTL."""

    def __init__(self, threshold: float = 0.92, ttl_s: int = 3600):
        self._threshold = threshold
        self._ttl_s = ttl_s
        self._entries: list[dict] = []
        self._exact: dict[str, dict] = {}

    def search(self, embedding: list[float]) -> dict | None:
        """Return best matching entry above threshold, or None."""
        best_score = 0.0
        best_entry: dict | None = None
        now = time.time()
        for entry in self._entries:
            if now - entry["ts"] > self._ttl_s:
                continue
            score = _cosine_similarity(embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score >= self._threshold:
            logger.info("SEMANTIC_CACHE hit score=%.4f", best_score)
            return {"response": best_entry["response"], "model_used": best_entry["model_used"],
                    "similarity": best_score}
        return None

    def exact_lookup(self, key: str) -> dict | None:
        """Exact hash lookup (bypasses similarity search)."""
        entry = self._exact.get(key)
        if entry and time.time() - entry["ts"] <= self._ttl_s:
            return {"response": entry["response"], "model_used": entry["model_used"], "similarity": 1.0}
        return None

    def store(self, embedding: list[float], query_key: str, system_prompt: str,
              user_query: str, response: str, model_used: str, tools_hash: str = "") -> None:
        now = time.time()
        entry = {
            "embedding": embedding,
            "query_key": query_key,
            "system_prompt": system_prompt,
            "user_query": user_query,
            "response": response,
            "model_used": model_used,
            "tools_hash": tools_hash,
            "ts": now,
        }
        self._entries.append(entry)
        self._exact[query_key] = entry
        # Simple GC: prune expired every 100 stores
        if len(self._entries) % 100 == 0:
            self._prune()

    def _prune(self) -> None:
        now = time.time()
        self._entries = [e for e in self._entries if now - e["ts"] <= self._ttl_s]
        self._exact = {k: v for k, v in self._exact.items()
                       if now - v["ts"] <= self._ttl_s}

    def clear(self) -> None:
        self._entries.clear()
        self._exact.clear()

    def __len__(self) -> int:
        self._prune()
        return len(self._entries)


# ---------------------------------------------------------------------------
# pgvector store (optional, requires psycopg2)
# ---------------------------------------------------------------------------

try:
    import psycopg2
    import psycopg2.extras
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False
    psycopg2 = None  # type: ignore


def _pgvector_ddl() -> str:
    return """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS anchor_semantic_cache (
        id SERIAL PRIMARY KEY,
        embedding vector(1024),
        query_key VARCHAR(64) UNIQUE,
        system_prompt TEXT DEFAULT '',
        user_query TEXT NOT NULL,
        response TEXT NOT NULL,
        model_used TEXT NOT NULL,
        tools_hash TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_anchor_semantic_cache_key
        ON anchor_semantic_cache (query_key);
    """


class PgVectorStore:
    """pgvector-backed persistent vector store."""

    def __init__(self, dsn: str, threshold: float = 0.92, ttl_s: int = 3600):
        if not _HAS_PGVECTOR:
            raise RuntimeError("psycopg2 not installed; cannot use PgVectorStore")
        self._dsn = dsn
        self._threshold = threshold
        self._ttl_s = ttl_s
        self._conn: Any = None

    def _ensure_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            with self._conn.cursor() as cur:
                cur.execute(_pgvector_ddl())
            self._conn.commit()

    def search(self, embedding: list[float]) -> dict | None:
        self._ensure_conn()
        vec = "[" + ",".join(str(v) for v in embedding) + "]"
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT response, model_used,
                          1 - (embedding <=> %s::vector) AS similarity
                   FROM anchor_semantic_cache
                   WHERE created_at > NOW() - INTERVAL '%s seconds'
                   ORDER BY embedding <=> %s::vector
                   LIMIT 1""",
                (vec, int(self._ttl_s), vec),
            )
            row = cur.fetchone()
        if row and row[2] >= self._threshold:
            logger.info("SEMANTIC_CACHE pgvector hit score=%.4f", row[2])
            return {"response": row[0], "model_used": row[1], "similarity": row[2]}
        return None

    def exact_lookup(self, key: str) -> dict | None:
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT response, model_used FROM anchor_semantic_cache
                   WHERE query_key = %s AND created_at > NOW() - INTERVAL '%s seconds'""",
                (key, int(self._ttl_s)),
            )
            row = cur.fetchone()
        if row:
            return {"response": row[0], "model_used": row[1], "similarity": 1.0}
        return None

    def store(self, embedding: list[float], query_key: str, system_prompt: str,
              user_query: str, response: str, model_used: str, tools_hash: str = "") -> None:
        self._ensure_conn()
        vec = "[" + ",".join(str(v) for v in embedding) + "]"
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO anchor_semantic_cache
                   (embedding, query_key, system_prompt, user_query, response, model_used, tools_hash)
                   VALUES (%s::vector, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (query_key) DO NOTHING""",
                (vec, query_key, system_prompt, user_query, response, model_used, tools_hash),
            )
        self._conn.commit()

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()


# ---------------------------------------------------------------------------
# Global cache instance
# ---------------------------------------------------------------------------

_CACHE: InMemoryVectorStore | PgVectorStore | None = None


def get_cache() -> InMemoryVectorStore | PgVectorStore | None:
    """Return the global cache instance, creating it on first call."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not _ENABLED:
        return None
    dsn = os.environ.get("ANCHOR_SEMANTIC_CACHE_DB_URL", "")
    if dsn and _HAS_PGVECTOR:
        logger.info("SEMANTIC_CACHE using pgvector: %s", dsn.split("@")[-1] if "@" in dsn else dsn)
        _CACHE = PgVectorStore(dsn=dsn, threshold=_SIMILARITY_THRESHOLD, ttl_s=_TTL_S)
    else:
        logger.info("SEMANTIC_CACHE using in-memory store (threshold=%.2f, ttl=%ds)",
                    _SIMILARITY_THRESHOLD, _TTL_S)
        _CACHE = InMemoryVectorStore(threshold=_SIMILARITY_THRESHOLD, ttl_s=_TTL_S)
    return _CACHE


async def lookup(system_prompt: str, user_query: str, tools: list | None = None,
                 salt: str = "") -> dict | None:
    """Lookup cache. Returns {response, model_used, similarity} or None."""
    cache = get_cache()
    if cache is None:
        return None
    tools_json = json.dumps(tools or [], sort_keys=True)
    key = _query_key(system_prompt, user_query, tools_json, salt=salt)
    exact = cache.exact_lookup(key)
    if exact:
        return exact
    text = f"{system_prompt}\n{user_query}".strip()
    embedding = await get_embedding(text)
    if embedding is None:
        return None
    return cache.search(embedding)


async def store(system_prompt: str, user_query: str, response: str,
                model_used: str, tools: list | None = None,
                salt: str = "") -> None:
    """Store a response in the cache.

    audit 2026-08-16 (D8): refuse to cache junk — empty, error, stub or
    placeholder responses (e.g. stored during an upstream outage) would
    otherwise poison the cache for its TTL.
    """
    cache = get_cache()
    if cache is None:
        return
    _r = (response or "").strip()
    if not _r:
        return
    _r_low = _r.lower()
    if _r.startswith("[error:") or _r.startswith("[stub:"):
        return
    for _marker in ("sorry, i cannot", "i’m sorry", "i'm sorry"):
        if _r_low.startswith(_marker) or (len(_r) < 60 and _marker in _r_low):
            logger.warning("SEMANTIC_CACHE store rejected placeholder (len=%d)", len(_r))
            return
    tools_json = json.dumps(tools or [], sort_keys=True)
    key = _query_key(system_prompt, user_query, tools_json, salt=salt)
    text = f"{system_prompt}\n{user_query}".strip()
    embedding = await get_embedding(text)
    if embedding is None:
        return
    cache.store(embedding, key, system_prompt, user_query, _r, model_used, tools_json)


def enabled() -> bool:
    return _ENABLED
