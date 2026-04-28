"""All non-Postgres ML lives here. Models load lazily on first call."""
from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from pgkg.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Proposition extraction prompt version.
# BUMP THIS whenever the extraction prompt changes — it invalidates the cache
# for all cached entries that used the old prompt.
# ---------------------------------------------------------------------------
PROMPT_VERSION = "v1"

# ---------------------------------------------------------------------------
# Cache protocol — keeps ml.py free of asyncpg imports.
# Implementations live in memory.py (PostgresExtractCache).
# ---------------------------------------------------------------------------

@runtime_checkable
class ExtractCache(Protocol):
    async def get(self, cache_key: str) -> "list[Proposition] | None": ...
    async def put(
        self,
        cache_key: str,
        chunk_hash: str,
        extractor_model: str,
        prompt_version: str,
        props: "list[Proposition]",
    ) -> None: ...


def compute_cache_key(chunk_text: str, extractor_model: str) -> str:
    """Deterministic cache key for (chunk_text, extractor_model, PROMPT_VERSION)."""
    raw = f"{chunk_text}\x9f{extractor_model}\x9f{PROMPT_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Lazy model singletons
# ---------------------------------------------------------------------------

_embed_model: "SentenceTransformer | None" = None
_rerank_model: "CrossEncoder | None" = None


def _get_embed_model() -> "SentenceTransformer":
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        _embed_model = SentenceTransformer(get_settings().embed_model)
    return _embed_model


def _get_rerank_model() -> "CrossEncoder":
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        _rerank_model = CrossEncoder(get_settings().rerank_model)
    return _rerank_model


def is_embed_loaded() -> bool:
    return _embed_model is not None


def is_rerank_loaded() -> bool:
    return _rerank_model is not None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using the configured model. Returns L2-normalized vectors."""
    if not texts:
        return []
    model = _get_embed_model()
    vecs = np.array(model.encode(texts, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float64)
    # Normalize explicitly so L2 norm = 1 regardless of model flags
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms < 1e-10, 1.0, norms)
    return vecs.tolist()


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def rerank(query: str, docs: list[str]) -> list[float]:
    """Score each doc against query. Returns relevance score per doc."""
    if not docs:
        return []
    model = _get_rerank_model()
    pairs = [[query, doc] for doc in docs]
    scores = model.predict(pairs)
    if hasattr(scores, "tolist"):
        return scores.tolist()
    return list(scores)


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance) — pure numpy
# ---------------------------------------------------------------------------

def mmr(
    query_emb: list[float],
    doc_embs: list[list[float]],
    k: int,
    lambda_: float = 0.5,
) -> list[int]:
    """Greedy MMR selection. Returns indices of selected docs in selection order."""
    if not doc_embs:
        return []

    q = np.array(query_emb, dtype=np.float64)
    docs = np.array(doc_embs, dtype=np.float64)
    n = len(docs)
    k = min(k, n)

    # Normalize for cosine similarity
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    doc_norms = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-10)

    query_sims = doc_norms @ q_norm  # shape (n,)

    selected: list[int] = []
    remaining = list(range(n))

    for _ in range(k):
        if not remaining:
            break
        if not selected:
            # First pick: highest query similarity
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            sel_vecs = doc_norms[selected]  # shape (m, d)
            best_score = float("-inf")
            best = remaining[0]
            for i in remaining:
                rel = lambda_ * query_sims[i]
                red = (1 - lambda_) * float(np.max(sel_vecs @ doc_norms[i]))
                score = rel - red
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(best)
        remaining.remove(best)

    return selected


# ---------------------------------------------------------------------------
# Proposition extraction
# ---------------------------------------------------------------------------

class Proposition(BaseModel):
    text: str
    subject: str
    predicate: str
    object: str
    object_is_literal: bool = False


async def _run_cache_get(cache: ExtractCache, cache_key: str) -> "list[Proposition] | None":
    return await cache.get(cache_key)


async def _run_cache_put(
    cache: ExtractCache,
    cache_key: str,
    chunk_hash: str,
    extractor_model: str,
    prompt_version: str,
    props: "list[Proposition]",
) -> None:
    await cache.put(cache_key, chunk_hash, extractor_model, prompt_version, props)


def extract_propositions(
    chunk_text: str,
    *,
    max_propositions: int = 20,
    cache: "ExtractCache | None" = None,
) -> "list[Proposition]":
    """Extract atomic propositions from chunk text using an LLM.

    When ``cache`` is provided and a cache hit is found, the LLM is not called.
    The offline-extract path (PGKG_OFFLINE_EXTRACT=1) bypasses the cache entirely
    so test/stub data never pollutes the production cache.
    """
    import asyncio  # noqa: PLC0415

    # Offline fallback for tests — do NOT touch the cache (keep prod cache clean).
    if os.environ.get("PGKG_OFFLINE_EXTRACT", "0") == "1":
        return [
            Proposition(
                text=chunk_text[:120],
                subject="?",
                predicate="states",
                object=chunk_text[:120],
                object_is_literal=True,
            )
        ]

    settings = get_settings()
    extractor_model = settings.extractor_model or settings.llm_model

    # Cache lookup (async protocol bridged via asyncio.run or existing loop)
    if cache is not None:
        cache_key = compute_cache_key(chunk_text, extractor_model)
        chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside an async context — callers should prefer the async path.
            # For now fall through to sync LLM call (cache will be used by async wrappers).
            # NOTE: if you want full async cache, call _extract_propositions_async() instead.
            cached = None
        else:
            cached = asyncio.run(_run_cache_get(cache, cache_key))

        if cached is not None:
            return cached

    system_prompt = (
        "You are a knowledge extraction assistant. Extract distinct atomic facts "
        "from the provided text as subject-predicate-object propositions. "
        "Rules:\n"
        "- Each proposition must be self-contained and atomic (one fact per proposition).\n"
        "- Resolve pronouns when context allows (use the actual entity name).\n"
        "- Set object_is_literal=true for non-entity objects: numbers, dates, "
        "  freeform strings, measurements, quotes.\n"
        "- Keep subject and object as concise noun phrases.\n"
        f"- Extract at most {max_propositions} propositions.\n"
        "Return a JSON object with a 'propositions' array."
    )

    prop_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "object_is_literal": {"type": "boolean"},
        },
        "required": ["text", "subject", "predicate", "object", "object_is_literal"],
        "additionalProperties": False,
    }

    if settings.llm_provider == "openai":
        props = _extract_openai(chunk_text, system_prompt, prop_schema, max_propositions, settings, extractor_model)
    elif settings.llm_provider == "anthropic":
        props = _extract_anthropic(chunk_text, system_prompt, prop_schema, max_propositions, settings, extractor_model)
    elif settings.llm_provider == "claude_code":
        import asyncio as _asyncio  # noqa: PLC0415
        props = _asyncio.run(
            _extract_claude_code(
                chunk_text,
                extractor_model=extractor_model,
                max_propositions=max_propositions,
                system_prompt=system_prompt,
            )
        )
    else:
        props = _extract_ollama(chunk_text, system_prompt, max_propositions, settings, extractor_model)

    # Populate cache after successful LLM call (sync path only)
    if cache is not None and loop is None:
        asyncio.run(_run_cache_put(cache, cache_key, chunk_hash, extractor_model, PROMPT_VERSION, props))

    return props


def _parse_propositions_json(text: str) -> "list[Proposition]":
    """Parse a JSON string containing a 'propositions' array into Proposition objects."""
    import json  # noqa: PLC0415
    data = json.loads(text)
    return [Proposition(**p) for p in data.get("propositions", [])]


async def _extract_claude_code(
    chunk_text: str,
    *,
    extractor_model: str = "claude-haiku-4-5-20251001",
    max_propositions: int = 20,
    system_prompt: str,
) -> "list[Proposition]":
    """Extract propositions via the local claude CLI using claude-agent-sdk.

    This is for local experimentation only. It requires the 'claude' CLI to be
    installed and logged in. NOT suitable for benchmark runs (rate limits + ToS).
    """
    try:
        import claude_agent_sdk  # noqa: PLC0415
        from claude_agent_sdk import ClaudeAgentOptions  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "claude_code provider requires 'claude-agent-sdk' to be installed. "
            "Run: uv sync --extra claude_agent"
        ) from exc

    user_prompt = (
        f"{chunk_text}\n\n"
        f"Return a JSON object with a 'propositions' array. "
        f"Extract at most {max_propositions} propositions."
    )

    accumulated_text = ""
    try:
        async for message in claude_agent_sdk.query(
            prompt=user_prompt,
            options=ClaudeAgentOptions(model=extractor_model, system_prompt=system_prompt),
        ):
            # Collect text from AssistantMessage content blocks
            if hasattr(message, "content") and isinstance(message.content, list):
                for block in message.content:
                    if hasattr(block, "text"):
                        accumulated_text += block.text
    except Exception as exc:
        raise RuntimeError(
            "claude_code provider requires the 'claude' CLI installed and logged in. "
            "Run `claude` once to authenticate, or set PGKG_LLM_PROVIDER=openai instead."
        ) from exc

    return _parse_propositions_json(accumulated_text)


def _extract_openai(
    text: str,
    system_prompt: str,
    prop_schema: dict,
    max_propositions: int,
    settings,
    extractor_model: str,
) -> list[Proposition]:
    import openai  # noqa: PLC0415
    client_kwargs: dict = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        client_kwargs["base_url"] = settings.openai_base_url
    client = openai.OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=extractor_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "propositions_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "propositions": {
                            "type": "array",
                            "items": prop_schema,
                            "maxItems": max_propositions,
                        }
                    },
                    "required": ["propositions"],
                    "additionalProperties": False,
                },
            },
        },
    )
    import json  # noqa: PLC0415
    data = json.loads(response.choices[0].message.content)
    return [Proposition(**p) for p in data["propositions"]]


def _extract_anthropic(
    text: str,
    system_prompt: str,
    prop_schema: dict,
    max_propositions: int,
    settings,
    extractor_model: str,
) -> list[Proposition]:
    import anthropic  # noqa: PLC0415
    import json  # noqa: PLC0415
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tool = {
        "name": "store_propositions",
        "description": "Store extracted propositions",
        "input_schema": {
            "type": "object",
            "properties": {
                "propositions": {
                    "type": "array",
                    "items": prop_schema,
                    "maxItems": max_propositions,
                }
            },
            "required": ["propositions"],
        },
    }
    response = client.messages.create(
        model=extractor_model,
        max_tokens=4096,
        system=system_prompt,
        tools=[tool],
        tool_choice={"type": "tool", "name": "store_propositions"},
        messages=[{"role": "user", "content": text}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "store_propositions":
            return [Proposition(**p) for p in block.input["propositions"]]
    return []


def _extract_ollama(
    text: str,
    system_prompt: str,
    max_propositions: int,
    settings,
    extractor_model: str,
) -> list[Proposition]:
    import json  # noqa: PLC0415
    import httpx  # noqa: PLC0415
    payload = {
        "model": extractor_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "format": "json",
        "stream": False,
    }
    resp = httpx.post(
        f"{settings.ollama_base_url}/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    data = json.loads(content)
    return [Proposition(**p) for p in data.get("propositions", [])]


# ---------------------------------------------------------------------------
# Async-friendly extraction entry point (used by memory.py ingest pipeline)
# ---------------------------------------------------------------------------

async def extract_propositions_async(
    chunk_text: str,
    *,
    max_propositions: int = 20,
    cache: "ExtractCache | None" = None,
) -> list[Proposition]:
    """Async wrapper around extract_propositions with full cache support.

    This is the preferred entry point from async contexts (e.g. Memory.ingest).
    The offline-extract path bypasses cache entirely.
    """
    import asyncio  # noqa: PLC0415

    if os.environ.get("PGKG_OFFLINE_EXTRACT", "0") == "1":
        return [
            Proposition(
                text=chunk_text[:120],
                subject="?",
                predicate="states",
                object=chunk_text[:120],
                object_is_literal=True,
            )
        ]

    settings = get_settings()
    extractor_model = settings.extractor_model or settings.llm_model

    if cache is not None:
        cache_key = compute_cache_key(chunk_text, extractor_model)
        cached = await cache.get(cache_key)
        if cached is not None:
            return cached

    if settings.llm_provider == "claude_code":
        # _extract_claude_code is natively async; call it directly rather than
        # bridging through run_in_executor (which would need asyncio.run inside a
        # running loop and would fail).
        system_prompt = (
            "You are a knowledge extraction assistant. Extract distinct atomic facts "
            "from the provided text as subject-predicate-object propositions. "
            "Rules:\n"
            "- Each proposition must be self-contained and atomic (one fact per proposition).\n"
            "- Resolve pronouns when context allows (use the actual entity name).\n"
            "- Set object_is_literal=true for non-entity objects: numbers, dates, "
            "  freeform strings, measurements, quotes.\n"
            "- Keep subject and object as concise noun phrases.\n"
            f"- Extract at most {max_propositions} propositions.\n"
            "Return a JSON object with a 'propositions' array."
        )
        props = await _extract_claude_code(
            chunk_text,
            extractor_model=extractor_model,
            max_propositions=max_propositions,
            system_prompt=system_prompt,
        )
    else:
        # Run sync LLM extraction in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        props = await loop.run_in_executor(
            None,
            lambda: _do_extract(chunk_text, max_propositions, settings, extractor_model),
        )

    if cache is not None:
        chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
        await cache.put(cache_key, chunk_hash, extractor_model, PROMPT_VERSION, props)

    return props


def _do_extract(
    chunk_text: str,
    max_propositions: int,
    settings,
    extractor_model: str,
) -> list[Proposition]:
    """Internal sync extraction dispatch — no cache logic."""
    system_prompt = (
        "You are a knowledge extraction assistant. Extract distinct atomic facts "
        "from the provided text as subject-predicate-object propositions. "
        "Rules:\n"
        "- Each proposition must be self-contained and atomic (one fact per proposition).\n"
        "- Resolve pronouns when context allows (use the actual entity name).\n"
        "- Set object_is_literal=true for non-entity objects: numbers, dates, "
        "  freeform strings, measurements, quotes.\n"
        "- Keep subject and object as concise noun phrases.\n"
        f"- Extract at most {max_propositions} propositions.\n"
        "Return a JSON object with a 'propositions' array."
    )
    prop_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {"type": "string"},
            "object_is_literal": {"type": "boolean"},
        },
        "required": ["text", "subject", "predicate", "object", "object_is_literal"],
        "additionalProperties": False,
    }
    if settings.llm_provider == "openai":
        return _extract_openai(chunk_text, system_prompt, prop_schema, max_propositions, settings, extractor_model)
    elif settings.llm_provider == "anthropic":
        return _extract_anthropic(chunk_text, system_prompt, prop_schema, max_propositions, settings, extractor_model)
    elif settings.llm_provider == "claude_code":
        import asyncio as _asyncio  # noqa: PLC0415
        return _asyncio.run(
            _extract_claude_code(
                chunk_text,
                extractor_model=extractor_model,
                max_propositions=max_propositions,
                system_prompt=system_prompt,
            )
        )
    else:
        return _extract_ollama(chunk_text, system_prompt, max_propositions, settings, extractor_model)
