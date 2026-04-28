"""Tests for the proposition extraction cache (PostgresExtractCache)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from pgkg.ml import (
    PROMPT_VERSION,
    ExtractCache,
    Proposition,
    compute_cache_key,
    extract_propositions_async,
)
from pgkg.memory import BackendExtractCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prop(**kwargs) -> Proposition:
    defaults = dict(
        text="Alice is a scientist.",
        subject="Alice",
        predicate="is",
        object="scientist",
        object_is_literal=False,
    )
    defaults.update(kwargs)
    return Proposition(**defaults)


async def _seed_cache(pool: asyncpg.Pool, cache_key: str, props: list[Proposition]) -> None:
    payload = json.dumps([p.model_dump() for p in props])
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO proposition_cache
                (cache_key, chunk_hash, extractor_model, prompt_version, propositions)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            ON CONFLICT (cache_key) DO NOTHING
            """,
            cache_key,
            hashlib.sha256(b"dummy-chunk").hexdigest(),
            "test-model",
            PROMPT_VERSION,
            payload,
        )


async def _get_hit_count(pool: asyncpg.Pool, cache_key: str) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT hit_count FROM proposition_cache WHERE cache_key = $1",
            cache_key,
        )
        return row["hit_count"] if row else 0


# ---------------------------------------------------------------------------
# test_cache_hit_returns_stored_props_without_llm
# ---------------------------------------------------------------------------

async def test_cache_hit_returns_stored_props_without_llm(
    pool: asyncpg.Pool, backend, monkeypatch
):
    """Pre-populate cache; LLM provider should never be called on cache hit."""
    chunk = "Alice is a brilliant scientist who works at CERN."
    model = "test-model-no-llm"
    cache_key = compute_cache_key(chunk, model)

    stored = [_make_prop(text="Alice works at CERN.", subject="Alice", predicate="works at", object="CERN")]
    await _seed_cache(pool, cache_key, stored)

    # Patch settings so extractor_model matches what we seeded
    import pgkg.config as config_module
    import pgkg.ml as ml_module

    fake_settings = MagicMock()
    fake_settings.extractor_model = model
    fake_settings.llm_model = model
    fake_settings.llm_provider = "openai"
    fake_settings.openai_api_key = None
    fake_settings.openai_base_url = None
    fake_settings.prompt_version = PROMPT_VERSION

    monkeypatch.setattr(ml_module, "get_settings", lambda: fake_settings)
    monkeypatch.delenv("PGKG_OFFLINE_EXTRACT", raising=False)

    # Patch OpenAI so if LLM is called, the test fails
    import openai
    monkeypatch.setattr(
        openai.OpenAI,
        "__init__",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("LLM was called — should have been a cache hit")),
    )

    cache = BackendExtractCache(backend)
    result = await extract_propositions_async(chunk, cache=cache)

    assert len(result) == 1
    assert result[0].subject == "Alice"
    assert result[0].object == "CERN"


# ---------------------------------------------------------------------------
# test_cache_miss_then_hit
# ---------------------------------------------------------------------------

async def test_cache_miss_then_hit(pool: asyncpg.Pool, backend, monkeypatch):
    """First call invokes LLM stub (count=1); second call hits cache (count still 1)."""
    chunk = "Bob loves hiking in the mountains near his home."
    model = "stub-model-v1"

    import pgkg.ml as ml_module

    fake_settings = MagicMock()
    fake_settings.extractor_model = model
    fake_settings.llm_model = model
    fake_settings.llm_provider = "openai"
    fake_settings.openai_api_key = "fake"
    fake_settings.openai_base_url = None
    fake_settings.prompt_version = PROMPT_VERSION

    monkeypatch.setattr(ml_module, "get_settings", lambda: fake_settings)
    monkeypatch.delenv("PGKG_OFFLINE_EXTRACT", raising=False)

    call_count = {"n": 0}
    stub_prop = _make_prop(text="Bob loves hiking.", subject="Bob", predicate="loves", object="hiking")

    def _fake_do_extract(chunk_text, max_propositions, settings, extractor_model):
        call_count["n"] += 1
        return [stub_prop]

    monkeypatch.setattr(ml_module, "_do_extract", _fake_do_extract)

    cache = BackendExtractCache(backend)

    # First call — cache miss, LLM called
    result1 = await extract_propositions_async(chunk, cache=cache)
    assert call_count["n"] == 1
    assert result1[0].subject == "Bob"

    # Second call — cache hit, LLM NOT called again
    result2 = await extract_propositions_async(chunk, cache=cache)
    assert call_count["n"] == 1, "LLM should not have been called a second time"
    assert result2[0].subject == "Bob"


# ---------------------------------------------------------------------------
# test_offline_extract_bypasses_cache
# ---------------------------------------------------------------------------

async def test_offline_extract_bypasses_cache(pool: asyncpg.Pool, monkeypatch):
    """PGKG_OFFLINE_EXTRACT=1 should bypass cache reads and writes entirely."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    chunk = "Carol studies quantum mechanics."
    model = "offline-model"

    import pgkg.ml as ml_module

    fake_settings = MagicMock()
    fake_settings.extractor_model = model
    fake_settings.llm_model = model

    monkeypatch.setattr(ml_module, "get_settings", lambda: fake_settings)

    cache_key = compute_cache_key(chunk, model)

    # Inject a spy cache
    get_calls = {"n": 0}
    put_calls = {"n": 0}

    class SpyCache:
        async def get(self, k):
            get_calls["n"] += 1
            return None

        async def put(self, k, ch, em, pv, props):
            put_calls["n"] += 1

    spy = SpyCache()
    result = await extract_propositions_async(chunk, cache=spy)

    assert get_calls["n"] == 0, "cache.get should NOT be called in offline mode"
    assert put_calls["n"] == 0, "cache.put should NOT be called in offline mode"
    # Offline extract returns stub proposition
    assert len(result) == 1
    assert result[0].predicate == "states"


# ---------------------------------------------------------------------------
# test_cache_key_changes_with_model
# ---------------------------------------------------------------------------

def test_cache_key_changes_with_model():
    """Same text with two different models produces distinct cache keys."""
    chunk = "The quick brown fox jumps over the lazy dog."
    key1 = compute_cache_key(chunk, "gpt-4o-mini-2024-07-18")
    key2 = compute_cache_key(chunk, "gpt-4o-2024-08-06")
    assert key1 != key2


# ---------------------------------------------------------------------------
# test_cache_key_changes_with_prompt_version
# ---------------------------------------------------------------------------

def test_cache_key_changes_with_prompt_version(monkeypatch):
    """Bumping PROMPT_VERSION produces a different cache key for same input."""
    import pgkg.ml as ml_module

    chunk = "Delta is the fourth letter of the Greek alphabet."
    model = "some-model"

    key_v1 = compute_cache_key(chunk, model)

    # Monkeypatch the module-level PROMPT_VERSION
    monkeypatch.setattr(ml_module, "PROMPT_VERSION", "v2")

    # compute_cache_key reads module-level PROMPT_VERSION at call time
    key_v2 = ml_module.compute_cache_key(chunk, model)

    assert key_v1 != key_v2


# ---------------------------------------------------------------------------
# test_postgres_cache_hit_count_increments
# ---------------------------------------------------------------------------

async def test_postgres_cache_hit_count_increments(pool: asyncpg.Pool, backend):
    """Two get() calls on a populated key → hit_count = 2."""
    chunk = "Eve is a cryptographer."
    model = "hit-count-model"
    cache_key = compute_cache_key(chunk, model)

    stored = [_make_prop(text="Eve is a cryptographer.", subject="Eve", predicate="is", object="cryptographer")]
    await _seed_cache(pool, cache_key, stored)

    cache = BackendExtractCache(backend)
    await cache.get(cache_key)
    await cache.get(cache_key)

    count = await _get_hit_count(pool, cache_key)
    assert count == 2
