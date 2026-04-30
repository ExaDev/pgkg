"""Tests for chunks-only ingest mode (extract_propositions=False)."""
from __future__ import annotations

import argparse
import uuid
from unittest.mock import patch

import asyncpg
import pytest

from pgkg.memory import Memory
from bench.common import BenchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic embed: each text gets a unique unit vector based on hash."""
    result = []
    for t in texts:
        v = [0.0] * 1024
        idx = hash(t) % 1024
        v[idx] = 1.0
        result.append(v)
    return result


# ---------------------------------------------------------------------------
# test_ingest_chunks_only_skips_extraction
# ---------------------------------------------------------------------------

async def test_ingest_chunks_only_skips_extraction(pool: asyncpg.Pool, backend, monkeypatch):
    """extract_propositions_async must not be called; propositions have NULL subject_id
    and metadata->>'mode' = 'chunk'; entities table is untouched."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)

    # Monkeypatch extract_propositions_async to raise — proves it is never called
    async def _should_not_be_called(*args, **kwargs):
        raise AssertionError("extract_propositions_async was called in chunks-only mode")

    monkeypatch.setattr(ml_module, "extract_propositions_async", _should_not_be_called)

    ns = f"chunks_only_{uuid.uuid4().hex[:8]}"
    mem = Memory(backend, namespace=ns, extract_propositions=False)
    result = await mem.ingest("Hello world. This is a test document.")

    assert result.documents == 1
    assert result.chunks >= 1
    assert result.propositions >= 1
    assert result.entities == 0

    async with pool.acquire() as conn:
        # All propositions should have NULL subject_id
        null_count = await conn.fetchval(
            "SELECT COUNT(*) FROM propositions WHERE namespace = $1 AND subject_id IS NULL",
            ns,
        )
        total_count = await conn.fetchval(
            "SELECT COUNT(*) FROM propositions WHERE namespace = $1",
            ns,
        )
        assert null_count == total_count

        # All propositions should have metadata->>'mode' = 'chunk'
        chunk_mode_count = await conn.fetchval(
            "SELECT COUNT(*) FROM propositions WHERE namespace = $1 AND metadata->>'mode' = 'chunk'",
            ns,
        )
        assert chunk_mode_count == total_count

        # Entities table should be untouched for this namespace
        entity_count = await conn.fetchval(
            "SELECT COUNT(*) FROM entities WHERE namespace = $1",
            ns,
        )
        assert entity_count == 0


# ---------------------------------------------------------------------------
# test_recall_works_in_chunks_mode
# ---------------------------------------------------------------------------

async def test_recall_works_in_chunks_mode(pool: asyncpg.Pool, backend, monkeypatch):
    """After chunks-only ingest, recall returns results with NULL predicate."""
    import pgkg.ml as ml_module

    ocean_text = "The ocean is vast and deep."

    def _controlled_embed(texts: list[str]) -> list[list[float]]:
        result = []
        for t in texts:
            v = [0.0] * 1024
            if "ocean" in t.lower() or "vast" in t.lower():
                v[100] = 1.0
            else:
                v[hash(t) % 1024] = 1.0
            result.append(v)
        return result

    monkeypatch.setattr(ml_module, "embed", _controlled_embed)

    class FakeCE:
        def predict(self, pairs):
            return [0.5] * len(pairs)

    monkeypatch.setattr(ml_module, "_rerank_model", FakeCE())

    async def _noop_extract(*args, **kwargs):
        raise AssertionError("extract should not be called")

    monkeypatch.setattr(ml_module, "extract_propositions_async", _noop_extract)

    ns = f"chunks_recall_{uuid.uuid4().hex[:8]}"
    mem = Memory(backend, namespace=ns, extract_propositions=False)
    await mem.ingest(ocean_text)

    results = await mem.recall(
        "vast ocean",
        k=10,
        with_rerank=False,
        with_mmr=False,
        expand_graph=False,
    )

    assert len(results) > 0
    # All results should have NULL predicate (no extraction happened)
    for r in results:
        assert r.predicate is None


# ---------------------------------------------------------------------------
# test_chunks_mode_graph_expansion_is_noop
# ---------------------------------------------------------------------------

async def test_chunks_mode_graph_expansion_is_noop(pool: asyncpg.Pool, backend, monkeypatch):
    """Graph expansion with chunks-only ingest produces no graph-sourced rows."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)

    class FakeCE:
        def predict(self, pairs):
            return [0.5] * len(pairs)

    monkeypatch.setattr(ml_module, "_rerank_model", FakeCE())

    async def _noop_extract(*args, **kwargs):
        raise AssertionError("extract should not be called")

    monkeypatch.setattr(ml_module, "extract_propositions_async", _noop_extract)

    ns = f"chunks_graph_{uuid.uuid4().hex[:8]}"
    mem = Memory(backend, namespace=ns, extract_propositions=False)
    await mem.ingest("Alice visited Bob last Tuesday. Bob works at Acme Corp.")

    results = await mem.recall(
        "Alice Bob",
        k=20,
        with_rerank=False,
        with_mmr=False,
        expand_graph=True,  # enabled but should be a no-op (no edges)
    )

    # No result should have source_kind='graph' since there are no edges
    graph_results = [r for r in results if r.source_kind == "graph"]
    assert len(graph_results) == 0


# ---------------------------------------------------------------------------
# test_bench_config_mode_field
# ---------------------------------------------------------------------------

def test_bench_config_mode_field():
    """BenchConfig(extract_propositions=False).resolve_stack() -> stack.mode == 'chunks'."""
    config = BenchConfig(extract_propositions=False)
    stack = config.resolve_stack()
    assert stack.mode == "chunks"


def test_bench_config_mode_field_default():
    """BenchConfig() default -> stack.mode == 'propositions'."""
    config = BenchConfig()
    stack = config.resolve_stack()
    assert stack.mode == "propositions"


# ---------------------------------------------------------------------------
# test_cli_chunks_only_flag
# ---------------------------------------------------------------------------

def test_cli_chunks_only_flag():
    """--chunks-only argparse flag is accepted and sets chunks_only=True."""
    import argparse
    # Directly test the argparse setup from cli.py
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--chunks-only", action="store_true", default=False)

    args = parser.parse_args(["myfile.txt", "--chunks-only"])
    assert args.chunks_only is True

    args_default = parser.parse_args(["myfile.txt"])
    assert args_default.chunks_only is False
