"""Integration tests for Memory using a real Postgres container."""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import asyncpg
import numpy as np
import pytest

from pgkg.memory import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int = 1024, *, hot: int = 0) -> list[float]:
    v = [0.0] * dim
    v[hot] = 1.0
    return v


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic embed: each text gets a unique unit vector based on hash."""
    result = []
    for i, t in enumerate(texts):
        v = [0.0] * 1024
        idx = hash(t) % 1024
        v[idx] = 1.0
        result.append(v)
    return result


# ---------------------------------------------------------------------------
# test_ingest_creates_rows
# ---------------------------------------------------------------------------

async def test_ingest_creates_rows(pool: asyncpg.Pool, monkeypatch):
    """PGKG_OFFLINE_EXTRACT=1: ingest populates documents, chunks, propositions tables."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)

    ns = f"ingest_test_{uuid.uuid4().hex[:8]}"
    mem = Memory(pool, namespace=ns)
    result = await mem.ingest("Hello world. This is a test document.")

    assert result.documents == 1
    assert result.chunks >= 1
    assert result.propositions >= 1

    async with pool.acquire() as conn:
        doc_count = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE namespace = $1", ns
        )
        chunk_count = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.namespace = $1",
            ns,
        )
        prop_count = await conn.fetchval(
            "SELECT COUNT(*) FROM propositions WHERE namespace = $1", ns
        )

    assert doc_count == 1
    assert chunk_count >= 1
    assert prop_count >= 1


# ---------------------------------------------------------------------------
# test_recall_returns_ingested
# ---------------------------------------------------------------------------

async def test_recall_returns_ingested(pool: asyncpg.Pool, monkeypatch):
    """After ingesting a doc, recalling a matching query returns the proposition."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module

    # Give the "ocean" text a specific hot vector
    ocean_text = "The ocean is vast and deep."

    def _controlled_embed(texts: list[str]) -> list[list[float]]:
        result = []
        for t in texts:
            if "ocean" in t.lower() or "vast" in t.lower():
                v = _unit_vec(hot=100)
            else:
                v = _unit_vec(hot=hash(t) % 1024)
            result.append(v)
        return result

    monkeypatch.setattr(ml_module, "embed", _controlled_embed)

    # Disable rerank/mmr for simplicity
    class FakeCE:
        def predict(self, pairs):
            return [0.5] * len(pairs)

    monkeypatch.setattr(ml_module, "_rerank_model", FakeCE())

    ns = f"recall_test_{uuid.uuid4().hex[:8]}"
    mem = Memory(pool, namespace=ns)
    await mem.ingest(ocean_text)

    results = await mem.recall(
        "vast ocean",
        k=10,
        with_rerank=False,
        with_mmr=False,
        expand_graph=False,
    )

    assert len(results) > 0
    texts = [r.text for r in results]
    assert any("ocean" in t.lower() or "vast" in t.lower() for t in texts)


# ---------------------------------------------------------------------------
# test_recall_session_scope
# ---------------------------------------------------------------------------

async def test_recall_session_scope(pool: asyncpg.Pool, monkeypatch):
    """Propositions ingested with session_id='A' don't appear in session_id='B' recall."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)

    ns = f"session_test_{uuid.uuid4().hex[:8]}"
    mem = Memory(pool, namespace=ns)

    # Ingest unique text with session A
    unique_text = f"Unique session A content xyzzy_{uuid.uuid4().hex}"
    await mem.ingest(unique_text, session_id="A")

    # Recall with session B — should return empty (no global facts)
    results = await mem.recall(
        unique_text[:20],
        k=10,
        session_id="B",
        with_rerank=False,
        with_mmr=False,
        expand_graph=False,
    )

    prop_ids = {r.proposition_id for r in results}
    # Session B should not see session A's propositions
    # We verify by checking the session_id in the DB
    if prop_ids:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT session_id FROM propositions WHERE id = ANY($1::uuid[])",
                [str(p) for p in prop_ids],
            )
        sessions = {r["session_id"] for r in rows}
        assert "A" not in sessions, f"Session B recall returned session A props: {sessions}"


# ---------------------------------------------------------------------------
# test_forget_supersedes
# ---------------------------------------------------------------------------

async def test_forget_supersedes(pool: asyncpg.Pool, monkeypatch):
    """After forget(), the proposition no longer appears in recall results."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module

    target_text = f"Forgotten fact about zebras {uuid.uuid4().hex}"

    def _targeted_embed(texts: list[str]) -> list[list[float]]:
        result = []
        for t in texts:
            if "zebra" in t.lower() or "forgotten" in t.lower():
                v = _unit_vec(hot=700)
            else:
                v = _unit_vec(hot=hash(t) % 1024)
            result.append(v)
        return result

    monkeypatch.setattr(ml_module, "embed", _targeted_embed)

    ns = f"forget_test_{uuid.uuid4().hex[:8]}"
    mem = Memory(pool, namespace=ns)
    result = await mem.ingest(target_text)

    # Get the proposition id
    async with pool.acquire() as conn:
        prop_id = await conn.fetchval(
            "SELECT id FROM propositions WHERE namespace = $1 LIMIT 1", ns
        )

    assert prop_id is not None

    # Forget it
    await mem.forget(prop_id)

    # Verify it no longer appears in recall
    results = await mem.recall(
        "zebra forgotten",
        k=10,
        with_rerank=False,
        with_mmr=False,
        expand_graph=False,
    )
    result_ids = {r.proposition_id for r in results}
    assert prop_id not in result_ids, "Forgotten proposition should not appear in recall"


# ---------------------------------------------------------------------------
# test_recall_default_flags_with_pgvector_embedding (regression)
# ---------------------------------------------------------------------------
#
# Regression for: bool(numpy.ndarray) ambiguity in memory.recall when the
# rerank+MMR path inspects pgvector's returned embeddings. Every other
# recall test bypassed rerank/MMR, so the truthiness check on the
# embedding column was never exercised against a real DB row.

async def test_recall_default_flags_with_pgvector_embedding(pool: asyncpg.Pool, monkeypatch):
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module

    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    monkeypatch.setattr(ml_module, "rerank", lambda q, docs: [1.0 / (i + 1) for i in range(len(docs))])

    ns = f"recall_default_{uuid.uuid4().hex[:8]}"
    mem = Memory(pool, namespace=ns, extract_propositions=False)
    await mem.ingest("The chunks-only ingest mode skips LLM extraction entirely.")
    await mem.ingest("Hybrid retrieval fuses BM25 and vector similarity via RRF.")

    # Default flags: rerank=True, mmr=True. This is the path the API uses
    # and the path that previously crashed on numpy embedding truthiness.
    results = await mem.recall("chunks-only mode", k=2)

    assert len(results) > 0
    assert all(r.text for r in results)
