"""Integration tests for Memory using a real Postgres container."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
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

async def test_ingest_creates_rows(pool: asyncpg.Pool, backend, monkeypatch):
    """PGKG_OFFLINE_EXTRACT=1: ingest populates documents, chunks, propositions tables."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)

    ns = f"ingest_test_{uuid.uuid4().hex[:8]}"
    mem = Memory(backend, namespace=ns)
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

async def test_recall_returns_ingested(pool: asyncpg.Pool, backend, monkeypatch):
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
    mem = Memory(backend, namespace=ns)
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

async def test_recall_session_scope(pool: asyncpg.Pool, backend, monkeypatch):
    """Propositions ingested with session_id='A' don't appear in session_id='B' recall."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)

    ns = f"session_test_{uuid.uuid4().hex[:8]}"
    mem = Memory(backend, namespace=ns)

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

async def test_forget_supersedes(pool: asyncpg.Pool, backend, monkeypatch):
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
    mem = Memory(backend, namespace=ns)
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

async def test_recall_default_flags_with_pgvector_embedding(pool: asyncpg.Pool, backend, monkeypatch):
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module

    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    monkeypatch.setattr(ml_module, "rerank", lambda q, docs: [1.0 / (i + 1) for i in range(len(docs))])

    ns = f"recall_default_{uuid.uuid4().hex[:8]}"
    mem = Memory(backend, namespace=ns, extract_propositions=False)
    await mem.ingest("The chunks-only ingest mode skips LLM extraction entirely.")
    await mem.ingest("Hybrid retrieval fuses BM25 and vector similarity via RRF.")

    # Default flags: rerank=True, mmr=True. This is the path the API uses
    # and the path that previously crashed on numpy embedding truthiness.
    results = await mem.recall("chunks-only mode", k=2)

    assert len(results) > 0
    assert all(r.text for r in results)


# ---------------------------------------------------------------------------
# test_ingest_propagates_asserted_at
# ---------------------------------------------------------------------------

async def test_ingest_propagates_asserted_at(pool: asyncpg.Pool, monkeypatch):
    """Ingest with asserted_at stores it in both chunk and proposition rows."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)

    expected_ts = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    ns = f"assertedat_ingest_{uuid.uuid4().hex[:8]}"
    mem = Memory(pool, namespace=ns, extract_propositions=False)

    await mem.ingest(
        "The sky is blue and the grass is green.",
        asserted_at=expected_ts,
    )

    async with pool.acquire() as conn:
        prop_row = await conn.fetchrow(
            "SELECT asserted_at FROM propositions WHERE namespace = $1 LIMIT 1", ns
        )
        chunk_row = await conn.fetchrow(
            """
            SELECT c.asserted_at FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.namespace = $1
            LIMIT 1
            """,
            ns,
        )

    assert prop_row is not None
    prop_ts = prop_row["asserted_at"]
    if prop_ts is not None and prop_ts.tzinfo is None:
        prop_ts = prop_ts.replace(tzinfo=timezone.utc)
    assert prop_ts == expected_ts, f"Proposition asserted_at {prop_ts!r} should equal {expected_ts!r}"

    assert chunk_row is not None
    chunk_ts = chunk_row["asserted_at"]
    if chunk_ts is not None and chunk_ts.tzinfo is None:
        chunk_ts = chunk_ts.replace(tzinfo=timezone.utc)
    assert chunk_ts == expected_ts, f"Chunk asserted_at {chunk_ts!r} should equal {expected_ts!r}"


# ---------------------------------------------------------------------------
# test_recall_returns_asserted_at_in_result
# ---------------------------------------------------------------------------

async def test_recall_returns_asserted_at_in_result(pool: asyncpg.Pool, monkeypatch):
    """Result.asserted_at is populated when ingested with an asserted_at timestamp."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module

    expected_ts = datetime(2025, 6, 20, 8, 30, 0, tzinfo=timezone.utc)
    target_text = f"Fact about temporal reasoning asserted {uuid.uuid4().hex}"

    def _controlled_embed(texts: list[str]) -> list[list[float]]:
        result = []
        for t in texts:
            v = [0.0] * 1024
            v[hash(t) % 1024] = 1.0
            result.append(v)
        return result

    monkeypatch.setattr(ml_module, "embed", _controlled_embed)

    ns = f"assertedat_recall_{uuid.uuid4().hex[:8]}"
    mem = Memory(pool, namespace=ns, extract_propositions=False)

    await mem.ingest(target_text, asserted_at=expected_ts)

    results = await mem.recall(
        target_text[:30],
        k=10,
        with_rerank=False,
        with_mmr=False,
        expand_graph=False,
    )

    assert len(results) > 0, "Recall should return results"
    # All ingested props were stamped with expected_ts; every returned result should carry it
    for r in results:
        ts = r.asserted_at
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        assert ts == expected_ts, f"Result.asserted_at {ts!r} should equal {expected_ts!r}"
