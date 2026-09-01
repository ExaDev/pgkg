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


# ---------------------------------------------------------------------------
# The access ledger is accounting, so a retry must not double-count
# ---------------------------------------------------------------------------

async def test_a_failed_access_flush_does_not_double_count(
    pool: asyncpg.Pool,
) -> None:
    """The flush writes one autocommitted statement per org and, on any
    failure, used to restore every org's counts — including the orgs whose
    statement had already committed.  Those were then applied again by the next
    flush, and the decay profile's frequency term reads that column, so one
    transient link failure permanently inflated the ranking of whichever tenant
    happened to flush first.
    """
    import asyncpg as pg

    async with pool.acquire() as conn:
        org_a = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
            f"flush_a_{uuid.uuid4().hex[:8]}",
        )
        org_b = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
            f"flush_b_{uuid.uuid4().hex[:8]}",
        )
        prop_a = await conn.fetchval(
            "INSERT INTO propositions (text, org_id) VALUES ('a', $1) RETURNING id",
            org_a,
        )
        prop_b = await conn.fetchval(
            "INSERT INTO propositions (text, org_id) VALUES ('b', $1) RETURNING id",
            org_b,
        )

    memory = Memory(pool)
    memory._access.record(org_a, [prop_a] * 3)
    memory._access.record(org_b, [prop_b] * 3)

    statements = {"n": 0}
    real_execute = pg.Connection.execute

    async def flaky(self, query, *args, **kwargs):
        if "access_count" in query:
            statements["n"] += 1
            if statements["n"] == 2:
                raise pg.PostgresConnectionError("link went away")
        return await real_execute(self, query, *args, **kwargs)

    pg.Connection.execute = flaky
    try:
        with pytest.raises(Exception):
            await memory.flush_access()
        pg.Connection.execute = real_execute
        await memory.flush_access()
    finally:
        pg.Connection.execute = real_execute
        await memory.aclose()

    async with pool.acquire() as conn:
        counts = dict(
            await conn.fetch(
                "SELECT id, access_count FROM propositions"
                " WHERE id = ANY($1::uuid[])",
                [prop_a, prop_b],
            )
        )

    assert counts[prop_a] == 3, (
        f"the org whose flush committed before the failure was counted "
        f"{counts[prop_a]} times, not 3"
    )
    assert counts[prop_b] == 3
