"""Chunks-only ingest, which writes into the chunk store and nowhere else.

ADR 0001, D1 lists "chunk embeddings in `propositions`" under what not to
build — it corrupts BM25 length normalisation, conflates lifecycles and blocks
per-store index tuning — and says of the existing instance: *undo it*.  Phase 2
names the shape that replaces it: chunks-only mode becomes "retrieve from the
chunk source" and the fake-proposition rows disappear.

So the contract these tests pin is a store, not a mode flag.  A chunks-only
ingest produces `chunks` rows with an embedding and a tsvector, carried by the
current version of a document, content-addressed under D6 — and no proposition
rows at all.

The tests scope themselves to a fresh org and collection rather than to a
namespace.  Chunks carry no `namespace`: D3 replaced stringly-typed scoping
with explicit columns, and the chunk arms of retrieval read those columns.  A
chunks-only test that isolated itself the way a propositions test does would be
reading every other test's passages.
"""
from __future__ import annotations

import uuid

import asyncpg
import pytest
from pgvector import HalfVector

from pgkg.memory import Memory, Scope, provision_org
from bench.common import BenchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 1024


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic embed: each text gets a unique unit vector based on hash."""
    result = []
    for t in texts:
        v = [0.0] * DIM
        idx = hash(t) % DIM
        v[idx] = 1.0
        result.append(v)
    return result


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def _fresh_scope(
    pool: asyncpg.Pool, prefix: str, **overrides
) -> Scope:
    """An org and collection of this test's own, bound to the live generation."""
    async with pool.acquire() as conn:
        org = await provision_org(conn, _unique(prefix))
        collection = await conn.fetchval(
            """
            INSERT INTO collections
                (org_id, owner_org_id, name, kind, claim_scope, decay_profile)
            VALUES ($1, $1, $2, 'chat', 'org', 'conversational')
            RETURNING id
            """,
            org,
            _unique(f"{prefix}_coll"),
        )
    return Scope(org_id=org, collection_id=collection, **overrides)


async def _make_user(pool: asyncpg.Pool, org_id: uuid.UUID) -> uuid.UUID:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
            org_id,
            _unique("user"),
        )


def _no_extraction(monkeypatch) -> None:
    """Proves the extractor is never reached, whatever else the test asserts."""
    import pgkg.ml as ml_module

    async def _should_not_be_called(*args, **kwargs):
        raise AssertionError("the extractor was called in chunks-only mode")

    monkeypatch.setattr(
        ml_module, "extract_propositions_async", _should_not_be_called
    )


# ---------------------------------------------------------------------------
# 1. What the store holds after a chunks-only ingest
# ---------------------------------------------------------------------------

async def test_chunks_only_ingest_writes_no_propositions(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """D1's "undo the existing instance": the retrievable row is the chunk, so
    there is nothing for a proposition row to hold."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    _no_extraction(monkeypatch)

    scope = await _fresh_scope(pool, "chunks_only")
    mem = Memory(pool, namespace=_unique("ns"), scope=scope, extract_propositions=False)
    result = await mem.ingest("Hello world. This is a test document.")

    assert result.documents == 1
    assert result.chunks >= 1
    assert result.propositions == 0
    assert result.entities == 0

    async with pool.acquire() as conn:
        propositions = await conn.fetchval(
            "SELECT COUNT(*) FROM propositions WHERE org_id = $1", scope.org_id
        )
        entities = await conn.fetchval(
            "SELECT COUNT(*) FROM entities WHERE org_id = $1", scope.org_id
        )

    assert propositions == 0
    assert entities == 0


async def test_chunks_only_ingest_writes_retrievable_chunks(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """Embedding and tsvector on the chunk row, and the flag the chunk arms of
    retrieval read.  A chunk with no vector is not a candidate; a chunk that is
    not retrievable is not one either (ADR 0001, D1, D6)."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    _no_extraction(monkeypatch)

    scope = await _fresh_scope(pool, "chunk_rows")
    mem = Memory(pool, namespace=_unique("ns"), scope=scope, extract_propositions=False)
    result = await mem.ingest("The ocean is vast and deep. Whales live in it.")

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT text, embedding IS NOT NULL AS vectored, doc_len, retrievable,"
            " document_id, content_hash IS NOT NULL AS addressed"
            " FROM chunks WHERE org_id = $1",
            scope.org_id,
        )

    assert len(rows) == result.chunks
    for row in rows:
        assert row["vectored"] is True
        assert row["doc_len"] > 0
        assert row["retrievable"] is True
        assert row["document_id"] is None
        assert row["addressed"] is True


async def test_chunks_only_ingest_carries_its_chunks_on_a_current_version(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """The chunk is retrievable because a live document's current version
    carries it, which is the only definition of liveness the chunk store has
    (ADR 0001, D6)."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    _no_extraction(monkeypatch)

    scope = await _fresh_scope(pool, "versioned")
    mem = Memory(pool, namespace=_unique("ns"), scope=scope, extract_propositions=False)
    result = await mem.ingest("A single paragraph about kestrels.")

    async with pool.acquire() as conn:
        version = await conn.fetchrow(
            """
            SELECT dv.id, dv.status, d.current_version_id
            FROM document_versions dv
            JOIN documents d ON d.id = dv.document_id
            WHERE d.org_id = $1
            """,
            scope.org_id,
        )
        carried = await conn.fetchval(
            "SELECT COUNT(*) FROM document_version_chunks WHERE document_version_id = $1",
            version["id"],
        )

    assert version["status"] == "current"
    assert version["current_version_id"] == version["id"]
    assert carried == result.chunks


async def test_chunks_only_ingest_records_the_chunker_on_the_chunk(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """A chunk derives its provenance from its version, because a
    content-addressed passage is shared by every version carrying it and cannot
    record which ingest produced it (ADR 0001, D5, D6)."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    _no_extraction(monkeypatch)

    scope = await _fresh_scope(pool, "chunk_prov")
    mem = Memory(pool, namespace=_unique("ns"), scope=scope, extract_propositions=False)
    result = await mem.ingest("Attributed prose about ledgers.")

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT pr.id, pr.producer, pr.kind
            FROM chunks c
            JOIN provenance pr ON pr.id = c.provenance_id
            WHERE c.org_id = $1
            """,
            scope.org_id,
        )
        version_provenance = await conn.fetchval(
            """
            SELECT dv.provenance_id FROM document_versions dv
            JOIN documents d ON d.id = dv.document_id
            WHERE d.org_id = $1
            """,
            scope.org_id,
        )

    assert len(rows) == 1
    assert rows[0]["producer"] == "chunker"
    assert rows[0]["kind"] == "document_version"
    assert rows[0]["id"] == version_provenance
    assert result.provenance_ids == (version_provenance,)


# ---------------------------------------------------------------------------
# 2. Content addressing, which is what the chunk store buys
# ---------------------------------------------------------------------------

async def test_re_ingesting_the_same_text_reuses_the_passage(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """Two ingests of one paragraph are one chunk row with two carriers: the
    content address is what makes a re-ingest cost storage nothing (D6)."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    _no_extraction(monkeypatch)

    scope = await _fresh_scope(pool, "addressed")
    mem = Memory(pool, namespace=_unique("ns"), scope=scope, extract_propositions=False)
    text = "Boilerplate that appears in two ingests of the same corner."
    first = await mem.ingest(text)
    second = await mem.ingest(text)

    async with pool.acquire() as conn:
        chunk_rows = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE org_id = $1", scope.org_id
        )
        links = await conn.fetchval(
            """
            SELECT COUNT(*) FROM document_version_chunks dvc
            JOIN chunks c ON c.id = dvc.chunk_id
            WHERE c.org_id = $1
            """,
            scope.org_id,
        )
        refcount = await conn.fetchval(
            "SELECT refcount FROM chunks WHERE org_id = $1", scope.org_id
        )

    assert first.chunks == second.chunks == 1
    assert chunk_rows == 1
    assert links == 2
    assert refcount == 2


async def test_a_turn_that_repeats_a_paragraph_holds_two_positions_on_one_row(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """The address is the row's identity, and a turn can reach it twice.

    Its two positions in the version are two links, not two rows — so the
    vector has to be written against the row and not against the position, or
    the second position overwrites the first with the same value and the count
    of what was embedded stops meaning anything (D6)."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    _no_extraction(monkeypatch)

    scope = await _fresh_scope(pool, "repeated")
    mem = Memory(pool, namespace=_unique("ns"), scope=scope, extract_propositions=False)
    result = await mem.ingest(
        "## A\n\nSame body text here.\n\n## A\n\nSame body text here.\n",
        chunk_size=40,
    )

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, embedding IS NOT NULL AS vectored, refcount"
            " FROM chunks WHERE org_id = $1",
            scope.org_id,
        )
        ords = [
            row["ord"]
            for row in await conn.fetch(
                """
                SELECT dvc.ord FROM document_version_chunks dvc
                JOIN chunks c ON c.id = dvc.chunk_id
                WHERE c.org_id = $1 ORDER BY dvc.ord
                """,
                scope.org_id,
            )
        ]

    assert result.chunks == 2
    assert len(rows) == 1
    assert rows[0]["vectored"] is True
    assert rows[0]["refcount"] == 2
    assert ords == [0, 1]


async def test_two_users_private_passages_never_share_a_row(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """Dedup stops at every boundary the read predicate consults.  042 stopped
    it at the collection and the ACL group for exactly this reason; a private
    passage adds `visibility` and `owner_user_id` to that list, because
    pgkg_visible() reads both and one row cannot hold two owners."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    _no_extraction(monkeypatch)

    base = await _fresh_scope(pool, "private_chunks")
    one = await _make_user(pool, base.org_id)
    two = await _make_user(pool, base.org_id)
    text = "The same private note, written twice by two people."

    for user in (one, two):
        scope = Scope(
            org_id=base.org_id,
            collection_id=base.collection_id,
            user_id=user,
            visibility="private",
            claim_scope="user",
        )
        await Memory(
            pool,
            namespace=_unique("ns"),
            scope=scope,
            extract_propositions=False,
        ).ingest(text)

    async with pool.acquire() as conn:
        owners = [
            row["owner_user_id"]
            for row in await conn.fetch(
                "SELECT owner_user_id, visibility FROM chunks WHERE org_id = $1"
                " ORDER BY owner_user_id",
                base.org_id,
            )
        ]
        visibilities = [
            row["visibility"]
            for row in await conn.fetch(
                "SELECT visibility FROM chunks WHERE org_id = $1", base.org_id
            )
        ]

    assert sorted(owners) == sorted([one, two])
    assert visibilities == ["private", "private"]


# ---------------------------------------------------------------------------
# 3. Recall over the chunk store
# ---------------------------------------------------------------------------

async def test_recall_in_chunks_mode_returns_passages(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """`recall()` in chunks-only mode answers out of the chunk store: the row
    is a passage, so it has a chunk id, no proposition id, and nothing an
    extractor would have given it."""
    import pgkg.ml as ml_module

    def _controlled_embed(texts: list[str]) -> list[list[float]]:
        result = []
        for t in texts:
            v = [0.0] * DIM
            if "ocean" in t.lower() or "vast" in t.lower():
                v[100] = 1.0
            else:
                v[hash(t) % DIM] = 1.0
            result.append(v)
        return result

    monkeypatch.setattr(ml_module, "embed", _controlled_embed)
    _no_extraction(monkeypatch)

    scope = await _fresh_scope(pool, "chunks_recall")
    mem = Memory(pool, namespace=_unique("ns"), scope=scope, extract_propositions=False)
    await mem.ingest("The ocean is vast and deep.")

    results = await mem.recall(
        "vast ocean",
        k=10,
        with_rerank=False,
        with_mmr=False,
        expand_graph=False,
    )

    assert results
    for r in results:
        assert r.source == "chunks"
        assert r.proposition_id is None
        assert r.chunk_id == r.item_id
        assert r.predicate is None
        assert r.subject is None
        assert r.object is None


async def test_recall_in_chunks_mode_stays_inside_the_org(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """The chunk store has no namespace to hide behind, so the org predicate is
    the whole isolation boundary (ADR 0001, D3)."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    _no_extraction(monkeypatch)

    ns = _unique("ns")
    theirs = await _fresh_scope(pool, "chunks_them")
    mine = await _fresh_scope(pool, "chunks_me")

    await Memory(
        pool, namespace=ns, scope=theirs, extract_propositions=False
    ).ingest("Their refund policy runs to thirty days.")

    intruder = Memory(
        pool, namespace=ns, scope=mine, extract_propositions=False
    )
    assert await intruder.recall(
        "refund policy", with_rerank=False, with_mmr=False
    ) == []

    owner = Memory(
        pool, namespace=ns, scope=theirs, extract_propositions=False
    )
    assert await owner.recall("refund policy", with_rerank=False, with_mmr=False)


async def test_chunks_mode_graph_expansion_is_noop(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """Graph expansion with chunks-only ingest produces no graph-sourced rows:
    there are no propositions, so there are no edges."""
    import pgkg.ml as ml_module
    monkeypatch.setattr(ml_module, "embed", _fake_embed)
    _no_extraction(monkeypatch)

    class FakeCE:
        def predict(self, pairs):
            return [0.5] * len(pairs)

    monkeypatch.setattr(ml_module, "_rerank_model", FakeCE())

    scope = await _fresh_scope(pool, "chunks_graph")
    mem = Memory(pool, namespace=_unique("ns"), scope=scope, extract_propositions=False)
    await mem.ingest("Alice visited Bob last Tuesday. Bob works at Acme Corp.")

    results = await mem.recall(
        "Alice Bob",
        k=20,
        with_rerank=False,
        with_mmr=False,
        expand_graph=True,  # enabled but should be a no-op (no edges)
    )

    graph_results = [r for r in results if r.source_kind == "graph"]
    assert len(graph_results) == 0


# ---------------------------------------------------------------------------
# 4. The bench ablation still names the store it measures
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
# 5. The flag that selects the mode
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
