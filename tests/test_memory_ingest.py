"""Write-path and read-path contracts for pgkg.memory.

Four defects from ADR 0001 "What the current implementation cannot carry" are
pinned here, plus the subject/object gap:

  * embeddings must travel as bound parameters, never as SQL text
  * ingest() must be one transaction — a failure leaves nothing behind
  * ingest() must issue a number of statements that does not grow with the
    number of propositions
  * recall() must not write access bookkeeping synchronously per row, while
    frequency weighting must still reach the ranking function
  * Result.subject / Result.object must carry the resolved entity names

The batch write path is checked against an oracle that reproduces the previous
per-row path statement for statement, so "identical rows" is a measurement
rather than an assertion about the new code alone.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from datetime import datetime, timezone

import asyncpg
import pytest
from pgvector import HalfVector

from pgkg import ml
from pgkg.chunking import chunk_document
from pgkg.memory import Memory
from pgkg.ml import Proposition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
async def embed_dim(pool: asyncpg.Pool) -> int:
    """The declared embedding width, read from the catalog rather than assumed."""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )


def _ns(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _make_embed(dim: int):
    """Deterministic one-hot embedding, unique per text."""

    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            v = [0.0] * dim
            v[int.from_bytes(digest[:4], "big") % dim] = 1.0
            out.append(v)
        return out

    return embed


def _make_constant_embed(dim: int):
    """Every text embeds to the same vector, so cosine similarity is always 1."""

    def embed(texts: list[str]) -> list[list[float]]:
        v = [0.0] * dim
        v[0] = 1.0
        return [list(v) for _ in texts]

    return embed


def _make_extractor(props_per_chunk: int):
    """Deterministic extraction: even-indexed propositions get an entity object
    (so an edge is written), odd-indexed ones get a literal object."""

    async def extract(chunk_text: str, *, max_propositions: int = 20, cache=None):
        tag = hashlib.sha256(chunk_text.encode()).hexdigest()[:6]
        return [
            Proposition(
                subject=f"subject {tag} {i}",
                predicate=f"relates-{i}",
                object=f"object {tag} {i}",
                object_is_literal=bool(i % 2),
            )
            for i in range(props_per_chunk)
        ]

    return extract


class _RecordingConnection:
    def __init__(self, conn, log: list[tuple[str, tuple]]) -> None:
        self._conn = conn
        self._log = log

    def __getattr__(self, name):
        return getattr(self._conn, name)

    async def execute(self, query, *args, **kwargs):
        self._log.append((query, args))
        return await self._conn.execute(query, *args, **kwargs)

    async def fetch(self, query, *args, **kwargs):
        self._log.append((query, args))
        return await self._conn.fetch(query, *args, **kwargs)

    async def fetchrow(self, query, *args, **kwargs):
        self._log.append((query, args))
        return await self._conn.fetchrow(query, *args, **kwargs)

    async def fetchval(self, query, *args, **kwargs):
        self._log.append((query, args))
        return await self._conn.fetchval(query, *args, **kwargs)


class _RecordingAcquire:
    def __init__(self, pool, log) -> None:
        self._pool = pool
        self._log = log
        self._ctx = None

    async def __aenter__(self):
        self._ctx = self._pool.acquire()
        conn = await self._ctx.__aenter__()
        return _RecordingConnection(conn, self._log)

    async def __aexit__(self, *exc):
        return await self._ctx.__aexit__(*exc)


class _RecordingPool:
    """Wraps a real pool and records every statement Memory sends."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self.log: list[tuple[str, tuple]] = []

    def acquire(self):
        return _RecordingAcquire(self._pool, self.log)

    @property
    def queries(self) -> list[str]:
        return [q for q, _ in self.log]


_NUMERIC_LITERAL_LIST = re.compile(r"\[\s*-?\d")


def _sql_literal_offenders(queries: list[str]) -> list[str]:
    return [q for q in queries if _NUMERIC_LITERAL_LIST.search(q)]


async def _ingest_per_row(
    pool: asyncpg.Pool,
    namespace: str,
    text: str,
    *,
    session_id: str | None = None,
    asserted_at: datetime | None = None,
) -> None:
    """Oracle: the pre-refactor per-row write path, reproduced statement for
    statement, including the vector-literal interpolation it used.

    Only the write path is the oracle; chunking is shared with the code under
    test, so the comparison isolates the batching change.
    """

    def vec_literal(emb: list[float]) -> str:
        return "[" + ",".join(str(v) for v in emb) + "]"

    chunks = chunk_document(text)

    async with pool.acquire() as conn:
        doc_id = await conn.fetchval(
            "INSERT INTO documents (source, namespace) VALUES ($1, $2) RETURNING id",
            None,
            namespace,
        )
        chunk_ids = []
        for chunk in chunks:
            chunk_ids.append(
                await conn.fetchval(
                    """
                    INSERT INTO chunks (document_id, text, span_start, span_end, asserted_at)
                    VALUES ($1, $2, $3, $4, $5) RETURNING id
                    """,
                    doc_id,
                    chunk.text,
                    chunk.span_start,
                    chunk.span_end,
                    asserted_at,
                )
            )

        for chunk_id, chunk in zip(chunk_ids, chunks):
            propositions = await ml.extract_propositions_async(chunk.text)
            if not propositions:
                continue

            entity_names: list[str] = []
            for prop in propositions:
                entity_names.append(prop.subject)
                if not prop.object_is_literal:
                    entity_names.append(prop.object)

            all_embs = ml.embed(entity_names + [p.text for p in propositions])
            entity_embs = all_embs[: len(entity_names)]
            prop_embs = all_embs[len(entity_names):]

            entity_idx = 0
            for prop, prop_emb in zip(propositions, prop_embs):
                subject_id = await conn.fetchval(
                    "SELECT pgkg_link_entity($1, $2, $3, "
                    f"'{vec_literal(entity_embs[entity_idx])}'::vector)",
                    namespace,
                    prop.subject,
                    "concept",
                )
                entity_idx += 1

                object_id = None
                object_literal = None
                if prop.object_is_literal:
                    object_literal = prop.object
                else:
                    object_id = await conn.fetchval(
                        "SELECT pgkg_link_entity($1, $2, $3, "
                        f"'{vec_literal(entity_embs[entity_idx])}'::vector)",
                        namespace,
                        prop.object,
                        "concept",
                    )
                    entity_idx += 1

                prop_id = await conn.fetchval(
                    f"""
                    INSERT INTO propositions
                        (text, embedding, subject_id, predicate, object_id,
                         object_literal, chunk_id, namespace, session_id, asserted_at)
                    VALUES ($1, '{vec_literal(prop_emb)}'::vector,
                            $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                    """,
                    prop.text,
                    subject_id,
                    prop.predicate,
                    object_id,
                    object_literal,
                    chunk_id,
                    namespace,
                    session_id,
                    asserted_at,
                )

                if object_id is not None:
                    await conn.execute(
                        """
                        INSERT INTO edges (src_entity, dst_entity, relation, proposition_id)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT DO NOTHING
                        """,
                        subject_id,
                        object_id,
                        prop.predicate,
                        prop_id,
                    )


async def _proposition_snapshot(pool: asyncpg.Pool, namespace: str) -> list[tuple]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT p.text, p.predicate, p.object_literal, p.session_id,
                   p.asserted_at, p.metadata, p.embedding::text AS emb,
                   s.name AS subject_name, o.name AS object_name,
                   c.text AS chunk_text, d.source AS doc_source
            FROM propositions p
            LEFT JOIN entities s ON s.id = p.subject_id
            LEFT JOIN entities o ON o.id = p.object_id
            LEFT JOIN chunks c ON c.id = p.chunk_id
            LEFT JOIN documents d ON d.id = c.document_id
            WHERE p.namespace = $1
            ORDER BY p.text, p.predicate
            """,
            namespace,
        )
    return [tuple(r.values()) for r in rows]


async def _edge_snapshot(pool: asyncpg.Pool, namespace: str) -> list[tuple]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT s.name, d.name, e.relation, e.weight, p.text
            FROM edges e
            JOIN entities s ON s.id = e.src_entity
            JOIN entities d ON d.id = e.dst_entity
            JOIN propositions p ON p.id = e.proposition_id
            WHERE p.namespace = $1
            ORDER BY s.name, d.name, e.relation, p.text
            """,
            namespace,
        )
    return [tuple(r.values()) for r in rows]


_DOC = (
    "Ada Lovelace wrote the first algorithm for the analytical engine.\n\n"
    "Charles Babbage designed the analytical engine in London.\n\n"
    "The engine was never completed during their lifetimes."
)


# ---------------------------------------------------------------------------
# 1. Transactional ingest
# ---------------------------------------------------------------------------

async def test_ingest_rolls_back_every_row_when_a_later_write_fails(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """ingest() is one transaction: a proposition insert that the database
    rejects must leave no document and no chunk behind."""
    monkeypatch.setattr(ml, "embed", lambda texts: [[0.5] * 4 for _ in texts])

    ns = _ns("rollback")
    mem = Memory(pool, namespace=ns, extract_propositions=False)

    with pytest.raises(Exception):
        await mem.ingest(_DOC)

    async with pool.acquire() as conn:
        docs = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE namespace = $1", ns
        )
        chunks = await conn.fetchval(
            """
            SELECT COUNT(*) FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.namespace = $1
            """,
            ns,
        )
        props = await conn.fetchval(
            "SELECT COUNT(*) FROM propositions WHERE namespace = $1", ns
        )

    assert (docs, chunks, props) == (0, 0, 0)


# ---------------------------------------------------------------------------
# 2. Parameter binding
# ---------------------------------------------------------------------------

async def test_ingest_never_puts_an_embedding_in_the_sql_text(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))
    monkeypatch.setattr(ml, "extract_propositions_async", _make_extractor(2))

    recording = _RecordingPool(pool)
    mem = Memory(recording, namespace=_ns("bind_ingest"))
    await mem.ingest(_DOC)

    assert _sql_literal_offenders(recording.queries) == []

    bound_arrays = [
        arg
        for _, args in recording.log
        for arg in args
        if isinstance(arg, list) and arg and isinstance(arg[0], HalfVector)
    ]
    assert bound_arrays, "embeddings should reach the server as bound arrays"
    assert all(len(v.to_list()) == embed_dim for vs in bound_arrays for v in vs)


async def test_recall_never_puts_the_query_vector_in_the_sql_text(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))
    monkeypatch.setattr(ml, "extract_propositions_async", _make_extractor(1))

    ns = _ns("bind_recall")
    await Memory(pool, namespace=ns).ingest(_DOC)

    recording = _RecordingPool(pool)
    mem = Memory(recording, namespace=ns)
    results = await mem.recall(
        "analytical engine", with_rerank=False, with_mmr=False
    )

    assert results
    assert _sql_literal_offenders(recording.queries) == []
    assert any(
        isinstance(arg, HalfVector) and len(arg.to_list()) == embed_dim
        for _, args in recording.log
        for arg in args
    ), "the query vector should be bound as a parameter"


# ---------------------------------------------------------------------------
# 3. Set-based ingest
# ---------------------------------------------------------------------------

async def test_batch_ingest_writes_the_same_rows_as_the_per_row_path(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """The batched write path must be observationally identical to the per-row
    path it replaces: same propositions, same entity resolution, same edges."""
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))
    monkeypatch.setattr(ml, "extract_propositions_async", _make_extractor(3))

    asserted = datetime(2021, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    ns_new = _ns("batch_new")
    ns_old = _ns("batch_old")

    await Memory(pool, namespace=ns_new).ingest(
        _DOC, session_id="s1", asserted_at=asserted
    )
    await _ingest_per_row(
        pool, ns_old, _DOC, session_id="s1", asserted_at=asserted
    )

    new_props = await _proposition_snapshot(pool, ns_new)
    old_props = await _proposition_snapshot(pool, ns_old)
    assert new_props
    assert new_props == old_props

    new_edges = await _edge_snapshot(pool, ns_new)
    old_edges = await _edge_snapshot(pool, ns_old)
    assert new_edges
    assert new_edges == old_edges


async def test_batch_ingest_reports_the_same_counts_as_the_per_row_path(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))
    monkeypatch.setattr(ml, "extract_propositions_async", _make_extractor(3))

    ns_new = _ns("counts_new")
    ns_old = _ns("counts_old")

    result = await Memory(pool, namespace=ns_new).ingest(_DOC)
    await _ingest_per_row(pool, ns_old, _DOC)

    async with pool.acquire() as conn:
        old_props = await conn.fetchval(
            "SELECT COUNT(*) FROM propositions WHERE namespace = $1", ns_old
        )
        old_entities = await conn.fetchval(
            "SELECT COUNT(*) FROM entities WHERE namespace = $1", ns_old
        )

    assert result.documents == 1
    assert result.propositions == old_props
    assert result.entities == old_entities


async def test_ingest_statement_count_does_not_grow_with_propositions(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """The write path is set-based: twenty propositions cost the same number of
    round trips as two."""
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    monkeypatch.setattr(ml, "extract_propositions_async", _make_extractor(2))
    small = _RecordingPool(pool)
    await Memory(small, namespace=_ns("rt_small")).ingest(_DOC)

    monkeypatch.setattr(ml, "extract_propositions_async", _make_extractor(20))
    large = _RecordingPool(pool)
    await Memory(large, namespace=_ns("rt_large")).ingest(_DOC)

    assert len(large.queries) == len(small.queries)
    assert len(small.queries) <= 8


async def test_ingest_statement_count_does_not_grow_with_chunks(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))
    monkeypatch.setattr(ml, "extract_propositions_async", _make_extractor(1))

    one = _RecordingPool(pool)
    await Memory(one, namespace=_ns("chunk_one")).ingest("A single paragraph.")

    many = _RecordingPool(pool)
    ns_many = _ns("chunk_many")
    long_document = "\n\n".join(
        f"Paragraph number {i} about analytical engines. " + ("filler words " * 40)
        for i in range(12)
    )
    result = await Memory(many, namespace=ns_many).ingest(long_document)

    assert result.chunks > 1, "the document must span several chunks for this to bite"
    assert len(many.queries) == len(one.queries)


async def test_batch_entity_linking_dedupes_near_duplicates_within_one_batch(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """Entity resolution inside one batched statement must still see entities
    created earlier in the same batch, or near-duplicate names would each get
    their own row."""
    monkeypatch.setattr(ml, "embed", _make_constant_embed(embed_dim))

    async def extract(chunk_text: str, *, max_propositions: int = 20, cache=None):
        return [
            Proposition(subject="William Shakespeare", predicate="wrote", object="Hamlet"),
            Proposition(subject="William Shakespear", predicate="wrote", object="Hamlet"),
        ]

    monkeypatch.setattr(ml, "extract_propositions_async", extract)

    ns = _ns("dedupe")
    result = await Memory(pool, namespace=ns).ingest("One paragraph.")

    async with pool.acquire() as conn:
        names = [
            r["name"]
            for r in await conn.fetch(
                "SELECT name FROM entities WHERE namespace = $1 ORDER BY name", ns
            )
        ]

    assert names == ["Hamlet", "William Shakespeare"]
    assert result.entities == 2


# ---------------------------------------------------------------------------
# 4. Access accounting off the read path
# ---------------------------------------------------------------------------

async def _access_counts(pool: asyncpg.Pool, namespace: str) -> list[int]:
    async with pool.acquire() as conn:
        return [
            r["access_count"]
            for r in await conn.fetch(
                "SELECT access_count FROM propositions WHERE namespace = $1 ORDER BY text",
                namespace,
            )
        ]


async def test_recall_does_not_write_access_counts_synchronously(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    ns = _ns("no_sync_bump")
    mem = Memory(pool, namespace=ns, extract_propositions=False, access_flush_interval=3600.0)
    await mem.ingest(_DOC)

    results = await mem.recall("analytical engine", with_rerank=False, with_mmr=False)
    assert results

    assert all(c == 0 for c in await _access_counts(pool, ns))


async def test_flush_access_applies_the_accumulated_counts(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    ns = _ns("flush_bump")
    mem = Memory(pool, namespace=ns, extract_propositions=False, access_flush_interval=3600.0)
    await mem.ingest(_DOC)

    seen: list[int] = []
    for _ in range(3):
        results = await mem.recall(
            "analytical engine", k=1, with_rerank=False, with_mmr=False
        )
        assert len(results) == 1
        seen.append(1)

    await mem.flush_access()

    counts = await _access_counts(pool, ns)
    assert sum(counts) == 3
    assert max(counts) == 3

    await mem.flush_access()
    assert sum(await _access_counts(pool, ns)) == 3


async def test_access_accounting_still_boosts_ranking(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """Frequency weighting must still reach pgkg_search: a proposition recalled
    once scores higher on the next recall."""
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    ns = _ns("freq_rank")
    mem = Memory(pool, namespace=ns, extract_propositions=False, access_flush_interval=3600.0)
    await mem.ingest("Ada Lovelace wrote the first algorithm.")

    first = await mem.recall("algorithm", k=1, with_rerank=False, with_mmr=False)
    assert len(first) == 1

    await mem.flush_access()

    second = await mem.recall("algorithm", k=1, with_rerank=False, with_mmr=False)
    assert len(second) == 1
    assert second[0].proposition_id == first[0].proposition_id
    assert second[0].score > first[0].score * 1.4


async def test_flush_access_keeps_pending_counts_when_the_write_fails(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """A failed flush must not silently drop the accounting it was carrying."""
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    ns = _ns("flush_fail")
    mem = Memory(pool, namespace=ns, extract_propositions=False, access_flush_interval=3600.0)
    await mem.ingest("Ada Lovelace wrote the first algorithm.")
    await mem.recall("algorithm", k=1, with_rerank=False, with_mmr=False)

    class _Boom:
        def acquire(self):
            raise RuntimeError("pool is gone")

    original_pool = mem._pool
    mem._pool = _Boom()
    with pytest.raises(RuntimeError):
        await mem.flush_access()
    mem._pool = original_pool

    await mem.flush_access()
    assert sum(await _access_counts(pool, ns)) == 1


# ---------------------------------------------------------------------------
# 5. Graph structure on the way out
# ---------------------------------------------------------------------------

async def test_recall_populates_subject_and_object_names(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    async def extract(chunk_text: str, *, max_propositions: int = 20, cache=None):
        return [
            Proposition(
                text="Ada Lovelace wrote the analytical engine notes",
                subject="Ada Lovelace",
                predicate="wrote",
                object="analytical engine notes",
            )
        ]

    monkeypatch.setattr(ml, "extract_propositions_async", extract)

    ns = _ns("spo")
    mem = Memory(pool, namespace=ns)
    await mem.ingest("Ada Lovelace wrote the analytical engine notes.")

    results = await mem.recall(
        "analytical engine notes", with_rerank=False, with_mmr=False
    )
    assert results
    row = next(r for r in results if r.predicate == "wrote")
    assert row.subject == "Ada Lovelace"
    assert row.object == "analytical engine notes"


async def test_recall_leaves_subject_and_object_none_for_chunk_rows(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    ns = _ns("spo_chunks")
    mem = Memory(pool, namespace=ns, extract_propositions=False)
    await mem.ingest("Ada Lovelace wrote the analytical engine notes.")

    results = await mem.recall(
        "analytical engine notes", with_rerank=False, with_mmr=False
    )
    assert results
    assert all(r.subject is None and r.object is None for r in results)



# ---------------------------------------------------------------------------
# 6. Concurrency: the transaction holds entity rows until commit
# ---------------------------------------------------------------------------

async def test_ingest_survives_a_concurrent_writer_creating_the_same_entity(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """Entity resolution is a read-then-insert, so two ingests that mention the
    same new entity race.  Wrapping ingest in a transaction widens that window
    from one statement to the whole document, so the loser must recover rather
    than surface a unique violation to the caller."""
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    async def extract(chunk_text: str, *, max_propositions: int = 20, cache=None):
        return [
            Proposition(subject="Ada Lovelace", predicate="wrote", object="algorithms")
        ]

    monkeypatch.setattr(ml, "extract_propositions_async", extract)

    ns = _ns("entity_race")
    competitor = await pool.acquire()
    try:
        tx = competitor.transaction()
        await tx.start()
        await competitor.execute(
            "INSERT INTO entities (name, type, embedding, namespace) "
            "VALUES ($1, 'concept', $2, $3)",
            "Ada Lovelace",
            HalfVector([0.0] * embed_dim),
            ns,
        )

        task = asyncio.create_task(
            Memory(pool, namespace=ns).ingest("Ada Lovelace wrote algorithms.")
        )

        # Wait until the ingest is genuinely blocked on the uncommitted row,
        # so the race is exercised rather than merely given the chance to occur.
        for _ in range(500):
            blocked = await competitor.fetchval(
                "SELECT COUNT(*) FROM pg_locks WHERE NOT granted"
            )
            if blocked:
                break
            await asyncio.sleep(0.01)
        else:
            await tx.rollback()
            task.cancel()
            pytest.fail("the ingest never blocked on the competing entity insert")

        await tx.commit()
        result = await task
    finally:
        await pool.release(competitor)

    assert result.propositions == 1

    async with pool.acquire() as conn:
        entities = await conn.fetchval(
            "SELECT COUNT(*) FROM entities WHERE namespace = $1 AND name = $2",
            ns,
            "Ada Lovelace",
        )
        docs = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE namespace = $1", ns
        )
    assert entities == 1
    assert docs == 1
