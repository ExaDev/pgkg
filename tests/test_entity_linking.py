"""Entity resolution under concurrency, and the branches nothing else reaches.

pgkg_link_entity() reads before it inserts, so two ingests that first mention
the same entity race: the loser's INSERT hits
entities_namespace_name_type_key.  Wrapping ingest in one transaction (ADR 0001
phase 0) widens that window from a single statement to a whole document, which
turns a rare interleaving into a routine one.  Resolution must therefore be
idempotent at the SQL level rather than only recoverable by the caller.
"""
from __future__ import annotations

import asyncio
import uuid

import asyncpg
import pytest
from pgvector import HalfVector


def _ns(tag: str) -> str:
    return f"link_{tag}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
async def embed_dim(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT pgkg_embedding_dim('entities', 'embedding')"
        )


def one_hot(dim: int, *, hot: int = 0) -> HalfVector:
    v = [0.0] * dim
    v[hot] = 1.0
    return HalfVector(v)


async def _wait_for_a_blocked_lock(conn: asyncpg.Connection) -> bool:
    for _ in range(500):
        if await conn.fetchval("SELECT COUNT(*) FROM pg_locks WHERE NOT granted"):
            return True
        await asyncio.sleep(0.01)
    return False


async def test_link_entity_yields_the_winning_row_when_a_concurrent_writer_inserts_first(
    pool: asyncpg.Pool, embed_dim: int
) -> None:
    """The loser of the create race must return the winner's id, not raise.

    The competitor holds an uncommitted insert of the same (namespace, name,
    type), so the call under test blocks on the unique index and then finds the
    row committed underneath it — exactly the interleaving that a caller-side
    retry exists to paper over.
    """
    namespace = _ns("race")
    competitor = await pool.acquire()
    other = await pool.acquire()
    try:
        tx = competitor.transaction()
        await tx.start()
        winner_id = await competitor.fetchval(
            """
            INSERT INTO entities (name, type, embedding, namespace)
            VALUES ('Ada Lovelace', 'concept', $1, $2)
            RETURNING id
            """,
            one_hot(embed_dim),
            namespace,
        )

        task = asyncio.create_task(
            other.fetchval(
                "SELECT pgkg_link_entity($1, 'Ada Lovelace', 'concept', $2)",
                namespace,
                one_hot(embed_dim),
            )
        )

        if not await _wait_for_a_blocked_lock(competitor):
            await tx.rollback()
            task.cancel()
            pytest.fail("pgkg_link_entity never blocked on the competing insert")

        await tx.commit()
        linked_id = await task
    finally:
        await pool.release(other)
        await pool.release(competitor)

    assert linked_id == winner_id

    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM entities WHERE namespace = $1", namespace
        )
    assert count == 1


async def test_link_entity_is_idempotent_under_parallel_first_mentions(
    pool: asyncpg.Pool, embed_dim: int
) -> None:
    """Four writers naming the same new entity at once agree on one id."""
    namespace = _ns("parallel")
    embedding = one_hot(embed_dim)

    async def link() -> uuid.UUID:
        async with pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT pgkg_link_entity($1, 'Analytical Engine', 'concept', $2)",
                namespace,
                embedding,
            )

    ids = await asyncio.gather(*(link() for _ in range(4)))

    assert len(set(ids)) == 1
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM entities WHERE namespace = $1", namespace
            )
            == 1
        )


async def test_link_entity_matches_an_untyped_row_when_no_type_is_given(
    pool: asyncpg.Pool, embed_dim: int
) -> None:
    """The (type IS NULL AND p_type IS NULL) branch resolves rather than
    inserting a second untyped row.

    Type only gates the exact-match stage: asking for the same name under a
    type, with no embedding to reach the similarity stage, is a different
    entity.
    """
    namespace = _ns("untyped")
    async with pool.acquire() as conn:
        created = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'untyped thing', NULL, $2)",
            namespace,
            one_hot(embed_dim),
        )
        again = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'untyped thing', NULL, $2)",
            namespace,
            one_hot(embed_dim),
        )
        typed = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'untyped thing', 'concept', NULL)",
            namespace,
        )

    assert again == created
    assert typed != created


async def test_link_entity_creates_without_an_embedding(
    pool: asyncpg.Pool, embed_dim: int
) -> None:
    """A NULL embedding skips the similarity stage; the exact-name stage still
    dedupes, so two calls yield one row."""
    namespace = _ns("noemb")
    async with pool.acquire() as conn:
        first = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'no vector here', 'concept', NULL)",
            namespace,
        )
        second = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'no vector here', 'concept', NULL)",
            namespace,
        )
        stored = await conn.fetchval(
            "SELECT embedding FROM entities WHERE id = $1", first
        )

    assert first == second
    assert stored is None


async def test_link_entity_default_threshold_rejects_a_distant_neighbour(
    pool: asyncpg.Pool, embed_dim: int
) -> None:
    """At the default 0.85 an orthogonal embedding is a different entity even
    when the names are near-identical, so the trigram stage cannot merge on
    spelling alone."""
    namespace = _ns("threshold")
    async with pool.acquire() as conn:
        original = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'William Shakespeare', 'concept', $2)",
            namespace,
            one_hot(embed_dim, hot=0),
        )
        distant = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'William Shakespear', 'concept', $2)",
            namespace,
            one_hot(embed_dim, hot=1),
        )
        near = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'William Shakespere', 'concept', $2)",
            namespace,
            one_hot(embed_dim, hot=0),
        )

    assert distant != original
    assert near == original
