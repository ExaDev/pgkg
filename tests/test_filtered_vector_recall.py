"""A scoped vector search must return the k it was asked for.

D3: "pgvector's HNSW does not know about the WHERE clause.  It walks the graph
for the nearest ef_search neighbours globally, then the executor discards rows
failing the filter.  With one org at 0.1% of the table, a top-20 query either
under-returns or degrades to a sequential scan.  This is the classic
multi-tenant pgvector failure and it arrives long before storage does."

Partitioning is the other half of the mitigation and is deferred to phase 4, so
until then hnsw.iterative_scan is the whole of it.  The failure is silent —
no error, just missing rows — and it gets worse with every tenant added, which
is why it needs a test rather than a runbook entry.
"""
from __future__ import annotations

import uuid

import asyncpg
import pytest

import pgkg.db

DIM = 1024
ORGS = 8
ROWS_PER_ORG = 40


def pg_vec(hot_index: int, spread: float = 0.0) -> str:
    """A vector on the shared axis, pushed off it by `spread`.

    Every row shares component 0 with the query and differs on a component of
    its own, so cosine similarity falls monotonically as `spread` grows and no
    two rows tie.  Ties are what made an earlier version of this module flaky:
    with every row equidistant from the query, which ones HNSW walked to first
    was arbitrary, and so was the count that survived the scope filter.
    """
    v = ["0.0"] * DIM
    v[0] = "1.0"
    if spread:
        v[hot_index % DIM] = f"{spread:.6f}"
    return "[" + ",".join(v) + "]"


def ns() -> str:
    return f"filtered_vec_{uuid.uuid4().hex[:10]}"


async def test_a_pooled_connection_configures_iterative_scan(
    pool: asyncpg.Pool, pg_dsn: str
) -> None:
    """The setting is per-connection, so it belongs in the pool's init.

    The pool fixture is requested only to guarantee the migrations have run
    against this DSN before pgkg.db builds a pool of its own.
    """
    pool = await pgkg.db.make_pool(pg_dsn)
    try:
        async with pool.acquire() as conn:
            setting = await conn.fetchval("SHOW hnsw.iterative_scan")
    finally:
        await pgkg.db.close_pool(pool)

    assert setting == "strict_order"


async def _seed(conn: asyncpg.Connection, namespace: str) -> list[uuid.UUID]:
    orgs = [
        await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
            f"vec_{uuid.uuid4().hex[:8]}",
        )
        for _ in range(ORGS)
    ]
    # Distance is assigned round-robin across the orgs, so the tenant under
    # test holds every eighth position in the global ordering — the shape D3
    # describes, one tenant at a small share of an index the others crowd,
    # with its rows spread through the ordering rather than clustered.
    #
    # The spreads are all distinct, which is what makes the measurement
    # repeatable: an earlier version gave every row the same distance from the
    # query, so which ones HNSW walked to first was arbitrary and so was the
    # count that survived the scope filter.
    for slot, org in enumerate(orgs):
        for i in range(ROWS_PER_ORG):
            spread = (i * ORGS + slot + 1) * 0.002
            hot = slot * ROWS_PER_ORG + i + 1
            await conn.execute(
                f"""
                INSERT INTO propositions (text, namespace, org_id, embedding)
                VALUES ($1, $2, $3, '{pg_vec(hot, spread)}')
                """,
                f"row {slot}-{i}",
                namespace,
                org,
            )
    await conn.execute("ANALYZE propositions")
    return orgs


async def _scoped_candidates(
    conn: asyncpg.Connection,
    *,
    namespace: str,
    org: uuid.UUID,
    iterative_scan: str,
) -> int:
    """Count what the vector arm returns with the exact paths priced out, so
    the vector index is the only one left — the plan a real corpus reaches on
    size alone."""
    async with conn.transaction():
        await conn.execute("SET LOCAL enable_seqscan = off")
        await conn.execute("SET LOCAL enable_sort = off")
        await conn.execute("SET LOCAL enable_bitmapscan = off")
        await conn.execute(
            f"SET LOCAL hnsw.iterative_scan = '{iterative_scan}'"
        )
        rows = await conn.fetch(
            f"""
            SELECT item_id FROM pgkg_vector_candidates(
                '{pg_vec(0)}'::halfvec, $1, NULL, $2, $3::uuid[],
                NULL, NULL, NULL
            )
            """,
            namespace,
            ROWS_PER_ORG,
            [org],
        )
    return len(rows)


async def test_a_scoped_vector_search_under_returns(pool: asyncpg.Pool) -> None:
    """The D3 failure, measured: ask for k, get a fraction of k, no error.

    Eight tenants share the index and the one under test holds every eighth
    position in the distance ordering, so a top-`ROWS_PER_ORG` walk reaches
    only about an eighth of its rows and the executor discards the rest.  The
    count is exact and repeatable because no two rows tie on distance.

    The planner hints price out the exact paths, leaving the vector index as
    the only way in — the plan a real corpus reaches on size alone, and the
    only plan in which this failure exists.

    What this does NOT yet assert is that `hnsw.iterative_scan` closes the
    gap.  Measured against these rows it does not: `off` and `strict_order`
    return the identical count, because the scan sits under the Merge Append
    that `pgkg_vector_candidates`' two source arms produce and the iterative
    scan does not resume across it.  The mitigation named in this module's
    docstring is therefore unproven, and partitioning — deferred to phase 4 —
    is the half that is known to work.  Asserting the under-return keeps the
    failure itself pinned while that is resolved.
    """
    namespace = ns()
    async with pool.acquire() as conn:
        orgs = await _seed(conn, namespace)
        mine = orgs[0]

        default = await _scoped_candidates(
            conn, namespace=namespace, org=mine, iterative_scan="off"
        )

    assert 0 < default < ROWS_PER_ORG, (
        f"asked for {ROWS_PER_ORG} rows of one tenant and got {default}; if "
        "this is now equal to the ask, the dataset no longer provokes "
        "filtered under-return and this test proves nothing"
    )


async def test_the_setting_survives_a_connection_going_back_to_the_pool(
    pool: asyncpg.Pool, pg_dsn: str
) -> None:
    """The mitigation has to hold for every acquire, not just the first.

    asyncpg issues RESET ALL when a connection is released, so a plain SET run
    from the pool's init callback is undone the moment that connection goes
    back — the second caller onwards silently gets the default and the
    under-return this whole module is about.  Reading the setting after a
    release is the only way to tell the two implementations apart; a freshly
    created connection reports the right answer either way.
    """
    pool = await pgkg.db.make_pool(pg_dsn)
    try:
        async with pool.acquire() as conn:
            first = await conn.fetchval("SHOW hnsw.iterative_scan")
        async with pool.acquire() as conn:
            after_release = await conn.fetchval("SHOW hnsw.iterative_scan")
    finally:
        await pgkg.db.close_pool(pool)

    assert first == "strict_order"
    assert after_release == "strict_order", (
        "the setting was lost when the connection was released, so every "
        "acquire after the first runs the unmitigated scan"
    )
