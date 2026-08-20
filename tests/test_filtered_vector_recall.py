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


def pg_vec(hot_index: int) -> str:
    v = ["0.0"] * DIM
    v[hot_index % DIM] = "1.0"
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
    for slot, org in enumerate(orgs):
        for i in range(ROWS_PER_ORG):
            await conn.execute(
                f"""
                INSERT INTO propositions (text, namespace, org_id, embedding)
                VALUES ($1, $2, $3, '{pg_vec(slot * ROWS_PER_ORG + i)}')
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


async def test_iterative_scan_recovers_rows_the_default_silently_drops(
    pool: asyncpg.Pool,
) -> None:
    """Both settings measured against the same rows, because the size of the
    gap is the whole point: the default is not slower, it is wrong, and it
    reports no error while being wrong.

    Asserting the improvement rather than an exact count keeps this honest
    about what the setting is — a mitigation that scans further, bounded by
    hnsw.max_scan_tuples.  Partitioning, deferred to phase 4, is what makes
    the filter into pruning and removes the gap entirely.
    """
    namespace = ns()
    async with pool.acquire() as conn:
        orgs = await _seed(conn, namespace)
        mine = orgs[0]

        default = await _scoped_candidates(
            conn, namespace=namespace, org=mine, iterative_scan="off"
        )
        iterative = await _scoped_candidates(
            conn, namespace=namespace, org=mine, iterative_scan="strict_order"
        )

    assert default < ROWS_PER_ORG, (
        "the dataset no longer provokes filtered under-return, so this test "
        "proves nothing about the setting"
    )
    assert iterative > default, (
        f"iterative scan returned {iterative}, no better than the default's "
        f"{default}"
    )
    assert iterative >= ROWS_PER_ORG // 2
