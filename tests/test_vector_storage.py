"""Integration tests for halfvec storage and the single-source dimension.

Nothing in this module writes the embedding dimension down: it is read from
the column through ``pgkg_embedding_dim()``, which is the property migration
012 exists to establish.  A test that hardcoded 1024 would be asserting the
opposite of the behaviour under test.
"""
from __future__ import annotations

import math
import uuid

import asyncpg
import pytest

PGKG_TABLES = ("propositions", "entities", "chunks", "documents")


def ns() -> str:
    return f"halfvec_{uuid.uuid4().hex[:10]}"


def pg_vec(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


def one_hot(dim: int, *, hot: int, value: float = 1.0) -> list[float]:
    v = [0.0] * dim
    v[hot] = value
    return v


async def embedding_dim(conn: asyncpg.Connection) -> int:
    return await conn.fetchval(
        "SELECT pgkg_embedding_dim('propositions', 'embedding')"
    )


async def insert_proposition(
    conn: asyncpg.Connection,
    *,
    text: str,
    namespace: str,
    embedding: list[float] | None = None,
) -> uuid.UUID:
    emb_expr = f"'{pg_vec(embedding)}'" if embedding is not None else "NULL"
    return await conn.fetchval(
        f"""
        INSERT INTO propositions (text, namespace, embedding)
        VALUES ($1, $2, {emb_expr})
        RETURNING id
        """,
        text,
        namespace,
    )


# ---------------------------------------------------------------------------
# Storage type and index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["propositions", "entities"])
async def test_embedding_column_is_halfvec(pool: asyncpg.Pool, table: str) -> None:
    async with pool.acquire() as conn:
        declared = await conn.fetchval(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            WHERE a.attrelid = $1::regclass
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
            """,
            table,
        )

    assert declared.startswith("halfvec(")


@pytest.mark.parametrize(
    "index_name", ["prop_emb_idx", "entities_embedding_idx"]
)
async def test_embedding_index_is_hnsw_over_halfvec_cosine(
    pool: asyncpg.Pool, index_name: str
) -> None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT am.amname, opc.opcname
            FROM pg_index x
            JOIN pg_class i ON i.oid = x.indexrelid
            JOIN pg_am am ON am.oid = i.relam
            JOIN pg_opclass opc ON opc.oid = x.indclass[0]
            WHERE i.relname = $1
            """,
            index_name,
        )

    assert row is not None
    assert row["amname"] == "hnsw"
    assert row["opcname"] == "halfvec_cosine_ops"


# ---------------------------------------------------------------------------
# One declaration of the dimension
# ---------------------------------------------------------------------------


async def test_every_embedding_column_agrees_on_the_dimension(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        prop_dim = await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )
        entity_dim = await conn.fetchval(
            "SELECT pgkg_embedding_dim('entities', 'embedding')"
        )

    assert prop_dim > 0
    assert prop_dim == entity_dim


async def test_no_pgkg_table_stores_fp32_vectors(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        remaining = await conn.fetch(
            """
            SELECT c.relname, a.attname
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = ANY($1)
              AND a.attnum > 0
              AND NOT a.attisdropped
              AND a.atttypid = 'vector'::regtype
            """,
            list(PGKG_TABLES),
        )

    assert remaining == []


@pytest.mark.parametrize(
    "function_name",
    ["pgkg_search", "pgkg_vector_candidates", "pgkg_link_entity"],
)
async def test_query_embedding_parameter_is_an_unmodified_halfvec(
    pool: asyncpg.Pool, function_name: str
) -> None:
    """A dimension in a function signature is discarded by Postgres, so the
    parameter must not pretend to declare one — and the fp32 overload must be
    gone, or a NULL-embedding call becomes ambiguous."""
    async with pool.acquire() as conn:
        signatures = await conn.fetch(
            """
            SELECT pg_get_function_identity_arguments(p.oid) AS args
            FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE p.proname = $1
              AND n.nspname = 'public'
            """,
            function_name,
        )

    assert len(signatures) == 1
    args = signatures[0]["args"]
    assert "halfvec" in args
    assert "vector" not in args


# ---------------------------------------------------------------------------
# Retrieval behaviour is unchanged
# ---------------------------------------------------------------------------


async def test_cosine_ordering_survives_halfvec_storage(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        target = one_hot(dim, hot=0)
        diagonal = [0.0] * dim
        diagonal[0] = 1.0 / math.sqrt(2.0)
        diagonal[1] = 1.0 / math.sqrt(2.0)
        orthogonal = one_hot(dim, hot=1)

        await insert_proposition(
            conn, text="identical", namespace=namespace, embedding=target
        )
        await insert_proposition(
            conn, text="diagonal", namespace=namespace, embedding=diagonal
        )
        await insert_proposition(
            conn, text="orthogonal", namespace=namespace, embedding=orthogonal
        )

        rows = await conn.fetch(
            f"""
            SELECT p.text, c.rank, c.raw_score
            FROM pgkg_vector_candidates('{pg_vec(target)}', $1) c
            JOIN propositions p ON p.id = c.item_id
            ORDER BY c.rank
            """,
            namespace,
        )

    assert [r["text"] for r in rows] == ["identical", "diagonal", "orthogonal"]
    scores = [r["raw_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-3)
    assert scores[1] == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-3)
    assert scores[2] == pytest.approx(0.0, abs=1e-3)


async def test_search_returns_the_stored_embedding_to_the_client(
    pool: asyncpg.Pool,
) -> None:
    """The wire type stays fp32: widening halfvec is exact, and every client
    codec already handles it."""
    namespace = ns()
    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        target = one_hot(dim, hot=3)

        await insert_proposition(
            conn, text="a stored fact", namespace=namespace, embedding=target
        )

        row = await conn.fetchrow(
            f"""
            SELECT text, embedding
            FROM pgkg_search(NULL, '{pg_vec(target)}'::vector, 10, 20, $1)
            """,
            namespace,
        )

    assert row["text"] == "a stored fact"
    assert list(row["embedding"]) == pytest.approx(target)


async def test_link_entity_accepts_an_fp32_query_vector(pool: asyncpg.Pool) -> None:
    """Callers still bind fp32 literals; the implicit widening cast is what
    keeps every existing call site working."""
    namespace = ns()
    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        emb = one_hot(dim, hot=5)

        created = await conn.fetchval(
            f"SELECT pgkg_link_entity($1, 'Ada Lovelace', 'person', '{pg_vec(emb)}'::vector)",
            namespace,
        )
        again = await conn.fetchval(
            f"SELECT pgkg_link_entity($1, 'Ada Lovelace', 'person', '{pg_vec(emb)}'::vector)",
            namespace,
        )
        stored = await conn.fetchval(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'entities'::regclass AND attname = 'embedding'"
        )

    assert created is not None
    assert again == created
    assert stored.startswith("halfvec(")


# ---------------------------------------------------------------------------
# The dimension is enforced
# ---------------------------------------------------------------------------


async def test_wrong_dimension_is_rejected_on_write(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        too_wide = one_hot(dim + 1, hot=0)

        with pytest.raises(asyncpg.PostgresError) as excinfo:
            await insert_proposition(
                conn, text="too wide", namespace=namespace, embedding=too_wide
            )

    assert "dimension" in str(excinfo.value).lower()


async def test_wrong_dimension_query_is_rejected(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        await insert_proposition(
            conn,
            text="a fact worth finding",
            namespace=namespace,
            embedding=one_hot(dim, hot=0),
        )

        with pytest.raises(asyncpg.PostgresError) as excinfo:
            await conn.fetch(
                "SELECT * FROM pgkg_search(NULL, '[1,2,3]'::vector, 10, 20, $1)",
                namespace,
            )

    assert "dimension" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# The forward path to a different-width generation (ADR 0001, D8)
# ---------------------------------------------------------------------------


async def test_helper_converts_a_column_at_another_width(pool: asyncpg.Pool) -> None:
    table = f"pgkg_dimprobe_{uuid.uuid4().hex[:8]}"
    async with pool.acquire() as conn:
        await conn.execute(f"CREATE TABLE {table} (id INT PRIMARY KEY, vec vector(8))")
        try:
            await conn.execute(f"INSERT INTO {table} VALUES (1, '[1,0,0,0,0,0,0,0]')")

            await conn.execute(
                "SELECT pgkg_set_embedding_storage($1, 'vec', $2)",
                table,
                f"{table}_vec_idx",
            )

            declared = await conn.fetchval(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = $1::regclass AND attname = 'vec'",
                table,
            )
            dim = await conn.fetchval(
                "SELECT pgkg_embedding_dim($1, 'vec')", table
            )
            index = await conn.fetchrow(
                """
                SELECT am.amname, opc.opcname
                FROM pg_index x
                JOIN pg_class i ON i.oid = x.indexrelid
                JOIN pg_am am ON am.oid = i.relam
                JOIN pg_opclass opc ON opc.oid = x.indclass[0]
                WHERE i.relname = $1
                """,
                f"{table}_vec_idx",
            )
            preserved = await conn.fetchval(f"SELECT vec::text FROM {table} WHERE id = 1")
        finally:
            await conn.execute(f"DROP TABLE {table}")

    assert declared == "halfvec(8)"
    assert dim == 8
    assert index["amname"] == "hnsw"
    assert index["opcname"] == "halfvec_cosine_ops"
    assert preserved == "[1,0,0,0,0,0,0,0]"


async def test_helper_requires_a_dimension_when_the_column_declares_none(
    pool: asyncpg.Pool,
) -> None:
    table = f"pgkg_dimprobe_{uuid.uuid4().hex[:8]}"
    async with pool.acquire() as conn:
        await conn.execute(f"CREATE TABLE {table} (id INT PRIMARY KEY, vec vector)")
        try:
            with pytest.raises(asyncpg.PostgresError):
                await conn.execute(
                    "SELECT pgkg_set_embedding_storage($1, 'vec', NULL)", table
                )

            await conn.execute(
                "SELECT pgkg_set_embedding_storage($1, 'vec', NULL, 8)", table
            )
            declared = await conn.fetchval(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = $1::regclass AND attname = 'vec'",
                table,
            )
        finally:
            await conn.execute(f"DROP TABLE {table}")

    assert declared == "halfvec(8)"
