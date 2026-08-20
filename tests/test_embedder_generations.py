"""Embedder generations: the registry, and the retrieval consequence.

The registry half is ordinary schema.  The half worth testing hard is what
happens during a cutover window, because that is the only time the design is
load-bearing: a partially backfilled second generation must add recall without
subtracting any, and it must be switchable off by a status flip rather than a
deploy.

Every assertion goes through pgkg_search(), the registry functions, or the
catalog.  Nothing here writes the primary dimension down — it is read from the
column, the property migration 012 established — while the second generation is
deliberately a different width, because a width difference is the only thing
that proves the parameterisation is real.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import asyncpg
import pytest

SECOND_GENERATION_DIM = 768

QUERYABLE_STATUSES = ("live", "primary", "retiring")


def ns() -> str:
    return f"generation_{uuid.uuid4().hex[:10]}"


def pg_vec(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


def one_hot(dim: int, hot: int) -> list[float]:
    v = [0.0] * dim
    v[hot] = 1.0
    return v


def graded(dim: int, *values: float) -> list[float]:
    v = [0.0] * dim
    for i, value in enumerate(values):
        v[i] = value
    return v


async def primary_dim(conn: asyncpg.Connection) -> int:
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


async def register_generation(
    conn: asyncpg.Connection,
    *,
    name: str,
    dim: int,
    status: str = "live",
    normalize: bool = True,
    query_prefix: str | None = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO embedder_generations
            (name, dim, storage_type, normalize, query_prefix, status)
        VALUES ($1, $2, 'halfvec', $3, $4, $5)
        RETURNING id
        """,
        name,
        dim,
        normalize,
        query_prefix,
        status,
    )


async def side_table(
    conn: asyncpg.Connection, generation_id: uuid.UUID, source: str = "prop"
) -> str:
    return await conn.fetchval(
        "SELECT pgkg_create_generation_storage($1, $2)", generation_id, source
    )


async def store_side_vector(
    conn: asyncpg.Connection,
    *,
    table: str,
    item_id: uuid.UUID,
    vec: list[float],
) -> None:
    await conn.execute(
        f"INSERT INTO {table} (item_id, vec) VALUES ($1, '{pg_vec(vec)}')",
        item_id,
    )


async def bind_org(
    conn: asyncpg.Connection,
    *,
    generation_id: uuid.UUID,
    role: str = "secondary",
    org_id: uuid.UUID | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO org_embedders (org_id, generation_id, role)
        VALUES (COALESCE($1, pgkg_default_org()), $2, $3)
        """,
        org_id,
        generation_id,
        role,
    )


async def search(
    conn: asyncpg.Connection,
    *,
    namespace: str,
    q_embedding: list[float] | None,
    gen_queries: tuple[tuple[uuid.UUID, list[float]], ...] = (),
    org_ids: list[uuid.UUID] | None = None,
) -> dict[uuid.UUID, asyncpg.Record]:
    q_sql = (
        f"'{pg_vec(q_embedding)}'::halfvec"
        if q_embedding is not None
        else "NULL::halfvec"
    )
    if gen_queries:
        elements = ", ".join(
            f"ROW('{gen_id}'::UUID, '{pg_vec(v)}'::halfvec)::pgkg_gen_query"
            for gen_id, v in gen_queries
        )
        gen_sql = f"ARRAY[{elements}]"
    else:
        gen_sql = "NULL::pgkg_gen_query[]"

    rows = await conn.fetch(
        f"""
        SELECT proposition_id, text, rrf_score, source_kind
        FROM pgkg_search(
            q_text := NULL,
            q_embedding := {q_sql},
            p_namespace := $1::TEXT,
            p_org_ids := $2::UUID[],
            p_gen_queries := {gen_sql}
        )
        """,
        namespace,
        org_ids,
    )
    return {r["proposition_id"]: r for r in rows}


@dataclass(frozen=True)
class Cutover:
    """A namespace mid-backfill: three items in both generations, one only in
    the old one, one only in the new one."""

    namespace: str
    generation_id: uuid.UUID
    table: str
    both: uuid.UUID
    old_only: uuid.UUID
    new_only: uuid.UUID
    q_old: list[float]
    q_new: list[float]


async def cutover(
    conn: asyncpg.Connection,
    *,
    status: str = "live",
    bind: bool = True,
) -> Cutover:
    namespace = ns()
    old_dim = await primary_dim(conn)
    assert SECOND_GENERATION_DIM != old_dim, (
        "the second generation must be a different width for this suite to "
        "prove anything"
    )

    both = await insert_proposition(
        conn, text="in both generations", namespace=namespace,
        embedding=one_hot(old_dim, 2),
    )
    old_only = await insert_proposition(
        conn, text="only in the old generation", namespace=namespace,
        embedding=one_hot(old_dim, 1),
    )
    new_only = await insert_proposition(
        conn, text="only in the new generation", namespace=namespace,
        embedding=None,
    )

    generation_id = await register_generation(
        conn, name=f"bge-m3@{SECOND_GENERATION_DIM}",
        dim=SECOND_GENERATION_DIM, status=status,
    )
    table = await side_table(conn, generation_id)
    if bind:
        await bind_org(conn, generation_id=generation_id)

    await store_side_vector(
        conn, table=table, item_id=new_only,
        vec=one_hot(SECOND_GENERATION_DIM, 0),
    )
    await store_side_vector(
        conn, table=table, item_id=both,
        vec=one_hot(SECOND_GENERATION_DIM, 1),
    )

    return Cutover(
        namespace=namespace,
        generation_id=generation_id,
        table=table,
        both=both,
        old_only=old_only,
        new_only=new_only,
        q_old=graded(old_dim, 0.0, 1.0, 0.5),
        q_new=graded(SECOND_GENERATION_DIM, 1.0, 0.5),
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


async def test_current_model_is_registered_as_generation_one(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT g.name, g.dim, g.storage_type, g.normalize, g.status
            FROM embedder_generations g
            WHERE g.id = pgkg_generation_1()
            """
        )
        declared = await primary_dim(conn)

    assert row["name"] == "bge-m3"
    assert row["dim"] == declared
    assert row["storage_type"] == "halfvec"
    assert row["normalize"] is True
    assert row["status"] == "primary"


async def test_every_org_is_bound_to_generation_one_as_primary(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        unbound = await conn.fetch(
            """
            SELECT o.id
            FROM orgs o
            WHERE NOT EXISTS (
                SELECT 1 FROM org_embedders oe
                WHERE oe.org_id = o.id
                  AND oe.generation_id = pgkg_generation_1()
                  AND oe.role = 'primary'
            )
            """
        )

    assert unbound == []


async def test_generation_status_vocabulary_is_closed(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await register_generation(
                conn, name="nonsense", dim=64, status="warming-up"
            )


async def test_embedder_role_vocabulary_is_closed(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        generation_id = await register_generation(
            conn, name=f"role-{uuid.uuid4().hex[:8]}", dim=64
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await bind_org(conn, generation_id=generation_id, role="tertiary")


async def test_an_org_has_at_most_one_primary_embedder(
    pool: asyncpg.Pool,
) -> None:
    """Two primaries means two inline vector widths on one column, which the
    schema cannot represent — so the registry refuses it rather than letting a
    cutover half-happen."""
    async with pool.acquire() as conn:
        generation_id = await register_generation(
            conn, name=f"rival-{uuid.uuid4().hex[:8]}", dim=64
        )
        with pytest.raises(asyncpg.UniqueViolationError):
            await bind_org(conn, generation_id=generation_id, role="primary")


async def test_generation_width_beyond_the_halfvec_index_ceiling_is_rejected(
    pool: asyncpg.Pool,
) -> None:
    """halfvec is HNSW-indexable to 4000 dimensions; a generation wider than
    that could be registered but never retrieved (ADR 0001, D7)."""
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await register_generation(conn, name="too-wide", dim=4096)


async def test_existing_content_rows_carry_generation_one(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        namespace = ns()
        prop_id = await insert_proposition(
            conn, text="a fact without a stated generation", namespace=namespace
        )
        prop_generation = await conn.fetchval(
            "SELECT embedder_generation_id FROM propositions WHERE id = $1",
            prop_id,
        )
        chunk_generation = await conn.fetchval(
            """
            INSERT INTO chunks (text)
            VALUES ('a chunk without a stated generation')
            RETURNING embedder_generation_id
            """
        )
        generation_1 = await conn.fetchval("SELECT pgkg_generation_1()")

    assert prop_generation == generation_1
    assert chunk_generation == generation_1


@pytest.mark.parametrize("status", QUERYABLE_STATUSES)
async def test_live_generations_lists_queryable_statuses(
    pool: asyncpg.Pool, status: str
) -> None:
    async with pool.acquire() as conn:
        generation_id = await register_generation(
            conn, name=f"listed-{uuid.uuid4().hex[:8]}", dim=64, status=status
        )
        await bind_org(conn, generation_id=generation_id)
        listed = await conn.fetch(
            """
            SELECT generation_id FROM pgkg_live_generations(pgkg_default_org())
            """
        )

    assert generation_id in {r["generation_id"] for r in listed}


@pytest.mark.parametrize("status", ["building", "retired"])
async def test_live_generations_omits_unqueryable_statuses(
    pool: asyncpg.Pool, status: str
) -> None:
    async with pool.acquire() as conn:
        generation_id = await register_generation(
            conn, name=f"hidden-{uuid.uuid4().hex[:8]}", dim=64, status=status
        )
        await bind_org(conn, generation_id=generation_id)
        listed = await conn.fetch(
            """
            SELECT generation_id FROM pgkg_live_generations(pgkg_default_org())
            """
        )

    assert generation_id not in {r["generation_id"] for r in listed}


async def test_live_generations_carries_what_the_caller_needs_to_embed(
    pool: asyncpg.Pool,
) -> None:
    """The query prefix and normalisation flag travel with the generation, not
    with the application config, because a dual window runs two of them."""
    async with pool.acquire() as conn:
        generation_id = await register_generation(
            conn, name=f"asymmetric-{uuid.uuid4().hex[:8]}", dim=64,
            normalize=False, query_prefix="query: ",
        )
        await bind_org(conn, generation_id=generation_id)
        row = await conn.fetchrow(
            """
            SELECT name, dim, "normalize", query_prefix, role
            FROM pgkg_live_generations(pgkg_default_org())
            WHERE generation_id = $1
            """,
            generation_id,
        )

    assert row["dim"] == 64
    assert row["normalize"] is False
    assert row["query_prefix"] == "query: "
    assert row["role"] == "secondary"


# ---------------------------------------------------------------------------
# The transitional side table
# ---------------------------------------------------------------------------


async def test_side_table_is_created_at_the_generation_width(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        generation_id = await register_generation(
            conn, name=f"sided-{uuid.uuid4().hex[:8]}",
            dim=SECOND_GENERATION_DIM,
        )
        table = await side_table(conn, generation_id)
        declared = await conn.fetchval(
            "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
            "WHERE a.attrelid = $1::regclass AND a.attname = 'vec'",
            table,
        )
        key = await conn.fetchval(
            """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid
                               AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = $1::regclass AND i.indisprimary
            """,
            table,
        )

    assert declared == f"halfvec({SECOND_GENERATION_DIM})"
    assert key == "item_id"


async def test_side_table_has_its_own_hnsw_cosine_index(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        generation_id = await register_generation(
            conn, name=f"indexed-{uuid.uuid4().hex[:8]}",
            dim=SECOND_GENERATION_DIM,
        )
        table = await side_table(conn, generation_id)
        row = await conn.fetchrow(
            """
            SELECT am.amname, opc.opcname
            FROM pg_index i
            JOIN pg_class ic ON ic.oid = i.indexrelid
            JOIN pg_am am ON am.oid = ic.relam
            JOIN pg_opclass opc ON opc.oid = i.indclass[0]
            WHERE i.indrelid = $1::regclass AND NOT i.indisprimary
            """,
            table,
        )

    assert row["amname"] == "hnsw"
    assert row["opcname"] == "halfvec_cosine_ops"


async def test_side_table_row_dies_with_its_item(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        state = await cutover(conn)
        await conn.execute(
            "DELETE FROM propositions WHERE id = $1", state.new_only
        )
        remaining = await conn.fetchval(
            f"SELECT count(*) FROM {state.table} WHERE item_id = $1",
            state.new_only,
        )

    assert remaining == 0


async def test_chunk_source_storage_references_chunks(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        generation_id = await register_generation(
            conn, name=f"chunked-{uuid.uuid4().hex[:8]}",
            dim=SECOND_GENERATION_DIM,
        )
        table = await side_table(conn, generation_id, source="chunk")
        referenced = await conn.fetchval(
            """
            SELECT confrelid::regclass::TEXT
            FROM pg_constraint
            WHERE conrelid = $1::regclass AND contype = 'f'
            """,
            table,
        )

    assert referenced == "chunks"


async def test_unknown_embedding_source_is_rejected(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        generation_id = await register_generation(
            conn, name=f"nowhere-{uuid.uuid4().hex[:8]}", dim=64
        )
        with pytest.raises(asyncpg.PostgresError):
            await side_table(conn, generation_id, source="sentence")


# ---------------------------------------------------------------------------
# Retrieval across two generations
# ---------------------------------------------------------------------------


async def test_second_generation_adds_a_candidate_source(
    pool: asyncpg.Pool,
) -> None:
    """An item embedded only in the new generation is unreachable through the
    old one and reachable through the new one, which is what "another RRF
    source" has to mean."""
    async with pool.acquire() as conn:
        state = await cutover(conn)
        old_only_search = await search(
            conn, namespace=state.namespace, q_embedding=state.q_old
        )
        both_generations = await search(
            conn,
            namespace=state.namespace,
            q_embedding=state.q_old,
            gen_queries=((state.generation_id, state.q_new),),
        )

    assert state.new_only not in old_only_search
    assert state.new_only in both_generations
    assert both_generations[state.new_only]["source_kind"] == "vec"


async def test_an_item_in_both_generations_gains_from_the_second_vote(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        state = await cutover(conn)
        one = await search(
            conn, namespace=state.namespace, q_embedding=state.q_old
        )
        two = await search(
            conn,
            namespace=state.namespace,
            q_embedding=state.q_old,
            gen_queries=((state.generation_id, state.q_new),),
        )

    assert two[state.both]["rrf_score"] > one[state.both]["rrf_score"]


async def test_an_item_missing_from_the_new_generation_keeps_its_old_rank(
    pool: asyncpg.Pool,
) -> None:
    """The graceful-degradation clause: a partial backfill must not cost recall
    or rank for anything it has not reached yet."""
    async with pool.acquire() as conn:
        state = await cutover(conn)
        one = await search(
            conn, namespace=state.namespace, q_embedding=state.q_old
        )
        two = await search(
            conn,
            namespace=state.namespace,
            q_embedding=state.q_old,
            gen_queries=((state.generation_id, state.q_new),),
        )

    assert state.old_only in two
    assert two[state.old_only]["rrf_score"] == pytest.approx(
        one[state.old_only]["rrf_score"]
    )


@pytest.mark.parametrize("status", ["building", "retired"])
async def test_a_generation_below_live_contributes_nothing(
    pool: asyncpg.Pool, status: str
) -> None:
    """Every step of the cutover protocol rolls back by flipping a status, so
    the status has to be what retrieval reads."""
    async with pool.acquire() as conn:
        state = await cutover(conn, status=status)
        results = await search(
            conn,
            namespace=state.namespace,
            q_embedding=state.q_old,
            gen_queries=((state.generation_id, state.q_new),),
        )

    assert state.new_only not in results


async def test_a_generation_the_org_is_not_bound_to_contributes_nothing(
    pool: asyncpg.Pool,
) -> None:
    """Generation is a property of the org, so an org that has not adopted one
    does not pay for it or read from it."""
    async with pool.acquire() as conn:
        state = await cutover(conn, bind=False)
        default_org = await conn.fetchval("SELECT pgkg_default_org()")
        unbound = await search(
            conn,
            namespace=state.namespace,
            q_embedding=state.q_old,
            gen_queries=((state.generation_id, state.q_new),),
            org_ids=[default_org],
        )

        await bind_org(conn, generation_id=state.generation_id)
        bound = await search(
            conn,
            namespace=state.namespace,
            q_embedding=state.q_old,
            gen_queries=((state.generation_id, state.q_new),),
            org_ids=[default_org],
        )

    assert state.new_only not in unbound
    assert state.new_only in bound


async def test_a_query_vector_of_the_wrong_width_is_refused(
    pool: asyncpg.Pool,
) -> None:
    """Silently skipping the arm would look like a completed backfill with
    catastrophic recall, and a dual window has more than one candidate
    generation to blame, so the error has to name the one at fault."""
    async with pool.acquire() as conn:
        state = await cutover(conn)
        with pytest.raises(asyncpg.PostgresError) as refusal:
            await search(
                conn,
                namespace=state.namespace,
                q_embedding=state.q_old,
                gen_queries=((state.generation_id, state.q_old),),
            )

    assert str(state.generation_id) in str(refusal.value)


async def test_generation_queries_are_the_last_search_parameter(
    pool: asyncpg.Pool,
) -> None:
    """Every existing caller binds pgkg_search positionally, so a new parameter
    that is not last silently mis-binds rather than failing."""
    async with pool.acquire() as conn:
        signatures = await conn.fetch(
            """
            SELECT pg_get_function_identity_arguments(p.oid) AS args
            FROM pg_proc p
            JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE p.proname = 'pgkg_search' AND n.nspname = 'public'
            """
        )

    assert len(signatures) == 1
    assert signatures[0]["args"].endswith("pgkg_gen_query[]")
