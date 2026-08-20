"""Centrality is computed per tenant, never globally.

D4's hard rule: "Ranking signals are never computed globally over shared
content.  pgkg_recompute_pagerank() must run per subscriber over that
subscriber's visible subgraph."  The out-degree term is the one that leaks:
divide a tenant's edge weight by a total that counted another tenant's edges
and that tenant's score now depends on data it cannot read, which is a real
cross-tenant inference channel rather than a rounding error.
"""
from __future__ import annotations

import uuid

import asyncpg

DIM = 1024


def pg_vec(hot_index: int = 0) -> str:
    v = [0.0] * DIM
    v[hot_index] = 1.0
    return "[" + ",".join(str(x) for x in v) + "]"


def ns() -> str:
    return f"pagerank_{uuid.uuid4().hex[:10]}"


async def new_org(conn: asyncpg.Connection, name: str) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", name
    )


async def entity(
    conn: asyncpg.Connection, *, name: str, namespace: str, org_id: uuid.UUID
) -> uuid.UUID:
    return await conn.fetchval(
        f"""
        INSERT INTO entities (name, type, namespace, org_id, embedding)
        VALUES ($1, 'concept', $2, $3, '{pg_vec()}')
        RETURNING id
        """,
        name,
        namespace,
        org_id,
    )


async def edge(
    conn: asyncpg.Connection,
    *,
    src: uuid.UUID,
    dst: uuid.UUID,
    namespace: str,
    org_id: uuid.UUID,
) -> None:
    prop = await conn.fetchval(
        "INSERT INTO propositions (text, namespace, org_id, subject_id, object_id)"
        " VALUES ('edge carrier', $1, $2, $3, $4) RETURNING id",
        namespace,
        org_id,
        src,
        dst,
    )
    await conn.execute(
        "INSERT INTO edges (src_entity, dst_entity, relation, proposition_id)"
        " VALUES ($1, $2, 'related_to', $3)",
        src,
        dst,
        prop,
    )


async def scores(
    conn: asyncpg.Connection, ids: list[uuid.UUID]
) -> dict[uuid.UUID, float]:
    rows = await conn.fetch(
        "SELECT entity_id, score FROM entity_pagerank WHERE entity_id = ANY($1::uuid[])",
        ids,
    )
    return {r["entity_id"]: r["score"] for r in rows}


async def test_another_tenants_edges_do_not_change_my_scores(
    pool: asyncpg.Pool,
) -> None:
    """The same subgraph, scored twice: once alone, once while a second tenant
    holds edges out of an entity of the same name.  The scores must match."""
    async with pool.acquire() as conn:
        clean_ns = ns()
        org_a = await new_org(conn, "a")
        a1 = await entity(conn, name="A1", namespace=clean_ns, org_id=org_a)
        a2 = await entity(conn, name="A2", namespace=clean_ns, org_id=org_a)
        await edge(conn, src=a1, dst=a2, namespace=clean_ns, org_id=org_a)
        await conn.execute(
            "SELECT pgkg_recompute_pagerank($1, 20, 0.85, $2)", clean_ns, org_a
        )
        alone = await scores(conn, [a1, a2])

        shared_ns = ns()
        org_b = await new_org(conn, "b")
        b1 = await entity(conn, name="A1", namespace=shared_ns, org_id=org_b)
        b2 = await entity(conn, name="A2", namespace=shared_ns, org_id=org_b)
        await edge(conn, src=b1, dst=b2, namespace=shared_ns, org_id=org_b)

        c1 = await entity(conn, name="A1", namespace=shared_ns, org_id=org_a)
        c2 = await entity(conn, name="A2", namespace=shared_ns, org_id=org_a)
        await edge(conn, src=c1, dst=c2, namespace=shared_ns, org_id=org_a)
        await conn.execute(
            "SELECT pgkg_recompute_pagerank($1, 20, 0.85, $2)", shared_ns, org_a
        )
        contended = await scores(conn, [c1, c2])

    assert contended[c2] == alone[a2], (
        "a second tenant's edges changed this tenant's centrality"
    )
    assert contended[c1] == alone[a1]


async def test_out_degree_ignores_edges_outside_the_namespace(
    pool: asyncpg.Pool,
) -> None:
    """Out-degree is the divisor, so an edge counted from elsewhere dilutes the
    score of every real neighbour."""
    async with pool.acquire() as conn:
        org = await new_org(conn, "solo")
        target_ns = ns()
        hub = await entity(conn, name="Hub", namespace=target_ns, org_id=org)
        spoke = await entity(conn, name="Spoke", namespace=target_ns, org_id=org)
        await edge(conn, src=hub, dst=spoke, namespace=target_ns, org_id=org)
        await conn.execute(
            "SELECT pgkg_recompute_pagerank($1, 20, 0.85, $2)", target_ns, org
        )
        before = await scores(conn, [spoke])

        other_ns = ns()
        far = await entity(conn, name="Far", namespace=other_ns, org_id=org)
        await conn.execute(
            "INSERT INTO edges (src_entity, dst_entity, relation, proposition_id)"
            " VALUES ($1, $2, 'leaks', $3)",
            hub,
            far,
            await conn.fetchval(
                "INSERT INTO propositions (text, namespace, org_id)"
                " VALUES ('far edge', $1, $2) RETURNING id",
                other_ns,
                org,
            ),
        )
        await conn.execute(
            "SELECT pgkg_recompute_pagerank($1, 20, 0.85, $2)", target_ns, org
        )
        after = await scores(conn, [spoke])

    assert after[spoke] == before[spoke], (
        "an edge leaving the namespace inflated the out-degree divisor"
    )


async def test_centrality_still_ranks_a_hub_above_a_leaf(
    pool: asyncpg.Pool,
) -> None:
    """The control arm: scoping the divisor must not flatten the algorithm."""
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn, "solo")
        a = await entity(conn, name="A", namespace=namespace, org_id=org)
        b = await entity(conn, name="B", namespace=namespace, org_id=org)
        c = await entity(conn, name="C", namespace=namespace, org_id=org)
        await edge(conn, src=a, dst=b, namespace=namespace, org_id=org)
        await edge(conn, src=b, dst=c, namespace=namespace, org_id=org)
        await edge(conn, src=a, dst=c, namespace=namespace, org_id=org)
        await conn.execute(
            "SELECT pgkg_recompute_pagerank($1, 20, 0.85, $2)", namespace, org
        )
        got = await scores(conn, [a, b, c])

    assert got[c] > got[a]
    assert len(got) == 3


async def test_the_org_defaults_to_the_request_scope(pool: asyncpg.Pool) -> None:
    """A one-argument call is what every existing caller makes, and it has to
    keep resolving to the same org the column defaults and the policies use."""
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn, "scoped")
        a = await entity(conn, name="A", namespace=namespace, org_id=org)
        b = await entity(conn, name="B", namespace=namespace, org_id=org)
        await edge(conn, src=a, dst=b, namespace=namespace, org_id=org)

        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('pgkg.org_id', $1, true)", str(org)
            )
            await conn.execute("SELECT pgkg_recompute_pagerank($1)", namespace)

        got = await scores(conn, [a, b])

    assert len(got) == 2
    assert got[b] > got[a]
