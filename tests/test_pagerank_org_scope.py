"""The out-degree divisor and the vote numerator are scoped by org, not only by
namespace.

025 scoped both halves of the PageRank recurrence to `(namespace, org_id)`, and
`test_out_degree_ignores_edges_outside_the_namespace` pins the namespace half.
The org half was untested: the existing cross-tenant case gives each tenant its
own entity rows, so their edges never share a `src_entity` and never meet in a
divisor.  A cross-org edge is what makes them meet, and `edges` carries no org
of its own — the only thing keeping one tenant's edge out of another tenant's
arithmetic is the join to `entities`.

The assertion is an unchanged score, which makes the tenant's own subgraph the
control arm: the number being compared is one the algorithm produced from data
the tenant can read.

Only the divisor is asserted.  The vote numerator carries the same org
predicate, but it joins the scoped out-degree subquery on `src_entity`, so a
foreign source has no divisor row and is dropped by that inner join before the
predicate is reached — belt and braces rather than a second observable rule,
and a test for it would pass with the predicate removed.
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
    return f"prorg_{uuid.uuid4().hex[:10]}"


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
    proposition = await conn.fetchval(
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
        proposition,
    )


async def score(conn: asyncpg.Connection, entity_id: uuid.UUID) -> float:
    return await conn.fetchval(
        "SELECT score FROM entity_pagerank WHERE entity_id = $1", entity_id
    )


async def test_an_edge_into_another_org_does_not_dilute_my_scores(
    pool: asyncpg.Pool,
) -> None:
    """Out-degree is the divisor.  An edge from my entity to a foreign one is
    not a vote inside my subgraph, so counting it halves the share every real
    neighbour receives — a score that moves with data the tenant cannot read."""
    namespace = ns()
    async with pool.acquire() as conn:
        mine = await new_org(conn, "mine")
        stranger = await new_org(conn, "stranger")

        hub = await entity(conn, name="Hub", namespace=namespace, org_id=mine)
        spoke = await entity(conn, name="Spoke", namespace=namespace, org_id=mine)
        await edge(conn, src=hub, dst=spoke, namespace=namespace, org_id=mine)
        await conn.execute(
            "SELECT pgkg_recompute_pagerank($1, 20, 0.85, $2)", namespace, mine
        )
        before = await score(conn, spoke)

        foreign = await entity(
            conn, name="Foreign", namespace=namespace, org_id=stranger
        )
        await edge(
            conn, src=hub, dst=foreign, namespace=namespace, org_id=stranger
        )
        await conn.execute(
            "SELECT pgkg_recompute_pagerank($1, 20, 0.85, $2)", namespace, mine
        )
        after = await score(conn, spoke)

    assert after == before, (
        "an edge leaving the org inflated the out-degree divisor"
    )
