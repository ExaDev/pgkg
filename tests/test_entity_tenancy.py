"""Entity resolution is confined to one org.

D3 makes org_id the hard isolation boundary and accepts that entity *names*
are shared across users inside one org.  Across orgs nothing is shared, so
the uniqueness constraint entity resolution relies on has to name the org:
while it does not, two tenants contend for one row, and which failure that
produces depends only on whether row-level security happens to be active.
"""
from __future__ import annotations

import uuid

import asyncpg
import pytest

DIM = 1024


def vec(*, hot_index: int = 0, value: float = 1.0, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[hot_index] = value
    return v


def pg_vec(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


def ns() -> str:
    return f"entity_tenancy_{uuid.uuid4().hex[:10]}"


async def new_org(conn: asyncpg.Connection, name: str) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", name
    )


async def link_as(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    namespace: str,
    name: str,
    type_: str | None = "concept",
    embedding: list[float] | None = None,
    threshold: float = 0.85,
    assume_app_role: bool = False,
) -> uuid.UUID | None:
    """Resolve an entity the way ingest does: GUC first, then the function."""
    emb = f"'{pg_vec(embedding)}'::halfvec" if embedding is not None else "NULL"
    async with conn.transaction():
        await conn.execute(
            "SELECT set_config('pgkg.org_id', $1, true)", str(org_id)
        )
        if assume_app_role:
            await conn.execute("SET LOCAL ROLE pgkg_app")
        return await conn.fetchval(
            f"SELECT pgkg_link_entity($1, $2, $3, {emb}, $4)",
            namespace,
            name,
            type_,
            threshold,
        )


# ---------------------------------------------------------------------------
# The regression that motivates the change
# ---------------------------------------------------------------------------


async def test_a_second_org_naming_the_same_entity_still_gets_an_id(
    pool: asyncpg.Pool,
) -> None:
    """Under RLS the loser of the name race used to get NULL back, silently
    dropping the subject of every proposition mentioning that entity."""
    namespace = ns()
    async with pool.acquire() as conn:
        org_a = await new_org(conn, "a")
        org_b = await new_org(conn, "b")

        first = await link_as(
            conn, org_id=org_a, namespace=namespace, name="Postgres",
            assume_app_role=True,
        )
        second = await link_as(
            conn, org_id=org_b, namespace=namespace, name="Postgres",
            assume_app_role=True,
        )

    assert first is not None
    assert second is not None, (
        "the second org's entity link returned NULL, which ingest would store "
        "as a proposition with no subject"
    )
    assert first != second


async def test_each_org_gets_its_own_entity_row(pool: asyncpg.Pool) -> None:
    """Same namespace, same name, two orgs: two rows, each in its own org."""
    namespace = ns()
    async with pool.acquire() as conn:
        org_a = await new_org(conn, "a")
        org_b = await new_org(conn, "b")

        a_id = await link_as(
            conn, org_id=org_a, namespace=namespace, name="Kingfisher"
        )
        b_id = await link_as(
            conn, org_id=org_b, namespace=namespace, name="Kingfisher"
        )

        owners = dict(
            await conn.fetch(
                "SELECT id, org_id FROM entities WHERE id = ANY($1::uuid[])",
                [a_id, b_id],
            )
        )

    assert a_id != b_id
    assert owners[a_id] == org_a
    assert owners[b_id] == org_b


async def test_resolution_is_still_idempotent_inside_one_org(
    pool: asyncpg.Pool,
) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn, "a")
        first = await link_as(
            conn, org_id=org, namespace=namespace, name="Repeated"
        )
        again = await link_as(
            conn, org_id=org, namespace=namespace, name="Repeated"
        )

    assert first == again


async def test_the_fuzzy_arm_does_not_reach_across_orgs(
    pool: asyncpg.Pool,
) -> None:
    """Stage 2 matches on trigram similarity plus cosine, which is exactly the
    arm that would otherwise adopt a near-identical name from another tenant."""
    namespace = ns()
    async with pool.acquire() as conn:
        org_a = await new_org(conn, "a")
        org_b = await new_org(conn, "b")

        theirs = await link_as(
            conn, org_id=org_a, namespace=namespace,
            name="William Shakespeare", embedding=vec(),
        )
        mine = await link_as(
            conn, org_id=org_b, namespace=namespace,
            name="William Shakespear", embedding=vec(value=0.99),
        )

    assert theirs != mine, "org B adopted org A's entity through the fuzzy arm"


async def test_the_fuzzy_arm_still_dedupes_within_one_org(
    pool: asyncpg.Pool,
) -> None:
    """The control arm: the same near-miss inside one org must still collapse,
    or the test above would pass merely because stage 2 stopped working."""
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn, "a")

        canonical = await link_as(
            conn, org_id=org, namespace=namespace,
            name="William Shakespeare", embedding=vec(),
        )
        typo = await link_as(
            conn, org_id=org, namespace=namespace,
            name="William Shakespear", embedding=vec(value=0.99),
            threshold=0.5,
        )

    assert typo == canonical


async def test_an_untyped_entity_cannot_be_duplicated(
    pool: asyncpg.Pool,
) -> None:
    """A NULL type used to escape the unique index, so the concurrency yield
    stage 3 relies on had nothing to block on (noted in migration 013)."""
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn, "a")
        first = await link_as(
            conn, org_id=org, namespace=namespace, name="Untyped", type_=None
        )

        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO entities (name, type, namespace, org_id) "
                    "VALUES ('Untyped', NULL, $1, $2)",
                    namespace,
                    org,
                )

    assert first is not None
