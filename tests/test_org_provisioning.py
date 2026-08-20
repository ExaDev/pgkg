"""Creating an org is enough to make it usable.

D8 makes the embedder generation a property of the org, and 022 binds every org
that existed when it ran.  Every org created afterwards has to be bound too, or
recall resolves no generation and embeds the query with nothing.  Leaving that
to application code makes a schema invariant depend on which code path created
the row: a direct INSERT, a fixture, or a migration in another branch each has
to remember.  Enforce it where the row is written instead.
"""
from __future__ import annotations

import uuid

import asyncpg

GENERATION_1 = uuid.UUID("00000000-0000-0000-0000-000000000010")


async def test_a_raw_org_insert_is_bound_to_the_primary_generation(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        org = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
            f"raw_{uuid.uuid4().hex[:8]}",
        )
        role = await conn.fetchval(
            "SELECT role FROM org_embedders WHERE org_id = $1", org
        )

    assert role == "primary"


async def test_every_org_has_a_primary_generation(pool: asyncpg.Pool) -> None:
    """The invariant 022 asserts, held by construction rather than by every
    caller remembering to bind."""
    async with pool.acquire() as conn:
        unbound = await conn.fetch(
            """
            SELECT o.id
            FROM orgs o
            WHERE NOT EXISTS (
                SELECT 1 FROM org_embedders oe
                WHERE oe.org_id = o.id AND oe.role = 'primary'
            )
            """
        )

    assert [r["id"] for r in unbound] == []


async def test_a_new_org_can_resolve_a_live_generation(
    pool: asyncpg.Pool,
) -> None:
    """The consequence that matters: recall asks the registry what to embed
    with, and an unbound org gets nothing back."""
    async with pool.acquire() as conn:
        org = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
            f"fresh_{uuid.uuid4().hex[:8]}",
        )
        generations = await conn.fetch(
            "SELECT generation_id, role FROM pgkg_live_generations($1)", org
        )

    assert [r["role"] for r in generations] == ["primary"]
    assert generations[0]["generation_id"] == GENERATION_1


async def test_an_explicit_binding_is_not_overwritten(
    pool: asyncpg.Pool,
) -> None:
    """The trigger seeds; it must not fight an operator who then re-roles the
    org onto a different generation."""
    async with pool.acquire() as conn:
        org = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
            f"rerole_{uuid.uuid4().hex[:8]}",
        )
        second = await conn.fetchval(
            """
            INSERT INTO embedder_generations
                (name, dim, storage_type, normalize, status)
            VALUES ($1, 768, 'halfvec', TRUE, 'live')
            RETURNING id
            """,
            f"bge-m3@768-{uuid.uuid4().hex[:6]}",
        )
        await conn.execute(
            "UPDATE org_embedders SET role = 'secondary'"
            " WHERE org_id = $1 AND generation_id = $2",
            org,
            GENERATION_1,
        )
        await conn.execute(
            "INSERT INTO org_embedders (org_id, generation_id, role)"
            " VALUES ($1, $2, 'primary')",
            org,
            second,
        )
        roles = dict(
            await conn.fetch(
                "SELECT generation_id, role FROM org_embedders WHERE org_id = $1",
                org,
            )
        )

    assert roles[second] == "primary"
    assert roles[GENERATION_1] == "secondary"


async def test_a_transitional_generation_table_is_reachable_by_the_app_role(
    pool: asyncpg.Pool,
) -> None:
    """The side table is created at run time by pgkg_create_generation_storage,
    so 020's GRANT ON ALL TABLES never saw it: without a grant of its own the
    transitional arm fails with permission denied for the one role the policies
    are written for."""
    async with pool.acquire() as conn:
        generation = await conn.fetchval(
            """
            INSERT INTO embedder_generations
                (name, dim, storage_type, normalize, status)
            VALUES ($1, 256, 'halfvec', TRUE, 'building')
            RETURNING id
            """,
            f"bge-m3@256-{uuid.uuid4().hex[:6]}",
        )
        table = await conn.fetchval(
            "SELECT pgkg_create_generation_storage($1, 'prop')", generation
        )
        can_read = await conn.fetchval(
            "SELECT has_table_privilege('pgkg_app', $1, 'SELECT')", table
        )

    assert can_read is True
