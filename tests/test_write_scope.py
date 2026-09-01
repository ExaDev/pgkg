"""Reads widen, writes never do — asserted on every write path there is.

Scope says it in one line: "Reads widen (the org's own rows plus the system
org's shared collections, plus subscribed collections); writes never do."  The
widening fields arrive from an unauthenticated HTTP request, so a write path
that consults read_org_ids or read_collection_ids hands the caller a tenant it
was never given.  These tests are the class, not the instance: every method
that writes is exercised under a scope widened as far as the API allows, and
the operator's reserved system org is checked for damage after each one.
"""
from __future__ import annotations

import uuid

import asyncpg
import pytest

from pgkg.config import SYSTEM_ORG_ID
from pgkg.memory import Memory, Scope


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def new_org(conn: asyncpg.Connection) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("org")
    )


async def system_proposition(conn: asyncpg.Connection) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO propositions (text, org_id, collection_id)
        VALUES ('operator published guidance', pgkg_system_org(),
                pgkg_default_collection())
        RETURNING id
        """
    )


def widened(org_id: uuid.UUID) -> Scope:
    """Every widening field an HTTP caller can set, all set."""
    return Scope(
        org_id=org_id,
        include_system_org=True,
        subscribed_collection_ids=(uuid.uuid4(),),
    )


async def test_forget_cannot_invalidate_a_row_outside_the_write_org(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        tenant = await new_org(conn)
        shared = await system_proposition(conn)

    memory = Memory(pool, scope=widened(tenant))
    await memory.forget(shared, reason="user_deleted")
    await memory.aclose()

    async with pool.acquire() as conn:
        state = await conn.fetchrow(
            "SELECT invalidated_at FROM propositions WHERE id = $1", shared
        )
        await conn.execute("DELETE FROM propositions WHERE id = $1", shared)

    assert state["invalidated_at"] is None


async def test_forget_still_invalidates_the_write_org_own_row(
    pool: asyncpg.Pool,
) -> None:
    """Non-vacuity: the same call on the tenant's own row must still work."""
    async with pool.acquire() as conn:
        tenant = await new_org(conn)
        await conn.execute("SELECT set_config('pgkg.org_id', $1, false)", str(tenant))
        mine = await conn.fetchval(
            "INSERT INTO propositions (text, org_id) VALUES ('mine', $1)"
            " RETURNING id",
            tenant,
        )

    memory = Memory(pool, scope=widened(tenant))
    await memory.forget(mine, reason="user_deleted")
    await memory.aclose()

    async with pool.acquire() as conn:
        state = await conn.fetchrow(
            "SELECT invalidated_at, invalidation_reason FROM propositions"
            " WHERE id = $1",
            mine,
        )

    assert state["invalidated_at"] is not None
    assert state["invalidation_reason"] == "user_deleted"


TENANT_TABLES = ("documents", "chunks", "propositions", "provenance")


async def rows_per_org(
    conn: asyncpg.Connection, org_id: uuid.UUID
) -> dict[str, int]:
    return {
        table: await conn.fetchval(
            f"SELECT count(*) FROM {table} WHERE org_id = $1", org_id
        )
        for table in TENANT_TABLES
    }


async def test_ingest_writes_every_row_into_the_write_org(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    from pgkg import ml

    async with pool.acquire() as conn:
        tenant = await new_org(conn)
        dim = await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )
        before_system = await rows_per_org(conn, SYSTEM_ORG_ID)

    monkeypatch.setattr(ml, "embed", lambda texts: [[0.01] * dim for _ in texts])

    memory = Memory(pool, scope=widened(tenant), namespace=unique("ns"))
    result = await memory.ingest(
        "The reimbursement window is thirty days for lodging claims."
    )
    await memory.aclose()

    async with pool.acquire() as conn:
        after_system = await rows_per_org(conn, SYSTEM_ORG_ID)
        mine = await rows_per_org(conn, tenant)

    assert result.chunks > 0
    assert all(count > 0 for count in mine.values())
    assert after_system == before_system


async def test_access_counts_are_written_under_the_org_that_read_them(
    pool: asyncpg.Pool,
) -> None:
    """The access ledger is a write too: it must not credit a widened org."""
    async with pool.acquire() as conn:
        tenant = await new_org(conn)
        shared = await system_proposition(conn)

    memory = Memory(pool, scope=widened(tenant))
    memory._record_access([shared])
    written = await memory.flush_access()

    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT access_count FROM propositions WHERE id = $1", shared
        )
        await conn.execute("DELETE FROM propositions WHERE id = $1", shared)

    assert written == 1
    assert count == 0, (
        "a widened read credited an access to a proposition in the system org"
    )


@pytest.mark.parametrize("method", ["forget", "ingest", "flush_access"])
def test_no_write_path_reads_the_widening_properties(method: str) -> None:
    """The structural half of the same claim, so a new write path cannot
    quietly reintroduce it: the body of each write method must not name a
    read-widening property."""
    import inspect

    source = inspect.getsource(getattr(Memory, method))
    assert "read_org_ids" not in source
    assert "read_collection_ids" not in source
