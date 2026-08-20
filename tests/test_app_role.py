"""The role the RLS policies are written for must be able to reach every table.

020 provisions pgkg_app and tells a deployment to connect as it, because
Postgres exempts table owners from row-level security.  The grant it issues is
GRANT ... ON ALL TABLES, which is a snapshot of the tables that existed at that
moment, not a standing rule: every migration that creates a table after it has
to grant again.  Forgetting is silent until a deployment connects as the role
the policies are for and finds a table it cannot read at all.
"""
from __future__ import annotations

import asyncpg


async def test_every_table_is_reachable_by_the_application_role(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        unreachable = await conn.fetch(
            """
            SELECT c.relname
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relkind = 'r'
              AND NOT has_table_privilege('pgkg_app', c.oid, 'SELECT')
            ORDER BY c.relname
            """
        )

    assert [r["relname"] for r in unreachable] == []


async def test_the_application_role_can_write_where_it_must(
    pool: asyncpg.Pool,
) -> None:
    """Retrieval writes: the access ledger updates propositions, and ingest
    writes provenance.  A read-only grant would fail closed at runtime."""
    async with pool.acquire() as conn:
        missing = await conn.fetch(
            """
            SELECT c.relname
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relkind = 'r'
              AND c.relname IN ('propositions', 'provenance', 'entities',
                                'chunks', 'documents', 'edges')
              AND NOT has_table_privilege('pgkg_app', c.oid, 'INSERT')
            ORDER BY c.relname
            """
        )

    assert [r["relname"] for r in missing] == []


async def test_the_application_role_is_not_exempt_from_its_own_policies(
    pool: asyncpg.Pool,
) -> None:
    """A superuser or BYPASSRLS role makes every policy inert, which is the
    failure mode that looks exactly like security."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = 'pgkg_app'"
        )

    assert row is not None, "pgkg_app was not provisioned"
    assert row["rolsuper"] is False
    assert row["rolbypassrls"] is False
