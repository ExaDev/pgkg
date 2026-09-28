"""A migration must apply as the role a managed Postgres hands you (issue #29).

Cloud SQL and RDS give the migrating role no superuser bit.  A function whose
definition pins an extension's GUC (`SET pg_trgm.similarity_threshold = ...`)
is only creatable by such a role once the extension's library is loaded in the
session: before that the GUC is an unknown placeholder, and Postgres refuses a
non-superuser the right to set a placeholder, since it cannot yet know the
parameter's real context.  A fresh migration session has loaded nothing, and
the suite's own migration run is a superuser's, so nothing else here would see
it.
"""
from __future__ import annotations

import pathlib
import uuid

import asyncpg

MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent / "migrations"


async def test_pinning_the_trigram_threshold_needs_no_superuser(
    pg_dsn: str, pool: asyncpg.Pool,
) -> None:
    """`pool` is requested for its migrations: pg_trgm must exist in public."""
    suffix = uuid.uuid4().hex[:8]
    role = f"pgkg_migrator_{suffix}"
    schema = f"pgkg_scratch_{suffix}"
    password = uuid.uuid4().hex

    async with pool.acquire() as admin:
        await admin.execute(
            f"CREATE ROLE {role} LOGIN NOSUPERUSER PASSWORD '{password}'"
        )
    try:
        async with pool.acquire() as admin:
            await admin.execute(f"CREATE SCHEMA {schema} AUTHORIZATION {role}")
        migrator = await asyncpg.connect(
            pg_dsn,
            user=role,
            password=password,
            server_settings={"search_path": f"{schema}, public"},
        )
        try:
            trgm_already_loaded = await migrator.fetchval(
                "SELECT count(*) > 0 FROM pg_settings"
                " WHERE name = 'pg_trgm.similarity_threshold'"
            )
            await migrator.execute(
                (MIGRATIONS_DIR / "051_entity_dedup_reaches_the_trigram_index.sql")
                .read_text()
            )
            proconfig = await migrator.fetchval(
                "SELECT p.proconfig FROM pg_proc p"
                " JOIN pg_namespace n ON n.oid = p.pronamespace"
                " WHERE n.nspname = $1 AND p.proname = 'pgkg_link_entity'",
                schema,
            )
        finally:
            await migrator.close()
    finally:
        async with pool.acquire() as admin:
            await admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            await admin.execute(f"DROP ROLE {role}")

    assert not trgm_already_loaded, "pg_trgm preloaded: the placeholder path is untested"
    assert proconfig == ["pg_trgm.similarity_threshold=0.6"]
