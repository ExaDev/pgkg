"""The schema's own functions, called where search_path cannot find them.

PostgreSQL 17 builds indexes, and runs REINDEX, VACUUM, ANALYZE and CLUSTER,
with search_path pinned to `pg_catalog, pg_temp`.  A function evaluated there —
one named in an index expression, or one that such a function calls — sees
nothing outside the catalog, so an unqualified call to another pgkg function
fails with "does not exist".  On 16 the same build uses the caller's path and
the defect hides, which is how 040 shipped a migration that cannot install on
17 (issue #28).

pg_restore has the same shape on every version: a dump sets search_path to ''
before it loads data, and a stored generated column is recomputed as each row
is copied in.  `entities` stores both gazetteer keys that way (047), so an
unqualified call in the normaliser is a dump that will not restore.

Both are exercised here by pinning the path the server would pin, which works
against any major version; the CI matrix is what runs the real PG17 build.
"""
from __future__ import annotations

import pathlib
import uuid

import asyncpg

MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent / "migrations"

DEFAULT_ORG = uuid.UUID("00000000-0000-0000-0000-000000000001")

# What PostgreSQL 17+ sets for the duration of a maintenance operation.
MAINTENANCE_SEARCH_PATH = "pg_catalog, pg_temp"


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def test_an_index_over_the_gazetteer_keys_builds_under_the_maintenance_search_path(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute(
                """
                INSERT INTO entities (name, type, namespace, org_id, aliases)
                VALUES ($1, 'concept', 'default', $2, $3)
                """,
                unique("Helios"), DEFAULT_ORG, ["Project Helios", "HLS"],
            )
            await conn.execute(f"SET LOCAL search_path = {MAINTENANCE_SEARCH_PATH}")

            await conn.execute(
                """
                CREATE INDEX restricted_path_alias_keys_idx
                    ON public.entities USING gin (public.pgkg_gazetteer_keys(aliases))
                """
            )
        finally:
            await tx.rollback()


async def test_the_stored_gazetteer_keys_compute_under_an_empty_search_path(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute("SET LOCAL search_path = ''")

            keys = await conn.fetchrow(
                """
                INSERT INTO public.entities (name, type, namespace, org_id, aliases)
                VALUES ($1, 'concept', 'default', $2, $3)
                RETURNING gazetteer_name_key, gazetteer_alias_keys
                """,
                "The Helios  Migration!", DEFAULT_ORG, ["Project-Helios", "AI"],
            )
        finally:
            await tx.rollback()

    assert keys["gazetteer_name_key"] == "the helios migration"
    assert keys["gazetteer_alias_keys"] == ["project helios"]


# The body 040 shipped with before issue #28, which every install that migrated
# before the fix is still running: 040 is recorded as applied by filename, so the
# corrected file is never re-read there.
UNQUALIFIED_GAZETTEER_KEYS = """
CREATE OR REPLACE FUNCTION pgkg_gazetteer_keys(p_texts TEXT[]) RETURNS TEXT[]
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
AS $$
    SELECT ARRAY(
        SELECT pgkg_gazetteer_key(t)
        FROM unnest(p_texts) AS t
        WHERE length(pgkg_gazetteer_key(t)) >= 3
    )
$$
"""


async def test_an_install_that_ran_the_original_040_is_repaired_by_057(
    pool: asyncpg.Pool,
) -> None:
    (repair,) = MIGRATIONS_DIR.glob("057_*.sql")
    async with pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute(UNQUALIFIED_GAZETTEER_KEYS)
            await conn.execute(repair.read_text())
            await conn.execute("SET LOCAL search_path = ''")

            keys = await conn.fetchval(
                "SELECT public.pgkg_gazetteer_keys($1)", ["Project-Helios"]
            )
        finally:
            await tx.rollback()

    assert keys == ["project helios"]
