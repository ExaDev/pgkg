"""Every row-level security policy is pinned by a read that crosses an org.

020's policies are the second line of defence behind the retrieval predicate,
and the failure mode of a second line is that nobody notices it stopped
working.  Before this module only `propositions` and `collections` had a test
that changed answer when their policy was neutered: the other eight tables
carried a policy no assertion could tell from `USING (TRUE)`.

Each case seeds a row in one org and reads it back as `pgkg_app` — the role the
policies are written for, since Postgres exempts the table owner — once under
the row's own org and once under a stranger's.  The first read is the control
arm: without it a policy that hid everything would look identical to a policy
that hid the right thing.
"""
from __future__ import annotations

import uuid

import asyncpg
import pytest

ORG_GUC = "pgkg.org_id"


async def new_org(conn: asyncpg.Connection) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
        f"rls_{uuid.uuid4().hex[:10]}",
    )


async def _seed_propositions(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO propositions (text, org_id) VALUES ('rls row', $1)", org
    )


async def _seed_chunks(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO chunks (text, org_id) VALUES ('rls chunk', $1)", org
    )


async def _seed_documents(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO documents (source, org_id) VALUES ('rls doc', $1)", org
    )


async def _seed_entities(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO entities (name, type, namespace, org_id)"
        " VALUES ($1, 'concept', 'rls', $2)",
        f"rls_{uuid.uuid4().hex[:8]}",
        org,
    )


async def _seed_users(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO users (org_id, external_id) VALUES ($1, $2)",
        org,
        f"rls_{uuid.uuid4().hex[:8]}",
    )


async def _seed_tenant_shards(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO tenant_shards (org_id, shard_key) VALUES ($1, 'pool_9')", org
    )


async def _seed_provenance(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO provenance (org_id, kind, producer)"
        " VALUES ($1, 'backfill', 'backfill')",
        org,
    )


async def _seed_collections(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO collections (org_id, owner_org_id, name) VALUES ($1, $1, $2)",
        org,
        f"rls_{uuid.uuid4().hex[:8]}",
    )


async def _seed_collection_subscriptions(
    conn: asyncpg.Connection, org: uuid.UUID
) -> None:
    collection = await conn.fetchval(
        "INSERT INTO collections (org_id, owner_org_id, name)"
        " VALUES ($1, $1, $2) RETURNING id",
        org,
        f"sub_{uuid.uuid4().hex[:8]}",
    )
    await conn.execute(
        "INSERT INTO collection_subscriptions (org_id, collection_id)"
        " VALUES ($1, $2)",
        org,
        collection,
    )


async def _seed_org_embedders(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    """026 binds a new org to the primary generation by trigger, so the row is
    already there; asserting it is what makes the read below non-vacuous."""
    bound = await conn.fetchval(
        "SELECT count(*) FROM org_embedders WHERE org_id = $1", org
    )
    assert bound > 0, "026's binding trigger did not fire, so nothing to hide"


async def _seed_document_versions(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    document = await conn.fetchval(
        "INSERT INTO documents (source, org_id) VALUES ('rls versioned', $1)"
        " RETURNING id",
        org,
    )
    await conn.execute(
        "INSERT INTO document_versions (document_id, org_id, version_no,"
        " content_hash) VALUES ($1, $2, 1, digest('rls body', 'sha256'))",
        document,
        org,
    )


async def _seed_document_version_chunks(
    conn: asyncpg.Connection, org: uuid.UUID
) -> uuid.UUID:
    document = await conn.fetchval(
        "INSERT INTO documents (source, org_id) VALUES ('rls linked', $1)"
        " RETURNING id",
        org,
    )
    version = await conn.fetchval(
        "INSERT INTO document_versions (document_id, org_id, version_no,"
        " content_hash) VALUES ($1, $2, 1, digest($3, 'sha256')) RETURNING id",
        document,
        org,
        f"rls linked {org}",
    )
    chunk = await conn.fetchval(
        "INSERT INTO chunks (text, org_id) VALUES ($1, $2) RETURNING id",
        f"rls linked passage {uuid.uuid4().hex[:8]}",
        org,
    )
    await conn.execute(
        "INSERT INTO document_version_chunks (document_version_id, chunk_id, ord)"
        " VALUES ($1, $2, 0)",
        version,
        chunk,
    )
    # The version, not the org: counting these rows through document_versions
    # would be laundering the answer through that table's policy, and the count
    # would come back zero for a stranger even with this table wide open.
    return version


async def _seed_ingest_jobs(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    collection = await conn.fetchval(
        "INSERT INTO collections (org_id, owner_org_id, name, kind)"
        " VALUES ($1, $1, $2, 'corpus') RETURNING id",
        org,
        f"jobs_{uuid.uuid4().hex[:8]}",
    )
    await conn.execute(
        "SELECT pgkg_enqueue_ingest_job($1, $2, $3, digest($4, 'sha256'), $4)",
        org,
        collection,
        f"external_{uuid.uuid4().hex[:8]}",
        "a queued document body",
    )


SEEDS = {
    "chunks": _seed_chunks,
    "collection_subscriptions": _seed_collection_subscriptions,
    "collections": _seed_collections,
    "document_version_chunks": _seed_document_version_chunks,
    "document_versions": _seed_document_versions,
    "documents": _seed_documents,
    "entities": _seed_entities,
    "ingest_jobs": _seed_ingest_jobs,
    "org_embedders": _seed_org_embedders,
    "propositions": _seed_propositions,
    "provenance": _seed_provenance,
    "tenant_shards": _seed_tenant_shards,
    "users": _seed_users,
}

# How the seeded row is found again, and what $1 is when it is.  Every table but
# one carries the org directly and is counted by it.  document_version_chunks is
# a link table and gains no org column of its own — the version already states
# it, and a second copy could only disagree with the first — so its seed returns
# the version id and the count names that version.  Counting it through
# document_versions instead would launder the answer through that table's
# policy: a stranger would read zero however wide open this table was left.
FOUND_BY = {
    "document_version_chunks": "document_version_id = $1",
}


def _row_predicate(table: str) -> str:
    return FOUND_BY.get(table, "org_id = $1")


async def test_every_rls_enabled_table_is_covered_here(pool: asyncpg.Pool) -> None:
    """A table that gains a policy without gaining a case here would ship an
    untested isolation boundary, which is the state this module exists to end."""
    async with pool.acquire() as conn:
        protected = [
            r["relname"]
            for r in await conn.fetch(
                """
                SELECT c.relname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relkind = 'r'
                  AND c.relrowsecurity
                ORDER BY c.relname
                """
            )
        ]

    assert protected == sorted(SEEDS)


async def _count_as_app(
    conn: asyncpg.Connection, *, table: str, handle: uuid.UUID, guc: uuid.UUID
) -> int:
    async with conn.transaction():
        await conn.execute("SET LOCAL ROLE pgkg_app")
        await conn.execute("SELECT set_config($1, $2, true)", ORG_GUC, str(guc))
        return await conn.fetchval(
            f"SELECT count(*) FROM {table} WHERE {_row_predicate(table)}",
            handle,
        )


@pytest.mark.parametrize("table", sorted(SEEDS))
async def test_the_application_role_reads_only_its_own_orgs_rows(
    pool: asyncpg.Pool, table: str
) -> None:
    async with pool.acquire() as conn:
        mine = await new_org(conn)
        stranger = await new_org(conn)
        # The org, unless the seed named something more specific to count.
        handle = await SEEDS[table](conn, mine) or mine

        own = await _count_as_app(conn, table=table, handle=handle, guc=mine)
        foreign = await _count_as_app(conn, table=table, handle=handle, guc=stranger)

    assert own > 0, (
        f"{table}: the row is invisible to its own org, so the negative "
        "assertion below would hold for the wrong reason"
    )
    assert foreign == 0, f"{table}: a stranger's session read another org's rows"
