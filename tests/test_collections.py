"""Collections: the row that carries decay profile, claim scope and ownership.

Phase 1 puts collection_id on every retrievable row.  A UUID with nothing
behind it is the stringly-typed namespace again with a different type, so the
column needs the table it names: D3 rejected the namespace string for having
"no referential integrity", and D6 makes the collection the thing that carries
a decay profile and a claim scope.  The subscription seam ships now and stays
empty, because it sits in the hot-path predicate (D4).
"""
from __future__ import annotations

import uuid

import asyncpg
import pytest

SYSTEM_ORG = uuid.UUID("00000000-0000-0000-0000-000000000000")
DEFAULT_ORG = uuid.UUID("00000000-0000-0000-0000-000000000001")
DEFAULT_COLLECTION = uuid.UUID("00000000-0000-0000-0000-000000000002")


def ns() -> str:
    return f"collections_{uuid.uuid4().hex[:10]}"


async def new_org(conn: asyncpg.Connection, name: str = "acme") -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", name
    )


async def new_collection(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    name: str = "notes",
    decay_profile: str = "conversational",
    claim_scope: str = "org",
    owner_org_id: uuid.UUID | None = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO collections
            (org_id, owner_org_id, name, decay_profile, claim_scope)
        VALUES ($1::uuid, COALESCE($2::uuid, $1::uuid), $3, $4, $5)
        RETURNING id
        """,
        org_id,
        owner_org_id,
        name,
        decay_profile,
        claim_scope,
    )


# ---------------------------------------------------------------------------
# The backfill target exists and is the one the columns already default to
# ---------------------------------------------------------------------------


async def test_the_default_collection_is_a_real_row(pool: asyncpg.Pool) -> None:
    """Every pre-tenancy row was backfilled to pgkg_default_collection() by a
    column default, so that id has to resolve to a row or the FK cannot land."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT org_id, owner_org_id, decay_profile, claim_scope "
            "FROM collections WHERE id = pgkg_default_collection()"
        )

    assert row is not None
    assert row["org_id"] == DEFAULT_ORG
    assert row["owner_org_id"] == DEFAULT_ORG
    assert row["decay_profile"] == "conversational"
    assert row["claim_scope"] == "org"


@pytest.mark.parametrize("table", ["propositions", "chunks", "documents"])
async def test_collection_id_is_a_foreign_key(
    pool: asyncpg.Pool, table: str
) -> None:
    """An unresolvable collection is rejected by Postgres, not by a caller
    remembering to check."""
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            async with conn.transaction():
                if table == "propositions":
                    await conn.execute(
                        "INSERT INTO propositions (text, namespace, collection_id)"
                        " VALUES ('orphan', $1, $2)",
                        ns(),
                        uuid.uuid4(),
                    )
                elif table == "chunks":
                    await conn.execute(
                        "INSERT INTO chunks (text, collection_id)"
                        " VALUES ('orphan', $1)",
                        uuid.uuid4(),
                    )
                else:
                    await conn.execute(
                        "INSERT INTO documents (source, collection_id)"
                        " VALUES ('orphan', $1)",
                        uuid.uuid4(),
                    )


async def test_a_pre_tenancy_insert_still_works(pool: asyncpg.Pool) -> None:
    """The whole point of the backfill: a caller that names no scope at all
    still lands, and lands in the reserved default collection."""
    namespace = ns()
    async with pool.acquire() as conn:
        prop_id = await conn.fetchval(
            "INSERT INTO propositions (text, namespace) VALUES ('legacy', $1)"
            " RETURNING id",
            namespace,
        )
        row = await conn.fetchrow(
            "SELECT org_id, collection_id FROM propositions WHERE id = $1",
            prop_id,
        )

    assert row["org_id"] == DEFAULT_ORG
    assert row["collection_id"] == DEFAULT_COLLECTION


# ---------------------------------------------------------------------------
# What the collection carries
# ---------------------------------------------------------------------------


async def test_decay_profile_is_a_closed_vocabulary(pool: asyncpg.Pool) -> None:
    """One constant cannot serve three kinds of content, and a free-text
    profile is a constant nobody can enumerate (D6)."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        for profile in ("conversational", "timeless", "perishable"):
            assert await new_collection(
                conn, org_id=org, name=f"c_{profile}", decay_profile=profile
            )

        with pytest.raises(asyncpg.CheckViolationError):
            async with conn.transaction():
                await new_collection(
                    conn, org_id=org, name="bad", decay_profile="whenever"
                )


async def test_claim_scope_is_a_closed_vocabulary(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        for scope in ("world", "org", "user"):
            assert await new_collection(
                conn, org_id=org, name=f"s_{scope}", claim_scope=scope
            )

        with pytest.raises(asyncpg.CheckViolationError):
            async with conn.transaction():
                await new_collection(
                    conn, org_id=org, name="bad", claim_scope="galactic"
                )


async def test_extraction_is_opt_in(pool: asyncpg.Pool) -> None:
    """D2: the corpus is not proposition-extracted by default."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        extract = await conn.fetchval(
            "SELECT extract_propositions FROM collections WHERE id = $1",
            collection,
        )

    assert extract is False


async def test_nothing_is_shared_by_default(pool: asyncpg.Pool) -> None:
    """D4: don't default anything to shared, and a tenant's own collection is
    owned by the tenant, never by the system org."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        row = await conn.fetchrow(
            "SELECT owner_org_id, visibility, public_source"
            " FROM collections WHERE id = $1",
            collection,
        )

    assert row["owner_org_id"] == org
    assert row["visibility"] == "private"
    assert row["public_source"] is False


# ---------------------------------------------------------------------------
# The sharing seam: present, and empty
# ---------------------------------------------------------------------------


async def test_no_collection_is_subscribed_implicitly(pool: asyncpg.Pool) -> None:
    """A subscription is a capability, so it has to be granted by a row.

    Asked of a fresh org and of the operator's shelf rather than of the whole
    table: a global count is an assertion about every other test in the suite,
    and it fails for the wrong reason the day one of them subscribes something.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        shared = await new_collection(
            conn, org_id=SYSTEM_ORG, name=f"shelf_{uuid.uuid4().hex[:6]}",
            claim_scope="world",
        )
        subscriptions = await conn.fetchval(
            "SELECT count(*) FROM collection_subscriptions"
            " WHERE org_id = $1 OR collection_id = $2",
            org,
            shared,
        )

    assert subscriptions == 0


async def test_a_subscription_carries_a_weight_that_can_turn_it_down(
    pool: asyncpg.Pool,
) -> None:
    """rrf_weight lets a tenant turn shared material down, or off, without a
    rebuild (D4)."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        shared = await new_collection(
            conn, org_id=SYSTEM_ORG, name=f"vendor_docs_{uuid.uuid4().hex[:6]}",
            claim_scope="world", decay_profile="perishable",
        )
        await conn.execute(
            "INSERT INTO collection_subscriptions (org_id, collection_id)"
            " VALUES ($1, $2)",
            org,
            shared,
        )
        row = await conn.fetchrow(
            "SELECT enabled, rrf_weight FROM collection_subscriptions"
            " WHERE org_id = $1 AND collection_id = $2",
            org,
            shared,
        )

    assert row["enabled"] is True
    assert row["rrf_weight"] == pytest.approx(1.0)


async def test_a_tenant_cannot_subscribe_to_a_collection_twice(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        shared = await new_collection(
            conn, org_id=SYSTEM_ORG, name=f"standards_{uuid.uuid4().hex[:6]}",
            claim_scope="world",
        )
        await conn.execute(
            "INSERT INTO collection_subscriptions (org_id, collection_id)"
            " VALUES ($1, $2)",
            org,
            shared,
        )

        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO collection_subscriptions (org_id, collection_id)"
                    " VALUES ($1, $2)",
                    org,
                    shared,
                )


async def test_collection_names_are_unique_within_an_org(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        await new_collection(conn, org_id=org, name="handbook")

        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await new_collection(conn, org_id=org, name="handbook")


async def test_collections_are_org_scoped_under_rls(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        org_a = await new_org(conn, "a")
        org_b = await new_org(conn, "b")
        mine = await new_collection(
            conn, org_id=org_a, name=f"mine_{uuid.uuid4().hex[:6]}"
        )
        await new_collection(
            conn, org_id=org_b, name=f"theirs_{uuid.uuid4().hex[:6]}"
        )

        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('pgkg.org_id', $1, true)", str(org_a)
            )
            await conn.execute("SET LOCAL ROLE pgkg_app")
            visible = await conn.fetch(
                "SELECT id FROM collections WHERE org_id IN ($1, $2)",
                org_a,
                org_b,
            )

    assert [r["id"] for r in visible] == [mine]
