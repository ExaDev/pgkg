"""Tenancy: explicit scoping columns, the shared retrieval predicate, RLS.

Every assertion goes through a public surface — the retrieval functions, the
tables' own constraints, or a session that has assumed the application role —
because the point of the phase is that isolation is enforced by Postgres and
by the one shared predicate, not by each caller remembering a WHERE clause.
"""
from __future__ import annotations

import re
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
    return f"tenancy_{uuid.uuid4().hex[:10]}"


async def new_org(conn: asyncpg.Connection, name: str = "acme") -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", name
    )


async def new_user(
    conn: asyncpg.Connection, org_id: uuid.UUID, external_id: str
) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
        org_id,
        external_id,
    )


async def new_collection(
    conn: asyncpg.Connection, *, org_id: uuid.UUID, name: str
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO collections (org_id, owner_org_id, name)
        VALUES ($1, $1, $2)
        RETURNING id
        """,
        org_id,
        name,
    )


async def insert_entity(
    conn: asyncpg.Connection, *, name: str, namespace: str
) -> uuid.UUID:
    return await conn.fetchval(
        f"""
        INSERT INTO entities (name, type, namespace, embedding)
        VALUES ($1, 'concept', $2, '{pg_vec(vec())}')
        RETURNING id
        """,
        name,
        namespace,
    )


async def insert_proposition(
    conn: asyncpg.Connection,
    *,
    text: str,
    namespace: str,
    org_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
    claim_scope: str | None = None,
    visibility: str | None = None,
    owner_user_id: uuid.UUID | None = None,
    acl_group_id: uuid.UUID | None = None,
    embedding: list[float] | None = None,
    subject_id: uuid.UUID | None = None,
    object_id: uuid.UUID | None = None,
    predicate: str | None = None,
) -> uuid.UUID:
    emb_expr = f"'{pg_vec(embedding)}'" if embedding is not None else "NULL"
    return await conn.fetchval(
        f"""
        INSERT INTO propositions
            (text, namespace, embedding, subject_id, object_id, predicate,
             org_id, collection_id, claim_scope, visibility,
             owner_user_id, acl_group_id)
        VALUES ($1, $2, {emb_expr}, $3, $4, $5,
                COALESCE($6, pgkg_default_org()),
                COALESCE($7, pgkg_default_collection()),
                COALESCE($8, 'org'), COALESCE($9, 'shared'), $10, $11)
        RETURNING id
        """,
        text,
        namespace,
        subject_id,
        object_id,
        predicate,
        org_id,
        collection_id,
        claim_scope,
        visibility,
        owner_user_id,
        acl_group_id,
    )


async def insert_edge(
    conn: asyncpg.Connection,
    *,
    src: uuid.UUID,
    dst: uuid.UUID,
    proposition_id: uuid.UUID,
    relation: str = "related_to",
) -> None:
    await conn.execute(
        """
        INSERT INTO edges (src_entity, dst_entity, relation, proposition_id)
        VALUES ($1, $2, $3, $4)
        """,
        src,
        dst,
        relation,
        proposition_id,
    )


SEARCH = """
SELECT proposition_id, source_kind
FROM pgkg_search($1, NULL, 50, 200, $2, NULL, 30.0, TRUE, 60, $3, $4, $5, $6)
"""


async def search(
    conn: asyncpg.Connection,
    *,
    q: str,
    namespace: str,
    org_ids: list[uuid.UUID] | None = None,
    collection_ids: list[uuid.UUID] | None = None,
    user_id: uuid.UUID | None = None,
    acl_groups: list[uuid.UUID] | None = None,
) -> list[asyncpg.Record]:
    return await conn.fetch(
        SEARCH, q, namespace, org_ids, collection_ids, user_id, acl_groups
    )


# ---------------------------------------------------------------------------
# Reserved partitions and defaults
# ---------------------------------------------------------------------------


async def test_reserved_orgs_are_seeded(pool: asyncpg.Pool) -> None:
    """The tenant default and the operator's system org both exist as rows, so
    every scoping column can be NOT NULL with a foreign key."""
    async with pool.acquire() as conn:
        default_is_system = await conn.fetchval(
            "SELECT is_system FROM orgs WHERE id = pgkg_default_org()"
        )
        system_is_system = await conn.fetchval(
            "SELECT is_system FROM orgs WHERE id = pgkg_system_org()"
        )

    assert default_is_system is False
    assert system_is_system is True


async def test_unscoped_insert_lands_in_the_default_partition(
    pool: asyncpg.Pool,
) -> None:
    """A caller that knows nothing about tenancy still writes a fully scoped
    row: that is what keeps the pre-tenancy surface working."""
    namespace = ns()
    async with pool.acquire() as conn:
        prop_id = await conn.fetchval(
            "INSERT INTO propositions (text, namespace) VALUES ('legacy', $1) "
            "RETURNING id",
            namespace,
        )
        row = await conn.fetchrow(
            """
            SELECT org_id = pgkg_default_org()               AS default_org,
                   collection_id = pgkg_default_collection() AS default_collection,
                   claim_scope, visibility, owner_user_id, acl_group_id
            FROM propositions WHERE id = $1
            """,
            prop_id,
        )

    assert row["default_org"] is True
    assert row["default_collection"] is True
    assert row["claim_scope"] == "org"
    assert row["visibility"] == "shared"
    assert row["owner_user_id"] is None
    assert row["acl_group_id"] is None


@pytest.mark.parametrize("table", ["chunks", "entities", "documents"])
async def test_every_retrievable_table_carries_an_org(
    pool: asyncpg.Pool, table: str
) -> None:
    """Chunks become retrievable in their own right and entity resolution is
    org-wide, so both need the column now rather than in a later rewrite."""
    async with pool.acquire() as conn:
        has_org = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = $1 AND column_name = 'org_id'
                  AND is_nullable = 'NO'
            )
            """,
            table,
        )

    assert has_org is True


async def test_private_rows_must_name_their_owner(pool: asyncpg.Pool) -> None:
    """owner_user_id is set if and only if visibility is private; a private row
    with no owner would be invisible to everyone including its author."""
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn)
        user = await new_user(conn, org, "u1")

        with pytest.raises(asyncpg.CheckViolationError):
            await insert_proposition(
                conn, text="orphan secret", namespace=namespace,
                org_id=org, visibility="private",
            )

        with pytest.raises(asyncpg.CheckViolationError):
            await insert_proposition(
                conn, text="shared but owned", namespace=namespace,
                org_id=org, visibility="shared", owner_user_id=user,
            )


async def test_owner_must_be_a_real_user(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn)
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await insert_proposition(
                conn, text="ghost owner", namespace=namespace, org_id=org,
                visibility="private", owner_user_id=uuid.uuid4(),
            )


async def test_claim_scope_is_a_closed_vocabulary(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await insert_proposition(
                conn, text="nonsense scope", namespace=namespace,
                claim_scope="galactic",
            )


# ---------------------------------------------------------------------------
# The shared retrieval predicate
# ---------------------------------------------------------------------------


async def test_search_returns_only_the_requested_orgs(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        org_a = await new_org(conn, "a")
        org_b = await new_org(conn, "b")
        mine = await insert_proposition(
            conn, text="kingfisher nests in riverbanks",
            namespace=namespace, org_id=org_a,
        )
        theirs = await insert_proposition(
            conn, text="kingfisher hunts from a perch",
            namespace=namespace, org_id=org_b,
        )

        rows = await search(conn, q="kingfisher", namespace=namespace, org_ids=[org_a])

    found = {r["proposition_id"] for r in rows}
    assert mine in found
    assert theirs not in found


async def test_search_treats_several_orgs_as_partitions(pool: asyncpg.Pool) -> None:
    """The array form is what makes a subscribed shared collection a second
    partition rather than a special case in the predicate."""
    namespace = ns()
    async with pool.acquire() as conn:
        org_a = await new_org(conn, "a")
        org_b = await new_org(conn, "b")
        mine = await insert_proposition(
            conn, text="quokka grazes at dusk", namespace=namespace, org_id=org_a
        )
        shared = await insert_proposition(
            conn, text="quokka is a marsupial", namespace=namespace, org_id=org_b
        )

        rows = await search(
            conn, q="quokka", namespace=namespace, org_ids=[org_a, org_b]
        )

    assert {mine, shared} <= {r["proposition_id"] for r in rows}


async def test_search_returns_only_the_requested_collections(
    pool: asyncpg.Pool,
) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn)
        subscribed = await new_collection(conn, org_id=org, name="subscribed")
        unsubscribed = await new_collection(conn, org_id=org, name="unsubscribed")
        wanted = await insert_proposition(
            conn, text="zymurgy is the study of fermentation",
            namespace=namespace, org_id=org, collection_id=subscribed,
        )
        unwanted = await insert_proposition(
            conn, text="zymurgy appears in a crossword",
            namespace=namespace, org_id=org, collection_id=unsubscribed,
        )

        rows = await search(
            conn, q="zymurgy", namespace=namespace,
            org_ids=[org], collection_ids=[subscribed],
        )

    found = {r["proposition_id"] for r in rows}
    assert wanted in found
    assert unwanted not in found


async def test_vector_arm_is_scoped_too(pool: asyncpg.Pool) -> None:
    """Both candidate sources share the predicate; a filter applied to only the
    keyword arm would leak through the HNSW arm."""
    namespace = ns()
    target = vec(hot_index=3)
    async with pool.acquire() as conn:
        org_a = await new_org(conn, "a")
        org_b = await new_org(conn, "b")
        mine = await insert_proposition(
            conn, text="mine", namespace=namespace, org_id=org_a, embedding=target
        )
        theirs = await insert_proposition(
            conn, text="theirs", namespace=namespace, org_id=org_b, embedding=target
        )

        rows = await conn.fetch(
            f"""
            SELECT item_id FROM pgkg_vector_candidates(
                '{pg_vec(target)}', $1, NULL, 200, $2, NULL, NULL, NULL)
            """,
            namespace,
            [org_a],
        )

    found = {r["item_id"] for r in rows}
    assert mine in found
    assert theirs not in found


async def test_private_facts_are_invisible_to_other_users(
    pool: asyncpg.Pool,
) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn)
        alice = await new_user(conn, org, "alice")
        bob = await new_user(conn, org, "bob")
        team = await insert_proposition(
            conn, text="the team uses postgres", namespace=namespace, org_id=org
        )
        alice_only = await insert_proposition(
            conn, text="postgres password rotation for alice",
            namespace=namespace, org_id=org, claim_scope="user",
            visibility="private", owner_user_id=alice,
        )

        as_alice = await search(
            conn, q="postgres", namespace=namespace, org_ids=[org], user_id=alice
        )
        as_bob = await search(
            conn, q="postgres", namespace=namespace, org_ids=[org], user_id=bob
        )

    assert {team, alice_only} <= {r["proposition_id"] for r in as_alice}
    bob_sees = {r["proposition_id"] for r in as_bob}
    assert team in bob_sees
    assert alice_only not in bob_sees


async def test_acl_grouped_rows_need_the_group(pool: asyncpg.Pool) -> None:
    """Document ACLs are filtered inside the candidate source, and a caller
    that names no groups sees no grouped rows."""
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn)
        group = uuid.uuid4()
        open_row = await insert_proposition(
            conn, text="mitochondria produce atp", namespace=namespace, org_id=org
        )
        gated = await insert_proposition(
            conn, text="mitochondria appear in the restricted deck",
            namespace=namespace, org_id=org, acl_group_id=group,
        )

        without = await search(
            conn, q="mitochondria", namespace=namespace, org_ids=[org]
        )
        with_group = await search(
            conn, q="mitochondria", namespace=namespace, org_ids=[org],
            acl_groups=[group],
        )

    ungated = {r["proposition_id"] for r in without}
    assert open_row in ungated
    assert gated not in ungated
    assert {open_row, gated} <= {r["proposition_id"] for r in with_group}


# ---------------------------------------------------------------------------
# Graph expansion is re-filtered, not trusted
# ---------------------------------------------------------------------------


async def test_graph_expansion_cannot_reach_another_users_private_fact(
    pool: asyncpg.Pool,
) -> None:
    """Entity resolution is org-wide, so a shared entity is a bridge between
    users.  Every neighbour reached over that bridge is re-filtered through the
    same predicate as the seed, or the graph arm is a laundering channel."""
    namespace = ns()
    async with pool.acquire() as conn:
        org = await new_org(conn)
        alice = await new_user(conn, org, "alice")
        bob = await new_user(conn, org, "bob")
        hub = await insert_entity(conn, name="Postgres", namespace=namespace)
        other = await insert_entity(conn, name="Rotation Policy", namespace=namespace)

        seed = await insert_proposition(
            conn, text="kingfisher runs on postgres", namespace=namespace,
            org_id=org, subject_id=hub, predicate="runs_on",
        )
        bobs_secret = await insert_proposition(
            conn, text="bob rotated the production credential on tuesday",
            namespace=namespace, org_id=org, claim_scope="user",
            visibility="private", owner_user_id=bob,
            subject_id=hub, object_id=other, predicate="rotated_by",
        )
        await insert_edge(conn, src=hub, dst=other, proposition_id=bobs_secret)

        as_alice = await search(
            conn, q="kingfisher", namespace=namespace, org_ids=[org], user_id=alice
        )
        as_bob = await search(
            conn, q="kingfisher", namespace=namespace, org_ids=[org], user_id=bob
        )

    alice_sees = {r["proposition_id"] for r in as_alice}
    assert seed in alice_sees, "the seed must be retrieved or the walk never starts"
    assert bobs_secret not in alice_sees

    bob_rows = {r["proposition_id"]: r["source_kind"] for r in as_bob}
    assert bob_rows.get(bobs_secret) == "graph", (
        "the edge must be traversable for its owner, or the test above passes "
        "for the wrong reason"
    )


async def test_graph_expansion_cannot_cross_the_org_boundary(
    pool: asyncpg.Pool,
) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        org_a = await new_org(conn, "a")
        org_b = await new_org(conn, "b")
        hub = await insert_entity(conn, name="Shared Hub", namespace=namespace)
        other = await insert_entity(conn, name="Neighbour", namespace=namespace)

        seed = await insert_proposition(
            conn, text="quokka visits the shared hub", namespace=namespace,
            org_id=org_a, subject_id=hub, predicate="visits",
        )
        foreign = await insert_proposition(
            conn, text="competitor pricing model", namespace=namespace,
            org_id=org_b, subject_id=hub, object_id=other, predicate="prices",
        )
        await insert_edge(conn, src=hub, dst=other, proposition_id=foreign)

        scoped = await search(
            conn, q="quokka", namespace=namespace, org_ids=[org_a]
        )
        both = await search(
            conn, q="quokka", namespace=namespace, org_ids=[org_a, org_b]
        )

    assert seed in {r["proposition_id"] for r in scoped}
    assert foreign not in {r["proposition_id"] for r in scoped}
    assert foreign in {r["proposition_id"] for r in both}


# ---------------------------------------------------------------------------
# Row-level security
# ---------------------------------------------------------------------------


async def test_application_role_sees_only_the_org_in_the_guc(
    pool: asyncpg.Pool,
) -> None:
    """Defence in depth: a query with no tenancy predicate at all still cannot
    read another org's rows."""
    namespace = ns()
    async with pool.acquire() as conn:
        org_a = await new_org(conn, "a")
        org_b = await new_org(conn, "b")
        mine = await insert_proposition(
            conn, text="ours", namespace=namespace, org_id=org_a
        )
        await insert_proposition(
            conn, text="theirs", namespace=namespace, org_id=org_b
        )

        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('pgkg.org_id', $1, true)", str(org_a)
            )
            await conn.execute("SET LOCAL ROLE pgkg_app")
            visible = await conn.fetch(
                "SELECT id FROM propositions WHERE namespace = $1", namespace
            )

    assert [r["id"] for r in visible] == [mine]


async def test_application_role_without_a_guc_sees_the_default_org(
    pool: asyncpg.Pool,
) -> None:
    """Absent scoping resolves to the backfill org, so there is no state in
    which the policy means "everything"."""
    namespace = ns()
    async with pool.acquire() as conn:
        other = await new_org(conn, "other")
        legacy = await conn.fetchval(
            "INSERT INTO propositions (text, namespace) VALUES ('legacy', $1) "
            "RETURNING id",
            namespace,
        )
        await insert_proposition(
            conn, text="foreign", namespace=namespace, org_id=other
        )

        async with conn.transaction():
            await conn.execute("SET LOCAL ROLE pgkg_app")
            visible = await conn.fetch(
                "SELECT id FROM propositions WHERE namespace = $1", namespace
            )

    assert [r["id"] for r in visible] == [legacy]


async def test_application_role_cannot_write_into_another_org(
    pool: asyncpg.Pool,
) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        org_a = await new_org(conn, "a")
        org_b = await new_org(conn, "b")

        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('pgkg.org_id', $1, true)", str(org_a)
                )
                await conn.execute("SET LOCAL ROLE pgkg_app")
                await conn.execute(
                    "INSERT INTO propositions (text, namespace, org_id) "
                    "VALUES ('smuggled', $1, $2)",
                    namespace,
                    org_b,
                )


# ---------------------------------------------------------------------------
# The predicate must stay prunable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "function_name,expected",
    [("pgkg_current_org", "s"), ("pgkg_visible", "i")],
)
async def test_predicate_functions_are_not_volatile(
    pool: asyncpg.Pool, function_name: str, expected: str
) -> None:
    """A volatile predicate cannot be an index qualifier and cannot prune a
    partition, which would defeat the entire point of the columns."""
    async with pool.acquire() as conn:
        volatility = await conn.fetchval(
            "SELECT provolatile::TEXT FROM pg_proc WHERE proname = $1",
            function_name,
        )

    assert volatility == expected


async def _tenancy_plan(
    conn: asyncpg.Connection, namespace: str, org: uuid.UUID, target: list[float]
) -> str:
    rows = await conn.fetch(
        f"""
        EXPLAIN SELECT p.id
        FROM propositions p
        WHERE p.embedding IS NOT NULL
          AND p.namespace = $1
          AND p.superseded_by IS NULL
          AND pgkg_visible(p.org_id, p.collection_id, p.visibility,
                           p.owner_user_id, p.acl_group_id,
                           $2, NULL, NULL, NULL)
        ORDER BY p.embedding <=> '{pg_vec(target)}'
        LIMIT 3
        """,
        namespace,
        [org],
    )
    return "\n".join(r["QUERY PLAN"] for r in rows)


async def _seed_for_plan(conn: asyncpg.Connection, namespace: str) -> uuid.UUID:
    org = await new_org(conn)
    for i in range(20):
        await insert_proposition(
            conn, text=f"row {i}", namespace=namespace, org_id=org,
            embedding=vec(hot_index=i),
        )
    return org


async def test_tenancy_predicate_inlines_into_column_comparisons(
    pool: asyncpg.Pool,
) -> None:
    """The planner has to see the columns.  An opaque call in the WHERE clause
    is neither an index qualifier nor a pruning constraint, which is the whole
    reason the scope stopped being a string."""
    namespace = ns()
    target = vec(hot_index=7)
    async with pool.acquire() as conn:
        org = await _seed_for_plan(conn, namespace)
        plan = await _tenancy_plan(conn, namespace, org, target)

    assert "pgkg_visible" not in plan
    assert "org_id = ANY" in plan
    assert "visibility = 'shared'" in plan


async def test_tenancy_predicate_leaves_the_hnsw_index_usable(
    pool: asyncpg.Pool,
) -> None:
    """pgvector's HNSW does not know about the WHERE clause, so a predicate it
    cannot see turns top-k into a scan over every tenant.  With the sequential
    and sort paths priced out, the scoped query still resolves to the vector
    index rather than failing to find a path at all."""
    namespace = ns()
    target = vec(hot_index=7)
    async with pool.acquire() as conn:
        org = await _seed_for_plan(conn, namespace)
        async with conn.transaction():
            await conn.execute("SET LOCAL enable_seqscan = off")
            await conn.execute("SET LOCAL enable_sort = off")
            plan = await _tenancy_plan(conn, namespace, org, target)

    assert "prop_emb_idx" in plan


# ---------------------------------------------------------------------------
# Shard placement
# ---------------------------------------------------------------------------


async def test_shard_key_defaults_to_a_pool(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        shard = await conn.fetchval("SELECT pgkg_tenant_shard($1)", org)

    assert re.fullmatch(r"pool_\d+", shard)


async def test_a_whale_tenant_can_be_promoted_to_its_own_shard(
    pool: asyncpg.Pool,
) -> None:
    """Placement is a row, not a hash, so a dedicated partition is an UPDATE
    rather than a migration."""
    async with pool.acquire() as conn:
        org = await new_org(conn, "whale")
        await conn.execute(
            "INSERT INTO tenant_shards (org_id, shard_key) VALUES ($1, 'whale_01')",
            org,
        )
        shard = await conn.fetchval("SELECT pgkg_tenant_shard($1)", org)

    assert shard == "whale_01"
