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

The guard runs in the direction that can fail.  Pinning the set of tables that
HAVE row security only notices a table that gains a policy; the likelier
accident is a table that ships with an org column and never gains one, which is
what 040 did with entity_mentions and entity_links while this module stayed
green.  So the enumeration below starts from the tables that carry an org and
requires each to be policied or explicitly excused, and the older assertion is
kept because a policy still has to arrive with a case here.

Its remaining blind spot, stated rather than left to be discovered: a table
whose rows belong to an org without carrying the column — edges,
proposition_provenance, corroborations — is org-scoped only through the row it
references, and this guard cannot see it.
"""
from __future__ import annotations

import uuid

import asyncpg
import pytest

ORG_GUC = "pgkg.org_id"
SYSTEM_ORG = uuid.UUID("00000000-0000-0000-0000-000000000000")


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


async def _seed_entity_mentions(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    entity = await conn.fetchval(
        "INSERT INTO entities (name, type, namespace, org_id)"
        " VALUES ($1, 'concept', 'rls', $2) RETURNING id",
        f"mention_{uuid.uuid4().hex[:8]}",
        org,
    )
    chunk = await conn.fetchval(
        "INSERT INTO chunks (text, org_id) VALUES ($1, $2) RETURNING id",
        f"rls mention passage {uuid.uuid4().hex[:8]}",
        org,
    )
    await conn.execute(
        "INSERT INTO entity_mentions (entity_id, chunk_id, org_id, span_start,"
        " span_end) VALUES ($1, $2, $3, 0, 4)",
        entity,
        chunk,
        org,
    )


async def _seed_entity_links(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    """The shared side must live in the operator's org and the org side must
    not, so this row exists only for a tenant: 040's direction trigger rejects
    it in either other arrangement, and the policy is single-org for the same
    reason."""
    mine = await conn.fetchval(
        "INSERT INTO entities (name, type, namespace, org_id)"
        " VALUES ($1, 'concept', 'rls', $2) RETURNING id",
        f"link_own_{uuid.uuid4().hex[:8]}",
        org,
    )
    shared = await conn.fetchval(
        "INSERT INTO entities (name, type, namespace, org_id)"
        " VALUES ($1, 'concept', 'rls', pgkg_system_org()) RETURNING id",
        f"link_shared_{uuid.uuid4().hex[:8]}",
    )
    await conn.execute(
        "INSERT INTO entity_links (org_entity_id, shared_entity_id)"
        " VALUES ($1, $2)",
        mine,
        shared,
    )


async def _seed_corpus_stats(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO corpus_stats (kind, namespace, org_id, collection_id,"
        " n_total, total_len) VALUES ('proposition', $1, $2, $3, 3, 30)",
        f"rls_{uuid.uuid4().hex[:8]}",
        org,
        uuid.uuid4(),
    )


async def _seed_lexeme_df(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO lexeme_df (kind, namespace, lexeme, org_id, collection_id,"
        " df) VALUES ('proposition', $1, 'zzqhelios', $2, $3, 3)",
        f"rls_{uuid.uuid4().hex[:8]}",
        org,
        uuid.uuid4(),
    )


async def _seed_proposition_cache(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    key = f"rls_{uuid.uuid4().hex}"
    await conn.execute(
        "INSERT INTO proposition_cache (cache_key, chunk_hash, extractor_model,"
        " prompt_version, propositions, org_id)"
        " VALUES ($1, $1, 'm', 'v1', '[]'::jsonb, $2)",
        key,
        org,
    )


SEEDS = {
    "chunks": _seed_chunks,
    "collection_subscriptions": _seed_collection_subscriptions,
    "collections": _seed_collections,
    "corpus_stats": _seed_corpus_stats,
    "document_version_chunks": _seed_document_version_chunks,
    "document_versions": _seed_document_versions,
    "documents": _seed_documents,
    "entities": _seed_entities,
    "entity_links": _seed_entity_links,
    "entity_mentions": _seed_entity_mentions,
    "ingest_jobs": _seed_ingest_jobs,
    "lexeme_df": _seed_lexeme_df,
    "org_embedders": _seed_org_embedders,
    "proposition_cache": _seed_proposition_cache,
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


# Tables whose read side widens to the operator's org, so that D3's
# `org_id = ANY([tenant_org, SYSTEM_ORG])` can match its second element under
# the app role.  A table is here because the read path resolves rows of another
# org in the caller's read scope, never because the table looks shared:
# entity_links is absent because its org_id is the org side of 040's bridge by
# trigger and is never the system org, and the four request-scoped tables
# (users, tenant_shards, ingest_jobs, collection_subscriptions) and provenance
# are absent because no read path names another org's rows in them.
WIDENED = (
    "chunks",
    "collections",
    "corpus_stats",
    "document_version_chunks",
    "document_versions",
    "documents",
    "entities",
    "entity_mentions",
    "lexeme_df",
    "org_embedders",
    "proposition_cache",
    "propositions",
)

# A table that carries an org column and deliberately has no policy, with the
# reason it is not an isolation boundary.  Empty is the correct state: it exists
# so that excusing a table is a decision someone writes down here, rather than
# an omission the older assertion could not see.
UNPROTECTED_BY_DESIGN: dict[str, str] = {}


async def test_every_table_with_an_org_column_is_policied(
    pool: asyncpg.Pool,
) -> None:
    """The guard in the direction that fails: a table shipping with an org
    column and no policy is the accident, and 040 made it twice while every
    assertion in this module stayed green."""
    async with pool.acquire() as conn:
        carries_org = {
            r["relname"]
            for r in await conn.fetch(
                """
                SELECT c.relname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_attribute a
                  ON a.attrelid = c.oid AND a.attname = 'org_id' AND a.attnum > 0
                WHERE n.nspname = 'public' AND c.relkind = 'r'
                """
            )
        }
        protected = {
            r["relname"]
            for r in await conn.fetch(
                """
                SELECT c.relname
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relkind = 'r'
                  AND c.relrowsecurity
                """
            )
        }

    assert carries_org, "no table carries an org column, so this guard is vacuous"
    unguarded = sorted(carries_org - protected - set(UNPROTECTED_BY_DESIGN))
    assert not unguarded, (
        f"{unguarded} carry an org column, are reachable by pgkg_app and have "
        "no row-level policy: add one, or name the table in "
        "UNPROTECTED_BY_DESIGN with the reason its org column is not an "
        "isolation boundary"
    )
    stale = sorted(set(UNPROTECTED_BY_DESIGN) - carries_org)
    assert not stale, f"{stale} are excused but no longer carry an org column"


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

# ---------------------------------------------------------------------------
# The sharing seam, at the policy level.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table", WIDENED)
async def test_the_operators_rows_are_readable_by_a_tenant(
    pool: asyncpg.Pool, table: str
) -> None:
    """D3's predicate reads `org_id = ANY([tenant_org, SYSTEM_ORG])` and D4
    ships that seam early because it sits in the hot path.  While every policy
    was a single-org equality the second element could never match, so a tenant
    reading with include_system_org got its own rows and silently nothing
    else."""
    async with pool.acquire() as conn:
        tenant = await new_org(conn)
        handle = await SEEDS[table](conn, SYSTEM_ORG) or SYSTEM_ORG

        seen = await _count_as_app(conn, table=table, handle=handle, guc=tenant)

    assert seen > 0, (
        f"{table}: a tenant cannot read the operator's shared rows, so "
        "include_system_org has no effect on this table"
    )


@pytest.mark.parametrize(
    "table,statement",
    [
        (
            "propositions",
            "INSERT INTO propositions (text, org_id)"
            " VALUES ('promoted upward', pgkg_system_org())",
        ),
        (
            "chunks",
            "INSERT INTO chunks (text, org_id)"
            " VALUES ('promoted upward', pgkg_system_org())",
        ),
        (
            "entities",
            "INSERT INTO entities (name, type, namespace, org_id)"
            " VALUES ('Promoted Upward', 'concept', 'rls', pgkg_system_org())",
        ),
    ],
)
async def test_a_tenant_cannot_write_into_the_operators_org(
    pool: asyncpg.Pool, table: str, statement: str
) -> None:
    """The other half of the widening, and D4's first hard rule: nothing a
    tenant ingests is ever promoted into a shared collection.  Reads widen,
    writes do not, so WITH CHECK stays a single-org equality."""
    async with pool.acquire() as conn:
        tenant = await new_org(conn)
        async with conn.transaction():
            await conn.execute("SET LOCAL ROLE pgkg_app")
            await conn.execute("SELECT set_config($1, $2, true)", ORG_GUC, str(tenant))
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await conn.execute(statement)


# ---------------------------------------------------------------------------
# What the policies cost the plan.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
async def keyword_corpus(pool: asyncpg.Pool):
    """Enough passages, and long enough ones, that a sequential scan is the
    more expensive plan — otherwise the planner would choose one for reasons
    that have nothing to do with the policy."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await conn.fetchval(
            "INSERT INTO collections (org_id, owner_org_id, name, kind)"
            " VALUES ($1, $1, $2, 'corpus') RETURNING id",
            org,
            f"kw_{uuid.uuid4().hex[:8]}",
        )
        await conn.execute(
            """
            INSERT INTO chunks (text, span_start, span_end, org_id, collection_id)
            SELECT 'the reimbursement policy for lodging and equipment states '
                   || 'that expense claim ' || g || ' needs approval before the '
                   || 'operator reconciles the ledger '
                   || repeat('filler phrase ' || (g % 91) || ' ', 12),
                   0, 400, $1, $2
            FROM generate_series(1, 4000) g
            """,
            org,
            collection,
        )
        await conn.execute(
            """
            INSERT INTO chunks (text, span_start, span_end, org_id, collection_id)
            SELECT 'zqxwv unobtainium sesquipedalian needle ' || g, 0, 40, $1, $2
            FROM generate_series(1, 3) g
            """,
            org,
            collection,
        )
        # VACUUM as well as ANALYZE: with fastupdate on, a freshly bulk-loaded
        # GIN index still holds its pending list, and the planner charges a
        # bitmap scan for reading it — which would make a sequential scan the
        # cheaper plan for reasons that have nothing to do with the policy.
        await conn.execute("VACUUM ANALYZE chunks")
    return org, collection


async def test_the_policy_does_not_cost_the_keyword_arm_its_index(
    pool: asyncpg.Pool, keyword_corpus
) -> None:
    """`tsvector @@ tsquery` is ts_match_vq, and a qual whose function is not
    leakproof may not be used as an index condition on a table with a policy:
    it would be asked about rows the policy hides.  So the GIN index was
    unreachable under the one role the policies are written for, and every BM25
    arm degraded to a sequential scan whose cost grows with the table.

    The plan under the role is asserted to carry the policy's own qual, because
    `SET LOCAL ROLE` outside a transaction block is a no-op and a test that
    measures the owner twice would pass whatever the planner did.
    """
    org, _ = keyword_corpus
    probe = (
        "EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF) SELECT count(*)"
        " FROM chunks c WHERE c.tsv @@ to_tsquery('simple', 'unobtainium')"
    )
    async with pool.acquire() as conn:
        as_owner = "\n".join(r[0] for r in await conn.fetch(probe))
        async with conn.transaction():
            await conn.execute("SET LOCAL ROLE pgkg_app")
            await conn.execute("SELECT set_config($1, $2, true)", ORG_GUC, str(org))
            role = await conn.fetchval("SELECT current_user")
            as_app = "\n".join(r[0] for r in await conn.fetch(probe))

    assert role == "pgkg_app", "the role never changed, so nothing was measured"
    assert "pgkg.org_id" in as_app, (
        f"the policy is not in the plan, so it was not in force:\n{as_app}"
    )
    assert "chunk_tsv_idx" in as_owner, (
        f"the index is not the cheaper plan even as owner:\n{as_owner}"
    )
    assert "chunk_tsv_idx" in as_app, (
        "under the application role the GIN index is unreachable and the "
        f"keyword arm falls back to a sequential scan:\n{as_app}"
    )


async def test_the_reversed_operand_order_reaches_the_index_too(
    pool: asyncpg.Pool, keyword_corpus
) -> None:
    """`tsquery @@ tsvector` is a different function — ts_match_qv — and the
    planner tests leakproofness on the clause as written, before it commutes
    anything.  So marking only the form today's arms happen to use leaves the
    next arm one operand order away from the sequential scan 043 fixed, with
    nothing to say so.  This is the test that the guard is on the operator
    rather than on the style of the SQL that reaches it.
    """
    org, _ = keyword_corpus
    probe = (
        "EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF) SELECT count(*)"
        " FROM chunks c WHERE to_tsquery('simple', 'unobtainium') @@ c.tsv"
    )
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SET LOCAL ROLE pgkg_app")
            await conn.execute("SELECT set_config($1, $2, true)", ORG_GUC, str(org))
            role = await conn.fetchval("SELECT current_user")
            as_app = "\n".join(r[0] for r in await conn.fetch(probe))

    assert role == "pgkg_app", "the role never changed, so nothing was measured"
    assert "pgkg.org_id" in as_app, (
        f"the policy is not in the plan, so it was not in force:\n{as_app}"
    )
    assert "chunk_tsv_idx" in as_app, (
        "with the tsquery on the left the GIN index is unreachable under the "
        f"application role, which is the defect 043 fixed one way round:\n{as_app}"
    )

# ---------------------------------------------------------------------------
# The same mechanism on the gazetteer's three arms (issue #10).
# ---------------------------------------------------------------------------
#
# `entities` has been under row security since 020 and 043 did not touch the
# operators the gazetteer probes with, so every arm of
# pgkg_match_entity_mentions() was a sequential scan of the whole entity table
# under the one role the policies are written for — per candidate phrase, per
# chunk, on the ingest path.
#
# Each arm is written here the way the matcher writes it: the phrase side is a
# row of literals standing in for one row of the matcher's `usable` CTE, so the
# plan under test is the nested-loop index probe the matcher is costed on and
# not a simpler shape a constant would allow.

ARM_SQL = {
    "name": """
        SELECT count(*)
        FROM (VALUES ($1::uuid, $2::text)) AS u(org_id, phrase)
        JOIN entities e
          ON e.org_id = u.org_id
         AND e.gazetteer_name_key = u.phrase
    """,
    "alias": """
        SELECT count(*)
        FROM (VALUES ($1::uuid, $2::text)) AS u(org_id, phrase)
        JOIN entities e
          ON e.org_id = u.org_id
         AND e.gazetteer_alias_keys @> ARRAY[u.phrase]
    """,
    "fuzzy": """
        SELECT count(*)
        FROM (VALUES ($1::uuid, $2::text)) AS u(org_id, phrase)
        JOIN entities e
          ON e.org_id = u.org_id
         AND e.name % u.phrase
         AND similarity(e.gazetteer_name_key, u.phrase) >= 0.9
    """,
}

# The index each arm exists to probe, and the phrase that reaches exactly one
# entity through it.
ARM_INDEX = {
    "name": "entities_gazetteer_name_idx",
    "alias": "entities_gazetteer_alias_idx",
    "fuzzy": "entities_name_trgm_idx",
}

ARM_PHRASE = {
    "name": "zorbulon unobtainium programme",
    "alias": "zorbulon needle alias",
    # One transposition away from the name, which is what the trigram arm is
    # for: the typo and the missing accent, not guessing.
    "fuzzy": "zorbulon unobtanium programme",
}

# What each arm's index condition is, and therefore what has to be leakproof
# for the policy not to demote it to a filter.  The name arm is absent on
# purpose: its remedy is a stored column, not a claim about a function.
ARM_OPERATOR = {
    "alias": "arraycontains(anyarray, anyarray)",
    "fuzzy": "similarity_op(text, text)",
}


@pytest.fixture(scope="module")
async def gazetteer_corpus(pool: asyncpg.Pool):
    """Enough entities that the index is the cheaper plan.

    At a few hundred rows a sequential scan is what the planner should pick and
    every arm looks the same, so the fixture is sized where the choice is real
    — the scale the finding was measured at.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        namespace = f"gaz_{uuid.uuid4().hex[:8]}"
        await conn.execute("SELECT set_config($1, $2, false)", ORG_GUC, str(org))
        await conn.execute(
            """
            INSERT INTO entities (name, type, aliases, namespace, org_id)
            SELECT 'filler entity number ' || g, 'thing',
                   ARRAY['filler alias alpha ' || g, 'filler alias beta ' || g],
                   $1, $2
            FROM generate_series(1, 40000) g
            """,
            namespace,
            org,
        )
        await conn.execute(
            """
            INSERT INTO entities (name, type, aliases, namespace, org_id)
            VALUES ('Zorbulon Unobtainium Programme', 'thing',
                    ARRAY['Zorbulon Needle Alias'], $1, $2)
            """,
            namespace,
            org,
        )
        # VACUUM as well as ANALYZE, for the reason the keyword fixture gives:
        # a freshly bulk-loaded GIN index still holds its pending list, and the
        # planner charges a bitmap scan for reading it.
        await conn.execute("VACUUM ANALYZE entities")
    return org, namespace


async def _plan(conn: asyncpg.Connection, sql: str, *args: object) -> str:
    rows = await conn.fetch(
        f"EXPLAIN (ANALYZE, BUFFERS, COSTS OFF, TIMING OFF) {sql}", *args
    )
    return "\n".join(row[0] for row in rows)


async def _plan_as_app(
    conn: asyncpg.Connection, org: uuid.UUID, sql: str, *args: object
) -> str:
    """The plan the application role gets, with the role change inside a
    transaction: `SET LOCAL ROLE` outside one is scoped to asyncpg's implicit
    single-statement transaction, and a test that measured the owner twice
    would pass whatever the planner did."""
    async with conn.transaction():
        await conn.execute("SET LOCAL ROLE pgkg_app")
        await conn.execute("SELECT set_config($1, $2, true)", ORG_GUC, str(org))
        role = await conn.fetchval("SELECT current_user")
        plan = await _plan(conn, sql, *args)
    assert role == "pgkg_app", "the role never changed, so nothing was measured"
    assert "pgkg.org_id" in plan, (
        f"the policy is not in the plan, so it was not in force:\n{plan}"
    )
    return plan


@pytest.mark.parametrize("arm", sorted(ARM_SQL))
async def test_the_policy_does_not_cost_the_gazetteer_its_indexes(
    pool: asyncpg.Pool, gazetteer_corpus, arm: str
) -> None:
    """A qual whose function is not leakproof may not become an index
    condition on a table with a policy, so each of the three gazetteer arms
    was demoted to a filter over the whole entity table under pgkg_app while
    the owner got the index.  This is the ingest hot path: the matcher runs
    per chunk and probes once per candidate phrase.
    """
    org, _ = gazetteer_corpus
    index = ARM_INDEX[arm]
    async with pool.acquire() as conn:
        as_owner = await _plan(conn, ARM_SQL[arm], org, ARM_PHRASE[arm])
        as_app = await _plan_as_app(conn, org, ARM_SQL[arm], org, ARM_PHRASE[arm])

    assert index in as_owner, (
        f"the {arm} arm does not reach {index} even as owner, so the "
        f"comparison below would be vacuous:\n{as_owner}"
    )
    assert index in as_app, (
        f"under the application role the {arm} arm cannot reach {index} and "
        f"falls back to a sequential scan of every entity:\n{as_app}"
    )
    assert "Seq Scan on entities" not in as_app, (
        f"the {arm} arm still scans the whole entity table under the "
        f"application role:\n{as_app}"
    )


@pytest.mark.parametrize("arm", sorted(ARM_OPERATOR))
async def test_unmarking_the_operator_returns_the_arm_to_a_sequential_scan(
    pool: asyncpg.Pool, gazetteer_corpus, arm: str
) -> None:
    """The marking is what buys the plan, and this is the probe that proves it:
    with the operator's function unmarked the same arm reverts to a sequential
    scan under the role, and marking it again restores the index.  Without
    this a test asserting the index could be passing for any other reason.
    """
    org, _ = gazetteer_corpus
    function = ARM_OPERATOR[arm]
    index = ARM_INDEX[arm]
    async with pool.acquire() as conn:
        await conn.execute(f"ALTER FUNCTION {function} NOT LEAKPROOF")
        try:
            unmarked = await _plan_as_app(
                conn, org, ARM_SQL[arm], org, ARM_PHRASE[arm]
            )
        finally:
            await conn.execute(f"ALTER FUNCTION {function} LEAKPROOF")
        restored = await _plan_as_app(conn, org, ARM_SQL[arm], org, ARM_PHRASE[arm])

    assert index not in unmarked, (
        f"the {arm} arm reached {index} with {function} not leakproof, so the "
        f"marking is not what the other test is measuring:\n{unmarked}"
    )
    assert "Seq Scan on entities" in unmarked, (
        f"unmarking {function} did not put the {arm} arm back on a sequential "
        f"scan:\n{unmarked}"
    )
    assert index in restored, (
        f"re-marking {function} leakproof did not restore the index:\n{restored}"
    )


async def test_the_gazetteer_key_is_a_stored_column_and_not_a_leakproof_claim(
    pool: asyncpg.Pool, gazetteer_corpus
) -> None:
    """Why the name and alias arms read columns rather than calling the
    normaliser.  A leakproof marking on pgkg_gazetteer_key() would buy
    nothing — the planner inlines a SQL function's body before it judges the
    qual, and the body is lower/regexp_replace/btrim, none of them leakproof —
    so the arm would need those three marked instead, which is a far wider
    claim than this schema needs.  The stored column moves the normalisation to
    write time and leaves the qual an equality on text, which is leakproof
    already.
    """
    org, _ = gazetteer_corpus
    as_expression = """
        SELECT count(*)
        FROM (VALUES ($1::uuid, $2::text)) AS u(org_id, phrase)
        JOIN entities e
          ON e.org_id = u.org_id
         AND pgkg_gazetteer_key(e.name) = u.phrase
    """
    async with pool.acquire() as conn:
        claimed = {
            row["proname"]: row["proleakproof"]
            for row in await conn.fetch(
                "SELECT proname, proleakproof FROM pg_proc"
                " WHERE proname = ANY($1::text[])",
                ["pgkg_gazetteer_key", "pgkg_gazetteer_keys"],
            )
        }
        expression_plan = await _plan_as_app(
            conn, org, as_expression, org, ARM_PHRASE["name"]
        )
        column_plan = await _plan_as_app(
            conn, org, ARM_SQL["name"], org, ARM_PHRASE["name"]
        )

    assert claimed and not any(claimed.values()), (
        f"a gazetteer normaliser is marked leakproof: {claimed}. The claim "
        "cannot be honoured — the body is inlined before the qual is judged — "
        "and it would outlive any review of what the body does."
    )
    assert "Seq Scan on entities" in expression_plan, (
        "the expression form of the name arm no longer needs the stored "
        f"column, so this test has stopped explaining anything:\n"
        f"{expression_plan}"
    )
    assert ARM_INDEX["name"] in column_plan, (
        "the stored column is what makes the name arm's qual promotable:\n"
        f"{column_plan}"
    )


async def test_the_stored_gazetteer_keys_agree_with_the_normaliser(
    pool: asyncpg.Pool, gazetteer_corpus
) -> None:
    """A stored key is only a valid substitute for the call if it says the same
    thing, so the two are compared over every seeded entity rather than on one
    row: the column is the function's answer for that row, maintained by the
    database and unable to drift.
    """
    org, namespace = gazetteer_corpus
    async with pool.acquire() as conn:
        disagreements = await conn.fetchval(
            """
            SELECT count(*) FROM entities e
            WHERE e.namespace = $1
              AND (e.gazetteer_name_key
                       IS DISTINCT FROM pgkg_gazetteer_key(e.name)
                OR e.gazetteer_alias_keys
                       IS DISTINCT FROM pgkg_gazetteer_keys(e.aliases))
            """,
            namespace,
        )
        seeded = await conn.fetchval(
            "SELECT count(*) FROM entities WHERE namespace = $1", namespace
        )

    assert seeded == 40001, f"the fixture seeded {seeded} entities"
    assert disagreements == 0, (
        f"{disagreements} entities carry a stored gazetteer key that differs "
        "from what the normaliser returns for that row"
    )


async def test_the_matcher_probes_the_stored_keys(pool: asyncpg.Pool) -> None:
    """And the arm that no test can reach through a plan: the matcher's own
    body has to name the columns, or the indexes above are reachable by a
    query nobody runs.
    """
    async with pool.acquire() as conn:
        body = await conn.fetchval(
            "SELECT prosrc FROM pg_proc WHERE proname = $1",
            "pgkg_match_entity_mentions",
        )

    assert "gazetteer_name_key" in body and "gazetteer_alias_keys" in body, (
        "pgkg_match_entity_mentions() does not probe the stored gazetteer "
        "keys, so its quals are still the non-promotable expression form"
    )
    assert "pgkg_gazetteer_key(e.name)" not in body, (
        "pgkg_match_entity_mentions() still normalises the entity's name per "
        "row, which is the qual the policy demotes to a filter"
    )
async def test_the_application_role_can_still_maintain_the_statistics(
    pool: asyncpg.Pool,
) -> None:
    """corpus_stats and lexeme_df are written by triggers on the content
    tables, so a policy on them is a policy on the ingest path: an INSERT the
    app role is allowed to make must not fail, or be silently lost, because the
    statistics row it moves is behind a boundary.  The row's org is the writer's
    org, which is why the single-org WITH CHECK is the right one here."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await conn.fetchval(
            "INSERT INTO collections (org_id, owner_org_id, name, kind)"
            " VALUES ($1, $1, $2, 'chat') RETURNING id",
            org,
            f"stats_{uuid.uuid4().hex[:8]}",
        )
        namespace = f"stats_{uuid.uuid4().hex[:8]}"

        async with conn.transaction():
            await conn.execute("SET LOCAL ROLE pgkg_app")
            await conn.execute("SELECT set_config($1, $2, true)", ORG_GUC, str(org))
            await conn.execute(
                "INSERT INTO propositions (text, namespace, org_id, collection_id)"
                " VALUES ('zzqhelios reconciles the ledger', $1, $2, $3)",
                namespace,
                org,
                collection,
            )

        totals = await conn.fetchval(
            "SELECT n_total FROM corpus_stats WHERE kind = 'proposition'"
            " AND namespace = $1 AND org_id = $2 AND collection_id = $3",
            namespace,
            org,
            collection,
        )
        df = await conn.fetchval(
            "SELECT df FROM lexeme_df WHERE kind = 'proposition'"
            " AND namespace = $1 AND lexeme = 'zzqhelio' AND org_id = $2"
            " AND collection_id = $3",
            namespace,
            org,
            collection,
        )

    assert totals == 1, f"the app role's write did not reach corpus_stats: {totals}"
    assert df == 1, f"the app role's write did not reach lexeme_df: {df}"
