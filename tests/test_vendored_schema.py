"""pgkg installed into a schema that is not `public`, called from outside it.

A host application that vendors pgkg puts it in a schema of its own, next to
its own tables, and does not put that schema on its search_path (issue #30).
Everything pgkg evaluates on the caller's behalf then has to find pgkg's
objects without the caller's help: an RLS policy's function is inlined under
the caller's path, a trigger runs under it, and so does every retrieval
function.  Before the bodies were qualified, the first of those failed with
`function pgkg_default_org() does not exist ... during inlining`.

So this module installs a second copy of the schema into `pgkg_host`, with its
extensions in `pgkg_ext`, in a database of its own, and drives it from
connections whose search_path is ''.  Nothing in public, and nothing on the
path: a reference that is not qualified has nowhere to resolve.

Qualified, and not `SET search_path` on the functions: a function carrying a
SET clause is never inlined, which would turn every predicate below from an
index condition into a per-row call.  The plan assertions are what hold that.
"""
from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from pgvector import HalfVector
from pgvector.asyncpg import register_vector

from body_lint import Names, unqualified_references
from pgkg.db import make_pool
from pgkg.migrate import MIGRATIONS_DIR, UntrackedInstallError, apply_migrations

SCHEMA = "pgkg_host"
EXTENSION_SCHEMA = "pgkg_ext"
DIM = 1024
DEFAULT_ORG = uuid.UUID("00000000-0000-0000-0000-000000000001")
DEFAULT_COLLECTION = uuid.UUID("00000000-0000-0000-0000-000000000002")


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def vec(hot_index: int) -> HalfVector:
    raw = [0.0] * DIM
    raw[hot_index] = 1.0
    return HalfVector(raw)


def _with_database(dsn: str, database: str) -> str:
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{database}"))


@asynccontextmanager
async def _scratch_database(pg_dsn: str, prefix: str) -> AsyncIterator[str]:
    """A database of its own, dropped afterwards, so an install in it cannot
    meet the suite's."""
    database = unique(prefix)
    admin = await asyncpg.connect(pg_dsn)
    try:
        await admin.execute(f"CREATE DATABASE {database}")
    finally:
        await admin.close()
    try:
        yield _with_database(pg_dsn, database)
    finally:
        admin = await asyncpg.connect(pg_dsn)
        try:
            await admin.execute(f"DROP DATABASE {database} WITH (FORCE)")
        finally:
            await admin.close()


@pytest.fixture(scope="module")
async def vendored_dsn(pg_dsn: str) -> AsyncIterator[str]:
    """pgkg in pgkg_host, with its extensions in a schema the operator made
    beforehand and granted nothing on, which is what `CREATE SCHEMA` leaves:
    the runner has to give the application role what it needs there."""
    async with _scratch_database(pg_dsn, "pgkg_vendored") as dsn:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(f"CREATE SCHEMA {EXTENSION_SCHEMA}")
            await apply_migrations(
                conn, schema=SCHEMA, extension_schema=EXTENSION_SCHEMA
            )
        finally:
            await conn.close()

        yield dsn


async def _connect_without_a_path(dsn: str) -> asyncpg.Connection:
    """A caller that has neither pgkg's schema nor the extensions' on its path."""
    conn = await asyncpg.connect(dsn, server_settings={"search_path": ""})
    await register_vector(conn, schema=EXTENSION_SCHEMA)
    return conn


@pytest.fixture
async def caller(vendored_dsn: str) -> AsyncIterator[asyncpg.Connection]:
    conn = await _connect_without_a_path(vendored_dsn)
    try:
        yield conn
    finally:
        await conn.close()


async def _plan(conn: asyncpg.Connection, sql: str, *args: object) -> str:
    rows = await conn.fetch(f"EXPLAIN (COSTS OFF) {sql}", *args)
    return "\n".join(row[0] for row in rows)


async def _insert_proposition(
    conn: asyncpg.Connection, *, text: str, namespace: str, embedding: HalfVector
) -> uuid.UUID:
    return await conn.fetchval(
        f"""
        INSERT INTO {SCHEMA}.propositions (text, namespace, embedding, org_id)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """,
        text, namespace, embedding, DEFAULT_ORG,
    )


# ---------------------------------------------------------------------------
# Where the install lands
# ---------------------------------------------------------------------------


async def test_the_install_lands_in_its_schema_and_leaves_public_empty(
    caller: asyncpg.Connection,
) -> None:
    placed = {
        row["relname"]: row["nspname"]
        for row in await caller.fetch(
            """
            SELECT c.relname, n.nspname
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname IN ('orgs', 'propositions', 'chunks',
                                'pgkg_schema_migrations')
            """
        )
    }
    in_public = await caller.fetchval(
        """
        SELECT count(*) FROM pg_catalog.pg_class c
        WHERE c.relnamespace = 'public'::pg_catalog.regnamespace
        """
    )
    extensions = {
        row["extname"]: row["nspname"]
        for row in await caller.fetch(
            """
            SELECT e.extname, n.nspname
            FROM pg_catalog.pg_extension e
            JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
            WHERE e.extname IN ('vector', 'pg_trgm', 'pgcrypto')
            """
        )
    }

    assert placed == {
        "orgs": SCHEMA,
        "propositions": SCHEMA,
        "chunks": SCHEMA,
        "pgkg_schema_migrations": SCHEMA,
    }
    assert in_public == 0
    assert extensions == {
        "vector": EXTENSION_SCHEMA,
        "pg_trgm": EXTENSION_SCHEMA,
        "pgcrypto": EXTENSION_SCHEMA,
    }


async def test_a_second_run_applies_nothing(vendored_dsn: str) -> None:
    conn = await asyncpg.connect(vendored_dsn)
    try:
        applied = await apply_migrations(
            conn, schema=SCHEMA, extension_schema=EXTENSION_SCHEMA
        )
    finally:
        await conn.close()

    assert applied == ()


async def test_pgkg_migrate_installs_into_the_configured_schema(
    pg_dsn: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI and the test fixture share one runner, so what this suite
    installs is what `pgkg migrate` installs."""
    from pgkg import cli
    from pgkg.config import Settings

    async with _scratch_database(pg_dsn, "pgkg_cli") as dsn:
        settings = Settings(
            _env_file=None, db_schema="pgkg_cli_host", extension_schema="pgkg_cli_ext"
        )
        monkeypatch.setattr("pgkg.config.get_settings", lambda: settings)

        await cli.run_migrate(dsn)

        conn = await asyncpg.connect(dsn)
        try:
            recorded = await conn.fetchval(
                "SELECT count(*) FROM pgkg_cli_host.pgkg_schema_migrations"
            )
        finally:
            await conn.close()

    assert recorded == len(list(MIGRATIONS_DIR.glob("*.sql")))
    assert "Applying 001_extensions.sql..." in capsys.readouterr().out


async def test_an_untracked_install_is_refused_rather_than_reapplied(
    pg_dsn: str,
) -> None:
    """pgkg objects in the schema and no tracking table there means the
    tracking table was lost or the schema was installed another way.  Running
    from 001 would fail part-way at best; the runner says what to do instead."""
    async with _scratch_database(pg_dsn, "pgkg_untracked") as dsn:
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute("CREATE SCHEMA pgkg_old")
            await conn.execute("CREATE TABLE pgkg_old.orgs (id UUID PRIMARY KEY)")

            with pytest.raises(UntrackedInstallError, match="baseline"):
                await apply_migrations(
                    conn, schema="pgkg_old", extension_schema="pgkg_old"
                )
            tracked = await conn.fetchval(
                "SELECT to_regclass('pgkg_old.pgkg_schema_migrations') IS NOT NULL"
            )
        finally:
            await conn.close()

    assert not tracked


async def test_an_extension_schema_nothing_is_created_in_is_not_created(
    pg_dsn: str,
) -> None:
    """Every extension already installed elsewhere leaves the configured
    extension schema with nothing to hold, so the runner does not make it."""
    async with _scratch_database(pg_dsn, "pgkg_ext_elsewhere") as dsn:
        conn = await asyncpg.connect(dsn)
        try:
            for extension in ("vector", "pg_trgm", "pgcrypto"):
                await conn.execute(f"CREATE EXTENSION {extension} SCHEMA public")
            await apply_migrations(
                conn, schema="pgkg_elsewhere", extension_schema="pgkg_unused"
            )
            created = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'pgkg_unused')"
            )
        finally:
            await conn.close()

    assert not created


async def test_the_application_pool_finds_pgkg_in_its_schema(vendored_dsn: str) -> None:
    """The application's own queries are unqualified, so its pool puts pgkg's
    schema on the path, and the extensions' after it for the vector codec and
    the casts.  Twice, because asyncpg runs RESET ALL between acquires and a
    path set any way but at startup would survive exactly one."""
    pool = await make_pool(vendored_dsn, schema=SCHEMA)
    try:
        for _ in range(2):
            async with pool.acquire() as conn:
                path = await conn.fetchval("SHOW search_path")
                orgs = await conn.fetchval("SELECT count(*) FROM orgs")
                echoed = await conn.fetchval("SELECT $1::halfvec", vec(5))
    finally:
        await pool.close()

    assert path == f"{SCHEMA}, {EXTENSION_SCHEMA}"
    assert orgs >= 2
    assert echoed == vec(5)


# ---------------------------------------------------------------------------
# What pgkg evaluates on the caller's behalf
# ---------------------------------------------------------------------------


async def test_a_row_security_policy_resolves_and_inlines_without_a_path(
    caller: asyncpg.Connection,
) -> None:
    """The policy calls pgkg_current_org(), which calls pgkg_default_org(): the
    second call is the one that failed during inlining."""
    async with caller.transaction():
        await caller.execute("SET LOCAL ROLE pgkg_app")
        await caller.execute(
            "SELECT pg_catalog.set_config('pgkg.org_id', $1, true)", str(DEFAULT_ORG)
        )
        visible = await caller.fetchval(f"SELECT count(*) FROM {SCHEMA}.propositions")
        plan = await _plan(caller, f"SELECT id FROM {SCHEMA}.propositions")

    assert visible >= 0
    assert "pgkg.org_id" in plan, f"the policy is not in the plan:\n{plan}"
    assert "pgkg_current_org" not in plan, (
        f"the policy's function was not inlined:\n{plan}"
    )


async def test_the_application_role_retrieves_through_an_operator_made_extension_schema(
    caller: asyncpg.Connection,
) -> None:
    """pgkg_app reaches halfvec, `<=>` and similarity() through the extension
    schema, so it needs USAGE there.  A schema the operator created has none
    for it, and the runner grants exactly that: to pgkg_app, not to PUBLIC."""
    async with caller.transaction():
        await caller.execute("SET LOCAL ROLE pgkg_app")
        await caller.execute(
            "SELECT pg_catalog.set_config('pgkg.org_id', $1, true)", str(DEFAULT_ORG)
        )
        rows = await caller.fetch(
            f"""
            SELECT item_id FROM {SCHEMA}.pgkg_retrieve(
                'helios', $1::{EXTENSION_SCHEMA}.halfvec,
                p_namespace => 'nowhere', p_org_ids => $2::uuid[]
            )
            """,
            vec(3), [DEFAULT_ORG],
        )
    public_usage = await caller.fetchval(
        "SELECT pg_catalog.has_schema_privilege('public', $1, 'USAGE')",
        EXTENSION_SCHEMA,
    )

    assert rows == []
    assert not public_usage


async def test_a_statistics_trigger_fires_without_a_path(
    caller: asyncpg.Connection,
) -> None:
    namespace = unique("stats")
    await _insert_proposition(
        caller, text="the helios migration ships", namespace=namespace,
        embedding=vec(1),
    )

    n_total = await caller.fetchval(
        f"""
        SELECT n_total FROM {SCHEMA}.corpus_stats
        WHERE namespace = $1 AND kind = 'proposition'
        """,
        namespace,
    )

    assert n_total == 1


async def test_retrieval_answers_without_a_path(caller: asyncpg.Connection) -> None:
    namespace = unique("retrieve")
    wanted = await _insert_proposition(
        caller, text="the helios migration ships in march", namespace=namespace,
        embedding=vec(3),
    )
    await _insert_proposition(
        caller, text="an unrelated fact about lunch", namespace=namespace,
        embedding=vec(9),
    )

    rows = await caller.fetch(
        f"""
        SELECT item_id FROM {SCHEMA}.pgkg_retrieve(
            $1, $2::{EXTENSION_SCHEMA}.halfvec,
            p_namespace => $3, p_org_ids => $4::uuid[]
        )
        """,
        "helios migration", vec(3), namespace, [DEFAULT_ORG],
    )

    assert rows and rows[0]["item_id"] == wanted


async def test_the_gazetteer_matches_without_a_path(caller: asyncpg.Connection) -> None:
    """The matcher's fuzzy arm is pg_trgm's `%`, an extension operator: under
    an empty path it resolves only if the body names its schema."""
    name = unique("Zorbulon Programme")
    entity = await caller.fetchval(
        f"""
        INSERT INTO {SCHEMA}.entities (name, type, namespace, org_id)
        VALUES ($1, 'thing', 'default', $2) RETURNING id
        """,
        name, DEFAULT_ORG,
    )
    chunk = await caller.fetchval(
        f"""
        INSERT INTO {SCHEMA}.chunks (text, org_id, collection_id)
        VALUES ($1, $2, $3) RETURNING id
        """,
        f"Notes on the {name} and its budget.", DEFAULT_ORG, DEFAULT_COLLECTION,
    )

    await caller.execute(
        f"SELECT {SCHEMA}.pgkg_match_entity_mentions($1::uuid[])", [chunk]
    )
    mentioned = await caller.fetchval(
        f"SELECT count(*) FROM {SCHEMA}.entity_mentions WHERE entity_id = $1",
        entity,
    )

    assert mentioned == 1


# ---------------------------------------------------------------------------
# Qualification must not cost the plans their indexes
# ---------------------------------------------------------------------------


VISIBLE_QUERY = f"""
    SELECT p.id
    FROM {SCHEMA}.propositions p
    WHERE p.embedding IS NOT NULL
      AND p.namespace = $1
      AND p.superseded_by IS NULL
      AND {SCHEMA}.pgkg_visible(p.org_id, p.collection_id, p.visibility,
                                p.owner_user_id, p.acl_group_id,
                                $2, NULL, NULL, NULL)
    ORDER BY p.embedding OPERATOR({EXTENSION_SCHEMA}.<=>) $3
    LIMIT 3
"""


async def test_the_visibility_predicate_still_inlines_into_column_comparisons(
    caller: asyncpg.Connection,
) -> None:
    plan = await _plan(caller, VISIBLE_QUERY, "ns", [DEFAULT_ORG], vec(7))

    assert "pgkg_visible" not in plan, f"pgkg_visible was not inlined:\n{plan}"
    assert "org_id = ANY" in plan


async def test_the_vector_arm_still_reaches_its_index(
    caller: asyncpg.Connection,
) -> None:
    async with caller.transaction():
        await caller.execute("SET LOCAL enable_seqscan = off")
        await caller.execute("SET LOCAL enable_sort = off")
        plan = await _plan(caller, VISIBLE_QUERY, "ns", [DEFAULT_ORG], vec(7))

    assert "prop_emb_idx" in plan, plan


async def test_the_keyword_arm_still_inlines_into_an_index_scan(
    caller: asyncpg.Connection,
) -> None:
    """A retrieval arm is only as prunable as the predicate it inlines: with
    the sequential scan priced out, the scope has to reach the proposition
    scan as an index condition, which a Function Scan over an opaque arm
    cannot give it."""
    sql = f"""
        SELECT b.item_id FROM {SCHEMA}.pgkg_bm25_candidates(
            'helios migration', 'ns', NULL, 10, $1::uuid[]
        ) b
    """
    async with caller.transaction():
        await caller.execute("SET LOCAL enable_seqscan = off")
        plan = await _plan(caller, sql, [DEFAULT_ORG])

    # 059 made pgkg_bm25_candidates a dispatcher over the policy path and a
    # gated owner arm; the gate stays a Function Scan behind a one-time filter,
    # and the dispatcher and the policy path must both inline.
    assert not re.search(
        r"Function Scan on pgkg_bm25_candidates(_under_policy)?\b", plan
    ), f"the keyword arm was not inlined:\n{plan}"
    assert re.search(r"Index Scan using \w+ on propositions p", plan), plan
    assert "Index Cond: (namespace = 'ns'::text)" in plan, plan


# ---------------------------------------------------------------------------
# Every body, not only the ones exercised above
# ---------------------------------------------------------------------------


async def test_every_sql_function_body_resolves_without_a_path(
    caller: asyncpg.Connection,
) -> None:
    """A SQL-language body is parsed and resolved when it is created, so
    re-creating each one under an empty path is an exact check of every name
    it uses — relations, functions, types and extension operators alike.
    pg_get_functiondef qualifies the header for the path it runs under; the
    body is the text as installed."""
    functions = await caller.fetch(
        """
        SELECT p.oid::pg_catalog.regprocedure::TEXT AS signature,
               pg_catalog.pg_get_functiondef(p.oid) AS definition
        FROM pg_catalog.pg_proc p
        JOIN pg_catalog.pg_language l ON l.oid = p.prolang
        WHERE p.pronamespace = $1::pg_catalog.regnamespace
          AND l.lanname = 'sql'
        """,
        SCHEMA,
    )
    unresolved: dict[str, str] = {}
    for function in functions:
        try:
            async with caller.transaction():
                await caller.execute("SET LOCAL check_function_bodies = on")
                await caller.execute(function["definition"])
                raise _RolledBack
        except _RolledBack:
            pass
        except asyncpg.PostgresError as error:
            unresolved[function["signature"]] = str(error)

    assert functions, "no SQL functions found, so nothing was checked"
    assert unresolved == {}


class _RolledBack(Exception):
    """Raised inside a transaction only to roll it back."""


# The names a body could reach without a schema, read rather than executed: a
# plpgsql body is resolved statement by statement as it runs, so re-creating it
# proves nothing.  tests/body_lint.py reads it the way the server would, with
# the strings handed to EXECUTE and format() read as the SQL they are.
#
# Before trusting it over the catalog, it is shown each kind of miss it exists
# to catch, on a fixed set of names shaped like the real ones.
LINT_NAMES = Names(
    relations=frozenset({"chunks", "orgs", "propositions", "provenance", "pgkg_candidate"}),
    also_columns=frozenset({"propositions", "provenance"}),
    functions=frozenset({"pgkg_current_org", "similarity", "vector_dims", "digest"}),
    types=frozenset({"vector", "halfvec", "sparsevec"}),
    opclasses=frozenset({"halfvec_cosine_ops"}),
    extension_operators=frozenset({"<=>", "<#>", "<%", "%>", "<<%", "%>>"}),
    shared_operators=frozenset({"%", "<->"}),
)


@pytest.mark.parametrize(
    "body, expected",
    [
        ("SELECT a%b FROM s.chunks", "operator %"),
        ("SELECT 1 WHERE lower(a) % lower(b)", "operator %"),
        ("SELECT 1 WHERE name <% p_name", "operator <%"),
        ("SELECT 1 WHERE name %> p_name", "operator %>"),
        ("SELECT a <=> b", "operator <=>"),
        ("SELECT similarity(a, b)", "function similarity()"),
        ("IF vector_dims(q) <> 3 THEN RETURN; END IF;", "function vector_dims()"),
        ("SELECT digest(t, 'sha256')", "function digest()"),
        ("SELECT s.pgkg_x() WHERE org = pgkg_current_org()", "function pgkg_current_org()"),
        ("SELECT '[1,2]'::halfvec", "type halfvec"),
        (
            "EXECUTE format('CREATE INDEX %I ON s.%I USING hnsw "
            "(vec halfvec_cosine_ops)', a, b);",
            "operator class halfvec_cosine_ops",
        ),
        (
            "EXECUTE format('CREATE TABLE s.%I (vec halfvec(%s))', t, d);",
            "type halfvec",
        ),
        ("PERFORM 'chunks'::regclass;", "name resolved at run time: 'chunks'"),
        (
            "IF to_regclass('orgs') IS NULL THEN RETURN; END IF;",
            "name resolved at run time: 'orgs'",
        ),
        ("EXECUTE 'ANALYZE chunks';", "relation chunks"),
        ("EXECUTE 'VACUUM ' || 'orgs';", "relation orgs"),
        ("DECLARE v chunks.id%TYPE; BEGIN END;", "relation chunks in %TYPE"),
        ("DECLARE r orgs%ROWTYPE; BEGIN END;", "relation orgs in %ROWTYPE"),
        ("SELECT (a, b)::pgkg_candidate", "relation pgkg_candidate"),
        ("INSERT INTO propositions (text) VALUES ('x')", "relation propositions"),
        ("RETURN QUERY EXECUTE format($q$ SELECT 1 FROM chunks $q$);", "relation chunks"),
    ],
)
def test_the_body_lint_finds_each_kind_of_unqualified_name(
    body: str, expected: str
) -> None:
    assert expected in unqualified_references(body, LINT_NAMES)


def test_the_body_lint_accepts_the_qualified_forms() -> None:
    body = """
    DECLARE
        v s.chunks.id%TYPE;
        r s.orgs%ROWTYPE;
    BEGIN
        WITH orgs AS (SELECT 1) SELECT * FROM orgs;
        SELECT e.similarity(a, b), x OPERATOR(e.<=>) y, x OPERATOR(e.%) y,
               '[1]'::e.halfvec, n % 64, 's.chunks'::regclass,
               to_regclass('s.' || v_table), c.propositions
        FROM s.chunks c JOIN s.propositions AS p ON TRUE;
        INSERT INTO s.provenance (propositions) VALUES (1);
        EXECUTE format('CREATE INDEX %I ON s.%I USING hnsw (vec e.halfvec_cosine_ops)', a, b);
        RAISE NOTICE 'chunks % and orgs %% gone: similarity(x)', v;
        -- a comment naming chunks, similarity() and <=>
        PERFORM s.pgkg_current_org();
    END;
    """

    assert unqualified_references(body, LINT_NAMES) == []


async def test_the_catalog_supplies_every_kind_of_name(caller: asyncpg.Connection) -> None:
    names = await Names.from_catalog(
        caller, schema=SCHEMA, extension_schemas=[EXTENSION_SCHEMA]
    )

    assert {"chunks", "orgs", "pgkg_candidate"} <= names.relations
    assert {"pgkg_current_org", "similarity", "vector_dims", "digest"} <= names.functions
    assert not {"avg", "sum", "gen_random_uuid"} & names.functions
    assert {"vector", "halfvec"} <= names.types
    assert "halfvec_cosine_ops" in names.opclasses
    assert {"<=>", "<%", "%>"} <= names.extension_operators
    assert "%" in names.shared_operators
    assert "=" not in names.shared_operators


async def test_no_function_body_names_a_pgkg_or_extension_object_without_its_schema(
    caller: asyncpg.Connection,
) -> None:
    names = await Names.from_catalog(
        caller, schema=SCHEMA, extension_schemas=[EXTENSION_SCHEMA]
    )
    bodies = await caller.fetch(
        """
        SELECT p.oid::pg_catalog.regprocedure::TEXT AS signature, p.prosrc
        FROM pg_catalog.pg_proc p
        JOIN pg_catalog.pg_language l ON l.oid = p.prolang
        WHERE p.pronamespace = $1::pg_catalog.regnamespace
          AND l.lanname IN ('plpgsql', 'sql')
        """,
        SCHEMA,
    )

    unqualified = {
        row["signature"]: found
        for row in bodies
        if (found := unqualified_references(row["prosrc"], names))
    }

    assert bodies, "no functions found, so nothing was checked"
    assert unqualified == {}
