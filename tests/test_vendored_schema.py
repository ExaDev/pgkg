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

import argparse
import re
import uuid
from collections.abc import AsyncIterator
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from pgvector import HalfVector
from pgvector.asyncpg import register_vector

from pgkg.db import make_pool
from pgkg.migrate import MIGRATIONS_DIR, apply_migrations

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


@pytest.fixture(scope="module")
async def vendored_dsn(pg_dsn: str) -> AsyncIterator[str]:
    database = unique("pgkg_vendored")
    admin = await asyncpg.connect(pg_dsn)
    try:
        await admin.execute(f"CREATE DATABASE {database}")
    finally:
        await admin.close()

    dsn = _with_database(pg_dsn, database)
    conn = await asyncpg.connect(dsn)
    try:
        await apply_migrations(conn, schema=SCHEMA, extension_schema=EXTENSION_SCHEMA)
    finally:
        await conn.close()

    yield dsn

    admin = await asyncpg.connect(pg_dsn)
    try:
        await admin.execute(f"DROP DATABASE {database} WITH (FORCE)")
    finally:
        await admin.close()


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

    database = unique("pgkg_cli")
    admin = await asyncpg.connect(pg_dsn)
    await admin.execute(f"CREATE DATABASE {database}")
    try:
        dsn = _with_database(pg_dsn, database)
        settings = Settings(
            _env_file=None,
            database_url=dsn,
            db_schema="pgkg_cli_host",
            extension_schema="pgkg_cli_ext",
        )
        monkeypatch.setattr("pgkg.config.get_settings", lambda: settings)

        await cli.run_migrate(argparse.Namespace())

        conn = await asyncpg.connect(dsn)
        try:
            recorded = await conn.fetchval(
                "SELECT count(*) FROM pgkg_cli_host.pgkg_schema_migrations"
            )
        finally:
            await conn.close()
    finally:
        await admin.execute(f"DROP DATABASE {database} WITH (FORCE)")
        await admin.close()

    assert recorded == len(list(MIGRATIONS_DIR.glob("*.sql")))
    assert "Applying 001_extensions.sql..." in capsys.readouterr().out


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

    assert "pgkg_bm25_candidates" not in plan, (
        f"the keyword arm was not inlined:\n{plan}"
    )
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


# The names a plpgsql body could reach without a schema, and cannot be checked
# by re-creating it: plpgsql resolves names when a statement first runs, not
# when the function is created.  So the bodies are read instead.  Relations and
# functions come from the catalog; the extension objects are the ones the
# migrations use.
_EXTENSION_OPERATORS = re.compile(r"(?<![A-Za-z0-9_.(])(<=>|<#>|<\+>|<~>|<%>|<->)(?![)])")
_TRIGRAM_OPERATOR = re.compile(r"\w\s+%\s+\w")
_EXTENSION_TYPES = re.compile(r"(?<![.\w])(halfvec|vector|sparsevec)\b(?!\s*\.)")
# Columns and INSERT targets that share a relation's name.
_ALSO_COLUMNS = {"propositions", "provenance"}
_RELATION_POSITION = re.compile(
    r"\b(FROM|JOIN|INTO|UPDATE|TABLE|ONLY|REFERENCES)\s+$", re.IGNORECASE
)


def _code_of(body: str) -> str:
    """The body with comments and quoted strings blanked out, dynamic SQL kept."""
    body = re.sub(r"--[^\n]*", " ", body)
    body = re.sub(r"/\*.*?\*/", " ", body, flags=re.S)
    body = re.sub(r"\$([A-Za-z_]\w*)\$", " ", body)
    body = re.sub(r"'(?:[^']|'')*'", lambda m: _dynamic_sql_or_blank(m.group(0)), body)
    return body


def _dynamic_sql_or_blank(literal: str) -> str:
    """A string passed to EXECUTE or format() is SQL and is checked too."""
    inner = literal[1:-1].replace("''", "'")
    if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|GRANT|DROP)\b", inner):
        return " " + inner + " "
    return " '' "


async def test_no_plpgsql_body_names_a_pgkg_object_without_its_schema(
    caller: asyncpg.Connection,
) -> None:
    relations = {
        row[0]
        for row in await caller.fetch(
            """
            SELECT c.relname FROM pg_catalog.pg_class c
            WHERE c.relnamespace = $1::pg_catalog.regnamespace
              AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'c', 'f')
            """,
            SCHEMA,
        )
    }
    function_names = {
        row[0]
        for row in await caller.fetch(
            "SELECT DISTINCT proname FROM pg_catalog.pg_proc"
            " WHERE pronamespace = $1::pg_catalog.regnamespace",
            SCHEMA,
        )
    }
    bodies = await caller.fetch(
        """
        SELECT p.proname, p.prosrc
        FROM pg_catalog.pg_proc p
        JOIN pg_catalog.pg_language l ON l.oid = p.prolang
        WHERE p.pronamespace = $1::pg_catalog.regnamespace
          AND l.lanname = 'plpgsql'
        """,
        SCHEMA,
    )

    unqualified: list[str] = []
    for row in bodies:
        code = _code_of(row["prosrc"])
        for match in re.finditer(r"(?<![.\w])([a-z_][a-z0-9_]*)\b(?!\s*\.)", code):
            word = match.group(1)
            before = code[: match.start()]
            after = code[match.end():].lstrip()
            if word in function_names and after.startswith("("):
                unqualified.append(f"{row['proname']}: {word}()")
            elif word in relations and (
                word not in _ALSO_COLUMNS or _RELATION_POSITION.search(before)
            ):
                unqualified.append(f"{row['proname']}: {word}")
        for pattern in (_EXTENSION_OPERATORS, _TRIGRAM_OPERATOR):
            for match in pattern.finditer(code):
                unqualified.append(f"{row['proname']}: operator {match.group(0)!r}")
        for match in _EXTENSION_TYPES.finditer(code):
            unqualified.append(f"{row['proname']}: type {match.group(1)}")

    assert bodies, "no plpgsql functions found, so nothing was checked"
    assert unqualified == []
