"""Apply the SQL migrations, into whichever schema pgkg is installed in.

A migration file cannot know that schema when it is written: a host
application may vendor pgkg into a schema of its own, and the extensions pgkg
depends on live wherever the database's operator put them.  So a file names
pgkg's own objects as `@pgkg_schema@.x` and an extension's as
`@extschema:vector@.x`, and `render_migration` substitutes both before the text
reaches the server.  The spelling is PostgreSQL's own for extension scripts,
and nothing else in SQL, psql or dollar quoting reads `@word@` as anything.

Why the bodies say it at all, rather than relying on search_path: a function,
trigger or policy body is resolved against the CALLER's search_path, and a
caller that does not have pgkg's schema on it gets "does not exist" from inside
an RLS policy.  `SET search_path` on the function would fix the lookup and cost
the inlining the retrieval plans depend on (#30), so the names are qualified.

The statements outside bodies — CREATE TABLE, CREATE INDEX, CREATE POLICY — are
not qualified and need not be: they are resolved once, when the migration runs,
under the search_path the runner sets for it, and a policy or default is stored
as a parsed tree that no later caller's path can change.
"""
from __future__ import annotations

import pathlib
import re
from collections.abc import Callable, Mapping
from typing import Protocol

import asyncpg
from asyncpg.transaction import Transaction

MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "migrations"

# What 001 creates.  Their schemas go on the migration session's search_path,
# because the DDL outside function bodies (a halfvec column, an opclass) is
# resolved when it runs and is not qualified.
REQUIRED_EXTENSIONS = ("vector", "pg_trgm", "pgcrypto")

# Substituted unquoted, into identifiers and into string literals alike
# ('@pgkg_schema@.chunks'::regclass), which is only sound for a name that needs
# quoting in neither: lower case, no punctuation, at most NAMEDATALEN - 1.
_PLAIN_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}")

# The words PostgreSQL reserves outright or allows only as a function or type
# name (pg_get_keywords() catcode R and T, as of 18): each is a syntax error as
# an unquoted schema name.  test_migration_rendering checks the list against
# the server it runs on.
_RESERVED_WORDS = frozenset({
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc",
    "asymmetric", "authorization", "binary", "both", "case", "cast", "check",
    "collate", "collation", "column", "concurrently", "constraint", "create",
    "cross", "current_catalog", "current_date", "current_role",
    "current_schema", "current_time", "current_timestamp", "current_user",
    "default", "deferrable", "desc", "distinct", "do", "else", "end", "except",
    "false", "fetch", "for", "foreign", "freeze", "from", "full", "grant",
    "group", "having", "ilike", "in", "initially", "inner", "intersect",
    "into", "is", "isnull", "join", "lateral", "leading", "left", "like",
    "limit", "localtime", "localtimestamp", "natural", "not", "notnull",
    "null", "offset", "on", "only", "or", "order", "outer", "overlaps",
    "placing", "primary", "references", "returning", "right", "select",
    "session_user", "similar", "some", "symmetric", "system_user", "table",
    "tablesample", "then", "to", "trailing", "true", "union", "unique", "user",
    "using", "variadic", "verbose", "when", "where", "window", "with",
})

_PLACEHOLDER = re.compile(r"@(pgkg_schema|extschema:([a-z_][a-z0-9_]*))@")


class MigrationRenderError(ValueError):
    """A migration cannot be rendered for the schema it was asked for."""


class UntrackedInstallError(RuntimeError):
    """pgkg's objects are in the schema but its migration record is not."""


def plain_identifier(name: str, what: str) -> str:
    """`name`, if it can be written into SQL unquoted; otherwise refuse."""
    if not _PLAIN_IDENTIFIER.fullmatch(name):
        raise MigrationRenderError(
            f"{what} {name!r} is not a plain lower-case identifier; pgkg "
            "substitutes it unquoted into identifiers and string literals"
        )
    if name in _RESERVED_WORDS:
        raise MigrationRenderError(
            f"{what} {name!r} is a reserved word in PostgreSQL, and pgkg writes "
            "schema names unquoted; choose another name"
        )
    if name.startswith("pg_") and name != "pg_catalog":
        raise MigrationRenderError(
            f"{what} {name!r} starts with pg_, which PostgreSQL reserves for "
            "system schemas; choose another name"
        )
    return name


def render_migration(
    sql: str, *, schema: str, extension_schemas: Mapping[str, str]
) -> str:
    """The migration text with every placeholder replaced by its schema."""
    plain_identifier(schema, "schema")

    def substitute(match: re.Match[str]) -> str:
        extension = match.group(2)
        if extension is None:
            return schema
        if extension not in extension_schemas:
            raise MigrationRenderError(
                f"migration names extension {extension!r}, which is neither "
                "installed nor known to the runner"
            )
        return plain_identifier(
            extension_schemas[extension], f"extension schema of {extension}"
        )

    return _PLACEHOLDER.sub(substitute, sql)


class _Connection(Protocol):
    """The part of an asyncpg connection the runner needs."""

    async def execute(self, query: str, *args: object) -> str: ...

    async def fetch(self, query: str, *args: object) -> list[asyncpg.Record]: ...

    async def fetchval(self, query: str, *args: object) -> object: ...

    async def fetchrow(self, query: str, *args: object) -> asyncpg.Record | None: ...

    def transaction(self) -> Transaction: ...


async def _installed_extension_schemas(conn: _Connection) -> dict[str, str]:
    return {
        row["extname"]: row["nspname"]
        for row in await conn.fetch(
            """
            SELECT e.extname, n.nspname
            FROM pg_catalog.pg_extension e
            JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
            """
        )
    }


async def extension_schemas(conn: _Connection, *, default: str) -> dict[str, str]:
    """Where each extension lives, or will: `default` for one not yet created.

    Read from the catalog rather than configured, because `CREATE EXTENSION IF
    NOT EXISTS` is a no-op for an extension the operator already installed
    elsewhere, and a body qualified with the configured schema would then name
    a schema that does not hold it.
    """
    installed = await _installed_extension_schemas(conn)
    return {**{name: default for name in REQUIRED_EXTENSIONS}, **installed}


async def render_for(conn: _Connection, sql: str, *, schema: str) -> str:
    """`sql` rendered as the runner would render it now, for this database."""
    extensions = await extension_schemas(conn, default=schema)
    return render_migration(sql, schema=schema, extension_schemas=extensions)


def search_path_for(schema: str, extensions: Mapping[str, str]) -> str:
    """pgkg's schema first, so unqualified DDL and queries find it, then the
    schemas of the extensions it depends on, so their types and operators
    resolve too."""
    ordered = dict.fromkeys(
        [schema, *(extensions[name] for name in REQUIRED_EXTENSIONS if name in extensions)]
    )
    return ", ".join(
        plain_identifier(name, "schema") for name in ordered if name != "pg_catalog"
    )


# Tables every pgkg install has had since the migration that created them
# (002, 020): finding one in a schema with no migration record means the
# record was lost or the schema was installed some other way.
_INSTALL_MARKERS = ("propositions", "entities", "orgs")

# The role 020 provisions for the application to connect as.
_APPLICATION_ROLE = "pgkg_app"


async def _refuse_an_untracked_install(conn: _Connection, schema: str) -> None:
    tracked = await conn.fetchval(
        "SELECT to_regclass($1) IS NOT NULL", f"{schema}.pgkg_schema_migrations"
    )
    if tracked:
        return
    found = [
        row["relname"]
        for row in await conn.fetch(
            """
            SELECT c.relname FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = $1 AND c.relname = ANY($2::TEXT[])
            ORDER BY c.relname
            """,
            schema,
            list(_INSTALL_MARKERS),
        )
    ]
    if found:
        raise UntrackedInstallError(
            f"schema {schema} already holds pgkg tables ({', '.join(found)}) but "
            f"no {schema}.pgkg_schema_migrations, so the runner cannot tell which "
            "migrations it has had. Re-running from 001 would fail part-way or "
            "adopt tables it did not create. Instead, baseline it: create "
            f"{schema}.pgkg_schema_migrations (filename TEXT PRIMARY KEY, "
            "applied_at TIMESTAMPTZ NOT NULL DEFAULT now()) and insert the "
            "filename of every migration already applied, or install into an "
            "empty schema."
        )


async def _grant_extension_usage(
    conn: _Connection, *, schema: str, on_progress: Callable[[str], None]
) -> None:
    """Give the application role USAGE on every schema an extension lives in.

    pgkg_app reaches the extensions' types, operators and functions through
    those schemas.  `public` grants USAGE to everyone, but a schema an operator
    creates grants nothing, so without this pgkg_app is refused at the first
    halfvec.  Granted to pgkg_app alone, and only where it is missing; a
    refusal is reported and left, as 020 leaves its own grants.
    """
    role_exists = await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = $1)",
        _APPLICATION_ROLE,
    )
    if not role_exists:
        return
    extensions = await extension_schemas(conn, default=schema)
    for extension_schema in sorted(
        {extensions[name] for name in REQUIRED_EXTENSIONS} - {"pg_catalog"}
    ):
        has_usage = await conn.fetchval(
            "SELECT pg_catalog.has_schema_privilege($1, $2, 'USAGE')",
            _APPLICATION_ROLE,
            extension_schema,
        )
        if has_usage:
            continue
        try:
            await conn.execute(
                f"GRANT USAGE ON SCHEMA {extension_schema} TO {_APPLICATION_ROLE}"
            )
        except asyncpg.InsufficientPrivilegeError as error:
            on_progress(
                f"{_APPLICATION_ROLE} not granted USAGE on schema "
                f"{extension_schema} ({error}); grant it, or {_APPLICATION_ROLE} "
                "cannot use the extensions there"
            )


async def apply_migrations(
    conn: _Connection,
    *,
    schema: str,
    extension_schema: str,
    migrations_dir: pathlib.Path = MIGRATIONS_DIR,
    on_progress: Callable[[str], None] = lambda _message: None,
) -> tuple[str, ...]:
    """Apply every migration not yet recorded, in filename order.

    `extension_schema` is where 001 creates an extension the database does not
    have yet; one it already has stays where it is and is used from there, and
    the schema is not created when nothing would be put in it.  Each file runs
    in its own transaction with the search_path set for it, and is recorded by
    filename in `<schema>.pgkg_schema_migrations`.  Returns the filenames this
    call applied.
    """
    plain_identifier(schema, "schema")
    plain_identifier(extension_schema, "extension schema")

    await _refuse_an_untracked_install(conn, schema)
    await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    installed = await _installed_extension_schemas(conn)
    if any(name not in installed for name in REQUIRED_EXTENSIONS):
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {extension_schema}")
    await conn.execute(
        f"CREATE TABLE IF NOT EXISTS {schema}.pgkg_schema_migrations ("
        "  filename TEXT PRIMARY KEY,"
        "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")"
    )
    already = {
        row["filename"]
        for row in await conn.fetch(
            f"SELECT filename FROM {schema}.pgkg_schema_migrations"
        )
    }

    applied: list[str] = []
    for migration in sorted(migrations_dir.glob("*.sql")):
        if migration.name in already:
            on_progress(f"Skipping {migration.name} (already applied).")
            continue
        on_progress(f"Applying {migration.name}...")
        async with conn.transaction():
            extensions = await extension_schemas(conn, default=extension_schema)
            await conn.execute(
                f"SET LOCAL search_path = {search_path_for(schema, extensions)}"
            )
            await conn.execute(
                render_migration(
                    migration.read_text(),
                    schema=schema,
                    extension_schemas=extensions,
                )
            )
            await conn.execute(
                f"INSERT INTO {schema}.pgkg_schema_migrations (filename) VALUES ($1)",
                migration.name,
            )
        applied.append(migration.name)

    await _grant_extension_usage(conn, schema=schema, on_progress=on_progress)
    return tuple(applied)


class AppRoleUnavailable(RuntimeError):
    """pgkg_app is missing and the migrating role cannot create it.

    Every RLS policy is written for pgkg_app and is inert for the table owner,
    so a schema migrated without it protects nothing for the caller that was
    meant to assume it.  Raised before any migration is applied.
    """


async def check_app_role(conn: _Connection) -> None:
    """Refuse, before anything is applied, a migration pgkg_app cannot survive."""
    row = await conn.fetchrow(
        "SELECT current_user AS migrator,"
        "       app.rolsuper AS app_super,"
        "       app.rolbypassrls AS app_bypassrls,"
        "       (SELECT rolsuper OR rolcreaterole FROM pg_catalog.pg_roles"
        "         WHERE rolname = current_user) AS can_create"
        "  FROM (SELECT 1) AS one"
        "  LEFT JOIN pg_catalog.pg_roles app ON app.rolname = 'pgkg_app'"
    )
    assert row is not None
    if row["app_super"] or row["app_bypassrls"]:
        raise AppRoleUnavailable(
            "role pgkg_app is exempt from row-level security (SUPERUSER or "
            "BYPASSRLS), so every policy written for it would be inert.  Have an "
            "administrator run\n\n"
            "    ALTER ROLE pgkg_app NOSUPERUSER NOBYPASSRLS;\n\n"
            "and re-run pgkg migrate."
        )
    if row["app_super"] is not None or row["can_create"]:
        return
    raise AppRoleUnavailable(
        f"role pgkg_app does not exist and {row['migrator']} lacks CREATEROLE, "
        "so the migrations cannot provision the role every row-level security "
        "policy is written for.  Have an administrator run\n\n"
        "    CREATE ROLE pgkg_app NOLOGIN;\n\n"
        "and re-run pgkg migrate: the migrations only grant to an existing role."
    )


async def install(
    conn: _Connection,
    *,
    schema: str,
    extension_schema: str,
    migrations_dir: pathlib.Path = MIGRATIONS_DIR,
    on_progress: Callable[[str], None] = lambda _message: None,
) -> tuple[str, ...]:
    """What `pgkg migrate` runs: the app-role preflight, then the migrations.

    apply_migrations alone is the runner without the preflight, for a caller
    that wants 020's own refusal rather than this one.
    """
    await check_app_role(conn)
    return await apply_migrations(
        conn,
        schema=schema,
        extension_schema=extension_schema,
        migrations_dir=migrations_dir,
        on_progress=on_progress,
    )
