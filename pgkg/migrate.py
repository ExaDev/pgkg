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

_PLACEHOLDER = re.compile(r"@(pgkg_schema|extschema:([a-z_][a-z0-9_]*))@")


class MigrationRenderError(ValueError):
    """A migration cannot be rendered for the schema it was asked for."""


def plain_identifier(name: str, what: str) -> str:
    """`name`, if it can be written into SQL unquoted; otherwise refuse."""
    if not _PLAIN_IDENTIFIER.fullmatch(name):
        raise MigrationRenderError(
            f"{what} {name!r} is not a plain lower-case identifier; pgkg "
            "substitutes it unquoted into identifiers and string literals"
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


async def extension_schemas(conn: _Connection, *, default: str) -> dict[str, str]:
    """Where each extension lives, or will: `default` for one not yet created.

    Read from the catalog rather than configured, because `CREATE EXTENSION IF
    NOT EXISTS` is a no-op for an extension the operator already installed
    elsewhere, and a body qualified with the configured schema would then name
    a schema that does not hold it.
    """
    installed = {
        row["extname"]: row["nspname"]
        for row in await conn.fetch(
            """
            SELECT e.extname, n.nspname
            FROM pg_catalog.pg_extension e
            JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
            """
        )
    }
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


async def _create_schema(conn: _Connection, schema: str) -> bool:
    """Create `schema` if it is missing; say whether this call created it."""
    exists = await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = $1)",
        schema,
    )
    if exists:
        return False
    await conn.execute(f"CREATE SCHEMA {schema}")
    return True


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
    have yet; one it already has stays where it is and is used from there.
    Each file runs in its own transaction with the search_path set for it, and
    is recorded by filename in `<schema>.pgkg_schema_migrations`.  Returns the
    filenames this call applied.
    """
    plain_identifier(schema, "schema")
    plain_identifier(extension_schema, "extension schema")

    await _create_schema(conn, schema)
    if await _create_schema(conn, extension_schema):
        # Created here, so it is pgkg's to open up: the extensions' types and
        # operators are used by every role that reads pgkg, as they would be
        # from public.  A schema the operator made is left as they made it.
        await conn.execute(f"GRANT USAGE ON SCHEMA {extension_schema} TO PUBLIC")
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
