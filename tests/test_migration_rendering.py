"""How a migration file names the schema it is installed into (issue #30).

A migration cannot know its schema when it is written: a host application may
vendor pgkg into a schema of its own, and the extensions it depends on live
wherever the database's operator put them.  So a file says `@pgkg_schema@` for
pgkg's own objects and `@extschema:vector@` for an extension's, and the runner
substitutes both before the file reaches the server — the spelling PostgreSQL
itself uses in extension scripts.

These are pure but for one: the rendering is text in, text out, and is tested
without a server, while the reserved-word list is checked against the server's.
"""
from __future__ import annotations

import asyncpg
import pytest
from pydantic import ValidationError

from pgkg.config import Settings
from pgkg.migrate import MigrationRenderError, plain_identifier, render_migration

EXTENSIONS = {"vector": "pgkg_ext", "pg_trgm": "extensions", "pgcrypto": "public"}


def test_the_schema_placeholder_becomes_the_install_schema() -> None:
    sql = "SELECT @pgkg_schema@.pgkg_current_org() FROM @pgkg_schema@.orgs"

    rendered = render_migration(sql, schema="pgkg_host", extension_schemas=EXTENSIONS)

    assert rendered == "SELECT pgkg_host.pgkg_current_org() FROM pgkg_host.orgs"


def test_an_extension_placeholder_becomes_that_extensions_schema() -> None:
    sql = (
        "SELECT a OPERATOR(@extschema:vector@.<=>) b, "
        "@extschema:pg_trgm@.similarity(x, y)"
    )

    rendered = render_migration(sql, schema="pgkg_host", extension_schemas=EXTENSIONS)

    assert rendered == (
        "SELECT a OPERATOR(pgkg_ext.<=>) b, extensions.similarity(x, y)"
    )


def test_sql_that_merely_resembles_a_placeholder_is_left_alone() -> None:
    sql = "SELECT tags @> ARRAY['a'] AND tsv @@ q AND '@pgkg' <> '@'"

    rendered = render_migration(sql, schema="pgkg_host", extension_schemas=EXTENSIONS)

    assert rendered == sql


def test_an_extension_the_database_does_not_have_is_refused_by_name() -> None:
    with pytest.raises(MigrationRenderError, match="postgis"):
        render_migration(
            "SELECT @extschema:postgis@.st_x(g)",
            schema="pgkg_host",
            extension_schemas=EXTENSIONS,
        )


@pytest.mark.parametrize(
    "schema",
    ["", "Pgkg", "pgkg-host", "pgkg host", 'x"; DROP TABLE orgs; --', "9lives", "a" * 64],
)
def test_a_schema_name_that_is_not_a_plain_identifier_is_refused(schema: str) -> None:
    """The name is substituted unquoted, into identifiers and into string
    literals alike (`'@pgkg_schema@.chunks'::regclass`), which is only sound
    for a name that needs no quoting in either."""
    with pytest.raises(MigrationRenderError, match="schema"):
        render_migration("SELECT 1", schema=schema, extension_schemas=EXTENSIONS)


@pytest.mark.parametrize("schema", ["user", "select", "table", "order", "left", "join"])
def test_a_reserved_word_is_refused_as_a_schema_name(schema: str) -> None:
    """Lower case and unpunctuated, but still not writable unquoted:
    `CREATE SCHEMA user` is a syntax error, and so is `left.orgs`."""
    with pytest.raises(MigrationRenderError, match="reserved"):
        render_migration("SELECT 1", schema=schema, extension_schemas=EXTENSIONS)


def test_the_catalog_prefix_is_refused_as_a_schema_name() -> None:
    """`CREATE SCHEMA pg_...` is refused by the server; say so before it is."""
    with pytest.raises(MigrationRenderError, match="pg_"):
        render_migration("SELECT 1", schema="pg_kg", extension_schemas=EXTENSIONS)


async def test_every_keyword_the_server_reserves_is_refused(
    pool: asyncpg.Pool,
) -> None:
    """The list is pgkg's own copy, so it is checked against the server's:
    a keyword a new major version reserves has to be added here."""
    async with pool.acquire() as conn:
        reserved = {
            row["word"]
            for row in await conn.fetch(
                "SELECT word FROM pg_get_keywords() WHERE catcode IN ('R', 'T')"
            )
        }

    accepted = sorted(word for word in reserved if _accepted_as_schema(word))

    assert accepted == []


def _accepted_as_schema(word: str) -> bool:
    try:
        plain_identifier(word, "schema")
    except MigrationRenderError:
        return False
    return True


def test_an_extension_schema_that_is_not_a_plain_identifier_is_refused() -> None:
    with pytest.raises(MigrationRenderError, match="Ext Schema"):
        render_migration(
            "SELECT @extschema:vector@.l2_normalize(v)",
            schema="pgkg_host",
            extension_schemas={"vector": "Ext Schema"},
        )


# ---------------------------------------------------------------------------
# Where the settings say pgkg lives
# ---------------------------------------------------------------------------


def test_pgkg_installs_into_public_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PGKG_DB_SCHEMA", raising=False)
    monkeypatch.delenv("PGKG_EXTENSION_SCHEMA", raising=False)

    settings = Settings(_env_file=None)

    assert (settings.db_schema, settings.extension_schema) == ("public", "public")


def test_the_schemas_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PGKG_DB_SCHEMA", "pgkg_host")
    monkeypatch.setenv("PGKG_EXTENSION_SCHEMA", "extensions")

    settings = Settings(_env_file=None)

    assert (settings.db_schema, settings.extension_schema) == ("pgkg_host", "extensions")


@pytest.mark.parametrize("field", ["db_schema", "extension_schema"])
def test_a_schema_setting_that_is_not_a_plain_identifier_is_refused(
    field: str,
) -> None:
    """Refused when the settings load, rather than when the first migration or
    the first pool is made from them."""
    with pytest.raises(ValidationError, match=field):
        Settings.model_validate({field: "Host Schema"})
