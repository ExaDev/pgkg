from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from pgvector.asyncpg import register_vector

from pgkg.config import KEYWORD_ARM_GUC, KeywordArm, get_settings
from pgkg.migrate import extension_schemas, plain_identifier, search_path_for


_ITERATIVE_SCAN = "strict_order"


def _registering_vector_in(
    schema: str,
) -> Callable[[asyncpg.Connection], Awaitable[None]]:
    async def init(conn: asyncpg.Connection) -> None:
        await register_vector(conn, schema=schema)

    return init


async def make_pool(
    dsn: str | None = None,
    *,
    schema: str | None = None,
    keyword_arm: KeywordArm | None = None,
    min_size: int = 1,
    max_size: int = 10,
) -> asyncpg.Pool:
    """A pool whose connections find pgkg in `schema` (default: settings).

    The application's own queries name pgkg's objects without a schema, so the
    pool puts pgkg's schema first on the path and the extensions' schemas after
    it.  Those are read from the catalog once, here, rather than configured:
    they are wherever the database has them, and the vector codec has to be
    registered against the schema that actually holds the type.
    """
    settings = get_settings()
    schema = plain_identifier(schema or settings.db_schema, "schema")
    if dsn is None:
        from pgkg.embedded import get_dsn
        dsn = get_dsn()
    probe = await asyncpg.connect(dsn)
    try:
        extensions = await extension_schemas(probe, default=settings.extension_schema)
    finally:
        await probe.close()
    # HNSW does not know about the WHERE clause: it walks the graph for the
    # nearest ef_search neighbours globally and the executor then discards the
    # rows failing the scope filter, so a tenant holding a small share of the
    # index silently under-returns rather than erroring.  Measured on eight
    # orgs' worth of rows, a scoped top-k came back at a fraction of k.
    # Partitioning is the other half of the mitigation and is deferred, so
    # until it lands this setting is all of it (ADR 0001, D3).
    #
    # A startup option rather than a SET from the init callback: asyncpg issues
    # RESET ALL when a connection returns to the pool, which undoes a plain SET
    # after exactly one acquire and leaves every later caller on the
    # unmitigated scan.  RESET ALL restores startup options to what they were,
    # so this is the form that survives.  The search_path is set the same way
    # for the same reason.
    #
    # The keyword arm travels the same way and for the same reason: 059's
    # dispatcher reads it on every call, and a SET would select the owner arm
    # for one acquire and the policy path for every one after.  Sent only when
    # it selects the owner arm: unset already means the policy path, and a
    # pooler such as PgBouncer refuses a startup parameter it does not know, so
    # a deployment that never opted in must not be made to send one.
    arm = keyword_arm or settings.keyword_arm
    server_settings = {
        "hnsw.iterative_scan": _ITERATIVE_SCAN,
        "search_path": search_path_for(schema, extensions),
    } | ({KEYWORD_ARM_GUC: arm} if arm == "owner" else {})
    pool = await asyncpg.create_pool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        init=_registering_vector_in(extensions["vector"]),
        server_settings=server_settings,
    )
    return pool  # type: ignore[return-value]


async def connect(
    dsn: str | None = None, *, schema: str | None = None
) -> asyncpg.Connection:
    """One connection with the pool's search_path, for a command that needs
    pgkg's objects but not a pool (`pgkg check`)."""
    settings = get_settings()
    schema = plain_identifier(schema or settings.db_schema, "schema")
    if dsn is None:
        from pgkg.embedded import get_dsn
        dsn = get_dsn()
    conn = await asyncpg.connect(dsn)
    extensions = await extension_schemas(conn, default=settings.extension_schema)
    await conn.execute(f"SET search_path = {search_path_for(schema, extensions)}")
    return conn


async def close_pool(pool: asyncpg.Pool) -> None:
    await pool.close()


@asynccontextmanager
async def pool_from_settings() -> AsyncGenerator[asyncpg.Pool, None]:
    settings = get_settings()
    pool = await make_pool(settings.database_url)
    try:
        yield pool
    finally:
        await close_pool(pool)
