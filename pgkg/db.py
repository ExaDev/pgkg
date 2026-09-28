from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from pgvector.asyncpg import register_vector

from pgkg.config import KEYWORD_ARM_GUC, KeywordArm, get_settings


_ITERATIVE_SCAN = "strict_order"


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def make_pool(
    dsn: str | None = None, *, keyword_arm: KeywordArm | None = None
) -> asyncpg.Pool:
    if dsn is None:
        from pgkg.embedded import get_dsn
        dsn = get_dsn()
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
    # so this is the form that survives.
    #
    # The keyword arm travels the same way and for the same reason: 059's
    # dispatcher reads it on every call, and a SET would select the owner arm
    # for one acquire and the policy path for every one after.  Sent only when
    # it selects the owner arm: unset already means the policy path, and a
    # pooler such as PgBouncer refuses a startup parameter it does not know, so
    # a deployment that never opted in must not be made to send one.
    arm = keyword_arm or get_settings().keyword_arm
    server_settings = {"hnsw.iterative_scan": _ITERATIVE_SCAN} | (
        {KEYWORD_ARM_GUC: arm} if arm == "owner" else {}
    )
    pool = await asyncpg.create_pool(
        dsn,
        min_size=1,
        max_size=10,
        init=_init_connection,
        server_settings=server_settings,
    )
    return pool  # type: ignore[return-value]


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
