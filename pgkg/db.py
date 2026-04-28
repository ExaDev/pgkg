from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from pgvector.asyncpg import register_vector

from pgkg.config import get_settings


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def make_pool(dsn: str) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10, init=_init_connection)
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
