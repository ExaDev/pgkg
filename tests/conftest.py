"""Pytest fixtures: spin up a testcontainers Postgres+pgvector, run migrations."""
from __future__ import annotations

import pathlib
from typing import AsyncGenerator

import asyncpg
import pytest
from testcontainers.postgres import PostgresContainer

MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent / "migrations"
PG_IMAGE = "pgvector/pgvector:pg16"


@pytest.fixture(scope="session")
def pg_container():
    """Start a pgvector Postgres container for the test session (sync)."""
    with PostgresContainer(image=PG_IMAGE, driver="asyncpg") as container:
        yield container


@pytest.fixture(scope="session")
async def pool(pg_container) -> AsyncGenerator[asyncpg.Pool, None]:
    """Asyncpg pool pointing at the test container with all migrations applied."""
    dsn = pg_container.get_connection_url()
    # testcontainers returns a SQLAlchemy-style URL; asyncpg needs postgresql://
    dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")

    conn_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)

    # Apply migrations in numeric order
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    async with conn_pool.acquire() as conn:
        for migration in migration_files:
            sql = migration.read_text()
            await conn.execute(sql)

    yield conn_pool
    await conn_pool.close()
