"""Pytest fixtures: spin up Postgres for integration tests.

Uses pgserver (embedded, no Docker) if available, otherwise falls back
to testcontainers (Docker).
"""
from __future__ import annotations

import pathlib
import tempfile
from typing import AsyncGenerator

import asyncpg
import pytest
from pgvector.asyncpg import register_vector

MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent / "migrations"


def _pgserver_available() -> bool:
    try:
        import pgserver  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def pg_dsn():
    """Return a Postgres DSN — pgserver if available, testcontainers otherwise."""
    if _pgserver_available():
        from pgkg.embedded import get_dsn

        tmpdir = tempfile.mkdtemp(prefix="pgkg_test_")
        dsn = get_dsn(pgdata=tmpdir, database="pgkg_test", cleanup_mode="delete")
        yield dsn
    else:
        from testcontainers.postgres import PostgresContainer

        with PostgresContainer(
            image="pgvector/pgvector:pg16", driver="asyncpg",
        ) as container:
            dsn = container.get_connection_url()
            dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
            yield dsn


@pytest.fixture(scope="session")
async def pool(pg_dsn) -> AsyncGenerator[asyncpg.Pool, None]:
    """Asyncpg pool pointing at the test database with all migrations applied."""
    # Apply migrations first — the vector extension must exist before
    # register_vector can be called in the pool's init callback.
    migrate_conn = await asyncpg.connect(pg_dsn)
    try:
        for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
            await migrate_conn.execute(migration.read_text())
    finally:
        await migrate_conn.close()

    # Now create the pool with pgvector codec registration.
    conn_pool = await asyncpg.create_pool(
        pg_dsn, min_size=1, max_size=5,
        init=lambda conn: register_vector(conn),
    )

    yield conn_pool
    await conn_pool.close()
