"""Pytest fixtures: spin up Postgres for integration tests.

Uses pgserver (embedded, no Docker) if available, otherwise falls back
to testcontainers (Docker).
"""
from __future__ import annotations

import os
import pathlib
import tempfile
from contextlib import contextmanager
from typing import AsyncGenerator

import asyncpg
import pytest
from pgvector.asyncpg import register_vector

MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent / "migrations"


REQUIRED_EXTENSIONS = ("vector", "pg_trgm", "pgcrypto")


def _pgserver_available() -> bool:
    try:
        import pgserver  # noqa: F401
        return True
    except ImportError:
        return False


def _pgserver_has_required_extensions(dsn: str) -> bool:
    """pgserver bundles pgvector but not always pg_trgm/pgcrypto.

    A DSN that cannot create every extension migration 001 needs is useless
    for the integration suite, so probe before committing to it.
    """
    import asyncio

    async def probe() -> bool:
        conn = await asyncpg.connect(dsn)
        try:
            for ext in REQUIRED_EXTENSIONS:
                await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
            return True
        except Exception:
            return False
        finally:
            await conn.close()

    return asyncio.run(probe())


@contextmanager
def _embedded_dsn():
    from pgkg.embedded import get_dsn

    tmpdir = tempfile.mkdtemp(prefix="pgkg_test_")
    yield get_dsn(pgdata=tmpdir, database="pgkg_test", cleanup_mode="delete")


@contextmanager
def _docker_dsn():
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        image="pgvector/pgvector:pg16", driver="asyncpg",
    ) as container:
        yield container.get_connection_url().replace(
            "postgresql+asyncpg://", "postgresql://"
        )


@pytest.fixture(scope="session")
def pg_dsn():
    """Return a Postgres DSN for the integration suite.

    Backend selection via PGKG_TEST_BACKEND:
      auto (default) — try embedded pgserver, fall back to Docker if the
                       bundled server is missing a required extension
      embedded       — pgserver only
      docker         — testcontainers only
    """
    backend = os.environ.get("PGKG_TEST_BACKEND", "auto").lower()

    if backend not in ("auto", "embedded", "docker"):
        raise ValueError(
            f"PGKG_TEST_BACKEND must be auto, embedded or docker; got {backend!r}"
        )

    if backend == "embedded" or (backend == "auto" and _pgserver_available()):
        with _embedded_dsn() as dsn:
            if backend == "embedded" or _pgserver_has_required_extensions(dsn):
                yield dsn
                return

    with _docker_dsn() as dsn:
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
