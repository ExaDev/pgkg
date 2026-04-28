"""Embedded Postgres via pgserver — Docker-free local development.

Manages a child Postgres process with pgvector bundled. The data
directory persists between runs (default: ``~/.local/share/pgkg/pgdata``)
so ingested data survives restarts.

Usage::

    from pgkg.embedded import get_server

    server = get_server()  # starts Postgres if not already running
    dsn = server.get_uri(database="pgkg")

Requires: ``uv sync --extra embedded`` (installs pgserver).
"""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pgserver as _pgserver


_DEFAULT_PGDATA = pathlib.Path.home() / ".local" / "share" / "pgkg" / "pgdata"


def get_server(
    *,
    pgdata: str | pathlib.Path | None = None,
    cleanup_mode: str = "stop",
) -> "_pgserver.PostgresServer":
    """Start (or reuse) an embedded Postgres instance.

    Parameters
    ----------
    pgdata:
        Data directory.  Defaults to ``~/.local/share/pgkg/pgdata``.
        Set to a temp dir for ephemeral use.
    cleanup_mode:
        ``"stop"`` (default) stops Postgres when the last handle closes.
        ``"delete"`` also removes ``pgdata`` — good for test fixtures.
        ``None`` leaves the server running after Python exits.
    """
    try:
        import pgserver
    except ImportError as exc:
        raise ImportError(
            "pgserver is not installed.  Install with: uv sync --extra embedded"
        ) from exc

    data_dir = pathlib.Path(pgdata) if pgdata is not None else _DEFAULT_PGDATA
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    server = pgserver.get_server(pgdata=str(data_dir), cleanup_mode=cleanup_mode)
    return server


def _ensure_database(server: "_pgserver.PostgresServer", database: str) -> None:
    """Create the target database if it does not already exist."""
    result = server.psql(
        f"SELECT 1 FROM pg_database WHERE datname = '{database}'",
    )
    # psql returns a row containing "1" when the database exists,
    # or just the header + "(0 rows)" when it does not.
    if "(0 rows)" in result:
        server.psql(f'CREATE DATABASE "{database}"')


def get_dsn(
    *,
    pgdata: str | pathlib.Path | None = None,
    database: str = "pgkg",
    cleanup_mode: str = "stop",
) -> str:
    """Start embedded Postgres and return an asyncpg-compatible DSN.

    Creates the target database if it does not exist.
    """
    server = get_server(pgdata=pgdata, cleanup_mode=cleanup_mode)
    _ensure_database(server, database)
    return server.get_uri(database=database)
