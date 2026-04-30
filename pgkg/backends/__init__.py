"""Backend factory — construct a StorageBackend from configuration."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pgkg.backend import StorageBackend


async def make_backend(backend_type: str = "postgres", **kwargs) -> "StorageBackend":
    """Construct and initialise a storage backend.

    Parameters
    ----------
    backend_type:
        One of ``"postgres"`` (default).  Future values: ``"sqlite"``.
    **kwargs:
        Backend-specific arguments.  For ``"postgres"``: ``dsn`` (str | None).
        When *dsn* is omitted or ``None``, pgserver auto-starts an embedded
        Postgres instance.
    """
    if backend_type == "postgres":
        from pgkg.backends.postgres import PostgresBackend

        dsn: str | None = kwargs.pop("dsn", None)
        return await PostgresBackend.create(dsn)

    raise ValueError(f"Unknown backend type: {backend_type!r}")
