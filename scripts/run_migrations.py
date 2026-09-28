#!/usr/bin/env python3
"""Apply the migrations not yet applied against DATABASE_URL.

The same runner as `pgkg migrate`, into PGKG_DB_SCHEMA (default public).
"""
from __future__ import annotations

import asyncio
import os
import sys

import asyncpg

from pgkg.config import get_settings
from pgkg.migrate import install


async def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    settings = get_settings()
    conn = await asyncpg.connect(dsn)
    try:
        await install(
            conn,
            schema=settings.db_schema,
            extension_schema=settings.extension_schema,
            on_progress=print,
        )
        print("All migrations applied successfully.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
