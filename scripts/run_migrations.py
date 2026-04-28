#!/usr/bin/env python3
"""Apply all migrations in numeric order against DATABASE_URL."""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys

import asyncpg

MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent / "migrations"


async def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    conn = await asyncpg.connect(dsn)
    try:
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        for migration in migration_files:
            print(f"Applying {migration.name}...")
            sql = migration.read_text()
            await conn.execute(sql)
            print(f"  OK: {migration.name}")
        print("All migrations applied successfully.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
