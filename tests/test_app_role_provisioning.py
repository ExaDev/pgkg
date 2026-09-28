"""The migrations either leave a pgkg_app a deployment can use, or they stop.

Every RLS policy is inert for the table owner, so the role the policies are
written for is part of the security decision.  020 used to create it
best-effort and degrade to a NOTICE when the migrating role lacked CREATEROLE,
which left a schema whose policies protected nothing for the caller that was
meant to use them.  These tests migrate as the kinds of role a managed Postgres
actually hands out — no CREATEROLE, CREATEROLE without superuser, and a role an
administrator provisioned out of band — none of which the rest of the suite,
migrating as superuser, can see.

Each test gets a fresh database and migrator, and starts without a pgkg_app,
on a cluster shared only within this module, because roles are cluster-wide.
"""
from __future__ import annotations

import pathlib
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from pgkg.cli import AppRoleUnavailable, run_migrate
from pgkg.config import get_settings
from pgkg.migrate import apply_migrations as _apply_migrations

# Where `pgkg migrate` installs, which is where these tests look: the suite
# runs once in public and once vendored into another schema (issue #30).
SCHEMA = get_settings().db_schema
LEDGER = f"{SCHEMA}.pgkg_schema_migrations"

MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent / "migrations"

# pgvector is not a trusted extension, so on a managed Postgres an
# administrator enables it; the migrations' own CREATE EXTENSION IF NOT EXISTS
# then finds it there.
ADMIN_EXTENSIONS = ("vector", "pg_trgm", "pgcrypto")


def dsn_for(dsn: str, *, user: str, password: str, database: str) -> str:
    parts = urlsplit(dsn)
    host = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit(
        (parts.scheme, f"{user}:{password}@{host}", f"/{database}",
         parts.query, parts.fragment)
    )


@dataclass(frozen=True)
class Deployment:
    admin_dsn: str
    migrator: str
    migrator_dsn: str
    database: str


async def _admin(dsn: str, database: str | None = None) -> asyncpg.Connection:
    if database is None:
        return await asyncpg.connect(dsn)
    parts = urlsplit(dsn)
    return await asyncpg.connect(
        urlunsplit((parts.scheme, parts.netloc, f"/{database}",
                    parts.query, parts.fragment))
    )


async def provision(
    admin_dsn: str,
    *,
    createrole: bool,
    pre_provisioned: bool = False,
    app_role_attrs: str = "",
) -> Deployment:
    suffix = uuid.uuid4().hex[:8]
    migrator = f"migrator_{suffix}"
    database = f"approle_{suffix}"
    password = f"pw_{suffix}"
    admin = await _admin(admin_dsn)
    try:
        attrs = "CREATEROLE" if createrole else "NOCREATEROLE"
        await admin.execute(
            f"CREATE ROLE {migrator} LOGIN NOSUPERUSER {attrs} PASSWORD '{password}'"
        )
        await admin.execute(f"CREATE DATABASE {database} OWNER {migrator}")
        if pre_provisioned:
            await admin.execute(f"CREATE ROLE pgkg_app NOLOGIN {app_role_attrs}")
    finally:
        await admin.close()

    in_db = await _admin(admin_dsn, database)
    try:
        for ext in ADMIN_EXTENSIONS:
            await in_db.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext}"')
    finally:
        await in_db.close()

    return Deployment(
        admin_dsn=admin_dsn,
        migrator=migrator,
        migrator_dsn=dsn_for(
            admin_dsn, user=migrator, password=password, database=database
        ),
        database=database,
    )


async def teardown(deployment: Deployment) -> None:
    admin = await _admin(deployment.admin_dsn)
    try:
        await admin.execute(f"DROP DATABASE IF EXISTS {deployment.database} WITH (FORCE)")
        await admin.execute("DROP ROLE IF EXISTS pgkg_app")
        await admin.execute(f"DROP ROLE IF EXISTS {deployment.migrator}")
    finally:
        await admin.close()


@pytest.fixture
async def deployments(
    fresh_cluster_dsn: str,
) -> AsyncGenerator[list[Deployment], None]:
    made: list[Deployment] = []
    yield made
    failures: list[BaseException] = []
    for deployment in made:
        try:
            await teardown(deployment)
        except Exception as exc:
            failures.append(exc)
    if failures:
        raise ExceptionGroup("teardown failed", failures)


async def deploy(
    deployments: list[Deployment], dsn: str, **kwargs: bool | str,
) -> Deployment:
    deployment = await provision(dsn, **kwargs)
    deployments.append(deployment)
    return deployment


async def role_exists(deployment: Deployment) -> bool:
    admin = await _admin(deployment.admin_dsn)
    try:
        return bool(
            await admin.fetchval("SELECT 1 FROM pg_roles WHERE rolname = 'pgkg_app'")
        )
    finally:
        await admin.close()


async def can_become_the_app_role(dsn: str) -> str:
    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            await conn.execute("SET LOCAL ROLE pgkg_app")
            return await conn.fetchval("SELECT current_user")
    finally:
        await conn.close()


async def tables_unreachable_by_the_app_role(deployment: Deployment) -> list[str]:
    conn = await asyncpg.connect(deployment.migrator_dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT c.relname
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = $1
              AND c.relkind = 'r'
              AND c.relname <> 'pgkg_schema_migrations'
              AND NOT has_table_privilege('pgkg_app', c.oid, 'SELECT')
            ORDER BY c.relname
            """,
            SCHEMA,
        )
    finally:
        await conn.close()
    return [r["relname"] for r in rows]


async def applied(deployment: Deployment) -> list[str]:
    conn = await asyncpg.connect(deployment.migrator_dsn)
    try:
        if await conn.fetchval("SELECT to_regclass($1)", LEDGER) is None:
            return []
        rows = await conn.fetch(
            f"SELECT filename FROM {LEDGER} ORDER BY filename"
        )
    finally:
        await conn.close()
    return [r["filename"] for r in rows]


async def forget_applied(deployment: Deployment, filename: str) -> None:
    conn = await asyncpg.connect(deployment.migrator_dsn)
    try:
        await conn.execute(
            f"DELETE FROM {LEDGER} WHERE filename = $1", filename
        )
    finally:
        await conn.close()


async def apply_migrations(conn: asyncpg.Connection) -> None:
    """The shared runner without `pgkg migrate`'s preflight, so what refuses
    is the SQL itself."""
    await _apply_migrations(
        conn, schema=SCHEMA, extension_schema=get_settings().extension_schema
    )


def last_migration() -> str:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))[-1].name


def repair_migration() -> str:
    return next(MIGRATIONS_DIR.glob("058_*.sql")).name


async def test_020_raises_rather_than_leaving_no_role(
    fresh_cluster_dsn: str, deployments: list[Deployment],
) -> None:
    """The SQL itself fails loudly, whatever runner applies it."""
    deployment = await deploy(deployments, fresh_cluster_dsn, createrole=False)

    conn = await asyncpg.connect(deployment.migrator_dsn)
    try:
        with pytest.raises(asyncpg.PostgresError) as excinfo:
            await apply_migrations(conn)
    finally:
        await conn.close()

    assert "pgkg_app" in str(excinfo.value)
    assert "CREATE ROLE pgkg_app NOLOGIN" in (excinfo.value.hint or "")
    assert "020_tenancy.sql" not in await applied(deployment)
    assert not await role_exists(deployment)


async def test_the_preflight_refuses_before_applying_anything(
    fresh_cluster_dsn: str, deployments: list[Deployment],
) -> None:
    deployment = await deploy(deployments, fresh_cluster_dsn, createrole=False)

    with pytest.raises(AppRoleUnavailable) as excinfo:
        await run_migrate(deployment.migrator_dsn)

    assert "CREATE ROLE pgkg_app NOLOGIN" in str(excinfo.value)
    assert deployment.migrator in str(excinfo.value)
    assert await applied(deployment) == []


async def test_a_createrole_migrator_can_become_the_role_it_created(
    fresh_cluster_dsn: str, deployments: list[Deployment],
) -> None:
    """PG16 gives the creator ADMIN only — no SET, no INHERIT — until it grants
    the role to itself."""
    deployment = await deploy(deployments, fresh_cluster_dsn, createrole=True)

    await run_migrate(deployment.migrator_dsn)

    assert await can_become_the_app_role(deployment.migrator_dsn) == "pgkg_app"
    assert await tables_unreachable_by_the_app_role(deployment) == []


async def test_an_externally_provisioned_role_is_granted_to_not_created(
    fresh_cluster_dsn: str, deployments: list[Deployment],
) -> None:
    deployment = await deploy(
        deployments, fresh_cluster_dsn, createrole=False, pre_provisioned=True,
    )

    await run_migrate(deployment.migrator_dsn)

    assert last_migration() in await applied(deployment)
    assert await tables_unreachable_by_the_app_role(deployment) == []


async def test_058_lets_an_existing_pg16_install_become_the_role(
    fresh_cluster_dsn: str, deployments: list[Deployment],
) -> None:
    """An install that ran the old 020 as a CREATEROLE non-superuser on PG16
    holds ADMIN on pgkg_app and nothing else."""
    deployment = await deploy(deployments, fresh_cluster_dsn, createrole=True)
    await run_migrate(deployment.migrator_dsn)

    admin = await _admin(deployment.admin_dsn)
    try:
        await admin.execute(
            f"REVOKE pgkg_app FROM {deployment.migrator} "
            f"GRANTED BY {deployment.migrator}"
        )
    finally:
        await admin.close()
    await forget_applied(deployment, repair_migration())
    with pytest.raises(asyncpg.PostgresError):
        await can_become_the_app_role(deployment.migrator_dsn)

    await run_migrate(deployment.migrator_dsn)

    assert await can_become_the_app_role(deployment.migrator_dsn) == "pgkg_app"


async def test_058_provisions_and_grants_for_an_install_020_left_without_a_role(
    fresh_cluster_dsn: str, deployments: list[Deployment],
) -> None:
    """The old 020 noticed and carried on, so did every later GRANT.  Once the
    role can be created, 058 creates it and restores every grant."""
    deployment = await deploy(
        deployments, fresh_cluster_dsn, createrole=False, pre_provisioned=True,
    )
    await run_migrate(deployment.migrator_dsn)

    admin = await _admin(deployment.admin_dsn, deployment.database)
    try:
        await admin.execute("DROP OWNED BY pgkg_app")
        await admin.execute("DROP ROLE pgkg_app")
        await admin.execute(f"ALTER ROLE {deployment.migrator} CREATEROLE")
    finally:
        await admin.close()
    await forget_applied(deployment, repair_migration())

    await run_migrate(deployment.migrator_dsn)

    assert await role_exists(deployment)
    assert await can_become_the_app_role(deployment.migrator_dsn) == "pgkg_app"
    assert await tables_unreachable_by_the_app_role(deployment) == []


async def test_the_preflight_refuses_an_install_left_without_a_role(
    fresh_cluster_dsn: str, deployments: list[Deployment],
) -> None:
    deployment = await deploy(
        deployments, fresh_cluster_dsn, createrole=False, pre_provisioned=True,
    )
    await run_migrate(deployment.migrator_dsn)
    admin = await _admin(deployment.admin_dsn, deployment.database)
    try:
        await admin.execute("DROP OWNED BY pgkg_app")
        await admin.execute("DROP ROLE pgkg_app")
    finally:
        await admin.close()
    await forget_applied(deployment, repair_migration())

    with pytest.raises(AppRoleUnavailable):
        await run_migrate(deployment.migrator_dsn)

    assert repair_migration() not in await applied(deployment)


async def app_role_privileges_on_the_ledger(deployment: Deployment) -> list[str]:
    conn = await asyncpg.connect(deployment.migrator_dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT p.privilege
            FROM unnest(ARRAY['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE'])
                 AS p(privilege)
            WHERE has_table_privilege('pgkg_app', $1, p.privilege)
            ORDER BY p.privilege
            """,
            LEDGER,
        )
    finally:
        await conn.close()
    return [r["privilege"] for r in rows]


async def test_the_app_role_cannot_touch_the_migration_ledger(
    fresh_cluster_dsn: str, deployments: list[Deployment],
) -> None:
    """The ledger sits in public before 020 runs, so ON ALL TABLES reached it.
    It carries no policy: a session as pgkg_app could delete a row and have
    the next run re-apply that migration."""
    deployment = await deploy(deployments, fresh_cluster_dsn, createrole=True)

    await run_migrate(deployment.migrator_dsn)

    assert await app_role_privileges_on_the_ledger(deployment) == []


async def test_058_takes_the_ledger_back_from_the_app_role(
    fresh_cluster_dsn: str, deployments: list[Deployment],
) -> None:
    deployment = await deploy(deployments, fresh_cluster_dsn, createrole=True)
    await run_migrate(deployment.migrator_dsn)
    conn = await asyncpg.connect(deployment.migrator_dsn)
    try:
        await conn.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {LEDGER} TO pgkg_app"
        )
    finally:
        await conn.close()
    await forget_applied(deployment, repair_migration())

    await run_migrate(deployment.migrator_dsn)

    assert await app_role_privileges_on_the_ledger(deployment) == []


EXEMPT_ROLE_ATTRS = pytest.mark.parametrize("attrs", ["BYPASSRLS", "SUPERUSER"])


@EXEMPT_ROLE_ATTRS
async def test_the_preflight_refuses_an_app_role_exempt_from_its_policies(
    fresh_cluster_dsn: str, deployments: list[Deployment], attrs: str,
) -> None:
    """A pgkg_app that bypasses row security is the failure that looks
    exactly like security: every grant lands and no policy applies."""
    deployment = await deploy(
        deployments, fresh_cluster_dsn,
        createrole=False, pre_provisioned=True, app_role_attrs=attrs,
    )

    with pytest.raises(AppRoleUnavailable) as excinfo:
        await run_migrate(deployment.migrator_dsn)

    assert f"NO{attrs}" in str(excinfo.value)
    assert await applied(deployment) == []


@EXEMPT_ROLE_ATTRS
async def test_020_refuses_an_app_role_exempt_from_its_policies(
    fresh_cluster_dsn: str, deployments: list[Deployment], attrs: str,
) -> None:
    deployment = await deploy(
        deployments, fresh_cluster_dsn,
        createrole=False, pre_provisioned=True, app_role_attrs=attrs,
    )

    conn = await asyncpg.connect(deployment.migrator_dsn)
    try:
        with pytest.raises(asyncpg.PostgresError) as excinfo:
            await apply_migrations(conn)
    finally:
        await conn.close()

    assert f"NO{attrs}" in (excinfo.value.hint or "")
    assert "020_tenancy.sql" not in await applied(deployment)


@EXEMPT_ROLE_ATTRS
async def test_058_refuses_an_app_role_made_exempt_after_020(
    fresh_cluster_dsn: str, deployments: list[Deployment], attrs: str,
) -> None:
    deployment = await deploy(
        deployments, fresh_cluster_dsn, createrole=False, pre_provisioned=True,
    )
    await run_migrate(deployment.migrator_dsn)
    admin = await _admin(deployment.admin_dsn)
    try:
        await admin.execute(f"ALTER ROLE pgkg_app {attrs}")
    finally:
        await admin.close()
    await forget_applied(deployment, repair_migration())

    conn = await asyncpg.connect(deployment.migrator_dsn)
    try:
        with pytest.raises(asyncpg.PostgresError) as excinfo:
            await apply_migrations(conn)
    finally:
        await conn.close()

    assert f"NO{attrs}" in (excinfo.value.hint or "")
    assert repair_migration() not in await applied(deployment)
