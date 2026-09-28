"""The keyword arm on a Postgres that will not let pgkg mark its operators (#32).

043, 046 and 047 mark four pg_catalog functions LEAKPROOF so that the index
conditions made of them survive a row-security policy.  That needs ownership of
a built-in, which a managed Postgres — Cloud SQL, RDS — does not hand out, so
each ALTER degrades to a NOTICE and the keyword arm scans the whole tenant under
the application role: ~80x on a 40k-row tenant, visible only at scale.

The suite migrates as a superuser and therefore only ever sees the marked
state.  The managed state is simulated here by unmarking the functions as the
superuser, inside a transaction that is rolled back — ALTER FUNCTION is
transactional, so the catalog change is visible to exactly one connection for
the length of one test and the rest of the suite never plans against it.  The
one test that has to commit (the CLI opens its own connection) restores the
mark in a `finally` and asserts it did.

Three things are pinned: the degraded state is reported rather than logged;
the opt-in owner-rights keyword arm returns exactly the rows the policy path
returns, for every scope a caller can ask for; and it reaches the GIN index in
the state where the policy path cannot.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
import pytest

ORG_GUC = "pgkg.org_id"
ARM_GUC = "pgkg.keyword_arm"
SYSTEM_ORG = uuid.UUID("00000000-0000-0000-0000-000000000000")

VQ = "ts_match_vq(tsvector,tsquery)"
QV = "ts_match_qv(tsquery,tsvector)"
SIMILARITY = "similarity_op(text,text)"
CONTAINS = "arraycontains(anyarray,anyarray)"

# What each operator's index condition serves, as 059 names it.
SERVES = {
    VQ: "keyword",
    QV: "keyword",
    SIMILARITY: "gazetteer",
    CONTAINS: "gazetteer",
}

OWNER_ARM = (
    "public.pgkg_bm25_candidates_as_owner"
    "(text, text, text, integer, uuid[], uuid[], uuid, uuid[],"
    " timestamp with time zone, text)"
)

# The tables the owner-rights arm reads, and the one read policy each carries.
# The arm bypasses row security for its owner, so it restates these policies in
# its own body; this is the list it restates and the predicate it restates.
OWNER_ARM_TABLES = ("chunks", "corpus_stats", "lexeme_df", "propositions")
ORG_READ = "((org_id = pgkg_current_org()) OR (org_id = pgkg_system_org()))"

NEEDLE = "zorblatt"


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@asynccontextmanager
async def unmarked(
    conn: asyncpg.Connection, *signatures: str
) -> AsyncIterator[None]:
    """The managed-Postgres catalog, for one transaction on one connection."""
    transaction = conn.transaction()
    await transaction.start()
    try:
        for signature in signatures:
            await conn.execute(f"ALTER FUNCTION {signature} NOT LEAKPROOF")
        yield
    finally:
        await transaction.rollback()


async def as_app(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    """Become the role the policies are written for, inside the open
    transaction: SET LOCAL ROLE outside one is a no-op under asyncpg."""
    await conn.execute("SET LOCAL ROLE pgkg_app")
    await conn.execute("SELECT set_config($1, $2, true)", ORG_GUC, str(org))
    assert await conn.fetchval("SELECT current_user") == "pgkg_app"


async def proleakproof(conn: asyncpg.Connection, signature: str) -> bool:
    return await conn.fetchval(
        "SELECT proleakproof FROM pg_proc WHERE oid = to_regprocedure($1)",
        signature,
    )


# ---------------------------------------------------------------------------
# Detection.
# ---------------------------------------------------------------------------

async def test_the_state_names_every_operator_a_policy_would_demote(
    pool: asyncpg.Pool,
) -> None:
    """046 reported the two `@@` functions and nothing else, so an operator
    watching /health could not see the gazetteer arms 047 marks.  One list,
    in the migration that makes the claim, with what each one serves."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT signature, serves, leakproof FROM pgkg_leakproof_state()"
        )

    assert {r["signature"]: r["serves"] for r in rows} == SERVES
    assert all(r["leakproof"] for r in rows), (
        f"the suite migrates as a superuser, so every mark applies: {rows}"
    )


@pytest.mark.parametrize("signature", sorted(SERVES))
async def test_an_unmarked_operator_is_a_warning_not_a_notice(
    pool: asyncpg.Pool, signature: str
) -> None:
    from pgkg.config import row_security_warnings

    async with pool.acquire() as conn:
        async with unmarked(conn, signature):
            warnings = await row_security_warnings(conn, keyword_arm="policy")
            fix = await conn.fetchval(
                "SELECT fix FROM pgkg_leakproof_state() WHERE signature = $1",
                signature,
            )
            # The statement the warning hands a superuser is the one that
            # works: executed here, it restores the mark.
            await conn.execute(fix)
            fixed = await proleakproof(conn, signature)
        assert await proleakproof(conn, signature) is True

    assert len(warnings) == 1, warnings
    (warning,) = warnings
    assert signature in warning
    assert fix in warning, (
        "the warning has to carry the statement that fixes it, for whoever "
        "holds the superuser"
    )
    assert fixed is True, f"{fix!r} did not mark {signature} leakproof"


async def test_the_fix_names_the_schema_the_function_lives_in(
    pool: asyncpg.Pool,
) -> None:
    """similarity_op is pg_trgm's and lives wherever the extension was
    installed, not in pg_catalog; a fix that guessed the schema would fail for
    exactly the operator it was printed for."""
    async with pool.acquire() as conn:
        fixes = {
            row["signature"]: row["fix"]
            for row in await conn.fetch("SELECT signature, fix FROM pgkg_leakproof_state()")
        }
        trgm_schema = await conn.fetchval(
            "SELECT n.nspname FROM pg_extension e"
            " JOIN pg_namespace n ON n.oid = e.extnamespace"
            " WHERE e.extname = 'pg_trgm'"
        )

    assert fixes[SIMILARITY] == f"ALTER FUNCTION {trgm_schema}.{SIMILARITY} LEAKPROOF"
    assert fixes[VQ] == f"ALTER FUNCTION pg_catalog.{VQ} LEAKPROOF"


async def test_a_marked_catalog_raises_no_warning(pool: asyncpg.Pool) -> None:
    from pgkg.config import row_security_warnings

    async with pool.acquire() as conn:
        assert await row_security_warnings(conn, keyword_arm="policy") == ()


async def test_the_owner_arm_silences_the_keyword_warning_only(
    pool: asyncpg.Pool,
) -> None:
    """Opting in is the remedy for the keyword operators, so a deployment that
    took it is not told to take it again — but the gazetteer is still on the
    policy path, and still says so."""
    from pgkg.config import row_security_warnings

    async with pool.acquire() as conn:
        async with unmarked(conn, VQ, QV, SIMILARITY):
            warnings = await row_security_warnings(conn, keyword_arm="owner")

    assert len(warnings) == 1, warnings
    assert "similarity_op" in warnings[0]


async def test_the_owner_arm_warns_when_its_owner_is_under_the_policy(
    pool: asyncpg.Pool,
) -> None:
    """FORCE ROW LEVEL SECURITY subjects the owner to the policies too.  The
    owner arm stays correct — the policy and its own restatement both apply —
    and stops reaching the index, which is the thing it was opted into for."""
    from pgkg.config import row_security_warnings

    async with pool.acquire() as conn:
        async with unmarked(conn, VQ, QV):
            # The superuser who owns everything here bypasses even FORCE, so
            # the owner has to become a role that does not.
            owner = unique("pgkg_owner")
            await conn.execute(f"CREATE ROLE {owner} NOLOGIN")
            await conn.execute(f"GRANT USAGE ON SCHEMA public TO {owner}")
            await conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {owner}")
            await conn.execute(f"ALTER FUNCTION {OWNER_ARM} OWNER TO {owner}")
            not_owner = await conn.fetchval("SELECT pgkg_owner_arm_bypasses_policy()")
            for table in OWNER_ARM_TABLES:
                await conn.execute(f"ALTER TABLE {table} OWNER TO {owner}")
            owning = await conn.fetchval("SELECT pgkg_owner_arm_bypasses_policy()")
            await conn.execute("ALTER TABLE chunks FORCE ROW LEVEL SECURITY")
            forced = await conn.fetchval("SELECT pgkg_owner_arm_bypasses_policy()")
            warnings = await row_security_warnings(conn, keyword_arm="owner")

    assert not_owner is False, "a non-owner is under the policy"
    assert owning is True, "the table owner is exempt from its own policies"
    assert forced is False, "FORCE puts the owner back under the policy"
    assert any("FORCE ROW LEVEL SECURITY" in w for w in warnings), warnings


async def test_pgkg_check_prints_the_warning_and_fails(
    pool: asyncpg.Pool, pg_dsn: str, monkeypatch, capsys
) -> None:
    """The CLI opens its own connection, so the unmarked state has to be
    committed for it to see.  ts_match_qv is the operator nothing in the tree
    writes, and the mark is restored whatever happens."""
    from pgkg import cli
    from pgkg.config import get_settings

    monkeypatch.setenv("PGKG_DATABASE_URL", pg_dsn)
    get_settings.cache_clear()
    async with pool.acquire() as conn:
        await conn.execute(f"ALTER FUNCTION {QV} NOT LEAKPROOF")
        try:
            with pytest.raises(SystemExit) as exited:
                await cli.run_check()
        finally:
            await conn.execute(f"ALTER FUNCTION {QV} LEAKPROOF")
            get_settings.cache_clear()
        assert await proleakproof(conn, QV) is True

    assert exited.value.code == 1
    err = capsys.readouterr().err
    assert "WARNING" in err and "ts_match_qv" in err, err


async def test_pgkg_check_is_quiet_on_a_marked_catalog(
    pg_dsn: str, pool: asyncpg.Pool, monkeypatch, capsys
) -> None:
    from pgkg import cli
    from pgkg.config import get_settings

    monkeypatch.setenv("PGKG_DATABASE_URL", pg_dsn)
    get_settings.cache_clear()
    try:
        await cli.run_check()
    finally:
        get_settings.cache_clear()

    assert "WARNING" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The owner-rights keyword arm: the same rows.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
async def tenancy(pool: asyncpg.Pool):
    """Every visibility boundary the keyword arm has to honour, each holding a
    fact and a passage that match the query, and enough filler that the index
    is the cheaper plan."""
    async with pool.acquire() as conn:
        mine = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("mp_mine")
        )
        stranger = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("mp_other")
        )

        async def collection(org: uuid.UUID) -> uuid.UUID:
            return await conn.fetchval(
                "INSERT INTO collections (org_id, owner_org_id, name, kind)"
                " VALUES ($1, $1, $2, 'corpus') RETURNING id",
                org,
                unique("mp_coll"),
            )

        home, elsewhere, theirs = (
            await collection(mine), await collection(mine), await collection(stranger)
        )
        user = await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
            mine,
            unique("mp_user"),
        )
        group = uuid.uuid4()
        namespace = unique("mp_ns")

        # (label, org, collection, visibility, owner, acl group)
        placements = [
            ("home", mine, home, "shared", None, None),
            ("elsewhere", mine, elsewhere, "shared", None, None),
            ("private", mine, home, "private", user, None),
            ("acl", mine, home, "shared", None, group),
            ("stranger", stranger, theirs, "shared", None, None),
            ("system", SYSTEM_ORG, None, "shared", None, None),
        ]
        ids: dict[tuple[str, str], uuid.UUID] = {}
        for label, org, coll, visibility, owner, acl in placements:
            await conn.execute("SELECT set_config($1, $2, false)", ORG_GUC, str(org))
            coll = coll or await conn.fetchval("SELECT pgkg_default_collection()")
            text = f"the {NEEDLE} reconciliation for {label} needs {NEEDLE} approval"
            ids[("chunks", label)] = await conn.fetchval(
                """
                INSERT INTO chunks (text, span_start, span_end, org_id,
                                    collection_id, visibility, owner_user_id,
                                    acl_group_id)
                VALUES ($1, 0, 60, $2, $3, $4, $5, $6) RETURNING id
                """,
                text, org, coll, visibility, owner, acl,
            )
            ids[("propositions", label)] = await conn.fetchval(
                """
                INSERT INTO propositions (text, namespace, org_id, collection_id,
                                          visibility, owner_user_id, acl_group_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id
                """,
                text, namespace, org, coll, visibility, owner, acl,
            )

        # A vocabulary of its own: filler sharing words with another module's
        # corpus would move that module's GIN selectivity, and its plan with it.
        await conn.execute("SELECT set_config($1, $2, false)", ORG_GUC, str(mine))
        await conn.execute(
            """
            INSERT INTO chunks (text, span_start, span_end, org_id, collection_id)
            SELECT 'the quarterly warehouse inventory audit notes that pallet '
                   || g || ' was counted twice before the stocktake closed '
                   || repeat('padding token ' || (g % 91) || ' ', 12),
                   0, 400, $1, $2
            FROM generate_series(1, 4000) g
            """,
            mine,
            home,
        )
        await conn.execute("SELECT pgkg_refresh_corpus_stats()")
        await conn.execute("SELECT pgkg_refresh_chunk_stats()")
        # VACUUM as well as ANALYZE: a freshly bulk-loaded GIN index still
        # holds its pending list, and the planner charges a bitmap scan for it.
        await conn.execute("VACUUM ANALYZE chunks")
        await conn.execute("VACUUM ANALYZE propositions")
    yield {
        "mine": mine,
        "stranger": stranger,
        "home": home,
        "elsewhere": elsewhere,
        "user": user,
        "group": group,
        "namespace": namespace,
        "ids": ids,
    }
    # Leave the tables the size they were found.  Four thousand rows are
    # enough to move another module's seq-scan-or-bitmap choice on `chunks`,
    # and test_retrieval_plan_shape asserts on exactly that choice.
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM chunks WHERE org_id = ANY($1::uuid[]) OR id = ANY($2::uuid[])",
            [mine, stranger],
            [i for (store, _), i in ids.items() if store == "chunks"],
        )
        await conn.execute(
            "DELETE FROM propositions WHERE namespace = $1", namespace
        )
        await conn.execute("VACUUM ANALYZE chunks")
        await conn.execute("VACUUM ANALYZE propositions")


KEYWORD_ARM = (
    "SELECT item_id, raw_score FROM pgkg_bm25_candidates("
    "$1, $2, NULL, 200, $3::uuid[], $4::uuid[], $5::uuid, $6::uuid[], NULL, $7)"
)

# Every scope a caller can put to the arm, by who is asking and what they ask
# for.  The org arrays are the caller's to choose and RLS is what bounds them,
# so the cases that matter most are the ones that name an org the session may
# not read.
SCOPES = {
    "unscoped": ("mine", None, None, False, False),
    "own org": ("mine", ["mine"], None, False, False),
    "own and shared": ("mine", ["mine", "system"], None, True, True),
    "one collection": ("mine", ["mine"], ["home"], True, True),
    "names a stranger": ("mine", ["stranger"], None, True, True),
    "names everyone": ("mine", ["mine", "stranger", "system"], None, True, True),
    "as the stranger": ("stranger", None, None, True, True),
}


def _scope_args(tenancy: dict, scope: str, source: str) -> tuple:
    guc, orgs, collections, with_user, with_group = SCOPES[scope]
    resolve = {"mine": tenancy["mine"], "stranger": tenancy["stranger"],
               "system": SYSTEM_ORG}
    return (
        resolve[guc],
        (
            f"{NEEDLE} reconciliation",
            tenancy["namespace"],
            [resolve[o] for o in orgs] if orgs is not None else None,
            [tenancy[c] for c in collections] if collections is not None else None,
            tenancy["user"] if with_user else None,
            [tenancy["group"]] if with_group else None,
            source,
        ),
    )


async def _rows_as_app(
    conn: asyncpg.Connection, guc: uuid.UUID, arm: str, args: tuple
) -> dict[uuid.UUID, float]:
    await conn.execute("SELECT set_config($1, $2, true)", ARM_GUC, arm)
    await conn.execute("SELECT set_config($1, $2, true)", ORG_GUC, str(guc))
    return {
        row["item_id"]: round(row["raw_score"], 4)
        for row in await conn.fetch(KEYWORD_ARM, *args)
    }


@pytest.mark.parametrize("source", ["chunks", "propositions"])
@pytest.mark.parametrize("scope", sorted(SCOPES))
async def test_the_owner_arm_returns_exactly_what_the_policy_returns(
    pool: asyncpg.Pool, tenancy, scope: str, source: str
) -> None:
    """The owner arm bypasses row security for its owner, so every row it
    returns is one the policies never looked at.  Equality with the policy path
    — row for row, score for score, because the statistics are policied too —
    is the claim that it restates the policies faithfully."""
    guc, args = _scope_args(tenancy, scope, source)
    async with pool.acquire() as conn:
        async with unmarked(conn, VQ, QV):
            await as_app(conn, guc)
            by_policy = await _rows_as_app(conn, guc, "policy", args)
            by_owner = await _rows_as_app(conn, guc, "owner", args)

    assert by_owner == by_policy
    if scope not in ("names a stranger",):
        assert by_policy, "nothing matched, so the equality above is vacuous"


@pytest.mark.parametrize("source", ["chunks", "propositions"])
async def test_the_owner_arm_keeps_every_boundary(
    pool: asyncpg.Pool, tenancy, source: str
) -> None:
    """The equality above could hold because both paths are wrong the same
    way, so each boundary is also asserted by name."""
    ids = {label: i for (s, label), i in tenancy["ids"].items() if s == source}

    async def owner_rows(scope: str) -> set[uuid.UUID]:
        guc, args = _scope_args(tenancy, scope, source)
        return set(await _rows_as_app(conn, guc, "owner", args))

    async with pool.acquire() as conn:
        async with unmarked(conn, VQ, QV):
            await as_app(conn, tenancy["mine"])
            unscoped = await owner_rows("unscoped")
            everyone = await owner_rows("names everyone")
            one_collection = await owner_rows("one collection")
            stranger_named = await owner_rows("names a stranger")
            as_stranger = await owner_rows("as the stranger")

    assert ids["stranger"] not in everyone, "another org's row, by naming it"
    assert stranger_named == set(), "an org the session may not read, by naming it"
    assert ids["home"] in unscoped and ids["system"] in unscoped
    assert ids["private"] not in unscoped, "a private row with no caller"
    assert ids["acl"] not in unscoped, "an ACL-gated row with no groups named"
    assert {ids["private"], ids["acl"], ids["system"]} <= everyone
    assert ids["elsewhere"] not in one_collection, "a collection not asked for"
    assert ids["home"] in one_collection
    assert ids["home"] not in as_stranger and ids["stranger"] in as_stranger


async def test_the_owner_arm_reads_the_org_the_policies_read(
    pool: asyncpg.Pool, tenancy
) -> None:
    """The org comes from the GUC, never from an argument, and resolves the way
    pgkg_current_org() resolves it — including the unset and blank GUC, which
    both mean the backfill org."""
    source = "chunks"
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SET LOCAL ROLE pgkg_app")
            for value in ("", str(tenancy["stranger"])):
                await conn.execute(
                    "SELECT set_config($1, $2, true)", ORG_GUC, value
                )
                _, args = _scope_args(tenancy, "names everyone", source)
                args = (args[0], args[1], None, None, None, None, source)
                await conn.execute("SELECT set_config($1, 'policy', true)", ARM_GUC)
                by_policy = {r["item_id"] for r in await conn.fetch(KEYWORD_ARM, *args)}
                await conn.execute("SELECT set_config($1, 'owner', true)", ARM_GUC)
                by_owner = {r["item_id"] for r in await conn.fetch(KEYWORD_ARM, *args)}
                assert by_owner == by_policy, f"GUC {value!r}"


# ---------------------------------------------------------------------------
# The owner-rights keyword arm: the index.
# ---------------------------------------------------------------------------

async def _nested_plans(
    dsn: str, org: uuid.UUID, arm: str, args: tuple
) -> str:
    """The plans of the statements inside the arm, which EXPLAIN cannot show:
    a SECURITY DEFINER function never inlines, so its body is one Function
    Scan to the caller.  auto_explain logs nested statements, and at NOTICE
    the log comes back on the connection that ran them.  A connection of its
    own, because LOAD outlives the transaction."""
    conn = await asyncpg.connect(dsn)
    notices: list[str] = []
    conn.add_log_listener(lambda _c, message: notices.append(str(message)))
    try:
        await conn.execute("LOAD 'auto_explain'")
        async with unmarked(conn, VQ, QV):
            for setting, value in (
                ("auto_explain.log_min_duration", "0"),
                ("auto_explain.log_nested_statements", "on"),
                ("auto_explain.log_level", "notice"),
            ):
                await conn.execute("SELECT set_config($1, $2, true)", setting, value)
            await as_app(conn, org)
            await conn.execute("SELECT set_config($1, $2, true)", ARM_GUC, arm)
            await conn.fetch(KEYWORD_ARM, *args)
    finally:
        await conn.close()
    return "\n".join(notices)


async def test_the_owner_arm_reaches_the_index_the_policy_path_cannot(
    pg_dsn: str, tenancy
) -> None:
    guc, args = _scope_args(tenancy, "own org", "chunks")
    by_policy = await _nested_plans(pg_dsn, guc, "policy", args)
    by_owner = await _nested_plans(pg_dsn, guc, "owner", args)

    assert "chunk_tsv_idx" not in by_policy, (
        "with the operators unmarked the policy path still reached the GIN "
        f"index, so this is not measuring the managed state:\n{by_policy}"
    )
    assert "chunk_tsv_idx" in by_owner, (
        f"the owner arm does not reach the GIN index either:\n{by_owner}"
    )


# ---------------------------------------------------------------------------
# The owner-rights keyword arm: what it is allowed to be.
# ---------------------------------------------------------------------------

async def test_the_owner_arm_is_hardened_as_a_security_definer(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT p.prosecdef, p.proconfig,
                   has_function_privilege('pgkg_app', p.oid, 'EXECUTE') AS app,
                   EXISTS (
                       SELECT 1 FROM aclexplode(p.proacl) a
                       WHERE a.grantee = 0 AND a.privilege_type = 'EXECUTE'
                   ) AS public
            FROM pg_proc p WHERE p.oid = $1::regprocedure
            """,
            OWNER_ARM,
        )

    assert row["prosecdef"] is True
    assert row["proconfig"] == ["search_path=pg_catalog, pg_temp"]
    assert row["app"] is True, "the application role cannot call it"
    assert row["public"] is False, "EXECUTE is still granted to PUBLIC"


async def test_the_owner_arm_depends_on_no_caller_search_path(
    pool: asyncpg.Pool, tenancy
) -> None:
    """Its path is pinned to pg_catalog, so every pgkg object in its body has
    to be named by schema — and so does every helper that inlines into it.
    Called from an empty path, anything unqualified fails to resolve."""
    _, args = _scope_args(tenancy, "own org", "chunks")
    async with pool.acquire() as conn:
        async with conn.transaction():
            await as_app(conn, tenancy["mine"])
            await conn.execute("SET LOCAL search_path = ''")
            rows = await conn.fetch(
                "SELECT item_id FROM public.pgkg_bm25_candidates_as_owner("
                "$1, $2, NULL, 200, $3::uuid[], $4::uuid[], $5::uuid,"
                " $6::uuid[], NULL, $7)",
                *args,
            )
    assert rows


async def test_the_policies_the_owner_arm_restates_are_the_policies_in_force(
    pool: asyncpg.Pool,
) -> None:
    """The owner arm is only as safe as its restatement of the read policies
    on the tables it reads.  A policy added, narrowed or made restrictive on
    one of them fails here, naming the function that has to follow it."""
    async with pool.acquire() as conn:
        policies = await conn.fetch(
            """
            SELECT tablename, permissive, cmd, qual
            FROM pg_policies
            WHERE schemaname = 'public' AND tablename = ANY($1::text[])
            ORDER BY tablename
            """,
            list(OWNER_ARM_TABLES),
        )

    assert [
        (p["tablename"], p["permissive"], p["cmd"], p["qual"]) for p in policies
    ] == [(t, "PERMISSIVE", "ALL", ORG_READ) for t in OWNER_ARM_TABLES], (
        "a read policy on a table pgkg_bm25_candidates_as_owner() reads has "
        "changed; restate it in that function's body before changing this list"
    )


# ---------------------------------------------------------------------------
# The switch.
# ---------------------------------------------------------------------------

def test_the_policy_arm_is_the_default(monkeypatch) -> None:
    from pgkg.config import Settings

    monkeypatch.delenv("PGKG_KEYWORD_ARM", raising=False)
    assert Settings().keyword_arm == "policy"


def test_the_owner_arm_is_opted_into_by_setting(monkeypatch) -> None:
    from pydantic import ValidationError

    from pgkg.config import Settings

    monkeypatch.setenv("PGKG_KEYWORD_ARM", "owner")
    assert Settings().keyword_arm == "owner"
    monkeypatch.setenv("PGKG_KEYWORD_ARM", "definer")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("arm", ["policy", "owner"])
async def test_the_pool_carries_the_arm_across_every_acquire(
    pg_dsn: str, arm: str
) -> None:
    """A startup option, for the reason hnsw.iterative_scan is one: asyncpg
    issues RESET ALL on release, and a SET would survive one acquire.

    The policy arm sends no parameter at all: it is what an unset GUC already
    means, and a pooler such as PgBouncer refuses a startup parameter it does
    not know, so a deployment that never opted in must not be made to send
    one."""
    from pgkg.db import make_pool

    expected = arm if arm == "owner" else None
    pool = await make_pool(pg_dsn, keyword_arm=arm)
    try:
        for _ in range(3):
            async with pool.acquire() as conn:
                seen = await conn.fetchval("SELECT current_setting($1, true)", ARM_GUC)
                assert seen == expected
    finally:
        await pool.close()


async def test_the_default_pool_takes_the_arm_from_settings(
    pg_dsn: str, monkeypatch
) -> None:
    from pgkg.config import get_settings
    from pgkg.db import make_pool

    monkeypatch.setenv("PGKG_KEYWORD_ARM", "owner")
    get_settings.cache_clear()
    try:
        pool = await make_pool(pg_dsn)
    finally:
        get_settings.cache_clear()
    try:
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT current_setting($1, true)", ARM_GUC) == "owner"
    finally:
        await pool.close()


@pytest.mark.parametrize(
    "setting,owner_runs",
    [(None, False), ("policy", False), ("definer", False), ("owner", True)],
)
async def test_the_keyword_arm_dispatches_on_the_setting(
    pool: asyncpg.Pool, tenancy, setting: str | None, owner_runs: bool
) -> None:
    """Every keyword caller — pgkg_retrieve(), pgkg_search() — names
    pgkg_bm25_candidates(), so that is where the switch has to be.  Only
    'owner' selects the owner arm: unset, or a value nobody meant, is the policy
    path, so nothing changes for a deployment that did not opt in."""
    _, args = _scope_args(tenancy, "own org", "chunks")
    async with pool.acquire() as conn:
        async with conn.transaction():
            await as_app(conn, tenancy["mine"])
            if setting is not None:
                await conn.execute("SELECT set_config($1, $2, true)", ARM_GUC, setting)
            plan = "\n".join(
                row[0]
                for row in await conn.fetch(
                    f"EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF) {KEYWORD_ARM}", *args
                )
            )
            rows = await conn.fetch(KEYWORD_ARM, *args)

    (owner_scan,) = [
        line for line in plan.splitlines()
        if "Function Scan on pgkg_bm25_candidates_owner_gate" in line
    ]
    assert ("never executed" not in owner_scan) is owner_runs, plan
    assert rows, "nothing matched, so either branch would look the same"
    assert "Function Scan on pgkg_bm25_candidates_under_policy" not in plan, (
        "the policy path no longer inlines into the dispatcher, so its plan is "
        f"no longer 041's:\n{plan}"
    )


# ---------------------------------------------------------------------------
# A role that is not pgkg_app.
# ---------------------------------------------------------------------------
#
# 020 says the policies bite for any non-exempt role the operator creates by
# hand, and 059 grants the owner arm to pgkg_app alone.  Postgres checks
# EXECUTE on a function in a FROM list when the plan is initialised, not when
# the node first runs.  So an owner branch gated only by a One-Time Filter
# denied every keyword call for such a role, on the default path with the GUC
# unset.

@asynccontextmanager
async def as_other_role(
    conn: asyncpg.Connection, org: uuid.UUID
) -> AsyncIterator[str]:
    """A hand-made application role: table and schema access, not pgkg_app."""
    async with conn.transaction():
        role = unique("other_app")
        await conn.execute(f"CREATE ROLE {role} NOLOGIN")
        await conn.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        await conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}")
        await conn.execute(f"SET LOCAL ROLE {role}")
        await conn.execute("SELECT set_config($1, $2, true)", ORG_GUC, str(org))
        yield role


async def test_a_role_that_is_not_pgkg_app_keeps_the_policy_path(
    pool: asyncpg.Pool, tenancy
) -> None:
    _, args = _scope_args(tenancy, "own org", "chunks")
    async with pool.acquire() as conn, as_other_role(conn, tenancy["mine"]):
        rows = await conn.fetch(KEYWORD_ARM, *args)
        retrieved = await conn.fetch(
            "SELECT item_id FROM pgkg_retrieve($1, NULL, 10, 200, $2)",
            args[0],
            tenancy["namespace"],
        )

    assert tenancy["ids"][("chunks", "home")] in {r["item_id"] for r in rows}
    assert retrieved, "pgkg_retrieve() returned nothing on the default path"


async def test_a_role_that_is_not_pgkg_app_is_refused_the_owner_arm(
    pool: asyncpg.Pool, tenancy
) -> None:
    """Selecting the owner arm without the grant fails closed, by name."""
    _, args = _scope_args(tenancy, "own org", "chunks")
    async with pool.acquire() as conn, as_other_role(conn, tenancy["mine"]):
        await conn.execute("SELECT set_config($1, 'owner', true)", ARM_GUC)
        with pytest.raises(
            asyncpg.InsufficientPrivilegeError,
            match="pgkg_bm25_candidates_as_owner",
        ):
            await conn.fetch(KEYWORD_ARM, *args)


async def test_the_owner_arm_itself_refuses_an_ungranted_role(
    pool: asyncpg.Pool, tenancy
) -> None:
    _, args = _scope_args(tenancy, "own org", "chunks")
    async with pool.acquire() as conn, as_other_role(conn, tenancy["mine"]):
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.fetch(
                "SELECT * FROM pgkg_bm25_candidates_as_owner("
                "$1, $2, NULL, 200, $3::uuid[], $4::uuid[], $5::uuid,"
                " $6::uuid[], NULL, $7)",
                *args,
            )


DISPATCHER = OWNER_ARM.replace("_as_owner", "")


@pytest.mark.parametrize("arm", ["policy", "owner"])
async def test_the_keyword_arm_returns_its_rows_in_rank_order(
    pool: asyncpg.Pool, tenancy, arm: str
) -> None:
    """A UNION ALL guarantees no order, and callers read the arm without an
    ORDER BY of their own, as they did when it was one SELECT with one."""
    _, args = _scope_args(tenancy, "names everyone", "chunks")
    async with pool.acquire() as conn:
        async with conn.transaction():
            await as_app(conn, tenancy["mine"])
            await conn.execute("SELECT set_config($1, $2, true)", ARM_GUC, arm)
            rows = await conn.fetch(
                "SELECT rank FROM pgkg_bm25_candidates("
                "$1, $2, NULL, 200, $3::uuid[], $4::uuid[], $5::uuid,"
                " $6::uuid[], NULL, $7)",
                *args,
            )
        source = await conn.fetchval(
            "SELECT prosrc FROM pg_proc WHERE oid = $1::regprocedure", DISPATCHER
        )

    ranks = [r["rank"] for r in rows]
    assert len(ranks) > 1
    assert ranks == sorted(ranks)
    assert "ORDER BY" in source.rsplit("UNION ALL", 1)[1], (
        "the order is what the planner happened to produce, not what the "
        "dispatcher asks for"
    )


# The org rule the owner arm restates, in the words it restates it from.  If
# pgkg_current_org() or pgkg_default_org() is redefined, the owner arm's copy
# in 059 has to be re-read against the new text before this is updated.
CURRENT_ORG_RULE = (
    "COALESCE(NULLIF(current_setting('pgkg.org_id', TRUE), '')::UUID,"
    " pgkg_default_org())"
)
DEFAULT_ORG_RULE = "SELECT '00000000-0000-0000-0000-000000000001'::UUID"


def _normalised(sql: str) -> str:
    """Whitespace, schema qualification and parenthesis padding removed, so
    the #30 qualification of these bodies does not read as a redefinition."""
    text = " ".join(sql.replace("public.", "").replace("pg_catalog.", "").split())
    return text.replace("( ", "(").replace(" )", ")")


async def test_the_owner_arm_restates_the_org_rule_the_policies_call(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        sources = {
            row["proname"]: row["prosrc"]
            for row in await conn.fetch(
                "SELECT proname, prosrc FROM pg_proc p"
                " JOIN pg_namespace n ON n.oid = p.pronamespace"
                " WHERE n.nspname = 'public' AND proname = ANY($1::text[])",
                ["pgkg_current_org", "pgkg_default_org",
                 "pgkg_bm25_candidates_as_owner"],
            )
        }

    message = (
        "pgkg_current_org() or pgkg_default_org() no longer reads the way "
        "pgkg_bm25_candidates_as_owner() restates it; update the owner arm's "
        "readable_orgs before this pin"
    )
    assert _normalised(sources["pgkg_current_org"]) == _normalised(
        f"SELECT {CURRENT_ORG_RULE}"
    ), message
    assert _normalised(sources["pgkg_default_org"]) == _normalised(
        DEFAULT_ORG_RULE
    ), message
    assert _normalised(CURRENT_ORG_RULE) in _normalised(
        sources["pgkg_bm25_candidates_as_owner"]
    ), message


async def test_a_schema_without_059_is_told_to_migrate(pool: asyncpg.Pool) -> None:
    """`pgkg check` against a schema that predates 059 says what to do rather
    than printing an UndefinedFunctionError traceback."""
    from pgkg.config import row_security_warnings

    async with pool.acquire() as conn:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute("DROP FUNCTION pgkg_owner_arm_bypasses_policy()")
            await conn.execute("DROP FUNCTION pgkg_leakproof_state() CASCADE")
            warnings = await row_security_warnings(conn, keyword_arm="policy")
        finally:
            await transaction.rollback()

    assert len(warnings) == 1, warnings
    assert "059" in warnings[0] and "pgkg migrate" in warnings[0]
