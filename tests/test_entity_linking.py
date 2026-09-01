"""Entity resolution under concurrency, and the branches nothing else reaches.

pgkg_link_entity() reads before it inserts, so two ingests that first mention
the same entity race: the loser's INSERT hits
entities_namespace_name_type_key.  Wrapping ingest in one transaction (ADR 0001
phase 0) widens that window from a single statement to a whole document, which
turns a rare interleaving into a routine one.  Resolution must therefore be
idempotent at the SQL level rather than only recoverable by the caller.
"""
from __future__ import annotations

import asyncio
import math
import pathlib
import uuid

import asyncpg
import pytest
from pgvector import HalfVector


def _ns(tag: str) -> str:
    return f"link_{tag}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
async def embed_dim(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT pgkg_embedding_dim('entities', 'embedding')"
        )


def one_hot(dim: int, *, hot: int = 0) -> HalfVector:
    v = [0.0] * dim
    v[hot] = 1.0
    return HalfVector(v)


async def _wait_for_a_blocked_lock(conn: asyncpg.Connection) -> bool:
    for _ in range(500):
        if await conn.fetchval("SELECT COUNT(*) FROM pg_locks WHERE NOT granted"):
            return True
        await asyncio.sleep(0.01)
    return False


async def test_link_entity_yields_the_winning_row_when_a_concurrent_writer_inserts_first(
    pool: asyncpg.Pool, embed_dim: int
) -> None:
    """The loser of the create race must return the winner's id, not raise.

    The competitor holds an uncommitted insert of the same (namespace, name,
    type), so the call under test blocks on the unique index and then finds the
    row committed underneath it — exactly the interleaving that a caller-side
    retry exists to paper over.
    """
    namespace = _ns("race")
    competitor = await pool.acquire()
    other = await pool.acquire()
    try:
        tx = competitor.transaction()
        await tx.start()
        winner_id = await competitor.fetchval(
            """
            INSERT INTO entities (name, type, embedding, namespace)
            VALUES ('Ada Lovelace', 'concept', $1, $2)
            RETURNING id
            """,
            one_hot(embed_dim),
            namespace,
        )

        task = asyncio.create_task(
            other.fetchval(
                "SELECT pgkg_link_entity($1, 'Ada Lovelace', 'concept', $2)",
                namespace,
                one_hot(embed_dim),
            )
        )

        if not await _wait_for_a_blocked_lock(competitor):
            await tx.rollback()
            task.cancel()
            pytest.fail("pgkg_link_entity never blocked on the competing insert")

        await tx.commit()
        linked_id = await task
    finally:
        await pool.release(other)
        await pool.release(competitor)

    assert linked_id == winner_id

    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM entities WHERE namespace = $1", namespace
        )
    assert count == 1


async def test_link_entity_is_idempotent_under_parallel_first_mentions(
    pool: asyncpg.Pool, embed_dim: int
) -> None:
    """Four writers naming the same new entity at once agree on one id."""
    namespace = _ns("parallel")
    embedding = one_hot(embed_dim)

    async def link() -> uuid.UUID:
        async with pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT pgkg_link_entity($1, 'Analytical Engine', 'concept', $2)",
                namespace,
                embedding,
            )

    ids = await asyncio.gather(*(link() for _ in range(4)))

    assert len(set(ids)) == 1
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM entities WHERE namespace = $1", namespace
            )
            == 1
        )


async def test_link_entity_matches_an_untyped_row_when_no_type_is_given(
    pool: asyncpg.Pool, embed_dim: int
) -> None:
    """The (type IS NULL AND p_type IS NULL) branch resolves rather than
    inserting a second untyped row.

    Type only gates the exact-match stage: asking for the same name under a
    type, with no embedding to reach the similarity stage, is a different
    entity.
    """
    namespace = _ns("untyped")
    async with pool.acquire() as conn:
        created = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'untyped thing', NULL, $2)",
            namespace,
            one_hot(embed_dim),
        )
        again = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'untyped thing', NULL, $2)",
            namespace,
            one_hot(embed_dim),
        )
        typed = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'untyped thing', 'concept', NULL)",
            namespace,
        )

    assert again == created
    assert typed != created


async def test_link_entity_creates_without_an_embedding(
    pool: asyncpg.Pool, embed_dim: int
) -> None:
    """A NULL embedding skips the similarity stage; the exact-name stage still
    dedupes, so two calls yield one row."""
    namespace = _ns("noemb")
    async with pool.acquire() as conn:
        first = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'no vector here', 'concept', NULL)",
            namespace,
        )
        second = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'no vector here', 'concept', NULL)",
            namespace,
        )
        stored = await conn.fetchval(
            "SELECT embedding FROM entities WHERE id = $1", first
        )

    assert first == second
    assert stored is None


async def test_link_entity_default_threshold_rejects_a_distant_neighbour(
    pool: asyncpg.Pool, embed_dim: int
) -> None:
    """At the default 0.85 an orthogonal embedding is a different entity even
    when the names are near-identical, so the trigram stage cannot merge on
    spelling alone."""
    namespace = _ns("threshold")
    async with pool.acquire() as conn:
        original = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'William Shakespeare', 'concept', $2)",
            namespace,
            one_hot(embed_dim, hot=0),
        )
        distant = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'William Shakespear', 'concept', $2)",
            namespace,
            one_hot(embed_dim, hot=1),
        )
        near = await conn.fetchval(
            "SELECT pgkg_link_entity($1, 'William Shakespere', 'concept', $2)",
            namespace,
            one_hot(embed_dim, hot=0),
        )

    assert distant != original
    assert near == original


# ---------------------------------------------------------------------------
# Stage 2 and the trigram index (issue #17, migration 051).
#
# pgkg_link_entity()'s second stage used to filter with
# `similarity(name, p_name) > 0.6` — pg_trgm's FUNCTION, whose index support
# lives on the `%` OPERATOR — so it reached entities_name_trgm_idx for no role,
# the table owner included.  That makes it unlike 047's finding, which was
# about promotion under a policy: there is nothing here for a leakproof marking
# to fix, because the operator never appeared.
#
# The measurements below are structured the way 047's are: a corpus large
# enough that the index is the cheaper plan, `enable_seqscan` untouched, and a
# probe that puts the previous shape back and shows the index goes away with
# it.  What is measured is the shipped function itself rather than a copy of
# its statement — pg_stat_user_indexes counts scans of the index no matter how
# deeply nested the statement that made them is, and a replica statement can
# reach an index the function's own cached plan does not.
# ---------------------------------------------------------------------------

ORG_GUC = "pgkg.org_id"
TRGM_INDEX = "entities_name_trgm_idx"

MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent / "migrations"

# The winner, the runner-up, and a row that passes the embedding test but not
# the name test.  Similarities to their probes, measured: WINNER 0.844,
# RUNNER_UP 0.706, and UNRELATED 0.538 against its own probe — above pg_trgm's
# default threshold of 0.3, so `%` alone offers it as a candidate and only the
# 0.6 recheck rejects it.
WINNER_NAME = "Vantablack Isodyne Consortium"
RUNNER_UP_NAME = "Vantablack Isodyne Consortia"
UNRELATED_NAME = "Perihelion Drydock Authority"
NEAR_PROBE = "Vantablak Isodyne Consortium"
UNRELATED_PROBE = "Perihelion Ironworks Authority"

# 024's stage 2, verbatim, for the probe that shows the rewrite is what buys
# the index.  Everything else about the function is the same in both shapes.
PRIOR_SHAPE_SQL = """
CREATE OR REPLACE FUNCTION pgkg_link_entity(
    p_namespace  TEXT,
    p_name       TEXT,
    p_type       TEXT,
    p_embedding  halfvec,
    p_threshold  REAL DEFAULT 0.85
) RETURNS UUID
LANGUAGE plpgsql
SECURITY INVOKER
AS $prior$
DECLARE
    v_id  UUID;
    v_org UUID := pgkg_current_org();
BEGIN
    SELECT id INTO v_id
    FROM entities
    WHERE org_id = v_org
      AND namespace = p_namespace
      AND name = p_name
      AND (type = p_type OR (type IS NULL AND p_type IS NULL))
    LIMIT 1;

    IF v_id IS NOT NULL THEN
        RETURN v_id;
    END IF;

    IF p_embedding IS NOT NULL THEN
        SELECT id INTO v_id
        FROM entities
        WHERE org_id = v_org
          AND namespace = p_namespace
          AND similarity(name, p_name) > 0.6
          AND (1 - (embedding <=> p_embedding)) > p_threshold
        ORDER BY (embedding <=> p_embedding)
        LIMIT 1;
    END IF;

    IF v_id IS NOT NULL THEN
        RETURN v_id;
    END IF;

    INSERT INTO entities (name, type, embedding, namespace, org_id)
    VALUES (p_name, p_type, p_embedding, p_namespace, v_org)
    ON CONFLICT (org_id, namespace, name, COALESCE(type, '')) DO NOTHING
    RETURNING id INTO v_id;

    IF v_id IS NULL THEN
        SELECT id INTO v_id
        FROM entities
        WHERE org_id = v_org
          AND namespace = p_namespace
          AND name = p_name
          AND (type = p_type OR (type IS NULL AND p_type IS NULL))
        LIMIT 1;
    END IF;

    RETURN v_id;
END;
$prior$;
"""


def mixed(dim: int, *, primary: int, secondary: int, weight: float) -> HalfVector:
    """A unit vector whose cosine similarity to one_hot(primary) is `weight`."""
    v = [0.0] * dim
    v[primary] = weight
    v[secondary] = math.sqrt(1.0 - weight * weight)
    return HalfVector(v)


def _shipped_migration() -> str:
    paths = sorted(MIGRATIONS_DIR.glob("051_*.sql"))
    assert len(paths) == 1, f"expected exactly one 051 migration, found {paths}"
    return paths[0].read_text()


@pytest.fixture(scope="module")
async def dedup_corpus(pool: asyncpg.Pool, embed_dim: int):
    """Enough entities that the trigram index is the cheaper plan.

    At a few hundred rows a sequential scan is what the planner should pick and
    both shapes of the predicate look identical, so the fixture is sized where
    the choice is real — the scale the finding was measured at.  Only the three
    named rows carry an embedding: stage 2 discards a NULL embedding anyway, and
    40,000 more of them would be a bulk HNSW build in a fixture that is not
    about vector search.

    The rows are removed again when the module finishes.  They are not
    incidental to other tests: doubling the size of `entities` and splitting it
    across two large orgs changes which plan the planner picks for the
    gazetteer arms 047 measures, so a fixture of this size that outlives its
    module makes another module's assertions fail for a reason that has nothing
    to do with what either module is about.
    """
    namespace = _ns("dedup")
    async with pool.acquire() as conn:
        org = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
            f"dedup_{uuid.uuid4().hex[:10]}",
        )
        await conn.execute(
            """
            INSERT INTO entities (name, type, namespace, org_id)
            SELECT 'filler entity number ' || g, 'thing', $1, $2
            FROM generate_series(1, 40000) g
            """,
            namespace,
            org,
        )
        await conn.execute(
            """
            INSERT INTO entities (name, type, embedding, namespace, org_id)
            VALUES ($1, 'thing', $4, $7, $8),
                   ($2, 'thing', $5, $7, $8),
                   ($3, 'thing', $6, $7, $8)
            """,
            WINNER_NAME,
            RUNNER_UP_NAME,
            UNRELATED_NAME,
            one_hot(embed_dim, hot=0),
            mixed(embed_dim, primary=0, secondary=3, weight=0.9),
            one_hot(embed_dim, hot=0),
            namespace,
            org,
        )
        # VACUUM as well as ANALYZE: a freshly bulk-loaded GIN index still holds
        # its pending list, and the planner charges a bitmap scan for reading it.
        await conn.execute("VACUUM ANALYZE entities")
        named = {
            row["name"]: row["id"]
            for row in await conn.fetch(
                "SELECT id, name FROM entities WHERE namespace = $1 AND name = ANY($2)",
                namespace,
                [WINNER_NAME, RUNNER_UP_NAME, UNRELATED_NAME],
            )
        }
    assert len(named) == 3, f"the fixture seeded {named}"

    yield org, namespace, named

    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM entities WHERE namespace = $1", namespace)
        await conn.execute("VACUUM ANALYZE entities")


async def _scoped(conn: asyncpg.Connection, org: uuid.UUID) -> None:
    await conn.execute("SELECT set_config($1, $2, false)", ORG_GUC, str(org))


async def _trigram_scans(conn: asyncpg.Connection) -> int:
    await conn.execute("SELECT pg_stat_force_next_flush()")
    return await conn.fetchval(
        "SELECT COALESCE(idx_scan, 0) FROM pg_stat_user_indexes"
        " WHERE indexrelname = $1",
        TRGM_INDEX,
    )


@pytest.mark.parametrize("role", ["owner", "pgkg_app"])
async def test_the_similarity_stage_reaches_the_trigram_index(
    pool: asyncpg.Pool, dedup_corpus, embed_dim: int, role: str
) -> None:
    """Stage 2 must probe entities_name_trgm_idx rather than read every entity.

    Both roles are measured because this defect is not the one 047 fixed: a
    function call over a column is not an index condition for anybody, so the
    owner lost the index too, and a test that only measured pgkg_app would
    leave the more surprising half of the claim unpinned.
    """
    org, namespace, named = dedup_corpus
    async with pool.acquire() as conn:
        await _scoped(conn, org)
        before = await _trigram_scans(conn)
        if role == "owner":
            linked = await conn.fetchval(
                "SELECT pgkg_link_entity($1, $2, 'thing', $3)",
                namespace,
                NEAR_PROBE,
                one_hot(embed_dim, hot=0),
            )
            acting = "postgres"
        else:
            async with conn.transaction():
                await conn.execute("SET LOCAL ROLE pgkg_app")
                await conn.execute(
                    "SELECT set_config($1, $2, true)", ORG_GUC, str(org)
                )
                acting = await conn.fetchval("SELECT current_user")
                linked = await conn.fetchval(
                    "SELECT pgkg_link_entity($1, $2, 'thing', $3)",
                    namespace,
                    NEAR_PROBE,
                    one_hot(embed_dim, hot=0),
                )
        after = await _trigram_scans(conn)

    if role == "pgkg_app":
        assert acting == "pgkg_app", "the role never changed, so nothing was measured"
    assert linked == named[WINNER_NAME], (
        "stage 2 did not resolve the near-duplicate at all, so no plan claim "
        "below means anything"
    )
    assert after > before, (
        f"as {acting}, pgkg_link_entity() resolved a near-duplicate without "
        f"ever probing {TRGM_INDEX}, so stage 2 read the whole entity table"
    )


async def test_the_prior_similarity_call_reached_no_index_for_any_role(
    pool: asyncpg.Pool, dedup_corpus, embed_dim: int
) -> None:
    """The probe that makes the test above non-vacuous.

    With 024's stage 2 back in place — `similarity(name, p_name) > 0.6`, the
    function and not the operator — the same call resolves the same entity and
    touches the trigram index not once, as the owner.  Restoring the shipped
    migration restores the probe.
    """
    org, namespace, named = dedup_corpus
    async with pool.acquire() as conn:
        await _scoped(conn, org)
        await conn.execute(PRIOR_SHAPE_SQL)
        try:
            before = await _trigram_scans(conn)
            prior = await conn.fetchval(
                "SELECT pgkg_link_entity($1, $2, 'thing', $3)",
                namespace,
                NEAR_PROBE,
                one_hot(embed_dim, hot=0),
            )
            after = await _trigram_scans(conn)
        finally:
            await conn.execute(_shipped_migration())
        restored = await _trigram_scans(conn)
        again = await conn.fetchval(
            "SELECT pgkg_link_entity($1, $2, 'thing', $3)",
            namespace,
            NEAR_PROBE,
            one_hot(embed_dim, hot=0),
        )
        restored_after = await _trigram_scans(conn)

    assert prior == named[WINNER_NAME], (
        "the prior shape resolved something else, so the two shapes are not "
        "being compared on the same question"
    )
    assert after == before, (
        f"the prior shape reached {TRGM_INDEX}, so the operator rewrite is not "
        f"what the index test is measuring ({before} -> {after} scans)"
    )
    assert again == named[WINNER_NAME]
    assert restored_after > restored, (
        "re-applying migration 051 did not restore the index probe"
    )


async def test_the_operator_rewrite_keeps_the_candidate_set_and_the_winner(
    pool: asyncpg.Pool, dedup_corpus, embed_dim: int
) -> None:
    """`%` and `similarity() > 0.6` are not the same predicate, so the rewrite
    is only safe if it decides every one of these the same way it used to.

    The probes cover the whole shape of stage 2: the near-duplicate that must
    resolve, the same name with an embedding nearer the runner-up (which is
    what pins the ORDER BY, and proves the candidate set has more than one
    member), a name close enough for pg_trgm's DEFAULT 0.3 threshold but not
    for 0.6, and a name too far to be a candidate at all.  Each call runs in a
    transaction that is rolled back, so a probe that creates an entity does not
    change the answer to the next one.
    """
    org, namespace, named = dedup_corpus
    probes = [
        (NEAR_PROBE, one_hot(embed_dim, hot=0)),
        (NEAR_PROBE, mixed(embed_dim, primary=0, secondary=3, weight=0.9)),
        (UNRELATED_PROBE, one_hot(embed_dim, hot=0)),
        ("Nothing Like Any Seeded Name", one_hot(embed_dim, hot=0)),
    ]
    by_id = {value: key for key, value in named.items()}

    async def outcomes(conn: asyncpg.Connection) -> list[str]:
        results = []
        for name, embedding in probes:
            tx = conn.transaction()
            await tx.start()
            linked = await conn.fetchval(
                "SELECT pgkg_link_entity($1, $2, 'thing', $3)",
                namespace,
                name,
                embedding,
            )
            await tx.rollback()
            results.append(by_id.get(linked, "created a new entity"))
        return results

    async with pool.acquire() as conn:
        await _scoped(conn, org)
        shipped = await outcomes(conn)
        await conn.execute(PRIOR_SHAPE_SQL)
        try:
            prior = await outcomes(conn)
        finally:
            await conn.execute(_shipped_migration())

    assert shipped == prior, (
        f"the rewrite changed which entity stage 2 resolves: {prior} became "
        f"{shipped}"
    )
    assert shipped == [
        WINNER_NAME,
        RUNNER_UP_NAME,
        "created a new entity",
        "created a new entity",
    ], f"stage 2 no longer decides these the way 024 did: {shipped}"


async def test_a_caller_cannot_narrow_entity_dedup_by_raising_the_threshold(
    pool: asyncpg.Pool, dedup_corpus, embed_dim: int
) -> None:
    """pg_trgm.similarity_threshold is session state, and `%` reads it, so an
    unpinned rewrite would let any caller redefine what counts as the same
    entity.  Raised above the candidate's similarity it would drop a match the
    function is supposed to make, and the duplicate entity that follows is
    silent.  051 pins the threshold in the function's own definition, which is
    also why the caller's value survives the call.
    """
    org, namespace, named = dedup_corpus
    async with pool.acquire() as conn:
        await _scoped(conn, org)
        async with conn.transaction():
            await conn.execute("SET LOCAL pg_trgm.similarity_threshold = 0.95")
            linked = await conn.fetchval(
                "SELECT pgkg_link_entity($1, $2, 'thing', $3)",
                namespace,
                NEAR_PROBE,
                one_hot(embed_dim, hot=0),
            )
            still_set = await conn.fetchval("SHOW pg_trgm.similarity_threshold")

    assert linked == named[WINNER_NAME], (
        "a caller holding a stricter pg_trgm threshold stopped stage 2 from "
        "resolving a near-duplicate it resolves at the default"
    )
    assert float(still_set) == pytest.approx(0.95), (
        "pgkg_link_entity() left the caller's pg_trgm threshold changed, so it "
        "has redefined every later % in this transaction"
    )


async def test_the_confirmation_holds_the_line_at_0_6_without_the_pin(
    pool: asyncpg.Pool, dedup_corpus, embed_dim: int
) -> None:
    """Why stage 2 keeps `similarity(name, p_name) > 0.6` behind the operator.

    With the threshold pinned, `%` alone is already the same predicate, so the
    confirmation is unreachable and no test of the shipped function can tell it
    is there.  What it defends against is the pin going missing: a later
    migration that replaces this function and drops the proconfig line hands
    the definition of "the same entity" to pg_trgm's default of 0.3, and the
    wrong merge that follows is silent.  So the probe takes the pin off the
    function, puts a caller's lowered threshold underneath it, and requires the
    0.6 rule to hold anyway — the operator offers this pair as a candidate at
    0.1, and the confirmation is the only thing left to reject it.
    """
    org, namespace, named = dedup_corpus
    signature = "pgkg_link_entity(TEXT, TEXT, TEXT, halfvec, REAL)"
    async with pool.acquire() as conn:
        await _scoped(conn, org)
        await conn.execute(
            f"ALTER FUNCTION {signature} RESET pg_trgm.similarity_threshold"
        )
        try:
            unpinned = await conn.fetchval(
                "SELECT proconfig FROM pg_proc WHERE oid = $1::regprocedure",
                signature,
            )
            async with conn.transaction():
                await conn.execute("SET LOCAL pg_trgm.similarity_threshold = 0.1")
                offered = await conn.fetchval(
                    "SELECT $1::text % $2::text", UNRELATED_NAME, UNRELATED_PROBE
                )
                tx = conn.transaction()
                await tx.start()
                linked = await conn.fetchval(
                    "SELECT pgkg_link_entity($1, $2, 'thing', $3)",
                    namespace,
                    UNRELATED_PROBE,
                    one_hot(embed_dim, hot=0),
                )
                await tx.rollback()
        finally:
            await conn.execute(_shipped_migration())
        repinned = await conn.fetchval(
            "SELECT proconfig FROM pg_proc WHERE oid = $1::regprocedure", signature
        )

    assert not unpinned, (
        f"the pin was not actually removed, so nothing was probed: {unpinned}"
    )
    assert offered, (
        "the operator no longer offers this pair as a candidate at a lowered "
        "threshold, so the probe is not exercising the confirmation"
    )
    assert linked != named[UNRELATED_NAME], (
        "with the pin gone, stage 2 adopted an entity whose name similarity is "
        "below 0.6: the candidate generator is deciding what the confirmation "
        "is there to decide"
    )
    assert repinned == ["pg_trgm.similarity_threshold=0.6"], (
        f"re-applying migration 051 did not restore the pin: {repinned}"
    )
