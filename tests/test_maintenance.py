"""The scheduled path: the jobs that were built and never run (issue #19).

Four things exist in the schema and were reachable only from a test or an
operator's psql session — the gazetteer sweep that populates `entity_mentions`
(ADR 0001, D2), `pgkg_recompute_pagerank()`, `pgkg_contradict()` and
`pgkg_expire_due()`.  All four are recorded as built-but-unscheduled in
docs/adrs/0001-implementation-notes.md §4, and the first is why `entity_mentions`
was empty in every deployment: nothing in the product called it.

So the tests here drive `pgkg.maintenance`, never `Gazetteer`, and reach the
graph through the two ingest pipelines a customer uses.  Driving the gazetteer
directly is what the existing tests in test_entity_mentions.py do, and it is
exactly why an empty table went unnoticed for a phase.

Two properties beyond "it runs".  A cron entry that overlaps itself is the
normal failure mode of anything on a timer, so an overlapping run must decline
its task rather than repeat it; and each task is selectable on its own, because
a pagerank pass and a mention sweep have nothing to do with each other and an
operator debugging one should not have to run the other.

The fifth task is issue #22's, and it is tested for two more (last section).  It
spends money, so it must not embed a row it cannot write; and it is the only one
that can be interrupted halfway through something worth keeping, so a batch is
its own transaction and a run that dies leaves what it finished behind it.
"""
from __future__ import annotations

import asyncio
import uuid

import asyncpg
import pytest
from pgvector import HalfVector

from pgkg.ml import Proposition


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def raw_vec(hot_index: int = 0, dim: int = 1024) -> list[float]:
    raw = [0.0] * dim
    raw[hot_index] = 1.0
    return raw


async def new_org(conn: asyncpg.Connection) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("org")
    )


async def new_collection(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    kind: str = "corpus",
    claim_scope: str = "org",
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO collections
            (org_id, owner_org_id, name, kind, visibility, claim_scope)
        VALUES ($1, $1, $2, $3, 'private', $4)
        RETURNING id
        """,
        org_id, unique("coll"), kind, claim_scope,
    )


async def mention_rows(
    pool: asyncpg.Pool,
    org_id: uuid.UUID,
    *,
    collection_id: uuid.UUID | None = None,
) -> list[asyncpg.Record]:
    """The mention edges of one org, optionally of one collection only.

    The collection filter matters in the two-pipeline tests: a chat turn is a
    passage in the chunk store too (049), and it names the entity it produced —
    so the sweep matches it as well, correctly, and a test about the corpus edge
    has to say which side of the join it is asserting about.
    """
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT e.name, c.text
            FROM entity_mentions m
            JOIN entities e ON e.id = m.entity_id
            JOIN chunks c   ON c.id = m.chunk_id
            WHERE m.org_id = $1
              AND ($2::uuid IS NULL OR c.collection_id = $2)
            ORDER BY e.name
            """,
            org_id, collection_id,
        )


def chat_extractor(*propositions: Proposition):
    """An extractor that returns fixed facts, standing in for the LLM.

    The propositions carry a subject name, which is what makes chat ingest
    create an entity — the offline fallback's '?' subject creates nothing, so a
    test that used it would be asserting about an empty name list.
    """

    async def _extract(chunk_text: str, *, max_propositions: int = 20, cache=None):
        return list(propositions)

    return _extract


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch):
    """No embedder, no extractor: D2 prices this whole path at no model call."""
    from pgkg import ml

    monkeypatch.setattr(ml, "embed", lambda texts: [raw_vec(1) for _ in texts])
    return monkeypatch


async def ingest_corpus(
    pool: asyncpg.Pool,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    text: str,
    external_id: str | None = None,
):
    from pgkg.corpus import CorpusIngest

    corpus = CorpusIngest(
        pool,
        org_id=org_id,
        collection_id=collection_id,
        embed=lambda texts: [raw_vec(2) for _ in texts],
    )
    return await corpus.upsert_document(
        external_id=external_id or unique("doc"), text=text
    )


async def ingest_chat(
    pool: asyncpg.Pool,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    namespace: str,
    text: str,
) -> None:
    from pgkg.memory import Memory, Scope

    memory = Memory(
        pool,
        namespace=namespace,
        scope=Scope(org_id=org_id, collection_id=collection_id),
        use_extract_cache=False,
    )
    try:
        await memory.ingest(text)
    finally:
        await memory.aclose()


# ---------------------------------------------------------------------------
# The mention edge, through the product surface
# ---------------------------------------------------------------------------


async def test_the_maintenance_path_joins_a_passage_to_the_names_chat_produced(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """The acceptance case from issue #19, with nothing called by hand.

    Chat facts name the Helios migration; a runbook page names it too and
    shares no other wording with the facts.  Nothing about ingest creates the
    edge between them — D7 forbids the cross-product on either online path — so
    the mention exists only if the scheduled path ran.
    """
    from pgkg import ml
    from pgkg.maintenance import Maintenance

    namespace = unique("ns")
    offline.setattr(
        ml,
        "extract_propositions_async",
        chat_extractor(
            Proposition(
                text="The Helios migration slipped to Q3.",
                subject="Helios migration",
                predicate="slipped to",
                object="Q3",
                object_is_literal=True,
            )
        ),
    )

    async with pool.acquire() as conn:
        org = await new_org(conn)
        chat = await new_collection(conn, org_id=org, kind="chat")
        corpus = await new_collection(conn, org_id=org, kind="corpus")

    await ingest_chat(
        pool, org_id=org, collection_id=chat, namespace=namespace,
        text="The Helios migration slipped to Q3.",
    )
    await ingest_corpus(
        pool, org_id=org, collection_id=corpus,
        text=(
            "# Runbook\n\nCutting over the ledger requires draining the queue "
            "before the Helios migration window opens."
        ),
    )

    report = await Maintenance(pool, org_id=org).run(tasks=["mentions"])

    rows = await mention_rows(pool, org, collection_id=corpus)
    assert [r["name"] for r in rows] == ["Helios migration"]
    assert "Runbook" in rows[0]["text"]
    assert report.task("mentions").ran is True
    # Two edges, not one: the chat turn is a passage as well, and it names the
    # entity it produced.  That is the reverse direction of the same edge and it
    # is wanted — a retrieved passage seeding the facts about what it mentions.
    assert report.task("mentions").changed == 2


async def test_the_maintenance_path_matches_a_new_name_against_a_settled_corpus(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """The order steady state actually produces, and the one a sweep can miss.

    A corpus is bulk-loaded and swept; the chat facts that name what it talks
    about arrive days later.  The passage's watermark says it has been matched,
    so a sweep that only looks at unmatched passages never sees the new name —
    which would make the edge exist only for corpora ingested after the facts.
    """
    from pgkg import ml
    from pgkg.maintenance import Maintenance

    namespace = unique("ns")
    offline.setattr(
        ml,
        "extract_propositions_async",
        chat_extractor(
            Proposition(
                text="Helios is behind schedule.",
                subject="Helios",
                predicate="is",
                object="behind schedule",
                object_is_literal=True,
            )
        ),
    )

    async with pool.acquire() as conn:
        org = await new_org(conn)
        chat = await new_collection(conn, org_id=org, kind="chat")
        corpus = await new_collection(conn, org_id=org, kind="corpus")

    await ingest_corpus(
        pool, org_id=org, collection_id=corpus,
        text="Helios is the event-sourced ledger behind interbank settlement.",
    )
    settled = await Maintenance(pool, org_id=org).run(tasks=["mentions"])
    assert settled.task("mentions").changed == 0

    await ingest_chat(
        pool, org_id=org, collection_id=chat, namespace=namespace,
        text="Helios is behind schedule.",
    )

    after = await Maintenance(pool, org_id=org).run(tasks=["mentions"])

    rows = await mention_rows(pool, org, collection_id=corpus)
    assert [r["name"] for r in rows] == ["Helios"]
    assert "interbank settlement" in rows[0]["text"]
    # The corpus passage and the chat turn that produced the name: the first is
    # reached only by the name-side sweep, the second by the passage side.
    assert after.task("mentions").changed == 2


async def test_a_settled_org_reports_no_work_rather_than_repeating_it(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """What makes the entry point safe on a timer at all."""
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        corpus = await new_collection(conn, org_id=org, kind="corpus")
        await conn.execute(
            "INSERT INTO entities (name, type, org_id) VALUES ('Helios', "
            "'concept', $1)",
            org,
        )

    await ingest_corpus(
        pool, org_id=org, collection_id=corpus,
        text="Helios is the ledger behind settlement.",
    )

    maintenance = Maintenance(pool, org_id=org)
    first = await maintenance.run(tasks=["mentions"])
    second = await maintenance.run(tasks=["mentions"])

    assert first.task("mentions").changed == 1
    assert second.task("mentions").changed == 0
    assert second.task("mentions").scanned == 0


async def test_the_sweep_drains_a_backlog_larger_than_one_batch(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """A batch is the unit of work; the run is responsible for the backlog."""
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        await conn.execute(
            "INSERT INTO entities (name, type, org_id) VALUES ('Helios', "
            "'concept', $1)",
            org,
        )
        for i in range(5):
            await conn.execute(
                "INSERT INTO chunks (text, org_id, collection_id) "
                "VALUES ($1, $2, $3)",
                f"Passage {i} explains why Helios exists.", org, collection,
            )

    report = await Maintenance(pool, org_id=org, batch=2).run(tasks=["mentions"])

    assert report.task("mentions").changed == 5


async def test_a_sweep_that_stops_advancing_fails_the_run_rather_than_spinning(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """The failure mode of draining against a watermark.

    A run drains until a batch reports no work, which is only a stop condition
    while the watermark advances.  If it stops — a trigger dropped, a policy
    that hides the stamp from the role sweeping — the honest outcome is a run
    that fails loudly, not a nightly job that never exits.
    """
    from pgkg.gazetteer import Gazetteer, MatchResult
    from pgkg.maintenance import Maintenance

    class StuckGazetteer(Gazetteer):
        async def sweep(self, **kwargs) -> MatchResult:
            return MatchResult(chunks_scanned=1, mentions_added=0)

    async with pool.acquire() as conn:
        org = await new_org(conn)

    maintenance = Maintenance(
        pool,
        org_id=org,
        max_batches=3,
        gazetteer=StuckGazetteer(pool, org_id=org),
    )

    with pytest.raises(RuntimeError, match="not advancing"):
        await maintenance.run(tasks=["mentions"])


# ---------------------------------------------------------------------------
# Selecting tasks, and refusing what does not exist
# ---------------------------------------------------------------------------


async def test_each_task_is_selectable_on_its_own(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """A pagerank pass must not drag a mention sweep along with it."""
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        await conn.execute(
            "INSERT INTO entities (name, type, org_id) VALUES ('Helios', "
            "'concept', $1)",
            org,
        )
        await conn.execute(
            "INSERT INTO chunks (text, org_id, collection_id) VALUES ($1, $2, $3)",
            "Helios is the ledger.", org, collection,
        )

    report = await Maintenance(pool, org_id=org).run(tasks=["pagerank"])

    assert [t.task for t in report.tasks] == ["pagerank"]
    assert await mention_rows(pool, org) == []


async def test_every_task_runs_when_none_is_named(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """The cron case: one entry point, all four jobs, one report."""
    from pgkg.maintenance import TASKS, Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)

    report = await Maintenance(pool, org_id=org).run()
    # The CLI's own default is an empty list of --task flags, which has to mean
    # the same thing as naming none at all.
    from_cli = await Maintenance(pool, org_id=org).run(tasks=[])

    assert [t.task for t in report.tasks] == list(TASKS)
    assert [t.task for t in from_cli.tasks] == list(TASKS)
    assert all(t.ran for t in report.tasks)


async def test_a_task_nobody_defined_is_refused_before_anything_runs(
    pool: asyncpg.Pool,
) -> None:
    """A typo in a crontab must not be a silent no-op."""
    from pgkg.maintenance import Maintenance

    with pytest.raises(ValueError, match="pagerankk"):
        await Maintenance(pool).run(tasks=["pagerankk"])


def test_every_named_task_has_something_behind_it(pool: asyncpg.Pool) -> None:
    """The list of tasks and the work they name cannot drift apart."""
    from pgkg.maintenance import TASKS, Maintenance

    runners = Maintenance(pool)._runners()

    assert sorted(runners) == sorted(TASKS)


# ---------------------------------------------------------------------------
# Overlapping runs
# ---------------------------------------------------------------------------


async def test_a_run_that_overlaps_another_declines_the_task(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """The cron-overlap case, made deterministic by holding the lock.

    The lock is the product's own function, taken on a connection of this
    test's: a run that cannot have it reports that it did not run, and writes
    nothing — so the backlog is still there for the run that holds it.
    """
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        await conn.execute(
            "INSERT INTO entities (name, type, org_id) VALUES ('Helios', "
            "'concept', $1)",
            org,
        )
        await conn.execute(
            "INSERT INTO chunks (text, org_id, collection_id) VALUES ($1, $2, $3)",
            "Helios is the ledger.", org, collection,
        )

    holder = await pool.acquire()
    try:
        held = await holder.fetchval(
            "SELECT pgkg_try_maintenance_lock('mentions', $1)", org
        )
        assert held is True

        blocked = await Maintenance(pool, org_id=org).run(tasks=["mentions"])
    finally:
        await holder.execute(
            "SELECT pgkg_release_maintenance_lock('mentions', $1)", org
        )
        await pool.release(holder)

    assert blocked.task("mentions").ran is False
    assert blocked.task("mentions").changed == 0
    assert await mention_rows(pool, org) == []

    resumed = await Maintenance(pool, org_id=org).run(tasks=["mentions"])
    assert resumed.task("mentions").changed == 1


async def test_the_lock_is_held_per_org_not_globally(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """One tenant's long sweep must not stop every other tenant's."""
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        busy = await new_org(conn)
        other = await new_org(conn)

    holder = await pool.acquire()
    try:
        await holder.fetchval(
            "SELECT pgkg_try_maintenance_lock('mentions', $1)", busy
        )
        report = await Maintenance(pool, org_id=other).run(tasks=["mentions"])
    finally:
        await holder.execute(
            "SELECT pgkg_release_maintenance_lock('mentions', $1)", busy
        )
        await pool.release(holder)

    assert report.task("mentions").ran is True


async def test_two_concurrent_runs_neither_corrupt_nor_double_count(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """Two runs genuinely inside each other, and the numbers a defect moves.

    The overlap is made real rather than hoped for: the first run's sweep waits
    until the second has started, so the second is asking for the lock while the
    first holds it.  Exactly one of them does the work; the table holds one row
    per (entity, passage) whichever one wrote it; and every row is reported
    exactly once, because a yield counted twice is a yield a scheduler cannot
    act on.
    """
    from pgkg.gazetteer import Gazetteer, MatchResult
    from pgkg.maintenance import Maintenance

    class WaitingGazetteer(Gazetteer):
        """Holds its first batch open until the other run has arrived."""

        def __init__(self, *args, arrived: asyncio.Event, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.arrived = arrived
            self.entered = asyncio.Event()

        async def sweep(self, **kwargs) -> MatchResult:
            self.entered.set()
            await self.arrived.wait()
            return await super().sweep(**kwargs)

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        for name in ("Helios", "Postgres"):
            await conn.execute(
                "INSERT INTO entities (name, type, org_id) VALUES ($1, "
                "'concept', $2)",
                name, org,
            )
        for i in range(6):
            await conn.execute(
                "INSERT INTO chunks (text, org_id, collection_id) "
                "VALUES ($1, $2, $3)",
                f"Passage {i} explains why Helios runs on Postgres.",
                org, collection,
            )

    arrived = asyncio.Event()
    waiting = WaitingGazetteer(pool, org_id=org, arrived=arrived)
    first = Maintenance(pool, org_id=org, batch=2, gazetteer=waiting)
    second = Maintenance(pool, org_id=org, batch=2)

    async def overlap():
        # The timeout and the finally are so that a run which never reaches the
        # sweep fails this test rather than hanging it.
        try:
            await asyncio.wait_for(waiting.entered.wait(), timeout=10)
            return await second.run(tasks=["mentions"])
        finally:
            arrived.set()

    left, right = await asyncio.gather(
        first.run(tasks=["mentions"]), overlap()
    )

    async with pool.acquire() as conn:
        stored = await conn.fetchval(
            "SELECT COUNT(*) FROM entity_mentions WHERE org_id = $1", org
        )
        distinct = await conn.fetchval(
            "SELECT COUNT(*) FROM (SELECT DISTINCT entity_id, chunk_id "
            "FROM entity_mentions WHERE org_id = $1) AS pairs",
            org,
        )

    reported = left.task("mentions").changed + right.task("mentions").changed
    ran = [report.task("mentions").ran for report in (left, right)]
    assert stored == 12
    assert distinct == stored
    assert reported == stored
    assert sorted(ran) == [False, True]


# ---------------------------------------------------------------------------
# The other three jobs
# ---------------------------------------------------------------------------


async def test_pagerank_scores_this_orgs_entities_and_says_how_many(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """Nothing scheduled pgkg_recompute_pagerank, and the graph arm reads it.

    The namespace is not a flag an operator should have to know: an org's
    entities state which namespaces they live in, and every one of them is a
    subgraph that needs scoring (D3, D4).
    """
    from pgkg.maintenance import Maintenance

    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        first = await conn.fetchval(
            "INSERT INTO entities (name, type, namespace, org_id) "
            "VALUES ($1, 'concept', $2, $3) RETURNING id",
            unique("Helios"), namespace, org,
        )
        second = await conn.fetchval(
            "INSERT INTO entities (name, type, namespace, org_id) "
            "VALUES ($1, 'concept', $2, $3) RETURNING id",
            unique("Postgres"), namespace, org,
        )
        carrier = await conn.fetchval(
            "INSERT INTO propositions (text, namespace, org_id, subject_id, "
            "object_id) VALUES ('edge carrier', $1, $2, $3, $4) RETURNING id",
            namespace, org, first, second,
        )
        await conn.execute(
            "INSERT INTO edges (src_entity, dst_entity, relation, "
            "proposition_id) VALUES ($1, $2, 'depends_on', $3)",
            first, second, carrier,
        )

    report = await Maintenance(pool, org_id=org).run(tasks=["pagerank"])

    async with pool.acquire() as conn:
        scored = await conn.fetchval(
            "SELECT COUNT(*) FROM entity_pagerank ep JOIN entities e "
            "ON e.id = ep.entity_id WHERE e.org_id = $1",
            org,
        )

    assert scored == 2
    assert report.task("pagerank").changed == 2


async def test_expiries_withdraw_only_this_orgs_due_facts(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """pgkg_expire_due() was global; a per-tenant schedule cannot be.

    A maintenance run for one tenant that withdrew another tenant's expired
    facts would make the job unrunnable per org, which is how it has to run:
    the TTL is a property of the claim, and the operator schedules per customer.
    """
    from pgkg.maintenance import Maintenance

    namespace = unique("ns")
    async with pool.acquire() as conn:
        mine = await new_org(conn)
        theirs = await new_org(conn)
        my_collection = await new_collection(conn, org_id=mine, kind="chat")
        their_collection = await new_collection(conn, org_id=theirs, kind="chat")
        my_prop = await conn.fetchval(
            "INSERT INTO propositions (text, namespace, org_id, collection_id, "
            "expires_at) VALUES ('mine', $1, $2, $3, now() - interval '1 day') "
            "RETURNING id",
            namespace, mine, my_collection,
        )
        their_prop = await conn.fetchval(
            "INSERT INTO propositions (text, namespace, org_id, collection_id, "
            "expires_at) VALUES ('theirs', $1, $2, $3, now() - interval '1 day')"
            " RETURNING id",
            namespace, theirs, their_collection,
        )
        held = await conn.fetchval(
            "INSERT INTO propositions (text, namespace, org_id, collection_id, "
            "expires_at, legal_hold) VALUES ('held', $1, $2, $3, "
            "now() - interval '1 day', TRUE) RETURNING id",
            namespace, mine, my_collection,
        )

    report = await Maintenance(pool, org_id=mine).run(tasks=["expiries"])

    async with pool.acquire() as conn:
        rows = {
            r["id"]: r["invalidation_reason"]
            for r in await conn.fetch(
                "SELECT id, invalidation_reason FROM propositions "
                "WHERE id = ANY($1::uuid[])",
                [my_prop, their_prop, held],
            )
        }

    assert rows[my_prop] == "ttl"
    assert rows[their_prop] is None
    assert rows[held] is None
    assert report.task("expiries").changed == 1


async def test_contradictions_close_the_validity_a_supersession_left_open(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """Scheduling pgkg_contradict() needs a candidate rule, and there is one.

    Recording a supersession withdraws belief (021's trigger) and says nothing
    about validity, so the as-of-validity mode still answers with the replaced
    claim for every instant after its replacement.  Closing `valid_to` at the
    replacement's own clock is what pgkg_contradict() is for, and a recorded
    supersession is the one contradiction candidate this schema can name
    without a predicate vocabulary — which D-level consolidation still lacks.
    """
    from pgkg.maintenance import Maintenance

    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="chat")
        newer = await conn.fetchval(
            "INSERT INTO propositions (text, namespace, org_id, collection_id, "
            "claim_scope) VALUES ('lives in Berlin', $1, $2, $3, 'org') "
            "RETURNING id",
            namespace, org, collection,
        )
        older = await conn.fetchval(
            "INSERT INTO propositions (text, namespace, org_id, collection_id, "
            "claim_scope) VALUES ('lives in Lisbon', $1, $2, $3, 'org') "
            "RETURNING id",
            namespace, org, collection,
        )
        await conn.execute(
            "UPDATE propositions SET superseded_by = $1 WHERE id = $2",
            newer, older,
        )
        before = await conn.fetchrow(
            "SELECT valid_to, invalidation_reason FROM propositions WHERE id = $1",
            older,
        )
    assert before["invalidation_reason"] == "superseded"
    assert before["valid_to"] is None

    report = await Maintenance(pool, org_id=org).run(tasks=["contradictions"])

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT p.valid_to, n.recorded_at FROM propositions p "
            "JOIN propositions n ON n.id = $2 WHERE p.id = $1",
            older, newer,
        )

    assert row["valid_to"] == row["recorded_at"]
    assert report.task("contradictions").changed == 1


async def test_a_contradiction_never_crosses_a_claim_scope(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """D4's first rule, on the job that would otherwise break it.

    "Contradiction resolution runs only within a claim scope."  A user-scoped
    note that supersedes an org-scoped policy is a personal exception, not a
    correction, so the policy's validity stays open and the disagreement stays
    visible to whatever answers the question.
    """
    from pgkg.maintenance import Maintenance

    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="chat")
        user = await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) "
            "RETURNING id",
            org, unique("user"),
        )
        personal = await conn.fetchval(
            "INSERT INTO propositions (text, namespace, org_id, collection_id, "
            "claim_scope, visibility, owner_user_id) VALUES ('I fly economy', "
            "$1, $2, $3, 'user', 'private', $4) RETURNING id",
            namespace, org, collection, user,
        )
        policy = await conn.fetchval(
            "INSERT INTO propositions (text, namespace, org_id, collection_id, "
            "claim_scope) VALUES ('staff fly business', $1, $2, $3, 'org') "
            "RETURNING id",
            namespace, org, collection,
        )
        await conn.execute(
            "UPDATE propositions SET superseded_by = $1 WHERE id = $2",
            personal, policy,
        )

    report = await Maintenance(pool, org_id=org).run(tasks=["contradictions"])

    async with pool.acquire() as conn:
        valid_to = await conn.fetchval(
            "SELECT valid_to FROM propositions WHERE id = $1", policy
        )

    assert valid_to is None
    assert report.task("contradictions").changed == 0


# ---------------------------------------------------------------------------
# The fifth job: passages that reached the store without a vector (issue #22)
# ---------------------------------------------------------------------------


async def stranded_passages(
    pool: asyncpg.Pool,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    text: str,
    external_id: str | None = None,
) -> list[uuid.UUID]:
    """A live passage carrying no vector: the row issue #22 is about.

    Written by the real pipeline and then emptied, because the window that
    produces it is a race between two crawls — phase 2 saw a vectored row at an
    address and phase 3 created a new one anyway — and a test cannot schedule
    that.  What a test can do is assert about the state it leaves, which is the
    state the repair is defined over: `embedding IS NULL` on a chunk a current
    version still links.
    """
    result = await ingest_corpus(
        pool, org_id=org_id, collection_id=collection_id, text=text,
        external_id=external_id,
    )
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            UPDATE chunks SET embedding = NULL
            WHERE id IN (
                SELECT chunk_id FROM document_version_chunks
                WHERE document_version_id = $1
            )
            RETURNING id
            """,
            result.version_id,
        )
    return [row["id"] for row in rows]


async def vectors_of(
    pool: asyncpg.Pool, chunk_ids: list[uuid.UUID]
) -> dict[uuid.UUID, object]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, embedding, embedder_generation_id FROM chunks "
            "WHERE id = ANY($1::uuid[])",
            chunk_ids,
        )
    return {row["id"]: row for row in rows}


async def new_generation(conn: asyncpg.Connection) -> uuid.UUID:
    """A second embedding space, at the same width as the first.

    Same width deliberately: a wrong-space vector that does not fit the column
    fails loudly on its own, and the case D8 is about is the one that does fit.
    """
    return await conn.fetchval(
        """
        INSERT INTO embedder_generations
            (name, dim, storage_type, normalize, status)
        VALUES ($1, (SELECT dim FROM embedder_generations WHERE id = $2),
                'halfvec', TRUE, 'retiring')
        RETURNING id
        """,
        unique("bge-m3@"), await conn.fetchval("SELECT pgkg_generation_1()"),
    )


async def test_the_scheduled_path_vectors_a_passage_ingest_left_stranded(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """The acceptance case from issue #22, including why nothing else fixes it.

    The docstring the defect was found under said such a chunk is left for the
    next crawl.  It is not: the next crawl short-circuits on the unchanged
    document hash and never reaches the chunk, so the row keeps
    `embedding IS NULL` for good — invisible to the vector arm and to MMR, and
    silently.  The re-crawl here is what makes that half an assertion rather
    than a claim.
    """
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)

    external = unique("doc")
    text = "Draining the queue is what opens the Helios cutover window."
    stranded = await stranded_passages(
        pool, org_id=org, collection_id=collection, text=text,
        external_id=external,
    )
    assert len(stranded) == 1

    await ingest_corpus(
        pool, org_id=org, collection_id=collection, text=text,
        external_id=external,
    )
    before = await vectors_of(pool, stranded)
    assert before[stranded[0]]["embedding"] is None

    report = await Maintenance(
        pool, org_id=org, embed=lambda texts: [raw_vec(3) for _ in texts]
    ).run(tasks=["vectors"])

    after = await vectors_of(pool, stranded)
    assert after[stranded[0]]["embedding"] is not None
    assert report.task("vectors").scanned == 1
    assert report.task("vectors").changed == 1


async def test_a_repaired_passage_keeps_the_generation_it_was_written_in(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """D8's rule, on the job most able to break it.

    Vectors from two generations are incomparable, so the repair may only fill
    a row it can embed for — the generation the row itself declares — and may
    never restamp a row into the generation this process happens to run.  A
    cutover window is exactly when a stranded row from the outgoing generation
    is sitting there, and vectorising it into the incoming space would be the
    silent wrong answer this issue is already about.
    """
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        outgoing = await new_generation(conn)
        primary = await conn.fetchval(
            "SELECT generation_id FROM org_embedders WHERE org_id = $1 "
            "AND role = 'primary'",
            org,
        )

    mine = await stranded_passages(
        pool, org_id=org, collection_id=collection,
        text="Helios is the ledger behind interbank settlement.",
    )
    theirs = await stranded_passages(
        pool, org_id=org, collection_id=collection,
        text="The queue drains before the cutover window opens.",
    )
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE chunks SET embedder_generation_id = $1 WHERE id = ANY($2::uuid[])",
            outgoing, theirs,
        )

    asked: list[str] = []

    def embed(texts: list[str]) -> list[list[float]]:
        asked.extend(texts)
        return [raw_vec(3) for _ in texts]

    report = await Maintenance(pool, org_id=org, embed=embed).run(
        tasks=["vectors"]
    )

    rows = await vectors_of(pool, mine + theirs)
    assert rows[mine[0]]["embedding"] is not None
    assert rows[mine[0]]["embedder_generation_id"] == primary
    assert rows[theirs[0]]["embedding"] is None
    assert rows[theirs[0]]["embedder_generation_id"] == outgoing
    # And it is not embedded either: a row this process has no model for is a
    # row it must not send to the model it does have.  The write guard would
    # discard the vector, and the embedder would still have charged for it.
    assert any("Helios" in text for text in asked)
    assert not any("cutover window" in text for text in asked)
    # The one it cannot serve is counted, because a row left unvectored and
    # unmentioned is the defect this task exists to end.
    assert report.task("vectors").scanned == 2
    assert report.task("vectors").changed == 1


async def test_a_row_cut_over_mid_batch_is_not_given_the_old_space_s_vector(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """The same rule, at the only instant it can be broken by accident.

    The batch is chosen, the embedder is called, and the write lands after —
    so between the two the row's generation can move.  The write is guarded by
    the generation it was embedded for, which is why the answer here is "wrote
    nothing" rather than "wrote a vector from the space the row just left".
    """
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        incoming = await new_generation(conn)

    stranded = await stranded_passages(
        pool, org_id=org, collection_id=collection,
        text="Helios is the ledger behind interbank settlement.",
    )

    class CutOverMidBatch(Maintenance):
        async def _unvectored(self, conn, generation_id):
            rows = await super()._unvectored(conn, generation_id)
            await conn.execute(
                "UPDATE chunks SET embedder_generation_id = $1 "
                "WHERE id = ANY($2::uuid[])",
                incoming, [row["chunk_id"] for row in rows],
            )
            return rows

    report = await CutOverMidBatch(
        pool, org_id=org, embed=lambda texts: [raw_vec(3) for _ in texts]
    ).run(tasks=["vectors"])

    rows = await vectors_of(pool, stranded)
    assert rows[stranded[0]]["embedding"] is None
    assert rows[stranded[0]]["embedder_generation_id"] == incoming
    assert report.task("vectors").changed == 0


async def test_a_run_that_dies_halfway_keeps_the_rows_it_finished(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """Interruptibility, which is what makes this safe to run at all.

    An embedder that fails on the third of five batches is the ordinary
    outcome of a rate limit, and the run that retries an hour later must not
    have to pay for the batches that already succeeded.  So a batch is its own
    unit of work and its own transaction: what it wrote stays written.
    """
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)

    stranded = []
    for i in range(3):
        stranded += await stranded_passages(
            pool, org_id=org, collection_id=collection,
            text=f"Passage {i} explains why Helios exists at all.",
        )

    calls: list[int] = []

    def failing(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        if len(calls) == 2:
            raise RuntimeError("the embedder is rate limited")
        return [raw_vec(3) for _ in texts]

    maintenance = Maintenance(pool, org_id=org, batch=1, embed=failing)

    with pytest.raises(RuntimeError, match="rate limited"):
        await maintenance.run(tasks=["vectors"])

    rows = await vectors_of(pool, stranded)
    vectored = [row for row in rows.values() if row["embedding"] is not None]
    assert len(vectored) == 1

    # And the retry finishes the job rather than starting it again.
    resumed = await Maintenance(
        pool, org_id=org, embed=lambda texts: [raw_vec(3) for _ in texts]
    ).run(tasks=["vectors"])

    rows = await vectors_of(pool, stranded)
    assert all(row["embedding"] is not None for row in rows.values())
    assert resumed.task("vectors").changed == 2


async def test_the_repair_drains_a_backlog_larger_than_one_batch(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """A batch is the unit of work; the run is responsible for the backlog."""
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)

    stranded = []
    for i in range(5):
        stranded += await stranded_passages(
            pool, org_id=org, collection_id=collection,
            text=f"Passage {i} explains why Helios exists at all.",
        )

    report = await Maintenance(
        pool, org_id=org, batch=2,
        embed=lambda texts: [raw_vec(3) for _ in texts],
    ).run(tasks=["vectors"])

    rows = await vectors_of(pool, stranded)
    assert all(row["embedding"] is not None for row in rows.values())
    assert report.task("vectors").changed == 5


async def test_a_second_run_neither_repeats_the_work_nor_overwrites_it(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """Idempotence, in the form a crontab produces it: the tick after.

    A vectored row is not a candidate, so the second run neither pays the
    embedder again nor reports work it did not do.  The second run embeds to a
    different coordinate, so a repair that re-embedded what it found would move
    the first vector and say nothing.
    """
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)

    stranded = await stranded_passages(
        pool, org_id=org, collection_id=collection,
        text="Helios is the ledger behind interbank settlement.",
    )

    first = await Maintenance(
        pool, org_id=org, embed=lambda texts: [raw_vec(3) for _ in texts]
    ).run(tasks=["vectors"])
    written = (await vectors_of(pool, stranded))[stranded[0]]["embedding"]

    second = await Maintenance(
        pool, org_id=org, embed=lambda texts: [raw_vec(7) for _ in texts]
    ).run(tasks=["vectors"])

    unchanged = (await vectors_of(pool, stranded))[stranded[0]]["embedding"]
    assert unchanged.to_list() == written.to_list()
    assert first.task("vectors").changed == 1
    assert second.task("vectors").scanned == 0
    assert second.task("vectors").changed == 0


async def test_a_vector_written_mid_batch_by_another_writer_survives(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """The overlap the advisory lock does not cover, and the silent one.

    The lock keeps two maintenance runs apart; it says nothing about the ingest
    path, which vectors its own stranded rows after the transaction that
    promoted them.  If that lands between this run's select and its write, the
    row already carries the vector computed from its content — and this run's
    is computed from the same content, so overwriting it is not wrong so much
    as unaccountable: `changed` would count a row nothing changed.
    """
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)

    stranded = await stranded_passages(
        pool, org_id=org, collection_id=collection,
        text="Helios is the ledger behind interbank settlement.",
    )

    class VectoredMidBatch(Maintenance):
        async def _unvectored(self, conn, generation_id):
            rows = await super()._unvectored(conn, generation_id)
            await conn.execute(
                "UPDATE chunks SET embedding = $1::halfvec "
                "WHERE id = ANY($2::uuid[])",
                HalfVector(raw_vec(9)),
                [row["chunk_id"] for row in rows],
            )
            return rows

    report = await VectoredMidBatch(
        pool, org_id=org, embed=lambda texts: [raw_vec(3) for _ in texts]
    ).run(tasks=["vectors"])

    rows = await vectors_of(pool, stranded)
    assert rows[stranded[0]]["embedding"].to_list() == raw_vec(9)
    assert report.task("vectors").changed == 0


async def test_an_embedder_that_returns_the_wrong_count_stops_the_run(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """The one way this task could write a wrong vector rather than none.

    Chunks and vectors are matched by position, so an embedder that drops one
    would shift every vector after it onto the wrong passage — content-addressed
    rows carrying vectors computed from someone else's content, which no later
    run would ever look at again because they are no longer NULL.  Failing the
    run is the only honest answer.
    """
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)

    stranded = await stranded_passages(
        pool, org_id=org, collection_id=collection,
        text="Helios is the ledger behind interbank settlement.",
    )

    maintenance = Maintenance(
        pool, org_id=org, embed=lambda texts: [raw_vec(3) for _ in texts[:-1]]
    )

    with pytest.raises(RuntimeError, match="0 vectors for 1 passages"):
        await maintenance.run(tasks=["vectors"])

    rows = await vectors_of(pool, stranded)
    assert rows[stranded[0]]["embedding"] is None


async def test_a_repair_that_overlaps_another_declines_rather_than_re_embedding(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """The cron-overlap case on the one task that spends money per row."""
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)

    stranded = await stranded_passages(
        pool, org_id=org, collection_id=collection,
        text="Helios is the ledger behind interbank settlement.",
    )

    def never(texts: list[str]) -> list[list[float]]:
        raise AssertionError("a declined task must not call the embedder")

    holder = await pool.acquire()
    try:
        assert await holder.fetchval(
            "SELECT pgkg_try_maintenance_lock('vectors', $1)", org
        )
        blocked = await Maintenance(pool, org_id=org, embed=never).run(
            tasks=["vectors"]
        )
    finally:
        await holder.execute(
            "SELECT pgkg_release_maintenance_lock('vectors', $1)", org
        )
        await pool.release(holder)

    assert blocked.task("vectors").ran is False
    assert blocked.task("vectors").changed == 0
    rows = await vectors_of(pool, stranded)
    assert rows[stranded[0]]["embedding"] is None


async def test_the_repair_leaves_another_tenants_stranded_rows_alone(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """A maintenance run belongs to a tenant, and so does its embedder bill."""
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        mine = await new_org(conn)
        theirs = await new_org(conn)
        my_collection = await new_collection(conn, org_id=mine)
        their_collection = await new_collection(conn, org_id=theirs)

    ours = await stranded_passages(
        pool, org_id=mine, collection_id=my_collection,
        text="Helios is the ledger behind interbank settlement.",
    )
    others = await stranded_passages(
        pool, org_id=theirs, collection_id=their_collection,
        text="Selene is the ledger behind interbank settlement.",
    )

    report = await Maintenance(
        pool, org_id=mine, embed=lambda texts: [raw_vec(3) for _ in texts]
    ).run(tasks=["vectors"])

    rows = await vectors_of(pool, ours + others)
    assert rows[ours[0]]["embedding"] is not None
    assert rows[others[0]]["embedding"] is None
    assert report.task("vectors").changed == 1


async def test_an_unreferenced_chunk_is_not_embedded_at_this_orgs_expense(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """`refcount > 0` is the scope, and it is a scope about money.

    A chunk no version links is either provenance — a chat turn, which is not
    a passage (D1) — or content a purge is about to collect.  Neither is
    reachable by the vector arm, so embedding either is a bill for a row
    nothing will ever retrieve.
    """
    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        orphan = await conn.fetchval(
            "INSERT INTO chunks (text, org_id, collection_id) "
            "VALUES ($1, $2, $3) RETURNING id",
            "Nothing links this passage.", org, collection,
        )

    report = await Maintenance(
        pool, org_id=org, embed=lambda texts: [raw_vec(3) for _ in texts]
    ).run(tasks=["vectors"])

    rows = await vectors_of(pool, [orphan])
    assert rows[orphan]["embedding"] is None
    assert report.task("vectors").scanned == 0
    assert report.task("vectors").changed == 0


# ---------------------------------------------------------------------------
# The operator surface
# ---------------------------------------------------------------------------


def test_the_cli_exposes_every_task_by_name() -> None:
    """The entry point an operator puts in a crontab."""
    from pgkg.cli import build_parser
    from pgkg.maintenance import TASKS

    args = build_parser().parse_args(["maintain", "--task", "mentions"])

    assert args.command == "maintain"
    assert args.task == ["mentions"]

    for task in TASKS:
        parsed = build_parser().parse_args(["maintain", "--task", task])
        assert parsed.task == [task]


def test_the_cli_refuses_a_task_that_does_not_exist() -> None:
    from pgkg.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["maintain", "--task", "pagerankk"])


def test_the_command_is_wired_to_the_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure this issue is about, one level up.

    A subcommand that parses and is not dispatched is a command that exits zero
    and does nothing — which is exactly how a sweep nobody called went unnoticed
    for a phase.
    """
    import sys

    from pgkg import cli

    dispatched: list[object] = []
    monkeypatch.setattr(cli, "cmd_maintain", dispatched.append)
    monkeypatch.setattr(sys, "argv", ["pgkg", "maintain", "--task", "expiries"])

    cli.main()

    assert [args.task for args in dispatched] == [["expiries"]]


def test_the_cli_runs_every_task_when_none_is_named() -> None:
    from pgkg.cli import build_parser

    args = build_parser().parse_args(["maintain"])

    assert args.task == []


async def test_the_cli_command_runs_the_selected_task_and_prints_its_report(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch, capsys
) -> None:
    """The whole operator path, from argv to stdout.

    Everything above drives `Maintenance`; this drives the command an operator
    installs, because the defect this issue is about was never in the machinery
    — it was that nothing called it.
    """
    import contextlib
    import json

    from pgkg import db
    from pgkg.cli import build_parser, run_maintain

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        await conn.execute(
            "INSERT INTO entities (name, type, org_id) VALUES ('Helios', "
            "'concept', $1)",
            org,
        )
        await conn.execute(
            "INSERT INTO chunks (text, org_id, collection_id) VALUES ($1, $2, $3)",
            "Helios is the ledger behind settlement.", org, collection,
        )

    @contextlib.asynccontextmanager
    async def _test_pool():
        yield pool

    offline.setattr(db, "pool_from_settings", _test_pool)

    args = build_parser().parse_args(
        ["maintain", "--task", "mentions", "--org", str(org)]
    )
    await run_maintain(args)

    printed = json.loads(capsys.readouterr().out)

    assert printed["org"] == str(org)
    assert printed["tasks"] == [
        {"task": "mentions", "ran": True, "scanned": 2, "changed": 1}
    ]


async def test_the_report_serialises_to_the_json_the_cli_prints(
    pool: asyncpg.Pool, offline: pytest.MonkeyPatch
) -> None:
    """A cron entry's output is read by a log scraper, so the shape is API."""
    import json

    from pgkg.maintenance import Maintenance

    async with pool.acquire() as conn:
        org = await new_org(conn)

    report = await Maintenance(pool, org_id=org).run(tasks=["expiries"])
    payload = json.loads(json.dumps(report.as_dict()))

    assert payload["org"] == str(org)
    assert payload["tasks"] == [
        {"task": "expiries", "ran": True, "scanned": None, "changed": 0}
    ]
