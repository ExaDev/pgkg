"""The scheduled path: one entry point for the jobs nothing was running.

Four jobs existed in the schema, were tested, and were reachable only from a
test or an operator's psql session.  The consequence was not theoretical: the
gazetteer populates `entity_mentions`, ADR 0001 D2 makes that edge the answer to
"is a corpus-graph relationship worth having", and because nothing called it the
table was empty in every deployment (issue #19).  `pgkg_recompute_pagerank()`,
`pgkg_contradict()` and `pgkg_expire_due()` are recorded as built-and-unscheduled
in docs/adrs/0001-implementation-notes.md §4 for the same reason.

The fifth job is here for the other half of the same reason.  A corpus ingest
can leave a live passage with `embedding IS NULL` — the window between phase 2
deciding a vector already existed at an address and phase 3 creating a new row
there, and the promoted-then-repaired ordering that a killed process interrupts
— and the next crawl of that document short-circuits on the unchanged hash
before it reaches a chunk.  So nothing revisits the row: it stays retrievable by
the keyword arm, invisible to the vector arm and to MMR, silently and for good
(issue #22).  Unlike the other four this one spends money, which is why it is
scoped to the rows a version still links and to the generation this process can
actually embed for.

One entry point rather than five, because the thing an operator actually
installs is a crontab line, and five of them is four chances to forget one.
Three properties make it safe to install:

*Selectable.*  A pagerank pass and a mention sweep have nothing to do with each
other.  Each task runs on its own, so an operator debugging one does not have to
run the others, and so a deployment can give them different intervals — the
sweep on a timer of minutes, pagerank nightly.

*Reporting.*  Every task says whether it ran and what it did.  "Ran and found
nothing" and "declined because another run holds it" are different facts and a
scheduler's log has to be able to tell them apart.

*Overlap-safe.*  A cron entry that overlaps itself is the normal failure mode of
anything on a timer, and it is the one this module is built against: each task
takes an advisory lock per (task, org) and reports `ran=False` rather than
repeating work someone else is doing.  Nothing here needs the lock to be
correct — the mention insert is ON CONFLICT DO NOTHING, both watermarks are set
under an IS NULL predicate, the contradiction candidate query takes its rows
FOR UPDATE SKIP LOCKED, and the vector write is guarded by the same IS NULL it
selected on — the lock is there so that the numbers a scheduler reads mean what
they say, and so that two ticks do not pay an embedder for the same batch.

Why the sweep and not an inline call.  D7 rules out both online placements: a
corpus ingest must not hold a pooled connection across a cross-product against
every name the org knows, and a chat ingest must not match one new name against
an unbounded corpus on the request path.  `MatchResult.chunks_scanned` was
written for the timer — "a settled corpus drives the first to zero, which is
what makes the sweep re-runnable" — and that is what this schedules.

An inline `match_chunks()` after a corpus version is promoted was considered as a
latency optimisation and rejected: it is not what makes the edge exist, so
nothing may depend on it, and what it would add is a cross-product against every
name the org knows on every changed document of a nightly crawl plus a
best-effort call whose failures are swallowed on the write path.  The freshness
it buys is bounded by an interval the operator already controls — the sweep is
cheap enough to run every few minutes.  Recorded in
docs/adrs/0001-implementation-notes.md §5 so the question does not have to be
reopened from scratch.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import UUID

import asyncpg
from pgvector import HalfVector

from pgkg import ml
from pgkg.config import DEFAULT_ORG_ID, ORG_GUC
from pgkg.corpus import EmbedFn
from pgkg.gazetteer import Gazetteer

# The five jobs, in the order a run performs them.  Mentions first because it is
# the one with a customer-visible consequence, and vectors second because it is
# the other one: a passage with no vector is retrievable by half the retriever.
# Pagerank after both, since neither writes an entity or an edge, so the
# ordering costs nothing either way.
TASKS = ("mentions", "vectors", "pagerank", "contradictions", "expiries")

# One batch of a sweep, not a whole corpus.  Matches the gazetteer's own default
# so an operator who changes neither gets the same unit of work everywhere.
DEFAULT_BATCH = 1000

DEFAULT_ITERATIONS = 20
DEFAULT_DAMPING = 0.85

# How many batches one run may drain per direction before it concludes it is not
# making progress.  A drain loop's stop condition is the watermark, and a
# watermark that stops advancing — a trigger dropped, a policy that hides the
# stamp from the role doing the sweep — turns a nightly job into a process that
# never exits and a table that grows nothing.  Reached only in that case: at the
# default batch this is ten million rows in one direction of one run.
DEFAULT_MAX_BATCHES = 10_000

_SET_ORG_SQL = f"SELECT set_config('{ORG_GUC}', $1, false)"
_TRY_LOCK_SQL = "SELECT pgkg_try_maintenance_lock($1, $2)"
_RELEASE_LOCK_SQL = "SELECT pgkg_release_maintenance_lock($1, $2)"

# Which subgraphs this org has.  A namespace is not a flag an operator should
# have to remember: the rows state which ones exist, and every one of them is a
# subgraph PageRank has to be computed over separately (D3, D4).
_NAMESPACES_SQL = "SELECT DISTINCT namespace FROM entities WHERE org_id = $1"
_PAGERANK_SQL = "SELECT pgkg_recompute_pagerank($1, $2, $3, $4)"
_SCORED_SQL = """
SELECT COUNT(*)
FROM entity_pagerank ep
JOIN entities e ON e.id = ep.entity_id
WHERE e.org_id = $1 AND e.namespace = ANY($2::text[])
"""
_CONTRADICT_SQL = "SELECT considered, closed FROM pgkg_contradict_superseded($1, $2)"
_EXPIRE_SQL = "SELECT pgkg_expire_due($1, $2)"

# How many passages this org has that a crawl left without a vector — all of
# them, not only the ones this run can serve, because a row this process cannot
# embed for is still a row the operator has to be told about.  Served by
# chunks_unvectored_idx (054), so a settled org pays an index probe per tick.
_STRANDED_SQL = """
SELECT COUNT(*) FROM chunks
WHERE org_id = $1 AND embedding IS NULL AND refcount > 0
"""

# Which generation this process embeds in.  Read to decide which stranded rows
# the run may serve, never to stamp one: the row states its own generation and a
# vector computed by another model is not comparable with it (D8).
_PRIMARY_GENERATION_SQL = """
SELECT oe.generation_id FROM org_embedders oe
WHERE oe.org_id = $1 AND oe.role = 'primary'
"""

_UNVECTORED_SQL = (
    "SELECT chunk_id, chunk_text FROM pgkg_unvectored_chunks($1, $2, $3)"
)

# The write, guarded by both facts that made the vector the right one for the
# row.  `embedding IS NULL` so a vector another writer computed in the meantime
# is never overwritten by this one — the reason the task is safe to run twice —
# and the generation so a row cut over between the select and the write is left
# for the model that now owns it rather than filled from the space it just left.
_WRITE_VECTORS_SQL = """
WITH written AS (
    UPDATE chunks c
    SET embedding = e.embedding
    FROM unnest($2::uuid[], $3::halfvec[]) AS e(id, embedding)
    WHERE c.id = e.id
      AND c.embedding IS NULL
      AND c.embedder_generation_id = $1
    RETURNING 1
)
SELECT COUNT(*) FROM written
"""


@dataclass(frozen=True)
class TaskReport:
    """What one task did, in the three facts a scheduler acts on.

    `ran` is False only when another run held the lock, which is a normal
    outcome and not a failure.  `scanned` is the work in the unit the task works
    in — passages and names for the sweep, unvectored passages for the vector
    repair, subgraphs for pagerank, candidate claims for contradictions — and
    None where the job cannot honestly report one: `pgkg_expire_due()` knows
    what it withdrew and not what it looked at.
    `changed` is the yield, and it is the number worth alerting on when it stays
    non-zero for a job that is supposed to settle.
    """

    task: str
    ran: bool
    scanned: int | None = None
    changed: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "ran": self.ran,
            "scanned": self.scanned,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class MaintenanceReport:
    """One run, as a crontab's output.

    The shape is API: a cron entry's stdout is read by a log scraper, and a
    scraper that has to parse prose is a scraper that breaks on a reworded
    docstring.
    """

    org_id: UUID
    tasks: tuple[TaskReport, ...]

    def task(self, name: str) -> TaskReport:
        for report in self.tasks:
            if report.task == name:
                return report
        raise KeyError(f"{name} did not run in this report")

    @property
    def changed(self) -> int:
        return sum(report.changed for report in self.tasks)

    def as_dict(self) -> dict[str, object]:
        return {
            "org": str(self.org_id),
            "tasks": [report.as_dict() for report in self.tasks],
        }


class Maintenance:
    """One org's scheduled jobs.

    Tenancy is bound to the object rather than passed per call, as it is on
    Memory, CorpusIngest and Gazetteer: a maintenance run belongs to a tenant —
    it withdraws that tenant's expired claims and rescores that tenant's graph —
    and a default argument cannot fail loudly.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        org_id: UUID = DEFAULT_ORG_ID,
        namespace: str | None = None,
        batch: int = DEFAULT_BATCH,
        iterations: int = DEFAULT_ITERATIONS,
        damping: float = DEFAULT_DAMPING,
        max_batches: int = DEFAULT_MAX_BATCHES,
        gazetteer: Gazetteer | None = None,
        embed: EmbedFn | None = None,
    ) -> None:
        if batch < 1:
            raise ValueError("a batch of no rows would never make progress")
        if max_batches < 1:
            raise ValueError("a drain of no batches would never do any work")
        self._pool = pool
        self._org_id = org_id
        self._namespace = namespace
        self._batch = batch
        self._iterations = iterations
        self._damping = damping
        self._max_batches = max_batches
        self._embed = embed
        # An injected gazetteer already pointed at this org is used as it is;
        # one pointed elsewhere is re-pointed, which is what for_org is for.
        # Re-pointing unconditionally would quietly replace a caller's own
        # object with a plain Gazetteer.
        if gazetteer is None:
            self._gazetteer = Gazetteer(pool, org_id=org_id)
        elif gazetteer.org_id == org_id:
            self._gazetteer = gazetteer
        else:
            self._gazetteer = gazetteer.for_org(org_id)

    @property
    def org_id(self) -> UUID:
        return self._org_id

    def for_org(self, org_id: UUID) -> Maintenance:
        return Maintenance(
            self._pool,
            org_id=org_id,
            namespace=self._namespace,
            batch=self._batch,
            iterations=self._iterations,
            damping=self._damping,
            max_batches=self._max_batches,
            gazetteer=self._gazetteer,
            embed=self._embed,
        )

    async def run(self, tasks: Iterable[str] | None = None) -> MaintenanceReport:
        """Run the named tasks, or all of them, and report on each.

        Sequentially, and every name validated before the first one runs: a
        typo in a crontab must not be a silent no-op, and must not leave half a
        run behind either.
        """
        selected = self._selection(tasks)
        return MaintenanceReport(
            org_id=self._org_id,
            tasks=tuple([await self.run_task(task) for task in selected]),
        )

    async def run_task(self, task: str) -> TaskReport:
        """One task, under its own lock."""
        if task not in TASKS:
            raise ValueError(f"unknown maintenance task {task!r}; expected {TASKS}")

        async with self._locked(task) as conn:
            if conn is None:
                return TaskReport(task=task, ran=False)
            scanned, changed = await self._runners()[task](conn)
            return TaskReport(
                task=task, ran=True, scanned=scanned, changed=changed
            )

    def _runners(self):
        """Task name to the coroutine that performs it.

        A mapping rather than a name built into a getattr: a name in TASKS with
        nothing behind it fails here, loudly, rather than at whatever hour the
        crontab first selects it.
        """
        return {
            "mentions": self._mentions,
            "vectors": self._vectors,
            "pagerank": self._pagerank,
            "contradictions": self._contradictions,
            "expiries": self._expiries,
        }

    def _selection(self, tasks: Iterable[str] | None) -> Sequence[str]:
        if tasks is None:
            return TASKS
        selected = list(tasks)
        if not selected:
            return TASKS
        unknown = [task for task in selected if task not in TASKS]
        if unknown:
            raise ValueError(
                f"unknown maintenance task(s) {unknown}; expected {TASKS}"
            )
        return selected

    @asynccontextmanager
    async def _locked(self, task: str):
        """The connection this task runs on, or None if another run holds it.

        One connection for the whole task, held for as long as the lock is: a
        second connection acquired inside the work would be two pool slots for
        one job, which is the thing D7 rules out on the batch path.  The lock is
        released explicitly rather than left to the pool's own reset, so a run
        that finishes early does not keep the next tick out for as long as the
        connection happens to be idle.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(_SET_ORG_SQL, str(self._org_id))
            if not await conn.fetchval(_TRY_LOCK_SQL, task, self._org_id):
                yield None
                return
            try:
                yield conn
            finally:
                await conn.fetchval(_RELEASE_LOCK_SQL, task, self._org_id)

    async def _mentions(self, conn: asyncpg.Connection) -> tuple[int, int]:
        """Both directions of the gazetteer, drained a batch at a time.

        Both, because neither is reachable from the other: a passage stamped by
        an earlier sweep never meets a name created afterwards, and that is the
        common order in steady state (migration 053).  Draining rather than one
        batch per tick because the watermarks guarantee progress — every batch
        stamps what it read — so the loop ends, and a backlog that only shrinks
        by one batch per tick is a backlog that outlives the corpus.
        """
        scanned = changed = 0
        for sweep in (self._sweep_passages, self._sweep_names):
            drained = False
            for _ in range(self._max_batches):
                result = await sweep(conn)
                scanned += result.chunks_scanned
                changed += result.mentions_added
                if result.chunks_scanned == 0:
                    drained = True
                    break
            if not drained:
                raise RuntimeError(
                    f"mentions sweep {sweep.__name__} made no progress in "
                    f"{self._max_batches} batches of {self._batch}: the "
                    "watermark it drains against is not advancing"
                )
        return scanned, changed

    async def _sweep_passages(self, conn: asyncpg.Connection):
        return await self._gazetteer.sweep(limit=self._batch, conn=conn)

    async def _sweep_names(self, conn: asyncpg.Connection):
        return await self._gazetteer.sweep_entities(
            limit=self._batch, max_chunks=self._batch, conn=conn
        )

    async def _vectors(self, conn: asyncpg.Connection) -> tuple[int, int]:
        """Passages a crawl left without a vector, embedded a batch at a time.

        The ingest path pays for its own stranded rows immediately after the
        transaction that created them (#16), which covers every run that gets
        that far.  This is the backstop for the ones that do not: the version is
        promoted and committed first, so an embedder that fails, a process that
        is killed or a connection that drops in between leaves the row committed
        with `embedding IS NULL` — and the next crawl of that document
        short-circuits on the unchanged hash before it ever reaches a chunk.
        Nothing else ever revisits it, which is what makes a small window a
        permanent one (#22).

        `scanned` is every stranded row this org has, not only the ones this run
        can serve, because a row this process cannot embed for is exactly the
        row that must not go unmentioned again — an operator reading
        `scanned: 3, changed: 0` every tick is reading the backlog this task
        exists to make visible.

        One generation, this org's primary, and never a restamp.  D8 makes
        vectors from two generations incomparable, and `chunks.embedding` is the
        single inline column the primary owns, so the rows a stranded backlog is
        made of during a cutover — the outgoing generation's — are rows this
        process has no model for.  Filling them from the incoming space, or
        relabelling them so that it fits, would restate the defect being fixed
        in a form nothing can detect afterwards.  They are counted and left.
        """
        stranded = await conn.fetchval(_STRANDED_SQL, self._org_id)
        generation = await conn.fetchval(_PRIMARY_GENERATION_SQL, self._org_id)
        if not stranded or generation is None:
            return stranded, 0

        changed = 0
        # No "made no progress" failure here, unlike the mention sweep: progress
        # is proved by the write itself rather than by a watermark that could
        # stall, so a batch that writes nothing means another writer got there
        # first and there is nothing to raise about.  The cap bounds one run,
        # not the backlog — what is left is still there for the next tick.
        for _ in range(self._max_batches):
            rows = await self._unvectored(conn, generation)
            if not rows:
                break
            written = await self._vectorise(conn, generation, rows)
            changed += written
            if written == 0 or len(rows) < self._batch:
                break
        return stranded, changed

    async def _unvectored(
        self, conn: asyncpg.Connection, generation_id: UUID
    ) -> list[asyncpg.Record]:
        return await conn.fetch(
            _UNVECTORED_SQL, self._org_id, generation_id, self._batch
        )

    async def _vectorise(
        self,
        conn: asyncpg.Connection,
        generation_id: UUID,
        rows: Sequence[asyncpg.Record],
    ) -> int:
        """Embed one batch and write it, as one statement of its own.

        A batch is the unit of work AND the unit of durability: no transaction
        spans two of them, so a run that dies on the third batch leaves the
        first two vectored rather than rolling back an hour of embedder spend.

        The pooled connection is held across the model call, which the ingest
        path may not do (D7).  It is not the same trade: the connection carries
        the advisory lock this task runs under, so releasing it for the duration
        of the embedder would release the lock and let the next tick embed the
        same batch.  A background job holding one slot per tenant is the price
        of the lock meaning anything, and the batch bounds how long it is held.
        """
        texts = [row["chunk_text"] for row in rows]
        vectors = self._embed_texts(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"the embedder returned {len(vectors)} vectors for "
                f"{len(texts)} passages: a vector is only ever written against "
                "the content it was computed from"
            )
        return await conn.fetchval(
            _WRITE_VECTORS_SQL,
            generation_id,
            [row["chunk_id"] for row in rows],
            [HalfVector(vector) for vector in vectors],
        )

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Resolved at call time so a spy on ml.embed is a spy on this path."""
        if not texts:
            return []
        if self._embed is not None:
            return self._embed(list(texts))
        return ml.embed(list(texts))

    async def _pagerank(self, conn: asyncpg.Connection) -> tuple[int, int]:
        """One PageRank pass per subgraph this org has entities in.

        The graph arm of retrieval reads `entity_pagerank`, and nothing was
        recomputing it, so a deployment's scores were whatever the last manual
        run left — or absent, which is a silent zero in the ranking.
        """
        namespaces = (
            [self._namespace]
            if self._namespace is not None
            else [
                row["namespace"]
                for row in await conn.fetch(_NAMESPACES_SQL, self._org_id)
            ]
        )
        if not namespaces:
            return 0, 0
        for namespace in namespaces:
            await conn.execute(
                _PAGERANK_SQL,
                namespace, self._iterations, self._damping, self._org_id,
            )
        scored = await conn.fetchval(_SCORED_SQL, self._org_id, namespaces)
        return len(namespaces), scored

    async def _contradictions(self, conn: asyncpg.Connection) -> tuple[int, int]:
        """Supersessions whose validity interval nobody closed.

        Drained like the sweep, but the stop condition is different: there is no
        watermark, so a batch that closed nothing is the only proof that another
        batch would close nothing either.
        """
        scanned = changed = 0
        while True:
            row = await conn.fetchrow(_CONTRADICT_SQL, self._org_id, self._batch)
            scanned += row["considered"]
            changed += row["closed"]
            if row["closed"] == 0 or row["considered"] < self._batch:
                break
        return scanned, changed

    async def _expiries(self, conn: asyncpg.Connection) -> tuple[int | None, int]:
        """The TTL sweep, for this org only.

        No scanned count: the function reports what it withdrew, and counting
        what was due in a separate statement would report a different instant's
        answer as if it were this one's.
        """
        withdrawn = await conn.fetchval(
            _EXPIRE_SQL, self._namespace, self._org_id
        )
        return None, withdrawn
