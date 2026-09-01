"""The scheduled path: one entry point for the jobs nothing was running.

Four jobs existed in the schema, were tested, and were reachable only from a
test or an operator's psql session.  The consequence was not theoretical: the
gazetteer populates `entity_mentions`, ADR 0001 D2 makes that edge the answer to
"is a corpus-graph relationship worth having", and because nothing called it the
table was empty in every deployment (issue #19).  `pgkg_recompute_pagerank()`,
`pgkg_contradict()` and `pgkg_expire_due()` are recorded as built-and-unscheduled
in docs/adrs/0001-implementation-notes.md §4 for the same reason.

One entry point rather than four, because the thing an operator actually
installs is a crontab line, and four of them is three chances to forget one.
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
under an IS NULL predicate, and the contradiction candidate query takes its rows
FOR UPDATE SKIP LOCKED — the lock is there so that the numbers a scheduler reads
mean what they say.

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

from pgkg.config import DEFAULT_ORG_ID, ORG_GUC
from pgkg.gazetteer import Gazetteer

# The four jobs, in the order a run performs them.  Mentions first because it is
# the one with a customer-visible consequence; pagerank after it, since a sweep
# writes no entity and no edge, so the ordering costs nothing either way.
TASKS = ("mentions", "pagerank", "contradictions", "expiries")

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


@dataclass(frozen=True)
class TaskReport:
    """What one task did, in the three facts a scheduler acts on.

    `ran` is False only when another run held the lock, which is a normal
    outcome and not a failure.  `scanned` is the work in the unit the task works
    in — passages and names for the sweep, subgraphs for pagerank, candidate
    claims for contradictions — and None where the job cannot honestly report
    one: `pgkg_expire_due()` knows what it withdrew and not what it looked at.
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
