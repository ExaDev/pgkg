"""The corpus ingest queue, and the workers that drain it.

Corpus ingest is batch (ADR 0001, D7): 600k chunks is one to two GPU-hours, or
a day-plus on CPU, and it must not compete with online recall for pool slots.
That is what this module is for.  A connector enqueues everything it can see —
idempotently, by content hash, so a nightly full crawl adds no work it already
has — and a worker drains the queue under a slot budget, at whatever rate the
operator allows.

Three properties are load bearing:

*Resumable.*  A worker that dies leaves its job in `running`, and nothing about
that row distinguishes it from a worker still working.  The heartbeat does, so
claiming a job reclaims one whose lease has lapsed.  No sweeper, no flag to get
out of step with reality.

*Observable.*  "Is my corpus indexed yet" is the first question a customer
asks, and per-document progress is the only honest answer.  Progress is
reported on a connection of its own, because progress written inside the ingest
transaction is invisible until commit — which is exactly when it stops being
progress.

*Throttleable.*  `slots` caps how many documents are in flight, and therefore
how many pool connections this pipeline can hold at once; `throttle_seconds`
puts a floor under the interval between them.  Both exist so that an operator
can slow a backfill down without stopping it, which is the thing an operator
actually wants at three in the morning.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

import asyncpg

from pgkg.config import DEFAULT_ORG_ID, ORG_GUC
from pgkg.corpus import (
    CorpusIngestResult,
    Provenance,
    document_hash,
    dump_provenance,
    load_provenance,
)

DEFAULT_LEASE_SECONDS = 300.0
DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class IngestJob:
    """One document, as the queue holds it.

    The text travels with the job: a connector that has already read a document
    should not have to still be alive when a worker reaches it.
    """

    id: UUID
    org_id: UUID
    collection_id: UUID
    external_id: str
    text: str
    content_hash: bytes
    uri: str | None
    attempts: int
    # What the connector knew about the document travels with it too (D5).  A
    # queue that carried only the bytes made the worker's clock the world-time
    # of every queued document, so a perishable article published in 2015 and
    # crawled tonight decayed from tonight.
    source: str | None = None
    asserted_at: datetime | None = None
    provenance: Provenance | None = None


@dataclass(frozen=True)
class JobState:
    """What an operator — or a customer — can see about one document."""

    status: str
    attempts: int
    chunks_total: int | None
    chunks_embedded: int | None
    document_id: UUID | None
    version_id: UUID | None
    error: str | None
    enqueued_at: datetime
    finished_at: datetime | None


class CorpusPipeline(Protocol):
    """What a worker needs of an ingest pipeline.

    Narrow on purpose: the worker owns claiming, progress, retries and the slot
    budget, and knows nothing about chunking or embedding.
    """

    def for_collection(
        self, *, org_id: UUID, collection_id: UUID
    ) -> CorpusPipeline: ...

    async def upsert_document(
        self,
        *,
        external_id: str,
        text: str,
        uri: str | None = ...,
        source: str | None = ...,
        asserted_at: datetime | None = ...,
        provenance: Provenance | None = ...,
        on_progress: Callable[[int, int], Awaitable[None]] | None = ...,
    ) -> CorpusIngestResult: ...


_ENQUEUE_SQL = (
    "SELECT pgkg_enqueue_ingest_job($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)"
)

_CLAIM_SQL = """
SELECT job_id, org_id, collection_id, external_id, uri, payload, content_hash,
       attempts, source, asserted_at, provenance
FROM pgkg_claim_ingest_job($1, make_interval(secs => $2))
"""

_PROGRESS_SQL = "SELECT pgkg_report_ingest_progress($1, $2, $3)"

_FINISH_SQL = "SELECT pgkg_finish_ingest_job($1, $2, $3, $4, $5)"

_FAIL_SQL = "SELECT pgkg_fail_ingest_job($1, $2, $3)"

_SET_ORG_SQL = f"SELECT set_config('{ORG_GUC}', $1, false)"

# A job UUID is a handle, not an authorisation: the org is stated in the read
# rather than left to the connection, because a status carries the document id,
# the attempt count and the extractor's error text (ADR 0001, D3).  The GUC is
# set as well, so the policy on ingest_jobs has something to compare against
# when the deployment connects as pgkg_app.
_STATE_SQL = """
SELECT status, attempts, chunks_total, chunks_embedded, document_id,
       version_id, error, enqueued_at, finished_at
FROM ingest_jobs WHERE id = $1 AND org_id = $2
"""


async def enqueue_document(
    pool: asyncpg.Pool,
    *,
    org_id: UUID,
    collection_id: UUID,
    external_id: str,
    text: str,
    uri: str | None = None,
    source: str | None = None,
    asserted_at: datetime | None = None,
    provenance: Provenance | None = None,
) -> UUID:
    """Offer one document to the queue, returning the job that holds it.

    Idempotent by content hash: a re-crawl of unchanged content gets a handle to
    the work already queued rather than a second copy of it.
    """
    async with pool.acquire() as conn:
        return await conn.fetchval(
            _ENQUEUE_SQL,
            org_id,
            collection_id,
            external_id,
            document_hash(text),
            text,
            uri,
            source,
            asserted_at,
            dump_provenance(provenance),
        )


async def claim_job(
    pool: asyncpg.Pool,
    *,
    org_id: UUID | None = None,
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
) -> IngestJob | None:
    """Take the oldest claimable job, or None when there is nothing to do."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(_CLAIM_SQL, org_id, lease_seconds)
    if row is None:
        return None
    return IngestJob(
        id=row["job_id"],
        org_id=row["org_id"],
        collection_id=row["collection_id"],
        external_id=row["external_id"],
        text=row["payload"],
        content_hash=row["content_hash"],
        uri=row["uri"],
        attempts=row["attempts"],
        source=row["source"],
        asserted_at=row["asserted_at"],
        provenance=load_provenance(row["provenance"]),
    )


async def job_state(
    pool: asyncpg.Pool, job_id: UUID, *, org_id: UUID = DEFAULT_ORG_ID
) -> JobState:
    """One job's status, as the org that enqueued it may see it.

    A job belonging to another org is not found rather than reported: the caller
    holding the UUID has no relationship to it, and "no such job" is the honest
    answer to a question it was not entitled to ask.
    """
    async with pool.acquire() as conn:
        await conn.execute(_SET_ORG_SQL, str(org_id))
        row = await conn.fetchrow(_STATE_SQL, job_id, org_id)
    if row is None:
        raise KeyError(f"no such ingest job {job_id}")
    return JobState(
        status=row["status"],
        attempts=row["attempts"],
        chunks_total=row["chunks_total"],
        chunks_embedded=row["chunks_embedded"],
        document_id=row["document_id"],
        version_id=row["version_id"],
        error=row["error"],
        enqueued_at=row["enqueued_at"],
        finished_at=row["finished_at"],
    )


class IngestWorker:
    """Drains the queue through a corpus pipeline, one slot at a time."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        ingest: CorpusPipeline,
        org_id: UUID | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        throttle_seconds: float = 0.0,
        slots: int = 1,
    ) -> None:
        if slots < 1:
            raise ValueError("a worker with no slots would never do any work")
        self._pool = pool
        self._ingest = ingest
        self._org_id = org_id
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._throttle_seconds = throttle_seconds
        self._slots = asyncio.Semaphore(slots)

    async def run(self, *, concurrency: int = 1, max_jobs: int | None = None) -> int:
        """Drain until the queue is empty, returning how many jobs were run.

        `concurrency` is how many loops ask for work; `slots` is how many of
        them may be inside a document at once.  They are separate numbers
        because the cost being rationed is connections held, not tasks alive.
        """
        if concurrency < 1:
            raise ValueError("a drain with no loops would never claim anything")
        remaining = Budget(max_jobs)
        counts = await asyncio.gather(
            *(self._drain(remaining) for _ in range(concurrency))
        )
        return sum(counts)

    async def _drain(self, remaining: Budget) -> int:
        done = 0
        while remaining.take():
            job = await self.run_once()
            if job is None:
                remaining.give()
                return done
            done += 1
            if self._throttle_seconds:
                await asyncio.sleep(self._throttle_seconds)
        return done

    async def run_once(self) -> IngestJob | None:
        """Claim one job and see it through, or return None if none is due."""
        job = await claim_job(
            self._pool, org_id=self._org_id, lease_seconds=self._lease_seconds
        )
        if job is None:
            return None

        async with self._slots:
            try:
                result = await self._process(job)
            except Exception as failure:  # noqa: BLE001 — recorded, then retried
                await self._fail(job, failure)
                return job
            await self._finish(job, result)
        return job

    async def _process(self, job: IngestJob) -> CorpusIngestResult:
        pipeline = self._ingest.for_collection(
            org_id=job.org_id, collection_id=job.collection_id
        )
        return await pipeline.upsert_document(
            external_id=job.external_id,
            text=job.text,
            uri=job.uri,
            source=job.source,
            asserted_at=job.asserted_at,
            provenance=job.provenance,
            on_progress=self._progress_reporter(job),
        )

    @asynccontextmanager
    async def _job_connection(self, job: IngestJob):
        """A connection carrying the org of the job being worked on.

        A worker draining several orgs runs as an owner or sets the org per
        claim (033's policy comment); it knows which org each job belongs to, so
        it sets it, and the queue's own isolation policy applies to the writes
        the worker makes on that job's behalf.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(_SET_ORG_SQL, str(job.org_id))
            yield conn

    def _progress_reporter(
        self, job: IngestJob
    ) -> Callable[[int, int], Awaitable[None]]:
        async def report(chunks_total: int, chunks_embedded: int) -> None:
            # A connection of its own, so the numbers are visible while the
            # document that produced them is still being written.  The pipeline
            # calls this between its phases rather than from inside its ingest
            # transaction, so this acquire is never a second connection held on
            # top of one — which at slots == pool size would never be granted.
            async with self._job_connection(job) as conn:
                await conn.execute(
                    _PROGRESS_SQL, job.id, chunks_total, chunks_embedded
                )

        return report

    async def _finish(self, job: IngestJob, result: CorpusIngestResult) -> None:
        async with self._job_connection(job) as conn:
            await conn.execute(
                _FINISH_SQL,
                job.id,
                result.document_id,
                result.version_id,
                result.chunks_total,
                result.chunks_new + result.chunks_carried,
            )

    async def _fail(self, job: IngestJob, failure: Exception) -> None:
        # The usual reason a corpus job dies is transient — an embedder restart,
        # a dropped connection — so the queue keeps the error and tries again
        # until the attempt budget is spent.
        async with self._job_connection(job) as conn:
            await conn.execute(
                _FAIL_SQL, job.id, f"{type(failure).__name__}: {failure}",
                self._max_attempts,
            )


class Budget:
    """How many more jobs the drain may run, shared by its loops."""

    def __init__(self, limit: int | None) -> None:
        self._remaining = limit

    def take(self) -> bool:
        if self._remaining is None:
            return True
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        return True

    def give(self) -> None:
        if self._remaining is not None:
            self._remaining += 1
