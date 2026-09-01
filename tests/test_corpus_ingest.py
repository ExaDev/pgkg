"""The batch corpus ingest pipeline: hash first, embed only what changed.

Two pipelines, not one (ADR 0001, D7).  Chat ingest is online and embeds
everything it is given.  Corpus ingest is a nightly full crawl of a 100k
document corpus, so the only affordable answer to "has this changed?" is a
hash — one for the document, which makes an unchanged crawl free, and one per
chunk, which makes a typo fixed in a 300 page handbook one embedding call
rather than 300.  Every test here is about work NOT done: the assertions are on
a spy standing in for the embedder, because the cost this pipeline exists to
avoid is invisible in the rows it writes.

The embedding cache (D4) shares the expensive half of that work between tenants
without sharing a single row, and is restricted to `public_source` collections
because a content-hash cache is probe-able: on a confidential document a cache
hit would confirm another tenant holds it.
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid

import asyncpg
import pytest

SYSTEM_ORG = uuid.UUID("00000000-0000-0000-0000-000000000000")

WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")

CHUNK_CAP = 300


def document_text(
    words: tuple[str, ...] = WORDS, edited: str | None = None
) -> str:
    """A document whose chunks are distinct and content-defined.

    Each section names itself, so no two chunks share content: repeated content
    inside one document is one chunk row by design, and a test that leaned on
    that would be measuring deduplication rather than carry-over.  `edited`
    rewrites one section, which is the local edit the whole pipeline is built
    around — every other chunk hashes to what it hashed to before.
    """
    return "\n\n".join(
        f"## Section {word}\n\nThe {word} chapter explains how the {word} "
        f"subsystem {'misbehaves' if word == edited else 'behaves'} when the "
        f"operator asks it to reconcile a ledger, and it names {word} "
        f"explicitly so no two chapters share text."
        for word in WORDS
        if word in words
    )


def sha256(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class EmbedSpy:
    """Stands in for ml.embed and counts what it was asked to embed.

    Deterministic per text so a cached vector and a freshly computed one are
    comparable: the interesting assertion is that a second tenant got the
    right vector without paying for it.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    @property
    def texts(self) -> list[str]:
        return [text for call in self.calls for text in call]

    def vector_for(self, text: str) -> list[float]:
        digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
        seed = int.from_bytes(digest, "big")
        return [float((seed >> (i % 40)) % 97) / 97.0 for i in range(self.dim)]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector_for(text) for text in texts]


async def embedding_dim(conn: asyncpg.Connection) -> int:
    return await conn.fetchval(
        "SELECT pgkg_embedding_dim('propositions', 'embedding')"
    )


async def new_org(conn: asyncpg.Connection) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("org")
    )


async def new_collection(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    public_source: bool = False,
    extract_propositions: bool = False,
) -> uuid.UUID:
    # public_source content is operator-licensed, so the collection is owned by
    # the system org even though the tenant holds its own copy of the rows.
    owner = SYSTEM_ORG if public_source else org_id
    return await conn.fetchval(
        """
        INSERT INTO collections
            (org_id, owner_org_id, name, kind, public_source,
             extract_propositions)
        VALUES ($1, $2, $3, 'corpus', $4, $5)
        RETURNING id
        """,
        org_id,
        owner,
        unique("coll"),
        public_source,
        extract_propositions,
    )


@pytest.fixture
async def tenant(pool: asyncpg.Pool):
    """An org, a private corpus collection, and an embed spy for them."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        dim = await embedding_dim(conn)
    return org, collection, EmbedSpy(dim)


def ingest_for(pool, org, collection, spy, **kwargs):
    from pgkg.corpus import CorpusIngest

    return CorpusIngest(
        pool,
        org_id=org,
        collection_id=collection,
        embed=spy,
        max_chars=CHUNK_CAP,
        **kwargs,
    )


async def chunk_rows(
    pool: asyncpg.Pool, document_id: uuid.UUID
) -> list[asyncpg.Record]:
    """The chunks the document's current version carries, in order."""
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT c.id, c.text, c.embedding, dvc.ord
            FROM documents d
            JOIN document_version_chunks dvc
                 ON dvc.document_version_id = d.current_version_id
            JOIN chunks c ON c.id = dvc.chunk_id
            WHERE d.id = $1
            ORDER BY dvc.ord
            """,
            document_id,
        )


# ---------------------------------------------------------------------------
# Document upsert: hash first
# ---------------------------------------------------------------------------

async def test_a_new_document_embeds_every_chunk(pool: asyncpg.Pool, tenant) -> None:
    org, collection, spy = tenant
    ingest = ingest_for(pool, org, collection, spy)

    result = await ingest.upsert_document(
        external_id="handbook", text=document_text()
    )

    assert result.changed is True
    assert result.chunks_total == len(WORDS)
    assert result.chunks_new == len(WORDS)
    assert result.chunks_carried == 0
    assert len(spy.texts) == len(WORDS)

    rows = await chunk_rows(pool, result.document_id)
    assert [row["ord"] for row in rows] == list(range(len(WORDS)))
    assert all(row["embedding"] is not None for row in rows)


async def test_an_unchanged_re_ingest_embeds_nothing(
    pool: asyncpg.Pool, tenant
) -> None:
    """The whole reason a nightly full crawl is affordable."""
    org, collection, spy = tenant
    ingest = ingest_for(pool, org, collection, spy)
    text = document_text()

    first = await ingest.upsert_document(external_id="handbook", text=text)
    embedded_once = len(spy.texts)
    second = await ingest.upsert_document(external_id="handbook", text=text)

    assert second.changed is False
    assert second.document_id == first.document_id
    assert second.version_id == first.version_id
    assert second.chunks_new == 0
    assert len(spy.texts) == embedded_once

    async with pool.acquire() as conn:
        versions = await conn.fetchval(
            "SELECT count(*) FROM document_versions WHERE document_id = $1",
            first.document_id,
        )
    assert versions == 1


async def test_an_edit_embeds_only_the_changed_chunks(
    pool: asyncpg.Pool, tenant
) -> None:
    org, collection, spy = tenant
    ingest = ingest_for(pool, org, collection, spy)

    first = await ingest.upsert_document(
        external_id="handbook", text=document_text()
    )
    before = await chunk_rows(pool, first.document_id)
    spy.calls.clear()

    second = await ingest.upsert_document(
        external_id="handbook", text=document_text(edited="hotel")
    )

    assert second.changed is True
    assert second.version_id != first.version_id
    assert second.chunks_carried == len(WORDS) - 1
    assert second.chunks_new == 1
    assert len(spy.texts) == 1

    after = await chunk_rows(pool, second.document_id)
    carried = {row["id"] for row in before} & {row["id"] for row in after}
    assert len(carried) == len(WORDS) - 1
    assert all(row["embedding"] is not None for row in after)


async def test_the_edited_version_becomes_current_and_the_old_one_retires(
    pool: asyncpg.Pool, tenant
) -> None:
    org, collection, spy = tenant
    ingest = ingest_for(pool, org, collection, spy)

    first = await ingest.upsert_document(
        external_id="handbook", text=document_text()
    )
    second = await ingest.upsert_document(
        external_id="handbook", text=document_text(edited="hotel")
    )

    async with pool.acquire() as conn:
        current = await conn.fetchval(
            "SELECT current_version_id FROM documents WHERE id = $1",
            first.document_id,
        )
        statuses = dict(
            await conn.fetch(
                "SELECT id, status FROM document_versions WHERE document_id = $1",
                first.document_id,
            )
        )

    assert current == second.version_id
    assert statuses[second.version_id] == "current"
    assert statuses[first.version_id] == "retired"


async def test_a_failed_ingest_leaves_the_current_version_alone(
    pool: asyncpg.Pool, tenant
) -> None:
    """The flip and everything it depends on is one transaction (D6)."""
    org, collection, spy = tenant
    ingest = ingest_for(pool, org, collection, spy)
    first = await ingest.upsert_document(
        external_id="handbook", text=document_text()
    )

    def exploding_embed(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("the embedder died mid document")

    failing = ingest_for(pool, org, collection, exploding_embed)
    with pytest.raises(RuntimeError):
        await failing.upsert_document(
            external_id="handbook", text=document_text(edited="hotel")
        )

    async with pool.acquire() as conn:
        current = await conn.fetchval(
            "SELECT current_version_id FROM documents WHERE id = $1",
            first.document_id,
        )
        versions = await conn.fetchval(
            "SELECT count(*) FROM document_versions WHERE document_id = $1",
            first.document_id,
        )
        orphan = await conn.fetchval(
            """
            SELECT count(*) FROM chunks
            WHERE org_id = $1 AND text LIKE '%misbehaves%'
            """,
            org,
        )

    assert current == first.version_id
    assert versions == 1
    assert orphan == 0


async def test_two_documents_sharing_a_passage_share_one_chunk_row(
    pool: asyncpg.Pool, tenant
) -> None:
    """Reference counting deduplicates boilerplate within an org (D6)."""
    org, collection, spy = tenant
    ingest = ingest_for(pool, org, collection, spy)
    shared = document_text(words=("alpha", "bravo"))

    first = await ingest.upsert_document(external_id="doc_one", text=shared)
    spy.calls.clear()
    second = await ingest.upsert_document(external_id="doc_two", text=shared)

    assert second.chunks_new == 0
    assert second.chunks_carried == first.chunks_total
    assert spy.texts == []

    ids_one = {row["id"] for row in await chunk_rows(pool, first.document_id)}
    ids_two = {row["id"] for row in await chunk_rows(pool, second.document_id)}
    assert ids_one == ids_two


# ---------------------------------------------------------------------------
# The embedding cache: share the computation, not the rows (D4)
# ---------------------------------------------------------------------------

async def cached_hashes(pool: asyncpg.Pool, texts: list[str]) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT count(*)
            FROM unnest($1::text[]) AS t(txt)
            JOIN embedding_cache ec
                 ON ec.content_hash = digest(t.txt, 'sha256')
            """,
            texts,
        )


async def test_public_source_content_is_embedded_once_across_tenants(
    pool: asyncpg.Pool,
) -> None:
    """The GPU cost is paid once per unique passage per generation (D4)."""
    text = document_text(words=("alpha", "bravo", "charlie"))
    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        first_org = await new_org(conn)
        second_org = await new_org(conn)
        first_collection = await new_collection(
            conn, org_id=first_org, public_source=True
        )
        second_collection = await new_collection(
            conn, org_id=second_org, public_source=True
        )

    first_spy, second_spy = EmbedSpy(dim), EmbedSpy(dim)
    first = await ingest_for(pool, first_org, first_collection, first_spy)\
        .upsert_document(external_id="vendor_doc", text=text)
    second = await ingest_for(pool, second_org, second_collection, second_spy)\
        .upsert_document(external_id="vendor_doc", text=text)

    assert len(first_spy.texts) == first.chunks_total
    assert second_spy.texts == []
    assert second.chunks_total == first.chunks_total

    mine = await chunk_rows(pool, first.document_id)
    theirs = await chunk_rows(pool, second.document_id)
    # Each tenant still holds its own row in its own partition: only the
    # expensive half was shared.
    assert {row["id"] for row in mine}.isdisjoint({row["id"] for row in theirs})
    assert [row["embedding"].to_list() for row in mine] == [
        row["embedding"].to_list() for row in theirs
    ]


async def test_the_cache_names_the_generation_that_produced_the_vector(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, public_source=True)
        primary = await conn.fetchval(
            """
            SELECT generation_id FROM org_embedders
            WHERE org_id = $1 AND role = 'primary'
            """,
            org,
        )

    text = document_text(words=("delta",))
    await ingest_for(pool, org, collection, EmbedSpy(dim)).upsert_document(
        external_id="vendor_doc", text=text
    )

    async with pool.acquire() as conn:
        generations = await conn.fetch(
            """
            SELECT DISTINCT generation_id FROM embedding_cache
            WHERE content_hash IN (
                SELECT content_hash FROM chunks WHERE org_id = $1
            )
            """,
            org,
        )

    assert [row["generation_id"] for row in generations] == [primary]


async def test_a_private_collection_never_populates_the_cache(
    pool: asyncpg.Pool, tenant
) -> None:
    """A content-hash cache is probe-able, so uploads stay out of it (D4)."""
    org, collection, spy = tenant
    result = await ingest_for(pool, org, collection, spy).upsert_document(
        external_id="confidential", text=document_text(words=("echo", "foxtrot"))
    )

    assert result.chunks_new == result.chunks_total
    texts = [row["text"] for row in await chunk_rows(pool, result.document_id)]
    assert await cached_hashes(pool, texts) == 0


async def test_a_private_collection_does_not_consult_the_cache(
    pool: asyncpg.Pool,
) -> None:
    """The flag gates both halves: a cache hit is itself the observation."""
    text = document_text(words=("golf", "hotel"))
    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        public_org = await new_org(conn)
        private_org = await new_org(conn)
        public_collection = await new_collection(
            conn, org_id=public_org, public_source=True
        )
        private_collection = await new_collection(conn, org_id=private_org)

    public_ingest = ingest_for(pool, public_org, public_collection, EmbedSpy(dim))
    await public_ingest.upsert_document(external_id="vendor_doc", text=text)
    private_spy = EmbedSpy(dim)
    private = await ingest_for(
        pool, private_org, private_collection, private_spy
    ).upsert_document(external_id="my_copy", text=text)

    assert len(private_spy.texts) == private.chunks_total


# ---------------------------------------------------------------------------
# The batch queue: throttleable, resumable, observable (D7)
# ---------------------------------------------------------------------------

class FakeIngest:
    """An ingest that records how many documents were in flight at once.

    The worker is what is under test here, so the expensive half is replaced by
    something that yields control — a slot budget that admits two jobs when it
    should admit one is invisible unless the work overlaps.
    """

    def __init__(
        self, *, on_progress_total: int = 0, gate: asyncio.Event | None = None
    ) -> None:
        self.depth = 0
        self.max_depth = 0
        self.seen: list[tuple[uuid.UUID, str]] = []
        self.progress_total = on_progress_total
        self.observed_progress: list[int | None] = []
        self.pool: asyncpg.Pool | None = None
        # Held open by the test rather than finishing on its own: a document
        # that completes in microseconds never overlaps another one, so a slot
        # budget that admits too many would look like one that admits one.
        self.gate = gate

    def for_collection(self, *, org_id, collection_id):
        return self

    async def upsert_document(
        self,
        *,
        external_id,
        text,
        uri=None,
        source=None,
        asserted_at=None,
        provenance=None,
        acl_group_id=None,
        on_progress=None,
    ):
        from pgkg.corpus import CorpusIngestResult

        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        self.seen.append((external_id, text))
        try:
            if on_progress is not None:
                await on_progress(self.progress_total, 0)
                if self.pool is not None:
                    self.observed_progress.append(
                        await self._progress_as_seen_outside()
                    )
            if self.gate is not None:
                await self.gate.wait()
            for _ in range(4):
                await asyncio.sleep(0)
            return CorpusIngestResult(
                document_id=uuid.uuid4(),
                version_id=uuid.uuid4(),
                changed=True,
                chunks_total=self.progress_total,
                chunks_new=self.progress_total,
                embedded=self.progress_total,
            )
        finally:
            self.depth -= 1

    async def _progress_as_seen_outside(self) -> int | None:
        """Read the job's progress on another connection, mid-job.

        Progress written inside the ingest transaction is invisible until
        commit, which is exactly when it stops being progress.
        """
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT max(chunks_total) FROM ingest_jobs WHERE status = 'running'"
            )


async def test_a_queued_document_is_ingested_by_the_worker(
    pool: asyncpg.Pool, tenant
) -> None:
    from pgkg.ingest_jobs import IngestWorker, enqueue_document, job_state

    org, collection, spy = tenant
    text = document_text(words=("alpha", "bravo"))
    job = await enqueue_document(
        pool,
        org_id=org,
        collection_id=collection,
        external_id="queued_doc",
        text=text,
    )

    # Bound to this org: an undrained job left by another test would otherwise
    # be claimed here and counted.
    worker = IngestWorker(
        pool, ingest=ingest_for(pool, org, collection, spy), org_id=org
    )
    assert await worker.run() == 1

    state = await job_state(pool, job, org_id=org)
    assert state.status == "done"
    assert state.chunks_total == 2
    assert state.chunks_embedded == 2
    assert state.error is None

    rows = await chunk_rows(pool, state.document_id)
    assert len(rows) == 2
    assert all(row["embedding"] is not None for row in rows)


async def test_re_enqueueing_open_work_queues_one_job(
    pool: asyncpg.Pool, tenant
) -> None:
    """A connector offers everything it can see every night (D6)."""
    from pgkg.ingest_jobs import enqueue_document

    org, collection, _ = tenant
    text = document_text(words=("charlie",))
    first = await enqueue_document(
        pool, org_id=org, collection_id=collection, external_id="nightly", text=text
    )
    second = await enqueue_document(
        pool, org_id=org, collection_id=collection, external_id="nightly", text=text
    )

    assert second == first
    async with pool.acquire() as conn:
        queued = await conn.fetchval(
            "SELECT count(*) FROM ingest_jobs WHERE org_id = $1", org
        )
    assert queued == 1


async def test_a_finished_document_can_be_offered_again(
    pool: asyncpg.Pool, tenant
) -> None:
    """A document that reverts to an earlier revision is work again, and the
    hash check is what makes that cost nothing."""
    from pgkg.ingest_jobs import IngestWorker, enqueue_document, job_state

    org, collection, spy = tenant
    text = document_text(words=("delta",))
    worker = IngestWorker(pool, ingest=ingest_for(pool, org, collection, spy))

    first = await enqueue_document(
        pool, org_id=org, collection_id=collection, external_id="reverted", text=text
    )
    await worker.run()
    embedded_once = len(spy.texts)

    second = await enqueue_document(
        pool, org_id=org, collection_id=collection, external_id="reverted", text=text
    )
    await worker.run()

    assert second != first
    assert (await job_state(pool, second, org_id=org)).status == "done"
    assert len(spy.texts) == embedded_once


async def test_a_job_a_live_worker_holds_is_not_claimed_twice(
    pool: asyncpg.Pool, tenant
) -> None:
    from pgkg.ingest_jobs import claim_job, enqueue_document

    org, collection, _ = tenant
    await enqueue_document(
        pool,
        org_id=org,
        collection_id=collection,
        external_id="held",
        text=document_text(words=("echo",)),
    )

    held = await claim_job(pool, org_id=org)
    assert held is not None
    assert await claim_job(pool, org_id=org) is None


async def test_an_abandoned_job_is_reclaimed_once_its_lease_lapses(
    pool: asyncpg.Pool, tenant
) -> None:
    """Resumability: a worker that died leaves `running` behind, and only the
    heartbeat tells that apart from a worker still working."""
    from pgkg.ingest_jobs import IngestWorker, claim_job, enqueue_document, job_state

    org, collection, spy = tenant
    job = await enqueue_document(
        pool,
        org_id=org,
        collection_id=collection,
        external_id="abandoned",
        text=document_text(words=("foxtrot", "golf")),
    )

    abandoned = await claim_job(pool, org_id=org)
    assert abandoned is not None and abandoned.id == job

    worker = IngestWorker(
        pool,
        ingest=ingest_for(pool, org, collection, spy),
        org_id=org,
        lease_seconds=0.0,
    )
    assert await worker.run() == 1

    state = await job_state(pool, job, org_id=org)
    assert state.status == "done"
    assert state.attempts == 2
    assert len(await chunk_rows(pool, state.document_id)) == 2


async def test_progress_is_visible_before_the_job_finishes(
    pool: asyncpg.Pool, tenant
) -> None:
    from pgkg.ingest_jobs import IngestWorker, enqueue_document

    org, collection, _ = tenant
    await enqueue_document(
        pool,
        org_id=org,
        collection_id=collection,
        external_id="watched",
        text=document_text(words=("hotel",)),
    )

    fake = FakeIngest(on_progress_total=7)
    fake.pool = pool
    await IngestWorker(pool, ingest=fake, org_id=org).run()

    assert fake.observed_progress == [7]


async def test_the_pipeline_reports_its_own_progress(
    pool: asyncpg.Pool, tenant
) -> None:
    """The counts a customer watches come from the pipeline, not the worker:
    only it knows how many chunks a document turned into."""
    org, collection, spy = tenant
    seen: list[tuple[int, int]] = []

    async def record(total: int, embedded: int) -> None:
        seen.append((total, embedded))

    await ingest_for(pool, org, collection, spy).upsert_document(
        external_id="progress_doc",
        text=document_text(words=("alpha", "bravo", "charlie")),
        on_progress=record,
    )

    assert seen[0] == (3, 0)
    assert seen[-1] == (3, 3)


async def test_a_failing_job_returns_to_the_queue_until_its_attempts_run_out(
    pool: asyncpg.Pool, tenant
) -> None:
    from pgkg.ingest_jobs import IngestWorker, enqueue_document, job_state

    org, collection, _ = tenant
    job = await enqueue_document(
        pool,
        org_id=org,
        collection_id=collection,
        external_id="doomed",
        text=document_text(words=("delta",)),
    )

    def exploding_embed(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("the embedder died mid document")

    worker = IngestWorker(
        pool,
        ingest=ingest_for(pool, org, collection, exploding_embed),
        org_id=org,
        max_attempts=2,
        lease_seconds=0.0,
    )

    await worker.run_once()
    after_one = await job_state(pool, job, org_id=org)
    assert after_one.status == "pending"
    assert "embedder died" in after_one.error

    await worker.run_once()
    after_two = await job_state(pool, job, org_id=org)
    assert after_two.status == "failed"
    assert after_two.attempts == 2


@pytest.mark.parametrize("slots", [1, 3])
async def test_the_worker_holds_exactly_its_slot_budget(
    pool: asyncpg.Pool, tenant, slots: int
) -> None:
    """Corpus ingest is batch and must not compete with recall for pool slots.

    Both directions are asserted: one slot admits one document with three
    queued, and three slots admit three. Without the second case a budget that
    admitted nothing would pass.
    """
    from pgkg.ingest_jobs import IngestWorker, enqueue_document

    org, collection, _ = tenant
    for word in ("alpha", "bravo", "charlie"):
        await enqueue_document(
            pool,
            org_id=org,
            collection_id=collection,
            external_id=f"batch_{word}_{slots}",
            text=document_text(words=(word,)),
        )

    gate = asyncio.Event()
    fake = FakeIngest(gate=gate)
    worker = IngestWorker(pool, ingest=fake, org_id=org, slots=slots)
    drain = asyncio.create_task(worker.run(concurrency=3))
    # Long enough for all three loops to claim and try to enter — a claim is
    # one round trip, and the first one on a fresh task also opens a connection.
    for _ in range(20):
        await asyncio.sleep(0.02)
    in_flight = fake.max_depth

    gate.set()
    assert await drain == 3
    assert in_flight == slots


# ---------------------------------------------------------------------------
# Per-collection extraction: off by default (D2)
# ---------------------------------------------------------------------------

async def propositions_for(
    pool: asyncpg.Pool, document_id: uuid.UUID
) -> list[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT p.id, p.text, p.chunk_id, p.collection_id, p.claim_scope,
                   p.provenance_id, p.invalidated_at
            FROM documents d
            JOIN document_version_chunks dvc
                 ON dvc.document_version_id = d.current_version_id
            JOIN propositions p ON p.chunk_id = dvc.chunk_id
            WHERE d.id = $1
            """,
            document_id,
        )


async def test_a_corpus_collection_does_not_extract_by_default(
    pool: asyncpg.Pool, tenant, monkeypatch
) -> None:
    """The corpus is retrievable in its own right; extraction is opt-in (D2)."""
    from pgkg import ml

    org, collection, spy = tenant

    async def forbidden(*args, **kwargs):
        raise AssertionError("the extractor ran for a collection that opted out")

    monkeypatch.setattr(ml, "extract_propositions_async", forbidden)

    result = await ingest_for(pool, org, collection, spy).upsert_document(
        external_id="not_extracted", text=document_text(words=("alpha", "bravo"))
    )

    assert result.propositions == 0
    assert await propositions_for(pool, result.document_id) == []


async def test_a_collection_that_opts_in_extracts_from_its_chunks(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        org = await new_org(conn)
        collection = await new_collection(
            conn, org_id=org, extract_propositions=True
        )
        claim_scope = await conn.fetchval(
            "SELECT claim_scope FROM collections WHERE id = $1", collection
        )

    result = await ingest_for(pool, org, collection, EmbedSpy(dim)).upsert_document(
        external_id="fact_dense", text=document_text(words=("charlie", "delta"))
    )

    rows = await propositions_for(pool, result.document_id)
    assert result.propositions == 2
    assert len(rows) == 2
    assert {row["collection_id"] for row in rows} == {collection}
    assert {row["claim_scope"] for row in rows} == {claim_scope}
    # The derivation edge D2 calls free: a fact cites the passage it came from.
    assert all(row["chunk_id"] is not None for row in rows)

    async with pool.acquire() as conn:
        linked = await conn.fetchval(
            """
            SELECT count(*) FROM proposition_provenance
            WHERE proposition_id = ANY($1::uuid[])
            """,
            [row["id"] for row in rows],
        )
    assert linked == 2


async def test_only_the_new_chunks_are_extracted_on_re_ingest(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """Extraction is the expensive half twice over: cost per token, recurring
    per prompt version. It follows the same new-chunks-only rule."""
    from pgkg import ml

    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        org = await new_org(conn)
        collection = await new_collection(
            conn, org_id=org, extract_propositions=True
        )

    real = ml.extract_propositions_async
    extracted: list[str] = []

    async def counting(chunk_text: str, **kwargs):
        extracted.append(chunk_text)
        return await real(chunk_text, **kwargs)

    monkeypatch.setattr(ml, "extract_propositions_async", counting)

    ingest = ingest_for(pool, org, collection, EmbedSpy(dim))
    await ingest.upsert_document(external_id="handbook", text=document_text())
    assert len(extracted) == len(WORDS)
    extracted.clear()

    second = await ingest.upsert_document(
        external_id="handbook", text=document_text(edited="hotel")
    )

    assert len(extracted) == 1
    assert second.propositions == 1


async def test_extraction_reuses_the_proposition_cache(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """proposition_cache is keyed on chunk text, so content-addressed chunks
    make a re-ingest after an edit a cache hit for every unchanged chunk (D2)."""
    import json
    from unittest.mock import MagicMock

    import openai

    from pgkg import ml
    from pgkg.chunking import chunk_document

    async with pool.acquire() as conn:
        dim = await embedding_dim(conn)
        org = await new_org(conn)
        collection = await new_collection(
            conn, org_id=org, extract_propositions=True
        )

    text = document_text(words=("echo",))
    chunk = chunk_document(text, max_chars=CHUNK_CAP)[0]
    model = "cached-extractor-model"
    cache_key = ml.compute_cache_key(chunk.text, model)
    stored = [
        {
            "text": "The echo subsystem reconciles ledgers.",
            "subject": "echo subsystem",
            "predicate": "reconciles",
            "object": "ledgers",
            "object_is_literal": True,
        }
    ]

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO proposition_cache
                (cache_key, chunk_hash, extractor_model, prompt_version,
                 propositions, org_id)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6)
            """,
            cache_key,
            hashlib.sha256(chunk.text.encode()).hexdigest(),
            model,
            ml.PROMPT_VERSION,
            json.dumps(stored),
            org,
        )

    settings = MagicMock()
    settings.extractor_model = model
    settings.llm_model = model
    settings.llm_provider = "openai"
    settings.openai_api_key = None
    settings.openai_base_url = None
    monkeypatch.setattr(ml, "get_settings", lambda: settings)
    monkeypatch.delenv("PGKG_OFFLINE_EXTRACT", raising=False)
    monkeypatch.setattr(
        openai.OpenAI,
        "__init__",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("the extractor called an LLM on a cache hit")
        ),
    )

    result = await ingest_for(pool, org, collection, EmbedSpy(dim)).upsert_document(
        external_id="cached_doc", text=text
    )

    rows = await propositions_for(pool, result.document_id)
    assert [row["text"] for row in rows] == [stored[0]["text"]]

    async with pool.acquire() as conn:
        hits = await conn.fetchval(
            "SELECT hit_count FROM proposition_cache WHERE cache_key = $1",
            cache_key,
        )
    assert hits == 1


# ---------------------------------------------------------------------------
# Publication date, not ingest time (D5, D6)
# ---------------------------------------------------------------------------

async def test_a_published_document_is_asserted_at_its_publication_date(
    pool: asyncpg.Pool, tenant
) -> None:
    """D6 keys the perishable profile on asserted_at and D5 says published_at
    feeds it, so the two cannot be independent parameters: left that way, an
    eleven-year-old article decays from the moment this crawl reached it, and
    'perishable' behaves exactly like the 'timeless' profile it is drawn against.
    """
    import math
    from datetime import datetime, timezone

    from pgkg.corpus import Provenance

    org, collection, spy = tenant
    published = datetime(2015, 1, 1, tzinfo=timezone.utc)

    result = await ingest_for(pool, org, collection, spy).upsert_document(
        external_id=unique("stale_article"),
        text=document_text(words=("alpha",)),
        provenance=Provenance(
            kind="document_version", producer="chunker", published_at=published
        ),
    )

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT c.id, c.asserted_at,
                   EXTRACT(EPOCH FROM (now() - c.asserted_at)) / 86400.0 AS age_days
            FROM document_version_chunks dvc
            JOIN chunks c ON c.id = dvc.chunk_id
            WHERE dvc.document_version_id = $1
            """,
            result.version_id,
        )
        await conn.execute(
            "UPDATE collections SET decay_profile = 'perishable' WHERE id = $1",
            collection,
        )
        adjusted = await conn.fetchval(
            """
            SELECT adjusted_score FROM pgkg_apply_profile(
                ARRAY[($1, 'fused', 0, 1.0)::pgkg_candidate]
            )
            """,
            row["id"],
        )

    assert row["asserted_at"] == published
    # The half-life the collection defaults to, read off the age the database
    # itself computed: an assertion on a fixed number would expire with time.
    assert adjusted == pytest.approx(
        math.exp(-float(row["age_days"]) / 730.0), rel=0.01
    )
    assert adjusted < 0.01


async def test_an_explicit_asserted_at_outranks_the_publication_date(
    pool: asyncpg.Pool, tenant
) -> None:
    """A caller stating the world-time of a claim knows something a publication
    date does not, so it is not overwritten by one."""
    from datetime import datetime, timezone

    from pgkg.corpus import Provenance

    org, collection, spy = tenant
    stated = datetime(2024, 6, 1, tzinfo=timezone.utc)

    result = await ingest_for(pool, org, collection, spy).upsert_document(
        external_id=unique("restated"),
        text=document_text(words=("bravo",)),
        asserted_at=stated,
        provenance=Provenance(published_at=datetime(2015, 1, 1, tzinfo=timezone.utc)),
    )

    async with pool.acquire() as conn:
        asserted = await conn.fetchval(
            """
            SELECT c.asserted_at
            FROM document_version_chunks dvc
            JOIN chunks c ON c.id = dvc.chunk_id
            WHERE dvc.document_version_id = $1
            """,
            result.version_id,
        )

    assert asserted == stated


# ---------------------------------------------------------------------------
# What the connector knew has to survive the queue (D5, D6, D7)
# ---------------------------------------------------------------------------

async def test_a_queued_document_keeps_its_publication_date(
    pool: asyncpg.Pool, tenant
) -> None:
    """Batch is the default posture for a corpus, so the queue is the path most
    documents take.

    A job carried only the org, the collection, the external id, the hash, the
    text and the uri, so everything the connector knew about the document —
    who published it, when, under what licence — was dropped at the door and
    the worker re-derived asserted_at from its own clock.  That is the
    perishable-decay defect one indirection further out: the fix on the inline
    path is worth nothing if the queued path still decays from ingest time.
    """
    from datetime import datetime, timezone

    from pgkg.corpus import Provenance
    from pgkg.ingest_jobs import IngestWorker, enqueue_document, job_state

    org, collection, spy = tenant
    published = datetime(2015, 1, 1, tzinfo=timezone.utc)

    job = await enqueue_document(
        pool,
        org_id=org,
        collection_id=collection,
        external_id=unique("queued_article"),
        text=document_text(words=("alpha",)),
        source="crawler",
        provenance=Provenance(
            kind="document_version",
            producer="chunker",
            publisher="The Wire",
            published_at=published,
        ),
    )

    worker = IngestWorker(
        pool, ingest=ingest_for(pool, org, collection, spy), org_id=org
    )
    assert await worker.run() == 1

    state = await job_state(pool, job, org_id=org)
    assert state.status == "done"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT c.asserted_at, pr.published_at, pr.publisher, d.source
            FROM document_versions dv
            JOIN documents d ON d.id = dv.document_id
            JOIN document_version_chunks dvc ON dvc.document_version_id = dv.id
            JOIN chunks c ON c.id = dvc.chunk_id
            LEFT JOIN provenance pr ON pr.id = dv.provenance_id
            WHERE dv.id = $1
            """,
            state.version_id,
        )

    assert row["published_at"] == published
    assert row["publisher"] == "The Wire"
    assert row["source"] == "crawler"
    assert row["asserted_at"] == published


async def test_a_withdrawn_document_keeps_the_id_it_was_ingested_under(
    pool: asyncpg.Pool, tenant
) -> None:
    """Re-ingesting a withdrawn external id is a new document, and the
    withdrawn one is the record of what was withdrawn.

    The unique index did not exclude soft-deleted rows, so the only way to let
    the new document claim the id was to take it off the old one — which erases
    the very thing a deletion audit asks for.  The index is what should have
    said 'live documents only'.
    """
    org, collection, spy = tenant
    external_id = unique("withdrawn")
    ingest = ingest_for(pool, org, collection, spy)

    first = await ingest.upsert_document(
        external_id=external_id, text=document_text(words=("alpha", "bravo"))
    )
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE documents SET deleted_at = now() WHERE id = $1", first.document_id
        )

    second = await ingest.upsert_document(
        external_id=external_id, text=document_text(words=("charlie", "delta"))
    )
    assert second.document_id != first.document_id

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, external_id, deleted_at FROM documents"
            " WHERE org_id = $1 AND collection_id = $2 ORDER BY created_at",
            org,
            collection,
        )

    withdrawn = next(row for row in rows if row["id"] == first.document_id)
    live = next(row for row in rows if row["id"] == second.document_id)
    assert withdrawn["external_id"] == external_id
    assert withdrawn["deleted_at"] is not None
    assert live["external_id"] == external_id
    assert live["deleted_at"] is None


# ---------------------------------------------------------------------------
# What the pipeline costs: round trips, connections held, and pool slots
# ---------------------------------------------------------------------------

def sectioned_document(sections: int) -> str:
    """A document of exactly `sections` distinct chunks."""
    return "\n\n".join(
        f"## Section {i}\n\nThe section {i} chapter explains how subsystem {i} "
        f"behaves when the operator reconciles a ledger, naming {i} so no two "
        f"chapters share text."
        for i in range(sections)
    )


class RecordingPool:
    """Counts round trips made through a pool, without changing what they do."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self.queries: list[str] = []

    def acquire(self):
        outer = self

        class _Ctx:
            async def __aenter__(self):
                self._cm = outer._pool.acquire()
                conn = await self._cm.__aenter__()
                return _RecordingConn(conn, outer.queries)

            async def __aexit__(self, *exc):
                return await self._cm.__aexit__(*exc)

        return _Ctx()


class _RecordingConn:
    def __init__(self, conn: asyncpg.Connection, log: list[str]) -> None:
        self._conn = conn
        self._log = log

    def __getattr__(self, name):
        attr = getattr(self._conn, name)
        if name in ("execute", "fetch", "fetchrow", "fetchval"):
            async def recorded(query, *args, **kwargs):
                self._log.append(query)
                return await attr(query, *args, **kwargs)

            return recorded
        return attr


async def test_ingest_round_trips_do_not_grow_with_chunk_count(
    pool: asyncpg.Pool, tenant
) -> None:
    """Chat ingest was made set-based in phase 0 — twelve chunks and sixty
    facts are five statements.  The pipeline built for a 300-page handbook
    called pgkg_add_version_chunk() once per chunk, each one a network round
    trip with the ingest transaction held open.
    """
    org, collection, spy = tenant

    small = RecordingPool(pool)
    await ingest_for(small, org, collection, spy).upsert_document(
        external_id=unique("rt_small"), text=sectioned_document(2)
    )
    large = RecordingPool(pool)
    result = await ingest_for(large, org, collection, spy).upsert_document(
        external_id=unique("rt_large"), text=sectioned_document(30)
    )

    assert result.chunks_total >= 20, "the document must span many chunks"
    assert len(large.queries) == len(small.queries), (
        f"{result.chunks_total} chunks cost {len(large.queries)} round trips; "
        f"a 2-chunk document cost {len(small.queries)}"
    )


async def test_ingest_does_not_embed_inside_its_transaction(
    pool: asyncpg.Pool, tenant
) -> None:
    """A pooled connection held across a model call is a pool slot spent at the
    embedder's rate.

    memory.py states the rule its own plan is built around — "holding the
    ingest connection across them starves the pool under concurrent ingest" —
    and the bulk pipeline, the one that runs for hours, opened its transaction
    first and then called the embedder and a per-chunk extractor from inside
    it.  A fifty-chunk document with a two-second extractor held one connection,
    one transaction and the cache rows it had touched for a hundred seconds.
    """
    org, collection, spy = tenant
    holder: list[asyncpg.Connection] = []
    seen: list[bool] = []

    class Watching(RecordingPool):
        def acquire(self):
            outer = self

            class _Ctx:
                async def __aenter__(self):
                    self._cm = outer._pool.acquire()
                    conn = await self._cm.__aenter__()
                    holder.append(conn)
                    return conn

                async def __aexit__(self, *exc):
                    return await self._cm.__aexit__(*exc)

            return _Ctx()

    class ProbingSpy(EmbedSpy):
        def __call__(self, texts: list[str]) -> list[list[float]]:
            seen.append(bool(holder) and holder[-1].is_in_transaction())
            return super().__call__(texts)

    probing = ProbingSpy(spy.dim)
    await ingest_for(Watching(pool), org, collection, probing).upsert_document(
        external_id=unique("tx_embed"), text=sectioned_document(6)
    )

    assert seen, "the embedder was never called"
    assert not any(seen), (
        "the embedder ran while the ingest transaction was open, holding a "
        "pooled connection across a model call"
    )


async def test_the_worker_does_not_deadlock_at_its_slot_budget(
    pg_dsn: str, pool: asyncpg.Pool, tenant
) -> None:
    """`slots` rations connections held, and a slot used to hold two.

    The ingest transaction took one and the progress reporter deliberately took
    a second, so that progress is visible while that transaction is open.  With
    slots equal to the pool size every slot then waited for a connection every
    other slot was holding, and `pgkg worker --slots 10` on the default pool
    hung for ever with no error.  The control below is the same worker with one
    slot spare: if that drains and this one hangs, the cause is the budget.
    """
    from pgvector.asyncpg import register_vector

    from pgkg.corpus import CorpusIngest
    from pgkg.ingest_jobs import IngestWorker, enqueue_document

    org, collection, spy = tenant
    for i in range(2):
        await enqueue_document(
            pool,
            org_id=org,
            collection_id=collection,
            external_id=unique(f"deadlock_{i}"),
            text=sectioned_document(3),
        )

    small = await asyncpg.create_pool(
        pg_dsn, min_size=2, max_size=2, init=lambda c: register_vector(c)
    )
    try:
        worker = IngestWorker(
            small,
            ingest=CorpusIngest(
                small, org_id=org, collection_id=collection, embed=spy,
                max_chars=200,
            ),
            org_id=org,
            slots=2,
        )
        drained = await asyncio.wait_for(worker.run(concurrency=2), timeout=15)
    finally:
        await small.close()

    assert drained == 2


async def test_control_the_worker_drains_when_a_slot_is_spare(
    pg_dsn: str, pool: asyncpg.Pool, tenant
) -> None:
    """The control for the test above: same pool, one fewer slot."""
    from pgvector.asyncpg import register_vector

    from pgkg.corpus import CorpusIngest
    from pgkg.ingest_jobs import IngestWorker, enqueue_document

    org, collection, spy = tenant
    for i in range(2):
        await enqueue_document(
            pool,
            org_id=org,
            collection_id=collection,
            external_id=unique(f"control_{i}"),
            text=sectioned_document(3),
        )

    small = await asyncpg.create_pool(
        pg_dsn, min_size=2, max_size=2, init=lambda c: register_vector(c)
    )
    try:
        worker = IngestWorker(
            small,
            ingest=CorpusIngest(
                small, org_id=org, collection_id=collection, embed=spy,
                max_chars=200,
            ),
            org_id=org,
            slots=1,
        )
        drained = await asyncio.wait_for(worker.run(concurrency=2), timeout=15)
    finally:
        await small.close()

    assert drained == 2


async def test_job_status_is_scoped_to_the_caller(pool: asyncpg.Pool) -> None:
    """A job UUID is a handle, not an authorisation.

    The status read named no org and set no GUC, so pgkg_current_org() fell
    back to the default org and the queue's own policy had nothing to compare
    against — on an owner connection it was inert entirely.  What the row
    carries is the document id, the version id, the attempt count and the
    extractor's error text, which is why "no such job" is the honest answer to
    a question the caller was not entitled to ask.
    """
    from pgkg.ingest_jobs import job_state

    async with pool.acquire() as conn:
        stranger = await new_org(conn)
        collection = await new_collection(conn, org_id=stranger)
        job = await conn.fetchval(
            "SELECT pgkg_enqueue_ingest_job($1, $2, $3, digest($4, 'sha256'), $4)",
            stranger,
            collection,
            unique("ext"),
            "a stranger's confidential document body",
        )

    leaked = None
    try:
        leaked = await job_state(pool, job)
    except KeyError:
        pass

    # Its own org still reads it, and the queue is left as it was found: an
    # undrained job is claimable by any worker a later test starts.
    assert (await job_state(pool, job, org_id=stranger)).status == "pending"
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM ingest_jobs WHERE id = $1", job)

    assert leaked is None, (
        f"job_state read another org's job with no scope argument: {leaked!r}"
    )
