"""The corpus, as the Memory and the HTTP surface expose it (ADR 0001, D1).

Migrations 030-032 built the corpus: versioned documents, content-addressed
chunks, the two-store retriever `pgkg_retrieve` and the ingest queue.  Nothing
in Python called any of it — `recall()` still went to `pgkg_search`, which is
proposition-shaped and can only ever answer half the question.  These tests
drive the other half:

  * one ranked list carrying both classes, each row naming the store and the
    collection it came from, so an agent can tell a remembered fact from a
    retrieved passage
  * per-class access as an explicit option, which D1 keeps alongside the fused
    default for the client that would rather route itself
  * the quota reaching the caller: the corpus must not drown the memory, and
    the assertion has to be made where a customer would feel it — over HTTP
  * a chat chunk is provenance, not a passage.  Chat ingest writes chunks so a
    fact can cite the text it came from; they are not the corpus and must not
    be retrieved as though they were
  * document lifecycle over HTTP — create a collection, upsert a document
    idempotently, delete it, and ask whether it is indexed yet, which is the
    first question a customer asks
"""
from __future__ import annotations

import hashlib
import uuid
from contextlib import asynccontextmanager

import asyncpg
import httpx
import pytest

from pgkg import ml
from pgkg.corpus import CorpusIngest
from pgkg.memory import Memory, Scope, provision_org


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

CHAT_FACT = "Our refund policy gives every customer fourteen days."
CORPUS_TEXT = (
    "Refund policy. A customer may return any item within fourteen days of "
    "delivery for a full refund, provided the packaging is intact."
)


@pytest.fixture(scope="session")
async def dim(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )


def _make_embed(width: int):
    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            v = [0.0] * width
            v[int.from_bytes(digest[:4], "big") % width] = 1.0
            out.append(v)
        return out

    return embed


@pytest.fixture(autouse=True)
def _offline_models(dim: int, monkeypatch):
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")
    monkeypatch.setattr(ml, "embed", _make_embed(dim))
    monkeypatch.setattr(ml, "rerank", lambda query, docs: [1.0] * len(docs))


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def _org(pool: asyncpg.Pool) -> uuid.UUID:
    async with pool.acquire() as conn:
        return await provision_org(conn, _unique("corpus_org"))


async def _collection(
    pool: asyncpg.Pool,
    org_id: uuid.UUID,
    *,
    kind: str = "corpus",
    claim_scope: str = "org",
    extract_propositions: bool = False,
) -> uuid.UUID:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO collections
                (org_id, owner_org_id, name, kind, claim_scope,
                 extract_propositions)
            VALUES ($1, $1, $2, $3, $4, $5)
            RETURNING id
            """,
            org_id,
            _unique(kind),
            kind,
            claim_scope,
            extract_propositions,
        )


class _SharedPool:
    """The session pool, handed to the app without letting it close it."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    def __getattr__(self, name):
        return getattr(self._pool, name)

    def acquire(self, *args, **kwargs):
        return self._pool.acquire(*args, **kwargs)


@asynccontextmanager
async def _api(pool: asyncpg.Pool, namespace: str, monkeypatch):
    """The real app, over real HTTP, on the test pool."""
    from pgkg import api

    class _Settings:
        database_url = "unused"
        default_namespace = namespace
        extract_propositions = False

    async def _make_pool(dsn):
        return _SharedPool(pool)

    async def _close_pool(_pool):
        return None

    monkeypatch.setattr(api, "get_settings", lambda: _Settings())
    monkeypatch.setattr(api, "make_pool", _make_pool)
    monkeypatch.setattr(api, "close_pool", _close_pool)

    async with api.lifespan(api.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://pgkg.test",
        ) as client:
            yield client


async def _seed_both_stores(
    pool: asyncpg.Pool, dim: int, *, namespace: str
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """One remembered fact and one retrieved passage, in one org."""
    org = await _org(pool)
    chat = await _collection(pool, org, kind="chat")
    corpus = await _collection(pool, org, kind="corpus")

    memory = Memory(
        pool,
        namespace=namespace,
        scope=Scope(
            org_id=org,
            collection_id=chat,
            subscribed_collection_ids=(corpus,),
        ),
    )
    await memory.ingest(CHAT_FACT)

    ingest = CorpusIngest(
        pool, org_id=org, collection_id=corpus, embed=_make_embed(dim)
    )
    await ingest.upsert_document(external_id=_unique("doc"), text=CORPUS_TEXT)
    return org, chat, corpus


def _memory(pool: asyncpg.Pool, namespace: str, org, chat, corpus) -> Memory:
    return Memory(
        pool,
        namespace=namespace,
        scope=Scope(
            org_id=org,
            collection_id=chat,
            subscribed_collection_ids=(corpus,),
        ),
    )


# ---------------------------------------------------------------------------
# 1. One ranked list, both stores, every row labelled
# ---------------------------------------------------------------------------

async def test_recall_returns_both_stores_and_names_each_source(
    pool: asyncpg.Pool, dim: int
) -> None:
    namespace = _unique("fused")
    org, chat, corpus = await _seed_both_stores(pool, dim, namespace=namespace)

    results = await _memory(pool, namespace, org, chat, corpus).recall(
        "refund policy", k=10, with_rerank=False, with_mmr=False
    )

    by_source = {r.source for r in results}
    assert by_source == {"propositions", "chunks"}, (
        "the fused default has to carry both classes: "
        f"{[(r.source, r.text) for r in results]}"
    )

    passage = next(r for r in results if r.source == "chunks")
    fact = next(r for r in results if r.source == "propositions")

    assert passage.collection_id == corpus
    assert passage.proposition_id is None, "a passage is not a proposition"
    assert passage.chunk_id == passage.item_id
    assert passage.predicate is None and passage.subject is None

    assert fact.collection_id == chat
    assert fact.proposition_id == fact.item_id


async def test_a_chat_chunk_is_provenance_not_a_passage(
    pool: asyncpg.Pool, dim: int
) -> None:
    """Chat ingest writes a chunk so a fact can cite its text.  Retrieving it
    as a passage returns the same content twice and, worse, keeps returning it
    after the fact derived from it has been forgotten."""
    namespace = _unique("chatchunk")
    org = await _org(pool)
    chat = await _collection(pool, org, kind="chat")

    memory = Memory(
        pool, namespace=namespace, scope=Scope(org_id=org, collection_id=chat)
    )
    await memory.ingest(CHAT_FACT)

    async with pool.acquire() as conn:
        chunk_ids = {
            r["id"]
            for r in await conn.fetch(
                "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id"
                " WHERE d.namespace = $1",
                namespace,
            )
        }
    assert chunk_ids, "chat ingest still has to write the chunk it cites"

    results = await memory.recall(
        "refund policy", k=10, with_rerank=False, with_mmr=False
    )

    assert results, "the fact itself is still retrievable"
    assert all(r.source == "propositions" for r in results)
    assert not chunk_ids & {r.item_id for r in results}


# ---------------------------------------------------------------------------
# 2. Per-class access, kept as an explicit option (D1)
# ---------------------------------------------------------------------------

async def test_sources_restricts_recall_to_one_class(
    pool: asyncpg.Pool, dim: int
) -> None:
    namespace = _unique("sources")
    org, chat, corpus = await _seed_both_stores(pool, dim, namespace=namespace)
    memory = _memory(pool, namespace, org, chat, corpus)

    passages = await memory.recall(
        "refund policy", k=10, sources=("chunks",), with_rerank=False, with_mmr=False
    )
    facts = await memory.recall(
        "refund policy",
        k=10,
        sources=("propositions",),
        with_rerank=False,
        with_mmr=False,
    )

    assert passages and {r.source for r in passages} == {"chunks"}
    assert facts and {r.source for r in facts} == {"propositions"}


async def test_an_unknown_source_is_refused_at_the_call_site(
    pool: asyncpg.Pool,
) -> None:
    memory = Memory(pool, namespace=_unique("badsource"))
    with pytest.raises(ValueError):
        await memory.recall("anything", sources=("summaries",))


# ---------------------------------------------------------------------------
# 3. The document lifecycle, over HTTP
# ---------------------------------------------------------------------------

async def test_http_creates_a_collection_the_corpus_can_be_ingested_into(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    org = await _org(pool)
    namespace = _unique("http_coll")

    async with _api(pool, namespace, monkeypatch) as client:
        created = await client.post(
            "/collections",
            json={
                "org_id": str(org),
                "name": _unique("handbook"),
                "kind": "corpus",
                "claim_scope": "org",
                "decay_profile": "timeless",
            },
        )

    assert created.status_code == 200
    collection_id = uuid.UUID(created.json()["collection_id"])

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT org_id, owner_org_id, kind, decay_profile, claim_scope"
            " FROM collections WHERE id = $1",
            collection_id,
        )

    assert row["org_id"] == org
    # A tenant's collection is owned by that tenant: only the operator publishes.
    assert row["owner_org_id"] == org
    assert row["kind"] == "corpus"
    assert row["decay_profile"] == "timeless"


async def test_http_document_upsert_is_idempotent(
    pool: asyncpg.Pool, dim: int, monkeypatch
) -> None:
    """A connector re-crawls everything it can see, every night.  The second
    crawl of unchanged content has to cost nothing — no new version, no
    embedder call — or the corpus is unaffordable (ADR 0001, D6)."""
    org = await _org(pool)
    corpus = await _collection(pool, org, kind="corpus")
    external_id = _unique("handbook_doc")
    namespace = _unique("http_upsert")

    async with _api(pool, namespace, monkeypatch) as client:
        body = {
            "org_id": str(org),
            "collection_id": str(corpus),
            "external_id": external_id,
            "text": CORPUS_TEXT,
            "uri": "https://example.invalid/handbook",
        }
        first = (await client.post("/documents", json=body)).json()
        second = (await client.post("/documents", json=body)).json()

    assert first["changed"] is True
    assert first["chunks_total"] >= 1
    assert first["embedded"] >= 1

    assert second["changed"] is False
    assert second["document_id"] == first["document_id"]
    assert second["version_id"] == first["version_id"]
    assert second["embedded"] == 0

    async with pool.acquire() as conn:
        versions = await conn.fetchval(
            "SELECT count(*) FROM document_versions dv"
            " JOIN documents d ON d.id = dv.document_id"
            " WHERE d.external_id = $1",
            external_id,
        )
    assert versions == 1


async def test_http_document_delete_withdraws_its_passages(
    pool: asyncpg.Pool, dim: int, monkeypatch
) -> None:
    org = await _org(pool)
    chat = await _collection(pool, org, kind="chat")
    corpus = await _collection(pool, org, kind="corpus")
    external_id = _unique("doomed")
    namespace = _unique("http_delete")

    async with _api(pool, namespace, monkeypatch) as client:
        await client.post(
            "/documents",
            json={
                "org_id": str(org),
                "collection_id": str(corpus),
                "external_id": external_id,
                "text": CORPUS_TEXT,
            },
        )
        recall_body = {
            "query": "refund policy",
            "org_id": str(org),
            "collection_id": str(chat),
            "subscribed_collection_ids": [str(corpus)],
            "with_rerank": False,
            "with_mmr": False,
        }
        before = (await client.post("/recall", json=recall_body)).json()

        removed = await client.post(
            "/documents/delete",
            json={
                "org_id": str(org),
                "collection_id": str(corpus),
                "external_id": external_id,
            },
        )
        after = (await client.post("/recall", json=recall_body)).json()

        missing = await client.post(
            "/documents/delete",
            json={
                "org_id": str(org),
                "collection_id": str(corpus),
                "external_id": _unique("never_ingested"),
            },
        )

    assert [r for r in before if r["source"] == "chunks"]
    assert removed.status_code == 204
    assert [r for r in after if r["source"] == "chunks"] == []
    assert missing.status_code == 404


async def test_http_job_status_answers_is_my_corpus_indexed_yet(
    pool: asyncpg.Pool, dim: int, monkeypatch
) -> None:
    from pgkg.ingest_jobs import IngestWorker

    org = await _org(pool)
    corpus = await _collection(pool, org, kind="corpus")
    external_id = _unique("queued_doc")
    namespace = _unique("http_job")

    async with _api(pool, namespace, monkeypatch) as client:
        queued = await client.post(
            "/documents",
            json={
                "org_id": str(org),
                "collection_id": str(corpus),
                "external_id": external_id,
                "text": CORPUS_TEXT,
                "queue": True,
            },
        )
        job_id = queued.json()["job_id"]
        pending = (await client.get(f"/jobs/{job_id}?org_id={org}")).json()

        worker = IngestWorker(
            pool,
            ingest=CorpusIngest(
                pool, org_id=org, collection_id=corpus, embed=_make_embed(dim)
            ),
            org_id=org,
        )
        await worker.run(max_jobs=1)

        done = (await client.get(f"/jobs/{job_id}?org_id={org}")).json()
        unknown = await client.get(f"/jobs/{uuid.uuid4()}?org_id={org}")

    assert queued.status_code == 200
    assert pending["status"] == "pending"
    assert done["status"] == "done"
    assert done["chunks_total"] >= 1
    assert done["document_id"] is not None
    assert done["version_id"] is not None
    assert unknown.status_code == 404


# ---------------------------------------------------------------------------
# 4. The quota, where a customer would feel it
# ---------------------------------------------------------------------------

async def test_http_recall_holds_slots_back_for_the_memory(
    pool: asyncpg.Pool, dim: int, monkeypatch
) -> None:
    """The failure mode D1 designs against: the corpus drowns the memory and
    the assistant stops remembering you and starts quoting the handbook."""
    org = await _org(pool)
    chat = await _collection(pool, org, kind="chat")
    corpus = await _collection(pool, org, kind="corpus")
    namespace = _unique("http_quota")

    async with _api(pool, namespace, monkeypatch) as client:
        for i in range(4):
            await client.post(
                "/documents",
                json={
                    "org_id": str(org),
                    "collection_id": str(corpus),
                    "external_id": _unique(f"policy_{i}"),
                    "text": (
                        f"Refund policy clause {i}. A refund is issued within "
                        f"fourteen days under clause {i} of the refund policy."
                    ),
                },
            )
        for i in range(4):
            await client.post(
                "/memorize",
                json={
                    "org_id": str(org),
                    "collection_id": str(chat),
                    "text": f"We agreed our own refund policy exception {i}.",
                },
            )

        recalled = await client.post(
            "/recall",
            json={
                "query": "refund policy",
                "org_id": str(org),
                "collection_id": str(chat),
                "subscribed_collection_ids": [str(corpus)],
                "k": 10,
                "k_rerank": 4,
                "corpus_fraction": 0.5,
                "memory_floor": 2,
                "with_rerank": False,
                "with_mmr": False,
            },
        )
        unquotaed = await client.post(
            "/recall",
            json={
                "query": "refund policy",
                "org_id": str(org),
                "collection_id": str(chat),
                "subscribed_collection_ids": [str(corpus)],
                "k": 10,
                "sources": ["chunks"],
                "with_rerank": False,
                "with_mmr": False,
            },
        )

    rows = recalled.json()
    passages = [r for r in rows if r["source"] == "chunks"]
    facts = [r for r in rows if r["source"] == "propositions"]

    assert len(passages) == 2, "the corpus ceiling is half of a budget of four"
    assert len(facts) == 2, "and the floor keeps the rest for the memory"

    # The ceiling is a quota, not a shortage: asked for passages alone, the same
    # corpus fills the whole budget.
    assert len(unquotaed.json()) == 4
    assert {r["source"] for r in unquotaed.json()} == {"chunks"}


# ---------------------------------------------------------------------------
# 5. Diversity is a cosine, and cosines need one model space (D8)
# ---------------------------------------------------------------------------

async def _insert_proposition(
    pool: asyncpg.Pool,
    *,
    namespace: str,
    org_id: uuid.UUID,
    text: str,
    embedding: list[float],
    generation_id: uuid.UUID | None = None,
) -> uuid.UUID:
    from pgvector import HalfVector

    async with pool.acquire() as conn:
        await conn.execute("SELECT set_config('pgkg.org_id', $1, false)", str(org_id))
        return await conn.fetchval(
            """
            INSERT INTO propositions
                (text, namespace, org_id, embedding, embedder_generation_id)
            VALUES ($1, $2, $3, $4::halfvec,
                    COALESCE($5, pgkg_generation_1()))
            RETURNING id
            """,
            text,
            namespace,
            org_id,
            HalfVector(embedding),
            generation_id,
        )


async def test_mmr_ignores_a_vector_from_another_embedder_generation(
    pool: asyncpg.Pool, dim: int
) -> None:
    """A vector from another model space is not comparable to the query's, and
    the failure is silent: it is a number, so the cosine returns one.  Here the
    foreign vector is identical to the query, which is what MMR would pick
    first if it were allowed to look (ADR 0001, D8)."""
    org = await _org(pool)
    namespace = _unique("mmr_gen")
    embed = _make_embed(dim)

    async with pool.acquire() as conn:
        generation = await conn.fetchval(
            """
            INSERT INTO embedder_generations
                (name, dim, storage_type, normalize, status)
            VALUES ($1, $2, 'halfvec', TRUE, 'live')
            RETURNING id
            """,
            _unique("other-space"),
            dim,
        )
        await conn.execute(
            "INSERT INTO org_embedders (org_id, generation_id, role)"
            " VALUES ($1, $2, 'secondary')",
            org,
            generation,
        )

    primary = {
        await _insert_proposition(
            pool,
            namespace=namespace,
            org_id=org,
            text=f"Kestrel fact number {i} about kestrels",
            embedding=embed([f"kestrel fact {i}"])[0],
        )
        for i in range(2)
    }
    await _insert_proposition(
        pool,
        namespace=namespace,
        org_id=org,
        text="Kestrel fact from another model space",
        embedding=embed(["kestrel"])[0],
        generation_id=generation,
    )

    memory = Memory(pool, namespace=namespace, scope=Scope(org_id=org))
    results = await memory.recall(
        "kestrel", k=2, with_rerank=False, with_mmr=True
    )

    assert {r.proposition_id for r in results} == primary


# ---------------------------------------------------------------------------
# 6. The operator's entry point into the queue
# ---------------------------------------------------------------------------

def test_cli_exposes_a_worker_with_a_slot_budget_and_a_throttle() -> None:
    """Corpus ingest is batch and must not run in the request process (D7), so
    an operator needs a way to drain the queue — and to slow it down at three
    in the morning without stopping it."""
    from pgkg.cli import build_parser

    args = build_parser().parse_args(
        ["worker", "--slots", "4", "--throttle", "0.25", "--max-jobs", "10"]
    )

    assert args.command == "worker"
    assert args.slots == 4
    assert args.throttle == 0.25
    assert args.max_jobs == 10

    defaults = build_parser().parse_args(["worker"])
    assert defaults.slots == 1
    assert defaults.throttle == 0.0
    assert defaults.max_jobs is None


# ---------------------------------------------------------------------------
# 7. Two gaps the phase-2 SQL left open
# ---------------------------------------------------------------------------

async def test_a_repeated_passage_keeps_every_position_it_appears_in(
    pool: asyncpg.Pool,
) -> None:
    """Boilerplate repeats inside one document — a disclaimer under every
    section.  Content addressing makes that one chunk row, which is the point,
    but it is still at several positions, and the link table is where the
    positions live.  Dropping the second link loses the ordering the window
    expansion reads."""
    org = await _org(pool)
    collection = await _collection(pool, org, kind="corpus")
    repeated = f"The same disclaimer, repeated. {uuid.uuid4().hex[:8]}"

    async with pool.acquire() as conn:
        await conn.execute("SELECT set_config('pgkg.org_id', $1, false)", str(org))
        document = await conn.fetchval(
            "INSERT INTO documents (source, org_id, collection_id, external_id)"
            " VALUES ('repeats', $1, $2, $3) RETURNING id",
            org,
            collection,
            _unique("repeats"),
        )
        version = await conn.fetchval(
            "SELECT version_id FROM pgkg_open_document_version($1,"
            " digest('body', 'sha256'))",
            document,
        )
        first = await conn.fetchval(
            "SELECT chunk_id FROM pgkg_add_version_chunk($1, 0, $2)",
            version,
            repeated,
        )
        await conn.execute(
            "SELECT chunk_id FROM pgkg_add_version_chunk($1, 1, $2)",
            version,
            "Something else entirely.",
        )
        again = await conn.fetchval(
            "SELECT chunk_id FROM pgkg_add_version_chunk($1, 2, $2)",
            version,
            repeated,
        )
        ords = [
            r["ord"]
            for r in await conn.fetch(
                "SELECT ord FROM document_version_chunks"
                " WHERE document_version_id = $1 ORDER BY ord",
                version,
            )
        ]
        refcount = await conn.fetchval(
            "SELECT refcount FROM chunks WHERE id = $1", first
        )

    assert again == first, "the same content is still one chunk row"
    assert ords == [0, 1, 2]
    assert refcount == 2, "both links have to be counted, or GC frees a live chunk"


async def test_the_test_pool_carries_the_production_scan_setting(
    pool: asyncpg.Pool,
) -> None:
    """HNSW walks the graph before the scope filter runs, so a tenant holding a
    small share of the index under-returns unless the scan is iterative.  A
    suite whose pool leaves it off is not exercising the search the product
    performs.

    Asserted after other tests have borrowed and returned connections, because
    that is where the setting is lost: asyncpg runs RESET ALL on release, so a
    SET issued from the pool's init callback survives exactly one acquire."""
    async with pool.acquire() as conn:
        assert await conn.fetchval("SHOW hnsw.iterative_scan") == "strict_order"
