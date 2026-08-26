"""What recall() passes down to the retriever, and what it writes on the way out.

Two behaviours that the two-store recall path owns and that no test above it
pins, each found by mutating the implementation and watching the suite stay
green:

  * the quota arguments actually reach SQL.  `memory_floor` is the personal
    slot reservation D1 sizes at eight, and a recall that accepted the argument
    and dropped it would report exactly the ranking the quota exists to
    prevent, with no error anywhere.
  * reading a passage does not touch the access ledger.  D6 turns the frequency
    boost off for both corpus profiles — "reading the handbook is not evidence
    the handbook is true" — and the ledger is keyed on proposition ids, so a
    passage id landing in it would bump nothing and inflate the batch.
"""
from __future__ import annotations

import hashlib
import uuid

import asyncpg
import pytest

from pgkg import ml
from pgkg.corpus import CorpusIngest
from pgkg.memory import Memory, Scope, provision_org

CHAT_FACT = "We agreed a fourteen day refund window for every customer."
CORPUS_BODY = (
    "Refund policy clause {i}. A refund is issued within fourteen days under "
    "clause {i} of the published refund policy."
)
QUERY = "refund policy refund window"


@pytest.fixture(scope="session")
async def dim(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )


# Every module in this suite seeds one-hot fixture vectors into one shared HNSW
# index, and any two that do not share a hot axis are at cosine distance exactly
# 1.0 from each other — so each module's rows are tied competitors in every
# other module's neighbourhood, and a module that adds enough of them can
# reorder a search it has nothing to do with.
#
# These vectors carry a small negative floor, which makes their dot product with
# a foreign one-hot query negative and their distance greater than 1.0.  They
# therefore sort behind every plain one-hot row and cannot displace one.  Among
# themselves the hot axis still dominates, so retrieval inside this module works
# exactly as it would otherwise.
_FLOOR = -0.01


def _make_embed(width: int):
    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            v = [_FLOOR] * width
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
        return await provision_org(conn, _unique("quota_org"))


async def _collection(pool: asyncpg.Pool, org_id: uuid.UUID, kind: str) -> uuid.UUID:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO collections
                (org_id, owner_org_id, name, kind, claim_scope,
                 extract_propositions)
            VALUES ($1, $1, $2, $3, 'org', FALSE)
            RETURNING id
            """,
            org_id,
            _unique(kind),
            kind,
        )


async def _seed(
    pool: asyncpg.Pool, dim: int, *, namespace: str, facts: int, passages: int
):
    """A corpus that outnumbers the memory, which is the only condition under
    which a floor or a ceiling can be observed at all."""
    org = await _org(pool)
    chat = await _collection(pool, org, "chat")
    corpus = await _collection(pool, org, "corpus")

    memory = Memory(
        pool,
        namespace=namespace,
        scope=Scope(
            org_id=org, collection_id=chat, subscribed_collection_ids=(corpus,)
        ),
    )
    for i in range(facts):
        await memory.ingest(f"{CHAT_FACT} Note {i}.")

    ingest = CorpusIngest(
        pool, org_id=org, collection_id=corpus, embed=_make_embed(dim)
    )
    for i in range(passages):
        await ingest.upsert_document(
            external_id=_unique(f"clause_{i}"), text=CORPUS_BODY.format(i=i)
        )
    return memory, org, chat, corpus


async def test_the_memory_floor_reaches_the_retriever(
    pool: asyncpg.Pool, dim: int
) -> None:
    """The same query and the same rows, asked twice: once with the personal
    slots reserved and once with the reservation surrendered.

    Both calls are made through recall(), so this fails if the argument is
    accepted and never forwarded — which is exactly what an ordinary
    pass-through looks like when it is wrong.
    """
    namespace = _unique("floor")
    memory, *_ = await _seed(pool, dim, namespace=namespace, facts=4, passages=12)

    reserved = await memory.recall(
        QUERY, k=6, k_rerank=6, corpus_fraction=1.0, memory_floor=4,
        with_rerank=False, with_mmr=False,
    )
    surrendered = await memory.recall(
        QUERY, k=6, k_rerank=6, corpus_fraction=1.0, memory_floor=0,
        with_rerank=False, with_mmr=False,
    )

    kept = sum(1 for r in reserved if r.source == "propositions")
    dropped = sum(1 for r in surrendered if r.source == "propositions")

    assert kept == 4, (
        "four slots were reserved for the caller's own memory and "
        f"{kept} came back: {[(r.source, r.text) for r in reserved]}"
    )
    assert kept > dropped, (
        "the floor made no difference, so it never reached the retriever: "
        f"reserved={kept} surrendered={dropped}"
    )


async def test_reading_a_passage_does_not_enter_the_access_ledger(
    pool: asyncpg.Pool, dim: int
) -> None:
    """The ledger is the read path's only write, and D6 keeps corpus reads out
    of it.  A passage id recorded here would bump no row — the counter lives on
    propositions — so the cost would be silent: a larger flush, and a frequency
    signal that counted reads it was never meant to count.
    """
    namespace = _unique("ledger")
    memory, *_ = await _seed(pool, dim, namespace=namespace, facts=1, passages=4)
    await memory.flush_access()

    results = await memory.recall(QUERY, k=10, with_rerank=False, with_mmr=False)

    assert [r for r in results if r.source == "chunks"], (
        "no passage came back, so this proves nothing about passages"
    )
    facts = {r.proposition_id for r in results if r.source == "propositions"}
    recorded = {prop_id for _org, prop_id in memory._access.pending}

    assert recorded == facts, (
        "the ledger must hold exactly the propositions that were read: "
        f"recorded={recorded} facts={facts}"
    )


async def test_the_corpus_fraction_reaches_the_retriever(
    pool: asyncpg.Pool, dim: int
) -> None:
    """The ceiling, measured where it is the binding constraint.

    A floor and a ceiling can each be the one that decides, and with slots
    reserved the floor usually gets there first — which is why the floor test
    above cannot see this argument at all.  Here the reservation is surrendered
    so the only thing limiting the corpus is the share it is allowed.
    """
    namespace = _unique("ceiling")
    memory, *_ = await _seed(pool, dim, namespace=namespace, facts=4, passages=12)

    capped = await memory.recall(
        QUERY, k=10, k_rerank=6, corpus_fraction=0.5, memory_floor=0,
        with_rerank=False, with_mmr=False,
    )
    opened = await memory.recall(
        QUERY, k=10, k_rerank=6, corpus_fraction=1.0, memory_floor=0,
        with_rerank=False, with_mmr=False,
    )

    narrow = sum(1 for r in capped if r.source == "chunks")
    wide = sum(1 for r in opened if r.source == "chunks")

    assert narrow == 3, (
        f"half of six reranker slots is three passages, got {narrow}"
    )
    assert wide > narrow, (
        "opening the ceiling admitted no more of the corpus, so the share "
        f"never reached the retriever: capped={narrow} opened={wide}"
    )
