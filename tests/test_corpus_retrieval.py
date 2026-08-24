"""The corpus as a second candidate source, quotas, and decay profiles.

D1 splits retrieval into two physical stores because one BM25 `avgdl` cannot
describe both a 12-token fact and a 600-token passage.  These tests pin that
the chunk store carries its own corpus statistics, that the source quotas keep
a large corpus from drowning a user's own memory before the reranker ever sees
it, that the three decay profiles of D6 are resolved from the collection, and
that a chunk retrieved small is returned big.
"""
from __future__ import annotations

import math
import uuid

import asyncpg
import pytest
from pgvector import HalfVector

DIM = 1024


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def vec(hot_index: int = 0, value: float = 1.0) -> HalfVector:
    raw = [0.0] * DIM
    raw[hot_index] = value
    return HalfVector(raw)


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
    decay_profile: str = "conversational",
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO collections
            (org_id, owner_org_id, name, kind, claim_scope, decay_profile)
        VALUES ($1, $1, $2, $3, $4, $5)
        RETURNING id
        """,
        org_id, unique("coll"), kind, claim_scope, decay_profile,
    )


async def insert_chunk(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    text: str,
    embedding: HalfVector | None = None,
    asserted_at: object = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO chunks (text, org_id, collection_id, embedding, asserted_at)
        VALUES ($1, $2, $3, $4::halfvec, $5)
        RETURNING id
        """,
        text, org_id, collection_id, embedding, asserted_at,
    )


async def insert_proposition(
    conn: asyncpg.Connection,
    *,
    text: str,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    namespace: str = "default",
    embedding: HalfVector | None = None,
    claim_scope: str = "org",
    asserted_at: object = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO propositions
            (text, namespace, org_id, collection_id, embedding, claim_scope,
             asserted_at)
        VALUES ($1, $2, $3, $4, $5::halfvec, $6, $7)
        RETURNING id
        """,
        text, namespace, org_id, collection_id, embedding, claim_scope,
        asserted_at,
    )


async def stats_row(
    conn: asyncpg.Connection, *, kind: str, domain: str
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT n_total, total_len, avgdl FROM corpus_stats
        WHERE kind = $1 AND namespace = $2
        """,
        kind, domain,
    )


async def chunk_domain(conn: asyncpg.Connection, collection_id: uuid.UUID) -> str:
    return await conn.fetchval("SELECT pgkg_stats_domain($1)", collection_id)


# ---------------------------------------------------------------------------
# Separate corpus statistics per source
# ---------------------------------------------------------------------------


async def test_chunk_inserts_register_their_own_corpus_statistics(
    pool: asyncpg.Pool,
) -> None:
    """A chunk store with no statistics of its own would be normalised against
    the proposition average, which is the mixture D1 rejects."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        for i in range(3):
            await insert_chunk(
                conn, org_id=org, collection_id=collection,
                text=f"{unique('passage')} the quarterly expenses policy "
                     f"requires receipts for every claim above ten pounds {i}",
            )

        row = await stats_row(
            conn, kind="chunk", domain=await chunk_domain(conn, collection),
        )

    assert row is not None
    assert row["n_total"] == 3
    assert row["avgdl"] > 5.0


async def test_avgdl_differs_per_source(pool: asyncpg.Pool) -> None:
    """The whole point of the split: long passages and short facts do not
    share a length normaliser."""
    ns = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        padding = " ".join(f"clause{i}" for i in range(60))
        for i in range(3):
            await insert_chunk(
                conn, org_id=org, collection_id=collection,
                text=f"{unique('doc')} mitochondria {padding} {i}",
            )
        for i in range(3):
            await insert_proposition(
                conn, text=f"mitochondria generate energy {i}",
                org_id=org, collection_id=collection, namespace=ns,
            )

        chunks = await stats_row(
            conn, kind="chunk", domain=await chunk_domain(conn, collection),
        )
        props = await stats_row(conn, kind="proposition", domain=ns)

    assert chunks["avgdl"] > 10 * props["avgdl"]


async def test_chunk_statistics_track_deletes(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        keep = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text=f"{unique('keep')} retained passage about receipts",
        )
        drop = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text=f"{unique('drop')} discarded passage about receipts",
        )
        domain = await chunk_domain(conn, collection)
        before = await stats_row(conn, kind="chunk", domain=domain)

        await conn.execute("DELETE FROM chunks WHERE id = $1", drop)
        after = await stats_row(conn, kind="chunk", domain=domain)

        remaining = await conn.fetchval(
            "SELECT df FROM lexeme_df WHERE kind = 'chunk' AND namespace = $1"
            "   AND lexeme = 'receipt'",
            domain,
        )

    assert before["n_total"] == 2
    assert after["n_total"] == 1
    assert remaining == 1
    assert keep is not None


# ---------------------------------------------------------------------------
# Chunk retrieval as its own candidate source
# ---------------------------------------------------------------------------


async def test_bm25_candidates_retrieve_chunks_for_the_chunk_source(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="filing an expense claim requires an itemised receipt",
        )
        await insert_proposition(
            conn, text="expense claim policy was discussed",
            org_id=org, collection_id=collection, namespace=unique("ns"),
        )

        rows = await conn.fetch(
            """
            SELECT item_id, kind, rank FROM pgkg_bm25_candidates(
                $1, 'default', NULL, 200, $2::uuid[], $3::uuid[], NULL, NULL,
                NULL, 'chunks'
            ) ORDER BY rank
            """,
            "expense claim receipt", [org], [collection],
        )

    assert [r["item_id"] for r in rows] == [chunk]
    assert rows[0]["kind"] == "kw"


async def test_vector_candidates_retrieve_chunks_for_the_chunk_source(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        near = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text=unique("near passage"), embedding=vec(0),
        )
        far = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text=unique("far passage"), embedding=vec(7),
        )

        rows = await conn.fetch(
            """
            SELECT item_id, rank FROM pgkg_vector_candidates(
                $1::halfvec, 'default', NULL, 200, $2::uuid[], $3::uuid[],
                NULL, NULL, NULL, 'chunks'
            ) ORDER BY rank
            """,
            vec(0), [org], [collection],
        )

    assert [r["item_id"] for r in rows] == [near, far]


async def test_the_proposition_source_stays_the_default(
    pool: asyncpg.Pool,
) -> None:
    """Every existing caller passes no source at all, and must keep getting
    propositions and nothing else."""
    ns = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        prop = await insert_proposition(
            conn, text="zymurgy is the study of fermentation",
            org_id=org, collection_id=collection, namespace=ns,
        )
        await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="zymurgy is the study of fermentation in brewing",
        )

        rows = await conn.fetch(
            "SELECT item_id FROM pgkg_bm25_candidates($1, $2)", "zymurgy", ns,
        )

    assert [r["item_id"] for r in rows] == [prop]


async def test_chunk_bm25_scores_are_computed_from_the_chunk_statistics(
    pool: asyncpg.Pool,
) -> None:
    """Not merely "the short one wins" — the score is the BM25 value implied by
    the chunk store's own n_total, avgdl and document frequency.  Normalising
    against the proposition average would land somewhere else entirely."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        short = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="mitochondria generate cellular energy",
        )
        padding = " ".join(f"aside{i}" for i in range(120))
        long = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text=f"mitochondria generate cellular energy {padding}",
        )

        domain = await chunk_domain(conn, collection)
        stats = await stats_row(conn, kind="chunk", domain=domain)
        df = await conn.fetchval(
            """
            SELECT df FROM lexeme_df
            WHERE kind = 'chunk' AND namespace = $1 AND lexeme = 'mitochondria'
            """,
            domain,
        )
        doc_lens = {
            r["id"]: r["doc_len"]
            for r in await conn.fetch(
                "SELECT id, doc_len FROM chunks WHERE id = ANY($1::uuid[])",
                [short, long],
            )
        }
        rows = await conn.fetch(
            """
            SELECT item_id, raw_score FROM pgkg_bm25_candidates(
                $1, 'default', NULL, 200, $2::uuid[], $3::uuid[], NULL, NULL,
                NULL, 'chunks'
            ) ORDER BY rank
            """,
            "mitochondria", [org], [collection],
        )

    n_total = float(stats["n_total"])
    avgdl = float(stats["avgdl"])
    idf = math.log((n_total - df + 0.5) / (df + 0.5) + 1.0)

    def expected(doc_len: int) -> float:
        return idf * (1.0 * 2.2) / (1.0 + 1.2 * (0.25 + 0.75 * doc_len / avgdl))

    assert [r["item_id"] for r in rows] == [short, long]
    for row in rows:
        assert row["raw_score"] == pytest.approx(
            expected(doc_lens[row["item_id"]]), rel=1e-3
        )


# ---------------------------------------------------------------------------
# The three decay profiles
# ---------------------------------------------------------------------------


async def profile_scores(
    conn: asyncpg.Connection,
    item_ids: list[uuid.UUID],
    **kwargs: object,
) -> dict[uuid.UUID, float]:
    rows = await conn.fetch(
        """
        SELECT item_id, adjusted_score FROM pgkg_apply_profile(
            ARRAY(
                SELECT (u.id, 'fused', 0, 1.0)::pgkg_candidate
                FROM unnest($1::uuid[]) AS u(id)
            ),
            COALESCE($2::real, 30.0),
            COALESCE($3::real, 730.0)
        )
        """,
        item_ids,
        kwargs.get("recency_half_life_days"),
        kwargs.get("perishable_half_life_days"),
    )
    return {r["item_id"]: float(r["adjusted_score"]) for r in rows}


async def test_conversational_collections_decay_on_asserted_at(
    pool: asyncpg.Pool,
) -> None:
    ns = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(
            conn, org_id=org, kind="chat", decay_profile="conversational"
        )
        fresh = await insert_proposition(
            conn, text="the refund window is fourteen days", org_id=org,
            collection_id=collection, namespace=ns,
            asserted_at=await conn.fetchval("SELECT now()"),
        )
        stale = await insert_proposition(
            conn, text="the refund window was seven days", org_id=org,
            collection_id=collection, namespace=ns,
            asserted_at=await conn.fetchval("SELECT now() - interval '400 days'"),
        )

        scores = await profile_scores(conn, [fresh, stale])

    assert scores[fresh] > scores[stale]


async def test_timeless_collections_do_not_decay(pool: asyncpg.Pool) -> None:
    """A 2019 expenses policy is the expenses policy."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(
            conn, org_id=org, kind="corpus", decay_profile="timeless"
        )
        recent = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text=unique("recent policy passage"),
            asserted_at=await conn.fetchval("SELECT now()"),
        )
        ancient = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text=unique("ancient policy passage"),
            asserted_at=await conn.fetchval(
                "SELECT now() - interval '4000 days'"
            ),
        )

        scores = await profile_scores(conn, [recent, ancient])

    assert scores[recent] == pytest.approx(1.0)
    assert scores[ancient] == pytest.approx(1.0)


async def test_perishable_collections_decay_on_the_publication_date(
    pool: asyncpg.Pool,
) -> None:
    """External guidance ages, but on a multi-year curve rather than a
    conversational one."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        perishable = await new_collection(
            conn, org_id=org, kind="corpus", decay_profile="perishable"
        )
        conversational = await new_collection(
            conn, org_id=org, kind="chat", decay_profile="conversational"
        )
        published_at = await conn.fetchval(
            "SELECT now() - interval '400 days'"
        )
        old_doc = await insert_chunk(
            conn, org_id=org, collection_id=perishable,
            text=unique("vendor guidance from a while back"),
            asserted_at=published_at,
        )
        new_doc = await insert_chunk(
            conn, org_id=org, collection_id=perishable,
            text=unique("vendor guidance published today"),
            asserted_at=await conn.fetchval("SELECT now()"),
        )
        old_chat = await insert_proposition(
            conn, text=unique("a thing said a while back"), org_id=org,
            collection_id=conversational, namespace=unique("ns"),
            asserted_at=published_at,
        )

        scores = await profile_scores(conn, [old_doc, new_doc, old_chat])

    assert scores[new_doc] > scores[old_doc] > 0.0
    assert scores[old_doc] > scores[old_chat]


async def test_the_frequency_boost_is_off_for_the_corpus_profiles(
    pool: asyncpg.Pool,
) -> None:
    """log(1 + access_count) on reference material is a popularity feedback
    loop, and on shared material it carries usage across tenants."""
    ns = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        timeless = await new_collection(
            conn, org_id=org, kind="corpus", decay_profile="timeless"
        )
        chat = await new_collection(
            conn, org_id=org, kind="chat", decay_profile="conversational"
        )
        popular_doc = await insert_proposition(
            conn, text=unique("a much read definition"), org_id=org,
            collection_id=timeless, namespace=ns,
        )
        quiet_doc = await insert_proposition(
            conn, text=unique("an unread definition"), org_id=org,
            collection_id=timeless, namespace=ns,
        )
        popular_fact = await insert_proposition(
            conn, text=unique("a much recalled fact"), org_id=org,
            collection_id=chat, namespace=ns,
        )
        quiet_fact = await insert_proposition(
            conn, text=unique("a seldom recalled fact"), org_id=org,
            collection_id=chat, namespace=ns,
        )
        await conn.execute(
            "UPDATE propositions SET access_count = 50 WHERE id = ANY($1::uuid[])",
            [popular_doc, popular_fact],
        )

        scores = await profile_scores(
            conn, [popular_doc, quiet_doc, popular_fact, quiet_fact]
        )

    assert scores[popular_doc] == pytest.approx(scores[quiet_doc])
    assert scores[popular_fact] > scores[quiet_fact]


# ---------------------------------------------------------------------------
# Per-claim-scope source quotas
# ---------------------------------------------------------------------------


async def seed_corpus(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    n: int,
    body: str,
    embedding: HalfVector | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO chunks (text, org_id, collection_id, embedding)
        SELECT format($fmt$%s %s$fmt$, i, $1::text), $2, $3, $4::halfvec
        FROM generate_series(1, $5) AS i
        """,
        body, org_id, collection_id, embedding, n,
    )


async def retrieve(
    conn: asyncpg.Connection, q_text: str, **kwargs: object
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT * FROM pgkg_retrieve(
            $1, $2::halfvec,
            k_retrieve => $3, k_initial => $4,
            p_namespace => $5, expand_graph => FALSE,
            p_org_ids => $6::uuid[], p_collection_ids => $7::uuid[],
            k_rerank => $8, corpus_fraction => $9::real,
            memory_floor => $10,
            window_before => $11, window_after => $12,
            w_scope_world => $13::real, w_scope_org => $14::real,
            w_scope_user => $15::real
        )
        """,
        q_text,
        kwargs.get("q_embedding"),
        kwargs.get("k_retrieve", 20),
        kwargs.get("k_initial", 200),
        kwargs.get("namespace", "default"),
        kwargs.get("org_ids"),
        kwargs.get("collection_ids"),
        kwargs.get("k_rerank", 20),
        kwargs.get("corpus_fraction", 0.6),
        kwargs.get("memory_floor", 8),
        kwargs.get("window_before", 1),
        kwargs.get("window_after", 1),
        kwargs.get("w_scope_world", 1.0),
        kwargs.get("w_scope_org", 1.0),
        kwargs.get("w_scope_user", 1.0),
    )


async def personal_and_corpus(
    conn: asyncpg.Connection,
) -> tuple[uuid.UUID, list[uuid.UUID], uuid.UUID, str]:
    """A large corpus, a modest set of personal facts, and the one personal
    fact that answers the query."""
    org = await new_org(conn)
    corpus = await new_collection(
        conn, org_id=org, kind="corpus", claim_scope="org",
        decay_profile="timeless",
    )
    chat = await new_collection(
        conn, org_id=org, kind="chat", claim_scope="user",
        decay_profile="conversational",
    )
    ns = unique("ns")

    await seed_corpus(
        conn, org_id=org, collection_id=corpus, n=200,
        body="refund window policy section the refund window is set out in the "
             "vendor agreement",
        embedding=vec(0),
    )
    await conn.execute(
        """
        INSERT INTO propositions (text, namespace, org_id, collection_id,
                                  claim_scope)
        SELECT format('the refund window came up in passing %s', i), $1, $2, $3,
               'user'
        FROM generate_series(1, 29) AS i
        """,
        ns, org, chat,
    )
    answer = await insert_proposition(
        conn, text="I decided our refund window should be fourteen days",
        org_id=org, collection_id=chat, namespace=ns, claim_scope="user",
    )
    return org, [chat, corpus], answer, ns


async def test_personal_facts_survive_a_large_corpus(pool: asyncpg.Pool) -> None:
    """The product requirement.  600k org chunks against a user's own facts win
    on candidate volume alone, and the symptom is the assistant quoting the
    handbook instead of remembering you."""
    async with pool.acquire() as conn:
        org, collections, answer, ns = await personal_and_corpus(conn)

        rows = await retrieve(
            conn, "what did I decide about the refund window",
            q_embedding=vec(0), namespace=ns, org_ids=[org],
            collection_ids=collections,
        )

    assert answer in [r["item_id"] for r in rows]
    assert sum(1 for r in rows if r["bucket"] == "corpus") == 12
    assert sum(1 for r in rows if r["bucket"] == "memory") == 8


async def test_without_the_quota_the_corpus_takes_every_slot(
    pool: asyncpg.Pool,
) -> None:
    """The same fixture with the cap opened and the floor removed: the corpus
    does drown the memory, which is what the defaults are defending against."""
    async with pool.acquire() as conn:
        org, collections, answer, ns = await personal_and_corpus(conn)

        rows = await retrieve(
            conn, "what did I decide about the refund window",
            q_embedding=vec(0), namespace=ns, org_ids=[org],
            collection_ids=collections, corpus_fraction=1.0, memory_floor=0,
        )

    assert answer not in [r["item_id"] for r in rows]
    assert all(r["bucket"] == "corpus" for r in rows)


async def test_the_corpus_allowance_is_split_across_claim_scopes(
    pool: asyncpg.Pool,
) -> None:
    """Quota by claim scope, not just by store: general knowledge is topically
    unbounded and competes on every query, so it must not consume the whole
    corpus allowance."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        world = await new_collection(
            conn, org_id=org, kind="corpus", claim_scope="world",
            decay_profile="perishable",
        )
        org_scope = await new_collection(
            conn, org_id=org, kind="corpus", claim_scope="org",
            decay_profile="timeless",
        )
        padding = " ".join(f"clause{i}" for i in range(60))
        await seed_corpus(
            conn, org_id=org, collection_id=world, n=100,
            body="succinct refund window note",
        )
        await seed_corpus(
            conn, org_id=org, collection_id=org_scope, n=100,
            body=f"refund window buried in {padding}",
        )

        rows = await retrieve(
            conn, "refund window", org_ids=[org],
            collection_ids=[world, org_scope],
            corpus_fraction=1.0, memory_floor=0,
        )

    scopes = [r["claim_scope"] for r in rows]
    assert len(rows) == 20
    assert scopes.count("world") == 10
    assert scopes.count("org") == 10


async def test_quotas_admit_everything_when_the_budget_is_not_binding(
    pool: asyncpg.Pool,
) -> None:
    """A quota is a cap, not a reservation: a small corpus and a small memory
    both pass through untouched."""
    ns = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        corpus = await new_collection(conn, org_id=org, kind="corpus")
        chat = await new_collection(conn, org_id=org, kind="chat")
        await seed_corpus(
            conn, org_id=org, collection_id=corpus, n=3,
            body="the refund window is in the agreement",
        )
        for i in range(2):
            await insert_proposition(
                conn, text=f"our refund window is fourteen days {i}",
                org_id=org, collection_id=chat, namespace=ns,
            )

        rows = await retrieve(
            conn, "refund window", namespace=ns, org_ids=[org],
            collection_ids=[chat, corpus],
        )

    assert sum(1 for r in rows if r["bucket"] == "corpus") == 3
    assert sum(1 for r in rows if r["bucket"] == "memory") == 2


async def test_retrieve_names_the_store_each_row_came_from(
    pool: asyncpg.Pool,
) -> None:
    ns = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        corpus = await new_collection(conn, org_id=org, kind="corpus")
        chat = await new_collection(conn, org_id=org, kind="chat")
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=corpus,
            text="the mitochondrion is the powerhouse of the cell",
        )
        prop = await insert_proposition(
            conn, text="mitochondrion came up in the lesson", org_id=org,
            collection_id=chat, namespace=ns,
        )

        rows = await retrieve(
            conn, "mitochondrion", namespace=ns, org_ids=[org],
            collection_ids=[chat, corpus],
        )

    by_id = {r["item_id"]: r for r in rows}
    assert by_id[chunk]["source"] == "chunks"
    assert by_id[prop]["source"] == "propositions"
    assert by_id[chunk]["text"].startswith("the mitochondrion")


# ---------------------------------------------------------------------------
# Small-to-big context expansion
# ---------------------------------------------------------------------------


async def new_document(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO documents (source, namespace, org_id, collection_id,
                               external_id)
        VALUES ($1, 'default', $2, $3, $4) RETURNING id
        """,
        unique("src"), org_id, collection_id, unique("ext"),
    )


async def add_version(
    conn: asyncpg.Connection, document: uuid.UUID, chunks: list[str]
) -> list[uuid.UUID]:
    version = await conn.fetchval(
        "SELECT version_id FROM pgkg_open_document_version($1, digest($2, 'sha256'))",
        document, "".join(chunks),
    )
    ids = [
        await conn.fetchval(
            "SELECT chunk_id FROM pgkg_add_version_chunk($1, $2, $3)",
            version, ord_, text,
        )
        for ord_, text in enumerate(chunks)
    ]
    await conn.execute("SELECT pgkg_promote_document_version($1)", version)
    return ids


async def build_document(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    chunks: list[str],
) -> list[uuid.UUID]:
    document = await new_document(
        conn, org_id=org_id, collection_id=collection_id
    )
    return await add_version(conn, document, chunks)


async def test_a_chunk_hit_returns_its_neighbouring_chunks(
    pool: asyncpg.Pool,
) -> None:
    """A 300-token passage is the right unit to match and the wrong unit to
    read, so the hit is widened by document order."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="corpus")
        bodies = [
            f"{unique('para0')} opening remarks about nothing much",
            f"{unique('para1')} the preceding paragraph of the section",
            f"{unique('para2')} zymurgy is the applied science of fermentation",
            f"{unique('para3')} the following paragraph of the section",
            f"{unique('para4')} closing remarks about nothing much",
        ]
        ids = await build_document(
            conn, org_id=org, collection_id=collection, chunks=bodies
        )

        rows = await retrieve(
            conn, "zymurgy", org_ids=[org], collection_ids=[collection],
        )

    assert [r["item_id"] for r in rows] == [ids[2]]
    context = rows[0]["context_text"]
    assert bodies[2] in context
    assert bodies[1] in context
    assert bodies[3] in context
    assert bodies[0] not in context
    assert bodies[4] not in context


async def test_the_context_window_stops_at_the_document_edge(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="corpus")
        bodies = [
            f"{unique('head')} zymology opens the handbook",
            f"{unique('body')} the second paragraph",
            f"{unique('tail')} the third paragraph",
        ]
        ids = await build_document(
            conn, org_id=org, collection_id=collection, chunks=bodies
        )

        rows = await retrieve(
            conn, "zymology", org_ids=[org], collection_ids=[collection],
        )
        span = await conn.fetchrow(
            "SELECT * FROM pgkg_chunk_window($1::uuid[], 1, 1)", [ids[0]],
        )

    assert [r["item_id"] for r in rows] == [ids[0]]
    assert bodies[2] not in rows[0]["context_text"]
    assert (span["ord_from"], span["ord_to"]) == (0, 1)


async def test_a_chunk_with_no_document_is_its_own_context(
    pool: asyncpg.Pool,
) -> None:
    """The pre-lifecycle ingest path writes chunks with no version links, and a
    passage with no neighbours is still a passage."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="corpus")
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="oenology is the study of wine",
        )

        rows = await retrieve(
            conn, "oenology", org_ids=[org], collection_ids=[collection],
        )

    assert [r["item_id"] for r in rows] == [chunk]
    assert rows[0]["context_text"] == rows[0]["text"]


async def test_retired_passages_leave_retrieval(pool: asyncpg.Pool) -> None:
    """Promotion withdraws the passages the new version dropped, without
    touching a chunk row."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="corpus")
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        await add_version(
            conn, document,
            [f"{unique('v1')} the sabbatical policy grants twelve weeks"],
        )
        replacement = await add_version(
            conn, document,
            [f"{unique('v2')} the sabbatical policy grants eight weeks"],
        )

        rows = await retrieve(
            conn, "sabbatical policy", org_ids=[org],
            collection_ids=[collection],
        )

    assert [r["item_id"] for r in rows] == replacement


async def test_the_corpus_ceiling_binds_without_a_floor(
    pool: asyncpg.Pool,
) -> None:
    """The ceiling alone: with no reserved slots at all, the corpus still
    cannot take more than its share of the budget."""
    async with pool.acquire() as conn:
        org, collections, answer, ns = await personal_and_corpus(conn)

        rows = await retrieve(
            conn, "what did I decide about the refund window",
            q_embedding=vec(0), namespace=ns, org_ids=[org],
            collection_ids=collections, corpus_fraction=0.6, memory_floor=0,
        )

    assert sum(1 for r in rows if r["bucket"] == "corpus") == 12
    assert answer in [r["item_id"] for r in rows]


async def test_the_memory_floor_binds_without_a_ceiling(
    pool: asyncpg.Pool,
) -> None:
    """The floor alone: with the ceiling opened, the reserved slots are still
    reserved.  A floor and a ceiling are different constraints and either one
    on its own has to hold."""
    async with pool.acquire() as conn:
        org, collections, answer, ns = await personal_and_corpus(conn)

        rows = await retrieve(
            conn, "what did I decide about the refund window",
            q_embedding=vec(0), namespace=ns, org_ids=[org],
            collection_ids=collections, corpus_fraction=1.0, memory_floor=8,
        )

    assert sum(1 for r in rows if r["bucket"] == "memory") == 8
    assert answer in [r["item_id"] for r in rows]


async def test_per_scope_weights_reorder_the_admitted_rows(
    pool: asyncpg.Pool,
) -> None:
    """The knob a tenant turns to push general knowledge down without a
    rebuild.  Weights ship at parity, so a difference has to be asked for."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        world = await new_collection(
            conn, org_id=org, kind="corpus", claim_scope="world",
            decay_profile="timeless",
        )
        org_scope = await new_collection(
            conn, org_id=org, kind="corpus", claim_scope="org",
            decay_profile="timeless",
        )
        # Deliberately asymmetric: the shorter passages score strictly higher,
        # so the parity ordering is decided by BM25 rather than by a tie break.
        await seed_corpus(
            conn, org_id=org, collection_id=world, n=20,
            body="refund window",
        )
        await seed_corpus(
            conn, org_id=org, collection_id=org_scope, n=20,
            body="refund window mentioned in a longer paragraph of prose "
                 "about vendor agreements and their many appendices",
        )

        at_parity = await retrieve(
            conn, "refund window", org_ids=[org],
            collection_ids=[world, org_scope],
            corpus_fraction=1.0, memory_floor=0,
        )
        world_down = await retrieve(
            conn, "refund window", org_ids=[org],
            collection_ids=[world, org_scope],
            corpus_fraction=1.0, memory_floor=0, w_scope_world=0.1,
        )

    def best(rows: list[asyncpg.Record], scope: str) -> float:
        return max(
            float(r["adjusted_score"]) for r in rows if r["claim_scope"] == scope
        )

    assert {r["claim_scope"] for r in at_parity} == {"world", "org"}
    assert best(at_parity, "world") > best(at_parity, "org")
    assert best(world_down, "world") == pytest.approx(
        best(at_parity, "world") * 0.1, rel=1e-3
    )
    assert best(world_down, "org") > best(world_down, "world")


async def test_the_window_follows_the_current_version_ordering(
    pool: asyncpg.Pool,
) -> None:
    """A carried-over passage keeps its chunk row across versions, so it is
    linked from the retired ordering as well as the live one.  The window is
    the live one: a retired ordering is not the document's ordering any more."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="corpus")
        carried = f"{unique('carried')} the introduction, unchanged"
        dropped = f"{unique('dropped')} a paragraph the edit removed"
        added = f"{unique('added')} a paragraph the edit introduced"

        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        first = await add_version(conn, document, [carried, dropped])
        await add_version(conn, document, [carried, added])

        windows = await conn.fetch(
            "SELECT * FROM pgkg_chunk_window($1::uuid[], 1, 1)", [first[0]],
        )

    assert len(windows) == 1
    assert added in windows[0]["context_text"]
    assert dropped not in windows[0]["context_text"]


async def test_soft_deleted_documents_leave_retrieval(pool: asyncpg.Pool) -> None:
    """A withdrawn document stops being retrievable long before its rows can be
    physically reclaimed, so the read path is what has to honour deleted_at."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="corpus")
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        ids = await add_version(
            conn, document,
            [
                f"{unique('keep')} the parental leave policy grants six months",
                f"{unique('next')} the following paragraph",
            ],
        )
        before = await retrieve(
            conn, "parental leave", org_ids=[org], collection_ids=[collection],
        )

        await conn.execute(
            "UPDATE documents SET deleted_at = now() WHERE id = $1", document,
        )
        after = await retrieve(
            conn, "parental leave", org_ids=[org], collection_ids=[collection],
        )
        windows = await conn.fetch(
            "SELECT * FROM pgkg_chunk_window($1::uuid[], 1, 1)", [ids[0]],
        )

    assert [r["item_id"] for r in before] == [ids[0]]
    assert after == []
    assert windows == []


async def test_refresh_repairs_drifted_chunk_statistics(
    pool: asyncpg.Pool,
) -> None:
    """The escape hatch 011 argued for, on the new store: the read path cannot
    tell which mechanism populated the tables, so a full recomputation has to
    agree with the incrementally maintained values."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="corpus")
        for i in range(4):
            await insert_chunk(
                conn, org_id=org, collection_id=collection,
                text=f"{unique('para')} a passage about receipts and expenses {i}",
            )
        domain = await chunk_domain(conn, collection)
        maintained = await stats_row(conn, kind="chunk", domain=domain)

        await conn.execute(
            "UPDATE corpus_stats SET n_total = 999, total_len = 99999 "
            "WHERE kind = 'chunk' AND namespace = $1",
            domain,
        )
        await conn.execute("SELECT pgkg_refresh_chunk_stats($1)", collection)
        repaired = await stats_row(conn, kind="chunk", domain=domain)

    assert (maintained["n_total"], maintained["total_len"]) == (
        repaired["n_total"], repaired["total_len"],
    )
    assert repaired["n_total"] == 4
