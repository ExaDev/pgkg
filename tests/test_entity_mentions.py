"""The mention edge: where the corpus meets the graph (ADR 0001, D2).

D2 prices three edges between a corpus and a knowledge graph and builds two of
them.  The one this module is about is `entity ← chunk`, populated by gazetteer
matching rather than by an extractor: the org's own entity names and aliases are
matched against each new passage with the indexes `entities` already carries, so
the join costs an index probe per candidate phrase and never a token of LLM.

The payoff is bidirectional.  A chat fact naming *the Helios migration* seeds an
entity, and that entity pulls in the architecture document that defines Helios
*even when the query wording matches that document not at all* — which is the
claim D2 makes and the reason the edge is worth having.  The same edge run the
other way lets a retrieved passage seed the facts about what it mentions.

Both directions cross a trust boundary, so both are re-filtered through the same
visibility predicate as their seed (D3): an entity name is shared inside an org,
and a walk across one must not resurrect a row the seed's own filter would have
withheld.

`entity_links` is the third piece: the bridge from an org's entity space into a
shared collection's, which D4 requires to be a bridge and never a merge.
"""
from __future__ import annotations

import uuid

import asyncpg
import pytest
from pgvector import HalfVector

DIM = 1024

SYSTEM_ORG = uuid.UUID("00000000-0000-0000-0000-000000000000")


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def raw_vec(hot_index: int = 0, value: float = 1.0) -> list[float]:
    raw = [0.0] * DIM
    raw[hot_index] = value
    return raw


def vec(hot_index: int = 0, value: float = 1.0) -> HalfVector:
    return HalfVector(raw_vec(hot_index, value))


async def new_org(conn: asyncpg.Connection) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("org")
    )


async def new_collection(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    kind: str = "corpus",
    visibility: str = "private",
    claim_scope: str = "org",
    owner_org_id: uuid.UUID | None = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO collections
            (org_id, owner_org_id, name, kind, visibility, claim_scope)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """,
        org_id, owner_org_id or org_id, unique("coll"), kind, visibility,
        claim_scope,
    )


async def insert_entity(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    name: str,
    namespace: str = "default",
    aliases: list[str] | None = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO entities (name, type, namespace, org_id, aliases)
        VALUES ($1, 'concept', $2, $3, $4)
        RETURNING id
        """,
        name, namespace, org_id, aliases or [],
    )


async def insert_chunk(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    text: str,
    embedding: HalfVector | None = None,
    visibility: str = "shared",
    owner_user_id: uuid.UUID | None = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO chunks
            (text, org_id, collection_id, embedding, visibility, owner_user_id)
        VALUES ($1, $2, $3, $4::halfvec, $5, $6)
        RETURNING id
        """,
        text, org_id, collection_id, embedding, visibility, owner_user_id,
    )


async def insert_proposition(
    conn: asyncpg.Connection,
    *,
    text: str,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    namespace: str = "default",
    subject_id: uuid.UUID | None = None,
    object_id: uuid.UUID | None = None,
    embedding: HalfVector | None = None,
    visibility: str = "shared",
    owner_user_id: uuid.UUID | None = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO propositions
            (text, namespace, org_id, collection_id, subject_id, object_id,
             embedding, visibility, owner_user_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7::halfvec, $8, $9)
        RETURNING id
        """,
        text, namespace, org_id, collection_id, subject_id, object_id,
        embedding, visibility, owner_user_id,
    )


async def mentions_of(
    conn: asyncpg.Connection, chunk_id: uuid.UUID
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT entity_id, span_start, span_end, match_kind
        FROM entity_mentions WHERE chunk_id = $1
        ORDER BY span_start
        """,
        chunk_id,
    )


async def match(
    conn: asyncpg.Connection, *chunk_ids: uuid.UUID, threshold: float = 0.9
) -> int:
    return await conn.fetchval(
        "SELECT pgkg_match_entity_mentions($1::uuid[], 5, $2)",
        list(chunk_ids), threshold,
    )


# ---------------------------------------------------------------------------
# Gazetteer matching: the org's own names, against the org's own passages
# ---------------------------------------------------------------------------


async def test_gazetteer_matches_an_entity_name_in_a_chunk(
    pool: asyncpg.Pool,
) -> None:
    """The mention edge exists at all, and it points at the text it matched."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        entity = await insert_entity(conn, org_id=org, name="Helios")
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="The Helios platform replaced the legacy stack.",
        )

        added = await match(conn, chunk)
        rows = await mentions_of(conn, chunk)

    assert added == 1
    assert [r["entity_id"] for r in rows] == [entity]
    assert (rows[0]["span_start"], rows[0]["span_end"]) == (4, 10)
    assert rows[0]["match_kind"] == "name"


async def test_gazetteer_matches_a_multiword_name_regardless_of_case(
    pool: asyncpg.Pool,
) -> None:
    """A gazetteer whose unit is the word matches no name made of several."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        entity = await insert_entity(
            conn, org_id=org, name="Helios Migration Programme"
        )
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Our HELIOS migration programme finishes in Q3.",
        )

        await match(conn, chunk)
        rows = await mentions_of(conn, chunk)

    assert [r["entity_id"] for r in rows] == [entity]
    assert (rows[0]["span_start"], rows[0]["span_end"]) == (4, 30)


async def test_gazetteer_matches_an_alias(pool: asyncpg.Pool) -> None:
    """entities.aliases was declared in 002 and never read. It is a gazetteer."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        entity = await insert_entity(
            conn, org_id=org, name="Project Helios", aliases=["HLS", "helios one"]
        )
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="The HLS rollout finished under budget.",
        )

        await match(conn, chunk)
        rows = await mentions_of(conn, chunk)

    assert [r["entity_id"] for r in rows] == [entity]
    assert rows[0]["match_kind"] == "alias"
    assert (rows[0]["span_start"], rows[0]["span_end"]) == (4, 7)


async def test_gazetteer_never_matches_across_the_org_boundary(
    pool: asyncpg.Pool,
) -> None:
    """D3's hard isolation boundary. An entity name is shared inside one org."""
    async with pool.acquire() as conn:
        owner = await new_org(conn)
        stranger = await new_org(conn)
        await insert_entity(conn, org_id=owner, name="Zephyr")
        stranger_collection = await new_collection(conn, org_id=stranger)
        chunk = await insert_chunk(
            conn, org_id=stranger, collection_id=stranger_collection,
            text="Zephyr is the codename of our next release.",
        )

        added = await match(conn, chunk)
        rows = await mentions_of(conn, chunk)

    assert added == 0
    assert rows == []


async def test_gazetteer_ignores_names_too_short_to_be_evidence(
    pool: asyncpg.Pool,
) -> None:
    """A two-character name matches half the English language."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        await insert_entity(conn, org_id=org, name="AI")
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="AI is mentioned in every deck we have ever written.",
        )

        added = await match(conn, chunk)

    assert added == 0


async def test_gazetteer_tolerates_a_typo_at_a_stated_threshold(
    pool: asyncpg.Pool,
) -> None:
    """The trigram index on entities.name is what makes near-misses affordable."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        entity = await insert_entity(conn, org_id=org, name="Helios Platform")
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="The Helios Platfrom shipped on Tuesday.",
        )

        strict = await match(conn, chunk, threshold=0.99)
        assert strict == 0

        await match(conn, chunk, threshold=0.6)
        rows = await mentions_of(conn, chunk)

    assert [r["entity_id"] for r in rows] == [entity]
    assert rows[0]["match_kind"] == "fuzzy"


async def test_matching_the_same_chunk_twice_adds_nothing_and_scans_nothing(
    pool: asyncpg.Pool,
) -> None:
    """The sweep has to be re-runnable, which means bounded on a settled corpus."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        await insert_entity(conn, org_id=org, name="Helios")
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios is the migration programme.",
        )

        first = await match(conn, chunk)
        second = await match(conn, chunk)
        pending = await conn.fetch(
            "SELECT * FROM pgkg_unmatched_chunks($1, 100)", org
        )
        total = await conn.fetchval(
            "SELECT count(*) FROM entity_mentions WHERE chunk_id = $1", chunk
        )

    assert first == 1
    assert second == 0
    assert pending == []
    assert total == 1


async def test_a_new_entity_finds_the_passages_that_already_mention_it(
    pool: asyncpg.Pool,
) -> None:
    """Corpus first, then the chat that names it — which is the ordinary order.

    Matching only at chunk-ingest time would miss every entity the graph learns
    afterwards, and the entity a chat fact creates is exactly that entity.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios is the internal name for the payments rewrite.",
        )
        await match(conn, chunk)
        assert await mentions_of(conn, chunk) == []

        entity = await insert_entity(conn, org_id=org, name="Helios")
        added = await conn.fetchval(
            "SELECT pgkg_match_chunk_mentions($1::uuid[])", [entity]
        )
        rows = await mentions_of(conn, chunk)

    assert added == 1
    assert [r["entity_id"] for r in rows] == [entity]
    assert (rows[0]["span_start"], rows[0]["span_end"]) == (0, 6)


# ---------------------------------------------------------------------------
# The Python surface, and what it costs
# ---------------------------------------------------------------------------


async def test_sweep_visits_each_passage_once_and_calls_no_model(
    pool: asyncpg.Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2 prices this edge at near zero. Two things have to hold for that.

    Nothing in the matching path may reach a language model — not the extractor
    and not the embedder — and a second sweep of a settled corpus may not scan
    the passages the first one already resolved.
    """
    from pgkg import ml
    from pgkg.gazetteer import Gazetteer

    def _forbidden(*args, **kwargs):
        raise AssertionError("gazetteer matching reached a model")

    monkeypatch.setattr(ml, "embed", _forbidden)
    monkeypatch.setattr(ml, "extract_propositions_async", _forbidden)
    monkeypatch.setattr(ml, "rerank", _forbidden)

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        await insert_entity(conn, org_id=org, name="Helios")
        for i in range(3):
            await insert_chunk(
                conn, org_id=org, collection_id=collection,
                text=f"Passage {i} explains why Helios exists.",
            )

    gazetteer = Gazetteer(pool, org_id=org)
    first = await gazetteer.sweep()
    second = await gazetteer.sweep()

    assert first.chunks_scanned == 3
    assert first.mentions_added == 3
    assert second.chunks_scanned == 0
    assert second.mentions_added == 0


async def test_sweep_respects_its_batch_limit(pool: asyncpg.Pool) -> None:
    """A sweep is a scheduled job over a corpus, so its unit of work is a batch."""
    from pgkg.gazetteer import Gazetteer

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        await insert_entity(conn, org_id=org, name="Helios")
        for i in range(5):
            await insert_chunk(
                conn, org_id=org, collection_id=collection,
                text=f"Passage {i} explains why Helios exists.",
            )

    gazetteer = Gazetteer(pool, org_id=org)
    batch = await gazetteer.sweep(limit=2)
    rest = await gazetteer.sweep(limit=100)

    assert batch.chunks_scanned == 2
    assert rest.chunks_scanned == 3


async def test_matching_new_entities_reaches_the_corpus_already_stored(
    pool: asyncpg.Pool,
) -> None:
    """The direction ingest order forces: passages first, names afterwards."""
    from pgkg.gazetteer import Gazetteer

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios is the internal name for the payments rewrite.",
        )
        gazetteer = Gazetteer(pool, org_id=org)
        await gazetteer.sweep()

        entity = await insert_entity(conn, org_id=org, name="Helios")

    result = await gazetteer.match_entities([entity])

    async with pool.acquire() as conn:
        rows = await mentions_of(conn, chunk)

    assert result.mentions_added == 1
    assert [r["entity_id"] for r in rows] == [entity]


async def test_a_name_the_corpus_predates_is_swept_exactly_once(
    pool: asyncpg.Pool,
) -> None:
    """The name side of the sweep, and why it needs a watermark of its own.

    `sweep()` asks which passages have never been matched, so a passage matched
    while the org knew nothing about Helios is finished forever — and every
    entity a chat turn creates afterwards has no way back to it.  Names carry
    the same watermark for the same reason: without one the reverse direction is
    either never run or run over the whole name list on every tick (issue #19).
    """
    from pgkg.gazetteer import Gazetteer

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios is the internal name for the payments rewrite.",
        )

    gazetteer = Gazetteer(pool, org_id=org)
    await gazetteer.sweep()

    async with pool.acquire() as conn:
        entity = await insert_entity(conn, org_id=org, name="Helios")

    first = await gazetteer.sweep_entities()
    second = await gazetteer.sweep_entities()

    async with pool.acquire() as conn:
        rows = await mentions_of(conn, chunk)

    assert first.mentions_added == 1
    assert [r["entity_id"] for r in rows] == [entity]
    assert second.chunks_scanned == 0
    assert second.mentions_added == 0


async def test_a_name_sweep_respects_its_batch_limit(pool: asyncpg.Pool) -> None:
    """One batch of a scheduled sweep, not a whole name list."""
    from pgkg.gazetteer import Gazetteer

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios and Selene both ship this quarter.",
        )
        for name in ("Helios", "Selene", "Artemis"):
            await insert_entity(conn, org_id=org, name=name)

    gazetteer = Gazetteer(pool, org_id=org)
    batch = await gazetteer.sweep_entities(limit=2)
    rest = await gazetteer.sweep_entities(limit=100)

    assert batch.chunks_scanned == 2
    assert rest.chunks_scanned == 1


async def test_an_alias_added_later_puts_the_name_back_in_the_sweep(
    pool: asyncpg.Pool,
) -> None:
    """A watermark that outlives the name it was stamped for is a stale answer.

    Entity resolution keeps adding aliases to rows that already exist, and an
    alias is a new phrase to match — so the stamp is cleared when either key
    the matcher probes changes, and the next sweep re-reads that name.
    """
    from pgkg.gazetteer import Gazetteer

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="The ledger rewrite is tracked as Project Sunrise.",
        )
        entity = await insert_entity(conn, org_id=org, name="Helios")

    gazetteer = Gazetteer(pool, org_id=org)
    await gazetteer.sweep()
    swept = await gazetteer.sweep_entities()
    assert swept.mentions_added == 0

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE entities SET aliases = ARRAY['Project Sunrise'] WHERE id = $1",
            entity,
        )

    after = await gazetteer.sweep_entities()

    async with pool.acquire() as conn:
        rows = await mentions_of(conn, chunk)

    assert after.chunks_scanned == 1
    assert after.mentions_added == 1
    assert [r["match_kind"] for r in rows] == ["alias"]


# ---------------------------------------------------------------------------
# Bidirectional graph expansion, and the re-filter that guards both directions
# ---------------------------------------------------------------------------


def candidate_array(rows: list[tuple[uuid.UUID, str, int, float]]) -> str:
    """A pgkg_candidate[] literal, for calling the expansion arm directly."""
    if not rows:
        return "'{}'::pgkg_candidate[]"
    parts = [
        f"ROW('{item_id}'::uuid, '{kind}', {rank}, {score}::REAL)"
        for item_id, kind, rank, score in rows
    ]
    return "ARRAY[" + ", ".join(parts) + "]::pgkg_candidate[]"


async def expand(
    conn: asyncpg.Connection,
    seeds: list[tuple[uuid.UUID, str, int, float]],
    *,
    namespace: str = "default",
    org_ids: list[uuid.UUID] | None = None,
    collection_ids: list[uuid.UUID] | None = None,
    user_id: uuid.UUID | None = None,
) -> list[asyncpg.Record]:
    return await conn.fetch(
        f"""
        SELECT * FROM pgkg_graph_candidates(
            {candidate_array(seeds)}, $1, 20, 10, 100, $2::uuid[], $3::uuid[],
            $4::uuid, NULL
        )
        """,
        namespace, org_ids, collection_ids, user_id,
    )


async def test_a_fact_seed_pulls_in_the_passages_that_mention_its_entity(
    pool: asyncpg.Pool,
) -> None:
    """D2's claim, at the level of the arm that makes it."""
    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        entity = await insert_entity(
            conn, org_id=org, name="Helios", namespace=namespace
        )
        seed = await insert_proposition(
            conn, text="The Helios migration slipped to Q3.",
            org_id=org, collection_id=collection, namespace=namespace,
            subject_id=entity,
        )
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios is the event-sourced ledger behind settlement.",
        )
        await match(conn, chunk)

        rows = await expand(
            conn, [(seed, "fused", 0, 0.04)], namespace=namespace, org_ids=[org]
        )

    assert [r["item_id"] for r in rows] == [chunk]
    assert rows[0]["kind"] == "graph"


async def test_a_passage_seed_pulls_in_the_facts_about_what_it_mentions(
    pool: asyncpg.Pool,
) -> None:
    """The same edge, walked the other way: a passage seeds its own entities."""
    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        entity = await insert_entity(
            conn, org_id=org, name="Helios", namespace=namespace
        )
        neighbour = await insert_proposition(
            conn, text="The Helios migration slipped to Q3.",
            org_id=org, collection_id=collection, namespace=namespace,
            subject_id=entity,
        )
        seed = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios is the event-sourced ledger behind settlement.",
        )
        await match(conn, seed)

        rows = await expand(
            conn, [(seed, "fused", 0, 0.04)], namespace=namespace, org_ids=[org]
        )

    assert [r["item_id"] for r in rows] == [neighbour]
    assert rows[0]["kind"] == "graph"


async def test_a_passage_seed_cannot_walk_into_another_users_private_fact(
    pool: asyncpg.Pool,
) -> None:
    """D3's hard requirement, on the direction phase 3 opens.

    An entity name is shared inside an org — that is the accepted risk — so a
    passage naming one is a bridge every user in the org can stand on. Every
    row reached across it is re-filtered through the seed's own predicate.
    """
    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        reader = await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
            org, unique("reader"),
        )
        owner = await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
            org, unique("owner"),
        )
        entity = await insert_entity(
            conn, org_id=org, name="Helios", namespace=namespace
        )
        secret = await insert_proposition(
            conn, text="Helios is being cancelled next quarter.",
            org_id=org, collection_id=collection, namespace=namespace,
            subject_id=entity, visibility="private", owner_user_id=owner,
        )
        shared = await insert_proposition(
            conn, text="Helios shipped its first release.",
            org_id=org, collection_id=collection, namespace=namespace,
            subject_id=entity,
        )
        seed = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios is the event-sourced ledger behind settlement.",
        )
        await match(conn, seed)

        rows = await expand(
            conn, [(seed, "fused", 0, 0.04)], namespace=namespace,
            org_ids=[org], user_id=reader,
        )

    reached = {r["item_id"] for r in rows}
    assert secret not in reached
    assert reached == {shared}


async def test_a_fact_seed_cannot_walk_into_another_users_private_passage(
    pool: asyncpg.Pool,
) -> None:
    """The mirror image: a mention edge is not a way around a passage's owner."""
    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        reader = await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
            org, unique("reader"),
        )
        owner = await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
            org, unique("owner"),
        )
        entity = await insert_entity(
            conn, org_id=org, name="Helios", namespace=namespace
        )
        seed = await insert_proposition(
            conn, text="The Helios migration slipped to Q3.",
            org_id=org, collection_id=collection, namespace=namespace,
            subject_id=entity,
        )
        secret = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios exposes the board's severance schedule.",
            visibility="private", owner_user_id=owner,
        )
        shared = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios is the event-sourced ledger behind settlement.",
        )
        await match(conn, secret, shared)

        rows = await expand(
            conn, [(seed, "fused", 0, 0.04)], namespace=namespace,
            org_ids=[org], user_id=reader,
        )

    reached = {r["item_id"] for r in rows}
    assert secret not in reached
    assert reached == {shared}


async def test_a_passage_that_is_not_retrievable_is_not_reachable_either(
    pool: asyncpg.Pool,
) -> None:
    """Expansion may not resurrect what the chunk arm would have withheld."""
    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        entity = await insert_entity(
            conn, org_id=org, name="Helios", namespace=namespace
        )
        seed = await insert_proposition(
            conn, text="The Helios migration slipped to Q3.",
            org_id=org, collection_id=collection, namespace=namespace,
            subject_id=entity,
        )
        document = await conn.fetchval(
            "INSERT INTO documents (source, org_id, collection_id) "
            "VALUES ($1, $2, $3) RETURNING id",
            unique("doc"), org, collection,
        )
        provenance = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="Helios is the event-sourced ledger behind settlement.",
        )
        await conn.execute(
            "UPDATE chunks SET document_id = $1 WHERE id = $2", document, provenance
        )
        await match(conn, provenance)

        rows = await expand(
            conn, [(seed, "fused", 0, 0.04)], namespace=namespace, org_ids=[org]
        )

    assert rows == []


# ---------------------------------------------------------------------------
# entity_links: a bridge into shared entity space, never a merge
# ---------------------------------------------------------------------------


async def link_entities(
    conn: asyncpg.Connection,
    *,
    org_entity_id: uuid.UUID,
    shared_entity_id: uuid.UUID,
    confidence: float = 1.0,
) -> None:
    await conn.execute(
        """
        INSERT INTO entity_links (org_entity_id, shared_entity_id, confidence)
        VALUES ($1, $2, $3)
        """,
        org_entity_id, shared_entity_id, confidence,
    )


async def test_a_link_into_shared_space_is_refused_when_it_is_really_a_merge(
    pool: asyncpg.Pool,
) -> None:
    """D4: don't merge shared entities into per-org entity space.

    A rule that lives only in an ADR is not a constraint, and the shape a merge
    would take here — two rows in one org tied together — is the shape the
    trigger refuses.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        mine = await insert_entity(conn, org_id=org, name="Helios")
        also_mine = await insert_entity(conn, org_id=org, name="Helios Programme")

        with pytest.raises(asyncpg.RaiseError):
            await link_entities(
                conn, org_entity_id=mine, shared_entity_id=also_mine
            )


async def test_a_bridged_entity_reaches_shared_material_without_being_copied(
    pool: asyncpg.Pool,
) -> None:
    """The bridge D4 asks for: the org keeps its entity, the operator keeps theirs."""
    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        own_collection = await new_collection(conn, org_id=org, kind="chat")
        shared_collection = await new_collection(
            conn, org_id=SYSTEM_ORG, kind="corpus", visibility="shared",
            claim_scope="world",
        )

        mine = await insert_entity(
            conn, org_id=org, name="Helios", namespace=namespace
        )
        theirs = await insert_entity(
            conn, org_id=SYSTEM_ORG, name="Helios", namespace=namespace
        )
        await link_entities(conn, org_entity_id=mine, shared_entity_id=theirs)

        seed = await insert_proposition(
            conn, text="The Helios migration slipped to Q3.",
            org_id=org, collection_id=own_collection, namespace=namespace,
            subject_id=mine,
        )
        reference = await insert_chunk(
            conn, org_id=SYSTEM_ORG, collection_id=shared_collection,
            text=f"Helios is an open event-sourcing standard. {namespace}",
        )
        await match(conn, reference)

        rows = await expand(
            conn, [(seed, "fused", 0, 0.04)], namespace=namespace,
            org_ids=[org, SYSTEM_ORG],
        )
        own_entities = await conn.fetchval(
            "SELECT count(*) FROM entities WHERE org_id = $1 AND namespace = $2",
            org, namespace,
        )

    assert [r["item_id"] for r in rows] == [reference]
    assert own_entities == 1, "the shared entity must not be copied into the org"


async def test_the_bridge_carries_no_visibility_of_its_own(
    pool: asyncpg.Pool,
) -> None:
    """A link is a way to reach subscribed material, not a way to subscribe."""
    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        own_collection = await new_collection(conn, org_id=org, kind="chat")
        shared_collection = await new_collection(
            conn, org_id=SYSTEM_ORG, kind="corpus", visibility="shared",
            claim_scope="world",
        )

        mine = await insert_entity(
            conn, org_id=org, name="Helios", namespace=namespace
        )
        theirs = await insert_entity(
            conn, org_id=SYSTEM_ORG, name="Helios", namespace=namespace
        )
        await link_entities(conn, org_entity_id=mine, shared_entity_id=theirs)

        seed = await insert_proposition(
            conn, text="The Helios migration slipped to Q3.",
            org_id=org, collection_id=own_collection, namespace=namespace,
            subject_id=mine,
        )
        reference = await insert_chunk(
            conn, org_id=SYSTEM_ORG, collection_id=shared_collection,
            text=f"Helios is an open event-sourcing standard. {namespace}",
        )
        await match(conn, reference)

        unsubscribed = await expand(
            conn, [(seed, "fused", 0, 0.04)], namespace=namespace, org_ids=[org]
        )

    assert unsubscribed == []


# ---------------------------------------------------------------------------
# The claim the whole design rests on
# ---------------------------------------------------------------------------


async def test_a_chat_fact_pulls_in_the_document_that_defines_what_it_names(
    pool: asyncpg.Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2, end to end, over the real ingest pipelines.

    A handbook page defines Helios in wording the query never uses. A chat turn
    names Helios and nothing else the page says. The query is asked in the chat's
    words, so the page loses every lexical and vector arm there is — and comes
    back anyway, because the fact seeded an entity and the entity is mentioned
    in the page.

    The control arm is the same query with expansion off: without the mention
    edge the page is unreachable, which is what makes the first assertion mean
    something.
    """
    from pgkg import ml
    from pgkg.corpus import CorpusIngest
    from pgkg.gazetteer import Gazetteer
    from pgkg.memory import Memory, Scope
    from pgkg.ml import Proposition

    namespace = unique("ns")
    definition = (
        "Helios is the event-sourced ledger behind interbank settlement. "
        "Every transfer is stored as an immutable append-only record."
    )
    chat_turn = "The Helios migration slipped to Q3."
    query = "why did the migration slip"

    async def _extract(chunk_text: str, *, max_propositions: int = 20, cache=None):
        return [
            Proposition(
                text=chat_turn,
                subject="Helios",
                predicate="migration slipped to",
                object="Q3",
                object_is_literal=True,
            )
        ]

    monkeypatch.setattr(ml, "embed", lambda texts: [raw_vec(1) for _ in texts])
    monkeypatch.setattr(ml, "extract_propositions_async", _extract)

    async with pool.acquire() as conn:
        org = await new_org(conn)
        chat_collection = await new_collection(conn, org_id=org, kind="chat")
        corpus_collection = await new_collection(conn, org_id=org, kind="corpus")

    corpus = CorpusIngest(
        pool, org_id=org, collection_id=corpus_collection,
        embed=lambda texts: [raw_vec(2) for _ in texts],
    )
    await corpus.upsert_document(external_id=unique("handbook"), text=definition)

    memory = Memory(
        pool,
        namespace=namespace,
        scope=Scope(org_id=org, collection_id=chat_collection),
        use_extract_cache=False,
    )
    await memory.ingest(chat_turn)

    swept = await Gazetteer(pool, org_id=org).sweep()
    assert swept.mentions_added >= 1, "the gazetteer found no name to join on"

    async def retrieve(*, expand_graph: bool) -> list[asyncpg.Record]:
        async with pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT * FROM pgkg_retrieve(
                    q_text           => $1,
                    p_namespace      => $2,
                    expand_graph     => $3,
                    p_org_ids        => $4::uuid[],
                    p_collection_ids => $5::uuid[]
                )
                """,
                query, namespace, expand_graph, [org],
                [chat_collection, corpus_collection],
            )

    joined = await retrieve(expand_graph=True)
    isolated = await retrieve(expand_graph=False)

    passages = [r for r in joined if r["source"] == "chunks"]
    assert [r["text"] for r in passages] == [definition]
    assert passages[0]["source_kind"] == "graph"
    assert any(r["text"] == chat_turn for r in joined)

    assert [r["source"] for r in isolated] == ["propositions"], (
        "without the mention edge the passage shares no term with the query"
    )


# ---------------------------------------------------------------------------
# What the mention edge must not cost the two surfaces that walk it
# ---------------------------------------------------------------------------

async def test_graph_expansion_refilters_in_both_directions(
    pool: asyncpg.Pool,
) -> None:
    """D3's hard requirement, checked on both stores at once.

    One org, two users, one shared entity.  User B's seed is a shared fact
    naming it; user A owns a private fact about it and a private passage
    mentioning it, the latter reachable only through the edge this module
    builds.  Neither may come back for B.  The non-vacuity check is the same
    query as A, which returns both — so the walk does reach them and the
    visibility predicate in each arm is what stops it.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        user_a = await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
            org, unique("a"),
        )
        user_b = await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
            org, unique("b"),
        )
        entity = await insert_entity(
            conn, org_id=org, name="Zzqhelios Programme"
        )

        await insert_proposition(
            conn, org_id=org, collection_id=collection, subject_id=entity,
            text="Zzqhelios Programme kickoff meeting was minuted",
        )
        await insert_proposition(
            conn, org_id=org, collection_id=collection, subject_id=entity,
            text="PRIVATE_FACT Zzqhelios Programme budget overrun is nine million",
            visibility="private", owner_user_id=user_a,
        )

        document = await conn.fetchval(
            "INSERT INTO documents (source, org_id, collection_id, external_id)"
            " VALUES ('probe', $1, $2, $3) RETURNING id",
            org, collection, unique("ext"),
        )
        version = await conn.fetchval(
            "SELECT version_id FROM pgkg_open_document_version($1, $2)",
            document, uuid.uuid4().bytes,
        )
        chunk = await conn.fetchval(
            "SELECT chunk_id FROM pgkg_add_version_chunk($1, 0, $2)",
            version,
            "PRIVATE_PASSAGE the Zzqhelios Programme steering notes, restricted",
        )
        await conn.execute("SELECT pgkg_promote_document_version($1)", version)
        await conn.execute(
            "UPDATE chunks SET visibility = 'private', owner_user_id = $1"
            " WHERE id = $2",
            user_a, chunk,
        )
        assert await match(conn, chunk) == 1, (
            "the mention edge was not built, so the walk is vacuous"
        )

        async def texts_for(user: uuid.UUID) -> str:
            rows = await conn.fetch(
                """
                SELECT text FROM pgkg_retrieve(
                    q_text => 'Zzqhelios Programme kickoff minuted',
                    k_retrieve => 50,
                    expand_graph => TRUE,
                    p_org_ids => $1::uuid[],
                    p_collection_ids => $2::uuid[],
                    p_user_id => $3::uuid
                )
                """,
                [org], [collection], user,
            )
            return " | ".join(row["text"] for row in rows)

        stranger = await texts_for(user_b)
        owner = await texts_for(user_a)

    assert "PRIVATE_FACT" not in stranger, stranger
    assert "PRIVATE_PASSAGE" not in stranger, stranger
    assert "PRIVATE_FACT" in owner and "PRIVATE_PASSAGE" in owner, (
        f"the walk never reached the private rows at all: {owner}"
    )


async def test_mentions_do_not_displace_facts_in_pgkg_search(
    pool: asyncpg.Pool,
) -> None:
    """pgkg_search() is proposition-shaped, and the graph arm now emits
    passages as well as facts.

    The chunk candidates were discarded only after they had taken places in the
    arm's own budget, so a proposition-only search lost 42% of its facts the
    moment gazetteer mentions existed and returned nothing in their place.  A
    cap is a budget: the caller says which stores it can consume, and the arm
    spends the budget on those.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        await conn.execute("SELECT set_config('pgkg.org_id', $1, false)", str(org))
        collection = await new_collection(conn, org_id=org, kind="mixed")
        namespace = unique("mention_ns")

        edges: list[tuple[uuid.UUID, uuid.UUID]] = []
        for i in range(20):
            entity = await insert_entity(
                conn, org_id=org, name=f"zorbulon{i}", namespace=namespace
            )
            # One seed per entity, so every entity is a graph seed.
            await insert_proposition(
                conn, org_id=org, collection_id=collection, namespace=namespace,
                subject_id=entity, text="zorbulon reconciles the ledger",
            )
            await conn.execute(
                """
                INSERT INTO propositions
                    (text, namespace, subject_id, org_id, collection_id)
                SELECT 'derived fact ' || g, $1, $2, $3, $4
                FROM generate_series(1, 12) g
                """,
                namespace, entity, org, collection,
            )
            rows = await conn.fetch(
                """
                INSERT INTO chunks (text, span_start, span_end, org_id,
                                    collection_id)
                SELECT 'a passage naming ' || $3 || ', number ' || g, 0, 30,
                       $1, $2
                FROM generate_series(1, 12) g
                RETURNING id
                """,
                org, collection, str(entity),
            )
            edges.extend((entity, row["id"]) for row in rows)

        async def facts_found() -> int:
            rows = await conn.fetch(
                """
                SELECT proposition_id FROM pgkg_search(
                    'zorbulon', NULL, 400, 400, $1, NULL, 30.0, TRUE, 60,
                    $2::uuid[], $3::uuid[])
                """,
                namespace, [org], [collection],
            )
            return len(rows)

        before = await facts_found()
        await conn.executemany(
            """
            INSERT INTO entity_mentions (entity_id, chunk_id, org_id,
                                         span_start, span_end)
            VALUES ($1, $2, $3, 0, 8)
            """,
            [(entity, chunk, org) for entity, chunk in edges],
        )
        after = await facts_found()

    assert before > 0, "the seeds were not retrievable at all"
    assert after == before, (
        f"pgkg_search() returned {before} propositions before the passages "
        f"were mentioned and {after} after; the passages themselves are not "
        f"part of its contract"
    )
