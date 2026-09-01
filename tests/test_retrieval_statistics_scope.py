"""The statistics domain has a tenant, and its population is what retrieval
returns.

Two rules, both from ADR 0001.  D4's second hard rule: ranking signals are
never computed globally over shared content, because a tenant that watches its
own scores move can measure what other tenants have written about a term.  And
021's rule for the proposition half, which the chunk half now follows too: the
maintenance predicate has to be the retrieval predicate, or BM25 normalises
against a population the caller can never see.
"""
from __future__ import annotations

import hashlib
import uuid

import asyncpg
import pytest


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def new_org(conn: asyncpg.Connection) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("org")
    )


async def new_collection(
    conn: asyncpg.Connection, *, org_id: uuid.UUID, kind: str = "corpus"
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO collections (org_id, owner_org_id, name, kind, claim_scope)
        VALUES ($1, $1, $2, $3, 'org')
        RETURNING id
        """,
        org_id, unique("coll"), kind,
    )


async def new_document(
    conn: asyncpg.Connection, *, org_id: uuid.UUID, collection_id: uuid.UUID
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO documents (source, namespace, org_id, collection_id)
        VALUES ('seed', 'default', $1, $2)
        RETURNING id
        """,
        org_id, collection_id,
    )


async def add_version(
    conn: asyncpg.Connection, *, document_id: uuid.UUID, texts: list[str]
) -> uuid.UUID:
    version = await conn.fetchval(
        "SELECT version_id FROM pgkg_open_document_version($1, $2)",
        document_id,
        hashlib.sha256("|".join(texts).encode()).digest(),
    )
    for ord_, text in enumerate(texts):
        await conn.fetchval(
            "SELECT chunk_id FROM pgkg_add_version_chunk($1, $2, $3)",
            version, ord_, text,
        )
    await conn.execute("SELECT pgkg_promote_document_version($1)", version)
    return version


async def chunk_stats(
    conn: asyncpg.Connection, collection_id: uuid.UUID
) -> tuple[int, int]:
    row = await conn.fetchrow(
        """
        SELECT n_total, total_len FROM corpus_stats
        WHERE kind = 'chunk' AND collection_id = $1
        """,
        collection_id,
    )
    return (0, 0) if row is None else (row["n_total"], row["total_len"])


async def chunk_df(
    conn: asyncpg.Connection, collection_id: uuid.UUID, lexeme: str
) -> int:
    return await conn.fetchval(
        """
        SELECT COALESCE(SUM(df), 0) FROM lexeme_df
        WHERE kind = 'chunk' AND collection_id = $1 AND lexeme = $2
        """,
        collection_id, lexeme,
    ) or 0


# ---------------------------------------------------------------------------
# The population is what retrieval returns
# ---------------------------------------------------------------------------


async def test_a_passage_enters_the_statistics_when_its_version_is_promoted(
    pool: asyncpg.Pool,
) -> None:
    """A chunk is not linked to its version when its own INSERT statement ends,
    so the statistics have to follow the link, not the insert."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(conn, org_id=org, collection_id=collection)

        assert await chunk_stats(conn, collection) == (0, 0)

        await add_version(
            conn, document_id=document,
            texts=["the zorblatt calibration procedure runs monthly"],
        )

        n_total, total_len = await chunk_stats(conn, collection)

    assert n_total == 1
    assert total_len > 0


async def test_a_retired_versions_passages_leave_the_statistics(
    pool: asyncpg.Pool,
) -> None:
    """Retrieval stops returning a passage the moment its version retires, so
    the passage must stop moving the ranking of the ones that replaced it."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(conn, org_id=org, collection_id=collection)

        await add_version(
            conn, document_id=document,
            texts=["the zorblatt calibration procedure runs monthly"],
        )
        await add_version(
            conn, document_id=document,
            texts=["the zorblatt calibration procedure was discontinued"],
        )

        n_total, _ = await chunk_stats(conn, collection)
        df = await chunk_df(conn, collection, "zorblatt")

        retrievable = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE collection_id = $1"
            "   AND retrievable",
            collection,
        )

    assert retrievable == 1, "only the current version's passage is retrievable"
    assert n_total == 1, (
        f"the retired version's passage is still one of {n_total} documents in "
        f"the chunk statistics"
    )
    assert df == 1


async def test_soft_deleting_a_document_withdraws_its_passages_from_the_statistics(
    pool: asyncpg.Pool,
) -> None:
    """POST /documents/delete only sets deleted_at, and retrieval honours it."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(conn, org_id=org, collection_id=collection)
        await add_version(
            conn, document_id=document,
            texts=["the zorblatt calibration procedure runs monthly",
                   "calibration is signed off by the duty engineer"],
        )

        before, _ = await chunk_stats(conn, collection)

        await conn.execute(
            "UPDATE documents SET deleted_at = now() WHERE id = $1", document
        )
        deleted, _ = await chunk_stats(conn, collection)
        deleted_df = await chunk_df(conn, collection, "zorblatt")

        await conn.execute(
            "UPDATE documents SET deleted_at = NULL WHERE id = $1", document
        )
        restored, _ = await chunk_stats(conn, collection)

    assert before == 2
    assert deleted == 0, f"a soft-deleted document still supplies {deleted} rows"
    assert deleted_df == 0
    assert restored == 2, "undeleting has to put the passages back"


async def test_chat_provenance_chunks_never_enter_the_chunk_statistics(
    pool: asyncpg.Pool,
) -> None:
    """Chat ingest writes one chunk per turn so a fact can cite its text. Those
    chunks are not passages and retrieval never returns them."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="mixed")
        chat_document = await new_document(
            conn, org_id=org, collection_id=collection
        )

        await conn.execute(
            """
            INSERT INTO chunks (text, org_id, collection_id, document_id)
            SELECT 'zorblatt came up in turn ' || g, $1, $2, $3
            FROM generate_series(1, 5) g
            """,
            org, collection, chat_document,
        )

        n_total, _ = await chunk_stats(conn, collection)
        df = await chunk_df(conn, collection, "zorblatt")

    assert n_total == 0, f"{n_total} chat turns entered the chunk statistics"
    assert df == 0


# ---------------------------------------------------------------------------
# The domain has a tenant
# ---------------------------------------------------------------------------


async def test_proposition_statistics_are_keyed_per_org(
    pool: asyncpg.Pool,
) -> None:
    """Every tenant's Memory uses one namespace, so the namespace alone cannot
    be the domain (D4)."""
    namespace = unique("ns")
    async with pool.acquire() as conn:
        mine = await new_org(conn)
        theirs = await new_org(conn)
        for org, n in ((mine, 3), (theirs, 11)):
            await conn.execute(
                """
                INSERT INTO propositions (text, namespace, org_id)
                SELECT 'zorblatt is discussed in note ' || g, $1, $2
                FROM generate_series(1, $3) g
                """,
                namespace, org, n,
            )

        rows = {
            r["org_id"]: r["n_total"]
            for r in await conn.fetch(
                """
                SELECT org_id, n_total FROM corpus_stats
                WHERE kind = 'proposition' AND namespace = $1
                """,
                namespace,
            )
        }

    assert rows == {mine: 3, theirs: 11}


async def test_another_tenants_writes_do_not_move_my_document_frequency(
    pool: asyncpg.Pool,
) -> None:
    term = f"zorblatt{uuid.uuid4().hex[:6]}"
    namespace = unique("ns")
    async with pool.acquire() as conn:
        mine = await new_org(conn)
        theirs = await new_org(conn)
        await conn.execute(
            """
            INSERT INTO propositions (text, namespace, org_id)
            SELECT $1 || ' appears in my own note ' || g, $2, $3
            FROM generate_series(1, 3) g
            """,
            term, namespace, mine,
        )

        async def my_df() -> int:
            return await conn.fetchval(
                """
                SELECT df FROM lexeme_df
                WHERE kind = 'proposition' AND namespace = $1 AND org_id = $2
                  AND lexeme = $3
                """,
                namespace, mine, term,
            )

        before = await my_df()
        await conn.execute(
            """
            INSERT INTO propositions (text, namespace, org_id)
            SELECT $1 || ' is discussed at length in note ' || g, $2, $3
            FROM generate_series(1, 40) g
            """,
            term, namespace, theirs,
        )
        after = await my_df()

    assert before == 3
    assert after == 3


async def test_a_full_refresh_agrees_with_the_incremental_maintenance(
    pool: asyncpg.Pool,
) -> None:
    """The escape hatch has to land on the same numbers the triggers did, for
    both halves of the key."""
    namespace = unique("ns")
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(conn, org_id=org, collection_id=collection)
        await add_version(
            conn, document_id=document,
            texts=["the zorblatt calibration procedure runs monthly",
                   "calibration is signed off by the duty engineer"],
        )
        await add_version(
            conn, document_id=document,
            texts=["the zorblatt calibration procedure runs weekly"],
        )
        await conn.execute(
            """
            INSERT INTO propositions (text, namespace, org_id, collection_id)
            SELECT 'zorblatt calibration is a monthly job ' || g, $1, $2, $3
            FROM generate_series(1, 4) g
            """,
            namespace, org, collection,
        )

        maintained_chunks = await chunk_stats(conn, collection)
        maintained_df = await chunk_df(conn, collection, "zorblatt")
        maintained_props = await conn.fetchrow(
            """
            SELECT n_total, total_len FROM corpus_stats
            WHERE kind = 'proposition' AND namespace = $1 AND org_id = $2
            """,
            namespace, org,
        )

        await conn.execute("SELECT pgkg_refresh_chunk_stats($1)", collection)
        await conn.execute("SELECT pgkg_refresh_corpus_stats($1)", namespace)

        refreshed_chunks = await chunk_stats(conn, collection)
        refreshed_df = await chunk_df(conn, collection, "zorblatt")
        refreshed_props = await conn.fetchrow(
            """
            SELECT n_total, total_len FROM corpus_stats
            WHERE kind = 'proposition' AND namespace = $1 AND org_id = $2
            """,
            namespace, org,
        )

    assert maintained_chunks == refreshed_chunks == (1, refreshed_chunks[1])
    assert maintained_df == refreshed_df == 1
    assert tuple(maintained_props) == tuple(refreshed_props)
    assert maintained_props["n_total"] == 4


# ---------------------------------------------------------------------------
# The quota bucket keys on structure
# ---------------------------------------------------------------------------


async def test_a_fact_extracted_from_a_passage_is_corpus_material(
    pool: asyncpg.Pool,
) -> None:
    """D2's opt-in extraction produces facts that are corpus material however
    they are stored, in a mixed collection as much as a corpus one."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="mixed")
        document = await new_document(conn, org_id=org, collection_id=collection)
        await add_version(
            conn, document_id=document,
            texts=["the zorblatt calibration procedure runs monthly"],
        )
        passage = await conn.fetchval(
            "SELECT id FROM chunks WHERE collection_id = $1", collection
        )
        derived = await conn.fetchval(
            """
            INSERT INTO propositions
                (text, namespace, org_id, collection_id, chunk_id)
            VALUES ('zorblatt calibration runs monthly', 'default', $1, $2, $3)
            RETURNING id
            """,
            org, collection, passage,
        )
        chat_fact = await conn.fetchval(
            """
            INSERT INTO propositions (text, namespace, org_id, collection_id)
            VALUES ('I prefer the calibration on a Friday', 'default', $1, $2)
            RETURNING id
            """,
            org, collection,
        )

        buckets = {
            r["item_id"]: r["bucket"]
            for r in await conn.fetch(
                "SELECT item_id, bucket FROM pgkg_item_scope($1::uuid[])",
                [passage, derived, chat_fact],
            )
        }

    assert buckets[passage] == "corpus"
    assert buckets[derived] == "corpus"
    assert buckets[chat_fact] == "memory"


async def test_chat_provenance_chunks_do_not_skew_a_passages_score(
    pool: asyncpg.Pool,
) -> None:
    """The end the population rule exists for: a score that does not move.

    The statistics were maintained over every row of `chunks` while retrieval
    returned only the rows passing the liveness test, so twenty chat turns in
    the collection a corpus document shares — which is the default path, since
    Memory and CorpusIngest both default to the reserved collection — collapsed
    the only retrievable passage's score by an order of magnitude without
    changing one byte of the corpus.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        coll = await new_collection(conn, org_id=org)
        document = await new_document(conn, org_id=org, collection_id=coll)
        await add_version(
            conn, document_id=document, texts=["zorblatt calibration procedure"]
        )
        passage = await conn.fetchval(
            "SELECT id FROM chunks WHERE collection_id = $1", coll
        )

        async def score() -> float | None:
            return await conn.fetchval(
                """
                SELECT raw_score FROM pgkg_bm25_candidates(
                    'zorblatt', 'default', NULL, 200,
                    ARRAY[$1]::UUID[], ARRAY[$2]::UUID[], NULL, NULL, NULL,
                    'chunks'
                ) WHERE item_id = $3
                """,
                org, coll, passage,
            )

        before = await score()
        assert before is not None, "the passage was not retrieved at all"

        # Chat ingest writes one chunk per turn so a fact can cite its text.
        # Those chunks belong to a document and to no version of it, which is
        # what makes them provenance rather than passages.
        chat_doc = await conn.fetchval(
            """
            INSERT INTO documents (source, org_id, collection_id)
            VALUES ('chat', $1, $2) RETURNING id
            """,
            org, coll,
        )
        for turn in range(20):
            await conn.execute(
                """
                INSERT INTO chunks (text, org_id, collection_id, document_id)
                VALUES ($1, $2, $3, $4)
                """,
                f"zorblatt came up in turn {turn} of the conversation",
                org, coll, chat_doc,
            )

        retrievable = [
            row["item_id"]
            for row in await conn.fetch(
                """
                SELECT item_id FROM pgkg_bm25_candidates(
                    'zorblatt', 'default', NULL, 200,
                    ARRAY[$1]::UUID[], ARRAY[$2]::UUID[], NULL, NULL, NULL,
                    'chunks'
                )
                """,
                org, coll,
            )
        ]
        assert retrievable == [passage], (
            "chat chunks became retrievable, which is a different defect"
        )

        after = await score()

    assert after == before, (
        f"the only retrievable passage's BM25 score moved from {before} to "
        f"{after} because 20 non-retrievable chat chunks entered the "
        f"statistics"
    )
