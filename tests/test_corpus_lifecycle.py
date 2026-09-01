"""Document versions, content-addressed chunks, and the subscription seam.

D6 makes re-ingest cheap by hashing twice: once for the document, so an
unchanged nightly crawl does no work at all, and once per chunk, so a typo
fixed in a 300-page handbook is one embedding call rather than 300.  That only
holds if a chunk row is identified by its content rather than by its position
in a document, which is what UNIQUE (org_id, content_hash) and the
document_version_chunks many-to-many buy.

D4's ownership seam is exercised here too, and only as a capability: the
resolution function widens a caller's org list when — and only when — a
subscription row says so, and nothing is subscribed by default.
"""
from __future__ import annotations

import hashlib
import re
import uuid

import asyncpg
import pytest
from pgvector import HalfVector

SYSTEM_ORG = uuid.UUID("00000000-0000-0000-0000-000000000000")


def sha256(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def new_org(conn: asyncpg.Connection) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("org")
    )


async def new_collection(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    owner_org_id: uuid.UUID | None = None,
    visibility: str = "private",
    kind: str = "corpus",
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO collections
            (org_id, owner_org_id, name, kind, visibility)
        VALUES ($1::uuid, COALESCE($2::uuid, $1::uuid), $3, $4, $5)
        RETURNING id
        """,
        org_id,
        owner_org_id,
        unique("coll"),
        kind,
        visibility,
    )


async def new_document(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    external_id: str | None = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO documents (source, namespace, org_id, collection_id, external_id)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """,
        unique("src"),
        unique("ns"),
        org_id,
        collection_id,
        external_id,
    )


async def ingest_version(
    conn: asyncpg.Connection, document_id: uuid.UUID, chunks: list[str]
) -> tuple[uuid.UUID, bool, list[tuple[uuid.UUID, bool]]]:
    """The lifecycle as a caller uses it: open, add chunks, promote.

    Returns the version, whether it was new, and one (chunk_id, is_new) pair
    per chunk — is_new being exactly the signal that says "embed this one".
    """
    document_hash = sha256("".join(chunks))
    version = await conn.fetchrow(
        "SELECT * FROM pgkg_open_document_version($1, $2)",
        document_id,
        document_hash,
    )
    if not version["is_new"]:
        return version["version_id"], False, []

    added = [
        await conn.fetchrow(
            "SELECT * FROM pgkg_add_version_chunk($1, $2, $3)",
            version["version_id"],
            ord_,
            text,
        )
        for ord_, text in enumerate(chunks)
    ]
    await conn.execute(
        "SELECT pgkg_promote_document_version($1)", version["version_id"]
    )
    return (
        version["version_id"],
        True,
        [(row["chunk_id"], row["is_new"]) for row in added],
    )


async def chunk_count(conn: asyncpg.Connection, org_id: uuid.UUID) -> int:
    return await conn.fetchval(
        "SELECT count(*) FROM chunks WHERE org_id = $1", org_id
    )


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


async def test_the_first_version_is_numbered_and_becomes_current(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )

        version, is_new, _ = await ingest_version(conn, document, ["alpha beta"])
        row = await conn.fetchrow(
            """
            SELECT dv.version_no, dv.status, dv.retired_at, dv.org_id,
                   d.current_version_id
            FROM document_versions dv
            JOIN documents d ON d.id = dv.document_id
            WHERE dv.id = $1
            """,
            version,
        )

    assert is_new is True
    assert row["version_no"] == 1
    assert row["status"] == "current"
    assert row["retired_at"] is None
    assert row["org_id"] == org
    assert row["current_version_id"] == version


async def test_re_ingesting_identical_content_is_a_no_op(
    pool: asyncpg.Pool,
) -> None:
    """The nightly full crawl of an unchanged corpus: no version, no chunk, no
    embedding call.  is_new = FALSE is what tells the caller to stop."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        chunks = ["the handbook opens", "the handbook closes"]

        first, _, _ = await ingest_version(conn, document, chunks)
        chunks_after_first = await chunk_count(conn, org)

        second, is_new, added = await ingest_version(conn, document, chunks)
        versions = await conn.fetchval(
            "SELECT count(*) FROM document_versions WHERE document_id = $1",
            document,
        )

    assert is_new is False
    assert second == first
    assert added == []
    assert versions == 1
    assert chunks_after_first == 2


async def test_an_edited_document_re_uses_unchanged_chunks(
    pool: asyncpg.Pool,
) -> None:
    """A typo fixed in the middle chunk re-embeds one chunk, not three."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )

        _, _, first = await ingest_version(
            conn, document, ["opening para", "middle para", "closing para"]
        )
        before = await chunk_count(conn, org)

        _, _, second = await ingest_version(
            conn, document, ["opening para", "middle parra", "closing para"]
        )
        after = await chunk_count(conn, org)

    assert [row[1] for row in first] == [True, True, True]
    assert [row[1] for row in second] == [False, True, False]
    assert after - before == 1
    assert second[0][0] == first[0][0]
    assert second[2][0] == first[2][0]
    assert second[1][0] != first[1][0]


async def test_a_carried_chunk_is_shared_not_copied(pool: asyncpg.Pool) -> None:
    """Two versions naming the same content name the same row, and the refcount
    says how many links point at it."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )

        _, _, first = await ingest_version(conn, document, ["kept", "dropped"])
        await ingest_version(conn, document, ["kept", "replaced"])

        kept, dropped = first[0][0], first[1][0]
        refcounts = dict(
            await conn.fetch(
                "SELECT id, refcount FROM chunks WHERE id = ANY($1::uuid[])",
                [kept, dropped],
            )
        )

    assert refcounts[kept] == 2
    assert refcounts[dropped] == 1


async def test_boilerplate_is_one_chunk_row_per_org(pool: asyncpg.Pool) -> None:
    """Reference counting deduplicates boilerplate across documents in the same
    org, and never across orgs: a shared row could not live in a tenant
    partition (D4)."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        first_doc = await new_document(
            conn, org_id=org, collection_id=collection
        )
        second_doc = await new_document(
            conn, org_id=org, collection_id=collection
        )
        boilerplate = "confidential do not distribute"

        _, _, first = await ingest_version(conn, first_doc, [boilerplate])
        _, _, second = await ingest_version(
            conn, second_doc, [boilerplate, "and something else"]
        )

        other_org = await new_org(conn)
        other_collection = await new_collection(conn, org_id=other_org)
        other_doc = await new_document(
            conn, org_id=other_org, collection_id=other_collection
        )
        _, _, elsewhere = await ingest_version(conn, other_doc, [boilerplate])

        refcount = await conn.fetchval(
            "SELECT refcount FROM chunks WHERE id = $1", first[0][0]
        )

    assert second[0][0] == first[0][0]
    assert second[0][1] is False
    assert refcount == 2
    assert elsewhere[0][0] != first[0][0]
    assert elsewhere[0][1] is True


# ---------------------------------------------------------------------------
# Chunks are immutable and content-addressed
# ---------------------------------------------------------------------------


async def test_a_chunk_cannot_be_rewritten(pool: asyncpg.Pool) -> None:
    """Content addressing is a lie the moment text can change under a hash."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        _, _, added = await ingest_version(conn, document, ["as written"])

        with pytest.raises(asyncpg.RaiseError):
            async with conn.transaction():
                await conn.execute(
                    "UPDATE chunks SET text = 'rewritten' WHERE id = $1",
                    added[0][0],
                )


async def test_the_content_hash_is_derived_not_supplied(
    pool: asyncpg.Pool,
) -> None:
    """A caller that could write the hash could write the wrong one."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        _, _, added = await ingest_version(conn, document, ["derive me"])

        stored = await conn.fetchval(
            "SELECT content_hash FROM chunks WHERE id = $1", added[0][0]
        )

        with pytest.raises(asyncpg.GeneratedAlwaysError):
            async with conn.transaction():
                await conn.execute(
                    "UPDATE chunks SET content_hash = $1 WHERE id = $2",
                    sha256("something else"),
                    added[0][0],
                )

    assert stored == sha256("derive me")


async def test_chunks_are_retrievable_in_their_own_right(
    pool: asyncpg.Pool,
) -> None:
    """D2: the corpus is retrieved as chunks by default, so a chunk needs the
    same two indexes a proposition has."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        _, _, added = await ingest_version(
            conn, document, ["mitochondria generate adenosine triphosphate"]
        )
        chunk_id = added[0][0]

        dim = await conn.fetchval(
            "SELECT pgkg_embedding_dim('chunks', 'embedding')"
        )
        embedding = HalfVector([0.1] * dim)
        await conn.execute(
            "UPDATE chunks SET embedding = $1::halfvec WHERE id = $2",
            embedding,
            chunk_id,
        )

        row = await conn.fetchrow(
            """
            SELECT c.doc_len,
                   c.tsv @@ plainto_tsquery('english', 'mitochondria') AS matches,
                   (c.embedding <=> $2::halfvec) AS distance
            FROM chunks c
            WHERE c.id = $1
            """,
            chunk_id,
            embedding,
        )

    assert dim > 0
    assert row["matches"] is True
    assert row["doc_len"] > 0
    assert row["distance"] == pytest.approx(0.0, abs=1e-3)


async def test_a_chunk_inherits_its_tenancy_from_its_document(
    pool: asyncpg.Pool,
) -> None:
    """Derived, not passed: a chunk that disagreed with its document about org
    or collection would be invisible to retrieval or visible to the wrong
    tenant.  Provenance comes from the version, because that is the ingest run
    that produced this text, and the ACL group is the one thing the caller
    knows and the schema cannot derive."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        provenance = await conn.fetchval(
            "INSERT INTO provenance (org_id, kind, producer)"
            " VALUES ($1, 'document_version', 'chunker') RETURNING id",
            org,
        )
        acl_group = uuid.uuid4()

        version = await conn.fetchrow(
            "SELECT * FROM pgkg_open_document_version($1, $2, $3)",
            document,
            sha256("first"),
            provenance,
        )
        added = await conn.fetchrow(
            "SELECT * FROM pgkg_add_version_chunk($1, $2, $3, $4)",
            version["version_id"],
            0,
            "first",
            acl_group,
        )
        row = await conn.fetchrow(
            "SELECT org_id, collection_id, document_id, provenance_id,"
            " acl_group_id FROM chunks WHERE id = $1",
            added["chunk_id"],
        )

    assert row["org_id"] == org
    assert row["collection_id"] == collection
    assert row["document_id"] is None
    assert row["provenance_id"] == provenance
    assert row["acl_group_id"] == acl_group


async def test_adding_a_chunk_to_an_unknown_version_is_an_error(
    pool: asyncpg.Pool,
) -> None:
    """Silently inserting an unparented chunk would leave a row nothing links
    and GC eventually eats."""
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.RaiseError):
            await conn.fetchrow(
                "SELECT * FROM pgkg_add_version_chunk($1, 0, 'orphan')",
                uuid.uuid4(),
            )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


async def test_an_external_id_identifies_a_document_within_a_collection(
    pool: asyncpg.Pool,
) -> None:
    """A connector re-crawls by the customer's own id, so that id has to be the
    thing a second crawl collides on."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        await new_document(
            conn, org_id=org, collection_id=collection, external_id="handbook"
        )

        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await new_document(
                    conn,
                    org_id=org,
                    collection_id=collection,
                    external_id="handbook",
                )

        elsewhere = await new_collection(conn, org_id=org)
        assert await new_document(
            conn, org_id=org, collection_id=elsewhere, external_id="handbook"
        )


async def test_documents_without_an_external_id_still_ingest(
    pool: asyncpg.Pool,
) -> None:
    """Every pre-lifecycle document has no external id, and two of them are not
    a collision."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        first = await new_document(conn, org_id=org, collection_id=collection)
        second = await new_document(conn, org_id=org, collection_id=collection)
        row = await conn.fetchrow(
            "SELECT current_version_id, deleted_at FROM documents WHERE id = $1",
            first,
        )

    assert first != second
    assert row["current_version_id"] is None
    assert row["deleted_at"] is None


async def test_a_pre_lifecycle_chunk_is_not_content_addressed(
    pool: asyncpg.Pool,
) -> None:
    """The content address governs retrievable content, and the pre-lifecycle
    path writes provenance: a whole document's chunks go in one statement with
    no conflict handling, so a repeated paragraph there is still two rows.

    052 moved the index off the parent pointer and onto the statement that
    replaced it, and the bridge is what keeps this writer — which states the
    pointer and nothing else — outside the address."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        repeated = "the same paragraph twice"

        await conn.execute(
            """
            INSERT INTO chunks (document_id, text, org_id, collection_id)
            SELECT $1, $2, $3, $4 FROM generate_series(1, 2)
            """,
            document,
            repeated,
            org,
            collection,
        )
        legacy = await conn.fetchval(
            "SELECT count(*) FROM chunks"
            " WHERE org_id = $1 AND document_id IS NOT NULL",
            org,
        )

        other = await new_document(conn, org_id=org, collection_id=collection)
        _, _, first = await ingest_version(conn, other, [repeated])
        _, _, second = await ingest_version(conn, other, [repeated, "and more"])

    assert legacy == 2
    assert second[0][0] == first[0][0]


# ---------------------------------------------------------------------------
# Provenance is stated, not inferred from parentage (052, issue #18)
# ---------------------------------------------------------------------------


async def insert_chunk(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    text: str,
    document_id: uuid.UUID | None = None,
    provenance_only: bool = False,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO chunks
            (text, org_id, collection_id, document_id, provenance_only)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """,
        text,
        org_id,
        collection_id,
        document_id,
        provenance_only,
    )


async def chunk_flags(
    conn: asyncpg.Connection, chunk_id: uuid.UUID
) -> tuple[bool, bool, bool]:
    row = await conn.fetchrow(
        "SELECT retrievable, provenance_only, version_scoped"
        " FROM chunks WHERE id = $1",
        chunk_id,
    )
    return row["retrievable"], row["provenance_only"], row["version_scoped"]


async def test_a_withheld_passage_needs_no_parent_pointer_to_be_withheld(
    pool: asyncpg.Pool,
) -> None:
    """The whole point of 052.

    A passage that exists as provenance for the facts extracted from it is not
    retrievable content (041), and until now the only way to say so was to point
    it at one document — which is what makes a total content address
    unrepresentable (#18).  The writer says so instead, and the row names no
    document at all."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        withheld = await insert_chunk(
            conn,
            org_id=org,
            collection_id=collection,
            text="the operator said the ledger reconciles nightly",
            provenance_only=True,
        )
        content = await insert_chunk(
            conn,
            org_id=org,
            collection_id=collection,
            text="the handbook says the ledger reconciles nightly",
        )
        document_id = await conn.fetchval(
            "SELECT document_id FROM chunks WHERE id = $1", withheld
        )
        withheld_flags = await chunk_flags(conn, withheld)
        content_flags = await chunk_flags(conn, content)
        counted = await conn.fetchval(
            "SELECT n_total FROM corpus_stats"
            " WHERE kind = 'chunk' AND org_id = $1 AND collection_id = $2",
            org,
            collection,
        )

    assert document_id is None
    assert withheld_flags == (False, True, False)
    assert content_flags == (True, False, False)
    assert counted == 1


async def test_the_content_address_covers_retrievable_content_only(
    pool: asyncpg.Pool,
) -> None:
    """Which rows the address governs is as much part of it as which columns key
    it, and the rows it governs are the retrievable ones.

    A provenance row must never be reused by another writer, and the extraction
    path repeats a paragraph as two rows on purpose — each carries its own span
    and its own derivation record.  So provenance rows are outside the index,
    which before 052 was said as `document_id IS NULL` and is now said in the
    terms that decide it."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        repeated = "the refund window is thirty days"

        first = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text=repeated, provenance_only=True,
        )
        second = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text=repeated, provenance_only=True,
        )

        document = await new_document(conn, org_id=org, collection_id=collection)
        _, _, added = await ingest_version(conn, document, [repeated, repeated])

        predicate = await conn.fetchval(
            "SELECT pg_get_expr(i.indpred, i.indrelid) FROM pg_index i"
            " JOIN pg_class ic ON ic.oid = i.indexrelid"
            " WHERE ic.relname = 'chunks_content_addressed_key'"
        )

    assert first != second
    assert added[0][0] == added[1][0]
    assert added[0][0] not in (first, second)
    assert "document_id" not in predicate
    assert "provenance_only" in predicate


async def test_a_carried_passage_stays_retrievable_when_it_states_provenance(
    pool: asyncpg.Pool,
) -> None:
    """provenance_only guards the standalone arm, in the position the pointer
    held — it is not an absolute veto.

    045's predicate is "carried by the current version of a live document, OR
    standalone", and `document_id IS NULL` guarded the second arm only.  Keeping
    the guard in that position is what makes 052 a substitution rather than a
    change of answer: a row that states provenance and is also carried by a live
    current version is retrievable today and stays retrievable."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(conn, org_id=org, collection_id=collection)
        version = await conn.fetchval(
            "SELECT version_id FROM pgkg_open_document_version($1, $2)",
            document,
            sha256("carried"),
        )
        chunk = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="the operator publishes the ledger every night",
            provenance_only=True,
        )
        before = await chunk_flags(conn, chunk)

        await conn.execute(
            "INSERT INTO document_version_chunks"
            " (document_version_id, chunk_id, ord) VALUES ($1, $2, 0)",
            version,
            chunk,
        )
        await conn.execute("SELECT pgkg_promote_document_version($1)", version)
        after = await chunk_flags(conn, chunk)

    assert before == (False, True, False)
    assert after == (True, True, True)


async def test_naming_one_document_is_translated_into_the_statement(
    pool: asyncpg.Pool,
) -> None:
    """The bridge, which is why 052 needs no writer to change at the same time.

    A migration is forward-only and cannot reach the callers, so a row that
    still states the pointer and nothing else has to keep the answer the pointer
    used to give it.  The bridge only ever sets TRUE, so it cannot make a
    withheld row retrievable, and its WHEN clause keeps every writer that has
    moved off the pointer out of the function."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(conn, org_id=org, collection_id=collection)

        legacy = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="a chat turn about the ledger", document_id=document,
        )
        translated = await chunk_flags(conn, legacy)

        acquired = await insert_chunk(
            conn, org_id=org, collection_id=collection,
            text="a standalone note about the ledger",
        )
        stood_alone = await chunk_flags(conn, acquired)
        await conn.execute(
            "UPDATE chunks SET document_id = $1 WHERE id = $2",
            document,
            acquired,
        )
        withdrawn = await chunk_flags(conn, acquired)

    assert translated == (False, True, False)
    assert stood_alone == (True, False, False)
    assert withdrawn == (False, True, False)


async def test_changing_the_statement_moves_the_derived_flag_and_the_statistics(
    pool: asyncpg.Pool,
) -> None:
    """provenance_only is an input and retrievable is derived from it, so the
    trigger set that maintains the derived one has to watch the input.

    041 narrowed the chunks UPDATE trigger to "the only column on the row that
    changes the answer and is not already covered by the link trigger", which
    was the parent pointer and is now this.  The BM25 statistics move with the
    flag, because a withheld passage that still counts toward n_total inflates
    every other passage's IDF."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        chunk = await insert_chunk(
            conn,
            org_id=org,
            collection_id=collection,
            text="the settlement window is reconciled by the operator",
            provenance_only=True,
        )

        async def counted() -> int:
            return await conn.fetchval(
                "SELECT COALESCE(SUM(n_total), 0) FROM corpus_stats"
                " WHERE kind = 'chunk' AND org_id = $1 AND collection_id = $2",
                org,
                collection,
            )

        withheld = (await chunk_flags(conn, chunk), await counted())

        await conn.execute(
            "UPDATE chunks SET provenance_only = FALSE WHERE id = $1", chunk
        )
        published = (await chunk_flags(conn, chunk), await counted())

        await conn.execute(
            "UPDATE chunks SET provenance_only = TRUE WHERE id = $1", chunk
        )
        withdrawn = (await chunk_flags(conn, chunk), await counted())

    assert withheld == ((False, True, False), 0)
    assert published == ((True, False, False), 1)
    assert withdrawn == ((False, True, False), 0)


async def test_the_full_refresh_rebuilds_the_flag_from_the_statement(
    pool: asyncpg.Pool,
) -> None:
    """The repair path has to agree with the definition, or it becomes the way
    drift is introduced rather than the way it is removed.

    031 named a chunk moved between collections unsupported and pointed at this
    function as the repair; 041 and 045 each had to move it when the predicate
    moved.  A derived flag written by hand is the same shape of drift, and is how
    this test reaches the function without waiting for one."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        withheld = await insert_chunk(
            conn,
            org_id=org,
            collection_id=collection,
            text="the operator reconciles the ledger before settlement",
            provenance_only=True,
        )

        await conn.execute(
            "UPDATE chunks SET retrievable = TRUE WHERE id = $1", withheld
        )
        drifted = await chunk_flags(conn, withheld)

        await conn.execute("SELECT pgkg_refresh_chunk_stats($1)", collection)
        repaired = await chunk_flags(conn, withheld)
        counted = await conn.fetchval(
            "SELECT COALESCE(SUM(n_total), 0) FROM corpus_stats"
            " WHERE kind = 'chunk' AND org_id = $1 AND collection_id = $2",
            org,
            collection,
        )

    assert drifted == (True, True, False)
    assert repaired == (False, True, False)
    assert counted == 0


async def test_liveness_takes_the_statement_and_not_the_pointer(
    pool: asyncpg.Pool,
) -> None:
    """A guard on the shape rather than on a behaviour, because the behaviour is
    deliberately unchanged.

    Read from the catalogue rather than from a copy of the definition: the
    argument list of the liveness function is where the pointer used to be, and
    the bridge trigger is meant to be the only schema object left that depends
    on chunks.document_id at all."""
    async with pool.acquire() as conn:
        arguments = await conn.fetchval(
            "SELECT pg_get_function_arguments(p.oid) FROM pg_proc p"
            " WHERE p.proname = 'pgkg_chunk_retrievable'"
        )
        dependants = await conn.fetch(
            """
            SELECT DISTINCT
                   COALESCE(t.tgname, ic.relname, co.conname) AS name
            FROM pg_depend d
            LEFT JOIN pg_trigger t ON t.oid = d.objid
            LEFT JOIN pg_class ic ON ic.oid = d.objid
            LEFT JOIN pg_constraint co ON co.oid = d.objid
            WHERE d.refobjid = 'chunks'::regclass
              AND d.refobjsubid = (
                  SELECT a.attnum FROM pg_attribute a
                  WHERE a.attrelid = 'chunks'::regclass
                    AND a.attname = 'document_id'
              )
            """
        )
        bodies = await conn.fetch(
            "SELECT p.proname, pg_get_functiondef(p.oid) AS src FROM pg_proc p"
            " JOIN pg_namespace n ON n.oid = p.pronamespace"
            " WHERE n.nspname = 'public' AND p.proname LIKE 'pgkg%'"
        )

    chunk_alias = re.compile(
        r"\b(?:c|chunks|n|o|t|new_rows|old_rows|delta_rows)\.document_id\b"
    )
    bodies_reading_the_pointer = sorted(
        row["proname"] for row in bodies if chunk_alias.search(row["src"])
    )

    assert "p_provenance_only" in arguments
    assert "document_id" not in arguments
    assert {row["name"] for row in dependants} <= {
        "chunks_document_id_fkey",
        "pgkg_chunks_provenance_bridge",
    }
    # pg_depend does not see inside a function body, and the trigger that does
    # read the pointer reads it from its WHEN clause rather than its body — so
    # the bodies are checked separately, by the aliases this schema gives the
    # chunks table.  Every surviving `document_id` in a pgkg function belongs to
    # document_versions or to ingest_jobs.
    assert bodies_reading_the_pointer == []


# ---------------------------------------------------------------------------
# Retirement, purge and garbage collection
# ---------------------------------------------------------------------------


async def test_a_retired_version_keeps_its_chunks(pool: asyncpg.Pool) -> None:
    """Promotion retires; it does not reclaim.  An in-flight reader may still
    resolve the outgoing version, and HNSW churn is something to schedule."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )

        first, _, chunks = await ingest_version(conn, document, ["kept", "gone"])
        await ingest_version(conn, document, ["kept"])

        row = await conn.fetchrow(
            "SELECT status, retired_at FROM document_versions WHERE id = $1",
            first,
        )
        links = await conn.fetchval(
            "SELECT count(*) FROM document_version_chunks"
            " WHERE document_version_id = $1",
            first,
        )
        refcount = await conn.fetchval(
            "SELECT refcount FROM chunks WHERE id = $1", chunks[1][0]
        )

    assert row["status"] == "retired"
    assert row["retired_at"] is not None
    assert links == 2
    assert refcount == 1


async def test_a_referenced_chunk_cannot_be_deleted(pool: asyncpg.Pool) -> None:
    """The only way to remove a chunk is to stop pointing at it first, which is
    what makes the refcount trustworthy enough to drive GC."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        _, _, chunks = await ingest_version(conn, document, ["referenced"])

        with pytest.raises(asyncpg.ForeignKeyViolationError):
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM chunks WHERE id = $1", chunks[0][0]
                )


async def test_the_grace_period_protects_a_freshly_retired_version(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        first, _, _ = await ingest_version(conn, document, ["v1"])
        await ingest_version(conn, document, ["v2"])

        purged = await conn.fetchval(
            "SELECT pgkg_purge_retired_versions(p_org_id => $1)", org
        )
        survives = await conn.fetchval(
            "SELECT count(*) FROM document_versions WHERE id = $1", first
        )

    assert purged == 0
    assert survives == 1


async def test_gc_removes_only_unreferenced_chunks(pool: asyncpg.Pool) -> None:
    """Purging the retired version drops its links, the refcount of the chunk it
    alone carried falls to zero, and only that chunk is collected."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )

        _, _, first = await ingest_version(conn, document, ["kept", "dropped"])
        kept, dropped = first[0][0], first[1][0]
        await ingest_version(conn, document, ["kept"])

        purged = await conn.fetchval(
            "SELECT pgkg_purge_retired_versions('0 seconds'::interval, $1)",
            org,
        )
        refcounts = dict(
            await conn.fetch(
                "SELECT id, refcount FROM chunks WHERE id = ANY($1::uuid[])",
                [kept, dropped],
            )
        )

        collected = await conn.fetchval("SELECT pgkg_gc_chunks($1)", org)
        survivors = [
            r["id"]
            for r in await conn.fetch(
                "SELECT id FROM chunks WHERE id = ANY($1::uuid[])",
                [kept, dropped],
            )
        ]

    assert purged == 1
    assert refcounts == {kept: 1, dropped: 0}
    assert collected == 1
    assert survivors == [kept]


async def test_gc_trusts_the_refcount_and_leaks_rather_than_deletes(
    pool: asyncpg.Pool,
) -> None:
    """refcount = 0 is the condition, and it is what makes the sweep an index
    scan rather than an anti-join over every chunk in the org.  A refcount that
    drifted upwards therefore leaks a row — the safe direction — instead of
    deleting content something still points at."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        chunk = await conn.fetchval(
            "INSERT INTO chunks (text, org_id, collection_id)"
            " VALUES ('linked by nothing', $1, $2) RETURNING id",
            org,
            collection,
        )
        await conn.execute(
            "UPDATE chunks SET refcount = 1 WHERE id = $1", chunk
        )

        while_claimed = await conn.fetchval("SELECT pgkg_gc_chunks($1)", org)

        await conn.execute(
            "UPDATE chunks SET refcount = 0 WHERE id = $1", chunk
        )
        once_released = await conn.fetchval("SELECT pgkg_gc_chunks($1)", org)

    assert while_claimed == 0
    assert once_released == 1


async def test_gc_leaves_a_chunk_that_a_proposition_was_derived_from(
    pool: asyncpg.Pool,
) -> None:
    """propositions.chunk_id cascades, so collecting a chunk takes the facts
    extracted from it with it.  "Delete this passage" and "delete everything we
    learned from it" are different decisions."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        namespace = unique("ns")
        chunk = await conn.fetchval(
            "INSERT INTO chunks (text, org_id, collection_id)"
            " VALUES ('unreferenced but extracted', $1, $2) RETURNING id",
            org,
            collection,
        )
        await conn.execute(
            "INSERT INTO propositions (text, namespace, org_id, collection_id,"
            " chunk_id) VALUES ('a fact', $1, $2, $3, $4)",
            namespace,
            org,
            collection,
            chunk,
        )

        collected = await conn.fetchval("SELECT pgkg_gc_chunks($1)", org)
        survives = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE id = $1", chunk
        )

    assert collected == 0
    assert survives == 1


# ---------------------------------------------------------------------------
# The flip
# ---------------------------------------------------------------------------


async def test_a_document_cannot_have_two_current_versions(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        await ingest_version(conn, document, ["v1"])
        pending = await conn.fetchrow(
            "SELECT * FROM pgkg_open_document_version($1, $2)",
            document,
            sha256("v2"),
        )

        with pytest.raises(asyncpg.UniqueViolationError):
            async with conn.transaction():
                await conn.execute(
                    "UPDATE document_versions SET status = 'current'"
                    " WHERE id = $1",
                    pending["version_id"],
                )


async def test_the_version_flip_is_atomic(pool: asyncpg.Pool) -> None:
    """A concurrent reader sees the old version or the new one — never both,
    never neither."""
    async with pool.acquire() as writer, pool.acquire() as reader:
        org = await new_org(writer)
        collection = await new_collection(writer, org_id=org)
        document = await new_document(
            writer, org_id=org, collection_id=collection
        )
        first, _, _ = await ingest_version(writer, document, ["v1"])

        second = await writer.fetchrow(
            "SELECT * FROM pgkg_open_document_version($1, $2)",
            document,
            sha256("v2"),
        )
        await writer.fetchrow(
            "SELECT * FROM pgkg_add_version_chunk($1, $2, $3)",
            second["version_id"],
            0,
            "v2",
        )

        async def resolved() -> list[uuid.UUID]:
            return [
                r["id"]
                for r in await reader.fetch(
                    """
                    SELECT dv.id
                    FROM documents d
                    JOIN document_versions dv ON dv.id = d.current_version_id
                    WHERE d.id = $1 AND dv.status = 'current'
                    """,
                    document,
                )
            ]

        tx = writer.transaction()
        await tx.start()
        await writer.execute(
            "SELECT pgkg_promote_document_version($1)", second["version_id"]
        )
        during = await resolved()
        await tx.commit()
        after = await resolved()

    assert during == [first]
    assert after == [second["version_id"]]


async def test_promoting_withdraws_facts_from_chunks_not_carried_forward(
    pool: asyncpg.Pool,
) -> None:
    """A retired version's dropped chunks stop being a source, so the facts
    derived from them stop being believed — source_updated, not superseded:
    nothing replaced them, their source stopped saying them."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        namespace = unique("ns")

        _, _, chunks = await ingest_version(conn, document, ["kept", "dropped"])
        facts = {}
        for label, (chunk_id, _) in zip(("kept", "dropped"), chunks):
            facts[label] = await conn.fetchval(
                "INSERT INTO propositions (text, namespace, org_id,"
                " collection_id, chunk_id) VALUES ($1, $2, $3, $4, $5)"
                " RETURNING id",
                f"a fact from the {label} chunk",
                namespace,
                org,
                collection,
                chunk_id,
            )

        await ingest_version(conn, document, ["kept"])

        rows = {
            r["id"]: r
            for r in await conn.fetch(
                "SELECT id, invalidated_at, invalidation_reason, superseded_by"
                " FROM propositions WHERE id = ANY($1::uuid[])",
                list(facts.values()),
            )
        }

    assert rows[facts["kept"]]["invalidated_at"] is None
    assert rows[facts["dropped"]]["invalidated_at"] is not None
    assert rows[facts["dropped"]]["invalidation_reason"] == "source_updated"
    assert rows[facts["dropped"]]["superseded_by"] is None


# ---------------------------------------------------------------------------
# The ownership seam: resolved, and still empty
# ---------------------------------------------------------------------------


async def shared_collection(conn: asyncpg.Connection) -> uuid.UUID:
    """Only the operator publishes, so a shared collection is owned by the
    system org (D4).  Created here and subscribed to by nobody else."""
    return await new_collection(
        conn, org_id=SYSTEM_ORG, visibility="shared", kind="corpus"
    )


async def test_resolution_returns_only_the_callers_own_org_by_default(
    pool: asyncpg.Pool,
) -> None:
    """Nothing is subscribed implicitly, so the hot-path org list is one
    element and prunes to one partition."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        await shared_collection(conn)

        orgs = await conn.fetchval("SELECT pgkg_subscribed_orgs($1)", org)
        collections = await conn.fetchval(
            "SELECT pgkg_subscribed_collections($1)", org
        )

    assert orgs == [org]
    assert collections == []


async def test_a_subscription_widens_the_org_list_to_the_owner(
    pool: asyncpg.Pool,
) -> None:
    async with pool.acquire() as conn:
        org = await new_org(conn)
        shared = await shared_collection(conn)
        await conn.execute(
            "INSERT INTO collection_subscriptions (org_id, collection_id)"
            " VALUES ($1, $2)",
            org,
            shared,
        )

        orgs = await conn.fetchval("SELECT pgkg_subscribed_orgs($1)", org)
        collections = await conn.fetchval(
            "SELECT pgkg_subscribed_collections($1)", org
        )

    assert sorted(orgs) == sorted([org, SYSTEM_ORG])
    assert collections == [shared]


async def test_a_disabled_subscription_stops_widening(
    pool: asyncpg.Pool,
) -> None:
    """rrf_weight turns shared material down; enabled turns it off, and off has
    to mean the second partition is not scanned at all."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        shared = await shared_collection(conn)
        await conn.execute(
            "INSERT INTO collection_subscriptions"
            " (org_id, collection_id, enabled) VALUES ($1, $2, FALSE)",
            org,
            shared,
        )

        orgs = await conn.fetchval("SELECT pgkg_subscribed_orgs($1)", org)
        collections = await conn.fetchval(
            "SELECT pgkg_subscribed_collections($1)", org
        )

    assert orgs == [org]
    assert collections == []


async def test_resolution_names_the_subscribed_collection_and_no_other(
    pool: asyncpg.Pool,
) -> None:
    """Widening the org list without naming the collections would make every
    other collection in the operator's org visible too."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        subscribed = await shared_collection(conn)
        await shared_collection(conn)
        await conn.execute(
            "INSERT INTO collection_subscriptions (org_id, collection_id)"
            " VALUES ($1, $2)",
            org,
            subscribed,
        )

        collections = await conn.fetchval(
            "SELECT pgkg_subscribed_collections($1)", org
        )

    assert collections == [subscribed]


async def test_a_subscription_to_a_private_collection_widens_nothing(
    pool: asyncpg.Pool,
) -> None:
    """A subscription row is not itself an authorisation: the collection has to
    have been published."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        unpublished = await new_collection(
            conn, org_id=SYSTEM_ORG, visibility="private"
        )
        await conn.execute(
            "INSERT INTO collection_subscriptions (org_id, collection_id)"
            " VALUES ($1, $2)",
            org,
            unpublished,
        )

        orgs = await conn.fetchval("SELECT pgkg_subscribed_orgs($1)", org)
        collections = await conn.fetchval(
            "SELECT pgkg_subscribed_collections($1)", org
        )

    assert orgs == [org]
    assert collections == []


async def test_a_tenant_cannot_publish_its_own_collection(
    pool: asyncpg.Pool,
) -> None:
    """The D4 hard rule, structurally: nothing a tenant ingests is ever promoted
    into a shared collection, so a tenant-owned collection cannot be shared and
    resolution has nothing to widen to."""
    async with pool.acquire() as conn:
        org = await new_org(conn)

        with pytest.raises(asyncpg.CheckViolationError):
            async with conn.transaction():
                await new_collection(conn, org_id=org, visibility="shared")


async def test_the_hard_rule_is_recorded_on_the_seam_itself(
    pool: asyncpg.Pool,
) -> None:
    """A rule that lives only in an ADR is a rule the next migration does not
    read (D4)."""
    async with pool.acquire() as conn:
        comment = await conn.fetchval(
            "SELECT obj_description('collection_subscriptions'::regclass)"
        )

    assert "promoted" in comment
    assert "operator" in comment


async def test_the_shipped_state_widens_to_nothing(pool: asyncpg.Pool) -> None:
    """The capability ships unused: this migration publishes no collection and
    subscribes nobody, so the backfill org still resolves to itself alone."""
    async with pool.acquire() as conn:
        default_org = await conn.fetchval("SELECT pgkg_default_org()")
        orgs = await conn.fetchval(
            "SELECT pgkg_subscribed_orgs(pgkg_default_org())"
        )
        seeded = await conn.fetchval(
            "SELECT visibility FROM collections WHERE id = pgkg_default_collection()"
        )

    assert orgs == [default_org]
    assert seeded == "private"


# ---------------------------------------------------------------------------
# Where the content address stops (D3, D4, D6)
# ---------------------------------------------------------------------------

async def test_a_passage_in_two_collections_is_a_row_in_each(
    pool: asyncpg.Pool,
) -> None:
    """Dedup stops at the collection boundary, as it already stops at the org.

    A chunk row is what the read predicate consults: collection_id selects the
    claim scope, the decay profile, the statistics domain and the quota bucket,
    and acl_group_id gates the read outright.  One row serving two collections
    is therefore permanently scoped, decayed and gated as whichever collection
    ingested it first, and the second collection cannot retrieve its own
    document's passage at all.  D6's sketch addresses a chunk by (org,
    content_hash); D3 requires collection_id on every retrievable row, and the
    two cannot both hold once a passage is in two collections.
    """
    from pgkg.corpus import CorpusIngest

    async with pool.acquire() as conn:
        org = await new_org(conn)
        coll_a = await conn.fetchval(
            "UPDATE collections SET claim_scope = 'world' WHERE id = $1"
            " RETURNING id",
            await new_collection(conn, org_id=org),
        )
        coll_b = await new_collection(conn, org_id=org)

    body = (
        "The reimbursement window for travel expenses closes thirty days "
        "after the trip concludes. Receipts must be legible."
    )
    embed = lambda texts: [[0.0] * 1024 for _ in texts]

    a = CorpusIngest(pool, org_id=org, collection_id=coll_a, embed=embed)
    b = CorpusIngest(pool, org_id=org, collection_id=coll_b, embed=embed)
    await a.upsert_document(external_id=unique("docA"), text=body)
    res_b = await b.upsert_document(external_id=unique("docB"), text=body)

    assert res_b.changed is True

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT c.id, c.collection_id
            FROM document_version_chunks dvc
            JOIN document_versions dv ON dv.id = dvc.document_version_id
            JOIN documents d ON d.id = dv.document_id
            JOIN chunks c ON c.id = dvc.chunk_id
            WHERE d.collection_id = $1
            """,
            coll_b,
        )
    assert rows, "document B linked no chunks"
    wrong = [row for row in rows if row["collection_id"] != coll_b]
    assert not wrong, (
        f"{len(wrong)} of {len(rows)} chunks reached by document B's current "
        f"version carry collection_id {wrong[0]['collection_id']} instead of "
        f"{coll_b}"
    )


async def test_two_concurrent_crawls_of_one_document_do_not_collide(
    pool: asyncpg.Pool,
) -> None:
    """Opening a version takes MAX(version_no) + 1, and (document_id,
    version_no) is unique.

    Two connectors crawling one document at once — or an upsert racing a
    re-crawl — both computed the same number, and the loser raised a unique
    violation out of upsert_document.  The row lock on the document is what
    makes the second one wait and number itself 2.
    """
    import asyncio

    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection, external_id=unique("race")
        )

    async def open_version(payload: bytes) -> None:
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('pgkg.org_id', $1, false)", str(org)
            )
            async with conn.transaction():
                await conn.fetchrow(
                    "SELECT version_id, is_new"
                    " FROM pgkg_open_document_version($1, $2)",
                    document,
                    sha256(payload.decode()),
                )
                await asyncio.sleep(0.1)

    await asyncio.gather(open_version(b"one"), open_version(b"two"))

    async with pool.acquire() as conn:
        numbers = [
            row["version_no"]
            for row in await conn.fetch(
                "SELECT version_no FROM document_versions"
                " WHERE document_id = $1 ORDER BY version_no",
                document,
            )
        ]
    assert numbers == [1, 2]


async def test_a_link_repointed_by_update_does_not_leave_a_ghost_passage(
    pool: asyncpg.Pool,
) -> None:
    """Derived state has to survive every write that can change it.

    041 moved chunk liveness off the hot retrieval path onto chunks.retrievable
    and moved the BM25 population with it, both reconciled by trigger.  The
    link table carried triggers for INSERT and DELETE — the ingest path and the
    purge — and none for UPDATE, so repointing a link left the passage it
    abandoned flagged retrievable for good: still counted in the statistics,
    still returned by the keyword arm, and with a refcount no garbage collector
    would ever bring to zero.  Before 041 liveness was recomputed per scan and
    the same UPDATE was merely untidy; a stored flag makes it withdrawn content
    that stays searchable.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="corpus")
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        kept = "The zzqghostly calibration procedure runs once every quarter."
        abandoned = "The zzqghostly appendix lists each discontinued fitting."
        version, _, added = await ingest_version(
            conn, document, [kept, abandoned]
        )
        kept_id, abandoned_id = added[0][0], added[1][0]

        await conn.execute(
            """
            UPDATE document_version_chunks SET chunk_id = $1
            WHERE document_version_id = $2 AND ord = 1
            """,
            kept_id,
            version,
        )

        state = {
            row["id"]: row
            for row in await conn.fetch(
                "SELECT id, refcount, retrievable FROM chunks"
                " WHERE id = ANY($1::uuid[])",
                [kept_id, abandoned_id],
            )
        }
        reachable = await conn.fetch(
            """
            SELECT item_id FROM pgkg_bm25_candidates(
                q_text => 'zzqghostly appendix discontinued fitting',
                k_initial => 50,
                p_org_ids => $1::uuid[],
                p_collection_ids => $2::uuid[],
                p_source => 'chunks')
            """,
            [org],
            [collection],
        )

    assert not state[abandoned_id]["retrievable"], (
        "a passage no version carries any more is still flagged retrievable, "
        "so withdrawn content stays in the statistics and in the scan"
    )
    assert state[abandoned_id]["refcount"] == 0, (
        "the abandoned passage keeps a refcount with no link behind it, so it "
        "can never be collected"
    )
    assert abandoned_id not in [row["item_id"] for row in reachable], (
        "the keyword arm still returns a passage that no live document carries"
    )
    assert state[kept_id]["refcount"] == 2, (
        "the passage the link was moved onto did not gain the reference, so "
        "the counter drifts in both directions"
    )
    assert state[kept_id]["retrievable"], (
        "the passage the version still carries stopped being retrievable"
    )


async def test_reclaiming_a_retired_version_does_not_resurrect_its_passages(
    pool: asyncpg.Pool,
) -> None:
    """Withdrawal has to outlive the version that carried the withdrawn text.

    A passage dropped by a new version stops being retrievable at promotion,
    correctly, because the only version still carrying it is retired.  Then
    pgkg_purge_retired_versions reclaims that version, its links go, and the
    passage matched the standalone arm of the liveness predicate — belongs to
    no document, carried by no version — so it was readmitted as a passage
    standing on its own.  Garbage collection resurrecting withdrawn content is
    the wrong way round, and it is permanent for any passage a fact was
    extracted from, since the collector refuses to take those.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, kind="corpus")
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )
        kept = "The zzqreclaim alpha clause covers the reclamation schedule."
        dropped = "The zzqreclaim beta clause covers the discontinued fittings."

        _, _, added = await ingest_version(conn, document, [kept, dropped])
        dropped_id = added[1][0]
        await ingest_version(conn, document, [kept])

        withdrawn = await conn.fetchval(
            "SELECT retrievable FROM chunks WHERE id = $1", dropped_id
        )
        purged = await conn.fetchval(
            "SELECT pgkg_purge_retired_versions(INTERVAL '0 seconds', $1)", org
        )
        after = await conn.fetchrow(
            "SELECT refcount, retrievable FROM chunks WHERE id = $1", dropped_id
        )
        reachable = await conn.fetch(
            """
            SELECT item_id FROM pgkg_bm25_candidates(
                q_text => 'zzqreclaim beta discontinued fittings',
                k_initial => 50,
                p_org_ids => $1::uuid[],
                p_collection_ids => $2::uuid[],
                p_source => 'chunks')
            """,
            [org],
            [collection],
        )

    assert withdrawn is False, (
        "the dropped passage was still retrievable before the purge, so the "
        "purge is not what this test is measuring"
    )
    assert purged == 1, "the retired version was not reclaimed"
    assert after["refcount"] == 0, "the reclaimed links left the count behind"
    assert after["retrievable"] is False, (
        "reclaiming the retired version made the withdrawn passage retrievable "
        "again"
    )
    assert dropped_id not in [row["item_id"] for row in reachable], (
        "the keyword arm returns a passage the current version dropped"
    )
