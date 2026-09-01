"""Context windowing and the lifecycle races (ADR 0001, D3, D6).

Small-to-big widens a hit after every scoring stage has scoped it, so the
window is the one place where a correctly filtered result can be handed prose
the caller was never granted.  These tests pin the two halves of the answer:
a passage is only ever reused by documents that agree with it about every
column the retrieval predicate reads, and the neighbours a window aggregates
are indistinguishable from the anchor to that same predicate.
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid

import asyncpg
import pytest


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def sha256(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


async def new_org(conn: asyncpg.Connection) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("org")
    )


async def new_collection(
    conn: asyncpg.Connection, *, org_id: uuid.UUID
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO collections
            (org_id, owner_org_id, name, kind, claim_scope, decay_profile)
        VALUES ($1, $1, $2, 'corpus', 'org', 'timeless')
        RETURNING id
        """,
        org_id, unique("coll"),
    )


async def new_document(
    conn: asyncpg.Connection, *, org_id: uuid.UUID, collection_id: uuid.UUID
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO documents (source, org_id, collection_id, external_id)
        VALUES ('window probe', $1, $2, $3)
        RETURNING id
        """,
        org_id, collection_id, unique("ext"),
    )


async def add_version(
    conn: asyncpg.Connection,
    document: uuid.UUID,
    bodies: list[str],
    *,
    acl_groups: list[uuid.UUID | None] | None = None,
    promote: bool = True,
) -> list[uuid.UUID]:
    version = await conn.fetchval(
        "SELECT version_id FROM pgkg_open_document_version($1, $2)",
        document, sha256("".join(bodies)),
    )
    groups = acl_groups or [None] * len(bodies)
    ids = [
        await conn.fetchval(
            "SELECT chunk_id FROM pgkg_add_version_chunk($1, $2, $3, $4)",
            version, ord_, body, group,
        )
        for ord_, (body, group) in enumerate(zip(bodies, groups))
    ]
    if promote:
        await conn.execute("SELECT pgkg_promote_document_version($1)", version)
    return ids


async def window(
    conn: asyncpg.Connection, chunk: uuid.UUID
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        "SELECT * FROM pgkg_chunk_window(ARRAY[$1]::UUID[], 1, 1)", chunk
    )


# ---------------------------------------------------------------------------
# The content address covers the columns a chunk derives from its document
# ---------------------------------------------------------------------------


async def test_boilerplate_is_still_one_row_inside_a_collection(
    pool: asyncpg.Pool,
) -> None:
    """The dedup D6 is built for: two documents of one collection sharing a
    passage share the row, so the second ingest embeds nothing."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        footer = unique("footer") + " confidential do not distribute"

        first = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=collection),
            [footer],
        )
        second_doc = await new_document(
            conn, org_id=org, collection_id=collection
        )
        version = await conn.fetchval(
            "SELECT version_id FROM pgkg_open_document_version($1, $2)",
            second_doc, sha256(footer + "2"),
        )
        carried = await conn.fetchrow(
            "SELECT chunk_id, is_new FROM pgkg_add_version_chunk($1, 0, $2)",
            version, footer,
        )

    assert carried["chunk_id"] == first[0]
    assert carried["is_new"] is False


async def test_a_shared_passage_is_a_row_per_collection(
    pool: asyncpg.Pool,
) -> None:
    """collection_id carries the claim scope, the decay profile, the statistics
    domain and the quota bucket, so a row shared across collections would be
    scoped as whichever document was ingested first."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        coll_a = await new_collection(conn, org_id=org)
        coll_b = await new_collection(conn, org_id=org)
        shared = unique("shared") + " the reimbursement window closes"

        in_a = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=coll_a),
            [shared],
        )
        in_b = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=coll_b),
            [shared],
        )

        rows = {
            r["id"]: r
            for r in await conn.fetch(
                "SELECT id, collection_id, refcount FROM chunks"
                " WHERE id = ANY($1::uuid[])",
                [in_a[0], in_b[0]],
            )
        }

    assert in_a[0] != in_b[0]
    assert rows[in_a[0]]["collection_id"] == coll_a
    assert rows[in_b[0]]["collection_id"] == coll_b
    assert rows[in_a[0]]["refcount"] == 1
    assert rows[in_b[0]]["refcount"] == 1


async def test_a_shared_passage_is_a_row_per_acl_group(
    pool: asyncpg.Pool,
) -> None:
    """Two documents of one collection in different ACL groups do not share a
    row either: the group is what gates the read, and the second document
    would otherwise be gated by the first document's grant."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        allowed, denied = uuid.uuid4(), uuid.uuid4()
        shared = unique("shared") + " every claim is filed quarterly"

        first = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=collection),
            [shared],
            acl_groups=[allowed],
        )
        second = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=collection),
            [shared],
            acl_groups=[denied],
        )

        groups = dict(
            await conn.fetch(
                "SELECT id, acl_group_id FROM chunks WHERE id = ANY($1::uuid[])",
                [first[0], second[0]],
            )
        )

    assert first[0] != second[0]
    assert groups[first[0]] == allowed
    assert groups[second[0]] == denied


# ---------------------------------------------------------------------------
# The window stays inside the grant that admitted the anchor
# ---------------------------------------------------------------------------


async def test_each_collections_copy_of_a_passage_reads_its_own_document(
    pool: asyncpg.Pool,
) -> None:
    """The deterministic form of the cross-collection window leak: whichever
    copy of a shared passage is retrieved, the context comes from a document of
    that copy's own collection."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        coll_a = await new_collection(conn, org_id=org)
        coll_b = await new_collection(conn, org_id=org)
        shared = unique("shared") + " the standard disclaimer"
        marker_a, marker_b = unique("onlyina"), unique("onlyinb")

        in_a = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=coll_a),
            [f"{marker_a} opening", shared, f"{marker_a} closing"],
        )
        in_b = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=coll_b),
            [f"{marker_b} opening", shared, f"{marker_b} closing"],
        )

        from_a = await window(conn, in_a[1])
        from_b = await window(conn, in_b[1])

    assert marker_a in from_a["context_text"]
    assert marker_b not in from_a["context_text"]
    assert marker_b in from_b["context_text"]
    assert marker_a not in from_b["context_text"]


async def test_a_window_never_crosses_an_acl_group(pool: asyncpg.Pool) -> None:
    """A version may carry passages from several ACL groups — a document whose
    sections mirror different SharePoint grants.  The neighbours a window
    aggregates are the ones the anchor's own grant already admits."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        allowed, denied = uuid.uuid4(), uuid.uuid4()
        readable = unique("readable") + " the footer everyone sees"
        restricted = unique("restricted") + " the acquisition price"

        ids = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=collection),
            [readable, restricted],
            acl_groups=[allowed, denied],
        )

        row = await window(conn, ids[0])

    assert readable in row["context_text"]
    assert restricted not in row["context_text"]
    assert (row["ord_from"], row["ord_to"]) == (0, 0)


async def test_a_passage_two_live_documents_carry_is_its_own_context(
    pool: asyncpg.Pool,
) -> None:
    """Boilerplate repeated across documents of one collection has no single
    context: the neighbours of a footer in one handbook are not the context of
    the same footer in another, so the passage stands alone rather than
    borrowing an arbitrary document's prose."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        footer = unique("footer") + " every claim needs a receipt"

        first = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=collection),
            [unique("first") + " opening", footer],
        )
        second = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=collection),
            [unique("second") + " opening", footer],
        )

        assert first[1] == second[1]

        windows = await conn.fetch(
            "SELECT * FROM pgkg_chunk_window(ARRAY[$1]::UUID[], 1, 1)", first[1]
        )
        anchored = await conn.fetchrow(
            """
            SELECT text, context_text FROM pgkg_retrieve(
                q_text => $1, k_retrieve => 5, expand_graph => FALSE,
                p_org_ids => $2::uuid[], p_collection_ids => $3::uuid[],
                p_sources => ARRAY['chunks']::text[]
            )
            """,
            footer, [org], [collection],
        )

    assert windows == []
    assert anchored["context_text"] == anchored["text"] == footer


async def test_the_window_ignores_a_version_from_another_document(
    pool: asyncpg.Pool,
) -> None:
    """A link is owned by the version, and 033's policy validates only that
    side, so the window checks that the document carrying it is the anchor's
    own tenant and collection rather than trusting the link."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        other = await new_collection(conn, org_id=org)
        passage = unique("passage") + " the parental leave policy"
        intruder = unique("intruder") + " the acquisition price"

        # The anchor's own version is never promoted, so the grafted link is
        # the only current version that carries it: the window is empty, not
        # borrowed from the other collection's document.
        ids = await add_version(
            conn,
            await new_document(conn, org_id=org, collection_id=collection),
            [passage],
            promote=False,
        )
        elsewhere = await new_document(conn, org_id=org, collection_id=other)
        version = await conn.fetchval(
            "SELECT version_id FROM pgkg_open_document_version($1, $2)",
            elsewhere, sha256(intruder),
        )
        await conn.execute(
            "SELECT pgkg_add_version_chunk($1, 0, $2)", version, intruder
        )
        await conn.execute(
            "INSERT INTO document_version_chunks"
            " (document_version_id, chunk_id, ord) VALUES ($1, $2, 1)",
            version, ids[0],
        )
        await conn.execute("SELECT pgkg_promote_document_version($1)", version)

        rows = await conn.fetch(
            "SELECT * FROM pgkg_chunk_window(ARRAY[$1]::UUID[], 1, 1)", ids[0]
        )

    assert [r["context_text"] for r in rows if intruder in r["context_text"]] == []
    assert rows == []


async def test_a_chunk_of_another_org_cannot_be_linked_to_a_version(
    pool: asyncpg.Pool,
) -> None:
    """A link between two orgs is meaningless in any session: it injects the
    stranger's prose into the victim's context and holds the victim's chunk
    live against GC."""
    async with pool.acquire() as conn:
        victim = await new_org(conn)
        attacker = await new_org(conn)
        v_collection = await new_collection(conn, org_id=victim)
        a_collection = await new_collection(conn, org_id=attacker)

        theirs = await add_version(
            conn,
            await new_document(conn, org_id=victim, collection_id=v_collection),
            [unique("victim") + " the passage"],
        )
        mine = await new_document(
            conn, org_id=attacker, collection_id=a_collection
        )
        version = await conn.fetchval(
            "SELECT version_id FROM pgkg_open_document_version($1, $2)",
            mine, sha256("attacker body"),
        )

        with pytest.raises(asyncpg.RaiseError):
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO document_version_chunks"
                    " (document_version_id, chunk_id, ord) VALUES ($1, $2, 0)",
                    version, theirs[0],
                )


# ---------------------------------------------------------------------------
# Two crawls of one document
# ---------------------------------------------------------------------------


async def test_two_overlapping_crawls_number_their_versions_in_sequence(
    pool: asyncpg.Pool,
) -> None:
    """Nightly full crawls are what connectors do, so two of them meeting on
    one document is the common case: they queue on the document row instead of
    computing the same version number."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        document = await new_document(
            conn, org_id=org, collection_id=collection
        )

    async def crawl(body: str) -> None:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.fetchrow(
                    "SELECT version_id FROM pgkg_open_document_version($1, $2)",
                    document, sha256(body),
                )
                await asyncio.sleep(0.1)

    await asyncio.gather(crawl("one"), crawl("two"))

    async with pool.acquire() as conn:
        numbers = [
            r["version_no"]
            for r in await conn.fetch(
                "SELECT version_no FROM document_versions"
                " WHERE document_id = $1 ORDER BY version_no",
                document,
            )
        ]

    assert numbers == [1, 2]


# ---------------------------------------------------------------------------
# The same rule read through pgkg_retrieve, which is where a caller meets it
# ---------------------------------------------------------------------------

SHARED = (
    "The standard reimbursement footer applies to every expense claim filed "
    "quarterly under the zzqarbitrage policy schedule."
)
SECRET_BEFORE = (
    "Acme Holdings will be acquired for four hundred million zzqarbitrage."
)
SECRET_AFTER = "The zzqarbitrage board vote is scheduled for the third of March."


async def retrieved_context(
    conn: asyncpg.Connection,
    *,
    org: uuid.UUID,
    collections: list[uuid.UUID],
    acl_groups: list[uuid.UUID] | None = None,
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT item_id, source, text, context_text
        FROM pgkg_retrieve(
            q_text => 'zzqarbitrage reimbursement footer',
            k_retrieve => 20,
            expand_graph => FALSE,
            p_org_ids => $1::uuid[],
            p_collection_ids => $2::uuid[],
            p_acl_groups => $3::uuid[],
            p_sources => ARRAY['chunks']::text[]
        )
        """,
        [org], collections, acl_groups,
    )


async def test_context_text_never_leaves_the_collections_the_query_named(
    pool: asyncpg.Pool,
) -> None:
    """The permission-laundering machine D3 names, at the point it was real.

    A boilerplate footer in a readable collection and in a restricted one used
    to be one chunk row, and the window walked that row's neighbours by ord and
    concatenated their text, so a correctly scoped hit arrived carrying the
    restricted document's surrounding prose in context_text — over HTTP, since
    /recall returns Result.context_text.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        readable = await new_collection(conn, org_id=org)
        secret = await new_collection(conn, org_id=org)

        readable_doc = await new_document(
            conn, org_id=org, collection_id=readable
        )
        await add_version(conn, readable_doc, [SHARED])

        secret_doc = await new_document(conn, org_id=org, collection_id=secret)
        await add_version(
            conn, secret_doc, [SECRET_BEFORE, SHARED, SECRET_AFTER]
        )

        rows = await retrieved_context(
            conn, org=org, collections=[readable]
        )
        # And over every window the readable copy has, not only the one
        # pgkg_retrieve picked: a rule that held for an arbitrary pick would be
        # a coin flip on which version UUID sorted lower.
        every_window = [
            row["context_text"]
            for row in await conn.fetch(
                """
                SELECT cw.context_text
                FROM chunks c
                JOIN pgkg_chunk_window(ARRAY[c.id]::UUID[], 1, 1) cw ON TRUE
                WHERE c.text = $1 AND c.collection_id = $2
                """,
                SHARED, readable,
            )
        ]

    assert not [
        context
        for context in every_window
        if SECRET_BEFORE in context or SECRET_AFTER in context
    ], f"a window of the readable copy was built from the restricted document"

    passages = [row for row in rows if row["source"] == "chunks"]
    assert passages, "the readable passage did not come back at all"
    leaked = [
        row["context_text"]
        for row in passages
        if SECRET_BEFORE in (row["context_text"] or "")
        or SECRET_AFTER in (row["context_text"] or "")
    ]
    assert not leaked, (
        f"context_text was drawn from a collection outside the query: {leaked!r}"
    )


async def test_context_text_never_launders_a_document_acl(
    pool: asyncpg.Pool,
) -> None:
    """The same rule on the axis D3 calls the harder one: one collection, two
    documents in different ACL groups, one shared footer.  The ACL predicate
    kept the denied passage out of `text` all along; the window handed the whole
    denied document back beside it."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        allowed = uuid.uuid4()
        denied = uuid.uuid4()

        allowed_doc = await new_document(
            conn, org_id=org, collection_id=collection
        )
        await add_version(conn, allowed_doc, [SHARED], acl_groups=[allowed])

        denied_doc = await new_document(
            conn, org_id=org, collection_id=collection
        )
        await add_version(
            conn,
            denied_doc,
            [SECRET_BEFORE, SHARED, SECRET_AFTER],
            acl_groups=[denied, denied, denied],
        )

        rows = await retrieved_context(
            conn, org=org, collections=[collection], acl_groups=[allowed]
        )

    assert SECRET_BEFORE not in [row["text"] for row in rows], (
        "the ACL predicate itself failed, which is a different defect"
    )
    leaked = [
        row["context_text"]
        for row in rows
        if SECRET_BEFORE in (row["context_text"] or "")
        or SECRET_AFTER in (row["context_text"] or "")
    ]
    assert not leaked, (
        f"context_text laundered another ACL group's document: {leaked!r}"
    )


async def test_no_shared_passage_reads_the_other_collections_document(
    pool: asyncpg.Pool,
) -> None:
    """Ten independent pairs rather than one, because the defect this replaces
    was a coin flip between two version UUIDs and showed up in 30-70% of pairs.

    Each collection's copy of the shared passage is anchored explicitly: a
    passage in two collections is a row in each, so a probe that took an
    arbitrary one of those rows would be measuring which row an unordered scan
    happened to return rather than where the window went.
    """
    async with pool.acquire() as conn:
        org = await new_org(conn)
        coll_a = await new_collection(conn, org_id=org)
        coll_b = await new_collection(conn, org_id=org)

        for _ in range(10):
            shared = f"the shared boilerplate disclaimer {uuid.uuid4().hex}"
            marker_a = f"collectionaonly{uuid.uuid4().hex[:8]}"
            marker_b = f"collectionbonly{uuid.uuid4().hex[:8]}"
            for collection, marker in (
                (coll_a, marker_a), (coll_b, marker_b)
            ):
                document = await new_document(
                    conn, org_id=org, collection_id=collection
                )
                await add_version(
                    conn,
                    document,
                    [f"{marker} opening section", shared, f"{marker} closing"],
                )

            for collection, mine, theirs in (
                (coll_a, marker_a, marker_b), (coll_b, marker_b, marker_a)
            ):
                chunk = await conn.fetchval(
                    "SELECT id FROM chunks WHERE text = $1 AND collection_id = $2",
                    shared, collection,
                )
                assert chunk is not None, "the collection holds no copy at all"
                context = (await window(conn, chunk))["context_text"]
                assert mine in context, (
                    "the window lost the anchor's own document"
                )
                assert theirs not in context, (
                    "the window was built from a document in the other "
                    f"collection: {context!r}"
                )
