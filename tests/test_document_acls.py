"""Document ACLs: the write half of the seam (ADR 0001, D3).

The read half has been built since 020: `pgkg_visible()` drops an ACL-gated row
unless the caller presents its group, and the ACL group is part of the content
address.  Nothing wrote it.  `acl_group_id` was NULL on every row every pipeline
produced, and a NULL group passes the predicate — so a collection could declare
`acl_mode = 'group'` and get no enforcement at all while every part of the
system read as wired up.  D3 names the consequence: ingesting a corporate
corpus without modelling its permissions "builds a permission-laundering
machine".

Full ACL support needs a permissions source to synchronise groups from, which
does not exist.  What these tests pin is that the seam fails closed until it
does: an ACL-bounded collection refuses content that names no group, whichever
pipeline offers it, and content that names one is invisible to a caller who
does not present it.
"""
from __future__ import annotations

import hashlib
import uuid

import asyncpg
import pytest

from pgkg.corpus import CorpusIngest
from pgkg.memory import Memory, Scope

CHUNK_CAP = 300

SECRET = (
    "The acquisition of Northwind Traders closes on the third of March for "
    "four hundred million, and the board vote is scheduled for the Friday "
    "before."
)


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def sha256(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


class EmbedSpy:
    """Stands in for ml.embed, and counts what it was asked to pay for."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.texts: list[str] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [
            [float((int.from_bytes(sha256(text)[:4], "big") >> i) % 7) / 7.0
             for i in range(self.dim)]
            for text in texts
        ]


async def new_org(conn: asyncpg.Connection) -> uuid.UUID:
    return await conn.fetchval(
        "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("acl_org")
    )


async def new_collection(
    conn: asyncpg.Connection,
    *,
    org_id: uuid.UUID,
    acl_mode: str = "group",
    kind: str = "corpus",
    extract_propositions: bool = False,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO collections
            (org_id, owner_org_id, name, kind, claim_scope, decay_profile,
             acl_mode, extract_propositions)
        VALUES ($1, $1, $2, $3, 'org', 'timeless', $4, $5)
        RETURNING id
        """,
        org_id, unique("coll"), kind, acl_mode, extract_propositions,
    )


@pytest.fixture
async def bounded(pool: asyncpg.Pool):
    """An org, a collection whose ACL mode is 'group', and an embed spy."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org)
        dim = await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )
    return org, collection, EmbedSpy(dim)


def ingest_for(pool, org, collection, spy, **kwargs) -> CorpusIngest:
    return CorpusIngest(
        pool,
        org_id=org,
        collection_id=collection,
        embed=spy,
        max_chars=CHUNK_CAP,
        **kwargs,
    )


async def chunk_groups(pool: asyncpg.Pool, document_id: uuid.UUID) -> set:
    """The ACL group of every chunk the document's current version carries."""
    async with pool.acquire() as conn:
        return {
            row["acl_group_id"]
            for row in await conn.fetch(
                """
                SELECT c.acl_group_id
                FROM documents d
                JOIN document_version_chunks dvc
                     ON dvc.document_version_id = d.current_version_id
                JOIN chunks c ON c.id = dvc.chunk_id
                WHERE d.id = $1
                """,
                document_id,
            )
        }


async def retrieve(
    pool: asyncpg.Pool,
    query: str,
    *,
    org_id: uuid.UUID,
    collection_id: uuid.UUID,
    acl_groups: list[uuid.UUID] | None,
    sources: list[str] | None = None,
) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT text FROM pgkg_retrieve(
                q_text => $1, k_retrieve => 20, expand_graph => FALSE,
                p_org_ids => $2::uuid[], p_collection_ids => $3::uuid[],
                p_acl_groups => $4::uuid[], p_sources => $5::text[]
            )
            """,
            query, [org_id], [collection_id], acl_groups,
            sources or ["chunks"],
        )
    return [row["text"] for row in rows]


# ---------------------------------------------------------------------------
# 1. An ACL-bounded collection refuses untagged content
# ---------------------------------------------------------------------------

async def test_an_acl_bounded_collection_refuses_untagged_content(
    pool: asyncpg.Pool, bounded
) -> None:
    """The whole point of the change: a collection that declares an ACL mode
    and receives content with no group would publish that content to every
    caller of the tenant, silently."""
    org, collection, spy = bounded

    with pytest.raises(ValueError, match="acl_group_id"):
        await ingest_for(pool, org, collection, spy).upsert_document(
            external_id=unique("untagged"), text=SECRET
        )

    # Refused before the embedder was called: an ingest that spends the model
    # budget and then refuses to write has paid for nothing.
    assert spy.texts == []


async def test_a_collection_without_an_acl_mode_still_takes_untagged_content(
    pool: asyncpg.Pool
) -> None:
    """'none' is the default and the common case; the refusal is the ACL
    collection's rule, not a new requirement on every ingest."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(conn, org_id=org, acl_mode="none")
        dim = await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )

    result = await ingest_for(
        pool, org, collection, EmbedSpy(dim)
    ).upsert_document(external_id=unique("open"), text=SECRET)

    assert result.changed is True
    assert await chunk_groups(pool, result.document_id) == {None}


async def test_a_tagged_document_carries_its_group_to_every_chunk(
    pool: asyncpg.Pool, bounded
) -> None:
    org, collection, spy = bounded
    group = uuid.uuid4()

    result = await ingest_for(pool, org, collection, spy).upsert_document(
        external_id=unique("tagged"), text=SECRET, acl_group_id=group
    )

    assert result.changed is True
    assert await chunk_groups(pool, result.document_id) == {group}


async def test_facts_extracted_from_a_tagged_passage_carry_its_group(
    pool: asyncpg.Pool
) -> None:
    """A proposition is the passage restated, so a fact that lost the group its
    passage carries would launder the document it came from — through the
    proposition arm, which reads the same predicate."""
    async with pool.acquire() as conn:
        org = await new_org(conn)
        collection = await new_collection(
            conn, org_id=org, extract_propositions=True
        )
        dim = await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )
    group = uuid.uuid4()

    result = await ingest_for(
        pool, org, collection, EmbedSpy(dim)
    ).upsert_document(
        external_id=unique("extracted"), text=SECRET, acl_group_id=group
    )

    assert result.propositions > 0
    async with pool.acquire() as conn:
        groups = await conn.fetch(
            """
            SELECT DISTINCT p.acl_group_id
            FROM propositions p
            JOIN chunks c ON c.id = p.chunk_id
            JOIN document_version_chunks dvc ON dvc.chunk_id = c.id
            JOIN documents d
                 ON d.current_version_id = dvc.document_version_id
            WHERE d.id = $1
            """,
            result.document_id,
        )
    assert [row["acl_group_id"] for row in groups] == [group]


# ---------------------------------------------------------------------------
# 2. A tagged passage answers only to a caller presenting its group
# ---------------------------------------------------------------------------

async def test_a_tagged_passage_is_invisible_without_its_group(
    pool: asyncpg.Pool, bounded
) -> None:
    org, collection, spy = bounded
    group = uuid.uuid4()

    await ingest_for(pool, org, collection, spy).upsert_document(
        external_id=unique("acquisition"), text=SECRET, acl_group_id=group
    )

    granted = await retrieve(
        pool, "Northwind Traders acquisition",
        org_id=org, collection_id=collection, acl_groups=[group],
    )
    assert any("Northwind" in text for text in granted)

    for presented in (None, [uuid.uuid4()]):
        withheld = await retrieve(
            pool, "Northwind Traders acquisition",
            org_id=org, collection_id=collection, acl_groups=presented,
        )
        assert not [text for text in withheld if "Northwind" in text], (
            f"the passage came back for a caller presenting {presented!r}"
        )


# ---------------------------------------------------------------------------
# 3. The refusal is the database's, so it holds for every writer
# ---------------------------------------------------------------------------

async def test_the_database_refuses_an_untagged_chunk(
    pool: asyncpg.Pool, bounded
) -> None:
    """One pipeline checking the collection is a pipeline's rule; the tables
    are where every writer passes, including the ones phase 4 will add."""
    org, collection, _ = bounded
    async with pool.acquire() as conn:
        document = await conn.fetchval(
            """
            INSERT INTO documents (source, org_id, collection_id, external_id)
            VALUES ('acl probe', $1, $2, $3)
            RETURNING id
            """,
            org, collection, unique("ext"),
        )
        version = await conn.fetchval(
            "SELECT version_id FROM pgkg_open_document_version($1, $2)",
            document, sha256(SECRET),
        )
        with pytest.raises(asyncpg.PostgresError, match="acl_group_id"):
            await conn.fetchval(
                "SELECT chunk_id FROM pgkg_add_version_chunk($1, 0, $2)",
                version, SECRET,
            )


async def test_the_database_refuses_an_untagged_proposition(
    pool: asyncpg.Pool, bounded
) -> None:
    org, collection, _ = bounded
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.PostgresError, match="acl_group_id"):
            await conn.execute(
                """
                INSERT INTO propositions (text, org_id, collection_id)
                VALUES ($1, $2, $3)
                """,
                SECRET, org, collection,
            )


async def test_a_tagged_chunk_cannot_be_untagged_by_an_update(
    pool: asyncpg.Pool, bounded
) -> None:
    """An INSERT-only guard is the gap 045 found on the link table: dropping
    the group from a row that has one is the same laundering reached by the one
    statement nobody watched."""
    org, collection, spy = bounded
    group = uuid.uuid4()

    result = await ingest_for(pool, org, collection, spy).upsert_document(
        external_id=unique("regroup"), text=SECRET, acl_group_id=group
    )

    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.PostgresError, match="acl_group_id"):
            await conn.execute(
                """
                UPDATE chunks SET acl_group_id = NULL
                WHERE id IN (
                    SELECT dvc.chunk_id
                    FROM documents d
                    JOIN document_version_chunks dvc
                         ON dvc.document_version_id = d.current_version_id
                    WHERE d.id = $1
                )
                """,
                result.document_id,
            )


async def test_chat_ingest_into_an_acl_bounded_collection_needs_a_group(
    pool: asyncpg.Pool, bounded
) -> None:
    """Chat ingest already writes the group from its Scope, so the seam it
    needs is the refusal when the Scope does not name one."""
    org, collection, _ = bounded
    namespace = unique("acl_chat")
    untagged = Memory(
        pool,
        namespace=namespace,
        scope=Scope(org_id=org, collection_id=collection),
        extract_propositions=False,
    )

    with pytest.raises(asyncpg.PostgresError, match="acl_group_id"):
        await untagged.ingest(SECRET)

    group = uuid.uuid4()
    tagged = Memory(
        pool,
        namespace=namespace,
        scope=Scope(
            org_id=org, collection_id=collection, acl_group_id=group
        ),
        extract_propositions=False,
    )
    await tagged.ingest(SECRET)

    async with pool.acquire() as conn:
        written = await conn.fetch(
            """
            SELECT c.acl_group_id, c.document_id
            FROM chunks c
            JOIN document_version_chunks dvc ON dvc.chunk_id = c.id
            JOIN document_versions dv ON dv.id = dvc.document_version_id
            JOIN documents d ON d.id = dv.document_id
            WHERE d.namespace = $1
            """,
            namespace,
        )
    assert written, "chunks-only chat ingest wrote no chunk under the version"
    assert {row["acl_group_id"] for row in written} == {group}
    # 049 routed chunks-only chat ingest through pgkg_add_version_chunk(), which
    # leaves document_id NULL so the row falls under the content address.  A
    # chunk's parentage is the link table now, which is why this reads through
    # it rather than through the single-parent pointer.
    assert all(row["document_id"] is None for row in written)


# ---------------------------------------------------------------------------
# 4. Through the queue, which is the default posture for a corpus (D7)
# ---------------------------------------------------------------------------

async def test_the_queue_carries_the_acl_group_to_the_worker(
    pool: asyncpg.Pool, bounded
) -> None:
    from pgkg.ingest_jobs import IngestWorker, enqueue_document, job_state

    org, collection, spy = bounded
    group = uuid.uuid4()
    job = await enqueue_document(
        pool,
        org_id=org,
        collection_id=collection,
        external_id=unique("queued"),
        text=SECRET,
        acl_group_id=group,
    )

    worker = IngestWorker(
        pool, ingest=ingest_for(pool, org, collection, spy), org_id=org
    )
    assert await worker.run() == 1

    state = await job_state(pool, job, org_id=org)
    assert state.status == "done"
    assert await chunk_groups(pool, state.document_id) == {group}


async def test_an_untagged_job_for_an_acl_bounded_collection_fails(
    pool: asyncpg.Pool, bounded
) -> None:
    """Fail closed rather than fail silently: the connector's mistake is
    recorded against the document it offered, where an operator will see it."""
    from pgkg.ingest_jobs import IngestWorker, enqueue_document, job_state

    org, collection, spy = bounded
    job = await enqueue_document(
        pool,
        org_id=org,
        collection_id=collection,
        external_id=unique("queued_untagged"),
        text=SECRET,
    )

    worker = IngestWorker(
        pool, ingest=ingest_for(pool, org, collection, spy), org_id=org
    )
    assert await worker.run_once() is not None

    state = await job_state(pool, job, org_id=org)
    assert state.status == "pending"
    assert "acl_group_id" in state.error
    assert state.document_id is None
    assert spy.texts == []
