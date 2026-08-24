"""Batch corpus ingest: hash first, embed only what moved.

Chat ingest and corpus ingest are two pipelines, not one (ADR 0001, D7).  A
chat turn arrives once, is small, and is embedded on the spot.  A corpus
arrives as a nightly full crawl of everything the connector can see — that is
what a connector is — so the first question this module answers is "has this
changed?", and it answers it twice before it spends anything.

The document hash short-circuits the whole document, which is what makes an
unchanged crawl of 100k documents cost nothing.  The per-chunk content hash
short-circuits the expensive half: a typo fixed in a 300 page handbook is one
embedding call, not 300, because every other chunk is the same content under
the same address and the vector it already carries is still the right vector.

Everything that writes runs in ONE transaction, which is what D6 requires of
the flip: retiring the outgoing version and promoting the incoming one in two
round trips leaves a window where retrieval sees both versions or neither.  The
embedder is called inside that transaction, deliberately.  It is the slow step
and holding a connection across it is exactly what the online path must not do
— but this path is batch, it runs under the ingest_jobs worker's slot budget
rather than alongside recall, and the alternative is deciding what to embed
from a read taken outside the transaction that writes it.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import asyncpg
from pgvector import HalfVector

from pgkg import ml
from pgkg.chunking import DEFAULT_MAX_CHARS, chunk_document
from pgkg.config import DEFAULT_COLLECTION_ID, DEFAULT_ORG_ID, ORG_GUC
from pgkg.memory import Provenance
from pgkg.ml import ExtractCache, Proposition

EmbedFn = Callable[[list[str]], list[list[float]]]

# What a document's ingest tells the outside world while it is still running:
# chunks in the version, and chunks that now carry a vector.
ProgressFn = Callable[[int, int], Awaitable[None]]


@dataclass(frozen=True)
class CorpusIngestResult:
    """What one document cost, and what it left behind.

    The counts are the point: `embedded` is the only field that maps to money,
    and a caller watching a crawl watches it stay at zero.
    """

    document_id: UUID
    version_id: UUID
    changed: bool
    chunks_total: int = 0
    chunks_new: int = 0
    chunks_carried: int = 0
    embedded: int = 0
    cache_hits: int = 0
    propositions: int = 0


@dataclass(frozen=True)
class _CollectionPolicy:
    """The three collection columns this pipeline obeys."""

    claim_scope: str
    extract_propositions: bool
    public_source: bool


@dataclass(frozen=True)
class _Generation:
    """Which embedding space the vectors written here belong to.

    No width: the column declares that, and a second statement of it here could
    only disagree with the catalog.
    """

    generation_id: UUID


@dataclass(frozen=True)
class _Vectorised:
    """What vectorising the new chunks of one version cost."""

    new: int
    embedded: int
    cache_hits: int


_COLLECTION_POLICY_SQL = """
SELECT claim_scope, extract_propositions, public_source
FROM collections WHERE id = $1
"""

_PRIMARY_GENERATION_SQL = """
SELECT oe.generation_id
FROM org_embedders oe
JOIN embedder_generations g ON g.id = oe.generation_id
WHERE oe.org_id = $1 AND oe.role = 'primary'
"""

# A connector re-crawls by external_id, so that — not the source string or the
# uri — is what a second crawl collides on.  A soft-deleted document is not a
# collision: re-ingesting a withdrawn external id is a new document.
_FIND_DOCUMENT_SQL = """
SELECT id FROM documents
WHERE org_id = $1 AND collection_id = $2 AND external_id = $3
  AND deleted_at IS NULL
"""

_INSERT_DOCUMENT_SQL = """
INSERT INTO documents (source, uri, org_id, collection_id, external_id)
VALUES ($1, $2, $3, $4, $5)
RETURNING id
"""

_UPDATE_DOCUMENT_SQL = """
UPDATE documents
SET source = COALESCE($2, source), uri = COALESCE($3, uri)
WHERE id = $1
"""

_OPEN_VERSION_SQL = "SELECT version_id, is_new FROM pgkg_open_document_version($1, $2)"

# One derivation record per version rather than per chunk: a content-addressed
# chunk is shared by every version and document carrying that content, so it
# cannot record which crawl produced it — the version can.
_INSERT_PROVENANCE_SQL = """
INSERT INTO provenance
    (org_id, kind, source_id, producer, producer_model, prompt_version,
     ingest_run_id, actor_user_id, source_url, publisher, published_at,
     retrieved_at, licence, source_authority)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
RETURNING id
"""

# The version is opened before its provenance exists so that the unchanged path
# — the common one, once a corpus is steady — writes no provenance row at all.
_ATTRIBUTE_VERSION_SQL = "UPDATE document_versions SET provenance_id = $2 WHERE id = $1"

_ADD_CHUNK_SQL = """
SELECT chunk_id, is_new FROM pgkg_add_version_chunk($1, $2, $3, $4, $5)
"""

# The immutability trigger fires on text and org only, so writing the vector of
# a chunk the database just created is not a rewrite of its content.
_WRITE_CHUNK_EMBEDDINGS_SQL = """
UPDATE chunks c
SET embedding = e.embedding, embedder_generation_id = $1
FROM unnest($2::uuid[], $3::halfvec[]) AS e(id, embedding)
WHERE c.id = e.id
"""

_PROMOTE_SQL = "SELECT pgkg_promote_document_version($1)"

# Extraction over corpus chunks is opt-in per collection, so this statement runs
# for the fact-dense minority only (D2).  No entity linking and no edges: the
# derivation edge is chunk_id and it is free, while the mention edge that joins
# a passage to the entities it names is gazetteer work and belongs with phase 3.
# The object of a claim therefore survives as text rather than as an entity, and
# is not dropped on the floor while waiting for that.
_INSERT_PROPOSITIONS_SQL = """
WITH inserted AS (
    INSERT INTO propositions
        (text, embedding, predicate, object_literal, chunk_id, org_id,
         collection_id, claim_scope, provenance_id, asserted_at,
         embedder_generation_id)
    SELECT p.text, p.embedding, p.predicate, p.object_literal, p.chunk_id,
           $1, $2, $3, $4, $5, $6
    FROM unnest($7::text[], $8::halfvec[], $9::text[], $10::text[], $11::uuid[])
         AS p(text, embedding, predicate, object_literal, chunk_id)
    RETURNING id
)
INSERT INTO proposition_provenance (proposition_id, provenance_id)
SELECT id, $4 FROM inserted
ON CONFLICT DO NOTHING
"""

# The same table the online pipeline uses, keyed the same way: with
# content-addressed chunks, a re-ingest after an edit is a cache hit for every
# unchanged chunk (D2).  Bound to the ingest connection rather than taking one
# of its own — a second acquire from inside an open ingest transaction is how a
# batch pipeline starves the pool it shares with recall.
_CACHE_GET_SQL = """
UPDATE proposition_cache
SET hit_count = hit_count + 1
WHERE cache_key = $1
RETURNING propositions
"""

_CACHE_PUT_SQL = """
INSERT INTO proposition_cache
    (cache_key, chunk_hash, extractor_model, prompt_version, propositions)
VALUES ($1, $2, $3, $4, $5::jsonb)
ON CONFLICT (cache_key) DO NOTHING
"""

# The cache is read and written under the same flag.  A cache HIT is itself the
# observation the public_source restriction exists to prevent, so consulting it
# for private content would be the probe whether or not that content ever went
# in (ADR 0001, D4).
_READ_CACHE_SQL = """
SELECT content_hash, vec
FROM embedding_cache
WHERE generation_id = $1 AND content_hash = ANY($2::bytea[])
"""

_WRITE_CACHE_SQL = """
INSERT INTO embedding_cache (content_hash, generation_id, vec)
SELECT c.content_hash, $1, c.vec
FROM unnest($2::bytea[], $3::halfvec[]) AS c(content_hash, vec)
ON CONFLICT (content_hash, generation_id) DO NOTHING
"""

_SET_ORG_SQL = f"SELECT set_config('{ORG_GUC}', $1, false)"


async def _report(sink: ProgressFn | None, total: int, ready: int) -> None:
    if sink is None:
        return
    await sink(total, ready)


def content_hash(text: str) -> bytes:
    """The address the database derives for this text.

    Must be the digest the generated column computes, or a cache keyed on it
    could not be looked up from the text a connector is holding.
    """
    return hashlib.sha256(text.encode("utf-8")).digest()


# The same digest at the other granularity: one hash decides whether a document
# is worth opening, the other whether a chunk is worth embedding.
document_hash = content_hash


class _ConnExtractCache:
    """ExtractCache over one held connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def get(self, cache_key: str) -> list[Proposition] | None:
        # One round trip: the hit count is bumped by the statement that reads
        # the payload, as it is on the online path.
        row = await self._conn.fetchrow(_CACHE_GET_SQL, cache_key)
        if row is None:
            return None
        raw = row["propositions"]
        items = json.loads(raw) if isinstance(raw, str) else raw
        return [Proposition(**item) for item in items]

    async def put(
        self,
        cache_key: str,
        chunk_hash: str,
        extractor_model: str,
        prompt_version: str,
        props: list[Proposition],
    ) -> None:
        await self._conn.execute(
            _CACHE_PUT_SQL,
            cache_key,
            chunk_hash,
            extractor_model,
            prompt_version,
            json.dumps([prop.model_dump() for prop in props]),
        )


class CorpusIngest:
    """One org's corpus, one collection of it.

    Tenancy is bound to the object rather than passed per call, as it is on
    Memory: in a multi-tenant product one call site that forgets the argument is
    a cross-customer write, and a default argument cannot fail loudly.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        org_id: UUID = DEFAULT_ORG_ID,
        collection_id: UUID = DEFAULT_COLLECTION_ID,
        embed: EmbedFn | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
        use_extract_cache: bool = True,
    ) -> None:
        self._pool = pool
        self._org_id = org_id
        self._collection_id = collection_id
        self._embed = embed
        self._max_chars = max_chars
        self._use_extract_cache = use_extract_cache

    @property
    def org_id(self) -> UUID:
        return self._org_id

    @property
    def collection_id(self) -> UUID:
        return self._collection_id

    def for_collection(self, *, org_id: UUID, collection_id: UUID) -> CorpusIngest:
        """The same pipeline pointed at another tenant's collection."""
        return CorpusIngest(
            self._pool,
            org_id=org_id,
            collection_id=collection_id,
            embed=self._embed,
            max_chars=self._max_chars,
            use_extract_cache=self._use_extract_cache,
        )

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Resolved at call time so a spy on ml.embed is a spy on this path."""
        if not texts:
            return []
        if self._embed is not None:
            return self._embed(list(texts))
        return ml.embed(list(texts))

    async def upsert_document(
        self,
        *,
        external_id: str,
        text: str,
        uri: str | None = None,
        source: str | None = None,
        asserted_at: datetime | None = None,
        provenance: Provenance | None = None,
        on_progress: ProgressFn | None = None,
    ) -> CorpusIngestResult:
        version_hash = document_hash(text)

        async with self._pool.acquire() as conn:
            await conn.execute(_SET_ORG_SQL, str(self._org_id))
            policy = await self._policy(conn)
            generation = await self._generation(conn)

            async with conn.transaction():
                document_id = await self._document(conn, external_id, uri, source)
                opened = await conn.fetchrow(
                    _OPEN_VERSION_SQL, document_id, version_hash
                )
                version_id = opened["version_id"]

                # The no-op path: the hash has not moved, so nothing below runs
                # — no chunking, no embedder, no extractor, no writes.
                if not opened["is_new"]:
                    return CorpusIngestResult(
                        document_id=document_id, version_id=version_id, changed=False
                    )

                provenance_id = await self._attribute(
                    conn, version_id, provenance, uri
                )
                await self._update_document(conn, document_id, source, uri)
                linked = await self._link_chunks(conn, version_id, text, asserted_at)
                # Distinct rows, not links: a passage repeated inside one
                # document is one content-addressed chunk row.
                distinct = len({chunk_id for chunk_id, _, _ in linked})
                # Reported before the expensive step rather than after it, and
                # from a caller-supplied sink rather than written here: a
                # progress row written inside this transaction is invisible
                # until it commits, which is when it stops being progress.
                await _report(on_progress, len(linked), 0)
                vectorised = await self._vectorise(conn, policy, generation, linked)
                await _report(on_progress, len(linked), distinct)
                # Before the flip, in the order D6 gives: the facts a version
                # carries have to exist by the time it becomes the current one.
                extracted = await self._extract(
                    conn, policy, generation, provenance_id, asserted_at, linked
                )
                await conn.execute(_PROMOTE_SQL, version_id)

        return CorpusIngestResult(
            document_id=document_id,
            version_id=version_id,
            changed=True,
            chunks_total=len(linked),
            chunks_new=vectorised.new,
            chunks_carried=distinct - vectorised.new,
            embedded=vectorised.embedded,
            cache_hits=vectorised.cache_hits,
            propositions=extracted,
        )

    async def _policy(self, conn: asyncpg.Connection) -> _CollectionPolicy:
        row = await conn.fetchrow(_COLLECTION_POLICY_SQL, self._collection_id)
        if row is None:
            raise ValueError(f"no such collection {self._collection_id}")
        return _CollectionPolicy(
            claim_scope=row["claim_scope"],
            extract_propositions=row["extract_propositions"],
            public_source=row["public_source"],
        )

    async def _generation(self, conn: asyncpg.Connection) -> _Generation:
        row = await conn.fetchrow(_PRIMARY_GENERATION_SQL, self._org_id)
        if row is None:
            raise ValueError(
                f"org {self._org_id} is bound to no primary embedder generation"
            )
        return _Generation(generation_id=row["generation_id"])

    async def _document(
        self,
        conn: asyncpg.Connection,
        external_id: str,
        uri: str | None,
        source: str | None,
    ) -> UUID:
        existing = await conn.fetchval(
            _FIND_DOCUMENT_SQL, self._org_id, self._collection_id, external_id
        )
        if existing is not None:
            return existing
        return await conn.fetchval(
            _INSERT_DOCUMENT_SQL,
            source,
            uri,
            self._org_id,
            self._collection_id,
            external_id,
        )

    async def _update_document(
        self,
        conn: asyncpg.Connection,
        document_id: UUID,
        source: str | None,
        uri: str | None,
    ) -> None:
        if source is None and uri is None:
            return
        await conn.execute(_UPDATE_DOCUMENT_SQL, document_id, source, uri)

    async def _attribute(
        self,
        conn: asyncpg.Connection,
        version_id: UUID,
        provenance: Provenance | None,
        uri: str | None,
    ) -> UUID:
        given = provenance if provenance is not None else Provenance()
        provenance_id = await conn.fetchval(
            _INSERT_PROVENANCE_SQL,
            self._org_id,
            given.kind,
            given.source_id,
            given.producer or "chunker",
            given.producer_model,
            given.prompt_version,
            given.ingest_run_id,
            given.actor_user_id,
            given.source_url or uri,
            given.publisher,
            given.published_at,
            given.retrieved_at,
            given.licence,
            given.source_authority,
        )
        await conn.execute(_ATTRIBUTE_VERSION_SQL, version_id, provenance_id)
        return provenance_id

    async def _link_chunks(
        self,
        conn: asyncpg.Connection,
        version_id: UUID,
        text: str,
        asserted_at: datetime | None,
    ) -> list[tuple[UUID, str, bool]]:
        """Chunk the document and link every chunk to the open version.

        The database decides which chunks are new: it holds the content
        addresses, and a look-then-insert from here would race a second
        connector crawling the same boilerplate.
        """
        chunks = chunk_document(text, max_chars=self._max_chars)
        linked: list[tuple[UUID, str, bool]] = []
        for chunk in chunks:
            row = await conn.fetchrow(
                _ADD_CHUNK_SQL,
                version_id,
                chunk.ordinal,
                chunk.text,
                None,
                asserted_at,
            )
            linked.append((row["chunk_id"], chunk.text, row["is_new"]))
        return linked

    async def _vectorise(
        self,
        conn: asyncpg.Connection,
        policy: _CollectionPolicy,
        generation: _Generation,
        linked: list[tuple[UUID, str, bool]],
    ) -> _Vectorised:
        """Give a vector to every chunk the database created, and to no other.

        A carried-over chunk already holds the vector for its content, and its
        content cannot have changed — that is what its address means.
        """
        new = list(
            {chunk_id: text for chunk_id, text, is_new in linked if is_new}.items()
        )
        if not new:
            return _Vectorised(new=0, embedded=0, cache_hits=0)

        cached = await self._cache_lookup(conn, policy, generation, new)
        missing = [text for _, text in new if text not in cached]
        fresh = {
            text: HalfVector(vector)
            for text, vector in zip(missing, self._embed_texts(missing))
        }
        await self._cache_store(conn, policy, generation, fresh)

        vectors = {**cached, **fresh}
        await conn.execute(
            _WRITE_CHUNK_EMBEDDINGS_SQL,
            generation.generation_id,
            [chunk_id for chunk_id, _ in new],
            [vectors[text] for _, text in new],
        )
        return _Vectorised(
            new=len(new), embedded=len(fresh), cache_hits=len(cached)
        )

    async def _cache_lookup(
        self,
        conn: asyncpg.Connection,
        policy: _CollectionPolicy,
        generation: _Generation,
        new: list[tuple[UUID, str]],
    ) -> dict[str, HalfVector]:
        if not policy.public_source:
            return {}
        by_hash = {content_hash(text): text for _, text in new}
        rows = await conn.fetch(
            _READ_CACHE_SQL, generation.generation_id, list(by_hash)
        )
        return {by_hash[row["content_hash"]]: row["vec"] for row in rows}

    async def _cache_store(
        self,
        conn: asyncpg.Connection,
        policy: _CollectionPolicy,
        generation: _Generation,
        fresh: dict[str, HalfVector],
    ) -> None:
        if not policy.public_source or not fresh:
            return
        await conn.execute(
            _WRITE_CACHE_SQL,
            generation.generation_id,
            [content_hash(text) for text in fresh],
            list(fresh.values()),
        )

    async def _extract(
        self,
        conn: asyncpg.Connection,
        policy: _CollectionPolicy,
        generation: _Generation,
        provenance_id: UUID,
        asserted_at: datetime | None,
        linked: list[tuple[UUID, str, bool]],
    ) -> int:
        """Extract from the chunks this version added, if the collection opted in.

        New chunks only, for the same reason only they are embedded: a carried
        chunk's content has not changed, so neither have the facts in it — and
        extraction is the expensive half twice over, priced per token and
        recurring on every prompt version bump.
        """
        if not policy.extract_propositions:
            return 0
        new = list(
            {chunk_id: text for chunk_id, text, is_new in linked if is_new}.items()
        )
        if not new:
            return 0

        cache: ExtractCache | None = (
            _ConnExtractCache(conn) if self._use_extract_cache else None
        )
        extracted: list[tuple[UUID, Proposition]] = [
            (chunk_id, prop)
            for chunk_id, text in new
            for prop in await ml.extract_propositions_async(text, cache=cache)
        ]
        if not extracted:
            return 0

        vectors = self._embed_texts([prop.text for _, prop in extracted])
        await conn.execute(
            _INSERT_PROPOSITIONS_SQL,
            self._org_id,
            self._collection_id,
            policy.claim_scope,
            provenance_id,
            asserted_at,
            generation.generation_id,
            [prop.text for _, prop in extracted],
            [HalfVector(vector) for vector in vectors],
            [prop.predicate for _, prop in extracted],
            [prop.object for _, prop in extracted],
            [chunk_id for chunk_id, _ in extracted],
        )
        return len(extracted)
