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
round trips leaves a window where retrieval sees both versions or neither.

The models run BEFORE that transaction opens.  D7 is explicit that corpus
ingest must not compete with online recall for pool slots, and a document with
fifty chunks and a two-second extractor would otherwise hold one pooled
connection — and the row locks its extract-cache reads take — for a hundred
seconds.  So a document is three phases: ask whether the hash moved, spend the
model budget with no transaction open, then write everything in one transaction
that contains no model call.  The per-chunk extractor runs holding nothing at
all, and progress is reported between phases rather than during one, because a
slot that holds a connection while its reporter asks for a second one is a slot
that cannot report at all.

What makes that ordering safe is content addressing.  The spend phase asks
which chunk texts are already stored under this org and already carry a vector
of the primary generation; the write phase asks the database the same question
again, authoritatively, through pgkg_add_version_chunk().  A concurrent crawl
that stores a passage in between can only cost us one redundant embedding of
text that hashes to the row we then reuse — never a wrong vector, because a
vector is only ever written against the content it was computed from.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from uuid import UUID

import asyncpg
from pgvector import HalfVector

from pgkg import ml
from pgkg.chunking import DEFAULT_MAX_CHARS, Chunk, chunk_document
from pgkg.config import DEFAULT_COLLECTION_ID, DEFAULT_ORG_ID, ORG_GUC
from pgkg.memory import Provenance
from pgkg.ml import Proposition

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

# The no-op path, asked before anything is held or spent: this is the predicate
# pgkg_open_document_version() applies to decide is_new, read here so that an
# unchanged crawl opens no transaction at all.  The write phase asks again.
_FIND_UNCHANGED_SQL = """
SELECT d.id AS document_id, dv.id AS version_id
FROM documents d
JOIN document_versions dv ON dv.id = d.current_version_id
WHERE d.org_id = $1 AND d.collection_id = $2 AND d.external_id = $3
  AND d.deleted_at IS NULL AND dv.content_hash = $4
"""

# What is already stored, at the granularity the reuse path works at: a chunk
# row with no document_id is the one pgkg_add_version_chunk() reuses, and its
# vector is only reusable if it belongs to the generation this org writes in.
_KNOWN_CONTENT_SQL = """
SELECT c.content_hash,
       (c.embedding IS NOT NULL AND c.embedder_generation_id = $2) AS has_vector
FROM chunks c
WHERE c.org_id = $1 AND c.document_id IS NULL
  AND c.content_hash = ANY($3::bytea[])
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

# Every chunk of the document in one round trip.  A loop over this function is
# one network round trip per chunk with the ingest transaction held open, which
# on a 300-page handbook is the whole cost of the write phase; the lateral join
# makes the pipeline set-based the way chat ingest already is.
_ADD_CHUNKS_SQL = """
SELECT t.n, added.chunk_id, added.is_new
FROM unnest($2::int[], $3::text[]) WITH ORDINALITY AS t(ord, chunk_text, n)
CROSS JOIN LATERAL
    pgkg_add_version_chunk($1, t.ord, t.chunk_text, $4, $5) AS added
ORDER BY t.n
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
# unchanged chunk (D2).  One statement for the whole document rather than one
# per chunk, and consulted outside the ingest transaction: the hit_count bump
# takes a row lock, and a fifty-chunk document held those locks for as long as
# its extractor ran.
# Keyed by the org as well as by the text and the model: a hit is the claims
# extracted from a passage, so it answers only to the org that paid for them.
# The row policy says the same thing, and cannot say it on the owner connection
# a reference deployment uses (D4).
_CACHE_GET_SQL = """
UPDATE proposition_cache
SET hit_count = hit_count + 1
WHERE cache_key = ANY($1::text[]) AND org_id = $2
RETURNING cache_key, propositions
"""

_CACHE_PUT_SQL = """
INSERT INTO proposition_cache
    (cache_key, chunk_hash, extractor_model, prompt_version, propositions,
     org_id)
SELECT *, $6 FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::jsonb[])
ON CONFLICT (cache_key, org_id) DO NOTHING
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


# A queued document is worked on by a process the connector never meets, so the
# account it gave of where the document came from has to travel with it (D5,
# D7).  The wire form is this module's to state rather than the queue's: the
# queue holds it as opaque JSONB, and these two functions are the only place
# that knows a published_at is a timestamp and an ingest_run_id is a UUID.
_PROVENANCE_UUID_FIELDS = ("ingest_run_id", "actor_user_id", "source_id")
_PROVENANCE_TIME_FIELDS = ("published_at", "retrieved_at")


def dump_provenance(given: Provenance | None) -> str | None:
    """The derivation record as JSON, or None when the caller stated none."""
    if given is None:
        return None
    stated = {
        field.name: getattr(given, field.name)
        for field in fields(given)
        if getattr(given, field.name) is not None
    }
    return json.dumps(stated, default=str)


def load_provenance(payload: object) -> Provenance | None:
    """The record a connector stated, rebuilt with its types.

    JSONB has no timestamp and no UUID, so a round trip through the queue that
    did not restore them would hand the pipeline strings and store an ingest
    date as text — which is the perishable profile reading a date it cannot
    subtract.
    """
    if payload is None:
        return None
    raw = json.loads(payload) if isinstance(payload, str) else dict(payload)
    known = {f.name for f in fields(Provenance)}
    stated = {name: value for name, value in raw.items() if name in known}
    for name in _PROVENANCE_UUID_FIELDS:
        if isinstance(stated.get(name), str):
            stated[name] = UUID(stated[name])
    for name in _PROVENANCE_TIME_FIELDS:
        if isinstance(stated.get(name), str):
            stated[name] = datetime.fromisoformat(stated[name])
    return Provenance(**stated)


def _propositions(payload: object) -> list[Proposition]:
    items = json.loads(payload) if isinstance(payload, str) else payload
    return [Proposition(**item) for item in items]


class _BatchExtractCache:
    """ExtractCache for one document: one read for all its chunks, one write.

    The read is deferred to the first get() rather than done up front, so a
    caller that never consults the cache — the offline extractor — pays for
    nothing and bumps nobody's hit count.  Writes are buffered and land in the
    ingest transaction, which is the only place in this module that writes.
    """

    def __init__(
        self, pool: asyncpg.Pool, texts: Sequence[str], *, org_id: UUID
    ) -> None:
        self._pool = pool
        self._org_id = org_id
        self._texts = list(dict.fromkeys(texts))
        self._hits: dict[str, list[Proposition]] | None = None
        self._puts: list[tuple[str, str, str, str, str]] = []

    async def get(self, cache_key: str) -> list[Proposition] | None:
        if self._hits is None:
            self._hits = await self._prefetch()
        return self._hits.get(cache_key)

    async def _prefetch(self) -> dict[str, list[Proposition]]:
        # The key is derived the way ml derives it, from the settings actually
        # in force: a prefetch under another model's key would miss every time.
        settings = ml.get_settings()
        model = settings.extractor_model or settings.llm_model
        keys = [ml.compute_cache_key(text, model) for text in self._texts]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_CACHE_GET_SQL, keys, self._org_id)
        return {row["cache_key"]: _propositions(row["propositions"]) for row in rows}

    async def put(
        self,
        cache_key: str,
        chunk_hash: str,
        extractor_model: str,
        prompt_version: str,
        props: list[Proposition],
    ) -> None:
        self._puts.append(
            (
                cache_key,
                chunk_hash,
                extractor_model,
                prompt_version,
                json.dumps([prop.model_dump() for prop in props]),
            )
        )

    async def flush(self, conn: asyncpg.Connection) -> None:
        if not self._puts:
            return
        columns = [list(column) for column in zip(*self._puts)]
        await conn.execute(_CACHE_PUT_SQL, *columns, self._org_id)


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
        given = provenance if provenance is not None else Provenance()
        # D6 keys the perishable profile on asserted_at, and D5 says
        # provenance.published_at is what feeds it: what makes an article stale
        # is when it was published, not when this crawl reached it.  An explicit
        # asserted_at still wins — a caller stating the world-time of a claim
        # knows something a publication date does not.
        asserted = asserted_at if asserted_at is not None else given.published_at

        # Phase 1, ask whether there is any work.  The document hash is what
        # makes an unchanged crawl of 100k documents free, so it is answered
        # before the text is even chunked.
        async with self._pool.acquire() as conn:
            await conn.execute(_SET_ORG_SQL, str(self._org_id))
            policy = await self._policy(conn)
            generation = await self._generation(conn)
            unchanged = await conn.fetchrow(
                _FIND_UNCHANGED_SQL,
                self._org_id,
                self._collection_id,
                external_id,
                version_hash,
            )
        if unchanged is not None:
            return CorpusIngestResult(
                document_id=unchanged["document_id"],
                version_id=unchanged["version_id"],
                changed=False,
            )

        chunks = chunk_document(text, max_chars=self._max_chars)
        # Distinct texts, not links: a passage repeated inside one document is
        # one content-addressed chunk row, embedded once and extracted once.
        texts = list(dict.fromkeys(chunk.text for chunk in chunks))
        # Reported before the expensive step rather than after it, with nothing
        # held: a progress row written inside the ingest transaction is
        # invisible until it commits, which is when it stops being progress, and
        # a slot that holds a connection while the reporter asks for a second
        # one is a slot that cannot report at all.
        await _report(on_progress, len(chunks), 0)

        # Phase 2, spend.  The embedding cache lives on this connection — read
        # what has already been paid for, embed the rest, put it back — and no
        # transaction is open across it: the vectors are content-addressed and
        # are valid whether or not the version below commits.
        async with self._pool.acquire() as conn:
            await conn.execute(_SET_ORG_SQL, str(self._org_id))
            stored, vectored = await self._known_content(conn, generation, texts)
            cached = await self._cache_lookup(
                conn, policy, generation, [t for t in texts if t not in vectored]
            )
            missing = [t for t in texts if t not in vectored and t not in cached]
            fresh = {
                chunk_text: HalfVector(vector)
                for chunk_text, vector in zip(missing, self._embed_texts(missing))
            }
            await self._cache_store(conn, policy, generation, fresh)

        vectors = {**cached, **fresh}
        await _report(on_progress, len(chunks), len(texts))
        # The extractor is a model call per chunk, so it runs with nothing held
        # at all: its cache takes a connection for one statement, once.
        extracted, extract_cache = await self._extract(
            policy, [t for t in texts if t not in stored]
        )
        proposition_vectors = [
            HalfVector(vector)
            for vector in self._embed_texts([prop.text for _, prop in extracted])
        ]

        # Phase 3, write.  One transaction, as D6 requires of the flip, and no
        # model call inside it.
        async with self._pool.acquire() as conn:
            await conn.execute(_SET_ORG_SQL, str(self._org_id))
            async with conn.transaction():
                document_id = await self._document(conn, external_id, uri, source)
                opened = await conn.fetchrow(
                    _OPEN_VERSION_SQL, document_id, version_hash
                )
                version_id = opened["version_id"]
                # A crawl that landed the same content while we were embedding.
                if not opened["is_new"]:
                    return CorpusIngestResult(
                        document_id=document_id, version_id=version_id, changed=False
                    )

                provenance_id = await self._attribute(conn, version_id, given, uri)
                await self._update_document(conn, document_id, source, uri)
                linked = await self._link_chunks(conn, version_id, chunks, asserted)
                new_chunks = list(
                    {
                        chunk_id: chunk_text
                        for chunk_id, chunk_text, is_new in linked
                        if is_new
                    }.items()
                )
                distinct = len({chunk_id for chunk_id, _, _ in linked})
                await self._write_vectors(conn, generation, new_chunks, vectors)
                # Before the flip, in the order D6 gives: the facts a version
                # carries have to exist by the time it becomes the current one.
                propositions = await self._write_propositions(
                    conn,
                    policy,
                    generation,
                    provenance_id,
                    asserted,
                    linked,
                    extracted,
                    proposition_vectors,
                )
                if extract_cache is not None:
                    await extract_cache.flush(conn)
                await conn.execute(_PROMOTE_SQL, version_id)

        return CorpusIngestResult(
            document_id=document_id,
            version_id=version_id,
            changed=True,
            chunks_total=len(linked),
            chunks_new=len(new_chunks),
            chunks_carried=distinct - len(new_chunks),
            embedded=len(fresh),
            cache_hits=len(cached),
            propositions=propositions,
        )

    async def _known_content(
        self,
        conn: asyncpg.Connection,
        generation: _Generation,
        texts: Sequence[str],
    ) -> tuple[set[str], set[str]]:
        """Which of these texts are stored already, and which carry a vector.

        Two answers from one round trip, because they drive two different
        decisions: a stored text is one this version will carry without
        extracting it again, and a vectored one is a text nobody has to embed.
        """
        by_hash = {content_hash(chunk_text): chunk_text for chunk_text in texts}
        rows = await conn.fetch(
            _KNOWN_CONTENT_SQL, self._org_id, generation.generation_id, list(by_hash)
        )
        stored = {by_hash[row["content_hash"]] for row in rows}
        vectored = {
            by_hash[row["content_hash"]] for row in rows if row["has_vector"]
        }
        return stored, vectored

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
        given: Provenance,
        uri: str | None,
    ) -> UUID:
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
        chunks: Sequence[Chunk],
        asserted_at: datetime | None,
    ) -> list[tuple[UUID, str, bool]]:
        """Link every chunk of the document to the open version, in one call.

        The database decides which chunks are new: it holds the content
        addresses, and a look-then-insert from here would race a second
        connector crawling the same boilerplate.
        """
        rows = await conn.fetch(
            _ADD_CHUNKS_SQL,
            version_id,
            [chunk.ordinal for chunk in chunks],
            [chunk.text for chunk in chunks],
            None,
            asserted_at,
        )
        return [
            (row["chunk_id"], chunks[row["n"] - 1].text, row["is_new"])
            for row in rows
        ]

    async def _write_vectors(
        self,
        conn: asyncpg.Connection,
        generation: _Generation,
        new_chunks: list[tuple[UUID, str]],
        vectors: dict[str, HalfVector],
    ) -> None:
        """Give a vector to every chunk the database created, and to no other.

        A carried-over chunk already holds the vector for its content, and its
        content cannot have changed — that is what its address means.  A chunk
        the decide phase expected to be carried and the write phase created is
        left for the next crawl rather than embedded here, because embedding
        here would be a model call inside this transaction.
        """
        vectorised = [
            (chunk_id, vectors[chunk_text])
            for chunk_id, chunk_text in new_chunks
            if chunk_text in vectors
        ]
        if not vectorised:
            return
        await conn.execute(
            _WRITE_CHUNK_EMBEDDINGS_SQL,
            generation.generation_id,
            [chunk_id for chunk_id, _ in vectorised],
            [vector for _, vector in vectorised],
        )

    async def _cache_lookup(
        self,
        conn: asyncpg.Connection,
        policy: _CollectionPolicy,
        generation: _Generation,
        texts: Sequence[str],
    ) -> dict[str, HalfVector]:
        if not policy.public_source or not texts:
            return {}
        by_hash = {content_hash(chunk_text): chunk_text for chunk_text in texts}
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
            [content_hash(chunk_text) for chunk_text in fresh],
            list(fresh.values()),
        )

    async def _extract(
        self, policy: _CollectionPolicy, new_texts: Sequence[str]
    ) -> tuple[list[tuple[str, Proposition]], _BatchExtractCache | None]:
        """Extract from the chunks this version adds, if the collection opted in.

        New content only, for the same reason only new content is embedded: a
        carried chunk's text has not changed, so neither have the facts in it —
        and extraction is the expensive half twice over, priced per token and
        recurring on every prompt version bump.

        Keyed by chunk text rather than by chunk id, because the ids do not
        exist yet: this runs before the transaction that creates them.
        """
        if not policy.extract_propositions or not new_texts:
            return [], None

        cache = (
            _BatchExtractCache(self._pool, new_texts, org_id=self._org_id)
            if self._use_extract_cache
            else None
        )
        extracted = [
            (chunk_text, prop)
            for chunk_text in new_texts
            for prop in await ml.extract_propositions_async(chunk_text, cache=cache)
        ]
        return extracted, cache

    async def _write_propositions(
        self,
        conn: asyncpg.Connection,
        policy: _CollectionPolicy,
        generation: _Generation,
        provenance_id: UUID,
        asserted_at: datetime | None,
        linked: list[tuple[UUID, str, bool]],
        extracted: list[tuple[str, Proposition]],
        vectors: list[HalfVector],
    ) -> int:
        """Land the extracted facts, each citing the passage it came from."""
        if not extracted:
            return 0
        chunk_ids = {chunk_text: chunk_id for chunk_id, chunk_text, _ in linked}
        rows = [
            (chunk_ids[chunk_text], prop, vector)
            for (chunk_text, prop), vector in zip(extracted, vectors)
            if chunk_text in chunk_ids
        ]
        if not rows:
            return 0
        await conn.execute(
            _INSERT_PROPOSITIONS_SQL,
            self._org_id,
            self._collection_id,
            policy.claim_scope,
            provenance_id,
            asserted_at,
            generation.generation_id,
            [prop.text for _, prop, _ in rows],
            [vector for _, _, vector in rows],
            [prop.predicate for _, prop, _ in rows],
            [prop.object for _, prop, _ in rows],
            [chunk_id for chunk_id, _, _ in rows],
        )
        return len(rows)
