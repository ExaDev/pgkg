from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from time import monotonic
from uuid import UUID

import asyncpg
from pgvector import HalfVector
from pydantic import BaseModel

from pgkg import ml
from pgkg.chunking import DEFAULT_MAX_CHARS, chunk_document
from pgkg.config import (
    DEFAULT_COLLECTION_ID,
    DEFAULT_ORG_ID,
    ORG_GUC,
    SYSTEM_ORG_ID,
    get_settings,
    live_generations,
)
from pgkg.ml import ExtractCache, Proposition


# How long a Memory may hold unwritten access counts before the next recall
# flushes them, and how many propositions it may hold before flushing early.
# The interval trades staleness of the frequency term for writes: at 100
# recalls/sec of 20 rows, five seconds turns 10,000 row updates into at most a
# few hundred, one statement.
ACCESS_FLUSH_INTERVAL_SECONDS = 5.0
ACCESS_FLUSH_MAX_PENDING = 512


CLAIM_SCOPES = frozenset({"world", "org", "user"})
VISIBILITIES = frozenset({"private", "shared"})

# The two retrievable classes: remembered facts and retrieved passages.  Adding
# a third — summaries, entity cards, tool output — is a candidate function in
# SQL and one more name here (ADR 0001, D1).
SOURCES = ("propositions", "chunks")

# The quota that stops the corpus drowning the memory, as D1 sizes it: the
# corpus takes at most 60% of the reranker's input and never the last eight
# slots of personal material.  Arguments rather than constants in SQL, because
# tuning them must not need a migration.
DEFAULT_K_RERANK = 64
DEFAULT_CORPUS_FRACTION = 0.6
DEFAULT_MEMORY_FLOOR = 8

# The reasons belief can end, as migration 021's CHECK constraint enumerates
# them.  Rejecting an unknown reason here rather than letting the constraint do
# it keeps the failure at the call site, where the caller knows what it meant.
INVALIDATION_REASONS = frozenset(
    {
        "superseded",
        "source_updated",
        "source_deleted",
        "ttl",
        "user_deleted",
        "contradicted",
        "retracted_run",
    }
)

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class Scope:
    """Who is asking, and what they may see.

    Bound to a Memory rather than passed per call, deliberately.  In a
    multi-tenant product one call site that forgets the org argument is a
    cross-customer read, and a default argument cannot fail loudly.  A Memory
    holds nothing but a pool reference, so serving another tenant is
    `memory.scoped(other)` — one object, not one pool.

    Reads widen (the org's own rows plus the system org's shared collections,
    plus subscribed collections); writes never do.  There is exactly one org and
    one collection a Memory can write into.
    """

    org_id: UUID = DEFAULT_ORG_ID
    collection_id: UUID = DEFAULT_COLLECTION_ID
    user_id: UUID | None = None
    claim_scope: str = "org"
    visibility: str = "shared"
    acl_group_id: UUID | None = None
    acl_groups: tuple[UUID, ...] = ()
    subscribed_collection_ids: tuple[UUID, ...] = ()
    include_system_org: bool = False

    def __post_init__(self) -> None:
        if self.claim_scope not in CLAIM_SCOPES:
            raise ValueError(
                f"claim_scope must be one of {sorted(CLAIM_SCOPES)};"
                f" got {self.claim_scope!r}"
            )
        if self.visibility not in VISIBILITIES:
            raise ValueError(
                f"visibility must be one of {sorted(VISIBILITIES)};"
                f" got {self.visibility!r}"
            )
        if self.visibility == "private" and self.user_id is None:
            raise ValueError("a private scope must name the user who owns the row")

    @property
    def read_org_ids(self) -> list[UUID]:
        if self.include_system_org:
            return [self.org_id, SYSTEM_ORG_ID]
        return [self.org_id]

    @property
    def write_org_id(self) -> UUID:
        return self.org_id

    @property
    def read_collection_ids(self) -> list[UUID]:
        return [self.collection_id, *self.subscribed_collection_ids]

    @property
    def write_collection_id(self) -> UUID:
        return self.collection_id

    @property
    def owner_user_id(self) -> UUID | None:
        """The owner column, which the CHECK constraint ties to `private`.

        A shared row written by a known user still records that user — as the
        provenance actor, not as an owner.
        """
        return self.user_id if self.visibility == "private" else None

    @property
    def read_acl_groups(self) -> list[UUID]:
        return list(self.acl_groups)


@dataclass(frozen=True)
class Provenance:
    """How a batch of rows came to exist (ADR 0001, D5).

    One row per chunk and model run, so its cardinality tracks chunk count and
    not fact count.  The producer, model and prompt version default from the
    ingest mode and the settings actually in force; everything else is the
    caller's to state, because nothing else can know it.
    """

    kind: str = "document_version"
    producer: str | None = None
    producer_model: str | None = None
    prompt_version: str | None = None
    ingest_run_id: UUID | None = None
    actor_user_id: UUID | None = None
    source_id: UUID | None = None
    source_url: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime | None = None
    licence: str | None = None
    source_authority: int | None = None


class BeliefRecord(BaseModel):
    """One proposition as the system held it at an instant.

    Deliberately not a Result: there is no ranking here and no score, because
    the question the audit path answers is what was believed, not what is
    relevant.
    """

    proposition_id: UUID
    text: str
    recorded_at: datetime
    invalidated_at: datetime | None
    invalidation_reason: str | None
    valid_from: datetime | None
    valid_to: datetime | None
    provenance_id: UUID


class Result(BaseModel):
    """One retrieved item, and which store it came out of.

    An agent has to be able to tell a remembered fact from a retrieved passage
    — they answer differently, they are trusted differently, and only one of
    them can be forgotten — so `source` and `collection_id` travel with every
    row rather than being inferred from which fields happen to be NULL.

    `proposition_id` is the fact's id and is NULL for a passage; `item_id` is
    the row's own id whichever store it came from.  The pair is deliberate: a
    caller holding a Result and wanting to forget it needs the first, and a
    caller deduplicating or citing needs the second.
    """

    item_id: UUID
    source: str
    proposition_id: UUID | None
    text: str
    context_text: str | None = None
    score: float
    rrf_score: float
    source_kind: str
    bucket: str | None = None
    claim_scope: str | None = None
    collection_id: UUID | None = None
    chunk_id: UUID | None
    subject: str | None
    predicate: str | None
    object: str | None
    asserted_at: datetime | None = None


@dataclass
class IngestResult:
    documents: int
    chunks: int
    propositions: int
    entities: int
    # The retraction handle: one bad extractor deploy is undone by
    # pgkg_retract_ingest_run(), and it needs the id the batch was written with.
    ingest_run_id: UUID | None = None
    provenance_ids: tuple[UUID, ...] = ()


@dataclass(frozen=True)
class _PendingChunk:
    id: UUID
    text: str
    span_start: int
    span_end: int
    ordinal: int = 0
    # Set by the chunks-only planner and by nobody else: in that mode the chunk
    # is the retrievable row, so it carries the vector the extracted mode puts
    # on a proposition (ADR 0001, D1).
    embedding: HalfVector | None = None


@dataclass(frozen=True)
class _PendingEntity:
    name: str
    embedding: HalfVector


@dataclass(frozen=True)
class _PendingProposition:
    id: UUID
    text: str
    embedding: HalfVector
    chunk_id: UUID
    predicate: str | None
    subject_name: str | None
    object_name: str | None
    object_literal: str | None
    metadata: str | None


@dataclass(frozen=True)
class _IngestPlan:
    """Everything ingest() will write, computed before a connection is taken.

    Extraction and embedding happen outside the transaction on purpose: they are
    the slow part, they can take minutes for a long document, and the extract
    cache needs a pool connection of its own — holding the ingest connection
    across them starves the pool under concurrent ingest.
    """

    chunks: tuple[_PendingChunk, ...]
    entities: tuple[_PendingEntity, ...]
    propositions: tuple[_PendingProposition, ...]


_INSERT_DOCUMENT_SQL = """
INSERT INTO documents (id, source, namespace, org_id, collection_id)
VALUES ($1, $2, $3, $4, $5)
"""

# One derivation record per chunk, which is what makes provenance cardinality
# track chunk count rather than fact count.  The locator is per chunk because
# that is what a citation needs: the span the claim came out of.
_INSERT_PROVENANCE_SQL = """
INSERT INTO provenance
    (id, org_id, kind, source_id, source_locator, producer, producer_model,
     prompt_version, ingest_run_id, actor_user_id, source_url, publisher,
     published_at, retrieved_at, licence, source_authority)
SELECT r.id, $1, $2, $3, r.locator::jsonb, $4, $5, $6, $7, $8, $9, $10, $11,
       $12, $13, $14
FROM unnest($15::uuid[], $16::text[]) AS r(id, locator)
"""

_INSERT_CHUNKS_SQL = """
INSERT INTO chunks
    (id, document_id, text, span_start, span_end, asserted_at, org_id,
     collection_id, visibility, owner_user_id, acl_group_id, provenance_id)
SELECT c.id, $1, c.text, c.span_start, c.span_end, $2, $3, $4, $5, $6, $7,
       c.provenance_id
FROM unnest($8::uuid[], $9::text[], $10::int[], $11::int[], $12::uuid[])
     AS c(id, text, span_start, span_end, provenance_id)
"""

# One statement, one plpgsql call per distinct name.  Resolution stays
# sequential — pgkg_link_entity is VOLATILE, so each invocation sees the
# entities the earlier ones inserted, which is what lets two spellings of the
# same name inside a single batch collapse onto one row.
_LINK_ENTITIES_SQL = """
SELECT n.name, pgkg_link_entity($1, n.name, 'concept', n.embedding) AS entity_id
FROM unnest($2::text[], $3::halfvec[]) AS n(name, embedding)
"""

# The many-to-one link is written by the same statement as the row it links, so
# a proposition never exists with a hot-path provenance_id and no entry in the
# table erasure counts.  One statement, not two: the constraint check happens at
# the end of the statement, by which time both inserts have run.
_INSERT_PROPOSITIONS_SQL = """
WITH inserted AS (
    INSERT INTO propositions
        (id, text, embedding, subject_id, predicate, object_id, object_literal,
         chunk_id, namespace, session_id, metadata, asserted_at, org_id,
         collection_id, claim_scope, visibility, owner_user_id, acl_group_id,
         provenance_id)
    SELECT p.id, p.text, p.embedding, p.subject_id, p.predicate, p.object_id,
           p.object_literal, p.chunk_id, $1, $2,
           COALESCE(p.metadata::jsonb, '{}'::jsonb), $3, $4, $5, $6, $7, $8, $9,
           p.provenance_id
    FROM unnest($10::uuid[], $11::text[], $12::halfvec[], $13::uuid[],
                $14::text[], $15::uuid[], $16::text[], $17::uuid[], $18::text[],
                $19::uuid[])
         AS p(id, text, embedding, subject_id, predicate, object_id,
              object_literal, chunk_id, metadata, provenance_id)
    RETURNING id, provenance_id
)
INSERT INTO proposition_provenance (proposition_id, provenance_id)
SELECT id, provenance_id FROM inserted
ON CONFLICT DO NOTHING
"""

# The chunks-only path, which writes into the chunk store instead (ADR 0001,
# D1 and phase 2).  It is the corpus pipeline's write phase over a chat turn:
# one document, one version, its passages added through the function that holds
# the content addresses, then the flip.
#
# ONE DERIVATION RECORD PER VERSION, not per chunk.  A content-addressed passage
# is shared by every version carrying it, so it cannot record which ingest
# produced it — the version can, and pgkg_add_version_chunk() reads it from
# there.  The locator carries no span for the same reason: a shared row has no
# one offset into no one text.
_INSERT_VERSION_PROVENANCE_SQL = """
INSERT INTO provenance
    (org_id, kind, source_id, source_locator, producer, producer_model,
     prompt_version, ingest_run_id, actor_user_id, source_url, publisher,
     published_at, retrieved_at, licence, source_authority)
VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
RETURNING id
"""

# The version hash is digested in SQL rather than in Python so that the one
# statement of what a version hash is stays the one the database applies.
_OPEN_VERSION_SQL = """
SELECT version_id
FROM pgkg_open_document_version($1, digest($2, 'sha256'), $3)
"""

# Every passage of the turn in one round trip, the shape corpus ingest uses: a
# loop over this function is one network round trip per chunk with the ingest
# transaction held open.
_ADD_VERSION_CHUNKS_SQL = """
SELECT t.n, added.chunk_id, added.is_new
FROM unnest($2::int[], $3::text[]) WITH ORDINALITY AS t(ord, chunk_text, n)
CROSS JOIN LATERAL
    pgkg_add_version_chunk($1, t.ord, t.chunk_text, $4, $5, $6, $7) AS added
ORDER BY t.n
"""

# The immutability trigger fires on text and org only, so writing the vector of
# a chunk the database just created is not a rewrite of its content.  Only the
# rows it created: a reused passage already holds the vector for its content,
# and its content cannot have changed — that is what its address means.
_WRITE_CHUNK_EMBEDDINGS_SQL = """
UPDATE chunks c
SET embedding = e.embedding
FROM unnest($1::uuid[], $2::halfvec[]) AS e(id, embedding)
WHERE c.id = e.id
"""

_PROMOTE_VERSION_SQL = "SELECT pgkg_promote_document_version($1)"

_INSERT_EDGES_SQL = """
INSERT INTO edges (src_entity, dst_entity, relation, proposition_id)
SELECT e.src, e.dst, e.relation, e.proposition_id
FROM unnest($1::uuid[], $2::uuid[], $3::text[], $4::uuid[])
     AS e(src, dst, relation, proposition_id)
ON CONFLICT DO NOTHING
"""

# The two-store surface, not pgkg_search(): a passage and a fact are both
# answers to "what did we agree about the refund policy", and deciding between
# them is the caller's job, not the retriever's.  pgkg_search() keeps its
# proposition-shaped contract for the callers that want exactly that.
#
# The embedding leaves as `vector` even though both columns are halfvec:
# widening is exact and the client codec for it is the one _parse_emb
# understands.  It comes from whichever store the row is in, and only when that
# row was embedded by the org's PRIMARY generation: a vector from another model
# space is still a vector, so a cosine against it returns a number rather than
# an error, and MMR would rank on it (ADR 0001, D8).  A row from a transitional
# generation therefore reaches the caller with no embedding, and MMR leaves it
# in the order the scorer gave it.
#
# The scope arrays are passed positionally, so every scoring knob between them
# has to be spelled out too.  The generation queries are aggregated into
# pgkg_gen_query[] server-side: a composite carrying a halfvec has no client
# codec, and two plain arrays do.
_RECALL_SQL = """
SELECT r.item_id, r.source, r.text, r.context_text, r.rrf_score,
       r.adjusted_score, r.source_kind, r.bucket, r.claim_scope,
       r.collection_id, r.asserted_at,
       CASE
           WHEN COALESCE(p.embedder_generation_id, c.embedder_generation_id)
                = primary_generation.generation_id
           THEN COALESCE(p.embedding, c.embedding)::vector
       END AS embedding,
       p.id AS proposition_id,
       COALESCE(p.chunk_id, c.id) AS chunk_id,
       p.predicate,
       subj.name AS subject_name,
       COALESCE(obj.name, p.object_literal) AS object_name
FROM pgkg_retrieve(
        $1, $2::halfvec, $3, $4, $5, $6, 30.0, $7, 60,
        $8::uuid[], $9::uuid[], $10::uuid, $11::uuid[], $12::timestamptz,
        $13, $14, $15, 1.0, 1.0, 1.0, 730.0, 1, 1, $16::text[],
        (SELECT ARRAY(
            SELECT ROW(g.generation_id, g.q_embedding)::pgkg_gen_query
            FROM unnest($17::uuid[], $18::halfvec[])
                 AS g(generation_id, q_embedding)
        ))
     ) r
LEFT JOIN propositions p
       ON r.source = 'propositions' AND p.id = r.item_id
LEFT JOIN chunks c ON r.source = 'chunks' AND c.id = r.item_id
LEFT JOIN entities subj ON subj.id = p.subject_id
LEFT JOIN entities obj ON obj.id = p.object_id
LEFT JOIN LATERAL (
    SELECT oe.generation_id
    FROM org_embedders oe
    WHERE oe.org_id = $19::uuid AND oe.role = 'primary'
) primary_generation ON TRUE
ORDER BY r.adjusted_score DESC
"""

_BELIEVED_AT_SQL = """
SELECT proposition_id, text, recorded_at, invalidated_at, invalidation_reason,
       valid_from, valid_to, provenance_id
FROM pgkg_believed_at($1, $2, $3, $4::uuid[], $5::uuid[], $6::uuid, $7::uuid[])
"""

# The reason is set, not defaulted by the trigger: a bulk withdrawal and a user
# deletion are both invisible to retrieval and only the reason tells them apart.
# COALESCE keeps the first withdrawal's reason, so forgetting twice does not
# rewrite history.
#
# One org, singular: withdrawal is a write, and a Scope's reads widen where its
# writes do not.  A widened read admits the system org's shared material, which
# every subscriber reads and only the operator may withdraw.
_FORGET_SQL = """
UPDATE propositions
SET superseded_by = COALESCE($1, superseded_by),
    invalidated_at = COALESCE(invalidated_at, now()),
    invalidation_reason = COALESCE(invalidation_reason, $2)
WHERE id = $3 AND org_id = $4
"""

# The org predicate is stated here as well as carried by the GUC: the counts are
# grouped by the org that read them, and a row belonging to another org — one a
# widened read returned — must not be credited even on a connection for which
# RLS is inert.
_FLUSH_ACCESS_SQL = """
UPDATE propositions p
SET access_count = p.access_count + a.n,
    last_accessed_at = now()
FROM unnest($1::uuid[], $2::int[]) AS a(id, n)
WHERE p.id = a.id AND p.org_id = $3
"""

# Session-level rather than transaction-local: it has to be in force for the
# whole time the connection is held, including outside any transaction.  asyncpg
# resets the session when the connection returns to the pool, and every acquire
# sets it again, so no statement ever reads another tenant's value.
_SET_ORG_SQL = f"SELECT set_config('{ORG_GUC}', $1, false)"

# Creating an org and binding it to an embedder are one operation, because an
# unbound org has no primary generation: its rows would be written with an
# inline vector that pgkg_generation_candidates() refuses to search.  Migration
# 022 seeds the binding for the orgs that existed then; every later org gets it
# here.
_PROVISION_ORG_SQL = """
WITH created AS (
    INSERT INTO orgs (name, is_system) VALUES ($1, $2) RETURNING id
),
primary_generation AS (
    SELECT id FROM embedder_generations
    WHERE status = 'primary'
    ORDER BY created_at, id
    LIMIT 1
),
bound AS (
    INSERT INTO org_embedders (org_id, generation_id, role)
    SELECT created.id, primary_generation.id, 'primary'
    FROM created, primary_generation
    RETURNING org_id
)
SELECT id FROM created
"""


async def provision_org(
    conn: asyncpg.Connection, name: str, *, is_system: bool = False
) -> UUID:
    """Create an org and bind it to the live primary embedder generation."""
    return await conn.fetchval(_PROVISION_ORG_SQL, name, is_system)


class _AccessLedger:
    """The access counts the read path deliberately did not write.

    Shared between a Memory and the scoped copies it hands out, so a
    per-request scope that lives for one HTTP call cannot take its accounting to
    the grave — the long-lived Memory that owns the ledger flushes it on close.
    Keyed by org as well as proposition, because the write has to run under the
    org whose RLS policy admits the row.
    """

    def __init__(self) -> None:
        self.pending: dict[tuple[UUID, UUID], int] = {}
        self.last_flush = monotonic()

    def record(self, org_id: UUID, proposition_ids: Iterable[UUID]) -> None:
        for prop_id in proposition_ids:
            key = (org_id, prop_id)
            self.pending[key] = self.pending.get(key, 0) + 1

    def drain(self) -> dict[tuple[UUID, UUID], int]:
        drained, self.pending = self.pending, {}
        self.last_flush = monotonic()
        return drained

    def restore(self, counts: dict[tuple[UUID, UUID], int]) -> None:
        for key, count in counts.items():
            self.pending[key] = self.pending.get(key, 0) + count


class PostgresExtractCache:
    """Postgres-backed implementation of ExtractCache.

    Stores extracted propositions in the proposition_cache table so re-ingesting
    the same chunk with the same extractor model and prompt version is free.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        namespace: str,
        *,
        org_id: UUID = DEFAULT_ORG_ID,
    ) -> None:
        self._pool = pool
        self._namespace = namespace
        # The org is part of the key, not only of the row policy: a hit is the
        # extracted claims themselves, and the deployments that connect as the
        # owning role are the ones RLS cannot help (ADR 0001, D4).
        self._org_id = org_id

    async def get(self, cache_key: str) -> list[Proposition] | None:
        # The hit count is bumped by the same statement that reads the payload:
        # a lookup on the ingest path should cost one round trip, not two.
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE proposition_cache
                SET hit_count = hit_count + 1
                WHERE cache_key = $1 AND org_id = $2
                RETURNING propositions
                """,
                cache_key,
                self._org_id,
            )
            if row is None:
                return None
            raw = row["propositions"]
            if isinstance(raw, str):
                items = json.loads(raw)
            else:
                items = raw  # asyncpg may already decode JSONB
            return [Proposition(**p) for p in items]

    async def put(
        self,
        cache_key: str,
        chunk_hash: str,
        extractor_model: str,
        prompt_version: str,
        props: list[Proposition],
    ) -> None:
        payload = json.dumps([p.model_dump() for p in props])
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO proposition_cache
                    (cache_key, chunk_hash, extractor_model, prompt_version,
                     propositions, org_id)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                ON CONFLICT (cache_key, org_id) DO NOTHING
                """,
                cache_key,
                chunk_hash,
                extractor_model,
                prompt_version,
                payload,
                self._org_id,
            )


class Memory:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        namespace: str = "default",
        scope: Scope | None = None,
        use_extract_cache: bool = True,
        extract_propositions: bool = True,
        access_flush_interval: float = ACCESS_FLUSH_INTERVAL_SECONDS,
        embedders: Mapping[str, EmbedFn] | None = None,
    ) -> None:
        self._pool = pool
        self._namespace = namespace
        self._scope = scope if scope is not None else Scope()
        self._use_extract_cache = use_extract_cache
        self._extract_propositions = extract_propositions
        self._extract_cache: ExtractCache | None = (
            PostgresExtractCache(
                pool, namespace, org_id=self._scope.write_org_id
            )
            if use_extract_cache and extract_propositions
            else None
        )
        self._access_flush_interval = access_flush_interval
        self._access = _AccessLedger()
        # Keyed by generation name.  Empty means one live generation, embedded
        # by ml.embed — the single-generation case, which pays for no registry
        # lookup.  A cutover window supplies the second embedder here.
        self._embedders: Mapping[str, EmbedFn] = dict(embedders or {})

    @property
    def scope(self) -> Scope:
        return self._scope

    def scoped(self, scope: Scope) -> Memory:
        """The same store, read and written as someone else.

        The access ledger is shared, not copied: a request-scoped Memory is
        discarded when the request ends, and the accounting it gathered has to
        outlive it.
        """
        other = Memory(
            self._pool,
            namespace=self._namespace,
            scope=scope,
            use_extract_cache=self._use_extract_cache,
            extract_propositions=self._extract_propositions,
            access_flush_interval=self._access_flush_interval,
            embedders=self._embedders,
        )
        other._access = self._access
        return other

    @asynccontextmanager
    async def _connection(self):
        """A pooled connection with this scope's org in the RLS GUC.

        Every path goes through here.  RLS is what stops a missing predicate
        being a cross-customer breach, and it has nothing to read unless the
        connection carries the org — as does the entities.org_id default, since
        pgkg_link_entity() takes no org argument.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(_SET_ORG_SQL, str(self._scope.org_id))
            yield conn

    async def ingest(
        self,
        text: str,
        *,
        source: str | None = None,
        session_id: str | None = None,
        asserted_at: datetime | None = None,
        chunk_size: int = DEFAULT_MAX_CHARS,
        provenance: Provenance | None = None,
    ) -> IngestResult:
        # Boundaries come from the text's own structure, so re-ingesting an
        # edited document rewrites only the chunks around the edit.  The spans
        # are real offsets into `text`, which is what small-to-big context
        # expansion needs to widen a hit back out to its neighbourhood.
        chunks = tuple(
            _PendingChunk(
                id=uuid.uuid4(),
                text=chunk.text,
                span_start=chunk.span_start,
                span_end=chunk.span_end,
                ordinal=chunk.ordinal,
            )
            for chunk in chunk_document(text, max_chars=chunk_size)
        )

        plan = (
            await self._plan_extracted(chunks)
            if self._extract_propositions
            else self._plan_chunks_only(chunks)
        )
        resolved = self._resolve(provenance)
        # D5: published_at feeds the perishable decay profile, and D6 keys that
        # profile on asserted_at.  Ingest time is what neither of them means, so
        # a caller that gave a publication date and no world-time gets the
        # publication date; one that gave both keeps its own.
        asserted = (
            asserted_at if asserted_at is not None else resolved.published_at
        )
        if not self._extract_propositions:
            return await self._write_chunk_store(plan, text, source, asserted, resolved)
        return await self._write(plan, source, session_id, asserted, resolved)

    def _resolve(self, provenance: Provenance | None) -> Provenance:
        """Fill in what the ingest mode and the settings in force already know.

        The producer follows the mode because it is the mode: an LLM extracted
        these propositions, or the chunker made them.  The model and prompt
        version are only meaningful for the former.
        """
        given = provenance if provenance is not None else Provenance()
        producer = given.producer or (
            "llm_extract" if self._extract_propositions else "chunker"
        )
        if producer == "llm_extract":
            settings = get_settings()
            producer_model = (
                given.producer_model
                or settings.extractor_model
                or settings.llm_model
            )
            prompt_version = given.prompt_version or ml.PROMPT_VERSION
        else:
            producer_model = given.producer_model
            prompt_version = given.prompt_version

        return replace(
            given,
            producer=producer,
            producer_model=producer_model,
            prompt_version=prompt_version,
            ingest_run_id=given.ingest_run_id or uuid.uuid4(),
            actor_user_id=given.actor_user_id or self._scope.user_id,
        )

    async def _write(
        self,
        plan: _IngestPlan,
        source: str | None,
        session_id: str | None,
        asserted_at: datetime | None,
        provenance: Provenance,
    ) -> IngestResult:
        doc_id = uuid.uuid4()
        provenance_ids = {chunk.id: uuid.uuid4() for chunk in plan.chunks}
        async with self._connection() as conn:
            # One transaction: a document, its chunks and its propositions are
            # either all visible or none are.  Phase 2's version flip depends on
            # this boundary existing.
            async with conn.transaction():
                await conn.execute(
                    _INSERT_DOCUMENT_SQL,
                    doc_id,
                    source,
                    self._namespace,
                    self._scope.write_org_id,
                    self._scope.write_collection_id,
                )
                await self._write_provenance(
                    conn, plan.chunks, provenance_ids, provenance, source
                )
                await self._write_chunks(
                    conn, doc_id, plan.chunks, asserted_at, provenance_ids
                )
                entity_ids = await self._link_entities(conn, plan.entities)
                await self._write_propositions(
                    conn,
                    plan.propositions,
                    entity_ids,
                    session_id,
                    asserted_at,
                    provenance_ids,
                )
                await self._write_edges(conn, plan.propositions, entity_ids)

        return IngestResult(
            documents=1,
            chunks=len(plan.chunks),
            propositions=len(plan.propositions),
            entities=len(set(entity_ids.values())),
            ingest_run_id=provenance.ingest_run_id,
            provenance_ids=tuple(provenance_ids.values()),
        )

    async def _plan_extracted(self, chunks: tuple[_PendingChunk, ...]) -> _IngestPlan:
        entities: dict[str, _PendingEntity] = {}
        propositions: list[_PendingProposition] = []

        for chunk in chunks:
            extracted = await ml.extract_propositions_async(
                chunk.text, cache=self._extract_cache
            )
            if not extracted:
                continue

            # Embedding stays per chunk so a long document never builds one
            # unbounded batch for the embedder.
            entity_names: list[str] = []
            for prop in extracted:
                entity_names.append(prop.subject)
                if not prop.object_is_literal:
                    entity_names.append(prop.object)

            embeddings = ml.embed(entity_names + [p.text for p in extracted])
            entity_embs = embeddings[: len(entity_names)]
            prop_embs = embeddings[len(entity_names):]

            entity_idx = 0
            for prop, prop_emb in zip(extracted, prop_embs):
                subject_name = prop.subject
                entities.setdefault(
                    subject_name,
                    _PendingEntity(subject_name, HalfVector(entity_embs[entity_idx])),
                )
                entity_idx += 1

                object_name: str | None = None
                object_literal: str | None = None
                if prop.object_is_literal:
                    object_literal = prop.object
                else:
                    object_name = prop.object
                    entities.setdefault(
                        object_name,
                        _PendingEntity(object_name, HalfVector(entity_embs[entity_idx])),
                    )
                    entity_idx += 1

                propositions.append(
                    _PendingProposition(
                        id=uuid.uuid4(),
                        text=prop.text,
                        embedding=HalfVector(prop_emb),
                        chunk_id=chunk.id,
                        predicate=prop.predicate,
                        subject_name=subject_name,
                        object_name=object_name,
                        object_literal=object_literal,
                        metadata=None,
                    )
                )

        return _IngestPlan(
            chunks=chunks,
            entities=tuple(entities.values()),
            propositions=tuple(propositions),
        )

    def _plan_chunks_only(self, chunks: tuple[_PendingChunk, ...]) -> _IngestPlan:
        """Chunks-only mode: the passage is the retrievable row.

        So the vector goes on the chunk, and the plan carries no propositions,
        no entities and no edges at all.  It used to carry one proposition per
        chunk with a NULL subject, predicate and object — the hack D1 lists
        under what not to build and tells us to undo: a 600-token passage
        BM25 length-normalised against 12-token facts, on a chat fact's
        lifecycle, under a chat fact's index.
        """
        embeddings = ml.embed([chunk.text for chunk in chunks])
        return _IngestPlan(
            chunks=tuple(
                replace(chunk, embedding=HalfVector(embedding))
                for chunk, embedding in zip(chunks, embeddings)
            ),
            entities=(),
            propositions=(),
        )

    async def _write_chunk_store(
        self,
        plan: _IngestPlan,
        text: str,
        source: str | None,
        asserted_at: datetime | None,
        provenance: Provenance,
    ) -> IngestResult:
        """Land a chunks-only ingest in the chunk store, through the lifecycle.

        The same ordered sequence D6 gives the corpus, because it is the same
        write: a document, a version opened against the hash of the whole turn,
        its passages added through the function that owns the content addresses,
        the vectors of the passages that function created, then the flip.  One
        transaction, so retrieval never sees a version whose passages are only
        half embedded.

        Nothing here calls a model.  The embedder already ran, in the planner,
        outside this connection — a chat turn is small, but the rule that keeps
        the pool free under concurrent ingest does not care how small.
        """
        doc_id = uuid.uuid4()
        async with self._connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    _INSERT_DOCUMENT_SQL,
                    doc_id,
                    source,
                    self._namespace,
                    self._scope.write_org_id,
                    self._scope.write_collection_id,
                )
                provenance_id = await self._write_version_provenance(
                    conn, provenance, source
                )
                version_id = await conn.fetchval(
                    _OPEN_VERSION_SQL, doc_id, text, provenance_id
                )
                added = await self._add_version_chunks(
                    conn, version_id, plan.chunks, asserted_at
                )
                await self._write_chunk_embeddings(conn, added)
                await conn.execute(_PROMOTE_VERSION_SQL, version_id)

        return IngestResult(
            documents=1,
            chunks=len(plan.chunks),
            propositions=0,
            entities=0,
            ingest_run_id=provenance.ingest_run_id,
            provenance_ids=(provenance_id,),
        )

    async def _write_version_provenance(
        self,
        conn: asyncpg.Connection,
        provenance: Provenance,
        source: str | None,
    ) -> UUID:
        return await conn.fetchval(
            _INSERT_VERSION_PROVENANCE_SQL,
            self._scope.write_org_id,
            provenance.kind,
            provenance.source_id,
            json.dumps({"source": source}),
            provenance.producer,
            provenance.producer_model,
            provenance.prompt_version,
            provenance.ingest_run_id,
            provenance.actor_user_id,
            provenance.source_url,
            provenance.publisher,
            provenance.published_at,
            provenance.retrieved_at,
            provenance.licence,
            provenance.source_authority,
        )

    async def _add_version_chunks(
        self,
        conn: asyncpg.Connection,
        version_id: UUID,
        chunks: tuple[_PendingChunk, ...],
        asserted_at: datetime | None,
    ) -> list[tuple[_PendingChunk, UUID, bool]]:
        """Which row each passage landed on, and which of them are new.

        The database decides, not this process: it holds the content addresses,
        and a look-then-insert from here would race a second ingest of the same
        text.  The scope travels with the call because it is part of the address
        — a passage is reused only by a writer that agrees about who may read it.

        Paired back by the ordinality the statement returns rather than by row
        order, so nothing here depends on how the lateral join is executed.
        """
        if not chunks:
            return []
        rows = await conn.fetch(
            _ADD_VERSION_CHUNKS_SQL,
            version_id,
            [chunk.ordinal for chunk in chunks],
            [chunk.text for chunk in chunks],
            self._scope.acl_group_id,
            asserted_at,
            self._scope.visibility,
            self._scope.owner_user_id,
        )
        return [
            (chunks[row["n"] - 1], row["chunk_id"], row["is_new"]) for row in rows
        ]

    async def _write_chunk_embeddings(
        self,
        conn: asyncpg.Connection,
        added: list[tuple[_PendingChunk, UUID, bool]],
    ) -> None:
        """A vector for every passage the database created, and for no other.

        A reused passage already holds the vector for its content, and its
        content cannot have changed — that is what its address means.  Keyed by
        the row rather than by the position, because a turn that repeats a
        paragraph holds two positions on one row.
        """
        vectored = {
            chunk_id: chunk.embedding
            for chunk, chunk_id, is_new in added
            if is_new and chunk.embedding is not None
        }
        if not vectored:
            return
        await conn.execute(
            _WRITE_CHUNK_EMBEDDINGS_SQL,
            list(vectored),
            list(vectored.values()),
        )

    async def _write_provenance(
        self,
        conn: asyncpg.Connection,
        chunks: tuple[_PendingChunk, ...],
        provenance_ids: dict[UUID, UUID],
        provenance: Provenance,
        source: str | None,
    ) -> None:
        if not chunks:
            return
        locators = [
            json.dumps(
                {
                    "source": source,
                    "span_start": chunk.span_start,
                    "span_end": chunk.span_end,
                }
            )
            for chunk in chunks
        ]
        await conn.execute(
            _INSERT_PROVENANCE_SQL,
            self._scope.write_org_id,
            provenance.kind,
            provenance.source_id,
            provenance.producer,
            provenance.producer_model,
            provenance.prompt_version,
            provenance.ingest_run_id,
            provenance.actor_user_id,
            provenance.source_url,
            provenance.publisher,
            provenance.published_at,
            provenance.retrieved_at,
            provenance.licence,
            provenance.source_authority,
            [provenance_ids[chunk.id] for chunk in chunks],
            locators,
        )

    async def _write_chunks(
        self,
        conn: asyncpg.Connection,
        doc_id: UUID,
        chunks: tuple[_PendingChunk, ...],
        asserted_at: datetime | None,
        provenance_ids: dict[UUID, UUID],
    ) -> None:
        if not chunks:
            return
        await conn.execute(
            _INSERT_CHUNKS_SQL,
            doc_id,
            asserted_at,
            self._scope.write_org_id,
            self._scope.write_collection_id,
            self._scope.visibility,
            self._scope.owner_user_id,
            self._scope.acl_group_id,
            [c.id for c in chunks],
            [c.text for c in chunks],
            [c.span_start for c in chunks],
            [c.span_end for c in chunks],
            [provenance_ids[c.id] for c in chunks],
        )

    async def _link_entities(
        self, conn: asyncpg.Connection, entities: tuple[_PendingEntity, ...]
    ) -> dict[str, UUID]:
        if not entities:
            return {}
        rows = await conn.fetch(
            _LINK_ENTITIES_SQL,
            self._namespace,
            [e.name for e in entities],
            [e.embedding for e in entities],
        )
        return {row["name"]: row["entity_id"] for row in rows}

    async def _write_propositions(
        self,
        conn: asyncpg.Connection,
        propositions: tuple[_PendingProposition, ...],
        entity_ids: dict[str, UUID],
        session_id: str | None,
        asserted_at: datetime | None,
        provenance_ids: dict[UUID, UUID],
    ) -> None:
        if not propositions:
            return
        await conn.execute(
            _INSERT_PROPOSITIONS_SQL,
            self._namespace,
            session_id,
            asserted_at,
            self._scope.write_org_id,
            self._scope.write_collection_id,
            self._scope.claim_scope,
            self._scope.visibility,
            self._scope.owner_user_id,
            self._scope.acl_group_id,
            [p.id for p in propositions],
            [p.text for p in propositions],
            [p.embedding for p in propositions],
            [_entity_id(entity_ids, p.subject_name) for p in propositions],
            [p.predicate for p in propositions],
            [_entity_id(entity_ids, p.object_name) for p in propositions],
            [p.object_literal for p in propositions],
            [p.chunk_id for p in propositions],
            [p.metadata for p in propositions],
            [provenance_ids[p.chunk_id] for p in propositions],
        )

    async def _write_edges(
        self,
        conn: asyncpg.Connection,
        propositions: tuple[_PendingProposition, ...],
        entity_ids: dict[str, UUID],
    ) -> None:
        edges = [
            (
                _entity_id(entity_ids, p.subject_name),
                _entity_id(entity_ids, p.object_name),
                p.predicate,
                p.id,
            )
            for p in propositions
            if p.subject_name is not None and p.object_name is not None
        ]
        if not edges:
            return
        await conn.execute(
            _INSERT_EDGES_SQL,
            [e[0] for e in edges],
            [e[1] for e in edges],
            [e[2] for e in edges],
            [e[3] for e in edges],
        )

    async def recall(
        self,
        query: str,
        *,
        k: int = 10,
        k_retrieve: int = 100,
        session_id: str | None = None,
        with_rerank: bool = True,
        with_mmr: bool = True,
        mmr_lambda: float = 0.5,
        expand_graph: bool = True,
        valid_at: datetime | None = None,
        sources: Iterable[str] | None = None,
        k_rerank: int = DEFAULT_K_RERANK,
        corpus_fraction: float = DEFAULT_CORPUS_FRACTION,
        memory_floor: int = DEFAULT_MEMORY_FLOOR,
    ) -> list[Result]:
        """One ranked list over both stores, or over one of them.

        `sources` is the per-class option D1 keeps alongside the fused default:
        an agent that already knows it wants the handbook and not the
        conversation says so, and the whole candidate budget is spent there
        rather than on the share the quota split left it.  Left unset, both
        classes compete and the quota decides.
        """
        wanted = _resolve_sources(sources)
        q_emb, gen_ids, gen_embs = await self._query_vectors(query)
        async with self._connection() as conn:
            rows = await conn.fetch(
                _RECALL_SQL,
                query,
                HalfVector(q_emb),
                k_retrieve,
                k_retrieve * 2,
                self._namespace,
                session_id,
                expand_graph,
                self._scope.read_org_ids,
                self._scope.read_collection_ids,
                self._scope.user_id,
                self._scope.read_acl_groups,
                valid_at,
                k_rerank,
                corpus_fraction,
                memory_floor,
                wanted,
                gen_ids,
                gen_embs,
                self._scope.org_id,
            )

        if not rows:
            return []

        scores = [float(r["adjusted_score"]) for r in rows]
        embs = [_parse_emb(r["embedding"], q_emb) for r in rows]

        if with_rerank:
            # The same budget the quota divided: k_rerank is defined as the
            # reranker's input, and two different numbers for it would mean the
            # split guarded slots the reranker never sees.
            cutoff = min(k_retrieve, k_rerank)
            candidate_rows = rows[:cutoff]
            candidate_embs = embs[:cutoff]
            rerank_scores = ml.rerank(query, [r["text"] for r in candidate_rows])

            # Min-max normalize both score lists
            def _normalize(vals: list[float]) -> list[float]:
                lo, hi = min(vals), max(vals)
                span = hi - lo
                if span < 1e-10:
                    return [1.0] * len(vals)
                return [(v - lo) / span for v in vals]

            adj_scores = [float(r["adjusted_score"]) for r in candidate_rows]
            rerank_norm = _normalize(rerank_scores)
            adj_norm = _normalize(adj_scores)
            blended = [0.7 * r + 0.3 * a for r, a in zip(rerank_norm, adj_norm)]

            # Sort candidate_rows by blended score
            sorted_indices = sorted(range(len(blended)), key=lambda i: blended[i], reverse=True)
            rows = [candidate_rows[i] for i in sorted_indices]
            scores = [blended[i] for i in sorted_indices]
            embs = [candidate_embs[i] for i in sorted_indices]

        if with_mmr and len(rows) > k:
            # Diversity is a cosine, and cosines from two model spaces are not
            # comparable, so MMR runs in the primary generation's space only.  A
            # row with no primary vector keeps the order the reranker gave it
            # instead of being treated as maximally distinct — or, as the
            # fallback used to make it, maximally similar (ADR 0001, D8).
            in_space = [i for i, row in enumerate(rows) if row["embedding"] is not None]
            out_of_space = [
                i for i, row in enumerate(rows) if row["embedding"] is None
            ]
            selected = ml.mmr(
                q_emb, [embs[i] for i in in_space], k, lambda_=mmr_lambda
            )
            order = ([in_space[i] for i in selected] + out_of_space)[:k]
            rows = [rows[i] for i in order]
            scores = [scores[i] for i in order]
        else:
            rows = rows[:k]
            scores = scores[:k]

        # Only propositions carry an access count.  A passage has no belief
        # clock and no frequency term in its decay profile — reading the
        # handbook is not evidence the handbook is true (ADR 0001, D6).
        self._record_access(
            row["proposition_id"] for row in rows if row["proposition_id"] is not None
        )
        await self._maybe_flush_access()

        return [
            Result(
                item_id=row["item_id"],
                source=row["source"],
                proposition_id=row["proposition_id"],
                text=row["text"],
                context_text=row["context_text"],
                score=score,
                rrf_score=float(row["rrf_score"]),
                source_kind=row["source_kind"],
                bucket=row["bucket"],
                claim_scope=row["claim_scope"],
                collection_id=row["collection_id"],
                chunk_id=row["chunk_id"],
                subject=row["subject_name"],
                predicate=row["predicate"],
                object=row["object_name"],
                asserted_at=row["asserted_at"],
            )
            for row, score in zip(rows, scores)
        ]

    async def _query_vectors(
        self, query: str
    ) -> tuple[list[float], list[UUID], list[HalfVector]]:
        """The query, embedded once per live generation.

        With one generation this is one ml.embed() call and no registry lookup,
        which is the common case and the one that must not pay for the dual
        window.  During a cutover the caller supplies an embedder per generation
        name; each generation's query prefix travels with the generation because
        two generations in flight may disagree about it.

        Embedding happens before the retrieval connection is taken: it is CPU
        work on a model, and holding a pooled connection across it is how a
        concurrent read path starves the pool.
        """
        if not self._embedders:
            return ml.embed([query])[0], [], []

        async with self._connection() as conn:
            generations = await live_generations(conn, self._scope.org_id)
        primary = next((g for g in generations if g.role == "primary"), None)
        primary_prefix = primary.query_prefix if primary is not None else None
        q_emb = ml.embed([(primary_prefix or "") + query])[0]

        gen_ids: list[UUID] = []
        gen_embs: list[HalfVector] = []
        for generation in generations:
            if generation.role == "primary":
                continue
            embed = self._embedders.get(generation.name)
            if embed is None:
                continue
            prefixed = (generation.query_prefix or "") + query
            gen_ids.append(generation.generation_id)
            gen_embs.append(HalfVector(embed([prefixed])[0]))

        return q_emb, gen_ids, gen_embs

    def _record_access(self, proposition_ids: Iterable[UUID]) -> None:
        """Accumulate access counts in process.  The read path performs no
        writes; the counts reach propositions.access_count on the next flush,
        which is what the frequency term in pgkg_search() reads."""
        self._access.record(self._scope.org_id, proposition_ids)

    async def _maybe_flush_access(self) -> None:
        if not self._access.pending:
            return
        due = monotonic() - self._access.last_flush >= self._access_flush_interval
        if due or len(self._access.pending) >= ACCESS_FLUSH_MAX_PENDING:
            await self.flush_access()

    async def flush_access(self) -> int:
        """Write the accumulated access counts, one statement per org.

        Grouped by org because the write has to run under that org's GUC: the
        RLS policy compares the row's org to the connection's, so a batch
        flushed under the wrong one would silently update nothing.  Returns the
        number of propositions written.  On failure the counts are put back and
        the error is raised rather than swallowed, so nothing is lost silently
        and the caller can retry.

        Each statement autocommits, so only the orgs whose statement did not
        commit are put back: restoring an org whose UPDATE already landed would
        apply its counts a second time on the next flush, and the frequency term
        in the decay profile reads that column.
        """
        pending = self._access.drain()
        if not pending:
            return 0

        by_org: dict[UUID, dict[UUID, int]] = {}
        for (org_id, prop_id), count in pending.items():
            by_org.setdefault(org_id, {})[prop_id] = count

        applied: set[UUID] = set()
        try:
            async with self._pool.acquire() as conn:
                for org_id, counts in by_org.items():
                    ids = list(counts)
                    await conn.execute(_SET_ORG_SQL, str(org_id))
                    await conn.execute(
                        _FLUSH_ACCESS_SQL,
                        ids,
                        [counts[prop_id] for prop_id in ids],
                        org_id,
                    )
                    applied.add(org_id)
        except BaseException:
            self._access.restore(
                {
                    key: count
                    for key, count in pending.items()
                    if key[0] not in applied
                }
            )
            raise
        return len(pending)

    async def aclose(self) -> None:
        await self.flush_access()

    async def forget(
        self,
        proposition_id: UUID,
        *,
        reason: str | None = None,
        supersede_with: UUID | None = None,
    ) -> None:
        """End belief in a proposition, and say why.

        Retrieval reads invalidated_at; superseded_by stays what its name says,
        the pointer to the replacement.  A caller that names a replacement is
        superseding by definition; one that names neither is a deletion.
        """
        resolved = reason or ("superseded" if supersede_with else "user_deleted")
        if resolved not in INVALIDATION_REASONS:
            raise ValueError(
                f"invalidation reason must be one of {sorted(INVALIDATION_REASONS)};"
                f" got {resolved!r}"
            )

        async with self._connection() as conn:
            await conn.execute(
                _FORGET_SQL,
                supersede_with,
                resolved,
                proposition_id,
                self._scope.write_org_id,
            )

    async def believed_at(
        self, belief_at: datetime, *, k: int = 100
    ) -> list[BeliefRecord]:
        """What the system held at an instant, including what it has since
        withdrawn.

        Separate from recall() on purpose: there is no ranking here and no
        vector arm, because the as-of-belief filter cannot use the partial index
        that current-state retrieval depends on.  Making it a parameter of
        recall() would leave the hot path one argument away.
        """
        async with self._connection() as conn:
            rows = await conn.fetch(
                _BELIEVED_AT_SQL,
                belief_at,
                self._namespace,
                k,
                self._scope.read_org_ids,
                self._scope.read_collection_ids,
                self._scope.user_id,
                self._scope.read_acl_groups,
            )

        return [
            BeliefRecord(
                proposition_id=row["proposition_id"],
                text=row["text"],
                recorded_at=row["recorded_at"],
                invalidated_at=row["invalidated_at"],
                invalidation_reason=row["invalidation_reason"],
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
                provenance_id=row["provenance_id"],
            )
            for row in rows
        ]


def _resolve_sources(sources: Iterable[str] | None) -> list[str] | None:
    """NULL for the fused default, a validated list otherwise.

    An unknown source retrieves nothing in SQL, which would reach the caller as
    an empty result rather than as the typo it is.
    """
    if sources is None:
        return None
    wanted = list(dict.fromkeys(sources))
    unknown = [source for source in wanted if source not in SOURCES]
    if unknown or not wanted:
        raise ValueError(
            f"sources must be a non-empty subset of {list(SOURCES)}; got {list(sources)!r}"
        )
    return wanted


def _entity_id(entity_ids: dict[str, UUID], name: str | None) -> UUID | None:
    return entity_ids.get(name) if name is not None else None


def _parse_emb(val: object, fallback: list[float]) -> list[float]:
    """Parse an embedding from asyncpg — may be a list, numpy array, or string."""
    if val is None:
        return fallback
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        return json.loads(val.replace("(", "[").replace(")", "]"))
    # numpy array or pgvector type
    return list(val)
