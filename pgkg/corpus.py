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
which chunk texts are already stored at the address this document writes at and
already carry a vector of the primary generation; the write phase asks the
database the same question again, authoritatively, through
pgkg_add_version_chunk().  That is one question, so it is asked in one place:
the address is read from the unique index that enforces it, because a copy of it
written out beside the lookup has now drifted twice (#16).  A concurrent crawl
that stores a passage in between can only cost us one redundant embedding of
text that hashes to the row we then reuse — never a wrong vector, because a
vector is only ever written against the content it was computed from.
"""
from __future__ import annotations

import hashlib
import json
import re
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
    """The four collection columns this pipeline obeys."""

    claim_scope: str
    extract_propositions: bool
    public_source: bool
    acl_mode: str


@dataclass(frozen=True)
class _Generation:
    """Which embedding space the vectors written here belong to.

    No width: the column declares that, and a second statement of it here could
    only disagree with the catalog.
    """

    generation_id: UUID


# The content address: the columns that decide whether two writers storing the
# same text share one chunk row.  Stated once, by the unique index — the reuse
# path's ON CONFLICT names that same index, so Postgres itself holds the writer
# to it — and read from the catalogue here rather than restated.
#
# Restating it is what #16 was.  This lookup asked "already stored and
# vectored?" keyed on (org_id, content_hash) while the address moved twice
# underneath it: 042 added collection_id and acl_group_id, 049 added visibility
# and owner_user_id.  A lookup broader than the address answers yes for a row
# that does not exist, the write phase then correctly creates that row, and
# _write_vectors() skips it — leaving a passage with embedding IS NULL that no
# later crawl revisits, because the document hash short-circuits first.
CONTENT_ADDRESS_INDEX = "chunks_content_addressed_key"

@dataclass(frozen=True)
class _AddressKey:
    """One key of the content address, as the index declares it.

    The KEY, not the column: 042 wraps the two nullable axes in COALESCE so that
    "no group" is a value of the axis rather than the absence of one, and the
    index enforces the address over those expressions.  So the expression comes
    with the column, and the comparison is built from it — a predicate over the
    bare column cannot be matched to an expression key however cheap it looks,
    and the address stops being index-usable at the first one that is.

    Nullability is still carried, because it is what decides the comparison for
    a key that is a PLAIN column: a nullable axis holds NULL as one of its
    values and `= NULL` matches nothing.
    """

    column: str
    type_name: str
    nullable: bool
    key: str

    def over(self, alias: str) -> str:
        """The key as it reads on a row of `alias`."""
        return self._substituted(f"{alias}.{self.column}")

    def asked(self, position: int) -> str:
        """The key as it reads on the value being asked about."""
        return self._substituted(f"${position}::{self.type_name}")

    def predicate(self, position: int) -> str:
        return f'{self.over("c")} {self._comparison} {self.asked(position)}'

    @property
    def _comparison(self) -> str:
        # An expression key is compared with `=` because that is what the index
        # can serve, and the expression is what makes it total: COALESCE over a
        # nullable column cannot itself be NULL.  A plain nullable column has no
        # such wrapper, so it needs the null-safe comparison — which no index
        # can serve, which is the honest reason 042 wrapped them.
        if self.key != self.column:
            return "="
        return "IS NOT DISTINCT FROM" if self.nullable else "="

    def _substituted(self, value: str) -> str:
        return re.sub(rf"\b{re.escape(self.column)}\b", value, self.key)


@dataclass(frozen=True)
class _ContentAddress:
    """The address the database enforces: these keys, over these rows."""

    keys: tuple[_AddressKey, ...]
    row_condition: str


@dataclass(frozen=True)
class _ChunkAddress:
    """Where this pipeline stores a passage, in the address's own columns.

    Stated once and used twice — the reuse lookup asks about this address and
    _link_chunks() writes at it — so the question and the write cannot disagree
    about which row the answer was about.  Every field is named after the column
    it fills, because the columns come from the catalogue and are matched to
    values by name.
    """

    org_id: UUID
    collection_id: UUID
    acl_group_id: UUID | None
    visibility: str
    owner_user_id: UUID | None

    def value_for(self, column: str) -> object:
        """This pipeline's value for one column of the address.

        Refuses rather than guesses.  A column in the address that the pipeline
        states no value for means the address has moved again, and the two ways
        to guess are both #16: dropping the column asks a broader question than
        the address answers and strands a chunk with no vector, while assuming a
        value asks about a row that is not the one being written.
        """
        stated = {field.name: getattr(self, field.name) for field in fields(self)}
        if column not in stated:
            raise ValueError(
                f"the chunk content address is keyed on {column!r}, which this"
                " pipeline states no value for: the reuse lookup cannot stand"
                f" in for an address it cannot name (see {CONTENT_ADDRESS_INDEX})"
            )
        return stated[column]


_COLLECTION_POLICY_SQL = """
SELECT claim_scope, extract_propositions, public_source, acl_mode
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

# The address as the catalogue holds it, one row per KEY of the index rather
# than per column.  pg_get_indexdef() with a column number returns exactly what
# the index is keyed on at that position — the bare column where it is one, the
# expression where 042 wrapped it — which is the only place the two forms are
# distinguishable.  The partial predicate travels with them: which rows are
# content-addressed is as much part of the address as which keys key it.
_ADDRESS_KEYS_SQL = """
SELECT n AS position,
       pg_get_indexdef(i.indexrelid, n::int, true) AS key,
       pg_get_expr(i.indpred, i.indrelid) AS row_condition
FROM pg_index i
JOIN pg_class ic ON ic.oid = i.indexrelid
CROSS JOIN generate_series(1, i.indnkeyatts) AS n
WHERE ic.relname = $1
ORDER BY n
"""

_CHUNK_COLUMNS_SQL = """
SELECT a.attname AS column_name,
       format_type(a.atttypid, a.atttypmod) AS column_type,
       NOT a.attnotnull AS nullable
FROM pg_attribute a
WHERE a.attrelid = 'chunks'::regclass AND a.attnum > 0 AND NOT a.attisdropped
"""

_HASH_COLUMN = "content_hash"

# Every corpus passage is shared with the org and owned by nobody: a nightly
# crawl has no owning user, and D3's private lane belongs to chat ingest (049).
# Stated here rather than left to pgkg_add_version_chunk()'s defaults, because
# 049 put both columns in the address — the value the reuse lookup asks about
# has to be the value the write uses, and a default is a second place for that
# to be decided.
CORPUS_VISIBILITY = "shared"
CORPUS_OWNER_USER_ID: UUID | None = None


async def content_address(conn: asyncpg.Connection) -> _ContentAddress:
    """The content address, read from the unique index that enforces it."""
    rows = await conn.fetch(_ADDRESS_KEYS_SQL, CONTENT_ADDRESS_INDEX)
    if not rows:
        raise ValueError(
            f"chunks carries no index {CONTENT_ADDRESS_INDEX}: the reuse lookup"
            " derives the content address from it and will not guess it"
        )
    columns = {
        row["column_name"]: row
        for row in await conn.fetch(_CHUNK_COLUMNS_SQL)
    }
    return _ContentAddress(
        keys=tuple(_address_key(row["key"], columns) for row in rows),
        # A total address is a WHERE TRUE.  052 keeps it partial on
        # `NOT provenance_only`: content addressing governs retrievable content,
        # and a passage stored as provenance for the facts extracted from it
        # must never be reused by another writer.  Which rows the address covers
        # is read from the index for the same reason its keys are — this
        # pipeline writes retrievable content, so it satisfies the predicate,
        # but it does not get to decide what the predicate is.
        row_condition=rows[0]["row_condition"] or "TRUE",
    )


def _address_key(key: str, columns: dict[str, asyncpg.Record]) -> _AddressKey:
    """One key of the index, matched to the column whose value fills it.

    A key is only useful to this lookup if the pipeline can put its own value
    into it, which means knowing which column it reads.  Word-bounded, so
    `acl_group_id` is not found inside a hypothetical `group_id`, and exactly
    one match is required: a key over two columns has no single value to bind
    and is refused rather than guessed, for the same reason value_for() refuses.
    """
    named = [
        column
        for column in columns
        if re.search(rf"\b{re.escape(column)}\b", key)
    ]
    if len(named) != 1:
        raise ValueError(
            f"index {CONTENT_ADDRESS_INDEX} is keyed on {key!r}, which names"
            f" {sorted(named)} of chunks: the reuse lookup can only stand in"
            " for a key it can put one value into"
        )
    column = columns[named[0]]
    return _AddressKey(
        column=named[0],
        type_name=column["column_type"],
        nullable=column["nullable"],
        key=key,
    )


def reuse_lookup(
    address: _ContentAddress,
) -> tuple[str, Callable[[_ChunkAddress], tuple[object, ...]]]:
    """What is already stored at an address, and which of it carries a vector.

    Generated from the address rather than written out beside it, because a
    written-out copy is exactly what drifted twice (#16).  $1 is the generation a
    vector has to belong to to be reusable, which is not part of the address —
    where a passage lives and which embedding space its vector belongs to are
    different questions — and $2 is the hashes being asked about; the rest of the
    address follows in the index's own key order.

    Returned with the binder that fills it, so the statement and its arguments
    are built from one reading of the catalogue and cannot be paired wrongly.

    WHY IT IS ONE PROBE PER HASH RATHER THAN ONE PASS WITH `= ANY`.  Every
    predicate is written in the index's own terms, which is what makes the
    address usable as an index condition instead of a prefix of it — a bare
    `owner_user_id IS NOT DISTINCT FROM $n` cannot be matched to the COALESCE key
    042 declared, and the address stops being usable at the first such column,
    which is the third of six.  Getting the leading five is not enough: the hash
    is the last key and the only selective one, and a btree will not take a
    trailing `= ANY(...)` as an index condition, so it is left to a filter that
    reads every passage of the collection to answer.  Driving the statement from
    the hashes turns that into an equality the index can serve, and the LATERAL
    is what stops the planner flattening it back — it estimates one row for the
    prefix, so left to itself it puts the whole collection on the outer side.
    The LIMIT is free: an address is unique, which is what makes it an address.

    Measured on 30,000 passages in one collection (issue #18 measured the first
    half of this and left it): 2096 buffers as a sequential scan, 30,698 as the
    index prefix with the hash filtered, 6 as one probe per hash.  This runs
    once per changed document of a nightly crawl, so on a 600k-passage store the
    difference is the cost of the crawl.
    """
    hashed = [key for key in address.keys if key.column == _HASH_COLUMN]
    if not hashed:
        raise ValueError(
            f"index {CONTENT_ADDRESS_INDEX} is not keyed on {_HASH_COLUMN}, so it"
            " is not a content address this lookup can stand in for"
        )
    hash_key = hashed[0]
    scoped = tuple(key for key in address.keys if key.column != _HASH_COLUMN)
    predicates = "".join(
        f"\n          AND {key.predicate(position)}"
        for position, key in enumerate(scoped, start=3)
    )
    sql = (
        f"SELECT h.{_HASH_COLUMN}, found.has_vector\n"
        f"FROM unnest($2::{hash_key.type_name}[]) AS h({_HASH_COLUMN})\n"
        "JOIN LATERAL (\n"
        "    SELECT (c.embedding IS NOT NULL"
        " AND c.embedder_generation_id = $1) AS has_vector\n"
        "    FROM chunks c\n"
        f"    WHERE ({address.row_condition})\n"
        # `=` and not the null-safe comparison whatever the catalogue says the
        # column allows: unnest supplies a hash for every element, so there is
        # no NULL to be careful of, and `=` is what the index can serve.
        f'          AND {hash_key.over("c")} = {hash_key.over("h")}'
        f"{predicates}\n"
        "    LIMIT 1\n"
        ") AS found ON TRUE\n"
    )

    def binder(stored: _ChunkAddress) -> tuple[object, ...]:
        return tuple(stored.value_for(key.column) for key in scoped)

    return sql, binder


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
#
# Visibility and owner are passed rather than defaulted: they joined the content
# address in 049, and the address this writes at is the address the reuse lookup
# asked about, from the same _ChunkAddress.
_ADD_CHUNKS_SQL = """
SELECT t.n, added.chunk_id, added.is_new
FROM unnest($2::int[], $3::text[]) WITH ORDINALITY AS t(ord, chunk_text, n)
CROSS JOIN LATERAL
    pgkg_add_version_chunk($1, t.ord, t.chunk_text, $4, $5, $6::text, $7::uuid)
    AS added
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
# a passage to the entities it names is gazetteer work and belongs on a timer —
# `pgkg maintain --task mentions`, not this pipeline.  A cross-product against
# every name the org knows, on every changed document of a nightly crawl, is
# what D7 separates the two ingests to avoid, and the sweep's watermarks are
# what make the edge exist whether or not this path ever calls the matcher
# (pgkg/maintenance.py; implementation notes §5).
# The object of a claim therefore survives as text rather than as an entity, and
# is not dropped on the floor while waiting for that.
_INSERT_PROPOSITIONS_SQL = """
WITH inserted AS (
    INSERT INTO propositions
        (text, embedding, predicate, object_literal, chunk_id, org_id,
         collection_id, claim_scope, provenance_id, asserted_at,
         embedder_generation_id, acl_group_id)
    SELECT p.text, p.embedding, p.predicate, p.object_literal, p.chunk_id,
           $1, $2, $3, $4, $5, $6, $12
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
        # Read once per pipeline: the catalogue cannot move under a running
        # process, and a crawl of 100k documents should not ask 100k times.
        self._address: _ContentAddress | None = None

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
        acl_group_id: UUID | None = None,
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
            self._require_acl_group(policy, acl_group_id)
            generation = await self._generation(conn)
            address = self._chunk_address(acl_group_id)
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
            stored, vectored = await self._known_content(
                conn, generation, texts, address
            )
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
                linked = await self._link_chunks(
                    conn, version_id, chunks, asserted, address
                )
                new_chunks = list(
                    {
                        chunk_id: chunk_text
                        for chunk_id, chunk_text, is_new in linked
                        if is_new
                    }.items()
                )
                distinct = len({chunk_id for chunk_id, _, _ in linked})
                stranded = await self._write_vectors(
                    conn, generation, new_chunks, vectors
                )
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
                    acl_group_id,
                )
                if extract_cache is not None:
                    await extract_cache.flush(conn)
                await conn.execute(_PROMOTE_SQL, version_id)

        # Phase 4, and only ever for the rows phase 3 disagreed with phase 2
        # about.  Outside the transaction because it is a model call, and after
        # the promotion because a vector is content-addressed: the row is
        # retrievable by the keyword arm the moment the version flips, and this
        # is what stops it being retrievable ONLY by the keyword arm, for good.
        repaired = await self._vector_the_stranded(generation, stranded)

        return CorpusIngestResult(
            document_id=document_id,
            version_id=version_id,
            changed=True,
            chunks_total=len(linked),
            chunks_new=len(new_chunks),
            chunks_carried=distinct - len(new_chunks),
            embedded=len(fresh) + len(repaired),
            cache_hits=len(cached),
            propositions=propositions,
        )

    def _require_acl_group(
        self, policy: _CollectionPolicy, acl_group_id: UUID | None
    ) -> None:
        """Refuse untagged content offered to an ACL-bounded collection (D3).

        Asked before anything is spent, and before the unchanged-hash
        short-circuit: a connector pointed at an ACL-bounded collection with no
        group is misconfigured whether or not tonight's crawl moved.  The
        database refuses the same write — this is the message that says why, and
        the one that arrives before the embedder bill.
        """
        if policy.acl_mode == "none" or acl_group_id is not None:
            return
        raise ValueError(
            f"collection {self._collection_id} has acl_mode"
            f" {policy.acl_mode!r}, so an ingest into it must name an"
            " acl_group_id: an untagged row passes the retrieval predicate for"
            " every caller"
        )

    def _chunk_address(self, acl_group_id: UUID | None) -> _ChunkAddress:
        """The address every passage of this document is stored at."""
        return _ChunkAddress(
            org_id=self._org_id,
            collection_id=self._collection_id,
            acl_group_id=acl_group_id,
            visibility=CORPUS_VISIBILITY,
            owner_user_id=CORPUS_OWNER_USER_ID,
        )

    async def _known_content(
        self,
        conn: asyncpg.Connection,
        generation: _Generation,
        texts: Sequence[str],
        address: _ChunkAddress,
    ) -> tuple[set[str], set[str]]:
        """Which of these texts are stored already, and which carry a vector.

        Two answers from one round trip, because they drive two different
        decisions: a stored text is one this version will carry without
        extracting it again, and a vectored one is a text nobody has to embed.

        Asked at the address the write phase will write at, and at no other.
        Answering for the org as a whole said "stored and vectored" about a row
        in another collection, which the write phase then rightly declined to
        reuse — and the chunk it created instead was never given a vector (#16).
        The reuse this pipeline is built on is unaffected: the same passage in
        the same collection is the same address, which is where the saving was.
        What the narrowing does cost is one extraction of a passage per address
        it is stored at, and that is the right price: propositions hang off a
        chunk id, so the second collection's row carries no facts until they are
        extracted against it.
        """
        by_hash = {content_hash(chunk_text): chunk_text for chunk_text in texts}
        sql, binder = reuse_lookup(await self._content_address(conn))
        rows = await conn.fetch(
            sql, generation.generation_id, list(by_hash), *binder(address)
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
            acl_mode=row["acl_mode"],
        )

    async def _content_address(self, conn: asyncpg.Connection) -> _ContentAddress:
        """The address, read once: the catalogue cannot move under a live
        process, and a crawl of 100k documents should not ask 100k times."""
        if self._address is None:
            self._address = await content_address(conn)
        return self._address

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
        address: _ChunkAddress,
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
            address.acl_group_id,
            asserted_at,
            address.visibility,
            address.owner_user_id,
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
    ) -> list[tuple[UUID, str]]:
        """Give a vector to every chunk the database created, and to no other.

        A carried-over chunk already holds the vector for its content, and its
        content cannot have changed — that is what its address means.

        Returns the chunks the database created that there is no vector in hand
        for, which is the only case this cannot settle: phase 2 answered that
        the content was already vectored, and by the time phase 3 asked the
        database the same question authoritatively the row it had seen was gone.
        They are named rather than embedded here because embedding here would be
        a model call inside this transaction, and returned rather than dropped
        because "left for the next crawl" is not true of them — phase 1
        short-circuits on the unchanged document hash, so there is no next crawl
        for this document and the row would keep `embedding IS NULL` for good.
        """
        vectorised = [
            (chunk_id, vectors[chunk_text])
            for chunk_id, chunk_text in new_chunks
            if chunk_text in vectors
        ]
        stranded = [
            (chunk_id, chunk_text)
            for chunk_id, chunk_text in new_chunks
            if chunk_text not in vectors
        ]
        if not vectorised:
            return stranded
        await conn.execute(
            _WRITE_CHUNK_EMBEDDINGS_SQL,
            generation.generation_id,
            [chunk_id for chunk_id, _ in vectorised],
            [vector for _, vector in vectorised],
        )
        return stranded

    async def _vector_the_stranded(
        self, generation: _Generation, stranded: list[tuple[UUID, str]]
    ) -> list[tuple[UUID, str]]:
        """Pay for the vectors the write phase turned out to need.

        Empty on every ordinary ingest, which is the point: the reuse the whole
        pipeline is built on is unaffected and this costs one comparison per new
        chunk.  It is not a substitute for the reuse lookup being exactly the
        address — a lookup broader than the address would land every second
        collection's passage here and pay the embedder for all of it — it is
        what makes the remaining window cost money instead of costing a vector.

        Its own connection, taken after the embedder has returned: D7's rule is
        that no pooled connection is held across a model call, and a repair that
        broke it would be a worse defect than the one it fixes.
        """
        if not stranded:
            return []
        vectors = self._embed_texts([chunk_text for _, chunk_text in stranded])
        async with self._pool.acquire() as conn:
            await conn.execute(_SET_ORG_SQL, str(self._org_id))
            await conn.execute(
                _WRITE_CHUNK_EMBEDDINGS_SQL,
                generation.generation_id,
                [chunk_id for chunk_id, _ in stranded],
                [HalfVector(vector) for vector in vectors],
            )
        return stranded

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
        acl_group_id: UUID | None,
    ) -> int:
        """Land the extracted facts, each citing the passage it came from.

        A fact carries the ACL group of the document it was extracted from: a
        proposition is the passage restated, so one that lost the group would
        launder the document through the other arm of the same retriever.
        """
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
            acl_group_id,
        )
        return len(rows)
