from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from uuid import UUID

import asyncpg
from pgvector import HalfVector
from pydantic import BaseModel

from pgkg import ml
from pgkg.chunking import DEFAULT_MAX_CHARS, chunk_document
from pgkg.ml import ExtractCache, Proposition


# How long a Memory may hold unwritten access counts before the next recall
# flushes them, and how many propositions it may hold before flushing early.
# The interval trades staleness of the frequency term for writes: at 100
# recalls/sec of 20 rows, five seconds turns 10,000 row updates into at most a
# few hundred, one statement.
ACCESS_FLUSH_INTERVAL_SECONDS = 5.0
ACCESS_FLUSH_MAX_PENDING = 512


class Result(BaseModel):
    proposition_id: UUID
    text: str
    score: float
    rrf_score: float
    source_kind: str
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


@dataclass(frozen=True)
class _PendingChunk:
    id: UUID
    text: str
    span_start: int
    span_end: int


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
INSERT INTO documents (id, source, namespace) VALUES ($1, $2, $3)
"""

_INSERT_CHUNKS_SQL = """
INSERT INTO chunks (id, document_id, text, span_start, span_end, asserted_at)
SELECT c.id, $1, c.text, c.span_start, c.span_end, $2
FROM unnest($3::uuid[], $4::text[], $5::int[], $6::int[])
     AS c(id, text, span_start, span_end)
"""

# One statement, one plpgsql call per distinct name.  Resolution stays
# sequential — pgkg_link_entity is VOLATILE, so each invocation sees the
# entities the earlier ones inserted, which is what lets two spellings of the
# same name inside a single batch collapse onto one row.
_LINK_ENTITIES_SQL = """
SELECT n.name, pgkg_link_entity($1, n.name, 'concept', n.embedding) AS entity_id
FROM unnest($2::text[], $3::halfvec[]) AS n(name, embedding)
"""

_INSERT_PROPOSITIONS_SQL = """
INSERT INTO propositions
    (id, text, embedding, subject_id, predicate, object_id, object_literal,
     chunk_id, namespace, session_id, metadata, asserted_at)
SELECT p.id, p.text, p.embedding, p.subject_id, p.predicate, p.object_id,
       p.object_literal, p.chunk_id, $1, $2,
       COALESCE(p.metadata::jsonb, '{}'::jsonb), $3
FROM unnest($4::uuid[], $5::text[], $6::halfvec[], $7::uuid[], $8::text[],
            $9::uuid[], $10::text[], $11::uuid[], $12::text[])
     AS p(id, text, embedding, subject_id, predicate, object_id,
          object_literal, chunk_id, metadata)
"""

_INSERT_EDGES_SQL = """
INSERT INTO edges (src_entity, dst_entity, relation, proposition_id)
SELECT e.src, e.dst, e.relation, e.proposition_id
FROM unnest($1::uuid[], $2::uuid[], $3::text[], $4::uuid[])
     AS e(src, dst, relation, proposition_id)
ON CONFLICT DO NOTHING
"""

# The embedding leaves as `vector` even though the column is halfvec: widening
# is exact and the client codec for it is the one _parse_emb understands.
_RECALL_SQL = """
SELECT s.proposition_id, s.text, s.embedding, s.rrf_score, s.adjusted_score,
       s.source_kind, s.chunk_id, s.predicate, s.asserted_at,
       subj.name AS subject_name,
       COALESCE(obj.name, p.object_literal) AS object_name
FROM pgkg_search($1, $2::halfvec, $3, $4, $5, $6, 30.0, $7) s
LEFT JOIN propositions p ON p.id = s.proposition_id
LEFT JOIN entities subj ON subj.id = s.subject_id
LEFT JOIN entities obj ON obj.id = s.object_id
ORDER BY s.adjusted_score DESC
"""

_FLUSH_ACCESS_SQL = """
UPDATE propositions p
SET access_count = p.access_count + a.n,
    last_accessed_at = now()
FROM unnest($1::uuid[], $2::int[]) AS a(id, n)
WHERE p.id = a.id
"""


class PostgresExtractCache:
    """Postgres-backed implementation of ExtractCache.

    Stores extracted propositions in the proposition_cache table so re-ingesting
    the same chunk with the same extractor model and prompt version is free.
    """

    def __init__(self, pool: asyncpg.Pool, namespace: str) -> None:
        self._pool = pool
        self._namespace = namespace

    async def get(self, cache_key: str) -> list[Proposition] | None:
        # The hit count is bumped by the same statement that reads the payload:
        # a lookup on the ingest path should cost one round trip, not two.
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE proposition_cache
                SET hit_count = hit_count + 1
                WHERE cache_key = $1
                RETURNING propositions
                """,
                cache_key,
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
                    (cache_key, chunk_hash, extractor_model, prompt_version, propositions)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (cache_key) DO NOTHING
                """,
                cache_key,
                chunk_hash,
                extractor_model,
                prompt_version,
                payload,
            )


class Memory:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        namespace: str = "default",
        use_extract_cache: bool = True,
        extract_propositions: bool = True,
        access_flush_interval: float = ACCESS_FLUSH_INTERVAL_SECONDS,
    ) -> None:
        self._pool = pool
        self._namespace = namespace
        self._extract_propositions = extract_propositions
        self._extract_cache: ExtractCache | None = (
            PostgresExtractCache(pool, namespace) if use_extract_cache and extract_propositions else None
        )
        self._access_flush_interval = access_flush_interval
        self._access_pending: dict[UUID, int] = {}
        self._access_last_flush = monotonic()

    async def ingest(
        self,
        text: str,
        *,
        source: str | None = None,
        session_id: str | None = None,
        asserted_at: datetime | None = None,
        chunk_size: int = DEFAULT_MAX_CHARS,
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
            )
            for chunk in chunk_document(text, max_chars=chunk_size)
        )

        plan = (
            await self._plan_extracted(chunks)
            if self._extract_propositions
            else self._plan_chunks_only(chunks)
        )
        return await self._write(plan, source, session_id, asserted_at)

    async def _write(
        self,
        plan: _IngestPlan,
        source: str | None,
        session_id: str | None,
        asserted_at: datetime | None,
    ) -> IngestResult:
        doc_id = uuid.uuid4()
        async with self._pool.acquire() as conn:
            # One transaction: a document, its chunks and its propositions are
            # either all visible or none are.  Phase 2's version flip depends on
            # this boundary existing.
            async with conn.transaction():
                await conn.execute(_INSERT_DOCUMENT_SQL, doc_id, source, self._namespace)
                await self._write_chunks(conn, doc_id, plan.chunks, asserted_at)
                entity_ids = await self._link_entities(conn, plan.entities)
                await self._write_propositions(
                    conn, plan.propositions, entity_ids, session_id, asserted_at
                )
                await self._write_edges(conn, plan.propositions, entity_ids)

        return IngestResult(
            documents=1,
            chunks=len(plan.chunks),
            propositions=len(plan.propositions),
            entities=len(set(entity_ids.values())),
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
        """Chunks-only mode: embed each chunk directly, no LLM extraction, no
        entity linking, no edge creation."""
        embeddings = ml.embed([chunk.text for chunk in chunks])
        return _IngestPlan(
            chunks=chunks,
            entities=(),
            propositions=tuple(
                _PendingProposition(
                    id=uuid.uuid4(),
                    text=chunk.text,
                    embedding=HalfVector(embedding),
                    chunk_id=chunk.id,
                    predicate=None,
                    subject_name=None,
                    object_name=None,
                    object_literal=None,
                    metadata='{"mode": "chunk"}',
                )
                for chunk, embedding in zip(chunks, embeddings)
            ),
        )

    async def _write_chunks(
        self,
        conn: asyncpg.Connection,
        doc_id: UUID,
        chunks: tuple[_PendingChunk, ...],
        asserted_at: datetime | None,
    ) -> None:
        if not chunks:
            return
        await conn.execute(
            _INSERT_CHUNKS_SQL,
            doc_id,
            asserted_at,
            [c.id for c in chunks],
            [c.text for c in chunks],
            [c.span_start for c in chunks],
            [c.span_end for c in chunks],
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
    ) -> None:
        if not propositions:
            return
        await conn.execute(
            _INSERT_PROPOSITIONS_SQL,
            self._namespace,
            session_id,
            asserted_at,
            [p.id for p in propositions],
            [p.text for p in propositions],
            [p.embedding for p in propositions],
            [_entity_id(entity_ids, p.subject_name) for p in propositions],
            [p.predicate for p in propositions],
            [_entity_id(entity_ids, p.object_name) for p in propositions],
            [p.object_literal for p in propositions],
            [p.chunk_id for p in propositions],
            [p.metadata for p in propositions],
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
    ) -> list[Result]:
        q_emb = ml.embed([query])[0]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _RECALL_SQL,
                query,
                HalfVector(q_emb),
                k_retrieve,
                k_retrieve * 2,
                self._namespace,
                session_id,
                expand_graph,
            )

        if not rows:
            return []

        scores = [float(r["adjusted_score"]) for r in rows]
        embs = [_parse_emb(r["embedding"], q_emb) for r in rows]

        if with_rerank:
            cutoff = min(k_retrieve, 64)
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
            selected_indices = ml.mmr(q_emb, embs, k, lambda_=mmr_lambda)
            rows = [rows[i] for i in selected_indices]
            scores = [scores[i] for i in selected_indices]
        else:
            rows = rows[:k]
            scores = scores[:k]

        self._record_access(row["proposition_id"] for row in rows)
        await self._maybe_flush_access()

        return [
            Result(
                proposition_id=row["proposition_id"],
                text=row["text"],
                score=score,
                rrf_score=float(row["rrf_score"]),
                source_kind=row["source_kind"],
                chunk_id=row["chunk_id"],
                subject=row["subject_name"],
                predicate=row["predicate"],
                object=row["object_name"],
                asserted_at=row["asserted_at"],
            )
            for row, score in zip(rows, scores)
        ]

    def _record_access(self, proposition_ids: Iterable[UUID]) -> None:
        """Accumulate access counts in process.  The read path performs no
        writes; the counts reach propositions.access_count on the next flush,
        which is what the frequency term in pgkg_search() reads."""
        for prop_id in proposition_ids:
            self._access_pending[prop_id] = self._access_pending.get(prop_id, 0) + 1

    async def _maybe_flush_access(self) -> None:
        if not self._access_pending:
            return
        due = monotonic() - self._access_last_flush >= self._access_flush_interval
        if due or len(self._access_pending) >= ACCESS_FLUSH_MAX_PENDING:
            await self.flush_access()

    async def flush_access(self) -> int:
        """Write the accumulated access counts as a single statement.

        Returns the number of propositions written.  On failure the counts are
        put back and the error is raised rather than swallowed, so nothing is
        lost silently and the caller can retry.
        """
        pending, self._access_pending = self._access_pending, {}
        self._access_last_flush = monotonic()
        if not pending:
            return 0

        ids = list(pending)
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    _FLUSH_ACCESS_SQL, ids, [pending[prop_id] for prop_id in ids]
                )
        except BaseException:
            for prop_id, count in pending.items():
                self._access_pending[prop_id] = (
                    self._access_pending.get(prop_id, 0) + count
                )
            raise
        return len(ids)

    async def aclose(self) -> None:
        await self.flush_access()

    async def forget(
        self,
        proposition_id: UUID,
        *,
        supersede_with: UUID | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            if supersede_with is not None:
                await conn.execute(
                    "UPDATE propositions SET superseded_by = $1 WHERE id = $2",
                    supersede_with,
                    proposition_id,
                )
            else:
                await conn.execute(
                    "UPDATE propositions SET superseded_by = id WHERE id = $1",
                    proposition_id,
                )


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
