from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg
from pydantic import BaseModel

from pgkg import ml
from pgkg.ml import ExtractCache, Proposition, PROMPT_VERSION


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


@dataclass
class IngestResult:
    documents: int
    chunks: int
    propositions: int
    entities: int


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text on paragraph boundaries, hard-cap at chunk_size with overlap."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # Para itself may exceed chunk_size — hard-split it
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[max(0, chunk_size - chunk_overlap):]
            current = para

    if current:
        chunks.append(current)

    return chunks or [text[:chunk_size]]


class PostgresExtractCache:
    """Postgres-backed implementation of ExtractCache.

    Stores extracted propositions in the proposition_cache table so re-ingesting
    the same chunk with the same extractor model and prompt version is free.
    """

    def __init__(self, pool: asyncpg.Pool, namespace: str) -> None:
        self._pool = pool
        self._namespace = namespace

    async def get(self, cache_key: str) -> list[Proposition] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT propositions FROM proposition_cache WHERE cache_key = $1",
                cache_key,
            )
            if row is None:
                return None
            # Bump hit count (best-effort; don't fail the main path)
            await conn.execute(
                "UPDATE proposition_cache SET hit_count = hit_count + 1 WHERE cache_key = $1",
                cache_key,
            )
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
    ) -> None:
        self._pool = pool
        self._namespace = namespace
        self._extract_cache: ExtractCache | None = (
            PostgresExtractCache(pool, namespace) if use_extract_cache else None
        )

    async def ingest(
        self,
        text: str,
        *,
        source: str | None = None,
        session_id: str | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ) -> IngestResult:
        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        entities_created: set[UUID] = set()
        total_propositions = 0

        async with self._pool.acquire() as conn:
            # Insert document
            doc_id: UUID = await conn.fetchval(
                "INSERT INTO documents (source, namespace) VALUES ($1, $2) RETURNING id",
                source,
                self._namespace,
            )

            chunk_ids: list[UUID] = []
            for i, chunk_text in enumerate(chunks):
                chunk_id: UUID = await conn.fetchval(
                    """
                    INSERT INTO chunks (document_id, text, span_start, span_end)
                    VALUES ($1, $2, $3, $4) RETURNING id
                    """,
                    doc_id,
                    chunk_text,
                    i * chunk_size,
                    (i + 1) * chunk_size,
                )
                chunk_ids.append(chunk_id)

            # Extract and embed per chunk
            for chunk_id, chunk_text in zip(chunk_ids, chunks):
                propositions = await ml.extract_propositions_async(
                    chunk_text, cache=self._extract_cache
                )
                if not propositions:
                    continue

                # Collect all texts for batch embedding
                entity_names: list[str] = []
                for prop in propositions:
                    entity_names.append(prop.subject)
                    if not prop.object_is_literal:
                        entity_names.append(prop.object)

                prop_texts = [p.text for p in propositions]
                all_texts = entity_names + prop_texts
                all_embs = ml.embed(all_texts)

                entity_embs = all_embs[: len(entity_names)]
                prop_embs = all_embs[len(entity_names):]

                # Link entities and insert propositions
                entity_idx = 0
                for prop, prop_emb in zip(propositions, prop_embs):
                    subj_emb = entity_embs[entity_idx]
                    entity_idx += 1

                    subject_id: UUID = await conn.fetchval(
                        _link_entity_sql(subj_emb),
                        self._namespace,
                        prop.subject,
                        "concept",
                    )
                    entities_created.add(subject_id)

                    object_id: UUID | None = None
                    object_literal: str | None = None

                    if prop.object_is_literal:
                        object_literal = prop.object
                    else:
                        obj_emb = entity_embs[entity_idx]
                        entity_idx += 1
                        object_id = await conn.fetchval(
                            _link_entity_sql(obj_emb),
                            self._namespace,
                            prop.object,
                            "concept",
                        )
                        entities_created.add(object_id)

                    prop_id: UUID = await conn.fetchval(
                        f"""
                        INSERT INTO propositions
                            (text, embedding, subject_id, predicate, object_id,
                             object_literal, chunk_id, namespace, session_id)
                        VALUES ($1, '{_vec_literal(prop_emb)}'::vector,
                                $2, $3, $4, $5, $6, $7, $8)
                        RETURNING id
                        """,
                        prop.text,
                        subject_id,
                        prop.predicate,
                        object_id,
                        object_literal,
                        chunk_id,
                        self._namespace,
                        session_id,
                    )
                    total_propositions += 1

                    if object_id is not None:
                        await conn.execute(
                            """
                            INSERT INTO edges (src_entity, dst_entity, relation, proposition_id)
                            VALUES ($1, $2, $3, $4)
                            ON CONFLICT DO NOTHING
                            """,
                            subject_id,
                            object_id,
                            prop.predicate,
                            prop_id,
                        )

        return IngestResult(
            documents=1,
            chunks=len(chunks),
            propositions=total_propositions,
            entities=len(entities_created),
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
                f"""
                SELECT proposition_id, text, embedding, rrf_score, adjusted_score,
                       source_kind, chunk_id, subject_id, predicate, object_id
                FROM pgkg_search($1, '{_vec_literal(q_emb)}'::vector,
                                 $2, $3, $4, $5, 30.0, $6)
                """,
                query,
                k_retrieve,
                k_retrieve * 2,
                self._namespace,
                session_id,
                expand_graph,
            )

        if not rows:
            return []

        texts = [r["text"] for r in rows]
        scores = [float(r["adjusted_score"]) for r in rows]
        embs = [_parse_emb(r["embedding"], q_emb) for r in rows]

        if with_rerank:
            candidate_rows = rows[: min(k_retrieve, 64)]
            candidate_texts = [r["text"] for r in candidate_rows]
            rerank_scores = ml.rerank(query, candidate_texts)

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
            embs = [list(candidate_rows[i]["embedding"]) if candidate_rows[i]["embedding"] else q_emb
                    for i in sorted_indices]

        if with_mmr and len(rows) > k:
            selected_indices = ml.mmr(q_emb, embs, k, lambda_=mmr_lambda)
            rows = [rows[i] for i in selected_indices]
            scores = [scores[i] for i in selected_indices]
        else:
            rows = rows[:k]
            scores = scores[:k]

        # Fire-and-forget bump access
        prop_ids = [str(r["proposition_id"]) for r in rows]
        asyncio.ensure_future(self._bump(prop_ids))

        results = []
        for row, score in zip(rows, scores):
            subject_name: str | None = None
            object_name: str | None = None

            results.append(
                Result(
                    proposition_id=row["proposition_id"],
                    text=row["text"],
                    score=score,
                    rrf_score=float(row["rrf_score"]),
                    source_kind=row["source_kind"],
                    chunk_id=row["chunk_id"],
                    subject=subject_name,
                    predicate=row["predicate"],
                    object=object_name,
                )
            )

        return results

    async def _bump(self, prop_ids: list[str]) -> None:
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT pgkg_bump_access($1::uuid[])", prop_ids)
        except Exception:
            pass

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


def _parse_emb(val: object, fallback: list[float]) -> list[float]:
    """Parse an embedding from asyncpg — may be a list, numpy array, or string."""
    if val is None:
        return fallback
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        import json
        return json.loads(val.replace("(", "[").replace(")", "]"))
    # numpy array or pgvector type
    return list(val)


def _vec_literal(emb: list[float]) -> str:
    """Format a vector as a Postgres literal string '[a,b,c,...]'."""
    return "[" + ",".join(str(v) for v in emb) + "]"


def _link_entity_sql(emb: list[float]) -> str:
    return f"SELECT pgkg_link_entity($1, $2, $3, '{_vec_literal(emb)}'::vector)"
