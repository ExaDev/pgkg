from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from pgkg import ml
from pgkg.backend import (
    StorageBackend,
    StoredChunk,
    StoredDocument,
    StoredProposition,
)
from pgkg.ml import ExtractCache, Proposition


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


class BackendExtractCache:
    """Adapts StorageBackend cache methods to the ExtractCache protocol."""

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend

    async def get(self, cache_key: str) -> list[Proposition] | None:
        raw = await self._backend.cache_get(cache_key)
        if raw is None:
            return None
        return [Proposition(**d) for d in raw]

    async def put(
        self,
        cache_key: str,
        chunk_hash: str,
        extractor_model: str,
        prompt_version: str,
        props: list[Proposition],
    ) -> None:
        await self._backend.cache_put(
            cache_key,
            chunk_hash,
            extractor_model,
            prompt_version,
            [p.model_dump() for p in props],
        )


class Memory:
    def __init__(
        self,
        backend: StorageBackend,
        *,
        namespace: str = "default",
        use_extract_cache: bool = True,
        extract_propositions: bool = True,
    ) -> None:
        self._backend = backend
        self._namespace = namespace
        self._extract_propositions = extract_propositions
        self._extract_cache: ExtractCache | None = (
            BackendExtractCache(backend) if use_extract_cache and extract_propositions else None
        )

    async def ingest(
        self,
        text: str,
        *,
        source: str | None = None,
        session_id: str | None = None,
        asserted_at: datetime | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ) -> IngestResult:
        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        entities_created: set[UUID] = set()
        total_propositions = 0

        doc_id = await self._backend.store_document(
            StoredDocument(source=source, namespace=self._namespace)
        )

        chunk_ids: list[UUID] = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = await self._backend.store_chunk(
                StoredChunk(
                    document_id=doc_id,
                    text=chunk_text,
                    span_start=i * chunk_size,
                    span_end=(i + 1) * chunk_size,
                    asserted_at=asserted_at,
                )
            )
            chunk_ids.append(chunk_id)

        if self._extract_propositions:
            for chunk_id, chunk_text in zip(chunk_ids, chunks):
                propositions = await ml.extract_propositions_async(
                    chunk_text, cache=self._extract_cache
                )
                if not propositions:
                    continue

                entity_names: list[str] = []
                for prop in propositions:
                    entity_names.append(prop.subject)
                    if not prop.object_is_literal:
                        entity_names.append(prop.object)

                prop_texts = [p.text for p in propositions]
                all_texts = entity_names + prop_texts
                all_embs = ml.embed(all_texts)

                entity_embs = all_embs[: len(entity_names)]
                prop_embs = all_embs[len(entity_names) :]

                entity_idx = 0
                for prop, prop_emb in zip(propositions, prop_embs):
                    subj_emb = entity_embs[entity_idx]
                    entity_idx += 1

                    subject_id = await self._backend.link_entity(
                        self._namespace, prop.subject, "concept", subj_emb
                    )
                    entities_created.add(subject_id)

                    object_id: UUID | None = None
                    object_literal: str | None = None

                    if prop.object_is_literal:
                        object_literal = prop.object
                    else:
                        obj_emb = entity_embs[entity_idx]
                        entity_idx += 1
                        object_id = await self._backend.link_entity(
                            self._namespace, prop.object, "concept", obj_emb
                        )
                        entities_created.add(object_id)

                    prop_id = await self._backend.store_proposition(
                        StoredProposition(
                            text=prop.text,
                            embedding=prop_emb,
                            subject_id=subject_id,
                            predicate=prop.predicate,
                            object_id=object_id,
                            object_literal=object_literal,
                            chunk_id=chunk_id,
                            namespace=self._namespace,
                            session_id=session_id,
                            asserted_at=asserted_at,
                        )
                    )
                    total_propositions += 1

                    if object_id is not None:
                        await self._backend.store_edge(
                            subject_id, object_id, prop.predicate, prop_id
                        )
        else:
            chunk_texts_list = list(chunks)
            chunk_embs = ml.embed(chunk_texts_list)
            for chunk_id, chunk_text_val, chunk_emb in zip(
                chunk_ids, chunk_texts_list, chunk_embs
            ):
                await self._backend.store_proposition(
                    StoredProposition(
                        text=chunk_text_val,
                        embedding=chunk_emb,
                        subject_id=None,
                        predicate=None,
                        object_id=None,
                        object_literal=None,
                        chunk_id=chunk_id,
                        namespace=self._namespace,
                        session_id=session_id,
                        metadata={"mode": "chunk"},
                        asserted_at=asserted_at,
                    )
                )
                total_propositions += 1

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

        candidates = await self._backend.fused_search(
            query_text=query,
            query_embedding=q_emb,
            k_retrieve=k_retrieve,
            k_initial=k_retrieve * 2,
            namespace=self._namespace,
            session_id=session_id,
            recency_half_life_days=30.0,
            expand_graph=expand_graph,
            rrf_k=60,
        )

        if not candidates:
            return []

        # Fetch full proposition data for embeddings (needed by rerank/MMR)
        prop_rows = await self._backend.get_propositions(
            [c.proposition_id for c in candidates]
        )
        emb_by_id = {p.id: p.embedding for p in prop_rows}

        texts = [c.text for c in candidates]
        scores = [c.adjusted_score for c in candidates]
        embs = [emb_by_id.get(c.proposition_id) or q_emb for c in candidates]
        embs = [list(e) if not isinstance(e, list) else e for e in embs]

        if with_rerank:
            candidate_slice = candidates[: min(k_retrieve, 64)]
            candidate_texts = [c.text for c in candidate_slice]
            rerank_scores = ml.rerank(query, candidate_texts)

            def _normalize(vals: list[float]) -> list[float]:
                lo, hi = min(vals), max(vals)
                span = hi - lo
                if span < 1e-10:
                    return [1.0] * len(vals)
                return [(v - lo) / span for v in vals]

            adj_scores = [c.adjusted_score for c in candidate_slice]
            rerank_norm = _normalize(rerank_scores)
            adj_norm = _normalize(adj_scores)
            blended = [0.7 * r + 0.3 * a for r, a in zip(rerank_norm, adj_norm)]

            sorted_indices = sorted(
                range(len(blended)), key=lambda i: blended[i], reverse=True
            )
            candidates = [candidate_slice[i] for i in sorted_indices]
            scores = [blended[i] for i in sorted_indices]
            embs = [
                emb_by_id.get(candidate_slice[i].proposition_id) or q_emb
                for i in sorted_indices
            ]
            embs = [list(e) if not isinstance(e, list) else e for e in embs]

        if with_mmr and len(candidates) > k:
            selected_indices = ml.mmr(q_emb, embs, k, lambda_=mmr_lambda)
            candidates = [candidates[i] for i in selected_indices]
            scores = [scores[i] for i in selected_indices]
        else:
            candidates = candidates[:k]
            scores = scores[:k]

        # Fire-and-forget bump access
        prop_ids = [c.proposition_id for c in candidates]
        asyncio.ensure_future(self._bump(prop_ids))

        results = []
        for cand, score in zip(candidates, scores):
            results.append(
                Result(
                    proposition_id=cand.proposition_id,
                    text=cand.text,
                    score=score,
                    rrf_score=cand.rrf_score,
                    source_kind=cand.source_kind,
                    chunk_id=cand.chunk_id,
                    subject=None,
                    predicate=cand.predicate,
                    object=None,
                    asserted_at=cand.asserted_at,
                )
            )

        return results

    async def _bump(self, prop_ids: list[UUID]) -> None:
        try:
            await self._backend.bump_access(prop_ids)
        except Exception:
            pass

    async def forget(
        self,
        proposition_id: UUID,
        *,
        supersede_with: UUID | None = None,
    ) -> None:
        await self._backend.forget(proposition_id, supersede_with=supersede_with)
