"""PostgresBackend — wraps asyncpg pool and Postgres SQL functions."""
from __future__ import annotations

import json
import pathlib
from uuid import UUID

import asyncpg
from pgvector.asyncpg import register_vector

from pgkg.backend import (
    Candidate,
    PropositionRow,
    ScoredId,
    StoredChunk,
    StoredDocument,
    StoredProposition,
)

MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "migrations"


def _vec_literal(emb: list[float]) -> str:
    """Format a vector as a Postgres literal string ``[a,b,c,...]``."""
    return "[" + ",".join(str(v) for v in emb) + "]"


class PostgresBackend:
    """StorageBackend backed by asyncpg + pgvector.

    Delegates search to the ``pgkg_search()`` CTE, entity linking to
    ``pgkg_link_entity()``, and access bumping to ``pgkg_bump_access()``.
    """

    def __init__(self, pool: asyncpg.Pool, *, dsn: str | None = None) -> None:
        self._pool = pool
        self._dsn = dsn

    @classmethod
    async def create(cls, dsn: str | None = None) -> PostgresBackend:
        """Create a backend with a fresh connection pool.

        When *dsn* is ``None``, an embedded Postgres is started
        automatically via pgserver (requires ``uv sync --extra embedded``).
        """
        if dsn is None:
            from pgkg.embedded import get_dsn
            dsn = get_dsn()
        pool = await asyncpg.create_pool(
            dsn, min_size=1, max_size=10, init=_init_connection,
        )
        return cls(pool, dsn=dsn)  # type: ignore[arg-type]

    @property
    def pool(self) -> asyncpg.Pool:
        """Expose the pool for health checks and legacy callers."""
        return self._pool

    # --- Lifecycle -------------------------------------------------------

    async def apply_migrations(self) -> None:
        async with self._pool.acquire() as conn:
            await self._apply_migrations_with_conn(conn)

    async def _apply_migrations_with_conn(self, conn: asyncpg.Connection) -> None:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS pgkg_schema_migrations ("
            "  filename TEXT PRIMARY KEY,"
            "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        applied = {
            r["filename"]
            for r in await conn.fetch("SELECT filename FROM pgkg_schema_migrations")
        }
        for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if migration.name in applied:
                continue
            async with conn.transaction():
                await conn.execute(migration.read_text())
                await conn.execute(
                    "INSERT INTO pgkg_schema_migrations (filename) VALUES ($1)",
                    migration.name,
                )

    async def close(self) -> None:
        await self._pool.close()

    # --- Retrieval primitives --------------------------------------------

    async def keyword_search(
        self,
        query: str,
        k: int,
        namespace: str,
        session_id: str | None = None,
    ) -> list[ScoredId]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT p.id, ts_rank_cd(p.tsv, plainto_tsquery('english', $1)) AS score
                FROM propositions p
                WHERE p.tsv @@ plainto_tsquery('english', $1)
                  AND p.namespace = $2
                  AND p.superseded_by IS NULL
                  AND ($3::text IS NULL OR p.session_id = $3 OR p.session_id IS NULL)
                ORDER BY score DESC
                LIMIT $4
                """,
                query, namespace, session_id, k,
            )
        return [ScoredId(id=r["id"], score=float(r["score"])) for r in rows]

    async def vector_search(
        self,
        embedding: list[float],
        k: int,
        namespace: str,
        session_id: str | None = None,
    ) -> list[ScoredId]:
        vec_lit = _vec_literal(embedding)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT p.id, (1.0 - (p.embedding <=> '{vec_lit}'::vector)) AS score
                FROM propositions p
                WHERE p.embedding IS NOT NULL
                  AND p.namespace = $1
                  AND p.superseded_by IS NULL
                  AND ($2::text IS NULL OR p.session_id = $2 OR p.session_id IS NULL)
                ORDER BY p.embedding <=> '{vec_lit}'::vector
                LIMIT $3
                """,
                namespace, session_id, k,
            )
        return [ScoredId(id=r["id"], score=float(r["score"])) for r in rows]

    async def graph_neighbors(
        self,
        entity_ids: list[UUID],
        namespace: str,
        limit: int = 100,
    ) -> list[ScoredId]:
        if not entity_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT e.proposition_id AS id, 1.0 AS score
                FROM edges e
                JOIN propositions p ON p.id = e.proposition_id
                WHERE (e.src_entity = ANY($1::uuid[]) OR e.dst_entity = ANY($1::uuid[]))
                  AND p.namespace = $2
                  AND p.superseded_by IS NULL
                LIMIT $3
                """,
                entity_ids, namespace, limit,
            )
        return [ScoredId(id=r["id"], score=float(r["score"])) for r in rows]

    async def fused_search(
        self,
        query_text: str,
        query_embedding: list[float],
        k_retrieve: int,
        k_initial: int,
        namespace: str,
        session_id: str | None,
        recency_half_life_days: float,
        expand_graph: bool,
        rrf_k: int,
    ) -> list[Candidate] | None:
        vec_lit = _vec_literal(query_embedding)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT proposition_id, text, embedding, rrf_score, adjusted_score,
                       source_kind, chunk_id, subject_id, predicate, object_id
                FROM pgkg_search($1, '{vec_lit}'::vector,
                                 $2, $3, $4, $5, $6, $7, $8)
                """,
                query_text,
                k_retrieve,
                k_initial,
                namespace,
                session_id,
                recency_half_life_days,
                expand_graph,
                rrf_k,
            )
        return [
            Candidate(
                proposition_id=r["proposition_id"],
                text=r["text"],
                rrf_score=float(r["rrf_score"]),
                adjusted_score=float(r["adjusted_score"]),
                source_kind=r["source_kind"],
                chunk_id=r["chunk_id"],
                subject_id=r["subject_id"],
                predicate=r["predicate"],
                object_id=r["object_id"],
            )
            for r in rows
        ]

    # --- Data resolution -------------------------------------------------

    async def get_propositions(
        self,
        proposition_ids: list[UUID],
    ) -> list[PropositionRow]:
        if not proposition_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, text, embedding, chunk_id, subject_id, predicate,
                       object_id, confidence, access_count,
                       last_accessed_at::text, namespace
                FROM propositions
                WHERE id = ANY($1::uuid[])
                """,
                proposition_ids,
            )
        return [
            PropositionRow(
                id=r["id"],
                text=r["text"],
                embedding=list(r["embedding"]) if r["embedding"] is not None else None,
                chunk_id=r["chunk_id"],
                subject_id=r["subject_id"],
                predicate=r["predicate"],
                object_id=r["object_id"],
                confidence=float(r["confidence"]),
                access_count=r["access_count"],
                last_accessed_at=r["last_accessed_at"],
                namespace=r["namespace"],
            )
            for r in rows
        ]

    async def get_proposition_entity_ids(
        self,
        proposition_ids: list[UUID],
    ) -> dict[UUID, list[UUID]]:
        if not proposition_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, subject_id, object_id
                FROM propositions
                WHERE id = ANY($1::uuid[])
                """,
                proposition_ids,
            )
        result: dict[UUID, list[UUID]] = {}
        for r in rows:
            entities: list[UUID] = []
            if r["subject_id"] is not None:
                entities.append(r["subject_id"])
            if r["object_id"] is not None:
                entities.append(r["object_id"])
            if entities:
                result[r["id"]] = entities
        return result

    # --- Ingest primitives -----------------------------------------------

    async def store_document(self, doc: StoredDocument) -> UUID:
        async with self._pool.acquire() as conn:
            doc_id: UUID = await conn.fetchval(
                "INSERT INTO documents (source, namespace) VALUES ($1, $2) RETURNING id",
                doc.source,
                doc.namespace,
            )
        return doc_id

    async def store_chunk(self, chunk: StoredChunk) -> UUID:
        async with self._pool.acquire() as conn:
            chunk_id: UUID = await conn.fetchval(
                """
                INSERT INTO chunks (document_id, text, span_start, span_end)
                VALUES ($1, $2, $3, $4) RETURNING id
                """,
                chunk.document_id,
                chunk.text,
                chunk.span_start,
                chunk.span_end,
            )
        return chunk_id

    async def store_proposition(self, prop: StoredProposition) -> UUID:
        vec_lit = _vec_literal(prop.embedding)
        metadata_json = json.dumps(prop.metadata) if prop.metadata else None
        async with self._pool.acquire() as conn:
            prop_id: UUID = await conn.fetchval(
                f"""
                INSERT INTO propositions
                    (text, embedding, subject_id, predicate, object_id,
                     object_literal, chunk_id, namespace, session_id,
                     confidence, metadata)
                VALUES ($1, '{vec_lit}'::vector,
                        $2, $3, $4, $5, $6, $7, $8, $9,
                        $10::jsonb)
                RETURNING id
                """,
                prop.text,
                prop.subject_id,
                prop.predicate,
                prop.object_id,
                prop.object_literal,
                prop.chunk_id,
                prop.namespace,
                prop.session_id,
                prop.confidence,
                metadata_json,
            )
        return prop_id

    async def link_entity(
        self,
        namespace: str,
        name: str,
        entity_type: str,
        embedding: list[float],
        threshold: float = 0.85,
    ) -> UUID:
        vec_lit = _vec_literal(embedding)
        async with self._pool.acquire() as conn:
            entity_id: UUID = await conn.fetchval(
                f"SELECT pgkg_link_entity($1, $2, $3, '{vec_lit}'::vector, $4)",
                namespace,
                name,
                entity_type,
                threshold,
            )
        return entity_id

    async def store_edge(
        self,
        src_entity: UUID,
        dst_entity: UUID,
        predicate: str,
        proposition_id: UUID,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO edges (src_entity, dst_entity, relation, proposition_id)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT DO NOTHING
                """,
                src_entity,
                dst_entity,
                predicate,
                proposition_id,
            )

    # --- Access tracking -------------------------------------------------

    async def bump_access(self, proposition_ids: list[UUID]) -> None:
        if not proposition_ids:
            return
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT pgkg_bump_access($1::uuid[])", proposition_ids)

    # --- Entity resolution -----------------------------------------------

    async def resolve_entity_names(
        self, entity_ids: list[UUID],
    ) -> dict[UUID, str]:
        if not entity_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name FROM entities WHERE id = ANY($1::uuid[])",
                entity_ids,
            )
        return {r["id"]: r["name"] for r in rows}

    # --- Forget ----------------------------------------------------------

    async def forget(
        self,
        proposition_id: UUID,
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

    # --- Cache -----------------------------------------------------------

    async def cache_get(self, cache_key: str) -> list[dict] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT propositions FROM proposition_cache WHERE cache_key = $1",
                cache_key,
            )
            if row is None:
                return None
            await conn.execute(
                "UPDATE proposition_cache SET hit_count = hit_count + 1 WHERE cache_key = $1",
                cache_key,
            )
            raw = row["propositions"]
            if isinstance(raw, str):
                return json.loads(raw)
            return raw

    async def cache_put(
        self,
        cache_key: str,
        chunk_hash: str,
        extractor_model: str,
        prompt_version: str,
        propositions: list[dict],
    ) -> None:
        payload = json.dumps(propositions)
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

    # --- Health ----------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)
