"""StorageBackend protocol — the abstraction boundary between pgkg and its storage layer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID


@dataclass(frozen=True)
class ScoredId:
    """A database row ID with an associated relevance score."""
    id: UUID
    score: float


@dataclass(frozen=True)
class Candidate:
    """A proposition candidate with full retrieval metadata."""
    proposition_id: UUID
    text: str
    rrf_score: float
    adjusted_score: float
    source_kind: str  # "kw" | "vec" | "both" | "graph"
    chunk_id: UUID | None
    subject_id: UUID | None
    predicate: str | None
    object_id: UUID | None
    asserted_at: datetime | None = None


@dataclass(frozen=True)
class PropositionRow:
    """Full proposition data needed for scoring and reranking."""
    id: UUID
    text: str
    embedding: list[float] | None
    chunk_id: UUID | None
    subject_id: UUID | None
    predicate: str | None
    object_id: UUID | None
    confidence: float
    access_count: int
    last_accessed_at: str  # ISO 8601 timestamp
    namespace: str


@dataclass(frozen=True)
class StoredProposition:
    """Data needed to store a proposition."""
    text: str
    embedding: list[float]
    subject_id: UUID | None
    predicate: str | None
    object_id: UUID | None
    object_literal: str | None
    chunk_id: UUID | None
    namespace: str
    session_id: str | None
    confidence: float = 1.0
    metadata: dict | None = None
    asserted_at: datetime | None = None


@dataclass(frozen=True)
class StoredDocument:
    """Data needed to store a document."""
    id: UUID | None = None
    source: str | None = None
    namespace: str = "default"


@dataclass(frozen=True)
class StoredChunk:
    """Data needed to store a chunk."""
    document_id: UUID
    text: str
    id: UUID | None = None
    span_start: int | None = None
    span_end: int | None = None
    asserted_at: datetime | None = None


@runtime_checkable
class StorageBackend(Protocol):
    """Abstract storage backend for pgkg.

    Implementations must support all methods.  The ``fused_search`` method
    is optional — backends that cannot fuse retrieval in a single query
    should return ``None``, and the caller falls back to primitive
    orchestration.
    """

    # --- Lifecycle -------------------------------------------------------

    async def apply_migrations(self) -> None:
        """Apply schema migrations.  Called once during initialisation."""
        ...

    async def close(self) -> None:
        """Release all resources (connections, pools, file handles)."""
        ...

    # --- Retrieval primitives --------------------------------------------

    async def keyword_search(
        self,
        query: str,
        k: int,
        namespace: str,
        session_id: str | None = None,
    ) -> list[ScoredId]:
        """Full-text search.  Returns (id, score) pairs ranked by relevance."""
        ...

    async def vector_search(
        self,
        embedding: list[float],
        k: int,
        namespace: str,
        session_id: str | None = None,
    ) -> list[ScoredId]:
        """Vector similarity search.  Returns (id, score) pairs ranked by closeness."""
        ...

    async def graph_neighbors(
        self,
        entity_ids: list[UUID],
        namespace: str,
        limit: int = 100,
    ) -> list[ScoredId]:
        """One-hop graph expansion from seed entities.  Returns proposition IDs."""
        ...

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
        """Push-down fused retrieval (optional optimisation).

        Backends that can execute the full retrieval pipeline in a single
        query (e.g. the Postgres CTE) return results here.  Other backends
        return ``None`` and the caller orchestrates via the primitive methods.
        """
        return None

    # --- Data resolution -------------------------------------------------

    async def get_propositions(
        self,
        proposition_ids: list[UUID],
    ) -> list[PropositionRow]:
        """Fetch full proposition data for the given IDs."""
        ...

    async def get_proposition_entity_ids(
        self,
        proposition_ids: list[UUID],
    ) -> dict[UUID, list[UUID]]:
        """Return ``{proposition_id: [entity_ids]}`` for the given propositions."""
        ...

    # --- Ingest primitives -----------------------------------------------

    async def store_document(self, doc: StoredDocument) -> UUID:
        """Insert a document.  Returns its ID."""
        ...

    async def store_chunk(self, chunk: StoredChunk) -> UUID:
        """Insert a chunk.  Returns its ID."""
        ...

    async def store_proposition(self, prop: StoredProposition) -> UUID:
        """Insert a proposition.  Returns its ID."""
        ...

    async def link_entity(
        self,
        namespace: str,
        name: str,
        entity_type: str,
        embedding: list[float],
        threshold: float = 0.85,
    ) -> UUID:
        """Find or create an entity by name/type/embedding similarity."""
        ...

    async def store_edge(
        self,
        src_entity: UUID,
        dst_entity: UUID,
        predicate: str,
        proposition_id: UUID,
    ) -> None:
        """Create an edge between two entities via a proposition."""
        ...

    # --- Access tracking -------------------------------------------------

    async def bump_access(self, proposition_ids: list[UUID]) -> None:
        """Increment access_count and update last_accessed_at."""
        ...

    # --- Entity resolution -----------------------------------------------

    async def resolve_entity_names(
        self, entity_ids: list[UUID],
    ) -> dict[UUID, str]:
        """Map entity IDs to names.  Returns ``{id: name}`` for found entities."""
        ...

    # --- Forget ----------------------------------------------------------

    async def forget(
        self,
        proposition_id: UUID,
        supersede_with: UUID | None = None,
    ) -> None:
        """Mark a proposition as superseded."""
        ...

    # --- Cache (optional) ------------------------------------------------

    async def cache_get(self, cache_key: str) -> list[dict] | None:
        """Retrieve cached extraction results as raw dicts.

        Returns ``None`` on cache miss.  The caller is responsible for
        deserialising dicts into ``Proposition`` objects.
        """
        return None

    async def cache_put(
        self,
        cache_key: str,
        chunk_hash: str,
        extractor_model: str,
        prompt_version: str,
        propositions: list[dict],
    ) -> None:
        """Store extraction results in cache as raw dicts."""
        ...

    # --- Health ----------------------------------------------------------

    async def health_check(self) -> bool:
        """Return True if the backend is healthy (can serve queries)."""
        ...
