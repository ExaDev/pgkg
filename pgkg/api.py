from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from pgkg import ml
from pgkg.config import (
    DEFAULT_COLLECTION_ID,
    DEFAULT_ORG_ID,
    ORG_GUC,
    embed_dim,
    get_settings,
    live_generations,
)
from pgkg.corpus import CorpusIngest, document_hash, dump_provenance
from pgkg.db import make_pool, close_pool
from pgkg.ingest_jobs import job_state
from pgkg.memory import (
    DEFAULT_CORPUS_FRACTION,
    DEFAULT_K_RERANK,
    DEFAULT_MEMORY_FLOOR,
    BeliefRecord,
    Memory,
    Provenance,
    Result,
    Scope,
)

_pool = None
_memory: Memory | None = None
_corpus: CorpusIngest | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _memory, _corpus
    settings = get_settings()
    _pool = await make_pool(settings.database_url)
    _memory = Memory(
        _pool,
        namespace=settings.default_namespace,
        extract_propositions=settings.extract_propositions,
    )
    # The corpus pipeline is bound to a tenant the same way a Memory is, and
    # rebound per request: two pipelines, not one (ADR 0001, D7), but one
    # ownership rule.
    _corpus = CorpusIngest(_pool)
    yield
    # The Memory carries access counts that recall() deliberately did not write;
    # closing it is what turns them into rows, so it happens before the pool
    # goes.  Per-request scopes share its ledger, so this drains theirs too.
    if _memory:
        await _memory.aclose()
    if _pool:
        await close_pool(_pool)


app = FastAPI(title="pgkg", lifespan=lifespan)


class ScopedRequest(BaseModel):
    """The tenant fields every request carries.

    Absent, they resolve to the reserved default org and collection — the
    partition migration 020 backfilled every pre-tenancy row into — so a client
    written before tenancy keeps working and keeps seeing its own data.
    """

    org_id: UUID | None = None
    collection_id: UUID | None = None
    user_id: UUID | None = None
    acl_groups: list[UUID] = []
    subscribed_collection_ids: list[UUID] = []
    include_system_org: bool = False


class ProvenanceRequest(BaseModel):
    kind: str = "document_version"
    source_id: UUID | None = None
    ingest_run_id: UUID | None = None
    actor_user_id: UUID | None = None
    source_url: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime | None = None
    licence: str | None = None
    source_authority: int | None = None


class MemorizeRequest(ScopedRequest):
    text: str
    session_id: str | None = None
    source: str | None = None
    asserted_at: datetime | None = None
    claim_scope: str = "org"
    visibility: str = "shared"
    provenance: ProvenanceRequest | None = None


class RecallRequest(ScopedRequest):
    query: str
    k: int = 10
    session_id: str | None = None
    with_rerank: bool = True
    with_mmr: bool = True
    expand_graph: bool = True
    # As-of-validity: the same belief, read at another world instant.
    valid_at: datetime | None = None
    # Which classes may answer.  Unset fuses both and lets the quota decide;
    # naming one is the per-class option D1 keeps for a caller that would
    # rather route itself.
    sources: list[str] | None = None
    # The quota, exposed because it is a tuning decision: how many rows the
    # split divides, what share of them the corpus may take, and how many are
    # held back for the caller's own memory whatever the corpus scores.
    k_rerank: int = DEFAULT_K_RERANK
    corpus_fraction: float = DEFAULT_CORPUS_FRACTION
    memory_floor: int = DEFAULT_MEMORY_FLOOR


class CollectionRequest(ScopedRequest):
    """A new collection for this tenant.

    `owner_org_id` is not a field: a tenant's collection is owned by that
    tenant.  Only the operator publishes, and nothing a tenant ingests is ever
    promoted into a shared collection (ADR 0001, D4).
    """

    name: str
    kind: str = "corpus"
    claim_scope: str = "org"
    decay_profile: str = "timeless"
    extract_propositions: bool = False
    acl_mode: str = "none"
    licence: str | None = None


class DocumentRequest(ScopedRequest):
    """One document as its connector sees it.

    Keyed on `external_id`, which is what a re-crawl collides on: matching on
    the source path would make a moved file a new document.
    """

    external_id: str
    text: str
    uri: str | None = None
    source: str | None = None
    asserted_at: datetime | None = None
    provenance: ProvenanceRequest | None = None
    # Batch ingest is the default posture for a corpus (D7); inline ingest is
    # for the single document a human just uploaded and is watching.
    queue: bool = False

    @property
    def effective_asserted_at(self) -> datetime | None:
        """The world-time of this document's content.

        Two fields, one clock: D5 says provenance.published_at feeds the
        perishable decay profile and D6 keys that profile on asserted_at, so a
        connector that states a publication date has stated the world-time as
        well.  Left independent, a 2015 article ingested today decays from
        today, which is the distinction between 'perishable' and 'timeless'.
        """
        if self.asserted_at is not None:
            return self.asserted_at
        return self.provenance.published_at if self.provenance else None


class DocumentDeleteRequest(ScopedRequest):
    external_id: str


class ForgetRequest(ScopedRequest):
    proposition_id: UUID
    supersede_with: UUID | None = None
    reason: str | None = None


class BelievedRequest(ScopedRequest):
    belief_at: datetime
    k: int = 100


def _scoped(req: ScopedRequest, **overrides: str) -> Memory:
    """The app's Memory, bound to this request's tenant.

    One place decides which org a request runs as, and everything downstream —
    the retrieval predicate, the RLS GUC, the columns a write lands in — follows
    from the Scope rather than from an argument each handler could forget.
    """
    assert _memory is not None
    try:
        scope = Scope(
            org_id=req.org_id or DEFAULT_ORG_ID,
            collection_id=req.collection_id or DEFAULT_COLLECTION_ID,
            user_id=req.user_id,
            acl_groups=tuple(req.acl_groups),
            subscribed_collection_ids=tuple(req.subscribed_collection_ids),
            include_system_org=req.include_system_org,
            **overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _memory.scoped(scope)


@app.post("/memorize", response_model=dict)
async def memorize(req: MemorizeRequest) -> dict:
    memory = _scoped(
        req, claim_scope=req.claim_scope, visibility=req.visibility
    )
    provenance = (
        Provenance(**req.provenance.model_dump()) if req.provenance else Provenance()
    )
    result = await memory.ingest(
        req.text,
        source=req.source,
        session_id=req.session_id,
        asserted_at=req.asserted_at,
        provenance=provenance,
    )
    return {
        "documents": result.documents,
        "chunks": result.chunks,
        "propositions": result.propositions,
        "entities": result.entities,
        "provenance": len(result.provenance_ids),
        "ingest_run_id": str(result.ingest_run_id),
    }


@app.post("/recall", response_model=list[Result])
async def recall(req: RecallRequest) -> list[Result]:
    memory = _scoped(req)
    try:
        return await memory.recall(
            req.query,
            k=req.k,
            session_id=req.session_id,
            with_rerank=req.with_rerank,
            with_mmr=req.with_mmr,
            expand_graph=req.expand_graph,
            valid_at=req.valid_at,
            sources=req.sources,
            k_rerank=req.k_rerank,
            corpus_fraction=req.corpus_fraction,
            memory_floor=req.memory_floor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/forget", status_code=204)
async def forget(req: ForgetRequest) -> Response:
    memory = _scoped(req)
    try:
        await memory.forget(
            req.proposition_id,
            reason=req.reason,
            supersede_with=req.supersede_with,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(status_code=204)


@app.post("/believed", response_model=list[BeliefRecord])
async def believed(req: BelievedRequest) -> list[BeliefRecord]:
    """The audit path: what this tenant's memory held at an instant.

    Its own endpoint rather than a flag on /recall, because as-of-belief cannot
    use the index current-state retrieval depends on and must not become the hot
    path by accident.
    """
    memory = _scoped(req)
    return await memory.believed_at(req.belief_at, k=req.k)


# The corpus surface.  These three statements sit here rather than beside
# CorpusIngest because they are all the corpus lifecycle needs that the ingest
# pipeline itself does not do; they belong next to it once that module grows a
# management API.
_CREATE_COLLECTION_SQL = """
INSERT INTO collections
    (org_id, owner_org_id, name, kind, claim_scope, decay_profile,
     extract_propositions, acl_mode, licence)
VALUES ($1, $1, $2, $3, $4, $5, $6, $7, $8)
RETURNING id
"""

# A soft delete, because retrieval has to stop returning a withdrawn document
# long before its rows can be physically reclaimed: pgkg_chunk_live() reads
# deleted_at, so one UPDATE withdraws every passage the document carries.
_DELETE_DOCUMENT_SQL = """
UPDATE documents SET deleted_at = now()
WHERE org_id = $1 AND collection_id = $2 AND external_id = $3
  AND deleted_at IS NULL
RETURNING id
"""

# The facts extracted from those passages are not withdrawn by the same read
# path — a proposition has its own belief clock — so the deletion has to say so,
# in the vocabulary migration 021 enumerates (ADR 0001, D6).
_WITHDRAW_DOCUMENT_CLAIMS_SQL = """
UPDATE propositions p
SET invalidated_at = COALESCE(p.invalidated_at, now()),
    invalidation_reason = COALESCE(p.invalidation_reason, 'source_deleted')
WHERE p.org_id = $1
  AND p.invalidated_at IS NULL
  AND p.chunk_id IN (
        SELECT dvc.chunk_id
        FROM document_version_chunks dvc
        JOIN document_versions dv ON dv.id = dvc.document_version_id
        WHERE dv.document_id = $2
      )
"""

# Enqueued through a connection carrying the org, because ingest_jobs holds the
# document text and is therefore under RLS like every other tenant table.
_ENQUEUE_SQL = (
    "SELECT pgkg_enqueue_ingest_job($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)"
)


@asynccontextmanager
async def _scoped_connection(org_id: UUID):
    """A pooled connection with this request's org in the RLS GUC."""
    assert _pool is not None
    async with _pool.acquire() as conn:
        await conn.execute(f"SELECT set_config('{ORG_GUC}', $1, false)", str(org_id))
        yield conn


def _corpus_for(req: ScopedRequest) -> CorpusIngest:
    assert _corpus is not None
    return _corpus.for_collection(
        org_id=req.org_id or DEFAULT_ORG_ID,
        collection_id=req.collection_id or DEFAULT_COLLECTION_ID,
    )


@app.post("/collections", response_model=dict)
async def create_collection(req: CollectionRequest) -> dict:
    org_id = req.org_id or DEFAULT_ORG_ID
    async with _scoped_connection(org_id) as conn:
        try:
            collection_id = await conn.fetchval(
                _CREATE_COLLECTION_SQL,
                org_id,
                req.name,
                req.kind,
                req.claim_scope,
                req.decay_profile,
                req.extract_propositions,
                req.acl_mode,
                req.licence,
            )
        except Exception as exc:  # a CHECK or a duplicate name is the caller's
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"collection_id": str(collection_id)}


@app.post("/documents", response_model=dict)
async def upsert_document(req: DocumentRequest) -> dict:
    """Offer a document to a collection, inline or through the queue.

    Idempotent either way: the document hash decides whether there is any work,
    so a nightly re-crawl of an unchanged corpus writes nothing and embeds
    nothing.
    """
    org_id = req.org_id or DEFAULT_ORG_ID
    collection_id = req.collection_id or DEFAULT_COLLECTION_ID

    if req.queue:
        async with _scoped_connection(org_id) as conn:
            job_id = await conn.fetchval(
                _ENQUEUE_SQL,
                org_id,
                collection_id,
                req.external_id,
                document_hash(req.text),
                req.text,
                req.uri,
                req.source,
                req.effective_asserted_at,
                dump_provenance(
                    Provenance(**req.provenance.model_dump())
                    if req.provenance
                    else None
                ),
            )
        return {"job_id": str(job_id)}

    provenance = (
        Provenance(**req.provenance.model_dump()) if req.provenance else None
    )
    try:
        result = await _corpus_for(req).upsert_document(
            external_id=req.external_id,
            text=req.text,
            uri=req.uri,
            source=req.source,
            asserted_at=req.effective_asserted_at,
            provenance=provenance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "document_id": str(result.document_id),
        "version_id": str(result.version_id),
        "changed": result.changed,
        "chunks_total": result.chunks_total,
        "chunks_new": result.chunks_new,
        "chunks_carried": result.chunks_carried,
        "embedded": result.embedded,
        "cache_hits": result.cache_hits,
        "propositions": result.propositions,
    }


@app.post("/documents/delete", status_code=204)
async def delete_document(req: DocumentDeleteRequest) -> Response:
    org_id = req.org_id or DEFAULT_ORG_ID
    async with _scoped_connection(org_id) as conn:
        async with conn.transaction():
            document_id = await conn.fetchval(
                _DELETE_DOCUMENT_SQL,
                org_id,
                req.collection_id or DEFAULT_COLLECTION_ID,
                req.external_id,
            )
            if document_id is None:
                raise HTTPException(
                    status_code=404, detail=f"no such document {req.external_id!r}"
                )
            await conn.execute(_WITHDRAW_DOCUMENT_CLAIMS_SQL, org_id, document_id)
    return Response(status_code=204)


@app.get("/jobs/{job_id}", response_model=dict)
async def read_job(job_id: UUID, org_id: UUID | None = None) -> dict:
    """Is my corpus indexed yet — the first question a customer asks.

    The org is a parameter of the question, not an inference from the job id: a
    status carries the document id and the extractor's error text, and a UUID a
    caller happens to hold is not a claim on another tenant's queue.
    """
    assert _pool is not None
    try:
        state = await job_state(_pool, job_id, org_id=org_id or DEFAULT_ORG_ID)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "job_id": str(job_id),
        "status": state.status,
        "attempts": state.attempts,
        "chunks_total": state.chunks_total,
        "chunks_embedded": state.chunks_embedded,
        "document_id": str(state.document_id) if state.document_id else None,
        "version_id": str(state.version_id) if state.version_id else None,
        "error": state.error,
        "enqueued_at": state.enqueued_at.isoformat(),
        "finished_at": state.finished_at.isoformat() if state.finished_at else None,
    }


@app.get("/health")
async def health() -> dict:
    db_ok = False
    # The embedding space is reported from the registry rather than from
    # settings, because the registry is what the stored vectors were written
    # against: a configured width could only ever disagree with them.
    embedding: dict = {"dim": None, "generations": []}
    if _pool:
        try:
            async with _pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                embedding = {
                    "dim": await embed_dim(conn, DEFAULT_ORG_ID),
                    "generations": [
                        generation.name
                        for generation in await live_generations(conn, DEFAULT_ORG_ID)
                    ],
                }
            db_ok = True
        except Exception:
            pass

    return {
        "status": "ok",
        "db": db_ok,
        "embedding": embedding,
        "models_loaded": {
            "embed": ml.is_embed_loaded(),
            "rerank": ml.is_rerank_loaded(),
        },
    }
