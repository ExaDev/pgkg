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
    embed_dim,
    get_settings,
    live_generations,
)
from pgkg.db import make_pool, close_pool
from pgkg.memory import BeliefRecord, Memory, Provenance, Result, Scope

_pool = None
_memory: Memory | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _memory
    settings = get_settings()
    _pool = await make_pool(settings.database_url)
    _memory = Memory(
        _pool,
        namespace=settings.default_namespace,
        extract_propositions=settings.extract_propositions,
    )
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
    return await memory.recall(
        req.query,
        k=req.k,
        session_id=req.session_id,
        with_rerank=req.with_rerank,
        with_mmr=req.with_mmr,
        expand_graph=req.expand_graph,
        valid_at=req.valid_at,
    )


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
