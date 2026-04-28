from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from pgkg import ml
from pgkg.config import get_settings
from pgkg.db import make_pool, close_pool
from pgkg.memory import Memory, IngestResult, Result

_pool = None
_memory: Memory | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _memory
    settings = get_settings()
    _pool = await make_pool(settings.database_url)
    _memory = Memory(_pool, namespace=settings.default_namespace)
    yield
    if _pool:
        await close_pool(_pool)


app = FastAPI(title="pgkg", lifespan=lifespan)


class MemorizeRequest(BaseModel):
    text: str
    session_id: str | None = None
    source: str | None = None


class RecallRequest(BaseModel):
    query: str
    k: int = 10
    session_id: str | None = None
    with_rerank: bool = True
    with_mmr: bool = True
    expand_graph: bool = True


class ForgetRequest(BaseModel):
    proposition_id: UUID
    supersede_with: UUID | None = None


@app.post("/memorize", response_model=dict)
async def memorize(req: MemorizeRequest) -> dict:
    assert _memory is not None
    result = await _memory.ingest(req.text, source=req.source, session_id=req.session_id)
    return {
        "documents": result.documents,
        "chunks": result.chunks,
        "propositions": result.propositions,
        "entities": result.entities,
    }


@app.post("/recall", response_model=list[Result])
async def recall(req: RecallRequest) -> list[Result]:
    assert _memory is not None
    return await _memory.recall(
        req.query,
        k=req.k,
        session_id=req.session_id,
        with_rerank=req.with_rerank,
        with_mmr=req.with_mmr,
        expand_graph=req.expand_graph,
    )


@app.post("/forget", status_code=204)
async def forget(req: ForgetRequest) -> Response:
    assert _memory is not None
    await _memory.forget(req.proposition_id, supersede_with=req.supersede_with)
    return Response(status_code=204)


@app.get("/health")
async def health() -> dict:
    db_ok = False
    if _pool:
        try:
            async with _pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
        except Exception:
            pass

    return {
        "status": "ok",
        "db": db_ok,
        "models_loaded": {
            "embed": ml.is_embed_loaded(),
            "rerank": ml.is_rerank_loaded(),
        },
    }
