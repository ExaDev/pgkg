"""Every entry point must close the Memory it opened.

Phase 0 moved access accounting off the read path: recall() accumulates counts
in process and a flush writes them in one statement.  That trade is only safe if
whoever opened the Memory closes it, so the counts a short-lived process
gathered are not silently discarded.  The HTTP app, the CLI and the benchmark
harness are the three owners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import pytest

from pgkg import ml
from pgkg.memory import Memory

_TEXT = "Ada Lovelace wrote the first algorithm for the analytical engine."


@pytest.fixture(scope="session")
async def embed_dim(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )


def _make_embed(dim: int):
    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            v = [0.0] * dim
            v[int.from_bytes(digest[:4], "big") % dim] = 1.0
            out.append(v)
        return out

    return embed


def _ns(tag: str) -> str:
    return f"lifecycle_{tag}_{uuid.uuid4().hex[:8]}"


async def _access_total(pool: asyncpg.Pool, namespace: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT COALESCE(SUM(access_count), 0)
            FROM propositions WHERE namespace = $1
            """,
            namespace,
        )


class _SharedPool:
    """Hands out the session pool without letting the caller close it."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    def acquire(self):
        return self._pool.acquire()


async def test_api_shutdown_flushes_pending_access_counts(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """The app holds one long-lived Memory; shutting it down must not throw away
    the accounting that Memory is carrying."""
    from pgkg import api

    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))
    monkeypatch.setattr(ml, "rerank", lambda query, docs: [1.0] * len(docs))

    namespace = _ns("api")

    class _Settings:
        database_url = "unused"
        default_namespace = namespace
        # The extraction path: access counts live on propositions and nowhere
        # else — a passage has no frequency term in its decay profile — so the
        # accounting this test is about only exists in this mode.  Chunks-only
        # mode writes into the chunk store now (ADR 0001, D1).
        extract_propositions = True

    async def _make_pool(dsn):
        return _SharedPool(pool)

    async def _close_pool(_pool):
        return None

    monkeypatch.setattr(api, "get_settings", lambda: _Settings())
    monkeypatch.setattr(api, "make_pool", _make_pool)
    monkeypatch.setattr(api, "close_pool", _close_pool)

    async with api.lifespan(api.app):
        await api.memorize(api.MemorizeRequest(text=_TEXT))
        results = await api.recall(
            api.RecallRequest(query="analytical engine", with_mmr=False)
        )
        assert results
        assert await _access_total(pool, namespace) == 0

    assert await _access_total(pool, namespace) == len(results)


async def test_cli_recall_flushes_pending_access_counts(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch, capsys
) -> None:
    """`pgkg recall` is a whole process lifetime, so it must flush before it
    exits or every CLI read is lost accounting."""
    from pgkg import cli, db

    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))
    monkeypatch.setattr(ml, "rerank", lambda query, docs: [1.0] * len(docs))

    namespace = _ns("cli")
    await Memory(pool, namespace=namespace).ingest(_TEXT)

    @asynccontextmanager
    async def _pool_from_settings():
        yield _SharedPool(pool)

    class _Settings:
        default_namespace = namespace
        extract_propositions = True

    monkeypatch.setattr(db, "pool_from_settings", _pool_from_settings)
    monkeypatch.setattr("pgkg.config.get_settings", lambda: _Settings())

    await cli.run_recall(argparse.Namespace(query="analytical engine", k=5))

    printed = json.loads(capsys.readouterr().out)
    assert printed
    assert await _access_total(pool, namespace) == len(printed)


async def test_bench_run_flushes_pending_access_counts(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch, tmp_path: Path
) -> None:
    """The harness opens a Memory per item; a completed run must leave no
    unwritten accounting behind."""
    from bench.common import QA, BenchConfig, BenchItem, run_bench

    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    namespace = _ns("bench")
    items = [
        BenchItem(
            id="one",
            namespace=namespace,
            conversation=[
                {"speaker": "Ada", "text": _TEXT, "session_id": "s0"},
            ],
            questions=[
                QA(id="q0", question="analytical engine", answer="Ada Lovelace"),
            ],
        )
    ]
    config = BenchConfig(
        dry_run=True,
        exact_match=True,
        with_rerank=False,
        with_mmr=False,
        expand_graph=False,
        extract_propositions=True,
        output_path=tmp_path,
    )

    report = await run_bench(
        name="lifecycle", items=items, config=config, pool=pool
    )

    assert report.total == 1
    assert await _access_total(pool, namespace) > 0
