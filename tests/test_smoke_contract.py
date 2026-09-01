"""The three requests `make smoke` sends, exactly as it sends them.

Smoke is how an operator finds out whether a deployment is alive, and its
payloads are a contract nothing else in the suite covers: every other API test
builds a request out of the current models, so a required field added to
`MemorizeRequest` or a renamed key in a response would pass the whole suite and
still leave `make smoke` failing against a healthy box.

The bodies below are copied from the Makefile rather than constructed, which is
the point — if the two drift, this fails.
"""
from __future__ import annotations

import hashlib
import uuid

import asyncpg
import httpx
import pytest

from pgkg import ml

HEALTH_PATH = "/health"
MEMORIZE_BODY = {"text": "pgkg smoke test memory", "source": "smoke"}
RECALL_BODY = {"query": "smoke test memory", "k": 3}


@pytest.fixture(scope="session")
async def smoke_dim(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )


@pytest.fixture(autouse=True)
def _offline_models(smoke_dim: int, monkeypatch):
    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            v = [0.0] * smoke_dim
            v[int.from_bytes(digest[:4], "big") % smoke_dim] = 1.0
            out.append(v)
        return out

    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")
    monkeypatch.setattr(ml, "embed", embed)
    monkeypatch.setattr(ml, "rerank", lambda query, docs: [1.0] * len(docs))


class _SharedPool:
    """The session pool, handed to the app without letting it close it."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    def __getattr__(self, name):
        return getattr(self._pool, name)

    def acquire(self, *args, **kwargs):
        return self._pool.acquire(*args, **kwargs)


@pytest.fixture
async def smoke_client(pool: asyncpg.Pool, monkeypatch):
    from pgkg import api

    namespace = f"smoke_{uuid.uuid4().hex[:10]}"

    class _Settings:
        database_url = "unused"
        default_namespace = namespace
        extract_propositions = True

    monkeypatch.setattr(api, "get_settings", lambda: _Settings())
    monkeypatch.setattr(api, "make_pool", lambda dsn: _shared(pool))
    monkeypatch.setattr(api, "close_pool", _noop)

    async with api.lifespan(api.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://pgkg.test",
        ) as client:
            yield client


async def _shared(pool: asyncpg.Pool) -> _SharedPool:
    return _SharedPool(pool)


async def _noop(_pool) -> None:
    return None


async def test_smoke_health_reports_a_json_object(smoke_client) -> None:
    """`curl -sf` fails the build on any non-2xx, and the result is piped
    straight into json.tool, so the body has to parse as JSON."""
    response = await smoke_client.get(HEALTH_PATH)

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


async def test_smoke_memorize_accepts_the_makefile_body(smoke_client) -> None:
    """Two keys, `text` and `source`, and nothing else.  A newly required
    field anywhere in MemorizeRequest breaks the operator's first command."""
    response = await smoke_client.post("/memorize", json=MEMORIZE_BODY)

    assert response.status_code == 200, response.text
    assert isinstance(response.json(), dict)


async def test_smoke_recall_accepts_the_makefile_body(smoke_client) -> None:
    """`query` and `k`, and a JSON body out.  Recall now spans two stores and
    carries a quota, and none of that may become mandatory for this caller."""
    await smoke_client.post("/memorize", json=MEMORIZE_BODY)

    response = await smoke_client.post("/recall", json=RECALL_BODY)

    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    assert len(body) <= RECALL_BODY["k"]


async def test_smoke_recall_returns_what_smoke_just_memorized(
    smoke_client,
) -> None:
    """The sequence is the assertion: an operator reads these three commands as
    one story, and a recall that answers 200 with nothing in it has not shown
    the deployment works."""
    await smoke_client.post("/memorize", json=MEMORIZE_BODY)

    body = (await smoke_client.post("/recall", json=RECALL_BODY)).json()

    assert any(MEMORIZE_BODY["text"] in row["text"] for row in body), body
