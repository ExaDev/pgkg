"""Tenancy, provenance and generations across the Python surface.

Migrations 020-022 put the scoping columns, the derivation record and the
embedder registry in the database.  Nothing enforces them until the layer that
takes a request decides which org is asking, so these tests drive that layer:

  * a Scope is bound to a Memory, not passed per call, so no call site can
    forget the isolation boundary
  * the org GUC reaches every connection, which is what gives RLS and the
    entity-org default anything to work with
  * ingest() writes a real provenance row, and the run id it returns is enough
    to retract the batch
  * forget() records why belief ended, not only what replaced it
  * the audit path is a separate method, reachable over HTTP as its own endpoint
  * and the cross-org negative is asserted through HTTP, because the API is
    where a real leak would happen
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import asyncpg
import httpx
import pytest
from pgvector import HalfVector

from pgkg import ml
from pgkg.config import (
    DEFAULT_COLLECTION_ID,
    DEFAULT_ORG_ID,
    GENERATION_1_ID,
    SYSTEM_ORG_ID,
    embed_dim as registry_embed_dim,
    live_generations,
)
from pgkg.memory import Memory, Provenance, Scope, provision_org


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
async def dim(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )


def _ns(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _make_embed(width: int):
    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            v = [0.0] * width
            v[int.from_bytes(digest[:4], "big") % width] = 1.0
            out.append(v)
        return out

    return embed


@pytest.fixture(autouse=True)
def _offline_models(dim: int, monkeypatch):
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")
    monkeypatch.setattr(ml, "embed", _make_embed(dim))
    monkeypatch.setattr(ml, "rerank", lambda query, docs: [1.0] * len(docs))


async def _make_org(pool: asyncpg.Pool, name: str) -> uuid.UUID:
    async with pool.acquire() as conn:
        return await provision_org(conn, name)


async def _make_user(pool: asyncpg.Pool, org_id: uuid.UUID) -> uuid.UUID:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
            org_id,
            uuid.uuid4().hex,
        )


_FACT = "Ada Lovelace wrote the first algorithm for the analytical engine."


# ---------------------------------------------------------------------------
# 1. The scope object
# ---------------------------------------------------------------------------

async def test_reserved_scope_ids_match_the_migration(pool: asyncpg.Pool) -> None:
    """The constants the Python layer defaults to are the rows the migration
    reserved; a drift between them would silently write into a partition that
    does not exist."""
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT pgkg_system_org()") == SYSTEM_ORG_ID
        assert await conn.fetchval("SELECT pgkg_default_org()") == DEFAULT_ORG_ID
        assert (
            await conn.fetchval("SELECT pgkg_default_collection()")
            == DEFAULT_COLLECTION_ID
        )
        assert await conn.fetchval("SELECT pgkg_generation_1()") == GENERATION_1_ID


async def test_a_provisioned_org_is_bound_to_the_primary_generation(
    pool: asyncpg.Pool,
) -> None:
    """An org with no embedder binding has no primary generation, so its own
    inline vectors are unsearchable.  Creating the org is where that is fixed,
    because the migration can only seed the orgs that already exist."""
    org = await _make_org(pool, "provisioned")

    async with pool.acquire() as conn:
        generations = await live_generations(conn, org)

    assert [g.generation_id for g in generations if g.role == "primary"] == [
        GENERATION_1_ID
    ]


def test_private_scope_without_an_owner_is_refused() -> None:
    with pytest.raises(ValueError):
        Scope(org_id=DEFAULT_ORG_ID, visibility="private")


def test_a_shared_write_by_a_known_user_leaves_the_owner_column_null() -> None:
    """The CHECK constraint ties owner_user_id to private visibility, but the
    requester's identity is still needed on the read side and as the provenance
    actor, so knowing the user is not the same as owning the row."""
    user = uuid.uuid4()
    scope = Scope(org_id=DEFAULT_ORG_ID, visibility="shared", user_id=user)
    assert scope.owner_user_id is None
    assert scope.user_id == user


def test_unknown_claim_scope_is_refused() -> None:
    with pytest.raises(ValueError):
        Scope(org_id=DEFAULT_ORG_ID, claim_scope="galaxy")


def test_unknown_visibility_is_refused() -> None:
    with pytest.raises(ValueError):
        Scope(org_id=DEFAULT_ORG_ID, visibility="secret")


def test_system_org_is_readable_but_never_written() -> None:
    """D4: a tenant reads shared collections and never writes into them."""
    scope = Scope(org_id=DEFAULT_ORG_ID, include_system_org=True)
    assert scope.read_org_ids == [DEFAULT_ORG_ID, SYSTEM_ORG_ID]
    assert scope.write_org_id == DEFAULT_ORG_ID


def test_subscribed_collections_widen_reads_only() -> None:
    other = uuid.uuid4()
    scope = Scope(org_id=DEFAULT_ORG_ID, subscribed_collection_ids=(other,))
    assert scope.read_collection_ids == [DEFAULT_COLLECTION_ID, other]
    assert scope.write_collection_id == DEFAULT_COLLECTION_ID


# ---------------------------------------------------------------------------
# 2. Scoping on the write path
# ---------------------------------------------------------------------------

async def test_ingest_writes_the_scope_columns(pool: asyncpg.Pool) -> None:
    org = await _make_org(pool, "scope_columns")
    user = await _make_user(pool, org)
    group = uuid.uuid4()
    ns = _ns("scope_cols")

    scope = Scope(
        org_id=org,
        collection_id=DEFAULT_COLLECTION_ID,
        user_id=user,
        visibility="private",
        claim_scope="user",
        acl_groups=(group,),
        acl_group_id=group,
    )
    mem = Memory(pool, namespace=ns, scope=scope, extract_propositions=False)
    await mem.ingest(_FACT)

    async with pool.acquire() as conn:
        prop = await conn.fetchrow(
            "SELECT org_id, collection_id, claim_scope, visibility, owner_user_id,"
            " acl_group_id FROM propositions WHERE namespace = $1",
            ns,
        )
        chunk = await conn.fetchrow(
            "SELECT c.org_id, c.collection_id, c.visibility, c.owner_user_id,"
            " c.acl_group_id FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.namespace = $1",
            ns,
        )
        doc = await conn.fetchrow(
            "SELECT org_id, collection_id FROM documents WHERE namespace = $1", ns
        )

    assert prop["org_id"] == org
    assert prop["collection_id"] == DEFAULT_COLLECTION_ID
    assert prop["claim_scope"] == "user"
    assert prop["visibility"] == "private"
    assert prop["owner_user_id"] == user
    assert prop["acl_group_id"] == group
    assert chunk["org_id"] == org
    assert chunk["visibility"] == "private"
    assert chunk["owner_user_id"] == user
    assert doc["org_id"] == org


async def test_entities_take_their_org_from_the_request_guc(
    pool: asyncpg.Pool,
) -> None:
    """pgkg_link_entity() takes no org argument, so entity rows get theirs from
    the column default, which reads pgkg.org_id.  If the GUC does not reach the
    connection the entities land in the backfill org — which is exactly the
    cross-tenant mistake RLS exists to catch."""
    org = await _make_org(pool, "entity_guc")
    ns = _ns("entity_guc")

    mem = Memory(pool, namespace=ns, scope=Scope(org_id=org))
    result = await mem.ingest(_FACT)
    assert result.entities > 0

    async with pool.acquire() as conn:
        orgs = [
            r["org_id"]
            for r in await conn.fetch(
                "SELECT DISTINCT org_id FROM entities WHERE namespace = $1", ns
            )
        ]

    assert orgs == [org]


# ---------------------------------------------------------------------------
# 3. Scoping on the read path
# ---------------------------------------------------------------------------

async def test_recall_cannot_reach_another_orgs_facts(pool: asyncpg.Pool) -> None:
    """One namespace, two orgs: the only thing keeping them apart is the org
    predicate."""
    org_a = await _make_org(pool, "read_a")
    org_b = await _make_org(pool, "read_b")
    ns = _ns("cross_org")

    await Memory(
        pool, namespace=ns, scope=Scope(org_id=org_b), extract_propositions=False
    ).ingest("Org B keeps its refund policy at thirty days.")

    mine = Memory(
        pool, namespace=ns, scope=Scope(org_id=org_a), extract_propositions=False
    )
    results = await mine.recall("refund policy", with_rerank=False, with_mmr=False)
    assert results == []

    theirs = Memory(
        pool, namespace=ns, scope=Scope(org_id=org_b), extract_propositions=False
    )
    assert await theirs.recall("refund policy", with_rerank=False, with_mmr=False)


async def test_recall_cannot_reach_another_users_private_fact(
    pool: asyncpg.Pool,
) -> None:
    org = await _make_org(pool, "private_org")
    owner = await _make_user(pool, org)
    other = await _make_user(pool, org)
    ns = _ns("private")

    await Memory(
        pool,
        namespace=ns,
        scope=Scope(org_id=org, user_id=owner, visibility="private", claim_scope="user"),
        extract_propositions=False,
    ).ingest("The owner prefers dark mode in the console.")

    intruder = Memory(
        pool,
        namespace=ns,
        scope=Scope(org_id=org, user_id=other, visibility="private", claim_scope="user"),
        extract_propositions=False,
    )
    assert await intruder.recall("dark mode", with_rerank=False, with_mmr=False) == []

    mine = Memory(
        pool,
        namespace=ns,
        scope=Scope(org_id=org, user_id=owner, visibility="private", claim_scope="user"),
        extract_propositions=False,
    )
    assert await mine.recall("dark mode", with_rerank=False, with_mmr=False)


# ---------------------------------------------------------------------------
# 4. Provenance
# ---------------------------------------------------------------------------

async def test_ingest_records_provenance(pool: asyncpg.Pool) -> None:
    org = await _make_org(pool, "prov_org")
    actor = await _make_user(pool, org)
    ns = _ns("prov")
    run_id = uuid.uuid4()

    mem = Memory(pool, namespace=ns, scope=Scope(org_id=org))
    result = await mem.ingest(
        _FACT,
        source="notebook.md",
        provenance=Provenance(
            kind="chat_turn", ingest_run_id=run_id, actor_user_id=actor
        ),
    )

    assert result.ingest_run_id == run_id
    assert len(result.provenance_ids) == result.chunks

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM provenance WHERE ingest_run_id = $1", run_id
        )
        unattributed = await conn.fetchval(
            "SELECT COUNT(*) FROM propositions p"
            " WHERE p.namespace = $1"
            "   AND p.provenance_id = pgkg_unattributed_provenance()",
            ns,
        )
        chunk_prov = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.namespace = $1"
            "   AND c.provenance_id = pgkg_unattributed_provenance()",
            ns,
        )

    assert len(rows) == result.chunks
    row = rows[0]
    assert row["org_id"] == org
    assert row["kind"] == "chat_turn"
    assert row["producer"] == "llm_extract"
    assert row["producer_model"]
    assert row["prompt_version"] == ml.PROMPT_VERSION
    assert row["actor_user_id"] == actor
    assert row["source_locator"] is not None
    assert unattributed == 0
    assert chunk_prov == 0


async def test_chunks_only_ingest_records_the_chunker_as_producer(
    pool: asyncpg.Pool,
) -> None:
    ns = _ns("prov_chunker")
    run_id = uuid.uuid4()
    mem = Memory(pool, namespace=ns, extract_propositions=False)
    await mem.ingest(_FACT, provenance=Provenance(ingest_run_id=run_id))

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT kind, producer, producer_model, prompt_version FROM provenance"
            " WHERE ingest_run_id = $1",
            run_id,
        )

    assert row["kind"] == "document_version"
    assert row["producer"] == "chunker"
    assert row["producer_model"] is None
    assert row["prompt_version"] is None


async def test_ingest_records_external_source_fields(pool: asyncpg.Pool) -> None:
    ns = _ns("prov_external")
    run_id = uuid.uuid4()
    published = datetime(2019, 4, 1, tzinfo=timezone.utc)
    retrieved = datetime(2026, 8, 1, tzinfo=timezone.utc)

    mem = Memory(pool, namespace=ns, extract_propositions=False)
    await mem.ingest(
        _FACT,
        provenance=Provenance(
            kind="document_version",
            ingest_run_id=run_id,
            source_url="https://example.invalid/handbook",
            publisher="Example Standards Body",
            published_at=published,
            retrieved_at=retrieved,
            licence="CC-BY-4.0",
            source_authority=7,
        ),
    )

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT source_url, publisher, published_at, retrieved_at, licence,"
            " source_authority FROM provenance WHERE ingest_run_id = $1",
            run_id,
        )

    assert row["source_url"] == "https://example.invalid/handbook"
    assert row["publisher"] == "Example Standards Body"
    assert row["published_at"] == published
    assert row["retrieved_at"] == retrieved
    assert row["licence"] == "CC-BY-4.0"
    assert row["source_authority"] == 7


async def test_ingest_links_every_proposition_to_its_provenance(
    pool: asyncpg.Pool,
) -> None:
    """The many-to-one table is what makes erasure a COUNT(*) rather than a
    special case, so the link is written at ingest and not on first dedup."""
    ns = _ns("prov_link")
    mem = Memory(pool, namespace=ns, extract_propositions=False)
    result = await mem.ingest(_FACT)

    async with pool.acquire() as conn:
        linked = await conn.fetchval(
            "SELECT COUNT(*) FROM proposition_provenance pp"
            " JOIN propositions p ON p.id = pp.proposition_id"
            " WHERE p.namespace = $1 AND pp.provenance_id = p.provenance_id",
            ns,
        )

    assert linked == result.propositions


async def test_retracting_an_ingest_run_withdraws_it_from_recall(
    pool: asyncpg.Pool,
) -> None:
    ns = _ns("prov_retract")
    mem = Memory(pool, namespace=ns, extract_propositions=False)
    result = await mem.ingest(_FACT)
    assert await mem.recall("analytical engine", with_rerank=False, with_mmr=False)

    async with pool.acquire() as conn:
        withdrawn = await conn.fetchval(
            "SELECT pgkg_retract_ingest_run($1)", result.ingest_run_id
        )

    assert withdrawn == result.propositions
    assert await mem.recall("analytical engine", with_rerank=False, with_mmr=False) == []


# ---------------------------------------------------------------------------
# 5. Forget records a reason
# ---------------------------------------------------------------------------

async def test_forget_records_a_reason(pool: asyncpg.Pool) -> None:
    ns = _ns("forget_reason")
    mem = Memory(pool, namespace=ns, extract_propositions=False)
    await mem.ingest(_FACT)
    results = await mem.recall("analytical engine", with_rerank=False, with_mmr=False)
    target = results[0].proposition_id

    await mem.forget(target)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT invalidated_at, invalidation_reason, superseded_by"
            " FROM propositions WHERE id = $1",
            target,
        )

    assert row["invalidated_at"] is not None
    assert row["invalidation_reason"] == "user_deleted"
    assert row["superseded_by"] is None
    assert await mem.recall("analytical engine", with_rerank=False, with_mmr=False) == []


async def test_forget_with_a_replacement_records_supersession(
    pool: asyncpg.Pool,
) -> None:
    ns = _ns("forget_supersede")
    mem = Memory(pool, namespace=ns, extract_propositions=False)
    await mem.ingest("The refund window is thirty days.")
    await mem.ingest("The refund window is sixty days.")
    rows = await mem.recall("refund window", k=10, with_rerank=False, with_mmr=False)
    old, new = rows[0].proposition_id, rows[1].proposition_id

    await mem.forget(old, supersede_with=new)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT invalidated_at, invalidation_reason, superseded_by"
            " FROM propositions WHERE id = $1",
            old,
        )

    assert row["invalidated_at"] is not None
    assert row["invalidation_reason"] == "superseded"
    assert row["superseded_by"] == new


async def test_forget_accepts_an_explicit_reason(pool: asyncpg.Pool) -> None:
    ns = _ns("forget_ttl")
    mem = Memory(pool, namespace=ns, extract_propositions=False)
    await mem.ingest(_FACT)
    target = (
        await mem.recall("analytical engine", with_rerank=False, with_mmr=False)
    )[0].proposition_id

    await mem.forget(target, reason="source_deleted")

    async with pool.acquire() as conn:
        reason = await conn.fetchval(
            "SELECT invalidation_reason FROM propositions WHERE id = $1", target
        )

    assert reason == "source_deleted"


async def test_forget_refuses_an_unknown_reason(pool: asyncpg.Pool) -> None:
    mem = Memory(pool, namespace=_ns("forget_bad"), extract_propositions=False)
    with pytest.raises(ValueError):
        await mem.forget(uuid.uuid4(), reason="because")


async def test_forget_cannot_reach_another_orgs_fact(pool: asyncpg.Pool) -> None:
    org_a = await _make_org(pool, "forget_a")
    org_b = await _make_org(pool, "forget_b")
    ns = _ns("forget_cross")

    theirs = Memory(
        pool, namespace=ns, scope=Scope(org_id=org_b), extract_propositions=False
    )
    await theirs.ingest(_FACT)
    target = (
        await theirs.recall("analytical engine", with_rerank=False, with_mmr=False)
    )[0].proposition_id

    await Memory(
        pool, namespace=ns, scope=Scope(org_id=org_a), extract_propositions=False
    ).forget(target)

    async with pool.acquire() as conn:
        still_live = await conn.fetchval(
            "SELECT invalidated_at IS NULL FROM propositions WHERE id = $1", target
        )

    assert still_live is True


# ---------------------------------------------------------------------------
# 6. The audit path
# ---------------------------------------------------------------------------

async def test_believed_at_returns_what_was_believed_before_a_forget(
    pool: asyncpg.Pool,
) -> None:
    ns = _ns("audit")
    mem = Memory(pool, namespace=ns, extract_propositions=False)
    await mem.ingest(_FACT)
    target = (
        await mem.recall("analytical engine", with_rerank=False, with_mmr=False)
    )[0].proposition_id
    await mem.forget(target, reason="contradicted")

    now = datetime.now(timezone.utc)
    current = await mem.recall("analytical engine", with_rerank=False, with_mmr=False)
    history = await mem.believed_at(now - timedelta(seconds=0))

    assert current == []
    assert target not in [r.proposition_id for r in history]

    async with pool.acquire() as conn:
        recorded = await conn.fetchval(
            "SELECT recorded_at FROM propositions WHERE id = $1", target
        )

    before = await mem.believed_at(recorded + timedelta(milliseconds=1))
    entry = next(r for r in before if r.proposition_id == target)
    assert entry.invalidated_at is not None
    assert entry.invalidation_reason == "contradicted"
    assert entry.provenance_id is not None


async def test_believed_at_is_org_scoped(pool: asyncpg.Pool) -> None:
    org_a = await _make_org(pool, "audit_a")
    org_b = await _make_org(pool, "audit_b")
    ns = _ns("audit_cross")

    await Memory(
        pool, namespace=ns, scope=Scope(org_id=org_b), extract_propositions=False
    ).ingest(_FACT)

    auditor = Memory(
        pool, namespace=ns, scope=Scope(org_id=org_a), extract_propositions=False
    )
    assert await auditor.believed_at(datetime.now(timezone.utc)) == []


# ---------------------------------------------------------------------------
# 7. The embedder registry replaces the ignored embed_dim setting
# ---------------------------------------------------------------------------

async def test_embed_dim_comes_from_the_registry(pool: asyncpg.Pool, dim: int) -> None:
    async with pool.acquire() as conn:
        assert await registry_embed_dim(conn) == dim
        assert await registry_embed_dim(conn, org_id=DEFAULT_ORG_ID) == dim


async def test_live_generations_lists_the_primary_generation(
    pool: asyncpg.Pool, dim: int
) -> None:
    async with pool.acquire() as conn:
        generations = await live_generations(conn, DEFAULT_ORG_ID)

    primary = [g for g in generations if g.role == "primary"]
    assert len(primary) == 1
    assert primary[0].generation_id == GENERATION_1_ID
    assert primary[0].dim == dim
    assert primary[0].normalize is True


# ---------------------------------------------------------------------------
# 8. The HTTP layer, which is where the GUC is set and where a leak would happen
# ---------------------------------------------------------------------------

class _SharedPool:
    """Hands the app the test pool without letting it close it."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    def acquire(self):
        return self._pool.acquire()


@asynccontextmanager
async def _api(pool: asyncpg.Pool, namespace: str, monkeypatch):
    """The real app, over real HTTP, on the test pool.

    ASGITransport does not send lifespan events, which is what makes it safe to
    run the app's own lifespan here instead: the pool belongs to the session
    event loop and a threaded test client would reach it from another one.
    """
    from pgkg import api

    class _Settings:
        database_url = "unused"
        default_namespace = namespace
        extract_propositions = False

    async def _make_pool(dsn):
        return _SharedPool(pool)

    async def _close_pool(_pool):
        return None

    monkeypatch.setattr(api, "get_settings", lambda: _Settings())
    monkeypatch.setattr(api, "make_pool", _make_pool)
    monkeypatch.setattr(api, "close_pool", _close_pool)

    async with api.lifespan(api.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://pgkg.test",
        ) as client:
            yield client


async def test_http_recall_cannot_cross_the_org_boundary(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """The acceptance case: a leak would happen here, not in the SQL, because
    this is the layer that decides which org is asking."""
    org_a = await _make_org(pool, "http_a")
    org_b = await _make_org(pool, "http_b")
    ns = _ns("http_cross")

    async with _api(pool, ns, monkeypatch) as client:
        stored = await client.post(
            "/memorize",
            json={
                "text": "Org B settles invoices in fourteen days.",
                "org_id": str(org_b),
            },
        )
        assert stored.status_code == 200

        intruder = await client.post(
            "/recall",
            json={
                "query": "settles invoices",
                "org_id": str(org_a),
                "with_rerank": False,
                "with_mmr": False,
            },
        )
        owner = await client.post(
            "/recall",
            json={
                "query": "settles invoices",
                "org_id": str(org_b),
                "with_rerank": False,
                "with_mmr": False,
            },
        )

    assert intruder.status_code == 200
    assert intruder.json() == []
    assert owner.status_code == 200
    assert [r["text"] for r in owner.json()]


async def test_http_recall_cannot_reach_another_users_private_fact(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    org = await _make_org(pool, "http_private")
    owner = await _make_user(pool, org)
    other = await _make_user(pool, org)
    ns = _ns("http_private")

    async with _api(pool, ns, monkeypatch) as client:
        await client.post(
            "/memorize",
            json={
                "text": "The owner keeps a spare key in the third drawer.",
                "org_id": str(org),
                "user_id": str(owner),
                "visibility": "private",
                "claim_scope": "user",
            },
        )
        intruder = await client.post(
            "/recall",
            json={
                "query": "spare key",
                "org_id": str(org),
                "user_id": str(other),
                "with_rerank": False,
                "with_mmr": False,
            },
        )
        mine = await client.post(
            "/recall",
            json={
                "query": "spare key",
                "org_id": str(org),
                "user_id": str(owner),
                "with_rerank": False,
                "with_mmr": False,
            },
        )

    assert intruder.json() == []
    assert mine.json()


async def test_http_memorize_records_provenance(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    org = await _make_org(pool, "http_prov")
    actor = await _make_user(pool, org)
    ns = _ns("http_prov")
    run_id = uuid.uuid4()

    async with _api(pool, ns, monkeypatch) as client:
        stored = await client.post(
            "/memorize",
            json={
                "text": _FACT,
                "org_id": str(org),
                "user_id": str(actor),
                "provenance": {
                    "kind": "chat_turn",
                    "ingest_run_id": str(run_id),
                    "actor_user_id": str(actor),
                    "source_url": "https://example.invalid/thread/1",
                },
            },
        )

    body = stored.json()
    assert body["ingest_run_id"] == str(run_id)
    assert body["provenance"] == body["chunks"]

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT org_id, kind, producer, actor_user_id, source_url"
            " FROM provenance WHERE ingest_run_id = $1",
            run_id,
        )
        orphans = await conn.fetchval(
            "SELECT COUNT(*) FROM propositions"
            " WHERE namespace = $1"
            "   AND provenance_id = pgkg_unattributed_provenance()",
            ns,
        )

    assert row["org_id"] == org
    assert row["kind"] == "chat_turn"
    assert row["producer"] == "chunker"
    assert row["actor_user_id"] == actor
    assert row["source_url"] == "https://example.invalid/thread/1"
    assert orphans == 0


async def test_http_forget_records_a_reason(pool: asyncpg.Pool, monkeypatch) -> None:
    org = await _make_org(pool, "http_forget")
    ns = _ns("http_forget")

    async with _api(pool, ns, monkeypatch) as client:
        await client.post(
            "/memorize", json={"text": _FACT, "org_id": str(org)}
        )
        found = await client.post(
            "/recall",
            json={
                "query": "analytical engine",
                "org_id": str(org),
                "with_rerank": False,
                "with_mmr": False,
            },
        )
        target = found.json()[0]["proposition_id"]

        dropped = await client.post(
            "/forget",
            json={
                "proposition_id": target,
                "org_id": str(org),
                "reason": "source_updated",
            },
        )
        after = await client.post(
            "/recall",
            json={
                "query": "analytical engine",
                "org_id": str(org),
                "with_rerank": False,
                "with_mmr": False,
            },
        )

    assert dropped.status_code == 204
    assert after.json() == []

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT invalidated_at, invalidation_reason FROM propositions"
            " WHERE id = $1",
            uuid.UUID(target),
        )

    assert row["invalidated_at"] is not None
    assert row["invalidation_reason"] == "source_updated"


async def test_http_audit_endpoint_returns_withdrawn_belief(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    org = await _make_org(pool, "http_audit")
    ns = _ns("http_audit")

    async with _api(pool, ns, monkeypatch) as client:
        await client.post("/memorize", json={"text": _FACT, "org_id": str(org)})
        found = await client.post(
            "/recall",
            json={
                "query": "analytical engine",
                "org_id": str(org),
                "with_rerank": False,
                "with_mmr": False,
            },
        )
        target = found.json()[0]["proposition_id"]
        await client.post(
            "/forget",
            json={
                "proposition_id": target,
                "org_id": str(org),
                "reason": "contradicted",
            },
        )

        async with pool.acquire() as conn:
            recorded = await conn.fetchval(
                "SELECT recorded_at FROM propositions WHERE id = $1",
                uuid.UUID(target),
            )

        audited = await client.post(
            "/believed",
            json={
                "belief_at": (recorded + timedelta(milliseconds=1)).isoformat(),
                "org_id": str(org),
            },
        )
        stranger = await client.post(
            "/believed",
            json={
                "belief_at": (recorded + timedelta(milliseconds=1)).isoformat(),
                "org_id": str(DEFAULT_ORG_ID),
            },
        )

    assert audited.status_code == 200
    entry = next(
        r for r in audited.json() if r["proposition_id"] == target
    )
    assert entry["invalidation_reason"] == "contradicted"
    assert entry["provenance_id"]
    assert target not in [r["proposition_id"] for r in stranger.json()]


async def test_http_endpoints_still_work_without_any_scope_fields(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    """bench/common.py and the Makefile smoke target post the old bodies."""
    ns = _ns("http_compat")

    async with _api(pool, ns, monkeypatch) as client:
        health = await client.get("/health")
        stored = await client.post("/memorize", json={"text": _FACT})
        found = await client.post(
            "/recall", json={"query": "analytical engine", "k": 3}
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert stored.json()["chunks"] >= 1
    assert [r["text"] for r in found.json()]

    async with pool.acquire() as conn:
        orgs = [
            r["org_id"]
            for r in await conn.fetch(
                "SELECT DISTINCT org_id FROM propositions WHERE namespace = $1", ns
            )
        ]

    assert orgs == [DEFAULT_ORG_ID]


async def test_http_rejects_a_private_write_with_no_user(
    pool: asyncpg.Pool, monkeypatch
) -> None:
    ns = _ns("http_bad_scope")

    async with _api(pool, ns, monkeypatch) as client:
        response = await client.post(
            "/memorize", json={"text": _FACT, "visibility": "private"}
        )

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# 9. The CLI keeps working, and can name a tenant
# ---------------------------------------------------------------------------

async def test_cli_ingest_writes_into_the_named_org(
    pool: asyncpg.Pool, monkeypatch, capsys
) -> None:
    from pgkg import cli, db

    org = await _make_org(pool, "cli_org")
    ns = _ns("cli_org")

    @asynccontextmanager
    async def _pool_from_settings():
        yield _SharedPool(pool)

    class _Settings:
        default_namespace = ns
        extract_propositions = False

    monkeypatch.setattr(db, "pool_from_settings", _pool_from_settings)
    monkeypatch.setattr("pgkg.config.get_settings", lambda: _Settings())

    args = argparse.Namespace(
        path="-",
        chunks_only=True,
        org=str(org),
        user=None,
        collection=None,
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(_FACT))
    await cli.run_ingest(args)

    async with pool.acquire() as conn:
        orgs = [
            r["org_id"]
            for r in await conn.fetch(
                "SELECT DISTINCT org_id FROM propositions WHERE namespace = $1", ns
            )
        ]

    assert orgs == [org]
    assert json.loads(capsys.readouterr().out)["chunks"] >= 1


# ---------------------------------------------------------------------------
# 10. Two generations at once, and MMR in the primary space only
# ---------------------------------------------------------------------------

async def _insert_proposition(
    pool: asyncpg.Pool,
    *,
    namespace: str,
    org_id: uuid.UUID,
    text: str,
    embedding: list[float] | None,
) -> uuid.UUID:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO propositions (text, embedding, namespace, org_id)
            VALUES ($1, $2::halfvec, $3, $4)
            RETURNING id
            """,
            text,
            None if embedding is None else HalfVector(embedding),
            namespace,
            org_id,
        )


async def test_recall_fuses_a_second_generation(
    pool: asyncpg.Pool, dim: int
) -> None:
    """During a cutover a row reachable only in the new model space must still
    be retrieved, at its new-generation rank."""
    org = await _make_org(pool, "gen_org")
    ns = _ns("gen")
    query = "qzzx unrelated probe"
    second_dim = 768

    q_primary = _make_embed(dim)([query])[0]
    near = await _insert_proposition(
        pool, namespace=ns, org_id=org, text="Primary neighbour alpha", embedding=q_primary
    )
    far_vector = [0.0] * dim
    far_vector[(q_primary.index(1.0) + 7) % dim] = 1.0
    only_in_gen_two = await _insert_proposition(
        pool, namespace=ns, org_id=org, text="Secondary neighbour beta", embedding=far_vector
    )

    second_embed = _make_embed(second_dim)
    async with pool.acquire() as conn:
        generation = await conn.fetchval(
            """
            INSERT INTO embedder_generations
                (name, dim, storage_type, normalize, status)
            VALUES ('probe-768', $1, 'halfvec', TRUE, 'live')
            RETURNING id
            """,
            second_dim,
        )
        await conn.execute(
            "INSERT INTO org_embedders (org_id, generation_id, role)"
            " VALUES ($1, $2, 'secondary')",
            org,
            generation,
        )
        side_table = await conn.fetchval(
            "SELECT pgkg_create_generation_storage($1, 'prop')", generation
        )
        await conn.execute(
            f"INSERT INTO {side_table} (item_id, vec) VALUES ($1, $2::halfvec)",
            only_in_gen_two,
            HalfVector(second_embed([query])[0]),
        )

    scope = Scope(org_id=org)
    single = Memory(pool, namespace=ns, scope=scope, extract_propositions=False)
    dual = Memory(
        pool,
        namespace=ns,
        scope=scope,
        extract_propositions=False,
        embedders={"probe-768": second_embed},
    )

    without = await single.recall(query, with_rerank=False, with_mmr=False)
    with_second = await dual.recall(query, with_rerank=False, with_mmr=False)

    assert [r.proposition_id for r in without][0] == near
    assert [r.proposition_id for r in with_second][0] == only_in_gen_two


async def test_mmr_leaves_out_rows_with_no_primary_vector(
    pool: asyncpg.Pool, dim: int
) -> None:
    """Diversity is a cosine, so a row with no vector in the primary space
    cannot take part; the old fallback made it look identical to the query and
    therefore the first thing MMR chose."""
    org = await _make_org(pool, "mmr_org")
    ns = _ns("mmr")

    embedded = [
        await _insert_proposition(
            pool,
            namespace=ns,
            org_id=org,
            text=f"Kestrel fact number {i} about kestrels",
            embedding=_make_embed(dim)([f"kestrel {i}"])[0],
        )
        for i in range(2)
    ]
    vectorless = await _insert_proposition(
        pool,
        namespace=ns,
        org_id=org,
        text="Kestrel fact with no vector at all",
        embedding=None,
    )

    mem = Memory(
        pool, namespace=ns, scope=Scope(org_id=org), extract_propositions=False
    )
    results = await mem.recall("kestrel", k=2, with_rerank=False, with_mmr=True)

    assert len(results) == 2
    assert vectorless not in [r.proposition_id for r in results]
    assert set(r.proposition_id for r in results) == set(embedded)


async def test_http_health_reports_the_registry_embedding_space(
    pool: asyncpg.Pool, dim: int, monkeypatch
) -> None:
    """The width is a property of the registry, not of configuration, so the
    only honest way to report it is to read it."""
    async with _api(pool, _ns("health"), monkeypatch) as client:
        body = (await client.get("/health")).json()

    assert body["embedding"]["dim"] == dim
    assert "bge-m3" in body["embedding"]["generations"]
