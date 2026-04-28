"""Integration tests for pgkg SQL migrations and the pgkg_search hero function."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import asyncpg
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vec(dim: int = 1024, *, hot_index: int = 0, value: float = 1.0) -> list[float]:
    """Return a deterministic unit-ish vector with `value` at `hot_index`."""
    v = [0.0] * dim
    v[hot_index] = value
    return v


def _list_to_pg(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
_OLD_TS = datetime(2000, 1, 1, tzinfo=timezone.utc)
_BUMP_TS = datetime(2020, 1, 1, tzinfo=timezone.utc)


async def insert_entity(
    conn: asyncpg.Connection,
    *,
    name: str,
    entity_type: str = "concept",
    namespace: str = "default",
    embedding: list[float] | None = None,
) -> uuid.UUID:
    emb = embedding if embedding is not None else vec()
    emb_str = _list_to_pg(emb)
    row = await conn.fetchrow(
        f"""
        INSERT INTO entities (name, type, namespace, embedding)
        VALUES ($1, $2, $3, '{emb_str}')
        RETURNING id
        """,
        name,
        entity_type,
        namespace,
    )
    return row["id"]


async def insert_proposition(
    conn: asyncpg.Connection,
    *,
    text: str,
    namespace: str = "default",
    session_id: str | None = None,
    embedding: list[float] | None = None,
    subject_id: uuid.UUID | None = None,
    object_id: uuid.UUID | None = None,
    predicate: str | None = None,
    superseded_by: uuid.UUID | None = None,
    last_accessed_at: datetime | None = None,
    access_count: int = 0,
    confidence: float = 1.0,
) -> uuid.UUID:
    emb_expr = f"'{_list_to_pg(embedding)}'" if embedding is not None else "NULL"
    # Pass last_accessed_at as a parameter to avoid SQL injection / quoting issues
    if last_accessed_at is not None:
        laa_placeholder = "$10"
        extra_param: list = [last_accessed_at]
    else:
        laa_placeholder = "now()"
        extra_param = []

    row = await conn.fetchrow(
        f"""
        INSERT INTO propositions
            (text, namespace, session_id, embedding, subject_id, object_id,
             predicate, superseded_by, last_accessed_at, access_count, confidence)
        VALUES ($1, $2, $3, {emb_expr}, $4, $5, $6, $7, {laa_placeholder}, $8, $9)
        RETURNING id
        """,
        text,
        namespace,
        session_id,
        subject_id,
        object_id,
        predicate,
        superseded_by,
        access_count,
        confidence,
        *extra_param,
    )
    return row["id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_schema_creates_extensions(pool: asyncpg.Pool) -> None:
    """vector, pg_trgm, and pgcrypto extensions must be present."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT extname FROM pg_extension WHERE extname = ANY($1::text[])",
            ["vector", "pg_trgm", "pgcrypto"],
        )
    found = {r["extname"] for r in rows}
    assert "vector" in found
    assert "pg_trgm" in found
    assert "pgcrypto" in found


async def test_insert_proposition_generates_tsv(pool: asyncpg.Pool) -> None:
    """Inserting a proposition should auto-populate the tsv generated column."""
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(
            conn, text="The quick brown fox jumps over the lazy dog"
        )
        row = await conn.fetchrow(
            "SELECT tsv FROM propositions WHERE id = $1", prop_id
        )
    assert row is not None
    assert row["tsv"]  # non-empty tsvector


async def test_pgkg_search_keyword_only(pool: asyncpg.Pool) -> None:
    """Keyword-only search (NULL embedding) returns cat propositions, not dog ones."""
    async with pool.acquire() as conn:
        ns = f"kw_test_{uuid.uuid4().hex[:8]}"
        for i in range(5):
            await insert_proposition(conn, text=f"cats are fluffy animals number {i}", namespace=ns)
        for i in range(5):
            await insert_proposition(conn, text=f"dogs are loyal companions number {i}", namespace=ns)

        rows = await conn.fetch(
            """
            SELECT proposition_id, text
            FROM pgkg_search($1, NULL, 10, 20, $2)
            """,
            "cat",
            ns,
        )

    texts = [r["text"] for r in rows]
    assert len(texts) > 0
    assert all("cat" in t for t in texts), f"Expected only cat results, got: {texts}"


async def test_pgkg_search_vector_only(pool: asyncpg.Pool) -> None:
    """Vector-only search (NULL q_text) returns the proposition whose embedding matches."""
    async with pool.acquire() as conn:
        ns = f"vec_test_{uuid.uuid4().hex[:8]}"
        target_emb = vec(hot_index=5, value=1.0)
        other_emb = vec(hot_index=500, value=1.0)

        target_id = await insert_proposition(
            conn, text="target proposition", namespace=ns, embedding=target_emb
        )
        await insert_proposition(
            conn, text="unrelated proposition", namespace=ns, embedding=other_emb
        )

        query_emb = vec(hot_index=5, value=0.9999)
        rows = await conn.fetch(
            f"""
            SELECT proposition_id
            FROM pgkg_search(NULL, '{_list_to_pg(query_emb)}', 10, 20, $1)
            """,
            ns,
        )

    ids = [r["proposition_id"] for r in rows]
    assert target_id in ids, "Target proposition should be in vector search results"


async def test_pgkg_search_rrf_combines(pool: asyncpg.Pool) -> None:
    """Proposition matching both keyword and vector gets source_kind='both' and ranks top."""
    async with pool.acquire() as conn:
        ns = f"rrf_test_{uuid.uuid4().hex[:8]}"
        both_emb = vec(hot_index=10, value=1.0)
        other_emb = vec(hot_index=900, value=1.0)

        both_id = await insert_proposition(
            conn,
            text="the magnificent elephant roams the savanna",
            namespace=ns,
            embedding=both_emb,
        )
        # keyword-only (different embedding)
        await insert_proposition(
            conn,
            text="elephants are magnificent creatures",
            namespace=ns,
            embedding=other_emb,
        )
        # vector-only (similar embedding, different text)
        await insert_proposition(
            conn,
            text="unrelated zoological fact",
            namespace=ns,
            embedding=vec(hot_index=10, value=0.99),
        )

        query_emb = vec(hot_index=10, value=1.0)
        rows = await conn.fetch(
            f"""
            SELECT proposition_id, source_kind, adjusted_score
            FROM pgkg_search('elephant', '{_list_to_pg(query_emb)}', 10, 20, $1)
            ORDER BY adjusted_score DESC
            """,
            ns,
        )

    assert rows, "Expected results from combined search"
    top = rows[0]
    assert top["proposition_id"] == both_id, "Both-match proposition should rank first"
    assert top["source_kind"] == "both"


async def test_pgkg_search_session_filter(pool: asyncpg.Pool) -> None:
    """Session filter: session='A' returns session-A and NULL-session props; 'B' returns only NULL."""
    async with pool.acquire() as conn:
        ns = f"sess_test_{uuid.uuid4().hex[:8]}"
        emb_a = vec(hot_index=1, value=1.0)
        emb_null = vec(hot_index=2, value=1.0)
        emb_b = vec(hot_index=3, value=1.0)

        id_a = await insert_proposition(
            conn, text="session A proposition about rivers", namespace=ns,
            session_id="A", embedding=emb_a,
        )
        id_null = await insert_proposition(
            conn, text="global fact about rivers for everyone", namespace=ns,
            session_id=None, embedding=emb_null,
        )
        # session B — should NOT appear when querying with session='A'
        await insert_proposition(
            conn, text="session B proposition about rivers", namespace=ns,
            session_id="B", embedding=emb_b,
        )

        rows_a = await conn.fetch(
            """
            SELECT proposition_id
            FROM pgkg_search('rivers', NULL, 20, 50, $1, 'A')
            """,
            ns,
        )
        rows_b = await conn.fetch(
            """
            SELECT proposition_id
            FROM pgkg_search('rivers', NULL, 20, 50, $1, 'B')
            """,
            ns,
        )

    ids_a = {r["proposition_id"] for r in rows_a}
    ids_b = {r["proposition_id"] for r in rows_b}

    assert id_a in ids_a, "session-A prop should appear in session-A query"
    assert id_null in ids_a, "global (NULL session) prop should appear in session-A query"

    assert id_null in ids_b, "global (NULL session) prop should appear in session-B query"
    assert id_a not in ids_b, "session-A prop must NOT appear in session-B query"


async def test_pgkg_search_excludes_superseded(pool: asyncpg.Pool) -> None:
    """Superseded propositions must never appear in search results."""
    async with pool.acquire() as conn:
        ns = f"sup_test_{uuid.uuid4().hex[:8]}"
        emb = vec(hot_index=20, value=1.0)

        replacement_id = await insert_proposition(
            conn, text="updated fact about mountains", namespace=ns, embedding=emb
        )
        old_id = await insert_proposition(
            conn,
            text="old fact about mountains",
            namespace=ns,
            embedding=emb,
            superseded_by=replacement_id,
        )

        rows = await conn.fetch(
            f"""
            SELECT proposition_id
            FROM pgkg_search('mountains', '{_list_to_pg(emb)}', 20, 50, $1)
            """,
            ns,
        )

    ids = {r["proposition_id"] for r in rows}
    assert old_id not in ids, "Superseded proposition must be excluded"
    assert replacement_id in ids, "Replacement proposition should be present"


async def test_pgkg_search_recency_decay(pool: asyncpg.Pool) -> None:
    """With a very short half-life, an old proposition ranks below a recent one."""
    async with pool.acquire() as conn:
        ns = f"decay_test_{uuid.uuid4().hex[:8]}"
        emb = vec(hot_index=30, value=1.0)

        recent_id = await insert_proposition(
            conn,
            text="fact about oceans",
            namespace=ns,
            embedding=emb,
            # last_accessed_at defaults to now()
        )
        old_id = await insert_proposition(
            conn,
            text="fact about oceans",
            namespace=ns,
            embedding=emb,
            last_accessed_at=_OLD_TS,
        )

        # half_life = 1 day makes the 25-year-old record score extremely low
        rows = await conn.fetch(
            f"""
            SELECT proposition_id, adjusted_score
            FROM pgkg_search('oceans', '{_list_to_pg(emb)}', 20, 50, $1,
                             NULL, 1.0)
            ORDER BY adjusted_score DESC
            """,
            ns,
        )

    ids_ordered = [r["proposition_id"] for r in rows]
    assert recent_id in ids_ordered
    assert old_id in ids_ordered
    assert ids_ordered.index(recent_id) < ids_ordered.index(old_id), (
        "Recent proposition should rank above old one with short recency half-life"
    )


async def test_pgkg_search_graph_expansion(pool: asyncpg.Pool) -> None:
    """A proposition linked via an edge to a seed entity is returned as source_kind='graph'."""
    async with pool.acquire() as conn:
        ns = f"graph_test_{uuid.uuid4().hex[:8]}"

        entity_a = await insert_entity(conn, name="entity_alpha", namespace=ns)
        entity_b = await insert_entity(conn, name="entity_beta", namespace=ns)

        # Seed proposition: matches keyword and vector; subject is entity_a
        seed_emb = vec(hot_index=40, value=1.0)
        seed_id = await insert_proposition(
            conn,
            text="alpha entity primary fact",
            namespace=ns,
            embedding=seed_emb,
            subject_id=entity_a,
        )

        # Graph proposition: zero-vector embedding and unmatched text — only reachable via graph
        # Use a zero-like vector that is orthogonal to the query
        graph_emb = vec(hot_index=41, value=1.0)  # orthogonal to seed_emb (index 40)
        graph_id = await insert_proposition(
            conn,
            text="unrelated zymurgy trivia xyzzy qux",
            namespace=ns,
            embedding=graph_emb,
            subject_id=entity_b,
        )

        # Edge: entity_a -> entity_b, proposition_id = graph_id
        await conn.execute(
            """
            INSERT INTO edges (src_entity, dst_entity, relation, proposition_id)
            VALUES ($1, $2, 'related_to', $3)
            """,
            entity_a,
            entity_b,
            graph_id,
        )

        # Search with k_initial=1 so only seed_id enters kw/vec; graph_id must come via graph
        rows = await conn.fetch(
            f"""
            SELECT proposition_id, source_kind
            FROM pgkg_search('alpha', '{_list_to_pg(seed_emb)}', 50, 1, $1,
                             NULL, 30.0, TRUE)
            """,
            ns,
        )

    result_map = {r["proposition_id"]: r["source_kind"] for r in rows}
    assert graph_id in result_map, "Graph-expanded proposition should be in results"
    assert result_map[graph_id] == "graph", (
        f"Expected source_kind='graph', got '{result_map.get(graph_id)}'"
    )


async def test_pgkg_link_entity_creates_and_dedupes(pool: asyncpg.Pool) -> None:
    """First pgkg_link_entity call creates an entity; second similar call returns same id."""
    async with pool.acquire() as conn:
        ns = f"link_test_{uuid.uuid4().hex[:8]}"
        emb = vec(hot_index=50, value=1.0)
        emb_str = _list_to_pg(emb)

        # First call — creates
        id1 = await conn.fetchval(
            f"SELECT pgkg_link_entity($1, 'William Shakespeare', 'person', '{emb_str}')",
            ns,
        )
        assert id1 is not None

        # Second call — exact same namespace + name + type → same id
        id2 = await conn.fetchval(
            f"SELECT pgkg_link_entity($1, 'William Shakespeare', 'person', '{emb_str}')",
            ns,
        )
        assert id1 == id2, "Exact-match call must return same entity id"

        # Third call — same name prefix (high trigram similarity) + very similar embedding
        # "William Shakespear" has high trigram overlap with "William Shakespeare"
        similar_emb = vec(hot_index=50, value=0.9999)
        similar_emb_str = _list_to_pg(similar_emb)
        id3 = await conn.fetchval(
            f"SELECT pgkg_link_entity($1, 'William Shakespear', 'person', '{similar_emb_str}', 0.5)",
            ns,
        )
        # "William Shakespear" vs "William Shakespeare": high trigram overlap, embedding ~identical
        assert id3 == id1, "Near-duplicate entity (typo) should resolve to existing id"


async def test_pgkg_bump_access_updates_columns(pool: asyncpg.Pool) -> None:
    """pgkg_bump_access should increment access_count and update last_accessed_at."""
    async with pool.acquire() as conn:
        ns = f"bump_test_{uuid.uuid4().hex[:8]}"
        prop_id = await insert_proposition(
            conn,
            text="bumpable proposition",
            namespace=ns,
            last_accessed_at=_BUMP_TS,
            access_count=0,
        )

        before = await conn.fetchrow(
            "SELECT access_count, last_accessed_at FROM propositions WHERE id = $1",
            prop_id,
        )

        await conn.execute(
            "SELECT pgkg_bump_access($1::uuid[])", [str(prop_id)]
        )

        after = await conn.fetchrow(
            "SELECT access_count, last_accessed_at FROM propositions WHERE id = $1",
            prop_id,
        )

    assert after["access_count"] == before["access_count"] + 1
    assert after["last_accessed_at"] > before["last_accessed_at"]


async def test_pgkg_recompute_pagerank_runs(pool: asyncpg.Pool) -> None:
    """pgkg_recompute_pagerank should execute and populate entity_pagerank."""
    async with pool.acquire() as conn:
        ns = f"pr_test_{uuid.uuid4().hex[:8]}"

        id_a = await insert_entity(conn, name="pr_entity_a", namespace=ns)
        id_b = await insert_entity(conn, name="pr_entity_b", namespace=ns)
        id_c = await insert_entity(conn, name="pr_entity_c", namespace=ns)

        p_ab = await insert_proposition(conn, text="a to b", namespace=ns, subject_id=id_a, object_id=id_b)
        p_bc = await insert_proposition(conn, text="b to c", namespace=ns, subject_id=id_b, object_id=id_c)
        p_ac = await insert_proposition(conn, text="a to c", namespace=ns, subject_id=id_a, object_id=id_c)

        await conn.execute(
            """
            INSERT INTO edges (src_entity, dst_entity, relation, proposition_id) VALUES
                ($1, $2, 'links', $4),
                ($2, $3, 'links', $5),
                ($1, $3, 'links', $6)
            """,
            id_a, id_b, id_c, p_ab, p_bc, p_ac,
        )

        await conn.execute("SELECT pgkg_recompute_pagerank($1)", ns)

        rows = await conn.fetch(
            "SELECT entity_id, score FROM entity_pagerank WHERE entity_id = ANY($1::uuid[])",
            [id_a, id_b, id_c],
        )

    assert len(rows) == 3, "All 3 entities should have pagerank scores"
    scores = {r["entity_id"]: r["score"] for r in rows}

    # C receives links from both A and B → higher PageRank than A (receives none)
    assert scores[id_c] > scores[id_a], (
        f"C (receives 2 links) should outrank A (receives 0). Got C={scores[id_c]}, A={scores[id_a]}"
    )
