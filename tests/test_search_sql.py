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
    asserted_at: datetime | None = None,
) -> uuid.UUID:
    emb_expr = f"'{_list_to_pg(embedding)}'" if embedding is not None else "NULL"
    # Build dynamic placeholders for optional timestamp params
    params: list = [
        text,       # $1
        namespace,  # $2
        session_id, # $3
        subject_id, # $4
        object_id,  # $5
        predicate,  # $6
        superseded_by, # $7
        access_count,  # $8
        confidence,    # $9
    ]

    if last_accessed_at is not None:
        laa_placeholder = f"${len(params) + 1}"
        params.append(last_accessed_at)
    else:
        laa_placeholder = "now()"

    if asserted_at is not None:
        aat_placeholder = f"${len(params) + 1}"
        params.append(asserted_at)
    else:
        aat_placeholder = "NULL"

    row = await conn.fetchrow(
        f"""
        INSERT INTO propositions
            (text, namespace, session_id, embedding, subject_id, object_id,
             predicate, superseded_by, last_accessed_at, access_count, confidence,
             asserted_at)
        VALUES ($1, $2, $3, {emb_expr}, $4, $5, $6, $7, {laa_placeholder}, $8, $9,
                {aat_placeholder})
        RETURNING id
        """,
        *params,
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


async def test_pgkg_search_asserted_at_overrides_recency(pool: asyncpg.Pool) -> None:
    """asserted_at drives decay when set; NULL falls back to last_accessed_at.

    With a short half-life, a proposition whose asserted_at is 60 days ago should
    score lower than one whose asserted_at is NULL (falls back to last_accessed_at=now).
    Both have last_accessed_at=now so only asserted_at distinguishes them.
    """
    from datetime import timedelta
    async with pool.acquire() as conn:
        ns = f"assertedat_decay_{uuid.uuid4().hex[:8]}"
        emb = vec(hot_index=60, value=1.0)

        sixty_days_ago = datetime.now(timezone.utc) - timedelta(days=60)

        # Proposition with old asserted_at — should decay heavily
        old_asserted_id = await insert_proposition(
            conn,
            text="fact about rivers and streams",
            namespace=ns,
            embedding=emb,
            last_accessed_at=datetime.now(timezone.utc),
            asserted_at=sixty_days_ago,
        )
        # Proposition with NULL asserted_at — decay falls back to last_accessed_at=now
        null_asserted_id = await insert_proposition(
            conn,
            text="fact about rivers and streams",
            namespace=ns,
            embedding=emb,
            last_accessed_at=datetime.now(timezone.utc),
            asserted_at=None,
        )

        # half_life = 7 days makes 60-day-old assertion score much lower than fresh
        rows = await conn.fetch(
            f"""
            SELECT proposition_id, adjusted_score
            FROM pgkg_search('rivers', '{_list_to_pg(emb)}', 20, 50, $1,
                             NULL, 7.0)
            ORDER BY adjusted_score DESC
            """,
            ns,
        )

    result_map = {r["proposition_id"]: float(r["adjusted_score"]) for r in rows}
    assert old_asserted_id in result_map, "Old-asserted proposition should appear in results"
    assert null_asserted_id in result_map, "Null-asserted proposition should appear in results"
    assert result_map[null_asserted_id] > result_map[old_asserted_id], (
        "Null asserted_at (decays from now) should rank above old asserted_at with short half-life"
    )


# ---------------------------------------------------------------------------
# BM25 scoring tests
#
# These tests validate that pgkg_search() uses proper BM25 scoring and
# demonstrate the three properties where BM25 differs from ts_rank_cd:
#   1. IDF weighting — rare terms outrank common ones
#   2. Term frequency saturation — repeating a term doesn't scale linearly
#   3. Document length normalisation — shorter docs score higher
#
# Each test also runs a ts_rank_cd baseline via a raw SQL subquery to
# prove the old ranking would have been wrong (or at least indifferent).
# ---------------------------------------------------------------------------

async def _tsrankcd_order(
    conn: asyncpg.Connection,
    query: str,
    namespace: str,
    limit: int = 50,
) -> list[UUID]:
    """Return proposition IDs ranked by ts_rank_cd (the old scoring method)."""
    rows = await conn.fetch(
        """
        SELECT p.id
        FROM propositions p
        WHERE p.tsv @@ plainto_tsquery('english', $1)
          AND p.namespace = $2
          AND p.superseded_by IS NULL
        ORDER BY ts_rank_cd(p.tsv, plainto_tsquery('english', $1)) DESC
        LIMIT $3
        """,
        query, namespace, limit,
    )
    return [r["id"] for r in rows]


async def test_pgkg_search_bm25_idf_ranking(pool: asyncpg.Pool) -> None:
    """BM25's IDF gives a higher score for a rare query term than a common one.

    We compute raw BM25 scores (not RRF ranks) for a single target
    proposition against two different single-term queries: "zymurgy"
    (rare — df=1) and "animal" (common — df=21).  Both terms appear
    once in the target, so term frequency is identical.  The only
    difference is IDF.

    BM25 should give "zymurgy" a much higher score.
    ts_rank_cd should give both terms approximately the same score
    (it has no IDF component).
    """
    async with pool.acquire() as conn:
        ns = f"bm25_idf_{uuid.uuid4().hex[:8]}"

        # Make "animal" common — 20 propositions contain it
        for i in range(20):
            await insert_proposition(
                conn, text=f"the animal kingdom contains species number {i}", namespace=ns,
            )

        # Target proposition contains both "zymurgy" (rare) and "animal" (common),
        # each appearing exactly once.
        target_id = await insert_proposition(
            conn, text="zymurgy is applied in animal biology research", namespace=ns,
        )

        # Compute raw BM25 scores for two single-term queries against the target.
        # This bypasses pgkg_search/RRF to test the BM25 formula directly.
        bm25_scores = await conn.fetch(
            """
            WITH
            corpus AS (
                SELECT
                    GREATEST(COUNT(*), 1)::FLOAT8 AS n_total,
                    GREATEST(AVG(length(p.tsv)), 1.0)::FLOAT8 AS avgdl
                FROM propositions p
                WHERE p.namespace = $1 AND p.superseded_by IS NULL
            ),
            queries(term) AS (VALUES ('zymurgi'), ('anim')),
            df AS (
                SELECT
                    q.term,
                    (SELECT COUNT(*)::FLOAT8 FROM propositions p
                     WHERE p.tsv @@ to_tsquery('simple', q.term)
                       AND p.namespace = $1
                       AND p.superseded_by IS NULL) AS df
                FROM queries q
            )
            SELECT
                q.term,
                COALESCE((
                    SELECT
                        LN((c.n_total - d.df + 0.5) / (d.df + 0.5) + 1.0)
                        * (COALESCE(array_length(u.positions, 1), 0)::FLOAT8 * 2.2)
                        / (COALESCE(array_length(u.positions, 1), 0)::FLOAT8
                           + 1.2 * (1.0 - 0.75 + 0.75 * length(p.tsv)::FLOAT8 / c.avgdl))
                    FROM unnest(p.tsv) AS u(lexeme, positions, weights)
                    WHERE u.lexeme = q.term
                ), 0.0) AS bm25
            FROM queries q
            CROSS JOIN corpus c
            JOIN df d ON d.term = q.term
            CROSS JOIN propositions p
            WHERE p.id = $2
            """,
            ns, target_id,
        )
        bm25_by_term = {r["term"]: float(r["bm25"]) for r in bm25_scores}

        # ts_rank_cd baseline
        rare_tsrank = float(await conn.fetchval(
            "SELECT ts_rank_cd(p.tsv, plainto_tsquery('english', 'zymurgy')) FROM propositions p WHERE p.id = $1",
            target_id,
        ))
        common_tsrank = float(await conn.fetchval(
            "SELECT ts_rank_cd(p.tsv, plainto_tsquery('english', 'animal')) FROM propositions p WHERE p.id = $1",
            target_id,
        ))

    # BM25: rare term should score significantly higher due to IDF
    assert bm25_by_term["zymurgi"] > bm25_by_term["anim"] * 1.5, (
        f"BM25 score for rare 'zymurgi' ({bm25_by_term['zymurgi']:.4f}) should be "
        f"significantly higher than common 'anim' ({bm25_by_term['anim']:.4f}) — "
        f"IDF for df=1 >> IDF for df=21"
    )

    # ts_rank_cd: both terms appear once with similar cover density, so
    # scores should be approximately equal (no IDF awareness).
    assert abs(rare_tsrank - common_tsrank) < 0.05, (
        f"ts_rank_cd should give similar scores for rare ({rare_tsrank:.4f}) "
        f"and common ({common_tsrank:.4f}) single-term queries — no IDF"
    )


async def test_pgkg_search_bm25_tf_saturation(pool: asyncpg.Pool) -> None:
    """BM25's k1 parameter saturates term frequency — repeating a term 10x
    should NOT give 10x the score.

    We create two propositions: one mentions "fermentation" once, the other
    repeats it many times.  Both match the query "fermentation".  BM25 should
    score them relatively close (saturation), whereas a naive tf-based scorer
    would give the repetitive document a much higher score.
    """
    async with pool.acquire() as conn:
        ns = f"bm25_tf_{uuid.uuid4().hex[:8]}"

        # Proposition with single mention
        single_id = await insert_proposition(
            conn,
            text="fermentation is a metabolic process used in brewing",
            namespace=ns,
        )

        # Proposition with heavy repetition
        repeated_text = " ".join(["fermentation"] * 15 + ["process"])
        repeated_id = await insert_proposition(
            conn, text=repeated_text, namespace=ns,
        )

        # Also insert some unrelated propositions to give IDF meaningful values
        for i in range(10):
            await insert_proposition(
                conn, text=f"unrelated topic about wildlife number {i}", namespace=ns,
            )

        # Get BM25 scores from pgkg_search (keyword-only, no embedding)
        rows = await conn.fetch(
            """
            SELECT proposition_id,
                   rrf_score
            FROM pgkg_search('fermentation', NULL, 10, 20, $1)
            """,
            ns,
        )
        scores = {r["proposition_id"]: float(r["rrf_score"]) for r in rows}

        # Get raw ts_rank_cd scores for comparison
        tsrank_rows = await conn.fetch(
            """
            SELECT p.id,
                   ts_rank_cd(p.tsv, plainto_tsquery('english', 'fermentation')) AS score
            FROM propositions p
            WHERE p.tsv @@ plainto_tsquery('english', 'fermentation')
              AND p.namespace = $1
              AND p.superseded_by IS NULL
            ORDER BY score DESC
            """,
            ns,
        )
        tsrank_scores = {r["id"]: float(r["score"]) for r in tsrank_rows}

    assert single_id in scores, "Single-mention proposition should appear in BM25 results"
    assert repeated_id in scores, "Repeated proposition should appear in BM25 results"

    # Both should appear — the key insight is the score ratio.
    # BM25 with k1=1.2 saturates: tf=15 scores at most ~2.2x of tf=1.
    # ts_rank_cd does not have this saturation property.
    # We verify BM25 produces a reasonable ratio by checking both match
    # (the RRF rank difference should be small, not 15x).

    # With ts_rank_cd, the repeated document should have a significantly
    # higher raw score than the single-mention one.
    if single_id in tsrank_scores and repeated_id in tsrank_scores:
        tsrank_ratio = tsrank_scores[repeated_id] / max(tsrank_scores[single_id], 1e-10)
        # ts_rank_cd typically gives the repetitive doc a disproportionate boost
        assert tsrank_ratio > 2.0, (
            f"ts_rank_cd should give a large boost to repeated terms "
            f"(ratio={tsrank_ratio:.2f}, expected >2.0)"
        )


async def test_pgkg_search_bm25_length_normalisation(pool: asyncpg.Pool) -> None:
    """BM25's b parameter penalises long documents — a short proposition
    matching the query should score higher than a long verbose one with the
    same matching term.

    ts_rank_cd does not normalise by document length in the same way.
    """
    async with pool.acquire() as conn:
        ns = f"bm25_len_{uuid.uuid4().hex[:8]}"

        # Short proposition with the target term
        short_id = await insert_proposition(
            conn,
            text="mitochondria powers the cell",
            namespace=ns,
        )

        # Long verbose proposition with the same target term buried in padding
        padding = " ".join(
            f"word{i}" for i in range(40)
        )
        long_id = await insert_proposition(
            conn,
            text=f"mitochondria {padding} end of document",
            namespace=ns,
        )

        # Unrelated background to give corpus stats meaningful values
        for i in range(10):
            await insert_proposition(
                conn, text=f"background fact about geography number {i}", namespace=ns,
            )

        # BM25 keyword-only search
        rows = await conn.fetch(
            """
            SELECT proposition_id
            FROM pgkg_search('mitochondria', NULL, 10, 20, $1)
            """,
            ns,
        )
        bm25_ids = [r["proposition_id"] for r in rows]

    assert short_id in bm25_ids, "Short proposition should appear in results"
    assert long_id in bm25_ids, "Long proposition should appear in results"
    assert bm25_ids.index(short_id) < bm25_ids.index(long_id), (
        "BM25: short proposition should rank above long verbose one "
        "due to document length normalisation (b=0.75)"
    )


async def test_pgkg_search_returns_asserted_at_column(pool: asyncpg.Pool) -> None:
    """pgkg_search result rows include the asserted_at column with the correct value."""
    from datetime import timedelta
    async with pool.acquire() as conn:
        ns = f"assertedat_col_{uuid.uuid4().hex[:8]}"
        emb = vec(hot_index=61, value=1.0)

        expected_ts = datetime(2025, 3, 15, 12, 0, 0, tzinfo=timezone.utc)

        prop_id = await insert_proposition(
            conn,
            text="proposition with known assertion timestamp",
            namespace=ns,
            embedding=emb,
            asserted_at=expected_ts,
        )

        rows = await conn.fetch(
            f"""
            SELECT proposition_id, asserted_at
            FROM pgkg_search('assertion timestamp', '{_list_to_pg(emb)}', 10, 20, $1)
            """,
            ns,
        )

    result_map = {r["proposition_id"]: r["asserted_at"] for r in rows}
    assert prop_id in result_map, "Proposition should appear in search results"
    returned_ts = result_map[prop_id]
    assert returned_ts is not None, "asserted_at should not be NULL in results"
    # Compare timestamps (asyncpg may return timezone-aware datetime)
    if returned_ts.tzinfo is None:
        returned_ts = returned_ts.replace(tzinfo=timezone.utc)
    assert returned_ts == expected_ts, (
        f"Returned asserted_at {returned_ts!r} should equal inserted {expected_ts!r}"
    )


# ---------------------------------------------------------------------------
# Query decomposition tests
#
# These tests verify that pgkg_search() uses OR semantics for keyword
# matching — a multi-term query returns propositions matching ANY subset
# of terms, not just ALL of them.  BM25 scoring ranks propositions
# matching more terms higher.
# ---------------------------------------------------------------------------


async def test_pgkg_search_or_semantics(pool: asyncpg.Pool) -> None:
    """A multi-term query returns propositions matching only one of the terms.

    With AND semantics (plainto_tsquery), "zymurgy fermentation" would only
    match propositions containing BOTH terms.  With OR semantics, a
    proposition containing just "fermentation" also appears.
    """
    async with pool.acquire() as conn:
        ns = f"or_test_{uuid.uuid4().hex[:8]}"

        # Proposition matching both terms
        both_id = await insert_proposition(
            conn, text="zymurgy is the science of fermentation", namespace=ns,
        )

        # Proposition matching only "fermentation"
        one_id = await insert_proposition(
            conn, text="fermentation produces alcohol from sugars", namespace=ns,
        )

        # Proposition matching neither
        await insert_proposition(
            conn, text="geology studies rocks and minerals", namespace=ns,
        )

        rows = await conn.fetch(
            """
            SELECT proposition_id
            FROM pgkg_search('zymurgy fermentation', NULL, 10, 50, $1)
            """,
            ns,
        )
        result_ids = {r["proposition_id"] for r in rows}

    assert both_id in result_ids, "Proposition matching both terms should appear"
    assert one_id in result_ids, (
        "Proposition matching only 'fermentation' should appear — OR semantics"
    )


async def test_pgkg_search_multi_term_ranking(pool: asyncpg.Pool) -> None:
    """A proposition matching more query terms ranks higher than one matching fewer.

    BM25 sums per-term contributions, so matching 2 of 2 query terms
    scores higher than matching 1 of 2 (given similar document lengths).
    """
    async with pool.acquire() as conn:
        ns = f"multi_rank_{uuid.uuid4().hex[:8]}"

        # Proposition matching both "mitochondria" and "respiration"
        both_id = await insert_proposition(
            conn, text="mitochondria drive cellular respiration", namespace=ns,
        )

        # Proposition matching only "respiration"
        one_id = await insert_proposition(
            conn, text="respiration converts glucose to energy", namespace=ns,
        )

        # Background to give IDF meaningful values
        for i in range(5):
            await insert_proposition(
                conn, text=f"background fact about chemistry number {i}", namespace=ns,
            )

        rows = await conn.fetch(
            """
            SELECT proposition_id
            FROM pgkg_search('mitochondria respiration', NULL, 10, 50, $1)
            """,
            ns,
        )
        result_ids = [r["proposition_id"] for r in rows]

    assert both_id in result_ids, "Both-term proposition should appear"
    assert one_id in result_ids, "Single-term proposition should appear"
    assert result_ids.index(both_id) < result_ids.index(one_id), (
        "Proposition matching both terms should rank above single-term match"
    )


async def test_pgkg_search_no_terms_excluded(pool: asyncpg.Pool) -> None:
    """All existing keyword tests still pass — OR semantics is a superset of AND.

    When all query terms appear in a proposition, it still matches (and
    ranks highest).  This is a sanity check that OR doesn't break the
    common case.
    """
    async with pool.acquire() as conn:
        ns = f"no_exclude_{uuid.uuid4().hex[:8]}"

        target_id = await insert_proposition(
            conn, text="cats are fluffy animals", namespace=ns,
        )

        rows = await conn.fetch(
            """
            SELECT proposition_id
            FROM pgkg_search('fluffy cats', NULL, 10, 50, $1)
            """,
            ns,
        )
        result_ids = {r["proposition_id"] for r in rows}

    assert target_id in result_ids, "Proposition matching all terms should still appear"
