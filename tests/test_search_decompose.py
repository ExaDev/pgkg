"""Integration tests for the decomposed retrieval pipeline (migration 010).

Each candidate source, the fusion stage and the scoring profile are exercised
in isolation — that isolation is the point of the decomposition.  The
equivalence tests at the bottom instantiate the pre-refactor monolith from
migration 009 under a different name and assert that the composed
``pgkg_search()`` returns the same rows.
"""
from __future__ import annotations

import pathlib
import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest

MIGRATIONS_DIR = pathlib.Path(__file__).parent.parent / "migrations"
PRE_REFACTOR_MIGRATION = MIGRATIONS_DIR / "009_query_decomposition.sql"

DIM = 1024


def vec(*, hot_index: int = 0, value: float = 1.0, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[hot_index] = value
    return v


def pg_vec(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


def ns() -> str:
    return f"decomp_{uuid.uuid4().hex[:10]}"


async def insert_entity(
    conn: asyncpg.Connection, *, name: str, namespace: str
) -> uuid.UUID:
    return await conn.fetchval(
        f"""
        INSERT INTO entities (name, type, namespace, embedding)
        VALUES ($1, 'concept', $2, '{pg_vec(vec())}')
        RETURNING id
        """,
        name,
        namespace,
    )


async def insert_proposition(
    conn: asyncpg.Connection,
    *,
    text: str,
    namespace: str,
    session_id: str | None = None,
    embedding: list[float] | None = None,
    subject_id: uuid.UUID | None = None,
    object_id: uuid.UUID | None = None,
    predicate: str | None = None,
    access_count: int = 0,
    confidence: float = 1.0,
    asserted_at: datetime | None = None,
) -> uuid.UUID:
    emb_expr = f"'{pg_vec(embedding)}'" if embedding is not None else "NULL"
    return await conn.fetchval(
        f"""
        INSERT INTO propositions
            (text, namespace, session_id, embedding, subject_id, object_id,
             predicate, access_count, confidence, asserted_at)
        VALUES ($1, $2, $3, {emb_expr}, $4, $5, $6, $7, $8, $9)
        RETURNING id
        """,
        text,
        namespace,
        session_id,
        subject_id,
        object_id,
        predicate,
        access_count,
        confidence,
        asserted_at,
    )


async def insert_edge(
    conn: asyncpg.Connection,
    *,
    src: uuid.UUID,
    dst: uuid.UUID,
    proposition_id: uuid.UUID,
    relation: str = "related_to",
) -> None:
    await conn.execute(
        """
        INSERT INTO edges (src_entity, dst_entity, relation, proposition_id)
        VALUES ($1, $2, $3, $4)
        """,
        src,
        dst,
        relation,
        proposition_id,
    )


def candidate_array(rows: list[tuple[uuid.UUID, str, int, float]]) -> str:
    """Render a pgkg_candidate[] literal for direct function calls."""
    if not rows:
        return "'{}'::pgkg_candidate[]"
    parts = [
        f"ROW('{item_id}'::uuid, '{kind}', {rank}, {score}::REAL)" for item_id, kind, rank, score in rows
    ]
    return "ARRAY[" + ", ".join(parts) + "]::pgkg_candidate[]"


# ---------------------------------------------------------------------------
# pgkg_bm25_candidates
# ---------------------------------------------------------------------------


async def test_bm25_candidates_rank_rare_terms_first(pool: asyncpg.Pool) -> None:
    """Rarer query terms carry more IDF, so their document takes rank 1."""
    namespace = ns()
    async with pool.acquire() as conn:
        rare_id = await insert_proposition(
            conn, text="zymurgy fermentation notes", namespace=namespace
        )
        for i in range(10):
            await insert_proposition(
                conn, text=f"elephant sighting number {i}", namespace=namespace
            )

        rows = await conn.fetch(
            "SELECT * FROM pgkg_bm25_candidates('zymurgy elephant', $1)", namespace
        )

    assert rows, "keyword arm must return candidates"
    assert {r["kind"] for r in rows} == {"kw"}
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert rows[0]["item_id"] == rare_id
    assert rows[0]["raw_score"] > rows[1]["raw_score"]


async def test_bm25_candidates_empty_query_returns_no_rows(pool: asyncpg.Pool) -> None:
    """An empty or stop-word-only query short-circuits instead of erroring."""
    namespace = ns()
    async with pool.acquire() as conn:
        await insert_proposition(conn, text="the cat sat", namespace=namespace)

        for q in ("", "the and of", None):
            rows = await conn.fetch(
                "SELECT * FROM pgkg_bm25_candidates($1, $2)", q, namespace
            )
            assert rows == [], f"query {q!r} should produce no candidates"


async def test_bm25_candidates_truncates_to_k_initial(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        for i in range(5):
            await insert_proposition(
                conn, text=f"mitochondria observation {i}", namespace=namespace
            )

        rows = await conn.fetch(
            "SELECT * FROM pgkg_bm25_candidates('mitochondria', $1, NULL, 2)",
            namespace,
        )

    assert len(rows) == 2
    assert [r["rank"] for r in rows] == [1, 2]


async def test_bm25_candidates_session_scope(pool: asyncpg.Pool) -> None:
    """Session-scoped rows are invisible to other sessions; NULL session is shared."""
    namespace = ns()
    async with pool.acquire() as conn:
        a_id = await insert_proposition(
            conn, text="quokka in session a", namespace=namespace, session_id="A"
        )
        shared_id = await insert_proposition(
            conn, text="quokka with no session", namespace=namespace
        )

        rows = await conn.fetch(
            "SELECT * FROM pgkg_bm25_candidates('quokka', $1, 'B')", namespace
        )

    ids = {r["item_id"] for r in rows}
    assert shared_id in ids
    assert a_id not in ids


# ---------------------------------------------------------------------------
# pgkg_vector_candidates
# ---------------------------------------------------------------------------


async def test_vector_candidates_rank_by_cosine_distance(pool: asyncpg.Pool) -> None:
    namespace = ns()
    target = vec(hot_index=3)
    async with pool.acquire() as conn:
        near_id = await insert_proposition(
            conn, text="near", namespace=namespace, embedding=target
        )
        far_id = await insert_proposition(
            conn, text="far", namespace=namespace, embedding=vec(hot_index=7)
        )

        rows = await conn.fetch(
            f"""
            SELECT * FROM pgkg_vector_candidates('{pg_vec(target)}', $1)
            """,
            namespace,
        )

    assert [r["item_id"] for r in rows] == [near_id, far_id]
    assert {r["kind"] for r in rows} == {"vec"}
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[0]["raw_score"] > rows[1]["raw_score"]


async def test_vector_candidates_null_query_returns_no_rows(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        await insert_proposition(
            conn, text="anything", namespace=namespace, embedding=vec()
        )

        rows = await conn.fetch(
            "SELECT * FROM pgkg_vector_candidates(NULL, $1)", namespace
        )

    assert rows == []


# ---------------------------------------------------------------------------
# pgkg_graph_candidates
# ---------------------------------------------------------------------------


async def test_graph_candidates_expand_one_hop_from_seeds(pool: asyncpg.Pool) -> None:
    """Neighbours of the seeds' entities come back as 'graph' at the seed floor score."""
    namespace = ns()
    async with pool.acquire() as conn:
        hub = await insert_entity(conn, name="hub", namespace=namespace)
        other = await insert_entity(conn, name="other", namespace=namespace)

        seed_id = await insert_proposition(
            conn, text="seed fact", namespace=namespace, subject_id=hub
        )
        neighbour_id = await insert_proposition(
            conn, text="neighbour fact", namespace=namespace, subject_id=other
        )
        await insert_edge(conn, src=hub, dst=other, proposition_id=neighbour_id)

        seeds = candidate_array([(seed_id, "fused", 0, 0.04)])
        rows = await conn.fetch(
            f"SELECT * FROM pgkg_graph_candidates({seeds}, $1)", namespace
        )

    assert [r["item_id"] for r in rows] == [neighbour_id]
    assert rows[0]["kind"] == "graph"
    assert rows[0]["rank"] == 1
    assert rows[0]["raw_score"] == pytest.approx(0.04, rel=1e-6)


async def test_graph_candidates_empty_seeds_return_no_rows(pool: asyncpg.Pool) -> None:
    """No seeds means no expansion — this is how expand_graph=FALSE is expressed."""
    namespace = ns()
    async with pool.acquire() as conn:
        hub = await insert_entity(conn, name="hub", namespace=namespace)
        other = await insert_entity(conn, name="other", namespace=namespace)
        neighbour_id = await insert_proposition(
            conn, text="neighbour fact", namespace=namespace, subject_id=other
        )
        await insert_edge(conn, src=hub, dst=other, proposition_id=neighbour_id)

        rows = await conn.fetch(
            f"SELECT * FROM pgkg_graph_candidates({candidate_array([])}, $1)",
            namespace,
        )

    assert rows == []


async def test_graph_candidates_exclude_seed_propositions(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        hub = await insert_entity(conn, name="hub", namespace=namespace)
        other = await insert_entity(conn, name="other", namespace=namespace)

        seed_id = await insert_proposition(
            conn, text="seed fact", namespace=namespace, subject_id=hub
        )
        await insert_edge(conn, src=hub, dst=other, proposition_id=seed_id)

        seeds = candidate_array([(seed_id, "fused", 0, 0.02)])
        rows = await conn.fetch(
            f"SELECT * FROM pgkg_graph_candidates({seeds}, $1)", namespace
        )

    assert rows == []


async def test_graph_candidates_cap_fanout_per_seed_entity(pool: asyncpg.Pool) -> None:
    """A hub entity cannot consume the whole neighbour budget (spec D7).

    The old global LIMIT let one high-degree entity crowd every other seed out
    of the results.  The cap is now applied per seed entity.
    """
    namespace = ns()
    async with pool.acquire() as conn:
        hub = await insert_entity(conn, name="hub", namespace=namespace)
        quiet = await insert_entity(conn, name="quiet", namespace=namespace)

        hub_seed = await insert_proposition(
            conn, text="hub seed", namespace=namespace, subject_id=hub
        )
        quiet_seed = await insert_proposition(
            conn, text="quiet seed", namespace=namespace, subject_id=quiet
        )

        hub_neighbours = []
        for i in range(12):
            target = await insert_entity(conn, name=f"hub_nbr_{i}", namespace=namespace)
            prop_id = await insert_proposition(
                conn, text=f"hub neighbour {i}", namespace=namespace, subject_id=target
            )
            await insert_edge(conn, src=hub, dst=target, proposition_id=prop_id)
            hub_neighbours.append(prop_id)

        quiet_neighbours = []
        for i in range(2):
            target = await insert_entity(
                conn, name=f"quiet_nbr_{i}", namespace=namespace
            )
            prop_id = await insert_proposition(
                conn, text=f"quiet neighbour {i}", namespace=namespace, subject_id=target
            )
            await insert_edge(conn, src=quiet, dst=target, proposition_id=prop_id)
            quiet_neighbours.append(prop_id)

        seeds = candidate_array(
            [(hub_seed, "fused", 0, 0.04), (quiet_seed, "fused", 0, 0.02)]
        )
        rows = await conn.fetch(
            f"SELECT * FROM pgkg_graph_candidates({seeds}, $1, 20, 3, 100)", namespace
        )

    returned = {r["item_id"] for r in rows}
    from_hub = returned & set(hub_neighbours)
    from_quiet = returned & set(quiet_neighbours)

    assert len(from_hub) == 3, "hub fan-out must be capped at k_per_seed"
    assert from_quiet == set(quiet_neighbours), (
        "the quiet seed keeps its own fan-out budget"
    )
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))


async def test_graph_candidates_respect_namespace(pool: asyncpg.Pool) -> None:
    namespace = ns()
    foreign = ns()
    async with pool.acquire() as conn:
        hub = await insert_entity(conn, name="hub", namespace=namespace)
        other = await insert_entity(conn, name="other", namespace=namespace)

        seed_id = await insert_proposition(
            conn, text="seed fact", namespace=namespace, subject_id=hub
        )
        foreign_id = await insert_proposition(
            conn, text="foreign fact", namespace=foreign, subject_id=other
        )
        await insert_edge(conn, src=hub, dst=other, proposition_id=foreign_id)

        seeds = candidate_array([(seed_id, "fused", 0, 0.02)])
        rows = await conn.fetch(
            f"SELECT * FROM pgkg_graph_candidates({seeds}, $1)", namespace
        )

    assert rows == []


# ---------------------------------------------------------------------------
# pgkg_fuse
# ---------------------------------------------------------------------------


async def test_fuse_sums_reciprocal_ranks_across_sources(pool: asyncpg.Pool) -> None:
    item = uuid.uuid4()
    candidates = candidate_array([(item, "kw", 1, 9.9), (item, "vec", 3, 0.7)])
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT * FROM pgkg_fuse({candidates})")

    assert row["item_id"] == item
    assert row["fused_score"] == pytest.approx(1 / 61 + 1 / 63, rel=1e-6)
    assert row["in_kw"] is True
    assert row["in_vec"] is True
    assert row["in_graph"] is False


async def test_fuse_applies_per_source_weights(pool: asyncpg.Pool) -> None:
    kw_only = uuid.uuid4()
    vec_only = uuid.uuid4()
    candidates = candidate_array([(kw_only, "kw", 1, 0.0), (vec_only, "vec", 1, 0.0)])
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM pgkg_fuse({candidates}, 60, 2.0, 0.25)"
        )

    scores = {r["item_id"]: r["fused_score"] for r in rows}
    assert scores[kw_only] == pytest.approx(2.0 / 61, rel=1e-6)
    assert scores[vec_only] == pytest.approx(0.25 / 61, rel=1e-6)


async def test_fuse_scales_graph_candidates_by_w_graph(pool: asyncpg.Pool) -> None:
    """Graph candidates carry a propagated score, weighted by w_graph."""
    item = uuid.uuid4()
    candidates = candidate_array([(item, "graph", 1, 0.02)])
    async with pool.acquire() as conn:
        default_score = await conn.fetchval(
            f"SELECT fused_score FROM pgkg_fuse({candidates})"
        )
        weighted_score = await conn.fetchval(
            f"SELECT fused_score FROM pgkg_fuse({candidates}, 60, 1.0, 1.0, 0.25)"
        )

    assert default_score == pytest.approx(0.01, rel=1e-6)
    assert weighted_score == pytest.approx(0.005, rel=1e-6)


async def test_fuse_rrf_k_flattens_rank_differences(pool: asyncpg.Pool) -> None:
    top = uuid.uuid4()
    tail = uuid.uuid4()
    candidates = candidate_array([(top, "kw", 1, 0.0), (tail, "kw", 50, 0.0)])
    async with pool.acquire() as conn:
        sharp = {
            r["item_id"]: r["fused_score"]
            for r in await conn.fetch(f"SELECT * FROM pgkg_fuse({candidates}, 1)")
        }
        flat = {
            r["item_id"]: r["fused_score"]
            for r in await conn.fetch(f"SELECT * FROM pgkg_fuse({candidates}, 1000)")
        }

    assert sharp[top] / sharp[tail] > flat[top] / flat[tail]


async def test_fuse_empty_candidate_set_returns_no_rows(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"SELECT * FROM pgkg_fuse({candidate_array([])})")

    assert rows == []


# ---------------------------------------------------------------------------
# pgkg_apply_profile
# ---------------------------------------------------------------------------


async def test_apply_profile_decays_by_recency(pool: asyncpg.Pool) -> None:
    namespace = ns()
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        fresh_id = await insert_proposition(
            conn, text="fresh", namespace=namespace, asserted_at=now
        )
        stale_id = await insert_proposition(
            conn,
            text="stale",
            namespace=namespace,
            asserted_at=now - timedelta(days=60),
        )

        candidates = candidate_array(
            [(fresh_id, "fused", 0, 0.02), (stale_id, "fused", 0, 0.02)]
        )
        rows = await conn.fetch(
            f"SELECT * FROM pgkg_apply_profile({candidates}, 7.0)"
        )

    scores = {r["item_id"]: r["adjusted_score"] for r in rows}
    assert scores[fresh_id] > scores[stale_id]


async def test_apply_profile_boosts_frequently_accessed(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        cold_id = await insert_proposition(
            conn, text="cold", namespace=namespace, access_count=0
        )
        hot_id = await insert_proposition(
            conn, text="hot", namespace=namespace, access_count=99
        )

        candidates = candidate_array(
            [(cold_id, "fused", 0, 0.02), (hot_id, "fused", 0, 0.02)]
        )
        rows = await conn.fetch(f"SELECT * FROM pgkg_apply_profile({candidates})")

    scores = {r["item_id"]: r["adjusted_score"] for r in rows}
    assert scores[hot_id] > scores[cold_id]


async def test_apply_profile_scales_by_confidence(pool: asyncpg.Pool) -> None:
    namespace = ns()
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        certain_id = await insert_proposition(
            conn, text="certain", namespace=namespace, confidence=1.0, asserted_at=now
        )
        doubtful_id = await insert_proposition(
            conn, text="doubtful", namespace=namespace, confidence=0.5, asserted_at=now
        )

        candidates = candidate_array(
            [(certain_id, "fused", 0, 0.02), (doubtful_id, "fused", 0, 0.02)]
        )
        rows = await conn.fetch(f"SELECT * FROM pgkg_apply_profile({candidates})")

    scores = {r["item_id"]: r["adjusted_score"] for r in rows}
    assert scores[doubtful_id] == pytest.approx(scores[certain_id] * 0.5, rel=1e-4)


async def test_apply_profile_clamps_extreme_age(pool: asyncpg.Pool) -> None:
    """A very old row with a tiny half-life underflows to zero, not to an error."""
    namespace = ns()
    async with pool.acquire() as conn:
        ancient_id = await insert_proposition(
            conn,
            text="ancient",
            namespace=namespace,
            asserted_at=datetime(1970, 1, 2, tzinfo=timezone.utc),
        )

        candidates = candidate_array([(ancient_id, "fused", 0, 0.02)])
        score = await conn.fetchval(
            f"SELECT adjusted_score FROM pgkg_apply_profile({candidates}, 0.001)"
        )

    assert score == pytest.approx(0.0, abs=1e-30)


# ---------------------------------------------------------------------------
# Behavioural equivalence with the pre-refactor monolith
# ---------------------------------------------------------------------------


async def _install_pre_refactor_search(conn: asyncpg.Connection) -> None:
    """Instantiate migration 009's monolith as pgkg_search_pre010()."""
    source = PRE_REFACTOR_MIGRATION.read_text()
    assert "FUNCTION pgkg_search(" in source
    await conn.execute(source.replace("FUNCTION pgkg_search(", "FUNCTION pgkg_search_pre010("))


def _comparable(rows: list[asyncpg.Record]) -> list[tuple]:
    return [
        (
            r["proposition_id"],
            r["text"],
            r["source_kind"],
            r["chunk_id"],
            r["subject_id"],
            r["predicate"],
            r["object_id"],
            r["asserted_at"],
        )
        for r in rows
    ]


async def _assert_equivalent(conn: asyncpg.Connection, call_args: str, *params) -> None:
    new_rows = await conn.fetch(f"SELECT * FROM pgkg_search({call_args})", *params)
    old_rows = await conn.fetch(
        f"SELECT * FROM pgkg_search_pre010({call_args})", *params
    )

    # Two empty result sets are equal, so an equivalence assertion over them
    # says nothing about either implementation.
    assert new_rows, "the query under comparison must retrieve something"
    assert _comparable(new_rows) == _comparable(old_rows)
    old_scores = {r["proposition_id"]: r["rrf_score"] for r in old_rows}
    for row in new_rows:
        assert row["rrf_score"] == pytest.approx(
            old_scores[row["proposition_id"]], rel=1e-6
        )


async def test_search_equivalent_keyword_only(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        await _install_pre_refactor_search(conn)
        await insert_proposition(conn, text="the cat sat on the mat", namespace=namespace)
        await insert_proposition(conn, text="a cat and a dog", namespace=namespace)
        await insert_proposition(conn, text="quantum chromodynamics", namespace=namespace)

        await _assert_equivalent(conn, "'cat mat', NULL, 10, 20, $1", namespace)


async def test_search_equivalent_vector_only(pool: asyncpg.Pool) -> None:
    namespace = ns()
    query_emb = vec(hot_index=11)
    async with pool.acquire() as conn:
        await _install_pre_refactor_search(conn)
        for i in range(4):
            await insert_proposition(
                conn,
                text=f"vector row {i}",
                namespace=namespace,
                embedding=vec(hot_index=11 + i),
            )

        await _assert_equivalent(
            conn, f"NULL, '{pg_vec(query_emb)}', 10, 20, $1", namespace
        )


async def test_search_equivalent_fused(pool: asyncpg.Pool) -> None:
    namespace = ns()
    query_emb = vec(hot_index=21)
    async with pool.acquire() as conn:
        await _install_pre_refactor_search(conn)
        await insert_proposition(
            conn,
            text="badger census results",
            namespace=namespace,
            embedding=vec(hot_index=21),
            access_count=3,
        )
        await insert_proposition(
            conn,
            text="badger migration paths",
            namespace=namespace,
            embedding=vec(hot_index=25),
        )
        await insert_proposition(
            conn,
            text="unrelated pottery glaze",
            namespace=namespace,
            embedding=vec(hot_index=22),
        )

        await _assert_equivalent(
            conn, f"'badger census', '{pg_vec(query_emb)}', 10, 20, $1", namespace
        )


async def test_search_equivalent_graph_expanded(pool: asyncpg.Pool) -> None:
    namespace = ns()
    seed_emb = vec(hot_index=40)
    async with pool.acquire() as conn:
        await _install_pre_refactor_search(conn)
        entity_a = await insert_entity(conn, name="alpha", namespace=namespace)
        entity_b = await insert_entity(conn, name="beta", namespace=namespace)

        await insert_proposition(
            conn,
            text="alpha entity primary fact",
            namespace=namespace,
            embedding=seed_emb,
            subject_id=entity_a,
        )
        graph_id = await insert_proposition(
            conn,
            text="unrelated zymurgy trivia xyzzy",
            namespace=namespace,
            embedding=vec(hot_index=41),
            subject_id=entity_b,
        )
        await insert_edge(
            conn, src=entity_a, dst=entity_b, proposition_id=graph_id
        )

        await _assert_equivalent(
            conn, f"'alpha', '{pg_vec(seed_emb)}', 50, 1, $1, NULL, 30.0, TRUE", namespace
        )


async def test_search_suppresses_graph_rows_when_expansion_disabled(
    pool: asyncpg.Pool,
) -> None:
    namespace = ns()
    seed_emb = vec(hot_index=50)
    async with pool.acquire() as conn:
        entity_a = await insert_entity(conn, name="alpha", namespace=namespace)
        entity_b = await insert_entity(conn, name="beta", namespace=namespace)

        await insert_proposition(
            conn,
            text="alpha entity primary fact",
            namespace=namespace,
            embedding=seed_emb,
            subject_id=entity_a,
        )
        graph_id = await insert_proposition(
            conn,
            text="unrelated zymurgy trivia xyzzy",
            namespace=namespace,
            embedding=vec(hot_index=51),
            subject_id=entity_b,
        )
        await insert_edge(conn, src=entity_a, dst=entity_b, proposition_id=graph_id)

        expanded = await conn.fetch(
            f"""
            SELECT proposition_id FROM pgkg_search(
                'alpha', '{pg_vec(seed_emb)}', 50, 1, $1, NULL, 30.0, TRUE)
            """,
            namespace,
        )
        suppressed = await conn.fetch(
            f"""
            SELECT proposition_id FROM pgkg_search(
                'alpha', '{pg_vec(seed_emb)}', 50, 1, $1, NULL, 30.0, FALSE)
            """,
            namespace,
        )

    assert graph_id in {r["proposition_id"] for r in expanded}
    assert graph_id not in {r["proposition_id"] for r in suppressed}


async def test_search_returns_each_proposition_once(pool: asyncpg.Pool) -> None:
    """A proposition reachable by several edges is one result row, not several.

    The pre-refactor UNION ALL emitted one row per traversed edge; fusion now
    groups by item, so multiply-reachable neighbours are deduplicated.
    """
    namespace = ns()
    seed_emb = vec(hot_index=60)
    async with pool.acquire() as conn:
        entity_a = await insert_entity(conn, name="alpha", namespace=namespace)
        entity_b = await insert_entity(conn, name="beta", namespace=namespace)
        entity_c = await insert_entity(conn, name="gamma", namespace=namespace)

        await insert_proposition(
            conn,
            text="alpha beta gamma primary fact",
            namespace=namespace,
            embedding=seed_emb,
            subject_id=entity_a,
            object_id=entity_b,
        )
        graph_id = await insert_proposition(
            conn,
            text="unrelated zymurgy trivia xyzzy",
            namespace=namespace,
            embedding=vec(hot_index=61),
            subject_id=entity_c,
        )
        await insert_edge(conn, src=entity_a, dst=entity_c, proposition_id=graph_id)
        await insert_edge(
            conn,
            src=entity_b,
            dst=entity_c,
            proposition_id=graph_id,
            relation="also_related",
        )

        rows = await conn.fetch(
            f"""
            SELECT proposition_id FROM pgkg_search(
                'alpha', '{pg_vec(seed_emb)}', 50, 1, $1, NULL, 30.0, TRUE)
            """,
            namespace,
        )

    ids = [r["proposition_id"] for r in rows]
    assert graph_id in ids
    assert len(ids) == len(set(ids))


async def test_search_truncates_to_k_retrieve(pool: asyncpg.Pool) -> None:
    namespace = ns()
    async with pool.acquire() as conn:
        for i in range(6):
            await insert_proposition(
                conn, text=f"kingfisher note {i}", namespace=namespace
            )

        rows = await conn.fetch(
            "SELECT * FROM pgkg_search('kingfisher', NULL, 2, 20, $1)", namespace
        )

    assert len(rows) == 2
