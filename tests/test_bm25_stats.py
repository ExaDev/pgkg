"""Materialised BM25 corpus statistics.

The keyword arm used to derive `avgdl` from an aggregate over every active
proposition in the namespace and document frequency from one correlated
COUNT(*) per query lexeme.  These tests pin the behaviour of the tables that
replace those scans: that they track inserts, deletes and supersession
exactly, that they are not rewritten by the recall-path access-count bump,
that a full refresh agrees with the incrementally maintained values, that
retrieval still ranks sanely when the statistics are missing entirely, and
that a keyword search no longer visits `propositions` more than once.
"""
from __future__ import annotations

import json
import uuid

import asyncpg
import pytest


async def insert_proposition(
    conn: asyncpg.Connection,
    *,
    text: str,
    namespace: str,
    session_id: str | None = None,
    chunk_id: uuid.UUID | None = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO propositions (text, namespace, session_id, chunk_id)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """,
        text, namespace, session_id, chunk_id,
    )


async def corpus_stats(conn: asyncpg.Connection, namespace: str) -> asyncpg.Record | None:
    return await conn.fetchrow(
        "SELECT n_total, total_len, avgdl, updated_at FROM corpus_stats WHERE namespace = $1",
        namespace,
    )


async def lexeme_df(conn: asyncpg.Connection, namespace: str) -> dict[str, int]:
    rows = await conn.fetch(
        "SELECT lexeme, df FROM lexeme_df WHERE namespace = $1", namespace,
    )
    return {r["lexeme"]: r["df"] for r in rows}


def _ns(tag: str) -> str:
    return f"{tag}_{uuid.uuid4().hex[:8]}"


def _relation_counts(plan_json: str) -> dict[str, int]:
    """Count how many plan nodes scan each relation, over the whole plan tree."""
    counts: dict[str, int] = {}

    def walk(node: dict) -> None:
        name = node.get("Relation Name")
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
        for key in ("Plans", "Subplans"):
            for child in node.get(key, []):
                walk(child)

    for entry in json.loads(plan_json):
        walk(entry["Plan"])
    return counts


# ---------------------------------------------------------------------------
# Stored document length
# ---------------------------------------------------------------------------

async def test_stored_doc_len_equals_tsvector_length(pool: asyncpg.Pool) -> None:
    """doc_len is the tsvector's distinct-lexeme count, stored at write time.

    The BM25 length-normalisation term needs it once per candidate row per
    query; recomputing length(tsv) was pure waste.
    """
    ns = _ns("doclen")
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(
            conn, text="the cat sat on the mat with another cat", namespace=ns,
        )
        row = await conn.fetchrow(
            "SELECT doc_len, length(tsv) AS tsv_len FROM propositions WHERE id = $1",
            prop_id,
        )

    assert row["doc_len"] == row["tsv_len"]
    assert row["doc_len"] > 0


async def test_stored_doc_len_follows_text_updates(pool: asyncpg.Pool) -> None:
    """doc_len is generated, so it cannot drift from the text it describes."""
    ns = _ns("doclen_upd")
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(conn, text="alpha", namespace=ns)
        before = await conn.fetchval(
            "SELECT doc_len FROM propositions WHERE id = $1", prop_id,
        )

        await conn.execute(
            "UPDATE propositions SET text = $2 WHERE id = $1",
            prop_id, "alpha beta gamma delta epsilon",
        )
        after = await conn.fetchrow(
            "SELECT doc_len, length(tsv) AS tsv_len FROM propositions WHERE id = $1",
            prop_id,
        )

    assert before == 1
    assert after["doc_len"] == after["tsv_len"] > before


# ---------------------------------------------------------------------------
# Incremental maintenance
# ---------------------------------------------------------------------------

async def test_corpus_stats_track_inserts(pool: asyncpg.Pool) -> None:
    """A multi-row insert lands as one exact (n_total, total_len) contribution.

    avgdl is derived from the stored sum rather than materialised directly,
    because a mean cannot be maintained incrementally and a sum can.
    """
    ns = _ns("stats_ins")
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO propositions (text, namespace)
            VALUES ('cat sat', $1), ('dog barked loudly today', $1)
            """,
            ns,
        )
        stats = await corpus_stats(conn, ns)
        lengths = await conn.fetchval(
            "SELECT SUM(doc_len) FROM propositions WHERE namespace = $1", ns,
        )

    assert stats["n_total"] == 2
    assert stats["total_len"] == lengths
    assert stats["avgdl"] == pytest.approx(lengths / 2)


async def test_corpus_stats_are_scoped_per_namespace(pool: asyncpg.Pool) -> None:
    """Statistics for one retrieval domain never leak into another."""
    ns_a, ns_b = _ns("scope_a"), _ns("scope_b")
    async with pool.acquire() as conn:
        await insert_proposition(conn, text="alpha beta", namespace=ns_a)
        await insert_proposition(conn, text="gamma", namespace=ns_b)
        await insert_proposition(conn, text="delta", namespace=ns_b)

        stats_a = await corpus_stats(conn, ns_a)
        stats_b = await corpus_stats(conn, ns_b)
        df_a = await lexeme_df(conn, ns_a)

    assert stats_a["n_total"] == 1
    assert stats_b["n_total"] == 2
    assert set(df_a) == {"alpha", "beta"}


async def test_lexeme_df_counts_documents_not_occurrences(pool: asyncpg.Pool) -> None:
    """Document frequency is a document count, so repetition inside one
    document must not inflate it."""
    ns = _ns("df_docs")
    async with pool.acquire() as conn:
        await insert_proposition(
            conn, text="fermentation fermentation fermentation", namespace=ns,
        )
        await insert_proposition(conn, text="fermentation of sugar", namespace=ns)
        await insert_proposition(conn, text="unrelated topic", namespace=ns)
        df = await lexeme_df(conn, ns)

    assert df["ferment"] == 2
    assert df["sugar"] == 1


async def test_stats_track_deletes(pool: asyncpg.Pool) -> None:
    """Deleting a proposition removes exactly its contribution."""
    ns = _ns("stats_del")
    async with pool.acquire() as conn:
        keep = await insert_proposition(conn, text="cat sat", namespace=ns)
        drop = await insert_proposition(conn, text="dog barked loudly", namespace=ns)

        await conn.execute("DELETE FROM propositions WHERE id = $1", drop)

        stats = await corpus_stats(conn, ns)
        keep_len = await conn.fetchval(
            "SELECT doc_len FROM propositions WHERE id = $1", keep,
        )
        df = await lexeme_df(conn, ns)

    assert stats["n_total"] == 1
    assert stats["total_len"] == keep_len
    assert df["cat"] == 1
    assert df.get("dog", 0) == 0


async def test_stats_track_cascading_deletes(pool: asyncpg.Pool) -> None:
    """Propositions deleted by the chunk cascade are accounted for too —
    nothing in the maintenance path assumes the delete was explicit."""
    ns = _ns("stats_cascade")
    async with pool.acquire() as conn:
        doc_id = await conn.fetchval(
            "INSERT INTO documents (namespace) VALUES ($1) RETURNING id", ns,
        )
        chunk_id = await conn.fetchval(
            """
            INSERT INTO chunks (document_id, text) VALUES ($1, 'chunk body')
            RETURNING id
            """,
            doc_id,
        )
        await insert_proposition(
            conn, text="cascaded claim", namespace=ns, chunk_id=chunk_id,
        )
        before = await corpus_stats(conn, ns)

        await conn.execute("DELETE FROM chunks WHERE id = $1", chunk_id)
        after = await corpus_stats(conn, ns)

    assert before["n_total"] == 1
    assert after["n_total"] == 0
    assert after["total_len"] == 0


async def test_deleting_an_already_superseded_proposition_changes_nothing(
    pool: asyncpg.Pool,
) -> None:
    """Supersession already withdrew the row's contribution, so the delete that
    eventually reclaims the row must not withdraw it a second time."""
    ns = _ns("stats_del_super")
    async with pool.acquire() as conn:
        await insert_proposition(conn, text="active claim", namespace=ns)
        stale = await insert_proposition(conn, text="stale claim", namespace=ns)
        await conn.execute(
            "UPDATE propositions SET superseded_by = id WHERE id = $1", stale,
        )
        before = dict(await corpus_stats(conn, ns))
        df_before = await lexeme_df(conn, ns)

        await conn.execute("DELETE FROM propositions WHERE id = $1", stale)

        after = dict(await corpus_stats(conn, ns))
        df_after = await lexeme_df(conn, ns)

    assert before["n_total"] == 1
    assert after["n_total"] == before["n_total"]
    assert after["total_len"] == before["total_len"]
    assert df_after["claim"] == df_before["claim"] == 1


async def test_superseding_a_proposition_removes_it_from_stats(pool: asyncpg.Pool) -> None:
    """Retrieval ignores superseded propositions, so the statistics it ranks
    against must ignore them as well."""
    ns = _ns("stats_super")
    async with pool.acquire() as conn:
        await insert_proposition(conn, text="current claim", namespace=ns)
        stale = await insert_proposition(conn, text="retracted claim", namespace=ns)

        await conn.execute(
            "UPDATE propositions SET superseded_by = id WHERE id = $1", stale,
        )
        stats = await corpus_stats(conn, ns)
        df = await lexeme_df(conn, ns)

    assert stats["n_total"] == 1
    assert df["current"] == 1
    assert df.get("retract", 0) == 0


async def test_unsuperseding_a_proposition_restores_it(pool: asyncpg.Pool) -> None:
    """The update path is a signed delta, not a one-way removal."""
    ns = _ns("stats_unsuper")
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(conn, text="revived claim", namespace=ns)
        await conn.execute(
            "UPDATE propositions SET superseded_by = id WHERE id = $1", prop_id,
        )
        await conn.execute(
            "UPDATE propositions SET superseded_by = NULL WHERE id = $1", prop_id,
        )

        stats = await corpus_stats(conn, ns)
        df = await lexeme_df(conn, ns)

    assert stats["n_total"] == 1
    assert df["reviv"] == 1


async def test_rewriting_text_replaces_its_lexeme_contribution(pool: asyncpg.Pool) -> None:
    """An update that changes the tsvector subtracts the old lexemes and adds
    the new ones."""
    ns = _ns("stats_rewrite")
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(conn, text="obsolete wording", namespace=ns)
        await conn.execute(
            "UPDATE propositions SET text = 'replacement wording' WHERE id = $1",
            prop_id,
        )

        stats = await corpus_stats(conn, ns)
        df = await lexeme_df(conn, ns)

    assert stats["n_total"] == 1
    assert df.get("obsolet", 0) == 0
    assert df["replac"] == 1
    assert df["word"] == 1


async def test_moving_a_proposition_between_namespaces_moves_its_stats(
    pool: asyncpg.Pool,
) -> None:
    """The domain key is part of the delta, so a namespace change is a
    subtraction from one domain and an addition to the other."""
    ns_from, ns_to = _ns("move_from"), _ns("move_to")
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(conn, text="portable claim", namespace=ns_from)
        await conn.execute(
            "UPDATE propositions SET namespace = $2 WHERE id = $1", prop_id, ns_to,
        )

        stats_from = await corpus_stats(conn, ns_from)
        stats_to = await corpus_stats(conn, ns_to)

    assert stats_from["n_total"] == 0
    assert stats_to["n_total"] == 1


async def test_access_count_bump_does_not_rewrite_stats(pool: asyncpg.Pool) -> None:
    """The access-count flush rewrites access_count in bulk after reads.
    Nothing it touches affects ranking statistics, so the maintenance path must
    skip such an update entirely rather than subtract and re-add each row.
    """
    ns = _ns("stats_bump")
    async with pool.acquire() as conn:
        ids = [
            await insert_proposition(conn, text=f"claim number {i}", namespace=ns)
            for i in range(3)
        ]
        before = await corpus_stats(conn, ns)

        await conn.execute("SELECT pgkg_bump_access($1::uuid[])", ids)

        after = await corpus_stats(conn, ns)
        bumped = await conn.fetchval(
            "SELECT MIN(access_count) FROM propositions WHERE namespace = $1", ns,
        )

    assert bumped == 1, "the bump itself must still have happened"
    assert after["updated_at"] == before["updated_at"]
    assert after["n_total"] == before["n_total"]
    assert after["total_len"] == before["total_len"]


# ---------------------------------------------------------------------------
# Full recomputation
# ---------------------------------------------------------------------------

async def test_refresh_agrees_with_incremental_maintenance(pool: asyncpg.Pool) -> None:
    """The incremental deltas and a from-scratch recomputation must not drift.

    This is the invariant that lets the triggers be dropped in favour of a
    scheduled refresh without the read path noticing.
    """
    ns = _ns("refresh_agree")
    async with pool.acquire() as conn:
        keep = await insert_proposition(conn, text="mitochondria produce energy", namespace=ns)
        gone = await insert_proposition(conn, text="chloroplasts capture light", namespace=ns)
        stale = await insert_proposition(conn, text="mitochondria are organelles", namespace=ns)
        await conn.execute("DELETE FROM propositions WHERE id = $1", gone)
        await conn.execute(
            "UPDATE propositions SET superseded_by = id WHERE id = $1", stale,
        )

        incremental_stats = dict(await corpus_stats(conn, ns))
        incremental_df = await lexeme_df(conn, ns)

        await conn.execute("SELECT pgkg_refresh_corpus_stats($1)", ns)

        refreshed_stats = dict(await corpus_stats(conn, ns))
        refreshed_df = await lexeme_df(conn, ns)

    assert incremental_stats["n_total"] == refreshed_stats["n_total"] == 1
    assert incremental_stats["total_len"] == refreshed_stats["total_len"]
    assert {k: v for k, v in incremental_df.items() if v > 0} == refreshed_df
    assert keep is not None


async def test_refresh_repairs_corrupted_stats(pool: asyncpg.Pool) -> None:
    """A full refresh restores statistics that were removed or falsified."""
    ns = _ns("refresh_repair")
    async with pool.acquire() as conn:
        await insert_proposition(conn, text="alpha beta", namespace=ns)
        await insert_proposition(conn, text="beta gamma", namespace=ns)
        truth = dict(await corpus_stats(conn, ns))

        await conn.execute("DELETE FROM corpus_stats WHERE namespace = $1", ns)
        await conn.execute("DELETE FROM lexeme_df WHERE namespace = $1", ns)
        await conn.execute("SELECT pgkg_refresh_corpus_stats($1)", ns)

        repaired = dict(await corpus_stats(conn, ns))
        df = await lexeme_df(conn, ns)

    assert repaired["n_total"] == truth["n_total"]
    assert repaired["total_len"] == truth["total_len"]
    assert df["beta"] == 2


async def test_refresh_is_scoped_to_one_namespace(pool: asyncpg.Pool) -> None:
    """Refreshing one domain must not disturb another — a whole-corpus rebuild
    is not an acceptable cost for repairing one tenant."""
    ns_target, ns_other = _ns("refresh_target"), _ns("refresh_other")
    async with pool.acquire() as conn:
        await insert_proposition(conn, text="target claim", namespace=ns_target)
        await insert_proposition(conn, text="other claim", namespace=ns_other)

        await conn.execute(
            "UPDATE corpus_stats SET n_total = 999 WHERE namespace = $1", ns_other,
        )
        await conn.execute("SELECT pgkg_refresh_corpus_stats($1)", ns_target)

        target = await corpus_stats(conn, ns_target)
        other = await corpus_stats(conn, ns_other)

    assert target["n_total"] == 1
    assert other["n_total"] == 999


# ---------------------------------------------------------------------------
# Retrieval against materialised statistics
# ---------------------------------------------------------------------------

async def test_bm25_ranks_by_materialised_idf(pool: asyncpg.Pool) -> None:
    """A rare query term outranks a common one, now that document frequency
    comes from lexeme_df instead of a correlated COUNT(*)."""
    ns = _ns("bm25_idf")
    async with pool.acquire() as conn:
        for i in range(20):
            await insert_proposition(
                conn, text=f"the animal kingdom contains species {i}", namespace=ns,
            )
        rare = await insert_proposition(conn, text="zymurgy explained", namespace=ns)
        common = await insert_proposition(conn, text="animal explained", namespace=ns)

        rows = await conn.fetch(
            "SELECT item_id, raw_score FROM pgkg_bm25_candidates($1, $2)",
            "zymurgy animal", ns,
        )

    scores = {r["item_id"]: float(r["raw_score"]) for r in rows}
    assert scores[rare] > scores[common] * 1.5


async def test_bm25_length_normalisation_uses_stored_doc_len(pool: asyncpg.Pool) -> None:
    """avgdl now comes from the materialised sum, so the b=0.75 length penalty
    must still put a concise document above a padded one."""
    ns = _ns("bm25_len")
    async with pool.acquire() as conn:
        short_id = await insert_proposition(
            conn, text="mitochondria generate energy", namespace=ns,
        )
        padding = " ".join(f"filler{i}" for i in range(80))
        long_id = await insert_proposition(
            conn, text=f"mitochondria generate energy {padding}", namespace=ns,
        )

        rows = await conn.fetch(
            "SELECT item_id, rank FROM pgkg_bm25_candidates($1, $2) ORDER BY rank",
            "mitochondria", ns,
        )

    ordered = [r["item_id"] for r in rows]
    assert ordered.index(short_id) < ordered.index(long_id)


async def test_bm25_ranks_sanely_on_missing_stats(pool: asyncpg.Pool) -> None:
    """BM25 is a heuristic, so absent or lagging statistics must degrade the
    ranking rather than empty the result set or raise.

    Deleting the domain's statistics rows is the worst case a scheduled
    refresh could produce.  IDF flattens to a constant, but term frequency and
    length normalisation still work, so a concise match still wins.
    """
    ns = _ns("bm25_stale")
    async with pool.acquire() as conn:
        short_id = await insert_proposition(
            conn, text="mitochondria generate energy", namespace=ns,
        )
        padding = " ".join(f"filler{i}" for i in range(80))
        long_id = await insert_proposition(
            conn, text=f"mitochondria generate energy {padding}", namespace=ns,
        )
        await insert_proposition(conn, text="entirely unrelated matter", namespace=ns)

        await conn.execute("DELETE FROM corpus_stats WHERE namespace = $1", ns)
        await conn.execute("DELETE FROM lexeme_df WHERE namespace = $1", ns)

        rows = await conn.fetch(
            "SELECT item_id, rank, raw_score FROM pgkg_bm25_candidates($1, $2) ORDER BY rank",
            "mitochondria", ns,
        )

    ordered = [r["item_id"] for r in rows]
    assert ordered == [short_id, long_id]
    assert all(float(r["raw_score"]) > 0.0 for r in rows)


async def test_bm25_survives_a_namespace_with_no_stats_row(pool: asyncpg.Pool) -> None:
    """A namespace that has never been written to has no statistics row at
    all; a query against it returns nothing rather than failing."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT item_id FROM pgkg_bm25_candidates($1, $2)",
            "anything at all", _ns("bm25_empty"),
        )

    assert rows == []


async def test_keyword_search_scans_propositions_once(pool: asyncpg.Pool) -> None:
    """The correlated per-term COUNT(*) is gone.

    Document frequency and the corpus average used to cost one scan of every
    active proposition in the namespace each — the average once, the document
    frequency once per query lexeme.  A keyword search must now touch
    `propositions` exactly once, for the index scan that finds the candidates,
    and read its statistics from the materialised tables instead.
    """
    ns = _ns("plan_shape")
    async with pool.acquire() as conn:
        for i in range(50):
            await insert_proposition(
                conn, text=f"the animal kingdom contains species {i}", namespace=ns,
            )
        await insert_proposition(conn, text="zymurgy and fermentation", namespace=ns)
        await conn.execute("ANALYZE propositions")

        plan = await conn.fetchval(
            """
            EXPLAIN (FORMAT JSON, COSTS OFF)
            SELECT * FROM pgkg_bm25_candidates($1, $2)
            """,
            "zymurgy animal fermentation", ns,
        )

    counts = _relation_counts(plan)
    assert counts.get("propositions", 0) == 1, (
        f"keyword search should visit propositions once, plan visits it "
        f"{counts.get('propositions', 0)} times: {counts}"
    )
    assert counts.get("corpus_stats", 0) >= 1
    assert counts.get("lexeme_df", 0) >= 1


# ---------------------------------------------------------------------------
# The IDF is a per-tenant quantity (ADR 0001, D4)
# ---------------------------------------------------------------------------

async def test_another_tenants_writes_do_not_move_my_bm25_score(
    pool: asyncpg.Pool,
) -> None:
    """N, avgdl and every term's document frequency were keyed on namespace and
    kind alone.

    Every tenant's Memory uses one namespace, so the whole IDF was computed
    across all of them: a tenant could measure how much others had written
    about a competitor's name or an acquisition codename purely by watching its
    own scores move, with no access to a single row.  D4 calls a ranking signal
    computed globally over shared content a real inference channel, and the
    chunk half of the same function was already keyed per collection, so the
    omission was asymmetric rather than a decision.
    """
    import uuid as _uuid

    term = f"zzqnovaline{_uuid.uuid4().hex[:6]}"

    async with pool.acquire() as conn:
        mine = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
            f"mine_{_uuid.uuid4().hex[:8]}",
        )
        theirs = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id",
            f"theirs_{_uuid.uuid4().hex[:8]}",
        )

        for i in range(3):
            await conn.execute(
                "INSERT INTO propositions (text, org_id) VALUES ($1, $2)",
                f"{term} appears in my own note number {i}",
                mine,
            )

        async def my_score() -> float:
            return await conn.fetchval(
                "SELECT raw_score FROM pgkg_bm25_candidates("
                " q_text => $1, p_org_ids => $2::uuid[]) ORDER BY rank LIMIT 1",
                term,
                [mine],
            )

        before = await my_score()
        for i in range(200):
            await conn.execute(
                "INSERT INTO propositions (text, org_id) VALUES ($1, $2)",
                f"{term} is discussed at length in a stranger's note {i}",
                theirs,
            )
        after = await my_score()

    assert before is not None and after is not None
    assert after == pytest.approx(before, rel=1e-4), (
        f"another tenant's writes changed this tenant's BM25 score: "
        f"{before} -> {after}"
    )
