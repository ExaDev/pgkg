"""What the retrieval arms cost, measured as plan shape rather than as seconds.

Correctness tests pin the rows an arm returns; nothing pins how many times it
touches the table to return them, and every defect in this module was invisible
to a passing suite for that reason.  A wall-clock assertion on shared CI is a
flake, so each test here reads the plan the planner actually chose: how many
times the scoring aggregate is planned, how much of the vocabulary the IDF
lookup reads, and whether liveness is a column comparison or a subquery per
candidate row.

The corpus is deliberately large enough for the planner to have a choice.  On
four hundred rows every shape looks the same.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid

import asyncpg
import pytest

LIVENESS = """
    (c.refcount = 0 AND c.document_id IS NULL)
    OR EXISTS (
        SELECT 1 FROM document_version_chunks dvc
        JOIN document_versions dv ON dv.id = dvc.document_version_id
        JOIN documents d ON d.id = dv.document_id
        WHERE dvc.chunk_id = c.id
          AND dv.status = 'current'
          AND d.deleted_at IS NULL)
"""


def unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@pytest.fixture(scope="module")
async def corpus(pool: asyncpg.Pool):
    """One org with 4000 passages in a live document version, and 4000 facts."""
    async with pool.acquire() as conn:
        org = await conn.fetchval(
            "INSERT INTO orgs (name) VALUES ($1) RETURNING id", unique("plan_org")
        )
        await conn.execute("SELECT set_config('pgkg.org_id', $1, false)", str(org))
        collection = await conn.fetchval(
            """
            INSERT INTO collections (org_id, owner_org_id, name, kind)
            VALUES ($1, $1, $2, 'corpus') RETURNING id
            """,
            org,
            unique("plan_coll"),
        )
        namespace = unique("plan_ns")
        document = await conn.fetchval(
            """
            INSERT INTO documents (source, namespace, org_id, collection_id,
                                   external_id)
            VALUES ('seed', $1, $2, $3, $4) RETURNING id
            """,
            namespace,
            org,
            collection,
            unique("plan_doc"),
        )
        version = await conn.fetchval(
            """
            INSERT INTO document_versions
                (document_id, org_id, version_no, content_hash, status)
            VALUES ($1, $2, 1, $3, 'current') RETURNING id
            """,
            document,
            org,
            hashlib.sha256(b"seed").digest(),
        )
        await conn.execute(
            "UPDATE documents SET current_version_id = $1 WHERE id = $2",
            version,
            document,
        )
        await conn.execute(
            """
            INSERT INTO chunks (document_id, text, span_start, span_end,
                                org_id, collection_id)
            SELECT $1,
                   'the reimbursement policy for lodging and equipment states '
                   || 'that expense claim ' || g || ' needs approval from the '
                   || 'finance team before the operator reconciles the ledger '
                   || repeat('filler phrase ' || (g % 91) || ' ', 12),
                   0, 400, $2, $3
            FROM generate_series(1, 4000) g
            """,
            document,
            org,
            collection,
        )
        await conn.execute(
            """
            INSERT INTO document_version_chunks (document_version_id, chunk_id, ord)
            SELECT $1, c.id, (row_number() OVER (ORDER BY c.id))::int - 1
            FROM chunks c WHERE c.document_id = $2
            """,
            version,
            document,
        )
        await conn.execute(
            """
            INSERT INTO propositions (text, namespace, org_id, collection_id,
                                      asserted_at)
            SELECT 'the operator prefers a reimbursement window of ' || g
                   || ' days and avoids lodging in city ' || (g % 300),
                   $1, $2, $3, now() - (g || ' minutes')::interval
            FROM generate_series(1, 4000) g
            """,
            namespace,
            org,
            collection,
        )
        for table in ("chunks", "propositions", "lexeme_df",
                      "document_version_chunks"):
            await conn.execute(f"ANALYZE {table}")
    return org, collection, namespace


async def explain(conn: asyncpg.Connection, sql: str, *args: object) -> str:
    rows = await conn.fetch(f"EXPLAIN (ANALYZE, BUFFERS) {sql}", *args)
    return "\n".join(row[0] for row in rows)


async def buffers(conn: asyncpg.Connection, sql: str, *args: object) -> int:
    plan = await conn.fetchval(
        f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}", *args
    )
    node = json.loads(plan)[0]["Plan"]
    return node.get("Shared Hit Blocks", 0) + node.get("Shared Read Blocks", 0)


KEYWORD_ARM = (
    "SELECT * FROM pgkg_bm25_candidates($1, $2, NULL, 200, $3::uuid[], $4::uuid[],"
    " NULL, NULL, NULL, $5)"
)


async def test_bm25_scores_each_candidate_once(
    pool: asyncpg.Pool, corpus
) -> None:
    """`WHERE score > 0 ... ORDER BY score`, where `score` is a correlated
    sub-select, makes Postgres plan the aggregate twice — once for the filter
    and once for the sort — and nothing shares the result, so every candidate
    row paid for its own BM25 score twice.
    """
    org, collection, namespace = corpus
    async with pool.acquire() as conn:
        plan = await explain(
            conn, KEYWORD_ARM, "reimbursement window lodging", namespace,
            [org], None, "propositions",
        )

    subplans = re.findall(r"^\s+SubPlan \d+$", plan, re.MULTILINE)
    assert len(subplans) <= 1, (
        f"the BM25 aggregate is planned {len(subplans)} times, not once:\n{plan}"
    )


async def test_idf_reads_only_the_query_terms(pool: asyncpg.Pool, corpus) -> None:
    """The per-row correlated COUNT(*) became one aggregate over lexeme_df, but
    an unrestricted one: the lexeme restriction was a join to a CTE, so the
    primary key on the statistics table was unusable and every query read the
    entire vocabulary of its corpus — a cost that grows with the corpus, on the
    query path.
    """
    org, collection, namespace = corpus
    async with pool.acquire() as conn:
        plan = await explain(
            conn, KEYWORD_ARM, "reimbursement window lodging", namespace,
            [org], None, "propositions",
        )
        vocabulary = await conn.fetchval(
            "SELECT count(*) FROM lexeme_df WHERE kind = 'proposition'"
            " AND namespace = $1",
            namespace,
        )

    scan = re.search(
        r"(Seq Scan|Index Scan|Bitmap Heap Scan)[^\n]*lexeme_df[^\n]*"
        r"actual time=[\d.]+\.\.[\d.]+ rows=(\d+)",
        plan,
    )
    assert scan is not None, f"no lexeme_df access in the plan:\n{plan}"
    assert int(scan.group(2)) < vocabulary, (
        f"the IDF lookup read {scan.group(2)} lexeme_df rows for a three-term "
        f"query; the corpus vocabulary is {vocabulary}. Full plan:\n{plan}"
    )


# ---------------------------------------------------------------------------
# Liveness is a property of the row, read as one (D1)
# ---------------------------------------------------------------------------

RETRIEVAL_FUNCTIONS = (
    "pgkg_bm25_candidates",
    "pgkg_vector_candidates",
    "pgkg_graph_candidates",
    "pgkg_chunk_window",
    "pgkg_retrieve",
)


async def test_no_retrieval_function_tests_liveness_by_calling_a_function(
    pool: asyncpg.Pool,
) -> None:
    """pgkg_chunk_live() cannot inline — a body that reads a table never can,
    whatever it is written as — so every call was a nested loop over the link
    table for one candidate row.

    Liveness is therefore maintained as a column on the row it describes, and
    the arms compare that column.  What is asserted here is the arms, not the
    function: the function remains the readable statement of what liveness
    means and the maintenance path's own test of it, and it is still what an
    offline caller should use.
    """
    async with pool.acquire() as conn:
        callers = {
            row["proname"]
            for row in await conn.fetch(
                """
                SELECT p.proname
                FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = 'public'
                  AND p.proname = ANY($1::text[])
                  AND p.prosrc ILIKE '%pgkg_chunk_live%'
                """,
                list(RETRIEVAL_FUNCTIONS),
            )
        }
    assert not callers, (
        f"these retrieval functions call pgkg_chunk_live() per candidate row: "
        f"{sorted(callers)}"
    )


async def test_the_liveness_flag_agrees_with_the_predicate_it_summarises(
    pool: asyncpg.Pool, corpus
) -> None:
    """A maintained flag is only cheaper than the predicate if it says the same
    thing, so the two are compared over the whole corpus rather than on one
    row: a passage of a live current version, and nothing else, is retrievable.
    """
    org, collection, namespace = corpus
    async with pool.acquire() as conn:
        disagreements = await conn.fetchval(
            f"""
            SELECT count(*) FROM chunks c
            WHERE c.org_id = $1 AND c.retrievable IS DISTINCT FROM ({LIVENESS})
            """,
            org,
        )
        retrievable = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE org_id = $1 AND retrievable", org
        )

    assert disagreements == 0
    assert retrievable == 4000, (
        f"{retrievable} of the 4000 seeded passages are flagged retrievable, "
        f"so the comparison above may be vacuous"
    )


async def test_the_liveness_flag_costs_less_than_the_predicate(
    pool: asyncpg.Pool, corpus
) -> None:
    """And the point of it: the keyword arm's own candidate scan, with liveness
    read off the row, touches no more of the table than the definitional
    predicate does — where the function form touched twenty times as much,
    because it ran a three-table join once per candidate.
    """
    org, collection, namespace = corpus
    scan = (
        "SELECT count(*) FROM chunks c WHERE c.collection_id = $1"
        " AND c.tsv @@ to_tsquery('simple', 'lodg | equip') AND "
    )
    async with pool.acquire() as conn:
        as_flag = await buffers(conn, scan + "c.retrievable", collection)
        as_predicate = await buffers(conn, scan + f"({LIVENESS})", collection)

    assert as_flag <= as_predicate, (
        f"the maintained flag touched {as_flag} buffers where the predicate it "
        f"replaces touched {as_predicate}"
    )
