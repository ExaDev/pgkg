"""Provenance: one append-only derivation record per chunk-and-model-run.

Every assertion goes through a public surface — the table's own constraints and
triggers, or the cascade functions — because the value of the column is that
retraction and re-extraction become set-based queries rather than traversals.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import asyncpg
import pytest


def ns() -> str:
    return f"prov_{uuid.uuid4().hex[:10]}"


async def new_provenance(
    conn: asyncpg.Connection,
    *,
    kind: str = "document_version",
    producer: str = "llm_extract",
    source_id: uuid.UUID | None = None,
    ingest_run_id: uuid.UUID | None = None,
    **fields: object,
) -> uuid.UUID:
    columns = {
        "org_id": await conn.fetchval("SELECT pgkg_default_org()"),
        "kind": kind,
        "producer": producer,
        "source_id": source_id,
        "ingest_run_id": ingest_run_id,
        **fields,
    }
    names = list(columns)
    placeholders = ", ".join(f"${i + 1}" for i in range(len(names)))
    return await conn.fetchval(
        f"INSERT INTO provenance ({', '.join(names)}) "
        f"VALUES ({placeholders}) RETURNING id",
        *[columns[n] for n in names],
    )


async def insert_proposition(
    conn: asyncpg.Connection,
    *,
    text: str,
    namespace: str,
    provenance_id: uuid.UUID | None = None,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO propositions (text, namespace, provenance_id)
        VALUES ($1, $2, COALESCE($3, pgkg_unattributed_provenance()))
        RETURNING id
        """,
        text,
        namespace,
        provenance_id,
    )


async def temporal(conn: asyncpg.Connection, prop_id: uuid.UUID) -> asyncpg.Record:
    return await conn.fetchrow(
        "SELECT invalidated_at, invalidation_reason, superseded_by "
        "FROM propositions WHERE id = $1",
        prop_id,
    )


SEARCH = """
SELECT proposition_id FROM pgkg_search($1, NULL, 50, 200, $2)
"""


async def search_ids(
    conn: asyncpg.Connection, *, q: str, namespace: str
) -> set[uuid.UUID]:
    rows = await conn.fetch(SEARCH, q, namespace)
    return {row["proposition_id"] for row in rows}


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------


async def test_provenance_carries_the_external_source_axes(pool: asyncpg.Pool) -> None:
    """An external source needs a citation, a licence and two clocks that an
    internal document does not have: when it was published and when we fetched
    it are different facts, and both differ from when we ingested it."""
    published = datetime(2019, 3, 1, tzinfo=timezone.utc)
    retrieved = datetime(2026, 1, 4, tzinfo=timezone.utc)
    async with pool.acquire() as conn:
        prov_id = await new_provenance(
            conn,
            source_url="https://example.test/handbook",
            publisher="Example Press",
            published_at=published,
            retrieved_at=retrieved,
            licence="CC-BY-4.0",
            source_authority=7,
            source_locator=json.dumps({"page": 11, "section": "4.2"}),
            producer_model="test-extractor",
            prompt_version="v1",
        )
        row = await conn.fetchrow(
            """
            SELECT source_url, publisher, published_at, retrieved_at, licence,
                   source_authority, source_locator, producer_model,
                   prompt_version, created_at
            FROM provenance WHERE id = $1
            """,
            prov_id,
        )

    assert row["source_url"] == "https://example.test/handbook"
    assert row["publisher"] == "Example Press"
    assert row["published_at"] == published
    assert row["retrieved_at"] == retrieved
    assert row["licence"] == "CC-BY-4.0"
    assert row["source_authority"] == 7
    assert json.loads(row["source_locator"])["page"] == 11
    assert row["created_at"] is not None


async def test_provenance_kind_vocabulary_is_closed(pool: asyncpg.Pool) -> None:
    """A derivation record whose kind is a typo is a record that no cascade
    will ever find, so the vocabulary is a constraint rather than a comment."""
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.IntegrityConstraintViolationError):
            await new_provenance(conn, kind="documnet_version")


async def test_provenance_rejects_updates(pool: asyncpg.Pool) -> None:
    """How a fact was derived is not a mutable field: rewriting it would make
    every citation the system has already emitted retrospectively wrong."""
    async with pool.acquire() as conn:
        prov_id = await new_provenance(conn, publisher="Example Press")
        with pytest.raises(asyncpg.RaiseError):
            await conn.execute(
                "UPDATE provenance SET publisher = 'Someone Else' WHERE id = $1",
                prov_id,
            )
        publisher = await conn.fetchval(
            "SELECT publisher FROM provenance WHERE id = $1", prov_id
        )

    assert publisher == "Example Press"


async def test_provenance_rejects_casual_deletes(pool: asyncpg.Pool) -> None:
    """Append-only means a stray DELETE cannot quietly detach a fact from its
    source; erasure is a deliberate act with its own entry point."""
    async with pool.acquire() as conn:
        prov_id = await new_provenance(conn)
        with pytest.raises(asyncpg.RaiseError):
            await conn.execute("DELETE FROM provenance WHERE id = $1", prov_id)
        survives = await conn.fetchval(
            "SELECT count(*) FROM provenance WHERE id = $1", prov_id
        )

    assert survives == 1


async def test_erasure_removes_provenance_through_its_own_entry_point(
    pool: asyncpg.Pool,
) -> None:
    """Subject erasure has to be possible: it removes the derivation records a
    user contributed, and a proposition left with none is the caller's to
    delete."""
    async with pool.acquire() as conn:
        prov_id = await new_provenance(conn, producer="user_assertion")
        erased = await conn.fetchval(
            "SELECT pgkg_erase_provenance(ARRAY[$1]::UUID[])", prov_id
        )
        remaining = await conn.fetchval(
            "SELECT count(*) FROM provenance WHERE id = $1", prov_id
        )

    assert erased == 1
    assert remaining == 0


# ---------------------------------------------------------------------------
# The hot-path column
# ---------------------------------------------------------------------------


async def test_propositions_always_carry_a_provenance_id(pool: asyncpg.Pool) -> None:
    """The column is NOT NULL because the cascade is an indexed equality on it;
    a nullable column would mean a fallback traversal for every source change."""
    namespace = ns()
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.NotNullViolationError):
            await conn.execute(
                "INSERT INTO propositions (text, namespace, provenance_id) "
                "VALUES ('unsourced', $1, NULL)",
                namespace,
            )


async def test_an_unscoped_insert_lands_on_the_backfill_record(
    pool: asyncpg.Pool,
) -> None:
    """A caller that predates provenance still writes a complete row: the
    reserved record names those rows as unattributed rather than pretending
    they came from a source."""
    namespace = ns()
    async with pool.acquire() as conn:
        prop_id = await conn.fetchval(
            "INSERT INTO propositions (text, namespace) VALUES ('legacy', $1) "
            "RETURNING id",
            namespace,
        )
        kind = await conn.fetchval(
            """
            SELECT pr.kind FROM propositions p
            JOIN provenance pr ON pr.id = p.provenance_id
            WHERE p.id = $1
            """,
            prop_id,
        )

    assert kind == "backfill"


async def test_chunks_carry_provenance_too(pool: asyncpg.Pool) -> None:
    """The chunker is a producer in its own right, and a retired source has to
    reach the chunks it produced as well as the facts extracted from them."""
    async with pool.acquire() as conn:
        prov_id = await new_provenance(conn, kind="document_version", producer="chunker")
        chunk_id = await conn.fetchval(
            "INSERT INTO chunks (text, provenance_id) VALUES ('body', $1) RETURNING id",
            prov_id,
        )
        stored = await conn.fetchval(
            "SELECT provenance_id FROM chunks WHERE id = $1", chunk_id
        )

    assert stored == prov_id


async def test_many_sources_corroborate_one_proposition(pool: asyncpg.Pool) -> None:
    """Deduplication makes the relation many-to-one, and N sources is the
    signal that should raise confidence rather than a row to throw away."""
    namespace = ns()
    async with pool.acquire() as conn:
        first = await new_provenance(conn, kind="chat_turn", producer="llm_extract")
        second = await new_provenance(conn, kind="chat_turn", producer="llm_extract")
        prop_id = await insert_proposition(
            conn, text="prefers dark mode", namespace=namespace, provenance_id=first
        )
        await conn.executemany(
            "INSERT INTO proposition_provenance (proposition_id, provenance_id) "
            "VALUES ($1, $2)",
            [(prop_id, first), (prop_id, second)],
        )
        corroboration = await conn.fetchval(
            "SELECT count(*) FROM proposition_provenance WHERE proposition_id = $1",
            prop_id,
        )
        await conn.execute("DELETE FROM propositions WHERE id = $1", prop_id)
        orphans = await conn.fetchval(
            "SELECT count(*) FROM proposition_provenance WHERE proposition_id = $1",
            prop_id,
        )

    assert corroboration == 2
    assert orphans == 0


# ---------------------------------------------------------------------------
# Cascade on source change
# ---------------------------------------------------------------------------


async def test_retiring_a_source_invalidates_exactly_its_propositions(
    pool: asyncpg.Pool,
) -> None:
    """One set-based UPDATE keyed on provenance_id, and nothing derived from
    any other source moves."""
    namespace = ns()
    retired_source = uuid.uuid4()
    kept_source = uuid.uuid4()
    async with pool.acquire() as conn:
        retired_prov = await new_provenance(conn, source_id=retired_source)
        kept_prov = await new_provenance(conn, source_id=kept_source)
        doomed = await insert_proposition(
            conn,
            text="turbine calibration is quarterly",
            namespace=namespace,
            provenance_id=retired_prov,
        )
        spared = await insert_proposition(
            conn,
            text="turbine calibration is annual",
            namespace=namespace,
            provenance_id=kept_prov,
        )

        affected = await conn.fetchval(
            "SELECT pgkg_invalidate_source($1, 'source_deleted')", retired_source
        )

        doomed_row = await temporal(conn, doomed)
        spared_row = await temporal(conn, spared)
        found = await search_ids(conn, q="turbine calibration", namespace=namespace)

    assert affected == 1
    assert doomed_row["invalidated_at"] is not None
    assert doomed_row["invalidation_reason"] == "source_deleted"
    assert spared_row["invalidated_at"] is None
    assert found == {spared}


async def test_retiring_a_source_leaves_an_earlier_reason_intact(
    pool: asyncpg.Pool,
) -> None:
    """A row already withdrawn keeps the reason it was withdrawn for: the
    cascade must not overwrite the audit trail of why belief changed."""
    namespace = ns()
    source_id = uuid.uuid4()
    async with pool.acquire() as conn:
        prov_id = await new_provenance(conn, source_id=source_id)
        prop_id = await insert_proposition(
            conn, text="stale reading", namespace=namespace, provenance_id=prov_id
        )
        await conn.execute(
            "UPDATE propositions SET superseded_by = id WHERE id = $1", prop_id
        )
        before = await temporal(conn, prop_id)

        affected = await conn.fetchval(
            "SELECT pgkg_invalidate_source($1, 'source_updated')", source_id
        )
        after = await temporal(conn, prop_id)

    assert affected == 0
    assert before["invalidation_reason"] == "superseded"
    assert after["invalidation_reason"] == "superseded"
    assert after["invalidated_at"] == before["invalidated_at"]


async def test_retracting_an_ingest_run_undoes_only_that_batch(
    pool: asyncpg.Pool,
) -> None:
    """A bad extractor deploy is one indexed UPDATE to undo, which is the whole
    reason ingest_run_id is on the derivation record."""
    namespace = ns()
    bad_run = uuid.uuid4()
    good_run = uuid.uuid4()
    async with pool.acquire() as conn:
        bad_prov = await new_provenance(conn, ingest_run_id=bad_run)
        good_prov = await new_provenance(conn, ingest_run_id=good_run)
        garbage = await insert_proposition(
            conn, text="regulator hallucination", namespace=namespace,
            provenance_id=bad_prov,
        )
        sound = await insert_proposition(
            conn, text="regulator inspection cadence", namespace=namespace,
            provenance_id=good_prov,
        )

        affected = await conn.fetchval("SELECT pgkg_retract_ingest_run($1)", bad_run)

        garbage_row = await temporal(conn, garbage)
        found = await search_ids(conn, q="regulator", namespace=namespace)

    assert affected == 1
    assert garbage_row["invalidation_reason"] == "retracted_run"
    assert found == {sound}


async def test_a_retracted_run_leaves_the_ranking_statistics(
    pool: asyncpg.Pool,
) -> None:
    """IDF is computed over the corpus retrieval can actually return, so a
    withdrawal that retrieval honours has to move the statistics with it."""
    namespace = ns()
    run_id = uuid.uuid4()
    async with pool.acquire() as conn:
        prov_id = await new_provenance(conn, ingest_run_id=run_id)
        await insert_proposition(
            conn, text="kept sentence", namespace=namespace,
        )
        await insert_proposition(
            conn, text="withdrawn sentence", namespace=namespace,
            provenance_id=prov_id,
        )

        await conn.execute("SELECT pgkg_retract_ingest_run($1)", run_id)

        n_total = await conn.fetchval(
            "SELECT n_total FROM corpus_stats WHERE namespace = $1 "
            "AND kind = 'proposition'",
            namespace,
        )
        withdrawn_df = await conn.fetchval(
            "SELECT df FROM lexeme_df WHERE namespace = $1 AND kind = 'proposition' "
            "AND lexeme = 'withdrawn'",
            namespace,
        )

    assert n_total == 1
    assert (withdrawn_df or 0) == 0


async def test_deleting_an_actor_is_refused_until_their_records_are_erased(
    pool: asyncpg.Pool,
) -> None:
    """Append-only has no exception for a foreign key's convenience: nulling
    the actor would be a rewrite.  Erasure therefore runs in D5's order — the
    derivation records go first, and only then the user."""
    async with pool.acquire() as conn:
        org_id = await conn.fetchval("SELECT pgkg_default_org()")
        user_id = await conn.fetchval(
            "INSERT INTO users (org_id, external_id) VALUES ($1, $2) RETURNING id",
            org_id,
            f"erasure_{uuid.uuid4().hex[:8]}",
        )
        prov_id = await new_provenance(
            conn, kind="chat_turn", producer="user_assertion", actor_user_id=user_id
        )

        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await conn.execute("DELETE FROM users WHERE id = $1", user_id)

        await conn.execute("SELECT pgkg_erase_provenance(ARRAY[$1]::UUID[])", prov_id)
        await conn.execute("DELETE FROM users WHERE id = $1", user_id)

        remaining = await conn.fetchval(
            "SELECT count(*) FROM users WHERE id = $1", user_id
        )

    assert remaining == 0
