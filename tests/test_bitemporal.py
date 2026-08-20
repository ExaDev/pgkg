"""Bitemporal lifecycle: four clocks, three query modes.

Validity is when a claim is true in the world; belief is when the system held
it.  The tests keep those separate on purpose, because collapsing them is what
makes "why did the agent say that last month?" unanswerable.  Everything is
asserted through pgkg_search() and the lifecycle functions rather than by
inspecting the filter, since the filter is the thing under test.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest


def ns() -> str:
    return f"bitemporal_{uuid.uuid4().hex[:10]}"


def ago(**delta: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(**delta)


def ahead(**delta: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(**delta)


async def insert_proposition(
    conn: asyncpg.Connection,
    *,
    text: str,
    namespace: str,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    recorded_at: datetime | None = None,
    expires_at: datetime | None = None,
    legal_hold: bool = False,
) -> uuid.UUID:
    return await conn.fetchval(
        """
        INSERT INTO propositions
            (text, namespace, valid_from, valid_to, recorded_at, expires_at,
             legal_hold)
        VALUES ($1, $2, $3, $4, COALESCE($5, now()), $6, $7)
        RETURNING id
        """,
        text,
        namespace,
        valid_from,
        valid_to,
        recorded_at,
        expires_at,
        legal_hold,
    )


async def temporal(conn: asyncpg.Connection, prop_id: uuid.UUID) -> asyncpg.Record:
    return await conn.fetchrow(
        """
        SELECT valid_from, valid_to, recorded_at, invalidated_at,
               invalidation_reason, superseded_by, expires_at, legal_hold
        FROM propositions WHERE id = $1
        """,
        prop_id,
    )


CURRENT = """
SELECT proposition_id FROM pgkg_search($1, NULL, 50, 200, $2)
"""

AS_OF_VALIDITY = """
SELECT proposition_id
FROM pgkg_search(
    q_text := $1::TEXT,
    q_embedding := NULL,
    p_namespace := $2::TEXT,
    p_valid_at := $3::TIMESTAMPTZ
)
"""

AS_OF_BELIEF = """
SELECT proposition_id, invalidation_reason
FROM pgkg_believed_at($1, $2)
"""


async def current_ids(
    conn: asyncpg.Connection, *, q: str, namespace: str
) -> set[uuid.UUID]:
    return {r["proposition_id"] for r in await conn.fetch(CURRENT, q, namespace)}


async def valid_at_ids(
    conn: asyncpg.Connection, *, q: str, namespace: str, at: datetime
) -> set[uuid.UUID]:
    rows = await conn.fetch(AS_OF_VALIDITY, q, namespace, at)
    return {r["proposition_id"] for r in rows}


async def believed_at_ids(
    conn: asyncpg.Connection, *, namespace: str, at: datetime
) -> set[uuid.UUID]:
    rows = await conn.fetch(AS_OF_BELIEF, at, namespace)
    return {r["proposition_id"] for r in rows}


# ---------------------------------------------------------------------------
# Belief: superseded_by is the reason, invalidated_at is the filter
# ---------------------------------------------------------------------------


async def test_supersession_sets_the_invalidation_the_filter_reads(
    pool: asyncpg.Pool,
) -> None:
    """Retrieval filters on one nullable timestamp so a version retirement can
    be a bulk UPDATE.  A self-referencing UUID cannot be set in bulk, so it
    stays as the pointer to the replacement and the timestamp follows it."""
    namespace = ns()
    async with pool.acquire() as conn:
        replacement = await insert_proposition(
            conn, text="badgers hibernate lightly", namespace=namespace
        )
        stale = await insert_proposition(
            conn, text="badgers hibernate deeply", namespace=namespace
        )
        await conn.execute(
            "UPDATE propositions SET superseded_by = $1 WHERE id = $2",
            replacement,
            stale,
        )

        row = await temporal(conn, stale)
        found = await current_ids(conn, q="badgers hibernate", namespace=namespace)

    assert row["invalidated_at"] is not None
    assert row["invalidation_reason"] == "superseded"
    assert row["superseded_by"] == replacement
    assert found == {replacement}


async def test_clearing_supersession_restores_the_row(pool: asyncpg.Pool) -> None:
    """The link between the pointer and the timestamp is a two-way delta, not a
    one-way latch: an operator who un-supersedes a row gets it back."""
    namespace = ns()
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(
            conn, text="revived heuristic", namespace=namespace
        )
        await conn.execute(
            "UPDATE propositions SET superseded_by = id WHERE id = $1", prop_id
        )
        await conn.execute(
            "UPDATE propositions SET superseded_by = NULL WHERE id = $1", prop_id
        )

        row = await temporal(conn, prop_id)
        found = await current_ids(conn, q="revived heuristic", namespace=namespace)

    assert row["invalidated_at"] is None
    assert row["invalidation_reason"] is None
    assert found == {prop_id}


async def test_invalidation_demands_a_reason(pool: asyncpg.Pool) -> None:
    """A withdrawal with no reason is a row nobody can audit and no cascade can
    distinguish from any other, so the two columns move together."""
    namespace = ns()
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(
            conn, text="reasonless withdrawal", namespace=namespace
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "UPDATE propositions SET invalidated_at = now() WHERE id = $1",
                prop_id,
            )


async def test_invalidation_reason_vocabulary_is_closed(pool: asyncpg.Pool) -> None:
    """Free text here would make the reason unqueryable, and every operational
    use of it — retract a run, expire a TTL, honour a deletion — is a query."""
    namespace = ns()
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(
            conn, text="mystery withdrawal", namespace=namespace
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "UPDATE propositions SET invalidated_at = now(), "
                "invalidation_reason = 'because' WHERE id = $1",
                prop_id,
            )


# ---------------------------------------------------------------------------
# Validity: current state and as-of
# ---------------------------------------------------------------------------


async def test_a_claim_not_yet_valid_is_not_retrieved(pool: asyncpg.Pool) -> None:
    """A policy that takes effect next quarter is recorded now and true later;
    answering with it today would be wrong."""
    namespace = ns()
    async with pool.acquire() as conn:
        future = await insert_proposition(
            conn,
            text="expenses policy raises the mileage rate",
            namespace=namespace,
            valid_from=ahead(days=30),
        )
        found = await current_ids(conn, q="expenses policy", namespace=namespace)

    assert future not in found


async def test_a_claim_whose_validity_has_closed_is_not_retrieved(
    pool: asyncpg.Pool,
) -> None:
    """Closed validity is the world having moved on, and the default mode
    answers about the world now."""
    namespace = ns()
    async with pool.acquire() as conn:
        expired = await insert_proposition(
            conn,
            text="vendor contract covers weekend support",
            namespace=namespace,
            valid_from=ago(days=400),
            valid_to=ago(days=10),
        )
        found = await current_ids(conn, q="vendor contract", namespace=namespace)

    assert expired not in found


async def test_as_of_validity_answers_about_the_past_world(pool: asyncpg.Pool) -> None:
    """Same belief, different instant: what was true then, according to what we
    know now.  This is the mode that makes a closed row still useful."""
    namespace = ns()
    async with pool.acquire() as conn:
        old = await insert_proposition(
            conn,
            text="tariff band is seventeen",
            namespace=namespace,
            valid_from=ago(days=400),
            valid_to=ago(days=100),
        )
        new = await insert_proposition(
            conn,
            text="tariff band is twenty",
            namespace=namespace,
            valid_from=ago(days=100),
        )

        now_found = await current_ids(conn, q="tariff band", namespace=namespace)
        then_found = await valid_at_ids(
            conn, q="tariff band", namespace=namespace, at=ago(days=200)
        )

    assert now_found == {new}
    assert then_found == {old}


async def test_contradiction_closes_validity_instead_of_deleting(
    pool: asyncpg.Pool,
) -> None:
    """A claim the world has contradicted was still true once.  Closing
    valid_to keeps the row auditable and answerable as-of; deleting it throws
    away the only evidence of what the answer used to be."""
    namespace = ns()
    effective = ago(days=1)
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(
            conn,
            text="office badge opens the north door",
            namespace=namespace,
            valid_from=ago(days=200),
        )

        closed = await conn.fetchval(
            "SELECT pgkg_contradict($1, $2)", prop_id, effective
        )

        row = await temporal(conn, prop_id)
        now_found = await current_ids(conn, q="office badge", namespace=namespace)
        then_found = await valid_at_ids(
            conn, q="office badge", namespace=namespace, at=ago(days=50)
        )

    assert closed == 1
    assert row["valid_to"] == effective
    assert row["invalidated_at"] is None
    assert prop_id not in now_found
    assert then_found == {prop_id}


async def test_validity_interval_must_run_forwards(pool: asyncpg.Pool) -> None:
    """An interval that ends before it starts matches in no mode at all, so it
    is a data error rather than an empty answer."""
    namespace = ns()
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await insert_proposition(
                conn,
                text="impossible interval",
                namespace=namespace,
                valid_from=ago(days=1),
                valid_to=ago(days=2),
            )


# ---------------------------------------------------------------------------
# Belief: the audit path, deliberately off the hot path
# ---------------------------------------------------------------------------


async def test_as_of_belief_returns_what_the_system_believed_then(
    pool: asyncpg.Pool,
) -> None:
    """The question is not what was true, it is what we would have answered.  A
    fact we have since withdrawn has to come back for that instant."""
    namespace = ns()
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(
            conn,
            text="deprecated endpoint accepts a query parameter",
            namespace=namespace,
            recorded_at=ago(days=60),
        )
        await conn.execute(
            """
            UPDATE propositions
            SET invalidated_at = $2, invalidation_reason = 'contradicted'
            WHERE id = $1
            """,
            prop_id,
            ago(days=10),
        )

        current = await current_ids(
            conn, q="deprecated endpoint", namespace=namespace
        )
        believed = await believed_at_ids(conn, namespace=namespace, at=ago(days=30))

    assert prop_id not in current
    assert prop_id in believed


async def test_as_of_belief_excludes_what_was_not_yet_recorded(
    pool: asyncpg.Pool,
) -> None:
    """Belief time runs forwards too: a fact learned yesterday cannot explain an
    answer given last month."""
    namespace = ns()
    async with pool.acquire() as conn:
        later = await insert_proposition(
            conn,
            text="new runbook step",
            namespace=namespace,
            recorded_at=ago(days=2),
        )
        believed = await believed_at_ids(conn, namespace=namespace, at=ago(days=30))

    assert later not in believed


async def test_as_of_belief_ignores_validity_and_reports_the_reason(
    pool: asyncpg.Pool,
) -> None:
    """The audit answer includes claims whose world-validity has since closed,
    with the reason belief changed, because that is the explanation being
    sought."""
    namespace = ns()
    async with pool.acquire() as conn:
        prop_id = await insert_proposition(
            conn,
            text="seasonal rota covers august",
            namespace=namespace,
            recorded_at=ago(days=90),
            valid_from=ago(days=90),
            valid_to=ago(days=30),
        )
        rows = await conn.fetch(AS_OF_BELIEF, ago(days=60), namespace)

    reasons = {r["proposition_id"]: r["invalidation_reason"] for r in rows}
    assert prop_id in reasons
    assert reasons[prop_id] is None


# ---------------------------------------------------------------------------
# Retention: policy, not truth
# ---------------------------------------------------------------------------


async def test_expiry_withdraws_due_rows_but_respects_legal_hold(
    pool: asyncpg.Pool,
) -> None:
    """Retention is a fourth clock precisely because it answers to policy: a
    row under legal hold outlives its own TTL."""
    namespace = ns()
    async with pool.acquire() as conn:
        due = await insert_proposition(
            conn,
            text="transient session note",
            namespace=namespace,
            expires_at=ago(days=1),
        )
        held = await insert_proposition(
            conn,
            text="transient session note under hold",
            namespace=namespace,
            expires_at=ago(days=1),
            legal_hold=True,
        )
        fresh = await insert_proposition(
            conn,
            text="transient session note still current",
            namespace=namespace,
            expires_at=ahead(days=30),
        )

        expired = await conn.fetchval("SELECT pgkg_expire_due($1)", namespace)

        due_row = await temporal(conn, due)
        held_row = await temporal(conn, held)
        found = await current_ids(conn, q="transient session", namespace=namespace)

    assert expired == 1
    assert due_row["invalidation_reason"] == "ttl"
    assert held_row["invalidated_at"] is None
    assert found == {held, fresh}
