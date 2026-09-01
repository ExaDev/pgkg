"""Gazetteer matching: the corpus joins the graph without a model (ADR 0001, D2).

D2 prices three edges between a corpus and a knowledge graph.  This module owns
the cheap one.  Extracting propositions from a corpus costs hundreds of dollars
per pass, recurs on every prompt version bump and multiplies by tenant; matching
an org's own entity names against each passage costs an index probe per
candidate phrase, and the indexes have been sitting on `entities` unread since
migration 002.

Nothing here calls an embedder, a reranker or an extractor, and that is a
property worth stating rather than merely observing: the moment this path takes
a model call it stops being affordable to run on every ingest, and the mention
edge stops existing for most of the corpus.

Two directions, because ingest order forces both.  A passage can only mention
entities that exist when it arrives, and the entity a chat fact creates arrives
afterwards — so `match_chunks` handles new passages against the standing name
list, and `match_entities` handles new names against the standing corpus.  The
work itself is one SQL function either way; only the candidate generator
differs.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

import asyncpg

from pgkg.config import DEFAULT_ORG_ID, ORG_GUC

# Longer than any entity name worth matching as one phrase, and short enough
# that phrase generation stays linear in the passage.
DEFAULT_MAX_WORDS = 5

# Conservative on purpose.  The trigram arm exists for the typo and the missing
# accent, not for guessing: a false mention is an edge that pulls an unrelated
# document into every query about the entity it was mistaken for.
DEFAULT_THRESHOLD = 0.9

# One batch of a scheduled sweep, not a whole corpus.
DEFAULT_BATCH = 1000

_SET_ORG_SQL = f"SELECT set_config('{ORG_GUC}', $1, false)"
_MATCH_CHUNKS_SQL = "SELECT pgkg_match_entity_mentions($1::uuid[], $2, $3)"
_MATCH_ENTITIES_SQL = "SELECT pgkg_match_chunk_mentions($1::uuid[], $2, $3, $4)"
_PENDING_SQL = "SELECT id FROM pgkg_unmatched_chunks($1, $2) AS t(id)"


@dataclass(frozen=True)
class MatchResult:
    """What one call did, in the two numbers a scheduler cares about.

    `chunks_scanned` is the work; `mentions_added` is the yield.  A settled
    corpus drives the first to zero, which is what makes the sweep re-runnable
    on a timer rather than something to remember not to run twice.
    """

    chunks_scanned: int
    mentions_added: int


class Gazetteer:
    """One org's name list, matched against one org's passages.

    Tenancy is bound to the object rather than passed per call, as it is on
    Memory and CorpusIngest: a default argument cannot fail loudly, and a call
    site that forgets the org would write edges across a customer boundary.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        org_id: UUID = DEFAULT_ORG_ID,
        max_words: int = DEFAULT_MAX_WORDS,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._pool = pool
        self._org_id = org_id
        self._max_words = max_words
        self._threshold = threshold

    @property
    def org_id(self) -> UUID:
        return self._org_id

    def for_org(self, org_id: UUID) -> Gazetteer:
        return Gazetteer(
            self._pool,
            org_id=org_id,
            max_words=self._max_words,
            threshold=self._threshold,
        )

    async def match_chunks(self, chunk_ids: Iterable[UUID]) -> MatchResult:
        """New passages, against every name this org already knows."""
        ids = list(chunk_ids)
        if not ids:
            return MatchResult(chunks_scanned=0, mentions_added=0)

        async with self._connection() as conn:
            added = await conn.fetchval(
                _MATCH_CHUNKS_SQL, ids, self._max_words, self._threshold
            )
        return MatchResult(chunks_scanned=len(ids), mentions_added=added)

    async def match_entities(
        self, entity_ids: Iterable[UUID], *, max_chunks: int = DEFAULT_BATCH
    ) -> MatchResult:
        """New names, against the passages already stored.

        The scan count is not the corpus: the chunk tsvector index proposes a
        short list per name and only that list is read.
        """
        ids = list(entity_ids)
        if not ids:
            return MatchResult(chunks_scanned=0, mentions_added=0)

        async with self._connection() as conn:
            added = await conn.fetchval(
                _MATCH_ENTITIES_SQL,
                ids, max_chunks, self._max_words, self._threshold,
            )
        return MatchResult(chunks_scanned=len(ids), mentions_added=added)

    async def sweep(self, *, limit: int = DEFAULT_BATCH) -> MatchResult:
        """One batch of the passages this org has never matched.

        The watermark lives on the chunk, so a corpus that has settled reports
        no work rather than repeating it.
        """
        async with self._connection() as conn:
            pending = [
                row["id"]
                for row in await conn.fetch(_PENDING_SQL, self._org_id, limit)
            ]
            if not pending:
                return MatchResult(chunks_scanned=0, mentions_added=0)
            added = await conn.fetchval(
                _MATCH_CHUNKS_SQL, pending, self._max_words, self._threshold
            )
        return MatchResult(chunks_scanned=len(pending), mentions_added=added)

    def _connection(self):
        return _ScopedConnection(self._pool, self._org_id)


class _ScopedConnection:
    """A pooled connection carrying this org in the RLS GUC.

    The same shape Memory uses, and for the same reason: the policies on
    entity_mentions and on both endpoint tables have nothing to read unless the
    connection says who is asking.
    """

    def __init__(self, pool: asyncpg.Pool, org_id: UUID) -> None:
        self._pool = pool
        self._org_id = org_id
        self._ctx = None

    async def __aenter__(self) -> asyncpg.Connection:
        self._ctx = self._pool.acquire()
        conn = await self._ctx.__aenter__()
        await conn.execute(_SET_ORG_SQL, str(self._org_id))
        return conn

    async def __aexit__(self, *exc) -> bool | None:
        return await self._ctx.__aexit__(*exc)
