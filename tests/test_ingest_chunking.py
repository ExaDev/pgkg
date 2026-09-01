"""ingest() must use content-defined chunk boundaries.

Greedy packing decides every boundary from how much text has accumulated, so a
paragraph inserted near the top of a document shifts every boundary after it.
Phase 0 of ADR 0001 replaces that with boundaries chosen from local content, so
re-ingesting an edited document rewrites only the chunks near the edit — which
is what chunk-level dedup and the extraction cache both need.

These tests exercise the property through Memory.ingest() and the rows it
writes, not through the chunker's own unit surface (tests/test_chunking.py).
"""
from __future__ import annotations

import hashlib
import random
import uuid

import asyncpg
import pytest

from pgkg import ml

WORDS = (
    "mitochondrion ribosome fermentation zymurgy catalysis substrate enzyme "
    "kinetics gradient membrane vesicle organelle chromatin telomere plasmid "
    "operon codon anneal denature buffer titrate reagent aliquot centrifuge"
).split()


@pytest.fixture(scope="session")
async def embed_dim(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT pgkg_embedding_dim('propositions', 'embedding')"
        )


def _make_embed(dim: int):
    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            v = [0.0] * dim
            v[int.from_bytes(digest[:4], "big") % dim] = 1.0
            out.append(v)
        return out

    return embed


def _ns(tag: str) -> str:
    return f"ingchunk_{tag}_{uuid.uuid4().hex[:8]}"


def _document(seed: int = 11, sections: int = 24) -> str:
    rng = random.Random(seed)
    parts: list[str] = []
    for section in range(sections):
        parts.append(f"## Section {section}")
        for _ in range(rng.randint(2, 5)):
            parts.append(
                " ".join(rng.choice(WORDS) for _ in range(rng.randint(10, 55))) + "."
            )
    return "\n\n".join(parts) + "\n"


async def _chunk_rows(
    pool: asyncpg.Pool, namespace: str
) -> list[tuple[str, int, int]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.text, c.span_start, c.span_end
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.namespace = $1
            ORDER BY c.span_start
            """,
            namespace,
        )
    return [(r["text"], r["span_start"], r["span_end"]) for r in rows]


async def _ingest(pool: asyncpg.Pool, namespace: str, text: str) -> None:
    """Through the extraction path, because that is the one that keeps spans.

    The subject here is the chunker, and either ingest mode reaches it — but a
    chunks-only ingest now writes into the chunk store, where a passage shared
    by two documents has no one offset into no one text and carries no
    span_start (ADR 0001, D6).  The offline extractor keeps this a no-LLM test.
    """
    from pgkg.memory import Memory

    await Memory(pool, namespace=namespace).ingest(text)


async def test_stored_spans_locate_the_chunk_in_the_source_document(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """span_start/span_end must be real offsets into the ingested text.

    They used to be i*chunk_size placeholders, which cannot address a chunk
    whose boundaries are content-defined, and which small-to-big context
    expansion needs in order to widen a hit back out to its neighbourhood.
    """
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    document = _document()
    namespace = _ns("spans")
    await _ingest(pool, namespace, document)

    rows = await _chunk_rows(pool, namespace)

    assert len(rows) > 1
    for text, start, end in rows:
        assert document[start:end] == text
    starts = [start for _, start, _ in rows]
    assert starts == sorted(starts)
    assert all(a_end <= b_start for (_, _, a_end), (_, b_start, _) in zip(rows, rows[1:]))


def _short_paragraphs(seed: int = 0, count: int = 200) -> list[str]:
    """A bullet-list shape: many short paragraphs, no headings.

    This is where a length-driven chunker is worst: every paragraph is far
    smaller than the cap, so one inserted line shifts the packing of every
    chunk after it and nothing pulls the boundaries back into step.
    """
    rng = random.Random(seed)
    return [
        "- " + " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 12))) + "."
        for _ in range(count)
    ]


async def test_inserting_a_paragraph_near_the_top_preserves_later_chunks(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """Re-ingesting an edited document must reuse most chunk texts verbatim.

    Measured on this document over these four insertion points: the greedy
    packer this replaces preserved 0% of chunk texts at every one of them,
    because each boundary depends on all preceding text; content-defined
    boundaries preserved 92-96%. The 85% floor therefore sits far below the new
    behaviour and far above anything a length-driven chunker reaches here.
    """
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    paragraphs = _short_paragraphs()
    original = "\n\n".join(paragraphs) + "\n"

    ns_before = _ns("stable_before")
    await _ingest(pool, ns_before, original)
    before = [text for text, _, _ in await _chunk_rows(pool, ns_before)]
    assert len(before) >= 10

    inserted = "- An inserted line about zymurgy fermentation kinetics."
    for position in (1, 4, 7, 12):
        edited = (
            "\n\n".join(
                paragraphs[:position] + [inserted] + paragraphs[position:]
            )
            + "\n"
        )
        namespace = _ns(f"stable_after_{position}")
        await _ingest(pool, namespace, edited)
        after = {text for text, _, _ in await _chunk_rows(pool, namespace)}

        survivors = [text for text in before if text in after]
        fraction = len(survivors) / len(before)
        assert fraction >= 0.85, (
            f"inserting at paragraph {position} preserved only "
            f"{fraction:.0%} of chunk texts"
        )


async def test_a_heading_starts_a_new_chunk(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    """Structural markers, not accumulated length, decide where a chunk opens."""
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    document = (
        "## Alpha\n\nA short paragraph about ribosomes.\n\n"
        "## Beta\n\nA short paragraph about plasmids.\n\n"
        "## Gamma\n\nA short paragraph about telomeres.\n"
    )
    namespace = _ns("headings")
    await _ingest(pool, namespace, document)

    texts = [text for text, _, _ in await _chunk_rows(pool, namespace)]

    assert len(texts) == 3
    assert [text.splitlines()[0] for text in texts] == [
        "## Alpha",
        "## Beta",
        "## Gamma",
    ]


async def test_no_chunk_exceeds_the_requested_size(
    pool: asyncpg.Pool, embed_dim: int, monkeypatch
) -> None:
    monkeypatch.setattr(ml, "embed", _make_embed(embed_dim))

    from pgkg.memory import Memory

    namespace = _ns("cap")
    await Memory(pool, namespace=namespace).ingest(
        _document(), chunk_size=400
    )

    rows = await _chunk_rows(pool, namespace)

    assert len(rows) > 1
    assert all(len(text) <= 400 for text, _, _ in rows)
