"""Behaviour of the content-defined chunker.

The property that matters is boundary stability: a local edit must perturb only
local chunks, otherwise chunk-level dedup and the extraction cache both collapse
to a 0% hit rate on document updates (ADR 0001, D6).
"""

from __future__ import annotations

import hashlib
import random
import re

import pytest

from pgkg.chunking import (
    DEFAULT_MAX_CHARS,
    Chunk,
    chunk_document,
    content_hash,
)

WORDS = (
    "mitochondrion ribosome fermentation zymurgy catalysis substrate enzyme "
    "kinetics gradient membrane vesicle organelle chromatin telomere plasmid "
    "operon codon anneal denature buffer titrate reagent aliquot centrifuge "
    "supernatant pellet lysate assay control replicate variance covariance"
).split()


def _paragraph(rng: random.Random, words: int | None = None) -> str:
    count = rng.randint(8, 60) if words is None else words
    return " ".join(rng.choice(WORDS) for _ in range(count)) + "."


def _document(seed: int = 7, sections: int = 30) -> str:
    """A sectioned document whose paragraphs vary from one line to a screenful."""
    rng = random.Random(seed)
    parts: list[str] = []
    for section in range(sections):
        parts.append(f"## Section {section}")
        for _ in range(rng.randint(2, 6)):
            parts.append(_paragraph(rng))
    return "\n\n".join(parts) + "\n"


def _prose(seed: int = 7, paragraphs: int = 160) -> str:
    """No headings at all — only the hash cut point can hold boundaries here."""
    rng = random.Random(seed)
    return "\n\n".join(_paragraph(rng) for _ in range(paragraphs)) + "\n"


def _hashes(text: str, **kwargs: int) -> list[str]:
    return [c.content_hash for c in chunk_document(text, **kwargs)]


def _insert_paragraph_after(text: str, marker: str, paragraph: str) -> str:
    index = text.index(marker) + len(marker)
    return text[:index] + "\n\n" + paragraph + text[index:]


def test_chunk_document_yields_ordinals_texts_and_hashes():
    chunks = chunk_document(_document())

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.content_hash == content_hash(c.text) for c in chunks)


def test_content_hash_identifies_content_not_position():
    text = _document()
    chunks = chunk_document(text)
    first = chunks[0]

    assert content_hash(first.text) == first.content_hash
    assert content_hash("alpha") == hashlib.sha256(b"alpha").hexdigest()
    assert content_hash("alpha") != content_hash("beta")


def test_local_insertion_preserves_overwhelming_majority_of_hashes():
    """A paragraph inserted near the top must not rehash the whole document.

    Only the chunk absorbing the edit — and at most its immediate successor,
    while the boundary rule resynchronises — may change. Measured over 15 seeds
    x 4 insertion points, the worst survival was 96% here and 93% on
    heading-free prose, so 90% is a floor with margin rather than a
    rediscovery of the current behaviour. It is still a bound a
    length-dependent chunker cannot meet: greedy paragraph packing measured 60%
    on this document shape and 0% on one built from short paragraphs, because
    every boundary after the edit shifts.
    """
    original = _document()
    edited = _insert_paragraph_after(
        original, "## Section 2", "An inserted paragraph about zymurgy and fermentation."
    )

    before = _hashes(original)
    after = set(_hashes(edited))
    survivors = [h for h in before if h in after]

    assert len(before) >= 20
    assert len(survivors) / len(before) >= 0.9


def test_insertion_only_perturbs_chunks_near_the_edit():
    original = _document()
    edited = _insert_paragraph_after(
        original, "## Section 2", "An inserted paragraph about zymurgy and fermentation."
    )

    before = _hashes(original)
    after = set(_hashes(edited))
    changed = [i for i, h in enumerate(before) if h not in after]

    assert changed
    assert max(changed) < len(before) * 0.2


def test_local_insertion_is_stable_without_structural_markers():
    original = _prose()
    paragraphs = original.split("\n\n")
    edited = "\n\n".join(
        paragraphs[:3] + ["An inserted paragraph about zymurgy."] + paragraphs[3:]
    )

    before = _hashes(original)
    after = set(_hashes(edited))
    changed = [i for i, h in enumerate(before) if h not in after]

    assert len(before) >= 20
    assert (len(before) - len(changed)) / len(before) >= 0.9
    assert max(changed) < len(before) * 0.2


def test_appending_content_leaves_earlier_chunks_untouched():
    original = _document()
    extended = original + "\n\n## Appendix\n\n" + _paragraph(random.Random(99))

    before = _hashes(original)
    after = _hashes(extended)

    assert after[: len(before) - 1] == before[: len(before) - 1]


def test_chunking_is_idempotent():
    text = _document()

    assert chunk_document(text) == chunk_document(text)


def test_size_cap_is_respected():
    for max_chars in (400, DEFAULT_MAX_CHARS):
        chunks = chunk_document(_document(), max_chars=max_chars)
        assert chunks
        assert all(len(c.text) <= max_chars for c in chunks)


def test_paragraph_longer_than_the_cap_is_split():
    rng = random.Random(3)
    giant = " ".join(_paragraph(rng, words=40) for _ in range(30))

    chunks = chunk_document(giant, max_chars=500)

    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_unbroken_run_longer_than_the_cap_is_split():
    chunks = chunk_document("x" * 2500, max_chars=300)

    assert all(len(c.text) <= 300 for c in chunks)
    assert "".join(c.text for c in chunks) == "x" * 2500


def test_no_chunk_is_empty_or_whitespace_only():
    chunks = chunk_document(_document())

    assert all(c.text.strip() for c in chunks)


def test_whitespace_only_input_yields_no_chunks():
    assert chunk_document("") == ()
    assert chunk_document("   \n\n\t  \n") == ()


def test_full_text_is_recoverable_from_chunks_in_order():
    text = _document()
    chunks = chunk_document(text)

    assert all(c.text == text[c.span_start : c.span_end] for c in chunks)
    assert not text[: chunks[0].span_start].strip()
    assert not text[chunks[-1].span_end :].strip()
    gaps = [
        text[a.span_end : b.span_start] for a, b in zip(chunks, chunks[1:])
    ]
    assert all(not gap.strip() for gap in gaps)
    joined = "".join(c.text for c in chunks)
    assert re.sub(r"\s+", "", joined) == re.sub(r"\s+", "", text)


def test_headings_begin_a_chunk():
    text = _document()
    chunks = chunk_document(text)

    for chunk in chunks:
        heading_lines = [
            i for i, line in enumerate(chunk.text.splitlines()) if line.startswith("## ")
        ]
        assert heading_lines in ([], [0])


def test_min_chars_cannot_exceed_max_chars():
    with pytest.raises(ValueError):
        chunk_document(_document(), max_chars=200, min_chars=400)
