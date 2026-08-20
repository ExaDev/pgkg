"""Content-defined chunking.

Greedy packing makes every chunk boundary depend on all preceding text, so one
paragraph inserted near the top of a document reshuffles every boundary after it
and rehashes every chunk. Content-addressed chunks (ADR 0001, D6) and the
extraction cache both need the opposite: boundaries chosen from local structure,
so a local edit perturbs only local chunks.

Boundaries here are decided per segment from that segment's own content —
a structural marker (a heading) or a hash cut point — never from how much text
has accumulated. Accumulated length only ever *forces* a cut at the size cap,
and the next content-defined cut point resynchronises the stream.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

DEFAULT_MAX_CHARS = 1200

# One segment in three is a cut point. Combined with min_chars this puts the
# expected chunk somewhere between the floor and the cap without the boundary
# decision ever depending on where the previous boundary fell.
_CUT_POINT_MODULUS = 3

_PARAGRAPH_SEPARATOR = re.compile(r"\n[ \t]*\n")
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+")
_HEADING = re.compile(r"[ \t]{0,3}#{1,6}[ \t]+\S")


@dataclass(frozen=True)
class Chunk:
    """An immutable slice of a document, addressed by the hash of its text."""

    ordinal: int
    text: str
    content_hash: str
    span_start: int
    span_end: int


@dataclass(frozen=True)
class _Segment:
    start: int
    end: int
    text: str


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int | None = None,
) -> tuple[Chunk, ...]:
    """Split text into content-addressed chunks, in document order."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    floor = max(1, max_chars // 3) if min_chars is None else min_chars
    if floor > max_chars:
        raise ValueError("min_chars must not exceed max_chars")

    groups = _group(tuple(_segments(text, max_chars)), max_chars, floor)
    return tuple(
        Chunk(
            ordinal=ordinal,
            text=text[start:end],
            content_hash=content_hash(text[start:end]),
            span_start=start,
            span_end=end,
        )
        for ordinal, (start, end) in enumerate(groups)
    )


def _segments(text: str, max_chars: int) -> list[_Segment]:
    """Atomic units a boundary may fall between: paragraphs, then sentences."""
    return [
        piece
        for start, end in _paragraph_spans(text)
        for stripped in (_strip_span(text, start, end),)
        if stripped is not None
        for piece in _split_to_cap(text, *stripped, max_chars)
    ]


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for separator in _PARAGRAPH_SEPARATOR.finditer(text):
        spans.append((cursor, separator.start()))
        cursor = separator.end()
    spans.append((cursor, len(text)))
    return spans


def _strip_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _split_to_cap(
    text: str, start: int, end: int, max_chars: int
) -> list[_Segment]:
    if end - start <= max_chars:
        return [_Segment(start, end, text[start:end])]

    pieces: list[_Segment] = []
    cursor = start
    for boundary in _sentence_boundaries(text, start, end):
        if boundary <= cursor:
            continue
        pieces.extend(_hard_split(text, cursor, boundary, max_chars))
        cursor = boundary
    pieces.extend(_hard_split(text, cursor, end, max_chars))
    return pieces


def _sentence_boundaries(text: str, start: int, end: int) -> list[int]:
    return [
        start + match.end() for match in _SENTENCE_END.finditer(text[start:end])
    ]


def _hard_split(
    text: str, start: int, end: int, max_chars: int
) -> list[_Segment]:
    """Last resort for a run with no internal boundary: cut at the cap."""
    return [
        _Segment(cut, min(cut + max_chars, end), text[cut : min(cut + max_chars, end)])
        for cut in range(start, end, max_chars)
    ]


def _group(
    segments: tuple[_Segment, ...], max_chars: int, min_chars: int
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    current: tuple[_Segment, ...] = ()

    for segment in segments:
        if current and _cuts_before(segment, current, max_chars, min_chars):
            spans.append((current[0].start, current[-1].end))
            current = ()
        current = current + (segment,)

    if current:
        spans.append((current[0].start, current[-1].end))
    return spans


def _cuts_before(
    segment: _Segment,
    current: tuple[_Segment, ...],
    max_chars: int,
    min_chars: int,
) -> bool:
    start = current[0].start
    if segment.end - start > max_chars:
        return True
    if _is_heading(segment.text):
        # A section start always opens a chunk, but never leaves its own heading
        # behind as an orphan chunk.
        return any(not _is_heading(held.text) for held in current)
    return current[-1].end - start >= min_chars and _is_cut_point(segment.text)


def _is_heading(segment_text: str) -> bool:
    return _HEADING.match(segment_text) is not None


def _is_cut_point(segment_text: str) -> bool:
    digest = hashlib.blake2b(segment_text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _CUT_POINT_MODULUS == 0
