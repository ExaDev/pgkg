#!/usr/bin/env python3
"""Convert a folder of documents to text and load it into a pgkg collection.

pgkg's ingest paths all take text; nothing in the package reads .docx, .xlsx,
.pptx or .pdf. This is the missing front half, kept out of `pgkg/` because it
drags in a conversion stack that a server serving retrieval does not need.

MARKDOWN IS THE WRONG INTERCHANGE FORMAT, AND THIS SCRIPT USES IT ANYWAY.
Markdown is a 1D serialisation of a document, which is lossy for anything with
a 2D layout: a spreadsheet cell means one thing with its row label and column
header attached and nothing without them, and any projection to a text stream
must choose which spatial relationships to discard. ADR-0002 works out what a
structured reader should provide instead, and
ExaDev/documents.js#823 requests it. Until that lands, this routes between two
markdown converters, and the routing is a comparison of two lossy projections
rather than a verdict on either reader.

ROUTED BY FORMAT, PROVISIONALLY. documents.js extracts substantially more text
from PDFs and recovers heading structure where markitdown recovers none — the
chunker keys boundaries on structure, so that is a chunk-quality difference and
not merely a volume one. It also reads OOXML packages that fail a strict zip
reader. markitdown is preferred for spreadsheets because its markdown keeps
table rows as table rows, where documents.js flattens cells; that is a fact
about the two markdown emitters and NOT evidence that either reader is
deficient, which is the mistake the first version of this comment made.

Each falls back to the other, because they fail on different files: a .docx
that is not a standard package internally may defeat one and not the other.

The per-file winner is reported, so a fidelity regression in either converter
is visible rather than silent.

--dump-tree writes documents.js's own DocumentTree beside the text, which is
the lossless form the structured path will consume. Nothing reads it yet; it is
there so the work in ADR-0002 can be developed against real trees rather than
against a guess about their shape.

    scripts/ingest_dir.py PATH --dry-run          # what would be loaded
    scripts/ingest_dir.py PATH                    # load it, no LLM involved
    scripts/ingest_dir.py PATH --extract          # also extract propositions

WHERE THE DATA GOES. Conversion, chunking, embedding and reranking are all
local: bge-m3 and bge-reranker run on this machine. Nothing leaves it unless
you pass --extract, which sends chunk text to the configured LLM provider.
ADR-0001 D2 makes corpus extraction opt-in per collection for cost and quality
reasons; for third-party material it is also a disclosure decision, so the flag
says so and defaults off.

External ids are `file://<path relative to the root>`, so re-running is a no-op
for unchanged files and re-converts only what changed — the property D6 relies
on for crawlers.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from uuid import UUID

DOCUMENTS_JS_VERSION = "6.1.2"

CONVERTIBLE = {".docx", ".xlsx", ".pptx", ".pdf", ".md", ".txt", ".html", ".htm", ".csv"}

# Preference order per format. Provisional: it ranks two markdown projections,
# not two readers. See the module docstring.
ROUTING = {
    ".pdf":  ("documents_js", "markitdown"),
    ".pptx": ("documents_js", "markitdown"),
    ".docx": ("documents_js", "markitdown"),
    ".xlsx": ("markitdown", "documents_js"),
    ".csv":  ("markitdown", "documents_js"),
}
DEFAULT_ROUTE = ("markitdown", "documents_js")

# Decks repeat slide-master chrome across slides. Deduplicating whole lines
# keeps the first occurrence of each, which is the slide that introduced it.
# Note this catches only exact repetition; near-duplicate blocks survive it.
DEDUP_FORMATS = {".pptx"}


@dataclass
class Converted:
    path: pathlib.Path
    text: str
    converter: str = "-"
    repaired: bool = False
    deduped: int = 0
    error: str | None = None
    fell_back: bool = False


def _repair_zip(path: pathlib.Path) -> pathlib.Path | None:
    """Rewrite an OOXML package without its unreadable members.

    Office files are zips, and one member with a bad CRC fails the whole
    document — a corrupt embedded font is enough to lose an entire deck. Fonts,
    thumbnails and media carry no text, so dropping what will not read loses
    nothing that ingest wants and recovers the slides.
    """
    try:
        with zipfile.ZipFile(path) as src:
            keep, dropped = [], []
            for info in src.infolist():
                try:
                    keep.append((info, src.read(info.filename)))
                except Exception:
                    dropped.append(info.filename)
            if not dropped:
                return None
            tmp = pathlib.Path(tempfile.mkdtemp()) / path.name
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
                for info, data in keep:
                    out.writestr(info, data)
            return tmp
    except Exception:
        return None


def _via_markitdown(path: pathlib.Path) -> tuple[str, bool]:
    """Returns (text, repaired). Retries once without unreadable zip members."""
    from markitdown import MarkItDown

    md = MarkItDown(enable_plugins=False)
    try:
        return md.convert(str(path)).text_content, False
    except Exception:
        mended = _repair_zip(path)
        if mended is None:
            raise
        try:
            return md.convert(str(mended)).text_content, True
        finally:
            shutil.rmtree(mended.parent, ignore_errors=True)


def dump_tree(path: pathlib.Path, dest_dir: pathlib.Path) -> str | None:
    """Write documents.js's DocumentTree for one file. Returns an error or None.

    The tree is the lossless form: cells with addresses, formulas, comments and
    merged ranges, rather than a text stream that had to discard them. Nothing
    in pgkg consumes it yet (ADR-0002 D1), so this exists to give that work real
    material instead of an assumption about the shape.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower().lstrip(".")
    tree = dest_dir / f"{path.stem}.tree.json"
    scratch = pathlib.Path(tempfile.mkdtemp()) / "ignored.md"
    try:
        proc = subprocess.run(
            ["npx", "-y", f"documents.js@{DOCUMENTS_JS_VERSION}",
             f"{ext}-to-markdown", str(path), str(scratch),
             "--dump-package", str(tree)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0 or not tree.exists():
            detail = (proc.stderr or proc.stdout or "no output").strip().splitlines()
            return detail[-1][:120] if detail else "failed"
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(scratch.parent, ignore_errors=True)


def _via_documents_js(path: pathlib.Path) -> tuple[str, bool]:
    ext = path.suffix.lower().lstrip(".")
    out = pathlib.Path(tempfile.mkdtemp()) / "out.md"
    try:
        proc = subprocess.run(
            ["npx", "-y", f"documents.js@{DOCUMENTS_JS_VERSION}",
             f"{ext}-to-markdown", str(path), str(out)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0 or not out.exists():
            detail = (proc.stderr or proc.stdout or "no output").strip().splitlines()
            raise RuntimeError(detail[-1][:120] if detail else "failed")
        return out.read_text(errors="ignore"), False
    finally:
        shutil.rmtree(out.parent, ignore_errors=True)


CONVERTERS = {"markitdown": _via_markitdown, "documents_js": _via_documents_js}


def _dedup_lines(text: str) -> tuple[str, int]:
    seen: set[str] = set()
    kept: list[str] = []
    dropped = 0
    for line in text.splitlines():
        key = line.strip()
        if key and key in seen:
            dropped += 1
            continue
        if key:
            seen.add(key)
        kept.append(line)
    return "\n".join(kept), dropped


def convert(path: pathlib.Path) -> Converted:
    """Try this format's preferred converter, then the other one.

    Both are tried because they fail on different files, and a corpus loader
    that gives up on the first error loses whole documents to one bad member or
    one non-standard package.
    """
    order = ROUTING.get(path.suffix.lower(), DEFAULT_ROUTE)
    errors = []
    for attempt, name in enumerate(order):
        try:
            text, repaired = CONVERTERS[name](path)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__} {str(exc)[:60]}")
            continue
        dropped = 0
        if path.suffix.lower() in DEDUP_FORMATS:
            text, dropped = _dedup_lines(text)
        return Converted(path, text, converter=name, repaired=repaired,
                         deduped=dropped, fell_back=attempt > 0)
    return Converted(path, "", error="; ".join(errors))


def gather(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in CONVERTIBLE
    )


async def run(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.path).expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    files = gather(root)
    if not files:
        print(f"no convertible files under {root}")
        return 1

    converted = [convert(p) for p in files]
    ok = [c for c in converted if not c.error]
    bad = [c for c in converted if c.error]

    print(f"\n{'file':40} {'via':13} {'chars':>8} {'chunks':>7}  note")
    for c in converted:
        rel = str(c.path.relative_to(root))[:40]
        if c.error:
            print(f"{rel:40} {'-':13} {'-':>8} {'-':>7}  SKIPPED {c.error}")
        else:
            note = []
            if c.fell_back:
                note.append("fell back")
            if c.repaired:
                note.append("repaired zip")
            if c.deduped:
                note.append(f"-{c.deduped} dup lines")
            print(f"{rel:40} {c.converter:13} {len(c.text):>8} "
                  f"{max(1, len(c.text)//1200):>7}  {', '.join(note)}")
    if args.dump_tree:
        dest = pathlib.Path(args.dump_tree).expanduser()
        wrote = 0
        print()
        for c in converted:
            err = dump_tree(c.path, dest)
            if err:
                print(f"  tree FAILED {str(c.path.relative_to(root))[:44]}: {err}")
            else:
                wrote += 1
        print(f"  {wrote}/{len(converted)} DocumentTrees written to {dest}")

    total = sum(len(c.text) for c in ok)
    print(f"\n  {len(ok)} convertible, {len(bad)} skipped, {total:,} chars "
          f"(~{total//4:,} tokens, ~{total//1200:,} chunks)")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    if args.extract:
        print("\n  --extract: chunk text WILL be sent to the configured LLM provider")

    from pgkg.config import DEFAULT_COLLECTION_ID, DEFAULT_ORG_ID, get_settings
    from pgkg.corpus import CorpusIngest
    from pgkg.db import close_pool, make_pool
    from pgkg.memory import Provenance

    pool = await make_pool(get_settings().database_url)
    try:
        ingest = CorpusIngest(
            pool,
            org_id=UUID(args.org) if args.org else DEFAULT_ORG_ID,
            collection_id=UUID(args.collection) if args.collection else DEFAULT_COLLECTION_ID,
        )
        print(f"\n{'file':44} {'new':>5} {'carr':>5} {'emb':>5} {'cache':>6} {'props':>6}")
        totals = dict(new=0, carried=0, embedded=0, cache=0, props=0, changed=0)
        for c in ok:
            rel = str(c.path.relative_to(root))
            result = await ingest.upsert_document(
                external_id=f"file://{rel}",
                text=c.text,
                uri=str(c.path),
                provenance=Provenance(kind="document_version", producer="chunker"),
            )
            totals["new"] += result.chunks_new
            totals["carried"] += result.chunks_carried
            totals["embedded"] += result.embedded
            totals["cache"] += result.cache_hits
            totals["props"] += result.propositions
            totals["changed"] += int(result.changed)
            print(f"{rel[:44]:44} {result.chunks_new:>5} {result.chunks_carried:>5} "
                  f"{result.embedded:>5} {result.cache_hits:>6} {result.propositions:>6}"
                  f"{'' if result.changed else '   (unchanged)'}")
        print(f"\n  {totals['changed']}/{len(ok)} changed; {totals['new']} new chunks, "
              f"{totals['carried']} carried, {totals['embedded']} embedded, "
              f"{totals['cache']} cache hits, {totals['props']} propositions")
    finally:
        await close_pool(pool)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", help="Directory to walk")
    p.add_argument("--dry-run", action="store_true", help="Convert and report; write nothing")
    p.add_argument("--extract", action="store_true",
                   help="Also extract propositions — sends chunk text to the LLM provider")
    p.add_argument("--dump-tree", metavar="DIR",
                   help="Also write documents.js's DocumentTree per file into DIR "
                        "(the lossless form; nothing consumes it yet — see ADR-0002)")
    p.add_argument("--org", help="Org UUID (default: the reserved default org)")
    p.add_argument("--collection", help="Collection UUID (default: the reserved default)")
    raise SystemExit(asyncio.run(run(p.parse_args())))


if __name__ == "__main__":
    main()
