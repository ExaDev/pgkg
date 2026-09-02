# ADR-0002: Spreadsheets are a canvas, not a table

**Status:** Proposed
**Date:** 2026-09-02
**Decision makers:** Will Faithfull
**Context:** The corpus pipeline of
[ADR-0001](0001-corpus-embeddings-and-knowledge-graph.md) has no sensible answer for `.xlsx`

---

## Summary

A spreadsheet is not a document, and it is not reliably a table either. It is **a canvas**: a
sparse grid on which someone put a table here, a paragraph of commentary there, a couple of
label/value pairs in the margin, and a stray note in `M42`. Any design that begins by asking
"what shape is this sheet" has already assumed more structure than the file has.

The governing principle of this ADR is therefore:

> **Classification adds retrieval paths. It never decides what survives.**

Every populated cell is captured losslessly first. Structure recognition then layers *better*
ways to reach that content on top, and when recognition fails — which it will, often — the
baseline still answers.

> **Revision note.** Two earlier drafts are recorded in "Alternatives rejected" because their
> reasoning is the useful part: the first materialised every cell as a graph proposition, the
> second classified whole sheets and then their columns. The first saturates the graph; the second
> assumes a tidiness that spreadsheets do not have.

---

## Context

### What prompted this

Loading a spreadsheet-heavy corpus. Two document converters disagreed about the same workbook by
more than an order of magnitude in extracted text, and in opposite directions on different sheets.

That comparison asked both of them for **markdown**, which was the mistake. Markdown is a 1D
serialisation of a 2D sparse canvas, and there is no faithful way to do that: any projection must
choose which spatial relationships to discard, and two reasonable implementations choose
differently. The disagreement was a property of the output format, not of either library.

So the useful question is not which markdown output is better. It is whether the canvas can be
had at all — which is a question for the reader, not for this pipeline. See "Where this belongs".

Then, more damningly: `40%` in a chunk is noise, where `Widget A | unit cost | 4.20` is a fact.
Flatten-and-chunk produces the former, because a cell's meaning lives in the cells around it and
chunking severs that.

### The questions people actually ask

| Question | What answers it |
|---|---|
| "What discount did we assume?" | A cell, with whatever labels its neighbours give it |
| "What are the stages?" | A column, coherently |
| "How is ROI calculated?" | The **formula** — converters strip these entirely |
| "Which rows are above a threshold?" | A filter over rows: this is SQL, not retrieval |
| "What did the reviewer note about Q3?" | A paragraph someone typed into a cell |
| "What's in this workbook?" | A summary — and permission to go look properly |

Six questions, at least four mechanisms. No single flattening serves them, which is why choosing
between converters felt arbitrary: both were answering a question nobody had asked.

### What real sheets look like

The cases the design has to survive, none of them pathological:

- A table in `A1:F20`, with a column of prose commentary in `H3:H10` that is not part of it.
- A sheet that is entirely narrative, laid out visually with merged cells and no header row.
- Label/value pairs scattered in margins: `B2: "Owner"`, `C2: "R. Chen"`, twelve rows down another pair.
- A calculator: inputs, formulas, outputs, no row semantics at all.
- Cells filled in ad hoc over months, with no organising structure whatsoever.
- Cell comments and notes — pure prose, dropped by every converter tested.

A sheet is frequently **several of these at once**. So the unit of analysis is a *region* of the
grid, and some content belongs to no region at all.

### Why structure must not gate capture

Region detection is a heuristic. Header detection is a heuristic. Both will be wrong, and the
failure mode of a structure-gated design is that unrecognised content is *silently dropped* —
the worst possible outcome, because nobody can tell the difference between "the sheet does not say
that" and "the parser did not understand that part of the sheet".

Inverting it costs almost nothing: capture everything, then enrich. A wrong classification then
degrades a retrieval path rather than losing data.

### Where this belongs

Most of this ADR is not pgkg's work to do, and saying so is part of the decision.

> **The reader answers "what is on this canvas." pgkg answers "how should this be retrievable."**

That seam puts the cell grid, the formulas, the comments, the merged ranges, the region
segmentation and the neighbour-derived labels on the **document reader's** side —
[ExaDev/documents.js#823](https://github.com/ExaDev/documents.js/issues/823) requests exactly
that. Three reasons beyond reuse: the reader already holds the grid in memory, so doing it here
would mean either re-parsing OOXML or accepting a poorer input; the reader emits *confidence*
while the consumer sets the *threshold* it trusts, which is a clean split of responsibility; and
regions generalise past spreadsheets, since a PDF has columns, tables and figures for the same
reason and by the same mechanism.

`document-schema.js` already carries the vocabulary — `a1.ts` with `CellPosition` and `CellRange`,
`SheetGroupNode` in the tree, `cellTable` and `merged` in the content schema, and a lossless
`DocumentTree` round trip through `--dump-package` / `from-package`. What appears to be missing is
the xlsx reader populating it: dumping the package for one spreadsheet produced a
`kind: wordprocessing` tree of sections, paragraphs and tables with no sheet nodes, no A1
references and no formulas.

**What stays in pgkg** is retrieval policy, and only that: which regions become chunks (D4, D5,
D9), the row cap against the corpus quota (D4), the mention edges into the graph (D10), the schema
of the cell store (D1), and `query_table` (D7). Those are decisions about a retrieval budget and a
graph's health, and they do not belong in a conversion library.

**Deployment consequence.** Consuming a structured reader means a Node dependency on the ingest
path. That is acceptable because ADR-0001 D7 already splits ingest in two: corpus ingest is a
batch worker and chat ingest is online. The dependency lands in the batch worker, not in the
retrieval API, so the two-container thesis survives. `--dump-package` JSON via the CLI, or
`document-mcp`, are both adequate contracts.

**Until that lands**, `scripts/ingest_dir.py` routes by format between two markdown converters.
That routing is a comparison of two lossy projections and is explicitly provisional: it says which
markdown is less bad per format, and nothing about which reader is better.

---

## Decision

### D1 — Capture every populated cell, losslessly, before interpreting anything

Every non-empty cell lands in a queryable store with its address (workbook, sheet, row, column),
its literal value, its type as authored, its formula if it has one, its number format, and any
comment or note attached to it. No interpretation, no classification, no exceptions.

This is the floor the rest of the design stands on. It alone answers cell lookups and filters via
D7, and it guarantees that no amount of misclassification can lose content.

### D2 — Derive a cell's labels from its neighbours, not from a required header row

A cell's meaning comes from the nearest text cells above it and to its left. That rule covers a
table (the header row is simply the nearest text above) *and* a scattered label/value pair *and* a
margin annotation, with no requirement that the sheet be tidy.

Store the derived labels with the cell, along with how far away they were found, because distance
is confidence: an adjacent label is strong, one eleven rows up is weak and should be marked so.

This replaces "detect the header row" as the primitive. Header detection becomes a special case of
neighbour derivation that happens to be strong and repeated down a column.

### D3 — Segment the grid into regions, and treat regions as advisory

Find connected components of populated cells, tolerating a blank row or column inside a block.
Classify each region — `table`, `prose`, `model`, `mixed`, `unknown` — with a confidence.

Regions are **advisory**: they add retrieval paths under D4–D6. Nothing depends on them being
right, and a sheet with five regions and no recognisable shape among them still works, via D1 and
D8.

### D4 — Confident table regions additionally get row chunks

Where a region is confidently tabular, each row becomes a chunk carrying its header context, so a
row is retrievable as a self-contained unit rather than as a fragment.

**Under a cap.** A few hundred rows becoming a few hundred chunks would swamp the 38-slot corpus
quota (`floor(k_rerank × corpus_fraction)`) and starve every other source — the failure ADR-0001
D1 exists to prevent, arriving from a direction D1 did not anticipate. Past the cap, the card (D8)
plus the query tool (D7) is the answer, not more chunks.

### D5 — Long-text cells additionally get prose chunks

A cell holding a paragraph is a document, wherever it sits: inside a table's Notes column, in a
prose region, in the margin, or alone in `M42`. It becomes a chunk carrying its derived labels
(D2) as context.

Participation is **not exclusive**: the same cell is a queryable value under D1 *and* a prose
chunk under D5. Nothing is served by forcing an either/or, and the earlier drafts' insistence on
one treatment per cell was a mistake.

Cell comments and notes are prose and belong here, attached to their cell's context.

### D6 — Formulas become retrievable text, once per distinct formula

`ROI = (revenue − cost) / cost` is the answer to "how is ROI calculated", and it is the same answer
for all ten thousand rows that share it. Render each *distinct* formula once, with the labels of
the cells it reads.

### D7 — Expose the captured cells through a query tool

The honest answer to filter-and-compare is that retrieval is the wrong mechanism. The data is
already in Postgres — which is the whole thesis — so expose `query_table` on the MCP surface
beside `search_corpus`.

Same scope-binding rules as the rest of that surface: the tenant comes from the server, never
from a tool argument, per `pgkg/mcp_server.py`. A read-only role and a column allowlist besides,
because this one runs queries shaped by model output.

### D8 — Every sheet and every region gets a card

A summary chunk: name, extent, what was recognised and what was not, column labels, types, ranges,
a couple of example rows. Cheap and always correct, because it describes rather than interprets.

Its more important job is routing — the card is what lets an agent decide to go and look properly
instead of guessing from fragments. This is ADR-0001's deferred summaries-as-a-source-class,
arriving first where it is needed most.

A card must state what it could not classify. "Sheet 3 has 40 populated cells in no recognised
structure" is useful; silence about them is not.

### D9 — Unrecognised cells are chunked by proximity, in reading order

For the ad-hoc sheet — cells filled in wherever, no structure to find — group nearby populated
cells into a chunk in reading order, each carrying its address and derived labels.

This is not a fallback so much as an admission that **for an unstructured sheet, spatial proximity
is the semantics.** Cells near each other are related; that is how the person who typed them
read them. A chunk reading `B4: Q3 target · B5: 40% · D9: chase the renewal` is retrievable,
useful, and honest about being unstructured.

### D10 — Connect the graph to the cells; never copy the cells into the graph

The graph gets **mention edges** — one per row key, or per region, for entities that appear in
conversation — in `entity_mentions`, alongside the chunk mentions ADR-0001 D2 already defines.
Not one proposition per cell.

That preserves the path the mention edge exists for: a chat fact seeds an entity, graph expansion
reaches the row or region, and the agent reads or queries it. What it avoids is duplicating a
table into the class ADR-0001 D1 protects with a floor.

Why copying is wrong, given the earlier draft proposed exactly that:

- **Saturation.** A 10,000 × 20 sheet is 200,000 triples from one file. ADR-0001's sizing model
  puts an entire organisation's conversational graph at ~3.6M propositions, so one mid-size sheet
  is about 5% of everything that organisation ever said.
- **It lands in the privileged class.** Propositions carry the protected floor in the rerank
  budget. Bulk triples starve the personal memory that floor defends, and a cap on chunks does not
  help because these are not chunks.
- **Predicate collapse into spurious contradictions.** 200,000 propositions share ~20 predicates.
  Contradiction resolution keys on `(subject, functional predicate)`, and entity dedup is gated by
  a cosine on short name strings that is noisy on exactly this input. Two row labels that merge
  when they should not turn their measures into competing assertions, and the newer
  mass-invalidates the older — silently and plausibly.
- **BM25 contamination.** Rows of near-identical text make every term in the table common, which
  strips discriminative power from the prose sharing the collection's statistics. The documents get
  worse because a spreadsheet arrived.
- **It is unnecessary.** Propositions exist because prose is not queryable: structure must be
  extracted before anything can be asked of it. A table already has structure. Copying it
  duplicates the data, degrades the statistics, and buys nothing D7 could not answer.

Materialising triples as propositions survives only as an explicit per-sheet opt-in under a hard
row budget, for a small human-designated reference table — a glossary, a code list, an org chart —
where none of the objections above has force. Default off.

---

## Consequences

**Positive.** Nothing is lost, whatever the classifiers do. Spreadsheet questions become answerable
at all, and the prose buried in spreadsheets stops being discarded. The structured path costs
nothing extra because the data is already in Postgres. Cards give the deferred summary class a
concrete first use. The graph gains a bounded route into structured data without inheriting its
volume. And a misclassification degrades one retrieval path instead of dropping content.

**Negative.** More machinery than flatten-and-chunk, and most of it heuristic: region segmentation,
neighbour-derived labels with confidence, prose detection, formula rendering. The lossless cell
store costs storage that a text-only pipeline does not pay — bounded by cell count, which for a
large workbook is millions of rows, so it needs the same partitioning thinking as everything else.
`query_table` is real new attack surface. And a cell participating in several paths can be
retrieved several ways, so deduplication at the result level matters more than it did.

**A cost worth naming.** Connect-not-copy means a question answerable from a cell now takes two
steps: retrieve, then query. That is slower and puts a tool call between the agent and the answer.
The alternative was one step over a graph the copy had degraded — a trade, not a win, made in
favour of everything else in the collection staying usable.

**Deferred.** Cross-sheet references and formula dependency graphs. Charts. Pivot tables.
Conditional formatting as signal. Formula *evaluation* rather than rendering. Row-level change
tracking across versions, which a crawler re-reading a monthly sheet will eventually want.

**Open.** Whether the lossless cell store belongs in the same database as retrieval or beside it.
Whether region segmentation is worth building before the D1/D7/D8 floor has been used in anger —
the floor alone may answer more than expected, and building it first would tell us.

---

## Alternatives rejected

**Flatten to markdown and chunk normally.** Today's behaviour. Preserves tables *within* a chunk
and destroys them across boundaries, which for any table longer than a chunk is most of them. Loses
formulas, comments and all spatial relationship.

**Materialise every cell as a graph proposition.** The first draft of this ADR. Rejected on
saturation, on landing in the protected class, on predicate collapse producing spurious
contradictions, on BM25 contamination, and above all on being unnecessary — see D10.

**Classify the sheet, then its columns.** The second draft. Better, and still too opinionated: it
assumes a sheet has one shape and that every cell belongs to a column with a consistent role. Real
sheets carry a table and unrelated prose side by side, or no structure at all. Columns are the
wrong unit; regions are closer, and even regions must be advisory rather than gating.

**Treat spreadsheets as unsupported.** Honest about current quality, and untenable: a spreadsheet
can be the majority of a corpus by volume and is often the part people ask about most.

**LLM extraction over the whole sheet.** Expensive, recurring, and worse than reading the cells for
anything tabular, since the structure it would infer is already present and exact. This does *not*
rule out LLM extraction over prose cells, which are prose and fall under ADR-0001 D2 like any other.
