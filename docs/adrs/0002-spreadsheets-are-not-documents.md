# ADR-0002: Spreadsheets are not documents

**Status:** Proposed
**Date:** 2026-09-02
**Decision makers:** Will Faithfull
**Context:** The corpus pipeline of
[ADR-0001](0001-corpus-embeddings-and-knowledge-graph.md) has no sensible answer for `.xlsx`

---

## Summary

A spreadsheet is not a document. It is **a database with presentation**, and the corpus pipeline
treats it as prose: flatten to text, chunk at a size cap, embed each chunk. That destroys the only
thing that made the data useful — a cell's meaning lives in its row label and column header, and
chunking severs both.

This ADR proposes treating spreadsheets as a distinct source class, with the **column** rather
than the sheet as the unit of classification, and it argues for the opposite of the obvious move:
**a table is already a fact store, so the graph should reference it rather than copy it.**

> **Revision note.** An earlier draft of this ADR proposed materialising every
> `(row label, column header, value)` triple as a proposition, on the grounds that structural
> extraction is cheap and lossless where ADR-0001 D2 rules out LLM extraction as expensive and
> lossy. That premise is true and the conclusion did not follow. §"Why the obvious move is wrong"
> records why, because the reasoning is the useful part of this document.

---

## Context

### What prompted this

Loading a spreadsheet-heavy corpus through the pipeline. Two document converters disagreed about
the same workbook by more than an order of magnitude in extracted text, and in opposite
directions on different sheets — one preserving markdown tables, the other flattening cells and
emitting no tables at all. That disagreement is not a bug in either: it is two reasonable answers
to a question nobody has decided, which is *what the text of a spreadsheet even is*.

Then, more damningly: `40%` in a chunk is noise. `Widget A | unit cost | 4.20` is a fact. Current
chunking produces the former.

### The questions people actually ask

Taking a status matrix and a pricing model as the two archetypes:

| Question | What answers it | Current pipeline |
|---|---|---|
| "What discount did we assume for this account?" | Cell lookup: label → value, with header context | Fragment without headers |
| "What are the stages?" | The column, coherently | Many fragments of one table |
| "How is ROI calculated?" | The **formula** | Formulas are stripped entirely |
| "Which rows are above a threshold?" | A filter over rows — this is SQL | Cannot answer |
| "What does the model assume about ramp?" | Label/value prose | Works, by luck |
| "Show me the definitions" | A coherent table | Fragments |

Those want three different mechanisms. One flattening strategy cannot serve them, which is why
choosing between converters felt arbitrary — both were answering the wrong question.

### Why the obvious move is wrong

The tempting design is to materialise every `(row label, column header, value)` triple as a
proposition. Structurally extracted, no LLM, exact. It fails three ways, and the third is the one
that matters.

**It saturates the graph.** A 10,000-row by 20-column sheet is 200,000 triples from one file.
ADR-0001's own sizing model puts an entire organisation's conversational graph at roughly 3.6M
propositions, so one mid-size spreadsheet is about 5% of everything that organisation ever said,
and ten of them are half of it.

**It poisons the signals the graph runs on.** Three effects, each worse than the volume:

- Propositions are the class ADR-0001 D1 protects with a floor in the rerank budget, precisely
  because rank-by-score starves a class that matters for reasons score cannot see. Bulk triples
  land *inside* that protected class and starve the personal memory the floor exists to defend.
  A cap on chunks does not help; these are not chunks.
- 200,000 propositions share about 20 predicates. Contradiction resolution keys on
  `(subject, functional predicate)`, and entity dedup is gated by a cosine on short name strings,
  which is noisy on exactly this input. Two row labels that merge when they should not turn their
  measures into competing assertions of a functional predicate, and the newer mass-invalidates the
  older — silently, plausibly, wrongly.
- BM25 statistics are per collection. Rows of near-identical text make every term in the table
  common, which strips discriminative power from the *prose* in the same collection. The documents
  get worse because a spreadsheet arrived.

**And it is unnecessary, which is the real objection.** Propositions exist because prose is not
queryable: structure has to be extracted before anything can be asked. **A table already has
structure.** Copying it into the proposition store duplicates the data, destroys the statistics,
and buys nothing that a query over the table could not answer. Cheapness was never the argument
for extraction, and losslessness is worth nothing when the data was already lossless where it sat.

So the rule is **connect, do not copy**: keep the table as a table, and give the graph a reference
into it. See D6.

### Rich text hides in spreadsheets

The other half of the error is treating a sheet as homogeneous. Plenty of sheets carry a Notes,
Description or Comments column holding paragraphs per row; some sheets are documents wearing a
grid, with merged cells and prose blocks and no header row at all; and cell comments and notes are
pure prose that every converter drops.

Triple extraction turns a 500-word description cell into
`Widget A | description | <500 words>`. Not atomic, embeds badly as a proposition, and it was a
**chunk** all along. A design that only sees numbers will silently discard the most document-like
content in the workbook.

**So the unit of classification is the column, not the sheet.** A sheet is a mixture, and the
mixture is the point.

### Sheet shapes, and the column typology inside them

Three shapes, wanting different treatment:

- A **table sheet** is rows of records: a status matrix, a log, a price list. Header row, repeated
  row structure, few or no formulas.
- A **model sheet** is a calculator: inputs feed formulas which produce outputs. No row semantics
  at all — the interesting content is the *assumptions* and the *formulas*, and nobody wants the
  intermediate cells.
- A **document sheet** is prose in a grid: merged cells, no header row, paragraphs laid out
  visually. It wants ordinary chunking and nothing else.

Formula density, repeated row structure and a detectable header row separate them. Treating them
alike is why one converter can produce an order of magnitude more text than another for sheets
from the same workbook, with neither being wrong about the sheet it was built for.

Within a table sheet, every column is one of:

| Column kind | Signal | Treatment |
|---|---|---|
| **Key** | high cardinality, identifies the row | Row identity; the anchor for links (D6) |
| **Dimension** | low cardinality, repeats, categorical | Queryable column. Never a fact, never an entity |
| **Measure** | numeric, currency, date | Queryable column only |
| **Prose** | mostly unique, long, sentence punctuation | **Chunk it** (D3a) |
| **Formula** | same formula down the column | One retrievable unit per column, not per row |

Cardinality, mean length, type-inference success and punctuation density separate these well
enough to act on, and badly enough to need a confidence gate.

---

## Decision

### D1 — Classify the sheet, then classify every column

`table` | `model` | `document` | `unknown` for the sheet; then for a table sheet,
`key` | `dimension` | `measure` | `prose` | `formula` per column.

Both classifiers report confidence, and `unknown` falls through to the sheet card alone (D2). No
structural claim is made about a shape that was not recognised: a wrong header assignment
produces *confidently wrong* output, which is worse than no output. A misread prose column is
merely a missed chunk; a misread key column corrupts every link built on it, so the confidence
bar is higher for keys.

### D2 — Every sheet gets a card

One summary chunk per sheet: sheet name, dimensions, column headers, column types, ranges of
numeric columns, and two example rows. Cheap, always correct, and it is what answers "what is in
this workbook" and "does this model account for ramp".

Its more important job is *routing*: the card is what lets an agent decide to go and look
properly, rather than guessing from fragments. This is ADR-0001's deferred summaries-as-a-source-
class, arriving first for the format that needs it most.

### D3 — Table sheets: chunk by row, with the header prepended to every chunk

The single highest-value change available, and nearly free. Each row becomes a self-contained
retrievable unit that carries its own column context. Small row-groups are acceptable where rows
are narrow.

**With a cap.** A few hundred rows becoming a few hundred chunks would swamp the 38-slot corpus
quota
(`floor(k_rerank × corpus_fraction)`) and starve every other source — the failure ADR-0001 D1
exists to prevent, arriving from a direction D1 did not anticipate. Beyond the cap, the card plus
the structured path (D5) is the answer, not more chunks.

### D3a — Prose columns become chunks, and document sheets are just documents

A column of paragraphs is a column of documents. Each cell becomes a chunk carrying its row
identity as context — the key columns prepended the way D3 prepends a header — so it is
retrievable as prose and readable as belonging to its row. These chunks are ordinary corpus
content: they take the corpus quota, they are eligible for LLM proposition extraction under
ADR-0001 D2's normal per-collection rule, and nothing about them is spreadsheet-specific once
they exist.

A `document` sheet is this case generalised: chunk it as prose and make no structural claims.

Cell comments and notes are prose and are dropped by every converter tested. They belong here
too, attached to the cell's row.

This is half the reason the first draft of this ADR was wrong: a design that only sees numbers
discards the most document-like content in the workbook.

### D4 — Model sheets: assumptions and formulas, not cells

Extract input label → value pairs, and formulas rendered as text: `ROI = (revenue − cost) / cost`.
A rendered formula is a good embedding target and it is the actual answer to "how does this work".
Converters strip formulas today, so this is new capability rather than better chunking.

### D5 — Land the cells as rows, and expose a query tool

The honest answer to a filter-and-compare question is that retrieval is the wrong mechanism. The
data is already going into Postgres; land table sheets as actual rows and expose a `query_table`
tool on the MCP surface beside `search_corpus`.

This follows the instinct recorded during ADR-0001's design discussion — that two tools beat one
magic endpoint, and that an agent with conversational context often knows which it wants better
than a router would. Excel defined names, sheet names and table names come along as free
human-authored semantic metadata.

### D6 — Connect the graph to the table; never copy the table into the graph

The graph gets **one mention edge per row key**, not one proposition per cell. An entity that
appears in conversation — a product, an account, a stage — links to the rows that name it, in
`entity_mentions` alongside the chunk mentions D2-of-ADR-0001 already defines.

That preserves exactly the path the mention edge exists for: a chat fact seeds an entity, graph
expansion reaches the row, and the agent reads or queries it. What it does not do is duplicate the
table into the class ADR-0001 D1 protects.

The properties that follow are the reason for the design:

- **Bounded by rows, not cells.** One link per row key per entity, against one proposition per
  cell — three orders of magnitude apart on a wide sheet.
- **Lands outside the privileged class.** Mentions are a side table. They cannot starve the
  personal-memory floor, because they are not competing for it.
- **No predicate collapse, so no spurious contradictions.** There are no bulk functional
  predicates to mass-invalidate.
- **No IDF contamination.** Nothing is added to the text corpus statistics.
- **Nothing is lost.** The values are still there, in the table, exactly as authored, reachable
  through D5.

**Materialising triples as propositions is rejected as a default.** It stays available as an
explicit per-sheet opt-in for a small, human-designated fact table — a glossary, an org chart, a
code list — under a hard row budget, because for a fifty-row reference table the objections above
have no force and the convenience is real. That is a deliberate narrow exception, and the default
is off.

---

## Consequences

**Positive.** Spreadsheet questions become answerable at all. A spreadsheet-heavy corpus stops
being mostly header-less fragments, and the prose buried in it stops being discarded. The
structured path costs nothing extra — the data is already in Postgres, which is the whole thesis.
Cards give the deferred summary class a first concrete use. The graph gains a bounded route into
structured data without inheriting its volume.

**Negative.** Two classifiers instead of none, both heuristic, both able to be wrong. Header
detection is genuinely hard — merged cells, multi-row headers and pivot tables defeat naive
detection — and the failure mode is confident wrongness, hence the confidence gate and the
`unknown` class. Prose-column detection has its own edge cases: a mostly-empty Notes column with
occasional paragraphs, an enum whose labels are long enough to look like prose, and any
non-Latin-script text where length heuristics do not transfer. `query_table` is new surface area
with its own injection considerations — a tool that runs SQL from a model's arguments needs the
same scope-binding treatment `pgkg/mcp_server.py` documents, and probably a read-only role and a
column allowlist besides.

**A cost worth naming.** Connecting rather than copying means a question answerable from a cell
now needs two steps: retrieve, then query. That is slower and it puts a tool call between the
agent and the answer. The alternative was one step over a graph the copy had degraded, so this is
a trade rather than a win — but it is a trade made in favour of everything else in the collection
staying usable.

**Deferred.** Cross-sheet references and dependency graphs between formulas. Charts. Pivot tables.
Conditional formatting as signal. Anything requiring formula evaluation rather than rendering.
Row-level change tracking across versions of a workbook, which is what a crawler re-reading a
monthly-updated sheet will eventually want.

**Open.** Whether calculator workbooks belong in a retrieval corpus at all, or whether the useful
corpus is the prose alone, with models reachable only through `query_table`. This ADR assumes they
belong and are handled by D4; the alternative is defensible and cheaper.

---

## Alternatives rejected

**One converter, chosen by benchmark.** What the evidence actually says is that the converters
disagree because the question is undecided. Picking a winner would freeze one arbitrary answer.

**Convert to markdown tables and chunk normally.** This is today's behaviour for `.xlsx`. It
preserves tables *within* a chunk and destroys them *across* chunk boundaries, which for any
table longer than a chunk is most of them.

**LLM extraction over the whole spreadsheet.** Expensive, recurring, and strictly worse than
reading the cells for anything tabular: the structure the LLM would infer is already present and
exact. Note this does *not* rule out LLM extraction over prose columns, which are prose and are
governed by ADR-0001 D2 like any other prose.

**Materialising every cell as a proposition.** The first draft of this ADR. Rejected on
saturation, on predicate collapse producing spurious contradictions, on BM25 contamination of the
prose sharing the collection, and above all on being unnecessary: a table is a fact store already.
Retained only as a narrow per-sheet opt-in under a row budget (D6).

**Treat spreadsheets as unsupported.** Tempting, and honest about current quality — but a
spreadsheet can be the majority of a corpus by volume, and it is often the part people ask about
most.
