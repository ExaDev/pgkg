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

This ADR proposes treating spreadsheets as a distinct source class with four mechanisms rather
than one, and records an observation that inverts ADR-0001's central corpus decision:
**for a table, structural extraction is cheap and lossless, which is the exact opposite of the
case D2 rules out for prose.**

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

### The classification that everything depends on

Two kinds of sheet want opposite treatment:

- A **table sheet** is rows of records: a status matrix, a log, a price list. It has a header row,
  repeated row structure, and few or no formulas.
- A **model sheet** is a calculator: inputs feed formulas which produce outputs. It has no row
  semantics at all — the interesting content is the *assumptions* and the *formulas*, and nobody
  wants the intermediate cells.

Formula density, repeated row structure and a detectable header row separate them. Treating them
alike is why one converter can produce an order of magnitude more text than another for sheets
from the same workbook, with neither being wrong about the sheet it was built for.

---

## Decision

### D1 — Classify every sheet before converting it

`table` | `model` | `unknown`. The classifier reports its confidence, and `unknown` falls through
to the sheet card alone (D2 below). No structural claim is made about a sheet whose shape was not
recognised, because a wrong header assignment produces *confidently wrong facts*, which is worse
than no extraction.

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

### D6 — Structural extraction, and why D2-of-ADR-0001 inverts here

ADR-0001 D2 rules out proposition extraction for the corpus: it is expensive, recurring, and lossy
on exactly the content corpora are made of. **For a table sheet, every reasoning behind that
decision reverses.**

Each `(row label, column header, value)` triple *is* a proposition —
`Widget A | unit cost | 4.20` — obtained structurally, with no LLM, losslessly, at zero marginal
cost. The thing ADR-0001 forbids for prose is the single best thing available for a
table.

So table sheets **may** be proposition-extracted, by the table extractor and never by an LLM,
gated on high-confidence header detection. This is a deliberate exception to ADR-0001 D2 and not a
reversal of it: the argument there was about LLM extraction from prose, and it still holds for
every other corpus format.

---

## Consequences

**Positive.** Spreadsheet questions become answerable at all. A spreadsheet-heavy corpus stops being mostly
header-less fragments. The structured path costs nothing extra — the data is already in Postgres,
which is the whole thesis. Cards give the deferred summary class a first concrete use. Table
propositions enrich the knowledge graph from a source that needs no LLM budget, which also gives
the entity layer far more to link against.

**Negative.** Header detection is genuinely hard: merged cells, multi-row headers and pivot
tables all defeat naive detection, and the failure mode is confident wrongness. Hence the
confidence gate and the `unknown` class. Sheet classification is a heuristic that will
misclassify. `query_table` is new surface area with its own injection considerations — a tool
that runs SQL from a model's arguments needs the same scope-binding treatment
`pgkg/mcp_server.py` documents.

**Deferred.** Cross-sheet references and dependency graphs between formulas. Charts. Pivot tables.
Conditional formatting as signal. Anything requiring formula evaluation rather than rendering.

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

**LLM extraction over spreadsheets.** Expensive, recurring, and strictly worse than reading the
cells: the structure the LLM would infer is already present and exact.

**Treat spreadsheets as unsupported.** Tempting, and honest about current quality — but a
spreadsheet can be the majority of a corpus by volume, and it is often the part people ask about
most.
