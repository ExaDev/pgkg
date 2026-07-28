# Giving engineers a straight answer out of the standards library

**A technical primer for non-technical stakeholders.**

We are building a system that reads a set of governing standards once, learns how
their clauses relate to each other, and then answers an engineer's question with
the specific clause behind it. This note explains how that works and assumes
nothing about software.

- **Audience:** commercial and engineering leadership
- **Status:** pre-build, for scoping
- **Read time:** ~8 minutes

---

## In one paragraph

Engineers currently find governing clauses by knowing where to look. We are
turning that knowledge into infrastructure: the standards are broken into
individual requirements, indexed by meaning as well as by wording, wired together
by what they refer to, and version-tracked as amendments land. An engineer then
asks a question in plain language, from inside the tools they already use, and
gets back the handful of clauses that actually bear on it — each one cited, with
its edition, so the answer can be checked. The engineer stays the design
authority. The system removes the searching, not the judgement.

---

## § 01 — The problem, as engineers experience it

A single design has to satisfy a stack of documents — connection conditions,
engineering recommendations, wiring regulations, calculation standards — that
between them run to thousands of pages, cross-reference each other constantly,
and are amended on their own separate schedules.

Finding the clause that governs a specific decision is therefore a research task,
not a lookup. In practice it depends on a small number of experienced engineers
who know where things live and which document takes precedence.

Two costs follow. Chargeable hours go into navigating documents rather than
designing. And the knowledge of *which* clause governs *what* stays in people's
heads, where it cannot easily be reviewed, handed over, or audited.

---

## § 02 — What we are building, in three parts

| Part | What it is |
|---|---|
| A standards library the machine can reason over | Not a folder of PDFs with a search box. The content of those documents, broken down into individual requirements, indexed, and connected by what they refer to. |
| A retrieval engine | Given a question, it finds the small set of clauses that genuinely bear on it — and is honest when there aren't any. |
| A standard socket, so any AI assistant can plug in | This is the **MCP** in the project name, and it is the part most worth understanding commercially. |

---

## § 03 — MCP: one connection point instead of bespoke wiring

> **MCP — Model Context Protocol —** is a published, agreed interface between AI
> assistants and the systems they need to reach.

The analogy is one a grid connection business already runs on. A generator does
not get a hand-negotiated physical interface to the network. It connects at a
defined connection point, to published conditions, and the network does not need
to know the make and model of the machine behind it.

MCP does that job for AI tools. We build the standards library once, expose it
through the standard interface, and any compliant assistant can use it.

Two consequences that matter commercially:

- Engineers get this **inside the tools they already have open**, rather than as
  another portal to log into and forget.
- When a better assistant appears next year, we re-point rather than rebuild —
  the library is the asset, not the chat window.

---

## § 04 — Building the library: what "ingesting a standard" involves

Seven stages, run once per document and then again whenever an amendment lands.
Each stage adds structure the next one depends on.

```
════════════════════════════════════════════════════════════════════
 │        │        │           │          │           │          │
 01       02       03          04         05          06         07
 Collect  Cut up   Index by    Reduce to  Wire        Build both Track
                   meaning     single     together    indexes    amendments
                               requirements
```

### 01 — Collect the documents · *ingestion*

We take the documents in and record what each one *is*: issuing body, edition,
publication date, amendment status, and where it sits in the hierarchy.
Provenance is captured before anything else, because every answer later on has to
be traceable back to it.

### 02 — Cut each document into passages · *chunking*

A four-hundred-page standard cannot be handed to an AI in one piece, and almost
none of it is relevant to any given question. We cut it along its own structure —
clause, sub-clause, table — so that a retrieved passage is a unit an engineer
would recognise and cite, not an arbitrary slice of text.

### 03 — Index each passage by meaning · *embedding*

Each passage is converted into a list of numbers that encodes what it is *about*.
Passages about the same subject end up numerically close together even when they
share no words at all.

This is what lets an engineer ask about "voltage step on connection" and be given
the clause that says "rapid voltage change", without having to guess the
document's vocabulary first.

### 04 — Reduce prose to single requirements · *proposition extraction*

Standards writing packs several distinct requirements into one sentence. We use a
language model to restate each passage as a set of short, self-contained
statements — one requirement each, with its subject and limit made explicit.

Two reasons this matters. Short focused statements are markedly easier to
retrieve accurately than paragraphs. And an engineer can check a one-line
requirement at a glance. The original passage is always kept alongside: the
extracted statements are a *route to* the source text, never a replacement for it.

### 05 — Wire the requirements together · *knowledge graph*

Each requirement names things: a standard, a clause, an item of plant, a
quantity, a limit. We link every requirement that names the same thing, resolving
the fact that one document's "Generating Unit" is another's "generator".

The result is a schematic of the standards library, serving the same purpose a
single-line diagram serves for a network: it shows what connects to what. It is
what allows the system to follow a cross-reference out of one document and into
another, instead of treating each PDF as an island.

### 06 — Build both indexes over the same content · *keyword + vector search*

One index for exact words and reference numbers, one for meaning. You need both,
and each covers the other's blind spot. Exact matching is the only dependable way
to find a specific document reference or table number; meaning-based matching is
the only way to find a clause whose wording nobody could have guessed.

### 07 — Track amendments rather than overwrite them · *versioning and supersession*

When an edition is superseded we do not delete the old text. We mark it
superseded and record the date the change took effect. So the default answer
reflects what is current, and the system can still answer "what applied when this
design was approved in 2023?"

That is revision control, applied to requirements instead of drawings — and for
anything that may later be disputed, it is the most valuable property in this
list.

---

## § 05 — Answering a question

The pattern here is one engineering teams already use daily: **screen wide and
cheap, then study narrow and expensive.** A fast pass over everything produces a
shortlist; a slow, accurate pass reads only the shortlist.

### 01 — Separate the question into its parts · *query decomposition*

A real engineering question usually contains several. "Can this site export at
4 MW without a new transformer?" is at least three. Each part gets its own search.

### 02 — Search twice, in parallel · *hybrid retrieval*

Two mechanisms with genuinely different failure modes — comparable to running two
protection schemes on different sensing principles. The point is not redundancy
for its own sake; it is that the two do not miss the same things.

| Lane | What it finds |
|---|---|
| **By wording** | Exact references, defined terms and table numbers. Reliable when the engineer already knows the vocabulary. |
| **By meaning** | Clauses that address the question in different words. Reliable when the engineer does not know where to look. |

### 03 — Merge the two rankings · *reciprocal rank fusion*

A passage that ranks well on both methods rises to the top; a passage only one
method found still gets a hearing rather than being discarded. It works on rank
position alone, so there is no hand-tuned weighting between the two searches for
anyone to argue about or re-tune later.

### 04 — Follow cross-references one step out · *graph expansion*

If a clause we found defers to another standard, that clause is pulled in as
well. This is the stage that catches the requirement nobody remembered was there,
and it only works because of the wiring built at ingestion stage 05.

### 05 — Close-read the shortlist · *cross-encoder reranking*

A slower, more accurate model reads each shortlisted passage *against the actual
question* and re-orders them. It is far too expensive to run across the whole
library — which is precisely why the cheap wide screen came first.

### 06 — Drop the near-duplicates · *diversity selection (MMR)*

Standards repeat themselves, and related documents restate each other. We keep
the best statement of each distinct point, so the engineer sees the breadth of
what applies rather than the same requirement five times.

### 07 — Draft the answer from those passages only · *grounded generation*

Only at this last step does a model write prose, and it may only use the passages
retrieved. Every claim carries its clause and edition, so the engineer reads the
answer and the evidence together.

---

## § 06 — What it can be relied on for, and what it cannot

**It will:**

- Put an engineer in front of the right clause in seconds, with its exact text.
- Surface the cross-reference nobody remembered.
- Answer "what changed between these two editions, and does it affect us?"
- Make the compliance trail explicit, reviewable and reusable rather than
  personal.

**It will not:**

- Sign anything off. The engineer remains the design authority and we will
  present it that way, in the interface as well as in writing.
- Replace judgement on precedence, derogations, or anything requiring negotiation
  with a network operator.
- Guarantee completeness. It is a research assistant with citations, not a
  compliance certificate.

### The three hard parts, stated up front

**Tables, figures and formulae.** A great deal of what matters in a standard
lives in a table rather than a sentence. These need dedicated handling and are
the most likely source of error in early versions. We will scope them explicitly
rather than discover them late.

**Conditional requirements.** "Where X applies, unless Y" is easy to retrieve and
easy to misread. Answers must show the conditions in full rather than summarise
them away.

**Silence.** Failing to find a governing clause is not evidence that none exists.
The system must say "I did not find one" instead of assembling a plausible
answer, and we test that behaviour deliberately.

---

## § 07 — How we will know it works

Before we build, we agree a set of real questions with the client's engineers
where the governing clause is already known and undisputed. That set — the *gold
set* — is the yardstick, and it is what turns "the AI seems impressive" into
something that can be accepted or rejected.

| Measure | Why it matters |
|---|---|
| Did the correct clause come back? | Measured as a hit rate within the top few results. The single most important number, reported every iteration. |
| Did the answer cite it, and claim only what it says? | Checked against the source text. An answer that is right for the wrong reason still counts as a failure. |
| When nothing governs, did it say so? | We deliberately include questions with no governing clause. Confidently inventing one is the most damaging failure mode available to a system like this. |
| How long did the engineer take, against doing it by hand? | The commercial case rests on this one, and it is the easiest of the four to measure honestly. |

---

## § 08 — What we need from the client to start

**Standards licensing position.** Standards are copyrighted, and publishers'
licences constrain how their text may be stored and reproduced. We need current
licences and, where necessary, the publishers' terms confirmed before we ingest
their text. Realistically this is the single thing most likely to shape the scope
of phase one, so it is worth starting now rather than at build time.

**Scope for phase one, in priority order.** Which documents, and which kinds of
question. Narrow and deep will demonstrate value far better than broad and
shallow, and gives the gold set something to bite on.

**A named reviewing engineer.** Someone senior enough to settle "is this answer
right", for a few hours a week. Without that, we are tuning against our own
guesses about the domain.

**The confidentiality boundary.** Whether design details may leave the client's
environment, and which model providers are acceptable. This decides where the
system runs, and it is much cheaper to answer before we build than after.

---

## § 09 — Glossary

| Term | In plain terms |
|---|---|
| **MCP** | Model Context Protocol. The agreed interface through which any AI assistant reaches our standards library — a defined connection point rather than bespoke wiring per tool. |
| **Ingestion** | Taking documents in and recording what they are, including edition and amendment status. |
| **Chunk** | One passage of a document, cut along clause boundaries so it is citable on its own. |
| **Embedding** | A numerical fingerprint of what a passage is about, letting the system match on meaning rather than wording. |
| **Vector search** | Searching by those fingerprints: "find me passages about this", not "find me passages containing these words". |
| **Proposition** | One requirement, restated as a single self-contained statement, always linked back to the passage it came from. |
| **Knowledge graph** | The map of which requirements refer to the same plant, quantities and documents. A single-line diagram for the standards library. |
| **Hybrid retrieval** | Running the wording search and the meaning search together, because each covers the other's blind spot. |
| **Rank fusion** | Combining the two result lists into one, on rank position alone, with no weighting to tune. |
| **Reranking** | The slow, accurate second pass that reads the shortlist against the question and re-orders it. |
| **Supersession** | Marking a clause as replaced rather than deleting it, so "what applied at the time" stays answerable. |
| **Grounded** | The model may only use the passages retrieved, and must cite them. The guard against confident invention. |
| **Gold set** | The agreed questions with known correct clauses, against which the system is measured and accepted. |

---

## How this maps onto pgkg

For internal reference, the stages above correspond to existing pgkg components:

| Primer stage | pgkg implementation |
|---|---|
| Chunking | `_chunk_text()` in `pgkg/memory.py` |
| Embedding | `embed()` in `pgkg/ml.py` (`bge-m3`, 1024-d), stored in `propositions.embedding`, HNSW index |
| Proposition extraction | `extract_propositions()` in `pgkg/ml.py`, cached in `proposition_cache` |
| Knowledge graph | `entities` / `edges` tables, `pgkg_link_entity()` |
| Keyword index | `propositions.tsv` generated column + BM25 scoring (`migrations/008_bm25_search.sql`) |
| Hybrid retrieval + rank fusion + graph expansion | `pgkg_search()` (`migrations/008_bm25_search.sql`) |
| Query decomposition | `migrations/009_query_decomposition.sql` |
| Reranking + diversity | `rerank()` and `mmr()` in `pgkg/ml.py` |
| Supersession / versioning | `propositions.superseded_by`, `propositions.asserted_at` |

The MCP layer and clause-aware (structure-preserving) chunking are the two pieces
this engagement adds on top of what pgkg already does.

---

*Prepared by ExaDev. Not a specification — scope subject to § 08.*
