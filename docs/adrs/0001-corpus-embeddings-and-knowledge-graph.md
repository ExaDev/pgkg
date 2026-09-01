# ADR-0001: Corpus embeddings alongside the proposition knowledge graph

**Status:** Accepted
**Date:** 2026-08-20
**Decision makers:** Will Faithfull
**Context:** Extending pgkg from a single-namespace proposition graph to a multi-tenant memory system that also holds document corpora

---

## Summary

pgkg today is one flat pipeline — `documents → chunks → propositions`, scoped by a single
`namespace TEXT` — with retrieval implemented as one 250-line SQL function. This ADR records
the design for growing it into a multi-tenant system that serves both **conversational memory**
(facts extracted from user chats) and **document corpora** (an organisation's own material and
whatever general reference knowledge helps its users do the job), with provenance, versioning
and expiry throughout.

Eight decisions are recorded below as **D1–D8**, followed by the phased implementation plan.

The load-bearing conclusions:

- **Two physical stores, one fusion layer.** Propositions and chunks keep separate tables,
  indexes and corpus statistics; retrieval fuses them by rank (RRF), not score.
- **The corpus is not proposition-extracted by default.** Extraction is a per-collection flag.
  The corpus joins the graph through a cheap *mention* edge instead.
- **`claim_scope` (`world` | `org` | `user`) is orthogonal to ownership.** Scope governs
  semantics; ownership governs storage and visibility. Everything is org-owned by default.
- **Provenance is a shared, immutable derivation record**, referenced by a hot-path
  `provenance_id` column.
- **Bitemporal time**: assertion, validity, belief and retention are four separate clocks.

---

## Context

### The requirement

A SaaS product used by organisations. An organisation has many users; each user has many chats.
Facts extracted from those chats form a knowledge graph. The organisation also has a corpus it
wants feeding the same agentic memory — its own documents, but equally vendor documentation,
industry guidance, curated articles, anything that helps its users work well.

### What the current implementation cannot carry

| Where | Current behaviour | Why it blocks this |
|---|---|---|
| `migrations/009_query_decomposition.sql` | BM25 recomputes `avgdl` across the namespace per query, and document frequency per query lexeme via a correlated `COUNT(*)` over active propositions | **O(corpus) per query.** Seconds at 10M rows. Unrelated to the corpus feature; blocks everything |
| `pgkg/memory.py` `_vec_literal` | Embeddings string-interpolated into SQL text | ~15 KB of SQL per recall, no plan reuse, `pg_stat_statements` useless. `register_vector` is already installed on every connection |
| `pgkg/memory.py` `ingest` | No transaction; 3–4 round trips per proposition | Partial data on failure; unusable at corpus scale |
| `pgkg/memory.py` `recall` | `asyncio.ensure_future(self._bump(...))` updates N rows per read | Write amplification on the read path; unawaited task swallowing errors |
| `migrations/003 → 007 → 008 → 009` | `pgkg_search()` redeclared in full by each migration; four ~250-line copies | Every retrieval change is a 270-line diff; no stage is unit-testable |
| `pgkg/memory.py` | Chunks-only mode writes chunks *into* `propositions` with NULL S/P/O; `Result.subject`/`.object` hardcoded to `None` | The table means two things, so BM25 length normalisation averages 12-token facts with 300-token chunks |
| SQL throughout | `vector(1024)` hardcoded in DDL, in the `q_embedding` parameter and in `RETURNS TABLE` | `config.embed_dim` exists and is ignored; changing embedder is a rewrite |

### Four content classes with different physics

| Class | Unit | Volume | Churn | Time semantics | Visibility |
|---|---|---|---|---|---|
| Chat facts | Proposition, 5–20 tokens | 10⁵–10⁷ / org | Append-only, high rate | Fresh wins; decay correct | Private to one user |
| Corpus chunks (org-owned) | Passage, 300–800 tokens | 10⁵–10⁶ / org | Versioned, bursty re-sync | Timeless; current until replaced | Org-wide, ACL-bounded |
| Corpus chunks (general knowledge) | Passage, 300–800 tokens | 10⁵–10⁶ / org | Crawled, changes unannounced | Perishable; ages on a multi-year curve | Org-owned by default |
| Entities | Canonical name + aliases | 10⁴–10⁵ / org | Merge/split, slow | None | Org-wide |

---

## D1 — Two physical stores, unified retrieval, shared entity hub

**Decision.** `propositions` and `chunks` remain separate tables, each with its own indexes,
corpus statistics and scoring profile. A thin fusion layer runs both retrievers and merges by
rank. The `entities` table is the join between them.

**Rejected — one polymorphic `memory_items` table.** Mixing 12-token facts with 600-token
passages makes BM25's `avgdl` meaningless; one HNSW index must serve both; lifecycle operations
must pick document rows out of a table that is mostly chat facts. This is what the code does
today via chunks-only mode, and the corpus feature *removes* the hack rather than extending it.

**Rejected — two separate endpoints, no fusion.** Defensible (the agent has context a router
would guess at) and retained as an option for clients that want control, but it pushes fusion
onto every caller and "what did we agree about the refund policy" needs one ranked list.

**Consequence — the SQL decomposes.** Candidate sources become set-returning functions with one
identical signature `(item_id UUID, kind TEXT, rank INT, raw_score REAL)`:

```
pgkg_bm25_candidates(source, ...)     ┐
pgkg_vector_candidates(source, ...)   ├─→ pgkg_fuse(weights) → quotas → profile → rerank → MMR
pgkg_graph_candidates(...)            ┘
```

Adding summaries, entity cards or tool outputs later means adding one function, not editing 270
lines. Each stage becomes independently testable.

**Consequence — source quotas before the reranker are mandatory, not a refinement.**

> **The failure mode being designed against:** the corpus drowns the memory. 600k org chunks
> against a user's 4k personal facts wins nearly every query on candidate volume. The product
> symptom is precise: *the assistant stops remembering you and starts quoting the handbook.*
> MMR does not fix it — MMR only diversifies what survived to the final stage. Quota **by
> `claim_scope`**, not just by store: general knowledge is topically unbounded and competes on
> every query, where org policy only matches queries about the org.

Defaults: `w_kw = 1.0`, `w_vec = 1.0`, `w_graph = 0.5`, per-scope RRF weights at parity, corpus
capped at 60% of reranker input with a floor of 8 personal-memory slots.

---

## D2 — The corpus is not proposition-extracted by default

**Decision.** Corpus chunks are retrievable in their own right — embedded and `tsvector`'d,
never extracted. `collections.extract_propositions` opts in per collection, for fact-dense
material only: glossaries, org charts, product catalogues, policy statements, CRM exports.

**Why.**

1. **Cost is recurring and multiplied.** 50k documents × 12 chunks ≈ 480M input tokens ≈
   $70–200 per extraction pass at `gpt-4o-mini` rates. It recurs on every `prompt_version` bump
   and multiplies by tenant. Embedding the same corpus costs ~$10 by API or 1–2 GPU-hours
   self-hosted.
2. **Extraction is lossy exactly where corpora are dense.** Procedures (order is dropped),
   tables (structure flattens), numbers with units and conditions, code, and anything where the
   caveat carries the meaning. A 700-token passage answering "how do I file an expense claim" is
   a better retrieval unit than nine atomised facts about expense claims.
3. **Documents are not memories.** No meaningful `asserted_at`, no trustworthy access-frequency
   signal, no supersession relationship to chat facts.

**Consequence — the corpus joins the graph through three edges, priced very differently:**

| Edge | Mechanism | Cost | Verdict |
|---|---|---|---|
| **Derivation** `proposition ← chunk` | Already `propositions.chunk_id`; formalise via provenance so a fact cites its passage and a passage exposes its facts | Free | Build |
| **Mention** `entity ← chunk` | New `entity_mentions`, populated by **gazetteer matching** — match the org's existing entity names and aliases against each new chunk using the GIN/trigram indexes already on `entities`. No LLM | Near-zero | Build |
| **Full extraction** `corpus → propositions` | The existing extractor over corpus chunks | High, recurring | Opt-in per collection |

The mention edge is why the corpus↔graph relationship is worth having. Chats generate facts
naming entities — *the Helios migration*, *the Acme renewal*; the corpus is where those entities
are **defined**. Without the join the two pools meet only topically. With it, a retrieved chat
fact seeds an entity which pulls in the Helios architecture doc *even when the query wording
matches that document not at all*.

**Synergy.** `proposition_cache` is keyed on `(chunk_text, extractor_model, prompt_version)`.
With content-addressed chunks (D6), re-ingesting a corpus after an edit hits cache for every
unchanged chunk.

---

## D3 — Tenancy: explicit columns, RLS, and partitioning for HNSW correctness

**Decision.** Replace stringly-typed scoping with explicit columns on every retrievable row:

```sql
org_id        UUID NOT NULL   -- hard isolation boundary
collection_id UUID NOT NULL   -- carries decay profile, claim scope, ACL mode
claim_scope   TEXT NOT NULL   -- 'world' | 'org' | 'user'
visibility    TEXT NOT NULL   -- 'private' | 'shared'
owner_user_id UUID NULL       -- set iff visibility = 'private'
session_id    TEXT NULL       -- chat thread, for narrow recall
acl_group_id  UUID NULL       -- low-cardinality group, filtered inside the CTE
```

The shared retrieval predicate:

```sql
WHERE org_id = ANY($1)              -- [tenant_org] or [tenant_org, SYSTEM_ORG]
  AND collection_id = ANY($2)       -- own + subscribed, resolved by the app
  AND (visibility = 'shared' OR owner_user_id = $3)
  AND (acl_group_id IS NULL OR acl_group_id = ANY($4))
  AND invalidated_at IS NULL
```

Plus **row-level security** keyed on a per-request GUC (`current_setting('pgkg.org_id')`) as
defence in depth: in a multi-tenant product one missing `WHERE` clause is a cross-customer
breach, and RLS makes that Postgres-enforced rather than a code-review responsibility. Wrap the
GUC read in a `STABLE` function and verify `EXPLAIN` still prunes — volatile predicates defeat
partition pruning.

**Rejected** — encoding hierarchy in the namespace string (`"acme:user_42:chat_9"`): no
referential integrity, no indexable way to express "this user's private facts plus the org's
shared corpus", nothing for the planner to prune on. **Rejected** — schema-per-tenant and
database-per-tenant: thousands of schemas breaks migrations, pooling and the catalog.

### Partitioning is about correctness, not just speed

pgvector's HNSW does not know about the `WHERE` clause. It walks the graph for the nearest
`ef_search` neighbours **globally**, then the executor discards rows failing the filter. With
one org at 0.1% of the table, a top-20 query either under-returns or degrades to a sequential
scan. This is the classic multi-tenant pgvector failure and it arrives long before storage does.

Two mitigations, both wanted:

- **Partition so the filter becomes pruning.** Each partition carries its own HNSW index.
  Prefer `PARTITION BY LIST (shard_key)` with `shard_key` assigned at insert from a `tenants`
  table (default `pool_<hash(org) mod 64>`) over plain `PARTITION BY HASH` — Postgres moves rows
  across partitions on partition-key `UPDATE`, so a whale tenant can be promoted to a dedicated
  partition later without a migration. This also keeps per-tenant physical isolation available
  as a sales option at zero present cost.
- **`hnsw.iterative_scan = strict_order`** (pgvector 0.8+) so filtered searches keep scanning
  until they have enough surviving rows instead of silently under-returning.

### Accepted risks

- **Entity names cross user boundaries within an org.** Entity resolution is org-wide, which is
  where most of the graph's value lives. Scoped-query isolation is the agreed bar, so the
  residual exposure — an entity *name*, never its content — is accepted.
  **Hard requirement:** every proposition reached by graph expansion is re-filtered through the
  same visibility predicate as the seed. Test with a fixture that deliberately tries to walk
  from a shared entity into another user's private fact.
- **Document ACLs are not optional.** Corporate corpora mirror SharePoint/Drive permissions;
  ingesting without modelling that builds a permission-laundering machine. Keep ACL cardinality
  low (groups, never per-user rows) because the filter must run *inside* the vector CTE.

---

## D4 — `claim_scope` is orthogonal to ownership; org-owned is the default

**Decision.** Two independent axes.

- **`claim_scope`** — what the claim is *about*. Governs decay profile, contradiction
  partitioning and retrieval quotas.
  - `world` — general knowledge, external provenance. True in general, about nobody in particular.
  - `org` — claims about this organisation. Policy, architecture, pricing.
  - `user` — claims by or about this user. Decisions, preferences, state, exceptions.
- **Ownership** — who holds the copy. Governs storage, partitioning and visibility.
  `collections.owner_org_id NOT NULL`, the tenant by default; the reserved `SYSTEM_ORG` only for
  collections the platform operator deliberately publishes.

|  | `owner = tenant` (default) | `owner = SYSTEM_ORG` (opt-in) |
|---|---|---|
| `world` | A tenant's curated reading list, guidance they gathered, their own crawl. **The common case** | Table-stakes reference: vendor docs, standards, specs |
| `org` | Policy, architecture, pricing. Always private | Only for a parent/franchise group sharing policy downward. Rare; the seam supports it |
| `user` | Chat facts, personal notes | Never |

**Why org-owned is the default.** *Which* general knowledge you collect, and how you organise
and leverage it, is itself competitive advantage. The underlying articles may be public; the
curated set is not. An org's judgement about which guidance to trust is exactly the asset a
tenant would be unhappy to find pooled with competitors'.

**Neither store outranks the other.** This rules out a `trust_tier` ordering — an ordering exists
to break ties in favour of one side and there is no side to favour. Two rules follow:

1. **Contradiction resolution runs only within a claim scope.** A user-scoped statement never
   invalidates an org-scoped one, or the reverse; general guidance never invalidates org policy.
   Cross-scope disagreement is recorded as **tension** and surfaced to the answering model with
   both sides labelled and cited — usually it is not a contradiction but a personal exception to
   a general rule, and that distinction is the answer the user wants. *"The vendor docs recommend
   X, our policy is Y, and you decided Z for this client"* is three true statements at three
   scopes, and often the complete answer.
2. **Agreement is corroboration and must not be merged away.** A user statement agreeing with a
   guidance article is independent evidence from two kinds of source. Record it in
   `corroborations(proposition_a, proposition_b, kind)` where `kind ∈ {agrees, tension}`.

**`source_authority` is not the rejected trust ordering.** Killing `trust_tier` settled that
chat does not outrank corpus and corpus does not outrank chat. External sources plainly do rank
each other — official vendor documentation beats an unattributed PDF. `source_authority` is a
tie-breaker operating strictly *inside* `world` scope, never a lever between scopes.

### Sharing: the seam ships early, the content whenever

Sharing sits in the hot-path predicate, so the seam must exist even while unused. Use a reserved
system org rather than `org_id IS NULL` — NULL breaks the NOT-NULL invariant, complicates every
RLS policy, and NULL semantics in partition keys are a trap.

```sql
collections              + owner_org_id NOT NULL   -- = SYSTEM_ORG only when shared
collection_subscriptions NEW (org_id, collection_id, enabled, rrf_weight)
                             -- empty by default; nothing is subscribed implicitly
```

Pruning still works: two partitions when subscribed, one when not (the default). `rrf_weight`
lets a tenant turn shared material down, or off, without a rebuild.

**Share the computation, not the rows.** Physical row dedup across tenants is tempting — chunks
are already content-addressed — but **rejected**: a row shared between tenants cannot live in a
tenant partition, so it lands in a content-hash-partitioned pool, and vector search over that
pool cannot prune by org. The recall lost exceeds the disk saved, and it couples tenants for
index rebuilds. Deduplicate the expensive half instead:

```sql
CREATE TABLE embedding_cache (
  content_hash  BYTEA,
  generation_id UUID,
  vec           halfvec,
  PRIMARY KEY (content_hash, generation_id)
);
-- Consulted before calling the embedder. Populated ONLY for collections flagged
-- public_source = true (crawled or licensed web content). Each tenant still stores
-- its own chunk row in its own partition: partitioning intact, retrieval untouched.
```

Every tenant keeps its own data and index; the GPU cost is paid once per unique passage per
generation. It also makes embedder cutovers far cheaper across a fleet with overlapping public
content.

*Deliberate boundary:* a content-hash cache is probe-able — observing cache-hit latency reveals
whether a passage is cached. Uninteresting for public web content; for a confidential document
it would confirm another tenant holds it. Hence the `public_source` restriction: user uploads
and chat text never enter the cache.

### Hard rules

- **Nothing a tenant ingests is ever promoted into a shared collection** — not automatically, not
  by heuristic, not as a "contribute back" default. Shared collections are populated by the
  operator from operator-licensed sources. Put this in the schema comment: it is the kind of
  feature someone adds later without knowing why the rule exists.
- **Ranking signals are never computed globally over shared content.** Access counts are already
  dropped for corpora; this is the second and stronger reason. `pgkg_recompute_pagerank()` must
  run per subscriber over that subscriber's visible subgraph, or over the shared subgraph alone
  with org edges excluded. Global centrality over shared entities is a real cross-tenant
  inference channel.

---

## D5 — Provenance is a shared, immutable derivation record

**Decision.** An append-only `provenance` table, deduplicated so all propositions extracted from
one chunk by one model run share one row — provenance cardinality tracks chunk count, not fact
count. `propositions.provenance_id UUID NOT NULL` and `chunks.provenance_id` are the hot-path
columns.

```sql
CREATE TABLE provenance (
  id             UUID PRIMARY KEY,
  org_id         UUID NOT NULL,
  kind           TEXT NOT NULL,   -- chat_turn | document_version | api_assertion
                                  -- | inference | consolidation
  source_id      UUID,            -- document_versions.id | chat_turns.id (polymorphic by kind)
  source_locator JSONB,           -- page, char span, turn index, cell ref — for citation
  producer       TEXT NOT NULL,   -- llm_extract | chunker | user_assertion | consolidation
  producer_model TEXT,
  prompt_version TEXT,
  ingest_run_id  UUID,            -- retract exactly one bad batch
  actor_user_id  UUID,
  -- external sources need more, and one axis internal documents don't:
  source_url       TEXT,          -- canonical URL, post-redirect
  publisher        TEXT,
  published_at     TIMESTAMPTZ,   -- feeds the 'perishable' decay profile
  retrieved_at     TIMESTAMPTZ,   -- distinct from published_at and from ingested_at
  licence          TEXT,
  source_authority SMALLINT,      -- tie-breaker WITHIN world scope only
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**What the column buys.** Cascade on source change becomes one set-based `UPDATE ... WHERE
provenance_id IN (SELECT id FROM provenance WHERE source_id = ...)` — no traversal. Retraction by
`ingest_run_id` undoes a bad extractor deploy. Re-extraction targeting ("everything from prompt
v1, with v2") is an indexed query, so prompt improvements are a backfill not a rebuild.
`source_locator` is what lets the agent say "handbook §4.2, page 11".

**Many-to-one arrives with deduplication.** Once facts are deduplicated — and they will be,
because *"Will prefers dark mode"* gets extracted forty times — one proposition has N sources,
and N sources is exactly the signal that should raise confidence:

```sql
CREATE TABLE proposition_provenance (
  proposition_id UUID REFERENCES propositions(id) ON DELETE CASCADE,
  provenance_id  UUID REFERENCES provenance(id),
  PRIMARY KEY (proposition_id, provenance_id)
);
```

Keep `propositions.provenance_id` as the primary source for the hot path. This yields the
erasure story: **deleting a user removes their provenance rows; a proposition left with zero
provenance is deleted; a proposition still corroborated by another user's chat survives, minus
one unit of confidence.** It falls out of the schema rather than needing special-case code.

---

## D6 — Lifecycle: versioned documents, immutable content-addressed chunks, four clocks

### Documents are versioned; chunks are immutable and reference-counted

```
collections               -- org_id, owner_org_id, name, kind, visibility, claim_scope,
                          -- decay_profile, extract_propositions, acl_mode, licence,
                          -- public_source
documents                 -- org_id, collection_id, external_id (stable customer-side ID),
                          -- uri, current_version_id, deleted_at
document_versions         -- document_id, version_no, content_hash, status, ingested_at,
                          -- retired_at
chunks                    -- org_id, content_hash, UNIQUE(org_id, content_hash), text, tsv,
                          -- embedding, refcount, acl_group_id, provenance_id
document_version_chunks   -- (document_version_id, chunk_id, ord)  many-to-many
```

Re-ingest hashes the document first: unchanged hash, no work — which is what makes a nightly full
crawl of a 100k-document corpus cost nothing, and nightly full crawls are what connectors do.
When content has changed, chunk-level hashing means only changed chunks are embedded: a typo
fixed in a 300-page handbook is one embedding call, not 300. Reference counting also deduplicates
boilerplate across documents in the same org, and GC is `refcount = 0`.

> **This only works if chunk boundaries are stable.** Today's `_chunk_text` greedily packs
> paragraphs to a 1,200-character budget, making every boundary dependent on all preceding text:
> insert one paragraph near the top and every subsequent chunk shifts, every hash changes, and
> both chunk dedup and the extraction cache collapse to a 0% hit rate. **Required:** a
> content-defined boundary rule — structural markers (headings, paragraph groups) with a size
> cap, or a rolling-hash boundary condition — so a local edit perturbs only local chunks.

### Four clocks, kept separate

| Clock | Columns | Answers |
|---|---|---|
| Assertion | `asserted_at` *(exists)* | When was this said? Drives decay. For a corpus chunk this **is** the publication date — no new column needed |
| Validity | `valid_from`, `valid_to` | When is it true *in the world*? |
| Belief | `recorded_at`, `invalidated_at`, `invalidation_reason`, `superseded_by` | When did *we* believe it? |
| Retention | `expires_at`, `legal_hold` | When must it be gone? Policy, not truth |

Replace the `superseded_by IS NULL` filter with `invalidated_at IS NULL` plus an enumerated
reason (`superseded`, `source_updated`, `source_deleted`, `ttl`, `user_deleted`, `contradicted`,
`retracted_run`). One nullable timestamp supports a cheap partial index and can be set in bulk
when a version retires, which a self-referencing UUID cannot.

**Full bitemporal is in scope** — three nullable timestamps and an enum, ~110 MB per mid-size
tenant, one hot-path predicate. It stays cheap by keeping the modes separate:

```sql
-- default: current state. Partial index on (invalidated_at IS NULL) serves it.
WHERE invalidated_at IS NULL
  AND (valid_from IS NULL OR valid_from <= now())
  AND (valid_to   IS NULL OR valid_to   >  now())

-- as-of validity: what was true on V, per what we believe now.
WHERE invalidated_at IS NULL
  AND (valid_from IS NULL OR valid_from <= $V)
  AND (valid_to   IS NULL OR valid_to   >  $V)

-- as-of belief: what the system believed on T. Audit path, NOT wired into /recall.
WHERE recorded_at <= $T
  AND (invalidated_at IS NULL OR invalidated_at > $T)
```

The belief-time query cannot use the partial index and defeats HNSW pre-filtering, so it belongs
behind a separate audit endpoint — "why did the agent say that last month?" — where a 200 ms
sequential path over one tenant's partition is acceptable.

### Three decay profiles

One constant cannot serve three kinds of content. Today's `exp(−Δt / 30d) × log(1 + access_count)`
is right for chat and wrong for corpora in *two different directions*.

| Profile | Keyed on | Half-life | Frequency boost | Content |
|---|---|---|---|---|
| `conversational` | `asserted_at` | ~30 days | yes | Chat facts. Fresh genuinely wins |
| `timeless` | — | ∞ (factor 1.0) | no | Policy, definitions, internal architecture. A 2019 expenses policy *is* the expenses policy |
| `perishable` | `asserted_at` = `published_at` | 12–36 months | no | External guidance, vendor docs, articles. A 2019 post about a framework's API is not stale, it is *wrong* |

**Turn the frequency boost off for both corpus profiles.** `log(1 + access_count)` on reference
material is a popularity feedback loop; on shared material it also carries usage across tenants.
Dropping it deletes the read-path write-amplification problem on the largest table for free.

### The update transaction

Retiring the old version and promoting the new one must be **one transaction**, or retrieval
briefly sees both versions or neither. In order: insert new version and its new chunks → link
carried-over chunks → extract for new chunks only (if the collection opts in) → flip
`current_version_id` → invalidate propositions whose provenance points at chunks *not* carried
forward → commit.

*Operational caveat:* pgvector's HNSW does not reclaim deleted-element space eagerly; churn
leaves tombstones and index quality degrades until `VACUUM`, needing periodic
`REINDEX CONCURRENTLY`. Soft-delete on the read path plus grace-period physical GC is right, and
partitioning makes the rebuild tractable — one partition at a time.

---

## D7 — Vector representation: `halfvec` now, binary index with exact rescore later

Take a mid-size tenant: 200 users × 300 chats × 30 turns × 2 facts = **3.6M propositions**, plus
a 50k-document corpus at 12 chunks = **600k chunks**. 4.2M indexed vectors.

| Vector representation | Index B/row | HNSW total | Heap total | Recall impact |
|---|---:|---:|---:|---|
| `vector(1024)` fp32 *(today)* | 4,268 | 16.7 GB | 19.9 GB | baseline |
| `halfvec(1024)` fp16 | 2,220 | 8.7 GB | 11.9 GB | negligible |
| `halfvec(512)` MRL-truncated | 1,196 | 4.7 GB | 11.9 GB | small, model-dependent |
| `halfvec(256)` MRL-truncated | 684 | 2.7 GB | 11.9 GB | noticeable without rescore |
| **`bit(1024)` + fp16 rescore** | **304** | **1.2 GB** | **11.9 GB** | ~95–98% at 4× oversample |

**Decision.** `halfvec` becomes the storage default immediately (phase 0) — it halves both heap
and index for negligible recall cost, and it is a prerequisite for D8 because HNSW indexes
`vector` only to 2,000 dimensions but `halfvec` to 4,000. Binary quantization with exact rescore
(Hamming top-`4k` via `<~>` on a `bit_hamming_ops` index, then exact cosine rescore against the
heap vector) is phase 4, driven by real tenant sizes. Both stages are pgvector primitives.

**RAM, not disk, is the ceiling.** Ten mid-size tenants is 12 GB of index — comfortable. A
hundred is 119 GB of index and 1.3 TB total, which is where you stop adding tenants to one
cluster and shard by `org_id`. The shard key from D3 makes that mechanical. Real SaaS
distributions are power-law, so one cluster serves hundreds of small tenants and whales get
dedicated shards early.

### Latency

| Stage | Now | Fixed | What changes |
|---|---|---|---|
| BM25 keyword | O(corpus), seconds @ 10M | 5–15 ms | Materialise `corpus_stats` (n, avgdl) per (collection, kind) and `lexeme_df` as incrementally-maintained tables; store `length(tsv)` as a generated column. BM25 is a heuristic — slightly stale statistics are fine. Stays vanilla SQL |
| Vector search | 5–20 ms, degrading under filter | <2 ms | Partition pruning + binary first stage. The correctness gain matters more than the speed |
| Graph expansion | low ms | low ms | Cap fan-out **per seed**, not globally, so one hub entity cannot consume the budget |
| **Cross-encoder rerank** | **300–800 ms CPU** | 30–80 ms | **This is p99, not the database.** `bge-reranker-v2-m3` is 568M params scoring 64 pairs. ONNX int8, a smaller reranker, 32 candidates, or GPU — and it must become a separate horizontally-scalable service |
| Access-count bump | N writes/recall | amortised | Append to a log table (or in-process counter) flushed periodically |

**Two ingest pipelines, not one.** Chat ingest is online: single turn, low latency, CPU embedding
fine. Corpus ingest is batch: 600k chunks is 1–2 GPU-hours or a day-plus on CPU, and must not
compete with recall for pool slots or GPU. That means `ingest_jobs` with workers, idempotency by
content hash, per-document progress, backpressure and resumability — visible to the customer,
because "is my corpus indexed yet" is the first question they will ask.

**Speculative, worth a bench arm:** `bge-m3` — already the default embedder — emits learned sparse
lexical weights alongside its dense vector, and pgvector 0.7+ ships `sparsevec` (HNSW-indexable
up to 1,000 non-zeros). That would replace hand-rolled BM25 with learned sparse retrieval and
eliminate document-frequency computation entirely, while keeping everything in Postgres.
Escape hatches (`pg_search`, VectorChord BM25) cost portability, and portability is the thesis.

---

## D8 — Embedder generations

The embedder will change, and two generations may need to coexist to bridge compatibility.
This lands in **phase 1** — retrofitting it across 4M vectors later is a project, not a migration.

```sql
CREATE TABLE embedder_generations (
  id            UUID PRIMARY KEY,
  name          TEXT NOT NULL,     -- 'bge-m3', 'bge-m3@256' (MRL truncation is a
                                   -- generation, not a special case)
  dim           INT  NOT NULL,
  storage_type  TEXT NOT NULL,     -- 'halfvec' | 'bit+rescore'
  normalize     BOOLEAN NOT NULL,
  query_prefix  TEXT,              -- asymmetric models need this, and it must travel
                                   -- with the generation rather than the config
  status        TEXT NOT NULL      -- building | live | primary | retiring | retired
);

CREATE TABLE org_embedders (
  org_id UUID, generation_id UUID, role TEXT,  -- primary | secondary
  PRIMARY KEY (org_id, generation_id)
);
```

**Storage is deliberately asymmetric.** The **primary generation stays inline** on the content
row, because the scoping columns the vector search must filter on live there — move it to a side
table and you re-create the filtered-HNSW problem with the filter on one relation and the index on
another. **Transitional generations go in side tables** (`emb_prop_g2(item_id PK, vec halfvec(768),
bits bit(768))`), partitioned identically, where the extra join is acceptable because the window
is temporary.

**Generation is just another RRF candidate source.** Because fusion is rank-based, no score
calibration is needed across model spaces — comparing cosines from different models is
meaningless, comparing ranks is not. Retrieval during a partial backfill degrades gracefully: an
item present only in the old generation is still retrieved at its old-generation rank.

### Cutover protocol

| Step | Action | Rollback |
|---:|---|---|
| 1 | Register generation, `building`. Create side table + indexes on one partition first to measure build cost | Drop table |
| 2 | Backfill through `ingest_jobs`, throttled below online traffic, resumable. Keyed by **chunk content hash**, so unique text is embedded once | Stop job |
| 3 | Coverage past threshold → `live`, role `secondary`. Retrieval fuses both | Flip status |
| 4 | Evaluate on the existing bench suite per tenant cohort. This is what the bench harness is *for* | — |
| 5 | Promote to `primary`: new writes inline in the new generation, old demoted to `retiring` but still queried | Flip roles |
| 6 | `retired`: drop the old column or table, reclaim space, one partition at a time | Re-backfill |

### Three consequences easy to miss

1. **Query embedding cost multiplies by live generation.** Two live generations means embedding
   the query twice per recall — run concurrently, but budget for it. This is the decisive argument
   that **generation is a property of the org, not the collection**: per-collection embedders
   would mean embedding the query once per collection on every recall.
2. **MMR breaks across generations.** Diversity is a cosine computation and cosines are not
   comparable between model spaces. During a dual window, compute MMR in the primary generation's
   space only; items lacking a primary vector are ordered by the reranker rather than silently
   treated as maximally distinct.
3. **The reranker is generation-independent** — it scores text pairs. That makes it the natural
   arbiter during a transition, and an argument for keeping the stage even under latency pressure.
   `proposition_cache` is keyed on the *extractor* model, so it is correctly unaffected.

---

## Consolidation, contradiction, and the predicate problem

A graph fed by chat accumulates near-duplicates relentlessly; storage and ranking both degrade.
A periodic per-org consolidation job, off the hot path:

- **Exact dedup** — identical normalised text within `(org, claim_scope, subject, predicate)`.
  Merge, add a provenance row, increment corroboration, keep the latest `asserted_at`.
- **Near-dup** — cosine > ~0.95 within the same subject entity. Merge the confident band
  automatically; batch the ambiguous band to an LLM adjudicator.
- **Contradiction** — same `(subject, predicate)`, different object, where the predicate is
  single-valued. Resolve temporally: the newer assertion closes the older's `valid_to`.

Every stage partitions by `claim_scope` and never crosses it (D4). Within `world` scope dedup is
both safe and valuable, because syndication and near-duplicate guidance are everywhere.

> **The blocker nobody expects: free-text predicates make automated expiry impossible.**
> `predicate` comes straight from the LLM today, so *"works at"*, *"is employed by"* and
> *"works for"* are three unrelated relations and no contradiction between them is detectable.
> A `predicates` table with canonical forms, aliases, and crucially `is_functional BOOLEAN` —
> single-valued relations (current employer, date of birth, current city) are the only ones where
> a new value *should* retire the old. Multi-valued relations (attended, mentioned, collaborated
> with) must never auto-invalidate. Small table, large payoff: it is the prerequisite for
> automated fact expiry, and it makes graph expansion relation-typed rather than blind.

Two nearly-free retrieval wins once the corpus tables exist. **Small-to-big**: retrieve on the
chunk, return the surrounding window via `document_version_chunks.ord`. **Summaries as a third
source class**: document- and session-level rollups, embedded and retrievable, answering "what
does this document say overall" that no individual chunk answers. Both drop into the fusion layer
as additional candidate sources — which is why decomposing it comes first.

---

## Target schema

```
-- tenancy -------------------------------------------------------------------
orgs                      NEW
users                     NEW  (or external IDs only, if identity lives upstream)
tenant_shards             NEW  org_id -> shard_key, for partition placement
collections               NEW  owner_org_id NOT NULL (tenant by default; SYSTEM_ORG only
                               for explicitly shared), claim_scope, decay_profile
                               (conversational|timeless|perishable), extract_propositions,
                               acl_mode, licence, public_source
collection_subscriptions  NEW  org_id -> shared collection, enabled, rrf_weight

-- corpus --------------------------------------------------------------------
documents                 EXTEND  + org_id, collection_id, external_id,
                                    current_version_id, deleted_at
document_versions         NEW
chunks                    EXTEND  + org_id, content_hash UNIQUE, embedding, tsv, refcount,
                                    acl_group_id, provenance_id
                                    (now retrievable in its own right)
document_version_chunks   NEW  many-to-many, carries ord
summaries                 NEW  (phase 4) doc- and session-level rollups

-- graph ---------------------------------------------------------------------
propositions              EXTEND  + org_id, collection_id, claim_scope, visibility,
                                    owner_user_id, provenance_id, valid_from/valid_to,
                                    invalidated_at + reason, corroboration_count,
                                    embedding -> halfvec/bit, embedder_generation_id
entities                  EXTEND  + org_id  (resolution is org-wide; expanded neighbours
                                    re-filtered by visibility)
edges                     EXTEND  + relation_id -> predicates
predicates                NEW  canonical relation vocabulary, is_functional
entity_mentions           NEW  the corpus <-> graph join (entity_id, chunk_id, span)
entity_links              NEW  org entity <-> shared entity bridge, per org
entity_pagerank           EXTEND  + per-subscriber recomputation

-- machinery ----------------------------------------------------------------
provenance                NEW
proposition_provenance    NEW  (phase 4, with dedup)
proposition_cache         KEEP  already correct
corpus_stats              NEW  materialised n_total, avgdl per (collection, kind)
lexeme_df                 NEW  materialised document frequency
ingest_jobs               NEW  batch corpus pipeline + re-embedding backfills
access_log                NEW  replaces synchronous access_count writes
embedder_generations      NEW
org_embedders             NEW
emb_<source>_<gen>        NEW  transitional side tables during cutover only
embedding_cache           NEW  content_hash + generation -> vector; public_source only
corroborations            NEW  agrees | tension, across claim scopes
```

SQL surface, decomposed from one function into testable units: `pgkg_bm25_candidates`,
`pgkg_vector_candidates`, `pgkg_graph_candidates`, `pgkg_fuse`, `pgkg_apply_profile`, and a thin
`pgkg_search()` that composes them.

---

## Implementation plan

Phases are numbered because the order is load-bearing: each is independently shippable, and each
removes a constraint the next would otherwise hit.

### Phase 0 — Fix what will break anyway

Materialised BM25 statistics. Bind vectors as parameters. `halfvec` as the storage default.
Parameterise the hardcoded `vector(1024)` out of the DDL and function signatures. Transactional,
set-based ingest. Batched access-count updates. Decompose `pgkg_search()` into composable SRFs.
Content-defined chunk boundaries.

*No schema semantics change, no API change. Unblocks everything below. The BM25, chunker and
dimension-parameterisation fixes are prerequisites, not polish.*

### Phase 1 — Tenancy, provenance, time, and the embedder registry

`org_id`, `collection_id`, `claim_scope`, `visibility`, `owner_user_id` on every retrievable row;
RLS policies; `provenance` with `provenance_id` columns; the full bitemporal column set with
`invalidated_at` replacing the `superseded_by` filter; `collections` carrying decay profile and
claim scope; `embedder_generations` + `org_embedders` with the current model registered as
generation 1. Backfill: existing `namespace` values map to one org, one collection.

*Migration touches every table — do it before there is production data. The registry and
bitemporal columns are here because both are cheap now and expensive to retrofit.*

### Phase 2 — The corpus as a first-class source

Document versions, content-addressed chunks, chunk embeddings and tsvectors, the corpus
retriever, weighted RRF with quotas *by claim scope*, the three decay profiles, small-to-big
context expansion, the batch ingest queue, and the atomic update/retire transaction. Chunks-only
mode becomes "retrieve from the chunk source" and the fake-proposition rows disappear. The
ownership seam lands here — `owner_org_id`, the system org, subscription resolution in the
predicate — because it sits on the hot path; shared collections stay empty.

### Phase 3 — Join the corpus to the graph

`entity_mentions` populated by gazetteer matching over new chunks; bidirectional graph expansion
(facts → entities → chunks and back); the `entity_links` bridge into shared collections;
per-collection `extract_propositions` opt-in; document ACLs. Web-source ingest hygiene — HTML
extraction, boilerplate stripping, syndication dedup — belongs here, because it only matters once
general knowledge is in scope.

### Phase 4 — Scale and hygiene

Partitioning by shard key. Binary-quantized index with exact rescore. Consolidation and
contradiction jobs, partitioned by claim scope. The predicate vocabulary and `corroborations`.
Summaries as a source class. Cold tiering for old sessions. Physical GC and index rebuild
schedule. First live embedder cutover exercised end-to-end on a staging tenant.

*Driven by real tenant sizes. Partition before the first whale; quantize when index RAM becomes
the bill; rehearse the cutover while it is still optional.*

---

## What not to build

- **Don't extract propositions from the whole corpus.** Recurring cost, lossy on exactly the
  content corpora are made of, and the mention edge gets most of the graph benefit for free.
- **Don't add a graph database.** One-hop expansion over an indexed edge table is a join. If
  4+ hop traversal with path semantics is ever genuinely needed, revisit — agentic memory
  retrieval does not need it, and the Postgres-only thesis is the product.
- **Don't put chunk embeddings in `propositions`.** It corrupts BM25 length normalisation,
  conflates lifecycles, and blocks per-store index tuning. Undo the existing instance.
- **Don't shard across clusters before partitioning within one.** Partitioning buys most of the
  isolation and all of the HNSW correctness, with one schema to migrate.
- **Don't auto-reconcile across claim scopes.** Surface the tension; let the model and the human
  decide.
- **Don't build a query router yet.** Quotas plus the reranker will do the job. Revisit only if
  benchmarks show a query class quotas cannot serve.
- **Don't make the embedder a per-collection property.** Generation belongs to the org.
- **Don't merge near-identical facts across claim scopes.** Record the agreement; never collapse it.
- **Don't default anything to shared.** Curation is the tenant's advantage; pooling it is a
  product failure, not an optimisation.
- **Don't dedupe chunk rows across tenants.** Dedupe the embedding computation instead.
- **Don't compute ranking signals globally over shared content.** Per-subscriber, or not at all.
- **Don't merge shared entities into per-org entity space.** That is a per-org copy of the shared
  graph. Bridge with `entity_links`.

---

## Consequences

**Positive.** The Postgres-only thesis survives intact — every mechanism here is vanilla SQL plus
pgvector. Rank-based fusion pays off three times over (cross-store, cross-scope, cross-embedder-
generation), which is the strongest signal the architecture is factored correctly. The corpus
feature removes an existing hack rather than adding one. Erasure and re-extraction become indexed
queries. Storage per tenant drops ~3× on phase 0 alone and ~14× on the index at phase 4.

**Negative.** Phase 1 touches every table, so it must land before production data exists. The
SQL surface grows from one function to six, and the migration discipline (never redeclare a whole
function) has to hold. Bitemporal adds real conceptual load for the team. Two live embedder
generations double query-embedding latency during cutover windows.

**Deferred, with the seam built.** Partitioning, binary quantization, per-tenant physical
isolation, summaries, the predicate vocabulary. All are additive given the columns this ADR
establishes.

**Open, needs data.** A realistic tenant size distribution — the phase 4 partitioning and
quantization decisions are entirely driven by whether the shape is "many small" or "a few
enormous". And whether corpus connectors are pull or push: pull makes the content-hash no-op path
the most valuable optimisation in the pipeline, push makes the atomic version-flip the thing most
likely to break under concurrency.
