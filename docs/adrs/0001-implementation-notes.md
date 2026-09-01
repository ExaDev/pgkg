# ADR-0001 implementation notes

What phases 0–3 actually built, where it deviates from
[`0001-corpus-embeddings-and-knowledge-graph.md`](0001-corpus-embeddings-and-knowledge-graph.md),
and what is left for phase 4.

The ADR is the design authority and has not been edited. This document is the record of the
build: it exists so that someone picking the work up does not have to reconstruct the reasoning
from `git log`, and so that every conscious departure from the ADR is a decision on the record
rather than a discrepancy to be discovered.

Companion documents:

- [`0001-verification-findings.md`](0001-verification-findings.md) — the 26 defects three
  adversarial verifiers found in phases 0–3, each with its reproduction and the commit that closed
  it, plus the re-verification pass that audited the fixes.

**State at the time of writing.** Phases 0–3 complete, migrations 001–045, 534 tests green.
Run them with:

```bash
PGKG_OFFLINE_EXTRACT=1 uv run --python 3.12 pytest -q
```

---

## 1. What shipped

Each phase is one or two commits, and each migration file's header states in prose why it exists.
The headers are the primary documentation for the SQL; this table is the index into them.

### Phase 0 — Fix what will break anyway (`9c6faba`)

| ADR item | Where it landed | Notes |
|---|---|---|
| Materialised BM25 statistics | `011_bm25_stats.sql` | `corpus_stats` + `lexeme_df`, maintained by statement-level triggers with transition tables. Re-keyed in 041 (see §3). |
| Decompose `pgkg_search()` into composable SRFs | `010_search_decompose.sql` | `pgkg_bm25_candidates`, `pgkg_vector_candidates`, `pgkg_graph_candidates`, `pgkg_fuse`, `pgkg_apply_profile` over a `pgkg_candidate` row type |
| `halfvec` as the storage default, dimension out of the DDL | `012_halfvec_dims.sql` | `pgkg_embedding_dim()` reads the width off the column; no `vector(1024)` anywhere |
| Bind vectors as parameters | `pgkg/memory.py` | The pool registers the pgvector codec and sets `hnsw.iterative_scan` at connection startup |
| Transactional, set-based ingest | `pgkg/memory.py` | Chat ingest is 5 statements for 12 chunks and 60 propositions |
| Batched access-count updates | `pgkg/memory.py` | An in-process ledger flushed per org |
| Content-defined chunk boundaries | `pgkg/chunking.py` | So a typo fixed mid-document does not reflow every chunk hash after it |
| Idempotent entity resolution | `013_link_entity_idempotent.sql` | Two concurrent linkers of one name cannot both insert |

### Phase 1 — Tenancy, provenance, time, embedder registry (`441f362`, `f744229`)

| ADR item | Where it landed | Notes |
|---|---|---|
| Scoping columns on every retrievable row | `020_tenancy.sql` | `org_id`, `collection_id`, `claim_scope`, `visibility`, `owner_user_id`, `acl_group_id` |
| One shared visibility predicate | `020_tenancy.sql` | `pgkg_visible()`, inlinable, used by every arm |
| RLS policies and the `pgkg_app` role | `020_tenancy.sql`, `033`, `043` | Read side widens to the operator's org, `WITH CHECK` never does |
| Provenance as a shared immutable derivation record | `021_provenance_bitemporal.sql` | `provenance`, `proposition_provenance`, append-only by trigger |
| Full bitemporal column set | `021_provenance_bitemporal.sql` | `invalidated_at` replaces the `superseded_by` filter; `pgkg_believed_at()` answers as-of |
| Collections carrying decay profile and claim scope | `023_collections.sql` | Plus `visibility`, `public_source`, `licence`, `owner_org_id` |
| Embedder registry | `022_embedder_generations.sql`, `026_org_binding.sql` | `embedder_generations` + `org_embedders`, current model registered as generation 1, one primary per org enforced by index |
| Entity scope and PageRank scope | `024`, `025` | Entity resolution is org-wide, PageRank per org |

### Phase 2 — The corpus as a first-class source (`3143b31`, `3eb3e88`)

| ADR item | Where it landed | Notes |
|---|---|---|
| Versioned documents | `030_corpus_lifecycle.sql` | `documents` + `document_versions`, one current version enforced by partial unique index |
| Immutable content-addressed chunks | `030`, narrowed in `042` | Address is `(org_id, collection_id, acl_group, content_hash)` — see §3 |
| Reference counting and two-pass reclamation | `030` | `pgkg_purge_retired_versions()` then `pgkg_gc_chunks()`, on separate clocks |
| The atomic version flip | `030` | `pgkg_promote_document_version()` takes the document row lock, retires then promotes, withdraws facts derived from dropped chunks |
| The corpus retriever | `031_corpus_retrieval.sql` | `pgkg_retrieve()` composes both stores |
| Weighted RRF with quotas by claim scope | `031`, re-bucketed in `041` | `pgkg_item_scope()` + `pgkg_apply_quotas()` |
| Three decay profiles | `031` | `conversational`, `timeless`, `perishable`, per collection |
| Small-to-big context expansion | `031`, scoped in `042` | `pgkg_chunk_window()` |
| Batch ingest queue | `032_corpus_ingest.sql`, `pgkg/ingest_jobs.py` | `ingest_jobs` with claim/progress/finish/fail, `pgkg worker` |
| The ownership seam | `023`, `033`, `043` | System org, `owner_org_id`, `collection_subscriptions`, subscription resolution |
| Python surface | `pgkg/corpus.py`, `pgkg/api.py` | `CorpusIngest`, `POST /collections`, `POST /documents`, `POST /documents/delete`, `GET /jobs/{id}` |
| `recall()` moves onto the two-store surface | `pgkg/memory.py` | `_RECALL_SQL` calls `pgkg_retrieve()`, not `pgkg_search()` |

### Phase 3 — Join the corpus to the graph (`2f8c498`)

| ADR item | Where it landed | Notes |
|---|---|---|
| `entity_mentions` by gazetteer matching | `040_entity_mentions.sql`, `pgkg/gazetteer.py` | Matched over new chunks, `chunks.mentions_matched_at` is the queue |
| Bidirectional graph expansion | `040` | Facts → entities → passages and back, every hop re-filtered through `pgkg_visible()` |
| The `entity_links` bridge | `040` | Org-side entity to shared-side entity, direction enforced by trigger |
| Per-collection `extract_propositions` opt-in | `031` | D2's default is that the corpus is *not* extracted |

### The fix pass (`ec44bd9`) and its completion (`976034e`)

| Migration | What it fixes |
|---|---|
| `041_retrieval_statistics.sql` | Statistics keyed per `(kind, namespace, org_id, collection_id)`; retrievability stored on `chunks.retrievable`; quota bucketing on structure; single-pass BM25 scoring; index-served IDF; generation-restricted vector arm |
| `042_window_scope_and_lifecycle_races.sql` | Content address narrowed to the scoping columns; the context window carries its own predicate; `pgkg_open_document_version()` locks the document; cross-org links refused by trigger |
| `043_isolation_and_the_graph_arm.sql` | `ts_match_vq` marked `LEAKPROOF`; read policies widened to the operator's org; four unpolicied tables policied; `proposition_cache` attributed to an org; the graph arm takes `p_sources` |
| `044_queue_provenance_and_two_keys.sql` | `documents_external_id_key` partial on `deleted_at IS NULL`; `proposition_cache` keyed `(cache_key, org_id)`; provenance carried through the ingest queue |
| `045_link_table_writes.sql` | The link table's write side is own-org on both sides; the missing `UPDATE` reconciliation; liveness stops being a function of `refcount` |

---

## 2. Two rounds of adversarial verification

Worth knowing about, because it is the reason a third of the SQL exists and because the pattern is
worth repeating.

Phases 0–3 were audited by three independent adversarial verifiers — SQL and retrieval
correctness, tenant and user isolation, performance and operational soundness — each instructed to
mark a finding confirmed only if it had actually been reproduced. They found **26 items: 4
critical, 10 high, 11 medium, 1 control**, and wrote them into three read-only
`tests/test_verify_*.py` files as 25 failing tests: an executable specification the fix agents were
not allowed to edit. Four fix agents then worked in parallel on disjoint file ownership, each
required to mutation-probe every fix by breaking it and confirming a test failed.

A **fourth pass re-audited the fixed tree**, and found one critical incompletely fixed, one
regression the fixes had introduced, and — by pulling on the second — one defect that had shipped
with phase 2 and that no verifier had caught. All are in
[`0001-verification-findings.md`](0001-verification-findings.md) under `## Re-verification`.

Two things this process taught that generalise:

- **A test can be vacuous in a way that looks rigorous.** `tests/test_verify_perf.py`'s
  keyword-index test switched to `pgkg_app` with `SET LOCAL ROLE` outside an explicit transaction
  block, where it is scoped to asyncpg's implicit single-statement transaction — so it measured the
  owner's plan twice and passed against the very defect it was written to catch. Its replacement
  asserts `current_user` and asserts the policy's own qual appears in the plan *before* asserting
  the index, so it cannot pass by measuring the wrong role.
- **Two read-only tests can be jointly unsatisfiable.** H2's verify test forces a passage shared by
  two collections to be a row in each; H4's then picks an arbitrary one of those rows with an
  unordered `SELECT` and calls that row's own document a leak. Three of the 28 audit tests were
  retired for reasons of this kind, each replaced by a test that fails against the pre-fix shape.
  Their disposition is in §7.

---

## 3. Conscious deviations from the ADR

Every item here is a place where the implementation does not do what the ADR says, on purpose.

### D6's content address is `(org_id, collection_id, acl_group, content_hash)`, not `(org_id, content_hash)`

**The ADR contradicts itself here.** D6's schema sketch writes `UNIQUE (org_id, content_hash)` on
`chunks` and lists no `collection_id`; D3 requires `collection_id NOT NULL` on every retrievable
row, and 020 duly added it. Those two cannot both hold once a passage is in two collections, and
the contradiction is what findings C2 and H2 cost.

It is resolved in favour of D3. A chunk derives its org, collection, ACL group and provenance from
its version's document, and `ON CONFLICT DO NOTHING` keeps the row it found — so the second
collection to contain a passage got the first collection's scope permanently, and `collection_id`
selects the claim scope, the decay profile, the BM25 statistics domain and the quota bucket as well
as being read by the visibility predicate. One row cannot hold two values of a column two
documents disagree about, and copying the second document's values over the first only moves the
wrong answer to the other document. So dedup stops at the collection and ACL boundary for exactly
the reason D4 already stops it at the org boundary: a row shared across a scoping boundary cannot
carry the boundary.

`provenance_id` and `asserted_at` are deliberately *not* in the address. They are derivation and
belief clocks, not scoping columns; no read predicate consults them, and D5 makes provenance a
shared immutable record, so the first derivation stays the true one — the same rule 021 applies to
an invalidation reason.

What the narrower address costs is disk, not model calls: the expensive half is shared through
`embedding_cache`, which is keyed on `content_hash` alone. The case the nightly full crawl depends
on — the same document re-crawled, and boilerplate repeated inside one collection — dedups exactly
as before. **The ADR should be amended to say this.**

### D1's corpus ceiling keys on structure, not on `collections.kind`

The ADR describes the 60% corpus ceiling and the per-claim-scope split in terms of the collection's
`kind`. The `kind` vocabulary is `('chat', 'corpus', 'mixed')` and the reserved default collection
is `'mixed'`, so every passage on the default path — which is what `POST /documents` and both
`Memory` and `CorpusIngest` fall back to — was bucketed `memory` and competed for the slots the
8-row personal-memory floor exists to protect.

041 buckets on structure instead. A chunk is always `corpus`: a retrievable chunk is a document
passage by construction, since chat provenance is not retrievable at all. A proposition is `memory`
unless it is corpus-derived, meaning it lives in a `kind = 'corpus'` collection (D2's extraction
opt-in) or it cites a chunk that a document version carries (D5's derivation edge). Bucketing on
`claim_scope` was considered and rejected: a user-scoped passage would escape the ceiling and a
chat fact about the org would compete for corpus slots, inverting both halves of D1's protection.
Same intent, stated in something with no vocabulary to extend.

### Liveness is not a function of `refcount`

D6 defines a live chunk in terms of the reference count. It cannot be: `refcount` 0 is live for a
standalone passage and not live for chat provenance, and `refcount` 1 is live for a current-version
link and not live for a retired-only link. 031's predicate papered over this with
`(refcount = 0 AND document_id IS NULL)` as the standalone case, which an *orphan* also matches —
so purging a retired version resurrected the passages its successor had dropped.

Liveness is now `chunks.retrievable`, a stored column reconciled by trigger, derived from "carried
by the current version of a live document, or belonging to no document and never carried by any
version". The second clause needs a record the purge used to destroy, which is `chunks.version_scoped`.
`refcount` keeps its documented meaning — a count of `document_version_chunks` links, which is what
the collector reads — and is out of the liveness predicate entirely.

### An ambiguous context window is no window

031 gave a passage carried by several live current versions the window of the lowest
`document_version_id`, calling it arbitrary but stable. It was neither: which of two random UUIDs
sorts lower decided whose prose was returned, and it leaked across collections at a measured
30–70% rate. There is no correct answer to pick — the neighbours of a footer in one handbook are
not the context of the same footer in another, and attributing them to a citation that does not
carry them is a fabrication whichever document wins.

So a passage with more than one carrier gets no window row, and `pgkg_retrieve()`'s existing
`COALESCE` returns the passage as its own context, which is already what 031 does for a chunk with
no version links. The ambiguity is measured row-locally, on the anchor row's own carriers, never
over content elsewhere in the org: a rule that suppressed the window because the same text exists
in a collection the caller cannot see would make the read path depend on out-of-scope content,
which is the same inference channel D4's statistics rule is about.

**The cost, and it is real:** deduplicated boilerplate loses its context entirely. Five documents
in one collection sharing a footer are one chunk row with five carriers, so that passage never has
a window. This applies to every repeated passage in a corpus, because the narrowed content address
deliberately keeps within-collection boilerplate on one row.

### The context window takes no caller scope, deliberately

The obvious fix for the laundering defect (C2) is to pass the caller's org, collection, user and
ACL group into `pgkg_chunk_window()`. It is the wrong fix. A window is built *after* ranking, so
the honest rule is not "what may the caller read" but "is this neighbour indistinguishable from the
anchor to every read predicate". Each neighbour must agree with the anchor chunk on `org_id`,
`collection_id`, `acl_group_id`, `visibility` and `owner_user_id`, and the version supplying the
window must belong to a live current document of the anchor's own org and collection. Any caller
for whom the anchor passed `pgkg_visible()` therefore passes it for every neighbour, whatever
arguments they passed — so `context_text` can never widen the grant that admitted the anchor, and
no caller has to remember to scope it. `pgkg_retrieve()` did not have to change.

**The cost:** the agreement is chunk-to-chunk, not caller-to-chunk. A caller who *is* granted ACL
group A, reading an `acl_group_id IS NULL` header in a document whose body carries group A, gets no
body in context — and `ord_from`/`ord_to` shrink to match, so a withheld neighbour is
indistinguishable from an absent one.

### The RLS read widening does not gate on a subscription

D3 makes the collection list "own plus subscribed, resolved by the app". The read policies widen to
the operator's system org unconditionally; which shared collections a tenant actually retrieves from
is the subscription term in the retrieval predicate, supplied by the caller's `Scope`. Expressing
the subscription in the policy would put a correlated `EXISTS` in the policy of the two largest
tables, evaluated for every row of every other org a scan touches, and two of the widened tables
(`corpus_stats`, `lexeme_df`) carry no `collection_id` to gate on. RLS stays the org boundary.

### Cross-org link integrity is a trigger, not a policy term

C4 asked for the chunk side to be validated in `document_version_chunks`' RLS policy. 042 added
statement-level triggers instead, which is strictly stronger: a link between two orgs is
meaningless for an owner or `BYPASSRLS` connection too, where a policy is inert. 045 then added the
own-org write policy as well, because the *write authorisation* question — may this tenant write
here — is a different question from the *integrity* question the trigger answers, and 043's read
widening made the first one reachable. Both are in place; neither is redundant.

### `ts_match_vq` is marked `LEAKPROOF`

Not in the ADR, and it is a change to a built-in function's catalog entry, so it deserves stating.
On a table with a policy Postgres may not use a qual of a higher security level as an index
condition, and `ts_match_vq` (the function behind `tsvector @@ tsquery`) is not leakproof — so
under any non-exempt role the keyword arm's `@@` was demoted from an Index Cond to a Filter and
every BM25 arm degraded to a sequential scan. 043 marks it leakproof in a `DO` block that degrades
to a `NOTICE` without superuser. The claim is defensible: it raises no data-dependent error, emits
no message carrying either argument, has no side effect, and the boolean it returns is not
observable because the policy qual still filters the row before anything is returned.

### The extraction cache is keyed per org, not gated on `public_source`

D4 restricts `embedding_cache` to operator-licensed material because "for a confidential document a
cache hit would confirm another tenant holds it". `proposition_cache` carries the extracted facts
themselves, which is worse. Two options were available: gate it on `policy.public_source` the way
the embedding cache is gated, or attribute each entry to the org that paid for it. The second was
chosen, because it keeps the cache useful on private collections (an org re-ingesting its own
overlapping content still hits) while making a cross-org hit impossible. The consequence is that
the second org to hold a private passage re-extracts rather than reading the first org's facts,
which is the direction D4 chooses.

### Chunks-only mode still writes proposition rows

Phase 2's ADR text says "chunks-only mode becomes 'retrieve from the chunk source' and the
fake-proposition rows disappear". Half of this shipped. The chunk source exists, is retrievable in
its own right, and `recall()` goes through `pgkg_retrieve()` with both stores — so the *corpus*
path never creates a fake proposition. But `Memory.ingest(extract_propositions=False)`, which is
what `--chunks-only` and `PGKG_EXTRACT_PROPOSITIONS=0` select, still writes one proposition row per
chunk with NULL `subject`/`predicate`/`object` (`_plan_chunks_only` in `pgkg/memory.py`).

This is a deferral, not a decision: removing it means changing what `POST /memorize` returns and
what the `Result` shape means for a chunk-mode caller, and the ablation benchmarks compare against
the current shape. It should be done before the fake rows are load-bearing anywhere else.
Documented in the README's "two modes" section so it does not mislead.

---

## 4. Deferred to phase 4

The ADR's Consequences section says "Deferred, with the seam built. Partitioning, binary
quantization, per-tenant physical isolation, summaries, the predicate vocabulary." All of that is
still deferred, and this is what the seam actually looks like.

| Deferred | The seam that exists | What phase 4 has to do |
|---|---|---|
| **Partitioning by shard key** | `tenant_shards` table and `pgkg_tenant_shard()`; every retrievable row carries `org_id` | Convert `propositions` and `chunks` to partitioned tables on the shard key and rebuild the HNSW indexes per partition. D3 calls this a correctness matter, not just speed: a scoped vector search over one global HNSW index under-returns. The interim mitigation is `hnsw.iterative_scan`, set at connection startup by the pool. |
| **Binary quantization with exact rescore** | `halfvec` storage everywhere, width read from the column by `pgkg_embedding_dim()`, `embedder_generations.storage_type` names the representation | Add a binary index and a two-stage retrieve-then-rescore arm. The ADR's ~14× index saving is the motivation; the decision needs real tenant sizes. |
| **Consolidation and contradiction jobs** | `pgkg_contradict()` and `pgkg_expire_due()` exist and are tested; `invalidated_at`, `valid_from`, `valid_to` and `invalidation_reason` are all populated | Schedule them, partitioned by claim scope. Nothing runs them today — they are called by tests and by an operator, not by a job. |
| **The predicate vocabulary and `corroborations`** | `propositions.predicate` is free text; `edges.relation` likewise | Both tables named in the ADR are absent. The ADR's rule — record agreement, never collapse it across claim scopes — has nothing to record it in yet. |
| **Summaries as a source class** | `pgkg_retrieve()` takes `p_sources TEXT[]`, and `pgkg_item_scope()`/`pgkg_apply_quotas()` bucket by kind | A third source class plus its own arm and quota bucket. The two-store plumbing generalises to three; the quota shape does not, since the ceiling is currently a single corpus fraction. |
| **Cold tiering for old sessions** | Nothing | Not started. |
| **Physical GC and index rebuild schedule** | `pgkg_purge_retired_versions()`, `pgkg_gc_chunks()`, `pgkg_erase_provenance()`, `pgkg_retract_ingest_run()` all exist with per-org limits | Schedule them. See the sharp edge in §6 about orphans a purge leaves behind. |
| **First live embedder cutover end-to-end** | The whole D8 registry: `embedder_generations`, `org_embedders`, one primary per org, `pgkg_live_generations()`, `pgkg_create_generation_storage()`, the dual-generation vector arm, and the primary-generation restriction on the main arm | Rehearse it on a staging tenant. No cutover has ever been run; the protocol is tested step by step but not as a sequence. |
| **Web-source ingest hygiene** | Nothing | Phase 3's ADR text puts HTML extraction, boilerplate stripping and syndication dedup here. None of it exists — `CorpusIngest` takes text. Relevant the moment a connector points at the open web. |
| **Document ACLs, end to end** | `acl_group_id` on `chunks` and `propositions`, read by `pgkg_visible()`, part of the content address, tested through SQL | No ingest path or HTTP field sets it: `corpus.py` passes `NULL` for the ACL group. The read half is done and the write half is a parameter away. |

---

## 5. Non-obvious decisions worth knowing

Things that will look wrong at first glance and are not.

- **Statement-level triggers with transition tables, everywhere.** The bulk path links every chunk
  of a document in one statement, so a per-row trigger would turn a 500-chunk ingest into 1,000
  extra queries. Every derived-state trigger in 011, 030, 041, 042 and 045 is
  `FOR EACH STATEMENT ... REFERENCING`.
- **Reconciliation, not signed deltas, wherever a statement can move a row both ways.** A repoint
  of a link is a decrement and an increment at once, and an ord-only update is neither. Reconciling
  from the source of truth also makes a trigger idempotent, which is what lets a chunk inserted and
  linked in one statement be visited twice and counted once.
- **`pgkg_chunk_live()` still exists and the retrieval arms are asserted not to call it.** It is the
  readable statement of what liveness means and the maintenance path's own test of it. What the
  hot path must not do is call a function whose body reads a table: `inline_function` rejects any
  body with a non-empty range table, so every such call is a nested loop per candidate row.
  `tests/test_retrieval_plan_shape.py` pins the arms, not the function.
- **The retrieval statistics filter mirrors the candidate scan's scope filter term for term.** Org
  and collection both, `NULL` meaning unrestricted in both places. A statistics domain wider than
  the scan is a cross-tenant inference channel; narrower is a wrong IDF.
- **`WITH CHECK` is a single-org equality on every table, without exception.** That is the
  schema-level statement of D4's first hard rule. Wherever reads widen, the widening is in `USING`
  only.
- **A worker draining several orgs sets the job's org on every connection it uses.** Progress,
  finish and fail included, not just the claim — otherwise `pgkg_current_org()` falls back to the
  default org and the `ingest_jobs` policy cannot help.

---

## 6. Known residuals and sharp edges

Not defects with a fix pending — things a maintainer should know before being surprised by them.

- **A passage already orphaned by a purge cannot be recovered.** `chunks.version_scoped` records
  that a version once carried a chunk, but on a database that ran a purge before migration 045 the
  record was already destroyed. Those chunks read as standalone passages and the collector is their
  only exit. Fresh installations are unaffected.
- **`pgkg_gc_chunks()` will not collect a chunk a proposition cites.** Deliberate — "delete this
  passage" and "delete everything we learned from it" are different decisions — but it means an
  orphaned, extracted-from passage stays in `chunks` indefinitely. It is not retrievable, so this
  is a storage matter, not a correctness one.
- **Moving a document to another collection after ingest silently removes its chunks' windows.**
  031 named cross-collection moves unsupported; the window's carrier CTE now requires the
  document's collection to match the chunk's, so what used to be a wrong window is now no window.
  `pgkg_refresh_chunk_stats()` repairs the statistics half.
- **A second offer of already-queued content gets a handle to the existing job and does not restate
  its provenance.** So a re-`POST` with a corrected `published_at` while the first is still queued
  does not correct it.
- **`tests/conftest.py` connects as the container's owning superuser**, for whom every RLS policy is
  inert. Any isolation test that does not explicitly `SET LOCAL ROLE pgkg_app` *inside a
  transaction* is testing the SQL predicate only, never the policy. `tests/test_rls_coverage.py`,
  `tests/test_app_role.py` and `tests/test_api_scoping.py` are the ones that do.
- **The RLS coverage guard cannot see a table whose rows belong to an org without carrying the
  column** — `edges`, `proposition_provenance`, and `corroborations` when it arrives. Stated in
  that module's docstring.
- **Two live embedder generations double query-embedding latency during a cutover window.** The
  ADR says so; it has never been measured, because no cutover has been run.

---

## 7. The audit suite, and where its assertions went

The three `tests/test_verify_*.py` files were an executable specification, not a permanent suite:
28 tests written by verifiers who owned no production code, in files the fix agents could not edit.
They were never committed. Once the findings were closed they were retired into the topical suite,
so that each assertion lives next to the code it constrains. Recorded here because deleting a test
file is the kind of thing that looks like a cover-up in six months.

**Relocated** — into the file that owns the subject:

| From | To |
|---|---|
| `test_carried_over_chunk_keeps_first_collection` | `test_corpus_lifecycle.py::test_a_passage_in_two_collections_is_a_row_in_each` |
| `test_chat_provenance_chunks_do_not_skew_chunk_bm25` | `test_retrieval_statistics_scope.py` |
| the three quota and bucket tests | `test_corpus_retrieval.py` |
| `test_primary_vector_arm_stays_in_the_primary_model_space` | `test_embedder_generations.py` |
| `test_chunk_window_stays_inside_the_queried_collections` | `test_context_window_scope.py`, in a deterministic form |
| both context-window isolation tests | `test_context_window_scope.py` |
| the cross-org IDF test | `test_bm25_stats.py` |
| the cross-org link test | `test_tenancy.py` |
| the graph re-filter control (L1) | `test_entity_mentions.py`, strengthened to both directions |
| the job-status test | `test_corpus_ingest.py` |
| the two BM25 plan-shape tests | `test_retrieval_plan_shape.py` (new file) |
| round-trip count, no-embed-in-transaction, worker deadlock and its control | `test_corpus_ingest.py` |
| the concurrent-crawl test | `test_corpus_lifecycle.py` |
| the access-flush test | `test_memory.py` |
| the `pgkg_search` recall test | `test_entity_mentions.py` |

**Dropped, with a reason:**

| Test | Why |
|---|---|
| `test_perishable_decays_on_publication_date` | Asserts a hard-coded factor computed for an age of exactly 11×365 days with `rel=0.2`; the article's real age passed out of the tolerance band on 2026-06-09. Replaced by tests in `test_corpus_ingest.py` that read the age off the database, so the specification cannot expire. |
| `test_the_keyword_index_survives_the_application_role` | Vacuous: `SET LOCAL ROLE` outside a transaction block measured the owner twice, and it passed at HEAD against the defect it was written for. Replaced by `test_rls_coverage.py::test_the_policy_does_not_cost_the_keyword_arm_its_index`. |
| `test_chunk_liveness_does_not_cost_a_subquery_per_candidate` | Asserts the *function* costs no more than 3× an inline predicate, which requires liveness to be decidable with no table access. Replaced by three assertions in `test_retrieval_plan_shape.py` about what the arms call and what the flag costs. |
| `test_reingesting_a_deleted_document_starts_a_new_one` | Subsumed by `test_a_withdrawn_document_keeps_the_id_it_was_ingested_under`, which asserts the same re-ingest plus the audit trail. |
| `test_forget_cannot_reach_into_the_shared_operator_collection` | Superseded by `test_write_scope.py`, which covers the class under a wider scope with a non-vacuity control. |
| `test_the_extraction_cache_does_not_answer_another_orgs_probe` | Split: the schema-and-policy half is `test_rls_coverage.py`'s `proposition_cache` case, the payload half is two tests in `test_extract_cache.py`. |

**One trap found while relocating, worth remembering.** Moving the vector-generation test out of
the audit file made it run *before* other tests instead of last, and the cutover it builds demotes
generation 1 and registers a second primary installation-wide — so every org provisioned afterwards
violated `org_embedders_one_primary_idx`. It now builds and reads the cutover inside one
transaction and rolls it back, which is honest because everything it asserts is a read. The same
latent pollution existed in the audit file and was hidden only by `test_verify` sorting last.
Global-count assertions have the same shape: `test_no_collection_is_subscribed_implicitly` used to
ask for a count of zero over the whole `collection_subscriptions` table, which is an assertion
about every other test in the suite; it now asks of its own fresh org.
