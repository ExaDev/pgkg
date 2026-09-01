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

**State at the time of writing.** Phases 0–3 complete, migrations 001–053, 630 tests green.
Phase 2's last outstanding item — chunks-only ingest writing into the chunk store rather than
faking proposition rows — landed in 049; see §3. Four defects found after phase 3 are closed in
051–053 and `pgkg/corpus.py`; see the second fix-pass table in §1. **Migration 050 does not
exist**: the number was assigned to the fix that turned out to need no DDL, and the gap is
deliberate rather than a lost file.
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
| `047_gazetteer_under_row_security.sql` | The same non-leakproof-qual mechanism on `entities`: `similarity_op` and `arraycontains` marked `LEAKPROOF`, and the gazetteer keys stored as generated columns so the name and alias arms compare columns instead of calling the normaliser |

### The second fix pass — four defects found after phase 3

Found by re-reading the shipped tree rather than by a verifier, and fixed in parallel on disjoint
file ownership like the first pass. Each entry is one issue.

| Issue | Where it landed | What it fixes |
|---|---|---|
| **#16 — a passage in two collections lost its vector** | `pgkg/corpus.py` | The reuse lookup restated the content address instead of reading it, and had drifted from it twice (042 added three columns, 049 added two). It answered "already stored and vectored" about a row at another address; the write phase then correctly created a new row and the vector write skipped it, and no later crawl revisits it because the document hash short-circuits first. The lookup is now generated from the unique index that enforces the address, and the residual race — phase 2 saw a row phase 3 did not — is paid for after the transaction instead of stranding the chunk. No DDL was needed. |
| **#17 — entity dedup read every name** | `051_entity_dedup_reaches_the_trigram_index.sql` | `pgkg_link_entity()` stage 2 generated its candidates with `similarity(name, p_name) > 0.6`, a function call over a column that no index can serve, so every near-duplicate check was a sequential scan of `entities` — for the owner as well as for `pgkg_app`, which is what makes it a different defect from the one 047 fixed. It now generates with `name % p_name` against `entities_name_trgm_idx`. Measured on 40,001 names in one org: 78.4 ms and 949 buffers becomes 0.65 ms and 70. |
| **#18 — retrievability was inferred from parentage** | `052_retrievability_stops_being_a_property_of_parentage.sql` | Liveness and the content address's partial predicate both read `chunks.document_id` to mean "this row is not retrievable content", which is not a fact about parentage at all. Both now read `chunks.provenance_only`, which the writer states. The column itself survives as the record of which document a chat-provenance chunk came out of, and 052 records what removing it still costs; see §3. |
| **#19 — nothing ran the gazetteer** | `053_scheduling_what_was_already_built.sql`, `pgkg/maintenance.py`, `pgkg maintain` | Four jobs existed in the schema, were tested, and were reachable only from a test or an operator's psql session. `entity_mentions` was therefore empty in every deployment, which is D2's corpus-to-graph edge missing entirely. One entry point now runs all four, each selectable and each guarded by an advisory lock per (task, org). 053 adds the name-side mention watermark, without which the reverse direction — a name created after the corpus was swept — could never meet the passages that predate it. |

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

### D2's mention payoff is built, wired, and unreachable through `recall()`

The mention edge exists end to end: `pgkg maintain --task mentions` populates `entity_mentions`,
and `pgkg_graph_candidates` emits the document a chat fact's entity points at. What D2 actually
promises — that the document comes back *for a query whose wording matches it not at all* — does
not happen at any corpus size, and the measurement is recorded here so it does not have to be
retaken. Issue #21 carries the decision.

Two independent effects meet. Above roughly 38 competing passages, a graph candidate scores
`w_graph (0.5) × MIN(seed fused score)` — 010's `pgkg_fuse`, the neighbour floor 043 names — so it
sits below every keyword and vector candidate by construction, and `pgkg_apply_quotas` keeps only
`floor(k_rerank × corpus_fraction)` = 38 corpus items, cutting it before the cross-encoder that is
the one stage able to promote it on merit. Below that size the passage is inside the arms' own
`k_initial = 200` list, which makes it a *seed*, and 043's `per_seed` branch excludes seeds from
expansion. Measured: byte-identical `recall()` output with the mention rows present and deleted at
11 items; absent entirely at 310; present only once `k_rerank` reaches 1000, the point at which
the quota stops cutting at all.

**The test that appears to cover this cannot.**
`tests/test_entity_mentions.py::test_a_chat_fact_pulls_in_the_document_that_defines_what_it_names`
passes, and its docstring claims "D2, end to end, over the real ingest pipelines". It calls
`pgkg_retrieve` with `q_text` and no `q_embedding`, so the vector arm returns nothing, the chunk
is never a candidate, and it is therefore never a seed — the one geometry where a graph candidate
survives, and not the one `Memory.recall` runs at, since it always passes an embedding. The test
is correct about the SQL mechanism and says less than a reader assumes. This is the second time a
test has passed for a reason unrelated to the property it names (see §2 on the eight unobservable
RLS policies); both were found by asking what the test does when the thing it guards is removed.

## 3. Conscious deviations from the ADR

Every item here is a place where the implementation does not do what the ADR says, on purpose.

### D6's content address is the scoping columns plus the hash, not `(org_id, content_hash)`

*(042 made it `(org_id, collection_id, acl_group, content_hash)`; 049 added `visibility` and
`owner_user_id` on the same rule — see the entry near the end of this section.)*

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
version". (052 keeps the shape and moves the second arm's guard off the parent pointer: it reads
`NOT provenance_only`, in the position `document_id IS NULL` held. Same answer for every existing
row — see the 052 entry below.) The second clause needs a record the purge used to destroy, which is `chunks.version_scoped`.
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

### The `@@` functions are marked `LEAKPROOF`, and the state is reported

Not in the ADR, and it is a change to a built-in function's catalog entry, so it deserves stating.
On a table with a policy Postgres may not use a qual of a higher security level as an index
condition, and `ts_match_vq` (the function behind `tsvector @@ tsquery`) is not leakproof — so
under any non-exempt role the keyword arm's `@@` was demoted from an Index Cond to a Filter and
every BM25 arm degraded to a sequential scan. 043 marks it leakproof in a `DO` block that degrades
to a `NOTICE` without superuser. The claim is defensible: it raises no data-dependent error, emits
no message carrying either argument, has no side effect, and the boolean it returns is not
observable because the policy qual still filters the row before anything is returned.

046 finishes it in two ways. `@@` is two functions: `tsquery @@ tsvector` is `ts_match_qv`, and the
planner tests leakproofness on the clause as written, before it considers commuting the clause
through the operator's commutator — so the reversed operand order still planned a Seq Scan under
`pgkg_app`, and a future arm was one keystroke away from reinstating the defect. The alternative
was a guard asserting that no retrieval function writes the reversed form; the operator is marked
instead, because a text search for `@@` cannot see a query built in application Python, in a bench
script, or in the next migration, and marking makes both forms correct by construction.

And because the mark cannot be enforced — a managed Postgres will not grant ownership of a built-in
— it is reported. `pgkg_keyword_match_leakproof()` returns the flag for both signatures and
`GET /health` carries it as `keyword_index`, next to the embedder registry, so a deployment where
the `DO` block fell through to its `NOTICE` is a monitorable fact rather than a line in a migration
log. The suite can only ever observe the marked state, so the test that matters drives the
endpoint through the unmarked state inside a transaction it rolls back.

### The same mechanism on the gazetteer, and where a leakproof claim is the wrong remedy

043 fixed the keyword arms and left `entities` — under row security since 020 — probed by three
quals that are all non-leakproof: `similarity_op` (`text % text`), `arraycontains` (`anyarray @>
anyarray`) and `pgkg_gazetteer_key()`. Every arm of `pgkg_match_entity_mentions()` was therefore a
sequential scan of the whole entity table under `pgkg_app`, on the ingest hot path (measured on
40,001 entities: 42.6 / 186.6 / 74.3 ms against 4-70 buffers for the owner's index plans).

047 marks the first two, with 043's argument and 043's `insufficient_privilege` handling.
`arraycontains` carries one stated reservation: it is polymorphic, so the marking covers arrays of
every element type, and its residual leak surface is the element equality it resolves — `texteq`,
itself leakproof, for the only containment this schema promotes.

The third is the interesting one, and the reason this note exists: marking
`pgkg_gazetteer_key()` leakproof was measured to change nothing. The planner inlines a simple SQL
function's body before it judges the qual, so the claim on the wrapper is never consulted, and what
the qual actually contains is `lower`/`regexp_replace`/`btrim` — marking *those* would be a claim
over every qual in the database that lowercases or trims a column. A leakproofness marking on a
function this schema owns also outlives the review of its body. So that arm gets a different
remedy: the two keys are stored generated columns, and the quals become an equality on text
(`texteq`, already leakproof) and containment on a plain `text[]`. The name arm becomes an Index
Only Scan, which the expression index could not offer either.
### The extraction cache is keyed per org, not gated on `public_source`

D4 restricts `embedding_cache` to operator-licensed material because "for a confidential document a
cache hit would confirm another tenant holds it". `proposition_cache` carries the extracted facts
themselves, which is worse. Two options were available: gate it on `policy.public_source` the way
the embedding cache is gated, or attribute each entry to the org that paid for it. The second was
chosen, because it keeps the cache useful on private collections (an org re-ingesting its own
overlapping content still hits) while making a cross-org hit impossible. The consequence is that
the second org to hold a private passage re-extracts rather than reading the first org's facts,
which is the direction D4 chooses.

### Chunks-only mode writes into the chunk store; the extraction path does not (049)

Phase 2's ADR text says "chunks-only mode becomes 'retrieve from the chunk source' and the
fake-proposition rows disappear". Both halves have now shipped.
`Memory.ingest(extract_propositions=False)` — what `--chunks-only` and `PGKG_EXTRACT_PROPOSITIONS=0`
select — runs the corpus pipeline's write phase over a chat turn: one `documents` row, a version
opened against the hash of the whole turn, its passages added through `pgkg_add_version_chunk()`,
the vectors of the passages that call created, then `pgkg_promote_document_version()`. One
transaction, no model call inside it, and **no proposition rows at all**.

`recall()` needed no change: `pgkg_retrieve()` already ran both stores, so a chunks-only caller now
gets rows with `source = 'chunks'`, a `chunk_id`, and a NULL `proposition_id`. The ablation
benchmark reads the chunk store, measured: a clean `--chunks-only` LoCoMo run leaves 419
retrievable content-addressed chunks and zero propositions.

**The extraction path deliberately did not move.** Its chunks exist as provenance for the facts
extracted from them and are not retrievable content (041's bucketing rule turns on exactly that),
so the only shape that would hold them under a document version without making them retrievable is
a version that is never promoted — which is not what 030 means by `pending`. Paying a
`document_versions` row and a `document_version_chunks` row per chat turn, on the hottest ingest
path, to content-address rows no read predicate will ever reach is the pipeline conflation D7
separates the two ingests to avoid. So the extraction path keeps `chunks.document_id`, the
pre-lifecycle single-parent pointer, and keeps its per-chunk provenance locators — the span a
citation names. (052 keeps the decision and removes the reason it needed the pointer: the path now
states `provenance_only` itself, and the pointer is only the record of which document a passage came
out of. See below.)

### The content address gains `visibility` and `owner_user_id` (049), and stays partial

Chunks-only chat ingest is the first writer of a *retrievable private* passage, and that exposed
the half of 042's rule the corpus never exercised. `pgkg_visible()` reads five columns; 042 put
three of them in the address. Every corpus chunk is `shared` with no owner, so the gap cost
nothing — but a user's own note and the org's copy of the same sentence were one row under the old
address, and one row cannot hold two owners. Whichever writer landed first would decide, and the
answer is wrong in both directions: a private note published to the org, or an org passage only one
user can retrieve. So the address is now
`(org_id, collection_id, acl_group, visibility, owner_user_id, content_hash)`, on the same rule and
for the same reason as `collection_id`. It costs disk on the private lane only: `shared`/NULL is one
address, so every corpus row and every default-scoped chat row dedups exactly as before.

030 said widening the address from partial (`WHERE document_id IS NULL`) to total is "one DROP and
CREATE once ingest moves onto the function below". Only half of ingest moved, so it stays partial,
and the claim was re-measured rather than assumed: a total index turns **19 tests red**, all but
one of them the extraction path colliding with itself on repeated text in the default collection,
and the one exception being `test_a_pre_lifecycle_chunk_is_not_content_addressed`, which pins the
partial index on purpose. While `chunks.document_id` remains a single-parent pointer a total
address is unrepresentable anyway: two pre-lifecycle documents in one collection sharing a
paragraph would have to be one row, and that row can name only one parent. Widening it is
therefore not a DROP and CREATE but the removal of `chunks.document_id`, which is a separate piece
of work.

### The claim above is wrong twice, and 052 corrects both halves

**A total address is not the goal, and never was.** The right question is not "which columns key
the address" but "which rows does content addressing govern", and the answer is: the rows that are
retrievable content. A passage stored as provenance for the facts extracted from it must never be
reused by another writer, and the extraction path repeats a paragraph as two rows on purpose — each
carries its own span and its own per-chunk derivation record. So the address is *permanently*
partial, and `WHERE document_id IS NULL` was only ever a proxy for the predicate that decides it.
052 says it directly: `WHERE NOT provenance_only`. Over the rows it covers the address is total,
which is what 030 was reaching for. Re-measured on top of 052: a genuinely total index — one that
content-addresses provenance rows too — turns **22 tests red**, the same 19 plus the three that now
pin the partial predicate on purpose. The number in the paragraph above is right; its conclusion is
not.

**Removing the column is also not what unblocks it.** Three things read `chunks.document_id`:
liveness (045's `document_id IS NULL AND NOT version_scoped`), the address's partial predicate, and
the join from a chat-provenance chunk back to the document it came out of. Only the third is about
parentage. The first two are the same claim — "this row is not retrievable content" — and it is not
a fact about parentage at all, so the answer is that a writer states it. `chunks.provenance_only` is
that statement: an input, in the position `document_id IS NULL` held, guarding the standalone arm
only, so a version that carries a passage still makes it retrievable and every existing row keeps
the answer it had. After 052 nothing in the schema decides retrievability or membership of the
address from the pointer.

The column is still there, and dropping it is still separate work, but the reason is now only the
third use. `chunks.document_id` is the only record of which document a chat-provenance chunk came
out of, and eleven test modules read it as that join. Giving those chunks their parentage through
`document_version_chunks` instead needs a `document_versions` row and a link per chat turn; it moves
the extraction path's provenance from per-chunk to per-version (049's rule: a shared row cannot
record which ingest produced it) and so gives up the span a citation names; and it makes every chat
chunk carried by a version, which flips 041's proposition quota bucket —
`EXISTS (document_version_chunks ...)` — from `memory` to `corpus` for every chat fact, restoring
D1's drowning failure mode. That re-keying belongs with the retrieval statistics.

One object still reads the pointer: `pgkg_chunks_provenance_bridge()`, a `BEFORE INSERT OR UPDATE`
trigger that sets `provenance_only` on a row which names one document and states nothing. Migrations
are forward-only and cannot reach the callers, so without it every writer that still states the
pointer would produce retrievable content-addressed passages the moment the predicates moved.
Measured by deleting the trigger: five tests across three modules assert that inference directly and
two more collide on the address. It only ever sets TRUE, so it cannot overrule a writer
that states its own answer, and its `WHEN` clause keeps every writer that has moved off the pointer
out of the function. Retiring the direct writers, dropping the trigger and dropping the column is
one mechanical change; it is also the prerequisite for making `provenance_only` an absolute veto,
which is the stronger property the extraction path would need if it ever did move onto versions.

**The bench harness now scopes each item to its own collection**, both arms. A namespace isolates
propositions and nothing else — chunks carry no namespace, because D3 replaced stringly-typed
scoping with columns — so without it every LoCoMo conversation's passages were candidates for every
other conversation's question, and the vector arm has no distance threshold to keep them out. Both
arms get one rather than only the chunks arm: an ablation whose two arms are isolated by different
mechanisms is measuring the mechanisms.

---

## 4. Deferred to phase 4

The ADR's Consequences section says "Deferred, with the seam built. Partitioning, binary
quantization, per-tenant physical isolation, summaries, the predicate vocabulary." All of that is
still deferred, and this is what the seam actually looks like.

| Deferred | The seam that exists | What phase 4 has to do |
|---|---|---|
| **Partitioning by shard key** | `tenant_shards` table and `pgkg_tenant_shard()`; every retrievable row carries `org_id` | Convert `propositions` and `chunks` to partitioned tables on the shard key and rebuild the HNSW indexes per partition. D3 calls this a correctness matter, not just speed: a scoped vector search over one global HNSW index under-returns. The interim mitigation is `hnsw.iterative_scan`, set at connection startup by the pool. |
| **Binary quantization with exact rescore** | `halfvec` storage everywhere, width read from the column by `pgkg_embedding_dim()`, `embedder_generations.storage_type` names the representation | Add a binary index and a two-stage retrieve-then-rescore arm. The ADR's ~14× index saving is the motivation; the decision needs real tenant sizes. |
| **Consolidation and contradiction jobs** | **Scheduled** since 053: `pgkg maintain` runs the mention sweep, PageRank, `pgkg_contradict_superseded()` and `pgkg_expire_due()`, each selectable and overlap-safe | What is left is the part that needs a vocabulary: the ADR's "same subject and predicate, different object" rule is undecidable over free-text predicates without a `predicates` table carrying `is_functional`, so the scheduled rule closes the contradiction someone already asserted rather than inferring one. Also unbuilt: a tenant loop — a run names one org (§6). |
| **The predicate vocabulary and `corroborations`** | `propositions.predicate` is free text; `edges.relation` likewise | Both tables named in the ADR are absent. The ADR's rule — record agreement, never collapse it across claim scopes — has nothing to record it in yet. |
| **Summaries as a source class** | `pgkg_retrieve()` takes `p_sources TEXT[]`, and `pgkg_item_scope()`/`pgkg_apply_quotas()` bucket by kind | A third source class plus its own arm and quota bucket. The two-store plumbing generalises to three; the quota shape does not, since the ceiling is currently a single corpus fraction. |
| **Cold tiering for old sessions** | Nothing | Not started. |
| **Physical GC and index rebuild schedule** | `pgkg_purge_retired_versions()`, `pgkg_gc_chunks()`, `pgkg_erase_provenance()`, `pgkg_retract_ingest_run()` all exist with per-org limits, and `pgkg maintain` is now the shape a scheduled job takes | Deliberately *not* added to `pgkg maintain` in 053: these four delete rows where the other four withdraw them, and a destructive job behind the same one-line crontab entry as four idempotent ones needs its own decision. See the sharp edge in §6 about orphans a purge leaves behind. |
| **First live embedder cutover end-to-end** | The whole D8 registry: `embedder_generations`, `org_embedders`, one primary per org, `pgkg_live_generations()`, `pgkg_create_generation_storage()`, the dual-generation vector arm, and the primary-generation restriction on the main arm | Rehearse it on a staging tenant. No cutover has ever been run; the protocol is tested step by step but not as a sequence. |
| **Web-source ingest hygiene** | Nothing | Phase 3's ADR text puts HTML extraction, boilerplate stripping and syndication dedup here. None of it exists — `CorpusIngest` takes text. Relevant the moment a connector points at the open web. |
| **Document ACLs, end to end** | Both halves. `acl_group_id` on `chunks` and `propositions`, read by `pgkg_visible()` and part of the content address; written by `CorpusIngest.upsert_document(acl_group_id=...)`, the `/documents` field, and the queue; `048`'s triggers refuse an untagged row in a collection whose `acl_mode` is not `none` | A permissions source to synchronise groups from: nothing populates a group automatically, so a connector states it or the ACL-bounded collection refuses the document. Also unbuilt: re-grouping content already ingested — the document hash short-circuits an unchanged crawl, so a document does not change group by being offered again. |

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
- **The corpus reuse lookup is generated from the index, not written beside it.** `pgkg/corpus.py`
  reads the content address out of `chunks_content_addressed_key` — key columns, key *expressions*,
  and the partial predicate — and builds the "already stored and vectored?" statement from it. The
  address is stated once, by the object that enforces it, because `pgkg_add_version_chunk()`'s
  `ON CONFLICT` names that same index; a copy of it in Python drifted twice and cost a defect each
  time (#16). A column in the address the pipeline states no value for stops the ingest by name,
  before it spends anything, rather than guessing.
- **The comparison for each key comes from the key expression, not from the column's nullability.**
  042 wraps the two nullable axes in `COALESCE`, so a bare `owner_user_id IS NOT DISTINCT FROM $n`
  cannot be matched to the key however cheap it looks, and the address would stop being
  index-usable at the third of its six columns. The lookup is also driven one probe per hash
  through a `LATERAL`, because a trailing `= ANY(...)` is not an index condition and the hash is
  the last key and the only selective one. Measured on 30,000 passages in one collection: 2,096
  buffers as a sequential scan, 30,688 as the index prefix with the hash filtered, 6 as one probe
  per hash.
- **`pgkg_link_entity()` pins its own trigram threshold in `proconfig`.** `SET
  pg_trgm.similarity_threshold = 0.6` on the function, not `set_limit()` and not `SET LOCAL` in the
  body: Postgres saves and restores it around each call, so the caller cannot narrow entity dedup
  by raising the threshold and the function cannot leave the caller's `%` redefined — the
  gazetteer's fuzzy arm included. The `similarity(name, p_name) > 0.6` recheck is kept behind the
  operator although it is provably redundant (no `float4` widens to exactly 0.6), so that a later
  migration that replaces the function and forgets the `SET` can only lose matches, never invent
  merges.
- **The gazetteer runs on a timer, not on the ingest path.** D7 rules out both online placements: a
  corpus ingest must not hold a pooled connection across a cross-product against every name the org
  knows, and a chat ingest must not match one new name against an unbounded corpus on the request
  path. `pgkg maintain` drains both directions of the sweep instead, and both watermarks are what
  make it re-runnable. An inline `match_chunks()` after a version is promoted was considered and
  rejected: it is a latency optimisation, it is not what makes the edge exist, and a best-effort
  call whose failures are swallowed on the hottest write path is a worse thing to own than a job
  whose report says what it did.

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
- **A chunk whose post-transaction repair fails keeps `embedding IS NULL` for good.** When phase 2
  sees a vectored row that phase 3 finds gone, `CorpusIngest` embeds the replacement after the
  version is promoted (#16). If that embed or its write raises, the caller sees the failure — but
  the document is already committed, so re-offering it short-circuits on the unchanged hash and the
  passage is retrievable by the keyword arm only. The durable answer is a sweep over
  `chunks WHERE retrievable AND embedding IS NULL`, which is a fifth maintenance task and a
  decision about a scheduled job that spends money at the embedder; it is deliberately not in
  `pgkg maintain`.
- **`pgkg maintain` runs one org and has to be installed.** Nothing schedules it: a deployment with
  no crontab entry still has an empty `entity_mentions`, which is the defect #19 closed the
  mechanism for and not the operational habit. `Maintenance.for_org()` builds the per-tenant runner,
  but iterating every tenant needs a tenant list and a fairness policy — the problem the ingest
  queue's claim-and-lease design already solves for its own workload.
- **`pgkg_chunks_provenance_bridge()` is the last reader of `chunks.document_id`.** A `BEFORE INSERT
  OR UPDATE OF document_id` trigger that sets `provenance_only` on a row which names a document and
  states nothing. It exists because a forward-only migration cannot reach the callers, it only ever
  sets TRUE so it cannot overrule a writer that states its own answer, and it is what makes dropping
  the column mechanical: retire the direct writers, drop the trigger, drop the column.

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
