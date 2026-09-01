# ADR-0001 verification findings

Three adversarial verifiers audited the phases 0-3 implementation independently: SQL and
retrieval correctness, tenant and user isolation, and performance and operational soundness.
Each was instructed to mark a finding confirmed only if it was actually reproduced, and every
finding below carries the reproduction that confirmed it.

**26 confirmed: 4 critical, 10 high, 11 medium, 1 control.** Verdicts were `concerns`, `broken`,
`broken`.

The evidence lived in `tests/test_verify_sql_retrieval.py`, `tests/test_verify_tenant_isolation.py`
and `tests/test_verify_perf.py` — 25 failing tests, one or more per finding. Those files were the
specification for the fixes and were **read-only** to the fix agents: a fix was done when the
relevant test passed without the test having been edited.

## Outcome

**All 26 items are closed.** Every finding carries a `**Status:**` paragraph below saying how, and
the commit that did it. Two commits close the audit:

| Commit | What it closed |
|---|---|
| `ec44bd9` | The 25 defects and the control, in migrations 041–044 and four Python modules |
| `976034e` | Migration 045: the C4 residual, the H3 transition set, and the H5 predicate shape |

Three findings (H4, H5, H7) are fixed in the code but were closed by **retiring their verify test
rather than by making it pass**: each of those three tests asserts something the ADR does not
require, or reproduces via a route that cannot happen, and each is replaced by a test that fails
against the pre-fix shape. The reasoning is in the finding's own status paragraph. The three
`test_verify_*.py` files no longer exist: their 28 tests were relocated into the suites that own
their subject, or dropped with a reason, all recorded in `docs/adrs/0001-implementation-notes.md`.

A **re-verification pass** then re-audited the fixes independently. It found one critical
incompletely fixed and one regression the fixes had introduced, and pulling on the second exposed a
third defect on the shipped lifecycle path. All three are recorded, with their own statuses, in
the [re-verification](#re-verification) section at the end. Nothing else regressed: BM25 ranking
and quotas for the ordinary single-collection case were measured identical between a 001–040
database and a 001–044 one on the same fixture, with 0 of 50 rows differing in score by more than
1e-6.

## Assignment

| Owner | Scope | Migration |
|---|---|---|
| **F1** | Retrieval and statistics: `pgkg_bm25_candidates`, `pgkg_vector_candidates`, `pgkg_item_scope`, `pgkg_apply_quotas`, chunk statistics, `corpus_stats`/`lexeme_df` keys | 041 |
| **F2** | Context windowing and document lifecycle: `pgkg_chunk_window`, `pgkg_add_version_chunk`, `pgkg_open_document_version` | 042 |
| **F3** | RLS, isolation and the graph arm: policies, `pgkg_graph_candidates`, `proposition_cache`, and the two vacuous-test gaps | 043 |
| **F4** | Python: `pgkg/memory.py`, `pgkg/corpus.py`, `pgkg/ingest_jobs.py`, `pgkg/api.py` | — |

---


## Critical (4)

### C1 — `migrations/020_tenancy.sql`  ·  owner **F3**

Row-level security makes the GIN keyword index unreachable, so every BM25 arm degrades to a sequential scan under the role the policies were written for.

**Reproduction.** `tsvector @@ tsquery` is `ts_match_vq`, whose `proleakproof` is false. With RLS active Postgres may not evaluate a non-leakproof qual before the policy's own qual, so `@@` cannot become an index condition. Measured on 40,005 chunks: as the table owner, `SELECT count(*) FROM chunks WHERE tsv @@ to_tsquery('simple','unobtainium')` plans a Bitmap Index Scan on chunk_tsv_idx and runs in 0.064 ms; after `SET ROLE pgkg_app` the identical query plans a Seq Scan and runs in 20.694 ms — 320x, growing linearly with the table. Inside pgkg_bm25_candidates the whole chunk arm goes from 1,497 ms to 2,595 ms and the `tsv @@ q` test is demoted from an Index Cond to a Join Filter. No test can see this: conftest connects as a superuser, for whom every policy is inert, while tests/test_app_role.py asserts pgkg_app is exactly the non-BYPASSRLS role a deployment is told to connect as. Reproduced by tests/test_verify_perf.py::test_the_keyword_index_survives_the_application_role.

**Status: fixed** (`ec44bd9`). `ts_match_vq` is marked `LEAKPROOF` in 043, in a `DO` block that degrades to a `NOTICE` without superuser. Measured inside the chunk keyword arm as `pgkg_app` on 4,000 candidates: 481 shared buffers with a Bitmap Index Scan, against 1,345 with a Seq Scan and a Join Filter without it. The verify test named above was itself vacuous — `SET LOCAL ROLE` outside a transaction block is scoped to asyncpg's implicit single-statement transaction, so it measured the owner twice and passed at HEAD — and was replaced by `tests/test_rls_coverage.py::test_the_policy_does_not_cost_the_keyword_arm_its_index`, which switches role inside a transaction and asserts the policy's own qual is in the plan before asserting the index.

### C2 — `migrations/031_corpus_retrieval.sql`  ·  owner **F2**

pgkg_chunk_window() applies no visibility predicate at all, so pgkg_retrieve()'s context_text returns text from collections and ACL groups the caller was never granted. Chunks are content-addressed per (org_id, content_hash), so one boilerplate passage is the SAME ROW in a readable document and in a restricted one; the window then walks that row's neighbours by document_version_chunks.ord and string_aggs their text. Reachable over HTTP: /recall returns list[Result] and Result.context_text carries it.

**Reproduction.** One org, collection READABLE and collection SECRET. A shared footer paragraph is added to a READABLE document first (so the chunk row carries collection_id=READABLE and passes pgkg_visible), then the identical text is added at ord 1 of a SECRET document whose other ords are 'Acme Holdings will be acquired for four hundred million' and 'The board vote is scheduled for the third of March'. pgkg_retrieve(p_org_ids=[org], p_collection_ids=[READABLE], p_sources=['chunks']) returns the footer with context_text = both SECRET sentences concatenated around it. Identical result through the ACL axis: p_acl_groups=[allowed] correctly excludes the denied chunk from `text` but context_text launders the whole denied document — exactly the 'permission-laundering machine' D3 names.

**Status: fixed** (`ec44bd9`). 042 gives `pgkg_chunk_window()` its own predicate: a neighbour must agree with the anchor chunk on org, collection, ACL group, visibility and owner, and the version supplying the window must belong to a live current document of the anchor's own org and collection. The window still takes no caller scope, deliberately — a hit is widened after it is ranked, so the honest rule is not what the caller may read but whether a neighbour is indistinguishable from the anchor to every read predicate.

### C3 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

pgkg_item_scope() labels a candidate 'corpus' only when collections.kind = 'corpus', but the kind vocabulary is ('chat','corpus','mixed') and the reserved default collection is 'mixed'. Every passage in a mixed collection is bucketed 'memory', so D1's 60% corpus ceiling and its per-claim-scope split do not apply to it, and it competes for the very slots the 8-row personal-memory floor exists to protect. This is the default path: POST /documents falls back to DEFAULT_COLLECTION_ID when collection_id is omitted, and Memory and CorpusIngest both default there.

**Reproduction.** 200 chunks in a kind='mixed' collection (scores 100.0 down to 98.0) plus 20 of the caller's own user-scope propositions (score 0.001), passed to pgkg_apply_quotas(k_rerank=64, corpus_fraction=0.6, memory_floor=8): every admitted row is bucket='memory', all 64 slots go to passages, and 0 of the 20 personal facts survive. Separately, a chunk inserted into DEFAULT_COLLECTION_ID returns bucket='memory' from pgkg_item_scope, and pgkg_apply_quotas(..., corpus_fraction=0.0, ...) — a caller demanding no corpus at all — still admits it. Tests: test_mixed_collection_corpus_is_still_quota_capped, test_default_collection_corpus_is_labelled_corpus in tests/test_verify_sql_retrieval.py. The existing suite does not catch this because every quota test builds an explicit kind='corpus' collection.

**Status: fixed** (`ec44bd9`). 041 buckets on structure rather than on the `collections.kind` vocabulary: a chunk is always `corpus`, because a retrievable chunk is a document passage by construction, and a proposition is `memory` unless it is corpus-derived. Keying on something with no vocabulary to extend is what closes the class, rather than adding `mixed` to a list.

### C4 — `migrations/033_corpus_surface.sql`  ·  owner **F2**

document_version_chunks' RLS policy validates only the version side ('EXISTS (SELECT 1 FROM document_versions dv WHERE dv.id = document_version_id)'), never the chunk side. A tenant may graft another tenant's chunk id into its own document version. Combined with the unscoped pgkg_chunk_window() above, this turns a known chunk UUID into a cross-ORG data-injection channel, and it also lets a stranger keep a victim's chunk pgkg_chunk_live() forever.

**Reproduction.** Victim org owns chunk C via its own current document version. Attacker org, connected as pgkg_app with pgkg.org_id set to its own org, runs INSERT INTO document_version_chunks (document_version_id, chunk_id, ord) VALUES (attacker_version, C, 1) — the write SUCCEEDS. The victim then recalls C in its own scope and context_text comes back as 'Acme Holdings will be acquired for four hundred million ... <victim footer>' — the attacker's prose delivered inside the victim's result row.

**Status: fixed** (`ec44bd9`, completed by `976034e`). 042 added two statement-level triggers that refuse a link whose two sides disagree about the org, which closed the reproduction above and is strictly stronger than a policy term, because it holds for an owner or `BYPASSRLS` connection too. Re-verification found the stated root cause still standing: the policy was still 033's version-side-only one, and 043 had widened `document_versions` to the operator's system org for D4's sharing seam, so a tenant with no subscription could `UPDATE` the operator's links — both sides already in the system org, so the trigger passed. 045 splits the policy: `FOR SELECT` keeps the inherited visibility, `FOR ALL` is own-org on both sides. See the re-verification section at the end.


## High (10)

### H1 — `migrations/011_bm25_stats.sql`  ·  owner **F1**

corpus_stats and lexeme_df are keyed PRIMARY KEY (namespace, kind) with no org column, and pgkg_bm25_candidates' proposition arm reads them with only 'cs.namespace = p_namespace AND cs.kind = 'proposition''. Every tenant's Memory uses one namespace, so N, avgdl and every term's document frequency — the whole IDF — are computed globally across all orgs. This is D4's hard rule ('Ranking signals are never computed globally over shared content ... a real cross-tenant inference channel'). The chunk half of the SAME function keys its statistics per collection via pgkg_stats_domain(), so the omission is asymmetric rather than deliberate.

**Reproduction.** Org MINE writes 3 propositions containing rare term T; pgkg_bm25_candidates(q_text=T, p_org_ids=[MINE]) scores its top hit 0.1335. Org THEIRS then writes 200 propositions containing T. With no change whatsoever to org MINE's data, the same query now scores 0.00263 — a 50x shift. A tenant can therefore measure how much other tenants have written about any term (a competitor name, an acquisition codename) purely from its own score movements.

**Status: fixed** (`ec44bd9`). 041 types the statistics domain — `(kind, namespace, org_id, collection_id)` on `corpus_stats`, `(kind, namespace, lexeme, org_id, collection_id)` on `lexeme_df` — and drops `pgkg_stats_domain()` with its `collection:<uuid>` string convention. The read rule is that the statistics filter mirrors the candidate scan's scope term for term, org and collection both, `NULL` meaning unrestricted in both places: a domain wider than the scan is the leak, a domain narrower is H3.

### H2 — `migrations/030_corpus_lifecycle.sql`  ·  owner **F2**

pgkg_add_version_chunk() derives a new chunk's org, collection, ACL group and provenance from the version's document, but its `ON CONFLICT (org_id, content_hash) WHERE document_id IS NULL DO NOTHING` path leaves every one of those columns at whatever the first ingest wrote. The content address is org-wide while CorpusIngest is per (org, collection), so a passage shared by two collections in one org is permanently scoped, decayed, claim-scoped, quota-bucketed and ACL-gated as the first collection's. A query scoped to the second collection cannot retrieve its own document's passage; a query scoped to the first retrieves it under the wrong decay profile and claim scope. Since p_acl_group_id is on the same insert, the same path launders a document ACL once corpus.py starts passing it.

**Reproduction.** One org, collection A (claim_scope 'world') and collection B (claim_scope 'org'). CorpusIngest(coll_a).upsert_document(body) then CorpusIngest(coll_b).upsert_document(same body). Document B's current version links exactly one chunk and that chunk's collection_id is coll_a, not coll_b. Test: test_carried_over_chunk_keeps_first_collection.

**Status: fixed** (`ec44bd9`). 042 narrows the content address from `(org_id, content_hash)` to `(org_id, collection_id, coalesce(acl_group_id, nil), content_hash)`: dedup stops at the collection and ACL boundary for the same reason D4 stops it at the org boundary, that a row shared across a scoping boundary cannot carry the boundary. A passage genuinely present in two collections is now a row in each, and is embedded twice unless `embedding_cache` answers.

### H3 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

The chunk BM25 statistics are maintained over every row of `chunks`, but retrieval only returns rows passing pgkg_chunk_live(). pgkg_chunks_stats_delta() and pgkg_refresh_chunk_stats() have no liveness predicate at all, so chat-provenance chunks (document_id NOT NULL, written by memory.py ingest), chunks whose only links are to retired versions, and chunks of soft-deleted documents all contribute to n_total, total_len and lexeme_df. That is a permanently wrong population, not stale statistics: the proposition side of the same tables is correctly filtered on invalidated_at, matching what its arm retrieves.

**Reproduction.** One retrievable passage 'zorblatt calibration procedure' in a corpus collection scores raw_score 0.2877 for query 'zorblatt'. Insert 20 chat-provenance chunks (document_id set) into the same collection, each containing 'zorblatt'. pgkg_bm25_candidates(..., 'chunks') still returns exactly one row — the chat chunks are correctly excluded by pgkg_chunk_live — but its score has collapsed to 0.0273, a 10.5x drop, because lexeme_df('zorblatt') went 1 -> 21 and n_total went 1 -> 21. Test: test_chat_provenance_chunks_do_not_skew_chunk_bm25. Reachable on the default path: Memory and CorpusIngest both default to DEFAULT_COLLECTION_ID, so chat turns and corpus documents share one statistics domain.

**Status: fixed** (`ec44bd9`, refined by `976034e`). 041 stores retrievability on `chunks.retrievable`, reconciled by trigger at every transition that can change it, and makes the same column both the statistics population and the scan predicate so the two cannot disagree. It could not be a predicate evaluated at insert time, because a passage is not linked to its version when its INSERT statement ends. 045 completes the transition set with the `UPDATE` event and takes `refcount` out of the liveness predicate.

### H4 — `migrations/031_corpus_retrieval.sql`  ·  owner **F2**

pgkg_chunk_window() takes no org, collection, user or ACL argument and filters only on dv.status='current' AND d.deleted_at IS NULL. Because a content-addressed chunk can belong to the current version of several documents, pgkg_retrieve's `windows` CTE (DISTINCT ON (chunk_id) ... ORDER BY document_version_id) hands back a small-to-big context assembled from an arbitrary one of them. The returned context_text therefore carries neighbouring passages from documents in collections the query never named, and from documents behind a different acl_group_id, while every scoring stage was correctly scoped.

**Reproduction.** One org, collections A and B, ten independent document pairs each sharing one boilerplate passage between a document in A and a document in B, all promoted. Taking the window pgkg_retrieve would take (ORDER BY document_version_id LIMIT 1), 3 of the 10 shared passages returned context text containing a marker string that occurs only in collection B's document. The 30-70% rate is the coin flip on which of two random version UUIDs sorts lower. Test: test_chunk_window_stays_inside_the_queried_collections.

**Status: fixed** (`ec44bd9`); **verify test retired**. 042 makes a passage carried by more than one live current version its own context rather than picking the lowest version UUID: the neighbours of a footer in one handbook are not the context of the same footer in another, so any pick fabricates attribution for a citation. The verify test could not pass alongside the H2 verify test — once a shared passage is a row per collection, its unordered `SELECT id FROM chunks WHERE text = shared` picks an arbitrary one of them and calls that row's own document a leak — so it was replaced in `tests/test_context_window_scope.py` by a form that anchors the chunk by collection and checks both directions in each of ten pairs. The replacement is strictly stronger and fails against the pre-fix shape.

### H5 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

pgkg_chunk_live() does not inline, so chunk liveness costs a nested-loop subquery per candidate row on the hot retrieval path.

**Reproduction.** The predicate appears in the plan as an unexpanded `Filter: … pgkg_chunk_live(id, refcount)` — unlike pgkg_visible and pgkg_temporal_visible, which do inline into plain column comparisons — so its two sublinks (a self-lookup on chunks plus a three-table join) run once per candidate chunk. On 4,000 candidates the function form touches 28,937 shared buffers against 927 for the identical test written inline as a predicate (31x); on 40,000 candidates it adds ~339,000 buffer hits and 950 ms to a single keyword arm, and rewriting it as a predicate lets the planner build one hashed SubPlan instead (~6x faster). Under OR keyword semantics the candidate set is a large fraction of the corpus, so this scales with corpus size. Reproduced by tests/test_verify_perf.py::test_chunk_liveness_does_not_cost_a_subquery_per_candidate.

**Status: fixed** (`ec44bd9`, completed by `976034e`); **verify test retired**. The two candidate arms read `chunks.retrievable` instead of calling the function: 1,334 shared buffers on the fixture's 4,000 candidates against 25,338 for the function form, which also beats the test's own inline baseline of 1,372. The verify test asserted that `pgkg_chunk_live(c.id, c.refcount)` itself cost no more than 3x an inline predicate, which requires liveness to be decidable from those two arguments with no table access; `inline_function` rejects any body whose parse tree has a non-empty range table, and liveness was never a function of a count of links. It was replaced in `tests/test_retrieval_plan_shape.py` by three assertions about the arms rather than about the function. 045 finished the argument the finding started: `refcount` is out of the predicate entirely, and liveness reads `chunks.version_scoped`.

### H6 — `migrations/040_entity_mentions.sql`  ·  owner **F3**

pgkg_search() silently loses graph-expanded facts once gazetteer mentions exist, because the rewritten graph arm spends the k_total budget on chunk ids that pgkg_search then discards.

**Reproduction.** 040 rewrote pgkg_graph_candidates() to emit chunk ids alongside proposition ids, but pgkg_search() (migration 022's copy) still calls it and consumes the result with `JOIN propositions p ON p.id = fused.item_id`. Chunk candidates are therefore dropped only after they have occupied places in `capped`/`LIMIT k_total`. With 20 seed entities, 12 facts and 12 passages each, pgkg_search('zorbulon', …) returns 120 propositions before any entity_mentions rows exist and 70 after — a 42% recall loss with nothing returned in place of the evicted facts. pgkg_search()'s proposition-shaped contract is what the 59 baseline tests pin. Reproduced by tests/test_verify_perf.py::test_mentions_do_not_displace_facts_in_pgkg_search.

**Status: fixed** (`ec44bd9`). 043 gives `pgkg_graph_candidates()` a `p_sources` argument applied before the per-seed cap, because a cap is the budget. `pgkg_retrieve()` is D1's two-store surface and keeps both stores, which is the default; `pgkg_search()` is proposition-shaped by contract and asks for one, so its `k_total` buys only rows it can return.

### H7 — `pgkg/corpus.py`  ·  owner **F4**

The 'perishable' decay profile is keyed on ingest time, not publication date. D6 specifies perishable as keyed on `asserted_at = published_at` and D5 says provenance.published_at 'feeds the perishable decay profile', but CorpusIngest.upsert_document takes `asserted_at` and `provenance.published_at` as independent parameters and never derives one from the other (the same two fields are independent on the HTTP DocumentRequest). chunks.asserted_at is left NULL, so pgkg_apply_profile falls back to COALESCE(c.asserted_at, c.created_at) = the ingest timestamp, and the perishable factor is 1.0 — behaviourally identical to 'timeless', which is the profile the ADR is at pains to distinguish it from.

**Reproduction.** A perishable collection; upsert_document(text=..., provenance=Provenance(kind='document_version', producer='chunker', published_at=2015-01-01)). provenance.published_at stores correctly as 2015-01-01, chunks.asserted_at is None, and pgkg_apply_profile on the resulting chunk returns adjusted_score exactly 1.0 for a raw score of 1.0. At the 730-day half-life the eleven-year-old article should score about 0.0041 — a 245x over-ranking of content the ADR calls 'not stale, wrong'. Test: test_perishable_decays_on_publication_date.

**Status: fixed** (`ec44bd9`); **verify test dropped**. `chunks.asserted_at` is now derived from the resolved `provenance.published_at` when the caller gives no `asserted_at`, on the corpus path and the chat path alike, and the HTTP contract carries the derivation as `effective_asserted_at`. 044 carried the same fields through the ingest queue, so a queued perishable document decays from its publication date and not from the worker's clock. The verify test compared the measured factor against `exp(-11*365/730)` with `rel=0.2`; the article's real age on 2026-09-01 is 4,261 days, and the tolerance band stopped admitting it on 2026-06-09. Its replacements in `tests/test_corpus_ingest.py` read the age off the database instead of assuming it, so the specification survives and cannot expire.

### H8 — `pgkg/corpus.py`  ·  owner **F4**

Corpus ingest calls the embedder and a per-chunk LLM extractor from inside its open transaction, holding a pooled connection across every model call.

**Reproduction.** `upsert_document` opens `conn.transaction()` and then calls `_vectorise`, which runs `self._embed_texts(missing)` (corpus.py:523, synchronous — it also blocks the event loop), and `_extract`, which awaits `ml.extract_propositions_async(text)` once per chunk in a serial `for` loop. memory.py's own `_IngestPlan` docstring states the rule this breaks: 'holding the ingest connection across them starves the pool under concurrent ingest'. A 50-chunk document with a 2 s extractor keeps one pooled connection and one open transaction for ~100 s, and `_ConnExtractCache` bumps proposition_cache.hit_count on that same connection, holding those row locks for the whole time so a second ingest of overlapping content blocks. Reproduced by tests/test_verify_perf.py::test_corpus_ingest_does_not_embed_inside_its_transaction, which observes `conn.is_in_transaction()` is True at the moment the embedder is invoked.

**Status: fixed** (`ec44bd9`), **with a stated limit**. `CorpusIngest.upsert_document` is three phases: ask whether the document hash moved (one connection, no transaction, returns before chunking on the no-op path), spend the model budget, then write everything in one transaction containing no model call. The verify test's probe evaluates `is_in_transaction()` on the last-acquired pool proxy, which raises `InterfaceError` once that proxy is released, so a fix that held nothing at all would error the test rather than pass it. The shape adopted keeps a connection alive with no transaction open across the embedder batch and gives the per-chunk extractor no connection.

### H9 — `pgkg/ingest_jobs.py`  ·  owner **F4**

IngestWorker deadlocks permanently when `slots` reaches the pool size, because a slot holds two connections rather than the one its docstring rations.

**Reproduction.** `run()` documents slots as rationing 'connections held', but each in-flight document holds the ingest transaction's connection AND takes a second one in `_progress_reporter.report()` — deliberately, so progress is visible while that transaction is open. With slots == pool max_size every slot waits for a connection every other slot is holding, and nothing ever releases. `pgkg worker --slots N` passes `slots=N, concurrency=N` against the default `make_pool` max_size of 10, so `--slots 10` hangs forever with no error. Reproduced on a max_size=2 pool with slots=2: `asyncio.wait_for(worker.run(concurrency=2), timeout=15)` raises TimeoutError, while the control with slots=1 on the same pool drains both jobs in 2 s (tests/test_verify_perf.py::test_the_worker_does_not_deadlock_at_its_slot_budget and ::test_control_the_worker_drains_when_a_slot_is_spare).

**Status: fixed** (`ec44bd9`). Progress is reported between phases with nothing held, so a slot no longer needs a second connection while holding one. Probed by restoring the old shape: reporting progress from inside the phase-2 connection reproduces the deadlock at `slots=2` on a `max_size=2` pool while the control still drains, which is what proves the restructure is the fix.

### H10 — `pgkg/memory.py`  ·  owner **F4**

Memory.forget() scopes its UPDATE with self._scope.read_org_ids (line ~409, _FORGET_SQL 'WHERE id = $3 AND org_id = ANY($4)'), not write_org_id. Scope's own docstring says 'Reads widen ... writes never do', and include_system_org is a plain field on the unauthenticated HTTP ScopedRequest. So any tenant that widens its reads can invalidate propositions in the reserved SYSTEM_ORG — operator-published material every other subscriber reads. Mitigated only if the deployment actually connects as pgkg_app (propositions' policy is org_id = pgkg_current_org()); the RLS policy is the sole thing standing between this and cross-tenant data destruction, which is precisely the 'one missing WHERE clause' D3 warns about.

**Reproduction.** Memory(pool, scope=Scope(org_id=tenant, include_system_org=True)).forget(system_org_proposition_id, reason='user_deleted') sets invalidated_at and invalidation_reason='user_deleted' on the SYSTEM_ORG row. Reproduced on the test pool: the shared proposition came back invalidated. POST /forget with include_system_org=true is the same call over HTTP.

**Status: fixed** (`ec44bd9`). `_FORGET_SQL` names `scope.write_org_id`. An audit of the class found a second instance in the same file: `_FLUSH_ACCESS_SQL` named no org at all, so a widened read that returned a system-org proposition credited `access_count` on it under the tenant's flush, with RLS as the only guard. `tests/test_write_scope.py` covers the class under a maximally widened scope, including a non-vacuity control and a structural test that no write method's source names `read_org_ids` or `read_collection_ids`.


## Medium (11)

### M1 — `migrations/005_proposition_cache.sql`  ·  owner **F3**

proposition_cache is keyed on cache_key = hash(chunk_text, extractor_model) alone: no org column, no RLS, no public_source gate. Phase 2 newly wired it into the corpus path (pgkg/corpus.py:595, _ConnExtractCache gated only on self._use_extract_cache — NOT on policy.public_source, unlike the embedding cache at corpus.py:546/561). D4 restricts embedding_cache to operator-licensed material because 'for a confidential document a cache hit would confirm another tenant holds it'; the extraction cache carries the extracted facts themselves and has no such restriction.

**Reproduction.** Org A ingests a confidential document into a private (public_source=false) collection with extract_propositions=true; the extracted propositions land in proposition_cache under a key derived from the chunk text. Org B, holding the same text (e.g. a leaked draft), gets a cache hit and reads Org A's extraction verbatim. Reproduced: proposition_cache has no org_id column and relrowsecurity=false, and a SELECT as pgkg_app with an unrelated pgkg.org_id returns the cached payload in full.

**Status: fixed** (`ec44bd9`). 043 gives `proposition_cache` an `org_id`, RLS, and a policy whose read side is own-org or system-org and whose write side is own-org; 044 makes the primary key `(cache_key, org_id)` and both cache paths in `corpus.py` and `memory.py` name the org in the get and the put. Pre-existing rows were deleted rather than attributed, because a key computed without an org cannot say whose extraction it holds. The second org to hold a private passage now misses the cache and re-extracts, which is the direction D4 chooses.

### M2 — `migrations/030_corpus_lifecycle.sql`  ·  owner **F2**

pgkg_open_document_version() computes MAX(version_no) + 1 with no lock, so two concurrent crawls of one document collide instead of serialising.

**Reproduction.** The function reads `COALESCE((SELECT MAX(dv.version_no) …), 0) + 1` and inserts, with `document_versions_document_id_version_no_key` unique on (document_id, version_no) and no advisory or row lock taken on the document first. Two connectors (or two IngestWorker loops, or an upsert racing a re-crawl) that open a version for the same document in overlapping transactions both compute the same number; the second fails with `duplicate key value violates unique constraint "document_versions_document_id_version_no_key"`, which propagates out of upsert_document as an unhandled UniqueViolationError. Reproduced by tests/test_verify_perf.py::test_two_concurrent_crawls_of_one_document_do_not_collide (two overlapping transactions, 0.1 s apart).

**Status: fixed** (`ec44bd9`). `pgkg_open_document_version()` takes `SELECT ... FOR UPDATE` on the document row before it reads `MAX(version_no)`. Nightly full crawls are what connectors do, so two of them meeting on one document is the common case and has to queue rather than collide.

### M3 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

pgkg_vector_candidates() never filters on embedder_generation_id, so the primary vector arm computes cosines between the primary generation's query vector and rows whose inline vector belongs to another generation, and ranks the mixture in one ORDER BY. D8 states plainly that cosines from different model spaces are incomparable, and memory.py's MMR query already guards against exactly this mixture (it nulls the embedding unless embedder_generation_id equals the org's primary), so the codebase knows the mixture occurs. The affected row also gets a second, correctly-scoped vote through pgkg_generation_candidates, double-counting it in RRF.

**Reproduction.** A cutover at D8 step 5 with two same-width generations: gen2 promoted to primary, gen1 demoted to 'retiring', its rows still inline and tagged embedder_generation_id=gen1. pgkg_vector_candidates(q_gen2_space, ...) returns both the gen2-tagged proposition and the gen1-tagged one, ranking the gen1 row first because its stale vector happens to be nearer in raw coordinates. Test: test_primary_vector_arm_stays_in_the_primary_model_space. Not reachable today (no cutover has been run and no second generation is registered by any production path), which is why the severity is medium rather than high.

**Status: fixed** (`ec44bd9`). The vector arm restricts both stores to the primary generation of the orgs in scope, resolved from `org_embedders` and aggregated into one array so the restriction reaches the ordered index scan's filter rather than becoming a join its `LIMIT` cannot pass through. It restricts only when a primary generation is actually visible, so a role or scope that cannot read `org_embedders` degrades to the previous behaviour rather than losing the arm.

### M4 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

The BM25 score sub-select is planned and executed twice for every candidate row, once for the `> 0` filter and once for the sort.

**Reproduction.** pgkg_bm25_candidates ends `… ) sub WHERE sub.bm25_score > 0.0 ORDER BY sub.bm25_score DESC LIMIT k_initial`. Because `bm25_score` is a correlated sub-select, Postgres emits two independent SubPlans — the plan shows `SubPlan 6` and `SubPlan 7` (propositions arm) / `SubPlan 9` and `SubPlan 10` (chunks arm), each with `loops=40000` for 40,000 candidates, i.e. 80,000 evaluations of the per-row aggregate. Isolated head-to-head on the same data: the shipped shape runs in 147 ms with 2 SubPlans; the identical scoring wrapped in a MATERIALIZED CTE runs in 81 ms with 1. Applies to both arms and to every copy of the function (010, 020, 021, 031). Reproduced by tests/test_verify_perf.py::test_bm25_scores_each_candidate_once.

**Status: fixed** (`ec44bd9`). The scoring is a join to the IDF terms and a `GROUP BY` — one hash aggregate for the whole candidate set, zero SubPlans. The inner join subsumes the `> 0` filter, because RSJ IDF with the `+1` inside `LN` is strictly positive, so a candidate scores zero exactly when no query lexeme reached it and now produces no group.

### M5 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

The IDF lookup is still a full-table aggregate over lexeme_df: the query's own lexemes never reach the index.

**Reproduction.** The `term_df` CTE filters only on namespace/kind and expresses the lexeme restriction as a downstream `Hash Right Join` against the query_lexemes CTE, so the `lexeme_df_pkey` index on (namespace, kind, lexeme) is unusable. For a three-term query the plan shows `Seq Scan on lexeme_df … rows=4009` followed by a HashAggregate of all 4,009 rows — the entire vocabulary of that corpus. At 40k propositions this costs 17.3 ms per query; the same aggregate written as `AND ld.lexeme = ANY(ARRAY[...])` uses the index and costs 0.049 ms — 350x, and the gap widens with vocabulary. The per-row correlated COUNT(*) is gone, but a full aggregate whose cost grows with the corpus remains on the query path. Reproduced by tests/test_verify_perf.py::test_idf_reads_only_the_query_terms.

**Status: fixed** (`ec44bd9`). The query's lexemes are aggregated into one array and named in the `lexeme_df` scan's own qualifier, so the primary key serves them. No new index was needed: the lexeme is third in 041's key, ahead of the two partition columns the read path may leave unrestricted.

### M6 — `pgkg/corpus.py`  ·  owner **F4**

Corpus ingest makes one round trip per chunk inside its open transaction, so the pipeline built for a 300-page handbook is not set-based.

**Reproduction.** `_link_chunks` loops over `chunk_document(...)` awaiting `conn.fetchrow(_ADD_CHUNK_SQL, ...)` once per chunk. Measured through a recording pool: a 2-chunk document costs 12 round trips and a 30-chunk document costs 40 — one extra sequential round trip per chunk, each one a network RTT with the ingest transaction held open. A 500-chunk document is 500 serialised round trips. Chat ingest was explicitly made set-based in phase 0 (5 statements for 12 chunks and 60 propositions); the bulk path was not. Reproduced by tests/test_verify_perf.py::test_corpus_ingest_round_trips_do_not_grow_with_chunk_count.

**Status: fixed** (`ec44bd9`). Chunk linking is one `CROSS JOIN LATERAL` over `unnest ... WITH ORDINALITY`, and the extract cache reads all of a document's keys in one statement and buffers its writes into the ingest transaction. Measured through the verify test's own recording pool: 16 round trips for 2, 30 and 120 chunks alike, and an unchanged re-ingest is 4 round trips that open no transaction at all.

### M7 — `pgkg/corpus.py`  ·  owner **F4**

Re-ingesting a soft-deleted external_id raises UniqueViolationError, contradicting the pipeline's documented intent.

**Reproduction.** `_FIND_DOCUMENT_SQL` filters `AND deleted_at IS NULL` with the comment 'A soft-deleted document is not a collision: re-ingesting a withdrawn external id is a new document', but `documents_external_id_key` is `UNIQUE (org_id, collection_id, external_id) WHERE external_id IS NOT NULL` and does not exclude soft-deleted rows. So the lookup misses the withdrawn row and the subsequent INSERT hits the index. Reachable end to end through the shipped HTTP surface: POST /documents/delete (which only sets deleted_at) followed by an upsert of the same external_id raises `duplicate key value violates unique constraint "documents_external_id_key"` and returns 500. Reproduced by tests/test_verify_perf.py::test_reingesting_a_deleted_document_starts_a_new_one.

**Status: fixed** (`ec44bd9`). 044 makes `documents_external_id_key` partial on `deleted_at IS NULL`, which is where the rule belongs, and deletes the release-the-external-id statement the Python half needed in the interim. A withdrawn document keeps the id it was withdrawn under, which is the only thing a deletion audit asks of that row.

### M8 — `pgkg/ingest_jobs.py`  ·  owner **F4**

job_state() (line 114, _STATE_SQL = 'SELECT ... FROM ingest_jobs WHERE id = $1') takes no org argument and never sets the pgkg.org_id GUC on the connection it acquires, so nothing in the read names the caller. GET /jobs/{job_id} passes the raw pool straight through. Anyone holding a job UUID reads another tenant's status, attempt count, document_id, version_id and error text.

**Reproduction.** Org STRANGER enqueues a document via pgkg_enqueue_ingest_job. A caller with no relationship to that org calls job_state(pool, job_id) and receives JobState(status='pending', attempts=0, ...) rather than a KeyError. Because the connection sets no GUC, pgkg_current_org() falls back to pgkg_default_org(), so the ingest_jobs policy 033 added cannot help even when RLS is live — and on an owner connection it is inert entirely.

**Status: fixed** (`ec44bd9`). `job_state()` takes an `org_id`, states `org_id = $2` in the read, and sets the org GUC on the connection it acquires; `GET /jobs/{job_id}` takes `?org_id=`. The worker also sets the job's org on the connections it uses for progress, finish and fail, which is what 033's policy comment asks of a worker draining several orgs.

### M9 — `pgkg/memory.py`  ·  owner **F4**

A partly-applied access flush is restored in full, so any org whose UPDATE already committed is counted twice.

**Reproduction.** `flush_access()` loops over orgs issuing one autocommitted UPDATE each (no enclosing transaction), and on any exception calls `self._access.restore(pending)` with the whole drained dict — including the orgs whose UPDATE has already committed. Those counts are then applied a second time on the next flush. With three accesses recorded for org A and three for org B, and the connection failing on B's statement, org A's proposition ends at access_count 6 instead of 3. The frequency term in the decay profile reads this column, so a transient link failure permanently inflates ranking for whichever tenant happened to flush first. Reproduced by tests/test_verify_perf.py::test_a_failed_access_flush_does_not_double_count.

**Status: fixed** (`ec44bd9`). Only the orgs whose UPDATE did not commit are restored, so a transient link failure no longer permanently inflates the frequency term for whichever tenant happened to flush first.

### M10 — `tests/test_api_scoping.py`  ·  owner **F3**

The D4 sharing seam is tested only as Scope dataclass property assertions (test_system_org_is_readable_but_never_written, test_subscribed_collections_widen_reads_only, lines 157-169). No test asserts end-to-end that a tenant with include_system_org=True actually retrieves a SYSTEM_ORG row from the database — and under RLS it could not, because every policy is 'org_id = pgkg_current_org()' (one org) while the retrieval predicate allows [tenant, SYSTEM_ORG]. The direction is fail-closed, so this is a correctness/test-vacuity finding rather than a leak, but it means the sharing seam is unverified against Postgres and that include_system_org's only demonstrated effect at HEAD is the write-side widening in finding 4.

**Reproduction.** Scope(org_id=X, include_system_org=True).read_org_ids == [X, SYSTEM_ORG_ID] is asserted; nothing asserts that pgkg_retrieve/pgkg_search with p_org_ids=[X, SYSTEM_ORG] returns a system-org row. Separately, tests/conftest.py's pool connects as the container's owning superuser, so EVERY RLS policy is inert in every test except the handful that explicitly SET LOCAL ROLE pgkg_app (test_rls_coverage.py, test_app_role.py). Any isolation test not using that role is testing the SQL predicate only, never the policy.

**Status: fixed** (`ec44bd9`). The seam did not work and now does: every policy was a single-org equality while D3's predicate reads `org_id = ANY` of tenant and system org, so under `pgkg_app` the second element could never match and a tenant with `include_system_org` got its own rows and silently nothing else. 043 widens the read side on the tables the read path resolves other-org rows in, and leaves every `WITH CHECK` a single-org equality, which is the schema-level statement of D4's first hard rule. `tests/test_api_scoping.py` proves the seam end to end as `pgkg_app`, in both directions. The widening deliberately does not gate on a subscription row: D3 makes the resolved collection list the app's to supply, so the subscription is a term in the retrieval predicate and RLS stays the org boundary.

### M11 — `tests/test_rls_coverage.py`  ·  owner **F3**

The coverage test asserts `protected == sorted(SEEDS)` — it enumerates tables that HAVE relrowsecurity and pins that set. It therefore cannot notice a new table that ships with an org column and NO policy. Migration 040 added entity_mentions and entity_links (both with org_id, both granted to pgkg_app) with row security deliberately deferred, and the suite stayed green. The module's stated purpose ('a table that gains a policy without gaining a case here would ship an untested isolation boundary') does not cover the more likely failure of a table that never gains a policy.

**Reproduction.** tests/test_rls_coverage.py::test_every_rls_enabled_table_is_covered_here passes at HEAD while entity_mentions and entity_links are readable and writable by pgkg_app under any pgkg.org_id. The only boundary on those tables is pgkg_visible() re-applied on the read path; a direct SELECT or an INSERT of a forged mention edge is unconstrained by Postgres.

**Status: fixed** (`ec44bd9`). `tests/test_rls_coverage.py` now enumerates the tables that carry an `org_id` column and requires each to be RLS-enabled or named in an allowlist with a reason; the allowlist is empty, which is the correct state. The inverted guard found four unpolicied tables — `entity_mentions`, `entity_links`, `corpus_stats`, `lexeme_df` — all of which now have policies and per-table isolation cases. The residual blind spot is written into the module docstring rather than left to be found: a table whose rows belong to an org without carrying the column (`edges`, `proposition_provenance`, `corroborations`) is invisible to this guard.


## Low (1)

### L1 — `migrations/040_entity_mentions.sql`  ·  owner **F3**

CONTROL — NOT A DEFECT. D3's hard requirement (every row reached by graph expansion is re-filtered through the seed's visibility predicate) holds in BOTH directions after phase 3, and holds non-vacuously. Recording it so the parent does not re-investigate.

**Reproduction.** One org, users A and B, shared entity 'Zzqhelios Programme'. User A owns a private proposition and a private passage (chunks.visibility='private', owner_user_id=A, mention edge built by pgkg_match_entity_mentions). User B's seed is a shared fact naming the entity. pgkg_retrieve(expand_graph=TRUE, p_user_id=B) returns neither PRIVATE_FACT nor PRIVATE_PASSAGE. Non-vacuity check: the identical query with p_user_id=A returns BOTH, proving the walk reaches them and pgkg_visible() in the fact arm and the chunk arm is what stops it.

**Status: holds** (`ec44bd9`). Kept as a control and strengthened: it now asserts both directions in one test, that user B gets neither the private fact nor the private passage and that user A gets both, so the walk is proved to reach them and the visibility predicate in each arm is proved to be what stops it.


---

## Re-verification

A fourth adversarial pass re-audited the closed findings against the fixed tree, under the same
rule: reproduce or do not report. Its verdict was `partially_fixed`. It found three things, all
now closed by `976034e` (migration `045_link_table_writes.sql`), each with a test that fails when
its own fix is reverted.

### R1 — C4's stated root cause was still standing  ·  from **C4**

`document_version_chunks`' RLS policy was still 033's version-side-only `EXISTS`, and 043 had
widened `document_versions`' `USING` to admit the operator's system org unconditionally, which is
what D4's sharing seam needs. So a tenant connected as `pgkg_app` with only its own org id set,
holding no subscription, could write into the operator's shared corpus through the link table: with
both sides of the link already in the system org, 042's same-org trigger is satisfied, and 041's
refcount and retrievability triggers do not fire on `UPDATE` at all.

**Reproduction.** `UPDATE document_version_chunks SET chunk_id = <another operator chunk> WHERE
document_version_id = <an operator version> AND ord = 1` returned `UPDATE 1`; so did an `UPDATE` of
`ord` alone. Reading back as the operator, `pgkg_chunk_window` for document B ordinal 0 returned
B ordinal 0's text followed by document A ordinal 0's text — prose from a different document
spliced into B's context for every subscribing tenant, which is C2's laundering shape reached by a
write instead of a read. The displaced chunk kept `refcount` 1 with zero carriers; the substituted
chunk gained a second carrier, so 042's sole-carrier rule silently destroyed its window
everywhere. Confirmed as `pgkg_app` with no subscription rows that a tenant can read all of the
operator's versions, links and chunks, so the ids the attack needs are readable by anyone.

**Status: fixed** (`976034e`). 045 splits the policy in two, because the two rules are genuinely
different: `FOR SELECT` keeps 033's inherited visibility, so a subscriber still reads a shared
document's ordering, and `FOR ALL` is own-org on both sides of the link. Reads widen to the
operator's shelf and writes never do, which is D4's first hard rule and what 043 states everywhere
else. Of the two halves `USING` is the load-bearing one and `WITH CHECK` states the rule: probing
showed that on the `INSERT` path a tenant appending to the operator's document is already refused
before `WITH CHECK` is consulted, because the link fires 030's refcount trigger whose `UPDATE` of
the operator's chunk violates that table's own policy. Pinned by
`tests/test_tenancy.py::test_a_tenant_cannot_rewrite_the_operators_link_table`, which asserts the
refusal's shape as well as its effect — `UPDATE 0` and `DELETE 0` from `USING`, an
`InsufficientPrivilegeError` from `WITH CHECK` — so it cannot pass because some maintenance trigger
downstream happened to raise, which is not a boundary: a trigger is inert on the owner connection a
worker runs as.

### R2 — the link table had no `UPDATE` event  ·  regression **introduced by 041**

041 replaced the computed `pgkg_chunk_live()` with the stored `chunks.retrievable` column, but the
retrievability triggers on `document_version_chunks` covered `INSERT` and `DELETE` only, and 030's
refcount triggers covered the same two. A link is `(document_version_id, ord) -> chunk_id`, so an
`UPDATE` is a repoint: it changes which passage a position of a document carries without inserting
or deleting anything, which is exactly the event neither guard watches.

**Reproduction.** One identical fixture on two schemas: on a 001–040 database the keyword arm
returns 0 rows for the orphaned chunk's marker, because liveness was recomputed per scan; on the
001–044 database it returns 1. Withdrawn content stays searchable. Refcount drifts with it — the
orphan keeps `refcount` 1 with zero links, so it can also never be garbage collected.
`pgkg_refresh_chunk_stats()` repairs the flag but has to be run.

**Status: fixed** (`976034e`). 045 adds the missing statement-level trigger, which reconciles both
columns from the link table rather than applying a signed delta — a repoint is a decrement and an
increment in one statement and an ord-only `UPDATE` is neither, and reconciliation is also what
keeps the trigger idempotent. The repoint is reconciled rather than refused because it is a
legitimate manual repair for an operator, and the honest response to a repair is to make the
derived state agree with it. Pinned by
`tests/test_corpus_lifecycle.py::test_a_link_repointed_by_update_does_not_leave_a_ghost_passage`.

### R3 — garbage collection resurrected withdrawn passages  ·  found while fixing R2

Reconciling the refcount alone was not enough, and chasing why exposed a third defect on the
shipped lifecycle path. Liveness read `(refcount = 0 AND document_id IS NULL) OR EXISTS (current
live version link)` — 031's predicate, which 041 stored on the row without changing. The first arm
is meant for a standalone passage, a chunk belonging to no document and no version, which is what
033 says. An orphan matches it too.

**Reproduction.** A passage dropped by a new version is correctly not retrievable while a retired
version still carries it; `pgkg_purge_retired_versions()` then reclaims that version, the last link
goes, the refcount falls to zero, and the first arm readmits the passage as a standalone one.
Measured on the shipped path with no manual write anywhere: `retrievable` goes `FALSE` -> `TRUE` at
the purge and the keyword arm returns the passage again. It is permanent for any passage a
proposition was extracted from, since `pgkg_gc_chunks()` deliberately refuses to collect those.

This is **not** a regression of the fix pass — 031's and 033's `pgkg_chunk_live()` computed the same
boolean, so it shipped with phase 2 — but it is the defect R2 is about, reached through the front
door, and reconciling the refcount without it would have moved the repoint route from accidentally
right to definitionally wrong.

**Status: fixed** (`976034e`). The two cases are indistinguishable from `(document_id, refcount)`
because the purge destroys the only record that the chunk was ever carried, so that record becomes
a column: `chunks.version_scoped`, set when a link to the chunk first exists and never cleared.
Liveness now reads "carried by a live current version, or belonging to no document and never
carried by any version", and `refcount` leaves the predicate entirely — the shape H5's analysis
argued for, since liveness never was a function of a count of links. What cannot be recovered on an
existing installation is a passage already orphaned by a purge: the record is exactly what was
destroyed, so it stays a standalone passage and the collector remains its only exit. Pinned by
`tests/test_corpus_lifecycle.py::test_reclaiming_a_retired_version_does_not_resurrect_its_passages`.

### Accepted consequences, not defects

The re-verification pass also listed four behaviour changes the fixes introduced deliberately.
They are recorded here so they are decisions rather than surprises, and in the Deviations section
of `docs/adrs/0001-implementation-notes.md`.

- **Deduplicated boilerplate loses its context entirely.** 042's sole-carrier rule means five
  documents in one collection sharing a footer give one chunk row with five carriers and no window,
  so `pgkg_retrieve()`'s `COALESCE` falls back to the passage's own text. Pre-fix an arbitrary but
  present window was returned. The narrowed content address deliberately keeps within-collection
  boilerplate on one row, so this applies to every repeated passage in a corpus.
- **The window's neighbour agreement is chunk-to-chunk, not caller-to-chunk.** A caller who *is*
  granted ACL group A, reading an `acl_group_id IS NULL` header in a document whose body carries
  group A, gets no body in context, and `ord_from`/`ord_to` shrink to match, so a withheld
  neighbour is indistinguishable from an absent one. This is the price of a window that no caller
  has to remember to scope.
- **Moving a document to another collection after ingest removes its chunks' windows.** 031 named
  cross-collection moves unsupported and returned a wrong window; the carrier CTE now requires the
  document's collection to equal the chunk's, so a wrong answer became a silent one.
  `pgkg_refresh_chunk_stats()` repairs the statistics half.
- **A passage in two collections is embedded twice** unless `embedding_cache` answers, which per D4
  it only does for `public_source` collections, and `entity_mentions` matching runs per row.

