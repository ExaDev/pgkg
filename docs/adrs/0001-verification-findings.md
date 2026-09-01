# ADR-0001 verification findings

Three adversarial verifiers audited the phases 0-3 implementation independently: SQL and
retrieval correctness, tenant and user isolation, and performance and operational soundness.
Each was instructed to mark a finding confirmed only if it was actually reproduced, and every
finding below carries the reproduction that confirmed it.

**26 confirmed: 4 critical, 10 high, 11 medium, 1 control.** Verdicts were `concerns`, `broken`,
`broken`.

The evidence lives in `tests/test_verify_sql_retrieval.py`, `tests/test_verify_tenant_isolation.py`
and `tests/test_verify_perf.py` — 25 failing tests, one or more per finding. Those files are the
specification for the fixes and are **read-only** to the fix agents: a fix is done when the
relevant test passes without the test having been edited.

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

### C2 — `migrations/031_corpus_retrieval.sql`  ·  owner **F2**

pgkg_chunk_window() applies no visibility predicate at all, so pgkg_retrieve()'s context_text returns text from collections and ACL groups the caller was never granted. Chunks are content-addressed per (org_id, content_hash), so one boilerplate passage is the SAME ROW in a readable document and in a restricted one; the window then walks that row's neighbours by document_version_chunks.ord and string_aggs their text. Reachable over HTTP: /recall returns list[Result] and Result.context_text carries it.

**Reproduction.** One org, collection READABLE and collection SECRET. A shared footer paragraph is added to a READABLE document first (so the chunk row carries collection_id=READABLE and passes pgkg_visible), then the identical text is added at ord 1 of a SECRET document whose other ords are 'Acme Holdings will be acquired for four hundred million' and 'The board vote is scheduled for the third of March'. pgkg_retrieve(p_org_ids=[org], p_collection_ids=[READABLE], p_sources=['chunks']) returns the footer with context_text = both SECRET sentences concatenated around it. Identical result through the ACL axis: p_acl_groups=[allowed] correctly excludes the denied chunk from `text` but context_text launders the whole denied document — exactly the 'permission-laundering machine' D3 names.

### C3 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

pgkg_item_scope() labels a candidate 'corpus' only when collections.kind = 'corpus', but the kind vocabulary is ('chat','corpus','mixed') and the reserved default collection is 'mixed'. Every passage in a mixed collection is bucketed 'memory', so D1's 60% corpus ceiling and its per-claim-scope split do not apply to it, and it competes for the very slots the 8-row personal-memory floor exists to protect. This is the default path: POST /documents falls back to DEFAULT_COLLECTION_ID when collection_id is omitted, and Memory and CorpusIngest both default there.

**Reproduction.** 200 chunks in a kind='mixed' collection (scores 100.0 down to 98.0) plus 20 of the caller's own user-scope propositions (score 0.001), passed to pgkg_apply_quotas(k_rerank=64, corpus_fraction=0.6, memory_floor=8): every admitted row is bucket='memory', all 64 slots go to passages, and 0 of the 20 personal facts survive. Separately, a chunk inserted into DEFAULT_COLLECTION_ID returns bucket='memory' from pgkg_item_scope, and pgkg_apply_quotas(..., corpus_fraction=0.0, ...) — a caller demanding no corpus at all — still admits it. Tests: test_mixed_collection_corpus_is_still_quota_capped, test_default_collection_corpus_is_labelled_corpus in tests/test_verify_sql_retrieval.py. The existing suite does not catch this because every quota test builds an explicit kind='corpus' collection.

### C4 — `migrations/033_corpus_surface.sql`  ·  owner **F2**

document_version_chunks' RLS policy validates only the version side ('EXISTS (SELECT 1 FROM document_versions dv WHERE dv.id = document_version_id)'), never the chunk side. A tenant may graft another tenant's chunk id into its own document version. Combined with the unscoped pgkg_chunk_window() above, this turns a known chunk UUID into a cross-ORG data-injection channel, and it also lets a stranger keep a victim's chunk pgkg_chunk_live() forever.

**Reproduction.** Victim org owns chunk C via its own current document version. Attacker org, connected as pgkg_app with pgkg.org_id set to its own org, runs INSERT INTO document_version_chunks (document_version_id, chunk_id, ord) VALUES (attacker_version, C, 1) — the write SUCCEEDS. The victim then recalls C in its own scope and context_text comes back as 'Acme Holdings will be acquired for four hundred million ... <victim footer>' — the attacker's prose delivered inside the victim's result row.


## High (10)

### H1 — `migrations/011_bm25_stats.sql`  ·  owner **F1**

corpus_stats and lexeme_df are keyed PRIMARY KEY (namespace, kind) with no org column, and pgkg_bm25_candidates' proposition arm reads them with only 'cs.namespace = p_namespace AND cs.kind = 'proposition''. Every tenant's Memory uses one namespace, so N, avgdl and every term's document frequency — the whole IDF — are computed globally across all orgs. This is D4's hard rule ('Ranking signals are never computed globally over shared content ... a real cross-tenant inference channel'). The chunk half of the SAME function keys its statistics per collection via pgkg_stats_domain(), so the omission is asymmetric rather than deliberate.

**Reproduction.** Org MINE writes 3 propositions containing rare term T; pgkg_bm25_candidates(q_text=T, p_org_ids=[MINE]) scores its top hit 0.1335. Org THEIRS then writes 200 propositions containing T. With no change whatsoever to org MINE's data, the same query now scores 0.00263 — a 50x shift. A tenant can therefore measure how much other tenants have written about any term (a competitor name, an acquisition codename) purely from its own score movements.

### H2 — `migrations/030_corpus_lifecycle.sql`  ·  owner **F2**

pgkg_add_version_chunk() derives a new chunk's org, collection, ACL group and provenance from the version's document, but its `ON CONFLICT (org_id, content_hash) WHERE document_id IS NULL DO NOTHING` path leaves every one of those columns at whatever the first ingest wrote. The content address is org-wide while CorpusIngest is per (org, collection), so a passage shared by two collections in one org is permanently scoped, decayed, claim-scoped, quota-bucketed and ACL-gated as the first collection's. A query scoped to the second collection cannot retrieve its own document's passage; a query scoped to the first retrieves it under the wrong decay profile and claim scope. Since p_acl_group_id is on the same insert, the same path launders a document ACL once corpus.py starts passing it.

**Reproduction.** One org, collection A (claim_scope 'world') and collection B (claim_scope 'org'). CorpusIngest(coll_a).upsert_document(body) then CorpusIngest(coll_b).upsert_document(same body). Document B's current version links exactly one chunk and that chunk's collection_id is coll_a, not coll_b. Test: test_carried_over_chunk_keeps_first_collection.

### H3 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

The chunk BM25 statistics are maintained over every row of `chunks`, but retrieval only returns rows passing pgkg_chunk_live(). pgkg_chunks_stats_delta() and pgkg_refresh_chunk_stats() have no liveness predicate at all, so chat-provenance chunks (document_id NOT NULL, written by memory.py ingest), chunks whose only links are to retired versions, and chunks of soft-deleted documents all contribute to n_total, total_len and lexeme_df. That is a permanently wrong population, not stale statistics: the proposition side of the same tables is correctly filtered on invalidated_at, matching what its arm retrieves.

**Reproduction.** One retrievable passage 'zorblatt calibration procedure' in a corpus collection scores raw_score 0.2877 for query 'zorblatt'. Insert 20 chat-provenance chunks (document_id set) into the same collection, each containing 'zorblatt'. pgkg_bm25_candidates(..., 'chunks') still returns exactly one row — the chat chunks are correctly excluded by pgkg_chunk_live — but its score has collapsed to 0.0273, a 10.5x drop, because lexeme_df('zorblatt') went 1 -> 21 and n_total went 1 -> 21. Test: test_chat_provenance_chunks_do_not_skew_chunk_bm25. Reachable on the default path: Memory and CorpusIngest both default to DEFAULT_COLLECTION_ID, so chat turns and corpus documents share one statistics domain.

### H4 — `migrations/031_corpus_retrieval.sql`  ·  owner **F2**

pgkg_chunk_window() takes no org, collection, user or ACL argument and filters only on dv.status='current' AND d.deleted_at IS NULL. Because a content-addressed chunk can belong to the current version of several documents, pgkg_retrieve's `windows` CTE (DISTINCT ON (chunk_id) ... ORDER BY document_version_id) hands back a small-to-big context assembled from an arbitrary one of them. The returned context_text therefore carries neighbouring passages from documents in collections the query never named, and from documents behind a different acl_group_id, while every scoring stage was correctly scoped.

**Reproduction.** One org, collections A and B, ten independent document pairs each sharing one boilerplate passage between a document in A and a document in B, all promoted. Taking the window pgkg_retrieve would take (ORDER BY document_version_id LIMIT 1), 3 of the 10 shared passages returned context text containing a marker string that occurs only in collection B's document. The 30-70% rate is the coin flip on which of two random version UUIDs sorts lower. Test: test_chunk_window_stays_inside_the_queried_collections.

### H5 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

pgkg_chunk_live() does not inline, so chunk liveness costs a nested-loop subquery per candidate row on the hot retrieval path.

**Reproduction.** The predicate appears in the plan as an unexpanded `Filter: … pgkg_chunk_live(id, refcount)` — unlike pgkg_visible and pgkg_temporal_visible, which do inline into plain column comparisons — so its two sublinks (a self-lookup on chunks plus a three-table join) run once per candidate chunk. On 4,000 candidates the function form touches 28,937 shared buffers against 927 for the identical test written inline as a predicate (31x); on 40,000 candidates it adds ~339,000 buffer hits and 950 ms to a single keyword arm, and rewriting it as a predicate lets the planner build one hashed SubPlan instead (~6x faster). Under OR keyword semantics the candidate set is a large fraction of the corpus, so this scales with corpus size. Reproduced by tests/test_verify_perf.py::test_chunk_liveness_does_not_cost_a_subquery_per_candidate.

### H6 — `migrations/040_entity_mentions.sql`  ·  owner **F3**

pgkg_search() silently loses graph-expanded facts once gazetteer mentions exist, because the rewritten graph arm spends the k_total budget on chunk ids that pgkg_search then discards.

**Reproduction.** 040 rewrote pgkg_graph_candidates() to emit chunk ids alongside proposition ids, but pgkg_search() (migration 022's copy) still calls it and consumes the result with `JOIN propositions p ON p.id = fused.item_id`. Chunk candidates are therefore dropped only after they have occupied places in `capped`/`LIMIT k_total`. With 20 seed entities, 12 facts and 12 passages each, pgkg_search('zorbulon', …) returns 120 propositions before any entity_mentions rows exist and 70 after — a 42% recall loss with nothing returned in place of the evicted facts. pgkg_search()'s proposition-shaped contract is what the 59 baseline tests pin. Reproduced by tests/test_verify_perf.py::test_mentions_do_not_displace_facts_in_pgkg_search.

### H7 — `pgkg/corpus.py`  ·  owner **F4**

The 'perishable' decay profile is keyed on ingest time, not publication date. D6 specifies perishable as keyed on `asserted_at = published_at` and D5 says provenance.published_at 'feeds the perishable decay profile', but CorpusIngest.upsert_document takes `asserted_at` and `provenance.published_at` as independent parameters and never derives one from the other (the same two fields are independent on the HTTP DocumentRequest). chunks.asserted_at is left NULL, so pgkg_apply_profile falls back to COALESCE(c.asserted_at, c.created_at) = the ingest timestamp, and the perishable factor is 1.0 — behaviourally identical to 'timeless', which is the profile the ADR is at pains to distinguish it from.

**Reproduction.** A perishable collection; upsert_document(text=..., provenance=Provenance(kind='document_version', producer='chunker', published_at=2015-01-01)). provenance.published_at stores correctly as 2015-01-01, chunks.asserted_at is None, and pgkg_apply_profile on the resulting chunk returns adjusted_score exactly 1.0 for a raw score of 1.0. At the 730-day half-life the eleven-year-old article should score about 0.0041 — a 245x over-ranking of content the ADR calls 'not stale, wrong'. Test: test_perishable_decays_on_publication_date.

### H8 — `pgkg/corpus.py`  ·  owner **F4**

Corpus ingest calls the embedder and a per-chunk LLM extractor from inside its open transaction, holding a pooled connection across every model call.

**Reproduction.** `upsert_document` opens `conn.transaction()` and then calls `_vectorise`, which runs `self._embed_texts(missing)` (corpus.py:523, synchronous — it also blocks the event loop), and `_extract`, which awaits `ml.extract_propositions_async(text)` once per chunk in a serial `for` loop. memory.py's own `_IngestPlan` docstring states the rule this breaks: 'holding the ingest connection across them starves the pool under concurrent ingest'. A 50-chunk document with a 2 s extractor keeps one pooled connection and one open transaction for ~100 s, and `_ConnExtractCache` bumps proposition_cache.hit_count on that same connection, holding those row locks for the whole time so a second ingest of overlapping content blocks. Reproduced by tests/test_verify_perf.py::test_corpus_ingest_does_not_embed_inside_its_transaction, which observes `conn.is_in_transaction()` is True at the moment the embedder is invoked.

### H9 — `pgkg/ingest_jobs.py`  ·  owner **F4**

IngestWorker deadlocks permanently when `slots` reaches the pool size, because a slot holds two connections rather than the one its docstring rations.

**Reproduction.** `run()` documents slots as rationing 'connections held', but each in-flight document holds the ingest transaction's connection AND takes a second one in `_progress_reporter.report()` — deliberately, so progress is visible while that transaction is open. With slots == pool max_size every slot waits for a connection every other slot is holding, and nothing ever releases. `pgkg worker --slots N` passes `slots=N, concurrency=N` against the default `make_pool` max_size of 10, so `--slots 10` hangs forever with no error. Reproduced on a max_size=2 pool with slots=2: `asyncio.wait_for(worker.run(concurrency=2), timeout=15)` raises TimeoutError, while the control with slots=1 on the same pool drains both jobs in 2 s (tests/test_verify_perf.py::test_the_worker_does_not_deadlock_at_its_slot_budget and ::test_control_the_worker_drains_when_a_slot_is_spare).

### H10 — `pgkg/memory.py`  ·  owner **F4**

Memory.forget() scopes its UPDATE with self._scope.read_org_ids (line ~409, _FORGET_SQL 'WHERE id = $3 AND org_id = ANY($4)'), not write_org_id. Scope's own docstring says 'Reads widen ... writes never do', and include_system_org is a plain field on the unauthenticated HTTP ScopedRequest. So any tenant that widens its reads can invalidate propositions in the reserved SYSTEM_ORG — operator-published material every other subscriber reads. Mitigated only if the deployment actually connects as pgkg_app (propositions' policy is org_id = pgkg_current_org()); the RLS policy is the sole thing standing between this and cross-tenant data destruction, which is precisely the 'one missing WHERE clause' D3 warns about.

**Reproduction.** Memory(pool, scope=Scope(org_id=tenant, include_system_org=True)).forget(system_org_proposition_id, reason='user_deleted') sets invalidated_at and invalidation_reason='user_deleted' on the SYSTEM_ORG row. Reproduced on the test pool: the shared proposition came back invalidated. POST /forget with include_system_org=true is the same call over HTTP.


## Medium (11)

### M1 — `migrations/005_proposition_cache.sql`  ·  owner **F3**

proposition_cache is keyed on cache_key = hash(chunk_text, extractor_model) alone: no org column, no RLS, no public_source gate. Phase 2 newly wired it into the corpus path (pgkg/corpus.py:595, _ConnExtractCache gated only on self._use_extract_cache — NOT on policy.public_source, unlike the embedding cache at corpus.py:546/561). D4 restricts embedding_cache to operator-licensed material because 'for a confidential document a cache hit would confirm another tenant holds it'; the extraction cache carries the extracted facts themselves and has no such restriction.

**Reproduction.** Org A ingests a confidential document into a private (public_source=false) collection with extract_propositions=true; the extracted propositions land in proposition_cache under a key derived from the chunk text. Org B, holding the same text (e.g. a leaked draft), gets a cache hit and reads Org A's extraction verbatim. Reproduced: proposition_cache has no org_id column and relrowsecurity=false, and a SELECT as pgkg_app with an unrelated pgkg.org_id returns the cached payload in full.

### M2 — `migrations/030_corpus_lifecycle.sql`  ·  owner **F2**

pgkg_open_document_version() computes MAX(version_no) + 1 with no lock, so two concurrent crawls of one document collide instead of serialising.

**Reproduction.** The function reads `COALESCE((SELECT MAX(dv.version_no) …), 0) + 1` and inserts, with `document_versions_document_id_version_no_key` unique on (document_id, version_no) and no advisory or row lock taken on the document first. Two connectors (or two IngestWorker loops, or an upsert racing a re-crawl) that open a version for the same document in overlapping transactions both compute the same number; the second fails with `duplicate key value violates unique constraint "document_versions_document_id_version_no_key"`, which propagates out of upsert_document as an unhandled UniqueViolationError. Reproduced by tests/test_verify_perf.py::test_two_concurrent_crawls_of_one_document_do_not_collide (two overlapping transactions, 0.1 s apart).

### M3 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

pgkg_vector_candidates() never filters on embedder_generation_id, so the primary vector arm computes cosines between the primary generation's query vector and rows whose inline vector belongs to another generation, and ranks the mixture in one ORDER BY. D8 states plainly that cosines from different model spaces are incomparable, and memory.py's MMR query already guards against exactly this mixture (it nulls the embedding unless embedder_generation_id equals the org's primary), so the codebase knows the mixture occurs. The affected row also gets a second, correctly-scoped vote through pgkg_generation_candidates, double-counting it in RRF.

**Reproduction.** A cutover at D8 step 5 with two same-width generations: gen2 promoted to primary, gen1 demoted to 'retiring', its rows still inline and tagged embedder_generation_id=gen1. pgkg_vector_candidates(q_gen2_space, ...) returns both the gen2-tagged proposition and the gen1-tagged one, ranking the gen1 row first because its stale vector happens to be nearer in raw coordinates. Test: test_primary_vector_arm_stays_in_the_primary_model_space. Not reachable today (no cutover has been run and no second generation is registered by any production path), which is why the severity is medium rather than high.

### M4 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

The BM25 score sub-select is planned and executed twice for every candidate row, once for the `> 0` filter and once for the sort.

**Reproduction.** pgkg_bm25_candidates ends `… ) sub WHERE sub.bm25_score > 0.0 ORDER BY sub.bm25_score DESC LIMIT k_initial`. Because `bm25_score` is a correlated sub-select, Postgres emits two independent SubPlans — the plan shows `SubPlan 6` and `SubPlan 7` (propositions arm) / `SubPlan 9` and `SubPlan 10` (chunks arm), each with `loops=40000` for 40,000 candidates, i.e. 80,000 evaluations of the per-row aggregate. Isolated head-to-head on the same data: the shipped shape runs in 147 ms with 2 SubPlans; the identical scoring wrapped in a MATERIALIZED CTE runs in 81 ms with 1. Applies to both arms and to every copy of the function (010, 020, 021, 031). Reproduced by tests/test_verify_perf.py::test_bm25_scores_each_candidate_once.

### M5 — `migrations/031_corpus_retrieval.sql`  ·  owner **F1**

The IDF lookup is still a full-table aggregate over lexeme_df: the query's own lexemes never reach the index.

**Reproduction.** The `term_df` CTE filters only on namespace/kind and expresses the lexeme restriction as a downstream `Hash Right Join` against the query_lexemes CTE, so the `lexeme_df_pkey` index on (namespace, kind, lexeme) is unusable. For a three-term query the plan shows `Seq Scan on lexeme_df … rows=4009` followed by a HashAggregate of all 4,009 rows — the entire vocabulary of that corpus. At 40k propositions this costs 17.3 ms per query; the same aggregate written as `AND ld.lexeme = ANY(ARRAY[...])` uses the index and costs 0.049 ms — 350x, and the gap widens with vocabulary. The per-row correlated COUNT(*) is gone, but a full aggregate whose cost grows with the corpus remains on the query path. Reproduced by tests/test_verify_perf.py::test_idf_reads_only_the_query_terms.

### M6 — `pgkg/corpus.py`  ·  owner **F4**

Corpus ingest makes one round trip per chunk inside its open transaction, so the pipeline built for a 300-page handbook is not set-based.

**Reproduction.** `_link_chunks` loops over `chunk_document(...)` awaiting `conn.fetchrow(_ADD_CHUNK_SQL, ...)` once per chunk. Measured through a recording pool: a 2-chunk document costs 12 round trips and a 30-chunk document costs 40 — one extra sequential round trip per chunk, each one a network RTT with the ingest transaction held open. A 500-chunk document is 500 serialised round trips. Chat ingest was explicitly made set-based in phase 0 (5 statements for 12 chunks and 60 propositions); the bulk path was not. Reproduced by tests/test_verify_perf.py::test_corpus_ingest_round_trips_do_not_grow_with_chunk_count.

### M7 — `pgkg/corpus.py`  ·  owner **F4**

Re-ingesting a soft-deleted external_id raises UniqueViolationError, contradicting the pipeline's documented intent.

**Reproduction.** `_FIND_DOCUMENT_SQL` filters `AND deleted_at IS NULL` with the comment 'A soft-deleted document is not a collision: re-ingesting a withdrawn external id is a new document', but `documents_external_id_key` is `UNIQUE (org_id, collection_id, external_id) WHERE external_id IS NOT NULL` and does not exclude soft-deleted rows. So the lookup misses the withdrawn row and the subsequent INSERT hits the index. Reachable end to end through the shipped HTTP surface: POST /documents/delete (which only sets deleted_at) followed by an upsert of the same external_id raises `duplicate key value violates unique constraint "documents_external_id_key"` and returns 500. Reproduced by tests/test_verify_perf.py::test_reingesting_a_deleted_document_starts_a_new_one.

### M8 — `pgkg/ingest_jobs.py`  ·  owner **F4**

job_state() (line 114, _STATE_SQL = 'SELECT ... FROM ingest_jobs WHERE id = $1') takes no org argument and never sets the pgkg.org_id GUC on the connection it acquires, so nothing in the read names the caller. GET /jobs/{job_id} passes the raw pool straight through. Anyone holding a job UUID reads another tenant's status, attempt count, document_id, version_id and error text.

**Reproduction.** Org STRANGER enqueues a document via pgkg_enqueue_ingest_job. A caller with no relationship to that org calls job_state(pool, job_id) and receives JobState(status='pending', attempts=0, ...) rather than a KeyError. Because the connection sets no GUC, pgkg_current_org() falls back to pgkg_default_org(), so the ingest_jobs policy 033 added cannot help even when RLS is live — and on an owner connection it is inert entirely.

### M9 — `pgkg/memory.py`  ·  owner **F4**

A partly-applied access flush is restored in full, so any org whose UPDATE already committed is counted twice.

**Reproduction.** `flush_access()` loops over orgs issuing one autocommitted UPDATE each (no enclosing transaction), and on any exception calls `self._access.restore(pending)` with the whole drained dict — including the orgs whose UPDATE has already committed. Those counts are then applied a second time on the next flush. With three accesses recorded for org A and three for org B, and the connection failing on B's statement, org A's proposition ends at access_count 6 instead of 3. The frequency term in the decay profile reads this column, so a transient link failure permanently inflates ranking for whichever tenant happened to flush first. Reproduced by tests/test_verify_perf.py::test_a_failed_access_flush_does_not_double_count.

### M10 — `tests/test_api_scoping.py`  ·  owner **F3**

The D4 sharing seam is tested only as Scope dataclass property assertions (test_system_org_is_readable_but_never_written, test_subscribed_collections_widen_reads_only, lines 157-169). No test asserts end-to-end that a tenant with include_system_org=True actually retrieves a SYSTEM_ORG row from the database — and under RLS it could not, because every policy is 'org_id = pgkg_current_org()' (one org) while the retrieval predicate allows [tenant, SYSTEM_ORG]. The direction is fail-closed, so this is a correctness/test-vacuity finding rather than a leak, but it means the sharing seam is unverified against Postgres and that include_system_org's only demonstrated effect at HEAD is the write-side widening in finding 4.

**Reproduction.** Scope(org_id=X, include_system_org=True).read_org_ids == [X, SYSTEM_ORG_ID] is asserted; nothing asserts that pgkg_retrieve/pgkg_search with p_org_ids=[X, SYSTEM_ORG] returns a system-org row. Separately, tests/conftest.py's pool connects as the container's owning superuser, so EVERY RLS policy is inert in every test except the handful that explicitly SET LOCAL ROLE pgkg_app (test_rls_coverage.py, test_app_role.py). Any isolation test not using that role is testing the SQL predicate only, never the policy.

### M11 — `tests/test_rls_coverage.py`  ·  owner **F3**

The coverage test asserts `protected == sorted(SEEDS)` — it enumerates tables that HAVE relrowsecurity and pins that set. It therefore cannot notice a new table that ships with an org column and NO policy. Migration 040 added entity_mentions and entity_links (both with org_id, both granted to pgkg_app) with row security deliberately deferred, and the suite stayed green. The module's stated purpose ('a table that gains a policy without gaining a case here would ship an untested isolation boundary') does not cover the more likely failure of a table that never gains a policy.

**Reproduction.** tests/test_rls_coverage.py::test_every_rls_enabled_table_is_covered_here passes at HEAD while entity_mentions and entity_links are readable and writable by pgkg_app under any pgkg.org_id. The only boundary on those tables is pgkg_visible() re-applied on the read path; a direct SELECT or an INSERT of a forged mention edge is unconstrained by Postgres.


## Low (1)

### L1 — `migrations/040_entity_mentions.sql`  ·  owner **F3**

CONTROL — NOT A DEFECT. D3's hard requirement (every row reached by graph expansion is re-filtered through the seed's visibility predicate) holds in BOTH directions after phase 3, and holds non-vacuously. Recording it so the parent does not re-investigate.

**Reproduction.** One org, users A and B, shared entity 'Zzqhelios Programme'. User A owns a private proposition and a private passage (chunks.visibility='private', owner_user_id=A, mention edge built by pgkg_match_entity_mentions). User B's seed is a shared fact naming the entity. pgkg_retrieve(expand_graph=TRUE, p_user_id=B) returns neither PRIVATE_FACT nor PRIVATE_PASSAGE. Non-vacuity check: the identical query with p_user_id=A returns BOTH, proving the walk reaches them and pgkg_visible() in the fact arm and the chunk arm is what stops it.
