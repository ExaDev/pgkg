# Changelog

## 0.6.0

ADR-0001 phases 0-3: the corpus becomes a first-class retrievable store alongside the proposition
graph, and the whole schema gains a tenancy boundary. Migrations 010-047. See
[`docs/adrs/0001-implementation-notes.md`](docs/adrs/0001-implementation-notes.md) for what was
built against what was specified, and every deliberate deviation.

**Phase 0 — retrieval foundations** (migrations 010-013)

- **Composable retrieval**: `pgkg_search()` decomposed into set-returning functions over a
  `pgkg_candidate` row type — `pgkg_bm25_candidates()`, `pgkg_vector_candidates()`,
  `pgkg_graph_candidates()`, `pgkg_fuse()`, `pgkg_apply_profile()`. An arm can now be tested,
  replaced or weighted on its own.
- **Materialised BM25 statistics**: `corpus_stats` and `lexeme_df`, maintained by statement-level
  triggers with transition tables. The per-row correlated `COUNT(*)` over the corpus is gone.
- **`halfvec` storage and no hardcoded dimension**: the embedding width is read off the column with
  `pgkg_embedding_dim()`; `vector(1024)` no longer appears in any DDL or signature.
- **Content-defined chunk boundaries** (`pgkg/chunking.py`), so a typo fixed mid-document does not
  reflow every chunk hash after it.
- **Set-based, transactional chat ingest** and batched access-count flushes.
- **Idempotent entity resolution**: two concurrent linkers of one name cannot both insert.

**Phase 1 — tenancy, provenance, time, embedder registry** (migrations 020-026)

- **Scoping columns on every retrievable row**: `org_id`, `collection_id`, `claim_scope`,
  `visibility`, `owner_user_id`, `acl_group_id`, read by one shared inlinable predicate
  `pgkg_visible()`.
- **Row-level security** on every table carrying an org, and a `pgkg_app` role that is not
  `BYPASSRLS`. A coverage test enumerates the tables and fails when a new one arrives without a
  policy.
- **Collections** (`collections`) carrying the decay profile, claim scope, visibility, licence and
  `public_source` flag; **subscriptions** (`collection_subscriptions`) and a reserved system org for
  operator-published material. Reads may widen to it; writes never do.
- **Provenance** (`provenance`, `proposition_provenance`) as a shared, append-only derivation
  record, so erasure is an indexed query.
- **Bitemporal facts**: `invalidated_at`, `invalidation_reason`, `valid_from`, `valid_to` replace the
  `superseded_by` filter. `pgkg_believed_at()` answers as of a past instant;
  `pgkg_contradict()` and `pgkg_expire_due()` withdraw.
- **Embedder registry**: `embedder_generations` + `org_embedders`, the current model registered as
  generation 1, one primary generation per org enforced by index, and a dual-generation vector arm
  for cutover windows.

**Phase 2 — the corpus as a first-class source** (migrations 030-033, `pgkg/corpus.py`,
`pgkg/ingest_jobs.py`)

- **Versioned documents and immutable content-addressed chunks**: an unchanged nightly crawl does no
  work at all, and an edited handbook re-embeds only the chunks that moved. One current version per
  document, enforced by partial unique index; reference-counted reclamation on a separate clock
  from retirement.
- **`pgkg_retrieve()`**: the two-store surface. Both keyword arms, both vector arms, graph
  expansion, weighted RRF, quotas, decay and context windows in one composition. `recall()` now
  goes through it; `pgkg_search()` keeps its proposition-shaped contract.
- **Quotas**: a ceiling on the corpus's share of a result set and a floor under personal memory, so
  a 300-page handbook cannot bury what the user said yesterday.
- **Three decay profiles** per collection: `conversational`, `timeless`, and `perishable` keyed on
  the publication date rather than the crawl date.
- **Small-to-big context**: `pgkg_chunk_window()` returns a hit with its neighbouring passages, and
  never with passages from a document the caller could not have read.
- **Batch ingest queue** (`ingest_jobs`) and a worker (`pgkg worker`), with provenance and the
  publication date carried through the queue.
- **HTTP surface**: `POST /collections`, `POST /documents` (with `queue=true`),
  `POST /documents/delete`, `GET /jobs/{id}?org_id=`, plus `POST /forget` and `POST /believed`.
- **Corpus extraction is opt-in per collection** and off by default: extracting a whole corpus is a
  recurring cost and lossy on exactly the content corpora are made of.

**Phase 3 — joining the corpus to the graph** (migration 040, `pgkg/gazetteer.py`)

- **`entity_mentions`**: gazetteer matching over new passages, so a passage is reachable from an
  entity without having been proposition-extracted. `chunks.mentions_matched_at` is the queue.
- **Bidirectional graph expansion**: facts to entities to passages and back, with every hop
  re-filtered through the seed's own visibility predicate.
- **`entity_links`**: a bridge from an org's entity to a shared one, direction enforced by trigger,
  rather than copying the shared graph per tenant.

**Verification and fixes** (migrations 041-045)

Phases 0-3 were audited by three independent adversarial verifiers, which confirmed 26 items —
4 critical, 10 high, 11 medium and 1 control — each with a reproduction. A fourth pass then
re-audited the fixes and found one critical incompletely fixed, one regression, and one defect that
had shipped with phase 2. All are closed and recorded in
[`docs/adrs/0001-verification-findings.md`](docs/adrs/0001-verification-findings.md). The changes
worth knowing about as a user of the schema:

- **Ranking signals are per tenant.** `corpus_stats` and `lexeme_df` are keyed
  `(kind, namespace, org_id, collection_id)`. Previously every tenant's IDF was computed globally,
  so a tenant could measure how much other tenants had written about any term from its own score
  movements. The statistics filter now mirrors the candidate scan's scope filter term for term.
- **`context_text` cannot launder a document.** `pgkg_chunk_window()` carries its own predicate: a
  neighbour must be indistinguishable from the anchor to every read predicate, and a passage carried
  by more than one live document is its own context rather than borrowing an arbitrary one's
  neighbours.
- **The content address narrows** from `(org_id, content_hash)` to
  `(org_id, collection_id, acl_group, content_hash)`. A passage present in two collections is now a
  row in each, because `collection_id` selects the claim scope, decay profile, statistics domain and
  quota bucket, and one row cannot hold two values of it.
- **Chunk liveness is a stored column** (`chunks.retrievable`), reconciled by trigger, and is both
  the scan predicate and the statistics population so the two cannot disagree. It no longer depends
  on `refcount`: `chunks.version_scoped` records that a version once carried a passage, so
  reclaiming a retired version no longer resurrects the passages its successor dropped.
- **Quota bucketing keys on structure**, not on the `collections.kind` vocabulary, so passages in
  the default (`mixed`) collection are subject to the corpus ceiling.
- **The vector arm stays in one model space**: candidates are restricted to the org's primary
  embedder generation, so cosines from different model spaces are never ranked together.
- **The sharing seam works end to end.** Read policies widen to the operator's system org; every
  `WITH CHECK` remains a single-org equality. Previously a tenant with `include_system_org` got its
  own rows and silently nothing else.
- **`document_version_chunks` is own-org to write** and visible-to-read, and cross-org links are
  refused by trigger as well. Previously the policy validated only the version side.
- **`proposition_cache` is keyed `(cache_key, org_id)`** with a policy, so a cache hit can no longer
  hand one tenant another tenant's extracted facts.
- **Both functions behind `@@` are marked `LEAKPROOF`**, without which every BM25 arm degraded to a
  sequential scan under the role the RLS policies were written for. The mark needs ownership of a
  built-in, so where it cannot be applied `GET /health` now reports it as `keyword_index`: a
  deployment that missed the fix is a monitorable fact rather than a silent order-of-magnitude
  regression under load.
- **The gazetteer reaches its indexes under the application role too**: `similarity_op` and
  `arraycontains` are marked `LEAKPROOF` and the gazetteer keys are stored generated columns on
  `entities`, so none of the three arms of `pgkg_match_entity_mentions()` is a sequential scan of
  the entity table any more. Measured on 40,001 entities: 42.6 / 186.6 / 74.3 ms per probe under
  `pgkg_app`, against a few buffers on the index plans. This runs per chunk on ingest.
- **Performance**: BM25 scores each candidate once instead of twice; the IDF lookup is served by the
  primary key instead of aggregating the whole vocabulary; corpus ingest is set-based (16 round
  trips regardless of chunk count) and makes no model call inside a transaction; the ingest worker
  no longer deadlocks at its slot budget.
- **Write scoping**: `forget()` and the access-count flush name the write org, not the widened read
  list.

**Also**

- `PGKG_DATABASE_URL` may be omitted entirely: pgkg then starts an embedded Postgres through
  `pgserver`, with no Docker.
- `documents_external_id_key` is partial on `deleted_at IS NULL`, so re-ingesting a withdrawn
  external id starts a new document instead of raising.
- `GET /jobs/{job_id}` takes `?org_id=`; a job's status is no longer readable by anyone holding its
  UUID.


## 0.5.0

- **Assertion timestamps (`asserted_at`)**: New nullable `asserted_at TIMESTAMPTZ` column on `propositions` and `chunks` (migration 006). When set, recency decay in `pgkg_search()` keys on when the fact was originally asserted rather than when it was indexed (`COALESCE(asserted_at, last_accessed_at)`). No change for rows where `asserted_at` is NULL.
- **API**: `POST /memorize` accepts optional `asserted_at` (ISO 8601) in the request body.
- **`Memory.ingest()`**: New `asserted_at: datetime | None = None` parameter; propagated to both chunk and proposition rows.
- **`Memory.recall()` / `Result`**: `Result` model gains `asserted_at: datetime | None`; populated from `pgkg_search()` output.
- **Bench harnesses**: `ingest_conversation()` in `bench/common.py` now parses and forwards `timestamp` fields from turn dicts as `asserted_at`. `bench/locomo.py` wires per-turn timestamps; `bench/longmemeval.py` populates `timestamp` from session-level metadata for temporal-reasoning category support.

## 0.4.1

- Promote zero-LLM chunks-only path as the lead local-experimentation flow. New `.env.local-chunks` preset and `make local-chunks` target — no API key, no `claude` CLI required.
- Replace ill-defined `local-claude-chunks` target (which still required `claude` CLI for no functional reason) with the cleaner `local-chunks`.
- Restructure README "Local experimentation" section into three explicit paths: zero-LLM, Claude subscription, paid API.

## 0.4.0

- Add chunks-only ingest mode (`--chunks-only`, `PGKG_EXTRACT_PROPOSITIONS=0`). Skip LLM extraction; store chunks directly as propositions. Equivalent to vanilla hybrid RAG. Enables `pgkg-chunks` vs `pgkg-propositions` ablation.
- New make targets: `bench-mem0-stack-chunks`, `local-claude-chunks`.

## 0.3.0

- Add `claude_code` provider for local experimentation via Claude Agent SDK (uses the local `claude` CLI; subscription-auth, no API key). Not for benchmarking.
- New `.env.local-claude` preset and `make local-claude` target.

## 0.2.0 (2026-04-28)

- **Proposition extraction cache**: Added `proposition_cache` table (migration 005) and `PostgresExtractCache` implementation. Re-ingesting the same chunk with the same extractor model and prompt version hits the cache instead of calling the LLM. Cache is bypassed in offline-extract mode.
- **Pinned model IDs**: Default `llm_model` changed to `gpt-4o-mini-2024-07-18`; added `judge_model = gpt-4o-2024-08-06`, `extractor_model` override, and `openai_base_url` for OpenRouter/Groq compatibility.
- **Stack presets**: Added `.env.bench-mem0-stack`, `.env.bench-zep-stack`, `.env.bench-openrouter-free` and corresponding `make bench-*-stack` targets with cost warnings. `BenchReport` now includes a full `StackInfo` snapshot (models, git SHA, retrieval parameters) for reproducibility.

## 0.1.0

- Initial release: hybrid retrieval (BM25 + HNSW + RRF), graph expansion, recency/frequency decay, MMR, cross-encoder reranking, FastAPI endpoints, LoCoMo and LongMemEval bench harnesses.
