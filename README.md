# pgkg — Postgres-native knowledge graph engine for agentic memory

**The thesis:** Vanilla Postgres with `pgvector` and `tsvector` can match the retrieval quality of complex agent-memory stacks (Mem0, Zep, MemGPT) and knowledge graph systems that bolt on Kafka, Pinecone, Neo4j, etc. The only non-SQL components are embedding, reranking, and LLM-based proposition extraction.

**Two containers, one CTE:** Postgres does the work. Everything else is glue — and the ratio shows it. The retrieval pipeline, the tenancy boundary, the document lifecycle and the graph walk are all SQL; the Python is an HTTP surface, a model client and an ingest loop.

## What it does

**Retrieval**

- **Two stores, one surface:** passages and extracted facts are both answers to "what did we agree about the refund policy". `pgkg_retrieve()` runs both and fuses them; deciding between them is the caller's job, not the retriever's.
- **Hybrid retrieval:** BM25 over `tsvector` with materialised corpus statistics + dense vector (HNSW over `halfvec`), fused via Reciprocal Rank Fusion.
- **Graph expansion:** retrieved entities seed a one-hop neighbour search in both directions — facts to entities to passages and back — with every hop re-filtered through the seed's own visibility predicate.
- **Quotas:** a ceiling on how much of a result set the corpus may take, and a floor under personal memory, so a 300-page handbook cannot bury what the user told you yesterday.
- **Decay profiles per collection:** `conversational` (30-day half-life), `timeless` (flat), `perishable` (multi-year half-life keyed on the publication date, not the crawl date).
- **MMR diversity** and **cross-encoder reranking** on the final candidate set.

**Tenancy and provenance**

- **Explicit scoping columns on every retrievable row** — org, collection, claim scope, visibility, owner, ACL group — read by one shared, inlinable predicate, and enforced again by row-level security.
- **A sharing seam:** an operator can publish a collection that every tenant reads and none can write, without pooling anyone's private material into it.
- **Bitemporal facts:** when a claim was recorded, when it was believed, when it was withdrawn and why. `pgkg_believed_at()` answers as of a past instant.
- **Provenance as a shared immutable derivation record**, so erasure is an indexed query rather than a crawl.

**Corpus lifecycle**

- **Versioned documents and immutable content-addressed chunks:** an unchanged nightly crawl of 100k documents does no work at all, and a typo fixed in a 300-page handbook is one embedding call rather than 300.
- **Atomic version flips** with reference-counted reclamation on a separate clock from retirement.
- **Small-to-big context:** a hit comes back with its neighbouring passages, and never with passages from a document the caller could not have read.
- **A batch ingest queue** with a worker, so a corpus crawl is not an HTTP request.
- **Embedder generations:** the model that produced a vector is recorded, cosines from different model spaces are never compared, and a cutover can serve two spaces at once.

## Quickstart

End-to-end in two HTTP calls. Pick a path.

### Path A — Vanilla RAG (zero LLM, no API key, no claude CLI)

Pure chunks-only mode: chunks → embedder → Postgres. Hybrid retrieval + rerank + MMR + recency, no proposition extraction.

```bash
cp .env.local-chunks .env
make local-chunks
```

```bash
curl -X POST http://localhost:8000/memorize \
  -H 'Content-Type: application/json' \
  -d '{"text":"pgkg is a Postgres-native knowledge graph engine for agentic memory. It was built by ExaDev. Chunks-only mode skips LLM extraction entirely."}'

curl -X POST http://localhost:8000/recall \
  -H 'Content-Type: application/json' \
  -d '{"query":"who built pgkg?","k":3}' | python3 -m json.tool
```

You'll get the chunk back with a high score — vector + tsvector hybrid retrieval working. The `subject`, `predicate`, and `object` fields will be `null` because nothing extracted facts.

### Path B — Propositions mode via Claude subscription (no API key)

LLM extracts atomic facts at ingest. You'll see `subject`/`predicate`/`object` populated and the `text` field will be a short atomic statement, not the source paragraph.

Prereqs: Claude Pro/Max subscription, `claude` CLI installed and logged in (run `claude` once to authenticate).

```bash
uv sync --extra claude_agent
cp .env.local-claude .env
make local-claude
```

Same curl as Path A. Compare the result shape:

```jsonc
// Path A (chunks): one big chunk back
{ "text": "pgkg is a Postgres-native knowledge graph engine...", "subject": null, "predicate": null }

// Path B (propositions): atomic fact, predicate populated
{ "text": "pgkg built by ExaDev", "predicate": "built by", "source_kind": "both" }
```

`source_kind: "both"` only fires for atomic propositions because keyword search has a chance against short focused text. That's the win.

### Switching between paths

```bash
make wipe                  # truncate ingested data, keep schema
cp .env.local-claude .env  # or .env.local-chunks
# Ctrl-C the running server, then re-run make local-* in the same terminal
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ Client (agentic, web, CLI)                          │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│ FastAPI app (pgkg/api.py)                           │
├─────────────────────────────────────────────────────┤
│ Chat memory                                         │
│   POST /memorize  → embed + extract + link          │
│   POST /recall    → embed + retrieve + rerank + MMR │
│   POST /forget    → withdraw a fact, with a reason  │
│   POST /believed  → what was believed as of <t>     │
│                                                     │
│ Corpus                                              │
│   POST /collections      → create, with its policy  │
│   POST /documents        → upsert; queue=true for   │
│                            the batch worker         │
│   POST /documents/delete → soft delete              │
│   GET  /jobs/{id}?org_id= → queue status            │
│                                                     │
│   GET  /health    → liveness, the embedder          │
│                     registry, and whether the       │
│                     keyword index is reachable      │
│                     under row security              │
│                                                     │
│ Every request carries a scope: org, collection,     │
│ user, ACL groups, subscribed collections. Reads     │
│ may widen to the operator's shared org; writes      │
│ never do.                                           │
│                                                     │
│ Models loaded once at startup (lazy singletons):    │
│ • SentenceTransformer (embeddings)                  │
│ • CrossEncoder (reranking)                          │
│ • LLM client (OpenAI / Anthropic / Ollama / claude) │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼ (asyncpg, one pooled connection per request,
                  │  pgkg.org_id set on every one of them)
┌─────────────────────────────────────────────────────┐
│ PostgreSQL (pgkg container, pgvector + tsvector)    │
├─────────────────────────────────────────────────────┤
│ Tenancy        orgs  users  collections             │
│                collection_subscriptions             │
│                tenant_shards                        │
│ Facts          propositions  entities  edges        │
│                entity_pagerank                      │
│ Corpus         documents  document_versions         │
│                document_version_chunks  chunks      │
│ Graph bridge   entity_mentions  entity_links        │
│ Time           provenance  proposition_provenance   │
│ Ranking        corpus_stats  lexeme_df              │
│ Embedders      embedder_generations  org_embedders  │
│ Operational    ingest_jobs  embedding_cache         │
│                proposition_cache                    │
│                                                     │
│ Row-level security on every table with an org       │
│ column; a test enumerates them and fails on a new   │
│ one that arrives without a policy.                  │
└─────────────────────────────────────────────────────┘
```

### The SQL surface

Everything below is a function a caller or an operator invokes. Each migration's header explains
why it exists; those headers are the primary documentation for the schema.

**Retrieval**

| Function | What it does |
|---|---|
| `pgkg_retrieve()` | The two-store surface: both arms, fusion, quotas, decay, context windows |
| `pgkg_search()` | The proposition-shaped surface, for callers that want exactly that |
| `pgkg_bm25_candidates()` | Keyword arm, over either store, against materialised statistics |
| `pgkg_vector_candidates()` | Vector arm, restricted to the org's primary embedding space |
| `pgkg_generation_candidates()` | The second live generation's arm during a cutover window |
| `pgkg_graph_candidates()` | One-hop expansion, in both directions, re-filtered per hop |
| `pgkg_fuse()` | Weighted reciprocal rank fusion |
| `pgkg_item_scope()` / `pgkg_apply_quotas()` | The corpus ceiling and the personal-memory floor |
| `pgkg_apply_profile()` | Recency, frequency and the three decay profiles |
| `pgkg_chunk_window()` | Small-to-big context, scoped to what the anchor itself carries |
| `pgkg_visible()` / `pgkg_temporal_visible()` | The one shared read predicate, and its bitemporal half |

**Corpus lifecycle**

| Function | What it does |
|---|---|
| `pgkg_open_document_version()` | Opens the next version, or says the content has not moved |
| `pgkg_add_version_chunk()` | Content-addressed insert; returns `is_new`, which is what says "embed this one" |
| `pgkg_promote_document_version()` | The atomic flip: retire, promote, withdraw orphaned facts |
| `pgkg_purge_retired_versions()` / `pgkg_gc_chunks()` | Reclamation, in two grace-period passes |
| `pgkg_refresh_chunk_stats()` / `pgkg_refresh_corpus_stats()` | Full rebuild and drift repair for the ranking statistics |
| `pgkg_enqueue_ingest_job()` / `pgkg_claim_ingest_job()` | The batch queue |

**Graph and time**

| Function | What it does |
|---|---|
| `pgkg_link_entity()` | Idempotent entity resolution |
| `pgkg_match_entity_mentions()` | Gazetteer matching over new passages |
| `pgkg_believed_at()` | What was believed as of a past instant |
| `pgkg_contradict()` / `pgkg_expire_due()` | Withdrawal by contradiction and by expiry |
| `pgkg_erase_provenance()` | Erasure as an indexed query |
| `pgkg_recompute_pagerank()` | Offline centrality, per org |

**Tenancy**

| Function | What it does |
|---|---|
| `pgkg_current_org()` | Reads the `pgkg.org_id` GUC; every policy is written against it |
| `pgkg_subscribed_orgs()` / `pgkg_subscribed_collections()` | Subscription resolution |
| `pgkg_live_generations()` | Which embedding spaces a query must be embedded into |

## The hero query: RRF + graph expansion

`pgkg_retrieve()` orchestrates keyword retrieval, vector retrieval, RRF fusion, graph expansion, quotas, decay and context windows — over both stores. Each stage is its own set-returning function over a `pgkg_candidate` row type, so an arm can be tested, replaced or weighted on its own; `pgkg_retrieve()` is the composition (see `migrations/010_search_decompose.sql` for the decomposition and `031_corpus_retrieval.sql` for the two-store version). `pgkg_search()` is the same shape restricted to facts, kept for callers that want a proposition-shaped answer.

Simplified to one store and one arm pair, the RRF and fusion logic reads:

```sql
-- 1. Keyword retrieval (tsvector + ts_rank_cd)
WITH kw AS (
    SELECT p.id AS prop_id, ROW_NUMBER() OVER (...) AS rank
    FROM propositions p
    WHERE p.tsv @@ plainto_tsquery('english', q_text)
      AND p.namespace = p_namespace
      AND p.invalidated_at IS NULL
      AND pgkg_visible(p.org_id, p.collection_id, ...)
    ORDER BY ts_rank_cd(p.tsv, ...) DESC
    LIMIT k_initial
),

-- 2. Vector retrieval (HNSW + distance operator)
vec AS (
    SELECT p.id AS prop_id, ROW_NUMBER() OVER (...) AS rank
    FROM propositions p
    WHERE p.embedding <=> q_embedding IS NOT NULL
      AND p.namespace = p_namespace
      AND p.invalidated_at IS NULL
      AND pgkg_visible(p.org_id, p.collection_id, ...)
    ORDER BY p.embedding <=> q_embedding
    LIMIT k_initial
),

-- 3. RRF fusion
fused AS (
    SELECT
        COALESCE(kw.prop_id, vec.prop_id) AS prop_id,
        COALESCE(1.0 / (60 + kw.rank), 0.0) +
        COALESCE(1.0 / (60 + vec.rank), 0.0) AS rrf_score
    FROM kw FULL OUTER JOIN vec USING (prop_id)
),

-- 4. Seed entities from top fused propositions
seed_entities AS (
    SELECT entity_id FROM (
        SELECT p.subject_id AS entity_id, MAX(f.rrf_score)
        FROM fused f JOIN propositions p ON p.id = f.prop_id
        WHERE p.subject_id IS NOT NULL
        GROUP BY entity_id
        ORDER BY MAX DESC
        LIMIT 20
    )
),

-- 5. Graph neighbors (one hop from seeds)
neighbor_props AS (
    SELECT np.id AS prop_id, 0.5 * MIN(rrf_score) AS rrf_score
    FROM edges e
    JOIN propositions np ON np.id = e.proposition_id
    WHERE e.src_entity IN (SELECT entity_id FROM seed_entities)
       OR e.dst_entity IN (SELECT entity_id FROM seed_entities)
    GROUP BY np.id
)

-- Unified candidates, scored with recency/frequency decay
SELECT ... FROM fused UNION ALL SELECT ... FROM neighbor_props
ORDER BY adjusted_score DESC
LIMIT k_retrieve;
```

Recency and frequency boost the score:

```
adjusted_score = rrf_score
               * exp(log_decay * days_since_access / half_life)
               * log(1 + access_count)
               * confidence
```

The real keyword arm scores with BM25 against materialised statistics rather than `ts_rank_cd`, the
real fusion is weighted and spans four arms over two stores, and every arm carries `pgkg_visible()`
so that a scoped query prunes before it ranks. See `migrations/031_corpus_retrieval.sql` and
`041_retrieval_statistics.sql` for the current functions.

## Quickstart

1. **Copy environment:**
   ```bash
   cp .env.example .env
   ```

2. **Bring up the stack** (Postgres + FastAPI app):
   ```bash
   make up
   ```
   This builds the Docker image, starts the services, and runs migrations automatically.

3. **Run smoke tests** (health + memorize + recall):
   ```bash
   make smoke
   ```

### Example: Memorize and recall

**Memorize** a memory:
```bash
curl -X POST http://localhost:8000/memorize \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "pgkg is a Postgres-native knowledge graph for agentic memory."
  }'
```

Returns:
```json
{
  "documents": 1,
  "chunks": 1,
  "propositions": 3,
  "entities": 5
}
```

**Recall** relevant memories:
```bash
curl -X POST http://localhost:8000/recall \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is pgkg?",
    "k": 5
  }'
```

Returns:
```json
[
  {
    "proposition_id": "p89b2e11-...",
    "text": "pgkg is a Postgres-native knowledge graph for agentic memory.",
    "score": 0.95,
    "rrf_score": 0.50,
    "source_kind": "both",
    "chunk_id": "c89b2e11-...",
    "subject": "pgkg",
    "predicate": "is",
    "object": "a Postgres-native knowledge graph"
  },
  ...
]
```

## Local experimentation

Two paths depending on whether you want LLM-extracted facts or just chunk-level RAG.

### Zero-LLM: chunks-only mode (fastest path)

No API key, no `claude` CLI, no provider config. Chunks are embedded and stored directly; you still get hybrid retrieval (BM25 + vector + RRF), reranking, MMR, recency decay, and session scoping. You lose entity-level recall and graph-based multi-hop expansion (those need extracted facts), but for a lot of "drop in some files and search" use cases this is plenty.

```bash
cp .env.local-chunks .env
make local-chunks         # spins up db, migrates, serves on host
```

Then ingest something:
```bash
curl -X POST http://localhost:8000/memorize \
  -H 'Content-Type: application/json' \
  -d '{"text":"pgkg is a Postgres-native knowledge graph engine. It supports a chunks-only mode that needs no LLM."}'
curl -X POST http://localhost:8000/recall \
  -H 'Content-Type: application/json' \
  -d '{"query":"can pgkg run without an LLM?","k":5}'
```

### With proposition extraction: Claude Pro/Max subscription

If you have a Claude Pro or Max subscription, you can drive extraction through the `claude` CLI — no OpenAI/Anthropic API key needed. **Local development only**: rate limits and ToS make it unsuitable for benchmark runs.

Prereqs: `claude` CLI installed and logged in (run `claude` once and complete the browser flow).

```bash
uv sync --extra claude_agent
cp .env.local-claude .env
make local-claude         # spins up db, migrates, serves on host
```

The app must run on the host (not in the Docker `app` container) because the SDK shells out to your local `claude` binary. The `db` container is fine to use as normal.

### With proposition extraction: paid API

For benchmark runs or if you don't have a Claude subscription, use OpenAI / Anthropic / Ollama / OpenRouter. See [Configuration](#configuration) and the `.env.bench-*` presets. Budget ~$50-100 for a full LongMemEval-S + LoCoMo bench pass on the Mem0 stack.

## Two stores, and the two ways in

There are two *stores* — passages and extracted facts — and two *ingest paths* that fill them.
These are different axes, and the older "two modes" framing conflated them.

**The stores.** `chunks` holds passages; `propositions` holds atomic facts. `pgkg_retrieve()` runs
a keyword arm and a vector arm over each, fuses all of it with RRF, and applies quotas so neither
store can crowd the other out. A result row says which store it came from (`source`), and a
passage arrives with its neighbouring passages as `context_text`. Retrieving passages is not a
degraded mode — it is one of the two stores the retriever is built around.

**Chat ingest** (`POST /memorize`, `Memory.ingest`) is for conversation. It chunks the turn, and by
default sends each chunk to an LLM extractor, links the entities it names, and writes the edges that
make graph expansion possible. Its chunks are written as *provenance* — a fact can cite the text it
came from — and are deliberately not retrievable in their own right, because returning both would
return the same content twice and would keep returning the passage after the fact had been
forgotten.

Set `PGKG_EXTRACT_PROPOSITIONS=0` (or `pgkg ingest --chunks-only`) and the extractor is skipped: no
API key, no `claude` CLI, no LLM cost. The turn goes into the **chunk store** instead — a versioned
document whose passages are content-addressed, embedded and `tsvector`'d, retrievable in their own
right and carrying no proposition rows at all. You still get hybrid retrieval, reranking, MMR and
recency decay; you lose entity-level recall and multi-hop expansion, which need extracted facts.
This is the fastest way to try the system, and for a lot of "drop in some files and search" cases
it is enough.

One thing to know before you scope it: a passage carries no `namespace`. Isolation for the chunk
store is `org_id` and `collection_id`, so two chunks-only writers sharing a collection share a
retrievable pool, however different their namespaces. Give each tenant — or each experiment — its
own collection.

**Corpus ingest** (`POST /documents`, `CorpusIngest.upsert_document`) is for documents. It versions
the document, content-addresses its chunks so an unchanged crawl is free, embeds only what is new,
and promotes the new version atomically. Extraction here is **opt-in per collection**, and off by
default: extracting a whole handbook is a recurring cost, lossy on exactly the content corpora are
made of, and the gazetteer mention edge gets most of the graph benefit for nothing. Turn it on for
the fact-dense minority of collections that earn it — and install the mention sweep below, which is
what makes that edge exist.

The two paths share the retriever, the rerank and MMR pass, the `Result` shape and the tenancy
boundary. Mix them freely: a collection can hold either, and a query can name both stores or one.

## Keeping it fresh: `pgkg maintain`

Four jobs are not on the request path and are not supposed to be. Nothing runs them unless you
schedule them, and one of them — the gazetteer mention sweep — is what joins a passage to the
entities it names, so without it `entity_mentions` stays empty and graph expansion has nothing to
expand through.

```bash
pgkg maintain --org "$ORG"                     # all four jobs, one tenant
pgkg maintain --org "$ORG" --task mentions     # just the sweep; repeatable flag
```

| Task | What it does |
|---|---|
| `mentions` | Gazetteer matching, both directions: passages this org has never matched, and names it has never matched. Each side has its own watermark, so a settled corpus reports no work rather than repeating it. |
| `pagerank` | One PageRank pass per namespace the org has entities in. |
| `contradictions` | Closes the validity interval a supersession left open, at the replacement's own clock, within one claim scope. |
| `expiries` | Withdraws this org's facts whose `valid_to` has passed. |

It prints one JSON report — `{"org": ..., "tasks": [{"task", "ran", "scanned", "changed"}]}` — and
is safe to overlap itself: each task takes an advisory lock per (task, org) and reports `ran: false`
rather than repeating work another run is doing. `ran: false` is a normal outcome, not a failure.

A crontab entry per tenant, with the sweep on a short interval and the rest nightly:

```cron
*/5 *  * * *  pgkg maintain --org ORG --task mentions
30  3  * * *  pgkg maintain --org ORG --task pagerank --task contradictions --task expiries
```

The physical reclamation functions (`pgkg_purge_retired_versions()`, `pgkg_gc_chunks()`,
`pgkg_erase_provenance()`) are deliberately *not* behind this command: they delete rows where these
four withdraw them, and that belongs behind a decision rather than behind the same crontab line.

## Configuration

All settings are environment variables. See `.env.example` and `pgkg/config.py` for defaults.

Every variable takes the `PGKG_` prefix — `PGKG_EMBED_MODEL`, not `EMBED_MODEL`.

| Variable | Default | Description |
|----------|---------|-------------|
| `PGKG_DATABASE_URL` | (unset) | Postgres connection string. **When unset, pgkg starts an embedded Postgres via `pgserver` — no Docker.** Set it to point at an external instance. |
| `PGKG_EMBED_MODEL` | `BAAI/bge-m3` | HuggingFace sentence-transformer model for embeddings |
| `PGKG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | HuggingFace cross-encoder for reranking |
| `PGKG_LLM_PROVIDER` | `openai` | One of `openai`, `anthropic`, `ollama`, `claude_code` (local dev only — requires the `claude` CLI) |
| `PGKG_LLM_MODEL` | `gpt-4o-mini-2024-07-18` | LLM for proposition extraction. Pinned with a dated suffix so benchmark runs are reproducible. |
| `PGKG_EXTRACTOR_MODEL` | (unset) | Overrides `PGKG_LLM_MODEL` for extraction only — the "extract with one model, answer with another" setup |
| `PGKG_JUDGE_MODEL` | `gpt-4o-2024-08-06` | Benchmark judge; pinned to match published LongMemEval/LoCoMo setups |
| `PGKG_JUDGE_PROVIDER` | `openai` | Provider for the judge |
| `PGKG_OPENAI_API_KEY` | (unset) | Required when the provider is `openai`. The plain `OPENAI_API_KEY` also works, via the SDK's own fallback. |
| `PGKG_ANTHROPIC_API_KEY` | (unset) | Required when the provider is `anthropic`; `ANTHROPIC_API_KEY` also works |
| `PGKG_OPENAI_BASE_URL` | (unset) | Point at OpenRouter, Groq, or any OpenAI-compatible endpoint |
| `PGKG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `PGKG_OFFLINE_EXTRACT` | `0` | Set to `1` to skip every LLM call and use deterministic dummy extraction. What the test suite runs with. |
| `PGKG_EXTRACT_PROPOSITIONS` | `true` | Set to `0` to skip LLM extraction on the chat path. Zero LLM cost at ingest. See [Two stores](#two-stores-and-the-two-ways-in). Corpus extraction is a per-collection property, not this flag. |
| `PGKG_DEFAULT_NAMESPACE` | `default` | Default namespace for memories |
| `PGKG_PROMPT_VERSION` | `v1` | Informational; logged into the benchmark report. The source of truth is `PROMPT_VERSION` in `ml.py`. |

The embedding width is deliberately **not** configurable: it is a property of the schema, read with
`pgkg_embedding_dim('propositions', 'embedding')` and owned by the embedder registry. A settings
field could only disagree with the column it described.

Scope — org, collection, user, ACL groups — is per request, not per deployment. It arrives on the
HTTP body or on a `Memory`/`CorpusIngest` instance, and the reserved default org and collection are
what an unscoped caller lands in.

The ACL group is the one scoping column the *writer* has to state. A row with no group passes the
read predicate for every caller of the tenant, so a collection created with `acl_mode: "group"`
refuses content that names none: pass `acl_group_id` on `POST /documents` or to
`CorpusIngest.upsert_document`. Nothing populates it automatically — mirroring a SharePoint or
Drive group tree is not built — so today it is the connector's to supply, and an ACL-bounded
collection fails closed rather than publishing untagged content.

### Development mode

The suite runs against a real Postgres — every assertion goes through SQL, because that is where the
behaviour lives. It picks a backend itself: an embedded `pgserver` if the bundled build has
`pg_trgm` and `pgcrypto`, otherwise a `pgvector/pgvector:pg16` testcontainer. Force one with
`PGKG_TEST_BACKEND=embedded|docker`.

```bash
PGKG_OFFLINE_EXTRACT=1 uv run --python 3.12 pytest -q
```

`PGKG_OFFLINE_EXTRACT=1` replaces every LLM call with deterministic dummy extraction, so no keys
are needed. Note that the suite connects as the container's owning superuser, for whom every RLS
policy is inert: an isolation test has to `SET LOCAL ROLE pgkg_app` **inside a transaction** to
exercise the policy rather than just the SQL predicate.

There is also an end-to-end check that drives a real `pgkg mcp` over stdio against a throwaway
Postgres:

```bash
PGKG_LLM_PROVIDER=claude_code PGKG_OFFLINE_EXTRACT=0 ./scripts/e2e_mcp.sh
```

Name every extra you want when syncing the venv — `uv sync` removes what you leave out, and
dropping `claude_agent` makes the run above fail as a provider error rather than a missing package:

```bash
uv sync --python 3.12 --extra dev --extra mcp --extra claude_agent
```

## Benchmarks

We benchmark pgkg against [LoCoMo](https://github.com/snap-research/locomo) and [LongMemEval](https://github.com/xiaowu0162/LongMemEval) using the same model stack as Mem0's published results so numbers are directly comparable.

### Methodology

| Role | Model | Notes |
|---|---|---|
| Embedder | BAAI/bge-m3 (1024-d) | Open-source, runs in-process |
| Reranker | BAAI/bge-reranker-v2-m3 | Open-source, runs in-process |
| Extractor | gpt-4o-mini-2024-07-18 | Pinned; matches Mem0 |
| Answerer | gpt-4o-mini-2024-07-18 | Pinned; matches Mem0 |
| Judge | gpt-4o-2024-08-06 | Pinned; standard for LongMemEval/LoCoMo papers |

Model IDs are pinned with dated suffixes so results are reproducible across OpenAI API updates.

### Reproducibility

Each benchmark run emits a `bench/results/{name}-{timestamp}-report.json` with the full stack snapshot:

| Parameter | Default | Description |
|---|---|---|
| `k` | 20 | Final propositions returned to the answerer |
| `k_retrieve` | 100 | Candidates retrieved before reranking/MMR |
| `rrf_k` | 60 | RRF smoothing constant |
| `recency_half_life_days` | 30 | Exponential recency decay |
| `expand_graph` | true | One-hop graph neighbor expansion |
| `with_rerank` | true | Cross-encoder reranking pass |
| `with_mmr` | true | Maximal Marginal Relevance deduplication |

Proposition extraction is deterministic w.r.t. model + `prompt_version`, cached in `proposition_cache`. Re-runs on the same dataset hit the cache and incur no additional LLM cost.

### Results

_TBD_ — run `make bench-mem0-stack` (propositions) or `make bench-mem0-stack-chunks` (chunks ablation) when you have an OpenAI key.

The two arms are what the ablation measures: the propositions arm answers out of `propositions`,
the chunks arm out of `chunks`. Each benchmark item gets a collection of its own (`bench:{namespace}`)
in both arms, because a namespace isolates propositions and nothing else — a passage carries no
namespace, so without it every conversation's passages would be candidates for every other
conversation's question. Wipe between arms (`make wipe`): the two runs share the item's namespace
and collection, and the second arm would otherwise retrieve the first arm's rows.

| Benchmark | pgkg (propositions) | pgkg (chunks) | Mem0 | Zep | MemGPT | Stack used |
|---|---|---|---|---|---|---|
| LoCoMo (overall) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | mem0-stack |
| LongMemEval-S (overall) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | mem0-stack |

**References:**
- Mem0 published numbers: https://docs.mem0.ai/research
- Zep GraphRAG paper: https://arxiv.org/abs/2501.13956
- LongMemEval paper: https://arxiv.org/abs/2410.10813
- LoCoMo paper: https://arxiv.org/abs/2402.17029

### How to run

```bash
# Default stack (matches Mem0's published config) — spends ~$2-5 on full datasets
make bench-mem0-stack

# Zep-equivalent stack (gpt-4o everywhere) — ~5x more expensive
make bench-zep-stack

# Free-tier via OpenRouter (rate-limited; use --limit 5)
make bench-openrouter-free

# Dry run (no API keys needed — uses testcontainers Postgres + offline extraction)
PGKG_OFFLINE_EXTRACT=1 make bench-smoke
```

Each `bench-*-stack` target will ask for confirmation before spending money (skipped when `CI=1`).

## Why Postgres?

We get scale, durability, and observability for free. The trade-offs are honest:

**Advantages:**
- Everything is durable and ACID by default.
- Query cost is transparent (run EXPLAIN ANALYZE).
- No additional infrastructure (no Kafka, Pinecone, Neo4j, Chroma licenses or operational burden).
- Full-text search (tsvector) and vector search (HNSW via pgvector) in one system.
- Complex queries (CTEs, window functions, graph traversal) are first-class.

**Limitations:**
- **tsvector ≠ true BM25:** It's simpler and doesn't handle phrase queries. For production BM25, consider [ParadeDB's `pg_bm25`](https://www.paradedb.com/docs/search/bm25) as an optional extension.
- **HNSW recall sensitivity:** HNSW is approximate; recall depends on `ef_search` tuning (we use sensible defaults). If you need 100% exact recall, use a brute-force sequential scan (slower).
- **PageRank is offline:** Graph centrality is precomputed; it's not updated live. Run `pgkg_recompute_pagerank()` periodically (e.g., nightly).
- **Recency decay is exponential:** Linear decay is not an option without schema changes.

## Project layout

```
pgkg/
├── migrations/                 # SQL schema and functions, applied in order.
│   │                           # Each file's header explains why it exists;
│   │                           # those headers are the schema documentation.
│   ├── 001–013                 # Base schema, search, PageRank, extract cache,
│   │                           #   asserted_at, BM25 statistics, composable
│   │                           #   candidate SRFs, halfvec, idempotent linking
│   ├── 020–026                 # Tenancy: scoping columns, RLS, provenance,
│   │                           #   bitemporal columns, collections, embedder
│   │                           #   generations, entity and PageRank scope
│   ├── 030–033                 # Corpus: document versions, content-addressed
│   │                           #   chunks, the two-store retriever, the ingest
│   │                           #   queue, the application surface
│   ├── 040                     # entity_mentions and the entity_links bridge
│   └── 041–045                 # The verification fix pass — see
│                               #   docs/adrs/0001-verification-findings.md
│
├── pgkg/                       # Python package
│   ├── config.py               # Settings (pydantic) + the embedder registry readers
│   ├── db.py                   # asyncpg pool, pgvector codec, iterative scan
│   ├── embedded.py             # Embedded Postgres via pgserver (no Docker)
│   ├── chunking.py             # Content-defined chunk boundaries
│   ├── memory.py               # Scope, chat ingest, recall, forget, believed_at
│   ├── corpus.py               # CorpusIngest: versions, chunks, embeddings
│   ├── ingest_jobs.py          # The batch queue and its worker
│   ├── gazetteer.py            # Entity mention matching over new passages
│   ├── ml.py                   # Embeddings, reranking, MMR, extraction
│   ├── api.py                  # FastAPI endpoints
│   ├── maintenance.py         # The scheduled jobs behind `pgkg maintain`
│   └── cli.py                  # migrate, serve, ingest, recall, worker, maintain
│
├── docs/adrs/                  # Architecture decisions
│   ├── 0001-corpus-embeddings-and-knowledge-graph.md   # the design authority
│   ├── 0001-implementation-notes.md                    # what was built, and why it differs
│   └── 0001-verification-findings.md                   # the audit and its outcome
│
├── tests/                      # Integration tests against a real Postgres
│   ├── conftest.py             # Embedded pgserver, or a testcontainer
│   └── test_*.py               # One module per subject
│
├── bench/                      # LoCoMo and LongMemEval harnesses
├── scripts/run_migrations.py   # Apply .sql migrations to the database
├── Dockerfile                  # Multi-stage build (builder + runtime)
├── docker-compose.yml          # Postgres (pgvector) + FastAPI app
├── Makefile                    # Common tasks (up, down, test, smoke, psql)
└── pyproject.toml              # Project metadata and dependencies
```

## Design decisions

The schema is not obvious, and the reasons are written down rather than implied.

- [**ADR-0001 — Corpus embeddings alongside the proposition knowledge graph**](docs/adrs/0001-corpus-embeddings-and-knowledge-graph.md)
  is the design authority: two physical stores, tenancy, provenance, the document lifecycle, vector
  representation, embedder generations, and what deliberately not to build.
- [**Implementation notes**](docs/adrs/0001-implementation-notes.md) — what phases 0–3 actually
  built, every conscious deviation from the ADR with its reason, and what is deferred to phase 4.
  Read this before changing the schema.
- [**Verification findings**](docs/adrs/0001-verification-findings.md) — 26 defects found by three
  independent adversarial audits of phases 0–3, each with its reproduction and the commit that
  closed it, plus a fourth pass that re-audited the fixes.

## Status

**Alpha, research-grade.** This is a working proof-of-concept. The schema and SQL functions are the
stable part — they are heavily tested and adversarially audited — and the Python APIs may still
change. Phases 0–3 of ADR-0001 are complete; phase 4 (partitioning, binary quantization,
consolidation jobs, the predicate vocabulary, summaries) is not started, and the seams for it are
listed in the [implementation notes](docs/adrs/0001-implementation-notes.md#4-deferred-to-phase-4).

## License

MIT

---

**To get started:** Copy `.env.example` to `.env`, run `make up`, then `make smoke`.
