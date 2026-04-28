# pgkg — Postgres-native knowledge graph engine for agentic memory

**The thesis:** Vanilla Postgres with `pgvector` and `tsvector` can match the retrieval quality of complex agent-memory stacks (Mem0, Zep, MemGPT) and knowledge graph systems that bolt on Kafka, Pinecone, Neo4j, etc. The only non-SQL components are embedding, reranking, and LLM-based proposition extraction — bundled in a single ~310-line file.

```
Python:          878 lines (config, db, ml, api, memory, cli)
SQL:             393 lines (schema, search CTEs, PageRank)
```

**Two containers, one CTE:** Postgres does the work. Everything else is glue.

## What it does

- **Hybrid retrieval:** BM25-ish keyword (tsvector) + dense vector (HNSW) fused via Reciprocal Rank Fusion.
- **Graph expansion:** Retrieved entities seed a one-hop neighbor search; graph neighbors also participate in RRF.
- **Recency and frequency decay:** Propositions get fresher over time and more valuable as they're accessed; exponential half-life (configurable, default 30 days).
- **MMR diversity:** Maximal Marginal Relevance to avoid redundant results.
- **Cross-encoder reranking:** Second-pass relevance scoring on top-k candidates.
- **Session and namespace scoping:** Isolate memories by user session and logical namespace.
- **Fact supersession:** Mark propositions as superseded when new facts replace old ones.
- **Optional PageRank:** Offline (precomputed) link-based importance weighting on the entity graph.

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
│ POST /memorize  → embed + extract propositions      │
│ POST /recall    → embed + search + rerank           │
│ GET  /health    → liveness                          │
│                                                     │
│ Models loaded once at startup (lazy singletons):    │
│ • SentenceTransformer (embeddings)                  │
│ • CrossEncoder (reranking)                          │
│ • LLM client (OpenAI / Anthropic / Ollama)          │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼ (asyncpg)
┌─────────────────────────────────────────────────────┐
│ PostgreSQL (pgkg container, pgvector + tsvector)    │
├─────────────────────────────────────────────────────┤
│ Tables:                                             │
│   documents       chunks     propositions           │
│   entities        edges      entity_pagerank        │
│                                                     │
│ Key functions (everything below the line is SQL):   │
│   pgkg_search()         → 250+ lines RRF + graph   │
│   pgkg_link_entity()    → entity dedup + linking    │
│   pgkg_bump_access()    → update recency counters   │
│   pgkg_recompute_pagerank() → offline scoring       │
└─────────────────────────────────────────────────────┘
```

## The hero query: RRF + graph expansion

The core of pgkg is a single SQL function that orchestrates keyword retrieval, vector retrieval, RRF fusion, and graph expansion. Here's the RRF and fusion logic (full function in `migrations/003_search.sql`):

```sql
-- 1. Keyword retrieval (tsvector + ts_rank_cd)
WITH kw AS (
    SELECT p.id AS prop_id, ROW_NUMBER() OVER (...) AS rank
    FROM propositions p
    WHERE p.tsv @@ plainto_tsquery('english', q_text)
      AND p.namespace = p_namespace
      AND p.superseded_by IS NULL
    ORDER BY ts_rank_cd(p.tsv, ...) DESC
    LIMIT k_initial
),

-- 2. Vector retrieval (HNSW + distance operator)
vec AS (
    SELECT p.id AS prop_id, ROW_NUMBER() OVER (...) AS rank
    FROM propositions p
    WHERE p.embedding <=> q_embedding IS NOT NULL
      AND p.namespace = p_namespace
      AND p.superseded_by IS NULL
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

See `migrations/003_search.sql` for the complete function including all window functions and decay logic.

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

## Two modes: chunks vs propositions

pgkg supports two ingest modes:

**Propositions mode (default)** — chunks are split into atomic facts via an LLM extractor; entities are linked across documents; edges enable graph expansion. Best retrieval quality, especially on multi-hop QA. Costs ~$X per 1M tokens of input.

**Chunks mode** (`--chunks-only` / `PGKG_EXTRACT_PROPOSITIONS=0`) — chunks are embedded and stored directly as propositions with no entity structure. Zero LLM cost at ingest. Equivalent to vanilla hybrid RAG (BM25 + vector + rerank + MMR + recency). Graph expansion is a no-op (no edges).

Both modes use the same retrieval pipeline (`pgkg_search()`), the same rerank+MMR, and the same `Result` shape. You can mix them per-namespace by configuring different `Memory` instances.

The point: chunk RAG is a perfectly fine starting point for a lot of agent-memory use cases (especially in pgkg, since you still get hybrid retrieval and rerank). Add proposition extraction when you need entity-level recall or multi-hop reasoning. The headline benchmark numbers below quantify the gap.

## Configuration

All settings are environment variables. See `.env.example` and `pgkg/config.py` for defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/pgkg` | Postgres connection string |
| `EMBED_MODEL` | `BAAI/bge-m3` | HuggingFace sentence-transformer model for embeddings |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | HuggingFace cross-encoder for reranking |
| `EMBED_DIM` | `1024` | Embedding dimension (must match the chosen model) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM for proposition extraction |
| `LLM_PROVIDER` | `openai` | One of: `openai`, `anthropic`, `ollama`, `claude_code` (local dev only — requires `claude` CLI) |
| `OPENAI_API_KEY` | (unset) | OpenAI API key (required if `LLM_PROVIDER=openai`) |
| `ANTHROPIC_API_KEY` | (unset) | Anthropic API key (required if `LLM_PROVIDER=anthropic`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (used if `LLM_PROVIDER=ollama`) |
| `OFFLINE_EXTRACT` | `0` | Set to `1` to skip LLM calls (test mode); uses dummy extraction |
| `EXTRACT_PROPOSITIONS` | `true` | Set to `0` or `false` to skip LLM extraction entirely; store chunks directly as propositions (NULL subject/predicate/object). Zero LLM cost at ingest. See [Two modes](#two-modes-chunks-vs-propositions). |
| `DEFAULT_NAMESPACE` | `default` | Default namespace for memories |

### Development mode

To run tests and avoid needing LLM keys:

```bash
export PGKG_OFFLINE_EXTRACT=1
uv run pytest -q
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
├── migrations/                # SQL schema and functions
│   ├── 001_extensions.sql     # Enable pgvector, pg_trgm
│   ├── 002_schema.sql         # Tables: documents, chunks, propositions, entities, edges
│   ├── 003_search.sql         # pgkg_search(), pgkg_link_entity(), pgkg_bump_access()
│   └── 004_pagerank.sql       # pgkg_recompute_pagerank()
│
├── pgkg/                       # Python package (~350 lines)
│   ├── config.py              # Settings (pydantic)
│   ├── db.py                  # asyncpg pool + pgvector init
│   ├── ml.py                  # Embeddings, reranking, MMR, proposition extraction
│   └── api.py                 # FastAPI endpoints (created by Phase 2b)
│
├── tests/                      # Integration tests
│   ├── conftest.py            # Fixtures (test Postgres container)
│   └── test_search_sql.py      # SQL function tests
│
├── scripts/                    # Utilities
│   └── run_migrations.py       # Apply .sql migrations to the database
│
├── Dockerfile                  # Multi-stage build (builder + runtime)
├── docker-compose.yml          # Postgres (pgvector) + FastAPI app
├── Makefile                    # Common tasks (up, down, test, smoke, psql)
└── pyproject.toml             # Project metadata and dependencies
```

## Status

**Alpha, research-grade.** This is a working proof-of-concept. The schema and SQL functions are stable; Python APIs may change.

## License

MIT

---

**To get started:** Copy `.env.example` to `.env`, run `make up`, then `make smoke`.
