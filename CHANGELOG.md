# Changelog

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
