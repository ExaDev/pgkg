# Changelog

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
