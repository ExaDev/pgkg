# Changelog

## 0.2.0 (2026-04-28)

- **Proposition extraction cache**: Added `proposition_cache` table (migration 005) and `PostgresExtractCache` implementation. Re-ingesting the same chunk with the same extractor model and prompt version hits the cache instead of calling the LLM. Cache is bypassed in offline-extract mode.
- **Pinned model IDs**: Default `llm_model` changed to `gpt-4o-mini-2024-07-18`; added `judge_model = gpt-4o-2024-08-06`, `extractor_model` override, and `openai_base_url` for OpenRouter/Groq compatibility.
- **Stack presets**: Added `.env.bench-mem0-stack`, `.env.bench-zep-stack`, `.env.bench-openrouter-free` and corresponding `make bench-*-stack` targets with cost warnings. `BenchReport` now includes a full `StackInfo` snapshot (models, git SHA, retrieval parameters) for reproducibility.

## 0.1.0

- Initial release: hybrid retrieval (BM25 + HNSW + RRF), graph expansion, recency/frequency decay, MMR, cross-encoder reranking, FastAPI endpoints, LoCoMo and LongMemEval bench harnesses.
