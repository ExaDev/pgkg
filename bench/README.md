# pgkg Benchmarks

Harnesses for LoCoMo and LongMemEval, proving pgkg competes with complex agent-memory stacks.

## Datasets

Both datasets are downloaded automatically on first run and cached under `bench/data/` (not committed).

**LoCoMo:** 10 long multi-session conversations with multi-hop QA (~200 questions each).
Source: https://github.com/snap-research/locomo

**LongMemEval:** Single-question memory eval over long conversation histories.
Variants: `longmemeval_s` (small), `longmemeval_m` (medium), `longmemeval_oracle`.
Source: https://huggingface.co/datasets/xiaowu0162/longmemeval

## Running

Dry-run (no LLM calls, no API keys needed):

```bash
PGKG_OFFLINE_EXTRACT=1 uv run python -m bench.locomo --dry-run --limit 1
PGKG_OFFLINE_EXTRACT=1 uv run python -m bench.longmemeval --dry-run --limit 1
```

Full smoke test (both harnesses):

```bash
make bench-smoke
```

Real runs (requires `PGKG_OPENAI_API_KEY` and a running Postgres):

```bash
uv run python -m bench.locomo --limit 2
uv run python -m bench.longmemeval --variant longmemeval_s --limit 5
```

Or via Make:

```bash
make bench-locomo
make bench-longmemeval
```

## CLI Options

Common flags for both harnesses:

| Flag | Default | Description |
|------|---------|-------------|
| `--limit N` | all | Cap number of conversations/records |
| `--dry-run` | off | Skip LLM answer generation (use first retrieved text) |
| `--exact-match` | off | Report exact-match metric alongside LLM-judge accuracy |
| `--judge-model` | gpt-4o-mini | Model for answer generation and grading |
| `--judge-provider` | openai | `openai`, `anthropic`, or `ollama` |
| `--no-rerank` | off | Disable cross-encoder reranking |
| `--no-mmr` | off | Disable MMR diversity |
| `--no-graph-expansion` | off | Disable graph neighbor expansion |
| `--concurrency N` | 4 | Parallel items |
| `--dataset-path PATH` | auto | Override dataset file location |

## Results

Results land in `bench/results/`:

- `{name}-{timestamp}.jsonl` — one line per question: `{item_id, qa_id, question, gold, predicted, correct, exact_match, category, recall_ms, answer_ms, n_retrieved}`
- `{name}-{timestamp}-report.json` — summary with per-category accuracy, latency stats, and config

## Caveats

**Judge LLM noise:** LLM-as-judge introduces non-determinism. The `--exact-match` flag provides a deterministic secondary metric (normalized substring match).

**No fine-tuning:** All results use pgkg out-of-box settings — no domain-specific tuning.

**offline_extract mode:** With `PGKG_OFFLINE_EXTRACT=1`, proposition extraction is replaced with a dummy that uses the raw text, avoiding LLM calls. This is useful for harness validation but underperforms real extraction.

**What a real run looks like:** Set `PGKG_OPENAI_API_KEY`, ensure Postgres is running (`make up`), and run without `--dry-run`. Full LoCoMo takes ~30–90 minutes depending on OpenAI latency.
