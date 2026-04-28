"""LongMemEval benchmark harness.

Dataset: xiaowu0162/LongMemEval on HuggingFace.
Variants: longmemeval_s (small), longmemeval_m (medium), longmemeval_oracle.

Download happens automatically via HuggingFace hub or direct HTTP; cached under bench/data/.

HuggingFace dataset page: https://huggingface.co/datasets/xiaowu0162/longmemeval
GitHub: https://github.com/xiaowu0162/LongMemEval
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).parent / "data"
HF_DATASET_ID = "xiaowu0162/longmemeval"
VALID_VARIANTS = ("longmemeval_s", "longmemeval_m", "longmemeval_oracle")

from bench.common import BenchConfig, BenchItem, QA


def _download_via_hf_hub(variant: str) -> Path:
    """Download dataset file using huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        cache_path = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=f"{variant}.json",
            repo_type="dataset",
            local_dir=str(CACHE_DIR),
        )
        return Path(cache_path)
    except Exception as e:
        raise RuntimeError(f"huggingface_hub download failed: {e}") from e


def _download_via_datasets(variant: str) -> Path:
    """Download dataset using HuggingFace datasets library."""
    from datasets import load_dataset  # type: ignore
    ds = load_dataset(HF_DATASET_ID, split="test")
    # Filter by variant if needed
    dest = CACHE_DIR / f"{variant}_from_datasets.json"
    records = list(ds)
    dest.write_text(json.dumps(records))
    return dest


def _download_longmemeval(variant: str, dataset_path: Path | None) -> Path:
    """Return path to dataset JSON, downloading if needed."""
    if dataset_path is not None:
        return dataset_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try cached file first
    cache_file = CACHE_DIR / f"{variant}.json"
    if cache_file.exists():
        return cache_file

    # Try huggingface_hub
    try:
        return _download_via_hf_hub(variant)
    except Exception:
        pass

    # Try datasets library
    try:
        return _download_via_datasets(variant)
    except Exception as e:
        raise RuntimeError(
            f"Could not download {variant}. Install 'huggingface-hub' or 'datasets', "
            f"or provide --dataset-path. Error: {e}"
        ) from e


def _parse_haystack_sessions(record: dict) -> list[dict]:
    """Parse haystack_sessions into normalized turn dicts with session_id tags.

    LongMemEval's small variant uses one timestamp per session (``session_timestamp``
    field on the record or per-session metadata); all turns within a session share
    that timestamp, which is forwarded as ``timestamp`` on each turn dict so that
    ``ingest_conversation`` can pass it to ``memory.ingest`` as ``asserted_at``.
    Per-turn vs per-session granularity differs by dataset; we plumb through
    whatever the source provides without over-engineering.
    """
    turns: list[dict] = []
    sessions = record.get("haystack_sessions") or []
    session_ids = record.get("haystack_session_ids") or []
    # Some LongMemEval variants supply per-session timestamps as a parallel list
    session_timestamps = record.get("session_timestamps") or record.get("haystack_session_timestamps") or []

    for sess_idx, session in enumerate(sessions):
        sid = session_ids[sess_idx] if sess_idx < len(session_ids) else f"sess-{sess_idx}"
        sid = str(sid)
        # Prefer per-session timestamp from parallel list; fallback to record-level field
        session_ts = (
            session_timestamps[sess_idx]
            if sess_idx < len(session_timestamps)
            else record.get("session_timestamp")
        )

        if isinstance(session, list):
            for turn in session:
                if isinstance(turn, dict):
                    speaker = turn.get("role") or turn.get("speaker") or "user"
                    text = turn.get("content") or turn.get("text") or ""
                    if text:
                        t: dict = {
                            "speaker": str(speaker),
                            "text": str(text),
                            "session_id": sid,
                        }
                        # Per-turn timestamp wins; fall back to session-level timestamp
                        ts = turn.get("timestamp") or turn.get("date") or session_ts
                        if ts is not None:
                            t["timestamp"] = ts
                        turns.append(t)
        elif isinstance(session, dict):
            for turn in session.get("messages") or session.get("turns") or []:
                speaker = turn.get("role") or turn.get("speaker") or "user"
                text = turn.get("content") or turn.get("text") or ""
                if text:
                    t = {
                        "speaker": str(speaker),
                        "text": str(text),
                        "session_id": sid,
                    }
                    ts = turn.get("timestamp") or turn.get("date") or session_ts
                    if ts is not None:
                        t["timestamp"] = ts
                    turns.append(t)

    return turns


def load_longmemeval(
    *,
    variant: str = "longmemeval_s",
    dataset_path: Path | None = None,
) -> list[BenchItem]:
    """Load LongMemEval dataset and return list of BenchItems.

    Each record becomes one BenchItem: haystack_sessions are ingested as conversation
    turns (with session_id from haystack_session_ids), and the single question/answer
    pair becomes the QA to evaluate.

    Question types (category field):
      single-session-user, single-session-assistant, multi-session,
      knowledge-update, temporal-reasoning
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant!r}")

    path = _download_longmemeval(variant, dataset_path)
    raw = json.loads(path.read_text())

    # raw may be a list of records or a dict with a 'data' key
    if isinstance(raw, dict):
        records: list[Any] = raw.get("data") or raw.get("rows") or list(raw.values())
        if isinstance(records, dict):
            records = list(records.values())
    else:
        records = raw

    items: list[BenchItem] = []
    for record in records:
        if not isinstance(record, dict):
            continue

        qid = str(record.get("question_id") or record.get("id") or len(items))
        question = str(record.get("question") or "")
        answer = str(record.get("answer") or "")
        q_type = str(record.get("question_type") or "general")

        if not question:
            continue

        namespace = f"lme-{qid}"
        turns = _parse_haystack_sessions(record)

        items.append(BenchItem(
            id=qid,
            namespace=namespace,
            session_id=None,
            conversation=turns,
            questions=[QA(
                id=f"{qid}-q0",
                question=question,
                answer=answer,
                category=q_type,
            )],
        ))

    return items


async def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval benchmark")
    parser.add_argument(
        "--variant",
        choices=list(VALID_VARIANTS),
        default="longmemeval_s",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exact-match", action="store_true")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-provider", default="openai")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-mmr", action="store_true")
    parser.add_argument("--no-graph-expansion", action="store_true")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Skip LLM extraction; store chunks directly as propositions (ablation baseline).",
    )
    args = parser.parse_args()

    config = BenchConfig(
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        dry_run=args.dry_run,
        exact_match=args.exact_match,
        with_rerank=not args.no_rerank,
        with_mmr=not args.no_mmr,
        expand_graph=not args.no_graph_expansion,
        dataset_path=args.dataset_path,
        concurrency=args.concurrency,
        extract_propositions=not args.chunks_only,
    )

    items = load_longmemeval(
        variant=args.variant,
        dataset_path=args.dataset_path,
    )
    if args.limit:
        items = items[: args.limit]

    print(f"Loaded {len(items)} LongMemEval ({args.variant}) records")

    from pgkg.backends.postgres import PostgresBackend
    from pgkg.config import get_settings
    from bench.common import run_bench

    settings = get_settings()
    backend = await PostgresBackend.create(settings.database_url)
    try:
        report = await run_bench(
            name=f"longmemeval-{args.variant}",
            items=items,
            config=config,
            backend=backend,
        )
    finally:
        await backend.close()

    print(f"\nFinal: {report.accuracy:.1%} accuracy over {report.total} questions")


if __name__ == "__main__":
    asyncio.run(main())
