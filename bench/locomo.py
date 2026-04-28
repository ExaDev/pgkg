"""LoCoMo benchmark harness.

Dataset: snap-research/locomo — 10 long multi-session conversations with multi-hop QA.
Source: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

Download happens automatically on first run; cached under bench/data/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CACHE_DIR = Path(__file__).parent / "data"
CACHE_FILE = CACHE_DIR / "locomo10.json"

from bench.common import BenchConfig, BenchItem, QA


def _download_locomo(dataset_path: Path | None) -> Path:
    """Return path to locomo10.json, downloading if needed."""
    if dataset_path is not None:
        return dataset_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_FILE.exists():
        print(f"Downloading LoCoMo dataset from {LOCOMO_URL} ...")
        urlretrieve(LOCOMO_URL, CACHE_FILE)
        print(f"Saved to {CACHE_FILE}")
    return CACHE_FILE


def _parse_turns(session: list | dict) -> list[dict]:
    """Parse a session (list of dialogue turns) into normalized turn dicts."""
    turns = []
    if isinstance(session, list):
        items = session
    elif isinstance(session, dict):
        items = session.get("conversation", [])
    else:
        return turns

    for turn in items:
        if isinstance(turn, dict):
            speaker = turn.get("speaker") or turn.get("role") or "speaker"
            text = turn.get("text") or turn.get("content") or ""
            if text:
                t: dict = {"speaker": str(speaker), "text": str(text)}
                # LoCoMo has per-turn timestamps on some utterances
                timestamp = turn.get("timestamp") or turn.get("date") or turn.get("time")
                if timestamp is not None:
                    t["timestamp"] = timestamp
                turns.append(t)
    return turns


def load_locomo(*, dataset_path: Path | None = None) -> list[BenchItem]:
    """Load LoCoMo dataset and return list of BenchItems.

    Each BenchItem represents one conversation with its QA pairs.
    Sessions within a conversation are ingested under separate session_ids.

    Dataset URL (as of 2024):
      https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
    If the URL has moved, check the GitHub API:
      https://api.github.com/repos/snap-research/locomo/contents/data
    """
    path = _download_locomo(dataset_path)
    raw = json.loads(path.read_text())

    # Dataset may be a list or a dict with a key
    if isinstance(raw, dict):
        # Try common keys
        conversations = (
            raw.get("conversations")
            or raw.get("data")
            or list(raw.values())
        )
        if isinstance(conversations, dict):
            conversations = list(conversations.values())
    else:
        conversations = raw

    items: list[BenchItem] = []
    for idx, conv in enumerate(conversations):
        if not isinstance(conv, dict):
            continue

        namespace = f"locomo-{idx}"

        # Collect all turns across sessions
        all_turns: list[dict] = []

        # Try various field names for sessions
        sessions_data = (
            conv.get("sessions")
            or conv.get("conversation")
            or conv.get("dialogues")
            or []
        )

        if isinstance(sessions_data, list):
            for sess_idx, session in enumerate(sessions_data):
                sess_turns = _parse_turns(session)
                for turn in sess_turns:
                    turn["session_id"] = f"sess-{sess_idx}"
                all_turns.extend(sess_turns)
        elif isinstance(sessions_data, dict):
            for sess_idx, (sess_key, session) in enumerate(sessions_data.items()):
                sess_turns = _parse_turns(session)
                for turn in sess_turns:
                    turn["session_id"] = f"sess-{sess_idx}"
                all_turns.extend(sess_turns)

        # If no sessions found, try direct conversation field
        if not all_turns:
            direct = _parse_turns(conv)
            for turn in direct:
                turn["session_id"] = "sess-0"
            all_turns.extend(direct)

        # Parse QA pairs
        qa_raw = conv.get("qa") or conv.get("questions") or conv.get("qas") or []
        questions: list[QA] = []
        for qa_idx, qa in enumerate(qa_raw):
            if not isinstance(qa, dict):
                continue
            question = qa.get("question") or qa.get("q") or ""
            answer = qa.get("answer") or qa.get("a") or ""
            category = qa.get("category") or qa.get("type") or "general"
            if question:
                questions.append(QA(
                    id=f"locomo-{idx}-qa-{qa_idx}",
                    question=str(question),
                    answer=str(answer),
                    category=str(category),
                ))

        items.append(BenchItem(
            id=f"locomo-{idx}",
            namespace=namespace,
            session_id=None,
            conversation=all_turns,
            questions=questions,
        ))

    return items


async def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of conversations")
    parser.add_argument("--dry-run", action="store_true", help="Skip judge LLM calls")
    parser.add_argument("--exact-match", action="store_true", help="Use exact-match grading")
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

    items = load_locomo(dataset_path=args.dataset_path)
    if args.limit:
        items = items[: args.limit]

    print(f"Loaded {len(items)} LoCoMo conversations")

    from pgkg.backends.postgres import PostgresBackend
    from pgkg.config import get_settings
    from bench.common import run_bench

    settings = get_settings()
    backend = await PostgresBackend.create(settings.database_url)
    try:
        report = await run_bench(
            name="locomo",
            items=items,
            config=config,
            backend=backend,
        )
    finally:
        await backend.close()

    print(f"\nFinal: {report.accuracy:.1%} accuracy over {report.total} questions")


if __name__ == "__main__":
    asyncio.run(main())
