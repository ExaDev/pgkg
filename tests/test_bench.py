"""Tests for the bench harness."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import asyncpg
import pytest

from bench.common import BenchConfig, BenchItem, QA, exact_match_grade, StackInfo


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# test_bench_item_validates
# ---------------------------------------------------------------------------

def test_bench_item_validates():
    """BenchItem and QA models round-trip through Pydantic."""
    qa = QA(id="q1", question="What is X?", answer="X is Y", category="factual")
    item = BenchItem(
        id="item-1",
        namespace="test-ns",
        session_id=None,
        conversation=[{"speaker": "Alice", "text": "Hello"}],
        questions=[qa],
    )
    dumped = item.model_dump()
    restored = BenchItem(**dumped)
    assert restored.id == "item-1"
    assert restored.questions[0].answer == "X is Y"
    assert restored.conversation[0]["speaker"] == "Alice"


# ---------------------------------------------------------------------------
# test_run_bench_dry_run_exact_match
# ---------------------------------------------------------------------------

async def test_run_bench_dry_run_exact_match(pool: asyncpg.Pool, monkeypatch):
    """run_bench with dry_run=True, exact_match=True completes without LLM calls."""
    monkeypatch.setenv("PGKG_OFFLINE_EXTRACT", "1")

    import pgkg.ml as ml_module

    def _fake_embed(texts):
        result = []
        for t in texts:
            v = [0.0] * 1024
            v[hash(t) % 1024] = 1.0
            result.append(v)
        return result

    monkeypatch.setattr(ml_module, "embed", _fake_embed)

    uid = uuid.uuid4().hex[:8]
    items = [
        BenchItem(
            id=f"test-{uid}-0",
            namespace=f"bench-dryrun-{uid}-0",
            session_id=None,
            conversation=[
                {"speaker": "Alice", "text": "The sky is blue.", "session_id": "s0"},
            ],
            questions=[
                QA(id="q0", question="What color is the sky?", answer="blue", category="factual"),
            ],
        ),
        BenchItem(
            id=f"test-{uid}-1",
            namespace=f"bench-dryrun-{uid}-1",
            session_id=None,
            conversation=[
                {"speaker": "Bob", "text": "Water boils at 100 degrees Celsius.", "session_id": "s0"},
            ],
            questions=[
                QA(id="q1", question="At what temperature does water boil?", answer="100 degrees Celsius", category="factual"),
            ],
        ),
    ]

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config = BenchConfig(
            dry_run=True,
            exact_match=True,
            with_rerank=False,
            with_mmr=False,
            expand_graph=False,
            output_path=Path(tmpdir),
        )

        from bench.common import run_bench
        report = await run_bench(
            name="test-dryrun",
            items=items,
            config=config,
            pool=pool,
        )

    assert report.total == 2
    assert report.accuracy >= 0.0
    assert report.exact_match_accuracy >= 0.0
    assert report.n_propositions_ingested >= 0
    # Stack info should be populated
    assert report.stack is not None
    assert report.stack.extractor_model
    assert report.stack.judge_model
    assert report.stack.pgkg_git_sha  # may be "unknown" in CI, but must be present


# ---------------------------------------------------------------------------
# test_bench_config_factory_resolves_stack
# ---------------------------------------------------------------------------

def test_bench_config_factory_resolves_stack():
    """BenchConfig.with_resolved_stack populates all stack fields."""
    config = BenchConfig.with_resolved_stack()
    stack = config.resolve_stack()
    assert isinstance(stack, StackInfo)
    assert stack.extractor_model
    assert stack.answerer_model
    assert stack.judge_model
    assert stack.embed_model
    assert stack.rerank_model
    assert stack.prompt_version
    assert stack.pgkg_git_sha  # "unknown" if not in a git repo, but must be a string


# ---------------------------------------------------------------------------
# test_load_locomo_parses_one_conversation
# ---------------------------------------------------------------------------

def test_load_locomo_parses_one_conversation():
    """load_locomo parses the mini fixture into BenchItem(s) with turns and questions."""
    from bench.locomo import load_locomo

    fixture = FIXTURES_DIR / "locomo_mini.json"
    items = load_locomo(dataset_path=fixture)

    assert len(items) >= 1
    item = items[0]
    assert item.id.startswith("locomo-")
    assert len(item.conversation) > 0, "Should have conversation turns"
    assert len(item.questions) > 0, "Should have at least one QA"
    assert item.questions[0].question
    assert item.questions[0].answer


# ---------------------------------------------------------------------------
# test_load_longmemeval_parses_one_record
# ---------------------------------------------------------------------------

def test_load_longmemeval_parses_one_record():
    """load_longmemeval parses the mini fixture into BenchItem(s) with turns and questions."""
    from bench.longmemeval import load_longmemeval

    fixture = FIXTURES_DIR / "longmemeval_mini.json"
    items = load_longmemeval(dataset_path=fixture)

    assert len(items) >= 1
    item = items[0]
    assert len(item.conversation) > 0, "Should have conversation turns"
    assert len(item.questions) == 1
    assert item.questions[0].category == "single-session-user"
    assert "hiking" in item.questions[0].answer.lower()


# ---------------------------------------------------------------------------
# test_exact_match_grade
# ---------------------------------------------------------------------------

def test_exact_match_grade():
    """exact_match_grade: substring matching, case-insensitive."""
    assert exact_match_grade(gold_answer="The Rockies", predicted="She went to the Rockies last summer")
    assert exact_match_grade(gold_answer="blue", predicted="The sky is blue.")
    assert not exact_match_grade(gold_answer="Paris", predicted="London is the capital")
