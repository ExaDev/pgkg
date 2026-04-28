"""Shared harness for LoCoMo and LongMemEval benchmarks."""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from pgkg.config import get_settings
from pgkg.memory import Memory, Result


# ---------------------------------------------------------------------------
# Config and data models
# ---------------------------------------------------------------------------

class StackInfo(BaseModel):
    """Resolved model/parameter snapshot for reproducibility."""
    extractor_model: str
    answerer_model: str
    judge_model: str
    embed_model: str
    rerank_model: str
    prompt_version: str
    pgkg_git_sha: str
    recency_half_life_days: float = 30.0
    rrf_k: int = 60


class BenchConfig(BaseModel):
    judge_model: str = "gpt-4o-2024-08-06"
    judge_provider: str = "openai"
    # Answerer model (used for RAG answer generation); defaults to judge_model when unset
    answerer_model: str | None = None
    # Extractor model override; when None, resolved from settings.extractor_model or settings.llm_model
    extractor_model: str | None = None
    k: int = 20
    k_retrieve: int = 100
    rrf_k: int = 60
    recency_half_life_days: float = 30.0
    with_rerank: bool = True
    with_mmr: bool = True
    expand_graph: bool = True
    concurrency: int = 4
    dry_run: bool = False
    exact_match: bool = False
    dataset_path: Path | None = None
    output_path: Path = Path("bench/results")
    namespace_prefix: str = "bench"
    dataset_variant: str = "unknown"

    @classmethod
    def with_resolved_stack(cls, settings=None, **kwargs) -> "BenchConfig":
        """Factory that auto-populates model fields from current Settings."""
        if settings is None:
            settings = get_settings()

        extractor = kwargs.pop("extractor_model", None) or settings.extractor_model or settings.llm_model
        answerer = kwargs.pop("answerer_model", None) or settings.llm_model
        judge = kwargs.pop("judge_model", None) or settings.judge_model
        judge_provider = kwargs.pop("judge_provider", None) or settings.judge_provider

        return cls(
            extractor_model=extractor,
            answerer_model=answerer,
            judge_model=judge,
            judge_provider=judge_provider,
            **kwargs,
        )

    def resolve_stack(self) -> StackInfo:
        """Build a StackInfo from this config + current settings."""
        settings = get_settings()

        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            sha = "unknown"

        return StackInfo(
            extractor_model=self.extractor_model or settings.extractor_model or settings.llm_model,
            answerer_model=self.answerer_model or settings.llm_model,
            judge_model=self.judge_model,
            embed_model=settings.embed_model,
            rerank_model=settings.rerank_model,
            prompt_version=settings.prompt_version,
            pgkg_git_sha=sha,
            recency_half_life_days=self.recency_half_life_days,
            rrf_k=self.rrf_k,
        )


class QA(BaseModel):
    id: str
    question: str
    answer: str
    category: str = "general"


class BenchItem(BaseModel):
    id: str
    namespace: str
    session_id: str | None = None
    conversation: list[dict] = Field(default_factory=list)
    questions: list[QA] = Field(default_factory=list)


class CategoryStats(BaseModel):
    correct: int = 0
    total: int = 0
    acc: float = 0.0


class BenchReport(BaseModel):
    name: str
    dataset_variant: str = "unknown"
    total: int
    correct: int
    accuracy: float
    per_category: dict[str, CategoryStats]
    config: BenchConfig
    stack: StackInfo | None = None
    started_at: str
    finished_at: str
    n_propositions_ingested: int
    mean_recall_latency_ms: float
    mean_answer_latency_ms: float
    exact_match_correct: int = 0
    exact_match_accuracy: float = 0.0
    # Retrieval parameters (for reproducibility)
    k: int = 20
    k_retrieve: int = 100
    rrf_k: int = 60
    recency_half_life_days: float = 30.0
    expand_graph: bool = True
    with_rerank: bool = True
    with_mmr: bool = True


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

async def ingest_conversation(
    memory: Memory,
    *,
    namespace: str,
    session_id: str,
    turns: list[dict],
) -> int:
    """Ingest a list of turns into memory. Returns number of turns ingested."""
    count = 0
    for turn in turns:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "").strip()
        if not text:
            continue
        formatted = f"{speaker}: {text}"
        await memory.ingest(formatted, session_id=session_id)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

async def answer_question(
    memory: Memory,
    *,
    question: str,
    namespace: str,
    session_id: str | None,
    k: int,
    k_retrieve: int = 100,
    with_rerank: bool = True,
    with_mmr: bool = True,
    expand_graph: bool = True,
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "openai",
    dry_run: bool = False,
) -> tuple[str, list[Result]]:
    """Recall relevant propositions and generate an answer via LLM."""
    results = await memory.recall(
        question,
        k=k,
        k_retrieve=k_retrieve,
        session_id=session_id,
        with_rerank=with_rerank,
        with_mmr=with_mmr,
        expand_graph=expand_graph,
    )

    if dry_run:
        # In dry_run, return first retrieved text as answer (no LLM)
        if results:
            return results[0].text, results
        return "I don't know", results

    context = "\n".join(f"- {r.text}" for r in results)
    prompt = (
        f"Facts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the facts above. "
        "If the answer is not in the facts, say 'I don't know'."
    )
    answer = _call_llm(prompt, model=judge_model, provider=judge_provider)
    return answer, results


def _call_llm(prompt: str, *, model: str, provider: str) -> str:
    """Call LLM synchronously and return the text response."""
    settings = get_settings()

    if provider == "openai":
        import openai
        client_kwargs: dict = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        client = openai.OpenAI(**client_kwargs)

        request_kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
        }
        if settings.openai_base_url and "openrouter.ai" in settings.openai_base_url:
            request_kwargs["extra_headers"] = {
                "HTTP-Referer": "https://github.com/exadev/pgkg",
                "X-Title": "pgkg-bench",
            }

        response = client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""
    else:
        import httpx
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# LLM Judge grading
# ---------------------------------------------------------------------------

def llm_judge_grade(
    *,
    question: str,
    gold_answer: str,
    predicted: str,
    judge_model: str,
    judge_provider: str = "openai",
) -> dict:
    """Ask an LLM judge whether predicted matches gold answer."""
    prompt = (
        f"Question: {question}\n"
        f"Gold answer: {gold_answer}\n"
        f"Predicted answer: {predicted}\n\n"
        "Is the predicted answer correct? "
        "Respond with JSON only: {\"correct\": true/false, \"reasoning\": \"...\"}. "
        "Be lenient about phrasing; focus on factual equivalence."
    )
    raw = _call_llm(prompt, model=judge_model, provider=judge_provider)
    try:
        # Extract JSON from response
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"correct": False, "reasoning": f"parse error: {raw[:100]}"}


# ---------------------------------------------------------------------------
# Exact-match grading
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def exact_match_grade(*, gold_answer: str, predicted: str) -> bool:
    """Normalized substring match: is gold in predicted or vice versa?"""
    gold = _normalize(gold_answer)
    pred = _normalize(predicted)
    return gold in pred or pred in gold


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_bench(
    *,
    name: str,
    items: Iterable[BenchItem],
    config: BenchConfig,
    pool,
) -> BenchReport:
    """Orchestrate the full benchmark run."""
    from pgkg.memory import Memory

    items_list = list(items)
    started_at = datetime.now(timezone.utc).isoformat()

    # Resolve and print stack summary
    stack = config.resolve_stack()
    print(
        f"Stack: extract={stack.extractor_model} answer={stack.answerer_model} "
        f"judge={stack.judge_model} embed={stack.embed_model} rerank={stack.rerank_model} "
        f"retrieval={{k={config.k}, rrf_k={stack.rrf_k}, "
        f"recency_half_life={stack.recency_half_life_days}d, "
        f"graph={config.expand_graph}, rerank={config.with_rerank}, mmr={config.with_mmr}}}"
    )

    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    results_path = output_dir / f"{name}-{timestamp}.jsonl"

    semaphore = asyncio.Semaphore(config.concurrency)

    total = 0
    correct = 0
    em_correct = 0
    per_category: dict[str, CategoryStats] = {}
    n_propositions = 0
    recall_latencies: list[float] = []
    answer_latencies: list[float] = []

    async def process_item(item: BenchItem) -> None:
        nonlocal total, correct, em_correct, n_propositions

        async with semaphore:
            ns = item.namespace
            memory = Memory(pool, namespace=ns)

            # Ingest all conversation turns grouped by session_id
            sessions: dict[str, list[dict]] = {}
            for turn in item.conversation:
                sid = turn.get("session_id") or item.session_id or "default"
                sessions.setdefault(sid, []).append(turn)

            prop_count = 0
            for sid, turns in sessions.items():
                prop_count += await ingest_conversation(
                    memory,
                    namespace=ns,
                    session_id=sid,
                    turns=turns,
                )
            n_propositions += prop_count

            # Answer each question
            for qa in item.questions:
                t0 = time.monotonic()
                results = await memory.recall(
                    qa.question,
                    k=config.k,
                    k_retrieve=config.k_retrieve,
                    session_id=item.session_id,
                    with_rerank=config.with_rerank,
                    with_mmr=config.with_mmr,
                    expand_graph=config.expand_graph,
                )
                recall_ms = (time.monotonic() - t0) * 1000
                recall_latencies.append(recall_ms)

                t1 = time.monotonic()
                if config.dry_run:
                    predicted = results[0].text if results else "I don't know"
                    grade = {"correct": False, "reasoning": "dry_run"}
                else:
                    context = "\n".join(f"- {r.text}" for r in results)
                    prompt = (
                        f"Facts:\n{context}\n\n"
                        f"Question: {qa.question}\n\n"
                        "Answer the question using only the facts above. "
                        "If the answer is not in the facts, say 'I don't know'."
                    )
                    predicted = _call_llm(
                        prompt,
                        model=config.judge_model,
                        provider=config.judge_provider,
                    )
                    grade = llm_judge_grade(
                        question=qa.question,
                        gold_answer=qa.answer,
                        predicted=predicted,
                        judge_model=config.judge_model,
                        judge_provider=config.judge_provider,
                    )
                answer_ms = (time.monotonic() - t1) * 1000
                answer_latencies.append(answer_ms)

                is_correct = bool(grade.get("correct", False))
                is_em = exact_match_grade(gold_answer=qa.answer, predicted=predicted)

                cat = qa.category
                if cat not in per_category:
                    per_category[cat] = CategoryStats()
                per_category[cat].total += 1
                if is_correct:
                    per_category[cat].correct += 1

                total += 1
                if is_correct:
                    correct += 1
                if is_em:
                    em_correct += 1

                record = {
                    "item_id": item.id,
                    "qa_id": qa.id,
                    "question": qa.question,
                    "gold": qa.answer,
                    "predicted": predicted,
                    "correct": is_correct,
                    "exact_match": is_em,
                    "category": cat,
                    "recall_ms": recall_ms,
                    "answer_ms": answer_ms,
                    "n_retrieved": len(results),
                }

                with open(results_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                status = "CORRECT" if is_correct else "wrong"
                em_status = "EM" if is_em else ""
                print(
                    f"[{total:4d}] {status:7s} {em_status:2s} | "
                    f"{qa.question[:60]!r} → {predicted[:40]!r}"
                )

    await asyncio.gather(*[process_item(item) for item in items_list])

    # Compute per-category accuracy
    for stats in per_category.values():
        stats.acc = stats.correct / stats.total if stats.total else 0.0

    finished_at = datetime.now(timezone.utc).isoformat()
    accuracy = correct / total if total else 0.0
    em_accuracy = em_correct / total if total else 0.0

    report = BenchReport(
        name=name,
        dataset_variant=config.dataset_variant,
        total=total,
        correct=correct,
        accuracy=accuracy,
        per_category=per_category,
        config=config,
        stack=stack,
        started_at=started_at,
        finished_at=finished_at,
        n_propositions_ingested=n_propositions,
        mean_recall_latency_ms=sum(recall_latencies) / len(recall_latencies) if recall_latencies else 0.0,
        mean_answer_latency_ms=sum(answer_latencies) / len(answer_latencies) if answer_latencies else 0.0,
        exact_match_correct=em_correct,
        exact_match_accuracy=em_accuracy,
        k=config.k,
        k_retrieve=config.k_retrieve,
        rrf_k=config.rrf_k,
        recency_half_life_days=config.recency_half_life_days,
        expand_graph=config.expand_graph,
        with_rerank=config.with_rerank,
        with_mmr=config.with_mmr,
    )

    report_path = output_dir / f"{name}-{timestamp}-report.json"
    report_path.write_text(report.model_dump_json(indent=2))
    print(f"\nReport: {report_path}")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})  EM: {em_accuracy:.1%} ({em_correct}/{total})")

    return report
