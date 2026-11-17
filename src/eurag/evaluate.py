"""Score answers: own metrics (no LLM) plus Ragas judge metrics (Claude Haiku)."""

import asyncio
import math

from eurag.answer import Answer
from eurag.metrics import (
    citation_precision,
    citation_recall,
    hit_at_k,
    reciprocal_rank,
    refusal_correct,
)
from eurag.testset import TestCase

ROW_FIELDS = ["question_id", "config_id", "language", "category", "hit_at_5", "rr", "citation_recall",
              "citation_precision", "refusal_correct", "faithfulness", "factual_correctness", "input_tokens",
              "output_tokens", "seconds", "errors"]


def judge_metrics(judge_llm) -> dict:
    from ragas.metrics.collections import FactualCorrectness, Faithfulness

    return {"faithfulness": Faithfulness(llm=judge_llm), "factual_correctness": FactualCorrectness(llm=judge_llm, mode="f1")}


def default_judge_llm(model: str = "claude-haiku-5-5"):
    from anthropic import AsyncAnthropic
    from ragas.llms import llm_factory

    # Ragas drops temperature/top_p for Anthropic, which current Claude models reject (checked in Ragas 0.4.4).
    return llm_factory(model, provider="anthropic", client=AsyncAnthropic(), max_tokens=4096)


def _metric_kwargs(name: str, answer: Answer, case: TestCase) -> dict:
    if name == "faithfulness":
        return {"user_input": case.user_input, "response": answer.text,
                "retrieved_contexts": [r.chunk.text for r in answer.retrieved]}
    return {"response": answer.text, "reference": case.reference}


async def _score_one(answer: Answer, case: TestCase, metrics: dict, sem: asyncio.Semaphore) -> dict:
    expected = case.reference_citations
    row = {
        "question_id": answer.question_id, "config_id": answer.config_id, "language": case.language,
        "category": case.category,
        "hit_at_5": hit_at_k(expected, answer.retrieved, k=5),
        "rr": reciprocal_rank(expected, answer.retrieved),
        "citation_recall": citation_recall(expected, answer.citations),
        "citation_precision": citation_precision(answer.citations, answer.retrieved),
        "refusal_correct": refusal_correct(case, answer.text),
        "faithfulness": None, "factual_correctness": None,
        "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens, "seconds": answer.seconds,
        "errors": [],
    }
    if case.category != "out_of_scope":
        for name, metric in metrics.items():
            try:
                async with sem:
                    result = await metric.ascore(**_metric_kwargs(name, answer, case))
                value = float(result.value)
                if not math.isfinite(value):  # e.g. faithfulness on an answer with no statements
                    row["errors"].append(f"{name}: non-finite score ({value})")
                    continue
                row[name] = value
            except Exception as exc:  # noqa: BLE001 - one failed judge call must not abort the evaluation
                row["errors"].append(f"{name}: {exc}")
    return row


async def evaluate_answers(answers: list[Answer], cases: dict[str, TestCase], metrics: dict,
                           concurrency: int = 4) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    return list(await asyncio.gather(*(_score_one(a, cases[a.question_id], metrics, sem) for a in answers)))
