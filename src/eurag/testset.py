"""Test set: schema, loading, Ragas generation and the human review flow.

Only rows reviewed by a person (origin "reviewed") or written by hand
(origin "handwritten") are evaluated.
"""

import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

from eurag.chunking import Chunk

COLUMNS = ["id", "language", "category", "origin", "user_input", "reference", "reference_contexts",
           "reference_citations", "review_note"]

Category = Literal["single_doc", "multi_doc", "cross_language", "revoked", "cross_supervisor", "out_of_scope"]
Origin = Literal["generated", "reviewed", "rejected", "handwritten"]
EVALUATED = {"reviewed", "handwritten"}


class TestCase(BaseModel):
    __test__ = False  # not a pytest class

    id: str
    language: Literal["pt", "en"]
    category: Category
    origin: Origin
    user_input: str
    reference: str
    reference_contexts: list[str] = []
    reference_citations: list[str] = []
    review_note: str = ""

    @field_validator("reference_contexts", "reference_citations", mode="before")
    @classmethod
    def _json_list(cls, v):
        return json.loads(v) if isinstance(v, str) else v


def load_testset(path: Path, include_unreviewed: bool = False) -> list[TestCase]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        cases = [TestCase(**row) for row in csv.DictReader(f)]
    return cases if include_unreviewed else [c for c in cases if c.origin in EVALUATED]


def write_testset(cases: list[TestCase], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for c in cases:
            row = c.model_dump()
            row["reference_contexts"] = json.dumps(c.reference_contexts, ensure_ascii=False)
            row["reference_citations"] = json.dumps(c.reference_citations)
            writer.writerow(row)


def citations_for(chunks: list[Chunk]) -> list[str]:
    """Expected citations ("DOC:article") for the chunks a reference was built from."""
    return list(dict.fromkeys(f"{c.doc_id}:{a}" for c in chunks for a in c.covers))


def merge_review(reviewed_path: Path, handwritten_path: Path, out: Path) -> list[TestCase]:
    """Combine the reviewed candidates and the handwritten cases into the evaluated test set."""
    cases = load_testset(reviewed_path, include_unreviewed=True) + load_testset(handwritten_path)
    write_testset(cases, out)
    return [c for c in cases if c.origin in EVALUATED]


TRANSLATE_PROMPT = (
    "Translate this question and its reference answer into European Portuguese (Portugal, not Brazil). "
    "Keep legal terms, article numbers and document names exact. Reply only with JSON with keys "
    '"user_input" and "reference".\n\nQuestion: {q}\n\nReference: {r}'
)


def chunks_for_contexts(contexts: list[str], chunks: list[Chunk]) -> list[Chunk]:
    """Chunks a reference context came from. Ragas may split long chunks, so match by containment."""
    found = []
    for ctx in contexts:
        needle = ctx.strip()
        for c in chunks:
            if needle and (needle in c.text or c.text.strip() in needle) and c not in found:
                found.append(c)
    return found


def extract_json(text: str) -> dict:
    """The first JSON object in a reply, tolerating code fences or a sentence before it."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:80]!r}")
    return json.loads(m.group(0))


def generator_llm(model: str, api_key: str | None = None):
    """Ragas LLM wrapper for test-set generation. bypass_temperature: current Claude models reject it."""
    from langchain_anthropic import ChatAnthropic
    from ragas.llms.base import LangchainLLMWrapper

    kwargs = {"api_key": api_key} if api_key else {}
    return LangchainLLMWrapper(ChatAnthropic(model=model, max_tokens=4096, **kwargs), bypass_temperature=True)


def generate_candidates(chunks: list[Chunk], judge_model: str, embeddings, n: int = 80) -> list[TestCase]:  # pragma: no cover - paid
    """Generate candidate questions with Ragas. Cases whose contexts match no chunk are reported and dropped."""
    from langchain_core.documents import Document
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.testset import TestsetGenerator

    docs = [Document(page_content=c.text, metadata={"chunk_id": c.chunk_id}) for c in chunks if c.covers]
    generator = TestsetGenerator(llm=generator_llm(judge_model), embedding_model=LangchainEmbeddingsWrapper(embeddings))
    frame = generator.generate_with_langchain_docs(docs, testset_size=n).to_pandas()

    cases, unmatched = [], 0
    for i, row in enumerate(frame.itertuples(), 1):
        contexts = list(row.reference_contexts)
        source = chunks_for_contexts(contexts, chunks)
        if not source:
            unmatched += 1
            continue
        cases.append(TestCase(
            id=f"g{i:03d}", language=source[0].language,
            category="multi_doc" if len({c.doc_id for c in source}) > 1 else "single_doc", origin="generated",
            user_input=row.user_input, reference=row.reference, reference_contexts=contexts,
            reference_citations=citations_for(source),
        ))
    if unmatched:
        print(f"warning: {unmatched} generated cases matched no chunk and were dropped", file=sys.stderr)
    return cases


def translate_cases(cases: list[TestCase], model: str, pt_share: float = 0.6,
                    seed: int = 7) -> list[TestCase]:  # pragma: no cover - paid
    """Translate English cases to European Portuguese until pt_share of the set is Portuguese.

    A reply that cannot be parsed leaves that case in English instead of aborting.
    """
    from anthropic import Anthropic

    rng = random.Random(seed)
    english = [c for c in cases if c.language == "en"]
    rng.shuffle(english)
    client = Anthropic()
    out = list(cases)
    while english and sum(c.language == "pt" for c in out) < pt_share * len(out):
        case = english.pop()
        reply = client.messages.create(
            model=model, max_tokens=2048,
            messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(q=case.user_input, r=case.reference)}],
        )
        try:
            translated = extract_json(next(b.text for b in reply.content if b.type == "text"))
        except (ValueError, StopIteration) as exc:
            print(f"warning: could not translate {case.id}: {exc}", file=sys.stderr)
            continue
        out[out.index(case)] = case.model_copy(update={"user_input": translated["user_input"],
                                                       "reference": translated["reference"], "language": "pt",
                                                       "category": "cross_language"})
    return out
