"""Test set: schema, loading, Ragas generation and the human review flow.

Only rows reviewed by a person (origin "reviewed") or written by hand
(origin "handwritten") are evaluated.
"""

import csv
import json
import random
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
    "Keep legal terms, article numbers and document names exact. Reply as JSON with keys "
    '"user_input" and "reference".\n\nQuestion: {q}\n\nReference: {r}'
)


def generate_candidates(chunks: list[Chunk], judge_model: str, embeddings, n: int = 80, pt_share: float = 0.6,
                        seed: int = 7) -> list[TestCase]:  # pragma: no cover - paid, exercised in Task 11
    """Generate candidate questions with Ragas, then translate English ones until pt_share is Portuguese."""
    from anthropic import Anthropic
    from langchain_anthropic import ChatAnthropic
    from langchain_core.documents import Document
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator

    by_text = {c.text: c for c in chunks}
    docs = [Document(page_content=c.text, metadata={"chunk_id": c.chunk_id}) for c in chunks if c.article]
    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(ChatAnthropic(model=judge_model, max_tokens=4096)),
        embedding_model=LangchainEmbeddingsWrapper(embeddings),
    )
    frame = generator.generate_with_langchain_docs(docs, testset_size=n).to_pandas()

    cases = []
    for i, row in enumerate(frame.itertuples(), 1):
        contexts = list(row.reference_contexts)
        source = [by_text[t] for t in contexts if t in by_text]
        language = next((c.language for c in source), "en")
        cases.append(TestCase(
            id=f"g{i:03d}", language=language, category="multi_doc" if len({c.doc_id for c in source}) > 1
            else "single_doc", origin="generated", user_input=row.user_input, reference=row.reference,
            reference_contexts=contexts, reference_citations=citations_for(source),
        ))

    rng = random.Random(seed)
    english = [c for c in cases if c.language == "en"]
    rng.shuffle(english)
    client = Anthropic()
    while english and sum(c.language == "pt" for c in cases) < pt_share * len(cases):
        case = english.pop()
        reply = client.messages.create(
            model=judge_model, max_tokens=2048,
            messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(q=case.user_input, r=case.reference)}],
        )
        translated = json.loads(next(b.text for b in reply.content if b.type == "text"))
        idx = cases.index(case)
        cases[idx] = case.model_copy(update={**translated, "language": "pt", "category": "cross_language"})
    return cases
