"""Metrics computed without an LLM: retrieval, citations and refusals.

Expected citations are "DOC:article" strings. A retrieved chunk matches when
any article it covers is expected, so a fixed-size chunk that contains the
article counts as well as a per-article chunk.
"""

import unicodedata

from eurag.citations import Citation
from eurag.index import Retrieved
from eurag.testset import TestCase


def _keys(r: Retrieved) -> set[str]:
    return {f"{r.chunk.doc_id}:{a}" for a in r.chunk.covers}


def hit_at_k(expected: list[str], retrieved: list[Retrieved], k: int) -> float | None:
    if not expected:
        return None
    wanted = set(expected)
    return 1.0 if any(_keys(r) & wanted for r in retrieved if r.rank <= k) else 0.0


def reciprocal_rank(expected: list[str], retrieved: list[Retrieved]) -> float | None:
    if not expected:
        return None
    wanted = set(expected)
    for r in sorted(retrieved, key=lambda r: r.rank):
        if _keys(r) & wanted:
            return 1.0 / r.rank
    return 0.0


def citation_recall(expected: list[str], cited: list[Citation]) -> float | None:
    """Share of expected citations that the answer cites."""
    if not expected:
        return None
    got = {c.key() for c in cited}
    return sum(e in got for e in expected) / len(expected)


def citation_precision(cited: list[Citation], retrieved: list[Retrieved]) -> float | None:
    """Share of the answer's citations that point to an excerpt it actually received."""
    if not cited:
        return None
    available = set().union(*(_keys(r) for r in retrieved)) if retrieved else set()
    return sum(c.key() in available for c in cited) / len(cited)


# Phrases that refer to the excerpts or to answering, so a factual negation ("o artigo não contém
# exceções") is not mistaken for a refusal.
_REFUSAL_PHRASES = (
    "nao encontrei", "nao consigo responder", "nao esta nos excertos", "os excertos nao",
    "excertos fornecidos nao", "could not find", "couldn't find", "not in the excerpts", "cannot answer",
    "excerpts do not contain", "excerpts don't contain", "excerpts provided do not",
)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def is_refusal(text: str) -> bool:
    folded = _fold(text)
    return any(_fold(p) in folded for p in _REFUSAL_PHRASES)


def refusal_correct(case: TestCase, text: str) -> float | None:
    if case.category != "out_of_scope":
        return None
    return 1.0 if is_refusal(text) else 0.0
