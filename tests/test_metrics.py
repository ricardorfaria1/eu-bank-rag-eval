from eurag.chunking import Chunk
from eurag.citations import Citation
from eurag.index import Retrieved
from eurag.metrics import (
    citation_precision,
    citation_recall,
    hit_at_k,
    is_refusal,
    reciprocal_rank,
    refusal_correct,
)
from eurag.testset import TestCase


def _r(doc, covers, rank):
    c = Chunk(chunk_id=f"{doc}:{covers[0] if covers else 'pre'}:{rank}", doc_id=doc, article=covers[0] if covers else None,
              covers=tuple(covers), language="en", url="u", text="t")
    return Retrieved(chunk=c, rank=rank, score=1.0)


def test_hit_when_any_part_of_article_retrieved():
    assert hit_at_k(["AMLR:20"], [_r("AMLR", ["19", "20"], 1)], k=5) == 1.0
    assert hit_at_k(["AMLR:20"], [_r("AMLR", ["21"], 1)], k=5) == 0.0


def test_hit_respects_k():
    retrieved = [_r("X", ["1"], 1), _r("AMLR", ["20"], 2)]
    assert hit_at_k(["AMLR:20"], retrieved, k=1) == 0.0


def test_hit_none_for_out_of_scope():
    assert hit_at_k([], [_r("AMLR", ["20"], 1)], k=5) is None


def test_reciprocal_rank_second_position():
    assert reciprocal_rank(["AMLR:20"], [_r("X", ["1"], 1), _r("AMLR", ["20"], 2)]) == 0.5
    assert reciprocal_rank(["AMLR:20"], [_r("X", ["1"], 1)]) == 0.0


def test_citation_recall_and_precision():
    cited = [Citation("AMLR", "20"), Citation("AMLR", "99")]
    assert citation_recall(["AMLR:20", "L83-2017:87"], cited) == 0.5
    assert citation_precision(cited, [_r("AMLR", ["20"], 1)]) == 0.5
    assert citation_precision([], [_r("AMLR", ["20"], 1)]) is None
    assert citation_recall([], cited) is None


def test_is_refusal_pt_en():
    for text in ("Não encontrei essa informação nos excertos.", "Nao consigo responder com base nos excertos.",
                 "The excerpts do not contain this information.", "I could not find the answer in the excerpts."):
        assert is_refusal(text), text
    assert not is_refusal("O prazo é 31 de março [BdP-A3-2024, art. 2].")


def test_refusal_correct_only_out_of_scope():
    oos = TestCase(id="x", language="pt", category="out_of_scope", origin="handwritten", user_input="q", reference="r")
    other = oos.model_copy(update={"category": "single_doc"})
    assert refusal_correct(oos, "Não encontrei essa informação.") == 1.0
    assert refusal_correct(oos, "A taxa é 2%.") == 0.0
    assert refusal_correct(other, "Não encontrei.") is None
