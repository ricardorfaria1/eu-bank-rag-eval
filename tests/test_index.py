from eurag.chunking import Chunk
from eurag.index import Retriever, build_index
from tests.fakes import WordHashSparse, fake_dense


def _chunks():
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    return [Chunk(chunk_id=f"D:{i}:1", doc_id="D", article=str(i), covers=(str(i),), language="en", url="u",
                  text=f"common text about obligations {w} zebra{i}") for i, w in enumerate(words, 1)]


def test_index_dense_roundtrip(tmp_path):
    build_index(_chunks(), "fixed-dense", tmp_path, fake_dense(), None)
    hits = Retriever(tmp_path, "fixed-dense", fake_dense()).search("common text about obligations charlie zebra3", k=2)
    assert hits[0].chunk.chunk_id == "D:3:1" and hits[0].rank == 1
    assert hits[0].chunk.covers == ("3",)


def test_index_hybrid_roundtrip(tmp_path):
    build_index(_chunks(), "article-hybrid", tmp_path, fake_dense(), WordHashSparse())
    hits = Retriever(tmp_path, "article-hybrid", fake_dense(), WordHashSparse()).search("common text about obligations delta zebra4", k=3)
    assert hits[0].chunk.chunk_id == "D:4:1"
    assert [h.rank for h in hits] == [1, 2, 3]
