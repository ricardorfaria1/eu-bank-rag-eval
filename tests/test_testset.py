import csv
from pathlib import Path

from eurag.chunking import Chunk
from eurag.testset import COLUMNS, citations_for, load_testset

HANDWRITTEN = Path("data/testset/handwritten.csv")


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({**{c: "" for c in COLUMNS}, "reference_contexts": "[]", "reference_citations": "[]", **r})


def test_load_excludes_generated_and_rejected(tmp_path):
    p = tmp_path / "t.csv"
    base = {"language": "pt", "category": "single_doc", "user_input": "q", "reference": "r"}
    _write(p, [{**base, "id": "a", "origin": "generated"}, {**base, "id": "b", "origin": "rejected"},
               {**base, "id": "c", "origin": "reviewed"}, {**base, "id": "d", "origin": "handwritten"}])
    assert [c.id for c in load_testset(p)] == ["c", "d"]
    assert [c.id for c in load_testset(p, include_unreviewed=True)] == ["a", "b", "c", "d"]


def test_columns_exact_order():
    assert COLUMNS == ["id", "language", "category", "origin", "user_input", "reference", "reference_contexts",
                       "reference_citations", "review_note"]
    with HANDWRITTEN.open(encoding="utf-8") as f:
        assert next(csv.reader(f)) == COLUMNS


def test_handwritten_file_valid():
    cases = load_testset(HANDWRITTEN)
    assert len(cases) >= 10 and all(c.origin == "handwritten" for c in cases)
    assert {c.category for c in cases} == {"multi_doc", "cross_language", "revoked", "cross_supervisor", "out_of_scope"}
    assert all(c.reference_citations == [] for c in cases if c.category == "out_of_scope")
    assert all(c.reference_citations for c in cases if c.category != "out_of_scope")


def test_citations_from_contexts():
    chunks = [Chunk(chunk_id="A:1:1", doc_id="A", article="1", covers=("1",), language="en", url="u", text="x"),
              Chunk(chunk_id="A:1:2", doc_id="A", article="1", covers=("1", "2"), language="en", url="u", text="y"),
              Chunk(chunk_id="A:pre:1", doc_id="A", article=None, covers=(), language="en", url="u", text="z")]
    assert citations_for(chunks) == ["A:1", "A:2"]
