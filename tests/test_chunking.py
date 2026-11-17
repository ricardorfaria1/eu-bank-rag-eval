from pathlib import Path

import pytest

from eurag.chunking import article_chunks, fixed_chunks, normalise_article
from eurag.convert import DocText

FIX = Path(__file__).parent / "fixtures"


def _doc(name, doc_id, lang):
    return DocText(doc_id=doc_id, language=lang, url="https://example.test", text=(FIX / name).read_text())


@pytest.mark.parametrize(("raw", "expected"), [("12.º", "12"), ("12-A", "12a"), ("12a", "12a"), ("5.o", "5"), ("3", "3")])
def test_normalise_article(raw, expected):
    assert normalise_article(raw) == expected


def test_article_chunks_en():
    chunks = article_chunks(_doc("articles_en.txt", "AMLR", "en"))
    assert sorted({c.article for c in chunks}, key=int) == ["1", "2", "3"]
    assert all(c.doc_id == "AMLR" and c.language == "en" for c in chunks)


def test_article_chunks_pt_ordinal():
    chunks = article_chunks(_doc("articles_pt.txt", "BdP-A1-2022", "pt"))
    assert sorted({c.article for c in chunks}, key=int) == ["1", "2", "3"]


def test_eba_guidelines_are_units():
    chunks = article_chunks(_doc("guidelines_eba.txt", "EBA-RF", "en"))
    assert {c.article for c in chunks} == {"1", "2"}


def test_long_article_split_keeps_heading():
    chunks = [c for c in article_chunks(_doc("articles_en.txt", "AMLR", "en"), max_chars=2000) if c.article == "2"]
    assert len(chunks) > 1
    assert all(c.text.startswith("Article 2") for c in chunks)
    assert all(len(c.text) <= 2000 + 200 for c in chunks)


def test_fixed_chunks_size_and_overlap():
    chunks = fixed_chunks(_doc("articles_en.txt", "AMLR", "en"))
    assert all(len(c.text) <= 1000 for c in chunks)
    first, second = chunks[0].text, chunks[1].text
    assert first[-60:].split()[-1] in second


def test_fixed_chunk_article_attribution():
    chunks = fixed_chunks(_doc("articles_en.txt", "AMLR", "en"))
    assert chunks[0].article == "1"
    assert chunks[-1].article == "2"  # the article the chunk starts in
    assert "3" in chunks[-1].covers  # but it also contains Article 3, so retrieval metrics credit it


def test_article_chunk_covers_its_article():
    chunks = article_chunks(_doc("articles_en.txt", "AMLR", "en"))
    assert all(c.covers == ((c.article,) if c.article else ()) for c in chunks)


def test_chunk_ids_unique():
    for fn in (fixed_chunks, article_chunks):
        chunks = fn(_doc("articles_en.txt", "AMLR", "en"))
        assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_reference_in_body_is_not_a_heading():
    doc = DocText(doc_id="X", language="pt", url="u",
                  text="Artigo 1.º Objeto\nTexto.\nArtigo 83.º do Aviso n.º 1/2022 prevê que as entidades enviem o relatório anual ao Banco de Portugal, nos termos do modelo definido.\nArtigo 2.º\nMais texto.")
    assert [c.article for c in article_chunks(doc)] == ["1", "2"]
