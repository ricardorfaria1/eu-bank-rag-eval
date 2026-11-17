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


def _lines_doc(doc_id, text):
    return DocText(doc_id=doc_id, language="en", url="u", text=text)


def test_guideline_heading_without_colon_or_long():
    long_title = "Guideline 18: Sectoral guideline for payment initiation service providers (PISPs) and account information service providers (AISPs)"
    doc = _lines_doc("EBA-RF", "Guideline 16: Funds\n16.1. Text.\nGuideline 17 Sectoral guideline for crowdfunding\n17.1. Text.\n"
                     + long_title + "\n18.1. Text.")
    assert [c.article for c in article_chunks(doc)] == ["16", "17", "18"]


def test_eba_section_headings_are_units():
    doc = _lines_doc("EBA-RO", "Intro text.\n4.1 Internal policies and procedures\n4.1.1 Policies\n9. Firms should.\n"
                     "4.2 Acquisition of information\n15. Firms should identify.")
    assert [c.article for c in article_chunks(doc) if c.article] == ["4.1", "4.2"]


def test_annex_articles_do_not_collide_with_main_articles():
    doc = DocText(doc_id="BdP-A1-2022", language="pt", url="u",
                  text="Artigo 1.º Objeto\nA.\nArtigo 2.º\nB.\nArtigo 3.º\nC.\nAnexo I ao Aviso\nArtigo 1.º\nD.\nArtigo 2.º\nE.")
    assert [c.article for c in article_chunks(doc)] == ["1", "2", "3", "anexo1-1", "anexo1-2"]
    fixed = fixed_chunks(doc, size=12, overlap=0)
    assert "anexo1-1" in {a for c in fixed for a in c.covers}
