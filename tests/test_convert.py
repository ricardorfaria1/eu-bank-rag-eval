import shutil
from pathlib import Path

import pytest

from eurag.convert import convert, markdown_to_text
from eurag.corpus import SourceDoc

FIXTURES = Path(__file__).parent / "fixtures"


def _doc(fmt, doc_id="AMLR", lang="en"):
    return SourceDoc(doc_id=doc_id, title="t", aliases=[doc_id], url="https://example.test", format=fmt,
                     language=lang, publisher="p", reuse_note="r")


def test_eurlex_html_keeps_article_lines(tmp_path):
    shutil.copy(FIXTURES / "eurlex_excerpt.html", tmp_path / "AMLR.html")
    out = convert(_doc("html"), tmp_path)
    lines = out.text.splitlines()
    assert "Article 1" in lines and "Article 2" in lines
    assert "<" not in out.text
    assert "Menu Search" not in out.text
    assert any(line.startswith("(a) the measures to be applied by obliged entities") for line in lines)
    assert out.doc_id == "AMLR" and out.language == "en"


def test_markdown_to_text_strips_markup():
    md = "## Artigo 1.º\n\n**Objeto**\n\n- 1 - O presente Aviso regulamenta"
    assert markdown_to_text(md).splitlines() == ["Artigo 1.º", "Objeto", "1 - O presente Aviso regulamenta"]


@pytest.mark.models
def test_pdf_conversion(tmp_path):
    shutil.copy(Path("data/raw/BdP-A3-2024.pdf"), tmp_path / "BdP-A3-2024.pdf")
    out = convert(_doc("pdf", "BdP-A3-2024", "pt"), tmp_path)
    assert "artigo 83.º" in out.text.lower()
