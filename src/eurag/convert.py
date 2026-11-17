"""Turn downloaded documents into plain text with one heading or paragraph per line.

Article headings ("Article 12", "Artigo 12.º") must stay on their own line,
because per-article chunking splits on them.
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict

from eurag.corpus import SourceDoc


class DocText(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    language: str
    url: str
    text: str


def _clean(s: str) -> str:
    return " ".join(s.replace("\xa0", " ").split())


def eurlex_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("#docHtml") or soup.body or soup
    # A table row in EUR-Lex is one numbered point: "(a)" | "text". Flatten it into one paragraph.
    for row in body.select("tr"):
        row.replace_with(soup.new_tag("p", string=_clean(row.get_text(" "))))
    lines = [_clean(p.get_text(" ")) for p in body.find_all("p")]
    return "\n".join(line for line in lines if line)




def markdown_to_text(md: str) -> str:
    out = []
    for line in md.splitlines():
        line = re.sub(r"^\s*#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = line.replace("**", "").replace("__", "")
        line = _clean(line)
        if line:
            out.append(line)
    return "\n".join(out)


def pdf_to_text(path: Path) -> str:
    from docling.document_converter import DocumentConverter  # heavy import, only when needed

    return markdown_to_text(DocumentConverter().convert(str(path)).document.export_to_markdown())


def convert(doc: SourceDoc, raw_dir: Path) -> DocText:
    path = doc.raw_path(raw_dir)
    if doc.format == "html":
        text = eurlex_html_to_text(path.read_text(encoding="utf-8"))
    else:
        text = pdf_to_text(path)
    return DocText(doc_id=doc.doc_id, language=doc.language, url=doc.url, text=text)
