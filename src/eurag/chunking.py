"""Two chunking strategies: fixed-size (baseline) and one chunk per article.

"Article" is the citable unit of each document: "Article 12" in EU acts,
"Artigo 12.º" in Portuguese acts, "Guideline 4" in EBA guidelines, and the
"4.1 Title" sections of EBA guidelines without numbered guidelines. Articles
of an annex that restarts at "Artigo 1.º" are labelled "anexo1-1", "anexo1-2"...
"""

import re
from bisect import bisect_right

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict

from eurag.convert import DocText

MAX_HEADING_CHARS = 100

ARTICLE_RE = re.compile(
    r"^(?:"
    r"Article\s+(?P<en>\d+[a-z]?)"
    r"|Art(?:igo|\.)\s*(?P<pt>\d+)(?:\.?\s*[ºo°])?(?:-(?P<pt_suffix>[A-Z]))?(?:\s+\S.*)?"
    r"|(?P<sec>\d+\.\d+)\s+[A-Z][^.]{2,90}"
    r")$"
)
GUIDELINE_RE = re.compile(r"^Guideline\s+(?P<gl>\d+[A-Z]?)\b\s*:?.*$")


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    article: str | None
    covers: tuple[str, ...] = ()  # every article whose text appears in the chunk
    language: str
    url: str
    text: str


def normalise_article(raw: str) -> str:
    """'12.º' -> '12', '12-A' -> '12a', '5.o' -> '5'."""
    m = re.match(r"(\d+)\s*(?:\.?\s*[ºo°])?(?:-?([A-Za-z]))?$", raw.strip())
    if not m:
        return raw.strip().lower()
    return m.group(1) + (m.group(2) or "").lower()


def heading_article(line: str) -> str | None:
    """The article number if the line is an article heading, else None (no annex handling)."""
    line = line.strip()
    g = GUIDELINE_RE.match(line)
    if g:  # guideline titles can be long
        return normalise_article(g.group("gl"))
    if len(line) > MAX_HEADING_CHARS:
        return None
    m = ARTICLE_RE.match(line)
    if not m:
        return None
    if m.group("en"):
        return normalise_article(m.group("en"))
    if m.group("pt"):
        return m.group("pt") + (m.group("pt_suffix") or "").lower()
    return m.group("sec")


def heading_labels(lines: list[str]) -> list[str | None]:
    """Article label per line (None for non-headings); articles restarting at 1 go into annexes."""
    labels: list[str | None] = []
    annex, highest = 0, 0
    for line in lines:
        article = heading_article(line)
        if article is not None:
            number = int(re.match(r"\d+", article).group())
            if number == 1 and highest > 1 and "." not in article:
                annex += 1
                highest = 0
            highest = max(highest, number)
            if annex:
                article = f"anexo{annex}-{article}"
        labels.append(article)
    return labels


def _sections(text: str) -> list[tuple[str | None, list[str]]]:
    """Split text into (article, lines) sections; text before the first heading has article None."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    lines = text.splitlines()
    for line, article in zip(lines, heading_labels(lines), strict=True):
        if article is not None:
            sections.append((article, [line]))
        else:
            sections[-1][1].append(line)
    return [(a, lines) for a, lines in sections if any(line.strip() for line in lines)]


_PARAGRAPH_RE = re.compile(r"^\(?\d+(?:\.\d+)*[\.\)]?\s*[-–]?\s")


def _split_long(heading: str, body: list[str], max_chars: int) -> list[str]:
    """Group numbered paragraphs into parts of at most max_chars, each starting with the heading."""
    groups: list[list[str]] = [[]]
    for line in body:
        if _PARAGRAPH_RE.match(line) and groups[-1]:
            groups.append([line])
        else:
            groups[-1].append(line)
    parts: list[str] = []
    current = ""
    for group in groups:
        block = "\n".join(group)
        if current and len(heading) + len(current) + len(block) + 2 > max_chars:
            parts.append(current)
            current = block
        else:
            current = f"{current}\n{block}" if current else block
    if current:
        parts.append(current)
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars - len(heading) - 1, chunk_overlap=0)
    out = []
    for part in parts:
        pieces = [part] if len(part) + len(heading) + 1 <= max_chars else splitter.split_text(part)
        out.extend(f"{heading}\n{p}" for p in pieces)
    return out


def article_chunks(doc: DocText, max_chars: int = 2000) -> list[Chunk]:
    chunks: list[Chunk] = []
    preamble_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=0)
    for article, lines in _sections(doc.text):
        if article is None:
            texts = preamble_splitter.split_text("\n".join(lines))
        else:
            heading, body = lines[0], lines[1:]
            whole = "\n".join(lines)
            texts = [whole] if len(whole) <= max_chars else _split_long(heading, body, max_chars)
        for n, text in enumerate(texts, 1):
            label = article or "pre"
            chunks.append(Chunk(chunk_id=f"{doc.doc_id}:{label}:{n}", doc_id=doc.doc_id, article=article,
                                covers=(article,) if article else (), language=doc.language, url=doc.url,
                                text=text))
    return _dedupe_ids(chunks)


def _dedupe_ids(chunks: list[Chunk]) -> list[Chunk]:
    """Articles repeated in a document (e.g. amending acts quoting articles) get distinct ids."""
    seen: dict[str, int] = {}
    out = []
    for c in chunks:
        count = seen.get(c.chunk_id, 0)
        seen[c.chunk_id] = count + 1
        out.append(c if count == 0 else c.model_copy(update={"chunk_id": f"{c.chunk_id}#{count}"}))
    return out


def fixed_chunks(doc: DocText, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    offsets, articles, pos = [], [], 0
    lines = doc.text.splitlines(keepends=True)
    for line, article in zip(lines, heading_labels([x.rstrip("\n") for x in lines]), strict=True):
        if article is not None:
            offsets.append(pos)
            articles.append(article)
        pos += len(line)
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    chunks = []
    for i, d in enumerate(splitter.create_documents([doc.text])):
        start = d.metadata["start_index"]
        first = bisect_right(offsets, start) - 1
        last = bisect_right(offsets, start + len(d.page_content) - 1) - 1
        article = articles[first] if first >= 0 else None
        covers = tuple(dict.fromkeys(articles[max(first, 0):last + 1])) if last >= 0 else ()
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}:{article or 'pre'}:f{i}", doc_id=doc.doc_id, article=article,
                            covers=covers, language=doc.language, url=doc.url, text=d.page_content))
    return chunks
