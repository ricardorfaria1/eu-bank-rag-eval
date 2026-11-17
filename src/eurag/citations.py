"""Parse citations like "[AMLR, art. 20]" or "[EBA-RF, guideline 4]" from answers."""

import re
from typing import NamedTuple

from eurag.chunking import normalise_article
from eurag.corpus import SourceDoc


class Citation(NamedTuple):
    doc_id: str
    article: str

    def key(self) -> str:
        return f"{self.doc_id}:{self.article}"


_CITATION_RE = re.compile(
    r"\[\s*(?P<doc>[^,\[\]]+?)\s*,\s*(?:art(?:igo|icle)?\.?|guideline|gl\.?)\s*(?P<num>\d+[^\]\s,;]*)\s*\]",
    re.IGNORECASE,
)


def alias_map(docs: list[SourceDoc]) -> dict[str, str]:
    """Lower-cased doc id and every alias -> doc id."""
    out = {}
    for d in docs:
        for name in [d.doc_id, *d.aliases]:
            out[name.lower()] = d.doc_id
    return out


def parse_citations(text: str, aliases: dict[str, str]) -> list[Citation]:
    found = []
    for m in _CITATION_RE.finditer(text):
        doc_id = aliases.get(m.group("doc").strip().lower())
        if doc_id:
            found.append(Citation(doc_id, normalise_article(m.group("num"))))
    return found
