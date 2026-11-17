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


_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_REF_RE = re.compile(
    r"^\s*(?P<doc>[^,]+?)\s*,\s*(?:arts?(?:igos?|icles?)?\.?|guidelines?|gl\.?)\s*(?P<nums>.+)$", re.IGNORECASE
)
# "20", "20.º", "20(1)", "12-A", "4.1", "anexo1-3"; qualifiers such as ", n.º 1" or ", para. 2" are ignored.
_NUM_RE = re.compile(r"(?:anexo\d+-)?\d+(?:\.\d+(?![º°o]))?(?:\.?\s*[º°o])?(?:-?[A-Za-z](?![a-z]))?", re.IGNORECASE)
_QUALIFIER_RE = re.compile(r",\s*(?:n\.?\s*º|n\.?\s*o|para(?:graph)?\.?|§|al\.|alínea|point).*$", re.IGNORECASE)


def alias_map(docs: list[SourceDoc]) -> dict[str, str]:
    """Lower-cased doc id and every alias -> doc id."""
    out = {}
    for d in docs:
        for name in [d.doc_id, *d.aliases]:
            out[name.lower()] = d.doc_id
    return out


def _numbers(nums: str) -> list[str]:
    nums = _QUALIFIER_RE.sub("", nums)
    nums = re.sub(r"\([^)]*\)", "", nums)  # "20(1)" -> "20"
    parts = re.split(r"\s*(?:,|\band\b|\be\b|&)\s*", nums)
    out = []
    for part in parts:
        m = _NUM_RE.match(part.strip())
        if m:
            raw = m.group(0)
            out.append(raw.lower() if raw.lower().startswith("anexo") else normalise_article(raw))
    return out


def parse_citations(text: str, aliases: dict[str, str]) -> list[Citation]:
    found = []
    for bracket in _BRACKET_RE.finditer(text):
        for ref in bracket.group(1).split(";"):
            m = _REF_RE.match(ref)
            if not m:
                continue
            doc_id = aliases.get(m.group("doc").strip().lower())
            if doc_id:
                found += [Citation(doc_id, n) for n in _numbers(m.group("nums"))]
    return found
