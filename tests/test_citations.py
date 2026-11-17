from eurag.citations import Citation, parse_citations

ALIASES = {"amlr": "AMLR", "regulation (eu) 2024/1624": "AMLR", "bdp-a1-2022": "BdP-A1-2022",
           "aviso 1/2022": "BdP-A1-2022", "eba-rf": "EBA-RF"}


def test_parse_citation_variants():
    text = "Ver [AMLR, art. 20], [amlr, Article 20], [BdP-A1-2022, artigo 5.º] e [Aviso 1/2022, art. 12-A]."
    assert parse_citations(text, ALIASES) == [
        Citation("AMLR", "20"), Citation("AMLR", "20"), Citation("BdP-A1-2022", "5"), Citation("BdP-A1-2022", "12a")]


def test_guideline_citation_and_unknown_doc_ignored():
    assert parse_citations("[EBA-RF, guideline 4] and [GDPR, art. 5]", ALIASES) == [Citation("EBA-RF", "4")]


def test_paragraph_qualifiers_and_multiple_refs():
    text = ("[AMLR, Article 20(1)] [AMLR, art. 20.º, n.º 1] [AMLR, art. 21, para. 2] "
            "[AMLR, art. 22; amlr, art. 23] [AMLR, arts. 24 and 25] [EBA-RF, guideline 4.1]")
    assert [c.article for c in parse_citations(text, ALIASES)] == ["20", "20", "21", "22", "23", "24", "25", "4.1"]
    assert parse_citations("[Aviso 1/2022, art. anexo1-3]", ALIASES) == [Citation("BdP-A1-2022", "anexo1-3")]
