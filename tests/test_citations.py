from eurag.citations import Citation, parse_citations

ALIASES = {"amlr": "AMLR", "regulation (eu) 2024/1624": "AMLR", "bdp-a1-2022": "BdP-A1-2022",
           "aviso 1/2022": "BdP-A1-2022", "eba-rf": "EBA-RF"}


def test_parse_citation_variants():
    text = "Ver [AMLR, art. 20], [amlr, Article 20], [BdP-A1-2022, artigo 5.º] e [Aviso 1/2022, art. 12-A]."
    assert parse_citations(text, ALIASES) == [
        Citation("AMLR", "20"), Citation("AMLR", "20"), Citation("BdP-A1-2022", "5"), Citation("BdP-A1-2022", "12a")]


def test_guideline_citation_and_unknown_doc_ignored():
    assert parse_citations("[EBA-RF, guideline 4] and [GDPR, art. 5]", ALIASES) == [Citation("EBA-RF", "4")]
