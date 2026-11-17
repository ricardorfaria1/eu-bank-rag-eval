from eurag.answer import ANSWER_SYSTEM_PROMPT, answer_question
from eurag.chunking import Chunk
from eurag.index import Retrieved
from tests.fakes import recording_chat


def _retrieved():
    c = Chunk(chunk_id="AMLR:20:1", doc_id="AMLR", article="20", covers=("20",), language="en", url="u",
              text="Article 20\nCustomer due diligence measures.")
    e = Chunk(chunk_id="EBA-RF:4:1", doc_id="EBA-RF", article="4", covers=("4",), language="en", url="u",
              text="Guideline 4: CDD")
    return [Retrieved(chunk=c, rank=1, score=0.9), Retrieved(chunk=e, rank=2, score=0.8)]


def test_answer_prompt_has_excerpt_headers():
    chat = recording_chat()
    text, usage = answer_question("Quais são as medidas de diligência?", _retrieved(), chat)
    prompt = "\n".join(m.content for m in chat.prompts[0])
    assert "[AMLR, art. 20]" in prompt and "[EBA-RF, guideline 4]" in prompt
    assert "Quais são as medidas de diligência?" in prompt
    assert chat.prompts[0][0].content == ANSWER_SYSTEM_PROMPT
    assert text == "Resposta [AMLR, art. 20]."
    assert usage == {"input_tokens": 100, "output_tokens": 10}
