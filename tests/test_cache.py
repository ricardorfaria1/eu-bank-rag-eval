from eurag.answer import answer_all
from eurag.cache import AnswerCache
from eurag.config import RagConfig
from tests.fakes import recording_chat
from tests.test_answer import _retrieved

CONFIG = RagConfig(id="baseline", chunking="fixed", retrieval="dense")


class StubRetriever:
    def search(self, query, k):
        return _retrieved()[:k]


def test_cache_skips_saved_answers(tmp_path):
    cache = AnswerCache(tmp_path)
    questions = [("q1", "Pergunta 1?"), ("q2", "Pergunta 2?"), ("q3", "Pergunta 3?")]
    answer_all(questions[:2], CONFIG, StubRetriever(), recording_chat(), cache, aliases={"amlr": "AMLR"})
    chat = recording_chat()
    answers = answer_all(questions, CONFIG, StubRetriever(), chat, cache, aliases={"amlr": "AMLR"})
    assert len(chat.prompts) == 1
    assert [a.question_id for a in answers] == ["q1", "q2", "q3"]
    assert answers[2].citations[0].doc_id == "AMLR"
    assert cache.get(CONFIG, "q1").text == "Resposta [AMLR, art. 20]."


def test_cache_key_changes_with_config(tmp_path):
    cache = AnswerCache(tmp_path)
    answer_all([("q1", "P?")], CONFIG, StubRetriever(), recording_chat(), cache, aliases={})
    assert cache.get(CONFIG.model_copy(update={"top_k": 3}), "q1") is None
    assert cache.get(CONFIG, "q1") is not None


def test_cache_misses_when_question_or_corpus_changes(tmp_path):
    cache = AnswerCache(tmp_path)
    answer_all([("q1", "Pergunta original?")], CONFIG, StubRetriever(), recording_chat(), cache, aliases={},
               corpus_key="corpus-v1")
    assert cache.get(CONFIG, "q1", "Pergunta original?", "corpus-v1") is not None
    assert cache.get(CONFIG, "q1", "Pergunta editada?", "corpus-v1") is None
    assert cache.get(CONFIG, "q1", "Pergunta original?", "corpus-v2") is None


def test_truncated_last_line_does_not_break_resume(tmp_path):
    cache = AnswerCache(tmp_path)
    answer_all([("q1", "P?")], CONFIG, StubRetriever(), recording_chat(), cache, aliases={})
    with (tmp_path / "baseline.jsonl").open("a") as f:
        f.write('{"question_id": "q2", "trunc')
    assert cache.get(CONFIG, "q1", "P?") is not None


def test_spend_recorded_per_answer_even_if_run_crashes(tmp_path):
    spent = []

    class Boom(Exception):
        pass

    chat = recording_chat()
    calls = {"n": 0}
    original = chat._generate

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise Boom
        return original(*a, **k)

    object.__setattr__(chat, "_generate", flaky)
    try:
        answer_all([("q1", "P1?"), ("q2", "P2?")], CONFIG, StubRetriever(), chat, AnswerCache(tmp_path), aliases={},
                   on_answer=lambda a: spent.append(a.question_id))
    except Boom:
        pass
    assert spent == ["q1"]
