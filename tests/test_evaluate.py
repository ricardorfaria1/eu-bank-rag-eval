import asyncio
from types import SimpleNamespace

from eurag.answer import Answer
from eurag.evaluate import ROW_FIELDS, evaluate_answers
from eurag.testset import TestCase
from tests.test_answer import _retrieved


class StubMetric:
    def __init__(self, value=0.8, fail_on=None):
        self.value, self.fail_on, self.calls = value, fail_on, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on and self.fail_on in kwargs.get("response", ""):
            raise RuntimeError("judge failed")
        return SimpleNamespace(value=self.value)


def _answer(qid, text="O prazo é 31 de março [AMLR, art. 20]."):
    return Answer(question_id=qid, config_id="baseline", config_fingerprint="f", question="q?", text=text,
                  citations=[("AMLR", "20")], retrieved=_retrieved(), input_tokens=100, output_tokens=10,
                  seconds=1.0, model="m")


def _case(qid, category="single_doc"):
    return TestCase(id=qid, language="pt", category=category, origin="reviewed", user_input="q?",
                    reference="31 de março", reference_citations=[] if category == "out_of_scope" else ["AMLR:20"])


def test_row_fields_exact():
    metrics = {"faithfulness": StubMetric(), "factual_correctness": StubMetric(0.5)}
    rows = asyncio.run(evaluate_answers([_answer("a")], {"a": _case("a")}, metrics))
    assert list(rows[0]) == ROW_FIELDS
    assert rows[0]["faithfulness"] == 0.8 and rows[0]["factual_correctness"] == 0.5
    assert rows[0]["hit_at_5"] == 1.0 and rows[0]["citation_recall"] == 1.0 and rows[0]["refusal_correct"] is None


def test_failed_metric_is_recorded_not_raised():
    metrics = {"faithfulness": StubMetric(fail_on="BAD"), "factual_correctness": StubMetric()}
    answers = [_answer("a"), _answer("b", text="BAD answer [AMLR, art. 20]")]
    rows = asyncio.run(evaluate_answers(answers, {"a": _case("a"), "b": _case("b")}, metrics))
    by_id = {r["question_id"]: r for r in rows}
    assert by_id["a"]["faithfulness"] == 0.8
    assert by_id["b"]["faithfulness"] is None and "faithfulness: judge failed" in by_id["b"]["errors"]
    assert by_id["b"]["factual_correctness"] == 0.8


def test_out_of_scope_skips_judge():
    faith = StubMetric()
    rows = asyncio.run(evaluate_answers([_answer("o", "Não encontrei essa informação.")],
                                        {"o": _case("o", "out_of_scope")},
                                        {"faithfulness": faith, "factual_correctness": StubMetric()}))
    assert faith.calls == []
    assert rows[0]["refusal_correct"] == 1.0 and rows[0]["hit_at_5"] is None


def test_nan_metric_becomes_none_with_error():
    rows = asyncio.run(evaluate_answers([_answer("a")], {"a": _case("a")},
                                        {"faithfulness": StubMetric(float("nan")), "factual_correctness": StubMetric()}))
    assert rows[0]["faithfulness"] is None and any("non-finite" in e for e in rows[0]["errors"])
