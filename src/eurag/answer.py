"""Generate cited answers from retrieved excerpts."""

import hashlib
import time
from collections.abc import Callable, Iterable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from eurag.cache import AnswerCache
from eurag.citations import Citation, parse_citations
from eurag.config import RagConfig
from eurag.index import Retrieved

ANSWER_SYSTEM_PROMPT = """\
You answer questions from a bank's compliance team about anti-money laundering rules.

Use only the numbered excerpts provided. Each excerpt starts with a header such as [AMLR, art. 20] \
or [EBA-RF, guideline 4]. After every claim, cite the excerpt it comes from by copying its header \
exactly, for example [AMLR, art. 20].

Answer in the language of the question. Be concise.

If the excerpts do not contain the answer, say so in one sentence and do not guess."""

PROMPT_VERSION = hashlib.sha256(ANSWER_SYSTEM_PROMPT.encode()).hexdigest()[:8]


def excerpt_header(retrieved: Retrieved) -> str:
    """One header per article the excerpt contains, so every article in it can be cited."""
    c = retrieved.chunk
    if not c.covers:
        return f"[{c.doc_id}, preamble]"
    unit = "guideline" if c.doc_id.startswith("EBA") else "art."
    return " ".join(f"[{c.doc_id}, {unit} {a}]" for a in c.covers)


def build_messages(question: str, retrieved: list[Retrieved]) -> list:
    excerpts = "\n\n".join(f"{i}. {excerpt_header(r)}\n{r.chunk.text}" for i, r in enumerate(retrieved, 1))
    return [SystemMessage(ANSWER_SYSTEM_PROMPT), HumanMessage(f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}")]


def answer_question(question: str, retrieved: list[Retrieved], chat: BaseChatModel) -> tuple[str, dict]:
    reply = chat.invoke(build_messages(question, retrieved))
    usage = reply.usage_metadata or {}
    return reply.text, {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}


class Answer(BaseModel):
    question_id: str
    config_id: str
    config_fingerprint: str
    question: str
    text: str
    citations: list[Citation]
    retrieved: list[Retrieved]
    input_tokens: int
    output_tokens: int
    seconds: float
    model: str
    corpus_key: str = ""
    prompt_version: str = PROMPT_VERSION


def answer_all(questions: Iterable[tuple[str, str]], config: RagConfig, retriever, chat: BaseChatModel,
               cache: AnswerCache, aliases: dict[str, str], corpus_key: str = "",
               on_answer: Callable[[Answer], None] | None = None) -> list[Answer]:
    """Answer (question_id, question) pairs, reusing cached answers; on_answer runs after each paid answer."""
    answers = []
    for qid, question in questions:
        cached = cache.get(config, qid, question, corpus_key)
        if cached is not None:
            answers.append(cached)
            continue
        retrieved = retriever.search(question, k=config.top_k)
        start = time.perf_counter()
        text, usage = answer_question(question, retrieved, chat)
        answer = Answer(
            question_id=qid, config_id=config.id, config_fingerprint=config.fingerprint(), question=question,
            text=text, citations=parse_citations(text, aliases), retrieved=retrieved,
            seconds=round(time.perf_counter() - start, 3), model=config.answer_model, corpus_key=corpus_key, **usage,
        )
        cache.put(answer)
        if on_answer:
            on_answer(answer)
        answers.append(answer)
    return answers


def default_chat(model: str) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # No temperature/top_p: current Claude models reject sampling parameters.
    return ChatAnthropic(model=model, max_tokens=4096)
