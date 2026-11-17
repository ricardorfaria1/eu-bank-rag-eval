"""Offline stand-ins for embedding and chat models."""

import hashlib

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_qdrant import SparseEmbeddings, SparseVector


class WordHashSparse(SparseEmbeddings):
    """Bag of words hashed into sparse indices, enough to make keyword search testable."""

    def _vec(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest()[:6], 16)
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return SparseVector(indices=list(counts), values=list(counts.values()))

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def fake_dense():
    return DeterministicFakeEmbedding(size=64)


class RecordingChat(GenericFakeChatModel):
    """Fake chat model that records every prompt it receives."""

    prompts: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def recording_chat(reply="Resposta [AMLR, art. 20].", n=20):
    msgs = iter([AIMessage(content=reply, usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
                 for _ in range(n)])
    return RecordingChat(messages=msgs, prompts=[])
