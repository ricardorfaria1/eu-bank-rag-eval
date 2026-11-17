"""Qdrant index (local mode, on disk) and retrieval."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from eurag.chunking import Chunk


class Retrieved(BaseModel):
    chunk: Chunk
    rank: int
    score: float


_CLIENTS: dict[str, QdrantClient] = {}


def _client(path: Path) -> QdrantClient:
    # Qdrant local mode allows one client per storage folder in a process.
    key = str(Path(path).resolve())
    if key not in _CLIENTS:
        Path(path).mkdir(parents=True, exist_ok=True)
        _CLIENTS[key] = QdrantClient(path=key)
    return _CLIENTS[key]


def build_index(chunks: list[Chunk], name: str, path: Path, embeddings: Embeddings,
                sparse: SparseEmbeddings | None) -> None:
    client = _client(path)
    if client.collection_exists(name):
        client.delete_collection(name)
    docs = [Document(page_content=c.text, metadata=c.model_dump(exclude={"text"})) for c in chunks]
    _create(client, name, embeddings, sparse).add_documents(docs)


def _create(client: QdrantClient, name: str, embeddings: Embeddings, sparse: SparseEmbeddings | None):
    size = len(embeddings.embed_query("dimension probe"))
    sparse_config = {"langchain-sparse": models.SparseVectorParams()} if sparse else None
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        sparse_vectors_config=sparse_config,
    )
    return QdrantVectorStore(
        client=client, collection_name=name, embedding=embeddings, sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID if sparse else RetrievalMode.DENSE,
    )


class Retriever:
    def __init__(self, path: Path, name: str, embeddings: Embeddings, sparse: SparseEmbeddings | None = None):
        self._store = QdrantVectorStore(
            client=_client(path), collection_name=name, embedding=embeddings, sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID if sparse else RetrievalMode.DENSE,
        )

    def search(self, query: str, k: int) -> list["Retrieved"]:
        results = self._store.similarity_search_with_score(query, k=k)
        return [
            Retrieved(chunk=Chunk(text=doc.page_content, **{key: v for key, v in doc.metadata.items()
                                                              if not key.startswith("_")}),
                      rank=i, score=float(score))
            for i, (doc, score) in enumerate(results, 1)
        ]


def default_embeddings(model: str = "BAAI/bge-m3") -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})


def default_sparse() -> SparseEmbeddings:
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name="Qdrant/bm25")
