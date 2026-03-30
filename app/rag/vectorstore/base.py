from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.chunking import Chunk


class VectorStore(ABC):
    """向量库抽象接口，便于切换 Chroma/Milvus/pgvector。"""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        raise NotImplementedError
