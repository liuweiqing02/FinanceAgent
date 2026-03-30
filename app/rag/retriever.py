from __future__ import annotations

import math
from collections import Counter

from app.models.schemas import Evidence, RetrievalResult
from app.rag.chunking import Chunk
from app.rag.embeddings import HashEmbedding
from app.rag.vectorstore.base import VectorStore


class HybridRetriever:
    """BM25 + 向量混合检索。"""

    def __init__(self, vector_store: VectorStore, embedding: HashEmbedding, bm25_weight: float, vector_weight: float) -> None:
        self.vector_store = vector_store
        self.embedding = embedding
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self._corpus: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()

    def index(self, chunks: list[Chunk]) -> None:
        self._corpus = chunks
        self._tf = []
        self._df = Counter()
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            tf = Counter(tokens)
            self._tf.append(tf)
            for t in set(tokens):
                self._df[t] += 1

    def retrieve(self, query: str, top_k: int) -> RetrievalResult:
        q_emb = self.embedding.embed(query)
        vector_hits = self.vector_store.query(q_emb, top_k=max(top_k * 2, 10))
        vector_score = {c.chunk_id: s for c, s in vector_hits}

        bm25_score: dict[str, float] = {}
        n = max(len(self._corpus), 1)
        avg_len = sum(sum(tf.values()) for tf in self._tf) / max(len(self._tf), 1)
        for chunk, tf in zip(self._corpus, self._tf, strict=True):
            score = _bm25(query, tf, self._df, n, avg_len)
            bm25_score[chunk.chunk_id] = score

        candidates = {c.chunk_id: c for c in self._corpus}
        merged: list[tuple[Chunk, float]] = []
        for cid, chunk in candidates.items():
            b = bm25_score.get(cid, 0.0)
            v = vector_score.get(cid, 0.0)
            score = self.bm25_weight * b + self.vector_weight * v
            if score > 0:
                merged.append((chunk, score))

        merged.sort(key=lambda x: x[1], reverse=True)
        items = [
            Evidence(source_id=c.source_id, title=c.title, content=c.text, score=round(s, 6), metadata={"chunk_id": c.chunk_id})
            for c, s in merged[:top_k]
        ]
        return RetrievalResult(query=query, items=items)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.replace("\n", " ").split() if t.strip()]


def _bm25(query: str, tf: Counter[str], df: Counter[str], n_docs: int, avg_len: float, k1: float = 1.5, b: float = 0.75) -> float:
    q_tokens = _tokenize(query)
    dl = sum(tf.values()) or 1
    score = 0.0
    for t in q_tokens:
        f = tf.get(t, 0)
        if f == 0:
            continue
        idf = math.log(1 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
        denom = f + k1 * (1 - b + b * dl / (avg_len or 1))
        score += idf * (f * (k1 + 1) / denom)
    return score
