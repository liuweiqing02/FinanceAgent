from __future__ import annotations

import math
from dataclasses import dataclass

from app.rag.chunking import Chunk
from app.rag.vectorstore.base import VectorStore


@dataclass(slots=True)
class _MemoryRow:
    chunk: Chunk
    embedding: list[float]


class ChromaVectorStore(VectorStore):
    """默认向量库实现。优先使用 Chroma，缺失时自动回退内存模式。"""

    def __init__(self, collection: str, persist_dir: str) -> None:
        self.collection_name = collection
        self.persist_dir = persist_dir
        self._mem: list[_MemoryRow] = []
        self._chroma = None
        self._col = None
        try:
            import chromadb

            self._chroma = chromadb.PersistentClient(path=persist_dir)
            self._col = self._chroma.get_or_create_collection(name=collection)
        except Exception:  # noqa: BLE001
            self._chroma = None
            self._col = None

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if self._col is not None:
            ids = [c.chunk_id for c in chunks]
            docs = [c.text for c in chunks]
            metas = [{"source_id": c.source_id, "title": c.title} for c in chunks]
            self._col.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
            return

        for c, emb in zip(chunks, embeddings, strict=True):
            self._mem.append(_MemoryRow(chunk=c, embedding=emb))

    def query(self, embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        if self._col is not None:
            res = self._col.query(query_embeddings=[embedding], n_results=top_k)
            rows: list[tuple[Chunk, float]] = []
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            dists = res.get("distances", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            for i, cid in enumerate(ids):
                meta = metas[i] if i < len(metas) and metas else {}
                ch = Chunk(
                    chunk_id=cid,
                    source_id=meta.get("source_id", "unknown"),
                    title=meta.get("title", "unknown"),
                    text=docs[i] if i < len(docs) else "",
                )
                distance = dists[i] if i < len(dists) else 1.0
                score = 1.0 / (1.0 + float(distance))
                rows.append((ch, score))
            return rows

        rows: list[tuple[Chunk, float]] = []
        for row in self._mem:
            rows.append((row.chunk, _cosine(embedding, row.embedding)))
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
