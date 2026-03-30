from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.infra.logger import JsonlLogger
from app.models.schemas import Evidence
from app.rag.chunking import chunk_text
from app.rag.embeddings import HashEmbedding
from app.rag.ingestion import clean_text, collect_documents
from app.rag.knowledge_builder import build_knowledge_base, ensure_sample_raw_data
from app.rag.reranker import SimpleReranker
from app.rag.retriever import HybridRetriever
from app.rag.vectorstore.chroma_store import ChromaVectorStore


class RagPipeline:
    """RAG 全链路管道：采集->清洗->切块->向量化->检索->重排。"""

    def __init__(self, config: AppConfig, logger: JsonlLogger) -> None:
        self.config = config
        self.logger = logger
        self.embedding = HashEmbedding(dim=64)
        self.vector_store = ChromaVectorStore(
            collection=config.vector_collection,
            persist_dir=str(config.chroma_persist_dir),
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding=self.embedding,
            bm25_weight=config.bm25_weight,
            vector_weight=config.vector_weight,
        )
        self.reranker = SimpleReranker()
        self._indexed = False

    def build_index(self) -> int:
        docs = collect_documents(self.config.knowledge_base_dir)
        chunks = []
        for d in docs:
            cleaned = clean_text(d.text)
            chunks.extend(chunk_text(d.source_id, d.title, cleaned))

        embeddings = self.embedding.embed_batch([c.text for c in chunks])
        self.vector_store.upsert(chunks, embeddings)
        self.retriever.index(chunks)
        self._indexed = True

        self.logger.log("index_built", {"docs": len(docs), "chunks": len(chunks)})
        return len(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        if not self._indexed:
            self.build_index()

        k = top_k or self.config.top_k
        raw = self.retriever.retrieve(query, top_k=max(k * 2, 8))
        reranked = self.reranker.rerank(query, raw.items, self.config.rerank_top_n)

        self.logger.log(
            "retrieval",
            {
                "query": query,
                "recall": [
                    {
                        "source_id": ev.source_id,
                        "title": ev.title,
                        "score": ev.score,
                        "chunk_id": ev.metadata.get("chunk_id", ""),
                    }
                    for ev in reranked
                ],
            },
        )
        return reranked[:k]


def ensure_sample_knowledge(path: Path, raw_dir: Path | None = None) -> None:
    """兼容旧接口：自动构建可检索知识库，而非写入单句样例。"""

    use_raw_dir = raw_dir or path.parent / "raw"
    ensure_sample_raw_data(use_raw_dir)
    build_knowledge_base(use_raw_dir, path)
