from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.infra.logger import JsonlLogger
from app.models.schemas import Evidence
from app.rag.chunking import chunk_text
from app.rag.embeddings import HashEmbedding
from app.rag.ingestion import clean_text, collect_documents
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


def ensure_sample_knowledge(path: Path) -> None:
    """初始化示例知识库，保证开箱可跑。"""

    samples = {
        "AAPL_fundamental.md": """# AAPL 基本面
苹果公司在高端消费电子市场维持较高毛利率。服务业务占比提升，提升现金流稳定性。""",
        "AAPL_technical.md": """# AAPL 技术面
近期价格处于中期上升通道，成交量温和放大。若跌破关键均线，需关注回撤风险。""",
        "AAPL_news.md": """# AAPL 新闻
近期供应链传出新机备货增加，市场预期下半年销量改善。""",
        "TSLA_fundamental.md": """# TSLA 基本面
特斯拉汽车业务利润率受价格策略影响波动，储能业务增速较快。""",
        "TSLA_news.md": """# TSLA 新闻
市场关注自动驾驶监管进展与产能爬坡效率。""",
    }

    path.mkdir(parents=True, exist_ok=True)
    for name, content in samples.items():
        p = path / name
        if not p.exists():
            p.write_text(content, encoding="utf-8")
