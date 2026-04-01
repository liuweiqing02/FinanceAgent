from app.config import AppConfig
from app.models.schemas import Evidence
from app.rag.embeddings import HashEmbedding, build_embedding_from_config
from app.rag.reranker import SimpleReranker, build_reranker_from_config


def test_embedding_unknown_provider_fallback_to_hash() -> None:
    cfg = AppConfig(embedding_provider="unknown-provider")
    emb = build_embedding_from_config(cfg)
    assert isinstance(emb, HashEmbedding)
    vec = emb.embed("hello finance")
    assert len(vec) == 64


def test_reranker_unknown_provider_fallback_to_simple() -> None:
    cfg = AppConfig(reranker_provider="unknown-provider")
    rr = build_reranker_from_config(cfg)
    assert isinstance(rr, SimpleReranker)
    items = [
        Evidence(source_id="a", title="A", content="AAPL growth", score=0.1),
        Evidence(source_id="b", title="B", content="TSLA drop", score=0.2),
    ]
    out = rr.rerank("AAPL", items, top_n=1)
    assert len(out) == 1
