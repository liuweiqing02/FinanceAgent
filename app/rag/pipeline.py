from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app.config import AppConfig
from app.infra.logger import JsonlLogger
from app.models.schemas import Evidence
from app.rag.chunking import chunk_text
from app.rag.embeddings import build_embedding_from_config
from app.rag.ingestion import clean_text, collect_documents
from app.rag.knowledge_builder import build_knowledge_base, ensure_sample_raw_data
from app.rag.reranker import build_reranker_from_config
from app.rag.retriever import HybridRetriever
from app.rag.vectorstore.chroma_store import ChromaVectorStore


class RagPipeline:
    """RAG 全链路管道：采集->清洗->切块->向量化->检索->重排。"""

    def __init__(self, config: AppConfig, logger: JsonlLogger) -> None:
        self.config = config
        self.logger = logger
        self.embedding = build_embedding_from_config(config, logger=self.logger)
        use_chroma = config.vector_store_provider.strip().lower() in {"chroma", "chromadb"}
        self.vector_store = ChromaVectorStore(
            collection=config.vector_collection,
            persist_dir=str(config.chroma_persist_dir),
            enable_chroma=use_chroma,
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding=self.embedding,
            bm25_weight=config.bm25_weight,
            vector_weight=config.vector_weight,
        )
        self.reranker = build_reranker_from_config(config, logger=self.logger)
        self._indexed = False

    def build_index(self) -> int:
        docs = collect_documents(self.config.knowledge_base_dir)
        all_chunks = []
        source_hash_now: dict[str, str] = {}
        for d in docs:
            cleaned = clean_text(d.text)
            source_hash_now[d.source_id] = _sha256(cleaned)
            all_chunks.extend(chunk_text(d.source_id, d.title, cleaned))

        old_manifest = self._load_index_manifest()
        old_sources = set(old_manifest.keys())
        now_sources = set(source_hash_now.keys())

        removed_sources = sorted(old_sources - now_sources)
        changed_sources = sorted(
            sid for sid, h in source_hash_now.items() if old_manifest.get(sid, {}).get("hash") != h
        )

        if removed_sources or changed_sources:
            self.vector_store.delete_by_source_ids(sorted(set(removed_sources + changed_sources)))

        changed_chunks = [c for c in all_chunks if c.source_id in set(changed_sources)]
        if changed_chunks:
            embeddings = self.embedding.embed_batch([c.text for c in changed_chunks])
            self.vector_store.upsert(changed_chunks, embeddings)

        # BM25 语料每次全量重建，保证召回一致性。
        self.retriever.index(all_chunks)
        self._indexed = True

        self._save_index_manifest(source_hash_now)
        self.logger.log(
            "index_built",
            {
                "docs": len(docs),
                "chunks": len(all_chunks),
                "changed_sources": len(changed_sources),
                "removed_sources": len(removed_sources),
                "changed_chunks": len(changed_chunks),
            },
        )
        return len(all_chunks)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        ticker: str | None = None,
        topic: str | None = None,
    ) -> list[Evidence]:
        if not self._indexed:
            self.build_index()

        k = top_k or self.config.top_k
        topic_key = (topic or "").lower().strip()

        raw_top_k = max(k * 2, 8)
        if topic_key == "news":
            # 新闻候选更稀疏，扩大初召回避免被财务类片段淹没。
            raw_top_k = max(raw_top_k, 40)

        raw = self.retriever.retrieve(query, top_k=raw_top_k)

        # 先按 ticker/topic 过滤，再重排，保证主题一致性。
        scoped_candidates = _apply_scope_filter(raw.items, ticker=ticker, topic=topic)
        if not scoped_candidates and topic_key != "news" and ticker:
            scoped_candidates = _apply_scope_filter(raw.items, ticker=ticker, topic=None)
        if not scoped_candidates and topic_key != "news":
            scoped_candidates = raw.items

        reranked = self.reranker.rerank(query, scoped_candidates, self.config.rerank_top_n) if scoped_candidates else []
        dedup_by_source = topic_key != "news"
        final = _postprocess_evidence(reranked, max_items=k, dedup_by_source=dedup_by_source)

        self.logger.log(
            "retrieval",
            {
                "query": query,
                "ticker_filter": (ticker or "").upper(),
                "topic_filter": (topic or "").lower(),
                "recall": [
                    {
                        "source_id": ev.source_id,
                        "title": ev.title,
                        "score": ev.score,
                        "chunk_id": ev.metadata.get("chunk_id", ""),
                    }
                    for ev in final
                ],
            },
        )
        return final

    def _index_manifest_path(self) -> Path:
        return self.config.chroma_persist_dir / f"{self.config.vector_collection}_index_manifest.json"

    def _load_index_manifest(self) -> dict[str, dict[str, str]]:
        p = self._index_manifest_path()
        if not p.exists():
            return {}
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                return {str(k): v for k, v in obj.items() if isinstance(v, dict)}
        except Exception:  # noqa: BLE001
            return {}
        return {}

    def _save_index_manifest(self, source_hash: dict[str, str]) -> None:
        p = self._index_manifest_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        out = {sid: {"hash": h} for sid, h in source_hash.items()}
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_sample_knowledge(path: Path, raw_dir: Path | None = None) -> None:
    """兼容旧接口：自动构建可检索知识库，而非写入单句样例。"""

    use_raw_dir = raw_dir or path.parent / "raw"
    ensure_sample_raw_data(use_raw_dir)
    build_knowledge_base(use_raw_dir, path)


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _apply_scope_filter(items: list[Evidence], *, ticker: str | None, topic: str | None) -> list[Evidence]:
    out = items
    if ticker:
        t = ticker.upper().strip()
        out = [x for x in out if _ticker_match(x, t)]
    if topic:
        tp = topic.lower().strip()
        out = [x for x in out if _topic_match(x, tp)]
    return out


def _ticker_match(ev: Evidence, ticker: str) -> bool:
    sid = (ev.source_id or "").upper()
    title = (ev.title or "").upper()
    content = (ev.content or "").upper()
    return sid.startswith(f"{ticker}_") or ticker in title or f" {ticker} " in f" {content} "


def _topic_match(ev: Evidence, topic: str) -> bool:
    sid = (ev.source_id or "").lower()
    title = (ev.title or "").lower()
    body = (ev.content or "").lower()
    if sid.endswith(f"_{topic}"):
        return True

    if topic == "news":
        strong_sid = ["_news", "news.md", "google_news", "newswire", "reuters", "barron", "cnbc", "wsj", "bloomberg"]
        if any(k in sid or k in title for k in strong_sid):
            return True
        strong_body = [
            "来源类型: google_news_rss",
            "source: google_news_rss",
            "来源类型: newswire",
            "来源类型: reuters",
            "来源类型: cnbc",
            "来源类型: barron",
        ]
        return any(k in body for k in strong_body)

    topic_alias = {
        "fundamental": ["fundamental", "10-k", "10-q", "营收", "利润", "现金流", "roe"],
        "technical": ["technical", "sma", "均线", "量能", "阻力", "支撑"],
        "valuation": ["valuation", "pe", "pb", "forward pe", "估值"],
    }
    keys = topic_alias.get(topic, [topic])
    blob = f"{sid} {title} {body}"
    return any(k in blob for k in keys)


def _postprocess_evidence(
    items: list[Evidence],
    max_items: int,
    *,
    dedup_by_source: bool = True,
) -> list[Evidence]:
    if not items:
        return []

    usable = [x for x in items if _is_usable_evidence(x)]
    base = usable if usable else [x for x in items if _is_minimal_evidence(x)]
    if not base:
        base = [x for x in items if not _is_empty_fundamental_snapshot(x)]
    if not base:
        base = items[: max(1, min(max_items, 2))]

    selected: list[Evidence] = []
    seen_source: set[str] = set()
    for ev in base:
        sid = (ev.source_id or "").strip()
        if dedup_by_source and sid and sid in seen_source:
            continue
        selected.append(ev)
        if dedup_by_source and sid:
            seen_source.add(sid)
        if len(selected) >= max_items:
            break
    return selected


def _is_empty_fundamental_snapshot(ev: Evidence) -> bool:
    low = (ev.content or "").lower()
    return ev.source_id.startswith("mcp_fundamental") and low.count("n/a") >= 6


def _is_minimal_evidence(ev: Evidence) -> bool:
    text = (ev.content or "").strip()
    if len(text) < 30:
        return False
    low = text.lower()
    if sum(1 for t in ["us-gaap", "xbrl", "taxonomy", "linkbase", "namespace"] if t in low) >= 2:
        return False
    if ev.source_id.startswith("mcp_fundamental") and low.count("n/a") >= 6:
        return False
    return True


def _is_usable_evidence(ev: Evidence) -> bool:
    text = (ev.content or "").strip()
    if len(text) < 80:
        return False

    low = text.lower()
    bad_tokens = [
        "us-gaap",
        "xbrl",
        "taxonomy",
        "linkbase",
        "schema",
        "dei",
        "namespace",
        "http://",
        "https://",
        "false fy",
        "p7y",
        "p2y",
    ]
    if sum(1 for t in bad_tokens if t in low) >= 3:
        return False

    tokens = re.findall(r"\S+", text)
    if not tokens:
        return False
    noisy = 0
    for tok in tokens:
        t = tok.strip(".,:;()[]{}")
        if not t:
            continue
        if ":" in t and re.match(r"^[A-Za-z-]+:[A-Za-z0-9_]+$", t):
            noisy += 1
            continue
        digit = sum(ch.isdigit() for ch in t)
        if digit >= max(5, int(len(t) * 0.6)):
            noisy += 1
    if noisy / max(len(tokens), 1) > 0.3:
        return False

    if ev.source_id.startswith("mcp_fundamental") and low.count("n/a") >= 8:
        return False
    return True