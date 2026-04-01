from __future__ import annotations

from typing import Any, Protocol

from app.config import AppConfig
from app.models.schemas import Evidence


class Reranker(Protocol):
    def rerank(self, query: str, items: list[Evidence], top_n: int) -> list[Evidence]: ...


class SimpleReranker:
    """基于关键词覆盖与原始分数的轻量重排器。"""

    def rerank(self, query: str, items: list[Evidence], top_n: int) -> list[Evidence]:
        q_tokens = {t.lower() for t in query.split() if t.strip()}
        rescored: list[tuple[Evidence, float]] = []
        for item in items:
            c_tokens = {t.lower() for t in item.content.split() if t.strip()}
            overlap = len(q_tokens & c_tokens)
            score = item.score + overlap * 0.05
            rescored.append((item, score))

        rescored.sort(key=lambda x: x[1], reverse=True)
        final: list[Evidence] = []
        for ev, s in rescored[:top_n]:
            ev.score = round(float(s), 6)
            final.append(ev)
        return final


class CrossEncoderReranker:
    """本地 cross-encoder 重排器。"""

    def __init__(self, *, model_name: str, device: str = "auto", batch_size: int = 16) -> None:
        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {}
        if device != "auto":
            kwargs["device"] = device
        self.model = CrossEncoder(model_name, **kwargs)
        self.batch_size = max(batch_size, 1)

    def rerank(self, query: str, items: list[Evidence], top_n: int) -> list[Evidence]:
        if not items:
            return []

        pairs = [(query, x.content[:3000]) for x in items]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        rescored: list[tuple[Evidence, float]] = []
        for i, ev in enumerate(items):
            ce = float(scores[i]) if i < len(scores) else 0.0
            # 保留少量原始分，避免极端分数导致排序跳变。
            score = ce * 0.9 + float(ev.score) * 0.1
            rescored.append((ev, score))

        rescored.sort(key=lambda x: x[1], reverse=True)
        out: list[Evidence] = []
        for ev, s in rescored[:top_n]:
            ev.score = round(float(s), 6)
            out.append(ev)
        return out


def build_reranker_from_config(config: AppConfig, logger: Any | None = None) -> Reranker:
    """根据配置构建重排器；失败自动回退 SimpleReranker。"""

    provider = (config.reranker_provider or "simple").strip().lower()
    if provider == "simple":
        _log(logger, provider, "SimpleReranker")
        return SimpleReranker()

    if provider in {"local", "cross_encoder", "cross-encoder", "bge", "auto"}:
        try:
            rr = CrossEncoderReranker(
                model_name=config.reranker_model_name,
                device=config.reranker_device,
                batch_size=config.reranker_batch_size,
            )
            _log(logger, provider, "CrossEncoderReranker", model=config.reranker_model_name)
            return rr
        except Exception as exc:  # noqa: BLE001
            _log(logger, provider, "SimpleReranker(fallback)", error=str(exc)[:220])
            return SimpleReranker()

    _log(logger, provider, "SimpleReranker(fallback)", error="unknown reranker provider")
    return SimpleReranker()


def _log(logger: Any | None, provider: str, runtime: str, model: str = "", error: str = "") -> None:
    if logger is None:
        return
    try:
        logger.log(
            "reranker_runtime",
            {
                "provider": provider,
                "runtime": runtime,
                "model": model,
                "error": error,
            },
        )
    except Exception:  # noqa: BLE001
        return
