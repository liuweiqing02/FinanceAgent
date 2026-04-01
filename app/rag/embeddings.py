from __future__ import annotations

import hashlib
import math
from typing import Any, Protocol

from app.config import AppConfig


class TextEmbedding(Protocol):
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedding:
    """离线可运行的哈希向量器。"""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in text.replace("\n", " ").split() if t.strip()]

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in self._tokenize(text):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class SentenceTransformerEmbedding:
    """基于本地 sentence-transformers 的语义向量器。"""

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "auto",
        batch_size: int = 16,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {}
        if device != "auto":
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **kwargs)
        self.batch_size = max(batch_size, 1)
        self.normalize = normalize

    def embed(self, text: str) -> list[float]:
        rows = self.embed_batch([text])
        return rows[0] if rows else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in arr]


def build_embedding_from_config(config: AppConfig, logger: Any | None = None) -> TextEmbedding:
    """根据配置构建向量器；失败自动回退 HashEmbedding。"""

    provider = (config.embedding_provider or "hash").strip().lower()
    if provider == "hash":
        _log(logger, "hash", "HashEmbedding")
        return HashEmbedding(dim=64)

    if provider in {"local", "sentence_transformer", "sentence-transformers", "bge", "e5", "auto"}:
        try:
            emb = SentenceTransformerEmbedding(
                model_name=config.embedding_model_name,
                device=config.embedding_device,
                batch_size=config.embedding_batch_size,
                normalize=config.embedding_normalize,
            )
            _log(logger, provider, "SentenceTransformerEmbedding", model=config.embedding_model_name)
            return emb
        except Exception as exc:  # noqa: BLE001
            _log(logger, provider, "HashEmbedding(fallback)", error=str(exc)[:220])
            return HashEmbedding(dim=64)

    _log(logger, provider, "HashEmbedding(fallback)", error="unknown embedding provider")
    return HashEmbedding(dim=64)


def _log(logger: Any | None, provider: str, runtime: str, model: str = "", error: str = "") -> None:
    if logger is None:
        return
    try:
        logger.log(
            "embedding_runtime",
            {
                "provider": provider,
                "runtime": runtime,
                "model": model,
                "error": error,
            },
        )
    except Exception:  # noqa: BLE001
        return
