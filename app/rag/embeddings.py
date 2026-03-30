from __future__ import annotations

import hashlib
import math


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
