from __future__ import annotations

from app.models.schemas import Evidence


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
            ev.score = round(s, 6)
            final.append(ev)
        return final
