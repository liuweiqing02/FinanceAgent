from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    text: str


def chunk_text(source_id: str, title: str, text: str, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
    """按字符切块，保证离线可用。"""

    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")

    cleaned = text.strip()
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        body = cleaned[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{source_id}#c{idx}",
                source_id=source_id,
                title=title,
                text=body,
            )
        )
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
        idx += 1
    return chunks
