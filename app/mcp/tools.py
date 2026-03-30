from __future__ import annotations

from datetime import datetime
from pathlib import Path


def get_market_snapshot(ticker: str) -> dict:
    """示例行情数据工具（可替换为真实行情源）。"""

    seed = sum(ord(c) for c in ticker.upper())
    price = round(20 + (seed % 200) / 3, 2)
    pe = round(8 + (seed % 50) / 2.7, 2)
    pb = round(0.8 + (seed % 30) / 10, 2)
    return {
        "ticker": ticker.upper(),
        "price": price,
        "pe": pe,
        "pb": pb,
        "timestamp": datetime.utcnow().isoformat(),
    }


def load_news_file(base_dir: str, ticker: str) -> list[dict]:
    """从知识库读取新闻文本，兼容离线场景。"""

    folder = Path(base_dir)
    rows: list[dict] = []
    for p in sorted(folder.glob(f"*{ticker.upper()}*.md")):
        content = p.read_text(encoding="utf-8", errors="ignore")
        rows.append({"source_id": p.stem, "title": p.name, "content": content})
    return rows
