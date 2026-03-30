from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    """RAG 可观测日志，记录检索与生成关键事件。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: dict[str, Any]) -> None:
        row = {"event": event, "payload": payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
