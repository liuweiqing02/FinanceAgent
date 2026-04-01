from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RawDocument:
    source_id: str
    title: str
    text: str


def collect_documents(path: Path) -> list[RawDocument]:
    docs: list[RawDocument] = []
    for file in sorted(path.glob("*.md")):
        content = file.read_text(encoding="utf-8", errors="ignore")
        docs.append(RawDocument(source_id=file.stem, title=file.name, text=content))
    return docs


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b(?:us-gaap|xbrl|taxonomy|linkbase|namespace|dei)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z0-9_-]{40,}", " ", text)
    text = re.sub(r"\b[A-Za-z-]+:[A-Za-z0-9_]+\b", " ", text)
    text = re.sub(r"\b\d{6,}\b", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
