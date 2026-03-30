from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class KnowledgeQualityReport:
    raw_records: int
    topic_coverage: dict[str, int]
    source_diversity: int
    source_counts: dict[str, int]
    doc_length_distribution: dict[str, int]
    noise_ratio_raw: float
    noise_ratio_kb: float


def analyze_knowledge_quality(raw_dir: Path, kb_dir: Path, raw_glob: str = "*.jsonl") -> KnowledgeQualityReport:
    raw_rows = _load_raw_rows(raw_dir, raw_glob)
    topic_counts = Counter(str(r.get("topic", "unknown")).lower() for r in raw_rows)
    source_counts = Counter(str(r.get("source", "unknown")) for r in raw_rows)

    kb_texts = [p.read_text(encoding="utf-8", errors="ignore") for p in sorted(kb_dir.glob("*.md"))]

    doc_length_distribution = {
        "lt_1000": sum(1 for t in kb_texts if len(t) < 1000),
        "1000_5000": sum(1 for t in kb_texts if 1000 <= len(t) < 5000),
        "ge_5000": sum(1 for t in kb_texts if len(t) >= 5000),
    }

    noise_ratio_raw = _noise_ratio("\n".join(str(r.get("content", "")) for r in raw_rows))
    noise_ratio_kb = _noise_ratio("\n".join(kb_texts))

    return KnowledgeQualityReport(
        raw_records=len(raw_rows),
        topic_coverage=dict(topic_counts),
        source_diversity=len(source_counts),
        source_counts=dict(source_counts),
        doc_length_distribution=doc_length_distribution,
        noise_ratio_raw=round(noise_ratio_raw, 4),
        noise_ratio_kb=round(noise_ratio_kb, 4),
    )


def _load_raw_rows(raw_dir: Path, raw_glob: str) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(raw_dir.glob(raw_glob)):
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _noise_ratio(text: str) -> float:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 0.0

    noisy = 0
    for tok in tokens:
        t = tok.strip()
        if ":" in t and re.match(r"^[A-Za-z-]+:[A-Za-z0-9_]+$", t):
            noisy += 1
            continue
        digits = sum(ch.isdigit() for ch in t)
        if digits >= max(4, int(len(t) * 0.6)):
            noisy += 1
            continue
        if len(t) > 20 and sum(ch.isalpha() for ch in t) < len(t) * 0.3:
            noisy += 1
    return noisy / len(tokens)
