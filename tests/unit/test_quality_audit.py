import json
from pathlib import Path

from app.rag.quality_audit import analyze_knowledge_quality


def test_analyze_knowledge_quality_metrics(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    kb = tmp_path / "kb"
    raw.mkdir()
    kb.mkdir()

    rows = [
        {
            "ticker": "AAPL",
            "topic": "fundamental",
            "source": "sec",
            "content": "Revenue grew with stable margin and cash flow.",
        },
        {
            "ticker": "AAPL",
            "topic": "news",
            "source": "google_news_rss",
            "content": "Supplier update suggests stronger demand next quarter.",
        },
        {
            "ticker": "AAPL",
            "topic": "valuation",
            "source": "sec",
            "content": "us-gaap:Revenue recognized with 0000320193 reference.",
        },
    ]

    (raw / "real_events.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    (raw / "sample.jsonl").write_text("", encoding="utf-8")

    (kb / "AAPL_fundamental.md").write_text("A" * 800, encoding="utf-8")
    (kb / "AAPL_news.md").write_text("B" * 1800, encoding="utf-8")
    (kb / "AAPL_valuation.md").write_text("C" * 5200, encoding="utf-8")

    report = analyze_knowledge_quality(raw, kb, raw_glob="real_*.jsonl")

    assert report.raw_records == 3
    assert report.topic_coverage.get("fundamental", 0) == 1
    assert report.topic_coverage.get("news", 0) == 1
    assert report.source_diversity == 2
    assert report.doc_length_distribution["lt_1000"] == 1
    assert report.doc_length_distribution["1000_5000"] == 1
    assert report.doc_length_distribution["ge_5000"] == 1
    assert report.noise_ratio_raw > 0
