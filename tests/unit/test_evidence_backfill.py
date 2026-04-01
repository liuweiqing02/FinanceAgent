from pathlib import Path

from app.rag.evidence_backfill import evaluate_kb_completeness


def test_evaluate_kb_completeness_counts_fields(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir(parents=True, exist_ok=True)

    (kb / "AAPL_fundamental.md").write_text(
        "营收增速=10.2%\n利润增速=8.1%\n毛利率=42.0%\nROE=N/A\n",
        encoding="utf-8",
    )
    (kb / "AAPL_valuation.md").write_text(
        "PE=20.1\nForward PE=18.3\nP/B=N/A\n",
        encoding="utf-8",
    )
    (kb / "AAPL_news.md").write_text(
        "### 事件1\n发布时间: 2026-03-01\n### 事件2\n发布时间: 2026-03-02\n",
        encoding="utf-8",
    )

    comp = evaluate_kb_completeness(
        kb,
        "AAPL",
        min_fundamental_metrics=3,
        min_valuation_metrics=2,
        min_news_events=2,
    )

    assert comp.fundamental_metrics >= 3
    assert comp.valuation_metrics >= 2
    assert comp.news_events >= 2
    assert comp.ok
