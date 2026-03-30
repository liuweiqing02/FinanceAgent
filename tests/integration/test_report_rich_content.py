from pathlib import Path

from app.agents.orchestrator import FinanceResearchOrchestrator
from app.config import AppConfig
from app.rag.pipeline import ensure_sample_knowledge


def _extract_section(markdown: str, title: str, next_title: str) -> str:
    start = markdown.index(title)
    end = markdown.index(next_title, start) if next_title in markdown[start + len(title):] else len(markdown)
    return markdown[start:end]


def test_integration_rich_sections_and_summary_citations(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    ensure_sample_knowledge(kb)
    cfg = AppConfig(
        knowledge_base_dir=kb,
        chroma_persist_dir=tmp_path / "chroma",
        report_output_dir=tmp_path / "out",
        enable_langgraph=False,
    )

    orchestrator = FinanceResearchOrchestrator(cfg)
    bundle = orchestrator.run("AAPL")
    md = bundle.markdown

    assert "### 基本面" in md
    assert "### 技术面" in md
    assert "### 估值" in md
    assert "### 新闻" in md

    sec_f = _extract_section(md, "### 基本面", "### 技术面")
    sec_t = _extract_section(md, "### 技术面", "### 估值")
    sec_v = _extract_section(md, "### 估值", "### 新闻")
    sec_n = _extract_section(md, "### 新闻", "## 综合结论")

    for sec in [sec_f, sec_t, sec_v, sec_n]:
        assert ("框架判断" in sec) or ("证据解读" in sec)
        assert sec.count("\n") >= 8

    summary = _extract_section(md, "## 综合结论", "## 证据清单")
    cited_lines = [line for line in summary.splitlines() if line.startswith("-") and "[E" in line]
    assert len(cited_lines) >= 4
