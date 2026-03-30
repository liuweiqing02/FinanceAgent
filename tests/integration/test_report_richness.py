from pathlib import Path

from app.agents.orchestrator import FinanceResearchOrchestrator
from app.config import AppConfig
from app.rag.pipeline import ensure_sample_knowledge


def _section_block(markdown: str, section: str) -> str:
    marker = f"### {section}"
    start = markdown.find(marker)
    assert start >= 0, f"未找到章节: {section}"

    next_positions = [
        p for p in [
            markdown.find("### ", start + len(marker)),
            markdown.find("## 综合结论", start + len(marker)),
        ] if p != -1
    ]
    end = min(next_positions) if next_positions else len(markdown)
    return markdown[start:end]


def test_integration_markdown_rich_sections_and_citations(tmp_path: Path) -> None:
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

    for section in ["基本面", "技术面", "估值", "新闻"]:
        block = _section_block(md, section)
        assert f"### {section}" in block
        assert "框架判断" in block or "证据解读" in block
        assert len([line for line in block.splitlines() if line.strip()]) >= 8

    assert "## 综合结论" in md
    summary_block = md[md.find("## 综合结论"): md.find("## 证据清单")]
    cited_lines = [line for line in summary_block.splitlines() if line.strip().startswith("-") and "[E" in line]
    assert len(cited_lines) >= 4
