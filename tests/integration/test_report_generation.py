from pathlib import Path

from app.agents.orchestrator import FinanceResearchOrchestrator
from app.config import AppConfig
from app.rag.pipeline import ensure_sample_knowledge


def test_integration_report_generation(tmp_path: Path) -> None:
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
    assert "综合结论" in bundle.markdown
    assert "[E1]" in bundle.markdown
    assert len(bundle.evidence) >= 1
