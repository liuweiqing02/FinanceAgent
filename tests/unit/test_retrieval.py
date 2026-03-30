from pathlib import Path

from app.config import AppConfig
from app.infra.logger import JsonlLogger
from app.rag.pipeline import RagPipeline, ensure_sample_knowledge


def test_retrieve_returns_evidence(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    chroma_dir = tmp_path / "chroma"
    ensure_sample_knowledge(kb)

    cfg = AppConfig(
        knowledge_base_dir=kb,
        chroma_persist_dir=chroma_dir,
        report_output_dir=tmp_path / "out",
    )
    logger = JsonlLogger(tmp_path / "trace.jsonl")
    rag = RagPipeline(cfg, logger)
    rag.build_index()

    items = rag.retrieve("AAPL 基本面 风险")
    assert items
    assert items[0].source_id
