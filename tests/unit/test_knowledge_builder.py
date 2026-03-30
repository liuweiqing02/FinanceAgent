from pathlib import Path

from app.rag.knowledge_builder import build_knowledge_base, ensure_sample_raw_data


def test_build_knowledge_base_from_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    kb = tmp_path / "kb"

    ensure_sample_raw_data(raw)
    outputs = build_knowledge_base(raw, kb)

    assert len(outputs) >= 4
    text = outputs[0].read_text(encoding="utf-8")
    assert "## 关键事件与解读" in text
    assert "## 聚合结论" in text
