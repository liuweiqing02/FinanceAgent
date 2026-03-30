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


def test_build_knowledge_base_can_filter_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    kb = tmp_path / "kb"
    raw.mkdir(parents=True, exist_ok=True)

    (raw / "a.jsonl").write_text(
        '{"ticker":"AAPL","topic":"news","title":"n1","content":"x"}\n',
        encoding="utf-8",
    )
    (raw / "b.jsonl").write_text(
        '{"ticker":"TSLA","topic":"news","title":"n2","content":"这个内容长度足够用于测试过滤能力并确保不会混入其他文件"}\n',
        encoding="utf-8",
    )

    outputs = build_knowledge_base(raw, kb, include_glob="b.jsonl")
    names = sorted(p.name for p in outputs)
    assert names == ["TSLA_news.md"]
