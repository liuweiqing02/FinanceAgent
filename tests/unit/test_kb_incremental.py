import json
from pathlib import Path

from app.rag.knowledge_builder import build_knowledge_base


def test_build_knowledge_base_writes_manifest_and_updates_hash(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    kb = tmp_path / "kb"
    raw.mkdir()

    row = {
        "ticker": "AAPL",
        "topic": "news",
        "title": "n1",
        "content": "content a",
        "source": "x",
        "url": "u",
        "published_at": "2026-03-01",
    }
    (raw / "a.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    outs1 = build_knowledge_base(raw, kb)
    assert any(p.name == "AAPL_news.md" for p in outs1)

    manifest = json.loads((kb / "_kb_manifest.json").read_text(encoding="utf-8"))
    h1 = manifest["AAPL_news"]["hash"]

    row["content"] = "content b updated"
    (raw / "a.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    build_knowledge_base(raw, kb)

    manifest2 = json.loads((kb / "_kb_manifest.json").read_text(encoding="utf-8"))
    h2 = manifest2["AAPL_news"]["hash"]
    assert h1 != h2
