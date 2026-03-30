from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_config
from app.rag.knowledge_builder import build_knowledge_base, ensure_sample_raw_data


def main() -> None:
    parser = argparse.ArgumentParser(description="构建金融知识库（原始数据 -> Markdown知识库）")
    parser.add_argument("--raw-dir", default=None, help="原始数据目录，默认读取配置 RAW_DATA_DIR")
    parser.add_argument("--kb-dir", default=None, help="知识库输出目录，默认读取配置 KNOWLEDGE_BASE_DIR")
    args = parser.parse_args()

    cfg = get_config()
    raw_dir = cfg.raw_data_dir if args.raw_dir is None else Path(args.raw_dir)
    kb_dir = cfg.knowledge_base_dir if args.kb_dir is None else Path(args.kb_dir)

    ensure_sample_raw_data(raw_dir)
    outputs = build_knowledge_base(raw_dir, kb_dir)

    print(f"知识库构建完成: {len(outputs)} 个文件")
    for p in outputs:
        print(f"- {p}")


if __name__ == "__main__":
    main()
