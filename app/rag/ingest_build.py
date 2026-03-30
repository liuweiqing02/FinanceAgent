from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_config
from app.rag.knowledge_builder import build_knowledge_base, ensure_sample_raw_data
from app.rag.real_collectors import collect_real_raw_data


def main() -> None:
    parser = argparse.ArgumentParser(description="构建金融知识库（原始数据 -> Markdown知识库）")
    parser.add_argument("--raw-dir", default=None, help="原始数据目录，默认读取配置 RAW_DATA_DIR")
    parser.add_argument("--kb-dir", default=None, help="知识库输出目录，默认读取配置 KNOWLEDGE_BASE_DIR")
    parser.add_argument("--mode", default="auto", choices=["auto", "real", "sample"], help="采集模式")
    parser.add_argument("--tickers", default="AAPL,TSLA", help="股票代码列表，逗号分隔")
    parser.add_argument("--filing-limit", type=int, default=4, help="每个ticker最多抓取财报/公告条数")
    parser.add_argument("--news-limit", type=int, default=4, help="每个ticker最多抓取新闻条数")
    parser.add_argument(
        "--sec-user-agent",
        default=None,
        help="SEC 抓取 User-Agent，如 'FinanceAgent/1.0 your@email.com'，默认从 SEC_USER_AGENT 读取",
    )
    args = parser.parse_args()

    cfg = get_config()
    raw_dir = cfg.raw_data_dir if args.raw_dir is None else Path(args.raw_dir)
    kb_dir = cfg.knowledge_base_dir if args.kb_dir is None else Path(args.kb_dir)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    sec_user_agent = args.sec_user_agent or cfg.sec_user_agent

    include_glob = "*.jsonl"
    used_mode = args.mode
    if args.mode in {"auto", "real"}:
        try:
            stats = collect_real_raw_data(
                raw_dir=raw_dir,
                tickers=tickers,
                sec_user_agent=sec_user_agent,
                filing_limit=args.filing_limit,
                news_limit=args.news_limit,
            )
            print(
                "真实数据采集完成: "
                f"tickers={stats.tickers}, filings={stats.filings}, news={stats.news}, technical={stats.technical}, out={stats.output_file}"
            )
            used_mode = "real"
            include_glob = "real_*.jsonl"
        except Exception as exc:  # noqa: BLE001
            if args.mode == "real":
                raise
            print(f"真实采集失败，自动降级样例模式。原因: {exc}")
            used_mode = "sample"

    if used_mode == "sample":
        ensure_sample_raw_data(raw_dir)
        print("已生成样例原始数据。")
        include_glob = "finance_events.jsonl"

    outputs = build_knowledge_base(raw_dir, kb_dir, include_glob=include_glob)

    print(f"知识库构建完成: {len(outputs)} 个文件")
    for p in outputs:
        print(f"- {p}")


if __name__ == "__main__":
    main()
