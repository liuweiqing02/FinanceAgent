from __future__ import annotations

import argparse
from pathlib import Path

from app.agents.orchestrator import FinanceResearchOrchestrator
from app.config import get_config
from app.rag.evidence_backfill import backfill_kb_until_ready, evaluate_kb_completeness
from app.rag.knowledge_builder import build_knowledge_base, ensure_sample_raw_data
from app.rag.real_collectors import collect_real_raw_data


def _has_real_raw(raw_dir: Path) -> bool:
    for p in raw_dir.glob("real_*.jsonl"):
        if p.exists() and p.stat().st_size > 0:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="金融智能投顾研报系统")
    parser.add_argument("--ticker", default="AAPL", help="股票代码")
    parser.add_argument("--kb-mode", default="auto", choices=["auto", "real", "sample"], help="知识库构建模式")
    parser.add_argument("--sec-user-agent", default=None, help="SEC 抓取 User-Agent，默认读取 SEC_USER_AGENT")
    parser.add_argument("--evidence-backfill", action="store_true", help="生成前执行证据补齐与完整度检查")
    parser.add_argument("--backfill-rounds", type=int, default=2, help="证据补齐最大轮数")
    args = parser.parse_args()

    config = get_config()
    mode = args.kb_mode
    include_glob = "*.jsonl"
    if mode in {"auto", "real"}:
        try:
            stats = collect_real_raw_data(
                raw_dir=config.raw_data_dir,
                tickers=[args.ticker.upper()],
                sec_user_agent=args.sec_user_agent or config.sec_user_agent,
                filing_limit=4,
                news_limit=8,
            )
            print(
                "真实数据采集完成: "
                f"filings={stats.filings}, news={stats.news}, technical={stats.technical}, out={stats.output_file}"
            )
            mode = "real"
            include_glob = "real_*.jsonl"
        except Exception as exc:  # noqa: BLE001
            if args.kb_mode == "real":
                raise
            if _has_real_raw(config.raw_data_dir):
                print(f"真实采集失败，改为复用已有真实数据。原因: {exc}")
                mode = "real"
                include_glob = "real_*.jsonl"
            else:
                print(f"真实采集失败，自动降级样例模式。原因: {exc}")
                mode = "sample"

    if mode == "sample":
        ensure_sample_raw_data(config.raw_data_dir)
        include_glob = "finance_events.jsonl"

    build_knowledge_base(config.raw_data_dir, config.knowledge_base_dir, include_glob=include_glob)

    if args.evidence_backfill:
        sec_ua = args.sec_user_agent or config.sec_user_agent
        comp = backfill_kb_until_ready(
            config,
            args.ticker.upper(),
            sec_user_agent=sec_ua,
            max_rounds=args.backfill_rounds,
        )
        print(
            "证据完整度: "
            f"fundamental={comp.fundamental_metrics}({'ok' if comp.fundamental_ok else 'missing'}), "
            f"valuation={comp.valuation_metrics}({'ok' if comp.valuation_ok else 'missing'}), "
            f"news={comp.news_events}({'ok' if comp.news_ok else 'missing'})"
        )
    else:
        comp = evaluate_kb_completeness(
            config.knowledge_base_dir,
            args.ticker.upper(),
            min_fundamental_metrics=config.fundamental_min_metrics,
            min_valuation_metrics=config.valuation_min_metrics,
            min_news_events=config.news_min_events,
        )
        print(
            "证据完整度(未补齐): "
            f"fundamental={comp.fundamental_metrics}, valuation={comp.valuation_metrics}, news={comp.news_events}"
        )

    orchestrator = FinanceResearchOrchestrator(config)
    report = orchestrator.run(args.ticker.upper())

    out = Path(config.report_output_dir) / f"{args.ticker.upper()}_report.md"
    out.write_text(report.markdown, encoding="utf-8")
    print(f"研报已生成: {out}")
    print(f"证据条数: {len(report.evidence)}")


if __name__ == "__main__":
    main()
