from __future__ import annotations

import argparse
from pathlib import Path

from app.agents.orchestrator import FinanceResearchOrchestrator
from app.config import get_config
from app.rag.pipeline import ensure_sample_knowledge


def main() -> None:
    parser = argparse.ArgumentParser(description="金融智能投顾研报系统")
    parser.add_argument("--ticker", default="AAPL", help="股票代码")
    args = parser.parse_args()

    config = get_config()
    ensure_sample_knowledge(config.knowledge_base_dir, config.raw_data_dir)

    orchestrator = FinanceResearchOrchestrator(config)
    report = orchestrator.run(args.ticker.upper())

    out = Path(config.report_output_dir) / f"{args.ticker.upper()}_report.md"
    out.write_text(report.markdown, encoding="utf-8")
    print(f"研报已生成: {out}")
    print(f"证据条数: {len(report.evidence)}")


if __name__ == "__main__":
    main()
