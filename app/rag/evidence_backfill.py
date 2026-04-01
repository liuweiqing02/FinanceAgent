from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig
from app.rag.knowledge_builder import build_knowledge_base
from app.rag.real_collectors import collect_real_raw_data


@dataclass(slots=True)
class EvidenceCompleteness:
    ticker: str
    fundamental_metrics: int
    valuation_metrics: int
    news_events: int
    fundamental_ok: bool
    valuation_ok: bool
    news_ok: bool

    @property
    def ok(self) -> bool:
        return self.fundamental_ok and self.valuation_ok and self.news_ok


def evaluate_kb_completeness(
    kb_dir: Path,
    ticker: str,
    *,
    min_fundamental_metrics: int,
    min_valuation_metrics: int,
    min_news_events: int,
) -> EvidenceCompleteness:
    t = ticker.upper().strip()
    f_txt = _read(kb_dir / f"{t}_fundamental.md")
    v_txt = _read(kb_dir / f"{t}_valuation.md")
    n_txt = _read(kb_dir / f"{t}_news.md")

    fund = _count_non_na_fields(
        f_txt,
        ["营收增速", "利润增速", "毛利率", "营业利润率", "净利率", "ROE", "自由现金流", "总现金"],
    )
    val = _count_non_na_fields(
        v_txt,
        ["PE", "Forward PE", "P/B", "Trailing PE", "price_to_book", "营收增速", "利润增速"],
    )
    news = _count_news_events(n_txt)

    return EvidenceCompleteness(
        ticker=t,
        fundamental_metrics=fund,
        valuation_metrics=val,
        news_events=news,
        fundamental_ok=fund >= min_fundamental_metrics,
        valuation_ok=val >= min_valuation_metrics,
        news_ok=news >= min_news_events,
    )


def backfill_kb_until_ready(
    config: AppConfig,
    ticker: str,
    *,
    sec_user_agent: str,
    max_rounds: int = 2,
) -> EvidenceCompleteness:
    rounds = max(1, int(max_rounds))
    t = ticker.upper().strip()

    latest = evaluate_kb_completeness(
        config.knowledge_base_dir,
        t,
        min_fundamental_metrics=config.fundamental_min_metrics,
        min_valuation_metrics=config.valuation_min_metrics,
        min_news_events=config.news_min_events,
    )

    for i in range(rounds):
        if latest.ok:
            return latest

        filing_limit = 4 + i * 2
        news_limit = 4 + i * 2
        collect_real_raw_data(
            raw_dir=config.raw_data_dir,
            tickers=[t],
            sec_user_agent=sec_user_agent,
            filing_limit=filing_limit,
            news_limit=news_limit,
        )
        build_knowledge_base(config.raw_data_dir, config.knowledge_base_dir, include_glob="real_*.jsonl")

        latest = evaluate_kb_completeness(
            config.knowledge_base_dir,
            t,
            min_fundamental_metrics=config.fundamental_min_metrics,
            min_valuation_metrics=config.valuation_min_metrics,
            min_news_events=config.news_min_events,
        )

    return latest


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _count_non_na_fields(text: str, keys: list[str]) -> int:
    if not text.strip():
        return 0
    found: set[str] = set()
    for k in keys:
        m = re.search(rf"{re.escape(k)}\s*[=:：]\s*([^，。;\n]+)", text, flags=re.IGNORECASE)
        if not m:
            continue
        v = m.group(1).strip()
        if v and v.upper() not in {"N/A", "NA", "NONE", "NULL"}:
            found.add(k)
    return len(found)


def _count_news_events(text: str) -> int:
    if not text.strip():
        return 0
    by_event_header = len(re.findall(r"^###\s*事件\d+", text, flags=re.MULTILINE))
    by_pub = len(re.findall(r"发布时间\s*[:：]", text))
    return max(by_event_header, by_pub)
