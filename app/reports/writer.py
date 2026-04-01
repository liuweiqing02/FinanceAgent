from __future__ import annotations

from datetime import datetime

from app.models.schemas import AgentOutput, Evidence, ReportBundle


def build_markdown_report(
    ticker: str,
    outputs: list[AgentOutput],
    summary: AgentOutput,
    evidence: list[Evidence],
    market_snapshot: dict | None = None,
) -> ReportBundle:
    """生成带引用的动态 Markdown 研报（证据驱动章节）。"""

    sections = [
        f"# {ticker} 智能投顾研报",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 执行摘要",
        _first_sentence(summary.analysis),
        "",
    ]

    if market_snapshot and _market_snapshot_ok(market_snapshot):
        sections.extend(
            [
                "## 公司概况",
                f"- 标的：{market_snapshot.get('ticker', ticker)}",
                f"- 最新价格：{market_snapshot.get('price')}",
                f"- 估值指标：PE={market_snapshot.get('pe')}，PB={market_snapshot.get('pb')}",
                f"- 数据时间：{market_snapshot.get('timestamp', 'N/A')}",
                "",
            ]
        )

    sections.append("## 多Agent分析")
    for out in outputs:
        sections.extend(
            [
                f"### {out.name}",
                out.analysis,
                "",
                "结论：",
                *[f"- {c}" for c in out.conclusions],
                "",
            ]
        )

    sections.extend(
        [
            "## 综合结论",
            summary.analysis,
            "",
            *[f"- {c}" for c in summary.conclusions],
            "",
        ]
    )

    quality = _assess_evidence_quality(evidence)
    sections.extend(
        [
            "## 数据充分性评估",
            f"- 证据总数：{quality['total']}",
            f"- 可读证据数：{quality['readable']}",
            f"- 高质量证据数：{quality['high_quality']}",
            f"- 主题覆盖：{quality['coverage'] or '不足'}",
            "",
        ]
    )

    if _can_emit_trade_plan(evidence, market_snapshot):
        sections.extend(
            [
                "## 交易计划（证据充分时生成）",
                "- 策略类型：分批建仓，分层风控。",
                "- 建议动作：先小仓位试探，信号确认后再增配。",
                "- 风控要求：跌破关键支撑位或出现负面事件扩散时，降低风险敞口。",
                "",
            ]
        )

    sections.append("## 证据清单")
    for i, ev in enumerate(evidence, start=1):
        sections.append(f"- [E{i}] {ev.title} | score={ev.score} | source={ev.source_id}")
        sections.append(f"  - 摘要：{_readable_snippet(ev.content)}")

    markdown = "\n".join(sections).strip() + "\n"
    return ReportBundle(ticker=ticker, markdown=markdown, evidence=evidence)


def _first_sentence(text: str) -> str:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return lines[0] if lines else "本次报告基于多 Agent 与检索证据自动生成。"


def _market_snapshot_ok(snapshot: dict) -> bool:
    need = ["price", "pe", "pb"]
    return all(k in snapshot and snapshot.get(k) is not None for k in need)


def _assess_evidence_quality(evidence: list[Evidence]) -> dict:
    readable = [ev for ev in evidence if _is_readable(ev.content)]
    high_quality = [ev for ev in readable if ev.score >= 1.0]

    coverage_tags: list[str] = []
    for ev in evidence:
        t = f"{ev.title} {ev.source_id}".lower()
        if "fundamental" in t:
            coverage_tags.append("基本面")
        if "technical" in t:
            coverage_tags.append("技术面")
        if "valuation" in t:
            coverage_tags.append("估值")
        if "news" in t:
            coverage_tags.append("新闻")

    coverage = "、".join(sorted(set(coverage_tags)))
    return {
        "total": len(evidence),
        "readable": len(readable),
        "high_quality": len(high_quality),
        "coverage": coverage,
    }


def _can_emit_trade_plan(evidence: list[Evidence], market_snapshot: dict | None) -> bool:
    if market_snapshot is None or not _market_snapshot_ok(market_snapshot):
        return False

    quality = _assess_evidence_quality(evidence)
    if quality["high_quality"] < 3:
        return False

    tags = 0
    for ev in evidence:
        t = f"{ev.title} {ev.source_id}".lower()
        if ("technical" in t or "valuation" in t) and _is_readable(ev.content) and ev.score >= 1.0:
            tags += 1
    return tags >= 2


def _is_readable(text: str) -> bool:
    body = text.strip()
    if len(body) < 40:
        return False
    low = body.lower()
    bad_tokens = ["xbrl", "us-gaap", "v95cux", "http", "0000", "taxonomy", "false fy", "p7y", "p2y"]
    hit = sum(1 for x in bad_tokens if x in low)
    if hit >= 2:
        return False
    if low.count("n/a") >= 8:
        return False
    return True


def _readable_snippet(text: str, max_len: int = 180) -> str:
    clean = " ".join(text.replace("\n", " ").split())
    if len(clean) > max_len:
        return clean[:max_len] + "..."
    return clean
