from __future__ import annotations

from datetime import datetime

from app.models.schemas import AgentOutput, Evidence, ReportBundle


def build_markdown_report(ticker: str, outputs: list[AgentOutput], summary: AgentOutput, evidence: list[Evidence]) -> ReportBundle:
    """生成带引用的 Markdown 研报。"""

    sections = [
        f"# {ticker} 智能投顾研报",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 多Agent分析",
    ]

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
            "## 证据清单",
        ]
    )

    for i, ev in enumerate(evidence, start=1):
        sections.append(f"- [E{i}] {ev.title} | score={ev.score} | source={ev.source_id}")
        sections.append(f"  - 摘要：{ev.content[:160].replace(chr(10), ' ')}")

    markdown = "\n".join(sections).strip() + "\n"
    return ReportBundle(ticker=ticker, markdown=markdown, evidence=evidence)
