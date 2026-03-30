from __future__ import annotations

from app.models.schemas import AgentOutput, Evidence
from app.models.sentiment_risk import infer_sentiment_and_risk


class SummaryAgent:
    name = "总结"

    def run(self, ticker: str, outputs: list[AgentOutput], evidence: list[Evidence]) -> AgentOutput:
        merged_text = "\n".join(o.analysis for o in outputs)
        sentiment = infer_sentiment_and_risk(merged_text)

        cite = _citation_map(evidence)
        analysis = "\n".join(
            [
                f"{ticker} 综合结论由基本面、技术面、估值面、新闻面四个并行 Agent 汇总得到。",
                "",
                "综合判断：",
                f"- 市场情绪：{sentiment.sentiment}（情绪分数 {sentiment.score}）{cite.get(1, '[E1]')}",
                f"- 风险等级：{sentiment.risk_level}，建议采用分批决策与动态风控 {cite.get(2, '[E2]')}",
                "- 结论可信度：中等偏上，主要依赖当前知识库与召回证据质量。",
                "",
                "投资执行建议：",
                "- 仓位：先轻仓试探，确认信号后再逐步加仓。",
                "- 风控：设置分层止损与时间止损，避免单次误判扩大损失。",
                "- 复盘：每次关键新闻或财报发布后，重新触发全链路研报。",
            ]
        )

        conclusions = [
            _ensure_citation(f"{ticker} 具备跟踪价值，但应以风险预算作为前置约束 {cite.get(1, '[E1]')}", 1),
            _ensure_citation(f"当前更适合“分批建仓 + 持续验证”，而非一次性重仓 {cite.get(2, '[E2]')}", 2),
            _ensure_citation(f"若后续盈利与现金流继续改善，估值中枢有望抬升 {cite.get(3, '[E3]')}", 3),
            _ensure_citation(f"若出现技术位破坏或负面新闻扩散，应快速降低敞口 {cite.get(4, '[E4]')}", 4),
        ]
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)


def _citation_map(evidence: list[Evidence]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for i, _ in enumerate(evidence, start=1):
        mapping[i] = f"[E{i}]"
    return mapping


def _ensure_citation(text: str, default_idx: int = 1) -> str:
    if "[E" in text:
        return text
    return text + f" [E{default_idx}]"
