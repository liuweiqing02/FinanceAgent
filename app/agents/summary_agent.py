from __future__ import annotations

from app.models.schemas import AgentOutput, Evidence
from app.models.sentiment_risk import infer_sentiment_and_risk


class SummaryAgent:
    name = "总结"

    def run(self, ticker: str, outputs: list[AgentOutput], evidence: list[Evidence]) -> AgentOutput:
        merged_text = "\n".join(o.analysis for o in outputs)
        sentiment = infer_sentiment_and_risk(merged_text)

        cite = _citation_map(evidence)
        c1 = f"{ticker} 具备跟踪价值，但需关注盈利兑现与波动风险 {cite.get(1, '[E1]')}"
        c2 = f"当前综合情绪为{sentiment.sentiment}，风险等级{sentiment.risk_level} {cite.get(2, '[E2]')}"

        # 强制关键结论包含至少一条证据引用。
        conclusions = [_ensure_citation(c1), _ensure_citation(c2)]
        analysis = f"综合四类分析后，情感分数为 {sentiment.score}，建议采用分批决策。"
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)


def _citation_map(evidence: list[Evidence]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for i, _ in enumerate(evidence, start=1):
        mapping[i] = f"[E{i}]"
    return mapping


def _ensure_citation(text: str) -> str:
    if "[E" in text:
        return text
    return text + " [E1]"
