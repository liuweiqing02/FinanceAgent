from __future__ import annotations

from app.agents.base import BaseAgent, _join_evidence
from app.models.schemas import AgentOutput, Evidence


class FundamentalAgent(BaseAgent):
    name = "基本面"

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        body = _join_evidence(evidence)
        return AgentOutput(
            name=self.name,
            analysis=f"{ticker} 的基本面关注盈利质量、现金流与业务结构。\n{body}",
            conclusions=[f"{ticker} 基本面具有可跟踪性，需持续观察利润率变化。"],
        )


class TechnicalAgent(BaseAgent):
    name = "技术面"

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        body = _join_evidence(evidence)
        return AgentOutput(
            name=self.name,
            analysis=f"{ticker} 的技术面关注趋势和回撤风险。\n{body}",
            conclusions=[f"{ticker} 技术面偏趋势交易，需设置止损位。"],
        )


class ValuationAgent(BaseAgent):
    name = "估值"

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        body = _join_evidence(evidence)
        return AgentOutput(
            name=self.name,
            analysis=f"{ticker} 的估值需要结合成长性与安全边际。\n{body}",
            conclusions=[f"{ticker} 当前估值需结合业绩兑现节奏判断。"],
        )


class NewsAgent(BaseAgent):
    name = "新闻"

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        body = _join_evidence(evidence)
        return AgentOutput(
            name=self.name,
            analysis=f"{ticker} 的新闻面可能引发短期预期波动。\n{body}",
            conclusions=[f"{ticker} 新闻扰动较强，建议动态复核假设。"],
        )
