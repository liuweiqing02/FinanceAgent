from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from app.agents.specialists import FundamentalAgent, NewsAgent, TechnicalAgent, ValuationAgent
from app.agents.summary_agent import SummaryAgent
from app.config import AppConfig
from app.infra.logger import JsonlLogger
from app.mcp.protocol import MCPToolRegistry, ToolRequest
from app.mcp.tools import get_market_snapshot, load_news_file
from app.models.schemas import AgentOutput, Evidence, ReportBundle
from app.rag.pipeline import RagPipeline
from app.reports.writer import build_markdown_report


@dataclass(slots=True)
class PipelineState:
    ticker: str
    query: str
    evidence: list[Evidence]
    outputs: list[AgentOutput]


class FinanceResearchOrchestrator:
    """多 Agent 研报编排器。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = JsonlLogger(Path("logs/rag_trace.jsonl"))
        self.rag = RagPipeline(config, self.logger)
        self.registry = MCPToolRegistry()
        self.registry.register("market_snapshot", get_market_snapshot)
        self.registry.register("load_news_file", load_news_file)

        self.fundamental = FundamentalAgent()
        self.technical = TechnicalAgent()
        self.valuation = ValuationAgent()
        self.news = NewsAgent()
        self.summary = SummaryAgent()

    def run(self, ticker: str) -> ReportBundle:
        query = f"{ticker} 基本面 技术面 估值 新闻 风险"
        evidence = self.rag.retrieve(query, top_k=self.config.top_k)

        market = self.registry.execute(ToolRequest(tool_name="market_snapshot", args={"ticker": ticker}))
        self.logger.log("mcp_call", {"tool": "market_snapshot", "ok": market.ok, "data": market.data})

        outputs = self._run_agents(ticker, evidence)
        summary = self.summary.run(ticker, outputs, evidence)
        bundle = build_markdown_report(ticker, outputs, summary, evidence)

        self.logger.log(
            "generation",
            {
                "ticker": ticker,
                "conclusions": summary.conclusions,
                "evidence_count": len(evidence),
            },
        )
        return bundle

    def _run_agents(self, ticker: str, evidence: list[Evidence]) -> list[AgentOutput]:
        if self.config.enable_langgraph:
            try:
                return self._order_outputs(self._run_with_langgraph(ticker, evidence))
            except Exception as exc:  # noqa: BLE001
                self.logger.log("langgraph_fallback", {"reason": str(exc)})
        return self._order_outputs(self._run_in_parallel(ticker, evidence))

    def _run_in_parallel(self, ticker: str, evidence: list[Evidence]) -> list[AgentOutput]:
        agents = [self.fundamental, self.technical, self.valuation, self.news]
        outputs: list[AgentOutput] = []
        with ThreadPoolExecutor(max_workers=len(agents)) as pool:
            futures = {pool.submit(agent.run, ticker, evidence): agent.name for agent in agents}
            for future in as_completed(futures):
                outputs.append(future.result())
        return outputs

    def _order_outputs(self, outputs: list[AgentOutput]) -> list[AgentOutput]:
        order = {"基本面": 0, "技术面": 1, "估值": 2, "新闻": 3}
        return sorted(outputs, key=lambda x: order.get(x.name, 99))

    def _run_with_langgraph(self, ticker: str, evidence: list[Evidence]) -> list[AgentOutput]:
        from operator import add
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph
        from typing_extensions import Annotated

        class S(TypedDict):
            outputs: Annotated[list[AgentOutput], add]

        graph = StateGraph(S)

        def run_fundamental(state: S) -> S:
            return {"outputs": [self.fundamental.run(ticker, evidence)]}

        def run_technical(state: S) -> S:
            return {"outputs": [self.technical.run(ticker, evidence)]}

        def run_valuation(state: S) -> S:
            return {"outputs": [self.valuation.run(ticker, evidence)]}

        def run_news(state: S) -> S:
            return {"outputs": [self.news.run(ticker, evidence)]}

        graph.add_node("fundamental", run_fundamental)
        graph.add_node("technical", run_technical)
        graph.add_node("valuation", run_valuation)
        graph.add_node("news", run_news)

        # 四个专家 Agent 从起点并行执行，再统一汇聚结束。
        graph.add_edge(START, "fundamental")
        graph.add_edge(START, "technical")
        graph.add_edge(START, "valuation")
        graph.add_edge(START, "news")
        graph.add_edge("fundamental", END)
        graph.add_edge("technical", END)
        graph.add_edge("valuation", END)
        graph.add_edge("news", END)

        app = graph.compile()
        state = app.invoke({"outputs": []})
        return state["outputs"]
