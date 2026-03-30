from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from app.agents.specialists import FundamentalAgent, NewsAgent, TechnicalAgent, ValuationAgent
from app.agents.summary_agent import SummaryAgent
from app.config import AppConfig
from app.infra.logger import JsonlLogger
from app.mcp.protocol import FASTMCP_AVAILABLE, FastMCPServerAdapter, LocalMCPServer, MultiServerMCPClient, ToolRequest
from app.mcp.tools import get_fundamental_snapshot, get_market_snapshot, load_news_file
from app.models.llm_client import build_llm_from_config
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

        self.mcp = MultiServerMCPClient()

        server_cls = FastMCPServerAdapter if FASTMCP_AVAILABLE else LocalMCPServer
        finance_server = server_cls("finance_data")
        finance_server.register_tool(
            name="market_snapshot",
            fn=get_market_snapshot,
            description="获取股票行情与估值快照",
            input_schema={"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        )
        finance_server.register_tool(
            name="fundamental_snapshot",
            fn=get_fundamental_snapshot,
            description="获取股票财务核心指标快照",
            input_schema={"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        )

        kb_server = server_cls("knowledge_base")
        kb_server.register_tool(
            name="load_news_file",
            fn=load_news_file,
            description="从知识库读取新闻文档",
            input_schema={
                "type": "object",
                "properties": {"base_dir": {"type": "string"}, "ticker": {"type": "string"}},
                "required": ["base_dir", "ticker"],
            },
        )

        self.mcp.register_server(finance_server)
        self.mcp.register_server(kb_server)

        self.llm = build_llm_from_config(config)
        self.fundamental = FundamentalAgent(self.llm)
        self.technical = TechnicalAgent(self.llm)
        self.valuation = ValuationAgent(self.llm)
        self.news = NewsAgent(self.llm)
        self.summary = SummaryAgent(self.llm)
        self.logger.log("llm_runtime", {"enabled": self.llm is not None, "model": config.llm_model})

    def run(self, ticker: str) -> ReportBundle:
        query = f"{ticker} 基本面 技术面 估值 新闻 风险"
        evidence = self.rag.retrieve(query, top_k=self.config.top_k)

        tools = [f"{t.server}.{t.name}" for t in self.mcp.list_tools()]
        self.logger.log("mcp_tools", {"count": len(tools), "tools": tools, "backend": "fastmcp" if FASTMCP_AVAILABLE else "local"})

        market = self.mcp.call_tool(ToolRequest(server="finance_data", tool_name="market_snapshot", args={"ticker": ticker}))
        fundamental = self.mcp.call_tool(ToolRequest(server="finance_data", tool_name="fundamental_snapshot", args={"ticker": ticker}))
        self.logger.log("mcp_call", {"server": market.server, "tool": market.tool_name, "ok": market.ok, "data": market.data})
        self.logger.log(
            "mcp_call",
            {"server": fundamental.server, "tool": fundamental.tool_name, "ok": fundamental.ok, "data": fundamental.data},
        )

        evidence = self._inject_mcp_evidence(evidence, market.data if market.ok else None, fundamental.data if fundamental.ok else None)

        outputs = self._run_agents(ticker, evidence)
        summary = self.summary.run(ticker, outputs, evidence)
        bundle = build_markdown_report(ticker, outputs, summary, evidence, market_snapshot=market.data if market.ok else None)

        self.logger.log(
            "generation",
            {
                "ticker": ticker,
                "conclusions": summary.conclusions,
                "evidence_count": len(evidence),
            },
        )
        return bundle

    def _inject_mcp_evidence(
        self,
        evidence: list[Evidence],
        market: dict | None,
        fundamental: dict | None,
    ) -> list[Evidence]:
        enhanced = list(evidence)
        if market and market.get("price") is not None:
            enhanced.insert(
                0,
                Evidence(
                    source_id="mcp_market_snapshot",
                    title=f"{market.get('ticker', '')} market snapshot",
                    content=(
                        f"最新价={_fmt_num(market.get('price'))} {market.get('currency', '')}，"
                        f"日内涨跌幅={_fmt_num(market.get('change_percent'))}% ，"
                        f"日高={_fmt_num(market.get('day_high'))}，"
                        f"日低={_fmt_num(market.get('day_low'))}，"
                        f"PE={_fmt_num(market.get('pe'))}，"
                        f"Forward PE={_fmt_num(market.get('forward_pe'))}，"
                        f"PB={_fmt_num(market.get('pb'))}，"
                        f"市值={_fmt_num(market.get('market_cap'))}。"
                    ),
                    score=5.0,
                    metadata={"from": "mcp", "tool": "market_snapshot", "server": "finance_data"},
                ),
            )

        if fundamental:
            enhanced.insert(
                1 if enhanced else 0,
                Evidence(
                    source_id="mcp_fundamental_snapshot",
                    title=f"{fundamental.get('ticker', '')} fundamental snapshot",
                    content=(
                        f"营收增速={_fmt_pct(fundamental.get('revenue_growth'))}，"
                        f"利润增速={_fmt_pct(fundamental.get('earnings_growth'))}，"
                        f"毛利率={_fmt_pct(fundamental.get('gross_margin'))}，"
                        f"营业利润率={_fmt_pct(fundamental.get('operating_margin'))}，"
                        f"净利率={_fmt_pct(fundamental.get('profit_margin'))}，"
                        f"ROE={_fmt_pct(fundamental.get('return_on_equity'))}，"
                        f"自由现金流={_fmt_num(fundamental.get('free_cash_flow'))}，"
                        f"总现金={_fmt_num(fundamental.get('total_cash'))}，"
                        f"Trailing PE={_fmt_num(fundamental.get('trailing_pe'))}，"
                        f"Forward PE={_fmt_num(fundamental.get('forward_pe'))}，"
                        f"P/B={_fmt_num(fundamental.get('price_to_book'))}。"
                    ),
                    score=4.8,
                    metadata={"from": "mcp", "tool": "fundamental_snapshot", "server": "finance_data"},
                ),
            )
        return enhanced

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


def _fmt_num(v: object) -> str:
    if v is None:
        return "N/A"
    try:
        num = float(v)
    except Exception:  # noqa: BLE001
        return "N/A"
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    return f"{num:.2f}"


def _fmt_pct(v: object) -> str:
    if v is None:
        return "N/A"
    try:
        num = float(v) * 100
    except Exception:  # noqa: BLE001
        return "N/A"
    return f"{num:.2f}%"





