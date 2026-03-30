from app.agents.specialists import FundamentalAgent, NewsAgent, TechnicalAgent, ValuationAgent
from app.agents.summary_agent import SummaryAgent
from app.models.schemas import AgentOutput, Evidence


def _mock_evidence() -> list[Evidence]:
    return [
        Evidence(source_id=f"s{i}", title=f"doc{i}", content=f"content {i} growth risk", score=0.9 - i * 0.1)
        for i in range(1, 6)
    ]


def test_specialist_agents_content_quality() -> None:
    evidence = _mock_evidence()
    ticker = "AAPL"
    agents = [FundamentalAgent(), TechnicalAgent(), ValuationAgent(), NewsAgent()]

    for agent in agents:
        out = agent.run(ticker, evidence)
        assert "框架判断" in out.analysis
        assert "证据解读" in out.analysis
        assert "风险提示" in out.analysis
        assert len(out.conclusions) >= 3
        assert all("[E" in c for c in out.conclusions)


def test_summary_agent_content_quality() -> None:
    evidence = _mock_evidence()
    ticker = "AAPL"
    specialists = [
        FundamentalAgent().run(ticker, evidence),
        TechnicalAgent().run(ticker, evidence),
        ValuationAgent().run(ticker, evidence),
        NewsAgent().run(ticker, evidence),
    ]

    summary = SummaryAgent().run(ticker, specialists, evidence)
    assert isinstance(summary, AgentOutput)
    assert len(summary.conclusions) >= 4
    assert all("[E" in c for c in summary.conclusions)
