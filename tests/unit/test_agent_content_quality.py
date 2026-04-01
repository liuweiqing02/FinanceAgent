from app.agents import specialists
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


def test_news_select_evidence_does_not_fallback_to_non_news() -> None:
    evidence = [
        Evidence(
            source_id="AAPL_news",
            title="AAPL_news.md",
            content="AAPL news event. source=rss timestamp=2026-04-01T12:00:00+00:00. This is a readable event block.",
            score=0.9,
            metadata={"chunk_id": "AAPL_news#c1"},
        ),
        Evidence(
            source_id="AAPL_fundamental",
            title="AAPL_fundamental.md",
            content="Fundamental readable content with lots of text for fallback testing." * 6,
            score=0.8,
            metadata={"chunk_id": "AAPL_fundamental#c1"},
        ),
    ]

    refs = specialists._select_evidence(evidence, "news", limit=3, ticker="AAPL")
    assert len(refs) == 1
    assert refs[0][1].source_id == "AAPL_news"


def test_news_event_signal_count_accepts_structured_news_chunks() -> None:
    refs = [
        (
            1,
            Evidence(
                source_id="AAPL_news",
                title="AAPL_news.md",
                content="### 事件1 2026-04-01T11:30:02+00:00 - 来源类型: google_news_rss - 内容解读: Apple launches update.",
                score=0.9,
                metadata={"chunk_id": "AAPL_news#c2"},
            ),
        ),
        (
            2,
            Evidence(
                source_id="AAPL_news",
                title="AAPL_news.md",
                content="### 事件2 2026-04-01T10:57:00+00:00 - 来源类型: yahoo_finance_rss - 内容解读: Apple turns 50.",
                score=0.8,
                metadata={"chunk_id": "AAPL_news#c3"},
            ),
        ),
    ]

    assert specialists._news_event_signal_count(refs) >= 2