from app.models.schemas import Evidence
from app.rag.pipeline import _postprocess_evidence


def _ev(source_id: str, content: str, score: float = 1.0) -> Evidence:
    return Evidence(source_id=source_id, title=source_id, content=content, score=score, metadata={})


def test_postprocess_filters_noise_and_dedups_source() -> None:
    items = [
        _ev("A", "us-gaap taxonomy namespace xbrl member linkbase", 9.0),
        _ev("B", "正常财务解读，营收与利润同比改善，现金流稳定。" * 4, 8.0),
        _ev("B", "同源重复证据，应该被去重。" * 10, 7.0),
        _ev("C", "技术面显示价格站上SMA20，量能温和放大。" * 4, 6.0),
    ]

    out = _postprocess_evidence(items, max_items=3)
    assert len(out) == 2
    assert out[0].source_id == "B"
    assert out[1].source_id == "C"


def test_postprocess_filters_empty_fundamental_snapshot() -> None:
    items = [
        _ev("mcp_fundamental_snapshot", "营收增速=N/A，利润增速=N/A，毛利率=N/A，ROE=N/A，自由现金流=N/A，总现金=N/A，Forward PE=N/A，P/B=N/A", 9.0),
        _ev("mcp_market_snapshot", "最新价=100，PE=20，PB=3，日内涨跌幅=0.8%", 8.0),
    ]

    out = _postprocess_evidence(items, max_items=2)
    assert len(out) == 1
    assert out[0].source_id == "mcp_market_snapshot"
