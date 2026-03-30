from app.models.sentiment_risk import infer_news_sentiment_and_risk_5level, infer_sentiment_and_risk


def test_sentiment_risk_positive() -> None:
    r = infer_sentiment_and_risk("growth improve 提升 增速")
    assert r.sentiment in {"正向", "中性"}


def test_sentiment_risk_negative() -> None:
    r = infer_sentiment_and_risk("risk decline 回撤 波动 监管")
    assert r.risk_level in {"中", "高"}


def test_news_model_extreme_negative_high_risk() -> None:
    r = infer_news_sentiment_and_risk_5level("公司被监管调查并可能退市，财务造假风险上升，市场恐慌抛售")
    assert r.risk_level >= 4
    assert r.sentiment_level <= 2


def test_news_model_positive_low_risk() -> None:
    r = infer_news_sentiment_and_risk_5level("公司获政策支持并发布利好订单，增长改善，经营稳定")
    assert r.sentiment_level >= 4
    assert r.risk_level <= 3
