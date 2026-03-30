from app.models.sentiment_risk import infer_sentiment_and_risk


def test_sentiment_risk_positive() -> None:
    r = infer_sentiment_and_risk("growth improve 提升 增速")
    assert r.sentiment in {"正向", "中性"}


def test_sentiment_risk_negative() -> None:
    r = infer_sentiment_and_risk("risk decline 回撤 波动 监管")
    assert r.risk_level in {"中", "高"}
