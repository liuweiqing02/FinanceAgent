from app.models.sentiment_risk import NewsSentimentRiskResult, infer_news_sentiment_and_risk_5level


class _FakeEngine:
    def infer(self, text: str) -> NewsSentimentRiskResult:
        return NewsSentimentRiskResult(
            sentiment_level=4,
            sentiment_label="正面",
            risk_level=2,
            risk_label="低风险",
            sentiment_score=4.0,
            risk_score=2.0,
            reasons=["fake"],
        )


class _BrokenEngine:
    def infer(self, text: str) -> NewsSentimentRiskResult:
        raise RuntimeError("boom")


def test_news_infer_prefers_engine() -> None:
    r = infer_news_sentiment_and_risk_5level("任意文本", engine=_FakeEngine())
    assert r.sentiment_level == 4
    assert r.risk_level == 2


def test_news_infer_fallback_when_engine_failed() -> None:
    r = infer_news_sentiment_and_risk_5level("公司获政策支持并发布利好订单", engine=_BrokenEngine())
    assert 1 <= r.sentiment_level <= 5
    assert 1 <= r.risk_level <= 5
