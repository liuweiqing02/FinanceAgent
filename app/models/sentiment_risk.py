from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SentimentRiskResult:
    sentiment: str
    risk_level: str
    score: float


def infer_sentiment_and_risk(text: str) -> SentimentRiskResult:
    """轻量情感与风险推理，便于离线运行与面试讲解。"""

    t = text.lower()
    pos = sum(k in t for k in ["improve", "growth", "increase", "稳定", "提升", "增速"])
    neg = sum(k in t for k in ["risk", "decline", "drop", "回撤", "波动", "监管"])

    score = (pos - neg) / max(pos + neg, 1)
    if score >= 0.25:
        sentiment = "正向"
    elif score <= -0.25:
        sentiment = "负向"
    else:
        sentiment = "中性"

    if neg >= pos + 2:
        risk = "高"
    elif neg > pos:
        risk = "中"
    else:
        risk = "低"

    return SentimentRiskResult(sentiment=sentiment, risk_level=risk, score=round(score, 3))
