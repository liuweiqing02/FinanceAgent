from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SentimentRiskResult:
    sentiment: str
    risk_level: str
    score: float


@dataclass(slots=True)
class NewsSentimentRiskResult:
    """新闻专用 1~5 级情感与风险推理结果。"""

    sentiment_level: int
    sentiment_label: str
    risk_level: int
    risk_label: str
    sentiment_score: float
    risk_score: float
    reasons: list[str] = field(default_factory=list)


def infer_sentiment_and_risk(text: str) -> SentimentRiskResult:
    """轻量情感与风险推理（兼容旧接口）。"""

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


def infer_news_sentiment_and_risk_5level(text: str) -> NewsSentimentRiskResult:
    """新闻情感/风险 1~5 级推理。"""

    t = text.lower()

    risk_kw = {
        5: ["财务造假", "退市", "重大违法", "st", "破产", "调查立案", "fraud", "delist", "bankruptcy"],
        4: ["监管调查", "业绩暴雷", "大幅下滑", "诉讼", "合规风险", "lawsuit", "probe", "downgrade"],
        3: ["不确定", "波动", "下修", "风险", "压力", "uncertain", "volatile", "headwind"],
        2: ["承压", "扰动", "谨慎", "回调", "moderate risk", "cautious"],
        1: ["政策落地", "稳定", "确定性", "低风险", "secure", "stable"],
    }

    sent_kw = {
        5: ["爆发", "大超预期", "强劲增长", "历史新高", "breakout", "surge", "record high"],
        4: ["利好", "增长", "改善", "上调", "positive", "beat", "upgrade", "strong"],
        3: ["中性", "符合预期", "观望", "neutral", "inline"],
        2: ["承压", "下滑", "悲观", "negative", "miss", "weak"],
        1: ["暴跌", "恐慌", "崩盘", "重大利空", "collapse", "panic", "crash"],
    }

    risk_score_raw, risk_reasons = _weighted_score(t, risk_kw)
    sent_score_raw, sent_reasons = _weighted_score(t, sent_kw)

    pos_hits = sum(k in t for k in sent_kw[4] + sent_kw[5])
    neg_hits = sum(k in t for k in sent_kw[1] + sent_kw[2])
    sentiment_score = (pos_hits - neg_hits) + sent_score_raw * 0.2

    risk_level = _map_risk_level(risk_score_raw)
    sentiment_level = _map_sentiment_level(sentiment_score)

    return NewsSentimentRiskResult(
        sentiment_level=sentiment_level,
        sentiment_label=_sentiment_label(sentiment_level),
        risk_level=risk_level,
        risk_label=_risk_label(risk_level),
        sentiment_score=round(sentiment_score, 3),
        risk_score=round(risk_score_raw, 3),
        reasons=(risk_reasons + sent_reasons)[:6],
    )


def _weighted_score(text: str, kw_map: dict[int, list[str]]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    for level, kws in kw_map.items():
        for kw in kws:
            if kw in text:
                score += level
                reasons.append(kw)
    return score, reasons


def _map_risk_level(score: float) -> int:
    if score >= 10:
        return 5
    if score >= 6:
        return 4
    if score >= 3:
        return 3
    if score >= 1:
        return 2
    return 1


def _map_sentiment_level(score: float) -> int:
    if score >= 3:
        return 5
    if score >= 1:
        return 4
    if score > -0.5:
        return 3
    if score > -2:
        return 2
    return 1


def _risk_label(level: int) -> str:
    mapping = {1: "极低风险", 2: "低风险", 3: "中等风险", 4: "高风险", 5: "极高风险"}
    return mapping[level]


def _sentiment_label(level: int) -> str:
    mapping = {1: "极负面", 2: "负面", 3: "中性", 4: "正面", 5: "极正面"}
    return mapping[level]
