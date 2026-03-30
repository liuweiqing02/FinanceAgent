from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Evidence:
    """证据片段。"""

    source_id: str
    title: str
    content: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """检索结果集合。"""

    query: str
    items: list[Evidence]


@dataclass(slots=True)
class AgentOutput:
    """单个 Agent 的输出。"""

    name: str
    analysis: str
    conclusions: list[str]


@dataclass(slots=True)
class ReportBundle:
    """最终研报对象。"""

    ticker: str
    markdown: str
    evidence: list[Evidence]
