from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.rag.ingestion import clean_text


@dataclass(slots=True)
class RawKnowledgeItem:
    ticker: str
    topic: str
    title: str
    content: str
    source: str
    url: str
    published_at: str


def ensure_sample_raw_data(raw_dir: Path) -> None:
    """初始化更接近真实业务的原始数据样例。"""

    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "finance_events.jsonl"
    if target.exists():
        return

    rows = [
        {
            "ticker": "AAPL",
            "topic": "fundamental",
            "title": "AAPL FY25Q1 财报结构拆解",
            "content": "公司季度收入保持韧性，服务业务收入占比继续提升，硬件业务受换机周期影响出现结构性分化。毛利率维持在历史中高位，经营现金流同比改善，反映盈利质量并非依赖单次会计调整。管理层对下一季度指引偏谨慎但未下修全年核心战略投入。",
            "source": "earnings_call",
            "url": "https://example.com/aapl/fy25q1",
            "published_at": "2026-02-15",
        },
        {
            "ticker": "AAPL",
            "topic": "fundamental",
            "title": "AAPL 现金流与资本开支跟踪",
            "content": "资本开支主要投向算力基础设施与供应链协同，研发投入保持增长。自由现金流覆盖回购与分红后仍有安全垫，资产负债表保持稳健。若后续需求端复苏延迟，预计先体现为库存周转放缓而非现金流断裂。",
            "source": "10q",
            "url": "https://example.com/aapl/cashflow",
            "published_at": "2026-02-20",
        },
        {
            "ticker": "AAPL",
            "topic": "technical",
            "title": "AAPL 周线趋势与成交结构",
            "content": "价格仍在中期上行通道中运行，20日与60日均线保持多头排列。过去三周成交量温和放大但并未出现异常放量长上影，说明资金分歧可控。若跌破60日均线并连续两日无法收复，技术上将进入防守区间。",
            "source": "market_data",
            "url": "https://example.com/aapl/technical-weekly",
            "published_at": "2026-03-01",
        },
        {
            "ticker": "AAPL",
            "topic": "valuation",
            "title": "AAPL 估值分位与增长匹配",
            "content": "当前动态估值位于近三年中枢附近，尚未进入极端高估区。市场给予溢价主要来自服务业务稳定性和生态壁垒。若未来两个季度盈利增速回到双位数，估值中枢有上修空间；反之将回归行业均值附近。",
            "source": "sellside_report",
            "url": "https://example.com/aapl/valuation",
            "published_at": "2026-03-03",
        },
        {
            "ticker": "AAPL",
            "topic": "news",
            "title": "AAPL 供应链备货与新品预期",
            "content": "多家供应链渠道反馈新机备货节奏环比加快，市场预期下半年销量改善。与此同时，地缘政策和关键零部件产能仍可能造成短期扰动。新闻层面短期偏正向，但需持续验证出货与毛利兑现。",
            "source": "newswire",
            "url": "https://example.com/aapl/news-supply-chain",
            "published_at": "2026-03-05",
        },
        {
            "ticker": "TSLA",
            "topic": "fundamental",
            "title": "TSLA 汽车与储能双引擎进展",
            "content": "汽车业务利润率受价格策略波动影响明显，但储能业务维持高增速，逐步平滑周期冲击。产能利用率提升带来单位成本下降，然而海外市场竞争仍在加剧。管理层强调自动驾驶商业化节奏将决定中期利润弹性。",
            "source": "earnings_call",
            "url": "https://example.com/tsla/fundamental",
            "published_at": "2026-02-18",
        },
        {
            "ticker": "TSLA",
            "topic": "technical",
            "title": "TSLA 波动率与关键支撑观察",
            "content": "股价波动率明显高于大型科技股均值，短线交易拥挤度较高。近期反弹过程伴随量能回升，但上方密集成交区压力仍在。若回撤失守关键支撑，需警惕情绪放大导致的连锁下跌。",
            "source": "market_data",
            "url": "https://example.com/tsla/technical",
            "published_at": "2026-03-02",
        },
        {
            "ticker": "TSLA",
            "topic": "valuation",
            "title": "TSLA 高估值与成长兑现压力",
            "content": "估值仍处于成长股高位区间，市场已定价部分自动驾驶远期预期。若FSD商业化与监管落地速度不及预期，估值压缩风险高于同业。若储能与软件收入占比持续提升，估值结构有望获得再平衡。",
            "source": "sellside_report",
            "url": "https://example.com/tsla/valuation",
            "published_at": "2026-03-04",
        },
        {
            "ticker": "TSLA",
            "topic": "news",
            "title": "TSLA 监管进展与产能爬坡",
            "content": "近期市场聚焦自动驾驶监管审批进度与海外工厂爬坡效率。正面消息可快速抬升风险偏好，但负面监管信号也可能触发估值重定价。新闻驱动属性较强，建议与交付数据联动跟踪。",
            "source": "newswire",
            "url": "https://example.com/tsla/news-regulation",
            "published_at": "2026-03-06",
        },
    ]

    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_knowledge_base(raw_dir: Path, knowledge_base_dir: Path, include_glob: str = "*.jsonl") -> list[Path]:
    """将原始事件流数据构建为可检索知识库文档。"""

    raw_items = _load_raw_items(raw_dir, include_glob)
    grouped: dict[tuple[str, str], list[RawKnowledgeItem]] = defaultdict(list)
    for item in raw_items:
        grouped[(item.ticker.upper(), item.topic.lower())].append(item)

    knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for (ticker, topic), items in grouped.items():
        items.sort(key=lambda x: x.published_at)
        text = _render_topic_markdown(ticker, topic, items)
        output = knowledge_base_dir / f"{ticker}_{topic}.md"
        output.write_text(text, encoding="utf-8")
        outputs.append(output)
    return outputs


def _load_raw_items(raw_dir: Path, include_glob: str) -> list[RawKnowledgeItem]:
    items: list[RawKnowledgeItem] = []
    for file in sorted(raw_dir.glob(include_glob)):
        for row in file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not row.strip():
                continue
            obj = json.loads(row)
            items.append(
                RawKnowledgeItem(
                    ticker=str(obj.get("ticker", "")).upper(),
                    topic=str(obj.get("topic", "general")).lower(),
                    title=str(obj.get("title", "未命名事件")).strip(),
                    content=clean_text(str(obj.get("content", "")).strip()),
                    source=str(obj.get("source", "unknown")).strip(),
                    url=str(obj.get("url", "")).strip(),
                    published_at=str(obj.get("published_at", "")).strip(),
                )
            )
    return [x for x in items if x.ticker and x.content]


def _render_topic_markdown(ticker: str, topic: str, items: list[RawKnowledgeItem]) -> str:
    title_map = {
        "fundamental": "基本面",
        "technical": "技术面",
        "valuation": "估值",
        "news": "新闻",
    }
    topic_cn = title_map.get(topic, topic)

    lines = [
        f"# {ticker} {topic_cn} 深度知识库",
        "",
        "## 文档说明",
        "本文件由原始事件流自动构建，供 RAG 检索与证据引用使用。",
        "",
        "## 关键事件与解读",
    ]

    for i, item in enumerate(items, start=1):
        lines.extend(
            [
                f"### 事件{i}: {item.title}",
                f"- 发布时间: {item.published_at}",
                f"- 来源类型: {item.source}",
                f"- 来源链接: {item.url or 'N/A'}",
                "- 内容解读:",
                f"  {item.content}",
                "",
            ]
        )

    lines.extend(
        [
            "## 聚合结论",
            f"- {ticker} 在 {topic_cn} 维度当前存在可跟踪信号，但需结合后续数据验证。",
            "- 建议将该维度与其它维度交叉验证，避免单一来源偏差。",
            "",
        ]
    )
    return "\n".join(lines)
