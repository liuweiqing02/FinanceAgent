from __future__ import annotations

import json
import re
from typing import Any

from app.agents.base import BaseAgent
from app.models.schemas import AgentOutput, Evidence
from app.models.sentiment_risk import infer_news_sentiment_and_risk_5level


def _citation(index: int) -> str:
    return f"[E{index}]"


def _ensure_cited(text: str, default_idx: int = 1) -> str:
    if "[E" in text:
        return text
    return f"{text} {_citation(default_idx)}"


def _is_readable(text: str) -> bool:
    if len(text.strip()) < 40:
        return False
    long_tokens = re.findall(r"[A-Za-z0-9_-]{35,}", text)
    if len(long_tokens) >= 2:
        return False
    latin = sum(ch.isalpha() for ch in text)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    readable_ratio = (latin + cjk) / max(len(text), 1)
    return readable_ratio >= 0.15


def _topic_like(ev: Evidence, topic: str) -> bool:
    tag = f"{ev.title} {ev.source_id}".lower()
    if topic == "fundamental":
        return "fundamental" in tag or "10-k" in tag or "10-q" in tag or "mcp_fundamental" in tag
    if topic == "technical":
        return "technical" in tag or "sma" in ev.content.lower() or "mcp_market" in tag
    if topic == "valuation":
        return "valuation" in tag or "p/e" in ev.content.lower() or "mcp_market" in tag or "mcp_fundamental" in tag
    if topic == "news":
        return "news" in tag or "8-k" in tag or "rss" in tag
    return False


def _select_evidence(evidence: list[Evidence], topic: str, limit: int = 3) -> list[tuple[int, Evidence]]:
    picked: list[tuple[int, Evidence]] = []
    for i, ev in enumerate(evidence, start=1):
        if _topic_like(ev, topic) and _is_readable(ev.content):
            picked.append((i, ev))
            if len(picked) >= limit:
                return picked
    for i, ev in enumerate(evidence, start=1):
        if _is_readable(ev.content) and (i, ev) not in picked:
            picked.append((i, ev))
            if len(picked) >= limit:
                break
    return picked


def _kv_from_text(text: str, keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in keys:
        m = re.search(rf"{re.escape(k)}\s*[=:]\s*([^，。;\n]+)", text, flags=re.IGNORECASE)
        if m:
            v = m.group(1).strip()
            if v and v.upper() not in {"N/A", "NA", "NONE", "NULL"}:
                out[k] = v
    return out


def _evidence_line(ref: int, ev: Evidence, max_len: int = 140) -> str:
    clean = " ".join(ev.content.replace("\n", " ").split())
    if len(clean) > max_len:
        clean = clean[:max_len] + "..."
    return f"- [E{ref}] {ev.title}（score={ev.score}）：{clean}"


def _extract_json_block(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _extract_from_plain_text(text: str, fallback_ref: int) -> tuple[str, list[str]] | None:
    s = text.strip()
    if not s:
        return None

    lines = [x.strip() for x in s.splitlines() if x.strip()]
    cands = [x for x in lines if (x.startswith("-") or x.startswith("1.") or x.startswith("2.") or x.startswith("3.")) and "[E" in x]
    if len(cands) >= 3:
        conc = [_ensure_cited(re.sub(r"^[-*\d.\s]+", "", x).strip(), fallback_ref) for x in cands[:4]]
        return s, conc

    m = re.search(r"结论[：:]?([\s\S]{0,1200})", s)
    if m:
        tail = [x.strip(" -") for x in m.group(1).splitlines() if x.strip()]
        conc2 = [_ensure_cited(x, fallback_ref) for x in tail[:4] if len(x) > 6]
        if len(conc2) >= 3:
            return s, conc2
    return None


def _validate_payload(obj: dict[str, Any], min_conclusions: int = 3) -> tuple[bool, str, list[str]]:
    analysis = str(obj.get("analysis", "")).strip()
    raw = obj.get("conclusions", [])
    if not isinstance(raw, list):
        return False, "conclusions 不是数组", []
    conc = [str(x).strip() for x in raw if str(x).strip()]
    if len(conc) < min_conclusions:
        return False, f"conclusions 数量不足: {len(conc)}", conc
    if len(analysis) < 80:
        return False, f"analysis 太短: {len(analysis)}", conc
    return True, "", conc


def _llm_rewrite(
    *,
    llm: Any | None,
    agent_name: str,
    ticker: str,
    topic: str,
    analysis: str,
    conclusions: list[str],
    refs: list[tuple[int, Evidence]],
    retry_max: int = 2,
) -> tuple[str, list[str]]:
    if llm is None:
        return analysis, conclusions

    fallback_ref = refs[0][0] if refs else 1
    evidence_text = "\n".join(
        f"[E{ref}] {ev.title} | {ev.content[:500].replace(chr(10), ' ')}" for ref, ev in refs[:3]
    )
    system_prompt = "你是资深股票分析师，请输出可审计、可落地、结论有证据编号的中文分析。"

    last_output = ""
    last_error = ""
    attempts = max(1, retry_max + 1)

    for i in range(attempts):
        fix_hint = ""
        if i > 0:
            fix_hint = (
                "\n上一次输出校验失败，请修复后严格只输出 JSON。"
                f"\n失败原因：{last_error}"
                f"\n上次输出：{last_output[:1200]}"
            )

        user_prompt = (
            f"请重写 {ticker} 的{agent_name}分析，主题={topic}。\n"
            "必须满足：\n"
            "1) analysis 至少 220 字，不能空话，必须出现具体数据字段与因果解释。\n"
            "2) conclusions 输出 3~4 条，每条必须包含 [E数字] 引用。\n"
            "3) 如果证据不足，明确写出不足项，但不要编造数据。\n"
            "4) 只返回 JSON：{\"analysis\":\"...\",\"conclusions\":[\"...\"]}，不要解释。\n"
            f"可用证据:\n{evidence_text}\n\n"
            f"当前草稿analysis:\n{analysis}\n\n"
            f"当前草稿conclusions:\n{json.dumps(conclusions, ensure_ascii=False)}"
            f"{fix_hint}"
        )

        try:
            text = llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            last_output = text
            obj = _extract_json_block(text)
            if not obj:
                last_error = "未解析到 JSON 对象"
                continue

            ok, reason, conc = _validate_payload(obj)
            if not ok:
                last_error = reason
                continue

            clean_conclusions = [_ensure_cited(x, fallback_ref) for x in conc[:4]]
            return str(obj.get("analysis", "")).strip(), clean_conclusions
        except Exception as exc:  # noqa: BLE001
            last_error = f"调用异常: {exc}"

    # JSON 重试全部失败后，先尝试对最后输出做文本抽取。
    if last_output:
        parsed = _extract_from_plain_text(last_output, fallback_ref)
        if parsed is not None:
            p_analysis, p_conc = parsed
            p_analysis = p_analysis + "\n\n注：本段由 LLM 文本抽取生成（JSON 校验失败后降级）。"
            return p_analysis, p_conc

    # 进一步降级：再让 LLM 直接输出自由文本，不再要求 JSON。
    try:
        free_prompt = (
            f"请直接输出 {ticker} 的{agent_name}分析正文（不少于220字）和3条结论。"
            "要求：每条结论都带 [E数字] 引用，不要 JSON，不要解释。"
            f"可用证据：\n{evidence_text}"
        )
        free_text = llm.generate(system_prompt=system_prompt, user_prompt=free_prompt)
        parsed2 = _extract_from_plain_text(free_text, fallback_ref)
        if parsed2 is not None:
            p_analysis, p_conc = parsed2
            p_analysis = p_analysis + "\n\n注：本段由 LLM 自由文本生成并抽取（JSON 多轮失败后降级）。"
            return p_analysis, p_conc
    except Exception as exc:  # noqa: BLE001
        last_error = f"{last_error}; 自由文本调用异常: {exc}" if last_error else f"自由文本调用异常: {exc}"

    # 最后兜底：保留模板并显式标注失败原因，避免看起来像“悄悄没用 LLM”。
    fallback_note = f"\n\n注：LLM 重写未生效，已回退模板。原因：{last_error[:180] or '未知'}。"
    return analysis + fallback_note, conclusions


class FundamentalAgent(BaseAgent):
    name = "基本面"

    def __init__(self, llm: Any | None = None, retry_max: int = 2) -> None:
        self.llm = llm
        self.retry_max = retry_max

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        refs = _select_evidence(evidence, "fundamental", limit=3)
        first_ref = refs[0][0] if refs else 1

        metrics: list[str] = []
        for ref, ev in refs:
            kv = _kv_from_text(
                ev.content,
                ["营收增速", "利润增速", "毛利率", "营业利润率", "净利率", "ROE", "自由现金流", "总现金"],
            )
            if kv:
                metrics.append(f"- [E{ref}] " + "，".join(f"{k}={v}" for k, v in kv.items()))

        if not metrics:
            metrics.append("- 当前召回文本未提取到结构化财务指标，建议补充最新 10-Q/10-K 原文段落。")

        analysis = "\n".join(
            [
                f"{ticker} 基本面围绕盈利质量、现金流与增长兑现进行量化判断。",
                "",
                "框架判断：",
                "- 盈利质量：优先看毛利率、营业利润率、净利率是否同步改善。",
                "- 现金流韧性：看自由现金流与总现金能否覆盖回购/资本开支。",
                "- 成长兑现：看营收增速与利润增速是否同向。",
                "",
                "关键数据：",
                *metrics,
                "",
                "证据解读：",
                *[_evidence_line(ref, ev) for ref, ev in refs[:2]],
                "",
                "风险提示：",
                "- 若利润增速显著低于营收增速，可能意味着成本端压力上升。",
                "- 若自由现金流走弱且资本开支持续抬升，需关注资金安全边际。",
            ]
        )

        conclusions = [
            _ensure_cited(f"{ticker} 基本面结论应以财务指标实测值为锚，而非仅凭叙述判断 [E{first_ref}]", first_ref),
            _ensure_cited(f"若营收与利润增速同步改善，估值中枢存在上修基础 [E{first_ref}]", first_ref),
            _ensure_cited(f"若现金流覆盖能力下降，需要下调仓位容忍区间 [E{first_ref}]", first_ref),
        ]
        analysis, conclusions = _llm_rewrite(
            llm=self.llm,
            agent_name=self.name,
            ticker=ticker,
            topic="fundamental",
            analysis=analysis,
            conclusions=conclusions,
            refs=refs,
            retry_max=self.retry_max,
        )
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)


class TechnicalAgent(BaseAgent):
    name = "技术面"

    def __init__(self, llm: Any | None = None, retry_max: int = 2) -> None:
        self.llm = llm
        self.retry_max = retry_max

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        refs = _select_evidence(evidence, "technical", limit=3)
        first_ref = refs[0][0] if refs else 1

        metric_lines: list[str] = []
        for ref, ev in refs:
            kv = _kv_from_text(ev.content, ["最新价", "SMA20", "SMA60", "5日均量", "20日均量", "日内涨跌幅"])
            if kv:
                metric_lines.append(f"- [E{ref}] " + "，".join(f"{k}={v}" for k, v in kv.items()))

        if not metric_lines:
            metric_lines.append("- 当前召回文本缺少可计算的均线/量能字段，建议补充行情快照。")

        analysis = "\n".join(
            [
                f"{ticker} 技术面聚焦趋势方向、量价配合和风险位管理。",
                "",
                "框架判断：",
                "- 趋势：看价格与 SMA20/SMA60 相对位置。",
                "- 量能：看 5 日均量相对 20 日均量是否放大。",
                "- 波动：看日内涨跌幅是否超出常态区间。",
                "",
                "关键数据：",
                *metric_lines,
                "",
                "证据解读：",
                *[_evidence_line(ref, ev) for ref, ev in refs[:2]],
                "",
                "风险提示：",
                "- 放量下跌通常比缩量下跌更需要快速降风险。",
                "- 当价格反复跌破中期均线，趋势策略胜率会明显下降。",
            ]
        )

        conclusions = [
            _ensure_cited(f"{ticker} 技术面应优先依据均线和量能数据执行，而非主观预测 [E{first_ref}]", first_ref),
            _ensure_cited(f"若量价出现背离，需要缩短持仓评估周期 [E{first_ref}]", first_ref),
            _ensure_cited(f"跌破关键支撑位时应先执行风控再讨论反弹 [E{first_ref}]", first_ref),
        ]
        analysis, conclusions = _llm_rewrite(
            llm=self.llm,
            agent_name=self.name,
            ticker=ticker,
            topic="technical",
            analysis=analysis,
            conclusions=conclusions,
            refs=refs,
            retry_max=self.retry_max,
        )
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)


class ValuationAgent(BaseAgent):
    name = "估值"

    def __init__(self, llm: Any | None = None, retry_max: int = 2) -> None:
        self.llm = llm
        self.retry_max = retry_max

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        refs = _select_evidence(evidence, "valuation", limit=3)
        first_ref = refs[0][0] if refs else 1

        metrics: list[str] = []
        for ref, ev in refs:
            kv = _kv_from_text(ev.content, ["PE", "Forward PE", "P/B", "Trailing PE", "price_to_book", "营收增速", "利润增速"])
            if kv:
                metrics.append(f"- [E{ref}] " + "，".join(f"{k}={v}" for k, v in kv.items()))

        if not metrics:
            metrics.append("- 当前召回文本缺少估值倍数字段，建议补充 market/fundamental MCP 快照。")

        analysis = "\n".join(
            [
                f"{ticker} 估值分析采用“倍数水平 + 增长匹配 + 安全边际”三步法。",
                "",
                "框架判断：",
                "- 倍数水平：PE/PB 在历史区间中的位置。",
                "- 增长匹配：估值扩张是否有营收/利润增速支撑。",
                "- 安全边际：悲观情景下估值压缩的容忍度。",
                "",
                "关键数据：",
                *metrics,
                "",
                "证据解读：",
                *[_evidence_line(ref, ev) for ref, ev in refs[:2]],
                "",
                "风险提示：",
                "- 高估值在增速放缓阶段容易触发双杀。",
                "- 利率中枢抬升时，高久期资产估值更脆弱。",
            ]
        )

        conclusions = [
            _ensure_cited(f"{ticker} 估值结论必须与增长兑现数据绑定，不能只看静态 PE/PB [E{first_ref}]", first_ref),
            _ensure_cited(f"若增长与利润率同步改善，估值溢价才更可持续 [E{first_ref}]", first_ref),
            _ensure_cited(f"建议按安全边际设定分批买入和减仓阈值 [E{first_ref}]", first_ref),
        ]
        analysis, conclusions = _llm_rewrite(
            llm=self.llm,
            agent_name=self.name,
            ticker=ticker,
            topic="valuation",
            analysis=analysis,
            conclusions=conclusions,
            refs=refs,
            retry_max=self.retry_max,
        )
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)


class NewsAgent(BaseAgent):
    name = "新闻"

    def __init__(self, llm: Any | None = None, retry_max: int = 2) -> None:
        self.llm = llm
        self.retry_max = retry_max

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        refs = _select_evidence(evidence, "news", limit=3)
        if not refs:
            refs = _select_evidence(evidence, "fundamental", limit=2)

        scored_rows: list[str] = []
        sent_lv: list[int] = []
        risk_lv: list[int] = []
        for ref, ev in refs[:3]:
            r = infer_news_sentiment_and_risk_5level(ev.content)
            sent_lv.append(r.sentiment_level)
            risk_lv.append(r.risk_level)
            scored_rows.append(
                f"- [E{ref}] 情感={r.sentiment_level}({r.sentiment_label})，风险={r.risk_level}({r.risk_label})，触发词={','.join(r.reasons[:3]) or '无'}"
            )

        avg_sent = round(sum(sent_lv) / max(len(sent_lv), 1), 2)
        avg_risk = round(sum(risk_lv) / max(len(risk_lv), 1), 2)
        first_ref = refs[0][0] if refs else 1
        second_ref = refs[1][0] if len(refs) > 1 else first_ref

        analysis = "\n".join(
            [
                f"{ticker} 新闻面分析聚焦事件强度、预期变化与情绪扩散。",
                "",
                "框架判断：",
                "- 事件强度：识别监管、产品、供应链等高冲击事件。",
                "- 预期变化：判断新闻是否改变盈利/估值假设。",
                "- 持续性：区分一次性噪音与可持续催化。",
                "",
                "情感/风险模型（1~5级）：",
                f"- 平均情感等级：{avg_sent}（1=极负面，5=极正面）",
                f"- 平均风险等级：{avg_risk}（1=极低风险，5=极高风险）",
                *scored_rows,
                "",
                "证据解读：",
                *[_evidence_line(ref, ev) for ref, ev in refs[:2]],
                "",
                "风险提示：",
                "- 新闻驱动阶段应提升复盘频次，避免单条消息放大偏见。",
                "- 若负面新闻与技术破位共振，优先处理风险敞口。",
            ]
        )

        conclusions = [
            _ensure_cited(f"{ticker} 新闻情感均值={avg_sent}，短期交易情绪影响显著 [E{first_ref}]", first_ref),
            _ensure_cited(f"{ticker} 新闻风险均值={avg_risk}，建议执行事件驱动风控 [E{second_ref}]", second_ref),
            _ensure_cited(f"重大仓位决策应至少基于两条独立新闻证据交叉验证 [E{first_ref}]", first_ref),
        ]
        analysis, conclusions = _llm_rewrite(
            llm=self.llm,
            agent_name=self.name,
            ticker=ticker,
            topic="news",
            analysis=analysis,
            conclusions=conclusions,
            refs=refs,
            retry_max=self.retry_max,
        )
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)

