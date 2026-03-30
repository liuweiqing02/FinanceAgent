from __future__ import annotations

import json
import re
from typing import Any

from app.models.schemas import AgentOutput, Evidence
from app.models.sentiment_risk import infer_sentiment_and_risk


class SummaryAgent:
    name = "总结"

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def run(self, ticker: str, outputs: list[AgentOutput], evidence: list[Evidence]) -> AgentOutput:
        merged_text = "\n".join(o.analysis for o in outputs)
        sentiment = infer_sentiment_and_risk(merged_text)

        cite = _citation_map(evidence)
        analysis = "\n".join(
            [
                f"{ticker} 综合结论由基本面、技术面、估值面、新闻面四个并行 Agent 汇总得到。",
                "",
                "综合判断：",
                f"- 市场情绪：{sentiment.sentiment}（情绪分数 {sentiment.score}）{cite.get(1, '[E1]')}",
                f"- 风险等级：{sentiment.risk_level}，建议采用分批决策与动态风控 {cite.get(2, '[E2]')}",
                "- 结论可信度：中等偏上，主要依赖当前知识库与召回证据质量。",
                "",
                "投资执行建议：",
                "- 仓位：先轻仓试探，确认信号后再逐步加仓。",
                "- 风控：设置分层止损与时间止损，避免单次误判扩大损失。",
                "- 复盘：每次关键新闻或财报发布后，重新触发全链路研报。",
            ]
        )

        conclusions = [
            _ensure_citation(f"{ticker} 具备跟踪价值，但应以风险预算作为前置约束 {cite.get(1, '[E1]')}", 1),
            _ensure_citation(f"当前更适合“分批建仓 + 持续验证”，而非一次性重仓 {cite.get(2, '[E2]')}", 2),
            _ensure_citation(f"若后续盈利与现金流继续改善，估值中枢有望抬升 {cite.get(3, '[E3]')}", 3),
            _ensure_citation(f"若出现技术位破坏或负面新闻扩散，应快速降低敞口 {cite.get(4, '[E4]')}", 4),
        ]

        analysis, conclusions = _llm_rewrite_summary(self.llm, ticker, outputs, evidence, analysis, conclusions)
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)


def _citation_map(evidence: list[Evidence]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for i, _ in enumerate(evidence, start=1):
        mapping[i] = f"[E{i}]"
    return mapping


def _ensure_citation(text: str, default_idx: int = 1) -> str:
    if "[E" in text:
        return text
    return text + f" [E{default_idx}]"


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


def _llm_rewrite_summary(
    llm: Any | None,
    ticker: str,
    outputs: list[AgentOutput],
    evidence: list[Evidence],
    analysis: str,
    conclusions: list[str],
) -> tuple[str, list[str]]:
    if llm is None:
        return analysis, conclusions

    block = "\n\n".join(f"## {o.name}\n{o.analysis}\n\n结论: {o.conclusions}" for o in outputs)
    ev = "\n".join(f"[E{i}] {x.title}: {x.content[:300].replace(chr(10), ' ')}" for i, x in enumerate(evidence[:5], start=1))

    user_prompt = (
        f"请基于四个并行专家输出，重写 {ticker} 的综合结论。\n"
        "要求：\n"
        "1) 输出 JSON：{\"analysis\":\"...\",\"conclusions\":[\"...\"]}。\n"
        "2) analysis 至少 260 字，必须覆盖一致点、分歧点、主要风险、执行建议。\n"
        "3) conclusions 给 4 条，每条必须有 [E数字]。\n"
        "4) 不得编造证据，不足就明确写证据不足。\n"
        f"专家输出:\n{block}\n\n证据:\n{ev}\n\n当前草稿:\n{analysis}\n{json.dumps(conclusions, ensure_ascii=False)}"
    )

    try:
        text = llm.generate(system_prompt="你是买方投研负责人，强调证据链与可执行性。", user_prompt=user_prompt)
        obj = _extract_json_block(text)
        if not obj:
            return analysis, conclusions
        new_analysis = str(obj.get("analysis", "")).strip()
        raw_c = obj.get("conclusions", [])
        if not isinstance(raw_c, list):
            return analysis, conclusions
        out = [_ensure_citation(str(x).strip(), i + 1) for i, x in enumerate(raw_c[:4]) if str(x).strip()]
        if len(out) < 4 or len(new_analysis) < 100:
            return analysis, conclusions
        return new_analysis, out
    except Exception:  # noqa: BLE001
        return analysis, conclusions
