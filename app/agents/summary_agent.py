from __future__ import annotations

import json
import re
from typing import Any

from app.models.schemas import AgentOutput, Evidence
from app.models.sentiment_risk import infer_sentiment_and_risk


class SummaryAgent:
    name = "总结"

    def __init__(self, llm: Any | None = None, retry_max: int = 2) -> None:
        self.llm = llm
        self.retry_max = retry_max

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

        analysis, conclusions = _llm_rewrite_summary(self.llm, ticker, outputs, evidence, analysis, conclusions, self.retry_max)
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



def _contains_forbidden_external_claims(text: str, evidence_text: str) -> bool:
    low = (text or "").lower()
    ev_low = (evidence_text or "").lower()
    markers = ["bloomberg", "factset", "consensus", "ref:"]
    return any((m in low) and (m not in ev_low) for m in markers)
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
    cands = [x for x in lines if (x.startswith("-") or x.startswith("1.") or x.startswith("2.") or x.startswith("3.") or x.startswith("4.")) and "[E" in x]
    if len(cands) >= 4:
        conc = [_ensure_citation(re.sub(r"^[-*\d.\s]+", "", x).strip(), fallback_ref) for x in cands[:4]]
        return s, conc
    return None


def _llm_rewrite_summary(
    llm: Any | None,
    ticker: str,
    outputs: list[AgentOutput],
    evidence: list[Evidence],
    analysis: str,
    conclusions: list[str],
    retry_max: int,
) -> tuple[str, list[str]]:
    if llm is None:
        return analysis, conclusions

    block = "\n\n".join(f"## {o.name}\n{o.analysis}\n\n结论: {o.conclusions}" for o in outputs)
    ev = "\n".join(f"[E{i}] {x.title}: {x.content[:300].replace(chr(10), ' ')}" for i, x in enumerate(evidence[:5], start=1))

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
            f"请基于四个并行专家输出，重写 {ticker} 的综合结论。\n"
            "要求：\n"
            "1) 输出 JSON：{\"analysis\":\"...\",\"conclusions\":[\"...\"]}。\n"
            "2) analysis 至少 260 字，必须覆盖一致点、分歧点、主要风险、执行建议。\n"
            "3) conclusions 给 4 条，每条必须有 [E数字]。\n"
            "4) 不得编造证据，不足就明确写证据不足。\n"
            "5) 只写投资分析与执行策略，不要评价系统管道、解析故障、编码问题、RAG实现细节。\n"
            "6) 语气务实克制，避免极端化表述。\n"
            f"专家输出:\n{block}\n\n证据:\n{ev}\n\n当前草稿:\n{analysis}\n{json.dumps(conclusions, ensure_ascii=False)}"
            f"{fix_hint}"
        )

        try:
            text = llm.generate(system_prompt="你是买方投研负责人，强调证据链与可执行性，输出克制、可落地结论。", user_prompt=user_prompt)
            last_output = text
            obj = _extract_json_block(text)
            if not obj:
                last_error = "未解析到 JSON"
                continue

            new_analysis = str(obj.get("analysis", "")).strip()
            raw_c = obj.get("conclusions", [])
            if not isinstance(raw_c, list):
                last_error = "conclusions 不是数组"
                continue
            out = [_ensure_citation(str(x).strip(), i + 1) for i, x in enumerate(raw_c[:4]) if str(x).strip()]
            if len(out) < 4:
                last_error = f"conclusions 数量不足: {len(out)}"
                continue
            if len(new_analysis) < 100:
                last_error = f"analysis 太短: {len(new_analysis)}"
                continue
            return new_analysis, out
        except Exception as exc:  # noqa: BLE001
            last_error = f"调用异常: {exc}"

    if last_output:
        parsed = _extract_from_plain_text(last_output, 1)
        if parsed is not None:
            p_analysis, p_conc = parsed
            p_analysis = p_analysis + "\n\n注：本段由 LLM 文本抽取生成（JSON 校验失败后降级）。"
            return p_analysis, p_conc

    try:
        free_prompt = (
            f"请直接输出 {ticker} 的综合结论：正文不少于260字，并给出4条结论。"
            "每条结论包含 [E数字]，不要 JSON，不要解释。"
            "仅写投资分析与执行策略，不要讨论系统实现细节。"
            f"专家输出：\n{block}\n\n证据：\n{ev}"
        )
        free_text = llm.generate(system_prompt="你是买方投研负责人，强调证据链与可执行性。", user_prompt=free_prompt)
        parsed2 = _extract_from_plain_text(free_text, 1)
        if parsed2 is not None:
            p_analysis, p_conc = parsed2
            p_analysis = p_analysis + "\n\n注：本段由 LLM 自由文本生成并抽取（JSON 多轮失败后降级）。"
            return p_analysis, p_conc
    except Exception as exc:  # noqa: BLE001
        last_error = f"{last_error}; 自由文本调用异常: {exc}" if last_error else f"自由文本调用异常: {exc}"

    fallback_note = f"\n\n注：LLM 重写未生效，已回退模板。原因：{last_error[:180] or '未知'}。"
    return analysis + fallback_note, conclusions

