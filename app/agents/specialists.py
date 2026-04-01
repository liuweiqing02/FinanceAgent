from __future__ import annotations

import asyncio
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
    body = text.strip()
    if len(body) < 60:
        return False
    low = body.lower()
    if low.count("n/a") >= 8:
        return False
    bad_tokens = ["us-gaap", "xbrl", "taxonomy", "linkbase", "namespace", "false fy", "p7y", "p2y"]
    if sum(1 for x in bad_tokens if x in low) >= 2:
        return False
    long_tokens = re.findall(r"[A-Za-z0-9_-]{35,}", body)
    if len(long_tokens) >= 2:
        return False
    latin = sum(ch.isalpha() for ch in body)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
    readable_ratio = (latin + cjk) / max(len(body), 1)
    return readable_ratio >= 0.2

def _topic_like(ev: Evidence, topic: str) -> bool:
    tag = f"{ev.title} {ev.source_id}".lower()
    content = (ev.content or "").lower()
    if topic == "fundamental":
        return "fundamental" in tag or "10-k" in tag or "10-q" in tag or "mcp_fundamental" in tag
    if topic == "technical":
        return "technical" in tag or "sma" in content or "mcp_market" in tag
    if topic == "valuation":
        return "valuation" in tag or "p/e" in content or "mcp_market" in tag or "mcp_fundamental" in tag
    if topic == "news":
        blob = f"{tag} {content}"
        return any(k in blob for k in ["news", "rss", "google_news", "newswire", "reuters", "barron", "cnbc"])
    return False


def _ticker_like(ev: Evidence, ticker: str) -> bool:
    t = ticker.upper().strip()
    if not t:
        return True
    sid = (ev.source_id or "").upper()
    title = (ev.title or "").upper()
    content = (ev.content or "").upper()
    if sid.startswith("MCP_"):
        return t in title or t in content or sid.endswith(t)
    return sid.startswith(f"{t}_") or t in title or f" {t} " in f" {content} "


def _select_evidence(evidence: list[Evidence], topic: str, limit: int = 3, ticker: str | None = None) -> list[tuple[int, Evidence]]:
    picked: list[tuple[int, Evidence]] = []
    for i, ev in enumerate(evidence, start=1):
        if ticker and not _ticker_like(ev, ticker):
            continue
        if _topic_like(ev, topic) and _is_readable(ev.content):
            picked.append((i, ev))
            if len(picked) >= limit:
                return picked

    # 对新闻维度，优先只保留新闻证据，避免被其它维度文本污染。
    if topic == "news" and picked:
        return picked

    for i, ev in enumerate(evidence, start=1):
        if ticker and not _ticker_like(ev, ticker):
            continue
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



def _count_core_fundamental_metrics(text: str) -> int:
    keys = ["营收增速", "利润增速", "毛利率", "营业利润率", "净利率", "ROE", "自由现金流", "总现金"]
    return len(_kv_from_text(text, keys))


def _fetch_runtime_fundamental_snapshot(mcp_client: Any | None, ticker: str) -> dict[str, str] | None:
    if mcp_client is None:
        return None
    try:
        from app.mcp.protocol import ToolRequest

        resp = mcp_client.call_tool(ToolRequest(server="finance_data", tool_name="fundamental_snapshot", args={"ticker": ticker.upper()}))
        if not resp.ok or not isinstance(resp.data, dict):
            return None
        data = resp.data

        def _fmt(v: float, pct: bool = False) -> str:
            if pct:
                return f"{float(v) * 100:.2f}%"
            num = float(v)
            if abs(num) >= 1_000_000_000:
                return f"{num / 1_000_000_000:.2f}B"
            if abs(num) >= 1_000_000:
                return f"{num / 1_000_000:.2f}M"
            return f"{num:.2f}"

        raw = {
            "营收增速": data.get("revenue_growth"),
            "利润增速": data.get("earnings_growth"),
            "毛利率": data.get("gross_margin"),
            "营业利润率": data.get("operating_margin"),
            "净利率": data.get("profit_margin"),
            "ROE": data.get("return_on_equity"),
            "自由现金流": data.get("free_cash_flow"),
            "总现金": data.get("total_cash"),
        }
        out: dict[str, str] = {}
        for k, v in raw.items():
            if not isinstance(v, (int, float)):
                continue
            out[k] = _fmt(float(v), pct=k not in {"自由现金流", "总现金"})
        return out or None
    except Exception:  # noqa: BLE001
        return None



def _fetch_runtime_valuation_metrics(mcp_client: Any | None, ticker: str) -> dict[str, str] | None:
    if mcp_client is None:
        return None
    try:
        from app.mcp.protocol import ToolRequest

        m = mcp_client.call_tool(ToolRequest(server="finance_data", tool_name="market_snapshot", args={"ticker": ticker.upper()}))
        f = mcp_client.call_tool(ToolRequest(server="finance_data", tool_name="fundamental_snapshot", args={"ticker": ticker.upper()}))
        data_m = m.data if m.ok and isinstance(m.data, dict) else {}
        data_f = f.data if f.ok and isinstance(f.data, dict) else {}

        merged = {
            "PE": data_m.get("pe") if isinstance(data_m.get("pe"), (int, float)) else data_f.get("trailing_pe"),
            "Forward PE": data_m.get("forward_pe") if isinstance(data_m.get("forward_pe"), (int, float)) else data_f.get("forward_pe"),
            "P/B": data_m.get("pb") if isinstance(data_m.get("pb"), (int, float)) else data_f.get("price_to_book"),
            "营收增速": data_f.get("revenue_growth"),
            "利润增速": data_f.get("earnings_growth"),
        }
        out: dict[str, str] = {}
        for k, v in merged.items():
            if not isinstance(v, (int, float)):
                continue
            num = float(v)
            if k in {"营收增速", "利润增速"}:
                out[k] = f"{num * 100:.2f}%"
            else:
                out[k] = f"{num:.2f}"
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _news_event_signal_count(refs: list[tuple[int, Evidence]]) -> int:
    cnt = 0
    for _, ev in refs:
        title = (ev.title or "").lower()
        source = (ev.source_id or "").lower()
        content = ev.content or ""
        low = content.lower()

        has_news_source = any(k in (title + " " + source) for k in ["news", "rss", "reuters", "barron", "cnbc", "google_news"])
        has_event_time = any(k in low for k in ["发布时间", "pubdate", "gmt", "发布", "published_at", "来源类型"]) or (
            re.search(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}", low) is not None
            or re.search(r"\d{1,2}-\d{2}t\d{2}:\d{2}", low) is not None
        )
        has_event_marker = ("### 事件" in content) or ("事件" in content and "来源" in content)
        too_template = ("深度知识库" in content and "文档说明" in content and (not has_event_time) and (not has_event_marker))

        if has_news_source and (has_event_time or has_event_marker) and (len(low.strip()) >= 80) and (not too_template):
            cnt += 1
    return cnt
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



def _contains_forbidden_external_claims(text: str, evidence_text: str) -> bool:
    low = (text or "").lower()
    ev_low = (evidence_text or "").lower()
    markers = ["bloomberg", "factset", "consensus", "ref:"]
    return any((m in low) and (m not in ev_low) for m in markers)
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
    system_prompt = "你是资深股票分析师，请输出可审计、可落地、语气克制的中文分析，不讨论系统实现细节。"

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
            "5) 只写投研结论与执行建议，不要输出系统管道故障、解析异常、编码问题等技术细节。\n"
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
            new_analysis = str(obj.get("analysis", "")).strip()
            if _contains_forbidden_external_claims(new_analysis + "\n" + "\n".join(clean_conclusions), evidence_text):
                last_error = "包含证据外引用来源（如 Bloomberg/FactSet/Ref）"
                continue
            return new_analysis, clean_conclusions
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
            "只写投研结论与执行建议，不要讨论系统实现细节。"
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

REACT_TOOL_ROUTES: dict[str, list[str]] = {
    "基本面": ["fundamental_snapshot", "market_snapshot", "load_news_file"],
    "技术面": ["market_snapshot", "load_news_file"],
    "估值": ["market_snapshot", "fundamental_snapshot"],
    "新闻": ["load_news_file", "market_snapshot", "fundamental_snapshot"],
}


def _react_enabled_for_agent(config: Any | None, agent_name: str) -> bool:
    if config is None:
        return False
    common = bool(getattr(config, "react_mcp_enabled", False))
    if agent_name == "新闻":
        return common or bool(getattr(config, "news_react_mcp_enabled", False))
    return common


def _react_step_budget(config: Any | None, agent_name: str) -> int:
    if config is None:
        return 4
    common = int(getattr(config, "react_mcp_max_steps", 0) or 0)
    if common > 0:
        return max(2, common)
    if agent_name == "新闻":
        return max(2, int(getattr(config, "news_react_max_steps", 4)))
    return 4


def _build_react_prompt(agent_name: str, ticker: str, refs: list[tuple[int, Evidence]], topic_hint: str) -> str:
    evidence_text = "\n".join(_evidence_line(ref, ev, max_len=220) for ref, ev in refs[:3])
    return (
        f"请分析 {ticker} 的{agent_name}，采用 ReAct 方式按需调用工具。\n"
        "输出必须是 JSON：{\"analysis\":\"...\",\"conclusions\":[\"...\"]}\n"
        "要求：\n"
        "1) analysis 至少 220 字，且必须包含小节：框架判断、证据解读、风险提示。\n"
        "2) conclusions 输出 3~4 条，每条必须包含 [E数字]。\n"
        f"3) 围绕主题提示展开：{topic_hint}。\n"
        "4) 不得编造数据，若工具返回不足请明确写证据不足。\n"
        f"可用已有证据（引用编号）：\n{evidence_text}\n"
    )


def _extract_last_message_text(out: Any) -> str:
    if isinstance(out, dict):
        msgs = out.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            content = getattr(last, "content", "") if not isinstance(last, dict) else last.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif isinstance(item, str):
                        parts.append(item)
                return "\n".join(parts).strip()
    return str(out).strip()


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
        has_loop = True
    except RuntimeError:
        has_loop = False

    if not has_loop:
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _log_react(logger: Any | None, agent_name: str, status: str, detail: str) -> None:
    if logger is None:
        return
    try:
        logger.log("react_mcp", {"agent": agent_name, "status": status, "detail": detail})
    except Exception:  # noqa: BLE001
        return


def _run_react_mcp_agent(
    *,
    agent_name: str,
    topic_hint: str,
    ticker: str,
    refs: list[tuple[int, Evidence]],
    llm: Any | None,
    config: Any | None,
    mcp_client: Any | None,
    kb_base_dir: str,
    logger: Any | None,
) -> tuple[str, list[str]] | None:
    if llm is None or mcp_client is None or not _react_enabled_for_agent(config, agent_name):
        return None

    try:
        from app.mcp.protocol import ToolRequest
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
    except Exception as exc:  # noqa: BLE001
        _log_react(logger, agent_name, "disabled", f"import_error: {exc}")
        return None

    def _mcp_call(server: str, tool_name: str, args: dict[str, Any]) -> str:
        resp = mcp_client.call_tool(ToolRequest(server=server, tool_name=tool_name, args=args))
        if not resp.ok:
            return json.dumps({"ok": False, "error": resp.error or "unknown"}, ensure_ascii=False)
        return json.dumps({"ok": True, "data": resp.data}, ensure_ascii=False)

    @tool
    def market_snapshot(ticker: str) -> str:
        """获取行情与估值快照。"""
        return _mcp_call("finance_data", "market_snapshot", {"ticker": ticker.upper()})

    @tool
    def fundamental_snapshot(ticker: str) -> str:
        """获取财务核心指标快照。"""
        return _mcp_call("finance_data", "fundamental_snapshot", {"ticker": ticker.upper()})

    @tool
    def load_news_file(ticker: str) -> str:
        """读取知识库新闻文档。"""
        return _mcp_call(
            "knowledge_base",
            "load_news_file",
            {"base_dir": kb_base_dir, "ticker": ticker.upper()},
        )

    tools_map = {
        "market_snapshot": market_snapshot,
        "fundamental_snapshot": fundamental_snapshot,
        "load_news_file": load_news_file,
    }
    route = REACT_TOOL_ROUTES.get(agent_name, ["market_snapshot", "fundamental_snapshot", "load_news_file"])
    tools = [tools_map[x] for x in route if x in tools_map]

    model = ChatOpenAI(
        base_url=config.llm_api_base,
        api_key=config.llm_api_key,
        model=config.llm_model,
        temperature=0.1,
        timeout=config.llm_timeout_seconds,
    )
    agent = create_react_agent(model=model, tools=tools)

    payload = {"messages": [("user", _build_react_prompt(agent_name, ticker, refs, topic_hint))]}
    step_budget = _react_step_budget(config, agent_name)
    recursion_limit = max(8, step_budget * 4)

    try:
        out = _run_async(agent.ainvoke(payload, config={"recursion_limit": recursion_limit}))
    except Exception as exc:  # noqa: BLE001
        _log_react(logger, agent_name, "failed", f"ainvoke_error: {exc}")
        return None

    text = _extract_last_message_text(out)
    obj = _extract_json_block(text)
    if not obj:
        _log_react(logger, agent_name, "failed", "react_output_not_json")
        return None

    ok, reason, conc = _validate_payload(obj, min_conclusions=3)
    if not ok:
        _log_react(logger, agent_name, "failed", f"react_output_invalid: {reason}")
        return None

    fallback_ref = refs[0][0] if refs else 1
    analysis = str(obj.get("analysis", "")).strip()
    conclusions = [_ensure_cited(x, fallback_ref) for x in conc[:4]]
    _log_react(logger, agent_name, "ok", f"conclusions={len(conclusions)}; tools={route}; steps={step_budget}")
    return analysis, conclusions

class FundamentalAgent(BaseAgent):
    name = "基本面"

    def __init__(
        self,
        llm: Any | None = None,
        retry_max: int = 2,
        config: Any | None = None,
        mcp_client: Any | None = None,
        kb_base_dir: str = "data/knowledge_base",
        logger: Any | None = None,
    ) -> None:
        self.llm = llm
        self.retry_max = retry_max
        self.config = config
        self.mcp_client = mcp_client
        self.kb_base_dir = kb_base_dir
        self.logger = logger

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        refs = _select_evidence(evidence, "fundamental", limit=3, ticker=ticker)
        react_out = _run_react_mcp_agent(
            agent_name=self.name,
            topic_hint="盈利质量、现金流、营收与利润增速",
            ticker=ticker,
            refs=refs,
            llm=self.llm,
            config=self.config,
            mcp_client=self.mcp_client,
            kb_base_dir=self.kb_base_dir,
            logger=self.logger,
        )
        if react_out is not None:
            analysis, conclusions = react_out
            return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)

        first_ref = refs[0][0] if refs else 1

        metrics: list[str] = []
        core_metric_hits = 0
        for ref, ev in refs:
            kv = _kv_from_text(
                ev.content,
                ["营收增速", "利润增速", "毛利率", "营业利润率", "净利率", "ROE", "自由现金流", "总现金"],
            )
            if kv:
                core_metric_hits += len(kv)
                metrics.append(f"- [E{ref}] " + "，".join(f"{k}={v}" for k, v in kv.items()))

        min_metrics = max(1, int(getattr(self.config, "fundamental_min_metrics", 3) or 3))
        if core_metric_hits < min_metrics:
            runtime_metrics = _fetch_runtime_fundamental_snapshot(self.mcp_client, ticker)
            if runtime_metrics:
                core_metric_hits += len(runtime_metrics)
                metrics.append("- [runtime] " + "，".join(f"{k}={v}" for k, v in runtime_metrics.items()))

        if core_metric_hits < min_metrics:
            analysis = "\n".join(
                [
                    f"{ticker} 当前基本面证据不足，暂不输出方向性基本面结论。",
                    "",
                    "框架判断：",
                    "- 基本面结论必须以营收、利润率、现金流等可审计字段为前提。",
                    "- 当前数据仅能支持‘观察状态’，不能支持‘方向判断’。",
                    "",
                    "证据解读：",
                    *([_evidence_line(ref, ev, max_len=110) for ref, ev in refs[:2]] or ["- 暂无可用基本面证据。"]),
                    "",
                    "关键缺口（至少需补齐其中3项）：",
                    "- 营收增速、利润增速、毛利率",
                    "- ROE、自由现金流、总现金",
                    "- 经营现金流/净利润比率或资本开支口径",
                    "",
                    "风险提示：",
                    "- 在核心财务字段缺失时做方向性判断，容易放大误判风险。",
                    "",
                    "执行建议：",
                    "- 本轮报告仅保留观察结论，不给出基本面驱动的加减仓建议。",
                    "- 下一轮优先抓取最新 10-Q/10-K 结构化财务字段再评估。",
                ]
            )
            conclusions = [
                _ensure_cited(f"{ticker} 基本面缺少可审计财务字段，当前结论仅可作观察性参考 [E{first_ref}]", first_ref),
                _ensure_cited("未拿到营收/利润/现金流核心指标前，不应做基本面方向判断 [E1]", first_ref),
                _ensure_cited("补齐结构化财务字段后再重启基本面打分与仓位建议 [E1]", first_ref),
            ]
            return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)

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

    def __init__(
        self,
        llm: Any | None = None,
        retry_max: int = 2,
        config: Any | None = None,
        mcp_client: Any | None = None,
        kb_base_dir: str = "data/knowledge_base",
        logger: Any | None = None,
    ) -> None:
        self.llm = llm
        self.retry_max = retry_max
        self.config = config
        self.mcp_client = mcp_client
        self.kb_base_dir = kb_base_dir
        self.logger = logger

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        refs = _select_evidence(evidence, "technical", limit=3, ticker=ticker)
        react_out = _run_react_mcp_agent(
            agent_name=self.name,
            topic_hint="趋势方向、均线关系、量价配合、关键支撑阻力",
            ticker=ticker,
            refs=refs,
            llm=self.llm,
            config=self.config,
            mcp_client=self.mcp_client,
            kb_base_dir=self.kb_base_dir,
            logger=self.logger,
        )
        if react_out is not None:
            analysis, conclusions = react_out
            return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)

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

    def __init__(
        self,
        llm: Any | None = None,
        retry_max: int = 2,
        config: Any | None = None,
        mcp_client: Any | None = None,
        kb_base_dir: str = "data/knowledge_base",
        logger: Any | None = None,
    ) -> None:
        self.llm = llm
        self.retry_max = retry_max
        self.config = config
        self.mcp_client = mcp_client
        self.kb_base_dir = kb_base_dir
        self.logger = logger

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        refs = _select_evidence(evidence, "valuation", limit=3, ticker=ticker)
        react_out = _run_react_mcp_agent(
            agent_name=self.name,
            topic_hint="PE/PB 与增长匹配、安全边际、估值压缩风险",
            ticker=ticker,
            refs=refs,
            llm=self.llm,
            config=self.config,
            mcp_client=self.mcp_client,
            kb_base_dir=self.kb_base_dir,
            logger=self.logger,
        )
        if react_out is not None:
            analysis, conclusions = react_out
            return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)

        first_ref = refs[0][0] if refs else 1

        metrics: list[str] = []
        metric_count = 0
        for ref, ev in refs:
            kv = _kv_from_text(ev.content, ["PE", "Forward PE", "P/B", "Trailing PE", "price_to_book", "营收增速", "利润增速"])
            if kv:
                metric_count += len(kv)
                metrics.append(f"- [E{ref}] " + "，".join(f"{k}={v}" for k, v in kv.items()))

        min_metrics = max(1, int(getattr(self.config, "valuation_min_metrics", 2) or 2))
        if metric_count < min_metrics:
            runtime = _fetch_runtime_valuation_metrics(self.mcp_client, ticker)
            if runtime:
                metric_count += len(runtime)
                metrics.append("- [runtime] " + "，".join(f"{k}={v}" for k, v in runtime.items()))

        if metric_count < min_metrics:
            analysis = "\n".join(
                [
                    f"{ticker} 当前估值证据不足，暂不输出方向性估值结论。",
                    "",
                    "框架判断：",
                    "- 估值分析至少需要倍数锚（PE/PB/EV类）与增长锚（营收/利润增速）。",
                    "- 当前证据仅支持观察，不支持估值驱动的仓位调整。",
                    "",
                    "证据解读：",
                    *([_evidence_line(ref, ev, max_len=110) for ref, ev in refs[:2]] or ["- 暂无可用估值证据。"]),
                    "",
                    "关键缺口：",
                    "- PE/Forward PE/PB 至少两项可用",
                    "- 营收增速或利润增速至少一项可用",
                    "",
                    "风险提示：",
                    "- 缺失估值锚时进行高低估判断，误差会显著放大。",
                ]
            )
            conclusions = [
                _ensure_cited(f"{ticker} 估值关键字段缺失，当前仅能维持观察结论 [E{first_ref}]", first_ref),
                _ensure_cited("缺少倍数锚与增长锚前，不应做估值驱动加减仓 [E1]", first_ref),
                _ensure_cited("补齐 PE/PB 与增长字段后再重启估值判断 [E1]", first_ref),
            ]
            return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)

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

    def __init__(
        self,
        llm: Any | None = None,
        retry_max: int = 2,
        news_engine: Any | None = None,
        config: Any | None = None,
        mcp_client: Any | None = None,
        kb_base_dir: str = "data/knowledge_base",
        logger: Any | None = None,
    ) -> None:
        self.llm = llm
        self.retry_max = retry_max
        self.news_engine = news_engine
        self.config = config
        self.mcp_client = mcp_client
        self.kb_base_dir = kb_base_dir
        self.logger = logger

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        refs = _select_evidence(evidence, "news", limit=3, ticker=ticker)
        if not refs:
            refs = _select_evidence(evidence, "fundamental", limit=2, ticker=ticker)

        event_count = _news_event_signal_count(refs)
        min_events = max(1, int(getattr(self.config, "news_min_events", 2) or 2))

        react_out = _run_react_mcp_agent(
            agent_name=self.name,
            topic_hint="事件强度、预期变化、情绪扩散与风险传导",
            ticker=ticker,
            refs=refs,
            llm=self.llm,
            config=self.config,
            mcp_client=self.mcp_client,
            kb_base_dir=self.kb_base_dir,
            logger=self.logger,
        )
        if react_out is not None and event_count >= min_events:
            analysis, conclusions = react_out
            return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)

        scored_rows: list[str] = []
        sent_lv: list[int] = []
        risk_lv: list[int] = []
        for ref, ev in refs[:3]:
            r = infer_news_sentiment_and_risk_5level(ev.content, engine=self.news_engine)
            sent_lv.append(r.sentiment_level)
            risk_lv.append(r.risk_level)
            scored_rows.append(
                f"- [E{ref}] 情感={r.sentiment_level}({r.sentiment_label})，风险={r.risk_level}({r.risk_label})，触发词={','.join(r.reasons[:3]) or '无'}"
            )

        avg_sent = round(sum(sent_lv) / max(len(sent_lv), 1), 2)
        avg_risk = round(sum(risk_lv) / max(len(risk_lv), 1), 2)
        first_ref = refs[0][0] if refs else 1
        second_ref = refs[1][0] if len(refs) > 1 else first_ref

        if event_count < min_events:
            analysis = "\n".join(
                [
                    f"{ticker} 当前新闻事件证据不足，暂不输出方向性新闻结论。",
                    "",
                    "框架判断：",
                    "- 新闻分析至少需要可定位事件（标题/时间/来源）与影响路径。",
                    "- 当前仅可给出观察级情绪信号，不支持事件驱动交易。",
                    "",
                    "证据解读：",
                    *([_evidence_line(ref, ev, max_len=110) for ref, ev in refs[:2]] or ["- 暂无可用新闻证据。"]),
                    "",
                    "风险提示：",
                    "- 缺乏事件粒度时，新闻情绪容易被噪声放大。",
                ]
            )
            conclusions = [
                _ensure_cited(f"{ticker} 新闻事件粒度不足，当前结论仅作观察参考 [E{first_ref}]", first_ref),
                _ensure_cited("未形成可验证事件链前，不应触发事件驱动加减仓 [E1]", first_ref),
                _ensure_cited("补齐事件时间线与来源后再重启新闻维度打分 [E1]", first_ref),
            ]
            return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)

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









