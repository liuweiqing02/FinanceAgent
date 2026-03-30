from __future__ import annotations

from app.agents.base import BaseAgent
from app.models.schemas import AgentOutput, Evidence


def _citation(index: int) -> str:
    return f"[E{index}]"


def _safe_evidence(evidence: list[Evidence], idx: int) -> Evidence | None:
    if 0 <= idx < len(evidence):
        return evidence[idx]
    return None


def _evidence_line(evidence: list[Evidence], idx: int) -> str:
    ev = _safe_evidence(evidence, idx)
    if ev is None:
        return f"- {_citation(1)} 暂无可用证据，建议补充最新财报和公告。"
    return f"- {_citation(idx + 1)} {ev.title}（score={ev.score}）：{ev.content[:110].replace(chr(10), ' ')}"


def _ensure_cited(text: str, default_idx: int = 1) -> str:
    if "[E" in text:
        return text
    return f"{text} {_citation(default_idx)}"


class FundamentalAgent(BaseAgent):
    name = "基本面"

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        analysis = "\n".join(
            [
                f"{ticker} 基本面从盈利质量、现金流稳定性与业务结构三条主线展开。",
                "",
                "框架判断：",
                "- 盈利质量：关注毛利率与净利率变化，判断业绩增长是否可持续。",
                "- 经营韧性：关注现金流与资本开支匹配，识别扩张期资金压力。",
                "- 成长兑现：关注核心业务增速与新业务贡献，判断估值支撑强度。",
                "",
                "证据解读：",
                _evidence_line(evidence, 0),
                _evidence_line(evidence, 1),
                "",
                "风险提示：",
                "- 若盈利改善主要依赖一次性因素，后续可能出现回落。",
                "- 若需求端不及预期，估值扩张空间可能受限。",
                "",
                "跟踪建议：",
                "- 下次财报重点跟踪利润率、经营现金流和分业务收入结构。",
            ]
        )

        conclusions = [
            _ensure_cited(f"{ticker} 基本面当前偏稳健，具备中期跟踪价值 {_citation(1)}", 1),
            _ensure_cited(f"若后续盈利质量持续改善，估值中枢有上修可能 {_citation(2)}", 2),
            _ensure_cited("需警惕需求波动对利润兑现节奏的扰动 [E1]", 1),
        ]
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)


class TechnicalAgent(BaseAgent):
    name = "技术面"

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        analysis = "\n".join(
            [
                f"{ticker} 技术面围绕趋势、动量与关键支撑阻力位进行判断。",
                "",
                "框架判断：",
                "- 趋势方向：观察中期均线斜率与价格位置。",
                "- 交易热度：观察成交量变化与价格共振情况。",
                "- 风险位置：识别关键支撑位与止损触发区域。",
                "",
                "证据解读：",
                _evidence_line(evidence, 1),
                _evidence_line(evidence, 2),
                "",
                "风险提示：",
                "- 放量滞涨通常意味着短线分歧加大。",
                "- 跌破关键均线后，回撤幅度可能扩大。",
                "",
                "交易建议：",
                "- 采用分批建仓与分层止损，避免单点决策。",
            ]
        )

        conclusions = [
            _ensure_cited(f"{ticker} 当前更适合趋势跟随而非逆势抄底 {_citation(2)}", 2),
            _ensure_cited(f"若跌破关键支撑位，应优先执行风控纪律 {_citation(3)}", 3),
            _ensure_cited("短期波动或加大，仓位管理优先级高于收益博弈 [E2]", 2),
        ]
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)


class ValuationAgent(BaseAgent):
    name = "估值"

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        analysis = "\n".join(
            [
                f"{ticker} 估值分析采用“增长-估值匹配度”与“安全边际”双维度评估。",
                "",
                "框架判断：",
                "- 相对估值：结合行业可比公司估值区间进行横向比较。",
                "- 绝对估值：结合未来现金流与增长预期进行纵向验证。",
                "- 安全边际：评估当前价格对悲观场景的容忍度。",
                "",
                "证据解读：",
                _evidence_line(evidence, 0),
                _evidence_line(evidence, 3),
                "",
                "风险提示：",
                "- 若增长预期下修，估值溢价会快速收敛。",
                "- 若利率中枢上行，高估值资产承压更明显。",
                "",
                "策略建议：",
                "- 在估值扩张阶段控制追高节奏，在回调阶段关注性价比。",
            ]
        )

        conclusions = [
            _ensure_cited(f"{ticker} 估值需要与盈利兑现同步验证，不能仅看静态倍数 {_citation(1)}", 1),
            _ensure_cited(f"若增长保持韧性，当前估值仍有一定容纳空间 {_citation(4)}", 4),
            _ensure_cited("建议以安全边际为核心设定买入与加仓阈值 [E1]", 1),
        ]
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)


class NewsAgent(BaseAgent):
    name = "新闻"

    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        analysis = "\n".join(
            [
                f"{ticker} 新闻面分析关注事件冲击、预期差与情绪扩散路径。",
                "",
                "框架判断：",
                "- 事件强度：识别政策、供应链、产品周期等高影响事件。",
                "- 预期变化：判断新闻是否改变市场对利润和增长的预期。",
                "- 持续性：区分一次性噪音与中期趋势信号。",
                "",
                "证据解读：",
                _evidence_line(evidence, 2),
                _evidence_line(evidence, 4),
                "",
                "风险提示：",
                "- 新闻驱动行情通常波动更快，回撤也更陡。",
                "- 单一消息源可能带来偏差，需交叉验证。",
                "",
                "跟踪建议：",
                "- 建议建立“事件-预期-价格”联动表，每日滚动复核。",
            ]
        )

        conclusions = [
            _ensure_cited(f"{ticker} 新闻面仍是短期波动主因，需高频复盘 {_citation(3)}", 3),
            _ensure_cited(f"当前信息对中期逻辑偏中性到正向，但需继续验证 {_citation(5)}", 5),
            _ensure_cited("建议避免基于单条新闻进行重仓决策 [E3]", 3),
        ]
        return AgentOutput(name=self.name, analysis=analysis, conclusions=conclusions)
