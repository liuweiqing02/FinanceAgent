# 04 多 Agent 编排与报告生成

## 4.1 模块边界

本章节对应：

- 编排器：`app/agents/orchestrator.py`
- 专家 Agent：`app/agents/specialists.py`
- 总结 Agent：`app/agents/summary_agent.py`
- 报告输出：`app/reports/writer.py`

## 4.2 编排器初始化

类：`FinanceResearchOrchestrator`

初始化动作：

1. 创建 `JsonlLogger`（`logs/rag_trace.jsonl`）
2. 创建 `RagPipeline`
3. 初始化 MCP 多服务器客户端
4. 注册工具服务器：
   - `finance_data.market_snapshot`
   - `finance_data.fundamental_snapshot`
   - `knowledge_base.load_news_file`
5. 构建 LLM 与新闻风险模型
6. 创建四个专家 Agent + SummaryAgent

这一步完成后，系统同时具备“RAG 证据”和“MCP 快照工具”两种证据通路。

## 4.3 运行主流程

入口：`orchestrator.run(ticker)`

主链路：

1. `_retrieve_scoped_evidence(ticker)` 按主题检索证据
2. 调用 MCP 获取市场/基本面快照
3. `_inject_mcp_evidence(...)` 将快照注入 Evidence 列表
4. `_run_agents(...)` 并行运行四个专家 Agent
5. `SummaryAgent.run(...)` 聚合四维结论
6. `build_markdown_report(...)` 生成最终 markdown

## 4.4 主题化证据检索策略

函数：`_retrieve_scoped_evidence`

- 固定四个 query 模板：fundamental/technical/valuation/news
- 每个主题分别调用 `rag.retrieve(..., ticker=..., topic=...)`
- 基于 `chunk_id` 去重后合并
- 总量上限 `max_total = max(8, top_k * 3)`

这样可以降低“某一主题证据缺失”导致的偏科输出。

## 4.5 MCP 证据注入

函数：`_inject_mcp_evidence`

注入项：

- `mcp_market_snapshot`（价格、日内波动、PE/PB、市值）
- `mcp_fundamental_snapshot`（增速、利润率、ROE、现金流）

降级逻辑：

- 若 fundamental 快照不可用，注入 `mcp_fundamental_unavailable` 提醒
- 避免在缺关键字段时误给强结论

## 4.6 专家 Agent 机制

### 4.6.1 证据选择

公共函数：`_select_evidence`

- 优先选择 topic 匹配且可读证据
- 支持 ticker 匹配约束
- 新闻主题避免回退到非新闻证据（降低污染）

### 4.6.2 生成模式

每个 Agent 都支持两种路径：

- ReAct + MCP（配置开启时）
- 模板分析 + LLM 重写（默认路径）

### 4.6.3 各 Agent 职责

- FundamentalAgent：盈利质量、增长、现金流、财务完整性
- TechnicalAgent：趋势、均线、量价关系、交易风险
- ValuationAgent：估值锚、增长匹配、安全边际
- NewsAgent：事件强度、情绪风险、触发条件

### 4.6.4 新闻专项门控

`NewsAgent` 使用 `_news_event_signal_count` 判断事件可用性：

- 来源是否为新闻源
- 是否具备时间标识/事件标记
- 文本是否不是模板噪声

若事件数低于阈值（`NEWS_MIN_EVENTS`），只输出“观察级结论”。

## 4.7 LLM 重写与兜底

公共逻辑：`_llm_rewrite(...)` / `_llm_rewrite_summary(...)`

约束：

- 必须输出 JSON（analysis + conclusions）
- conclusions 数量和引用格式受校验
- 禁止引用证据外来源（如 Bloomberg/FactSet 未在证据中）

降级：

1. JSON 校验失败 -> 纯文本抽取
2. 再失败 -> 模板结果 + 失败原因注记

## 4.8 SummaryAgent 聚合逻辑

输入：四个专家输出 + 证据列表

动作：

- 合并文本后调用 `infer_sentiment_and_risk`
- 生成统一“执行摘要 + 风险建议 + 操作建议”
- 结论统一补齐 `[E#]` 引用
- 可选 LLM 重写提升可读性

## 4.9 报告生成与质量门控

函数：`build_markdown_report(...)`

输出结构：

1. 标题与时间
2. 执行摘要
3. 公司概况（若 market snapshot 可用）
4. 多 Agent 分析章节
5. 综合结论
6. 数据充分性评估
7. 交易计划（满足条件时）
8. 证据清单

交易计划门控：`_can_emit_trade_plan`

- 必须有可用 market snapshot
- 高质量证据数 >= 3
- 技术或估值可读证据至少 2 条

## 4.10 运行模式与并行策略

`_run_agents(...)` 支持：

- LangGraph 并行执行（`ENABLE_LANGGRAPH=true`）
- ThreadPool 并发回退（LangGraph 异常时）

并发收益：

- 降低总体报告延迟
- 各主题分析互不阻塞

## 4.11 关键配置项

- `ENABLE_LANGGRAPH`
- `REACT_MCP_ENABLED` / `REACT_MCP_MAX_STEPS`
- `NEWS_REACT_MCP_ENABLED` / `NEWS_REACT_MAX_STEPS`
- `LLM_ENABLED` 与 LLM 连接参数
- `NEWS_MIN_EVENTS`（新闻结论门控）

## 4.12 典型排查路径

- 结论无引用：先检查 Agent 输出 JSON 与 Writer 拼接逻辑
- 新闻结论偏弱：检查 `news_events` 是否达阈值
- 维度缺失：检查 `_retrieve_scoped_evidence` 的 topic 命中情况
- 报告无交易计划：检查 `_can_emit_trade_plan` 的三项门控是否满足