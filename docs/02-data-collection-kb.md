# 02 真实采集与知识库构建

## 2.1 模块边界

本章节对应两条主链路：

- 采集链路：`app/rag/real_collectors.py`
- 构建链路：`app/rag/knowledge_builder.py`

输入是 ticker 列表，输出是两层数据：

1. 原始事件流：`data/raw/real_events.jsonl`
2. 聚合知识库：`data/knowledge_base/<TICKER>_<topic>.md`

## 2.2 采集入口与执行顺序

主入口函数：`collect_real_raw_data(...)`

按 ticker 执行顺序如下：

1. SEC 财报/公告采集（10-K/10-Q/8-K）
2. 新闻采集（Google News RSS + Yahoo Finance RSS）
3. 技术面快照（Yahoo chart，失败时可回退 runtime tool）
4. 基本面快照（runtime tool 或 Yahoo quoteSummary）
5. 估值快照（runtime tool 或 Yahoo quote/quoteSummary）
6. 统一字段标准化与增量合并

关键实现点：

- `news_limit` 为每个 ticker 限制，内部会放大候选再截断。
- 所有事件在落盘前统一走 `_enrich_raw_row`，补齐 `doc_id/hash/updated_at`。
- 落盘前执行 `_merge_incremental_rows`，避免全量覆盖历史 ticker 数据。

## 2.3 SEC 采集实现细节

### 2.3.1 映射与拉取

- `_load_cik_mapping`：读取 `https://www.sec.gov/files/company_tickers.json`，构建 `ticker -> cik`。
- `_collect_sec_filings`：读取 `submissions/CIK{cik}.json`，筛选表单类型：`10-K`、`10-Q`、`8-K`。

### 2.3.2 内容抽取与主题映射

- 抓取 filing HTML 后，先 `_extract_text_from_html` 去标签与噪声 token。
- 再 `_summarize_financial_text` 做句级筛选，保留金融相关可读句。
- `8-K` 统一归入 `fundamental` 主题，避免新闻主题被公告文本挤占。
- 对 `10-K/10-Q` 额外生成一条 `valuation` 事件（`_extract_valuation_paragraphs`）。

## 2.4 新闻采集实现细节

### 2.4.1 多源召回

函数：`_collect_news_rows(...)`

- 源 1：`_collect_google_news_rss`
- 源 2：`_collect_yahoo_finance_rss`

候选放大策略：

- `per_source_limit = max(8, news_limit * 2)`
- 先放大量召回，再做时效与去重，再按 `news_limit` 截断。

### 2.4.2 相关性与清洗

- `_news_queries_for_ticker`：ticker + alias + 主题词（stock/earnings/outlook）。
- `_is_ticker_relevant`：判断标题/摘要/链接是否与 ticker 或 alias 相关。
- `_collect_article_snippet`：尝试补抓正文摘要；对 script-heavy 页面直接跳过。
- `_is_web_noise`：过滤前端脚本噪声文本（如 `window.yahoo`、`googletag` 等）。

### 2.4.3 时效与去重

- `_is_recent_news`：默认 7 天窗口（`_NEWS_LOOKBACK_DAYS = 7`）。
- `_normalize_published_at`：统一 RFC822 / ISO 时间到 UTC ISO 格式。
- `_news_dedup_key`：基于 `ticker/title/url/date` 做稳定 hash 去重。
- 去重冲突时，保留 `content` 更长的版本。

## 2.5 原始事件标准化

统一字段函数：`_enrich_raw_row(row)`

标准化内容：

- ticker/topic 大小写归一
- `updated_at` 自动补全（优先 `updated_at`，其次 `collected_at`）
- `hash` 自动生成（内容归一后 SHA256）
- `doc_id` 自动生成（`ticker_topic_hash`）

这一层是后续增量合并和可追溯的基础。

## 2.6 增量合并策略

函数：`_merge_incremental_rows(out_file, new_rows, target_tickers)`

行为规则：

1. 读取已有 `real_events.jsonl`
2. 保留非目标 ticker 的历史数据
3. 用本轮目标 ticker 新数据覆盖同 `doc_id` 项
4. 结果按 `(ticker, topic, published_at)` 排序落盘

收益：

- 重复运行只更新目标 ticker
- 不会破坏其他 ticker 事件
- 支持持续追加与定时重建

## 2.7 知识库构建流程

入口：`build_knowledge_base(raw_dir, knowledge_base_dir, include_glob)`

### 2.7.1 读取与分组

- `_load_raw_items` 读取 jsonl 并转 `RawKnowledgeItem`
- 按 `(ticker, topic)` 分组

### 2.7.2 文档渲染

- `_render_topic_markdown` 生成主题文档
- 默认主题映射：`fundamental/technical/valuation/news`
- 每个事件写入：标题、发布时间、来源、链接、内容解读

### 2.7.3 增量输出与清理

- 对每个分组计算 `content_hash`
- 与 `_kb_manifest.json` 比较，未变化则跳过重写
- 清理已失效旧文档（manifest 中有，本轮无）

## 2.8 完整度评估与回填

实现文件：`app/rag/evidence_backfill.py`

- `evaluate_kb_completeness`：统计三类指标
  - `fundamental_metrics`
  - `valuation_metrics`
  - `news_events`
- `backfill_kb_until_ready`：不满足阈值时自动提升抓取轮次
  - `filing_limit/news_limit` 逐轮增加
  - 每轮后重建 KB 再评估

这层用于解决“报告能跑但新闻/指标不足”的问题。

## 2.9 关键配置与参数

- `SEC_USER_AGENT`：SEC 抓取必填
- `RAW_DATA_DIR` / `KNOWLEDGE_BASE_DIR`
- `FUNDAMENTAL_MIN_METRICS`
- `VALUATION_MIN_METRICS`
- `NEWS_MIN_EVENTS`

建议先在 `auto` 模式运行，确认网络与权限后再切到 `real`。

## 2.10 典型故障与排查

- 新闻为空：先查 `real_events.jsonl` 是否存在 `topic=news` 记录。
- 构建失败：检查 `data/knowledge_base` 写权限与文件占用。
- 采集失败：检查 `SEC_USER_AGENT`、网络连通、代理设置。
- 指标不足：启用 `--evidence-backfill` 自动补齐。