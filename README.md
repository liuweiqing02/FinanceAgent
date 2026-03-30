# 金融智能投顾研报系统（Multi-Agent + RAG + 向量数据库）

本项目提供一个可运行、可测试、可展示的金融研报系统，输入股票代码即可生成**带证据引用**的 Markdown 研报。

## 1. 功能特性

- LangGraph 多 Agent（基本面、技术面、估值、新闻并行，最后总结）
- MCP 工具协议与数据工具调用
- 情感/风险推理能力
- Markdown 研报输出
- RAG 全链路：真实采集 -> 清洗 -> 聚合构建 -> 切块 -> 向量化 -> 检索 -> 重排 -> 生成 -> 引用
- 真实数据源支持：SEC 财报/公告（10-K/10-Q/8-K）+ Google News RSS
- 向量数据库：默认 Chroma（不可用时自动回退内存模式）
- 向量库抽象接口 `VectorStore`（便于切换 Milvus/pgvector）
- Hybrid Retrieval（BM25 + 向量检索）
- 强制证据引用与可追溯报告
- 检索与生成可观测日志（query、召回、得分、结论）

## 2. 目录结构

```text
app/
  agents/       # 多 Agent 及编排
  mcp/          # MCP 协议与工具
  rag/          # RAG 管道、真实采集、构建、检索、重排、向量库
  models/       # 数据结构与风险模型
  infra/        # 基础设施（日志等）
  reports/      # Markdown 报告生成
  main.py       # 统一启动入口

data/
  raw/            # 原始事件流数据（jsonl）
  knowledge_base/ # 构建后的知识库文档（md）

tests/
  unit/
  integration/
  e2e/
```

## 3. 快速启动

### 3.1 安装

```bash
pip install -r requirements.txt
```

### 3.2 配置

```bash
cp .env.example .env
```

建议把 `SEC_USER_AGENT` 改成你自己的标识（SEC 接口要求）。

### 3.3 构建真实知识库（批量）

```bash
python -m app.rag.ingest_build --mode real --tickers AAPL,TSLA
```

如果你在离线环境，也可自动回退：

```bash
python -m app.rag.ingest_build --mode auto --tickers AAPL,TSLA
```

### 3.4 生成研报

```bash
python -m app.main --ticker AAPL --kb-mode auto
```

输出文件默认在 `reports_output/`，例如：`reports_output/AAPL_report.md`。

## 4. 真实采集流程说明

1. `real_collectors.py` 拉取 SEC `company_tickers.json` 建立 ticker-cik 映射。  
2. 拉取 `submissions/CIKxxxxx.json` 获取最新 10-K/10-Q/8-K。  
3. 下载 filing 正文并抽取纯文本，写入 `data/raw/real_events.jsonl`。  
4. 额外从 Google News RSS 拉取相关新闻摘要并统一落盘。  
5. `knowledge_builder.py` 按 `ticker + topic` 聚合为 `data/knowledge_base/*.md` 多段文档。  
6. RAG 管道对这些文档做切块、向量化、检索、重排和引用生成。  

## 5. 测试

```bash
pytest
```

测试覆盖：
- 单元测试：切块、检索、情感/风险、知识库构建、真实采集器解析、内容质量
- 集成测试：报告生成与富文本结构验收
- e2e smoke：命令行全流程
