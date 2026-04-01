# 金融智能投顾研报系统

本项目是一个可运行、可测试、可扩展的投研自动化系统：输入股票代码后，自动完成真实数据采集、RAG 检索、多 Agent 分析与证据引用报告生成，输出可追溯的 Markdown 研报。

## 1. 功能特性

- LangGraph 多 Agent（基本面、技术面、估值、新闻并行，最后总结）
- MCP 工具协议与数据工具调用
- RAG 全链路：采集 -> 清洗 -> 构建 -> 检索 -> 重排 -> 生成 -> 引用
- 真实数据源支持：SEC 财报/公告 + 新闻 RSS
- 向量数据库：默认 Chroma（不可用自动回退内存模式）
- Hybrid Retrieval（BM25 + 向量检索）
- 可切换 Embedding（Hash / BGE / E5）与 Reranker（Simple / Cross-Encoder）
- 强制证据引用与可追溯报告
- 检索与生成日志可观测

## 2. 详细实现文档

- [01 系统架构总览](docs/01-system-architecture.md)
- [02 真实采集与知识库构建](docs/02-data-collection-kb.md)
- [03 RAG 检索与重排实现](docs/03-rag-retrieval.md)
- [04 多 Agent 编排与报告生成](docs/04-agent-workflow.md)
- [05 新闻情感/风险模型（本地 LoRA）](docs/05-news-model.md)
- [06 配置、运行与可观测性](docs/06-config-ops.md)
- [07 测试策略与质量保障](docs/07-testing.md)

## 3. 目录结构

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

## 4. 快速启动

### 4.1 安装

```bash
pip install -r requirements.txt
```

### 4.2 配置

```bash
cp .env.example .env
```

建议把 `SEC_USER_AGENT` 改成你自己的标识（SEC 接口要求）。

### 4.3 大模型接口（百炼 / OpenAI 兼容）

本项目支持 OpenAI-Compatible 协议，只要兼容 `/chat/completions` 即可接入。

阿里云百炼示例：

```bash
LLM_ENABLED=true
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的百炼API Key
LLM_MODEL=qwen-plus
```

OpenAI 示例：

```bash
LLM_ENABLED=true
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=你的OpenAI API Key
LLM_MODEL=gpt-4.1-mini
```

说明：
- 未开启 `LLM_ENABLED` 时，系统自动回退为规则模板模式（离线可跑）。

### 4.4 构建真实知识库（批量）

```bash
python -m app.rag.ingest_build --mode real --tickers AAPL,TSLA
```

离线环境可使用自动回退：

```bash
python -m app.rag.ingest_build --mode auto --tickers AAPL,TSLA
```

### 4.5 生成研报

```bash
python -m app.main --ticker AAPL --kb-mode auto
```

输出文件默认在 `reports_output/`，例如：`reports_output/AAPL_report.md`。

## 5. 测试

```bash
pytest
```

## 6. 备注

- 本地新闻 LoRA 训练与启用步骤见：[05 新闻情感/风险模型（本地 LoRA）](docs/05-news-model.md)
- 如果你需要“部署版文档”（Docker、CI、生产参数建议），可在 `docs/` 继续补充。
