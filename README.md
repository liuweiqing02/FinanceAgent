# 金融智能投顾研报系统

本项目是一个可运行、可测试、可扩展的金融投研自动化系统。输入股票代码后，系统自动完成真实数据采集、RAG 检索、多 Agent 分析与证据引用报告生成，输出可追溯的 Markdown 研报。

## 核心亮点

| 能力 | 说明 |
| --- | --- |
| 多 Agent 并行分析 | 基本面、技术面、估值、新闻四个专家 Agent 并行执行，最后统一汇总 |
| 证据驱动研报 | 报告结论绑定证据编号，支持回溯到来源与内容 |
| 完整 RAG 链路 | 采集 -> 构建 -> 切块 -> 向量化 -> 混合检索 -> 重排 -> 生成 |
| 真实数据接入 | SEC 财报/公告 + 新闻 RSS + 运行时快照 |
| 组件可替换 | Embedding、Reranker、VectorStore、LLM 均可按配置切换 |
| 异常可回退 | 向量库、模型、网络异常时具备兜底路径，保证流程可运行 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化配置

```bash
cp .env.example .env
```

建议将 `SEC_USER_AGENT` 改成你自己的标识（SEC 接口要求）。

### 3. 构建知识库

```bash
python -m app.rag.ingest_build --mode auto --tickers AAPL,TSLA
```

### 4. 生成研报

```bash
python -m app.main --ticker AAPL --kb-mode auto
```

输出示例：`reports_output/AAPL_report.md`

## 模型配置示例

### 阿里云百炼（OpenAI-Compatible）

```bash
LLM_ENABLED=true
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的百炼API Key
LLM_MODEL=qwen-plus
```

### OpenAI

```bash
LLM_ENABLED=true
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=你的OpenAI API Key
LLM_MODEL=gpt-4.1-mini
```

说明：未开启 `LLM_ENABLED` 时，系统自动回退为规则模板模式（离线可跑）。

## 文档导航

详细实现已拆分到 `docs/`，建议按以下顺序阅读：

| 章节 | 内容 |
| --- | --- |
| [01-system-architecture](docs/01-system-architecture.md) | 系统分层、端到端流程、关键设计原则 |
| [02-data-collection-kb](docs/02-data-collection-kb.md) | 真实采集、增量合并、知识库构建、回填机制 |
| [03-rag-retrieval](docs/03-rag-retrieval.md) | 索引构建、混合检索、重排策略、证据后处理 |
| [04-agent-workflow](docs/04-agent-workflow.md) | 编排器、专家 Agent、总结逻辑、报告门控 |
| [05-news-model](docs/05-news-model.md) | 本地新闻情感/风险 LoRA 模型训练与启用 |
| [06-config-ops](docs/06-config-ops.md) | 配置项、运行命令、可观测性与运维建议 |
| [07-testing](docs/07-testing.md) | 测试分层、回归重点与质量基线 |

## 常用命令速查

```bash
# 批量构建真实知识库
python -m app.rag.ingest_build --mode real --tickers AAPL,TSLA

# 自动模式（在线失败自动回退）
python -m app.rag.ingest_build --mode auto --tickers AAPL,TSLA

# 生成单标的研报
python -m app.main --ticker AAPL --kb-mode auto

# 运行测试
pytest
```

## 项目结构

```text
app/
  agents/       # 多 Agent 与编排
  mcp/          # MCP 协议与工具调用
  rag/          # 采集、构建、检索、重排、向量库
  models/       # LLM 与本地模型
  infra/        # 日志等基础设施
  reports/      # 报告输出
  main.py       # 统一入口

data/
  raw/            # 原始事件流数据（jsonl）
  knowledge_base/ # 构建后的知识库文档（md）

tests/
  unit/
  integration/
  e2e/
```
