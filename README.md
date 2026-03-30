# 金融智能投顾研报系统（Multi-Agent + RAG + 向量数据库）

本项目提供一个可运行、可测试、可展示的金融研报系统，输入股票代码即可生成**带证据引用**的 Markdown 研报。

## 1. 功能特性

- LangGraph 多 Agent（基本面、技术面、估值、新闻并行，最后总结）
- MCP 工具协议与数据工具调用
- 情感/风险推理能力
- Markdown 研报输出
- RAG 全链路：原始数据采集 -> 清洗 -> 聚合构建 -> 切块 -> 向量化 -> 检索 -> 重排 -> 生成 -> 引用
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
  rag/          # RAG 管道、构建、检索、重排、向量库
  models/       # 数据结构与风险模型
  infra/        # 基础设施（日志等）
  reports/      # Markdown 报告生成
  main.py       # 统一启动入口

data/
  raw/          # 原始事件流数据（jsonl）
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

### 3.3 构建知识库（真实流程）

```bash
python -m app.rag.ingest_build
```

### 3.4 生成研报

```bash
python -m app.main --ticker AAPL
```

输出文件默认在 `reports_output/`，例如：`reports_output/AAPL_report.md`。

## 4. RAG 流程说明

1. 原始数据采集：读取 `data/raw/*.jsonl`（财报、技术、估值、新闻事件）。
2. 规范化清洗：字段标准化、文本清洗、来源保留。
3. 知识库构建：按 `ticker + topic` 聚合为 `data/knowledge_base/*.md`。
4. 切块：按长度与重叠切分 chunk。
5. 向量化：`HashEmbedding` 离线向量器。
6. 存储：写入 `ChromaVectorStore`。
7. 检索：Hybrid（BM25 + 向量相似度融合）。
8. 重排：基于关键词覆盖的轻量重排。
9. 生成：多 Agent 输出 + 总结 Agent 汇总。
10. 引用：关键结论强制带 `[E#]`，并输出证据清单。

## 5. 演示说明

1. 执行 `python -m app.rag.ingest_build` 生成知识库。
2. 执行 `python -m app.main --ticker AAPL` 生成研报。
3. 检查综合结论是否均带 `[E#]` 引用。
4. 查看 `logs/rag_trace.jsonl` 观测检索与生成过程。

## 6. 测试

```bash
pytest
```

测试覆盖：
- 单元测试：切块、检索、情感/风险、知识库构建、内容质量
- 集成测试：报告生成与富文本结构验收
- e2e smoke：命令行全流程
