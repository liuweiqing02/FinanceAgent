# 金融智能投顾研报系统（Multi-Agent + RAG + 向量数据库）

本项目从零构建一个可运行、可测试、可部署展示的金融研报系统，支持输入股票代码后一键生成**带证据引用**的 Markdown 研报。

## 1. 功能特性

- LangGraph 多 Agent（基本面、技术面、估值、新闻、总结）
- MCP 工具协议与数据工具调用
- 情感/风险推理能力
- Markdown 研报输出
- RAG 全链路：采集 -> 清洗 -> 切块 -> 向量化 -> 检索 -> 重排 -> 生成 -> 引用
- 向量数据库：默认 Chroma（不可用时自动回退内存模式，保证可运行）
- 向量库抽象接口 `VectorStore`（便于切换 Milvus/pgvector）
- Hybrid Retrieval（BM25 + 向量检索）
- 强制证据引用与可追溯报告
- 可观测性日志（query、召回、得分、生成）

## 2. 目录结构

```text
app/
  agents/       # 多 Agent 及编排
  mcp/          # MCP 协议与工具
  rag/          # RAG 管道、检索、重排、向量库
  models/       # 数据结构与风险模型
  infra/        # 基础设施（日志等）
  reports/      # Markdown 报告生成
  main.py       # 统一启动入口

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

### 3.3 运行

```bash
python -m app.main --ticker AAPL
```

输出文件默认在 `reports_output/`，例如：`reports_output/AAPL_report.md`。

## 4. RAG 流程说明

1. 采集：从 `data/knowledge_base/*.md` 读取原始文本。  
2. 清洗：标准化换行与空白。  
3. 切块：按长度与重叠切分 chunk。  
4. 向量化：`HashEmbedding` 离线向量器。  
5. 存储：写入 `ChromaVectorStore`。  
6. 检索：Hybrid（BM25 + 向量相似度融合）。  
7. 重排：基于关键词覆盖的轻量重排。  
8. 生成：多 Agent 输出 + 总结 Agent 汇总。  
9. 引用：关键结论强制带 `[E#]`，末尾输出证据清单。  

## 5. 评估与验收指标（建议）

- 引用覆盖率：关键结论中包含证据引用的比例（目标 100%）
- 检索命中质量：TopK 中与 query 高相关片段占比
- 报告完整性：是否包含多 Agent 分析、综合结论、证据清单
- 可观测性完备度：日志是否记录 query、召回、score、生成结论

## 6. 演示说明

1. 准备知识库文档：`data/knowledge_base/*.md`
2. 运行 `python -m app.main --ticker AAPL`
3. 打开生成研报，检查综合结论中是否含 `[E1]`、`[E2]`
4. 查看 `logs/rag_trace.jsonl` 观测检索与生成过程

## 7. 测试

```bash
pytest
```

测试包含：
- 单元测试：切块、检索、情感/风险模型
- 集成测试：从 orchestrator 到研报生成
- e2e smoke：命令行启动全流程
