# 01 系统架构总览

## 目标

系统核心目标是生成“可追溯证据链”的金融研报，而不是只输出自然语言总结。  
每条关键结论都应可回溯到证据编号与来源。

## 架构分层

1. 数据层：真实采集（SEC、RSS、市场快照）与增量事件流存储。  
2. 知识层：将原始事件聚合为可检索知识库文档。  
3. 检索层：切块、向量化、混合召回、重排。  
4. 分析层：多 Agent 并行分析（基本面/技术面/估值/新闻）。  
5. 生成层：总结归因、风险建议、引用绑定、Markdown 报告输出。  

## 端到端流程

1. 输入 `ticker`。  
2. 采集器写入 `data/raw/real_events.jsonl`。  
3. 构建器生成 `data/knowledge_base/<ticker>_<topic>.md`。  
4. RAG 管道检索并返回结构化证据集。  
5. 专家 Agent 输出分主题分析。  
6. Summary Agent 汇总为最终报告。  
7. Writer 输出 `reports_output/<ticker>_report.md`。  

## 关键设计原则

- 可回退：网络、模型、向量库异常时均有兜底路径。  
- 可替换：Embedding、Reranker、VectorStore、LLM 均可插拔。  
- 可观察：检索、重排、生成链路均可记录日志。  
- 可测试：核心链路覆盖 unit/integration/e2e。  
