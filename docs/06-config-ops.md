# 06 配置、运行与可观测性

## 核心环境变量

- `LLM_ENABLED`：是否启用 LLM。
- `LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL`：模型连接参数。
- `SEC_USER_AGENT`：SEC 抓取标识。
- `VECTOR_STORE_PROVIDER`：向量库实现选择。

## 常用命令

构建知识库：

```bash
python -m app.rag.ingest_build --mode auto --tickers AAPL,TSLA
```

生成研报：

```bash
python -m app.main --ticker AAPL --kb-mode auto
```

## 可观测性

建议观察以下日志维度：

- query 与召回候选数量
- 重排得分和最终证据序列
- Agent 输出与结论引用映射
- 失败回退原因（网络、模型、向量库）

## 稳定性建议

- 生产环境建议固定依赖版本。
- 定期清理或归档历史日志和临时数据。
- 将大模型权重和运行产物从 Git 仓库隔离。
