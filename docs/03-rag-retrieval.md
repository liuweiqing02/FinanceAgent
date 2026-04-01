# 03 RAG 检索与重排实现

## 3.1 模块边界

本章节对应以下模块：

- `app/rag/pipeline.py`（总管道）
- `app/rag/retriever.py`（HybridRetriever）
- `app/rag/reranker.py`（Simple/CrossEncoder）
- `app/rag/embeddings.py`（Hash/SentenceTransformer）
- `app/rag/vectorstore/chroma_store.py`（Chroma + 内存回退）
- `app/rag/chunking.py`（切块）

## 3.2 建索引流程

入口：`RagPipeline.build_index()`

执行步骤：

1. `collect_documents` 读取知识库 markdown 文档
2. `clean_text` 清洗噪声（url、xbrl/us-gaap、长 token 等）
3. `chunk_text` 按字符切块（默认 `chunk_size=400`，`overlap=80`）
4. 计算 source 级哈希，对比 index manifest
5. 仅对变更 source 执行向量 upsert
6. BM25 语料全量重建

增量机制：

- `*_index_manifest.json` 记录 source hash
- 删除已移除 source 的向量条目
- 仅更新 changed source 的 embedding

## 3.3 向量层实现

类：`ChromaVectorStore`

### 3.3.1 双后端策略

- 优先 Chroma 持久化（`PersistentClient`）
- 不可用时自动回退内存数组 `_mem`

### 3.3.2 查询与得分

- Chroma 返回 distance，转为 `score = 1 / (1 + distance)`
- 内存模式使用余弦相似度 `_cosine`

### 3.3.3 删除策略

- 支持 `delete_by_source_ids`，用于增量重建前清理旧 source

## 3.4 Embedding 组件

入口：`build_embedding_from_config(config)`

支持：

- `HashEmbedding`：完全离线，无模型依赖
- `SentenceTransformerEmbedding`：本地语义向量

失败回退：

- 本地模型加载异常时，自动回退 `HashEmbedding`
- 并在日志写入 `embedding_runtime` 事件

## 3.5 混合召回（HybridRetriever）

类：`HybridRetriever`

计算逻辑：

1. 向量召回：`vector_store.query(q_emb, top_k=max(top_k*2, 10))`
2. 词法召回：BM25（内部维护 `tf/df`）
3. 融合得分：

`score = bm25_weight * bm25 + vector_weight * vector`

4. 过滤 `score > 0` 后按分数排序，返回 Evidence

## 3.6 重排器（Reranker）

实现：

- `SimpleReranker`：关键词覆盖 + 原始分，轻量稳定
- `CrossEncoderReranker`：本地交叉编码器重排

策略：

- provider 为 `simple` 时直接使用轻量重排
- provider 为 `local/cross_encoder/...` 时尝试加载 CE
- CE 失败自动回退 Simple，并记录 `reranker_runtime`

## 3.7 查询阶段的主题感知策略

入口：`RagPipeline.retrieve(...)`

核心机制：

- `raw_top_k = max(k*2, 8)`
- 若 `topic == news`，提升到至少 40（防止新闻候选稀疏）
- 先做 ticker/topic scope 过滤，再重排
- 非新闻主题允许 fallback 到更宽松候选
- 新闻主题禁用跨主题 fallback，保证语义纯度
- 新闻主题默认 `dedup_by_source=False`，保留多事件

## 3.8 证据后处理

函数：`_postprocess_evidence(...)`

多层过滤：

1. `_is_usable_evidence`：长度、噪声 token 比例、数字噪声占比
2. `_is_minimal_evidence`：宽松可读性过滤
3. `_is_empty_fundamental_snapshot`：剔除 `N/A` 过多快照

输出控制：

- 默认按 source 去重（新闻可关闭）
- 截断到 `max_items`

## 3.9 关键参数建议

- `TOP_K`：最终返回证据数
- `BM25_WEIGHT` / `VECTOR_WEIGHT`：融合比例
- `RERANK_TOP_N`：重排后保留条数
- `EMBEDDING_PROVIDER` / `RERANKER_PROVIDER`
- `VECTOR_STORE_PROVIDER`（`chroma` 或回退内存）

建议在数据质量不稳阶段优先 `simple + hash`，先保证稳定与可解释性。

## 3.10 可观测性

`RagPipeline` 会记录：

- `index_built`：docs/chunks/change/removed 等统计
- `retrieval`：query、过滤条件、最终 recall 列表
- embedding/reranker 运行时信息

这些日志用于定位“检索到了什么、为何没被选中”。