# 05 新闻情感/风险模型（本地 LoRA）

## 方案定位

本地新闻模型用于结构化情感/风险判断，目标是降低在线推理成本并提升时延稳定性。

## 推荐基座

- `Qwen/Qwen2.5-1.5B-Instruct`
- QLoRA 4bit 训练（适配中等显存环境）

## 安装

```bash
pip install -r requirements-ml.txt
```

## 训练示例

```bash
python -m app.models.train_news_lora \
  --risk-csv "C:/path/to/risk.csv" \
  --sentiment-csv "C:/path/to/sentiment.csv" \
  --base-model "Qwen/Qwen2.5-1.5B-Instruct" \
  --output-dir "models/news_qwen_lora" \
  --max-samples 80000 \
  --num-train-epochs 1 \
  --batch-size 1 \
  --grad-accum 16
```

## 启用参数

```bash
NEWS_MODEL_ENABLED=true
NEWS_MODEL_BASE=Qwen/Qwen2.5-1.5B-Instruct
NEWS_MODEL_ADAPTER=models/news_qwen_lora
NEWS_MODEL_DEVICE=auto
NEWS_MODEL_MAX_NEW_TOKENS=96
```

## 运行时策略

- 优先使用本地 LoRA 模型。
- 本地模型异常时自动回退规则判断。
- 运行日志记录模型是否生效。
