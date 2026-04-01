from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen LoRA 微调用于新闻情感/风险双任务分类")
    parser.add_argument("--risk-csv", type=Path, required=True, help="风险标签数据集 CSV")
    parser.add_argument("--sentiment-csv", type=Path, required=True, help="情感标签数据集 CSV")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("models/news_qwen_lora"))
    parser.add_argument("--max-samples", type=int, default=80000, help="训练样本上限，0 表示全量")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--warmup-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true", default=True, help="仅使用本地缓存模型")
    parser.add_argument("--resume-from-checkpoint", type=str, default="", help="完整断点恢复目录（要求 torch>=2.6）")
    parser.add_argument("--resume-adapter-path", type=str, default="", help="仅恢复 LoRA 权重目录（不恢复优化器）")
    return parser.parse_args()


@dataclass(slots=True)
class TrainRow:
    text: str
    sentiment_level: int
    risk_level: int


def build_training_rows(risk_csv: Path, sentiment_csv: Path, max_samples: int) -> list[TrainRow]:
    import pandas as pd

    risk_df = pd.read_csv(
        risk_csv,
        usecols=["Date", "Article_title", "Stock_symbol", "Article", "risk_deepseek"],
    )
    sent_df = pd.read_csv(
        sentiment_csv,
        usecols=["Date", "Article_title", "Stock_symbol", "Article", "sentiment_deepseek"],
    )

    keys = ["Date", "Article_title", "Stock_symbol", "Article"]
    merged = risk_df.merge(sent_df, on=keys, how="inner")

    merged = merged.dropna(subset=["Article", "risk_deepseek", "sentiment_deepseek"])
    merged["risk_level"] = merged["risk_deepseek"].map(_to_level)
    merged["sentiment_level"] = merged["sentiment_deepseek"].map(_to_level)
    merged = merged.dropna(subset=["risk_level", "sentiment_level"])

    if max_samples > 0 and len(merged) > max_samples:
        merged = merged.sample(n=max_samples, random_state=42)

    rows: list[TrainRow] = []
    for _, row in merged.iterrows():
        text = str(row["Article"]).strip()
        if len(text) < 80:
            continue
        rows.append(
            TrainRow(
                text=text[:3500],
                sentiment_level=int(row["sentiment_level"]),
                risk_level=int(row["risk_level"]),
            )
        )
    return rows


def _to_level(v: object) -> int | None:
    try:
        n = int(round(float(v)))
        return max(1, min(5, n))
    except Exception:  # noqa: BLE001
        return None


def to_prompt(row: TrainRow) -> str:
    target = {
        "sentiment_level": row.sentiment_level,
        "risk_level": row.risk_level,
        "reasons": [],
    }
    return (
        "<|im_start|>system\n你是金融新闻情感与风险分析器，只返回JSON。<|im_end|>\n"
        "<|im_start|>user\n"
        "请对下面新闻进行双任务分类：\n"
        "1) sentiment_level: 1~5（1极负面，5极正面）\n"
        "2) risk_level: 1~5（1极低风险，5极高风险）\n"
        "输出 JSON：{\"sentiment_level\":x,\"risk_level\":y,\"reasons\":[\"...\"]}\n"
        f"新闻文本：\n{row.text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{json.dumps(target, ensure_ascii=False)}"
        "<|im_end|>"
    )


def main() -> None:
    args = parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("请先安装训练依赖：pip install -r requirements-ml.txt") from exc

    rows = build_training_rows(args.risk_csv, args.sentiment_csv, args.max_samples)
    if not rows:
        raise RuntimeError("训练数据为空，请检查 CSV 路径和字段")

    prompts = [{"text": to_prompt(r)} for r in rows]
    ds = Dataset.from_list(prompts)

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=qconf,
        device_map="auto",
        dtype=compute_dtype,
        local_files_only=args.local_files_only,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 恢复策略：
    # 1) resume_adapter_path：只恢复 LoRA 权重（兼容 torch<2.6）
    # 2) 否则从头初始化 LoRA
    if args.resume_adapter_path:
        resume_path = Path(args.resume_adapter_path)
        if not resume_path.exists():
            raise RuntimeError(f"resume-adapter-path 不存在: {resume_path}")
        model = PeftModel.from_pretrained(model, str(resume_path), is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targs = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=False,
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"训练完成，LoRA adapter 已保存到: {out_dir}")


if __name__ == "__main__":
    main()
