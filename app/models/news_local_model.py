from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.models.sentiment_risk import NewsSentimentRiskResult


@dataclass(slots=True)
class LocalNewsModelConfig:
    base_model: str
    adapter_path: str
    device: str = "auto"
    max_new_tokens: int = 96


class LocalNewsClassifier:
    """本地 LoRA 新闻分类器（Qwen 指令微调）。"""

    def __init__(self, config: LocalNewsModelConfig) -> None:
        self.config = config

    def infer(self, text: str) -> NewsSentimentRiskResult:
        generated = _generate_json(
            base_model=self.config.base_model,
            adapter_path=self.config.adapter_path,
            device=self.config.device,
            max_new_tokens=self.config.max_new_tokens,
            text=text,
        )
        obj = _extract_json(generated)
        if obj is None:
            raise RuntimeError("local news model 输出无法解析为 JSON")

        sent = _clip_int(obj.get("sentiment_level"), lo=1, hi=5, default=3)
        risk = _clip_int(obj.get("risk_level"), lo=1, hi=5, default=3)
        reasons = obj.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []

        return NewsSentimentRiskResult(
            sentiment_level=sent,
            sentiment_label=_sentiment_label(sent),
            risk_level=risk,
            risk_label=_risk_label(risk),
            sentiment_score=float(sent),
            risk_score=float(risk),
            reasons=[str(x).strip() for x in reasons[:6] if str(x).strip()],
        )


@lru_cache(maxsize=1)
def _load_generator(base_model: str, adapter_path: str, device: str):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("缺少本地模型依赖，请安装 requirements-ml.txt") from exc

    use_4bit = device == "auto" or "cuda" in device
    quant = None
    if use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map=device,
        quantization_config=quant,
        torch_dtype=torch.float16 if use_4bit else None,
    )

    adapter_dir = Path(adapter_path)
    if adapter_dir.exists():
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    return tokenizer, model


def _generate_json(base_model: str, adapter_path: str, device: str, max_new_tokens: int, text: str) -> str:
    tokenizer, model = _load_generator(base_model, adapter_path, device)

    prompt = (
        "你是金融新闻情感与风险分析器。\n"
        "任务：根据新闻文本输出 1~5 级情感和 1~5 级风险。\n"
        "输出必须是 JSON，格式："
        '{"sentiment_level":3,"risk_level":3,"reasons":["..."]}。\n'
        "不要输出其他解释。\n"
        f"新闻文本：\n{text[:3000]}"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "你是严格的JSON输出助手。"},
            {"role": "user", "content": prompt},
        ]
        model_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        model_input = prompt

    inputs = tokenizer(model_input, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    out_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(out_tokens, skip_special_tokens=True).strip()


def _extract_json(text: str) -> dict | None:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\\n", "", s)
        s = re.sub(r"\\n```$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _clip_int(v: object, lo: int, hi: int, default: int) -> int:
    try:
        iv = int(round(float(v)))
        return max(lo, min(hi, iv))
    except Exception:  # noqa: BLE001
        return default


def _risk_label(level: int) -> str:
    mapping = {1: "极低风险", 2: "低风险", 3: "中等风险", 4: "高风险", 5: "极高风险"}
    return mapping[level]


def _sentiment_label(level: int) -> str:
    mapping = {1: "极负面", 2: "负面", 3: "中性", 4: "正面", 5: "极正面"}
    return mapping[level]
