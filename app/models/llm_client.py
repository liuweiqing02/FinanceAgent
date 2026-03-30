from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import AppConfig


class LLMClient(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(slots=True)
class OpenAICompatibleLLMClient:
    """通用 OpenAI-Compatible 客户端（可接百炼/OpenAI/其他兼容网关）。"""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout_seconds: int = 45
    max_tokens: int = 1400

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8", errors="ignore")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"LLM HTTPError: {detail[:400]}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM URLError: {exc}") from exc

        obj = json.loads(raw)
        choices = obj.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response has no choices: {raw[:400]}")

        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(parts).strip()
        return str(content).strip()


def build_llm_from_config(config: AppConfig) -> LLMClient | None:
    """根据配置构建 LLM 客户端；未启用或缺参数时返回 None。"""

    if not config.llm_enabled:
        return None
    if not config.llm_api_key.strip() or not config.llm_model.strip() or not config.llm_api_base.strip():
        return None

    return OpenAICompatibleLLMClient(
        base_url=config.llm_api_base,
        api_key=config.llm_api_key,
        model=config.llm_model,
        temperature=config.llm_temperature,
        timeout_seconds=config.llm_timeout_seconds,
        max_tokens=config.llm_max_tokens,
    )
