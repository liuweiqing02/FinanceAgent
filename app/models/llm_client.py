from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
    trace_enabled: bool = True
    trace_file: Path = Path("logs/llm_trace.jsonl")
    trace_max_chars: int = 6000

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
            self._trace(
                status="http_error",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=detail,
                final_content="",
                error=f"HTTPError: {detail[:400]}",
            )
            raise RuntimeError(f"LLM HTTPError: {detail[:400]}") from exc
        except URLError as exc:
            self._trace(
                status="url_error",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response="",
                final_content="",
                error=f"URLError: {exc}",
            )
            raise RuntimeError(f"LLM URLError: {exc}") from exc

        try:
            obj = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            self._trace(
                status="decode_error",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=raw,
                final_content="",
                error=f"JSON decode error: {exc}",
            )
            raise RuntimeError(f"LLM response decode error: {exc}") from exc

        choices = obj.get("choices", [])
        if not choices:
            self._trace(
                status="no_choices",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_response=raw,
                final_content="",
                error="LLM response has no choices",
            )
            raise RuntimeError(f"LLM response has no choices: {raw[:400]}")

        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        final = ""
        if isinstance(content, str):
            final = content.strip()
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            final = "\n".join(parts).strip()
        else:
            final = str(content).strip()

        self._trace(
            status="ok",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=raw,
            final_content=final,
            error="",
        )
        return final

    def _trace(
        self,
        *,
        status: str,
        system_prompt: str,
        user_prompt: str,
        raw_response: str,
        final_content: str,
        error: str,
    ) -> None:
        if not self.trace_enabled:
            return
        try:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "model": self.model,
                "base_url": self.base_url,
                "system_prompt": _clip(system_prompt, self.trace_max_chars),
                "user_prompt": _clip(user_prompt, self.trace_max_chars),
                "raw_response": _clip(raw_response, self.trace_max_chars),
                "final_content": _clip(final_content, self.trace_max_chars),
                "error": _clip(error, 1200),
            }
            with self.trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            return


def _clip(text: str, max_chars: int) -> str:
    t = text or ""
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "...[truncated]"


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
        trace_enabled=config.llm_trace_enabled,
        trace_file=config.llm_trace_file,
        trace_max_chars=config.llm_trace_max_chars,
    )
