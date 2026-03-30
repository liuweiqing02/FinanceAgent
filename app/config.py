from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    def load_dotenv() -> None:  # type: ignore[no-redef]
        return None


load_dotenv()


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppConfig:
    """全局配置对象。"""

    app_env: str = os.getenv("APP_ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    report_output_dir: Path = Path(os.getenv("REPORT_OUTPUT_DIR", "reports_output"))
    raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
    knowledge_base_dir: Path = Path(os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base"))
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "FinanceAgent/1.0 contact@example.com")
    vector_store_provider: str = os.getenv("VECTOR_STORE_PROVIDER", "chroma")
    vector_collection: str = os.getenv("VECTOR_COLLECTION", "finance_docs")
    chroma_persist_dir: Path = Path(os.getenv("CHROMA_PERSIST_DIR", ".chroma"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.4"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.6"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    enable_langgraph: bool = _env_bool("ENABLE_LANGGRAPH", "true")

    # LLM（OpenAI-Compatible）配置：支持百炼兼容接口与 OpenAI。
    llm_enabled: bool = _env_bool("LLM_ENABLED", "false")
    llm_api_base: str = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1400"))


def get_config() -> AppConfig:
    """读取配置并确保必要目录存在。"""

    config = AppConfig()
    config.report_output_dir.mkdir(parents=True, exist_ok=True)
    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    config.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return config
