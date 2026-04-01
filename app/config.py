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

    # RAG embedding/reranker 后端配置（默认离线安全）
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "hash")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "auto")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    embedding_normalize: bool = _env_bool("EMBEDDING_NORMALIZE", "true")
    reranker_provider: str = os.getenv("RERANKER_PROVIDER", "simple")
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    reranker_device: str = os.getenv("RERANKER_DEVICE", "auto")
    reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))

    # 通用 ReAct + MCP 能力（统一工具路由与步数预算）
    react_mcp_enabled: bool = _env_bool("REACT_MCP_ENABLED", "false")
    react_mcp_max_steps: int = int(os.getenv("REACT_MCP_MAX_STEPS", "4"))

    # NewsAgent ReAct + MCP 开关（默认关闭，稳定优先）
    news_react_mcp_enabled: bool = _env_bool("NEWS_REACT_MCP_ENABLED", "false")
    news_react_max_steps: int = int(os.getenv("NEWS_REACT_MAX_STEPS", "4"))

    # LLM（OpenAI-Compatible）配置：支持百炼兼容接口与 OpenAI。
    llm_enabled: bool = _env_bool("LLM_ENABLED", "false")
    llm_api_base: str = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1400"))
    llm_json_retry_max: int = int(os.getenv("LLM_JSON_RETRY_MAX", "2"))
    llm_trace_enabled: bool = _env_bool("LLM_TRACE_ENABLED", "true")
    llm_trace_file: Path = Path(os.getenv("LLM_TRACE_FILE", "logs/llm_trace.jsonl"))
    llm_trace_max_chars: int = int(os.getenv("LLM_TRACE_MAX_CHARS", "6000"))

    # 本地新闻情感/风险模型（Qwen + LoRA）配置
    news_model_enabled: bool = _env_bool("NEWS_MODEL_ENABLED", "false")
    news_model_base: str = os.getenv("NEWS_MODEL_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
    news_model_adapter: str = os.getenv("NEWS_MODEL_ADAPTER", "models/news_qwen_lora")
    news_model_device: str = os.getenv("NEWS_MODEL_DEVICE", "auto")
    news_model_max_new_tokens: int = int(os.getenv("NEWS_MODEL_MAX_NEW_TOKENS", "96"))

    # 数据完整度门控
    fundamental_min_metrics: int = int(os.getenv("FUNDAMENTAL_MIN_METRICS", "3"))
    valuation_min_metrics: int = int(os.getenv("VALUATION_MIN_METRICS", "2"))
    news_min_events: int = int(os.getenv("NEWS_MIN_EVENTS", "2"))


def get_config() -> AppConfig:
    """读取配置并确保必要目录存在。"""

    config = AppConfig()
    config.report_output_dir.mkdir(parents=True, exist_ok=True)
    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    config.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return config
