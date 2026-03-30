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


@dataclass(slots=True)
class AppConfig:
    """全局配置对象。"""

    app_env: str = os.getenv("APP_ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    report_output_dir: Path = Path(os.getenv("REPORT_OUTPUT_DIR", "reports_output"))
    knowledge_base_dir: Path = Path(os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base"))
    vector_store_provider: str = os.getenv("VECTOR_STORE_PROVIDER", "chroma")
    vector_collection: str = os.getenv("VECTOR_COLLECTION", "finance_docs")
    chroma_persist_dir: Path = Path(os.getenv("CHROMA_PERSIST_DIR", ".chroma"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.4"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.6"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    enable_langgraph: bool = os.getenv("ENABLE_LANGGRAPH", "true").lower() == "true"


def get_config() -> AppConfig:
    """读取配置并确保必要目录存在。"""

    config = AppConfig()
    config.report_output_dir.mkdir(parents=True, exist_ok=True)
    config.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return config
