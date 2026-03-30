from app.config import AppConfig
from app.models.llm_client import OpenAICompatibleLLMClient, build_llm_from_config


def test_build_llm_from_config_disabled() -> None:
    cfg = AppConfig(llm_enabled=False)
    assert build_llm_from_config(cfg) is None


def test_build_llm_from_config_enabled() -> None:
    cfg = AppConfig(
        llm_enabled=True,
        llm_api_base="https://example.com/v1",
        llm_api_key="k",
        llm_model="demo-model",
    )
    cli = build_llm_from_config(cfg)
    assert isinstance(cli, OpenAICompatibleLLMClient)
    assert cli.base_url == "https://example.com/v1"
    assert cli.model == "demo-model"
