from app.agents.specialists import _llm_rewrite


class _FakeLLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.i = 0

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
        out = self.outputs[min(self.i, len(self.outputs) - 1)]
        self.i += 1
        return out


def test_llm_rewrite_retry_success() -> None:
    json_text = (
        "{\"analysis\":\""
        + ("这是足够长的分析文本" + "，" * 120)
        + "\",\"conclusions\":[\"结论1 [E1]\",\"结论2 [E2]\",\"结论3 [E3]\"]}"
    )
    llm = _FakeLLM([
        "这不是json",
        json_text,
    ])
    a, c = _llm_rewrite(
        llm=llm,
        agent_name="基本面",
        ticker="AAPL",
        topic="fundamental",
        analysis="draft",
        conclusions=["d1 [E1]", "d2 [E2]", "d3 [E3]"],
        refs=[],
        retry_max=2,
    )
    assert len(a) > 80
    assert len(c) >= 3


def test_llm_rewrite_text_fallback() -> None:
    llm = _FakeLLM([
        "分析正文\n- 结论A [E1]\n- 结论B [E2]\n- 结论C [E3]"
    ])
    a, c = _llm_rewrite(
        llm=llm,
        agent_name="新闻",
        ticker="AAPL",
        topic="news",
        analysis="draft",
        conclusions=["d1 [E1]", "d2 [E2]", "d3 [E3]"],
        refs=[],
        retry_max=0,
    )
    assert "文本抽取" in a
    assert len(c) >= 3
