from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import AgentOutput, Evidence


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(self, ticker: str, evidence: list[Evidence]) -> AgentOutput:
        raise NotImplementedError


def _join_evidence(evidence: list[Evidence], top_n: int = 2) -> str:
    return "\n".join(f"- [{i+1}] {e.title}: {e.content[:80]}" for i, e in enumerate(evidence[:top_n]))
