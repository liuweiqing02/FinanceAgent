from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class ToolRequest:
    """MCP 工具请求。"""

    tool_name: str
    args: dict[str, Any]


@dataclass(slots=True)
class ToolResponse:
    """MCP 工具响应。"""

    ok: bool
    data: Any
    error: str | None = None


class MCPToolRegistry:
    """简化版 MCP 工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    def execute(self, req: ToolRequest) -> ToolResponse:
        fn = self._tools.get(req.tool_name)
        if fn is None:
            return ToolResponse(ok=False, data=None, error=f"unknown tool: {req.tool_name}")
        try:
            return ToolResponse(ok=True, data=fn(**req.args))
        except Exception as exc:  # noqa: BLE001
            return ToolResponse(ok=False, data=None, error=str(exc))
