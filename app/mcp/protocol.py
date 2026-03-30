from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

try:
    from fastmcp import Client as FastMCPClient
    from fastmcp import FastMCP

    FASTMCP_AVAILABLE = True
except Exception:  # noqa: BLE001
    FastMCP = None  # type: ignore[assignment]
    FastMCPClient = None  # type: ignore[assignment]
    FASTMCP_AVAILABLE = False


@dataclass(slots=True)
class ToolSpec:
    """统一工具元数据描述。"""

    server: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolRequest:
    """统一 MCP 工具请求。"""

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    server: str | None = None


@dataclass(slots=True)
class ToolResponse:
    """统一 MCP 工具响应。"""

    ok: bool
    data: Any
    error: str | None = None
    server: str | None = None
    tool_name: str | None = None


class MCPServerAdapter(Protocol):
    """MCP server 适配器协议。"""

    name: str

    def list_tools(self) -> list[ToolSpec]: ...

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResponse: ...


class LocalMCPServer:
    """本地工具 server，作为 FastMCP 不可用时的兜底实现。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools: dict[str, tuple[Callable[..., Any], ToolSpec]] = {}

    def register_tool(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        spec = ToolSpec(
            server=self.name,
            name=name,
            description=description,
            input_schema=input_schema or {},
        )
        self._tools[name] = (fn, spec)

    def list_tools(self) -> list[ToolSpec]:
        return [spec for _, spec in self._tools.values()]

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResponse:
        row = self._tools.get(tool_name)
        if row is None:
            return ToolResponse(ok=False, data=None, error=f"unknown tool: {tool_name}", server=self.name, tool_name=tool_name)
        fn, _ = row
        try:
            data = fn(**args)
            return ToolResponse(ok=True, data=data, server=self.name, tool_name=tool_name)
        except Exception as exc:  # noqa: BLE001
            return ToolResponse(ok=False, data=None, error=str(exc), server=self.name, tool_name=tool_name)


class FastMCPServerAdapter:
    """基于 FastMCP 的真实 server 适配器（in-memory transport）。"""

    def __init__(self, name: str) -> None:
        if not FASTMCP_AVAILABLE:
            raise RuntimeError("fastmcp is not installed")
        self.name = name
        self._mcp = FastMCP(name)
        self._registered: dict[str, ToolSpec] = {}

    def register_tool(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        # FastMCP 会自动从签名生成 schema；这里保留 input_schema 作为统一层元数据。
        self._mcp.tool(name=name, description=description)(fn)
        self._registered[name] = ToolSpec(
            server=self.name,
            name=name,
            description=description,
            input_schema=input_schema or {},
        )

    def list_tools(self) -> list[ToolSpec]:
        return self._run(self._list_tools_async)

    async def _list_tools_async(self) -> list[ToolSpec]:
        client = FastMCPClient(self._mcp)
        async with client:
            tools = await client.list_tools()
        result: list[ToolSpec] = []
        for t in tools:
            schema = getattr(t, "inputSchema", None) or {}
            desc = getattr(t, "description", "") or self._registered.get(t.name, ToolSpec(self.name, t.name)).description
            result.append(ToolSpec(server=self.name, name=t.name, description=desc, input_schema=schema))
        return result

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResponse:
        return self._run(lambda: self._call_tool_async(tool_name, args))

    async def _call_tool_async(self, tool_name: str, args: dict[str, Any]) -> ToolResponse:
        client = FastMCPClient(self._mcp)
        try:
            async with client:
                result = await client.call_tool(tool_name, args)
                data = getattr(result, "data", None)
                if data is None:
                    data = getattr(result, "structured_content", None)
                if data is None:
                    raw_parts = [getattr(c, "text", str(c)) for c in getattr(result, "content", [])]
                    data = _normalize_fastmcp_content(raw_parts)
                return ToolResponse(ok=True, data=data, server=self.name, tool_name=tool_name)
        except Exception as exc:  # noqa: BLE001
            return ToolResponse(ok=False, data=None, error=str(exc), server=self.name, tool_name=tool_name)

    def _run(self, coro_factory: Callable[[], Any]) -> Any:
        try:
            asyncio.get_running_loop()
            has_loop = True
        except RuntimeError:
            has_loop = False

        if not has_loop:
            return asyncio.run(coro_factory())

        # 极少数嵌套 loop 场景：创建新 loop 同步执行，避免污染现有上下文。
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_factory())
        finally:
            loop.close()


class MultiServerMCPClient:
    """多 server MCP 客户端：统一发现工具与调用入口。"""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerAdapter] = {}

    def register_server(self, server: MCPServerAdapter) -> None:
        self._servers[server.name] = server

    def list_tools(self, server: str | None = None) -> list[ToolSpec]:
        if server is not None:
            srv = self._servers.get(server)
            return [] if srv is None else srv.list_tools()

        tools: list[ToolSpec] = []
        for srv in self._servers.values():
            tools.extend(srv.list_tools())
        return tools

    def call_tool(self, req: ToolRequest) -> ToolResponse:
        if req.server:
            srv = self._servers.get(req.server)
            if srv is None:
                return ToolResponse(ok=False, data=None, error=f"unknown server: {req.server}", server=req.server, tool_name=req.tool_name)
            return srv.call_tool(req.tool_name, req.args)

        hit: list[MCPServerAdapter] = []
        for srv in self._servers.values():
            if any(spec.name == req.tool_name for spec in srv.list_tools()):
                hit.append(srv)

        if not hit:
            return ToolResponse(ok=False, data=None, error=f"unknown tool: {req.tool_name}", tool_name=req.tool_name)
        if len(hit) > 1:
            servers = ", ".join(s.name for s in hit)
            return ToolResponse(
                ok=False,
                data=None,
                error=f"ambiguous tool '{req.tool_name}', specify server. candidates: {servers}",
                tool_name=req.tool_name,
            )
        return hit[0].call_tool(req.tool_name, req.args)


class MCPToolRegistry:
    """兼容旧接口：单 server 包装器。"""

    def __init__(self) -> None:
        self._server = LocalMCPServer("default")
        self._client = MultiServerMCPClient()
        self._client.register_server(self._server)

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._server.register_tool(name=name, fn=fn)

    def execute(self, req: ToolRequest) -> ToolResponse:
        if req.server is None:
            req = ToolRequest(tool_name=req.tool_name, args=req.args, server="default")
        return self._client.call_tool(req)


def _normalize_fastmcp_content(parts: list[str]) -> Any:
    if len(parts) == 1:
        one = parts[0].strip()
        try:
            import json

            return json.loads(one)
        except Exception:  # noqa: BLE001
            pass
        try:
            if one.isdigit() or (one.startswith('-') and one[1:].isdigit()):
                return int(one)
            return float(one)
        except Exception:  # noqa: BLE001
            return one
    return parts
