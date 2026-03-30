from app.mcp.protocol import (
    FASTMCP_AVAILABLE,
    FastMCPServerAdapter,
    LocalMCPServer,
    MultiServerMCPClient,
    ToolRequest,
)


def test_multiserver_list_and_call() -> None:
    c = MultiServerMCPClient()
    s1 = LocalMCPServer("finance")
    s2 = LocalMCPServer("kb")

    s1.register_tool("echo", lambda msg: {"msg": msg}, description="echo msg")
    s2.register_tool("strlen", lambda text: {"n": len(text)}, description="len text")

    c.register_server(s1)
    c.register_server(s2)

    tools = c.list_tools()
    names = {f"{t.server}.{t.name}" for t in tools}
    assert "finance.echo" in names
    assert "kb.strlen" in names

    r = c.call_tool(ToolRequest(server="finance", tool_name="echo", args={"msg": "ok"}))
    assert r.ok is True
    assert r.server == "finance"
    assert r.tool_name == "echo"
    assert r.data["msg"] == "ok"


def test_multiserver_ambiguous_tool_requires_server() -> None:
    c = MultiServerMCPClient()
    a = LocalMCPServer("a")
    b = LocalMCPServer("b")
    a.register_tool("same", lambda: 1)
    b.register_tool("same", lambda: 2)
    c.register_server(a)
    c.register_server(b)

    r = c.call_tool(ToolRequest(tool_name="same"))
    assert r.ok is False
    assert "ambiguous" in (r.error or "")


def test_multiserver_unknown_server() -> None:
    c = MultiServerMCPClient()
    r = c.call_tool(ToolRequest(server="missing", tool_name="x"))
    assert r.ok is False
    assert "unknown server" in (r.error or "")


def test_fastmcp_adapter_if_available() -> None:
    if not FASTMCP_AVAILABLE:
        return

    server = FastMCPServerAdapter("finance_fast")
    server.register_tool("add", lambda a, b: a + b, description="add numbers")

    tools = server.list_tools()
    assert any(t.name == "add" for t in tools)

    r = server.call_tool("add", {"a": 1, "b": 2})
    assert r.ok is True
    assert r.data == 3
