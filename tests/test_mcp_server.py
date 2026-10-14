"""MCP Server 协议最小闭环单测。

覆盖：初始化握手 / 能力协商 / ``tools/list`` 的 ``inputSchema`` 命名 /
``tools/call`` 的 ``content`` 结果封装与 ``isError`` / 错误分支 /
握手前调用拒绝 / Streamable HTTP 单端点的 JSON 与 SSE 两种响应。
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import math_solver
from mcp_tools.protocol import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    LATEST_PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    MCPServer,
)
from mcp_tools.registry import build_default_registry
from mcp_tools.server import _wants_sse, http_router


def _server() -> MCPServer:
    """用无外部依赖的默认注册表：不加载 RAG 文档，测试快且离线可跑。"""
    return MCPServer(build_default_registry())


async def _handshake(server: MCPServer, rpc_id: int = 1) -> dict:
    """走一遍 MCP 握手：initialize → notifications/initialized。"""
    init = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        }
    )
    assert await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    return init


async def test_initialize_negotiates_protocol_and_capabilities() -> None:
    server = _server()
    init = await _handshake(server)

    result = init["result"]
    assert result["protocolVersion"] == LATEST_PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "education-tools"
    assert "tools" in result["capabilities"]


async def test_unknown_client_version_falls_back_to_latest() -> None:
    server = _server()
    init = await server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}}
    )
    assert init["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION


async def test_tools_list_uses_mcp_input_schema() -> None:
    """MCP 的字段名是 ``inputSchema``（不是注册表里的 ``parameters``）。"""
    server = _server()
    await _handshake(server)

    listed = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = {t["name"]: t for t in listed["result"]["tools"]}

    assert {"math_solve", "math_verify", "ocr_parse", "knowledge_search"} <= set(tools)
    assert all("inputSchema" in t for t in tools.values())
    assert all("parameters" not in t for t in tools.values())
    assert tools["math_solve"]["inputSchema"]["required"] == ["equation"]


async def test_tools_call_returns_content_array() -> None:
    server = _server()
    await _handshake(server)

    called = await server.handle(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "whiteboard_clear"}}
    )
    result = called["result"]
    assert result["isError"] is False
    assert result["content"] and result["content"][0]["type"] == "text"
    assert "clear" in result["content"][0]["text"]


async def test_tools_call_math_solve_when_sympy_available() -> None:
    if not math_solver.available():
        return  # 无 sympy 时该工具会空结果，跳过（见 core/math_solver.py 降级设计）

    server = _server()
    await _handshake(server)

    called = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "math_solve", "arguments": {"equation": "3x+5=20"}},
        }
    )
    result = called["result"]
    assert result["isError"] is False
    assert "5" in result["content"][0]["text"]


async def test_tools_call_marks_unavailable_tool_as_error() -> None:
    """handler 返回 ``{"error": ...}``（如未注入白板）应体现为 ``isError=True``。"""
    server = _server()
    await _handshake(server)

    called = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "whiteboard_write", "arguments": {"text": "x", "x": 0.1, "y": 0.1}},
        }
    )
    result = called["result"]
    assert result["isError"] is True
    assert "board_unavailable" in result["content"][0]["text"]


async def test_unknown_tool_and_method_errors() -> None:
    server = _server()
    await _handshake(server)

    unknown_tool = await server.handle(
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "no_such_tool"}}
    )
    assert unknown_tool["error"]["code"] == INVALID_PARAMS

    unknown_method = await server.handle({"jsonrpc": "2.0", "id": 7, "method": "resources/list"})
    assert unknown_method["error"]["code"] == METHOD_NOT_FOUND


async def test_requires_initialize_before_tools_call() -> None:
    server = _server()
    too_early = await server.handle({"jsonrpc": "2.0", "id": 8, "method": "tools/list"})
    assert too_early["error"]["code"] == INVALID_REQUEST


async def test_tools_list_works_after_initialize_without_notification() -> None:
    """只发 initialize（省掉 initialized 通知）也应放行，避免对简单客户端过于苛刻。"""
    server = _server()
    await server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": LATEST_PROTOCOL_VERSION}}
    )
    listed = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert "result" in listed and listed["result"]["tools"]


async def test_notification_without_id_has_no_response() -> None:
    server = _server()
    assert await server.handle({"jsonrpc": "2.0", "method": "ping"}) is None
    assert await server.handle({"jsonrpc": "2.0", "method": "tools/list"}) is None


def test_wants_sse_only_when_client_rejects_json() -> None:
    assert _wants_sse("text/event-stream") is True
    assert _wants_sse("application/json, text/event-stream") is False
    assert _wants_sse("application/json") is False
    assert _wants_sse("") is False


def test_http_endpoint_json_and_sse() -> None:
    server = _server()
    app = FastAPI()
    app.include_router(http_router(server))
    client = TestClient(app)

    # GET 信息（人读）
    info = client.get("/mcp")
    assert info.status_code == 200
    assert info.json()["transport"] == "streamable-http"

    # 通知 → 202 且无正文
    notif = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert notif.status_code == 202

    # initialize → application/json
    res = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": LATEST_PROTOCOL_VERSION},
        },
    )
    assert res.status_code == 200
    assert res.json()["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION

    # 只收 SSE 的客户端 → text/event-stream
    sse = client.post(
        "/mcp",
        headers={"accept": "text/event-stream"},
        json={"jsonrpc": "2.0", "id": 9, "method": "ping"},
    )
    assert sse.status_code == 200
    assert sse.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"jsonrpc"' in sse.text

    # 非法 JSON 请求体 → Parse error
    bad = client.post("/mcp", content=b"{not json", headers={"content-type": "application/json"})
    assert bad.status_code == 200
    assert bad.json()["error"]["code"] == -32700
