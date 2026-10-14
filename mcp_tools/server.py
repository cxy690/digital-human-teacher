"""MCP Server 入口：stdio 与 Streamable HTTP 双模式。

把 :mod:`mcp_tools.registry` 里的工具用**真 MCP 报文**暴露出去（协议实现见
:mod:`mcp_tools.protocol`）。三种用法：

1) **stdio**（本地子进程，给 Claude Desktop / IDE 等 MCP Host 用）::

       python -m mcp_tools.server

   Host 配置形如::

       {"mcpServers": {"education": {
           "command": "python", "args": ["-m", "mcp_tools.server"],
           "cwd": "<项目根目录>"}}}

2) **Streamable HTTP**（远程共享，单端点 POST）::

       POST http://127.0.0.1:8000/mcp     # 已由 server/app.py 自动挂载
       # 或独立起服务：
       python -m mcp_tools.server --http --port 8020

   客户端需先 ``initialize`` → 发 ``notifications/initialized``，再 ``tools/list`` / ``tools/call``。

3) **自检**（不发网络请求，跑一遍握手 + 列工具 + 调工具）::

       python -m mcp_tools.server --selftest
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# 项目根目录加入模块搜索路径并固定工作目录（与 server/app.py 同款引导）：
# 保证 `python -m mcp_tools.server` / 直接运行脚本 / IDE 运行三种方式都能读到 .env 与 data/。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from fastapi import APIRouter, Request  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

from config.logging import get_logger, setup_logging  # noqa: E402
from mcp_tools.protocol import (  # noqa: E402
    LATEST_PROTOCOL_VERSION,
    PARSE_ERROR,
    MCPServer,
    error_payload,
)
from mcp_tools.registry import ToolRegistry, build_default_registry  # noqa: E402

log = get_logger("mcp_tools.server")

_SSE_MEDIA_TYPE = "text/event-stream"

_INSTRUCTIONS = (
    "education 教学系统工具集：OCR 识别(ocr_parse) / 知识检索(knowledge_search) / "
    "白板写入(whiteboard_write·whiteboard_clear) / 数学求解与校验(math_solve·math_verify)。"
)


def build_registry() -> ToolRegistry:
    """装配带真实依赖的工具注册表（依赖缺失自动降级，不阻断启动）。

    * ``ocr_parse``        ← ``document_parsing.pipeline.QuestionParser``
    * ``knowledge_search`` ← ``rag.retriever.build_retriever``
    * 白板工具是**会话态**的，MCP 侧无会话，故不注入 board（调用返回 ``board_unavailable``）
    """
    parser: Any | None = None
    retriever: Any | None = None
    try:
        from document_parsing.pipeline import QuestionParser

        parser = QuestionParser()
    except Exception as err:  # noqa: BLE001 - 可选组件
        log.warning("MCP：QuestionParser 不可用（ocr_parse 将降级）：{}", err)
    try:
        from rag.retriever import build_retriever

        retriever = build_retriever()
    except Exception as err:  # noqa: BLE001 - 可选组件
        log.warning("MCP：检索器不可用（knowledge_search 将返回空）：{}", err)
    return build_default_registry(parser=parser, retriever=retriever)


_server: MCPServer | None = None


def get_server() -> MCPServer:
    """进程级 MCP Server（懒建单例）。"""
    global _server
    if _server is None:
        _server = MCPServer(build_registry(), instructions=_INSTRUCTIONS)
    return _server


# ------------------------------------------------------------------ stdio 传输
async def serve_stdio(server: MCPServer | None = None) -> None:
    """stdio 传输：stdin 逐行读 JSON-RPC，stdout 逐行回；日志只走 stderr。"""
    srv = server or get_server()
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:  # EOF：宿主关闭了管道
            break
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as err:
            _write_line(error_payload(PARSE_ERROR, f"JSON 解析失败：{err}"))
            continue
        try:
            response = await srv.handle(message)
        except Exception as err:  # noqa: BLE001 - 单条报文异常不应终止 stdio 循环
            log.exception("MCP stdio 处理异常：{}", err)
            continue
        if response is not None:  # 通知不回包
            _write_line(response)


def _write_line(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# ------------------------------------------------------- Streamable HTTP 传输
def http_router(server: MCPServer | None = None) -> APIRouter:
    """Streamable HTTP 传输：单端点 ``POST /mcp``（响应 JSON 或 SSE）。"""
    router = APIRouter(tags=["mcp"])

    @router.get("/mcp", summary="MCP Server 信息")
    async def mcp_info() -> dict[str, Any]:
        srv = server or get_server()
        return {
            "transport": "streamable-http",
            "endpoint": "/mcp",
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "serverInfo": srv.server_info(),
            "capabilities": srv.capabilities(),
        }

    @router.post("/mcp", summary="MCP JSON-RPC 端点")
    async def mcp_rpc(request: Request) -> Response:
        srv = server or get_server()
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 - 请求体不是 JSON
            return _json_response(error_payload(PARSE_ERROR, "请求体不是合法 JSON"))
        response = await srv.handle(payload)
        if response is None:  # 通知：HTTP 层回 202 Accepted（无正文）
            return Response(status_code=202)
        if _wants_sse(request.headers.get("accept", "")):
            body = f"event: message\ndata: {json.dumps(response, ensure_ascii=False)}\n\n"
            return Response(content=body, media_type=_SSE_MEDIA_TYPE)
        return _json_response(response)

    return router


def _json_response(payload: Any) -> JSONResponse:
    return JSONResponse(payload)


def _wants_sse(accept: str) -> bool:
    """客户端明确只收 SSE 时才走 SSE；能收 JSON 就回 JSON（更省事）。"""
    accept = (accept or "").lower()
    return _SSE_MEDIA_TYPE in accept and "application/json" not in accept


# ------------------------------------------------------------------ 自检
async def _selftest() -> int:
    """不发网络请求，直接跑一遍「握手 → 列工具 → 调工具 → 错误分支」。"""
    from mcp_tools.protocol import INVALID_PARAMS, METHOD_NOT_FOUND

    server = MCPServer(build_registry())

    init = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "selftest", "version": "0"},
            },
        }
    )
    assert init["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION, init

    # 握手完成通知（无 id）
    assert await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    listed = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = listed["result"]["tools"]
    assert tools and all("inputSchema" in t and "name" in t for t in tools), listed

    called = await server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "math_solve", "arguments": {"equation": "3x+5=20"}},
        }
    )
    result = called["result"]
    assert result["isError"] is False and "5" in result["content"][0]["text"], called

    unknown_tool = await server.handle(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "no_such_tool"}}
    )
    assert unknown_tool["error"]["code"] == INVALID_PARAMS, unknown_tool

    unknown_method = await server.handle({"jsonrpc": "2.0", "id": 5, "method": "foo/bar"})
    assert unknown_method["error"]["code"] == METHOD_NOT_FOUND, unknown_method

    print(f"OK  MCP selftest 通过：{len(tools)} 个工具 -> {[t['name'] for t in tools]}")
    return 0


def main() -> None:
    """命令行入口：``education-mcp`` / ``python -m mcp_tools.server``。"""
    setup_logging()
    ap = argparse.ArgumentParser(
        prog="python -m mcp_tools.server",
        description="education MCP Server（stdio / Streamable HTTP）",
    )
    ap.add_argument("--http", action="store_true", help="以 Streamable HTTP 模式独立启动（默认 stdio）")
    ap.add_argument("--host", default="127.0.0.1", help="--http 时监听地址（默认 127.0.0.1）")
    ap.add_argument("--port", type=int, default=8020, help="--http 时监听端口（默认 8020）")
    ap.add_argument("--selftest", action="store_true", help="本地自检后退出")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(asyncio.run(_selftest()))

    if args.http:
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI(title="education MCP Server", version="0.1.0")
        app.include_router(http_router())
        print(f"MCP Streamable HTTP 端点： http://{args.host}:{args.port}/mcp", flush=True)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    asyncio.run(serve_stdio())


if __name__ == "__main__":
    main()
