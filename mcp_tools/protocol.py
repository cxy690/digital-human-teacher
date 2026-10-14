"""MCP（Model Context Protocol）协议核心：JSON-RPC 2.0 + 方法分发。

与 :mod:`a2a.rpc` 同构，只是走 MCP 的方法集。传输无关——stdio 与
Streamable HTTP（见 :mod:`mcp_tools.server`）都复用这里的 :class:`MCPServer`：

* ``initialize``                —— 握手 + 能力协商
* ``notifications/initialized`` —— 客户端握手完成通知（无 id、无响应）
* ``tools/list``                —— 列出工具（``inputSchema``，MCP 命名）
* ``tools/call``                —— 调用工具（结果封装为 ``content`` 数组）
* ``ping``                      —— 连通性探测

工具实现直接复用 :mod:`mcp_tools.registry`，既有工具无需改动；注册表里的
``parameters`` 在这里映射为 MCP 要求的 ``inputSchema``。
"""
from __future__ import annotations

import json
from typing import Any

from config.logging import get_logger
from mcp_tools.registry import ToolRegistry, build_default_registry

log = get_logger("mcp_tools.protocol")

# --- 协议版本（受支持清单；客户端版本不在其中时回落到最新版）---
LATEST_PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

# --- JSON-RPC 2.0 标准错误码（与 a2a/errors.py 保持一致）---
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

_ERROR_MESSAGES: dict[int, str] = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid Request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
}

# 握手前允许调用的方法（其余方法在 initialize 之前调用会被拒绝）
_PRE_INIT_METHODS = {"initialize", "notifications/initialized", "initialized", "ping"}


class MCPError(Exception):
    """带 JSON-RPC 错误码的业务异常（由分发层翻译成 error 响应）。"""

    def __init__(self, code: int, message: str = "") -> None:
        self.code = code
        self.message = message or _ERROR_MESSAGES.get(code, "error")
        super().__init__(f"[{code}] {self.message}")


def error_payload(code: int, message: str = "", rpc_id: Any = None) -> dict[str, Any]:
    """构造 JSON-RPC 错误响应体。"""
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": code, "message": message or _ERROR_MESSAGES.get(code, "error")},
    }


def _empty_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}}


class MCPServer:
    """一个 MCP Server：持有工具注册表，处理 JSON-RPC 报文。

    ``registry`` 缺省时用 :func:`build_default_registry` 装配（工具无外部依赖，
    但 ``ocr_parse`` / ``knowledge_search`` 会因缺 parser/retriever 而返回占位错误）。
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        name: str = "education-tools",
        version: str = "0.1.0",
        instructions: str = "",
    ) -> None:
        self.registry: ToolRegistry = registry if registry is not None else build_default_registry()
        self.name = name
        self.version = version
        self.instructions = instructions
        self._protocol_version = LATEST_PROTOCOL_VERSION
        self._client_info: dict[str, Any] | None = None
        # 握手是否已开始：收到 initialize 即置位（notifications/initialized 只作确认）
        self._handshaken = False

    # ------------------------------------------------------------ 元信息
    def server_info(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}

    def capabilities(self) -> dict[str, Any]:
        """声明的能力：只暴露 tools（无 resources / prompts / logging）。"""
        return {"tools": {"listChanged": False}}

    def tools(self) -> list[dict[str, Any]]:
        """工具清单（MCP 形状：``name`` / ``description`` / ``inputSchema``）。"""
        return [
            {
                "name": str(spec["name"]),
                "description": str(spec.get("description", "")),
                "inputSchema": spec.get("parameters") or _empty_schema(),
            }
            for spec in self.registry.specs()
        ]

    # ------------------------------------------------------------ 报文入口
    async def handle(self, message: Any) -> Any:
        """处理单条或批量 JSON-RPC 报文；通知（无 ``id``）返回 ``None``。"""
        if isinstance(message, list):
            if not message:
                return error_payload(INVALID_REQUEST, "批量请求不能为空")
            responses = [await self._handle_one(item) for item in message]
            return [r for r in responses if r is not None]
        return await self._handle_one(message)

    async def _handle_one(self, request: Any) -> dict[str, Any] | None:
        if not isinstance(request, dict):
            return error_payload(INVALID_REQUEST, "请求必须是 JSON 对象")
        rpc_id = request.get("id")
        response = await self._respond(rpc_id, request)
        if rpc_id is None:
            return None  # 通知：无论成功或失败都不回包
        return response

    async def _respond(self, rpc_id: Any, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("jsonrpc") not in (None, "2.0"):
            return error_payload(INVALID_REQUEST, 'jsonrpc 必须是 "2.0"', rpc_id)
        method = str(request.get("method") or "")
        params = request.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return error_payload(INVALID_PARAMS, "params 必须是对象", rpc_id)
        try:
            result = await self._dispatch(method, params)
        except MCPError as err:
            return error_payload(err.code, err.message, rpc_id)
        except Exception as err:  # noqa: BLE001 - 兜底，避免把异常抛给 client
            log.exception("MCP 请求处理异常：{}", err)
            return error_payload(INTERNAL_ERROR, str(err), rpc_id)
        return {"jsonrpc": "2.0", "id": rpc_id, "result": result}

    async def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return self._initialize(params)
        if method in ("notifications/initialized", "initialized"):
            self._handshaken = True
            return {}
        if method == "ping":
            return {}
        if method == "tools/list":
            self._require_initialized(method)
            return {"tools": self.tools()}
        if method == "tools/call":
            self._require_initialized(method)
            return await self._tools_call(params)
        raise MCPError(METHOD_NOT_FOUND, f"不支持的方法：{method or '(空)'}")

    def _require_initialized(self, method: str) -> None:
        """MCP 要求先握手；未 initialize 就调业务方法按错误处理（ping / initialize 除外）。

        只要求「发过 initialize」：``notifications/initialized`` 属客户端礼貌性确认，
        缺失也放行，以免对「initialize 后直接 tools/list」的简单客户端过于苛刻。
        """
        if not self._handshaken and method not in _PRE_INIT_METHODS:
            raise MCPError(INVALID_REQUEST, "尚未 initialize（请先发送 initialize 握手）")

    # ------------------------------------------------------------ 具体方法
    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = str(params.get("protocolVersion") or "")
        # 客户端版本受支持则沿用，否则回落到服务端最新版（MCP 规范允许）
        self._protocol_version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
        client_info = params.get("clientInfo")
        self._client_info = client_info if isinstance(client_info, dict) else None
        self._handshaken = True
        result: dict[str, Any] = {
            "protocolVersion": self._protocol_version,
            "capabilities": self.capabilities(),
            "serverInfo": self.server_info(),
        }
        if self.instructions:
            result["instructions"] = self.instructions
        log.info("MCP initialize: client={} protocol={}", self._client_info, self._protocol_version)
        return result

    async def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = str(params.get("name") or "")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not name:
            raise MCPError(INVALID_PARAMS, "tools/call 缺少 name")
        if not isinstance(arguments, dict):
            raise MCPError(INVALID_PARAMS, "arguments 必须是对象")
        if name not in {str(t["name"]) for t in self.registry.specs()}:
            raise MCPError(INVALID_PARAMS, f"未知工具：{name}")
        try:
            raw = await self.registry.call(name, **arguments)
        except Exception as err:  # noqa: BLE001 - 工具执行失败按 MCP 约定回 isError，而非协议错误
            log.warning("MCP 工具 {} 执行失败：{}", name, err)
            return {"content": [{"type": "text", "text": f"{type(err).__name__}: {err}"}], "isError": True}
        return _as_tool_result(raw)


def _as_tool_result(raw: Any) -> dict[str, Any]:
    """把 handler 返回值封装成 MCP 的 tool result（``content`` 数组 + ``isError``）。

    约定：handler 返回 ``{"error": ...}`` 视为「工具不可用」→ ``isError=True``。
    """
    is_error = isinstance(raw, dict) and bool(raw.get("error"))
    if isinstance(raw, str):
        text = raw
    elif raw is None:
        text = ""
    else:
        text = json.dumps(raw, ensure_ascii=False, default=str)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}
