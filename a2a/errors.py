"""A2A / JSON-RPC 错误码。

标准段（-327xx / -326xx）来自 JSON-RPC 2.0；``-320xx`` 段是 A2A 规范里
任务相关错误的保留区间。
"""
from __future__ import annotations

# --- JSON-RPC 2.0 标准错误 ---
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# --- A2A 任务错误 ---
TASK_NOT_FOUND = -32001        # TaskNotFoundError
TASK_NOT_CANCELABLE = -32002   # TaskNotCancelableError

_MESSAGES: dict[int, str] = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid Request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
    TASK_NOT_FOUND: "Task not found",
    TASK_NOT_CANCELABLE: "Task not cancelable",
}


class A2AError(Exception):
    """带 JSON-RPC 错误码的业务异常（由 RPC 层翻译成 error 响应）。"""

    def __init__(self, code: int, message: str = "") -> None:
        self.code = code
        self.message = message or _MESSAGES.get(code, "error")
        super().__init__(f"[{code}] {self.message}")


def error_payload(code: int, message: str, rpc_id: object = None) -> dict[str, object]:
    """构造 JSON-RPC 错误响应体。"""
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": code, "message": message or _MESSAGES.get(code, "error")},
    }
