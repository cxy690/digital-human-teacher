"""A2A 的 JSON-RPC 2.0 分发层。

已实现的方法（与 Agent Card 声明保持一致）：

* ``message/send``  —— 阻塞式执行一次回合，返回 Task（含 artifacts / status）
* ``tasks/get``     —— 查询任务（外部 client 取产物）
* ``tasks/cancel``  —— 取消未结束的任务

**未实现**：``message/stream``（SSE 流式）与 push notification —— 卡片里
``capabilities.streaming=False`` 已声明，客户端据此选择阻塞式调用。
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from a2a.errors import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    A2AError,
    error_payload,
)
from a2a.executor import SupervisorAgentExecutor
from config.logging import get_logger

log = get_logger("a2a.rpc")

SUPPORTED_METHODS = ("message/send", "tasks/get", "tasks/cancel")


async def handle_rpc(payload: Any, executor: SupervisorAgentExecutor) -> Any:
    """处理一条（或一批）JSON-RPC 请求；通知（无 id）返回 ``None``。"""
    if isinstance(payload, list):
        if not payload:
            return error_payload(INVALID_REQUEST, "批量请求不能为空")
        responses = [await _handle_one(item, executor) for item in payload]
        return [r for r in responses if r is not None]
    return await _handle_one(payload, executor)


async def _handle_one(request: Any, executor: SupervisorAgentExecutor) -> dict[str, Any] | None:
    """处理单条请求；**无 id 的通知不产生任何响应（成功的和报错的都不返回）**。"""
    if not isinstance(request, dict):
        return error_payload(INVALID_REQUEST, "请求必须是 JSON 对象")

    rpc_id = request.get("id")
    response = await _response_for(rpc_id, request, executor)
    if rpc_id is None:
        return None
    return response


async def _response_for(
    rpc_id: Any, request: dict[str, Any], executor: SupervisorAgentExecutor
) -> dict[str, Any]:
    version = request.get("jsonrpc")
    if version not in (None, "2.0"):
        return error_payload(INVALID_REQUEST, 'jsonrpc 必须是 "2.0"', rpc_id)

    method = str(request.get("method") or "")
    params = request.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return error_payload(INVALID_PARAMS, "params 必须是对象", rpc_id)

    try:
        result = await _dispatch(method, params, executor)
    except A2AError as err:
        return error_payload(err.code, err.message, rpc_id)
    except ValidationError as err:
        return error_payload(INVALID_PARAMS, f"参数不合法：{err.errors()[:2]}", rpc_id)
    except Exception as err:  # noqa: BLE001 - 兜底，避免把 500 抛给 client
        log.exception("A2A 请求处理异常：{}", err)
        return error_payload(INTERNAL_ERROR, str(err), rpc_id)
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


async def _dispatch(
    method: str, params: dict[str, Any], executor: SupervisorAgentExecutor
) -> dict[str, Any]:
    if method == "message/send":
        task = await executor.send(params)
        return task.model_dump(exclude_none=True)
    if method == "tasks/get":
        return executor.get(params.get("id")).model_dump(exclude_none=True)
    if method == "tasks/cancel":
        return executor.cancel(params.get("id")).model_dump(exclude_none=True)
    raise A2AError(METHOD_NOT_FOUND, f"不支持的方法：{method or '(空)'}")
