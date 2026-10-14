"""A2A 接入路由：Agent Card 发现 + JSON-RPC 端点。

挂在主应用上（见 ``server/app.py``）：

* ``GET  /.well-known/agent-card.json``  —— A2A 0.3.x 的卡片发现路径
* ``GET  /.well-known/agent.json``       —— A2A 0.2.x 的旧路径（同一份卡片）
* ``POST /a2a``                          —— JSON-RPC 2.0（message/send · tasks/get · tasks/cancel）
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from a2a.card import build_agent_card
from a2a.errors import PARSE_ERROR, error_payload
from a2a.executor import SupervisorAgentExecutor
from a2a.rpc import handle_rpc
from config.logging import get_logger

log = get_logger("a2a.routes")

router = APIRouter(tags=["a2a"])

_executor: SupervisorAgentExecutor | None = None


def get_executor() -> SupervisorAgentExecutor:
    """进程级执行器（懒建）。

    优先复用 ``server.websocket`` 里的装配根（``EducationHarness``），
    这样 A2A 与 WS 共用同一套配置 / 检索器 / 检查点；延迟导入是为了避免
    ``server.app -> a2a.routes -> server.websocket`` 的导入环。
    """
    global _executor
    if _executor is None:
        try:
            from server.websocket import harness  # noqa: PLC0415

            _executor = SupervisorAgentExecutor(harness)
        except Exception as err:  # noqa: BLE001 - 拿不到装配根也要能跑
            log.warning("A2A 未能复用装配根（{}），改用自建会话", err)
            _executor = SupervisorAgentExecutor()
    return _executor


@router.get("/.well-known/agent-card.json", summary="A2A Agent Card（0.3.x）")
async def agent_card(request: Request) -> dict[str, Any]:
    """返回 Agent Card；``url`` 按当前请求的 Host 生成（隧道域名下也正确）。"""
    return build_agent_card(str(request.base_url)).model_dump(exclude_none=True)


@router.get("/.well-known/agent.json", summary="A2A Agent Card（0.2.x 旧路径）")
async def agent_card_legacy(request: Request) -> dict[str, Any]:
    """老客户端仍会来这里找卡片，给同一份内容。"""
    return await agent_card(request)


@router.post("/a2a", summary="A2A JSON-RPC 2.0 端点")
async def a2a_endpoint(request: Request) -> JSONResponse:
    """JSON-RPC 2.0 入口：支持单请求与批量请求。"""
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - 请求体不是 JSON
        return JSONResponse(error_payload(PARSE_ERROR, "请求体不是合法 JSON"), status_code=400)

    response = await handle_rpc(payload, get_executor())
    if response is None:
        return JSONResponse(status_code=204, content=None)   # 通知：无响应体
    return JSONResponse(response)
