"""WebSocket 会话端点：把客户端与 :class:`EducationHarness` 连接起来。

入站消息 -> SessionLoop.submit；出站由 SessionState.emit 回调推入发送队列，
由独立 sender 任务异步下发，避免回调是同步的而 WS 发送是异步的冲突。
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from config.logging import get_logger
from config.settings import settings
from config.subjects import resolve_subject
from core.auth.deps import user_from_token
from core.harness import EducationHarness
from core.state.store import session_store

log = get_logger("server.websocket")

# 进程级装配根（与 server 同生命周期）
harness = EducationHarness()


def _query_token(websocket: WebSocket) -> str:
    """从 query string 取令牌（浏览器 WS 无法自定义 Header，故走 URL）。"""
    params = websocket.query_params
    return (params.get("token") or params.get("access_token") or "").strip()


async def _reject(websocket: WebSocket, code: int, message: str) -> None:
    """已 accept 后拒绝连接：先下发 error 事件，再按业务码关闭。"""
    try:
        await websocket.send_json(
            {"type": "error", "payload": {"code": code, "message": message}}
        )
        await websocket.close(code=code)
    except RuntimeError:
        pass


async def websocket_endpoint(websocket: WebSocket) -> None:
    """单条 WebSocket 连接 = 一个学生答疑会话。

    身份：先按 ``?token=<JWT>``（或 init 消息里的 ``token``）校验，通过后
    ``student_id`` 一律取自令牌主体（不再信任前端自报）；未登录时按灰度策略：
    ``AUTH_REQUIRED=true`` 直接拒绝，``false`` 则允许匿名并以客户端自报 id 兜底。
    """
    await websocket.accept()
    student_id = "anonymous"
    out_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def emit(payload: dict[str, Any]) -> None:
        # 同步回调：仅入队，由 sender 任务真正下发
        out_queue.put_nowait(payload)

    state = None
    supervisor = None
    loop = None
    sender: asyncio.Task | None = None
    runner: asyncio.Task | None = None
    resumed = False

    try:
        # 等待 init 消息（可携带 student_id / session_id / 学科 / token）
        first = await websocket.receive_json()
        if not isinstance(first, dict):
            first = {}

        # ---- 身份校验（认证）----
        token = _query_token(websocket) or str(first.get("token") or "").strip()
        user = await asyncio.to_thread(user_from_token, token) if token else None
        if settings.auth_enable:
            if token and user is None:
                await _reject(websocket, 4401, "身份凭证无效或已过期")
                return
            if not token and settings.auth_required:
                await _reject(websocket, 4401, "需要登录后才能连接")
                return

        if first.get("type") == "init":
            # 已认证时以令牌主体为准，忽略客户端自报的 student_id
            student_id = str((user or {}).get("id") or first.get("student_id") or student_id)
            # ---- 限流（防滥用）----
            if not harness.anti_rev.allow(student_id):
                await _reject(websocket, 4429, "请求过于频繁，请稍后再试")
                return
            requested_session_id = str(first.get("session_id") or "").strip() or None
            existing = session_store.get(requested_session_id) if requested_session_id else None
            resumed = bool(existing is not None and existing.student_id == student_id)
            state, supervisor, loop = harness.create_session(
                student_id,
                session_id=requested_session_id,
                emit=emit,
            )
            if user is not None:
                state.scratch["auth"] = {
                    "id": user["id"],
                    "role": user["role"],
                    "username": user["username"],
                }
                # 学段注入：出题按学生年级从真题库（ai_teacher.exercise_item）选题，
                # 提示词（teaching_layers / 教师人格）也从 scratch["grade"] 读学段。
                # 用户未填年级时不写，下游按「不限年级」处理。
                _grade = str(user.get("grade") or "").strip()
                if _grade:
                    state.scratch["grade"] = _grade
            raw_subject = str(first.get("subject") or "").strip()
            if raw_subject:
                state.subject = resolve_subject(raw_subject)
            # 同科多老师：学生在选择界面挑中的老师档案 key（为空=学科默认老师）
            raw_teacher = str(first.get("teacher") or "").strip()
            if raw_teacher:
                state.teacher = raw_teacher
                state.scratch["teacher"] = raw_teacher
        else:
            if not harness.anti_rev.allow(student_id):
                await _reject(websocket, 4429, "请求过于频繁，请稍后再试")
                return
            state, supervisor, loop = harness.create_session(student_id, emit=emit)

        await websocket.send_json({
            "type": "ready",
            "payload": {"session_id": state.session_id, "resumed": resumed},
        })

        async def sender_loop() -> None:
            while True:
                payload = await out_queue.get()
                await websocket.send_json(payload)

        sender = asyncio.create_task(sender_loop())
        runner = asyncio.create_task(loop.run())

        # 主接收循环
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "close":
                break
            loop.submit(message)

    except WebSocketDisconnect:
        log.info("client disconnected: {}", student_id)
    except Exception as err:  # noqa: BLE001 - 保证连接异常不拖垮进程
        log.exception("websocket error: {}", err)
    finally:
        if loop is not None:
            loop.stop()
        for task in (runner, sender):
            if task is not None:
                task.cancel()
        # 先等取消真正落地，再做会话收尾：否则 runner 里正在跑的回合（答疑 / 课件连讲 /
        # 数字人渲染）会与 on_session_end 并发改同一份 SessionState，状态可能互相覆盖。
        await asyncio.gather(
            *(task for task in (runner, sender) if task is not None),
            return_exceptions=True,
        )
        # Agent 生命周期收尾（Supervisor 是 BaseAgent 子类）：停掉本会话后台任务
        if supervisor is not None:
            try:
                await supervisor.on_session_end(supervisor.session)
            except Exception as err:  # noqa: BLE001 - 收尾失败不该影响断连
                log.warning("on_session_end failed: {}", err)
        if state is not None:
            harness.checkpoint(state)
        try:
            await websocket.close()
        except RuntimeError:
            pass
