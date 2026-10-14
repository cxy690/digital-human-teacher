"""A2A 协议最小闭环单测（Agent Card / message/send / tasks/get / tasks/cancel）。"""
from __future__ import annotations

from a2a.card import build_agent_card
from a2a.errors import METHOD_NOT_FOUND, TASK_NOT_CANCELABLE, TASK_NOT_FOUND
from a2a.executor import SupervisorAgentExecutor
from a2a.models import Message, Part, Task
from a2a.rpc import handle_rpc


def _send(text: str, *, task_id: str | None = None, **extra) -> dict:
    """构造一条 message/send 请求。"""
    message: dict = {"kind": "message", "role": "user", "parts": [{"kind": "text", "text": text}]}
    if task_id:
        message["taskId"] = task_id
    message.update(extra)
    return {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": message}}


def test_agent_card_declares_only_implemented_capabilities() -> None:
    """卡片必须如实声明：阻塞式可用、流式未实现。"""
    card = build_agent_card("https://example.com/")
    assert card.url == "https://example.com/a2a"
    assert card.capabilities.streaming is False
    assert card.capabilities.pushNotifications is False
    ids = {s.id for s in card.skills}
    assert {"tutor_qa", "intent_triage", "whiteboard"} <= ids


async def test_message_send_returns_task_with_artifacts() -> None:
    executor = SupervisorAgentExecutor()
    response = await handle_rpc(_send("老师，这道方程题怎么解？"), executor)

    assert response["jsonrpc"] == "2.0" and response["id"] == 1
    task = response["result"]
    assert task["kind"] == "task" and task["contextId"].startswith("ctx-")
    assert task["status"]["state"] in ("completed", "input-required")
    # 讲解话术与白板板书都应作为产物返回
    names = [a["name"] for a in task["artifacts"]]
    assert "讲解话术" in names
    assert "白板板书" in names
    # 边生成边上板会推很多次累计快照：对外只应聚合成 1 份完整板书，不能刷出一堆
    assert names.count("白板板书") == 1
    assert names.count("讲解话术") == 1
    # 学生这条消息留在 history 里
    assert task["history"][0]["role"] == "user"
    assert task["history"][0]["parts"][0]["text"].startswith("老师")

    # text Part 不应带 data 键（exclude_none 生效，符合 A2A 的 Part 形状）
    speech = next(a for a in task["artifacts"] if a["name"] == "讲解话术")
    assert "data" not in speech["parts"][0]
    board = next(a for a in task["artifacts"] if a["name"] == "白板板书")
    assert "text" not in board["parts"][0]


async def test_tasks_get_and_unknown_id() -> None:
    executor = SupervisorAgentExecutor()
    created = (await handle_rpc(_send("老师，这道题怎么解？"), executor))["result"]

    got = await handle_rpc(
        {"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": created["id"]}},
        executor,
    )
    assert got["result"]["id"] == created["id"]

    missing = await handle_rpc(
        {"jsonrpc": "2.0", "id": 3, "method": "tasks/get", "params": {"id": "task-ghost"}},
        executor,
    )
    assert missing["error"]["code"] == TASK_NOT_FOUND


async def test_tasks_cancel_then_not_cancelable() -> None:
    executor = SupervisorAgentExecutor()
    task = Task(contextId="ctx-cancel")
    task.touch("working")
    executor.store.put(task)

    canceled = await handle_rpc(
        {"jsonrpc": "2.0", "id": 4, "method": "tasks/cancel", "params": {"id": task.id}},
        executor,
    )
    assert canceled["result"]["status"]["state"] == "canceled"

    again = await handle_rpc(
        {"jsonrpc": "2.0", "id": 5, "method": "tasks/cancel", "params": {"id": task.id}},
        executor,
    )
    assert again["error"]["code"] == TASK_NOT_CANCELABLE


async def test_unsupported_method_and_bad_params() -> None:
    executor = SupervisorAgentExecutor()

    stream = await handle_rpc(
        {"jsonrpc": "2.0", "id": 6, "method": "message/stream", "params": {}}, executor
    )
    assert stream["error"]["code"] == METHOD_NOT_FOUND

    # 没有文本 Part：参数不合法（当前只支持文本输入）
    empty = await handle_rpc(
        {"jsonrpc": "2.0", "id": 7, "method": "message/send", "params": {"message": {"parts": []}}},
        executor,
    )
    assert empty["error"]["code"] == -32602

    notifications = await handle_rpc(
        {"jsonrpc": "2.0", "method": "tasks/get", "params": {"id": "task-x"}}, executor
    )
    assert notifications is None, "无 id 的通知不应产生响应"


async def test_resume_input_required_task() -> None:
    """老师追问后任务进入 input-required，学生再发一条即续跑同一个任务。"""
    executor = SupervisorAgentExecutor()
    waiting = Task(contextId="ctx-resume")
    waiting.history.append(Message(role="user", parts=[Part.of_text("老师，这道方程题怎么解？")]))
    waiting.touch("input-required", Message(role="agent", parts=[Part.of_text("x 等于几？")]))
    executor.store.put(waiting)

    response = await handle_rpc(_send("x 等于 5", task_id=waiting.id), executor)

    assert response["result"]["id"] == waiting.id, "等待作答的任务应当续跑而不是新建"
    # 新旧两条消息留在同一任务的历史里
    roles = [m["role"] for m in response["result"]["history"]]
    assert len(response["result"]["history"]) == 2
    assert roles == ["user", "user"]


async def test_distinct_contexts_use_distinct_sessions() -> None:
    """不同 contextId 隔离会话（A2A 调用不会串到学生浏览器里的课堂）。"""
    executor = SupervisorAgentExecutor()
    first = (await handle_rpc(_send("第一题怎么做？"), executor))["result"]
    second = (await handle_rpc(_send("第二题怎么做？"), executor))["result"]

    assert first["contextId"] != second["contextId"]
    assert set(executor._sessions) == {first["contextId"], second["contextId"]}  # noqa: SLF001
    # 会话本身用 a2a- 前缀命名，与外面前端 WS 会话隔离开
    session_ids = [state.session_id for state, _ in executor._sessions.values()]  # noqa: SLF001
    assert all(sid.startswith("a2a-") for sid in session_ids)
