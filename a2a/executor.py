"""A2A 任务执行器：把 :class:`Supervisor` 的一次回合包成 A2A Task。

映射思路
--------
本项目的 ``SessionState.emit`` 回调正好是「这一回合 agent 产出了什么」的完整记录
（讲解话术、白板指令、追问、判题…）。执行时临时接管 ``emit`` 收集事件，
回合结束再翻译成 A2A Artifact —— 不需要改动 Supervisor 一行代码：

======================  ==========================================================
事件                     A2A 产物
======================  ==========================================================
``ai_speech``           文本 Part（老师说了什么）
``board_ops``           data Part（白板指令列表，调用方自行渲染）
``ai_ask`` / ``ai_quiz`` / ``ai_check``   data Part（追问 / 模拟题 / 判题结果）
======================  ==========================================================

任务状态
--------
* 老师讲完若留下待作答的追问（``session.pending_question``）-> ``input-required``，
  并把问题放进 ``status.message``；学生再发一条带同一 ``taskId`` 的消息即**续跑同一个任务**
  （走既有判题链路），这正是 A2A 任务生命周期在答疑场景里的自然映射；
* 否则 -> ``completed``，``status.message`` 放本回合话术；
* 执行异常 -> ``failed``。

会话隔离
--------
一个 ``contextId`` 对应一个独立会话，且 session_id 前缀 ``a2a-``，
**与前端 WebSocket 会话完全隔离** —— 外部 agent 调进来不会打断学生浏览器里的课堂。
"""
from __future__ import annotations

import asyncio
from typing import Any

from a2a.errors import A2AError, INVALID_PARAMS, TASK_NOT_CANCELABLE, TASK_NOT_FOUND
from a2a.models import Artifact, Message, Part, Task, new_id
from a2a.store import A2ATaskStore
from config.logging import get_logger
from config.subjects import SUBJECT_BY_NAME, resolve_subject
from core.state.models import StudentTurn
from core.state.session import SessionState
from agents.supervisor import Supervisor

log = get_logger("a2a.executor")

# 事件名 -> 产物名（只登记对外有意义的产出；ai_audio/ai_avatar 等前端专用事件不入产物）
_EVENT_ARTIFACTS: dict[str, str] = {
    "ai_speech": "讲解话术",
    "board_ops": "白板板书",
    "ai_ask": "追问",
    "ai_quiz": "模拟题",
    "ai_check": "判题结果",
}

_TERMINAL_STATES = ("completed", "failed", "canceled")


class SupervisorAgentExecutor:
    """A2A 任务执行器（一个 contextId 一个会话 + 一个 Supervisor）。"""

    def __init__(
        self,
        harness: Any | None = None,
        *,
        store: A2ATaskStore | None = None,
        student_id: str = "a2a-client",
    ) -> None:
        self.store = store or A2ATaskStore()
        self._harness = harness
        self._student_id = student_id
        self._sessions: dict[str, tuple[SessionState, Supervisor]] = {}
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------- 对外方法
    async def send(self, params: dict[str, Any]) -> Task:
        """``message/send``：执行一次回合（或续跑等待作答的任务）。"""
        raw = params.get("message")
        if not isinstance(raw, dict):
            raise A2AError(INVALID_PARAMS, "message 字段缺失或不是对象")
        message = Message.model_validate(raw)
        text = message.text()
        if not text:
            raise A2AError(INVALID_PARAMS, "message.parts 里没有文本（当前只支持文本输入）")

        task = self._resolve_task(message)
        task.history.append(message)
        task.touch("working")

        state, supervisor = await self._session_for(task.contextId)
        self._apply_metadata(state, message.metadata)
        events: list[tuple[str, Any]] = []
        try:
            response = await self._run_turn(state, supervisor, text, events)
        except Exception as err:  # noqa: BLE001 - 单次任务失败要落成 failed 而不是 500
            log.exception("A2A 任务执行失败：{}", err)
            task.touch("failed", Message(role="agent", parts=[Part.of_text(f"执行失败：{err}")]))
            return task

        task.artifacts.extend(self._to_artifacts(events))
        pending = state.pending_question or {}
        question = str(pending.get("question") or "").strip()
        if question:
            # 老师追问 -> 等学生作答（续跑同一个任务）
            task.touch("input-required", Message(role="agent", parts=[Part.of_text(question)]))
        else:
            speech = str(getattr(response, "speech", "") or "").strip()
            task.touch(
                "completed",
                Message(role="agent", parts=[Part.of_text(speech)]) if speech else None,
            )
        return task

    def get(self, task_id: str | None) -> Task:
        """``tasks/get``。"""
        task = self.store.get(task_id)
        if task is None:
            raise A2AError(TASK_NOT_FOUND, f"任务不存在或已过期：{task_id or '(空)'}")
        return task

    def cancel(self, task_id: str | None) -> Task:
        """``tasks/cancel``。"""
        task = self.get(task_id)
        if task.status.state in _TERMINAL_STATES:
            raise A2AError(TASK_NOT_CANCELABLE, f"任务当前状态不可取消：{task.status.state}")
        task.touch("canceled")
        return task

    # ---------------------------------------------------------------- 内部
    def _resolve_task(self, message: Message) -> Task:
        """带 ``taskId`` 且该任务在等作答 -> 续跑；否则开新任务。"""
        if message.taskId:
            existing = self.store.get(message.taskId)
            if existing is not None and existing.status.state == "input-required":
                return existing
            if existing is not None:
                log.info(
                    "A2A 任务 {} 状态为 {}，不续跑，另开新任务",
                    existing.id, existing.status.state,
                )
        task = Task(contextId=message.contextId or new_id("ctx"))
        return self.store.put(task)

    async def _session_for(self, context_id: str) -> tuple[SessionState, Supervisor]:
        entry = self._sessions.get(context_id)
        if entry is not None:
            return entry
        async with self._lock:
            entry = self._sessions.get(context_id)
            if entry is None:
                entry = self._create_session(context_id)
                self._sessions[context_id] = entry
        return entry

    def _create_session(self, context_id: str) -> tuple[SessionState, Supervisor]:
        session_id = f"a2a-{context_id}"
        if self._harness is not None:
            state, supervisor, _ = self._harness.create_session(
                self._student_id, session_id=session_id, with_loop=False
            )
            return state, supervisor
        state = SessionState(session_id=session_id, student_id=self._student_id)
        return state, Supervisor(state)

    async def _run_turn(
        self,
        state: SessionState,
        supervisor: Supervisor,
        text: str,
        sink: list[tuple[str, Any]],
    ) -> Any:
        """接管 ``emit`` 收集本回合产出，再原样归还。"""
        previous = state.emit
        state.emit = lambda event: sink.append((str(event.get("type")), event.get("payload")))
        try:
            return await supervisor.handle_turn(StudentTurn(text=text))
        finally:
            state.emit = previous

    @staticmethod
    def _to_artifacts(events: list[tuple[str, Any]]) -> list[Artifact]:
        """把事件流**聚合成少量产物**，而不是一条事件一个产物。

        必须聚合的两个原因：
        * ``board_ops`` 在「边生成边上板」时会被推很多次（每次都是**累计快照**），
          直接逐条转产物会得到 7 个几乎一样的「白板板书」—— 对外只需要**最后一份**完整板书；
        * ``ai_speech`` 在理解先行模式下分「理解 / 解答」两次播报，对调用方来说
          就是「老师这一回合说了什么」，拼成一段更可用。

        追问 / 模拟题 / 判题这类一次性事件仍各自成产物（它们本就是独立信息）。
        """
        speeches: list[str] = []
        board: list[Any] = []
        extras: list[Artifact] = []
        for event_type, payload in events:
            if event_type not in _EVENT_ARTIFACTS:
                continue
            if event_type == "ai_speech":
                text = str((payload or {}).get("text") or "").strip()
                if text:
                    speeches.append(text)
                continue
            if event_type == "board_ops":
                ops = payload if isinstance(payload, list) else []
                if ops:
                    board = ops          # 累计快照：后面的覆盖前面的
                continue
            if isinstance(payload, dict) and payload:
                extras.append(
                    Artifact(
                        name=_EVENT_ARTIFACTS[event_type],
                        parts=[Part.of_data(payload)],
                        metadata={"event": event_type},
                    )
                )

        artifacts: list[Artifact] = []
        if speeches:
            artifacts.append(
                Artifact(
                    name=_EVENT_ARTIFACTS["ai_speech"],
                    parts=[Part.of_text("\n".join(speeches))],
                    metadata={"event": "ai_speech", "segments": len(speeches)},
                )
            )
        if board:
            artifacts.append(
                Artifact(
                    name=_EVENT_ARTIFACTS["board_ops"],
                    parts=[Part.of_data({"ops": board})],
                    metadata={"event": "board_ops", "op_count": len(board)},
                )
            )
        artifacts.extend(extras)
        return artifacts

    @staticmethod
    def _apply_metadata(state: SessionState, metadata: dict[str, Any]) -> None:
        """允许调用方在 metadata 里指定学科 / 老师（语义与前端 ``course_start`` 一致）。"""
        subject = str(metadata.get("subject") or "").strip()
        if subject:
            state.subject = resolve_subject(subject)
            state.scratch["subject"] = state.subject
            meta = SUBJECT_BY_NAME.get(state.subject)
            state.scratch["subject_id"] = str((meta or {}).get("key") or "")
        teacher = str(metadata.get("teacher") or "").strip()
        if teacher:
            state.teacher = teacher
            state.scratch["teacher"] = teacher
