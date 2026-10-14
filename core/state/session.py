"""会话运行时状态：一个学生一次连接对应一个 SessionState。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from config.constants import MicState
from core.state.models import (
    ChatMessage,
    WhiteboardOp,
)


@dataclass
class SessionState:
    """单会话可变状态容器（线程/协程内使用，跨模块共享同一实例）。"""

    session_id: str
    student_id: str
    created_at: float = field(default_factory=time.time)

    # 麦控 / 语音
    mic_state: MicState = MicState.IDLE
    ai_speaking: bool = False
    student_speaking: bool = False
    speech_epoch: int = 0                       # 学生插话即 +1，作废上一段未播完的播报

    # 白板
    board_seq: int = 0
    board_ops: list[WhiteboardOp] = field(default_factory=list)
    board_images: dict[str, dict[str, Any]] = field(default_factory=dict)

    # 课件（PPT）讲解：当前打开到第几页（deck_id 为空 = 未打开课件）
    deck_id: str = ""
    page_index: int = 0
    auto_playing: bool = False          # 是否正在「讲完一页自动翻下一页」的连讲中

    # 对话历史
    history: list[ChatMessage] = field(default_factory=list)

    # 当前正在讲解的知识点
    teaching: bool = False

    # 互动追问：待学生作答的问题（None 表示无）；followup_asked 为本题已追问次数
    pending_question: dict[str, Any] | None = None
    followup_asked: int = 0

    # 1v1 学科辅导课（学科选择页进入后开启）
    subject: str = "综合"                       # 当前学科（中文名）
    # 同科多老师：学生选中的老师档案 key（如 math / math_b）。
    # 为空时按学科默认老师解析，保持旧的「一科一师」行为完全不变。
    teacher: str = ""
    course_active: bool = False                 # 是否处于辅导课模式
    course_phase: str = "opening"               # 当前阶段（ClassPhase.value）
    course_started_at: float | None = None      # 课程开始时间（epoch 秒）
    course_deadline: float | None = None        # 课程结束时间（epoch 秒）
    quiz: dict[str, Any] | None = None          # 当前模拟题（答完置空）
    quiz_count: int = 0                         # 本次课已出题数
    last_topic: str = ""                        # 最近讲解的知识点（用于出题）

    # 会话级扩展位（缓存意图、临时题目解析结果等）
    scratch: dict[str, Any] = field(default_factory=dict)

    # 输出回调：向客户端推送（由 server/loop 注入）
    emit: Callable[[dict[str, Any]], None] | None = None

    def next_board_seq(self) -> int:
        """白板指令自增序号，保证两端顺序一致。"""
        self.board_seq += 1
        return self.board_seq

    def push_board(self, ops: list[WhiteboardOp], *, append: bool = False) -> None:
        """记录并向客户端同步白板指令。

        ``append=True``：这批指令**叠加**到当前白板上，不清屏、不重放。
        用于「课件换页后的标注」——页面刚上，不能被自己的标注指令清掉
        （前端收到 board_ops 默认会先清上一轮讲解）。
        """
        for op in ops:
            op.seq = self.next_board_seq()
            self.board_ops.append(op)
        if self.emit and ops:
            self.emit({
                "type": "board_ops",
                "payload": [op.model_dump() for op in ops],
                "append": append,
            })

    def add_message(self, msg: ChatMessage) -> None:
        self.history.append(msg)
        if self.emit:
            self.emit({"type": "chat_message", "payload": msg.model_dump()})

    def push_status(self, **fields: Any) -> None:
        """推送会话状态（麦控、AI 答疑运行状态等 UI 提示）。"""
        if self.emit:
            self.emit({"type": "status", "payload": fields})

    def push_event(self, event_type: str, payload: Any) -> None:
        if self.emit:
            self.emit({"type": event_type, "payload": payload})
