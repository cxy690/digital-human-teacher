"""会话与事件数据模型（pydantic），贯穿 core / agents / server 各层。"""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from config.constants import BoardOp, IntentType


def _now_ms() -> int:
    return int(time.time() * 1000)


class WhiteboardOp(BaseModel):
    """白板操作指令，前端据此渲染。"""

    op: BoardOp
    seq: int = 0
    ts_ms: int = Field(default_factory=_now_ms)
    # 文本 / 形状参数
    text: str | None = None
    points: list[list[float]] = Field(default_factory=list)  # 归一化坐标 [[x,y], ...]
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    color: str = "#1677ff"
    size: float = 2.0
    image_id: str | None = None
    # 课件讲解（BoardOp.SLIDE）：切到第几页。其余 op 可带上 deck_id 表明
    # 它属于哪一页的标注，翻页时前端据此丢弃过期标注。
    deck_id: str | None = None
    page_index: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class StudentTurn(BaseModel):
    """学生的一次输入（语音转文字 / 文字 / 图片）。"""

    ts_ms: int = Field(default_factory=_now_ms)
    intent: IntentType = IntentType.UNKNOWN
    text: str = ""
    image_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def image_region(self) -> tuple[float, float, float, float] | None:
        """本题图片的**题目区域**（归一化 x1,y1,x2,y2）；没有或非法时返回 ``None``。

        由 Supervisor 在 OCR 解析后写入 ``meta["image_region"]``
        （算法见 :mod:`document_parsing.region`）。白板据此圈画 —— ``None`` 时下游
        **不画框**：画错比不画更糟，学生盯着错的地方反而听不懂。
        """
        raw = self.meta.get("image_region")
        if not raw or len(raw) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in raw)
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return (max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2))


class ChatMessage(BaseModel):
    """会话消息记录。"""

    ts_ms: int = Field(default_factory=_now_ms)
    role: Literal["student", "ai", "system"]
    text: str = ""
    image_id: str | None = None
    board_ops: list[WhiteboardOp] = Field(default_factory=list)


class AIResponse(BaseModel):
    """AI 教师一次答疑的完整产物。"""

    speech: str = ""            # 语音播报文本
    board_ops: list[WhiteboardOp] = Field(default_factory=list)
    knowledge_points: list[str] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class QuizItem(BaseModel):
    """AI 生成的模拟题（选择 / 填空 / 简答），用于辅导课互动出题。"""

    qid: str = ""
    type: Literal["choice", "blank", "short"] = "choice"
    question: str = ""
    options: list[str] = Field(default_factory=list)  # 选择题选项（其余题型为空）
    answer: str = ""                                   # 标准答案
    hint: str = ""                                     # 答错时的引导提示
    topic: str = ""                                    # 关联知识点（便于板书/复盘）
