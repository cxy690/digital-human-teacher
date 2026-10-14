"""意图路由：把学生输入归一到 IntentType，决定后续处理链路。

语音转写文本、文字消息、图片、控制指令都会先经过这里。默认用轻量规则，
可在配置中接入 LLM 分类器做兜底（此处通过 ``llm_classifier`` 注入口子）。
"""
from __future__ import annotations

import re
from typing import Awaitable, Callable

from config.constants import IntentType
from core.state.models import StudentTurn

#: 设备/白板控制指令。注意「继续/接着讲/再讲一遍」这类**课件讲解指令不在此表**：
#: 它们由 Supervisor.lecture_command 在 route 节点最优先处理（走 resume_lecture 分支，
#: 能真正触发断点续讲/重讲）；曾放在这里但没有任何执行节点，等于把学生输入吞掉。
_CONTROL_KEYWORDS = ["打开摄像头", "关闭摄像头", "打开麦克风", "关闭麦克风", "清空白板"]
_QUESTION_HINTS = ["为什么", "怎么", "如何", "请问", "帮我", "这题", "这道", "不懂", "不会", "解释", "讲解", "思路", "答案"]
_CHITCHAT_HINTS = ["你好", "在吗", "谢谢", "再见", "你是谁"]

_QUESTION_MARKS = re.compile(r"[?？]")


class IntentRouter:
    """意图路由器。"""

    # 关键词表以只读属性透出：意图识别 Agent（agents/router/agent.py）复用同一份规则，
    # 避免两处各写一份关键词，导致判定口径漂移。
    control_keywords = _CONTROL_KEYWORDS
    question_hints = _QUESTION_HINTS
    chitchat_hints = _CHITCHAT_HINTS

    def __init__(
        self,
        llm_classifier: Callable[[str], Awaitable[IntentType]] | None = None,
    ) -> None:
        self._llm_classifier = llm_classifier

    def route(self, turn: StudentTurn) -> IntentType:
        """同步规则路由（覆盖绝大多数输入）。"""
        text = (turn.text or "").strip()

        if turn.image_id:
            return IntentType.IMAGE_QUESTION
        if any(kw in text for kw in _CONTROL_KEYWORDS):
            return IntentType.CONTROL
        if self._looks_like_question(text):
            return IntentType.VOICE_QUESTION if turn.meta.get("from_voice") else IntentType.TEXT_QUESTION
        if any(kw in text for kw in _CHITCHAT_HINTS):
            return IntentType.CHITCHAT
        if text:
            return IntentType.TEXT_QUESTION
        return IntentType.UNKNOWN

    async def route_async(self, turn: StudentTurn) -> IntentType:
        """规则优先，规则无法判断时调用 LLM 兜底分类。"""
        intent = self.route(turn)
        if intent is IntentType.UNKNOWN and self._llm_classifier and turn.text:
            try:
                return await self._llm_classifier(turn.text)
            except Exception:  # noqa: BLE001 - 分类失败不影响主流程
                return IntentType.TEXT_QUESTION
        return intent

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        if not text:
            return False
        if _QUESTION_MARKS.search(text):
            return True
        return any(hint in text for hint in _QUESTION_HINTS)
