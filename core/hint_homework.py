"""作业提示策略：分层渐进式提示（先启发、后给步骤、最后给答案）。

对应 PRD「知识答疑 + 习惯督导」双模式中的习惯养成诉求：鼓励学生先思考，
AI 不直接甩答案，而是逐层给线索。
"""
from __future__ import annotations

from enum import Enum


class HintLevel(str, Enum):
    """提示层级，由轻到重。"""

    CONCEPT = "concept"   # 只点知识点
    METHOD = "method"     # 给方法/思路
    STEP = "step"         # 给解题步骤（不给最终答案）
    ANSWER = "answer"     # 给标准答案 + 讲解


_LEVEL_ORDER: list[HintLevel] = [HintLevel.CONCEPT, HintLevel.METHOD, HintLevel.STEP, HintLevel.ANSWER]

# 每层提示对 LLM 的指令模板
_LEVEL_INSTRUCTION: dict[HintLevel, str] = {
    HintLevel.CONCEPT: "只提示本题涉及的核心知识点，不要给出方法或答案，引导学生回忆概念。",
    HintLevel.METHOD: "给出解题思路和方法方向，不要写具体步骤，也不要给答案。",
    HintLevel.STEP: "给出清晰的解题步骤，但最后一步留白，让学生自己算出结果。",
    HintLevel.ANSWER: "给出完整解题步骤与标准答案，并总结易错点。",
}


class HomeworkHintPolicy:
    """按题目维度记录已给到的提示层级，实现逐层放开。"""

    def __init__(self, max_level: HintLevel = HintLevel.ANSWER) -> None:
        self.max_level = max_level
        self._level_by_question: dict[str, HintLevel] = {}

    def peek(self, question_id: str) -> HintLevel | None:
        return self._level_by_question.get(question_id)

    def next_level(self, question_id: str, escalate: bool = True) -> HintLevel:
        """返回本次应给的提示层级；``escalate=False`` 时保持在当前层。"""
        current = self._level_by_question.get(question_id)
        if current is None:
            level = HintLevel.CONCEPT
        elif not escalate:
            level = current
        else:
            idx = min(_LEVEL_ORDER.index(current) + 1, _LEVEL_ORDER.index(self.max_level))
            level = _LEVEL_ORDER[idx]
        self._level_by_question[question_id] = level
        return level

    def instruction(self, level: HintLevel) -> str:
        return _LEVEL_INSTRUCTION[level]

    def reset(self, question_id: str) -> None:
        self._level_by_question.pop(question_id, None)
