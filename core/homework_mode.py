"""作业模式：把「一道题的答疑」升级为「一份作业的助学流程」。

职责：
* 维护作业题目队列与进度；
* 通过 :class:`HomeworkHintPolicy` 控制提示层级（先思考、再给提示、最后给答案）；
* 判定对错、串联错题记录（错题本）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.hint_homework import HintLevel, HomeworkHintPolicy


@dataclass
class HomeworkQuestion:
    """作业中的一道题。"""

    qid: str
    content: str
    answer: str | None = None
    image_id: str | None = None
    attempts: int = 0
    solved: bool = False
    hint_level: HintLevel | None = None


@dataclass
class HomeworkSession:
    """一次作业助学会话。"""

    homework_id: str
    questions: list[HomeworkQuestion] = field(default_factory=list)
    cursor: int = 0
    hint_policy: HomeworkHintPolicy = field(default_factory=HomeworkHintPolicy)

    @property
    def total(self) -> int:
        return len(self.questions)

    @property
    def solved_count(self) -> int:
        return sum(1 for q in self.questions if q.solved)

    def current(self) -> HomeworkQuestion | None:
        if 0 <= self.cursor < len(self.questions):
            return self.questions[self.cursor]
        return None

    def next(self) -> HomeworkQuestion | None:
        if self.cursor + 1 < len(self.questions):
            self.cursor += 1
        return self.current()

    def request_hint(self, escalate: bool = True) -> tuple[HintLevel | None, str]:
        """请求一次提示，返回 (层级, 对 LLM 的指令)。"""
        q = self.current()
        if q is None:
            return None, ""
        level = self.hint_policy.next_level(q.qid, escalate=escalate)
        q.hint_level = level
        return level, self.hint_policy.instruction(level)

    def submit(self, is_correct: bool) -> dict[str, object]:
        """提交一次作答结果，返回给上层用于播报 / 错题登记的结果。"""
        q = self.current()
        if q is None:
            return {"ok": False, "reason": "no_current_question"}
        q.attempts += 1
        q.solved = is_correct
        result: dict[str, object] = {
            "ok": True,
            "qid": q.qid,
            "correct": is_correct,
            "attempts": q.attempts,
            "progress": f"{self.solved_count}/{self.total}",
        }
        if is_correct:
            self.hint_policy.reset(q.qid)
        return result

    def progress_report(self) -> dict[str, object]:
        return {
            "homework_id": self.homework_id,
            "cursor": self.cursor,
            "total": self.total,
            "solved": self.solved_count,
            "current_qid": (self.current().qid if self.current() else None),
        }
