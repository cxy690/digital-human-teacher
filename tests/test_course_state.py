"""任务2 专项测试：课程数据模型与运行时状态。

覆盖：
* ``core/state/models.py::QuizItem``（字段、类型约束、默认值）；
* ``core/state/session.py::SessionState`` 新增的辅导课字段；
* ``core/course_session.py`` 的话术与阶段解析（``greeting_text`` / ``phase_intro_text`` /
  ``to_phase`` / ``phase_label``）。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.constants import ClassPhase
from core.course_session import greeting_text, phase_intro_text, phase_label, to_phase
from core.state.models import QuizItem
from core.state.session import SessionState


# ---------------------------------------------------------------- QuizItem
def test_quiz_item_defaults() -> None:
    quiz = QuizItem()
    assert quiz.qid == ""
    assert quiz.type == "choice"       # 默认题型为选择
    assert quiz.options == []
    assert quiz.answer == ""
    assert quiz.hint == ""
    assert quiz.topic == ""


def test_quiz_item_payload_roundtrip() -> None:
    quiz = QuizItem(
        qid="quiz-1",
        type="choice",
        question="3x = 12，x 等于多少？",
        options=["x = 4", "x = 3", "x = 6"],
        answer="x = 4",
        hint="两边同除以 3",
        topic="一元一次方程",
    )
    dumped = quiz.model_dump()
    assert set(dumped) == {"qid", "type", "question", "options", "answer", "hint", "topic"}
    assert QuizItem(**dumped) == quiz
    assert dumped["type"] == "choice"


def test_quiz_item_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        QuizItem(type="essay")  # type: ignore[arg-type]


# ---------------------------------------------------------------- SessionState
def test_session_state_course_fields_defaults() -> None:
    state = SessionState(session_id="s1", student_id="stu1")
    assert state.subject == "综合"
    assert state.course_active is False
    assert state.course_phase == ClassPhase.OPENING.value
    assert state.course_started_at is None
    assert state.course_deadline is None
    assert state.quiz is None
    assert state.quiz_count == 0
    assert state.last_topic == ""


def test_session_state_course_fields_are_per_instance() -> None:
    """dataclass 字段默认值不能跨实例共享可变对象。"""
    a = SessionState(session_id="a", student_id="a")
    b = SessionState(session_id="b", student_id="b")
    a.quiz = {"qid": "x"}
    a.quiz_count = 3
    b.scratch["k"] = 1
    assert b.quiz is None
    assert b.quiz_count == 0
    assert "k" not in a.scratch


# ---------------------------------------------------------------- 话术与阶段解析
def test_greeting_text_contains_subject() -> None:
    """开课问候：从**通用问候池**里随机取一条并带上学科名。

    刻意不断言具体措辞与结尾标点 —— 该池就是用来避免「每节课第一句都一模一样」的，
    咬死某一句会把「多样性」这个设计意图测反（见 ``config/teachers.GENERIC_GREETINGS``）。
    """
    for _ in range(8):                       # 随机池：多抽几次覆盖多条模板
        text = greeting_text("物理")
        assert "物理" in text
        assert text.strip()
    assert greeting_text("物理", "同学你好呀，我是你的物理老师。") == "同学你好呀，我是你的物理老师。"


def test_phase_intro_text_for_each_phase() -> None:
    for phase in (
        ClassPhase.OPENING,
        ClassPhase.LECTURE,
        ClassPhase.INTERACTION,
        ClassPhase.SUMMARY,
        ClassPhase.ENDED,
    ):
        assert phase_intro_text(phase, "英语"), f"{phase} 缺少过渡话术"
    # 前两个阶段会带上学科信息
    assert "英语" in phase_intro_text(ClassPhase.LECTURE, "英语")
    # 非法/空阶段返回空串，不抛异常
    assert phase_intro_text(None, "英语") == ""
    assert phase_intro_text("not-a-phase", "英语") == ""


def test_to_phase_and_phase_label() -> None:
    assert to_phase("lecture") is ClassPhase.LECTURE
    assert to_phase(ClassPhase.SUMMARY) is ClassPhase.SUMMARY
    assert to_phase(None) is None
    assert to_phase("") is None
    assert to_phase("bogus") is None          # 非法值安全返回 None

    assert phase_label(ClassPhase.INTERACTION) == "互动出题"
    assert phase_label("interaction") == "互动出题"
    assert phase_label(None) == ""
    assert phase_label("bogus") == "bogus"    # 无法解析时回显原值，便于排查
