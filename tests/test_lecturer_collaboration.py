"""讲课（LecturerAgent）与答疑（TeacherAgent）拆分后的协作契约与回退单测。

钉住四件事：
1. **契约隔离**：两条链路的 system 里不能再出现对方的输出字段；
2. **共享人格**：同一位老师在两条链路上必须拿到同一份学科人格与风格；
3. **上下文边界**：讲课不吃答疑的对话历史与作业分层提示；
4. **异常回退**：讲课失败要退到答疑的等价实现，再失败也要有兜底话术，且全程可追溯。
"""
from __future__ import annotations

from agents.lecturer.agent import LecturerAgent
from agents.protocol import (
    KIND_ERROR,
    KIND_FALLBACK,
    KIND_HANDOFF,
    KIND_REQUEST,
    KIND_RESPONSE,
    ROLE_LECTURER,
    ROLE_TEACHER,
    AgentBus,
)
from agents.supervisor import Supervisor
from agents.shared.teaching_layers import LECTURE_BOARD_OVERRIDE
from agents.teacher.agent import TeacherAgent
from core.courseware.models import Slide, SlideDeck, SlideElement
from core.hint_homework import HintLevel
from core.state.models import AIResponse, ChatMessage, StudentTurn
from core.state.session import SessionState
from models.llm import MockLLM


# ---------------------------------------------------------------- 夹具
def _deck() -> SlideDeck:
    return SlideDeck(
        deck_id="demo",
        title="三角形的内角和",
        slides=[
            Slide(
                index=0,
                title="三角形的内角和",
                text="三角形三个内角的和是 180 度。",
                elements=[
                    SlideElement(
                        id="e1",
                        text="三角形三个内角的和是 180 度。",
                        width=0.5,
                        height=0.1,
                    ),
                ],
            ),
            Slide(index=1, title="例题", text="已知两个角，求第三个角。"),
        ],
    )


def _session(session_id: str = "lect1") -> SessionState:
    state = SessionState(session_id=session_id, student_id="stu")
    state.scratch.update(grade="初中", subject="数学", subject_id="math")
    return state


def _joined(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


# ---------------------------------------------------------------- 1. 契约隔离
def test_lecture_stack_carries_only_lecture_contract() -> None:
    """讲课栈只带课件契约（points / annotations），不带答疑契约的字段。"""
    state, deck = _session(), _deck()
    messages = LecturerAgent(llm=MockLLM())._build_messages(state, deck, 0, deck.page(0))
    joined = _joined(messages)

    assert '"points"' in joined and '"annotations"' in joined
    # 答疑契约的独占字段一个都不许出现（这正是拆分前互相打架的地方）
    for field in ('"followup"', '"knowledge_points"', '"equation"', '"steps"'):
        assert field not in joined, f"讲课栈泄漏了答疑契约字段 {field}"


def test_answer_stack_carries_only_answer_contract() -> None:
    """答疑栈只带答疑契约，不带课件契约的字段（隔离是双向的）。"""
    state = _session("ans1")
    messages = TeacherAgent(llm=MockLLM())._build_messages(
        StudentTurn(text="什么是三角形的内角和？"), state, hint_level=None, context_notes=None
    )
    joined = _joined(messages)

    assert '"steps"' in joined and '"followup"' in joined
    for field in ('"annotations"', '"points"'):
        assert field not in joined, f"答疑栈泄漏了课件契约字段 {field}"


# ---------------------------------------------------------------- 2. 共享人格
def test_lecturer_shares_subject_personality_with_teacher() -> None:
    """讲课与答疑必须拿到同一份学科人格（同一位老师，不能两副面孔）。"""
    state, deck = _session("shared1"), _deck()
    lecture = _joined(
        LecturerAgent(llm=MockLLM())._build_messages(state, deck, 0, deck.page(0))
    )
    answer = _joined(
        TeacherAgent(llm=MockLLM())._build_messages(
            StudentTurn(text="什么是三角形内角和？"), state, hint_level=None, context_notes=None
        )
    )

    marker = "严谨、清晰、耐心的数学老师"
    assert marker in lecture and marker in answer
    assert "通用教师规则" in lecture


def test_lecture_board_override_comes_after_personality() -> None:
    """讲课的「白板约定」覆盖层必须排在学科人格之后，否则盖不住人格里的白板组织。"""
    state, deck = _session("order1"), _deck()
    joined = _joined(
        LecturerAgent(llm=MockLLM())._build_messages(state, deck, 0, deck.page(0))
    )

    assert joined.index("白板组织") < joined.index(LECTURE_BOARD_OVERRIDE)


# ---------------------------------------------------------------- 3. 上下文边界
def test_lecture_does_not_inherit_answer_history_or_homework_hint() -> None:
    """讲课不吃答疑对话历史；作业分层提示只在答疑栈里出现。"""
    state, deck = _session("ctx1"), _deck()
    state.history.append(ChatMessage(role="student", text="老师这道题怎么做"))
    state.history.append(ChatMessage(role="ai", text="我们先看已知条件"))

    lecture = _joined(
        LecturerAgent(llm=MockLLM())._build_messages(state, deck, 0, deck.page(0))
    )
    assert "老师这道题怎么做" not in lecture
    assert "作业模式提示层级" not in lecture
    # 但课件上下文必须在（讲课替代历史的依据）
    assert "【教材位置】" in lecture

    answer = _joined(
        TeacherAgent(llm=MockLLM())._build_messages(
            StudentTurn(text="这道题怎么做"), state, hint_level=HintLevel.STEP, context_notes=None
        )
    )
    assert "老师这道题怎么做" in answer
    assert "作业模式提示层级" in answer


# ---------------------------------------------------------------- 4. 标注映射
def test_slide_annotations_map_target_and_skip_oversize() -> None:
    """标注意图映射到真实元素框；罩住大半页的框宁可丢弃。"""
    deck = _deck()
    slide = deck.page(0)
    slide.elements.append(SlideElement(id="big", text="整页大框", width=1.0, height=1.0))
    lecturer = LecturerAgent(llm=MockLLM())

    ops = lecturer._slide_annotations(
        deck, 0, slide,
        [
            {"target": "三角形三个内角的和是 180 度。", "kind": "annotate"},
            {"target": "整页大框", "kind": "highlight"},
            {"target": "页面上根本没这句话"},
        ],
    )
    assert len(ops) == 1
    assert ops[0].op.value == "annotate"


# ---------------------------------------------------------------- 5. 协作记录
async def test_lecturer_records_response_in_collaboration_trace() -> None:
    """讲课完成后要留下可追溯记录，并扇出成客户端事件。"""
    state, deck = _session("trace1"), _deck()
    events: list[dict] = []
    state.emit = events.append

    response = await LecturerAgent(llm=MockLLM()).explain_slide(state, deck, 0)

    assert response.speech
    trace = AgentBus.trace(state)
    assert any(
        t["kind"] == KIND_RESPONSE and t["sender"] == ROLE_LECTURER for t in trace
    )
    assert any(e.get("type") == "agent_message" for e in events)


# ---------------------------------------------------------------- 6. 异常回退
class _BoomLecturer:
    async def explain_slide(self, session, deck, index, *, extra_notes=None):
        raise RuntimeError("lecturer down")


class _BoomTeacher:
    async def explain_slide(self, session, deck, index, *, extra_notes=None):
        raise RuntimeError("teacher down")


class _OkTeacher:
    async def explain_slide(self, session, deck, index, *, extra_notes=None):
        return AIResponse(speech="回退讲解")


async def test_supervisor_falls_back_to_teacher_when_lecturer_fails() -> None:
    """讲课失败 -> 回退答疑链路的等价实现，且回退动作被记录。"""
    state, deck = _session("fb1"), _deck()
    supervisor = Supervisor(state, teacher=_OkTeacher(), lecturer=_BoomLecturer())

    response = await supervisor._explain_slide(deck, 0)

    assert response.speech == "回退讲解"
    assert [t["kind"] for t in AgentBus.trace(state)] == [KIND_REQUEST, KIND_FALLBACK]
    assert AgentBus.trace(state)[-1]["recipient"] == ROLE_TEACHER


async def test_supervisor_degrades_gracefully_when_both_paths_fail() -> None:
    """两条链路都失败时也要有话可说 —— 连讲不能被一页的异常打断。"""
    state, deck = _session("fb2"), _deck()
    supervisor = Supervisor(state, teacher=_BoomTeacher(), lecturer=_BoomLecturer())

    response = await supervisor._explain_slide(deck, 0)

    assert response.speech
    assert [t["kind"] for t in AgentBus.trace(state)] == [
        KIND_REQUEST, KIND_FALLBACK, KIND_ERROR,
    ]


# ---------------------------------------------------------------- 7. 抢占转交
async def test_question_during_lecture_records_handoff() -> None:
    """讲课中学生提问 = 一次转交：讲课件 agent 让出话语权，答疑 agent 接手。"""
    state = _session("ho1")
    state.deck_id = "demo"
    state.page_index = 0
    supervisor = Supervisor(
        state,
        teacher=TeacherAgent(llm=MockLLM()),
        lecturer=LecturerAgent(llm=MockLLM()),
    )

    await supervisor.handle_turn(StudentTurn(text="老师，这里为什么要减 180？"))

    handoffs = [t for t in AgentBus.trace(state) if t["kind"] == KIND_HANDOFF]
    assert handoffs, "讲课中提问应留下转交记录"
    assert handoffs[0]["recipient"] == ROLE_TEACHER


# ---------------------------------------------------------------- 8. 兼容转发
async def test_teacher_explain_slide_still_works_and_shares_llm() -> None:
    """老调用方直接调 teacher.explain_slide 仍可用，且复用同一个 llm/白板。"""
    state, deck = _session("compat1"), _deck()
    teacher = TeacherAgent(llm=MockLLM())

    response = await teacher.explain_slide(state, deck, 0)

    assert response.speech
    assert teacher._lecturer_agent is not None
    assert teacher._lecturer_agent.llm is teacher.llm
    assert teacher._lecturer_agent.board is teacher.board
