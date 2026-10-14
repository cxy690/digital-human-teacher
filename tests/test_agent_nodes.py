"""各 Agent 的流程节点（``agents/<角色>/nodes.py`` + 注册表 ``agents/nodes.py``）单测。

钉住四件事：
1. **地图完整**：四个角色都有图，每条图的节点声明自洽（依赖存在、不重名、有功能说明）；
2. **声明 = 实现**：按声明构建出的图，执行顺序 / 校验 / 自省输出与声明逐字一致；
3. **分支语义**：回合主图只执行命中的那一个分支（其余 ``__skip__``），且与原 if/elif 等价；
4. **降级链**：讲课件、设备控制、语音、判题、出题这些链路图真的按声明的顺序跑。
"""
from __future__ import annotations

from agents import nodes as agent_nodes
from agents.lecturer.agent import LecturerAgent
from agents.protocol import (
    KIND_ERROR,
    KIND_FALLBACK,
    KIND_REQUEST,
    AgentBus,
    ROLE_LECTURER,
    ROLE_STUDENT,
    ROLE_SUPERVISOR,
    ROLE_TEACHER,
)
from agents.supervisor import Supervisor
from agents.teacher.agent import TeacherAgent
from config.settings import settings
from core.courseware.models import Slide, SlideDeck
from core.state.models import AIResponse, StudentTurn
from core.state.session import SessionState
from models.asr import MockASR
from models.llm import MockLLM


# ---------------------------------------------------------------- 夹具
def _session(session_id: str = "nodes") -> SessionState:
    state = SessionState(session_id=session_id, student_id="stu")
    state.scratch.update(grade="初中", subject="数学", subject_id="math")
    return state


def _supervisor(session_id: str = "nodes") -> tuple[Supervisor, SessionState]:
    state = _session(session_id)
    supervisor = Supervisor(
        state,
        teacher=TeacherAgent(llm=MockLLM()),
        lecturer=LecturerAgent(llm=MockLLM()),
    )
    supervisor.asr = MockASR()
    return supervisor, state


def _deck() -> SlideDeck:
    return SlideDeck(
        deck_id="demo",
        title="三角形的内角和",
        slides=[Slide(index=0, title="三角形的内角和", text="三角形内角和是 180 度。")],
    )


class _BoomLecturer:
    async def explain_slide(self, session, deck, index, *, extra_notes=None):
        raise RuntimeError("lecturer down")


class _BoomTeacher:
    async def explain_slide(self, session, deck, index, *, extra_notes=None):
        raise RuntimeError("teacher down")


class _OkTeacher:
    async def explain_slide(self, session, deck, index, *, extra_notes=None):
        return AIResponse(speech="回退讲解")


class _BoomASR:
    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000) -> str:
        raise AssertionError("关麦时不该调用 ASR")


# ---------------------------------------------------------------- 1. 地图完整
def test_node_map_covers_every_agent_role() -> None:
    assert set(agent_nodes.AGENT_NODES) == {
        ROLE_SUPERVISOR, ROLE_TEACHER, ROLE_LECTURER, ROLE_STUDENT,
    }
    assert set(agent_nodes.BUILDERS) == set(agent_nodes.AGENT_NODES)

    for role, graphs in agent_nodes.AGENT_NODES.items():
        assert graphs, f"{role} 至少要有一条流程"
        for graph, specs in graphs.items():
            assert graph in agent_nodes.BUILDERS[role], f"{graph} 没有构建函数"
            names = [spec.name for spec in specs]
            assert names and len(names) == len(set(names)), f"{graph} 节点名重复"
            for spec in specs:
                assert spec.summary, f"{graph}.{spec.name} 缺功能说明"
                assert spec.impl, f"{graph}.{spec.name} 缺实现绑定说明"


def test_declared_dependencies_are_satisfied() -> None:
    """声明级自检：每条连接都指向本图内存在的节点（注册表据此报问题）。"""
    for role, graphs in agent_nodes.describe_agent_nodes().items():
        for graph, info in graphs.items():
            assert info["problem"] == [], f"{role}.{graph} 连接有问题：{info['problem']}"


def test_built_graph_matches_declaration() -> None:
    """按声明构建的图：执行顺序、校验、自省输出都与声明逐字一致。"""
    supervisor, _ = _supervisor()
    for role, graphs in agent_nodes.AGENT_NODES.items():
        for graph, specs in graphs.items():
            built = agent_nodes.build_agent_graph(role, graph, supervisor)
            assert built.name == graph
            assert built.order() == [spec.name for spec in specs]
            assert built.validate() == []
            assert built.describe() == [spec.to_dict() for spec in specs]


def test_render_agent_nodes_lists_chains() -> None:
    """``render_agent_nodes`` 是「节点名称 + 顺序 + 输入输出」的可读视图。"""
    text = agent_nodes.render_agent_nodes(ROLE_TEACHER)
    assert "teacher.answer: retrieve → generate → validate → degrade → solve" in text
    assert "teacher.judge: evaluate → decide → announce" in text
    assert "teacher.quiz: generate → publish → announce" in text
    # 依赖与配置说明都在
    assert "依赖:" in text and "deps" in text


# ---------------------------------------------------------------- 2. 回合主图
async def test_turn_graph_runs_only_selected_branch() -> None:
    """闲聊回合：route 选 chitchat，其余分支节点全部跳过。"""
    supervisor, _ = _supervisor("turn-chitchat")
    graph = agent_nodes.build_agent_graph(ROLE_SUPERVISOR, "supervisor.turn", supervisor)

    run = await graph.run({"turn": StudentTurn(text="你好")})

    assert run.context["route"] == "chitchat"
    assert run.context["response"] is not None
    assert set(run.skipped) >= {"answer_check", "multi_intent", "control", "teach"}


async def test_turn_graph_control_branch_uses_device_graph() -> None:
    """控制回合：只跑 control 分支，设备副作用由学生端设备图落地（返回 None）。"""
    supervisor, _ = _supervisor("turn-control")
    graph = agent_nodes.build_agent_graph(ROLE_SUPERVISOR, "supervisor.turn", supervisor)

    run = await graph.run({"turn": StudentTurn(text="关闭麦克风")})

    assert run.context["route"] == "control"
    assert run.context["response"] is None
    assert supervisor.student.microphone_on is False
    assert set(run.skipped) >= {"answer_check", "multi_intent", "chitchat", "teach"}


async def test_turn_graph_answer_check_branch_wins_when_pending() -> None:
    """有待作答且输入像作答：answer_check 优先，不再重新路由。"""
    supervisor, state = _supervisor("turn-pending")
    state.pending_question = {
        "question": "1+1 等于几？", "expected": "2", "hint": "", "options": ["2", "3"],
        "attempts": 0,
    }
    graph = agent_nodes.build_agent_graph(ROLE_SUPERVISOR, "supervisor.turn", supervisor)

    run = await graph.run({"turn": StudentTurn(text="2")})

    assert run.context["route"] == "answer_check"
    assert run.context["response"] is not None
    assert state.pending_question is None, "答对后应收起待答问题"


# ---------------------------------------------------------------- 3. 学生端
async def test_device_control_keeps_first_match_semantics() -> None:
    """设备控制图与原 if/elif 等价：按声明顺序**先匹配先执行**，命中即收工。"""
    supervisor, _ = _supervisor("control")

    supervisor._handle_control("关闭摄像头，另外打开麦克风")
    assert supervisor.student.camera_on is False
    assert supervisor.student.microphone_on is True, "先匹配 camera_off，不该再执行 mic_on"

    supervisor._handle_control("打开麦克风")
    assert supervisor.student.microphone_on is True

    supervisor._handle_control("清空白板")
    assert [op.op.value for op in supervisor.session.board_ops] == ["clear"]


async def test_voice_graph_stops_before_asr_when_mic_off() -> None:
    """关麦时语音图在 guard_mic 收工，绝不调用 ASR。"""
    supervisor, state = _supervisor("voice-off")
    supervisor.asr = _BoomASR()
    supervisor.student.microphone_on = False

    assert await supervisor.on_voice(b"audio") is None
    assert state.history[-1].role == "ai"


async def test_voice_graph_builds_turn_and_dispatches() -> None:
    """开麦且识别有文本：转写 -> 构造回合 -> 交回总控（这里落到答疑分支）。"""
    supervisor, state = _supervisor("voice-on")
    supervisor.asr = MockASR()

    response = await supervisor.on_voice(b"audio")

    assert response is not None and response.speech
    assert state.board_ops, "答疑分支应产出白板指令"


def test_followup_without_choices_is_not_sent_as_a_choice_question(monkeypatch) -> None:
    """没有比较范围的“哪个数”追问不能让学生进入无法作答的状态。"""
    monkeypatch.setattr(settings, "interaction_enable", True)
    supervisor, state = _supervisor("invalid-followup")
    events: list[dict] = []
    state.emit = events.append

    supervisor._ask_followup(AIResponse(
        speech="讲解结束。",
        meta={"followup": {"question": "哪个数是有理数？", "answer": "0.333…"}},
    ))

    assert state.pending_question is None
    assert state.followup_asked == 0
    assert not any(event["type"] == "ai_ask" for event in events)


def test_followup_with_choices_is_sent_to_the_student(monkeypatch) -> None:
    """完整的选择题仍应保留题干、选项和标准答案。"""
    monkeypatch.setattr(settings, "interaction_enable", True)
    supervisor, state = _supervisor("valid-followup")
    events: list[dict] = []
    state.emit = events.append

    supervisor._ask_followup(AIResponse(
        speech="讲解结束。",
        meta={"followup": {
            "question": "哪个数是有理数？",
            "answer": "0.333…",
            "hint": "循环小数可以化成分数。",
            "options": ["π", "√2", "0.333…"],
        }},
    ))

    assert (state.pending_question or {})["expected"] == "0.333…"
    assert (state.pending_question or {})["options"] == ["π", "√2", "0.333…"]
    ask_events = [event for event in events if event["type"] == "ai_ask"]
    assert ask_events and ask_events[-1]["payload"]["options"] == ["π", "√2", "0.333…"]


# ---------------------------------------------------------------- 4. 教师三条流水线
def test_teacher_answer_graph_matches_langgraph_node_order() -> None:
    """答疑流水线的节点名必须与 LangGraph 版子图同名同序（两套编排强绑定）。"""
    supervisor, _ = _supervisor("answer")
    graph = supervisor._build_answer_graph()
    assert graph.name == "teacher.answer"
    assert graph.order() == ["retrieve", "generate", "validate", "degrade", "solve"]


async def test_teacher_quiz_graph_publishes_pending_question() -> None:
    """出题流水线：生成 -> 入库（pending_question/quiz_count）-> 推 ai_quiz -> 播报。"""
    supervisor, state = _supervisor("quiz")
    events: list[dict] = []
    state.emit = events.append

    response = await supervisor.on_quiz_request("三角形内角和")

    assert response is not None and response.speech
    assert state.quiz_count == 1 and state.quiz is not None
    assert (state.pending_question or {}).get("kind") == "quiz"
    assert any(e["type"] == "ai_quiz" for e in events)
    assert state.teaching is False and state.ai_speaking is False, "收尾必须复位状态"


async def test_teacher_judge_graph_feedback_and_retry() -> None:
    """判题流水线：答对收起题目；答错可重试（保留待答并累计 attempts）。"""
    supervisor, state = _supervisor("judge")
    events: list[dict] = []
    state.emit = events.append
    state.pending_question = {
        "question": "三角形的内角和是多少？",
        "expected": "180度",
        "hint": "想想平角",
        "options": ["180度", "90度"],
        "attempts": 0,
    }

    response = await supervisor.handle_turn(StudentTurn(text="180度"))

    assert response is not None
    check = [e for e in events if e["type"] == "ai_check"]
    assert check and check[-1]["payload"]["correct"] is True
    assert state.pending_question is None and state.quiz is None
async def test_teacher_judge_graph_reveals_answer_after_first_wrong_attempt() -> None:
    """互动题第一次答错即公布答案与思路，不要求学生再次作答。"""
    supervisor, state = _supervisor("judge-wrong")
    events: list[dict] = []
    state.emit = events.append
    state.pending_question = {
        "question": "三角形的内角和是多少？",
        "expected": "180度",
        "hint": "三角形三个内角相加等于平角。",
        "options": ["180度", "90度"],
        "attempts": 0,
        "kind": "quiz",
    }

    response = await supervisor.handle_turn(StudentTurn(text="90度"))

    assert response is not None and "180度" in response.speech
    assert "解题思路" in response.speech
    assert state.pending_question is None and state.quiz is None
    check = [event for event in events if event["type"] == "ai_check"]
    assert check and check[-1]["payload"]["allow_retry"] is False


# ---------------------------------------------------------------- 5. 讲课件降级链
async def test_lecture_graph_falls_back_then_returns_teacher_result() -> None:
    supervisor, state = _supervisor("lecture-fallback")
    supervisor.lecturer = _BoomLecturer()
    supervisor.teacher = _OkTeacher()

    response = await supervisor._explain_slide(_deck(), 0)

    assert response.speech == "回退讲解"
    kinds = [t["kind"] for t in AgentBus.trace(state)]
    assert kinds == [KIND_REQUEST, KIND_FALLBACK]
    assert AgentBus.trace(state)[-1]["recipient"] == ROLE_TEACHER


async def test_lecture_graph_degrades_when_both_paths_fail() -> None:
    supervisor, state = _supervisor("lecture-degrade")
    supervisor.lecturer = _BoomLecturer()
    supervisor.teacher = _BoomTeacher()

    response = await supervisor._explain_slide(_deck(), 0)

    assert response.speech
    assert [t["kind"] for t in AgentBus.trace(state)] == [
        KIND_REQUEST, KIND_FALLBACK, KIND_ERROR,
    ]


def test_slide_graph_keeps_present_then_explain_order() -> None:
    """单页呈现：present_page -> explain -> annotate -> speak -> push_audio。"""
    supervisor, _ = _supervisor("slide-order")
    graph = agent_nodes.build_agent_graph(ROLE_SUPERVISOR, "supervisor.slide", supervisor)
    assert graph.order() == ["present_page", "explain", "annotate", "speak", "push_audio"]


async def test_slide_graph_stops_before_explain_when_previewing() -> None:
    """``with_speech=False``（预览某页）：只上页面，不讲解、不出声。"""
    supervisor, state = _supervisor("slide-preview")
    graph = agent_nodes.build_agent_graph(ROLE_SUPERVISOR, "supervisor.slide", supervisor)

    run = await graph.run({"deck": _deck(), "index": 0, "with_speech": False})

    assert run.context.get("response") is None
    assert run.stopped == ["explain"], "预览在 explain 节点收工"
    assert state.page_index == 0 and state.board_ops, "页面仍然要亮出来"
    assert "push_audio" not in run.timings_ms, "收工后不应再出声"
