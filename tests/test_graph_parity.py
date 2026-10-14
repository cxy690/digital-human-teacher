"""编排一致性测试：builtin（自研 TaskGraph）与 langgraph（LangGraph StateGraph）两条链路行为应一致。

为什么单独成文件：``ORCHESTRATOR=langgraph`` 在生产可切换（langgraph 已随
``[langgraph]`` 可选依赖安装），但两套编排的分派/答疑链路是**两份实现**
（``Supervisor._handle_turn_builtin`` / ``_run_answer_pipeline`` 与
``core/turn_graph.py`` / ``core/langgraph_flow.py``），必须有测试真实跑 langgraph 分支，
否则「开关一拨就变行为」没有任何守线。

本文件由 ``_cloud_pull/tests/test_{turn,answer}_graph_parity.py`` 迁移而来，并修正：
* 总控已从 ``core/supervisor.py`` 迁到 ``agents/supervisor/agent.py``，
  导入路径随之更新（``from core.supervisor import Supervisor`` 已不存在）；
* 依赖缺失回退用例改为打 ``agents.supervisor.agent.LANGGRAPH_AVAILABLE``
  （``handle_turn`` 用的是该模块的全局，而不是旧 ``core.supervisor`` 模块的）。
"""
from __future__ import annotations

import pytest

import agents.supervisor.agent as supervisor_module
from agents.supervisor import Supervisor
from agents.teacher.agent import TeacherAgent
from config.constants import IntentType
from config.settings import settings
from core.langgraph_flow import LANGGRAPH_AVAILABLE
from core.state.models import AIResponse, StudentTurn
from core.state.session import SessionState
from models.asr import MockASR
from models.llm import MockLLM
from models.tts import MockTTS

_requires_langgraph = pytest.mark.skipif(
    not LANGGRAPH_AVAILABLE, reason="langgraph not installed"
)


def _new_supervisor() -> Supervisor:
    session = SessionState(session_id="parity", student_id="stu")
    supervisor = Supervisor(session, teacher=TeacherAgent(llm=MockLLM()), retriever=None)
    supervisor.tts = MockTTS()  # 避免测试触发真实 TTS 网络请求
    supervisor.asr = MockASR()
    return supervisor


async def _run(
    text: str, intent: IntentType | None = None
) -> tuple[Supervisor, AIResponse | None]:
    """用离线 MockLLM 跑一次回合，返回 ``(supervisor, AI 产物)``。"""
    supervisor = _new_supervisor()
    turn = StudentTurn(text=text, intent=intent) if intent is not None else StudentTurn(text=text)
    return supervisor, await supervisor.handle_turn(turn)


# --------------------------------------------------------------- 回合主图
@_requires_langgraph
async def test_resume_lecture_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """「请继续」：两条编排都走课件断点续讲（同一次 handle_slide_play(-1) 调用）。"""
    for orch in ("builtin", "langgraph"):
        monkeypatch.setattr(settings, "orchestrator", orch)
        supervisor = _new_supervisor()
        supervisor.session.deck_id = "d1"
        calls: list[int] = []

        async def fake_play(index: int = -1) -> None:
            calls.append(index)

        monkeypatch.setattr(supervisor, "handle_slide_play", fake_play)
        response = await supervisor.handle_turn(StudentTurn(text="请继续"))

        assert response is None, f"{orch}: 续讲指令不出额外语音"
        assert calls == [-1], f"{orch}: 应触发断点续讲"


@_requires_langgraph
async def test_control_turn_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """控制指令：两条链路都返回 None，且产生相同的设备副作用。"""
    monkeypatch.setattr(settings, "orchestrator", "builtin")
    sup_b, resp_b = await _run("关闭麦克风")

    monkeypatch.setattr(settings, "orchestrator", "langgraph")
    sup_l, resp_l = await _run("关闭麦克风")

    assert resp_b is None and resp_l is None
    assert sup_b.student.microphone_on is False
    assert sup_b.student.microphone_on == sup_l.student.microphone_on


@_requires_langgraph
async def test_chitchat_turn_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """闲聊：两条链路都产出同样的温柔回应。"""
    monkeypatch.setattr(settings, "orchestrator", "builtin")
    _, resp_b = await _run("你好")

    monkeypatch.setattr(settings, "orchestrator", "langgraph")
    _, resp_l = await _run("你好")

    assert resp_b is not None and resp_l is not None
    assert resp_b.speech and resp_b.speech == resp_l.speech


@_requires_langgraph
async def test_question_turn_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """答疑：两条链路的 speech 与白板指令序列一致。"""
    monkeypatch.setattr(settings, "orchestrator", "builtin")
    _, resp_b = await _run("为什么 1+1=2？")

    monkeypatch.setattr(settings, "orchestrator", "langgraph")
    _, resp_l = await _run("为什么 1+1=2？")

    assert resp_b is not None and resp_l is not None
    assert resp_b.speech and resp_l.speech
    assert [op.op for op in resp_b.board_ops] == [op.op for op in resp_l.board_ops]


@_requires_langgraph
async def test_multi_question_dispatch_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """多意图逐条作答：两条编排的分发结果一致（各自答完两条子问题）。"""
    monkeypatch.setattr(settings, "orchestrator", "builtin")
    sup_b, resp_b = await _run("这道方程题怎么解？另外这道函数题怎么解？")

    monkeypatch.setattr(settings, "orchestrator", "langgraph")
    sup_l, resp_l = await _run("这道方程题怎么解？另外这道函数题怎么解？")

    assert resp_b is not None and resp_l is not None
    assert resp_b.meta["multi_intent_count"] == 2
    assert resp_l.meta["multi_intent_count"] == 2
    assert len(sup_b.session.history) == len(sup_l.session.history)
    assert [op.op for op in resp_b.board_ops] == [op.op for op in resp_l.board_ops]


@_requires_langgraph
async def test_langgraph_turn_graph_compiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """开关指向 langgraph 时，应真正构建并缓存回合主图。"""
    monkeypatch.setattr(settings, "orchestrator", "langgraph")
    supervisor, _ = await _run("讲讲这道题")
    assert supervisor._turn_graph is not None


# --------------------------------------------------------------- 答疑子图
@_requires_langgraph
async def test_langgraph_answer_graph_compiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """开关指向 langgraph 时，答疑子图也应被构建并缓存。"""
    monkeypatch.setattr(settings, "orchestrator", "langgraph")
    supervisor, response = await _run("讲讲这道题", IntentType.TEXT_QUESTION)
    assert response is not None and response.speech
    assert supervisor._lg_graph is not None


async def test_langgraph_switch_falls_back_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """开关指向 langgraph 但依赖缺失时，应自动回退自研图并正常产出话术。

    这里打的必须是 ``agents.supervisor.agent.LANGGRAPH_AVAILABLE`` ——
    ``handle_turn`` / ``_run_answer_pipeline`` 读的是该模块的全局绑定。
    """
    monkeypatch.setattr(settings, "orchestrator", "langgraph")
    monkeypatch.setattr(supervisor_module, "LANGGRAPH_AVAILABLE", False)

    supervisor, response = await _run("这道题怎么做？", IntentType.TEXT_QUESTION)
    assert response is not None and response.speech
    assert supervisor._lg_graph is None      # 回退自研图：不该编译 LangGraph 子图
