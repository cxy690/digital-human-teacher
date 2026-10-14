"""意图识别 / 分流 Agent 单测（纯规则 + 模型分类桩，无需外部依赖）。"""
from __future__ import annotations

import pytest

from agents.supervisor import Supervisor
from config.constants import IntentType
from config.settings import settings
from core.intent import IntentAnalyzer, IntentLlmClassifier
from core.langgraph_flow import LANGGRAPH_AVAILABLE
from core.state.models import AIResponse, StudentTurn
from core.state.session import SessionState
from models.llm import MockLLM


def test_simple_control_is_classified_directly() -> None:
    agent = IntentAnalyzer()
    decision = agent.analyze(StudentTurn(text="打开麦克风"))
    assert decision.intent is IntentType.CONTROL
    assert decision.simple and not decision.delegate


def test_single_question_is_simple() -> None:
    agent = IntentAnalyzer()
    decision = agent.analyze(StudentTurn(text="老师，这道方程题怎么解？"))
    assert decision.intent is IntentType.TEXT_QUESTION
    assert decision.simple and not decision.delegate
    assert decision.delegation_notes() == []


def test_voice_question_keeps_voice_intent() -> None:
    agent = IntentAnalyzer()
    decision = agent.analyze(
        StudentTurn(text="为什么天空是蓝色的？", meta={"from_voice": True})
    )
    assert decision.intent is IntentType.VOICE_QUESTION
    assert decision.simple


def test_multi_intent_is_delegated_with_sub_questions() -> None:
    agent = IntentAnalyzer()
    decision = agent.analyze(StudentTurn(text="这道方程题怎么解？另外帮我打开麦克风"))
    assert decision.multi and decision.delegate and not decision.simple
    # 主意图取「非控制类」的那个，避免答疑被控制指令带跑
    assert decision.intent is IntentType.TEXT_QUESTION
    assert decision.control_clauses == ["帮我打开麦克风"]
    notes = decision.delegation_notes()
    assert notes and "逐个作答" in notes[0]
    # 控制指令不进转交清单（入口层自己就能办掉）
    assert all("麦克风" not in line for line in notes)


def test_long_text_is_complex_and_delegated() -> None:
    agent = IntentAnalyzer()
    decision = agent.analyze(StudentTurn(text="这道题" * 25))
    assert decision.complex and decision.delegate and not decision.multi
    assert "分步" in decision.delegation_notes()[0]


def test_empty_text_falls_back_to_unknown() -> None:
    agent = IntentAnalyzer()
    decision = agent.analyze(StudentTurn(text=""))
    assert decision.intent is IntentType.UNKNOWN
    assert decision.simple and not decision.delegate


def test_router_interface_is_compatible() -> None:
    """可直接替换 Supervisor.router：route / route_async / _looks_like_question 均可用。"""
    agent = IntentAnalyzer()
    assert agent.route(StudentTurn(text="你好")) is IntentType.CHITCHAT
    assert agent._looks_like_question("这题怎么做?")  # noqa: SLF001
    assert agent._looks_like_question("你好") is False  # noqa: SLF001


async def test_analyze_async_uses_router_fallback() -> None:
    agent = IntentAnalyzer()
    decision = await agent.analyze_async(StudentTurn(text="帮我看看这个"))
    assert decision.intent is IntentType.TEXT_QUESTION
    assert decision.simple and not decision.delegate


async def test_supervisor_executes_control_inside_multi_intent() -> None:
    """多意图：控制指令（简单意图）当场执行，提问部分转交下游答疑。"""
    state = SessionState(session_id="i1", student_id="s1")
    supervisor = Supervisor(state)
    await supervisor.on_start()
    assert supervisor.student.microphone_on is True

    turn = StudentTurn(text="这道方程题怎么解？另外帮我关闭麦克风")
    response = await supervisor.handle_turn(turn)
    assert response is not None
    assert supervisor.student.microphone_on is False
    assert turn.meta["intent_decision"]["delegate"] is True


# --------------------------------------------------------------- 小模型复核
class _SpyClassifier:
    """分类桩：记录被调用的文本，返回预设结果或抛错。"""

    def __init__(self, data: dict | None = None, *, error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self._data = data
        self._error = error

    async def classify(self, text: str, *, base=None):
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return self._data


async def test_simple_input_does_not_call_model() -> None:
    """常见输入（控制 / 闲聊 / 单一提问）走规则直判，不付模型开销。"""
    spy = _SpyClassifier()
    agent = IntentAnalyzer(classifier=spy)
    for text in ("打开麦克风", "你好", "老师，这道方程题怎么解？"):
        decision = await agent.analyze_async(StudentTurn(text=text))
        assert decision.simple and not decision.delegate
    assert spy.calls == [], "简单意图不应触发模型"


async def test_multi_intent_triggers_model_and_merges_sub_questions() -> None:
    spy = _SpyClassifier({
        "intent": IntentType.TEXT_QUESTION,
        "multi": True,
        "complex": False,
        "sub_questions": ["这道方程题怎么解", "帮我打开麦克风"],
    })
    agent = IntentAnalyzer(classifier=spy)
    decision = await agent.analyze_async(
        StudentTurn(text="这道方程题怎么解？另外帮我打开麦克风")
    )
    assert spy.calls, "多意图应触发小模型复核"
    assert decision.multi and decision.delegate and not decision.simple
    assert decision.clauses == ["这道方程题怎么解", "帮我打开麦克风"]
    # 控制指令始终由规则关键词判定（不交给模型猜），入口层据此当场执行
    assert decision.control_clauses == ["帮我打开麦克风"]
    assert "小模型复核" in decision.reason


async def test_uncertain_input_is_handed_to_model() -> None:
    """规则一个线索都没命中 -> 标 uncertain，交小模型判定。"""
    spy = _SpyClassifier({
        "intent": IntentType.CHITCHAT, "multi": False, "complex": False,
        "sub_questions": ["嗯那个"],
    })
    agent = IntentAnalyzer(classifier=spy)
    assert agent.analyze(StudentTurn(text="嗯那个")).uncertain, "规则应标出「没学到线索」"

    decision = await agent.analyze_async(StudentTurn(text="嗯那个"))
    assert spy.calls == ["嗯那个"]
    assert decision.intent is IntentType.CHITCHAT
    assert not decision.uncertain


async def test_model_failure_falls_back_to_rules() -> None:
    """模型超时/报错 -> 原样使用规则结果，绝不阻断答疑。"""
    spy = _SpyClassifier(error=TimeoutError("classifier timeout"))
    agent = IntentAnalyzer(classifier=spy)
    decision = await agent.analyze_async(
        StudentTurn(text="这道方程题怎么解？另外帮我打开麦克风")
    )
    assert decision.multi and decision.delegate
    assert decision.control_clauses == ["帮我打开麦克风"]


async def test_classifier_returns_none_on_unparsable_output() -> None:
    """Mock / 非 JSON 输出 -> 返回 None（调用方回退规则，不抛异常）。"""
    assert await IntentLlmClassifier(llm=MockLLM()).classify("你好") is None


async def test_classifier_normalizes_model_json() -> None:
    """模型返回的 JSON 被归一化：意图名转枚举、子问题清洗、缺失项兜底。"""

    class _JsonLLM(MockLLM):
        async def chat(self, messages, *, temperature: float = 0.3, max_tokens: int = 1024) -> str:
            return (
                '{"intent": "text_question", "multi": true, "complex": false,'
                ' "sub_questions": ["A？", "  B？  ", ""]}'
            )

    data = await IntentLlmClassifier(llm=_JsonLLM()).classify("A？B？")
    assert data == {
        "intent": IntentType.TEXT_QUESTION,
        "multi": True,
        "complex": False,
        "sub_questions": ["A？", "B？"],
    }
    # 完全无法解析（缺意图且无子问题）时返回 None
    assert IntentLlmClassifier._normalize({"foo": "bar"}) is None


# --------------------------------------------------------------- 多意图逐条分发
def _install_teach_spy(supervisor: Supervisor) -> list[tuple[str, bool]]:
    """替换答疑链路为记录桩，返回 ``[(子问题文本, 是否追问), ...]``。"""
    calls: list[tuple[str, bool]] = []

    async def _fake_teach(turn, intent, *, ocr_text: str = "", ask_followup: bool = True):
        calls.append((turn.text, ask_followup))
        return AIResponse(speech=f"讲解：{turn.text}")

    supervisor._turn_teach = _fake_teach  # type: ignore[method-assign]
    return calls


def test_plan_puts_control_first_then_questions() -> None:
    agent = IntentAnalyzer()
    decision = agent.analyze(
        StudentTurn(text="这道方程题怎么解？另外这道函数题怎么解？然后帮我关闭麦克风")
    )
    plan = agent.plan(decision)
    assert [seg.intent for seg in plan] == [
        IntentType.CONTROL, IntentType.TEXT_QUESTION, IntentType.TEXT_QUESTION
    ]
    assert plan[0].text == "帮我关闭麦克风"
    assert [plan[1].text, plan[2].text] == ["这道方程题怎么解", "这道函数题怎么解"]
    # clauses/control_clauses 保持向后兼容
    assert decision.control_clauses == ["帮我关闭麦克风"]


async def test_multi_question_is_answered_one_by_one() -> None:
    """两个提问 -> 逐条独立答疑，且只对最后一条追问。"""
    state = SessionState(session_id="m1", student_id="s1")
    supervisor = Supervisor(state)
    await supervisor.on_start()
    calls = _install_teach_spy(supervisor)

    response = await supervisor.handle_turn(
        StudentTurn(text="这道方程题怎么解？另外这道函数题怎么解？")
    )

    assert response is not None
    assert [c[0] for c in calls] == ["这道方程题怎么解", "这道函数题怎么解"]
    # 只有最后一条追问：否则前一题的追问会被下一题覆盖，学生答了也判不上
    assert [c[1] for c in calls] == [False, True]
    assert response.meta["multi_intent_count"] == 2


async def test_multi_intent_executes_control_before_questions() -> None:
    """多意图里夹带的控制指令先当场执行，再逐条答疑。"""
    state = SessionState(session_id="m2", student_id="s2")
    supervisor = Supervisor(state)
    await supervisor.on_start()
    order: list[str] = []
    calls = _install_teach_spy(supervisor)
    original_control = supervisor._handle_control

    def _spy_control(text: str) -> None:
        order.append(f"control:{text}")
        original_control(text)

    supervisor._handle_control = _spy_control  # type: ignore[method-assign]

    await supervisor.handle_turn(
        StudentTurn(text="这道方程题怎么解？另外这道函数题怎么解？然后帮我关闭麦克风")
    )

    assert order == ["control:帮我关闭麦克风"], "控制指令应先于提问执行"
    assert supervisor.student.microphone_on is False
    assert len(calls) == 2


async def test_multi_question_parity_builtin_orchestrator(monkeypatch) -> None:
    """内建 TaskGraph 与 LangGraph 两套编排的多意图分发行为必须一致。

    必须**真实跑** langgraph 分支：只把开关设回 builtin 再断言一次，
    等于没有守线（langgraph 是生产可切换的编排）。
    """
    if not LANGGRAPH_AVAILABLE:
        pytest.skip("langgraph not installed")

    question = "这道方程题怎么解？另外这道函数题怎么解？"

    async def _dispatch(orchestrator: str) -> list[tuple[str, bool]]:
        monkeypatch.setattr(settings, "orchestrator", orchestrator)
        state = SessionState(session_id=f"m4-{orchestrator}", student_id="s4")
        supervisor = Supervisor(state)
        await supervisor.on_start()
        calls = _install_teach_spy(supervisor)
        await supervisor.handle_turn(StudentTurn(text=question))
        return calls

    builtin_calls = await _dispatch("builtin")
    langgraph_calls = await _dispatch("langgraph")

    assert builtin_calls == [("这道方程题怎么解", False), ("这道函数题怎么解", True)]
    assert langgraph_calls == builtin_calls


async def test_single_question_multi_intent_uses_normal_path() -> None:
    """只有一条提问的多意图：控制当场办，提问走常规单意图链路（不必逐条）。"""
    state = SessionState(session_id="m3", student_id="s3")
    supervisor = Supervisor(state)
    await supervisor.on_start()
    calls = _install_teach_spy(supervisor)

    turn = StudentTurn(text="这道方程题怎么解？另外帮我关闭麦克风")
    await supervisor.handle_turn(turn)

    assert supervisor.student.microphone_on is False
    assert len(calls) == 1
    # 常规路径下 teacher 拿到的是整句（含转交清单），不是被切开的子句
    assert calls[0][0] == turn.text

