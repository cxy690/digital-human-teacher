"""表情技能单测（注入 Fake LLM，不依赖外部网络）。"""
from __future__ import annotations

from core.skills.expression_skill import EXPRESSION_MATERIALS, ExpressionSkill
from core.state.models import ChatMessage
from core.state.session import SessionState
from models.base import BaseLLMClient, ChatMessages


class _FakeLLM(BaseLLMClient):
    """返回固定文本的假 LLM。"""

    def __init__(self, reply: str = "") -> None:
        self.reply = reply
        self.calls = 0

    async def chat(
        self, messages: ChatMessages, *, temperature: float = 0.3, max_tokens: int = 1024
    ) -> str:
        self.calls += 1
        return self.reply


def _session_with_teaching() -> SessionState:
    """构造带 AI 讲解上下文的会话。"""
    state = SessionState(session_id="exp", student_id="s")
    state.add_message(
        ChatMessage(role="ai", text="我们看这道题：12 乘 3，先算 10 乘 3 等于 30，再加 2 乘 3 等于 6。")
    )
    return state


def test_should_try_rejects_before_llm() -> None:
    skill = ExpressionSkill(llm=_FakeLLM())
    state = _session_with_teaching()
    # 问句、求助、寒暄、超长都不进入判定
    assert not skill.should_try("为什么是这样？", state)
    assert not skill.should_try("帮我讲讲", state)
    assert not skill.should_try("你好", state)
    assert not skill.should_try("好" * 50, state)
    # 无 AI 讲解上下文时也不判定
    empty = SessionState(session_id="e2", student_id="s2")
    assert not skill.should_try("42", empty)


async def test_judge_skips_llm_when_prescreen_fails() -> None:
    fake = _FakeLLM('{"is_answer": true, "correct": true, "level": "great", "praise": "好！"}')
    skill = ExpressionSkill(llm=fake)
    state = _session_with_teaching()
    assert await skill.judge("这道题怎么做？", state) is None
    assert fake.calls == 0  # 预筛未过不应调用 LLM


async def test_judge_hits_praise() -> None:
    reply = '{"is_answer": true, "correct": true, "level": "normal", "praise": "回答正确，真棒！"}'
    skill = ExpressionSkill(llm=_FakeLLM(reply))
    decision = await skill.judge("36", _session_with_teaching())
    assert decision is not None
    assert decision.kind == "praise"
    assert decision.expression == EXPRESSION_MATERIALS["praise"]
    assert decision.praise


async def test_judge_hits_celebrate() -> None:
    reply = '{"is_answer": true, "correct": true, "level": "great", "praise": "太厉害了！"}'
    skill = ExpressionSkill(llm=_FakeLLM(reply))
    decision = await skill.judge("36", _session_with_teaching())
    assert decision is not None
    assert decision.expression == EXPRESSION_MATERIALS["celebrate"]


async def test_judge_miss_returns_none() -> None:
    # 答错 / 非作答 / 非法回复都返回 None（走常规答疑链路）
    for reply in (
        '{"is_answer": true, "correct": false, "level": "normal", "praise": ""}',
        '{"is_answer": false, "correct": false, "level": "normal", "praise": ""}',
        "抱歉，我不确定。",
    ):
        skill = ExpressionSkill(llm=_FakeLLM(reply))
        assert await skill.judge("36", _session_with_teaching()) is None


async def test_judge_llm_error_returns_none() -> None:
    class _BoomLLM(BaseLLMClient):
        async def chat(
            self, messages: ChatMessages, *, temperature: float = 0.3, max_tokens: int = 1024
        ) -> str:
            raise RuntimeError("boom")

    skill = ExpressionSkill(llm=_BoomLLM())
    assert await skill.judge("36", _session_with_teaching()) is None
