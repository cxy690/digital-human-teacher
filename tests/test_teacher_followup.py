"""互动追问：followup 解析与作答判定单测。"""
from __future__ import annotations

import json

from agents.teacher.agent import TeacherAgent
from core import math_solver
from core.state.models import StudentTurn
from core.state.session import SessionState
from models.base import BaseLLMClient
from models.llm import MockLLM


class FakeLLM(BaseLLMClient):
    """答疑返回带 followup 的 JSON；判题返回可控结果。"""

    is_mock = True

    def __init__(self, followup: dict | None = None, judge: dict | None = None) -> None:
        self.followup = followup
        self.judge = judge

    async def chat(
        self, messages, *, temperature: float = 0.3, max_tokens: int = 1024, **kwargs
    ) -> str:
        """``**kwargs`` 吞掉 ``json_mode`` 之类的调用参数 —— 生产链路会传它们。"""
        joined = " ".join(str(m.get("content")) for m in messages)
        if "批改老师" in joined:
            return json.dumps(self.judge or {"correct": False, "feedback": "再想想", "hint": "看提示"})
        return json.dumps(
            {
                "speech": "我们来看这道题。",
                "title": "测试题",
                "steps": ["读题", "求解"],
                "knowledge_points": [],
                "pitfalls": [],
                "equation": "",
                "answer": "",
                "followup": self.followup,
            }
        )


async def test_answer_parses_followup() -> None:
    llm = FakeLLM(
        followup={"question": "2+3等于几？", "answer": "5", "hint": "掰手指数", "options": ["4", "5", "6"]}
    )
    teacher = TeacherAgent(llm=llm)
    state = SessionState(session_id="s1", student_id="x")
    resp = await teacher.answer(StudentTurn(text="2+3"), state)
    fq = resp.meta.get("followup")
    assert fq is not None
    assert fq["question"] == "2+3等于几？"
    assert fq["answer"] == "5"
    assert fq["options"] == ["4", "5", "6"]


async def test_answer_without_followup_returns_none() -> None:
    teacher = TeacherAgent(llm=FakeLLM(followup=None))
    state = SessionState(session_id="s2", student_id="x")
    resp = await teacher.answer(StudentTurn(text="2+3"), state)
    assert resp.meta.get("followup") is None


async def test_evaluate_answer_text_via_llm() -> None:
    teacher = TeacherAgent(llm=FakeLLM(judge={"correct": True, "feedback": "完全正确！", "hint": ""}))
    out = await teacher.evaluate_answer("水的沸点是多少？", "100℃", "100摄氏度")
    assert out["correct"] is True


async def test_evaluate_answer_fallback_without_json() -> None:
    # MockLLM 返回非 JSON 文本 -> 走归一化字符串兜底
    teacher = TeacherAgent(llm=MockLLM())
    out = await teacher.evaluate_answer("这是什么颜色？", "红色", "红色的")
    assert out["correct"] is True


async def test_evaluate_answer_equation_uses_sympy() -> None:
    if not math_solver.available():
        return  # sympy 缺失时跳过（整体降级为 LLM/字符串判定）
    teacher = TeacherAgent(llm=MockLLM())
    ok = await teacher.evaluate_answer("解方程 3x+5=20", "x=5", "x=5")
    bad = await teacher.evaluate_answer("解方程 3x+5=20", "x=5", "x=4")
    assert ok["correct"] is True
    assert bad["correct"] is False
