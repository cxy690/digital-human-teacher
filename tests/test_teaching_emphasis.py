"""讲解重点标红：契约约定 + 标记解析 + 「白板留着、语音摘掉」这条边界。

为什么值得钉住：标红靠的是**模型在文字里写 ``**…**`**，所以有两条容易静默失效的边：
1. 契约里没写清楚 -> 模型不标（或者把整句都标上，等于没重点）；
2. 标记漏进朗读文本 -> 语音把星号念出来，聊天记录里也留着两个符号。
"""
from __future__ import annotations

from agents.shared.emphasis import has_emphasis, split_emphasis, strip_emphasis
from agents.shared.teaching_layers import ANSWER_CONTRACT
from agents.supervisor import Supervisor
from agents.teacher.agent import TeacherAgent
from core.state.models import AIResponse
from core.state.session import SessionState
from models.llm import MockLLM


# ---------------------------------------------------------------- 标记解析
def test_strip_emphasis_removes_all_marks() -> None:
    """摘标记要连「未配对」的那种一起清掉（模型偶尔只写一半）。"""
    assert strip_emphasis("移项要**变号**，得到 **3x=15**。") == "移项要变号，得到 3x=15。"
    assert strip_emphasis("移项要**变号") == "移项要变号"
    assert strip_emphasis("没有标记") == "没有标记"


def test_split_emphasis_marks_which_segments_are_key() -> None:
    assert split_emphasis("移项要**变号**，得到 3x=15。") == [
        ("移项要", False),
        ("变号", True),
        ("，得到 3x=15。", False),
    ]
    assert has_emphasis("**重点**") is True
    assert has_emphasis("普通文字") is False


# ---------------------------------------------------------------- 契约不漂移
def test_contract_tells_model_what_to_mark() -> None:
    """契约必须写明「用什么标、标多少、哪里不要标」，否则模型只会随机加粗。"""
    assert "重点标红" in ANSWER_CONTRACT
    assert "双星号" in ANSWER_CONTRACT
    assert "每行最多标 1~2 处" in ANSWER_CONTRACT, "不限制数量 -> 整段飘红，等于没有重点"
    assert "不要**把整句、整个步骤都包起来" in ANSWER_CONTRACT.replace(" ——", " ——")


# ---------------------------------------------------------------- 白板 vs 语音
def test_board_keeps_marks_but_speech_drops_them() -> None:
    """板书步骤/知识点保留标记（前端据此标红），话术里必须没有星号。"""
    teacher = TeacherAgent(llm=MockLLM())
    reply = (
        '{"speech": "移项要**变号**，先把常数项移过去。",'
        ' "title": "用移项解方程",'
        ' "steps": ["两边同时减 5，得到 **3x=15**", "两边同时除以 3，得 **x=5**"],'
        ' "knowledge_points": ["移项法则：移项要**变号**"],'
        ' "pitfalls": ["忘记**变号**，整题都会错"],'
        ' "followup": {"question": "为什么移项要**变号**？", "answer": "等式两边同加减",'
        '              "hint": "想想等式性质", "options": []}}'
    )

    response = teacher._compose_response(reply)  # noqa: SLF001 - 直接验证产物组装

    assert response.speech and "**" not in response.speech, "话术是朗读文本，不能带标记"
    texts = [op.text or "" for op in response.board_ops]
    assert any("**3x=15**" in text for text in texts), "步骤里的重点必须原样保留"
    assert any("**变号**" in text for text in texts), "知识点/易错点同样要能标重点"


def test_spoken_text_strips_marks_from_steps() -> None:
    """朗读文本 = 话术 + 白板步骤 + 追问：三处都不许把星号念出来。"""
    state = SessionState(session_id="emp", student_id="s")
    supervisor = Supervisor(state)
    response = AIResponse(
        speech="先说结论。",
        meta={"steps": ["移项要**变号**"], "title": "t",
              "followup": {"question": "为什么**要变号**？"}},
    )

    spoken = supervisor._spoken_text(response)  # noqa: SLF001

    assert "变号" in spoken
    assert "**" not in spoken


def test_speak_strips_marks_from_chat_text() -> None:
    """聊天/字幕走的是 response.speech，模型误写标记时也要兜住。"""
    state = SessionState(session_id="emp2", student_id="s")
    events: list[dict] = []
    state.emit = events.append
    supervisor = Supervisor(state)

    supervisor._speak(AIResponse(speech="移项要**变号**。"))  # noqa: SLF001

    speech_events = [e for e in events if e["type"] == "ai_speech"]
    assert speech_events and "**" not in speech_events[-1]["payload"]["text"]
    assert state.history[-1].text == "移项要变号。"
