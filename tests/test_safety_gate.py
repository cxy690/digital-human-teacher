"""P1 安全网关测试：三级处置口径、旁路收口、危机专线。

这一组测试的重点不是「有没有过滤」，而是三个**容易做错**的地方：

1. **误伤**：接线后每段 AI 话术都会过词库，裸「滚」会把物理课的「滚动摩擦」
   打成「**动摩擦」；「赌博/毒品/暴力」在道法、生物、健康教育里是正当教学内容。
2. **旁路**：流式话术与白板文字都会**绕过 _speak** 直接下行，必须各有收口。
3. **处置方式**：危机信号不能「打码后照常讲课」，要走专线且话术自身幂等。
"""
from __future__ import annotations

import pytest

from config.constants import BoardOp
from config.settings import settings
from core.safety_gate import SafetyGate
from core.guardrail import TeachingGuardrail
from core.safety_audit import SafetyAudit
from core.state.models import AIResponse, StudentTurn, WhiteboardOp
from core.state.session import SessionState
from agents.supervisor import Supervisor
from models.tts import MockTTS

CRISIS_TEXT = "老师，我不想活了，觉得活着没意思"


# ============================================================ guardrail 三级口径
def test_crisis_detected() -> None:
    verdict = TeachingGuardrail().screen_crisis(CRISIS_TEXT)
    assert verdict.crisis
    assert verdict.allowed is False
    assert verdict.speech == TeachingGuardrail.CRISIS_SPEECH


def test_crisis_speech_is_idempotent() -> None:
    """关怀话术自身不能命中任何一级词表，否则会被自己过滤掉。"""
    safe, verdict = TeachingGuardrail().filter_output(TeachingGuardrail.CRISIS_SPEECH)
    assert safe == TeachingGuardrail.CRISIS_SPEECH
    assert not verdict.hit


def test_rolling_is_not_masked() -> None:
    """误伤回归：物理课的「滚动摩擦」不能被打成「**动摩擦」。"""
    text = "小球在斜面上滚动，此时摩擦力的方向是怎样的？"
    safe, verdict = TeachingGuardrail().filter_output(text)
    assert safe == text
    assert not verdict.hit


def test_sensitive_teaching_content_is_flagged_not_masked() -> None:
    """道法课讲「赌博的危害」是正当教学内容，只标记不打码。"""
    text = "赌博会让家庭倾家荡产，我们要远离它。"
    safe, verdict = TeachingGuardrail().filter_output(text)
    assert safe == text, "敏感教学内容不能被遮蔽"
    assert "赌博" in verdict.flagged


def test_abuse_is_masked() -> None:
    safe, verdict = TeachingGuardrail().filter_output("你怎么这么笨蛋，去死吧")
    assert "笨蛋" not in safe
    assert "去死" not in safe
    assert verdict.unsafe
    assert verdict.allowed is False


def test_crisis_replaces_text_instead_of_masking() -> None:
    """危机内容不遮蔽，而是整段顶替为关怀话术。"""
    safe, verdict = TeachingGuardrail().filter_output("我想自杀")
    assert safe == TeachingGuardrail.CRISIS_SPEECH
    assert verdict.crisis


# ============================================================ SafetyGate 对象形态
def test_screen_student_text_reports_crisis() -> None:
    seen: list[dict] = []
    gate = SafetyGate(reporter=seen.append)
    verdict = gate.screen_student_text(CRISIS_TEXT)
    assert verdict.crisis
    assert seen and seen[0]["kind"] == "crisis"
    assert seen[0]["source"] == "student"
    assert seen[0]["text"] == CRISIS_TEXT, "上报必须带原文，否则教师端只看到一个命中词"


def test_screen_student_text_keeps_normal_text_untouched() -> None:
    gate = SafetyGate()
    assert not gate.screen_student_text("这道方程怎么解？").hit


def test_screen_response_masks_speech_and_board() -> None:
    gate = SafetyGate()
    response = AIResponse(
        speech="你怎么这么笨蛋",
        board_ops=[WhiteboardOp(op=BoardOp.TEXT, text="笨蛋", x=0.1, y=0.1)],
    )
    verdict = gate.screen_response(response)
    assert "笨蛋" not in response.speech
    assert "笨蛋" not in response.board_ops[0].text
    assert verdict.unsafe
    assert response.meta["safety"]["unsafe"]


def test_screen_response_masks_steps_and_followup() -> None:
    """steps 与 followup 会被 _spoken_text 念出来，属同一话术通道。"""
    gate = SafetyGate()
    response = AIResponse(
        speech="我们来看这道题",
        meta={"steps": ["先把蠢货条件圈出来"], "followup": {"question": "你是不是笨蛋？"}},
    )
    gate.screen_response(response)
    assert "蠢货" not in response.meta["steps"][0]
    assert "笨蛋" not in response.meta["followup"]["question"]


def test_screen_response_clean_text_untouched() -> None:
    gate = SafetyGate()
    response = AIResponse(speech="分数的分子和分母同时乘 2，大小不变。")
    verdict = gate.screen_response(response)
    assert response.speech == "分数的分子和分母同时乘 2，大小不变。"
    assert not verdict.hit
    assert "safety" not in response.meta, "未命中时不应写入 meta，避免污染产物"


def test_screen_response_crisis_clears_board() -> None:
    """危机命中：整段作废——板书与追问都要清掉，只留关怀话术。"""
    gate = SafetyGate()
    response = AIResponse(
        speech="我想自杀",
        board_ops=[WhiteboardOp(op=BoardOp.TEXT, text="例题", x=0.1, y=0.1)],
        meta={"steps": ["第一步"], "followup": {"question": "会了吗"}},
    )
    gate.screen_response(response)
    assert response.speech == SafetyGate.CRISIS_SPEECH
    assert response.board_ops == []
    assert "steps" not in response.meta
    assert "followup" not in response.meta


def test_screen_text_used_for_stream_and_board() -> None:
    gate = SafetyGate()
    assert "笨蛋" not in gate.screen_text("你这个笨蛋")
    assert gate.screen_text("小球沿斜面滚动") == "小球沿斜面滚动"


# ============================================================ 事件形态（未纳入本次融合）
# 云端另有 ``SafetyGate`` 的 A2A 事件形态（screen_outbound / screen_inbound），
# 它依赖 ``core/a2a`` 进程内总线；本仓库尚未引入 A2A 编排层，故这一组用例
# 随架构议题一起延后，见融合报告「未做的架构议题」。


# ============================================================ Supervisor 集成
async def _supervisor(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Supervisor:
    """离线 Supervisor：关掉数字人链路，并把审计写到临时文件（不污染 data/raw）。"""
    monkeypatch.setattr(settings, "digital_human_enable", False)
    state = SessionState(session_id="safety", student_id="stu")
    supervisor = Supervisor(state)
    supervisor.tts = MockTTS()
    supervisor.safety_audit = SafetyAudit(tmp_path / "safety.jsonl")
    await supervisor.on_start()
    return supervisor


async def test_crisis_turn_takes_the_caring_lane(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """危机回合：不答疑、不出板书，只回关怀话术并上报。"""
    supervisor = await _supervisor(monkeypatch, tmp_path)
    response = await supervisor.handle_turn(StudentTurn(text=CRISIS_TEXT))

    assert response is not None
    assert response.speech == SafetyGate.CRISIS_SPEECH
    assert response.board_ops == [], "危机回合不应有任何板书"
    assert supervisor.session.board_ops == [], "白板通道也不应被写入"
    assert supervisor.session.teaching is False
    assert supervisor.session.ai_speaking is False


async def test_speak_masks_abuse_even_on_the_normal_lane(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """常规答疑的下行收口：即使模型输出了不当用语，也必须被遮蔽。"""
    supervisor = await _supervisor(monkeypatch, tmp_path)
    response = supervisor._speak(AIResponse(speech="你怎么这么笨蛋"))  # noqa: SLF001 - 直测下行收口
    assert "笨蛋" not in response.speech
    assert supervisor.session.history[-1].text == response.speech


async def test_push_board_is_a_choke_point(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """白板通道收口：绕过 _speak 的上板路径（流式答疑/理解先行）同样被过滤。"""
    supervisor = await _supervisor(monkeypatch, tmp_path)
    supervisor._push_board([  # noqa: SLF001 - 直测白板收口
        WhiteboardOp(op=BoardOp.TEXT, text="笨蛋", x=0.1, y=0.1)
    ])
    assert "笨蛋" not in supervisor.session.board_ops[-1].text


async def test_normal_turn_still_passes_through(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """回归：加了网关之后，正常答疑链路不受影响。"""
    supervisor = await _supervisor(monkeypatch, tmp_path)
    response = await supervisor.handle_turn(StudentTurn(text="老师，这道方程题怎么解？"))
    assert response is not None
    assert response.speech
    assert supervisor.session.teaching is False


async def test_crisis_turn_writes_audit_entry(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """危机事件必须落盘：push_event 只到当前连接，没人在线就等于没上报。"""
    supervisor = await _supervisor(monkeypatch, tmp_path)
    await supervisor.handle_turn(StudentTurn(text=CRISIS_TEXT))

    entries = supervisor.safety_audit.recent(10)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["kind"] == "crisis"
    assert entry["source"] == "student"
    assert entry["hits"]
    assert entry["session_id"] == "safety"
    assert entry["student_id"] == "stu"
    # 原文必须留下：人工复核的第一件事就是「学生到底说了什么」
    assert entry["text"] == CRISIS_TEXT


# ============================================================ 危机词变体（P1.5）
@pytest.mark.parametrize(
    "text",
    [
        "我想自杀",
        "有时候真的想不开",
        "我觉得活着没意思",
        "真的撑不下去了",
        "没人在乎我",
        "一了百了算了",
        "我想伤害自己",
        "我好累，活够了",
    ],
)
def test_crisis_variants_detected(text: str) -> None:
    """补变体是为了提高召回——漏报的代价不可挽回。"""
    assert TeachingGuardrail().screen_crisis(text).crisis, text


@pytest.mark.parametrize(
    "text",
    [
        "老师我想死你了，好久不见",         # 「想死」+ 亲昵后缀
        "这道题想死我了，终于做出来了",     # 「想死」+ 抱怨后缀
        "下雨天不想活动，体育课改室内吧",   # 「不想活」+ 动
        "这道题没意思，换一道吧",           # 只有「没意思」，不成句
        "这道题我不会，想不通",             # 「想不通」≠「想不开」
        "这道题我终于做出来了，好解脱",     # 「解脱」不该被当成危机
    ],
)
def test_no_false_positive_on_teaching_talk(text: str) -> None:
    """误伤回归：这些是课堂里正常会出现的说法，一旦误报就会打断正常教学。"""
    assert not TeachingGuardrail().screen_crisis(text).crisis, text


def test_crisis_hits_are_human_readable() -> None:
    """``hits`` 会进审计文件并显示在教师端——不能漏出正则源码。"""
    verdict = TeachingGuardrail().screen_crisis("我不想活了")
    assert verdict.crisis == ["不想活"]
    for hit in verdict.crisis:
        assert not any(ch in hit for ch in r"()[]?!|\\"), f"命中标签不该含正则元字符：{hit}"
