"""智能麦控状态机单测。"""
from __future__ import annotations

from agents.shared.mic_control import MicController
from config.constants import MicState


def test_speech_then_silence_allows_ai() -> None:
    mc = MicController()
    assert mc.on_voice_frame(True, 0) is MicState.IDLE
    assert mc.on_voice_frame(True, 200) is MicState.STUDENT_SPEAKING
    assert not mc.can_ai_respond()
    # 静默 800ms（> 700ms 阈值）
    assert mc.on_voice_frame(False, 1000) is MicState.STUDENT_SILENT
    assert mc.can_ai_respond()


def test_barge_in_when_ai_speaking() -> None:
    mc = MicController()
    mc.begin_ai_speak(1000)
    assert mc.state is MicState.AI_SPEAKING
    mc.on_voice_frame(True, 1100)
    assert mc.on_voice_frame(True, 1300) is MicState.BARGED_IN
    assert mc.should_yield()
    # AI 让出话语权
    mc.end_ai_speak(1400)
    assert mc.state is MicState.STUDENT_SPEAKING


def test_ai_speaking_blocks_ai_response() -> None:
    mc = MicController()
    mc.begin_ai_speak(0)
    assert not mc.can_ai_respond()
