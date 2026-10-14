"""智能麦控：基于 VAD 的对话状态机，保证「学生说话 AI 静默、学生静默 AI 答疑」。

对应 PRD：
* 学生开口提问 / 说话互动时，AI 自动静默倾听，不抢占人声、无声音重叠；
* 学生停止提问、进入语音静默后，AI 自动触发语音答疑；
* 学生中途插话（barge-in）时，AI 立即让出话语权。
"""
from __future__ import annotations

from config import constants as C
from config.constants import MicState
from config.logging import get_logger

log = get_logger("agents.mic_control")


class MicController:
    """单会话麦控状态机（由音频 VAD 帧驱动，逐帧 ``on_voice_frame``）。"""

    def __init__(self) -> None:
        self.state: MicState = MicState.IDLE
        self.ai_speaking: bool = False
        self._voice_start_ms: int | None = None
        self._last_voice_ms: int = 0
        self._barge_in_ms: int = 0

    # ------------------------------------------------------------------ 输入
    def on_voice_frame(self, has_voice: bool, ts_ms: int) -> MicState:
        """喂入一帧 VAD 结果，推进状态机并返回当前状态。"""
        if has_voice:
            if self._voice_start_ms is None:
                self._voice_start_ms = ts_ms
            self._last_voice_ms = ts_ms
            speech_ms = ts_ms - self._voice_start_ms

            if self.ai_speaking:
                # 学生插话：超过最短语音长度即判定抢话，AI 需让出话语权
                if speech_ms >= C.VAD_SPEECH_MS:
                    self.state = MicState.BARGED_IN
                    self._barge_in_ms = ts_ms
            elif speech_ms >= C.VAD_SPEECH_MS:
                self.state = MicState.STUDENT_SPEAKING
            return self.state

        # 无语音
        self._voice_start_ms = None
        if self.state in (MicState.STUDENT_SPEAKING, MicState.IDLE):
            silence_ms = ts_ms - self._last_voice_ms
            if self._last_voice_ms and silence_ms >= C.VAD_SILENCE_MS:
                # 学生说完，进入静默：此时才允许 AI 触发答疑
                self.state = MicState.STUDENT_SILENT
        return self.state

    def begin_ai_speak(self, ts_ms: int) -> MicState:
        """AI 开始讲解。"""
        self.ai_speaking = True
        self.state = MicState.AI_SPEAKING
        return self.state

    def end_ai_speak(self, ts_ms: int) -> MicState:
        """AI 讲解结束；若之前被插话，则回到学生说话态。"""
        self.ai_speaking = False
        if self.state is MicState.BARGED_IN:
            self.state = MicState.STUDENT_SPEAKING
        else:
            self.state = MicState.STUDENT_SILENT
        return self.state

    # ------------------------------------------------------------------ 判定
    @property
    def student_speaking(self) -> bool:
        return self.state in (MicState.STUDENT_SPEAKING, MicState.BARGED_IN)

    def can_ai_respond(self) -> bool:
        """AI 是否可以开口答疑：学生已静默且 AI 未在讲话。"""
        return self.state is MicState.STUDENT_SILENT and not self.ai_speaking

    def should_yield(self, ts_ms: int | None = None) -> bool:
        """AI 是否应立即停止播报（被学生插话）。"""
        if self.state is not MicState.BARGED_IN:
            return False
        if ts_ms is None:
            return True
        return ts_ms - self._barge_in_ms >= C.BARGE_IN_GRACE_MS
