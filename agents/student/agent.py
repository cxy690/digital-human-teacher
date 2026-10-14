"""学生端代理：代表学生侧的设备状态与输入，驱动麦控、生成学生回合。

对应 PRD 学生端功能：音视频通道开关、语音连麦、文字提问、上传题目图片。
"""
from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from agents.shared.mic_control import MicController
from config.constants import IntentType, MicState
from core.state.models import StudentTurn
from core.state.session import SessionState


class StudentAgent(BaseAgent):
    """学生端智能体（持有本会话的麦控状态机）。"""

    role = "student"

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.mic = MicController()
        self.camera_on = True
        self.microphone_on = True

    async def on_session_start(self, session: SessionState) -> None:
        session.push_status(camera_on=self.camera_on, mic_on=self.microphone_on)

    # ---------------------------------------------------------------- 设备
    def set_camera(self, on: bool, session: SessionState) -> None:
        self.camera_on = on
        session.push_status(camera_on=on)
        self.log.info("camera {}", "on" if on else "off")

    def set_microphone(self, on: bool, session: SessionState) -> None:
        self.microphone_on = on
        session.push_status(mic_on=on)
        self.log.info("microphone {}", "on" if on else "off")

    # ---------------------------------------------------------------- 麦控
    def on_vad(self, has_voice: bool, ts_ms: int, session: SessionState) -> MicState:
        """驱动麦控状态机并同步到会话状态。"""
        state = self.mic.on_voice_frame(has_voice, ts_ms)
        session.mic_state = state
        session.student_speaking = self.mic.student_speaking
        session.push_status(mic_state=state.value)
        return state

    # ---------------------------------------------------------------- 回合
    # 客户端是**不可信输入源**：旧前端、第三方客户端、误触发送都可能送来
    # ``text: null`` / 缺字段。``StudentTurn.text`` 声明为 ``str``，
    # 一旦真的塞进 None，后面 ``ChatMessage(text=None)`` 会抛 pydantic
    # ValidationError，整条 WebSocket 会话直接断掉。
    # 所以在**构造边界**就把脏值归一成空串，让"脏输入"止步于此。
    @staticmethod
    def _clean(text: Any) -> str:
        return text if isinstance(text, str) else ("" if text is None else str(text))

    def build_turn_from_voice(self, transcript: str) -> StudentTurn:
        return StudentTurn(
            intent=IntentType.VOICE_QUESTION, text=self._clean(transcript), meta={"from_voice": True}
        )

    def build_turn_from_text(self, text: str) -> StudentTurn:
        return StudentTurn(intent=IntentType.TEXT_QUESTION, text=self._clean(text))

    def build_turn_from_image(self, image_id: str, text: str = "") -> StudentTurn:
        return StudentTurn(
            intent=IntentType.IMAGE_QUESTION, text=self._clean(text), image_id=image_id
        )
