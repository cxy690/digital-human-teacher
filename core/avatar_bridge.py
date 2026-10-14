"""数字人播报桥接：把 AI 教师的 speech 文本推给数字人服务驱动口型。

对接 digital_human 自研服务（MuseTalk 推理核心 + 自研 WebRTC 服务层）：
- 播报：POST /api/echo {"text", "voice", "expression"}（服务内部 edge-tts 合成 + 口型渲染；
  expression 为表情素材目录名，可空）
- 说话状态：GET /api/status → {"speaking": bool}
- 打断：POST /api/interrupt（立即闭嘴，并清空未开始的排队播报）

要点：
- 音频由数字人服务合成并随 WebRTC 音轨下发，教育后端不落音频；
- say() 送 TTS 前做符号口语化（×→乘、²→的平方…），仅影响朗读，页面显示不变；
- 服务未启动时全部静默跳过（返回 False），不阻塞答疑主链路。
"""
from __future__ import annotations

import re

import httpx

from config.logging import get_logger
from config.settings import settings

log = get_logger("core.avatar_bridge")

# 数学符号口语化（与"听到的=看到的"配套：显示保符号，朗读保通顺）
_TTS_SYMBOLS = {
    "×": "乘", "÷": "除以", "²": "的平方", "³": "的立方",
    "π": "派", "≈": "约等于", "≥": "大于等于", "≤": "小于等于",
    "√": "根号", "°": "度", "+": "加", "=": "等于",
}
_TTS_REGEX = re.compile("|".join(map(re.escape, _TTS_SYMBOLS)))


def for_tts(text: str) -> str:
    """符号口语化（数字人朗读用；其余字符原样保留）。"""
    return _TTS_REGEX.sub(lambda m: _TTS_SYMBOLS[m.group()], text)


class AvatarBridge:
    """数字人服务 HTTP 桥接（无状态，全局共用一个实例）。"""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=settings.avatar_api_base, timeout=5.0)

    async def say(
        self,
        text: str,
        *,
        interrupt: bool = True,
        expression: str | None = None,
        voice: str | None = None,
        avatar: str | None = None,
    ) -> bool:
        """驱动数字人播报一段文本。

        interrupt=True 先打断旧播报（回答语义）；
        expression 指定表情素材目录名（如 ai3_2），None=当前激活素材；
        voice 指定音色（按学科匹配教师形象时使用），None=全局默认音色；
        avatar 指定本次播报要切换到的激活素材（如数学 ai4），None=沿用当前激活素材。
        """
        spoken = for_tts(text).strip()
        if not spoken:
            return False
        try:
            if interrupt:
                await self.interrupt()
            payload: dict[str, str] = {"text": spoken, "voice": voice or settings.avatar_tts_voice}
            if expression:
                payload["expression"] = expression
            if avatar:
                payload["avatar"] = avatar
            resp = await self._client.post("/api/echo", json=payload)
            if resp.status_code != 200:
                log.debug("avatar say rejected: {} {}", resp.status_code, resp.text[:120])
                return False
            return True
        except Exception as err:  # noqa: BLE001 - 数字人未启动不阻塞主链路
            log.debug("avatar say skipped: {}", err)
            return False

    async def interrupt(self) -> None:
        """立即打断当前播报（学生重新提问时调用）。"""
        try:
            await self._client.post("/api/interrupt")
        except Exception as err:  # noqa: BLE001
            log.debug("avatar interrupt skipped: {}", err)

    async def set_avatar(self, avatar_id: str | None) -> bool:
        """切换数字人「激活素材」（待机动画与默认播报同源）。

        按学生所选科目下发：数学 → ai4，其余科目沿用默认素材。切换成功后服务端的待机帧
        也随之换成该素材（避免"待机是 A 老师、开口是 B 老师"）。数字人未启动或素材缺失时
        返回 False，服务端保持原素材，不影响答疑主链路。
        """
        name = (avatar_id or "").strip()
        if not name:
            return False
        try:
            resp = await self._client.post("/api/avatar", json={"avatar_id": name})
            if resp.status_code != 200:
                log.debug("avatar switch rejected: {} {}", resp.status_code, resp.text[:120])
                return False
            return True
        except Exception as err:  # noqa: BLE001 - 数字人未启动不阻塞主链路
            log.debug("avatar switch skipped: {}", err)
            return False

    async def is_speaking(self) -> bool:
        """数字人当前是否正在播音（音频实际在播）。

        取 ``playing``（口型/音频真正开播），而非 ``speaking``（含 TTS 合成、特征提取、
        预滚等准备阶段）——准备阶段画面仍是待机动画，按 speaking 判定会误认为已在说话。
        旧版服务端未下发 playing 时回落到 speaking，保持兼容。
        """
        try:
            resp = await self._client.get("/api/status")
            data = resp.json()
            return bool(data.get("playing", data.get("speaking")))
        except Exception:  # noqa: BLE001
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


_bridge: AvatarBridge | None = None


def get_bridge() -> AvatarBridge:
    """进程级单例（懒加载，首次调用时创建）。"""
    global _bridge
    if _bridge is None:
        _bridge = AvatarBridge()
    return _bridge
