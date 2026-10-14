"""数字人引擎：把 TTS 音频渲染成 AI 教师形象视频（1对1 实时答疑的口型驱动）。

提供可插拔的实时口型 backend（``get_avatar``，低延迟逐句渲染）：

* ``musetalk``：MuseTalk 1.5（首选，MIT，V100 级 30fps+ 实时）；
* ``latentsync``：LatentSync（备选，Apache-2.0，中文唇形对齐强，显存≈6.5GB）；
* ``http``：通用 ``POST {api_base}/render`` 渲染服务（如 Wav2Lip 兜底）；
* ``mock``（默认）：服务端不渲染，前端走「照片木偶」模式。

设计要点：
* 逐句渲染 —— 配合 ``BaseTTSClient.synthesize_stream``，首句音频一出即渲染推流；
* **返回的视频必须自带音轨**：视频模式下前端只播视频、不再单独播 TTS 音频；
  若引擎只能返回无声视频，把 ``AVATAR_VIDEO_HAS_AUDIO`` 置 false，
  Supervisor 会额外下发 ``ai_audio`` 补声音；
* 渲染失败不阻断播报 —— 调用方（Supervisor）捕获异常后回退照片木偶。
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from config.logging import get_logger
from config.settings import settings
from models.base import BaseAvatarClient

log = get_logger("models.avatar")


class MockAvatar(BaseAvatarClient):
    """离线 Mock：不渲染视频，告知前端使用照片木偶模式驱动口型与眨眼。"""

    is_mock = True
    produces_video = False

    async def render(
        self, audio: bytes, *, text: str = "", gesture: str | None = None,
        teacher_id: str | None = None,
    ) -> dict[str, Any]:
        return {"mode": "puppet", "text": text}


class PlaceholderAvatar(BaseAvatarClient):
    """占位数字人后端（Mock 降级）：静态形象图 + 音频 -> 合成带音轨 mp4。

    用途：MuseTalk 不可用时仍让下游拿到 ``mode:video``（保持契约一致），
    画面为静态人像（不说话/不动），仅保证“有画面、有声音”。
    ffmpeg 或形象图缺失时回退为 ``mode:puppet``（前端照片木偶），不阻断播报。

    形象图来源：``AVATAR_SOURCE_IMAGE``（显式指定）优先；否则用 ffmpeg 从
    ``fallback_video``（AI 教师底片）抽首帧作为静态图。
    """

    is_mock = True
    produces_video = True

    def __init__(self, image_path: str = "", fallback_video: str = "", timeout: float = 30.0) -> None:
        self.image_path = image_path
        self.fallback_video = fallback_video
        self.timeout = timeout
        # 惰性解析结果：None=未解析，""=解析失败，其他=可用图路径
        self._still: str | None = None

    @staticmethod
    def _resolve_ffmpeg() -> str | None:
        """定位 ffmpeg：FFMPEG_PATH -> imageio_ffmpeg -> PATH 中的 ffmpeg。"""
        env = os.getenv("FFMPEG_PATH", "").strip()
        if env and Path(env).is_file():
            return env
        try:
            import imageio_ffmpeg  # noqa: PLC0415 - 可选依赖

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001 - 可选依赖缺失
            pass
        return shutil.which("ffmpeg")

    def _resolve_still(self, ffmpeg: str | None) -> str | None:
        """解析静态形象图：显式图 > 从底片抽首帧；均不可用返回 None。"""
        if self._still is not None:
            return self._still or None
        if self.image_path and Path(self.image_path).is_file():
            self._still = self.image_path
            return self._still
        if ffmpeg and self.fallback_video and Path(self.fallback_video).is_file():
            out = Path(tempfile.mkdtemp(prefix="ph_still_")) / "still.png"
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", self.fallback_video, "-frames:v", "1", str(out)],
                    check=True,
                    capture_output=True,
                    timeout=self.timeout,
                )
                if out.is_file():
                    self._still = str(out)
                    return self._still
            except Exception as err:  # noqa: BLE001 - 抽帧失败则回退木偶
                log.warning("placeholder 抽首帧失败：{}", err)
        self._still = ""
        return None

    async def render(
        self, audio: bytes, *, text: str = "", gesture: str | None = None,
        teacher_id: str | None = None,
    ) -> dict[str, Any]:
        """静态图 + 音频 -> 带音轨 mp4；任何依赖缺失/异常均回退 ``mode:puppet``。"""
        if not audio:
            return {"mode": "puppet", "text": text}
        ffmpeg = self._resolve_ffmpeg()
        still = self._resolve_still(ffmpeg)
        if not ffmpeg or not still:
            return {"mode": "puppet", "text": text}
        try:
            tmp = tempfile.mkdtemp(prefix="ph_")
            in_audio = os.path.join(tmp, "in.bin")
            Path(in_audio).write_bytes(audio)
            out = os.path.join(tmp, "out.mp4")
            subprocess.run(
                [
                    ffmpeg, "-y",
                    "-loop", "1", "-i", still,
                    "-i", in_audio,
                    "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-r", "24",
                    "-c:a", "aac", "-shortest", "-movflags", "+faststart",
                    out,
                ],
                check=True,
                capture_output=True,
                timeout=self.timeout,
            )
            data = Path(out).read_bytes()
            if not data:
                return {"mode": "puppet", "text": text}
            return {
                "mode": "video",
                "video": base64.b64encode(data).decode("ascii"),
                "mime": "video/mp4",
                "text": text,
                "placeholder": True,
            }
        except Exception as err:  # noqa: BLE001 - 渲染失败回退木偶
            log.warning("placeholder avatar render failed, fallback to puppet: {}", err)
            return {"mode": "puppet", "text": text}


class HttpAvatar(BaseAvatarClient):
    """HTTP 数字人渲染服务基类（MuseTalk / metahuman-stream / 离线渲染类）。

    子类通过覆写 :attr:`render_path` 与 :meth:`_payload` 适配不同引擎的接口形态，
    返回结果统一归一化为 ``{"mode": "video", "video": <b64> | "stream_url": ...}``。
    """

    produces_video = True
    render_path = "/render"

    def __init__(
        self,
        api_base: str,
        api_key: str = "",
        timeout: float = 60.0,
        source_image: str = "",
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.source_image = source_image

    def _payload(self, audio: bytes, text: str, gesture: str | None = None,
                 teacher_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "audio": base64.b64encode(audio).decode("ascii"),
            "text": text,
        }
        if self.source_image:
            payload["image"] = self.source_image
        # 手势基元名（仅在指定时下发；不支持的引擎会忽略该字段）
        if gesture:
            payload["gesture"] = gesture
        # 多老师：教师标识（= 学科 key），引擎据此切换底片
        if teacher_id:
            payload["teacher_id"] = teacher_id
        return payload

    async def render(
        self, audio: bytes, *, text: str = "", gesture: str | None = None,
        teacher_id: str | None = None,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}{self.render_path}",
                json=self._payload(audio, text, gesture, teacher_id),
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        result: dict[str, Any] = {"mode": "video", "text": text}
        if data.get("stream_url"):
            result["stream_url"] = data["stream_url"]
        else:
            result["video"] = data.get("video", "")
            result["mime"] = data.get("mime", "video/mp4")
        return result


class MuseTalkAvatar(HttpAvatar):
    """MuseTalk 1.5（L3 实时口型首选）：消费级/专业卡实时逐句渲染。

    参数：
    * ``fps``：渲染帧率，**必须与底片视频实际 fps 一致**（默认 24，与
      ``scripts/avatar_engine_musetalk_v15.py`` 的底片兜底值对齐）。传 30 而底片是 24 时，
      whisper 特征仍按 24fps 切块，但输出 mp4 容器按 30fps 播放，会出现音画不同步/画面加速。
    * ``bbox_shift``：人脸框上下平移量（正下负上，官方推荐 [-10, 10]），底片脸型偏长
      或下巴被切时用负值（如 -5）；不传则用引擎侧的 ``MUSETALK_BBOX_SHIFT`` 环境变量默认值。

    手势：通过 ``render(..., gesture="nod")`` 逐句下发（nod/shake/shrug/point），引擎会
    交叉淡化切到对应动作基元再淡回主底片；不传则纯讲解。
    """

    render_path = "/render"

    def __init__(
        self,
        api_base: str,
        api_key: str = "",
        timeout: float = 60.0,
        fps: int = 24,
        bbox_shift: int | None = None,
    ) -> None:
        super().__init__(api_base, api_key, timeout)
        self.fps = fps
        self.bbox_shift = bbox_shift

    def _payload(self, audio: bytes, text: str, gesture: str | None = None,
                 teacher_id: str | None = None) -> dict[str, Any]:
        payload = super()._payload(audio, text, gesture, teacher_id)
        payload["model"] = "musetalk-1.5"
        payload["fps"] = self.fps
        # None 表示沿用引擎侧默认（MUSETALK_BBOX_SHIFT 环境变量），不下发字段
        if self.bbox_shift is not None:
            payload["bbox_shift"] = self.bbox_shift
        return payload


class LatentSyncAvatar(HttpAvatar):
    """LatentSync（实时口型备选）：中文唇形对齐能力强，显存≈6.5GB。"""

    render_path = "/render"

    def _payload(self, audio: bytes, text: str, gesture: str | None = None) -> dict[str, Any]:
        payload = super()._payload(audio, text, gesture)
        payload["model"] = "latentsync"
        return payload


@lru_cache
def get_avatar() -> BaseAvatarClient:
    """L3 实时口型引擎工厂；未配置时回退 Mock（前端照片木偶）。"""
    backend = (settings.avatar_backend or "mock").lower()
    if backend == "musetalk" and settings.avatar_api_base:
        log.info(
            "use MuseTalk 1.5 realtime avatar: {} fps={} bbox_shift={}",
            settings.avatar_api_base, settings.avatar_fps, settings.avatar_bbox_shift,
        )
        return MuseTalkAvatar(
            settings.avatar_api_base,
            settings.avatar_api_key,
            settings.avatar_timeout_seconds,
            fps=settings.avatar_fps,
            bbox_shift=settings.avatar_bbox_shift,
        )
    if backend == "latentsync" and settings.avatar_api_base:
        log.info("use LatentSync realtime avatar: {}", settings.avatar_api_base)
        return LatentSyncAvatar(settings.avatar_api_base, settings.avatar_api_key, settings.avatar_timeout_seconds)
    if backend in ("http",) and settings.avatar_api_base:
        log.info("use HTTP avatar renderer: {}", settings.avatar_api_base)
        return HttpAvatar(
            settings.avatar_api_base,
            settings.avatar_api_key,
            settings.avatar_timeout_seconds,
            source_image=settings.avatar_source_image,
        )
    if backend == "placeholder":
        log.info("use Placeholder avatar (static image + audio)")
        return PlaceholderAvatar(
            image_path=settings.avatar_source_image,
            fallback_video=settings.ai_teacher_video_path,
        )
    if backend not in ("mock", ""):
        log.warning("avatar backend {} not fully configured, fallback to MockAvatar", backend)
    return MockAvatar()
