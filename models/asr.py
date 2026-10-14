"""ASR 客户端：把学生语音转成文本。默认对接 OpenAI 兼容 ``/audio/transcriptions``。"""
from __future__ import annotations

from functools import lru_cache

import httpx

from config.logging import get_logger
from config.settings import settings
from models.base import BaseASRClient

log = get_logger("models.asr")


class OpenAICompatASR(BaseASRClient):
    """OpenAI Whisper 兼容接口。"""

    def __init__(self, api_base: str, api_key: str, model: str, timeout: float = 20.0) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"file": ("audio.wav", audio, "audio/wav")}
        data = {"model": self.model}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}/audio/transcriptions", headers=headers, files=files, data=data
            )
            resp.raise_for_status()
        return resp.json().get("text", "")


class MockASR(BaseASRClient):
    """离线 Mock：返回占位文本（真实链路应接入流式 ASR + VAD）。"""

    is_mock = True

    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000) -> str:
        return "老师，这道题我不会，可以帮我讲讲吗？"


class FunASRClient(BaseASRClient):
    """FunASR 适配器（立项方案 3.3 指定）：中文课堂场景识别优于通用模型。

    对接 FunASR runtime 的 HTTP 网关（约定 ``POST {api_base}/recognize``，
    入参音频字节 + 模型名 + 热词，返回 ``{"text": ...}``）。
    流式识别与 VAD 打断检测由 FunASR runtime 侧提供，本客户端负责整段识别入口。
    """

    def __init__(
        self,
        api_base: str,
        *,
        model: str = "paraformer-zh",
        hotwords: str = "",
        api_key: str = "",
        timeout: float = 20.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.hotwords = hotwords
        self.api_key = api_key
        self.timeout = timeout

    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000) -> str:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        files = {"file": ("audio.wav", audio, "audio/wav")}
        data = {"model": self.model, "sample_rate": str(sample_rate)}
        if self.hotwords:
            data["hotwords"] = self.hotwords
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.api_base}/recognize", headers=headers, files=files, data=data)
            resp.raise_for_status()
        return resp.json().get("text", "")


@lru_cache
def get_asr() -> BaseASRClient:
    """按 ``asr_provider`` 选择 ASR backend；未配置时降级为 Mock。"""
    provider = (settings.asr_provider or "openai").lower()
    if provider == "funasr" and settings.asr_api_base:
        log.info("use FunASR: {}", settings.funasr_model)
        return FunASRClient(
            settings.asr_api_base,
            model=settings.funasr_model,
            hotwords=settings.funasr_hotwords,
            api_key=settings.asr_api_key,
        )
    if provider != "mock" and settings.asr_api_key and settings.asr_api_base:
        log.info("use OpenAI-compatible ASR: {}", settings.asr_model)
        return OpenAICompatASR(settings.asr_api_base, settings.asr_api_key, settings.asr_model)
    if provider not in ("mock", ""):
        log.warning("asr provider {} not fully configured, fallback to MockASR", provider)
    else:
        log.warning("no asr config, fallback to MockASR")
    return MockASR()
