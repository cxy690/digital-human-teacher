"""TTS 客户端：把 AI 话术合成语音播放给学生。默认对接 OpenAI 兼容 ``/audio/speech``。"""
from __future__ import annotations

import asyncio
import re
from functools import lru_cache
from typing import AsyncIterator

import httpx

from config.logging import get_logger
from config.settings import settings
from models.base import BaseTTSClient

log = get_logger("models.tts")

_SENT_END = re.compile(r"[^。！？!?；;…\n]+[。！？!?；;…]*")

# 朗读前必须剔除的字符：TTS 会把这些当正文念出来，破坏讲课听感。
#   * 填空题的连续下划线（____）念出来像乱码或长停顿；
#   * 错误标记 ❌ / ✗ / ✘ 念出来会打断讲解（画面已用红叉表达）。
_RE_TTS_DROP = re.compile(r"[_＿]{2,}|[❌✗✘☓]")
_RE_TTS_SPACES = re.compile(r"[ \t]{2,}")
_RE_TTS_COMMAS = re.compile(r"[，,]{2,}")


def sanitize_for_tts(text: str) -> str:
    """清洗**待朗读**文本（只影响「念什么」，不改动画面上的文字）。

    * 填空题的连续下划线（``____``）不朗读——念出来像乱码或长停顿；
    * 错误标记 ``❌`` / ``✗`` / ``✘`` 不朗读——画面已用红叉表达；
      表示「错」请统一用 ``❌``，不要用 ``X`` / ``×``，否则容易被念成「乘」。

    乘号 ``×`` 与字母 ``x`` 保持原样：前者本就该念「乘」，后者是未知数。
    """
    if not text:
        return ""
    cleaned = _RE_TTS_DROP.sub("", text)
    cleaned = _RE_TTS_SPACES.sub(" ", cleaned)
    cleaned = _RE_TTS_COMMAS.sub("，", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """把话术按句切分（标点保留在句尾），供流式 TTS / 数字人逐句渲染。"""
    sentences = [m.group(0).strip() for m in _SENT_END.finditer(sanitize_for_tts(text))]
    return [s for s in sentences if s]


def _teacher_voice_ref(teacher_id: str | None) -> tuple[str, str, str]:
    """按老师取音色配置，返回 ``(edge 音色名, 参考干声路径, 参考文本)``。

    无档案 / 缺文件时返回空串，调用方回退到全局 TTS 配置（旧单老师行为）。
    """
    tid = (teacher_id or "").strip()
    if not tid:
        return "", "", ""
    try:
        from config.teachers import load_teacher  # 局部导入避免循环依赖

        profile = load_teacher(tid)
    except Exception:  # noqa: BLE001 - 档案异常不应阻断播报
        return "", "", ""
    if profile is None:
        return "", "", ""
    ref = profile.voice_ref_path          # 绝对路径：仅用于本地存在性校验
    # 传给引擎的用**档案原值**（通常是相对项目根路径）：云端 / SSH 隧道模式下，
    # 引擎在它自己的项目根下找同名文件（两端同构 data/teachers/<key>/...）才能命中；
    # 传本机绝对路径（E:\...）会让引擎端 os.path.isfile 失败而退化成内置音色。
    rel = profile.voice_ref_audio if ref else ""
    return profile.voice_name(), rel, profile.voice_ref_text


def _teacher_gender(teacher_id: str | None) -> str:
    """老师档案里的形象性别（``male`` / ``female`` / 空串）。

    用于「老师没有自己的干声」时挑选**同性别**的预置音色，避免落到
    全局 ``TTS_REF_AUDIO`` 那份固定性别的参考上（男老师因此发出女声）。
    """
    tid = (teacher_id or "").strip()
    if not tid:
        return ""
    try:
        from config.teachers import load_teacher  # 局部导入避免循环依赖

        profile = load_teacher(tid)
    except Exception:  # noqa: BLE001 - 档案异常不应阻断播报
        return ""
    return str(getattr(profile, "gender", "") or "").strip().lower()


def _tts_speed() -> float:
    """全局语速系数（``<1`` 更慢）。越界或非法值一律按 1.0（原速）处理。"""
    try:
        v = float(getattr(settings, "tts_speed", 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return v if 0.5 <= v <= 2.0 else 1.0


# 中文 TTS 的原速基线：约 4.2 字/秒（课件讲稿的「字数预算」用的也是这个数）。
_CHARS_PER_SEC = 4.2


def estimate_speech_seconds(text: str) -> float:
    """估算一段中文话术**在客户端真正播完**需要的秒数（已按全局语速折算）。

    用途：课件连讲时给「等这一页播完再翻页」一个**兜底上限**——
    真实播完以客户端上报的 ``slide_spoken`` 为准，但客户端掉线 / 旧前端不上报时
    绝不能把课堂卡死，所以用这个估算值当超时线。

    只是估算（±15% 量级）：零样本克隆会模仿参考干声的节奏，各老师快慢略有差异。
    """
    n = len(sanitize_for_tts(text or "").strip())
    if n <= 0:
        return 0.0
    return n / (_CHARS_PER_SEC * _tts_speed())


def _apply_speed(audio: bytes, speed: float) -> bytes:
    """对 TTS 音频做「**变速不变调**」（ffmpeg atempo）。

    为什么不用重采样：重采样会连带改变音高，放慢后老师会变得低沉发闷；
    ``atempo`` 只改时长、保音高，听感仍是同一位老师。

    同一段音频后续同时用于「数字人口型渲染」与「播放」，两者同源，
    所以变速不会造成音画不同步。

    失败（无 ffmpeg / 超时 / 解码异常）时**原样返回**，绝不因此中断播报。
    """
    if not audio or abs(speed - 1.0) < 1e-3:
        return audio
    import os
    import subprocess
    import tempfile

    from core.audio_utils import _resolve_ffmpeg  # 局部导入避免循环依赖

    ff = _resolve_ffmpeg()
    if not ff:
        log.warning("TTS 变速跳过：未找到 ffmpeg（speed={}）", speed)
        return audio
    ext = "wav" if audio[:4] == b"RIFF" else "mp3"
    try:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, f"in.{ext}")
            dst = os.path.join(td, f"out.{ext}")
            with open(src, "wb") as fh:
                fh.write(audio)
            # atempo 单次仅支持 0.5~2.0，超出则级联多级
            chain: list[str] = []
            s = speed
            while s < 0.5:
                chain.append("atempo=0.5")
                s /= 0.5
            while s > 2.0:
                chain.append("atempo=2.0")
                s /= 2.0
            chain.append(f"atempo={s:.4f}")
            subprocess.run(
                [ff, "-y", "-i", src, "-filter:a", ",".join(chain), dst],
                check=True, capture_output=True, timeout=30,
            )
            with open(dst, "rb") as fh:
                out = fh.read()
        return out or audio
    except Exception as err:  # noqa: BLE001 - 变速失败不能影响播报
        log.warning("TTS 变速失败（speed={}）：{}", speed, err)
        return audio


class OpenAICompatTTS(BaseTTSClient):
    """OpenAI TTS 兼容接口，返回音频字节流。"""

    def __init__(self, api_base: str, api_key: str, model: str, voice: str, timeout: float = 20.0) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.timeout = timeout

    async def synthesize(self, text: str, *, teacher_id: str | None = None) -> bytes:
        payload = {"model": self.model, "voice": self.voice, "input": text, "format": "mp3"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.api_base}/audio/speech", json=payload, headers=headers)
            resp.raise_for_status()
        return resp.content

    async def synthesize_stream(
        self, text: str, *, teacher_id: str | None = None
    ) -> AsyncIterator[tuple[bytes, str]]:
        """按句并发度为 1 的流式合成：首句一出即产出，降低数字人首帧延迟。"""
        for sentence in split_sentences(text):
            yield await self.synthesize(sentence, teacher_id=teacher_id), sentence


class EdgeTTS(BaseTTSClient):
    """微软 Edge 在线 TTS（edge-tts，免密钥）：单一中文女声，音色稳定。"""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%", volume: str = "+0%") -> None:
        self.voice = voice or "zh-CN-XiaoxiaoNeural"
        self.rate = rate
        self.volume = volume

    async def synthesize(self, text: str, *, teacher_id: str | None = None) -> bytes:
        import edge_tts

        voice, _ref, _txt = _teacher_voice_ref(teacher_id)   # 每位老师一个音色
        communicate = edge_tts.Communicate(text, voice or self.voice, rate=self.rate, volume=self.volume)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf += chunk["data"]
        return bytes(buf)


class MockTTS(BaseTTSClient):
    """离线 Mock：返回空音频占位；前端可退回浏览器 SpeechSynthesis 播报。"""

    is_mock = True

    async def synthesize(self, text: str, *, teacher_id: str | None = None) -> bytes:
        return b""

    async def synthesize_stream(
        self, text: str, *, teacher_id: str | None = None
    ) -> AsyncIterator[tuple[bytes, str]]:
        for sentence in split_sentences(text):
            yield b"", sentence


class CosyVoiceTTS(BaseTTSClient):
    """CosyVoice2 适配器（立项方案 3.3 首选）：流式推理、克隆授课老师音色。

    对接 CosyVoice2 推理服务（约定 ``POST {api_base}/api/v1/tts``，入参文本 + 参考干声/音色，
    返回音频字节或 ``{"audio": "<base64>"}``）。首包延迟目标 200-300ms，逐句流式合成。
    """

    def __init__(
        self,
        api_base: str,
        *,
        speaker: str = "",
        ref_audio: str = "",
        ref_text: str = "",
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.speaker = speaker
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.api_key = api_key
        self.timeout = timeout

    def _payload(self, text: str, teacher_id: str | None = None) -> dict:
        payload: dict = {"text": text, "stream": True}
        _voice, ref_audio, ref_text = _teacher_voice_ref(teacher_id)

        if not ref_audio:
            # 老师没有自己的干声：**按形象性别挑预置音色**，不要落到全局
            # TTS_REF_AUDIO —— 那份参考性别固定（女声），会让男老师发出女声。
            gender = _teacher_gender(teacher_id)
            if gender:
                payload["speaker"] = "中文男" if gender == "male" else "中文女"
                return payload

        # 该老师的克隆干声优先；没有则回退全局 TTS_REF_AUDIO
        audio = ref_audio or self.ref_audio
        text_ref = ref_text or self.ref_text
        if self.speaker and not audio:
            payload["speaker"] = self.speaker
        if audio:                               # zero-shot 克隆：参考干声 + 文本
            payload["ref_audio"] = audio
            payload["ref_text"] = text_ref
        return payload

    async def synthesize(self, text: str, *, teacher_id: str | None = None) -> bytes:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.api_base}/api/v1/tts",
                                     json=self._payload(text, teacher_id), headers=headers)
            resp.raise_for_status()
            return _audio_from_response(resp)


class GptSoVitsTTS(BaseTTSClient):
    """GPT-SoVITS 适配器（立项方案 3.3 备选）：少样本音色克隆。

    对接 GPT-SoVITS 官方推理 API（``POST {api_base}/tts``，入参 text/text_lang/
    ref_audio_path/prompt_text/prompt_lang 等，直接返回 wav 音频字节）。
    """

    def __init__(
        self,
        api_base: str,
        *,
        ref_audio: str = "",
        prompt_text: str = "",
        text_lang: str = "zh",
        prompt_lang: str = "zh",
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.ref_audio = ref_audio
        self.prompt_text = prompt_text
        self.text_lang = text_lang
        self.prompt_lang = prompt_lang
        self.api_key = api_key
        self.timeout = timeout

    def _payload(self, text: str, teacher_id: str | None = None) -> dict:
        payload: dict = {"text": text, "text_lang": self.text_lang, "media_type": "wav"}
        _voice, ref_audio, ref_text = _teacher_voice_ref(teacher_id)
        audio = ref_audio or self.ref_audio
        prompt = ref_text or self.prompt_text
        if audio:
            payload["ref_audio_path"] = audio
            payload["prompt_text"] = prompt
            payload["prompt_lang"] = self.prompt_lang
        return payload

    async def synthesize(self, text: str, *, teacher_id: str | None = None) -> bytes:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.api_base}/tts",
                                     json=self._payload(text, teacher_id), headers=headers)
            resp.raise_for_status()
            return _audio_from_response(resp)


def _parse_teacher_speakers(raw: str) -> dict[str, str]:
    """解析 ``DOUBAO_TEACHER_SPEAKERS``（形如 ``math=S_abc123,chinese=S_def456``）。

    分隔符同时接受中英文逗号/分号，方便手写。写坏的条目只告警跳过 ——
    一个配置写错不该让整堂课的语音起不来（与项目其它可选配置一致）。
    """
    out: dict[str, str] = {}
    for chunk in (raw or "").replace("；", ",").replace(";", ",").split(","):
        item = chunk.strip()
        if not item:
            continue
        key, sep, val = item.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not key or not val:
            log.warning("DOUBAO_TEACHER_SPEAKERS 条目非法已忽略：{}", item)
            continue
        out[key] = val
    return out


class DoubaoTS(BaseTTSClient):
    """豆包（火山引擎）大模型语音合成 · v3 HTTP 单向流式接口。

    以下每一条都是 2026-10 实测确认过的，**不要按"文档印象"改**：

    * 端点：``POST {api_base}/api/v3/tts/unidirectional``。
    * 鉴权：``X-Api-Key``（语音控制台「API Key 管理」创建的那串，新版控制台仅需 key）。
      **不是** AppID + Access Token —— 那套在这条链路上恒定 401
      ``load grant: requested grant not found in SaaS storage``。
    * ``X-Api-Resource-Id``：``volc.service_type.10029``（大模型语音合成，实测可用）；
      复刻音色走 ``seed-icl-2.0``，且 speaker 必须是 ``S_`` 开头的复刻音色，
      否则 500 ``resource ID is mismatched with speaker related resource``。
    * ``X-Api-Request-Id``（uuid）**必带**，缺了直接 400 路由失败。
    * 响应是**换行分隔的 JSON 流**，每行 ``{"code":0,"data":"<base64 mp3 分块>"}``；
      每个分块都是**独立可解码**的 base64，逐块解码后拼接字节即可
      （实测「逐块解码拼接」与「整体拼接后解码」字节数完全一致）。
    * **文本必须含句号「。」**，否则服务端判 45002001 ``No readable text!`` 不合成。
      实测 ``答对了！`` / ``对！`` / ``测试`` 全被拒，补句号后立刻正常出声 ——
      课堂里"答对了！"这类短反馈正好会踩中，故由 :meth:`_readable_text` 兜底。

    失败处理遵循本项目既有原则：**宁可这一句没声音，也不偷偷换音色**
    （音色是学生识别"同一位老师"最直接的线索），所以这里只记日志、返回已拿到的音频。
    """

    def __init__(
        self,
        api_key: str,
        *,
        api_base: str = "https://openspeech.bytedance.com",
        resource_id: str = "volc.service_type.10029",
        clone_resource_id: str = "seed-icl-2.0",
        speaker: str = "zh_female_shuangkuaisisi_moon_bigtts",
        teacher_speakers: dict[str, str] | None = None,
        audio_format: str = "mp3",
        sample_rate: int = 24000,
        uid: str = "education",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.resource_id = resource_id
        self.clone_resource_id = clone_resource_id
        self.speaker = speaker
        # 老师 -> 复刻音色 ID（``S_xxxxxx``）。没配的老师回落到全局 speaker。
        self.teacher_speakers = {
            str(k).strip(): str(v).strip()
            for k, v in (teacher_speakers or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.uid = uid
        self.timeout = timeout

    @staticmethod
    def _is_clone_speaker(speaker: str) -> bool:
        """复刻音色 ID 一律以 ``S_`` 开头。

        为什么要靠前缀判断：**预置音色与复刻音色走的是两个不同资源位**
        （``volc.service_type.10029`` vs ``seed-icl-2.0``），实测混用会直接
        500 ``resource ID is mismatched with speaker related resource``。
        """
        return (speaker or "").strip().upper().startswith("S_")

    def _voice_for(self, teacher_id: str | None) -> str:
        """按老师挑音色：该老师配了复刻音色就用它，否则回落全局预置音色。"""
        tid = (teacher_id or "").strip()
        return self.teacher_speakers.get(tid) or self.speaker

    def _resource_for(self, speaker: str) -> str:
        """音色决定资源位——必须成对出现，否则服务端直接拒。"""
        return self.clone_resource_id if self._is_clone_speaker(speaker) else self.resource_id

    @staticmethod
    def _readable_text(text: str) -> str:
        """保证待合成文本含句号「。」——服务端没有句号会直接判「无可读文本」。

        只做**最小补齐**（末尾补一个句号），不增删要念的字词：
        ``答对了！`` -> ``答对了！。``（实测可用），念出来的内容不变。
        """
        body = (text or "").strip()
        if body and "。" not in body:
            body += "。"
        return body

    def _payload(self, text: str, speaker: str) -> dict:
        return {
            "user": {"uid": self.uid},
            "req_params": {
                "text": text,
                "speaker": speaker,
                "audio_params": {"format": self.audio_format, "sample_rate": self.sample_rate},
            },
        }

    async def synthesize(self, text: str, *, teacher_id: str | None = None) -> bytes:
        import base64
        import json
        import uuid

        body = self._readable_text(sanitize_for_tts(text))
        if not body:
            return b""
        # 音色与资源位必须成对：复刻音色(S_) 配 seed-icl-2.0，预置音色配 10029。
        voice = self._voice_for(teacher_id)
        resource = self._resource_for(voice)
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": resource,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        chunks: list[bytes] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.api_base}/api/v3/tts/unidirectional",
                json=self._payload(body, voice),
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    raw = (line or "").strip()
                    if not raw or not raw.startswith("{"):
                        continue
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    data = item.get("data")
                    if isinstance(data, str) and data:
                        try:
                            chunks.append(base64.b64decode(data))
                        except Exception:  # noqa: BLE001 - 单块坏了不该丢整句
                            log.warning("豆包 TTS 分块解码失败，已跳过该块")
                        continue
                    # 没有 data 的行未必是错误：正常结束时服务端会补一帧
                    # ``code=20000000, message=OK``（实测），当错误打会刷一堆假告警。
                    # 只有非"成功码"才记日志，便于定位 45002001 / 45000030 这类真错误。
                    code = item.get("code")
                    if code not in (None, 0, 20000000):
                        log.warning(
                            "豆包 TTS 错误帧 code={} message={}",
                            code, item.get("message"),
                        )
        if not chunks:
            log.warning("豆包 TTS 本句无音频（resource_id={} speaker={} voice={}）",
                        resource, self.speaker, voice)
        return b"".join(chunks)


class TeacherTTS(BaseTTSClient):
    """按老师分派 TTS 引擎（多老师混合发声）。

    为什么需要：本项目的音色克隆依赖「老师素材里的干声」。若某些老师的素材里
    有效语音太少（例如整段几乎静音），零样本克隆会得不到稳定音色（听起来可能
    性别都不对）。此时按档案里的 ``voice_engine`` 退回 edge-tts 的自带音色，
    保证「每位老师都有一段自然、性别正确的普通话」。

    * ``voice_engine = "clone"``（默认）-> 走 :attr:`clone`（CosyVoice2 零样本克隆）；
    * ``voice_engine = "edge"``  -> 走 :attr:`edge`（edge-tts，音色取档案的 voice）。
    """

    # edge 端点偶发失败后的**冷却时长**（秒）：冷却期内不再白等重试，直接按既定音色处理。
    EDGE_RETRY_AFTER = 300.0

    # edge 单句失败时的**同音色重试**次数与退避基数（秒）。
    # 「抖一下」是 edge 最常见的失败形态，重试几次就能吸收掉，不必切引擎换声。
    EDGE_ATTEMPTS = 3
    EDGE_RETRY_SLEEP = 0.4

    # 连续多少次「整句都失败」才认定 edge 长期不可用、允许换一次音色。
    # 低于阈值的失败只让该句静音（保持音色）；达到阈值才切克隆并永久锁定，
    # 否则 edge 长时间挂掉会变成整堂课静音。阈值取 3 是为了把偶发抖动
    # （重试后仍失败的零星几次）和真正的长期故障区分开。
    EDGE_SWITCH_AFTER_FAILS = 3

    def __init__(self, clone: BaseTTSClient, edge: BaseTTSClient | None) -> None:
        self.clone = clone
        self.edge = edge
        self._edge_dead: dict[str, float] = {}   # teacher_id -> 最近一次 edge 失败时间
        # teacher_id -> 本会话已**定死**的引擎名。一旦写入就不再改变：
        # 音色是学生识别「同一位老师」最直观的线索，一堂课里换音色 ≈ 换了个人。
        self._lock: dict[str, str] = {}
        # teacher_id -> 本会话是否已经**成功**用 edge 出过声。
        # 只有「一次都没成功过」才允许退到克隆引擎（否则整堂课都没声音）；
        # 已经出过声就绝不再换，宁可这一句没有音频，也不换音色。
        self._spoke: dict[str, bool] = {}
        # teacher_id -> edge 连续整句失败的次数（成功一次即清零）
        self._streak: dict[str, int] = {}

    def begin_utterance(self, teacher_id: str | None = None) -> None:
        """标记「一段新播报开始」（编排层每段播报前调用）。

        注意：**这里不再清掉音色锁定**。历史实现是每段重新选一次引擎，结果是
        edge 抖动一次就把后续 5 分钟全部换成克隆引擎，听感变成「讲着讲着换了个
        老师」。现在引擎在一堂课内定死，音色恒定。
        需要重新选（换老师 / 重新开课）请显式调用 :meth:`reset_voice`。
        """
        return None

    def reset_voice(self, teacher_id: str | None = None) -> None:
        """显式解除某位老师的音色锁定（换老师 / 重新开课时调用）。"""
        tid = (teacher_id or "").strip()
        self._lock.pop(tid, None)
        self._spoke.pop(tid, None)
        self._streak.pop(tid, None)
        self._edge_dead.pop(tid, None)

    def _use_edge(self, teacher_id: str | None) -> bool:
        """该老师是否应改用 edge-tts 兜底音色。"""
        tid = (teacher_id or "").strip()
        if not tid or self.edge is None:
            return False
        try:
            from config.teachers import load_teacher  # 局部导入避免循环依赖

            profile = load_teacher(tid)
        except Exception:  # noqa: BLE001 - 档案异常时按克隆处理
            return False
        return bool(profile and profile.voice_engine == "edge")

    def _edge_alive(self, teacher_id: str | None) -> bool:
        """该老师的 edge 是否不在熔断期内。"""
        import time

        tid = (teacher_id or "").strip()
        dead_at = self._edge_dead.get(tid)
        return not (dead_at and time.time() - dead_at < self.EDGE_RETRY_AFTER)

    def _pick_engine(self, teacher_id: str | None) -> BaseTTSClient:
        """挑本节课该用的引擎：**一旦定下就整堂课不再改变**。

        为什么不每句 / 每段重新判断：edge 抖动一次就会把后续 5 分钟全部换成克隆
        引擎，学生听到的是「换了位老师」——比偶尔一句没声音糟糕得多。
        """
        tid = (teacher_id or "").strip()
        locked = self._lock.get(tid)
        if locked == "edge" and self.edge is not None:
            return self.edge
        if locked == "clone":
            return self.clone
        use_edge = (
            self._use_edge(teacher_id) and self.edge is not None and self._edge_alive(teacher_id)
        )
        self._lock[tid] = "edge" if use_edge else "clone"
        return self.edge if use_edge else self.clone

    async def synthesize(self, text: str, *, teacher_id: str | None = None) -> bytes:
        """合成一句话，并按**全局语速**统一做一次「变速不变调」。

        语速快的根因：零样本克隆会**模仿参考干声的说话节奏**，所以不同老师快慢不一。
        这里在 TTS 唯一汇聚点统一处理，对「克隆 / edge」两种引擎、所有老师都生效。
        """
        audio = await self._synthesize_voice(text, teacher_id=teacher_id)
        speed = _tts_speed()
        if audio and abs(speed - 1.0) >= 1e-3:
            audio = await asyncio.to_thread(_apply_speed, audio, speed)
        return audio

    async def _synthesize_voice(self, text: str, *, teacher_id: str | None = None) -> bytes:
        tid = (teacher_id or "").strip()
        engine = self._pick_engine(teacher_id)
        # 克隆引擎：必须**保留 teacher_id** —— 它会据此按形象性别挑预置音色，
        # 传 None 会落到全局 TTS_REF_AUDIO（一份固定女声），男老师就变女声了。
        if engine is not self.edge or self.edge is None:
            return await engine.synthesize(text, teacher_id=teacher_id)

        # 选定 edge：先**用同一音色重试**，把「抖一下」吸收掉，而不是换引擎换声
        import time

        last = ""
        for attempt in range(1, self.EDGE_ATTEMPTS + 1):
            try:
                audio = await self.edge.synthesize(text, teacher_id=teacher_id)
                if audio:
                    self._spoke[tid] = True
                    self._streak[tid] = 0
                    return audio
                last = "返回空音频"
            except Exception as err:  # noqa: BLE001 - edge 端点不稳定时不应中断播报
                last = f"{type(err).__name__}: {err}"
            log.warning("edge tts 第 {}/{} 次失败（{}）：{}",
                        attempt, self.EDGE_ATTEMPTS, tid, last)
            if attempt < self.EDGE_ATTEMPTS and self.EDGE_RETRY_SLEEP > 0:
                await asyncio.sleep(self.EDGE_RETRY_SLEEP * attempt)

        self._edge_dead[tid] = time.time()
        self._streak[tid] = self._streak.get(tid, 0) + 1

        # 换不换音色，看「是否已经证明 edge 能出声」+「是否持续故障」：
        #  1) 本课一次都没成功过（典型是开课就断网）-> 立刻切，否则开场就是哑巴；
        #  2) 已经出声过但连续失败达到阈值 -> 判定 edge 长期不可用，切一次并永久锁定；
        #  3) 其余（零星抖动）-> **保持音色**，本句不下发音频（上层走静音口型，
        #     文字仍在屏幕上），听感上还是同一位老师。
        need_switch = (
            not self._spoke.get(tid)
            or self._streak.get(tid, 0) >= self.EDGE_SWITCH_AFTER_FAILS
        )
        if need_switch:
            log.warning(
                "edge 持续不可用（{}，连续失败 {} 次），改用克隆引擎并**永久锁定音色**",
                tid, self._streak.get(tid, 0),
            )
            self._lock[tid] = "clone"
            return await self.clone.synthesize(text, teacher_id=teacher_id)

        log.warning("edge 本句失败但未达切换阈值（{}），保持音色、本句不下发音频", tid)
        return b""



def _audio_from_response(resp: "httpx.Response") -> bytes:
    """归一化 TTS 返回：兼容裸音频字节与 ``{"audio": "<base64>"}`` 两种形态。"""
    import base64

    ctype = resp.headers.get("content-type", "")
    if "application/json" in ctype:
        data = resp.json()
        audio_b64 = data.get("audio") or data.get("data") or ""
        if isinstance(audio_b64, str) and audio_b64:
            if audio_b64.startswith("data:"):
                audio_b64 = audio_b64.partition(",")[2]
            return base64.b64decode(audio_b64)
        return b""
    return resp.content


@lru_cache
def get_tts() -> BaseTTSClient:
    provider = (settings.tts_provider or "openai").lower()
    if provider == "cosyvoice" and settings.cosyvoice_api_base:
        log.info("use CosyVoice2 TTS: {}（素材语音不足的老师自动退回 edge 音色）",
                 settings.cosyvoice_api_base)
        return TeacherTTS(
            CosyVoiceTTS(
                settings.cosyvoice_api_base,
                speaker=settings.tts_voice,
                ref_audio=settings.tts_ref_audio,
                ref_text=settings.tts_ref_text,
                api_key=settings.tts_api_key,
            ),
            EdgeTTS(settings.tts_voice or "zh-CN-XiaoxiaoNeural"),
        )
    if provider == "gptsovits" and settings.gptsovits_api_base:
        log.info("use GPT-SoVITS TTS: {}", settings.gptsovits_api_base)
        return TeacherTTS(
            GptSoVitsTTS(
                settings.gptsovits_api_base,
                ref_audio=settings.tts_ref_audio,
                prompt_text=settings.tts_ref_text,
                api_key=settings.tts_api_key,
            ),
            EdgeTTS(settings.tts_voice or "zh-CN-XiaoxiaoNeural"),
        )
    if provider == "doubao" and settings.doubao_api_key:
        teacher_speakers = _parse_teacher_speakers(settings.doubao_teacher_speakers)
        log.info("use 豆包（火山引擎）TTS: {} resource={} 预置音色={} 已配复刻音色 {} 位老师",
                 settings.doubao_api_base, settings.doubao_resource_id,
                 settings.doubao_speaker, len(teacher_speakers))
        return DoubaoTS(
            settings.doubao_api_key,
            api_base=settings.doubao_api_base,
            resource_id=settings.doubao_resource_id,
            clone_resource_id=settings.doubao_clone_resource_id,
            speaker=settings.doubao_speaker,
            teacher_speakers=teacher_speakers,
        )
    if provider == "edge":
        voice = settings.tts_voice or "zh-CN-XiaoxiaoNeural"
        log.info("use Edge TTS: {}", voice)
        return EdgeTTS(voice)
    if provider == "mock":
        log.warning("tts provider=mock, use MockTTS")
        return MockTTS()
    if settings.tts_api_key and settings.tts_api_base:
        log.info("use OpenAI-compatible TTS: {}", settings.tts_model)
        return OpenAICompatTTS(
            settings.tts_api_base, settings.tts_api_key, settings.tts_model, settings.tts_voice
        )
    log.warning("no tts config, fallback to MockTTS")
    return MockTTS()
