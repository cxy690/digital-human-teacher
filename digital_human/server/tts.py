"""语音合成（edge-tts）：文本 → 语音数据。

产物两路：
- y16:   float32 numpy，16kHz 单声道（whisper 特征提取用）
- pcm48: int16 numpy，48kHz 单声道（WebRTC 音轨推送用，aiortc 的 opus 编码固定 48kHz）

实现说明：edge-tts 输出 mp3 字节流，由 soundfile（libsndfile，原生支持 MP3）直接解码，
无需 ffmpeg 可执行文件、无需磁盘落盘。
在渲染线程（同步环境）中调用；内部用 asyncio.run 驱动 edge-tts 协程，不依赖外部事件循环。

音色容错：科目目录里每个科目可配 edge-tts 音色。若配了 edge-tts 不支持的音色
（实测如 zh-CN-XiaomoNeural / zh-CN-XiaohanNeural / zh-CN-XiaoxuanNeural 均不存在，
表现为 edge-tts 返回 "No audio was received"），此处会自动退回默认音色重试 ——
否则该科目的整段讲解会彻底没声音，前端还会一直停在"准备中"。
"""
import asyncio
import io
import logging

import librosa
import numpy as np
import soundfile as sf

SR_16K = 16000
SR_48K = 48000

# 单次合成超时（秒）：edge-tts 走外网流式合成，偶发「连接建立后长时间无数据」；
# 渲染线程不能因此永久卡死（卡死会导致播报状态永远为 true），故设时限 + 重试
TTS_TIMEOUT_SECONDS = 20.0
TTS_ATTEMPTS = 2

# 保底音色：所有候选都失败时仍不可用则直接报错（该音色是 edge-tts 长期稳定的中文音色）
TTS_FALLBACK_VOICE = "zh-CN-XiaoxiaoNeural"

logger = logging.getLogger("digital_human.tts")


class EdgeTTS:
    def __init__(self, voice="zh-CN-XiaoxiaoNeural", rate="+0%", volume="+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume

    def synthesize(self, text, voice=None):
        """合成语音，返回 (y16, pcm48)。

        音色候选顺序：请求音色（科目配置）→ 引擎默认音色 → 保底音色。
        每个音色最多重试 TTS_ATTEMPTS 次（覆盖网络抖动/超时）；全部失败才抛异常
        （引擎按「播报失败」处理并恢复待机，不会拖死渲染线程）。
        """
        wanted = (voice or self.voice or "").strip()
        candidates = []
        for v in (wanted, (self.voice or "").strip(), TTS_FALLBACK_VOICE):
            if v and v not in candidates:
                candidates.append(v)

        last_err = None
        for idx, cand in enumerate(candidates):
            for attempt in range(1, TTS_ATTEMPTS + 1):
                buf = io.BytesIO()
                try:
                    self._edge_stream(text, cand, buf)
                    if buf.getbuffer().nbytes <= 0:
                        raise RuntimeError("edge-tts 未返回音频数据")
                    if cand != wanted:
                        logger.warning("已改用音色 %s 合成（原音色 %s 不可用）", cand, wanted)
                    return self._decode(buf)
                except Exception as err:  # noqa: BLE001 - 网络抖动/超时/音色不支持均重试
                    last_err = err
                    logger.warning("edge-tts 合成失败（音色 %s，第 %d/%d 次）：%s",
                                   cand, attempt, TTS_ATTEMPTS, err)
            if idx < len(candidates) - 1:
                logger.warning("音色 %s 不可用，改用备用音色重试", cand)
        raise RuntimeError(
            f"edge-tts 合成失败（已尝试音色 {candidates}，每个 {TTS_ATTEMPTS} 次）：{last_err}")

    @staticmethod
    def _decode(buf):
        """mp3 字节流 → (y16 16k float32, pcm48 int16)。"""
        buf.seek(0)
        y, sr = sf.read(buf, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y[:, 0]
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR_16K) if sr != SR_16K else y
        # 音轨直接从原始采样率重采样到 48k。旧实现走 24k→16k→48k 中转：16k 只是 whisper
        # 特征要用的中间格式，却把听感带宽砍到 8kHz 再把发闷的素材插值升回去，齿音/辅音
        # 丢失、两次重采样叠加重采样失真（听感偏"杂"）。whisper 仍用 16k 单路，互不影响。
        y48 = y if sr == SR_48K else librosa.resample(y, orig_sr=sr, target_sr=SR_48K)
        pcm48 = (np.clip(y48, -1.0, 1.0) * 32767.0).astype(np.int16)
        # 首尾各 10ms 淡入/淡出：播报段与静音之间是硬切，波形从 0 突变到语音（或反之）
        # 会听到"咔"声，段尾尤其明显（语音样本被直接截断）。在源头处理一次即覆盖所有播报。
        edge = min(int(SR_48K * 0.01), len(pcm48) // 2)
        if edge > 0:
            ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
            faded = pcm48.astype(np.float32)
            faded[:edge] *= ramp
            faded[-edge:] *= ramp[::-1]
            pcm48 = faded.astype(np.int16)
        return y16.astype(np.float32), pcm48

    def _edge_stream(self, text, voice, buf):
        import edge_tts

        async def _collect():
            communicate = edge_tts.Communicate(text, voice, rate=self.rate, volume=self.volume)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])

        async def _run():
            # 超时覆盖「建连 + 全量收流」：edge-tts 自身无超时，卡住即永久等待
            await asyncio.wait_for(_collect(), timeout=TTS_TIMEOUT_SECONDS)

        asyncio.run(_run())
