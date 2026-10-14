"""WebRTC 输出轨道（aiortc）。

- VideoTrack：节拍跟随引擎口型帧率；播报中按音频时钟向引擎取对应口型帧（未产出时沿用上一帧），
  待机时读引擎最新帧。时间戳取墙钟（非逐次累加）：发送管线（VP8 编码）偶发积压时
  戳不整体漂移，避免嘴画落后声音逐秒拉大；发送分辨率降采样一半降低编码耗时。
- AudioTrack：48kHz 20ms 块节拍；从引擎音频段快照按自己的读指针取样本（无语音时输出静音），
  段序号变化（新播报开始/结束）时重置读指针。
"""
import asyncio
import fractions
import logging
import time

import cv2
import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame, VideoFrame

logger = logging.getLogger("digital_human.webrtc")

VIDEO_CLOCK_RATE = 90000


class VideoTrack(MediaStreamTrack):
    kind = "video"

    # 发送分辨率降采样：720x1180 原始尺寸 VP8 软编码撑不满帧率（recv 积压→时间戳漂移→
    # 嘴画与声音渐进错位+卡顿）；缺省半分辨率。本机 config.yaml 置 send_scale: 1.0
    # 与待机 mp4（全分辨率）画质对齐；云端 config.cloud.yaml 保持 0.5 省上行/算力
    SEND_SCALE = 0.5

    def __init__(self, engine):
        super().__init__()
        self._engine = engine
        self.SEND_SCALE = float((getattr(engine, "cfg", None) or {}).get(
            "send_scale", self.SEND_SCALE))
        self._ptime = 1.0 / max(10, int(getattr(engine, "fps", 25)))
        self._pts = 0
        self._start = None
        self._last = None
        self._fallback = np.zeros((1280, 720, 3), dtype=np.uint8)
        # 交付埋点：每 5s 汇总一次实际交付帧率/欠帧数（卡顿诊断）
        self._stat_t = None
        self._stat_n = 0
        self._stat_miss = 0

    def _downscale(self, frame):
        h, w = frame.shape[:2]
        nw, nh = w // 2 * 2, h // 2 * 2
        sw, sh = int(nw * self.SEND_SCALE) // 2 * 2, int(nh * self.SEND_SCALE) // 2 * 2
        if (sw, sh) != (w, h):
            return cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        return frame

    async def recv(self):
        now = time.time()
        if self._start is None:
            self._start = now
            self._pts = 0
        else:
            # 节拍槽锚定墙钟（不逐次累加）：积压后下一拍直接对齐真实时间，不欠拍越欠越多
            slot = int((now - self._start) / self._ptime) + 1
            wait = self._start + slot * self._ptime - now
            if wait > 0:
                await asyncio.sleep(wait)
            # 时间戳取墙钟：编码/发送偶发积压时浏览器仍按真实时间轴播放，嘴画不渐进落后声音
            self._pts = max(self._pts + 1, int((time.time() - self._start) * VIDEO_CLOCK_RATE))
        frame, playing = self._engine.frame_at(time.time())
        if frame is None:
            # 播报中但本帧尚未推理完：沿用上一帧；首帧未就绪时先垫待机/兼容帧。
            # 注意：numpy 数组不能参与 or/and 真值判断（会抛 ValueError 弄死发送任务），
            # 必须显式 is None 分支
            self._stat_miss += 1
            if playing and self._last is not None:
                frame = self._last
            else:
                frame = self._engine.current_frame()
                if frame is None:
                    frame = self._fallback
        else:
            self._last = frame if playing else None
        now2 = time.time()
        if self._stat_t is None:
            self._stat_t = now2
        self._stat_n += 1
        if now2 - self._stat_t >= 5.0:
            logger.info("视频轨道交付：%.1f fps（欠帧 %d，playing=%s）",
                        self._stat_n / (now2 - self._stat_t), self._stat_miss, playing)
            self._stat_t = now2
            self._stat_n = 0
            self._stat_miss = 0
        frame = self._downscale(frame)
        vf = VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        vf.pts = self._pts
        vf.time_base = fractions.Fraction(1, VIDEO_CLOCK_RATE)
        return vf


class AudioTrack(MediaStreamTrack):
    kind = "audio"

    SAMPLE_RATE = 48000
    CHUNK = 960                # 20ms
    SILENCE = np.zeros(CHUNK, dtype=np.int16)
    MAX_SEG_LAG = 2.0          # 段内位置落后上限（s）：仅作保护，正常情况下永不触发

    def __init__(self, engine):
        super().__init__()
        self._engine = engine
        self._pts = 0
        self._start = None
        self._seq = -1
        self._seq_t0 = None    # 当前音频段起播墙钟（仅用于极端落后的保护判断）
        self._pos = 0
        # 交付埋点：每 5s 汇总交付帧率与时间轴落后量（音频杂音/卡顿诊断）
        self._stat_t = None
        self._stat_n = 0

    async def recv(self):
        if self._start is None:
            self._start = time.time()
            self._pts = 0
        else:
            wait = self._start + self._pts / self.SAMPLE_RATE - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            # 时间戳严格按采样数累加（与发出的数据一一对应）。
            # 旧实现取 max(累计, 墙钟)：积压时时间戳一次跳增几十毫秒，而这一帧只带 20ms
            # 数据，播放端时间轴被拉长/压缩 → 听感"跳音/杂音"。改成纯采样时钟后，
            # 积压由下方 wait<=0 的连续补发自然追平（浏览器抖动缓冲吸收），时间轴不再断层。
            self._pts += self.CHUNK

        pcm, seq = self._engine.audio_snapshot()
        now = time.time()
        if seq != self._seq:
            self._seq = seq
            self._seq_t0 = now
            self._pos = 0
        if pcm is None:
            chunk = self.SILENCE
        else:
            # 读指针纯顺序前进（不跳样本）：波形连续，从根上消除断裂处的"咔哒/爆"音。
            # 只有长期跟不上（段内位置落后 > MAX_SEG_LAG）才硬重置一次，避免永久错位。
            if self._seq_t0 is not None and \
                    now - self._seq_t0 - self._pos / self.SAMPLE_RATE > self.MAX_SEG_LAG:
                self._pos = min(int((now - self._seq_t0) * self.SAMPLE_RATE),
                                max(0, len(pcm) - self.CHUNK))
            chunk = pcm[self._pos:self._pos + self.CHUNK]
            self._pos += self.CHUNK
            if len(chunk) < self.CHUNK:
                chunk = np.concatenate([chunk, np.zeros(self.CHUNK - len(chunk), dtype=np.int16)])

        stereo = np.empty((1, self.CHUNK * 2), dtype=np.int16)   # s16 packed 交错：LRLR...
        stereo[0, 0::2] = chunk
        stereo[0, 1::2] = chunk
        frame = AudioFrame.from_ndarray(stereo, format="s16", layout="stereo")
        frame.sample_rate = self.SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self.SAMPLE_RATE)

        # 交付埋点：落后 >0 表示发送端被积压拖慢（持续增大才需要关注）
        if self._stat_t is None:
            self._stat_t = now
        self._stat_n += 1
        if now - self._stat_t >= 5.0:
            lag_ms = ((now - self._start) - self._pts / self.SAMPLE_RATE) * 1000.0
            logger.info("音频轨道交付：%.1f fps（时间轴落后 %.0f ms）",
                        self._stat_n / (now - self._stat_t), lag_ms)
            self._stat_t = now
            self._stat_n = 0
        return frame
