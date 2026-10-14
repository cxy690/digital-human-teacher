# -*- coding: utf-8 -*-
"""本地"对上口型"演示（无需 MuseTalk/GPU/下载）。

忠实复刻 clients/student_tablet/index.html 的「照片木偶」口型算法：
  真实音频 RMS 能量 -> base=min(1.9,0.25+rms*6.0) -> 嘴部椭圆区域按 scaleY(base) 开合。
用 edge-tts 合成真实语音（联网，不下载模型），输出一段带声音、嘴型随语音开合的回放 mp4。

这就是项目在 AVATAR_BACKEND=mock 时的本地口型路径；云端 MuseTalk 则是音素级精确口型。
用法： D:\\An\\envs\\tea\\python.exe scripts/preview_lipsync.py
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import wave

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_IMG = os.path.join(ROOT, "clients", "student_tablet", "ai_teacher_smile.png")
TMP_MP3 = os.path.join(ROOT, "_demo_speech.mp3")
TMP_WAV = os.path.join(ROOT, "_demo_speech.wav")
OUT = os.path.join(ROOT, "preview_lipsync.mp4")

# 与前端 #mouth-layer 完全一致的口型参数
CX, CY = 0.505, 0.365          # 嘴部中心（transform-origin）
RX, RY = 0.06, 0.032           # 椭圆遮罩半径（相对宽/高）
FPS = 24
TEXT = ("同学你好，我们一起来看这道数学题。先读题，再列式，最后计算，"
        "一步一步来，你一定能做对。")
VOICE = "zh-CN-XiaoxiaoNeural"


def _ffmpeg() -> str:
    """优先用 imageio_ffmpeg 自带的 ffmpeg；tea 环境没有则回退系统 ffmpeg。"""
    cand = r"D:\An\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
    return cand if os.path.isfile(cand) else "ffmpeg"


async def _synth() -> None:
    import edge_tts
    await edge_tts.Communicate(TEXT, voice=VOICE).save(TMP_MP3)


def _pcm_envelope(wav_path: str, fps: int) -> tuple[np.ndarray, int]:
    """解码为 16k 单声道 PCM，按视频帧长求 RMS 包络（归一化到 0~1 量级）。"""
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    win = max(1, int(sr / fps))
    pad = (-len(pcm)) % win
    if pad:
        pcm = np.concatenate([pcm, np.zeros(pad, np.float32)])
    frames = pcm.reshape(-1, win)
    rms = np.sqrt((frames ** 2).mean(axis=1))     # 每帧 RMS
    return rms, len(rms)


def _mouth_mask(h: int, w: int) -> np.ndarray:
    """软边椭圆遮罩：r<=0.55 全开，0.55~1 线性淡出（对应 CSS radial-gradient #000 55%->transparent 100%）。"""
    ys, xs = np.mgrid[0:h, 0:w]
    dx = (xs - CX * w) / (RX * w)
    dy = (ys - CY * h) / (RY * h)
    r = np.sqrt(dx * dx + dy * dy)
    mask = np.clip((1.0 - (r - 0.55) / 0.45), 0.0, 1.0)
    return (mask[:, :, None]).astype(np.float32)


def _scale_y(img: np.ndarray, sy: float, h: int) -> np.ndarray:
    """绕嘴部中心 cy 做纵向缩放（等价 CSS transform: scaleY(sy), origin-y=CY）。"""
    cy = CY * h
    M = np.array([[1, 0, 0], [0, sy, cy * (1 - sy)]], dtype=np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def main() -> None:
    if not os.path.isfile(BASE_IMG):
        raise SystemExit(f"缺少头像底图：{BASE_IMG}")
    print(f"[1/4] edge-tts 合成语音：{TEXT[:20]}...")
    asyncio.run(_synth())
    ff = _ffmpeg()
    subprocess.run([ff, "-y", "-i", TMP_MP3, "-ac", "1", "-ar", "16000", TMP_WAV],
                   check=True, capture_output=True)
    rms, n_frames = _pcm_envelope(TMP_WAV, FPS)
    print(f"[2/4] 音频时长 ≈ {n_frames / FPS:.2f}s，共 {n_frames} 帧 @ {FPS}fps")

    base = cv2.imread(BASE_IMG, cv2.IMREAD_COLOR)
    h, w = base.shape[:2]
    base_f = base.astype(np.float32)
    mask = _mouth_mask(h, w)
    inv = 1.0 - mask

    print("[3/4] 逐帧渲染口型（能量驱动嘴部开合）...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    silent = os.path.join(ROOT, "_demo_silent.mp4")
    writer = cv2.VideoWriter(silent, fourcc, FPS, (w, h))
    for i in range(n_frames):
        mouth_open = min(1.9, 0.25 + float(rms[i]) * 6.0)   # 前端同款公式
        scaled = _scale_y(base, mouth_open, h).astype(np.float32)
        frame = base_f * inv + scaled * mask
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()

    print("[4/4] 合成音轨...")
    subprocess.run([ff, "-y", "-i", silent, "-i", TMP_MP3, "-c:v", "copy",
                    "-c:a", "aac", "-shortest", OUT], check=True, capture_output=True)
    for p in (silent, TMP_WAV):
        try:
            os.remove(p)
        except OSError:
            pass
    print(f"\n✅ 对口型回放已生成：{OUT}")
    print(f"   {n_frames} 帧 ≈ {n_frames / FPS:.1f}s  {w}x{h} @ {FPS}fps（嘴型随语音能量开合）")


if __name__ == "__main__":
    main()
