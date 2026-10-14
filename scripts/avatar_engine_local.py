# -*- coding: utf-8 -*-
"""本地 CPU 版"新老师"数字人渲染引擎（MuseTalk 的轻量替身，无需 GPU/下载）。

实现与 scripts/avatar_engine_musetalk.py **完全相同的 HTTP 契约**：
    POST /render  {"audio": "<base64>", "text": "...", "gesture": "nod|shake|shrug|point"}
    响应 {"mode":"video","video":"<base64 mp4>","mime":"video/mp4","text":...}

渲染内容（全部 CPU + OpenCV）：
* 画面 = 新老师**原始视频帧**：talk 主底片 + 手势基元，按 ``_plan_frames`` 交叉淡化切换
  （与 MuseTalk 引擎同一套帧规划逻辑，含 _bounce 往复、talk 游标连续）；
* 不做任何嘴部覆盖/重绘——直接用原视频自带的自然嘴部动作，仅把音频按时间对齐；
  （音素级无缝重绘嘴部需 MuseTalk，见云端部署；本地无模型不做覆盖层以免出现贴片破绽）
* **静音闭嘴**：检测音频 RMS，静音持续≥ MIN_SILENT_FRAMES 帧时整帧交叉淡化到闭嘴待机帧
  （talk 第 0 帧，闭嘴微笑），有声再淡回说话画面——避免“音频没声但嘴还在动”；
* 音轨 = 输入音频直接合流。

用途：本地交互式演示（AVATAR_BACKEND=http 指向本引擎），云端仍用真 MuseTalk 音素级口型。
用法： D:\\An\\envs\\tea\\python.exe scripts/avatar_engine_local.py
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CLIP_DIR = Path(os.getenv("AVATAR_CLIP_DIR", str(ROOT / "data" / "ai_teacher")))
FACE_VIDEO = os.getenv("AVATAR_FACE_VIDEO", str(CLIP_DIR / "talk.mp4"))
# 动作基元名（含可选 idle 待机）；与云端 MuseTalk 引擎保持一致。
ACTION_CLIPS = ("nod", "shake", "shrug", "point", "idle")
PORT = int(os.getenv("ENGINE_PORT", "8012"))
HOST = os.getenv("ENGINE_HOST", "127.0.0.1")
CROSSFADE_FRAMES = int(os.getenv("MUSETALK_CROSSFADE", "6"))

# 新老师基元视频同机位同帧率；本地不做嘴部重绘（无模型），故无需嘴部坐标
_lock = threading.Lock()
_clips: dict[str, "_Clip"] = {}
_fps = 24.0

# 静音闭嘴参数：RMS 低于阈视为静音；静音连续≥ MIN_SILENT_FRAMES 帧才切待机（防抖），
# 切换用 SILENT_RAMP 帧交叉淡化（整帧，非嘴部贴片）。
SIL_RMS_THRESHOLD = 0.03
MIN_SILENT_FRAMES = 8
SILENT_RAMP = 4


def _ffmpeg() -> str:
    """定位 ffmpeg：FFMPEG_PATH -> imageio-ffmpeg 自带 -> 历史硬编码路径 -> PATH。"""
    env = os.getenv("FFMPEG_PATH", "").strip()
    if env and os.path.isfile(env):
        return env
    try:
        import imageio_ffmpeg  # noqa: PLC0415 - 可选依赖

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - 可选依赖缺失
        pass
    cand = r"D:\An\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
    return cand if os.path.isfile(cand) else "ffmpeg"


class _Clip:
    __slots__ = ("name", "frames", "cursor")

    def __init__(self, name: str, frames: list[np.ndarray]) -> None:
        self.name = name
        self.frames = frames
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.frames)


def _read_clip_frames(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"底片无有效帧：{path}")
    return frames, float(fps)


def _init() -> None:
    global _fps
    talk_frames, fps = _read_clip_frames(FACE_VIDEO)
    _clips["talk"] = _Clip("talk", talk_frames)
    _fps = fps
    for g in ACTION_CLIPS:
        p = CLIP_DIR / f"{g}.mp4"
        if p.is_file():
            _clips[g] = _Clip(g, _read_clip_frames(str(p))[0])
        else:
            print(f"[local-avatar] 动作基元缺失，跳过：{p}", flush=True)
    print(f"[local-avatar] clips={list(_clips)} fps={_fps:.2f} port={PORT}", flush=True)


# ---------------------------- 帧规划（与 MuseTalk 引擎一致） ----------------------------
def _bounce(idx: int, n: int) -> int:
    if n <= 1:
        return 0
    cycle = 2 * (n - 1)
    m = idx % cycle
    return m if m < n else cycle - m


def _plan_frames(n_frames: int, gesture: str | None):
    talk = _clips["talk"]
    g = _clips.get(gesture) if gesture else None
    plan = []
    if g is None or g is talk:
        for i in range(n_frames):
            plan.append([(talk, (talk.cursor + i) % len(talk), 1.0)])
        talk.cursor = (talk.cursor + n_frames) % len(talk)
        return plan
    k = max(0, min(CROSSFADE_FRAMES, n_frames // 2, len(g) // 2))
    for i in range(n_frames):
        t_idx = (talk.cursor + i) % len(talk)
        if i < k:
            a = (i + 1) / (k + 1)
            plan.append([(talk, t_idx, 1.0 - a), (g, _bounce(i, len(g)), a)])
        elif i < n_frames - k:
            plan.append([(g, _bounce(i - k, len(g)), 1.0)])
        else:
            a = (i - (n_frames - k) + 1) / (k + 1)
            plan.append([(g, _bounce(i - k, len(g)), 1.0 - a), (talk, t_idx, a)])
    talk.cursor = (talk.cursor + n_frames) % len(talk)
    return plan
# 每个采样元组: (clip, frame_idx, blend_weight)


# ---------------------------- 渲染 ----------------------------
def _audio_to_mp3(audio_bytes: bytes) -> tuple[str, np.ndarray]:
    """音频字节 -> (临时 mp3 路径, 每帧 RMS 包络)。"""
    tmp = tempfile.mkdtemp(prefix="lav_")
    mp3 = os.path.join(tmp, "in.mp3")
    wav = os.path.join(tmp, "a.wav")
    Path(mp3).write_bytes(audio_bytes)
    subprocess.run([_ffmpeg(), "-y", "-i", mp3, "-ac", "1", "-ar", "16000", wav],
                   check=True, capture_output=True)
    with wave.open(wav, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    win = max(1, int(sr / _fps))
    pad = (-len(pcm)) % win
    if pad:
        pcm = np.concatenate([pcm, np.zeros(pad, np.float32)])
    rms = np.sqrt((pcm.reshape(-1, win) ** 2).mean(axis=1))
    return mp3, rms


def _idle_weights(rms: np.ndarray) -> np.ndarray:
    """逐帧的“待机（闭嘴）混合权重” 0~1：静音段淡入到 1，有声淡回 0。"""
    n = len(rms)
    silent = rms < SIL_RMS_THRESHOLD
    idle = np.zeros(n, bool)
    i = 0
    while i < n:                                    # 只把“足够长的静音run”当待机，防抖
        if silent[i]:
            j = i
            while j < n and silent[j]:
                j += 1
            if j - i >= MIN_SILENT_FRAMES:
                idle[i:j] = True
            i = j
        else:
            i += 1
    w = np.zeros(n)
    cur = 0.0
    step = 1.0 / max(1, SILENT_RAMP)
    for k in range(n):                              # 线性斜坡交叉淡化
        tgt = 1.0 if idle[k] else 0.0
        cur = min(tgt, cur + step) if cur < tgt else max(tgt, cur - step)
        w[k] = cur
    return w


def _render(audio_bytes: bytes, out_mp4: str, gesture: str | None) -> None:
    mp3, rms = _audio_to_mp3(audio_bytes)
    n_frames = len(rms)
    plan = _plan_frames(n_frames, gesture)
    idle_w = _idle_weights(rms)
    idle_still = _clips["talk"].frames[0].astype(np.float32)   # 闭嘴微笑待机帧

    h, w = _clips["talk"].frames[0].shape[:2]
    silent = out_mp4 + ".silent.mp4"
    writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), _fps, (w, h))
    for i, spec in enumerate(plan):
        canvas = None
        for clip, idx, bw in spec:
            layer = clip.frames[idx].astype(np.float32) * bw
            canvas = layer if canvas is None else canvas + layer
        if idle_w[i] > 0:                          # 静音：整帧淡向闭嘴待机帧（非贴片）
            canvas = canvas * (1.0 - idle_w[i]) + idle_still * idle_w[i]
        writer.write(np.clip(canvas, 0, 255).astype(np.uint8))   # 原视频帧，无嘴部贴片
    writer.release()

    subprocess.run([_ffmpeg(), "-y", "-i", silent, "-i", mp3, "-c:v", "copy",
                    "-c:a", "aac", "-shortest", out_mp4], check=True, capture_output=True)
    for p in (silent,):
        try:
            os.remove(p)
        except OSError:
            pass


# ---------------------------- HTTP ----------------------------
app = FastAPI(title="本地新老师数字人引擎", version="0.1.0")


class RenderRequest(BaseModel):
    audio: str = ""
    text: str = ""
    gesture: str | None = None
    fps: int = 0
    bbox_shift: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "local-cpu", "fps": _fps,
            "clips": {k: len(v) for k, v in _clips.items()}}


@app.post("/render")
def render(req: RenderRequest) -> dict:
    if not req.audio:
        raise HTTPException(status_code=400, detail="audio(base64) 不能为空")
    audio = base64.b64decode(req.audio)
    out = os.path.join(tempfile.mkdtemp(prefix="lavout_"), f"{uuid.uuid4().hex}.mp4")
    gesture = req.gesture if req.gesture in _clips else None
    t0 = time.time()
    with _lock:
        _render(audio, out, gesture)
    data = Path(out).read_bytes()
    return {"mode": "video", "video": base64.b64encode(data).decode("ascii"),
            "mime": "video/mp4", "text": req.text, "model": "local-cpu",
            "latency_ms": int((time.time() - t0) * 1000)}


def main() -> None:
    import uvicorn

    _init()
    print(f"[local-avatar] http://{HOST}:{PORT}  (POST /render)", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
