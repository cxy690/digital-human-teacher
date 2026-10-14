"""Wav2Lip 数字人渲染服务（真唇形同步）。

在**独立** conda 环境（torch cu128 + torchvision + librosa + imageio-ffmpeg）中运行，
与主项目 tea 环境隔离：

    set WAV2LIP_DIR=D:\\wav2lip
    python scripts/avatar_engine_wav2lip.py

监听 127.0.0.1:8011，符合本项目最小契约：
    请求  POST /render  {"audio": "<base64 音频>", "text": "这一句的话术"}
    响应  {"mode":"video","video":"<base64 mp4>","mime":"video/mp4","text":...}

用教师底片（默认 data/raw/teacher_base.mp4）做基底，用 Wav2Lip 逐帧重建嘴部区域，
使口型与传入语音逐字同步；底片帧数不足音频时长时循环取帧。输出 mp4 自带音轨。
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
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import librosa
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ------------------------------------------------------------------ 路径与配置
ROOT = Path(__file__).resolve().parent.parent                      # education/
WAV2LIP_DIR = Path(os.getenv("WAV2LIP_DIR", r"D:\wav2lip"))
CHECKPOINT = os.getenv("WAV2LIP_CHECKPOINT", str(WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"))
FACE_VIDEO = os.getenv("AVATAR_FACE_VIDEO", str(ROOT / "data" / "ai_teacher" / "talk.mp4"))
CLIP_DIR = Path(os.getenv("AVATAR_CLIP_DIR", str(ROOT / "data" / "ai_teacher")))
# 问候/打招呼片段：仅当话术含“你好”类问候词时，用它作底片对口型（可 .env 配 AVATAR_GREET_CLIP）
GREETING_CLIP = Path(os.getenv("AVATAR_GREET_CLIP", str(CLIP_DIR / "idle.mp4")))
PORT = int(os.getenv("ENGINE_PORT", "8011"))
HOST = os.getenv("ENGINE_HOST", "127.0.0.1")              # 设 0.0.0.0 可让局域网内其他机器访问
RESIZE = max(1, int(os.getenv("WAV2LIP_RESIZE", "2")))             # 降分辨率加速（1=原图）
BATCH = int(os.getenv("WAV2LIP_BATCH", "64"))
IMG_SIZE = 96
MEL_STEP = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DET_DEVICE = os.getenv("WAV2LIP_DET_DEVICE", "cpu")           # 人脸检测(一次性)默认 CPU；配合降采样避免显存/内存峰值
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()                           # imageio 自带 ffmpeg，免装

sys.path.insert(0, str(WAV2LIP_DIR))                              # 让 face_detection / models 可导入

# ------------------------------------------------------------------ 梅尔谱（对齐 Wav2Lip hparams）
# 注意：librosa>=0.10 起 librosa.filters.mel 的 sr / n_fft 变为关键字参数，
# 官方 audio.py 用位置传参会报错，这里用关键字参数自行实现，避免依赖官方 audio.py。
from scipy import signal as _signal

_HP = dict(n_fft=800, hop=200, win=800, sr=16000, n_mels=80,
           fmin=55, fmax=7600, preemph=0.97,
           min_level_db=-100, ref_level_db=20, max_abs=4.0)
_mel_basis = None


def _melspectrogram(wav: np.ndarray) -> np.ndarray:
    global _mel_basis
    y = _signal.lfilter([1, -_HP["preemph"]], [1], wav)            # pre-emphasis
    D = librosa.stft(y=y, n_fft=_HP["n_fft"], hop_length=_HP["hop"], win_length=_HP["win"])
    S = np.abs(D)
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(sr=_HP["sr"], n_fft=_HP["n_fft"],
                                         n_mels=_HP["n_mels"], fmin=_HP["fmin"], fmax=_HP["fmax"])
    S = np.dot(_mel_basis, S)
    min_level = np.exp(_HP["min_level_db"] / 20 * np.log(10))
    S = 20 * np.log10(np.maximum(min_level, S)) - _HP["ref_level_db"]
    # symmetric + clip 归一化（与官方 hparams 默认一致）
    S = np.clip(2 * _HP["max_abs"] * ((S - _HP["min_level_db"]) / (-_HP["min_level_db"])) - _HP["max_abs"],
                -_HP["max_abs"], _HP["max_abs"])
    return S


# ------------------------------------------------------------------ 启动时一次性加载
_lock = threading.Lock()
_model = None
_frames: list[np.ndarray] = []              # 主底片（talk）
_boxes: list[tuple[int, int, int, int]] = []
_greet_frames: list[np.ndarray] = []        # 问候/打招呼底片（可选）
_greet_boxes: list[tuple[int, int, int, int]] = []
_fps = 24.0
_frame_cursor = 0          # 跨请求连续取帧游标：讲解中画面动作连续、不逐步重启


def _load_model():
    from models import Wav2Lip
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model = Wav2Lip()
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def _read_frames(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if RESIZE > 1:
            frame = cv2.resize(frame, (frame.shape[1] // RESIZE, frame.shape[0] // RESIZE))
        if frame.shape[1] % 2 or frame.shape[0] % 2:               # libx264 要求宽高为偶数
            frame = frame[:frame.shape[0] - frame.shape[0] % 2, :frame.shape[1] - frame.shape[1] % 2]
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"底片无有效帧：{path}")
    return frames, float(fps)


def _detect_boxes(frames: list[np.ndarray], pads=(0, 10, 0, 0), det_batch: int = 4, det_scale: int = 2) -> list[tuple[int, int, int, int]]:
    import face_detection
    # 人脸检测只跑一次：先对帧再降采样(det_scale)检测以大幅省显存/内存，再把框坐标还原
    det = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                       flip_input=False, device=DET_DEVICE)
    h, w = frames[0].shape[:2]
    small = [cv2.resize(f, (max(1, w // det_scale), max(1, h // det_scale))) for f in frames]
    raw: list = []
    for s in range(0, len(small), det_batch):
        raw.extend(det.get_detections_for_batch(np.array(small[s:s + det_batch])))
    pady1, pady2, padx1, padx2 = pads
    boxes = []
    for rect in raw:
        if rect is None:
            raise RuntimeError("底片有帧未检测到人脸，请换更清晰的正脸底片")
        x1, y1, x2, y2 = (int(v * det_scale) for v in rect)    # 还原到原始帧尺寸
        boxes.append([max(0, x1 - padx1), max(0, y1 - pady1), min(w, x2 + padx2), min(h, y2 + pady2)])
    arr = np.array(boxes, dtype=np.float64)
    # 时序平滑（T=5），与官方一致，减少抖动
    for i in range(len(arr)):
        win = arr[len(arr) - 5:] if i + 5 > len(arr) else arr[i:i + 5]
        arr[i] = np.mean(win, axis=0)
    return [tuple(int(v) for v in b) for b in arr]           # (x1,y1,x2,y2)


def _init() -> None:
    global _model, _fps, _frames, _boxes, _greet_frames, _greet_boxes
    print(f"[engine] device={DEVICE} ffmpeg={FFMPEG}", flush=True)
    print(f"[engine] checkpoint={CHECKPOINT}", flush=True)
    if not os.path.isfile(CHECKPOINT):
        raise SystemExit(f"缺少权重文件：{CHECKPOINT}")
    _model = _load_model()
    _frames, _fps = _read_frames(FACE_VIDEO)
    _boxes = _detect_boxes(_frames)
    if GREETING_CLIP.is_file():        # 问候/打招呼底片（可选）：含“你好”类问候时对口型用
        _greet_frames, _ = _read_frames(str(GREETING_CLIP))
        _greet_boxes = _detect_boxes(_greet_frames)
        print(f"[engine] greet clip: {GREETING_CLIP.name} frames={len(_greet_frames)}", flush=True)
    else:
        print(f"[engine] greet clip not found: {GREETING_CLIP}", flush=True)
    print(f"[engine] base frames={len(_frames)} fps={_fps:.2f} size={_frames[0].shape[1]}x{_frames[0].shape[0]}", flush=True)
    print("[engine] 就绪", flush=True)


# ------------------------------------------------------------------ 逐帧推理
def _render(audio_bytes: bytes, out_mp4: str, gesture: str | None = None) -> None:
    global _frame_cursor
    tmp = tempfile.mkdtemp(prefix="w2l_")
    in_audio = os.path.join(tmp, "in.mp3")
    Path(in_audio).write_bytes(audio_bytes)
    wav16 = os.path.join(tmp, "a.wav")
    subprocess.run([FFMPEG, "-y", "-i", in_audio, "-ac", "1", "-ar", "16000", wav16],
                   check=True, capture_output=True)
    wav, _ = librosa.load(wav16, sr=16000)
    mel = _melspectrogram(wav)
    mel = np.nan_to_num(mel)

    mult = 80.0 / _fps
    chunks: list[np.ndarray] = []
    i = 0
    while True:
        start = int(i * mult)
        if start + MEL_STEP > mel.shape[1]:
            chunks.append(mel[:, mel.shape[1] - MEL_STEP:])
            break
        chunks.append(mel[:, start:start + MEL_STEP])
        i += 1

    use_greet = gesture == "greet" and len(_greet_frames) > 0
    base_frames = _greet_frames if use_greet else _frames
    base_boxes = _greet_boxes if use_greet else _boxes
    n = len(base_frames)
    silent = os.path.join(tmp, "silent.mp4")
    writer = imageio.get_writer(silent, fps=_fps, codec="libx264", quality=8, macro_block_size=1)

    img_b, mel_b, fr_b, co_b = [], [], [], []

    def flush() -> None:
        if not img_b:
            return
        img = np.asarray(img_b)
        mb = np.asarray(mel_b)
        masked = img.copy()
        masked[:, IMG_SIZE // 2:] = 0
        inp = np.concatenate((masked, img), axis=3) / 255.0
        mb = mb.reshape(len(mb), mb.shape[1], mb.shape[2], 1)
        t_img = torch.FloatTensor(np.transpose(inp, (0, 3, 1, 2))).to(DEVICE)
        t_mel = torch.FloatTensor(np.transpose(mb, (0, 3, 1, 2))).to(DEVICE)
        with torch.no_grad():
            pred = _model(t_mel, t_img)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        for p, f, c in zip(pred, fr_b, co_b):
            x1, y1, x2, y2 = c
            patch = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = patch
            writer.append_data(np.asarray(f)[:, :, ::-1])          # BGR -> RGB
        img_b.clear(); mel_b.clear(); fr_b.clear(); co_b.clear()

    for idx_i, m in enumerate(chunks):
        # 问候：从打招呼片段第 0 帧开始（做一次动作）；讲解：从上次结束处继续，画面连续
        fi = idx_i % n if use_greet else (_frame_cursor + idx_i) % n
        frame = base_frames[fi].copy()
        x1, y1, x2, y2 = base_boxes[fi]
        face = cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
        img_b.append(face); mel_b.append(m); fr_b.append(frame); co_b.append((x1, y1, x2, y2))
        if len(img_b) >= BATCH:
            flush()
    flush()
    writer.close()
    if not use_greet:
        _frame_cursor = (_frame_cursor + len(chunks)) % n   # 推进游标，供下一句无缝衔接

    subprocess.run([FFMPEG, "-y", "-i", silent, "-i", in_audio, "-c:v", "copy",
                    "-c:a", "aac", "-shortest", out_mp4], check=True, capture_output=True)


# ------------------------------------------------------------------ HTTP 服务
app = FastAPI(title="Wav2Lip 数字人渲染服务", version="0.1.0")


class RenderRequest(BaseModel):
    audio: str = ""
    text: str = ""
    gesture: str | None = None          # "greet" 时用打招呼底片（对口型）


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "wav2lip", "device": DEVICE,
            "frames": len(_frames), "greet_frames": len(_greet_frames),
            "fps": _fps, "resize": RESIZE}


@app.post("/render")
def render(req: RenderRequest) -> dict:
    if not req.audio:
        raise HTTPException(status_code=400, detail="audio(base64) 不能为空")
    audio = base64.b64decode(req.audio)
    out_mp4 = os.path.join(tempfile.mkdtemp(prefix="w2lout_"), f"{uuid.uuid4().hex}.mp4")
    t0 = time.time()
    with _lock:                                                    # 模型非线程安全，串行化
        _render(audio, out_mp4, gesture=req.gesture)
    data = Path(out_mp4).read_bytes()
    return {"mode": "video", "video": base64.b64encode(data).decode("ascii"),
            "mime": "video/mp4", "text": req.text,
            "latency_ms": int((time.time() - t0) * 1000)}


def main() -> None:
    import uvicorn
    _init()
    print(f"[engine] http://{HOST}:{PORT}  (POST /render)", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
