"""MuseTalk 1.5 实时口型数字人渲染服务（L3 自由问答层首选引擎）。

对应《AI 教师自研项目立项方案》3.1：实时口型数字人首选 **MuseTalk 1.5**（开源、MIT 可商用，
V100 级显卡 30fps+ 实时推理），是当前速度与质量最均衡的开源实时口型方案。
本引擎替换旧的 Wav2Lip（`scripts/avatar_engine_wav2lip.py` 保留为兼容兜底）。

与 wav2lip 引擎一样，在**独立** conda 环境（torch + MuseTalk 依赖）中运行，与主项目隔离：

    set MUSETALK_DIR=D:\\MuseTalk
    python scripts/avatar_engine_musetalk.py

监听 127.0.0.1:8012，符合本项目最小契约（与 models/avatar.py 的 MuseTalkAvatar 对齐）：
    请求  POST /render  {"audio": "<base64 音频>", "text": "...", "model": "musetalk-1.5", "fps": 30}
    响应  {"mode":"video","video":"<base64 mp4>","mime":"video/mp4","text":...}

工程提示（立项方案 3.1）：消费级显卡上画质与延迟不可兼得，服务端建议 **L40S / A10 起步**，
按并发班数横向扩容（多起几个引擎实例，前面用负载均衡，`.env` 的 AVATAR_API_BASE 指向 LB 即可）。

依赖：需先克隆 MuseTalk 仓库（MUSETALK_DIR）并下载权重（musetalk/pytorch_model.bin 等），
本脚本通过 sys.path 引入 MuseTalk 自带推理工具，逐帧重建口型区域，输出自带音轨的 mp4。
"""
from __future__ import annotations

import base64
import hashlib
import json
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
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ------------------------------------------------------------------ 路径与配置
ROOT = Path(__file__).resolve().parent.parent                        # education/
MUSETALK_DIR = Path(os.getenv("MUSETALK_DIR", r"D:\MuseTalk"))
MODEL_NAME = os.getenv("MUSETALK_MODEL", "musetalk-1.5")             # 立项方案 3.1 首选版本
# 动作基元库目录：主底片 talk.mp4 + 手势 nod/shake/shrug/point.mp4
# （须同机位/光线/背景/服装/分辨率/帧率，切换才不"跳"，见 数字人素材规格.md）。
def _abs_dir(value: str) -> Path:
    """把（可能相对的）路径解析为绝对路径：相对路径按项目根 ROOT 解析。

    必要性：``_init()`` 会 ``os.chdir(MUSETALK_DIR)``，若不在此刻固化绝对路径，
    后续 ``CLIP_DIR / "x.mp4"`` 会相对新工作目录解析而找不到文件。
    """
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p)


# 动作基元库目录：主底片 talk.mp4 + 手势 nod/shake/shrug/point.mp4（+ 可选 idle.mp4）。
CLIP_DIR = _abs_dir(os.getenv("AVATAR_CLIP_DIR", str(ROOT / "data" / "ai_teacher")))
# 主底片（讲解循环）；默认用规范化命名的 talk.mp4，仍可用 AVATAR_FACE_VIDEO 覆盖为任意路径。
FACE_VIDEO = str(_abs_dir(os.getenv("AVATAR_FACE_VIDEO", str(CLIP_DIR / "talk.mp4"))))
# 动作基元名：请求 RenderRequest.gesture 命中则播放对应 {name}.mp4，播完交叉淡化回主底片。
# idle（待机/倾听）为可选：不叠加“动作”，用于倾听/思考态；缺失自动跳过。
ACTION_CLIPS = ("nod", "shake", "shrug", "point", "idle")
# 人脸预处理缓存根目录（按 video_hash 分子目录，换源自动失效重建）。
CACHE_DIR = _abs_dir(os.getenv("AVATAR_CACHE_DIR", str(ROOT / "data" / "cache")))
CACHE_VERSION = "v1"                    # 缓存格式版本；结构变更时递增以强制重建
PORT = int(os.getenv("ENGINE_PORT", "8012"))
HOST = os.getenv("ENGINE_HOST", "127.0.0.1")                         # 设 0.0.0.0 供局域网/多实例扩容
BATCH = int(os.getenv("MUSETALK_BATCH", "8"))                        # 单次送入 UNet 的帧数
IMG_SIZE = 256                                                       # MuseTalk 人脸裁剪尺寸（高于 Wav2Lip 的 96）
# 人脸框上下平移（正下负上，官方推荐 [-10, 10]）；可被单次请求 RenderRequest.bbox_shift 覆盖。
BBOX_SHIFT = int(os.getenv("MUSETALK_BBOX_SHIFT", "0"))
# 手势 <-> 主底片 的交叉淡化帧数（越大越"丝滑"但过渡越慢）；0 = 关闭淡化直接切。
CROSSFADE_FRAMES = int(os.getenv("MUSETALK_CROSSFADE", "6"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

sys.path.insert(0, str(MUSETALK_DIR))                                # 让 musetalk.* 可导入


class _Clip:
    """一段底片：帧序列（RGB）+ 每帧人脸框 + 独立取帧游标（讲解循环跨请求连续、不重启）。"""

    __slots__ = ("name", "frames", "boxes", "fps", "cursor")

    def __init__(self, name: str, frames: list[np.ndarray], fps: float) -> None:
        self.name = name
        self.frames = frames
        self.boxes: list[tuple[int, int, int, int]] = []   # 每帧人脸框 (x1,y1,x2,y2)
        self.fps = fps
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.frames)


# ------------------------------------------------------------------ 启动时一次性加载
_lock = threading.Lock()          # 模型非线程安全，串行化渲染
_audio_processor = None           # Whisper 音频特征提取器
_vae = None
_unet = None
_pe = None                        # pose extractor
_clips: dict[str, _Clip] = {}     # 基元库：talk(主底片) + 各手势片段
_fps = 24.0                       # 主底片帧率（所有基元须一致，否则切换会变速）


def _load_models() -> None:
    """加载 MuseTalk 全套模型（audio_processor/vae/unet/pe）。

    使用 MuseTalk 仓库自带的 ``load_all_model``；不同版本若签名不同，
    在此处对齐一次即可（这是唯一与 MuseTalk 版本耦合的入口）。
    """
    global _audio_processor, _vae, _unet, _pe
    from musetalk.utils.utils import load_all_model  # noqa: PLC0415

    # 不同版本返回 3 个(audio_processor, vae, unet)或 4 个(多一个 pose extractor)
    models = load_all_model()
    _audio_processor, _vae, _unet = models[0], models[1], models[2]
    _pe = models[3] if len(models) > 3 else None


def _read_clip_frames(path: str) -> tuple[list[np.ndarray], float]:
    """读取一段底片帧（RGB）与其 fps。"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"底片无有效帧：{path}")
    return frames, float(fps)


def _detect_boxes(clip: _Clip) -> None:
    """对一段底片逐帧检测人脸框（一次性），用于裁剪口型重建区域。

    使用 MuseTalk 自带的 landmark/bbox 预处理；版本差异在此对齐。

    实现要点：
    * MuseTalk API ``get_landmark_and_bbox(img_list, upperbondrange=0)`` 要求传入 **文件路径列表**（内部会 ``cv2.imread``），
      不接受 numpy 数组。这里把 ``clip.frames`` 临时写为 PNG（BGR）后传路径。
    * ``upperbondrange`` 就是 MuseTalk 官方的 **bbox_shift**（正下负上），在 landmark 阶段就调整 half_face_coord，
      比“检测后再平移框”更准。直接把引擎侧 ``BBOX_SHIFT`` 透传。
    * 返回值是 ``(coords_list, frames)``——coords_list 才是人脸框；旧实现写反了（当 landmarks 用）。
    * 无人脸帧会得到 ``coord_placeholder=(0.0,0.0,0.0,0.0)``；遇到时退回全图作兵底（避免后续 crop 崩）。
    """
    from musetalk.utils.preprocessing import get_landmark_and_bbox  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="mt_frames_") as tmpdir:
        paths: list[str] = []
        for i, frame in enumerate(clip.frames):
            p = os.path.join(tmpdir, f"{i:06d}.png")
            # clip.frames 存的是 RGB（见 _read_clip_frames），cv2.imwrite 需 BGR
            cv2.imwrite(p, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            paths.append(p)
        # upperbondrange = 官方 bbox_shift（正下负上）
        coords_list, _ = get_landmark_and_bbox(paths, upperbondrange=BBOX_SHIFT)

    h, w = clip.frames[0].shape[:2]
    clip.boxes.clear()
    last_good: tuple[int, int, int, int] = (0, 0, w, h)
    for c in coords_list:
        x1, y1, x2, y2 = (int(v) for v in c)
        if x2 - x1 <= 0 or y2 - y1 <= 0 or x1 < 0:
            # coord_placeholder 或异常框 -> 沿用上一个有效框（底片通常全程有脸，开首就失效时退回全图）
            clip.boxes.append(last_good)
            continue
        # clip 到画面内
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        last_good = (x1, y1, x2, y2)
        clip.boxes.append(last_good)


# ------------------------------------------------------------------ 人脸预处理缓存
def _file_hash(path: str) -> str:
    """文件内容 SHA-256 前 16 位，作为缓存 key；内容一变 key 即变（换源自动失效）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _cache_file(clip_path: str) -> Path:
    """人脸框缓存文件：``<CACHE_DIR>/<版本>/<内容hash>_s<bbox_shift>.json``。

    把 bbox_shift 纳入文件名：改动人脸框平移量时旧缓存不再复用。
    """
    return CACHE_DIR / CACHE_VERSION / f"{_file_hash(clip_path)}_s{BBOX_SHIFT}.json"


def _load_boxes_cache(clip_path: str, clip: _Clip) -> list[tuple[int, int, int, int]] | None:
    """命中有效缓存则返回人脸框列表；否则返回 None（由调用方重建）。"""
    p = _cache_file(clip_path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if int(data.get("frame_count", -1)) != len(clip.frames):
            return None
        boxes = [tuple(int(v) for v in b) for b in data.get("boxes", [])]
        return boxes if len(boxes) == len(clip.frames) else None
    except Exception:  # noqa: BLE001 - 缓存损坏则忽略并重建
        return None


def _save_boxes_cache(clip_path: str, clip: _Clip) -> None:
    """把人脸框连同元信息写入缓存，便于换源/改 bbox_shift 时校验失效。"""
    p = _cache_file(clip_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        h, w = clip.frames[0].shape[:2]
        payload = {
            "version": CACHE_VERSION,
            "video": Path(clip_path).name,
            "bbox_shift": BBOX_SHIFT,
            "fps": clip.fps,
            "frame_count": len(clip.frames),
            "size": [w, h],
            "boxes": [list(b) for b in clip.boxes],
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as err:  # noqa: BLE001 - 缓存写失败不影响渲染
        print(f"[musetalk] 人脸缓存写入失败（忽略）：{err}", flush=True)


def _load_clip(name: str, path: str) -> _Clip:
    """加载一段基元：读帧 + 人脸框（优先命中 video_hash 缓存，换源自动重建）。"""
    frames, fps = _read_clip_frames(path)
    clip = _Clip(name, frames, fps)
    cached = _load_boxes_cache(path, clip)
    if cached is not None:
        clip.boxes = cached
        print(f"[musetalk] {name}: 命中人脸缓存 {_cache_file(path).name}（{len(cached)} 帧）", flush=True)
    else:
        print(f"[musetalk] {name}: 正在为 AI 教师建立人脸缓存...（{len(frames)} 帧）", flush=True)
        _detect_boxes(clip)
        _save_boxes_cache(path, clip)
    return clip


def _init() -> None:
    global _fps
    print(f"[musetalk] device={DEVICE} model={MODEL_NAME} dir={MUSETALK_DIR} "
          f"clip_dir={CLIP_DIR} cache_dir={CACHE_DIR} bbox_shift={BBOX_SHIFT} "
          f"crossfade={CROSSFADE_FRAMES}", flush=True)
    if not MUSETALK_DIR.is_dir():
        raise SystemExit(f"未找到 MuseTalk 仓库：{MUSETALK_DIR}（请设 MUSETALK_DIR）")
    os.chdir(MUSETALK_DIR)      # MuseTalk 内部用相对路径 models/...，必须切到仓库根目录
    _load_models()
    talk = _load_clip("talk", FACE_VIDEO)
    _clips["talk"] = talk
    _fps = talk.fps
    for g in ACTION_CLIPS:                        # 动作基元可选：缺失只警告不阻断启动
        p = CLIP_DIR / f"{g}.mp4"
        if p.is_file():
            _clips[g] = _load_clip(g, str(p))
        else:
            print(f"[musetalk] 动作基元缺失，跳过：{p}", flush=True)
    # 问候/打招呼（gesture="greet"）：复用 idle.mp4（老师打招呼动作）；缺失则降级为纯讲解
    if "greet" not in _clips and "idle" in _clips:
        _clips["greet"] = _clips["idle"]
    primitives = [k for k in _clips if k != "talk"]
    print(f"[musetalk] talk frames={len(talk)} fps={_fps:.2f} "
          f"size={talk.frames[0].shape[1]}x{talk.frames[0].shape[0]} primitives={primitives}", flush=True)
    print("[musetalk] 就绪", flush=True)


# ------------------------------------------------------------------ 音频特征
def _audio_to_features(wav_path: str, fps: float) -> list[np.ndarray]:
    """音频 -> Whisper 特征 -> 按帧切块（每帧一个 whisper chunk），供 UNet 条件输入。

    ``fps`` 必须与输出 mp4 容器帧率一致：否则 chunk 数量与帧数不匹配，
    会出现画面加速/口型对不上。旧实现写死底片 ``_fps``，导致客户端 fps_override 只影响
    容器、不影响 chunk 数 -> 这里修正为使用有效 fps。
    """
    whisper_features = _audio_processor.audio2feature([wav_path])
    # feature2chunks：把 whisper 特征按视频帧率切成逐帧条件块
    whisper_chunks = _audio_processor.feature2chunks(whisper_features, fps / 25)
    return whisper_chunks[0] if whisper_chunks and isinstance(whisper_chunks[0], list) else whisper_chunks


def _apply_bbox_shift(
    box: tuple[int, int, int, int], shift: int, height: int
) -> tuple[int, int, int, int]:
    """对人脸框作上下平移并 clip 到画面内。

    MuseTalk 语义：正值向下（多包下巴/脖子），负值向上（多包额头）。
    """
    if not shift:
        return box
    x1, y1, x2, y2 = box
    y1 = max(0, y1 + shift)
    y2 = min(height, y2 + shift)
    if y2 <= y1:  # 极端 shift 导致框反转，退回原框
        return box
    return (x1, y1, x2, y2)


# ------------------------------------------------------------------ 帧规划（手势切换 + 交叉淡化）
def _bounce(idx: int, n: int) -> int:
    """往复取帧索引：0..n-1..0..，避免手势片段循环播放时的硬切跳变。"""
    if n <= 1:
        return 0
    cycle = 2 * (n - 1)
    m = idx % cycle
    return m if m < n else cycle - m


def _plan_frames(
    n_frames: int, gesture: str | None
) -> list[list[tuple[_Clip, int, float]]]:
    """规划每个输出帧的来源，返回长度 ``n_frames`` 的列表。

    每个元素是该输出帧的加权采样 ``[(clip, frame_idx, weight), ...]``（weight 之和=1）：
    * 单源帧 -> ``[(clip, idx, 1.0)]``；
    * 交叉淡化帧 -> ``[(clipA, idxA, wA), (clipB, idxB, wB)]``，合成时按权重叠加两张画布。

    有手势时：主底片 --淡入--> 手势（往复播放）--淡出--> 主底片；无手势时主底片从其
    游标处连续取帧（跨请求不重启）。主底片游标在手势期间照常推进，淡出时无缝衔接。
    """
    talk = _clips["talk"]
    g = _clips.get(gesture) if gesture else None
    plan: list[list[tuple[_Clip, int, float]]] = []
    if g is None or g is talk:                         # 无手势：主底片连续循环
        for i in range(n_frames):
            plan.append([(talk, (talk.cursor + i) % len(talk), 1.0)])
        talk.cursor = (talk.cursor + n_frames) % len(talk)
        return plan

    # 淡化帧数不超过 总帧/手势长度 的一半，避免整段都在过渡；CROSSFADE_FRAMES=0 则直接切
    k = max(0, min(CROSSFADE_FRAMES, n_frames // 2, len(g) // 2))
    for i in range(n_frames):
        t_idx = (talk.cursor + i) % len(talk)
        if i < k:                                      # 淡入：talk -> gesture
            alpha = (i + 1) / (k + 1)
            plan.append([(talk, t_idx, 1.0 - alpha), (g, _bounce(i, len(g)), alpha)])
        elif i < n_frames - k:                         # 手势主体（往复播放）
            plan.append([(g, _bounce(i - k, len(g)), 1.0)])
        else:                                          # 淡出：gesture -> talk
            alpha = (i - (n_frames - k) + 1) / (k + 1)
            plan.append([(g, _bounce(i - k, len(g)), 1.0 - alpha), (talk, t_idx, alpha)])
    talk.cursor = (talk.cursor + n_frames) % len(talk)
    return plan


# ------------------------------------------------------------------ 逐帧推理
def _render(
    audio_bytes: bytes,
    out_mp4: str,
    fps_override: float | None = None,
    bbox_shift_override: int | None = None,
    gesture: str | None = None,
) -> None:
    """核心渲染：基元库帧 + 音频 -> 逐帧重建口型 -> 加权合成 -> 带音轨 mp4。

    * ``fps_override``：客户端指定的输出帧率（None/0 = 沿用底片 fps）。
      同时驱动 whisper chunk 切块与 mp4 容器帧率，两者必须一致。
    * ``bbox_shift_override``：客户端指定的人脸框平移量（None = 用引擎侧 ``BBOX_SHIFT`` 默认）。
    * ``gesture``：本次播报要叠加的手势基元名（nod/shake/shrug/point）；None = 纯讲解。
    """
    from musetalk.utils.blending import get_image_blending  # noqa: PLC0415

    fps = fps_override or _fps
    # 人脸框已在 _detect_boxes 阶段用 BBOX_SHIFT 经 MuseTalk 原生 upperbondrange 调过，
    # 客户端传入的 bbox_shift_override 在这里以 **delta** 方式叠加（避免重复平移）。
    delta_shift = 0 if bbox_shift_override is None else int(bbox_shift_override) - BBOX_SHIFT
    tmp = tempfile.mkdtemp(prefix="mt_")
    in_audio = os.path.join(tmp, "in.mp3")
    Path(in_audio).write_bytes(audio_bytes)
    wav16 = os.path.join(tmp, "a.wav")
    subprocess.run([FFMPEG, "-y", "-i", in_audio, "-ac", "1", "-ar", "16000", wav16],
                   check=True, capture_output=True)

    whisper_chunks = _audio_to_features(wav16, fps)
    n_frames = len(whisper_chunks)                     # 音频决定输出帧数
    plan = _plan_frames(n_frames, gesture)

    # 展平所有采样：一个输出帧可能含 1~2 个采样（交叉淡化）；口型由该输出帧对应的音频驱动。
    samples: list[tuple[int, _Clip, int, float]] = []  # (out_i, clip, frame_idx, weight)
    for out_i, spec in enumerate(plan):
        for clip, idx, w in spec:
            samples.append((out_i, clip, idx, w))

    silent = os.path.join(tmp, "silent.mp4")
    writer = imageio.get_writer(silent, fps=fps, codec="libx264", quality=8, macro_block_size=1)

    face_batch: list[np.ndarray] = []                  # 裁剪并归一化的人脸块
    feat_batch: list[np.ndarray] = []                  # 各采样对应的 whisper 条件特征
    meta_batch: list[tuple[int, _Clip, int, tuple[int, int, int, int], float]] = []

    # 增量合成：按 out_i 升序累加加权画布，out_i 变化即说明上一帧已完整 -> 立即写出（省内存）。
    acc: np.ndarray | None = None
    acc_i = -1

    def emit(out_i: int, canvas: np.ndarray, weight: float) -> None:
        nonlocal acc, acc_i
        if out_i != acc_i:
            if acc is not None:
                writer.append_data(np.clip(acc, 0, 255).astype(np.uint8))
            acc = canvas.astype(np.float32) * weight
            acc_i = out_i
        else:
            acc += canvas.astype(np.float32) * weight

    def flush() -> None:
        if not face_batch:
            return
        faces = torch.from_numpy(
            np.asarray(face_batch, dtype=np.float32).transpose(0, 3, 1, 2)
        ).to(DEVICE)
        feats = torch.from_numpy(np.asarray(feat_batch, dtype=np.float32)).to(DEVICE)
        with torch.no_grad():
            pred = _unet.infer(faces, feats)           # MuseTalk UNet：人脸块 + 音频条件 -> 重建口型
        pred = pred.cpu().numpy()
        for face_out, (out_i, clip, idx, box, weight) in zip(pred, meta_batch):
            x1, y1, x2, y2 = box
            canvas = clip.frames[idx].copy()
            resized = cv2.resize(face_out.astype(np.uint8), (x2 - x1, y2 - y1))
            canvas[y1:y2, x1:x2] = get_image_blending(canvas, resized, box)
            emit(out_i, canvas, weight)
        face_batch.clear()
        feat_batch.clear()
        meta_batch.clear()

    for out_i, clip, idx, weight in samples:
        frame = clip.frames[idx]
        box = _apply_bbox_shift(clip.boxes[idx], delta_shift, frame.shape[0])
        x1, y1, x2, y2 = box
        face = cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE))
        face_batch.append(face / 255.0)                # 归一化到 [0,1]
        feat_batch.append(np.asarray(whisper_chunks[out_i], dtype=np.float32))
        meta_batch.append((out_i, clip, idx, box, weight))
        if len(face_batch) >= BATCH:
            flush()
    flush()
    if acc is not None:                                # 写出最后一帧
        writer.append_data(np.clip(acc, 0, 255).astype(np.uint8))
    writer.close()

    # 合流音轨：视频只存画面，音轨用输入音频（口型已与之同步）
    subprocess.run([FFMPEG, "-y", "-i", silent, "-i", in_audio, "-c:v", "copy",
                    "-c:a", "aac", "-shortest", out_mp4], check=True, capture_output=True)


# ------------------------------------------------------------------ HTTP 服务
app = FastAPI(title="MuseTalk 数字人渲染服务", version="0.1.0")


class RenderRequest(BaseModel):
    audio: str = ""
    text: str = ""
    model: str = MODEL_NAME
    fps: int = 0                 # 0 表示沿用底片 fps；非 0 时需与底片一致，否则音画错位
    # None 表示沿用引擎侧 MUSETALK_BBOX_SHIFT 环境变量默认值；传入则本次请求覆盖
    bbox_shift: int | None = None
    # 本次播报叠加的动作基元名（nod/shake/shrug/point/idle）；None/未知 = 纯讲解
    gesture: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "musetalk", "model": MODEL_NAME, "device": DEVICE,
            "fps": _fps, "bbox_shift": BBOX_SHIFT, "crossfade": CROSSFADE_FRAMES,
            "clips": {name: len(clip) for name, clip in _clips.items()}}


@app.post("/render")
def render(req: RenderRequest) -> dict:
    if not req.audio:
        raise HTTPException(status_code=400, detail="audio(base64) 不能为空")
    audio = base64.b64decode(req.audio)
    out_mp4 = os.path.join(tempfile.mkdtemp(prefix="mtout_"), f"{uuid.uuid4().hex}.mp4")
    # 未知手势名降级为纯讲解（不报错），避免编排端下发未加载的基元时中断播报
    gesture = req.gesture if req.gesture in _clips else None
    t0 = time.time()
    with _lock:
        _render(
            audio,
            out_mp4,
            fps_override=float(req.fps) if req.fps else None,
            bbox_shift_override=req.bbox_shift,
            gesture=gesture,
        )
    data = Path(out_mp4).read_bytes()
    return {"mode": "video", "video": base64.b64encode(data).decode("ascii"),
            "mime": "video/mp4", "text": req.text, "model": MODEL_NAME,
            "latency_ms": int((time.time() - t0) * 1000)}


def main() -> None:
    import uvicorn

    _init()
    print(f"[musetalk] http://{HOST}:{PORT}  (POST /render)", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
