"""MuseTalk v1.5 引擎（适配**新版 API**，**不依赖 mmcv/mmpose**，支持多底片/手势）。

背景：项目自带的 `scripts/avatar_engine_musetalk.py` 是为**旧版 MuseTalk API**写的
（`models[0]` 直接当 AudioProcessor / `audio2feature()`），与本仓库
（`vae/unet/pe + AudioProcessor.get_audio_feature + WhisperModel`）不兼容，一渲染就 500。
本脚本照抄本仓库 `scripts/inference.py` 的正确流程，并做本地化改造：

1. **绕过 mmcv**：改用 **MediaPipe** 求人脸框，不 import `musetalk.utils.preprocessing`。
2. **常驻服务化**：模型/底片/人脸框/latent/mask 只在启动时加载一次，`/render` 逐句推理。
3. **多底片（手势）**：主底片 `talk` + 问候底片 `greet`（默认 `idle.mp4`）。
   `gesture="greet"` 时用问候底片（从第 0 帧起做一次打招呼动作），其余用主底片连续取帧。

HTTP 契约与项目一致（与 models/avatar.py 的 MuseTalkAvatar/HttpAvatar 对齐）：
    请求  POST /render  {"audio": "<base64 音频>", "text": "...", "gesture": ..., "fps": ..., "bbox_shift": ...}
    响应  {"mode":"video","video":"<base64 mp4>","mime":"video/mp4","text":...}

启动：
    MUSETALK_DIR=/root/education/education/models_zoo/MuseTalk  python scripts/avatar_engine_musetalk_v15.py
"""
from __future__ import annotations

import base64
import hashlib
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 - 非交互 stdout 可能不支持
    pass

# ------------------------------------------------------------------ 路径与配置
def _abs(p: str) -> Path:
    """相对路径按项目根解析（_init 会 chdir(MUSETALK_DIR)，必须提前固化）。"""
    x = Path(p)
    return x if x.is_absolute() else (ROOT / x)


ROOT = Path(__file__).resolve().parent.parent                       # education/
MUSETALK_DIR = Path(os.getenv("MUSETALK_DIR", r"D:\models_zoo\MuseTalk"))
CLIP_DIR = Path(os.getenv("AVATAR_CLIP_DIR", str(ROOT / "data" / "teachers")))
# 旧单底片路径：**默认留空，不再回退到任何 底片.mp4 / talk.mp4**。
# 多老师模式一律走 TEACHERS_DIR/<teacher_id>/talk.mp4（见 _get_clips），
# 仅当显式配置 AVATAR_FACE_VIDEO 时才启用「旧单底片」兼容路径。
TALK_CLIP = os.getenv("AVATAR_FACE_VIDEO", "").strip()
GREET_CLIP = os.getenv("AVATAR_GREET_CLIP", "").strip()
# 多老师：底片位于 <TEACHERS_DIR>/<teacher_id>/talk.mp4（按需懒加载，见 _get_clips）
TEACHERS_DIR = _abs(os.getenv("AVATAR_TEACHERS_DIR", str(ROOT / "data" / "teachers")))
MAX_TEACHERS = int(os.getenv("MUSETALK_MAX_TEACHERS", "2"))        # 同时常驻的教师底片数（LRU 淘汰）
# 启动时是否预加载「旧单底片」；多老师模式默认关闭，省显存与启动时间
LEGACY_PRELOAD = os.getenv("AVATAR_LEGACY_PRELOAD", "0") == "1"
PORT = int(os.getenv("ENGINE_PORT", "8012"))
HOST = os.getenv("ENGINE_HOST", "127.0.0.1")
BATCH = int(os.getenv("MUSETALK_BATCH", "8"))                       # 单批帧数
EXTRA_MARGIN = int(os.getenv("MUSETALK_EXTRA_MARGIN", "10"))        # v1.5 下巴向下扩展像素
PARSING_MODE = os.getenv("MUSETALK_PARSING_MODE", "jaw")            # jaw / raw / neck / lips
# 融合 mask 起点（占裁剪框高度比例）：越大 -> 替换区域越低、保留越多上半脸原貌（官方默认 0.5）
UPPER_BOUNDARY_RATIO = float(os.getenv("MUSETALK_UPPER_BOUNDARY_RATIO", "0.5"))
# lips 模式：嘴部区域外扩 / 边缘羽化强度（占裁剪框宽度比例）
LIPS_DILATE = float(os.getenv("MUSETALK_LIPS_DILATE", "0.06"))
LIPS_BLUR = float(os.getenv("MUSETALK_LIPS_BLUR", "0.05"))
# TTS 首尾静音裁剪：消除「口型整体早于声音」（Edge TTS 常带 0.1~0.3s 前导静音）
TRIM_SILENCE = os.getenv("MUSETALK_TRIM_SILENCE", "1") == "1"
TRIM_DB = float(os.getenv("MUSETALK_TRIM_DB", "-45"))               # 静音判定阈值(dBFS)
TRIM_PAD = float(os.getenv("MUSETALK_TRIM_PAD", "0.04"))            # 首尾保留余量(秒)
# whisper 音频窗口：left 调大 -> 口型更滞后；调小 -> 口型更提前（官方 2/2 已基本对齐）
AUDIO_PAD_LEFT = int(os.getenv("MUSETALK_AUDIO_PAD_LEFT", "2"))
AUDIO_PAD_RIGHT = int(os.getenv("MUSETALK_AUDIO_PAD_RIGHT", "2"))
FP16 = os.getenv("MUSETALK_FP16", "1") == "1"                       # 半精度加速
BBOX_SHIFT = int(os.getenv("MUSETALK_BBOX_SHIFT", "0"))            # 人脸框上下平移（正下负上，官方推荐 [-10,10]）
IMG_SIZE = 256                                                       # MuseTalk 人脸裁剪尺寸
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# 输出视频码率（imageio quality，0~10，越高码率越高）。
# quality=8（≈crf10）时单句可达 10MB —— 公网下载每句要几十秒，播放就会一卡一卡。
# 默认 4（≈crf30）：单句约 2MB，人脸画面肉眼几乎无差别，低带宽也能流畅播。
VIDEO_QUALITY = int(os.getenv("MUSETALK_VIDEO_QUALITY", "4"))
# 预处理结果磁盘缓存目录（按底片指纹分文件）。命中后可跳过「人脸框 + VAE + 分割」三步，
# 把「首次切到某位老师」从 30~60s 降到 ~1s —— 与融合项目离线 genavatar 素材包的收益等价。
CACHE_DIR = _abs(os.getenv("AVATAR_CACHE_DIR", str(ROOT / "data" / "cache"))) / "precompute"

sys.path.insert(0, str(MUSETALK_DIR))   # 让 musetalk 包可导入

# PyTorch>=2.6 默认 weights_only=True；MuseTalk 部分权重是 legacy .tar 格式，强制旧行为加载
_ORIG_TORCH_LOAD = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_trusted

_lock = threading.Lock()
_vae = _unet = _pe = _whisper = _audio_processor = _fp = None
_weight_dtype = torch.float32
# 多老师：teacher_id -> {"talk": _Clip, "greet": _Clip?}；按需加载 + LRU 淘汰
_teacher_clips: dict[str, dict[str, "_Clip"]] = {}
_teacher_lock = threading.Lock()


class _Clip:
    """一段底片及其预计算结果（帧/人脸框/latent/mask/裁剪框）。"""

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path
        self.frames: list[np.ndarray] = []
        self.boxes: list[tuple[int, int, int, int]] = []
        self.latents: list[torch.Tensor] = []
        self.masks: list[np.ndarray] = []
        self.crops: list[tuple[int, int, int, int]] = []
        self.cursor = 0            # 讲解连续取帧游标（问候不用）
        self.fps = 24.0


_clips: dict[str, _Clip] = {}


# ------------------------------------------------------------------ 模型加载
def _load_models() -> None:
    """加载 VAE/UNet/PE/Whisper/FaceParsing（照 inference.py 的 v1.5 用法）。"""
    global _vae, _unet, _pe, _whisper, _audio_processor, _fp, _weight_dtype
    from musetalk.utils.audio_processor import AudioProcessor  # noqa: PLC0415
    from musetalk.utils.face_parsing import FaceParsing  # noqa: PLC0415
    from musetalk.utils.utils import load_all_model  # noqa: PLC0415
    from transformers import WhisperModel  # noqa: PLC0415

    _vae, _unet, _pe = load_all_model(
        unet_model_path=os.path.join("models", "musetalkV15", "unet.pth"),
        vae_type="sd-vae",
        unet_config=os.path.join("models", "musetalkV15", "musetalk.json"),
        device=DEVICE,
    )
    _pe = _pe.to(DEVICE)
    _vae.vae = _vae.vae.to(DEVICE)
    _unet.model = _unet.model.to(DEVICE)
    if FP16:
        _pe = _pe.half()
        _vae.vae = _vae.vae.half()
        _unet.model = _unet.model.half()
    _weight_dtype = _unet.model.dtype

    _audio_processor = AudioProcessor(feature_extractor_path=os.path.join("models", "whisper"))
    _whisper = WhisperModel.from_pretrained(os.path.join("models", "whisper"))
    _whisper = _whisper.to(device=DEVICE, dtype=_weight_dtype).eval()
    _whisper.requires_grad_(False)
    _fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    print(f"[musetalk-v15] models ready device={DEVICE} fp16={FP16}", flush=True)


def _read_frames(clip: _Clip) -> None:
    cap = cv2.VideoCapture(clip.path)
    clip.fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        clip.frames.append(frame)
    cap.release()
    if not clip.frames:
        raise SystemExit(f"底片无有效帧：{clip.path}")


def _detect_boxes(clip: _Clip) -> None:
    """检测每帧人脸框：优先 MuseTalk 自带 mmpose（支持 bbox_shift）；本地无 mmcv 时回退 MediaPipe。"""
    try:
        from musetalk.utils.preprocessing import get_landmark_and_bbox  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="mt_frames_") as tmpdir:
            paths: list[str] = []
            for i, frame in enumerate(clip.frames):       # clip.frames 为 BGR，直接写盘
                p = os.path.join(tmpdir, f"{i:06d}.png")
                cv2.imwrite(p, frame)
                paths.append(p)
            coords_list, _ = get_landmark_and_bbox(paths, upperbondrange=BBOX_SHIFT)
        h, w = clip.frames[0].shape[:2]
        last_good = (0, 0, w, h)
        for c in coords_list:
            x1, y1, x2, y2 = (int(v) for v in c)
            if x2 - x1 <= 0 or y2 - y1 <= 0 or x1 < 0:    # 占位/异常框 -> 沿用上一帧
                clip.boxes.append(last_good)
                continue
            x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
            last_good = (x1, y1, x2, y2)
            clip.boxes.append(last_good)
        print(f"[musetalk-v15] {clip.name} 人脸框 {len(clip.boxes)} 帧（mmpose, bbox_shift={BBOX_SHIFT}）", flush=True)
        return
    except Exception as err:  # noqa: BLE001 - 无 mmcv 时回退
        print(f"[musetalk-v15] mmpose 不可用（{err}），回退 MediaPipe", flush=True)
    _detect_boxes_mediapipe(clip)


def _detect_boxes_mediapipe(clip: _Clip) -> None:
    """MediaPipe 检测每帧人脸框（替代 mmpose，规避 mmcv 编译）。"""
    import mediapipe as mp  # noqa: PLC0415

    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    h0, w0 = clip.frames[0].shape[:2]
    last_good = (0, 0, w0, h0)
    for frame in clip.frames:
        h, w = frame.shape[:2]
        res = fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.detections:
            clip.boxes.append(last_good)
            continue
        bb = res.detections[0].location_data.relative_bounding_box
        x1 = int(max(0, bb.xmin * w))
        y1 = int(max(0, bb.ymin * h))
        x2 = int(min(w, (bb.xmin + bb.width) * w))
        y2 = int(min(h, (bb.ymin + bb.height) * h))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            clip.boxes.append(last_good)
            continue
        last_good = (x1, y1, x2, y2)
        clip.boxes.append(last_good)
    fd.close()
    print(f"[musetalk-v15] {clip.name} 人脸框 {len(clip.boxes)} 帧（MediaPipe）", flush=True)


def _build_latents(clip: _Clip) -> None:
    for frame, (x1, y1, x2, y2) in zip(clip.frames, clip.boxes):
        crop = cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        clip.latents.append(_vae.get_latents_for_unet(crop))
    print(f"[musetalk-v15] {clip.name} latents {len(clip.latents)} 帧", flush=True)


def _parse_labels(image_pil) -> np.ndarray:
    """BiSeNet 原始 19 类标签图（复刻 MuseTalk ``FaceParsing.__call__`` 的前向）。"""
    from PIL import Image  # noqa: PLC0415

    img = _fp.preprocess(image_pil.resize((512, 512), Image.BILINEAR))
    with torch.no_grad():
        tensor = torch.unsqueeze(img, 0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return _fp.net(tensor)[0].squeeze(0).cpu().numpy().argmax(0)


def _mask_lips(frame: np.ndarray, box: list[int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """只把「嘴 + 上下唇」交给模型重建，其余像素逐像素保留底片原貌。

    相比官方 ``jaw`` 模式（整个下半脸替换）可显著减少「脸变了」。
    BiSeNet(CelebAMask-HQ) 标签：11=mouth 12=u_lip 13=l_lip（10=nose 14=neck）。
    """
    from PIL import Image  # noqa: PLC0415

    from musetalk.utils.blending import get_crop_box  # noqa: PLC0415

    crop_box, _s = get_crop_box(box, 1.5)
    face_large = Image.fromarray(frame[:, :, ::-1]).crop(crop_box)
    w, h = face_large.size

    lip = (np.isin(_parse_labels(face_large), [11, 12, 13]) * 255).astype(np.uint8)
    if not lip.any():
        raise RuntimeError("lips 区域为空")
    lip = cv2.resize(lip, (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(3, int(LIPS_DILATE * w) // 2 * 2 + 1)
    lip = cv2.dilate(lip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    b = max(3, int(LIPS_BLUR * w) // 2 * 2 + 1)
    return cv2.GaussianBlur(lip, (b, b), 0), crop_box


def _build_materials(clip: _Clip) -> None:
    """预计算每帧融合 mask 与裁剪框（底片固定 -> 只算一次）。

    mask 决定「哪些像素用模型重建的脸替换 / 哪些逐像素保留底片」：
      - raw（整脸）/ neck（含脖子）/ jaw（下半脸，官方默认）
      - lips（本项目新增）：只替换嘴+上下唇，其余 100% 保留底片原貌
    """
    from musetalk.utils.blending import get_image_prepare_material  # noqa: PLC0415

    for frame, (x1, y1, x2, y2) in zip(clip.frames, clip.boxes):
        y2m = min(y2 + EXTRA_MARGIN, frame.shape[0])
        box = [x1, y1, x2, y2m]
        try:
            if PARSING_MODE == "lips":
                mask_array, crop_box = _mask_lips(frame, box)
            else:
                mask_array, crop_box = get_image_prepare_material(
                    frame, box, mode=PARSING_MODE, fp=_fp,
                    upper_boundary_ratio=UPPER_BOUNDARY_RATIO,
                )
        except Exception as err:  # noqa: BLE001 - 分割失败回退官方 jaw，不中断启动
            print(f"[musetalk-v15] mask({PARSING_MODE}) 失败：{err} -> 回退 jaw", flush=True)
            mask_array, crop_box = get_image_prepare_material(frame, box, mode="jaw", fp=_fp)
        clip.masks.append(mask_array)
        clip.crops.append(crop_box)
    print(f"[musetalk-v15] {clip.name} mask {len(clip.masks)} 帧 mode={PARSING_MODE}", flush=True)


def _load_clip(name: str, path: str) -> _Clip:
    clip = _Clip(name, path)
    _read_frames(clip)
    if _load_precomputed(clip):
        print(f"[musetalk-v15] {name} 命中预处理缓存（跳过人脸框/VAE/分割三步）", flush=True)
    else:
        _detect_boxes(clip)
        _build_latents(clip)
        _build_materials(clip)
        _save_precomputed(clip)
    print(f"[musetalk-v15] {name} frames={len(clip.frames)} fps={clip.fps:.2f} "
          f"size={clip.frames[0].shape[1]}x{clip.frames[0].shape[0]} path={path}", flush=True)
    return clip


# ------------------------------------------------------------------ 预处理磁盘缓存
# 为什么需要：latents（VAE）与 masks（BiSeNet 分割）都只跟**底片本身**有关，
# 与具体要说什么话无关。算一次就要落盘，否则「老师被 LRU 淘汰后再切回来」要重算 30~60s。
# 融合项目之所以切老师快，正是因为它的素材包里带离线算好的 latents.pt —— 这里用等价手段
# 在当前引擎上拿到同样收益，无需引入第二套服务。
def _cache_path(src: str) -> Path:  # noqa: D401
    """底片指纹（路径+大小+mtime）-> 缓存文件；换源/换底片自动失效。"""
    st = os.stat(src)
    key = hashlib.md5(f"{src}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")).hexdigest()[:32]
    return CACHE_DIR / f"{key}.npz"


def _load_precomputed(clip: _Clip) -> bool:
    """命中缓存时回填 latents/masks/boxes/crops，返回是否命中。"""
    p = _cache_path(clip.path)
    if not p.is_file():
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            lat = torch.from_numpy(z["latents"]).float()          # [N,1,4,H,W]
            clip.latents = [lat[i:i + 1] for i in range(lat.shape[0])]
            masks = z["masks"]
            shapes = z["shapes"]
            clip.masks = [masks[i, :int(h), :int(w)].copy() for i, (h, w) in enumerate(shapes)]
            clip.boxes = [tuple(int(v) for v in b) for b in z["boxes"]]
            clip.crops = [tuple(int(v) for v in c) for c in z["crops"]]
            clip.fps = float(z["fps"])
        ok = (bool(clip.latents) and len(clip.masks) == len(clip.frames)
              and len(clip.boxes) == len(clip.frames))
        return ok
    except Exception as err:  # noqa: BLE001 - 缓存损坏就当没有，现算一份
        print(f"[musetalk-v15] 预处理缓存读取失败（忽略，改为现算）：{err}", flush=True)
        return False


def _save_precomputed(clip: _Clip) -> None:
    """把本次预计算结果落盘（原子写，失败只告警不中断启动）。"""
    if not clip.latents or not clip.masks:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _cache_path(clip.path)
        # 注意：np.savez_compressed 会给不以 .npz 结尾的文件名自动追加 .npz，
        # 临时名必须自己带上 .npz，否则后续原子替换会因文件名对不上而失败
        tmp = p.with_suffix(".tmp.npz")
        # mask 是逐帧裁剪区域的（各帧尺寸不同）-> 补零到统一形状 + 记录原形状；
        # 直接 np.stack 会因形状不一致而失败
        shapes = np.array([[m.shape[0], m.shape[1]] for m in clip.masks], dtype=np.int32)
        hmax, wmax = int(shapes[:, 0].max()), int(shapes[:, 1].max())
        pad = np.zeros((len(clip.masks), hmax, wmax), dtype=clip.masks[0].dtype)
        for i, m in enumerate(clip.masks):
            h, w = m.shape[:2]
            pad[i, :h, :w] = m
        np.savez_compressed(
            tmp,
            latents=torch.cat(clip.latents, dim=0).detach().cpu().numpy(),
            masks=pad, shapes=shapes,
            boxes=np.array(clip.boxes, dtype=np.int32),
            crops=np.array([list(c) for c in clip.crops], dtype=np.int32),
            fps=np.float64(clip.fps),
        )
        tmp.replace(p)
        size_mb = p.stat().st_size / 1048576
        print(f"[musetalk-v15] 已缓存预处理结果 {clip.name} -> {p.name}（{size_mb:.1f} MB）", flush=True)
    except Exception as err:  # noqa: BLE001 - 缓存写不进去不影响推理
        print(f"[musetalk-v15] 预处理缓存写入失败（忽略）：{err}", flush=True)


def _get_clips(teacher_id: str) -> dict[str, "_Clip"]:
    """取某位老师的底片集合（首次访问时懒加载；超过 MAX_TEACHERS 按 LRU 淘汰）。

    ``teacher_id`` 为空时返回「旧单底片」（_clips），保持向后兼容。
    """
    tid = (teacher_id or "").strip()
    if not tid:
        if not _clips:
            raise FileNotFoundError("请求未携带 teacher_id，且已移除旧单底片配置")
        return _clips

    with _teacher_lock:
        got = _teacher_clips.get(tid)
        if got is not None:                     # 命中：刷新 LRU 顺序
            _teacher_clips.pop(tid)
            _teacher_clips[tid] = got
            return got

        talk = TEACHERS_DIR / tid / "talk.mp4"
        if not talk.is_file():
            raise FileNotFoundError(f"教师底片不存在：{talk}")
        clips: dict[str, _Clip] = {"talk": _load_clip("talk", str(talk))}
        greet = TEACHERS_DIR / tid / "idle.mp4"
        if greet.is_file():
            clips["greet"] = _load_clip("greet", str(greet))
        _teacher_clips[tid] = clips
        print(f"[musetalk-v15] 已加载教师底片 {tid}（缓存 {len(_teacher_clips)}/{MAX_TEACHERS}）", flush=True)
        while len(_teacher_clips) > MAX_TEACHERS:
            old, _c = next(iter(_teacher_clips.items()))
            _teacher_clips.pop(old)
            print(f"[musetalk-v15] 淘汰教师底片缓存：{old}", flush=True)
        return clips


def _init() -> None:
    print(f"[musetalk-v15] dir={MUSETALK_DIR} talk={TALK_CLIP} greet={GREET_CLIP} "
          f"device={DEVICE} batch={BATCH} teachers_dir={TEACHERS_DIR}", flush=True)
    if not MUSETALK_DIR.is_dir():
        raise SystemExit(f"未找到 MuseTalk 仓库：{MUSETALK_DIR}（请设 MUSETALK_DIR）")
    os.chdir(MUSETALK_DIR)                     # 模型路径是相对仓库根
    _load_models()
    if LEGACY_PRELOAD and TALK_CLIP and os.path.isfile(TALK_CLIP):
        _clips["talk"] = _load_clip("talk", TALK_CLIP)
        if GREET_CLIP and os.path.isfile(GREET_CLIP):
            _clips["greet"] = _load_clip("greet", GREET_CLIP)
        print(f"[musetalk-v15] 旧单底片已预加载 primitives={list(_clips)}", flush=True)
    else:
        print("[musetalk-v15] 多老师按需加载模式（未配置旧单底片）", flush=True)
    print("[musetalk-v15] 就绪", flush=True)


def _trim_silence_inplace(wav_path: str) -> bool:
    """裁掉 wav 首尾静音（原地覆盖），返回是否发生裁剪。

    Edge TTS 输出常带 0.1~0.3s 前导静音；不裁的话帧数按音频总长计算，
    整段口型会「提前」于声音。只用标准库 wave + numpy，无额外依赖。
    """
    import wave  # noqa: PLC0415

    try:
        with wave.open(wav_path, "rb") as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if sw != 2 or not raw:
            return False
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            y = y.reshape(-1, n_ch).mean(axis=1)
        loud = np.where(np.abs(y) > 10 ** (TRIM_DB / 20))[0]
        if loud.size == 0:
            return False
        pad = int(TRIM_PAD * sr)
        a = max(0, int(loud[0]) - pad)
        b = min(len(y), int(loud[-1]) + 1 + pad)
        if b - a < sr * 0.05 or (a == 0 and b == len(y)):
            return False                      # 没有静音 / 裁完太短 -> 不动
        seg = np.clip(y[a:b] * 32768.0, -32768, 32767).astype(np.int16)
        if n_ch > 1:
            seg = np.repeat(seg[:, None], n_ch, axis=1).ravel()
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(n_ch)
            wf.setsampwidth(sw)
            wf.setframerate(sr)
            wf.writeframes(seg.tobytes())
    except Exception as err:  # noqa: BLE001 - 裁剪失败不应中断渲染
        print(f"[musetalk-v15] 静音裁剪跳过：{err}", flush=True)
        return False
    return True


# ------------------------------------------------------------------ 逐句渲染
def _render(audio_bytes: bytes, out_mp4: str, gesture: str | None = None,
            clips: dict[str, "_Clip"] | None = None) -> None:
    """一句音频 -> 对口型视频；按 gesture 在给定底片集合里选择。"""
    from musetalk.utils.blending import get_image_blending  # noqa: PLC0415

    clips = clips or _clips
    use_greet = gesture == "greet" and "greet" in clips
    clip = clips["greet"] if use_greet else clips["talk"]
    n_frames = len(clip.frames)
    fps = clip.fps

    tmp = tempfile.mkdtemp(prefix="mt15_")
    in_audio = os.path.join(tmp, "in.mp3")
    Path(in_audio).write_bytes(audio_bytes)
    wav16 = os.path.join(tmp, "a.wav")
    subprocess.run([FFMPEG, "-y", "-i", in_audio, "-ac", "1", "-ar", "16000", wav16],
                   check=True, capture_output=True)
    if TRIM_SILENCE and _trim_silence_inplace(wav16):
        print(f"[musetalk-v15] 已裁剪首尾静音（阈值 {TRIM_DB}dB）", flush=True)

    features, librosa_length = _audio_processor.get_audio_feature(wav16)
    weight_dtype = _unet.model.dtype
    whisper_chunks = _audio_processor.get_whisper_chunk(
        features, DEVICE, weight_dtype, _whisper, librosa_length,
        fps=fps,
        audio_padding_length_left=AUDIO_PAD_LEFT,
        audio_padding_length_right=AUDIO_PAD_RIGHT,
    )
    n_out = int(whisper_chunks.shape[0])
    # 问候：从第 0 帧做起（做一次打招呼动作）；讲解：从上次结束处继续，画面连续
    base = 0 if use_greet else clip.cursor
    idxs = [(base + i) % n_frames for i in range(n_out)]

    silent = os.path.join(tmp, "silent.mp4")
    writer = imageio.get_writer(silent, fps=fps, codec="libx264",
                                quality=VIDEO_QUALITY, macro_block_size=1)
    timesteps = torch.tensor([0], device=DEVICE)

    t_unet = 0.0
    with torch.no_grad():
        for s in range(0, n_out, BATCH):
            _tu = time.time()
            chunk = whisper_chunks[s:s + BATCH].to(DEVICE).to(weight_dtype)
            latent_batch = torch.cat([clip.latents[i] for i in idxs[s:s + BATCH]], dim=0).to(DEVICE).to(weight_dtype)
            audio_feat = _pe(chunk)
            pred = _unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feat).sample
            recon = _vae.decode_latents(pred)
            t_unet += time.time() - _tu
            for j, res_frame in enumerate(recon):
                gi = idxs[s + j]
                frame = clip.frames[gi]
                x1, y1, x2, y2 = clip.boxes[gi]
                y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                combined = get_image_blending(frame, res_frame, [x1, y1, x2, y2], clip.masks[gi], clip.crops[gi])
                writer.append_data(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"[musetalk-v15] render clip={clip.name} n_out={n_out} unet={t_unet:.1f}s", flush=True)
    if not use_greet:
        clip.cursor = (clip.cursor + n_out) % n_frames

    # 音轨与口型帧同源：裁剪过静音时用裁剪后的 wav，避免声画错位
    mux_audio = wav16 if TRIM_SILENCE else in_audio
    subprocess.run([FFMPEG, "-y", "-i", silent, "-i", mux_audio, "-c:v", "copy",
                    "-c:a", "aac", "-shortest", out_mp4], check=True, capture_output=True)


# ------------------------------------------------------------------ HTTP 服务
app = FastAPI(title="MuseTalk v1.5 渲染服务", version="0.1.0")


class RenderRequest(BaseModel):
    audio: str = ""
    text: str = ""
    gesture: str | None = None      # "greet" -> 用问候底片（idle.mp4）对口型
    fps: int = 0
    bbox_shift: int | None = None
    teacher_id: str | None = None   # 多老师：按学科 key 选择该老师的底片（空=旧单底片）


@app.get("/health")
def health() -> dict:
    def _fps() -> float:
        if "talk" in _clips:
            return _clips["talk"].fps
        if _teacher_clips:
            return next(iter(_teacher_clips.values()))["talk"].fps
        return 24.0

    return {"status": "ok", "engine": "musetalk-v15", "device": DEVICE,
            "clips": {k: len(v.frames) for k, v in _clips.items()},
            "fps": _fps(),
            "extra_margin": EXTRA_MARGIN, "parsing_mode": PARSING_MODE,
            "upper_boundary_ratio": UPPER_BOUNDARY_RATIO,
            "trim_silence": TRIM_SILENCE,
            "audio_pad": [AUDIO_PAD_LEFT, AUDIO_PAD_RIGHT],
            "teachers_dir": str(TEACHERS_DIR),
            "loaded_teachers": list(_teacher_clips),
            "legacy_preload": LEGACY_PRELOAD}


@app.post("/render")
def render(req: RenderRequest) -> dict:
    if not req.audio:
        raise HTTPException(status_code=400, detail="audio(base64) 不能为空")
    audio = base64.b64decode(req.audio)
    out_mp4 = os.path.join(tempfile.mkdtemp(prefix="mt15out_"), f"{uuid.uuid4().hex}.mp4")
    t0 = time.time()
    try:
        with _lock:                              # 模型非线程安全，串行化（含底片切换/懒加载）
            clips = _get_clips(req.teacher_id or "")
            _render(audio, out_mp4, gesture=req.gesture, clips=clips)
    except FileNotFoundError as err:             # 该教师的底片缺失 -> 明确 404，便于上层回退
        raise HTTPException(status_code=404, detail=str(err)) from err
    data = Path(out_mp4).read_bytes()
    return {"mode": "video", "video": base64.b64encode(data).decode("ascii"),
            "mime": "video/mp4", "text": req.text, "model": "musetalk-1.5",
            "teacher_id": req.teacher_id or "", "latency_ms": int((time.time() - t0) * 1000)}


def main() -> None:
    import uvicorn

    _init()
    print(f"[musetalk-v15] http://{HOST}:{PORT}  (POST /render)", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
