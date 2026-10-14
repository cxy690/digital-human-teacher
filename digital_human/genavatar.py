# -*- coding: utf-8 -*-
"""数字人素材生成工具（MuseTalk v15 预处理数据，官方流程移植）。

将"讲课视频/帧序列"处理为数字人服务可直接加载的素材目录：
    full_imgs/      全帧序列（正+倒 ping-pong，无缝循环）
    mask/           人脸下半区解析蒙版（jaw 模式）
    coords.pkl      每帧人脸 bbox
    mask_coords.pkl 每帧蒙版裁剪框
    latents.pt      每帧 VAE 编码（供 UNet 推理）
    avator_info.json

用法（在 digital_human 目录下，使用 CUDA 环境解释器）：
    python genavatar.py --video <video.mp4> --avatar-id ai3
    python genavatar.py --imgs <frames_dir> --avatar-id ai3 --output ../data/avatars/ai3_next

说明：
- 素材要求每一帧都能检测到人脸（与官方一致）；有失败帧时脚本报错退出。
- 输出目录已存在素材时需显式 --overwrite（防误覆盖在用素材）。
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)  # FaceParsing/VAE 内部使用 ./models/... 相对路径
sys.path.insert(0, BASE)
# preprocessing.py 以裸包名导入 face_detection（教程既有代码），需补充该目录
sys.path.insert(0, os.path.join(BASE, "musetalk", "utils"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from musetalk.models.vae import VAE  # noqa: E402
from musetalk.utils.blending import get_image_prepare_material  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402
from musetalk.utils.preprocessing import get_landmark_and_bbox  # noqa: E402

COORD_PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)


def smooth_coords(coords, window=9):
    """人脸框时序平滑（移动平均，window 取奇数；<3 不处理）。

    逐帧 landmarks 的 bbox 上/下缘带 ±数像素高频噪声，直接用于 VAE 裁剪/贴回
    会导致口型尺寸逐帧闪烁、贴回区域跳动；平均掉高频、保留头部慢速运动。
    """
    a = np.asarray(coords, dtype=np.float32)
    if window < 3 or len(a) < window:
        return [list(map(float, c)) for c in a]
    half = window // 2
    pad = np.pad(a, ((half, half), (0, 0)), mode="edge")
    csum = np.cumsum(pad, axis=0)
    csum = np.vstack([np.zeros((1, a.shape[1]), dtype=np.float32), csum])
    sm = (csum[window:] - csum[:-window]) / window
    return [list(map(float, c)) for c in sm]


def imread(path):
    """读图（PIL 解码，规避 cv2.imread 不支持非 ASCII 路径的问题）。"""
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def imwrite(path, img):
    """PNG 写出（imencode + Python 文件接口，兼容非 ASCII 路径）。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"PNG 编码失败：{path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def load_input_frames(args):
    if args.imgs:
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            files.extend(glob.glob(os.path.join(args.imgs, ext)))
        files.sort()
        if not files:
            raise SystemExit(f"输入帧目录为空：{args.imgs}")
        return [imread(p) for p in files]
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频：{args.video}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"视频未读取到任何帧：{args.video}")
    return frames


def main():
    parser = argparse.ArgumentParser(description="数字人素材生成（MuseTalk v15 预处理）")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="输入视频文件")
    src.add_argument("--imgs", help="输入帧目录（PNG/JPG 序列）")
    parser.add_argument("--avatar-id", default="ai3", help="形象 ID（默认 ai3）")
    parser.add_argument("--avatar-root", default=None,
                        help="素材根目录（默认项目级 data/avatars）")
    parser.add_argument("--output", default=None,
                        help="输出目录（默认 <avatar-root>/<avatar-id>）")
    parser.add_argument("--bbox-shift", type=int, default=0, help="bbox 纵向偏移（v15 固定 0）")
    parser.add_argument("--extra-margin", type=int, default=10, help="bbox 下缘外扩像素（v15）")
    parser.add_argument("--mode", default="jaw", help="脸部解析模式（v15 用 jaw）")
    parser.add_argument("--left-cheek-width", type=int, default=90)
    parser.add_argument("--right-cheek-width", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true",
                        help="输出目录已有素材时清空重建")
    parser.add_argument("--smooth-window", type=int, default=9,
                        help="人脸框时序移动平均窗口（奇数，0 关闭）：消除逐帧检测抖动")
    args = parser.parse_args()

    if args.imgs:
        args.imgs = os.path.abspath(args.imgs)

    root = args.avatar_root or os.path.abspath(os.path.join(BASE, "..", "data", "avatars"))
    out = args.output or os.path.join(root, args.avatar_id)
    out = os.path.abspath(out)
    full_dir = os.path.join(out, "full_imgs")
    mask_dir = os.path.join(out, "mask")
    if os.path.isdir(full_dir) and os.listdir(full_dir):
        if not args.overwrite:
            raise SystemExit(f"输出目录已有素材：{out}\n如需重建请加 --overwrite")
        shutil.rmtree(out)
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # ---- 1. 读入帧并写盘（人脸检测从磁盘读取）----
    frames = load_input_frames(args)
    h, w = frames[0].shape[:2]
    print(f"输入 {len(frames)} 帧，{w}x{h} → 输出 {out}")
    for i, f in enumerate(frames):
        imwrite(os.path.join(full_dir, f"{i:08d}.png"), f)

    # ---- 2. 人脸检测 → coords ----
    img_list = sorted(glob.glob(os.path.join(full_dir, "*.png")))
    coords, frames_read = get_landmark_and_bbox(img_list, args.bbox_shift)
    bad = [i for i, c in enumerate(coords) if tuple(c) == COORD_PLACEHOLDER]
    if bad:
        raise SystemExit(f"有 {len(bad)} 帧未检测到人脸（帧号: {bad[:20]}...），"
                         "素材要求全帧可检测，请更换视频或调整 bbox-shift")
    coords = smooth_coords(coords, args.smooth_window)

    # ---- 3. VAE 编码 latents（v15：bbox 下缘外扩 extra_margin 后裁剪 256）----
    vae = VAE(model_path=os.path.join(BASE, "models", "sd-vae"))
    latents = []
    for idx, (bbox, frame) in enumerate(tqdm(list(zip(coords, frames_read)),
                                             desc="VAE 编码", ncols=80)):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        y2 = min(y2 + args.extra_margin, frame.shape[0])
        coords[idx] = [x1, y1, x2, y2]
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents.append(vae.get_latents_for_unet(resized).cpu())
    del vae
    torch.cuda.empty_cache()

    # ---- 4. 正+倒 ping-pong（首尾同帧，用于无缝循环）----
    frames_cyc = frames_read + frames_read[::-1]
    coords_cyc = coords + coords[::-1]
    latents_cyc = latents + latents[::-1]
    n = len(frames_cyc)

    # ---- 5. 写 full_imgs（覆盖为 ping-pong 序列）+ 生成 mask ----
    fp = FaceParsing(left_cheek_width=args.left_cheek_width,
                     right_cheek_width=args.right_cheek_width)
    mask_coords = []
    for i, frame in enumerate(tqdm(frames_cyc, desc="生成 mask", ncols=80)):
        imwrite(os.path.join(full_dir, f"{i:08d}.png"), frame)
        x1, y1, x2, y2 = [int(v) for v in coords_cyc[i]]
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2],
                                                    fp=fp, mode=args.mode)
        imwrite(os.path.join(mask_dir, f"{i:08d}.png"), mask)
        mask_coords.append(crop_box)
    del fp
    torch.cuda.empty_cache()

    # ---- 6. 保存数据文件 ----
    with open(os.path.join(out, "coords.pkl"), "wb") as f:
        pickle.dump(coords_cyc, f)
    with open(os.path.join(out, "mask_coords.pkl"), "wb") as f:
        pickle.dump(mask_coords, f)
    torch.save(latents_cyc, os.path.join(out, "latents.pt"))
    with open(os.path.join(out, "avator_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "avatar_id": args.avatar_id,
            "video_path": os.path.join(out, "full_imgs"),
            "bbox_shift": args.bbox_shift,
        }, f)

    # ---- 7. 一致性校验 ----
    assert len(frames_cyc) == len(coords_cyc) == len(mask_coords) == len(latents_cyc) == n
    n_mask = len(glob.glob(os.path.join(mask_dir, "*.png")))
    assert n_mask == n, f"mask 文件数不一致：{n_mask} != {n}"
    print(f"完成：{n} 帧素材（原始 {len(frames)} 帧 ping-pong）→ {out}")


if __name__ == "__main__":
    main()
