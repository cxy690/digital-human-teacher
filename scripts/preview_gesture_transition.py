# -*- coding: utf-8 -*-
"""本地手势过渡预览（无需 MuseTalk/torch）。

复用 avatar_engine_musetalk.py 里同一套帧规划逻辑（_bounce / _plan_frames / 交叉淡化），
用真实基元视频（talk + nod/shake/shrug/point）合成一段演示 mp4，直观查看：
  讲解 → 点头 → 摇头 → 摊手 → 指点 → 讲解 的丝滑过渡。

区别：不做 MuseTalk 口型重绘（嘴型不对音频），只看动作切换与过渡。
用法： D:\\An\\envs\\tea\\python.exe scripts/preview_gesture_transition.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_DIR = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "preview_gesture_transition.mp4")
CROSSFADE_FRAMES = 6          # 与引擎 MUSETALK_CROSSFADE 默认一致
FPS = 24


class Clip:
    """与引擎 _Clip 等价的最小实现：帧序列 + 取帧游标。"""

    def __init__(self, name: str, frames: list[np.ndarray]) -> None:
        self.name = name
        self.frames = frames
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.frames)


def read_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)          # 保持 BGR，直接给 VideoWriter
    cap.release()
    if not frames:
        raise SystemExit(f"无有效帧：{path}")
    return frames


# ---------------------------- 以下逻辑与 avatar_engine_musetalk.py 保持一致 ----------------------------
def _bounce(idx: int, n: int) -> int:
    if n <= 1:
        return 0
    cycle = 2 * (n - 1)
    m = idx % cycle
    return m if m < n else cycle - m


def _plan_frames(n_frames: int, gesture: str | None, clips: dict[str, Clip]):
    talk = clips["talk"]
    g = clips.get(gesture) if gesture else None
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
            alpha = (i + 1) / (k + 1)
            plan.append([(talk, t_idx, 1.0 - alpha), (g, _bounce(i, len(g)), alpha)])
        elif i < n_frames - k:
            plan.append([(g, _bounce(i - k, len(g)), 1.0)])
        else:
            alpha = (i - (n_frames - k) + 1) / (k + 1)
            plan.append([(g, _bounce(i - k, len(g)), 1.0 - alpha), (talk, t_idx, alpha)])
    talk.cursor = (talk.cursor + n_frames) % len(talk)
    return plan
# ----------------------------------------------------------------------------------------------------


def main() -> None:
    names = ["talk", "nod", "shake", "shrug", "point"]
    clips: dict[str, Clip] = {}
    for n in names:
        p = os.path.join(CLIP_DIR, f"{n}.mp4")
        if not os.path.isfile(p):
            raise SystemExit(f"缺少基元视频：{p}")
        clips[n] = Clip(n, read_frames(p))
        print(f"loaded {n}: {len(clips[n])} frames")

    h, w = clips["talk"].frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT, fourcc, FPS, (w, h))

    # 时间线：讲解 → 各手势（每个含淡入/淡出回讲解）→ 讲解
    timeline = [
        (None, 36), ("nod", 60), (None, 18),
        ("shake", 60), (None, 18),
        ("shrug", 60), (None, 18),
        ("point", 60), (None, 36),
    ]
    total = 0
    for gesture, n in timeline:
        for spec in _plan_frames(n, gesture, clips):
            canvas = None
            for clip, idx, weight in spec:
                layer = clip.frames[idx].astype(np.float32) * weight
                canvas = layer if canvas is None else canvas + layer
            writer.write(np.clip(canvas, 0, 255).astype(np.uint8))
            total += 1
        print(f"segment gesture={gesture or 'talk':5} frames={n} done")
    writer.release()
    print(f"\n✅ 预览已生成：{OUT}")
    print(f"   共 {total} 帧 ≈ {total / FPS:.1f}s  分辨率 {w}x{h} @ {FPS}fps")


if __name__ == "__main__":
    main()
