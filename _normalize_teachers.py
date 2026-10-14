# -*- coding: utf-8 -*-
"""临时：把 10 位老师底片规范化为「人脸居中 + 统一大小（放大）」。

做法：OpenCV(Haar) 稳测人脸框 -> 按目标脸宽算裁剪窗口 -> 裁剪后缩放回 1280x720。
结果：所有老师的脸中心落在同一位置、脸宽一致，于是「嘴位/眼位/眨眼」可共用一套标定。
"""
from __future__ import annotations

import os
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1280, 720
SRC = "data/teaching_materials"
DST = "data/teachers"

# 目标：脸中心 (0.50, 0.374)，脸宽占比 0.238
# （0.238 必须 >= 实测最大脸宽 0.237，否则“把脸缩小”无解）
TARGET_FACE_W = 0.238
TARGET_CX, TARGET_CY = 0.50, 0.374

FILES = {
    "chinese": "语文老师.mp4", "math": "数学老师.mp4", "english": "英语老师.mp4",
    "physics": "物理老师.mp4", "chemistry": "化学老师.mp4", "biology": "生物老师.mp4",
    "politics": "政治老师.mp4", "morality": "道德与法治老师.mp4",
    "geography": "地理老师.mp4", "history": "历史老师.mp4",
}

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def measure_face(path: str):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    boxes = []
    for i in list(range(5, total, max(1, total // 12)))[:12]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.equalizeHist(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        b = CASCADE.detectMultiScale(g, 1.08, 6, minSize=(int(W * 0.12), int(H * 0.16)))
        if len(b):
            boxes.append(max(b, key=lambda x: int(x[2]) * int(x[3])))
    cap.release()
    if not boxes:
        return None
    return (float(np.median([b[0] for b in boxes])), float(np.median([b[1] for b in boxes])),
            float(np.median([b[2] for b in boxes])), float(np.median([b[3] for b in boxes])))


def main() -> None:
    print(f"{'学科':<11}{'原脸宽':<9}{'放大':<8}{'裁剪窗口':<15}{'残余偏移(dx,dy)':<18}{'输出'}")
    print("-" * 100)
    for key, name in FILES.items():
        src = os.path.join(SRC, name)
        if not os.path.isfile(src):
            print(f"{key:<11}缺失 {src}")
            continue
        m = measure_face(src)
        if m is None:
            print(f"{key:<11}人脸检测失败，跳过")
            continue
        fx, fy, fw, fh = m
        fcx, fcy = fx + fw / 2, fy + fh / 2

        cw = fw / TARGET_FACE_W
        ch = cw * 9 / 16
        if cw > W:
            cw, ch = float(W), float(H)
        x0 = float(np.clip(fcx - TARGET_CX * cw, 0, W - cw))
        y0 = float(np.clip(fcy - TARGET_CY * ch, 0, H - ch))

        rcx = (fcx - x0) / cw
        rcy = (fcy - y0) / ch
        zoom = W / cw

        out_dir = os.path.join(DST, key)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "talk.mp4")
        r = subprocess.run(
            [FF, "-y", "-v", "error", "-i", src,
             "-vf", f"crop={cw:.0f}:{ch:.0f}:{x0:.0f}:{y0:.0f},scale={W}:{H}",
             "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "24",
             "-c:a", "aac", "-b:a", "128k", out],
            capture_output=True, text=True, timeout=300,
        )
        ok = os.path.isfile(out) and os.path.getsize(out) > 0
        if not ok and r.stderr.strip():
            print("   ffmpeg:", r.stderr.strip()[:200])
        # 音色素材：同时抽 16k 单声道干声
        wav = os.path.join(out_dir, "voice_ref.wav")
        subprocess.run([FF, "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav],
                       capture_output=True, timeout=180)

        print(f"{key:<11}{fw / W:<9.3f}{zoom:<8.2f}{f'{cw:.0f}x{ch:.0f}':<15}"
              f"({rcx - TARGET_CX:+.3f},{rcy - TARGET_CY:+.3f})   "
              f"{'OK' if ok else 'FAIL'}  {'+wav' if os.path.isfile(wav) else ''}")

    print("\n[完成] 已输出到 data/teachers/<学科>/talk.mp4 与 voice_ref.wav")
    print(f"[目标] 脸中心 ({TARGET_CX},{TARGET_CY})  脸宽占比 {TARGET_FACE_W}")


if __name__ == "__main__":
    main()
