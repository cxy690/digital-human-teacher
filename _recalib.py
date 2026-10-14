"""重新标定 10 位老师的待机帧 / 闭眼帧 / 眼位 / 眨眼对齐偏移，并分析干声基频。

用法（云端，musetalk venv，在 data/teachers 下执行）：
    python _recalib.py            # 只报告
    python _recalib.py --apply    # 写回 teacher.json

判定依据（全部是可验证的几何量）：
* EAR（眼纵横比）= (上睑-下睑) / 左右眼角间距；睁眼大、闭眼小；
* 待机帧 = 全片内「EAR 最大 且 嘴张开度不大于中位数」的帧（睁眼 + 闭嘴 + 自然）；
* 闭眼帧 = EAR 最小、且头位与待机帧接近（瞳距偏差 <3%、鼻尖位移 <6px）；
  并要求 EAR 明显低于待机帧（<0.70 倍），否则视为「片中没有真眨眼」→ blink_at=null，
  由前端退化为「眼区压合」合成（几何完全对齐，不会错位）；
* blink_shift = 闭眼帧两只眼睛中心相对待机帧的像素偏移，前端贴图时据此对齐，
  这是「眨眼不协调」的根因修复。
"""
from __future__ import annotations

import glob
import json
import os
import sys
import wave

import cv2
import mediapipe as mp
import numpy as np

APPLY = "--apply" in sys.argv
# --only=english,math ：只重选指定老师（避免影响已调好的其它老师）
ONLY: set[str] = set()
FORCE_FRAME: float | None = None
for _a in sys.argv:
    if _a.startswith("--only="):
        ONLY = {x.strip() for x in _a.split("=", 1)[1].split(",") if x.strip()}
    elif _a.startswith("--force-frame="):
        FORCE_FRAME = float(_a.split("=", 1)[1])
W, H = 1280, 720
FPS = 24.0

R_EYE = {"corner_a": 33, "corner_b": 133, "top": 160, "bottom": 144, "top2": 158, "bottom2": 153}
L_EYE = {"corner_a": 362, "corner_b": 263, "top": 385, "bottom": 373, "top2": 387, "bottom2": 380}
LIPS = {"upper": 13, "lower": 14}
MOUTH = {"left": 291, "right": 61}       # 嘴角（左/右）
NOSE = 1


def f0_median(path: str) -> float:
    """轻量自相关法估计基频中位数（男声 ~85-155Hz，女声 ~165-255Hz）。"""
    try:
        with wave.open(path) as w:
            sr = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return 0.0
    if data.size < sr:
        return 0.0
    lo, hi = int(sr / 350), int(sr / 70)
    win = int(sr * 0.04)
    f0s: list[float] = []
    for start in range(0, data.size - win, win):
        seg = data[start:start + win]
        if float(np.sqrt(np.mean(seg ** 2))) < 0.02:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        hi_i = min(hi, len(ac) - 1)
        if hi_i <= lo + 1:
            continue
        peak = lo + int(np.argmax(ac[lo:hi_i]))
        if ac[peak] > 0.35:
            f0s.append(sr / peak)
    return float(np.median(f0s)) if f0s else 0.0


def landmarks(video: str) -> list[dict]:
    """逐帧取关键点（像素坐标）与几何量。"""
    cap = cv2.VideoCapture(video)
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    out: list[dict] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            out.append({})
            continue
        lm = res.multi_face_landmarks[0].landmark

        def px(i: int) -> tuple[float, float]:
            return lm[i].x * W, lm[i].y * H

        def eye(e: dict) -> tuple[float, float, float, float]:
            ax, ay = px(e["corner_a"]); bx, by = px(e["corner_b"])
            tx, ty = px(e["top"]); b2x, b2y = px(e["bottom"])
            t2x, t2y = px(e["top2"]); b3x, b3y = px(e["bottom2"])
            wid = float(np.hypot(ax - bx, ay - by)) or 1.0
            hgt = (float(np.hypot(tx - b2x, ty - b2y)) + float(np.hypot(t2x - b3x, t2y - b3y))) / 2
            return hgt / wid, (ax + bx) / 2, (ay + by) / 2, wid

        r_ear, rcx, rcy, rw = eye(R_EYE)
        l_ear, lcx, lcy, _lw = eye(L_EYE)
        ux, uy = px(LIPS["upper"]); dx, dy = px(LIPS["lower"])
        mlx, mly = px(MOUTH["left"]); mrx, mry = px(MOUTH["right"])
        nx, ny = px(NOSE)
        # 全脸形状（用于衡量「离中性表情有多远」）：按瞳距归一化，消除头位缩放影响
        ied = float(np.hypot(rcx - lcx, rcy - lcy)) or 1.0
        pts = np.array([[lm[i].x * W, lm[i].y * H] for i in range(len(lm))], dtype=np.float32)
        out.append({
            "ear": (r_ear + l_ear) / 2,
            "rc": (rcx, rcy), "lc": (lcx, lcy),
            "rw": rw,
            "mouth": float(np.hypot(ux - dx, uy - dy)),          # 上下唇间距：张嘴度
            "mouth_w": float(np.hypot(mlx - mrx, mly - mry)),    # 嘴角间距：撅嘴时明显变窄
            # 嘴角相对唇缝中点的竖直偏移：微笑上提（负）、撅嘴/撇嘴下沉（正）
            "corner_dy": float((mly + mry) / 2 - (uy + dy) / 2),
            "nose": (nx, ny),
            "ied": ied,
            "pts": (pts - np.array([rcx, rcy], dtype=np.float32)) / ied,
        })
    cap.release()
    mesh.close()
    return out


def main() -> None:
    rows = []
    for p in sorted(glob.glob("*/teacher.json")):
        key = os.path.dirname(p)
        if ONLY and key not in ONLY:
            continue
        d = json.load(open(p, encoding="utf-8"))
        frames = landmarks(os.path.join(key, "talk.mp4"))
        valid = [i for i, f in enumerate(frames) if f]
        if not valid:
            print(f"{key:<10} 未检测到人脸，跳过")
            continue

        # ---- 先选「睁眼 + 闭嘴 + 表情中性」的天然待机帧（只看正脸帧，避免侧头）----
        #
        # 只用「上下唇间距」判断闭嘴是不够的：说话中的双唇闭合（/m/ /b/ /p/ 那类）
        # 也是闭合的，但嘴角收拢、嘴唇前突，静态图看起来就是「撅嘴」。
        # 因此再加两条约束：
        #   1) 嘴角间距不能明显窄于本片中位数（撅嘴会显著变窄）；
        #   2) 全脸关键点尽量接近本片的中性形态（离中性越远 = 表情越夸张）。
        med_ied = float(np.median([frames[i]["ied"] for i in valid]))
        front = [i for i in valid if abs(frames[i]["ied"] - med_ied) / med_ied < 0.05] or valid

        gaps = [frames[i]["mouth"] for i in valid]
        gap_max = float(np.percentile(gaps, 10))          # 唇缝要闭合（取全片最闭的 10%）
        ears = [frames[i]["ear"] for i in valid]
        ear_min = float(np.percentile(ears, 60))          # 眼睛要明显睁开
        # 嘴宽基准必须只在「闭嘴帧」里取：说话张嘴会让嘴宽明显变大，
        # 若拿全片中位数当基准，闭嘴的正常宽度会被误判成「过窄（撅嘴）」。
        closed = [i for i in valid if frames[i]["mouth"] <= gap_max]
        w_med = float(np.median([frames[i]["mouth_w"] for i in (closed or valid)]))

        # 逐帧运动量：与前一帧关键点的平均位移（按瞳距归一化）。
        # 说话时口型/头部一直在动，静息帧的运动量明显更低，用它把「说话间歇」挑出来。
        motion: dict[int, float] = {}
        for n, i in enumerate(valid):
            motion[i] = (0.0 if n == 0
                         else float(np.mean(np.linalg.norm(
                             frames[i]["pts"] - frames[valid[n - 1]]["pts"], axis=1))))
        mv = [motion[i] for i in valid]
        motion_max = float(np.percentile(mv, 45))

        # 以「唇闭合 + 低运动」的稳定帧为基准形态，再挑最接近该形态的一帧：
        # 这样得到的是「放松说话间歇」的常态表情，而不是视频里占比最大的微笑表情。
        calm = [i for i in front if frames[i]["mouth"] <= gap_max and motion[i] <= motion_max]
        ref_shape = np.median(np.stack([frames[i]["pts"] for i in calm or front]), axis=0)

        # 反「撅嘴」：撅嘴时嘴角收拢、嘴宽明显窄于常态。
        # 逐级放宽：优先要求「嘴宽 >= 常态」，再放宽到 0.97 / 0.94，最后才不限制。
        pool: list[int] = []
        for floor in (1.00, 0.97, 0.94, None):
            base = [i for i in front
                    if frames[i]["mouth"] <= gap_max
                    and frames[i]["ear"] >= ear_min
                    and motion[i] <= motion_max * 1.4]
            if floor is not None:
                base = [i for i in base if frames[i]["mouth_w"] >= w_med * floor]
            if base:
                pool = base
                break
        if not pool:
            pool = calm or front

        def rest_score(i: int) -> float:
            """待机帧打分（越小越好）：既贴近「放松常态」，又避免撅嘴/大笑。"""
            f = frames[i]
            shape_dev = float(np.mean(np.linalg.norm(f["pts"] - ref_shape, axis=1)))
            # 嘴宽低于常态（撅嘴）时额外重罚：这是用户最容易看出来的瑕疵。
            narrow = max(0.0, (w_med - f["mouth_w"]) / w_med)
            return (shape_dev
                    + abs(f["corner_dy"]) / f["ied"] * 2.0
                    + abs(f["mouth_w"] - w_med) / w_med * 1.5
                    + narrow * 3.0
                    + motion[i] / max(1e-6, motion_max) * 0.5)

        if FORCE_FRAME is not None:      # 人工指定（自动打分个别老师仍不理想时用）
            _target = int(round(FORCE_FRAME * FPS))
            still_i = min(valid, key=lambda i: abs(i - _target))
        else:
            still_i = min(pool, key=rest_score)
        s = frames[still_i]
        # 关键指标（可量化核对：嘴宽不得明显窄于常态，否则就是「撅嘴」）
        print(f"  > {key} 待机帧={still_i / FPS:.2f}s EAR={s['ear']:.3f} "
              f"嘴宽={s['mouth_w']:.0f}(常态{w_med:.0f}, 比值{s['mouth_w'] / w_med:.2f}) "
              f"唇缝={s['mouth']:.1f} 嘴角偏移={s['corner_dy']:+.1f} 运动={motion[still_i]:.3f}")

        # ---- 在待机帧附近找「真眨眼帧」：眼位偏移必须小，否则贴图必错位 ----
        def eye_dist(i: int) -> float:
            f = frames[i]
            return max(float(np.hypot(f["rc"][0] - s["rc"][0], f["rc"][1] - s["rc"][1])),
                       float(np.hypot(f["lc"][0] - s["lc"][0], f["lc"][1] - s["lc"][1])))

        donor_i, dn = None, None
        for limit in (8.0, 16.0, 26.0):               # 先严格，找不到再逐级放宽
            cand = [i for i in front if i != still_i and eye_dist(i) <= limit]
            if not cand:
                continue
            j = min(cand, key=lambda i: frames[i]["ear"])
            if frames[j]["ear"] < s["ear"] * 0.70:    # 明显闭眼 = 真眨眼
                donor_i, dn = j, frames[j]
                break
        has_blink = donor_i is not None

        # ---- 眼位 / 眼区 / 眨眼对齐偏移 ----
        eye_points = [[int(round(s["rc"][0])), int(round(s["rc"][1]))],
                      [int(round(s["lc"][0])), int(round(s["lc"][1]))]]
        eye_r = [int(round(s["rw"] * 0.60)), int(round(s["rw"] * 0.28))]
        shift = ([[int(round(dn["rc"][0] - s["rc"][0])), int(round(dn["rc"][1] - s["rc"][1]))],
                  [int(round(dn["lc"][0] - s["lc"][0])), int(round(dn["lc"][1] - s["lc"][1]))]]
                 if has_blink else None)

        frame_at = round(still_i / FPS, 2)
        blink_at = round(donor_i / FPS, 2) if has_blink else None

        rows.append((key, frame_at, blink_at, eye_points, eye_r, shift,
                     s["ear"], (dn["ear"] if has_blink else 0.0), len(pool),
                     frames[still_i]["mouth"]))
        if APPLY:
            d["eye_points"] = eye_points
            d["eye_r"] = eye_r
            d["frame_at"] = frame_at
            d["blink_at"] = blink_at
            d["blink_shift"] = shift
            d["calib_source"] = "关键点(EAR)"
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(d, fh, ensure_ascii=False, indent=2)

    print(f"\n{'学科':<10}{'待机帧':<8}{'EAR开':<7}{'闭眼帧':<8}{'EAR闭':<7}{'嘴':<6}"
          f"{'eye_r':<10}{'偏移':<22}{'候选':<5}")
    print("-" * 108)
    for key, fa, ba, _ep, er, sh, se, de, nc, mo in rows:
        print(f"{key:<10}{fa:<8}{se:<7.3f}{str(ba):<8}{de:<7.3f}{mo:<6.1f}{str(er):<10}{str(sh):<22}{nc:<5}")

    print(f"\n{'学科':<10}{'干声F0(Hz)':<12}{'性别推测':<10}{'ref_text'}")
    print("-" * 108)
    for p in sorted(glob.glob("*/teacher.json")):
        key = os.path.dirname(p)
        f0 = f0_median(os.path.join(key, "voice_ref.wav"))
        guess = "男声" if 0 < f0 < 160 else ("女声" if f0 >= 160 else "无法判定")
        d = json.load(open(p, encoding="utf-8"))
        print(f"{key:<10}{f0:<12.1f}{guess:<10}{str(d.get('voice_ref_text'))[:36]}")
    if APPLY:
        print("\n[已写回 teacher.json]")


if __name__ == "__main__":
    main()
