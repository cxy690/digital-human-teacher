"""待机动画换源处理：把外部源片加工为项目内待机循环素材（<name>_idle.mp4 + 兜底图）。

同一工具兼顾「换形象」与「换待机源」两种诉求，参数化后每套素材各自独立输出，互不影响：
    python scripts/proc_idle_replace.py --src "...ai3.7.mp4"                        # 默认：ai3_idle（720x1180）
    python scripts/proc_idle_replace.py --src "...ai4.mp4" --out-name ai4_idle \
        --target-ed 0 --wm-rects "1040,635,1260,705;38,38,228,84" \
        --auto-window 2 --frames-out "data/avatars/_src/ai4_frames"

工艺（沿用 2026-09-16 定稿并扩展）：
1. 水印修复：优先取「整个 ROI 内无水印」的帧组做逐像素时序中值回填（纯色背景上残影为 0），
   找不到这样的帧组时退化为「逐像素挑未覆盖帧取中值」；均带 6px 羽化（移动水印亦可修复，
   见 patch_roi 的两种路径与失效场景说明）。
2. 取景缩放（--target-ed > 0）：源片为特写构图，脸过大既放大口型贴回缝隠、又使融合窗
   变大拖低生成速率（冻帧根因）。按 --target-ed 将全帧等比缩小后底对齐合成到画布，
   上/侧边距补背景灰。**--target-ed 0 = 不缩放，保持源片原始分辨率**（横版源片宜用，
   前端按人脸锚点取景即可，且与口型素材同构图、待机↔播报切换不跳变）。
3. 待机片段（--window START:COUNT 或 --auto-window 秒数，可选）：只取一段做循环（默认全片）。
   待机既要"活着"又不能"自言自语"，故 --auto-window 以**眨眼段为锚点**取窗口
   （整段无眨眼时才退回"嘴型最闭合"帧），再在候选锚点中挑嘴部开合度最低者。
   只按"嘴最闭合"取段会得到一段毫无表情的死帧（2026-09-21 语文/数学待机"不眨眼"根因：
   原参数 --window 34:17 恰好落在两次眨眼之间，剪辑后整段无眨眼）。
   源片里可用的"睁眼+静立"段被不可用段隔开时，可用 --segments "START:COUNT;START:COUNT"
   把多段拼成一个更长的循环体（段间交叉淡化，避免拼缝处位置硬跳；化学 aihx 即此用法）。
4. 无缝循环：正+倒 ping-pong（可保证接缝处位置/速度连续）。是否需要 ping-pong 由
   **人脸区域**的接缝差分判定（见 face_seam_stats）：待机素材常是"背景几乎不动、人脸缓慢漂移"，
   用整帧平均差分判"首尾同帧"会漏判（2026-09-23 化学/英语待机"一跳一跳"根因：整帧均值仅 ~1，
   而人脸实际漂了 4px，直接循环每圈回跳一次，接缝处分差高达段内相邻帧的 8~9 倍）。
5. 编码：livetalking 环境 PyAV libx264 crf18 preset medium + faststart（project_env 无 PyAV）。
6. 兜底图：取处理后首帧导出 <name>.png（与视频首帧同画面，加载替换零感知）。
7. 人脸锚点：mediapipe FaceLandmarker 测虹膜中点/瞳距，打印 IDLE_FACE 常量供前端更新。
8. 帧导出（--frames-out，可选）：把「水印修复后、缩放前」的原始分辨率帧写成 PNG 序列，
   直接喂给 genavatar.py --imgs 生成口型素材包——二者同源同构图，前端缩放比才为 1。

源片同时归档到 data/avatars/_src/（换机器/重加工时唯一不可替代的是裁剪前原始视频）。
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(_ROOT, "clients", "student_tablet")
SRC_ARCHIVE_DIR = os.path.join(_ROOT, "data", "avatars", "_src")
DEFAULT_CANVAS = (720, 1180)          # 竖版画布（历史待机约定）
_WM_DEFAULT = "37,34,232,87;505,1200,695,1245"


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit("视频未读取到任何帧：%s" % path)
    return frames


def imwrite(path, img):
    """PNG 写出（imencode + 文件接口，兼容非 ASCII 路径）。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 编码失败：%s" % path)
    with open(path, "wb") as f:
        f.write(buf.tobytes())


_WM_BRIGHT = 18          # 高于「无字底板」该灰阶即判为水印覆盖（水印为半透明白字，只增亮）
_WM_ABSENT = 20          # 整帧 ROI 内最大增亮低于该灰阶即判「本帧无水印」


def patch_roi(frames, roi):
    """ROI 内用「干净帧」的时序中值回填 + 羽化融合（移动水印亦可修复）。

    首选按**整帧**挑干净帧：水印多为移动/淡入淡出，只要存在一组「整个 ROI 内都没有
    水印」的帧，直接取这组帧的逐像素中值即为真底板（残影为 0）。判定用「本帧 ROI 内
    最大增亮 ≤ _WM_ABSENT」——水印本体增亮几十灰阶，而压缩噪声只有个位数，
    阈值取 20 可稳定分辨。

    退化路径（水印全程压住该 ROI、找不到干净帧组时）：按**像素**挑干净帧 —— 先以时间轴
    10 分位估无字底板（水印只增亮，低分位必落在未覆盖帧上），偏离底板超阈值即判该像素
    在本帧被覆盖，再对每个像素只在未覆盖帧上取中值（无干净帧时退回底板）。

    为什么整帧法优先（2026-09-23 化学 aihxn 源片「待机/讲解底色里仍浮着水印」根因）：
    该源片水印在**纯色墙面**上移动且带半透明淡入淡出，边缘增亮只有 10~18 灰阶 ——
    逐像素阈值法（_WM_BRIGHT=18）会把淡出边缘误判成"干净"，中值里混进边缘灰阶，
    在纯色背景上留下 +9 灰阶、可见的水印浮雕（旧法残影 mean 8.9/max 18.4；
    ai4/aiyw 等片背景有书架/实验台纹理，同样残影被纹理掩盖才没暴露）。
    而按整帧判「有无水印」，淡入淡出帧一律被判"有"，干净帧组纯度足够，残影为 0。
    """
    x1, y1, x2, y2 = roi
    stack = np.asarray([f[y1:y2, x1:x2] for f in frames], dtype=np.float32)
    plate = np.percentile(stack, 10, axis=0)                          # 无字底板（退化路径与兜底用）
    med = np.median(stack, axis=0)
    absent = (stack - med[None]).max(axis=3).max(axis=(1, 2)) <= _WM_ABSENT   # 整帧无水印
    if int(absent.sum()) >= max(8, len(frames) // 5):
        patch = np.median(stack[absent], axis=0)
    else:
        covered = (stack - plate[None]).max(axis=3) > _WM_BRIGHT      # 逐帧逐像素覆盖判定
        n_clean = (~covered).sum(axis=0)
        masked = np.where(covered[..., None], np.inf, stack)          # 被覆盖的值排到时间轴末尾
        masked.sort(axis=0)
        mid = np.clip(n_clean - 1, 0, len(frames) - 1)[None, :, :, None]
        picked = np.take_along_axis(masked, np.broadcast_to(mid, masked.shape), axis=0)[0]
        patch = np.where((n_clean > 0)[..., None], picked, plate)
    patch = patch.astype(np.uint8)
    h, w = patch.shape[:2]
    feather = 6
    alpha = np.ones((h, w), dtype=np.float32)
    ramp = np.linspace(0, 1, feather, dtype=np.float32)
    alpha[:feather, :] *= ramp[:, None]
    alpha[-feather:, :] *= ramp[::-1][:, None]
    alpha[:, :feather] *= ramp[None, :]
    alpha[:, -feather:] *= ramp[::-1][None, :]
    out = []
    for f in frames:
        win = f[y1:y2, x1:x2].astype(np.float32)
        blended = patch.astype(np.float32) * alpha[..., None] + win * (1 - alpha[..., None])
        g = f.copy()
        g[y1:y2, x1:x2] = blended.astype(np.uint8)
        out.append(g)
    return out


def encode(frames, fps, out_mp4):
    import av

    out_c = av.open(out_mp4, "w")
    s = out_c.add_stream("libx264", rate=round(fps))
    s.width, s.height = frames[0].shape[1], frames[0].shape[0]
    s.pix_fmt = "yuv420p"
    s.options = {"crf": "18", "preset": "medium", "movflags": "+faststart"}
    for img in frames:
        pf = av.VideoFrame.from_ndarray(img, format="bgr24")
        for pkt in s.encode(pf):
            out_c.mux(pkt)
    for pkt in s.encode():
        out_c.mux(pkt)
    out_c.close()


_MODEL_ASSET = os.path.join(_ROOT, "data", "models", "face_landmarker.task")


def measure_img(img_bgr):
    """mediapipe 虹膜中点/瞳距 → 前端 FACE 常量元组 (w,h,ex,ey,ed)。"""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_MODEL_ASSET),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lm.detect(mp_img)
    if not res.face_landmarks:
        return None
    pts = res.face_landmarks[0]
    h, w = img_bgr.shape[:2]
    li, ri = pts[468], pts[473]          # 左/右虹膜中心
    ex = (li.x + ri.x) / 2 * w
    ey = (li.y + ri.y) / 2 * h
    ed = ((li.x - ri.x) ** 2 + (li.y - ri.y) ** 2) ** 0.5 * w
    return w, h, round(ex, 1), round(ey, 1), round(ed, 1)


def measure_face(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return measure_img(img)


def make_landmarker():
    """FaceLandmarker 每次构造都要重新加载模型（秒级），批量测帧必须复用一个实例。"""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_MODEL_ASSET),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def _first_face(lm, img_bgr):
    import mediapipe as mp

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return res.face_landmarks[0] if res.face_landmarks else None


def _face_metrics(img_bgr, lm=None):
    """(嘴部开合度, 眼睛开合度)：唇缝/眼睑间距 ÷ 瞳距 归一化；未检出人脸返回 (None, None)。"""
    own = lm is None
    lm = lm or make_landmarker()
    try:
        p = _first_face(lm, img_bgr)
    finally:
        if own:
            lm.close()
    if p is None:
        return None, None
    li, ri = p[468], p[473]              # 左/右虹膜中心
    ed = ((li.x - ri.x) ** 2 + (li.y - ri.y) ** 2) ** 0.5
    if ed <= 0:
        return None, None

    def gap(a, b):
        return (((p[a].x - p[b].x) ** 2 + (p[a].y - p[b].y) ** 2) ** 0.5) / ed

    mouth = gap(13, 14)                  # 内唇 上/下
    ear = (gap(159, 145) + gap(386, 374)) / 2   # 右眼 / 左眼 上/下眼睑
    return round(mouth, 4), round(ear, 4)


def measure_mouth_open(img_bgr, lm=None):
    """嘴部开合度：内唇上下点(13/14)欧氏距 / 瞳距（归一化，跨素材可比）。"""
    return _face_metrics(img_bgr, lm)[0]


def measure_eye_open(img_bgr, lm=None):
    """眼睛开合度 EAR：上下眼睑间距 / 瞳距。睁眼约 0.30、闭眼约 0.01，眨眼时骤降。"""
    return _face_metrics(img_bgr, lm)[1]


def blink_segments(ears, low_ratio=0.72):
    """睁闭眼时序 → 眨眼区间 [(首帧, 末帧), ...]（连续低于阈值的帧聚为一段）。

    阈值取 EAR 低分位的 low_ratio 倍：睁/闭眼相差一个量级，按分位数取阈即可自适应
    不同素材的取景与瞳距，无需逐素材调参。
    """
    items = sorted((i, e) for i, e in ears.items() if e is not None)
    if len(items) < 6:
        return []
    thr = float(np.percentile([e for _, e in items], 25)) * low_ratio
    segs, cur = [], None
    for i, e in items:
        if e < thr:
            cur = [i, i] if cur is None else [cur[0], i]
        elif cur is not None:
            segs.append((cur[0], cur[1]))
            cur = None
    if cur is not None:
        segs.append((cur[0], cur[1]))
    return segs


def zoom_out(frames, target_ed, canvas_w, canvas_h):
    """等比缩小到底对齐的画布；上/侧边距用边缘外延 + 中值 inpainting 填充。

    边距不能用固定背景色（源片带暗角，角点色≠中部背景色会出现矩形色差）；
    底对齐保证下缘无边距，故边缘外延不会复制人物（此前缩待机的条纹伪影根因）。
    返回 (frames, s, x0, y0) 供水印矩形坐标映射。
    """
    face = measure_img(frames[0])
    if not face:
        raise SystemExit("首帧未检测到人脸，无法计算缩放")
    _, _, ex, ey, ed = face
    s = target_ed / ed
    h0, w0 = frames[0].shape[:2]
    if h0 * s > canvas_h:      # 目标瞳距过大时以画布高度为限（底对齐不允许上边距为负）
        s = canvas_h / h0
    sw, sh = int(round(w0 * s)) // 2 * 2, int(round(h0 * s))
    if sw > canvas_w:
        raise SystemExit("缩放后宽度 %d 超出画布 %d：横版源片请用 --target-ed 0（保持原分辨率）"
                         % (sw, canvas_w))
    x0 = (canvas_w - sw) // 2
    y0 = canvas_h - sh                    # 底对齐：下缘不留边距（下缘是人物肩胸）
    margin = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    margin[:y0, :] = 255
    margin[:, :x0] = 255
    margin[:, x0 + sw:] = 255
    bordered = [cv2.copyMakeBorder(cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA),
                                   y0, 0, x0, canvas_w - sw - x0, cv2.BORDER_REPLICATE)
                for f in frames]
    if y0 > 0 or x0 > 0:
        # 边距只 inpaint 一次：对时序中值帧补边得底板，全帧复用（时序稳定不闪、且快）；
        # 中值帧边距内边界：背景行=纯背景，肩胸行=柔和涂抹，扩散后无特征。
        # REPLICATE 出横向条纹、REFLECT 出镜像鬼影发丝、逐帧 inpaint 太慢，均不可用
        plate = np.median(np.asarray(bordered, dtype=np.float32), axis=0).astype(np.uint8)
        plate = cv2.inpaint(plate, margin, 8, cv2.INPAINT_TELEA)
        # 底板与内区交界高斯羽化：消除补边/内区的色调台阶（硬切换会出现横向缝）
        alpha = cv2.GaussianBlur((margin / 255.0).astype(np.float32), (0, 0), 12)
        bordered = [(plate.astype(np.float32) * alpha[..., None]
                     + c.astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)
                    for c in bordered]
    print("取景缩放 s=%.4f（瞳距 %.1f→%.1f），边距 上%d 侧%d" % (s, ed, ed * s, y0, x0))
    return bordered, s, x0, y0


def concat_segments(segs, blend):
    """把多段可用素材拼成一个循环体；段与段之间做 blend 帧线性交叉淡化。

    源片里可用的「睁眼+静立」段常被不可用段（人脸检测失败/长时间闭眼）隔开，拼接能把
    零散可用段合成更长的待机循环（化学 aihx：可用段只有 18 帧与 32 帧，单段不足 1.4s）。
    硬切会在拼缝处出现人物位置跳变，故让前段末尾与后段开头逐帧互相过渡。
    """
    out = list(segs[0])
    for nxt in segs[1:]:
        k = min(blend, len(out), len(nxt))
        for i in range(k):
            a = (i + 1) / (k + 1)          # 前段权重 1→0、后段 0→1，拼缝两端都不留硬边
            out[-k + i] = cv2.addWeighted(out[-k + i], 1 - a, nxt[i], a, 0.0)
        out.extend(nxt[k:])
    return out


def _face_crop(img, anchor):
    """以虹膜中点为心、按瞳距取头部区域（裁到画布内）：接缝比较只在人脸区做。

    anchor 即 measure_img 的返回值 (w, h, ex, ey, ed)。
    """
    _, _, ex, ey, ed = anchor
    h, w = img.shape[:2]
    x1, x2 = max(0, int(ex - 1.5 * ed)), min(w, int(ex + 1.5 * ed))
    y1, y2 = max(0, int(ey - 1.2 * ed)), min(h, int(ey + 2.0 * ed))
    return img[y1:y2, x1:x2].astype(np.float32)


def face_seam_stats(frames, anchor, samples=20):
    """人脸区的 (循环接缝差分, 段内相邻帧差分中位数) → 判断回跳会不会被肉眼看到。

    待机素材常是「背景几乎不动、人脸缓慢漂移」，只比整帧平均差分会把这种漂移误判成
    "首尾同帧"（见文件头工艺 4）。接缝差分明显大于平时的逐帧变化（比值 >3）时，
    循环回跳肉眼可见，必须 ping-pong 镜像。
    """
    first = _face_crop(frames[0], anchor)
    step = max(1, len(frames) // samples)
    steps = []
    for i in range(0, len(frames) - 1, step):
        s = float(np.abs(_face_crop(frames[i + 1], anchor) - _face_crop(frames[i], anchor)).mean())
        steps.append(s)
    seam = float(np.abs(_face_crop(frames[-1], anchor) - first).mean())
    return seam, (float(np.median(steps)) if steps else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="输出目录（默认 clients/student_tablet）")
    ap.add_argument("--out-name", default="ai3_idle",
                    help="输出文件名前缀：<name>.mp4 / <name>.png（默认 ai3_idle）")
    ap.add_argument("--target-ed", type=float, default=139.3,
                    help="目标瞳距 px；0 = 不缩放，保持源片原始分辨率（横版源片用）")
    ap.add_argument("--canvas", default="720x1180",
                    help="缩放目标画布 WxH（默认 720x1180；--target-ed 0 时忽略）")
    ap.add_argument("--fps", type=float, default=24.0, help="输出帧率（默认 24）")
    ap.add_argument("--wm-rects", default=_WM_DEFAULT,
                    help="源片水印矩形（x1,y1,x2,y2 分号分隔，源片坐标系；空串=不处理）")
    ap.add_argument("--window", default="",
                    help="待机循环取用的片段 START:COUNT（源片帧序；空=全片）")
    ap.add_argument("--segments", default="",
                    help="拼接多个可用片段做循环体 \"START:COUNT;START:COUNT\""
                         "（可用段被不可用段隔开时用；优先于 --window）")
    ap.add_argument("--seg-blend", type=int, default=8,
                    help="--segments 拼接处的交叉淡化帧数（默认 8；0 = 硬切）")
    ap.add_argument("--auto-window", type=float, default=0.0,
                    help="自动取片段时长秒数（>0 时以眨眼段为锚点取段，覆盖 --window；"
                         "待机无眨眼说明取段落在两次眨眼之间，此时应加大该值")
    ap.add_argument("--frames-out", default="",
                    help="把水印修复后、缩放前的原始分辨率帧导出为 PNG 序列（喂 genavatar --imgs）")
    args = ap.parse_args()

    cw, ch = [int(v) for v in args.canvas.lower().split("x")]
    out_mp4 = os.path.join(args.out_dir, args.out_name + ".mp4")
    out_png = os.path.join(args.out_dir, args.out_name + ".png")
    os.makedirs(args.out_dir, exist_ok=True)

    frames = read_frames(args.src)
    fps = float(args.fps)
    print("源帧 %d，%dx%d" % (len(frames), frames[0].shape[1], frames[0].shape[0]))

    # ---- 1) 水印修复（源片坐标）----
    for rect in [r for r in args.wm_rects.split(";") if r.strip()]:
        rx1, ry1, rx2, ry2 = [int(float(v)) for v in rect.split(",")]
        w, h = frames[0].shape[1], frames[0].shape[0]
        rx1, ry1 = max(0, rx1 - 3), max(0, ry1 - 3)
        rx2, ry2 = min(w, rx2 + 3), min(h, ry2 + 3)
        print("水印 ROI：(%d,%d,%d,%d)" % (rx1, ry1, rx2, ry2))
        frames = patch_roi(frames, (rx1, ry1, rx2, ry2))

    # ---- 2) 导出修复后原始帧（供 genavatar 生成同构图口型素材）----
    if args.frames_out:
        os.makedirs(args.frames_out, exist_ok=True)
        for i, f in enumerate(frames):
            imwrite(os.path.join(args.frames_out, "%08d.png" % i), f)
        print("[ok] 修复后帧导出 %d 张 → %s" % (len(frames), args.frames_out))

    # ---- 3) 取景缩放（可选）----
    if args.target_ed and args.target_ed > 0:
        frames, s, x0, y0 = zoom_out(frames, args.target_ed, cw, ch)
    else:
        s, x0, y0 = 1.0, 0, 0
        print("保持源片分辨率 %dx%d（不做缩放）" % (frames[0].shape[1], frames[0].shape[0]))

    # ---- 4) 选取待机片段（眨眼为锚、嘴型闭合为优；--segments 拼接多个可用段）----
    start, count = 0, len(frames)
    if args.window.strip():
        start, count = [int(v) for v in args.window.split(":")]
    if args.auto_window > 0:
        step = max(1, int(fps / 4))          # 每 0.25s 采样测一次
        idxs = list(range(0, len(frames), step))
        count = min(len(frames), max(8, int(round(args.auto_window * fps))))
        lm = make_landmarker()               # 复用同一实例：逐帧构造要反复加载模型
        try:
            mouths, ears = {}, {}
            for i in idxs:
                m, e = _face_metrics(frames[i], lm)
                if m is not None:
                    mouths[i] = m
                if e is not None:
                    ears[i] = e
        finally:
            lm.close()
        blinks = blink_segments(ears)
        if blinks:
            anchors = sorted({(a + b) // 2 for a, b in blinks})
            why = "眨眼 %d 段（%s）" % (len(blinks), "、".join("#%d" % a for a in anchors))
        else:                                # 整段无眨眼时退回唯一判据：嘴型最闭合帧
            anchors = [min(mouths, key=mouths.get)] if mouths else [0]
            why = "未检测到眨眼，退回嘴型最闭合帧 #%d" % anchors[0]
        start, score = 0, float("inf")
        for c in anchors:
            # 眨眼落在窗口前 1/3：ping-pong 镜像后两次眨眼间隔更长，不至"连眨"
            s = max(0, min(len(frames) - count, c - count // 3))
            win = [mouths[i] for i in idxs if s <= i < s + count and i in mouths]
            sc = float(np.mean(win)) if win else float("inf")
            if sc < score:
                start, score = s, sc
        print("自动待机片段：%s → 帧 %d..%d（%d 帧，嘴部开合均值 %.4f）"
              % (why, start, start + count - 1, count, score))
    if args.segments.strip():
        segs = []
        for part in [p for p in args.segments.split(";") if p.strip()]:
            s0, c0 = [int(v) for v in part.split(":")]
            if c0 <= 0 or s0 + c0 > len(frames):
                raise SystemExit("--segments 片段越界：%s（源片共 %d 帧）" % (part, len(frames)))
            segs.append(frames[s0:s0 + c0])
        frames = concat_segments(segs, args.seg_blend)
        print("拼接 %d 段（%s），交叉淡化 %d 帧 → 循环体 %d 帧（%.2fs）"
              % (len(segs), "、".join("#%s 起 %d 帧" % (p.split(":")[0], len(s))
                                     for p, s in zip(args.segments.split(";"), segs)),
                 args.seg_blend, len(frames), len(frames) / fps))
    else:
        seg = frames[start:start + count]
        if start or count != len(frames):
            print("待机片段：源帧 %d..%d（%d 帧）" % (start, start + len(seg) - 1, len(seg)))
        frames = seg

    # ---- 5) 无缝循环：人脸区接缝有明显回跳则正+倒 ping-pong ----
    anchor = measure_img(frames[0])
    if not anchor:
        print("警告：首帧未检测到人脸，无法判定接缝 → 直接 ping-pong")
        frames = frames + frames[-2::-1]
    else:
        seam, step = face_seam_stats(frames, anchor)
        whole = float(np.abs(frames[0].astype(int) - frames[-1].astype(int)).mean())
        ratio = seam / step if step > 1e-6 else float("inf")
        print("接缝差分：整帧 %.2f，人脸区 %.2f（段内相邻帧 %.3f → %.1fx）"
              % (whole, seam, step, ratio))
        if whole > 2.0 or seam > max(1.5, 3.0 * step):
            frames = frames + frames[-2::-1]
            print("接缝回跳可见 → ping-pong，共 %d 帧（%.2fs 循环）" % (len(frames), len(frames) / fps))
        else:
            print("接缝与段内变化相当 → 直接循环，共 %d 帧（%.2fs 循环）" % (len(frames), len(frames) / fps))

    # ---- 6) 编码 + 兜底图 ----
    encode(frames, fps, out_mp4)
    imwrite(out_png, frames[0])
    print("[ok] %s / %s" % (out_mp4, out_png))

    # ---- 7) 源片归档 ----
    os.makedirs(SRC_ARCHIVE_DIR, exist_ok=True)
    dst = os.path.join(SRC_ARCHIVE_DIR, os.path.basename(args.src))
    # normcase：Windows 盘符大小写不同（y:\ vs Y:\）时 abspath 字符串不等，源片已在归档目录内会误判
    if os.path.normcase(os.path.abspath(args.src)) != os.path.normcase(os.path.abspath(dst)):
        shutil.copyfile(args.src, dst)
        print("[ok] 源片归档 %s" % dst)

    # ---- 8) 人脸锚点（复用步骤 5 量过的首帧；输出 png 与首帧同画面）----
    face = anchor or measure_face(out_png)
    if face:
        print("%s_FACE = { w: %s, h: %s, ex: %s, ey: %s, ed: %s }" % (args.out_name, *face))
    else:
        print("警告：未检测到人脸，无法给出 %s_FACE 锚点" % args.out_name)


if __name__ == "__main__":
    main()
