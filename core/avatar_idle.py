"""待机素材：静态「睁眼 + 闭嘴」帧 + 一张同构图闭眼帧（供前端眨眼）。

为什么需要：
* 原始待机底片是「有人在说话」的视频，嘴一直在动，不能直接当待机画面；
* 讲解视频是引擎按音频重绘的，讲完回到待机时要做平滑过渡，待机画面越接近讲解
  画面越好。

静态帧来源（**关键**）：
* 首选：底片 ``talk.mp4`` 的 ``frame_at`` —— 该时刻由标定保证是「睁眼 + 闭嘴 + 中性」，
  且底片是未压缩原图，**最清晰、嘴最自然**；
* 兜底：引擎静音渲染视频的 ``_STILL_FROM_RENDER_AT``。它与讲解视频同域，但**静音段模型
  仍会把嘴渲得微张/撅起**（MuseTalk 只按音频重建「嘴 + 上下唇」，对纯静音没有闭唇约束），
  实测观感明显更差 —— 所以不再默认用它，只在底片抽帧失败时启用。

为了不让待机变成「一张死图」，再合成一张**闭眼帧**供前端做自然眨眼：
* 首选：底片里**真实闭眼的那一帧**（只把其眼区抠回静态帧，最自然）；无真帧时用
  「上眼睑皮肤下压 + 保留原下眼睑/睫毛 + 柔和睫线」合成，避免旧的纵向压合糊成一条带。

前端在「睁眼帧 ↔ 闭眼帧」两张**整帧图**之间按真人三阶段节奏短促切换即可。
用整帧图而非 CSS 局部位移，是因为头像走 ``object-fit:cover`` 裁切，
百分比坐标会随窗口宽高比漂移；整帧图则与裁切/窗口尺寸完全无关。

产物（缓存于 ``data/avatar_stream/``，文件名带算法版本号，升级后旧缓存自动失效）：
* ``idle_still{suffix}.mp4`` 引擎静音渲染视频（**可选**，仅作无静态帧时的兜底）
* ``idle_still_v2{suffix}.png`` 睁眼静态帧（优先取自底片 ``frame_at``）
* ``idle_blink_v2{suffix}.png`` 闭眼帧（眨眼用）

失败一律返回 None，前端保持原底片。
"""
from __future__ import annotations

import asyncio
import base64
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from config.logging import get_logger
from config.teachers import load_teacher
from core.audio_utils import _resolve_ffmpeg
from core.avatar_stream import STREAM_DIR, STREAM_URL_PREFIX
from models.avatar import get_avatar

log = get_logger("core.avatar_idle")

_IDLE_MP4 = "idle_still.mp4"
_IDLE_PNG = "idle_still.png"
_IDLE_BLINK_PNG = "idle_blink_v4.png"
_SILENCE_SECONDS = 1.6
_FRAME_AT = 0.13          # 抽帧时间点：该处为「睁眼、闭嘴」的安静状态
_STILL_FROM_RENDER_AT = 0.4  # 待机静态帧改从「引擎静音渲染视频」取帧的时刻（秒）：
                              # 与讲解视频同域，消除交叉淡化时的亚像素重影

# 眼位标定（源图像素坐标，按 1280x720 底图量出）：左眼 / 右眼中心。
# 需要重新标定时，改默认值或设环境变量 AVATAR_EYE_POINTS="x1,y1;x2,y2"。
_EYES = ((605, 235), (730, 230))
_EYE_RX, _EYE_RY = 46, 22     # 眼区半宽 / 半高（用于闭眼帧抠取 / 合成）

_BLINK_SRC_AT = 6.25          # 底片中「真实闭眼」时刻（秒，= 第 150 帧 @24fps），已人工核对


def _env_eyes() -> tuple[tuple[int, int], tuple[int, int]]:
    """"x1,y1;x2,y2" 形式的眼位覆盖（可选）。"""
    raw = os.getenv("AVATAR_EYE_POINTS", "").strip()
    if not raw:
        return _EYES
    try:
        pts = [tuple(int(v) for v in seg.split(",")) for seg in raw.split(";") if seg.strip()]
        if len(pts) == 2:
            return ((pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]))
    except Exception:  # noqa: BLE001 - 配置错误则用默认标定
        log.warning("AVATAR_EYE_POINTS 解析失败，用默认眼位：{}", raw)
    return _EYES


def _film_path() -> Path:
    """旧单底片路径：只认显式环境变量；**不再回退到任何 底片.mp4**。"""
    return Path((os.getenv("AVATAR_FACE_VIDEO", "") or "").strip())


def _make_silence() -> bytes:
    """生成一小段纯静音 mp3（供引擎渲染出「闭嘴」画面）。"""
    ff = _resolve_ffmpeg()
    if not ff:
        return b""
    try:
        STREAM_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STREAM_DIR / "_silence_tmp.mp3"
        subprocess.run(
            [
                ff, "-y",
                "-f", "lavfi", "-t", str(_SILENCE_SECONDS), "-i", "anullsrc=r=44100:cl=mono",
                "-c:a", "libmp3lame", "-q:a", "4", str(tmp),
            ],
            check=True, capture_output=True, timeout=30,
        )
        data = tmp.read_bytes()
        try:
            tmp.unlink()
        except OSError:
            pass
        return data
    except Exception as err:  # noqa: BLE001 - 失败则回退原底片
        log.warning("make silence failed: {}", err)
        return b""


def _extract_frame(video: Path, out: Path, at: float) -> bool:
    """从视频中抽取一帧为 PNG（用于「完全静止」的待机画面）。"""
    ff = _resolve_ffmpeg()
    if not ff or not video.is_file():
        return False
    try:
        subprocess.run(
            [ff, "-y", "-ss", str(at), "-i", str(video), "-frames:v", "1", str(out)],
            check=True, capture_output=True, timeout=30,
        )
        return out.is_file() and out.stat().st_size > 0
    except Exception as err:  # noqa: BLE001
        log.warning("extract idle frame failed: {}", err)
        return False


def _detect_blink_at(calib: "_Calib") -> float | None:
    """从底片里自动定位「眼睛真正闭上」的那一帧，返回其时间（秒）；找不到返回 None。

    背景：部分底片的 ``blink_at`` 标定是 null（自动标定漏检），但底片里**其实有真实眨眼**
    （眼皮是真闭上的）。真闭眼帧贴回来最自然，远胜任何合成。
    指标：眼区「暗部占比」—— 睁眼时瞳孔+睫毛让眼区偏暗，闭眼时暗部明显减少；
    取全片暗部最少的一帧，若明显低于中位数即判为真眨眼。

    只用 ffmpeg + numpy/Pillow（主应用环境没有 cv2）：一次性把底片缩放导出为 24fps 的
    帧序列，再逐帧打分。失败一律返回 None（调用方退回合成）。
    """
    ff = _resolve_ffmpeg()
    if not ff or not calib.film.is_file():
        return None
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    rx, ry = calib.eye_r
    tmp = tempfile.mkdtemp(prefix="_blinkscan_")
    try:
        subprocess.run(
            [ff, "-y", "-i", str(calib.film), "-t", "30",
             "-vf", "scale=480:-1,fps=24", os.path.join(tmp, "f%05d.png")],
            check=True, capture_output=True, timeout=180,
        )
        files = sorted(Path(tmp).glob("f*.png"))
        if len(files) < 8:
            return None
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        k = 480.0 / max(1, _probe_width(calib.film))
        boxes = [
            (int((cx - rx) * k), int((cy - ry) * k), int((cx + rx) * k), int((cy + ry) * k))
            for cx, cy in calib.eyes
        ]
        scores: list[tuple[float, int]] = []
        for idx, png in enumerate(files):
            try:
                g = np.asarray(Image.open(png).convert("L"), dtype=np.float32)
            except Exception:  # noqa: BLE001 - 单帧读失败不影响整体
                continue
            vals = []
            for (a, b, c, d) in boxes:
                roi = g[max(0, b):d, max(0, a):c]
                if roi.size >= 16:
                    vals.append(float((roi < (roi.mean() - 22)).mean()))
            if vals:
                scores.append((sum(vals) / len(vals), idx))
        if len(scores) < 8:
            return None
        arr = np.array([s for s, _ in scores], dtype=np.float32)
        med = float(np.median(arr))
        if med <= 0:
            return None
        best, idx = scores[int(np.argmin(arr))]
        if best >= 0.85 * med:                      # 暗部没明显减少 -> 底片没真眨眼
            return None
        return idx / 24.0                           # 导出时统一按 24fps
    except Exception as err:  # noqa: BLE001 - 检测失败不影响主流程
        log.warning("自动检测眨眼帧失败：{}", err)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _probe_width(film: Path) -> int:
    """读底片宽度（像素），用于把眼位坐标缩放到「打分用的低分辨率帧」上。失败回退 1280。"""
    ff = _resolve_ffmpeg()
    if not ff or not str(film) or not Path(film).is_file():
        return 1280
    try:
        import re  # noqa: PLC0415

        out = subprocess.run([ff, "-i", str(film)], capture_output=True, text=True, timeout=20)
        m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", out.stderr or "")
        if m:
            return int(m.group(1))
    except Exception:  # noqa: BLE001 - 探测失败用默认值
        pass
    return 1280


def _blink_source_time() -> float:
    """取得底片中「真实闭眼」的时刻（秒）。

    * 可用 ``AVATAR_BLINK_SRC_AT``（秒）覆盖，换底片后重新标定即可；
    * 默认值已在本项目底片上人工核对（该帧眼睑闭合、且头位与静态帧重合）。
    """
    env = os.getenv("AVATAR_BLINK_SRC_AT", "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            log.warning("AVATAR_BLINK_SRC_AT 非数字，改用默认值：{}", env)
    return _BLINK_SRC_AT


def _frame_at_env() -> float:
    """抽帧点（秒）：可用 ``AVATAR_FRAME_AT`` 覆盖。"""
    env = os.getenv("AVATAR_FRAME_AT", "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            log.warning("AVATAR_FRAME_AT 非数字，改用默认值：{}", env)
    return _FRAME_AT


@dataclass(frozen=True)
class _Calib:
    """一位老师的待机标定：底片 / 眼位 / 眼区半径 / 抽帧点 / 闭眼时刻 / 眨眼对齐偏移。"""

    film: Path
    eyes: tuple[tuple[int, int], tuple[int, int]]
    eye_r: tuple[int, int]
    frame_at: float
    blink_at: float | None
    shift: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))
    blink_src: Path | None = None      # 独立闭眼帧素材（底片本身没有眨眼时用）


def _calib_for(teacher_key: str = "") -> _Calib | None:
    """取某位老师的标定；无档案且未显式配置旧底片时返回 ``None``（不再用旧老师素材）。"""
    profile = load_teacher(teacher_key) if teacher_key else None
    if profile is not None:
        return _Calib(profile.talk_path, profile.eye_points, profile.eye_r,
                      profile.frame_at, profile.blink_at, profile.blink_shift,
                      profile.blink_src_path)
    film = _film_path()
    if not str(film) or not film.is_file():
        log.warning("无教师档案（key={}）且未配置 AVATAR_FACE_VIDEO，跳过待机素材生成",
                    teacher_key or "default")
        return None
    return _Calib(film, _env_eyes(), (_EYE_RX, _EYE_RY),
                  _frame_at_env(), _blink_source_time())


def _feather_mask(w: int, h: int, inset: float, blur_ratio: float):
    """生成居中、四边羽化到 0 的椭圆蒙版（贴回时不留矩形边）。"""
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new("L", (w, h), 0)
    ix, iy = int(w * inset), int(h * inset)
    ImageDraw.Draw(mask).ellipse((ix, iy, w - 1 - ix, h - 1 - iy), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(1, int(min(w, h) * blur_ratio))))


def _eye_crop_box(im, cx: int, cy: int, rx: int, ry: int, pad_x: int, pad_y: int):
    return (
        max(0, cx - rx - pad_x), max(0, cy - ry - pad_y),
        min(im.width, cx + rx + pad_x), min(im.height, cy + ry + pad_y),
    )


def _paste_eye(im, patch, box, rx: int, ry: int) -> None:
    """把眼部补丁按「轻羽化椭圆」贴回：羽化半径随眼区大小自适应（小眼区不再糊成一团）。"""
    from PIL import Image, ImageDraw, ImageFilter  # noqa: PLC0415 - 可选依赖

    w, h = patch.size
    mask = Image.new("L", (w, h), 0)
    inset = max(2, min(rx, ry) // 3)
    ImageDraw.Draw(mask).ellipse((inset, inset, w - 1 - inset, h - 1 - inset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1.0, min(rx, ry) * 0.22)))
    im.paste(patch, (box[0], box[1]), mask)


def _donor_eye(donor, im, cx: int, cy: int, rx: int, ry: int, shift):
    """从真闭眼素材裁出这只眼睛的对应区域（含少量眼周皮肤）。返回 (patch, box)。"""
    from PIL import Image  # noqa: PLC0415

    box = _eye_crop_box(im, cx, cy, rx, ry, max(6, rx // 3), max(4, ry // 3))
    dx, dy = shift
    sx0, sy0, sx1, sy1 = box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy
    if 0 <= sx0 and 0 <= sy0 and sx1 <= donor.width and sy1 <= donor.height:
        eye = donor.crop((sx0, sy0, sx1, sy1))
    else:
        eye = donor.crop(box)
    size = (box[2] - box[0], box[3] - box[1])
    if eye.size != size:
        eye = eye.resize(size, Image.LANCZOS)
    return eye, box


def _match_tone(donor_patch, still_patch):
    """把 donor 眼部补丁的整体色调对齐到 still 同位置。

    真闭眼帧取自**底片**、待机静态帧取自**引擎渲染画面**，两者曝光/色温常有细微差异，
    直接贴回会在眼周留下一圈「光晕」。这里用补丁**外圈皮肤**的均值差做一次整体平移，
    把 donor 拉到 still 的色调上，消除光晕而不改变闭眼结构。
    """
    try:
        import numpy as np  # noqa: PLC0415 - 可选依赖

        d = np.asarray(donor_patch, dtype=np.float32)
        s = np.asarray(still_patch, dtype=np.float32)
        if d.shape != s.shape or d.ndim != 3:
            return donor_patch
        h, w = d.shape[:2]
        m = max(2, min(h, w) // 6)
        ring = np.zeros((h, w), dtype=bool)
        ring[:m, :] = ring[-m:, :] = ring[:, :m] = ring[:, -m:] = True
        if int(ring.sum()) < 16:
            return donor_patch
        shift = s[ring].mean(axis=0) - d[ring].mean(axis=0)
        from PIL import Image  # noqa: PLC0415

        return Image.fromarray(np.clip(d + shift, 0, 255).astype(np.uint8))
    except Exception:  # noqa: BLE001 - 色调对齐失败则用原补丁
        return donor_patch


def _lid_skin(im, cx: int, cy: int, rx: int, ry: int):
    """取「眼裂正上方」的上眼睑皮肤，作为合成闭眼时的填充源。

    为什么不用眼睛外侧（太阳穴）的皮肤：那块离眼睛远、光照/色调往往不同，铺上去像贴了一块
    假皮肤。紧贴上眼睑的那一条皮肤色调最接近，且自带眼睑的明暗，铺下来最像"眼皮垂下来"。
    取样带取在眉下、眼裂之上的一条窄带，尽量避免把眉毛带进来。
    """
    from PIL import ImageFilter  # noqa: PLC0415

    top = cy - ry
    y0 = max(0, int(top - ry * 0.80))
    y1 = max(y0 + 2, int(top - ry * 0.05))
    x0 = max(0, int(cx - rx * 1.15))
    x1 = min(im.width, int(cx + rx * 1.15))
    band = im.crop((x0, y0, x1, y1))
    if band.width < 4 or band.height < 2:        # 越界/太窄时退回眼区自身
        band = im.crop((max(0, cx - rx), max(0, cy - ry), min(im.width, cx + rx), min(im.height, cy + ry)))
    return band.filter(ImageFilter.GaussianBlur(1.0))


def _synth_closed_eye(im, cx: int, cy: int, rx: int, ry: int, side: int):
    """把一只「睁眼」合成为自然闭眼。返回 (patch, box)。

    要点（踩过的坑）：
    * **必须把整个眼裂都用眼睑皮肤盖住**。早期版本只盖上半、保留原下眼睑与下睫毛，
      结果原下睫毛 + 新画的睫线 = **两条线**，看起来像"下眼皮翻到上面来"。
    * 睫线**只画一条**，位置在眼裂约 52% 高度（闭眼时上下睑在中间合拢），向下微弓；
      上方补一条很淡的褶影增加立体感，不再做任何纵向压合（压合会把瞳孔拖成糊带）。
    """
    from PIL import Image, ImageDraw, ImageFilter  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    pad_x, pad_y = 5, 3
    box = _eye_crop_box(im, cx, cy, rx, ry, pad_x, pad_y)
    w, h = box[2] - box[0], box[3] - box[1]
    lid = im.crop(box)
    gray = np.asarray(lid.convert("L"), dtype=np.float32)   # 供探测「本帧的上睫线」用
    skin = _lid_skin(im, cx, cy, rx, ry).resize((w, h), Image.LANCZOS)

    ap = (pad_x, pad_y, w - 1 - pad_x, h - 1 - pad_y)     # 眼裂椭圆
    cover = Image.new("L", (w, h), 0)
    ImageDraw.Draw(cover).ellipse(ap, fill=255)
    cover = cover.filter(ImageFilter.GaussianBlur(max(1, int(ry * 0.18))))
    # 眼睑弧面：越靠眼裂上方越暗（眉骨下的阴影），越靠下越亮
    shade = Image.new("L", (w, h), 0)
    sd = ImageDraw.Draw(shade)
    for yy in range(h):
        t = (yy - pad_y) / max(1, h - 2 * pad_y)
        sd.line([(0, yy), (w, yy)], fill=int(max(0.0, 104 * (0.55 - t))))
    skinned = Image.composite(Image.new("RGB", (w, h), (60, 46, 42)), skin, shade)
    lid = Image.composite(skinned, lid, cover)

    # ---- 睫线：沿用「本帧自己的上睫线形状」，整体下移到眼裂中部 ----
    # 直接画一条对称圆弧会变成"横着的一道线"（眼睛本身是斜的，形状对不上）。
    # 这里逐列探测原图的上睫线（眼裂内自上而下第一个偏暗的行），把它平移下来，
    # 于是斜度/弧度都跟这位老师一致。
    xs = np.arange(w, dtype=np.float32)
    within = (xs >= pad_x) & (xs <= w - 1 - pad_x)
    band1 = int(pad_y + (h - 2 * pad_y) * 0.72)
    y_lash = np.full(w, np.nan, dtype=np.float32)
    for i in range(w):
        if not within[i]:
            continue
        col = gray[pad_y:band1, i]
        if col.size < 3:
            continue
        hit = np.where(col < (float(np.median(col)) - 24.0))[0]
        if hit.size:
            y_lash[i] = pad_y + hit[0]
    valid = ~np.isnan(y_lash)
    center = pad_y + (h - 2 * pad_y) * 0.52          # 闭眼时上下睑合拢的位置
    if int(valid.sum()) >= 3:
        y_lash = np.interp(xs, xs[valid], y_lash[valid])
        y_lash = np.convolve(y_lash, np.ones(7, dtype=np.float32) / 7.0, mode="same")
        shift = center - float(y_lash[within].mean())
    else:                                            # 探测失败：退回几何估计
        y_lash = pad_y + (h - 2 * pad_y) * 0.30 * np.ones(w, dtype=np.float32)
        shift = center - float(y_lash.mean())
    line_y = y_lash + shift

    t = np.clip((xs - pad_x) / max(1.0, float(xs[within].max() - xs[within].min())), 0, 1)
    taper = np.clip(np.minimum(t, 1 - t) / 0.18, 0, 1)   # 两端渐隐，避免端头生硬

    def _line_mask(half_h: float, y_off: float, blur: float, strength: float):
        m = np.zeros((h, w), dtype=np.float32)
        for i in range(w):
            if not within[i]:
                continue
            y = line_y[i] + y_off
            a = int(max(0, round(y - half_h)))
            b = int(min(h, round(y + half_h) + 1))
            if b > a:
                m[a:b, i] = 255.0
        m *= taper[None, :]
        m = Image.fromarray(m.astype("uint8")).filter(ImageFilter.GaussianBlur(blur))
        a = np.asarray(m, dtype=np.float32) * strength
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))

    # 睫线上方的淡褶影（给闭眼一点立体感，不是一整块死皮肤）
    lid = Image.composite(Image.new("RGB", (w, h), (96, 78, 72)), lid,
                          _line_mask(max(0.8, ry * 0.34), -max(1.4, ry * 0.48),
                                     max(1.0, ry * 0.20), 0.26))
    dark = tuple(int(c * 0.62) for c in lid.getpixel((w // 2, int(max(0, line_y[w // 2])))))
    lid = Image.composite(Image.new("RGB", (w, h), dark), lid,
                          _line_mask(max(0.8, ry * 0.13), 0.0, max(0.7, ry * 0.10), 0.55))
    return lid, box


def _eyes_differ(a, b) -> bool:
    """两组眼部裁图是否有明显差异（用来判断「真闭眼帧」是否被误标定成睁眼）。"""
    try:
        from PIL import ImageChops  # noqa: PLC0415

        ga = a.convert("L")
        gb = b.convert("L")
        diff = ImageChops.difference(ga, gb)
        hist = diff.histogram()
        total = sum(hist) or 1
        mean = sum(i * n for i, n in enumerate(hist)) / total
        return mean >= 6.0
    except Exception:  # noqa: BLE001
        return True


def _make_blink_frame(still_png: Path, out_png: Path, calib: _Calib) -> bool:
    """由「睁眼」静态帧生成「闭眼」帧（供前端眨眼切换）。

    优先级：
      1) 底片/独立素材里的**真实闭眼帧** —— 只把其眼区贴回静态帧（最自然），
         羽化保持很轻（旧版羽化过重，会把好端端的真闭眼糊成一条带）。若该眼区与
         睁眼帧几乎无差异（autocalib 误标定），视为没有真闭眼帧，转 2)；
      2) **合成**：把上眼睑皮肤下压、保留下眼睑与睫毛、补一条柔和睫线 ——
         对「底片里根本没有眨眼」的老师同样能得到自然的闭眼。
    纯图像域处理，与前端窗口尺寸 / cover 裁切无关。
    """
    if not still_png.is_file():
        return False
    try:
        from PIL import Image  # noqa: PLC0415 - 可选依赖
    except Exception:  # noqa: BLE001
        log.info("PIL 不可用，跳过闭眼帧合成（待机将无眨眼）")
        return False
    try:
        im = Image.open(still_png).convert("RGB")
        rx, ry = calib.eye_r

        # ---- 1) 真实闭眼帧 ----
        donor = None
        blink_t = calib.blink_at
        if blink_t is None and (calib.blink_src is None or not calib.blink_src.is_file()):
            # 标定缺失：底片里**可能其实有真眨眼**（自动标定漏检）。现场扫一遍全片，
            # 找到「眼睛真正闭上」的那一帧就直接用 —— 真闭眼远比合成自然。
            blink_t = _detect_blink_at(calib)
            if blink_t is not None:
                log.info("闭眼帧：标定缺失，自动检测到真眨眼 t={:.2f}s", blink_t)
        if calib.blink_src is not None and calib.blink_src.is_file():
            donor = Image.open(calib.blink_src).convert("RGB")
            log.info("闭眼帧：使用独立闭眼素材 {} 偏移={}", calib.blink_src.name, calib.shift)
        elif blink_t is not None:
            donor_png = STREAM_DIR / "_blink_donor.png"
            if _extract_frame(calib.film, donor_png, blink_t):
                donor = Image.open(donor_png).convert("RGB")
                log.info("闭眼帧：使用底片真实闭眼帧 t={:.2f}s 偏移={}", blink_t, calib.shift)
                try:
                    donor_png.unlink()          # 用完即删，不在 stream 目录留临时帧
                except OSError:
                    pass

        if donor is not None and donor.size == im.size:
            patches = []
            for idx, (cx, cy) in enumerate(calib.eyes):
                eye, box = _donor_eye(donor, im, cx, cy, rx, ry, calib.shift[idx])
                still_eye = im.crop(box)
                eye = _match_tone(eye, still_eye)          # 消除底片域 vs 渲染域的曝光差光晕
                patches.append((eye, box, still_eye))
            if any(_eyes_differ(p[0], p[2]) for p in patches):
                for eye, box, _ in patches:
                    _paste_eye(im, eye, box, rx, ry)
                out_png.parent.mkdir(parents=True, exist_ok=True)
                im.save(out_png)
                return out_png.is_file() and out_png.stat().st_size > 0
            log.info("闭眼帧：底片标注的闭眼帧与睁眼无差异（疑似误标定），改用合成")

        # ---- 2) 合成闭眼（**默认关闭**）----
        # 实测：从睁眼硬合成闭眼，结果普遍像"糊了一块"（比不眨眼更差），而且绝大多数的
        # 底片其实都能扫到真眨眼（见 _detect_blink_at）。所以默认**宁可不眨眼**，也不糊眼。
        # 需要时置 AVATAR_BLINK_SYNTH=1 打开合成兜底。
        if os.getenv("AVATAR_BLINK_SYNTH", "") != "1":
            log.info("闭眼帧：底片无真眨眼，跳过眨眼（如需合成兜底请设 AVATAR_BLINK_SYNTH=1）")
            return False
        for idx, (cx, cy) in enumerate(calib.eyes):
            side = -1 if idx == 0 else 1
            lid, box = _synth_closed_eye(im, cx, cy, rx, ry, side)
            _paste_eye(im, lid, box, rx, ry)
        log.info("闭眼帧：使用上眼睑合成（底片无可用真闭眼帧）")

        out_png.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_png)
        return out_png.is_file() and out_png.stat().st_size > 0
    except Exception as err:  # noqa: BLE001
        log.warning("make blink frame failed: {}", err)
        return False


def _url(p: Path) -> str:
    """待机 / 封面素材的访问 URL（带 ``?v=<mtime>`` 版本号）。

    这些素材的文件名是**固定**的（``idle_still_<老师key>.png`` 等），而
    ``/media/avatar_stream`` 是普通静态挂载（浏览器会按 URL 缓存）：换了老师底片后
    服务端文件确实重建了，但浏览器仍会拿同名 URL 的旧图 —— 表现为「待机层 / 封面
    还是上一位老师」。附上文件 mtime 当版本号，素材一更新 URL 就变，
    既能在更新时自动失效，又不影响平时的缓存复用。
    """
    try:
        version = int(p.stat().st_mtime)
    except OSError:
        version = 0
    return f"{STREAM_URL_PREFIX}/{p.name}?v={version}"


async def ensure_teacher_portrait(teacher_key: str = "") -> str | None:
    """老师形象缩略图（底片待机帧，缓存于 ``avatar_stream/``），返回 URL | None。

    供「同科多老师」选择界面展示真人形象：直接抽底片的 ``frame_at``（已标定为
    「睁眼 + 闭嘴」），比走引擎静音渲染快得多（毫秒级 vs 数秒），
    因此可以放心地在选择界面一次性取回同科所有老师的头像。
    """
    key = (teacher_key or "").strip()
    if not key:
        return None
    calib = _calib_for(key)
    if calib is None:
        return None
    png = STREAM_DIR / f"portrait_{key}.png"
    if not (png.is_file() and png.stat().st_size > 0):
        try:
            STREAM_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        ok = await asyncio.to_thread(_extract_frame, calib.film, png, calib.frame_at)
        if not ok:
            return None
    return _url(png)


async def ensure_idle_assets(teacher_key: str = "") -> dict[str, str | None]:
    """确保存在静音待机素材，返回 ``{'video', 'still', 'blink'}``（URL | None，幂等）。

    多老师：按 ``teacher_key``（= 学科 key）生成并复用**该老师专属**的待机素材
    （文件名带该 key 后缀，互不覆盖）；为空时沿用旧的单老师文件名（向后兼容）。
    """
    suffix = f"_{teacher_key}" if teacher_key else ""
    mp4 = STREAM_DIR / f"idle_still{suffix}.mp4"
    # 静态帧文件名带版本号：来源改为「底片 frame_at」后旧缓存自动失效（无需删文件）
    png = STREAM_DIR / f"idle_still_v2{suffix}.png"
    # 眨眼帧文件名带算法版本号：合成算法升级后旧缓存自动失效（无需删文件、无破坏性操作）
    blink = STREAM_DIR / f"idle_blink_v4{suffix}.png"
    calib = _calib_for(teacher_key)
    if calib is None:
        return {"video": None, "still": None, "blink": None}

    async def _assets() -> dict[str, str | None]:
        """按需补齐静态帧 / 闭眼帧，返回三个 URL。

        待机静态帧**优先从老师底片 ``talk.mp4`` 的 ``frame_at`` 抽帧** —— 该时刻由
        标定保证是「睁眼 + 闭嘴 + 中性」，且底片是未压缩原图，**最清晰、嘴最自然**。
        引擎静音渲染帧（``_STILL_FROM_RENDER_AT``）虽与讲解同域，但静音段模型仍会把嘴
        渲得微张/撅起（MuseTalk 只按音频重建嘴+上下唇，无闭唇约束），观感更差，
        故降为兜底（底片抽帧失败时才用）。
        """
        if not (png.is_file() and png.stat().st_size > 0):
            picked = await asyncio.to_thread(_extract_frame, calib.film, png, calib.frame_at)
            if not picked and mp4.is_file() and mp4.stat().st_size > 0:
                picked = await asyncio.to_thread(
                    _extract_frame, mp4, png, _STILL_FROM_RENDER_AT)
                if picked:
                    log.info("静态帧：底片抽帧失败，退回引擎渲染帧 t={}", _STILL_FROM_RENDER_AT)
        # 眨眼帧必须与静态帧同源同姿态：缺失、为空、或**比静态帧旧**（静态帧换了来源/重建过）
        # 都要重新生成，否则会拿旧姿态的闭眼图去眨，整帧跳一下。
        need_blink = (
            png.is_file() and (
                not (blink.is_file() and blink.stat().st_size > 0)
                or png.stat().st_mtime > blink.stat().st_mtime
            )
        )
        if need_blink:
            await asyncio.to_thread(_make_blink_frame, png, blink, calib)
        return {
            "video": _url(mp4) if mp4.is_file() else None,
            "still": _url(png) if png.is_file() else None,
            "blink": _url(blink) if blink.is_file() else None,
        }

    # 先用「底片抽帧」这条路：毫秒级、无需引擎，绝大多数老师在这里就拿到 still/blink，
    # 不必等数秒的静音渲染。仅当底片拿不到静态帧时才回退到引擎静音渲染（见下）。
    try:
        fast = await _assets()
        if fast.get("still"):
            return fast
    except OSError:
        pass

    avatar = get_avatar()
    if not getattr(avatar, "produces_video", False):
        return {"video": None, "still": None, "blink": None}
    try:
        audio = await asyncio.to_thread(_make_silence)
        if not audio:
            return {"video": None, "still": None, "blink": None}
        # 用该老师自己的底片渲染静音画面（待机与讲解同源，切换不跳）
        result = await avatar.render(audio, text="", teacher_id=teacher_key)
        video_b64 = str(result.get("video") or "")
        if result.get("mode") != "video" or not video_b64:
            log.warning("idle render skipped: mode={} teacher={}", result.get("mode"), teacher_key)
            return {"video": None, "still": None, "blink": None}
        STREAM_DIR.mkdir(parents=True, exist_ok=True)
        mp4.write_bytes(base64.b64decode(video_b64))
        assets = await _assets()
        log.info("静音待机素材已生成（{}）：{} / {} / {}",
                 teacher_key or "default", mp4, png, blink)
        return assets
    except Exception as err:  # noqa: BLE001 - 失败则前端保持原底片
        log.warning("idle assets failed（{}）：{}", teacher_key or "default", err)
        return {"video": None, "still": None, "blink": None}
