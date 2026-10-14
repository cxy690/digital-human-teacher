"""新增一位「同科风格变体」老师：底片归一化 + 干声抽取 + 待机帧标定 + 写档案。

背景
----
项目原本约定「学科 key = 教师档案 key = 数字人 avatar_id」，一个学科只有一位老师。
本脚本用于在同一学科下**追加**一位风格不同的老师（如数学的「严谨型」/「亲和型」），
档案放在 ``data/teachers/<key>/``（key 不再是学科名，如 ``math_strict``），
由 ``teacher.json`` 的 ``subject`` 字段把它归到对应学科。

流程（与既有 ``_normalize_teachers.py`` / ``_recalib.py`` 一脉相承）
--------------------------------------------------------------
1. **底片归一化**：Haar 测人脸 → 裁剪并缩放到 1280x720，使「脸中心 / 脸宽」统一。
   归一化之后各老师的眼位高度一致，因此可以**复用已标定的共享眼位**
   （见 ``_normalize_teachers.py`` 的设计说明）。
2. **抽干声**：ffmpeg 抽 16k 单声道 ``voice_ref.wav``，供 CosyVoice 零样本克隆。
3. **待机帧**：在归一化底片里挑「人脸 + 双眼可检出 + 帧间运动最小」的一帧作为
   ``frame_at``（对应「睁眼 + 张嘴幅度小」的放松帧）。找不到人眼时回退 0.13s。
4. **眨眼**：默认 ``blink_at=null`` —— 前端退化为「眼区压合」合成眨眼，
   几何完全对齐、不会错位；若素材里确有真眨眼，可用 ``_recalib.py --only=<key>`` 精标。
5. **发声方式**：按干声有效语音占比判定 ``clone`` / ``edge``（阈值 0.30，
   与 ``_set_voice_engine.py`` 一致）。

用法
----
    python scripts/add_teacher_variant.py \\
        --subject math --key math_strict \\
        --source ../education_fusion_副本/data/avatars/_src/ai4.mp4 \\
        --variant-name 严谨型 --variant-desc "逻辑严密、步步追问依据" \\
        --persona "你是一位极其严谨的数学老师，讲课像写证明……" \\
        --voice zh-CN-YunjianNeural
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEACHERS_DIR = ROOT / "data" / "teachers"

W, H = 1280, 720
FPS = 24.0

# 目标人脸几何：与 _normalize_teachers.py 完全一致（归一化后各老师眼位可共用标定）
TARGET_FACE_W = 0.238
TARGET_CX, TARGET_CY = 0.50, 0.374

# 归一化底片的共享眼位标定（取现有 9 位归一化老师的共识值，见 data/teachers/*/teacher.json）
SHARED_EYE_POINTS = ((595, 242), (699, 243))
SHARED_EYE_R = (27, 13)

MIN_SPEECH_RATIO = 0.30

# 同学的默认 edge 音色（变体自带一份不同的，保证「换老师＝换声音」）
EDGE_VOICES = {
    "chinese": "zh-CN-XiaoxiaoNeural", "math": "zh-CN-YunxiNeural",
    "english": "zh-CN-XiaoyiNeural", "physics": "zh-CN-YunyangNeural",
    "chemistry": "zh-CN-XiaoyiNeural", "biology": "zh-CN-XiaoxiaoNeural",
    "politics": "zh-CN-YunxiaNeural", "morality": "zh-CN-YunjianNeural",
    "geography": "zh-CN-YunxiNeural", "history": "zh-CN-XiaoxiaoNeural",
    "science": "zh-CN-YunxiNeural",
}

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


# ---------------------------------------------------------------- 1. 底片归一化
def measure_face(path: Path) -> tuple[float, float, float, float] | None:
    """在多帧上检测人脸框，取中位数（单帧检测会受表情/遮挡影响）。"""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    boxes: list[tuple[float, float, float, float]] = []
    for i in list(range(5, total, max(1, total // 12)))[:12]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = FACE_CASCADE.detectMultiScale(
            gray, 1.08, 6, minSize=(int(W * 0.12), int(H * 0.16))
        )
        if len(found):
            boxes.append(tuple(float(v) for v in max(found, key=lambda b: int(b[2]) * int(b[3]))))  # type: ignore[arg-type]
    cap.release()
    if not boxes:
        return None
    return tuple(float(np.median([b[i] for b in boxes])) for i in range(4))  # type: ignore[return-value]


def _src_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or W
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or H
    cap.release()
    return w, h


def normalize(src: Path, dst: Path, wav: Path) -> bool:
    """裁剪 + 缩放，把脸中心/脸宽归一化到与其它老师一致；顺带抽 16k 干声。

    裁剪窗口必须按**源视频尺寸**夹取（素材里有竖屏 720x1280 的源，
    按 1280x720 夹取会得到超出画面的非法 crop 而转码失败）。
    """
    sw, sh = _src_size(src)
    m = measure_face(src)
    if m is None:
        print(f"[WARN] 人脸检测失败，跳过硬裁剪，直接转码：{src.name}", file=sys.stderr)
        cw, ch, x0, y0 = float(sw), float(sh), 0.0, 0.0
    else:
        fx, fy, fw, fh = m
        fcx, fcy = fx + fw / 2, fy + fh / 2
        cw = fw / TARGET_FACE_W
        ch = cw * 9 / 16
        if cw > sw:                      # 宽度不够：退而求其次，先锁宽度再定高度
            cw, ch = float(sw), float(sw) * 9 / 16
        if ch > sh:                      # 高度不够（竖屏）：锁高度再回推宽度
            ch = float(sh)
            cw = min(float(sw), ch * 16 / 9)
            ch = cw * 9 / 16
        x0 = float(np.clip(fcx - TARGET_CX * cw, 0, max(0.0, sw - cw)))
        y0 = float(np.clip(fcy - TARGET_CY * ch, 0, max(0.0, sh - ch)))

    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg()
    vf = f"crop={cw:.0f}:{ch:.0f}:{x0:.0f}:{y0:.0f},scale={W}:{H}"
    proc = subprocess.run(
        [ff, "-y", "-v", "error", "-i", str(src), "-vf", vf,
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(int(FPS)),
         "-c:a", "aac", "-b:a", "128k", str(dst)],
        capture_output=True, text=True, timeout=600,
    )
    if not (dst.is_file() and dst.stat().st_size > 0):
        print(f"[FAIL] 底片转码失败：{proc.stderr.strip()[:300]}", file=sys.stderr)
        return False
    subprocess.run(
        [ff, "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        capture_output=True, timeout=300,
    )
    print(f"[OK] 底片归一化 -> {dst}（crop {cw:.0f}x{ch:.0f}@{x0:.0f},{y0:.0f}）")
    return True


# ---------------------------------------------------------------- 2. 待机帧标定
def _eye_boxes(fx: int, fy: int, fw: int, fh: int):
    """按人脸框几何比例切左右眼区域（比 Haar eye 级联稳，且不受眼镜影响）。"""
    y0, y1 = fy + int(fh * 0.22), fy + int(fh * 0.48)
    lx0, lx1 = fx + int(fw * 0.12), fx + int(fw * 0.46)
    rx0, rx1 = fx + int(fw * 0.54), fx + int(fw * 0.88)
    return (lx0, y0, lx1, y1), (rx0, y0, rx1, y1)


def pick_still_frame(video: Path) -> float:
    """挑待机帧，返回时间（秒）。

    两段式（纯几何，不依赖 mediapipe）：
    1. **眼部纹理** ``Laplacian.var`` 取前 25% —— 睁眼（虹膜/睫毛/眼睑边缘）纹理高，
       闭眼（平滑眼睑）纹理低。这一步用来**排除眨眼帧**：项目文档明确警告过
       「只按运动量最小来抽帧会把闭眼瞬间抽成待机图」。
    2. 在候选帧里取**帧间运动最小** —— 避开说话/晃动造成的高频运动帧。

    首帧没有前一帧可比，其运动量用第 2 帧的值代替（否则恒为 0，会永远胜出）。
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0.13
    prev_gray: np.ndarray | None = None
    tex: dict[int, float] = {}
    motion: dict[int, float] = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            motion[idx] = float(np.mean(cv2.absdiff(gray, prev_gray)))
        prev_gray = gray
        faces = FACE_CASCADE.detectMultiScale(
            cv2.equalizeHist(gray), 1.1, 6, minSize=(int(W * 0.12), int(H * 0.16))
        )
        if len(faces):
            fx, fy, fw, fh = (int(v) for v in max(faces, key=lambda b: int(b[2]) * int(b[3])))
            vals = []
            for (x0, y0, x1, y1) in _eye_boxes(fx, fy, fw, fh):
                if x1 > x0 and y1 > y0:
                    vals.append(float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var()))
            if len(vals) == 2:
                tex[idx] = min(vals)          # 取两眼较小者：一只眼闭了就不算睁眼帧
        idx += 1
    cap.release()

    if not tex:
        print("[WARN] 未检出人脸，待机帧回退 0.13s（可用 _recalib.py 人工精标）", file=sys.stderr)
        return 0.13
    if 0 in motion or 1 in motion:            # 首帧运动量用第 2 帧代替
        motion[0] = motion.get(1, 0.0)

    order = sorted(tex, key=lambda i: -tex[i])
    pool = order[: max(1, len(order) // 4)]   # 纹理前 25%
    best = min(pool, key=lambda i: motion.get(i, float("inf")))
    print(f"[INFO] 待机帧候选 {len(pool)} 帧，选中 t={best / FPS:.2f}s "
          f"(眼部纹理={tex[best]:.0f}, 运动={motion.get(best, float('inf')):.3f})")
    return round(best / FPS, 2)


# ---------------------------------------------------------------- 3. 发声方式判定
def validate_edge_voice(voice: str) -> tuple[bool, str]:
    """校验 edge 音色是否真实存在。

    为什么必须校验：edge-tts 的语音列表很小（zh-CN 只有 8 个），写错音色名时
    ``Communicate`` 不会报错，而是合成出**空音频**，上层悄悄退回默认音色——
    结果就是「换了老师却听不出声音区别」，非常难排查。
    """
    try:
        import asyncio

        import edge_tts

        names = {v["ShortName"] for v in asyncio.run(edge_tts.list_voices())}
    except Exception as err:  # noqa: BLE001 - 离线环境跳过校验
        return True, f"（跳过音色校验：{type(err).__name__}）"
    if voice in names:
        return True, ""
    zh = sorted(n for n in names if n.startswith("zh-CN"))
    return False, f"音色 {voice} 在 edge-tts 中不存在。可用中文音色：{', '.join(zh)}"


def speech_ratio(path: Path) -> float:
    try:
        import soundfile as sf
    except Exception:
        return 0.0
    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception:
        return 0.0
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    win = int(sr * 0.03)
    n = len(data) // win
    if n == 0:
        return 0.0
    rms = np.array([float(np.sqrt(np.mean(data[i * win:(i + 1) * win] ** 2))) for i in range(n)])
    return float(np.mean(rms > max(0.008, rms.max() * 0.08)))


def audio_peak(path: Path) -> float:
    """干声峰值电平：静音/只有背景噪声的素材峰值极低，不能用于音色克隆。"""
    try:
        import soundfile as sf

        data, _ = sf.read(str(path), dtype="float32")
        return float(np.max(np.abs(data))) if len(data) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="新增一位同科风格变体老师")
    ap.add_argument("--subject", required=True, help="学科（key 或中文名），如 math / 数学")
    ap.add_argument("--key", required=True, help="新老师的档案 key（目录名），如 math_strict")
    ap.add_argument("--source", required=True, help="源视频（真人录像 / 数字人素材）")
    ap.add_argument("--name", default="", help="显示名，默认「<学科>老师」")
    ap.add_argument("--variant-name", default="", help="风格名，如「严谨型」")
    ap.add_argument("--variant-desc", default="", help="一句话介绍（选择界面副标题）")
    ap.add_argument("--persona", default="", help="说话风格提示词（替换学科技能里的 persona）")
    ap.add_argument("--greeting", default="", help="专属开场白（进入课堂第一句；留空用学科通用模板）")
    ap.add_argument("--voice", default="", help="edge 兜底音色；留空则用学科默认")
    ap.add_argument("--order", type=int, default=10, help="同科排序（默认老师为 0，变体建议 >=10）")
    ap.add_argument("--skill", default="", help="继承哪个学科技能模块（默认按 subject 推导）")
    ap.add_argument("--eye-points", default="", help="覆盖共享眼位，格式 x1,y1;x2,y2")
    ap.add_argument(
        "--voice-engine", choices=("edge", "auto", "clone"), default="edge",
        help="发声方式：edge=直接用 edge-tts 音色（默认，最稳）；"
             "auto=按干声语音占比判定；clone=强制零样本克隆。"
             "注意：素材音轨常是静音或背景音，宁可 edge 也不要克隆出噪声。",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印将写入的档案，不落盘")
    ap.add_argument("--force", action="store_true", help="音色校验不通过时仍然写入")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from config.subjects import SUBJECT_BY_KEY, SUBJECT_BY_NAME, resolve_subject

    meta = SUBJECT_BY_KEY.get(args.subject) or SUBJECT_BY_NAME.get(args.subject)
    subject_cn = resolve_subject(args.subject)
    skill_key = args.skill or (str((meta or {}).get("key") or args.subject))

    src = Path(args.source)
    if not src.is_absolute():
        src = (ROOT / src).resolve()
    if not src.is_file():
        print(f"[FAIL] 源视频不存在：{src}", file=sys.stderr)
        raise SystemExit(2)

    out_dir = TEACHERS_DIR / args.key
    talk = out_dir / "talk.mp4"
    wav = out_dir / "voice_ref.wav"

    if not args.dry_run:
        if not normalize(src, talk, wav):
            raise SystemExit(1)
    frame_at = pick_still_frame(talk) if talk.is_file() else 0.13
    ratio = speech_ratio(wav) if wav.is_file() else 0.0
    peak = audio_peak(wav) if wav.is_file() else 0.0
    if args.voice_engine == "clone":
        engine = "clone"
    elif args.voice_engine == "auto":
        # 双重门槛：占比达标 + 电平足够（否则「占比高」可能只是恒定背景噪声）
        engine = "clone" if (ratio >= MIN_SPEECH_RATIO and peak >= 0.10) else "edge"
    else:
        engine = "edge"
    voice = args.voice or EDGE_VOICES.get(skill_key, "zh-CN-XiaoxiaoNeural")
    voice_ok, voice_why = validate_edge_voice(voice)
    if not voice_ok:
        print(f"[FAIL] {voice_why}", file=sys.stderr)
        if not args.force:
            raise SystemExit(2)
        print("[WARN] 已指定 --force，仍继续写入该音色", file=sys.stderr)
    elif voice_why:
        print(f"[INFO] {voice_why}")
    # 不用克隆时，留着 wav 只会误导（可能被误判为可克隆音色），直接删掉并清空登记。
    if engine == "edge" and not args.dry_run and wav.is_file():
        wav.unlink()
        print(f"[INFO] 该老师走 edge-tts 音色，已删除 voice_ref.wav"
              f"（原干声 占比={ratio:.3f} 峰值={peak:.3f}）")

    eyes = SHARED_EYE_POINTS
    if args.eye_points:
        right, left = args.eye_points.split(";")
        eyes = (tuple(int(v) for v in right.split(",")), tuple(int(v) for v in left.split(",")))  # type: ignore[assignment]

    cfg_path = out_dir / "teacher.json"
    cfg: dict = {}
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    cfg.update({
        "key": args.key,
        "subject": subject_cn,
        "name": args.name or f"{subject_cn}老师",
        "talk": f"data/teachers/{args.key}/talk.mp4",
        # 有可用干声（语音占比达标）才登记参考音频；否则留空走 edge 音色
        "voice_ref_audio": (
            f"data/teachers/{args.key}/voice_ref.wav" if engine == "clone" else ""
        ),
        "voice_ref_text": cfg.get("voice_ref_text", ""),
        "eye_points": [list(p) for p in eyes],
        "eye_r": list(SHARED_EYE_R),
        "frame_at": frame_at,
        # 变体默认不做真眨眼帧标定 -> 前端走「眼区压合」合成眨眼（几何对齐，不会错位）
        "blink_at": cfg.get("blink_at"),
        "blink_shift": cfg.get("blink_shift", [[0, 0], [0, 0]]),
        "teaching_style": cfg.get("teaching_style", "plain"),
        "calib_source": "归一化底片 + 共享眼位（scripts/add_teacher_variant.py）",
        "voice_engine": engine,
        "voice_speech_ratio": round(ratio, 3),
        "voice": voice,
        "variant_name": args.variant_name,
        "variant_desc": args.variant_desc,
        "variant_order": args.order,
        "skill_key": skill_key,
    })
    if args.persona:
        cfg["persona"] = args.persona
    if args.greeting:
        cfg["greeting"] = args.greeting

    print(f"\n[key={args.key}] 学科={subject_cn} 技能继承={skill_key}")
    print(f"  待机帧 frame_at={frame_at}s  干声占比={ratio:.3f} 峰值={peak:.3f}  发声={engine}  音色={voice}")
    print(f"  风格：{args.variant_name or '(未命名)'} — {args.variant_desc}")

    if args.dry_run:
        print("\n[dry-run] 未写入。")
        return
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 档案已写入 {cfg_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 非交互 stdout 可能不支持
        pass
    main()
