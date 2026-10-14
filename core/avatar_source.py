"""AI 教师驱动视频源：路径解析、文件哈希与底片规格校验。

换教师时只需替换 ``data/ai_teacher/talk.mp4``（或改 ``.env`` 的
``AI_TEACHER_VIDEO_PATH``）。文件缺失或校验不通过时自动回退到
``AI_TEACHER_FALLBACK_PATH``（默认与主底片同源），保证「缺素材也能开箱即跑」。

依赖策略（延续项目 Mock 降级风格，缺依赖不阻断启动）：
* opencv 缺失 -> 跳过分辨率 / 时长 / 帧率检查（记录 WARNING，仍判定通过）；
* mediapipe 缺失 -> 跳过人脸检查（记录 WARNING，仍判定通过）；
* 文件读不到 / 无有效帧 -> 判定失败，交由调用方回退。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from config.logging import get_logger
from config.settings import settings

log = get_logger("core.avatar_source")

# 人脸校验时均匀抽取的帧数（兼顾耗时与覆盖度）
_SAMPLE_FRAMES = 12


def compute_file_hash(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """计算文件内容的 SHA-256（截取前 16 位）作为缓存 key。

    以内容而非路径做 key：即便换了文件名，只要内容相同缓存仍可复用；
    内容一变（换教师）缓存 key 自动改变 -> 天然失效重建。

    入参：``path`` 目标文件；``chunk_size`` 分块读取大小（默认 1MB）。
    返回：16 位十六进制摘要。
    异常：文件不存在时抛 ``FileNotFoundError``。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _check_single_face(cap: object, frame_count: int) -> tuple[bool, str]:
    """抽样校验人脸：要求全程稳定检测到 1 张脸。

    返回 ``(是否通过, 说明)``。mediapipe 缺失时跳过（判定通过）。
    """
    try:
        import cv2  # noqa: PLC0415 - 可选依赖
        import mediapipe as mp  # noqa: PLC0415 - 可选依赖
    except Exception as err:  # noqa: BLE001 - 可选依赖缺失不阻断
        log.warning("mediapipe/cv2 不可用（{}），跳过人脸校验", err)
        return True, "mediapipe 不可用，已跳过人脸校验"

    step = max(1, frame_count // _SAMPLE_FRAMES)
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    counts: list[int] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = detector.process(rgb)
                counts.append(len(res.detections) if res.detections else 0)
            idx += 1
    finally:
        detector.close()

    if not counts:
        return False, "未能抽取到有效帧进行人脸检测"
    if all(c == 1 for c in counts):
        return True, f"人脸稳定 1 张（采样 {len(counts)} 帧）"
    if all(c == 0 for c in counts):
        return False, "未检测到人脸"
    return False, f"人脸数量不稳定：采样 {counts}（需全程 1 张）"


def validate_teacher_video(path: Path) -> tuple[bool, str]:
    """校验驱动视频是否为可用的 MuseTalk 底片。

    检查项：
    * 文件存在且可被 opencv 打开、有有效帧、时长 > 0；
    * 分辨率最短边 >= ``settings.ai_teacher_min_side``（默认 512）；
    * 帧率 >= ``settings.ai_teacher_min_fps``（默认 24，MuseTalk 标准）；
    * 均匀抽样帧稳定检测到 1 张人脸。

    返回 ``(是否通过, 原因说明)``。opencv 缺失时跳过规格校验（判定通过）。
    """
    p = Path(path)
    if not p.is_file():
        return False, f"文件不存在：{p}"

    try:
        import cv2  # noqa: PLC0415 - 可选依赖
    except Exception as err:  # noqa: BLE001 - 可选依赖缺失不阻断
        log.warning("opencv 不可用（{}），跳过分辨率/帧率校验：{}", err, p)
        return True, "opencv 不可用，已跳过规格校验"

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return False, f"无法用 opencv 打开：{p}"
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0:
            return False, f"视频无有效帧或参数异常（{width}x{height} {fps}fps {frame_count}帧）"

        duration = frame_count / fps
        if duration <= 0:
            return False, "视频时长 <= 0"

        min_side = min(width, height)
        if min_side < settings.ai_teacher_min_side:
            return False, (
                f"分辨率过低：最短边 {min_side} < {settings.ai_teacher_min_side}"
                f"（{width}x{height}）"
            )

        if fps < settings.ai_teacher_min_fps:
            return False, f"帧率过低：{fps:.2f} < {settings.ai_teacher_min_fps}（MuseTalk 要求）"

        face_ok, face_msg = _check_single_face(cap, frame_count)
        if not face_ok:
            return False, face_msg

        return True, f"通过：{width}x{height} {fps:.2f}fps {duration:.2f}s，{face_msg}"
    finally:
        cap.release()


def _resolve_with_reason() -> tuple[Path, bool, bool, str]:
    """内部：解析生效视频源，返回 ``(路径, 是否回退, 是否可用, 原因)``。"""
    primary = Path(settings.ai_teacher_video_path)
    ok, reason = validate_teacher_video(primary)
    if ok:
        log.info("AI 教师驱动视频：{}（{}）", primary, reason)
        return primary, False, True, reason

    fallback = Path(settings.ai_teacher_fallback_path)
    log.warning("主驱动视频不可用：{} -> {}；回退到 {}", primary, reason, fallback)
    fb_ok, fb_reason = validate_teacher_video(fallback)
    if fb_ok:
        log.warning("已回退到回退底片：{}（{}）", fallback, fb_reason)
    else:
        log.error("回退素材同样不可用：{}（{}）；请检查 data/ai_teacher/ 或配置", fallback, fb_reason)
    return fallback, True, fb_ok, fb_reason


def resolve_teacher_video() -> tuple[Path, bool]:
    """解析最终使用的驱动视频：优先主源，失败则回退。

    返回 ``(最终使用的路径, 是否使用了回退)``。两端都不可用时仍返回回退路径
    并记录 ERROR，交由引擎侧自行降级，不阻断应用启动。
    """
    path, used_fallback, _ok, _reason = _resolve_with_reason()
    return path, used_fallback


class AvatarSourceInfo(BaseModel):
    """AI 教师视频源的可观测描述（作为编排/监控侧的“源节点”输出契约）。"""

    path: str
    video_hash: str = ""          # 内容哈希：换源即变，用于缓存失效与下游重跑判定
    clip_dir: str
    cache_dir: str
    used_fallback: bool
    valid: bool
    detail: str = ""


def get_avatar_source_info() -> AvatarSourceInfo:
    """解析当前生效的驱动视频并返回结构化信息。

    ``video_hash`` 变化即代表“AI 教师视频源”变更 -> 人脸缓存自动失效、下游可据此重跑。
    """
    path, used_fallback, ok, reason = _resolve_with_reason()
    vhash = compute_file_hash(path) if path.is_file() else ""
    return AvatarSourceInfo(
        path=str(path),
        video_hash=vhash,
        clip_dir=settings.ai_teacher_clip_dir,
        cache_dir=settings.avatar_cache_dir,
        used_fallback=used_fallback,
        valid=ok,
        detail=reason,
    )


__all__ = [
    "compute_file_hash",
    "validate_teacher_video",
    "resolve_teacher_video",
    "AvatarSourceInfo",
    "get_avatar_source_info",
]
