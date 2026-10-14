"""端到端链路验证：Qwen3 -> 文本清洗/分句 -> CosyVoice2 -> MuseTalk -> mp4。

设计原则：延续项目「缺依赖 Mock 降级、开箱即跑」——
未配置的服务自动走 Mock，脚本仍能跑完并如实标注每一步用的是真实还是 Mock。

覆盖三条路径：
* ``normal``         ：按当前配置跑全链路（真实服务不在则自动降级）；
* ``mock``           ：强制禁用 MuseTalk（用 MockAvatar），验证降级不中断；
* ``missing-source`` ：把 AI 教师视频指向不存在的路径，验证自动回退旧素材。

用法：
    python -m scripts.e2e_test
    python -m scripts.e2e_test --question "请用一句话介绍牛顿第一定律"
    python -m scripts.e2e_test --out data/e2e
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path

# 项目根加入模块搜索路径，兼容 `python scripts/e2e_test.py` 直接运行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pydantic import BaseModel  # noqa: E402

from config.logging import get_logger, setup_logging  # noqa: E402
from config.settings import settings  # noqa: E402
from core.avatar_source import resolve_teacher_video  # noqa: E402
from models.avatar import MockAvatar, get_avatar  # noqa: E402
from models.llm import get_llm  # noqa: E402
from models.tts import get_tts, split_sentences  # noqa: E402

log = get_logger("scripts.e2e_test")

DEFAULT_QUESTION = "请用一句话介绍牛顿第一定律"

# --------------------------------------------------------------- 文本清洗（可测）
_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_MD_LIST = re.compile(r"^\s{0,3}[-*+]\s+", re.M)
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_)")
_WS = re.compile(r"\s+")


def clean_markdown(text: str) -> str:
    """清洗 Markdown：去代码块/行内代码/链接/标题/列表/强调符，压缩空白。"""
    t = _CODE_BLOCK.sub(" ", text or "")
    t = _INLINE_CODE.sub(r"\1", t)
    t = _MD_LINK.sub(r"\1", t)
    t = _MD_HEADER.sub("", t)
    t = _MD_LIST.sub("", t)
    t = _MD_EMPHASIS.sub("", t)
    return _WS.sub(" ", t).strip()


def _audio_ext(data: bytes) -> str:
    """按魔数猜测音频扩展名。"""
    if data[:4] == b"RIFF":
        return "wav"
    return "mp3"


# --------------------------------------------------------------- 结果模型（Pydantic）
class StepResult(BaseModel):
    name: str
    seconds: float
    ok: bool
    detail: str = ""


class E2EReport(BaseModel):
    label: str
    question: str
    answer: str = ""
    audio_path: str = ""
    video_path: str = ""
    degraded: bool = False
    ok: bool = False
    detail: str = ""
    total_seconds: float = 0.0
    steps: list[StepResult] = []


# --------------------------------------------------------------- 人脸一致性（代理校验）
def check_face_consistency(src_video: Path, out_video: Path) -> tuple[bool, str]:
    """检查输出视频首帧人脸是否与源视频首帧一致（框位置/尺寸代理）。

    说明：MediaPipe 只给检测框、无身份 embedding，故用**归一化人脸框中心距离 +
    尺寸比**做代理判定；真正的身份一致性需人脸识别模型（本项目未引入）。
    opencv/mediapipe 缺失或任一视频无人脸时返回 (False, 原因)。
    """
    try:
        import cv2  # noqa: PLC0415 - 可选依赖
        import mediapipe as mp  # noqa: PLC0415 - 可选依赖
    except Exception as err:  # noqa: BLE001 - 可选依赖缺失
        return False, f"缺少 opencv/mediapipe，跳过人脸一致性检查（{err}）"

    det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def _first_face_box(path: Path):
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        h, w = frame.shape[:2]
        res = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.detections:
            return None
        bb = res.detections[0].location_data.relative_bounding_box
        return (bb.xmin + bb.width / 2, bb.ymin + bb.height / 2, bb.width, bb.height, w, h)

    try:
        s = _first_face_box(src_video)
        o = _first_face_box(out_video)
    finally:
        det.close()
    if s is None:
        return False, f"源视频检测不到人脸：{src_video.name}"
    if o is None:
        return False, f"输出视频检测不到人脸：{out_video.name}"
    center_dist = ((s[0] - o[0]) ** 2 + (s[1] - o[1]) ** 2) ** 0.5
    size_ratio = (o[2] * o[3]) / max(s[2] * s[3], 1e-9)
    ok = center_dist < 0.08 and 0.5 <= size_ratio <= 2.0
    detail = f"人脸框中心距={center_dist:.4f}（<0.08）、面积比={size_ratio:.2f}（0.5~2.0）"
    return ok, detail


# --------------------------------------------------------------- 单条链路
async def run_pipeline(
    question: str, out_dir: Path, *, avatar_override=None, label: str = "normal"
) -> E2EReport:
    """跑一次「LLM -> 清洗 -> TTS -> 数字人 -> mp4」，返回结构化报告。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = E2EReport(label=label, question=question)
    t_start = time.perf_counter()

    def step(name: str, t0: float, ok: bool, detail: str = "") -> None:
        report.steps.append(StepResult(name=name, seconds=round(time.perf_counter() - t0, 3), ok=ok, detail=detail))

    def fail(idx: int, name: str, detail: str) -> E2EReport:
        report.detail = f"失败于第 {idx} 步：{name}（{detail}）"
        return _finish(report, out_dir, t_start)

    # 1) LLM 生成回答
    t0 = time.perf_counter()
    llm = get_llm()
    try:
        answer = await llm.chat([{"role": "user", "content": question}])
    except Exception as err:  # noqa: BLE001
        step("Qwen3 生成回答", t0, False, f"异常：{err}")
        return fail(1, "Qwen3 生成回答", str(err))
    report.answer = answer or ""
    step("Qwen3 生成回答", t0, bool(answer), "MockLLM" if getattr(llm, "is_mock", False) else "真实 LLM")
    (out_dir / "answer.txt").write_text(report.answer, encoding="utf-8")
    if not answer:
        return fail(1, "Qwen3 生成回答", "未返回内容")

    # 2) 文本清洗 + 分句
    t0 = time.perf_counter()
    cleaned = clean_markdown(answer)
    sentences = split_sentences(cleaned)
    step("文本清洗/分句", t0, bool(sentences), f"{len(sentences)} 句")
    if not sentences:
        return fail(2, "文本清洗/分句", "清洗后无有效文本")

    # 3) TTS 合成
    t0 = time.perf_counter()
    tts = get_tts()
    try:
        audio = await tts.synthesize(cleaned)
    except Exception as err:  # noqa: BLE001
        step("CosyVoice2 合成语音", t0, False, f"异常：{err}")
        return fail(3, "CosyVoice2 合成语音", str(err))
    provider = "MockTTS" if getattr(tts, "is_mock", False) else type(tts).__name__
    if audio:
        audio_path = out_dir / f"speech.{_audio_ext(audio)}"
        audio_path.write_bytes(audio)
        report.audio_path = str(audio_path)
    step("CosyVoice2 合成语音", t0, bool(audio), f"{provider}，{len(audio)} 字节")

    # 4) 数字人渲染
    avatar = avatar_override if avatar_override is not None else get_avatar()
    t0 = time.perf_counter()
    if not getattr(avatar, "produces_video", False):
        step("MuseTalk 生成视频", t0, False, f"{type(avatar).__name__}：不产视频（puppet）")
        report.degraded = True
        return fail(4, "MuseTalk 生成视频", "数字人降级为 puppet（无视频）")
    if not audio:
        step("MuseTalk 生成视频", t0, False, "无音频（TTS 未产出），无法渲染")
        report.degraded = True
        return fail(4, "MuseTalk 生成视频", "无音频")
    render_err = ""
    try:
        result = await avatar.render(audio, text=cleaned)
    except Exception as err:  # noqa: BLE001
        render_err = str(err)
        result = {"mode": "puppet"}
    if result.get("mode") != "video" or not result.get("video"):
        detail = f"异常：{render_err}" if render_err else f"返回 mode={result.get('mode')}（降级）"
        step("MuseTalk 生成视频", t0, False, detail)
        report.degraded = True
        return fail(4, "MuseTalk 生成视频", detail)
    video_path = out_dir / "avatar.mp4"
    video_path.write_bytes(base64.b64decode(result["video"]))
    report.video_path = str(video_path)
    step("MuseTalk 生成视频", t0, True, f"{type(avatar).__name__}，{video_path.stat().st_size} 字节")

    # 5) 人脸一致性（输出 vs 源视频）
    t0 = time.perf_counter()
    src, _used_fallback = resolve_teacher_video()
    face_ok, face_detail = check_face_consistency(Path(src), video_path)
    step("人脸一致性检查", t0, face_ok, face_detail)

    report.ok = all(s.ok for s in report.steps[:4])
    report.detail = "全链路通过" if report.ok else "存在失败步骤"
    return _finish(report, out_dir, t_start)


def _finish(report: E2EReport, out_dir: Path, t_start: float) -> E2EReport:
    """收尾：写总耗时并落盘 report.json。"""
    report.total_seconds = round(time.perf_counter() - t_start, 3)
    (out_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


# --------------------------------------------------------------- 场景三：源缺失回退
def run_missing_source_scenario() -> E2EReport:
    """把主源指向不存在的文件，验证自动回退到 fallback。"""
    report = E2EReport(label="missing-source", question="(源路径校验)")
    t0 = time.perf_counter()
    primary_bak = settings.ai_teacher_video_path
    try:
        settings.ai_teacher_video_path = "data/ai_teacher/__not_exist__.mp4"
        path, used_fallback = resolve_teacher_video()
    finally:
        settings.ai_teacher_video_path = primary_bak
    report.steps.append(
        StepResult(
            name="源缺失自动回退",
            seconds=round(time.perf_counter() - t0, 3),
            ok=used_fallback and Path(path).is_file(),
            detail=f"使用 {path}，used_fallback={used_fallback}",
        )
    )
    report.ok = report.steps[0].ok
    report.total_seconds = round(time.perf_counter() - t0, 3)
    report.detail = "回退生效" if report.ok else "回退未按预期生效"
    return report


# --------------------------------------------------------------- 报告输出
def _print_report(report: E2EReport) -> None:
    log.info("—— 场景【{}】——", report.label)
    for i, s in enumerate(report.steps, 1):
        mark = "OK " if s.ok else "FAIL"
        log.info("  步骤{} [{}] {}：{:.3f}s {}", i, mark, s.name, s.seconds, s.detail)
    if report.degraded:
        log.warning("  ↳ 降级（degraded）：{}", report.detail)
    if report.video_path:
        log.info("  最终视频：{}", report.video_path)
    log.info("  总耗时：{:.3f}s", report.total_seconds)


def _verdict(reports: list[E2EReport]) -> tuple[bool, str]:
    """判定整体通过与否，并给出「失败于第 N 步」信息。

    mock 场景本就是验证降级（不产视频是预期），不参与通过判定。
    """
    for r in reports:
        if r.label == "mock":
            continue
        if not r.ok:
            for i, s in enumerate(r.steps, 1):
                if not s.ok:
                    return False, f"失败于第 {i} 步（场景 {r.label}：{s.name}）"
            return False, f"失败于场景 {r.label}"
    return True, "链路打通"


async def _amain(args: argparse.Namespace) -> int:
    out_root = Path(args.out)
    log.info("== E2E 链路验证开始（问题：{}）==", args.question)

    reports: list[E2EReport] = []
    reports.append(await run_pipeline(args.question, out_root / "normal", label="normal"))
    reports.append(
        await run_pipeline(
            args.question, out_root / "mock", avatar_override=MockAvatar(), label="mock"
        )
    )
    reports.append(run_missing_source_scenario())

    for r in reports:
        _print_report(r)

    ok, msg = _verdict(reports)
    if ok:
        log.info("✅ {}", msg)
        return 0
    log.error("❌ {}", msg)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 教师数字人端到端链路验证")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="测试问题")
    parser.add_argument("--out", default="data/e2e", help="输出目录（默认 data/e2e）")
    args = parser.parse_args()

    # 强制 UTF-8 输出，保证 ✅/❌ 等符号在 Windows 控制台正常显示
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    setup_logging()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
