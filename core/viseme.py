"""视素时间线：把 AI 话术转成粗略口型序列，供前端数字人驱动嘴唇形状。

零重依赖：优先用可选依赖 ``pypinyin`` 取拼音，缺失时返回空列表（前端回退能量驱动）。
输出为 ``[[viseme, dur_ms], ...]``，前端可按音频实际时长等比缩放。
"""
from __future__ import annotations

import re

from config.logging import get_logger

log = get_logger("core.viseme")

# 视素类（与前端 SVG 口型分组 data-viseme 一一对应）
VISEMES = ("rest", "mbp", "a", "i", "o", "u", "e")

try:
    from pypinyin import Style, lazy_pinyin

    _HAS_PYPINYIN = True
except ImportError:  # pragma: no cover - 取决于环境
    _HAS_PYPINYIN = False

_CJK = re.compile(r"[\u4e00-\u9fff]")
_PAUSE = re.compile(r"[，。！？、；：,.!?;:…\s]")

_CHAR_MS = 200      # 一个正常汉字的基准时长
_PAUSE_MS = 140     # 标点/停顿的基准时长


def _final_viseme(pinyin: str) -> str:
    """由拼音粗略推断口型类别（取主元音，双唇音优先级最高）。"""
    py = (pinyin or "").lower()
    if not py:
        return "rest"
    if py[0] in ("b", "p", "m"):
        return "mbp"  # 双唇音：起始闭唇
    if "a" in py:
        return "a"
    if "o" in py:
        return "o"
    if "e" in py:
        return "e"
    if "i" in py:
        return "i"
    if "u" in py or "v" in py or "ü" in py:
        return "u"
    return "rest"


def _pinyin_of(ch: str) -> str:
    if not _HAS_PYPINYIN:
        return ""
    try:
        return lazy_pinyin(ch, style=Style.NORMAL)[0]
    except Exception:  # noqa: BLE001 - 单字取拼音失败不影响整体
        return ""


def text_to_visemes(text: str) -> list[str]:
    """文本 -> 视素类序列（标点/停顿记为 rest）。"""
    seq: list[str] = []
    for ch in text or "":
        if _CJK.match(ch):
            seq.append(_final_viseme(_pinyin_of(ch)))
        elif _PAUSE.match(ch):
            seq.append("rest")
    return seq


def build_viseme_timeline(text: str, *, ms_per_char: int = _CHAR_MS) -> list[list]:
    """文本 -> ``[[viseme, dur_ms], ...]``；pypinyin 缺失或无有效内容时返回空列表。"""
    if not _HAS_PYPINYIN:
        return []
    timeline: list[list] = []
    for ch in text or "":
        if _CJK.match(ch):
            timeline.append([_final_viseme(_pinyin_of(ch)), ms_per_char])
        elif _PAUSE.match(ch):
            timeline.append(["rest", _PAUSE_MS])
    return timeline
