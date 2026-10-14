"""题目区域定位：从 OCR 行框里圈出「这一道题」的外接框。

为什么需要它：拍照讲题要在线图上圈出题目位置。但学生拍来的照片往往**不止一道题**
（整页作业/试卷），把所有行框合并只会得到一个「罩住整页」的大框 —— 没有指向性，
学生看不出老师在讲哪一道。所以这里：

1. 按**垂直间距**把行框聚成块（一道题 ≈ 一块，行与行贴得近、题与题之间留白大）；
2. 用题干文本挑出对应的那一块（字符 bigram 重叠度最高者）；
3. 取该块所有行框的并集，外扩一点边距作为「题目区域」。

边界与取舍：
* 挑不出块 / OCR 什么都没识别到 → 返回 ``None``，**宁可不画框**（画错了比不画更糟：
  学生盯着错的地方，反而听不懂）；
* 并集面积超过 ``max_area``（如分块失败、一页挤了三道题）→ 同样返回 ``None``：
  一个罩住整页的框等于没框，还挡住了讲解文字。
  这与 ``LecturerAgent._MAX_ANNOTATE_AREA``（课件标注过大就不标）是同一取舍。

纯几何 + 纯文本匹配，不依赖 OCR / 模型，因此可离线单测。
"""
from __future__ import annotations

from statistics import median
from typing import Any, Iterable

#: 归一化外接框 (x1, y1, x2, y2)，取值 0~1
BBox = tuple[float, float, float, float]

#: 块面积上限：超过就认为「没切出单题」，不画框
DEFAULT_MAX_AREA = 0.85
#: 断块阈值 = max(行中心间距中位数 × gap_ratio, min_gap)
DEFAULT_GAP_RATIO = 2.2
DEFAULT_MIN_GAP = 0.03
#: 区域外扩边距（让框不要把字切掉）
DEFAULT_PAD = 0.008


def question_region(
    boxes: Iterable[Any],
    stem: str = "",
    *,
    max_area: float = DEFAULT_MAX_AREA,
    gap_ratio: float = DEFAULT_GAP_RATIO,
    min_gap: float = DEFAULT_MIN_GAP,
    pad: float = DEFAULT_PAD,
) -> BBox | None:
    """从 OCR 行框里算出「这一道题」的外接框；算不出时返回 ``None``。

    ``boxes``：任何带 ``text`` 与 ``bbox``（``x1, y1, x2, y2``，0~1 归一化）的对象
    （即 :class:`document_parsing.ocr.OCRBox`）。``stem`` 用于挑块，可为空。
    """
    rows = _clean(boxes)
    if not rows:
        return None

    rows.sort(key=lambda r: (r[1] + r[3]) / 2)
    blocks = _split_blocks(rows, max(_median_gap(rows) * gap_ratio, min_gap))
    target = _pick_block(blocks, stem)
    if target is None:
        return None

    x1 = max(0.0, min(r[0] for r in target) - pad)
    y1 = max(0.0, min(r[1] for r in target) - pad)
    x2 = min(1.0, max(r[2] for r in target) + pad)
    y2 = min(1.0, max(r[3] for r in target) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) * (y2 - y1) > max_area:
        return None
    return (round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))


# ---------------------------------------------------------------- 内部
def _clean(boxes: Iterable[Any]) -> list[tuple[float, float, float, float, str]]:
    """过滤空文本/退化框，并把坐标规整为左上-右下。"""
    out: list[tuple[float, float, float, float, str]] = []
    for box in boxes or []:
        text = str(getattr(box, "text", "") or "").strip()
        raw = getattr(box, "bbox", None)
        if not text or not raw or len(raw) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in raw)
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if right - left <= 0 or bottom - top <= 0:
            continue
        out.append((left, top, right, bottom, text))
    return out


def _median_gap(rows: list[tuple[float, float, float, float, str]]) -> float:
    """相邻行「中心间距」的中位数（rows 需已按 y 排序）。"""
    centers = [(r[1] + r[3]) / 2 for r in rows]
    gaps = [b - a for a, b in zip(centers, centers[1:]) if b - a > 0]
    return median(gaps) if gaps else 0.0


def _split_blocks(
    rows: list[tuple[float, float, float, float, str]], threshold: float
) -> list[list[tuple[float, float, float, float, str]]]:
    """按垂直间距断块：间距超过 ``threshold`` 视为换了一道题。"""
    blocks: list[list[tuple[float, float, float, float, str]]] = []
    current: list[tuple[float, float, float, float, str]] = []
    last_center: float | None = None
    for row in rows:
        center = (row[1] + row[3]) / 2
        if last_center is not None and center - last_center > threshold and current:
            blocks.append(current)
            current = []
        current.append(row)
        last_center = center
    if current:
        blocks.append(current)
    return blocks


def _pick_block(
    blocks: list[list[tuple[float, float, float, float, str]]], stem: str
) -> list[tuple[float, float, float, float, str]] | None:
    """挑出题干所在的块；题干缺失或完全匹配不上时取第一块。

    题干可能只是 OCR 全文（未做结构化），此时所有块得分都很高，取最高分即第一块 ——
    与「学生先拍的通常是第一道题」的直觉一致。
    """
    if not blocks:
        return None
    stem = (stem or "").strip()
    if not stem:
        return blocks[0]
    scores = [_coverage("".join(r[4] for r in block), stem) for block in blocks]
    best = max(range(len(blocks)), key=lambda i: scores[i])
    return blocks[best] if scores[best] > 0 else blocks[0]


def _coverage(text: str, stem: str) -> float:
    """``stem`` 的字符 bigram 有多大比例出现在 ``text`` 里（0~1）。"""
    want = _bigrams(stem)
    if not want:
        return 0.0
    return len(_bigrams(text) & want) / len(want)


def _bigrams(text: str) -> set[str]:
    text = "".join(str(text or "").split())
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}
