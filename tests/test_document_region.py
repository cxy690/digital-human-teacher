"""题目区域定位（document_parsing/region.py）单测：纯几何 + 文本匹配，无需 OCR。"""
from __future__ import annotations

from document_parsing.ocr import OCRBox
from document_parsing.region import question_region


def _box(text: str, x1: float, y1: float, x2: float, y2: float) -> OCRBox:
    return OCRBox(text=text, bbox=(x1, y1, x2, y2), confidence=0.99)


# 一页两道题：上面语文（y≈0.05~0.14），下面数学（y≈0.55~0.64）
_TWO_QUESTIONS = [
    _box("请背诵并默写古诗静夜思", 0.10, 0.05, 0.90, 0.09),
    _box("然后说说诗人此刻的心情", 0.10, 0.10, 0.80, 0.14),
    _box("解方程 3x+5=20 求 x 的值", 0.10, 0.55, 0.90, 0.59),
    _box("写出完整的解题步骤", 0.10, 0.60, 0.70, 0.64),
]


def test_empty_boxes_returns_none() -> None:
    assert question_region([]) is None


def test_blank_text_lines_are_ignored() -> None:
    boxes = [_box("   ", 0.1, 0.1, 0.5, 0.2), _box("真题目", 0.1, 0.3, 0.5, 0.4)]
    region = question_region(boxes)
    assert region is not None
    x1, y1, x2, y2 = region
    assert round(y1, 3) == 0.292 and round(y2, 3) == 0.408  # 只按有文字的那行算（含外扩）


def test_single_box_returns_box_with_padding() -> None:
    region = question_region([_box("求长方形面积", 0.10, 0.20, 0.80, 0.30)])
    assert region is not None
    x1, y1, x2, y2 = region
    assert 0 <= x1 < 0.10 and y1 < 0.20 and x2 > 0.80 and y2 > 0.30


def test_stem_picks_the_matching_block() -> None:
    """两题同页：题干匹配到哪一块，就框哪一块（而不是罩住整页）。"""
    region = question_region(_TWO_QUESTIONS, stem="解方程 3x+5=20 求 x 的值")
    assert region is not None
    _x1, y1, _x2, y2 = region
    assert y1 > 0.5, "应框住下面那道数学题"
    assert y2 < 0.70


def test_without_stem_takes_first_block() -> None:
    region = question_region(_TWO_QUESTIONS)
    assert region is not None
    _x1, y1, _x2, y2 = region
    assert y1 < 0.10 and y2 < 0.20, "题干缺失时取第一块"


def test_unrelated_stem_falls_back_to_first_block() -> None:
    region = question_region(_TWO_QUESTIONS, stem="完全无关的一句话")
    assert region is not None
    _x1, y1, _x2, _y2 = region
    assert y1 < 0.10


def test_oversized_region_is_rejected() -> None:
    """整页文字被合成一块（分块失败）→ 宁可不画框，也不罩住整页。"""
    page = [
        _box("第一行", 0.02, 0.02, 0.98, 0.20),
        _box("第二行", 0.02, 0.22, 0.98, 0.40),
        _box("第三行", 0.02, 0.42, 0.98, 0.60),
        _box("第四行", 0.02, 0.62, 0.98, 0.96),
    ]
    assert question_region(page) is None


def test_coordinates_are_clamped_into_unit_square() -> None:
    region = question_region([_box("贴边题目", 0.0, 0.0, 1.0, 0.05)])
    assert region is not None
    x1, y1, x2, y2 = region
    assert x1 == 0.0 and y1 == 0.0
    assert x2 <= 1.0 and y2 <= 1.0


def test_degenerate_boxes_are_ignored() -> None:
    """零宽/零高的框（OCR 偶发）不能污染结果。"""
    boxes = [_box("坏框", 0.5, 0.5, 0.5, 0.5), _box("好框", 0.1, 0.1, 0.4, 0.2)]
    region = question_region(boxes)
    assert region is not None
    x1, _y1, x2, _y2 = region
    assert x1 < 0.1 and x2 > 0.4


# ---------------------------------------------------------------- 回合载体
def test_turn_image_region_reads_meta() -> None:
    """StudentTurn.image_region 是白板画框的唯一入口，坏值必须被挡住。"""
    from core.state.models import StudentTurn

    turn = StudentTurn(text="讲讲这道题", image_id="img-1")
    assert turn.image_region is None                      # 没写 → 不画框

    turn.meta["image_region"] = [0.1, 0.2, 0.9, 0.5]
    assert turn.image_region == (0.1, 0.2, 0.9, 0.5)

    for bad in ("abc", [1, 2, 3], [0.5, 0.5, 0.5, 0.5], ["x", 0.1, 0.5, 0.2]):
        turn.meta["image_region"] = bad
        assert turn.image_region is None, f"坏值 {bad!r} 应被拒绝"
