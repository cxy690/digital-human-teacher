"""课件标注样式测试：单行文字画圈、跨行/非文字用虚线框。"""
from __future__ import annotations

from agents.shared.whiteboard import (
    ANNOTATE_BOX,
    ANNOTATE_ELLIPSE,
    ANNOTATE_UNDERLINE,
    Whiteboard,
)
from config.constants import BoardOp
from core.courseware.models import SlideElement


def _el(**kw) -> SlideElement:
    base = dict(id="e1", text="1．频率的稳定性", kind="text", x=0.1, y=0.2, width=0.2, height=0.05)
    base.update(kw)
    return SlideElement(**base)


def test_single_line_text_is_circled() -> None:
    op = Whiteboard().annotate_element(_el(), deck_id="数学", page_index=3)
    assert op.op is BoardOp.ANNOTATE
    assert op.meta["style"] == ANNOTATE_ELLIPSE
    assert op.meta["element_kind"] == "text"
    assert op.meta["element_id"] == "e1"


def test_wrapped_paragraph_keeps_box() -> None:
    """跨行的整块内容圈不住，必须退回避线框。"""
    el = _el(text="一般地，随着试验次数n的增加，频率偏离概率的幅度会缩小，事件A发生的频率",
             height=0.24)
    op = Whiteboard().annotate_element(el, deck_id="数学", page_index=3)
    assert op.meta["style"] == ANNOTATE_BOX


def test_overlong_single_line_keeps_box() -> None:
    """一行但特别长，圈出来会横跨大半页，也用框。"""
    el = _el(text="a" * 40, height=0.05)
    assert Whiteboard.annotate_style(el) == ANNOTATE_BOX


def test_non_text_element_keeps_box() -> None:
    el = _el(text="", kind="image", width=0.5, height=0.4)
    assert Whiteboard.annotate_style(el) == ANNOTATE_BOX


def test_explicit_style_wins() -> None:
    op = Whiteboard().annotate_element(
        _el(), deck_id="数学", page_index=3, style=ANNOTATE_UNDERLINE
    )
    assert op.meta["style"] == ANNOTATE_UNDERLINE


def test_highlight_still_uses_highlight_op() -> None:
    op = Whiteboard().annotate_element(_el(), deck_id="数学", page_index=3, kind="highlight")
    assert op.op is BoardOp.HIGHLIGHT
    assert "style" not in op.meta          # 高亮不需要样式分支


def test_annotate_coords_are_clamped() -> None:
    op = Whiteboard().annotate_element(
        _el(x=-0.5, y=1.8, width=2.0, height=0.05), deck_id="数学", page_index=1
    )
    assert op.x == 0.0 and op.y == 1.0 and op.width == 1.0 and op.height == 0.05
