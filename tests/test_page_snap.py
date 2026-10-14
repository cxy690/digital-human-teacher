"""页图吸附回归测试。

背景：pptx 字体的真实字号/行距定义在母版里，python-pptx 读不到，解析出的元素框
会随行号线性漂移（实测到第 7 行偏出 0.38 个页高）——表现为「虚线框圈不准」。
这里用行/列投影把框吸附到页图真实文字行上。
"""
from __future__ import annotations

import pytest

from core.courseware.image_snap import PageInk, snap_shape_elements
from core.courseware.models import SlideElement

pytest.importorskip("PIL")
pytest.importorskip("numpy")

from PIL import Image, ImageDraw  # noqa: E402

W, H = 400, 300
# 三条「文字行」：(上, 下, 左, 右)
LINES = [(40, 65, 40, 200), (100, 125, 40, 160), (160, 185, 40, 250)]


def _page(path, lines=LINES, bg: int = 255) -> str:
    img = Image.new("L", (W, H), bg)
    d = ImageDraw.Draw(img)
    for top, bot, left, right in lines:
        d.rectangle([left, top, right, bot], fill=0)
    img.save(path)
    return str(path)


def test_bands_found_at_real_text_rows(tmp_path) -> None:
    ink = PageInk.load(tmp_path / "p1.png")  # 不存在 -> 安全返回 None
    assert ink is None

    ink = PageInk.load(_page(tmp_path / "p2.png"))  # 传 str 也能用
    assert ink is not None
    assert ink.w == W and ink.h == H
    assert len(ink.bands) == len(LINES)
    for (y0, y1), (top, bot, _, _) in zip(ink.bands, LINES):
        assert abs(y0 - top) <= 2 and abs(y1 - bot) <= 2


def test_snaps_element_to_its_real_line(tmp_path) -> None:
    """元素估值偏上时，吸附后必须落到真实文字行上（这是「圈不准」的根因）。"""
    ink = PageInk.load(_page(tmp_path / "p.png"))
    el = SlideElement(id="a", text="标题行", kind="text", x=0.05, y=0.02, width=0.30, height=0.05)
    assert snap_shape_elements([el], ink, top=0.0, height=float(H), page_h=float(H)) == 1

    top, bot, left, right = LINES[0]
    assert abs(el.y - top / H) < 0.02
    assert abs((el.y + el.height) - bot / H) < 0.02
    assert abs(el.x - left / W) < 0.02
    assert abs((el.x + el.width) - right / W) < 0.02


def test_sub_line_element_picks_nearest_cluster(tmp_path) -> None:
    """行内多个独立文字块时，只圈离元素估值最近的那一簇，而不是整行。"""
    lines = [(100, 125, 40, 100), (100, 125, 200, 260)]
    ink = PageInk.load(_page(tmp_path / "p.png", lines))
    # 估值指向右侧那一块
    el = SlideElement(id="b", text="稳定于", kind="text", x=0.50, y=0.34, width=0.05, height=0.04)
    assert snap_shape_elements([el], ink, top=0.0, height=float(H), page_h=float(H)) == 1
    assert el.x > 0.45, f"应吸附到右侧文字簇，实际 x={el.x}"
    assert el.x + el.width < 0.75


def test_keeps_original_box_when_nothing_matches(tmp_path) -> None:
    """文字带完全落在文本框窗口之外时，保持原框、不做破坏性改写。"""
    ink = PageInk.load(_page(tmp_path / "p.png"))
    el = SlideElement(id="c", text="无关", kind="text", x=0.05, y=0.02, width=0.30, height=0.05)
    # 把窗口限制在页面上半部分之外（末行以下），池子里没有文字带
    assert snap_shape_elements([el], ink, top=0.98 * H, height=0.02 * H, page_h=float(H)) == 0
    assert (el.y, el.width) == (0.02, 0.30)


def test_photo_page_is_skipped(tmp_path) -> None:
    """深色/照片页分不出文字行，必须整页放弃吸附（避免乱框）。"""
    ink = PageInk.load(_page(tmp_path / "dark.png", lines=[], bg=40))
    assert ink is None


def test_photo_page_does_not_touch_elements(tmp_path) -> None:
    ink = PageInk.load(_page(tmp_path / "dark.png", lines=[], bg=40))
    el = SlideElement(id="d", text="标题", kind="text", x=0.5, y=0.5, width=0.2, height=0.05)
    assert snap_shape_elements([el], ink, top=0.0, height=float(H), page_h=float(H)) == 0
    assert (el.x, el.y) == (0.5, 0.5)
