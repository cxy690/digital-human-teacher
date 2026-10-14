"""PPT 元素框排版单测：标注横线必须落在**字形下方**，不能穿字。

背景（这条链路上已经踩过好几次，每次表现都是「横线划到字上」）：
元素框的底边按设计就贴在字形底部（见 ``pptx_source._TEXT_BOX_RATIO`` 的标定），
所以**框的位置算错一点点，横线就会横穿文字**。

本文件钉住最新一次回归：
段前间距写成 ``<a:spcBef><a:spcPct val="50000"/></a:spcBef>``（按字号百分比）时，
python-pptx 的 ``paragraph.space_before`` 返回 ``None``（它只解析 ``spcPts`` 点值），
于是段间距被当成 0 —— 多段落文本框里第 2 段起的位置越算越靠上。

实测实例（``data/courseware/数学-七上-第01章-1.2.1-有理数的概念`` 第 1 页，48pt 两段标题）：
两段各带 50% 段前距，解析行距只有 104px、真实行距 144px，第 2 段的框整整高了 40px
（= 48pt × 50%），框底边正好落在文字中部。
"""
from __future__ import annotations

from lxml import etree

from core.courseware.pptx_source import _space_emu, _layout_paragraphs

_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_EMU_PER_PT = 12700.0
_PAGE = 6858000.0          # 一张 16:9 页面的 EMU 高度（与真实课件一致）


class _Para:
    """最小替身：``_space_emu`` 只用到 ``_p`` 与 ``_tx_body`` 的 ``getparent()``。"""

    def __init__(self, xml: str) -> None:
        self._p = etree.fromstring(f'<a:p xmlns:a="{_A}">{xml}</a:p>')


def _para_with(xml: str) -> _Para:
    return _Para(xml)


# ---------------------------------------------------------------- 段间距解析
def test_space_before_reads_percent_of_font_size() -> None:
    """``spcPct`` 的基准是**字号**（不是行高）：48pt × 50% = 24pt。

    基准搞错会整体偏：按行高算（62.4pt × 50% = 31.2pt）与实测差 7pt（约 12px）。
    """
    paragraph = _para_with('<a:pPr><a:spcBef><a:spcPct val="50000"/></a:spcBef></a:pPr>')
    assert _space_emu(paragraph, 48.0, "spcBef") == 24.0 * _EMU_PER_PT


def test_space_reads_point_value() -> None:
    """``spcPts`` 是百分之一磅（2400 = 24pt），与 python-pptx 的取值一致。"""
    paragraph = _para_with('<a:pPr><a:spcAft><a:spcPts val="2400"/></a:spcAft></a:pPr>')
    assert _space_emu(paragraph, 48.0, "spcAft") == 24.0 * _EMU_PER_PT


def test_space_missing_returns_zero() -> None:
    assert _space_emu(_para_with("<a:pPr/>"), 48.0, "spcBef") == 0.0
    assert _space_emu(_para_with(""), 48.0, "spcAft") == 0.0


# ---------------------------------------------------------------- 段落累加
class _FakeParagraph:
    """只满足 ``_layout_paragraphs`` 读取需要的最小段落对象。"""

    def __init__(self, text: str, *, space_before_pct: int | None = None) -> None:
        self.text = text
        self.runs: list[object] = []
        self.font = None
        self.alignment = None
        ppr = ""
        if space_before_pct is not None:
            ppr = f'<a:pPr><a:spcBef><a:spcPct val="{space_before_pct}"/></a:spcBef></a:pPr>'
        self._p = etree.fromstring(f'<a:p xmlns:a="{_A}">{ppr}</a:p>')


class _FakeTextFrame:
    margin_top = 0
    margin_bottom = 0
    vertical_anchor = None

    def __init__(self, paragraphs: list[_FakeParagraph]) -> None:
        self.paragraphs = paragraphs


class _FakeShape:
    left = 0
    top = 0
    width = 6858000
    height = 6858000

    def __init__(self, paragraphs: list[_FakeParagraph]) -> None:
        self.text_frame = _FakeTextFrame(paragraphs)


def test_first_paragraph_ignores_space_before() -> None:
    """首段不加段前距（按文本框顶部起排），第 2 段才加上 —— 跟着页图渲染结果走。

    实测：同一份课件里单段大标题 / 两段标题的所有段落都写着 ``spcBef=50%``，
    但页图里首段文字都从 ``shape.top + margin_top`` 起排。首段若也加，整块文字整体下沉。
    """
    shape = _FakeShape([
        _FakeParagraph("第一段", space_before_pct=50000),
        _FakeParagraph("第二段", space_before_pct=50000),
    ])

    lines = _layout_paragraphs(shape, _PAGE, _PAGE, default_pt=48.0)
    assert [text for text, *_ in lines] == ["第一段", "第二段"]

    line_h = 48.0 * 1.3 * _EMU_PER_PT          # 字号 × 字体行高系数
    space_before = 24.0 * _EMU_PER_PT          # 48pt × 50%
    assert lines[0][2] == 0.0, "首段必须从顶部起排（不吞段前距）"
    assert lines[1][2] == (line_h + space_before) / _PAGE, "第 2 段要落在 行高 + 段前距 处"


def test_paragraphs_without_space_stack_by_line_height_only() -> None:
    """没有段间距时，段落位置就是逐个行高累加（不应被误加固定间距）。"""
    shape = _FakeShape([_FakeParagraph("A"), _FakeParagraph("B"), _FakeParagraph("C")])

    lines = _layout_paragraphs(shape, _PAGE, _PAGE, default_pt=48.0)
    line_h = 48.0 * 1.3 * _EMU_PER_PT
    assert [round(line[2] * _PAGE / line_h, 3) for line in lines] == [0.0, 1.0, 2.0]
