"""真实 .pptx 解析：把 PowerPoint 文件转成 :class:`SlideDeck`。

当前状态：**元素级解析可用**（需要 ``python-pptx``），**页面背景图**依赖
已导出的图片文件（见 :func:`_pick_page_image`）。

为什么要页面图而不是直接渲染 pptx？白板是「图片 + 上层标注」的模型，
页面图是**唯一**能保证「所见即 PPT 原样」的载体（字体、排版、图表都不会跑偏）；
而元素框只用来给标注定位，不参与画面渲染。二者分工如下：

* 页面图（``Slide.image``）：决定学生**看到什么**；
* 元素框（``SlideElement``）：决定 AI **圈哪里**。

接入步骤（后续你上传真实 PPT 时按这个走）：
1. ``pip install python-pptx``；
2. 用 LibreOffice 导出页图：``soffice --headless --convert-to pdf a.pptx``
   再 ``pdftoppm -png a.pdf p``，把 ``p-1.png / p-2.png …`` 放进课件目录；
   （或直接用 PPT 的「另存为图片」逐页导出）
3. 目录里放 ``a.pptx`` + 页图，后端自动解析出文字与元素框并关联页图。
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

from config.logging import get_logger
from core.courseware.models import Slide, SlideDeck, SlideElement

log = get_logger("core.courseware.pptx")

# PPT 文本里常混入控制字符：\x0b（竖排换行）最典型，直接渲染会显示成乱码方块
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(text: str) -> str:
    """清洗 PPT 文本：控制字符转空格并去掉首尾空白。"""
    return _CONTROL_CHARS.sub(" ", text or "").strip()


def _is_wide_char(ch: str) -> bool:
    """是否全角字符（中日韩文字与全角标点）：宽度按 1 个字号计。"""
    code = ord(ch)
    return (
        0x1100 <= code <= 0x115F
        or 0x2E80 <= code <= 0xA4CF
        or 0xAC00 <= code <= 0xD7A3
        or 0xF900 <= code <= 0xFAFF
        or 0xFE30 <= code <= 0xFE6F
        or 0xFF00 <= code <= 0xFF60
        or 0xFFE0 <= code <= 0xFFE6
        or 0x3000 <= code <= 0x303F
    )


def _text_width_emu(text: str, pt: float) -> float:
    """估算一行文字的宽度（EMU）：全角 1em、半角 0.55em。

    PPT 里文本框常比文字宽得多（居中排版 + 两侧留白），若直接用 shape 宽度
    当标注框，红框会甩到文字右侧一大截——所以按字数估算宽度来收窄。
    """
    units = sum(1.0 if _is_wide_char(ch) else 0.55 for ch in text)
    return units * pt * 12700.0


# DrawingML 命名空间：读 pptx 原生 XML 用。这里不 import pptx，
# 保持本模块「没装 python-pptx 也能导入」的性质（真正解析时才会 import）。
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _tx_body(paragraph: Any) -> Any:
    """段落所属的 ``<a:txBody>``（找不到返回 None）。"""
    node = getattr(paragraph, "_p", None)
    return None if node is None else node.getparent()


def _shape_def_rpr(paragraph: Any) -> Any:
    """形状级默认字符属性 ``<a:lstStyle><a:lvl1pPr><a:defRPr>``。

    很多标题把字号只写在这里（段落与 run 都不写），python-pptx 的
    ``paragraph.font.size`` 读不到 —— 于是字号回退成默认 18pt，
    标注框宽度只有实际的 1/3（实测 60pt 标题被当 18pt，横线短一大截）。
    """
    body = _tx_body(paragraph)
    if body is None:
        return None
    lst = body.find(f"{_A}lstStyle")
    ppr = None if lst is None else lst.find(f"{_A}lvl1pPr")
    return None if ppr is None else ppr.find(f"{_A}defRPr")


def _shape_alignment(paragraph: Any) -> str:
    """形状级默认段落对齐（``lstStyle/lvl1pPr/@algn``），没有返回空串。

    标题常把"居中"只写在这里，漏掉会让标注框按左对齐铺开 —— 横线比文字
    偏左一大截，右半边没盖住。
    """
    body = _tx_body(paragraph)
    if body is None:
        return ""
    lst = body.find(f"{_A}lstStyle")
    ppr = None if lst is None else lst.find(f"{_A}lvl1pPr")
    return "" if ppr is None else str(ppr.get("algn") or "")


def _shape_line_spacing(paragraph: Any) -> float:
    """形状级默认行距倍数（``lstStyle/lvl1pPr/lnSpc/spcPct``），没有返回 0。"""
    body = _tx_body(paragraph)
    if body is None:
        return 0.0
    lst = body.find(f"{_A}lstStyle")
    ppr = None if lst is None else lst.find(f"{_A}lvl1pPr")
    ln = None if ppr is None else ppr.find(f"{_A}lnSpc")
    pct = None if ln is None else ln.find(f"{_A}spcPct")
    try:
        return float(int(pct.get("val"))) / 100000.0 if pct is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _autofit_line_reduce(paragraph: Any) -> float:
    """PPT「自动缩小行距」的缩减比例（0~1）：``bodyPr/normAutofit/@lnSpcReduction``。

    该属性会让实际行高比标称值再小一截（实测某标题标称 150% 实际只有 120%），
    不算进去后续行的标注会整体偏低。单位为千分之一百分点（20000 = 20%）。
    """
    body = _tx_body(paragraph)
    if body is None:
        return 0.0
    bpr = body.find(f"{_A}bodyPr")
    naf = None if bpr is None else bpr.find(f"{_A}normAutofit")
    raw = None if naf is None else naf.get("lnSpcReduction")
    try:
        return min(0.5, max(0.0, int(raw or 0) / 100000.0))
    except (TypeError, ValueError):
        return 0.0


def _font_pt(paragraph: Any, default: float = 18.0) -> float:
    """段落字号（pt）：run 级 -> 段落级 -> 形状级 lstStyle -> 默认值。"""
    for run in getattr(paragraph, "runs", []) or []:
        size = getattr(getattr(run, "font", None), "size", None)
        if size is not None:
            return float(size.pt)
    size = getattr(getattr(paragraph, "font", None), "size", None)
    if size is not None:
        return float(size.pt)
    dpr = _shape_def_rpr(paragraph)
    raw = None if dpr is None else dpr.get("sz")
    if raw:
        try:
            return float(int(raw)) / 100.0      # sz 单位为 1/100 pt
        except ValueError:
            pass
    return default


# 字体「单倍行高」相对字号的倍数。
# 老代码写死 1.2：用它去累加同一段的第 2、3 行时会越算越短（行号越大偏得越多），
# 横线因此压到字上。用 24 份课件 159 页的「页图墨迹带行距 / 模型行高」标定，
# 真实值中位数 ≈1.3em，这里按实测取值。
_FONT_LINE_FACTOR = 1.3


def _line_height_emu(paragraph: Any, pt: float) -> float:
    """行高（EMU）：段落行距 -> 形状级默认行距 -> 单倍，再乘字号与字体行高系数。

    注意倍数行距的语义：OOXML 的 ``spcPct`` 是「字体单倍行高 × 百分比」，
    不是「字号 × 百分比」——漏乘这个单倍行高（≈1.3em）会让 1.5 倍行距的
    标题少算 20% 高度，同一段第二行的标注就压到字上（实测 21px @1200）。
    另外 ``normAutofit/lnSpcReduction``（自动缩小行距）也要按比例扣掉。
    """
    value = 0.0
    spacing = getattr(paragraph, "line_spacing", None)
    if spacing is not None:
        try:
            value = float(spacing)          # Length -> EMU；倍数行距则是个小数
        except (TypeError, ValueError):
            value = 0.0
        if value >= 100:                    # 明显是绝对行距（EMU）
            return value
    if value <= 0:
        value = _shape_line_spacing(paragraph) or 1.0
    return (
        pt * 12700.0 * _FONT_LINE_FACTOR * value
        * (1.0 - _autofit_line_reduce(paragraph))
    )


# 行框内的字形是**偏下**的（行高含上半部分留白）。用页图墨迹带实测标定：
# 字形底部 ≈ 行首 + 0.9 个行高，所以框底边贴住 0.9 行高 = 横线正好落在字下。
# （旧值是 offset 0.2 / ratio 0.9，底边落在 1.1 行高处，行高修正后会偏低约 30px。）
_TEXT_VERTICAL_OFFSET = 0.0
_TEXT_BOX_RATIO = 0.9


def _hard_lines(text: str) -> list[str]:
    """把一个「段落」按硬换行拆成多行。

    PPT 段落里用 ``\\x0b``（Shift+Enter 软回车）或换行符塞多行文字很常见
    （WPS 另存出来的 pptx 尤其多）。不拆开的话，这一段只会得到**一个**行高的框，
    而文字实际占好几行 —— 标注（横线）就会落在文字中部。
    """
    parts = (_clean(p) for p in re.split(r"[\x0b\n\r]", text or ""))
    return [p for p in parts if p]


def _wrap_visual_lines(text: str, max_w: float, pt: float) -> list[str]:
    """按估算宽度把一行文字折成若干「视觉行」，还原 PPT 的实际排版。

    CJK 可在任意字之间断行；西文优先在空格处断，避免把单词劈开。
    ``max_w`` <= 0（拿不到文本框宽度）时不折行，保持原样。
    """
    if max_w <= 0 or not text:
        return [text] if text else []
    out: list[str] = []
    cur = ""
    cur_w = 0.0
    for ch in text:
        ch_w = (1.0 if _is_wide_char(ch) else 0.55) * pt * 12700.0
        if cur and cur_w + ch_w > max_w:
            stripped = cur.rstrip()
            if not _is_wide_char(ch) and " " in stripped:
                head, _, tail = stripped.rpartition(" ")   # 西文在空格处断，不拆词
                out.append(head)
                cur = tail + ch
                cur_w = _text_width_emu(cur, pt)
                continue
            out.append(stripped)
            cur, cur_w = ch, ch_w
        else:
            cur += ch
            cur_w += ch_w
    if cur.strip():
        out.append(cur.strip())
    return out or [text]


def _space_emu(paragraph: Any, pt: float, tag: str) -> float:
    """段落前/后间距（EMU），``tag`` 取 ``spcBef`` / ``spcAft``。

    为什么不能只用 ``paragraph.space_before``：python-pptx **只解析 ``<a:spcPts>``（点值）**，
    遇到 ``<a:spcPct val="50000"/>``（按字号百分比给间距）直接返回 ``None`` —— 于是段前
    间距被当成 0，多段落文本框里第 2 段起的位置越算越靠上。

    实测（``数学-七上-第01章-1.2.1`` 第 1 页，48pt 两段标题）：两段各带 50% 段前间距，
    解析出的行距只有 104px 而真实行距 144px，第 2 段的框整整高了 40px（=48pt×50%），
    框底边恰好落在文字中部 —— 学生看到的就是「横线划在字上」。

    ``spcPct`` 的基准是**字号**（不是行高）：上面这个例子 48pt×50%=24pt=40px，与实测吻合；
    若按行高算（62.4pt×50%=52px）则对不上。
    """
    node = getattr(paragraph, "_p", None)
    holder = None
    if node is not None:
        ppr = node.find(f"{_A}pPr")
        if ppr is not None:
            holder = ppr.find(f"{_A}{tag}")
    if holder is None:
        # 回退到形状级默认：<a:txBody><a:lstStyle><a:lvl1pPr><a:spcBef>
        body = _tx_body(paragraph)
        lst = None if body is None else body.find(f"{_A}lstStyle")
        ppr = None if lst is None else lst.find(f"{_A}lvl1pPr")
        holder = None if ppr is None else ppr.find(f"{_A}{tag}")
    if holder is None:
        return 0.0

    pts = holder.find(f"{_A}spcPts")
    if pts is not None:
        try:
            return float(pts.get("val")) / 100.0 * 12700.0      # 百分之一磅 -> EMU
        except (TypeError, ValueError):
            return 0.0
    pct = holder.find(f"{_A}spcPct")
    if pct is not None:
        try:
            return float(pct.get("val")) / 100000.0 * pt * 12700.0   # 千分之一百分点 × 字号
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _layout_paragraphs(
    shape: Any, page_w: float, page_h: float, default_pt: float = 18.0
) -> list[tuple[str, float, float, float, float]]:
    """列出文本框里每个**视觉行**的真实框 ``(文本, x, y, w, h)``（归一化）。

    三个必须做对的地方（都会让「贴底边画的横线」跑到内容中部）：
    1. **按视觉行给框**：段落会折行，40 字的段落放进"一行只容 20 字"的框里实际占
       2 行；只给一个行高的框，框底边就落在第 1 行下方（实测 13% 的段落属于这类）。
       拿到行级框，LLM 也才能引用"那一句"而不是"那一整段"。
    2. **垂直对齐**：MIDDLE/BOTTOM 的文本框文字不在顶部，一律从 ``shape.top`` 起排会
       整体偏低（实测有 top=0.09 / 高 0.38 的"底部对齐"标题，偏差可达页高 30%）。
    3. **行高与段间距**：按 ``line_spacing``（倍数行距要乘字体单倍行高）或字号 ×1.3
       逐行累加，并计入 ``space_before/space_after``；不要用 shape 高度均分
       （文本框常有额外留白，均分会让越靠后的行偏得越多）。空段落照样占一行。
    """
    tf = shape.text_frame
    left = float(getattr(shape, "left", 0) or 0)
    width = float(getattr(shape, "width", 0) or 0)
    top = float(getattr(shape, "top", 0) or 0)
    height = float(getattr(shape, "height", 0) or 0)
    margin_top = float(getattr(tf, "margin_top", 0) or 0)
    margin_bottom = float(getattr(tf, "margin_bottom", 0) or 0)

    # 第一遍：拆视觉行并累计整段文字高度（垂直对齐要用到总高）
    blocks: list[tuple[Any, float, float, list[str], float, float]] = []
    total_h = 0.0
    for para_index, paragraph in enumerate(tf.paragraphs):
        pt = _font_pt(paragraph, default_pt)
        line_h = _line_height_emu(paragraph, pt)
        lines = [
            line
            for hard in _hard_lines(paragraph.text)
            for line in _wrap_visual_lines(hard, width, pt)
        ]
        # 段后间距：先按 XML 读（spcPts 点值 / spcPct 百分比都认），再退到 python-pptx 的属性
        # （后者只认 spcPts，遇到 WPS/新 Office 爱用的 spcPct 会返回 None）。
        after = _space_emu(paragraph, pt, "spcAft") or float(
            getattr(paragraph, "space_after", 0) or 0
        )
        # 段前间距：**只从第 2 段起生效** —— 首段按文本框顶部起排。
        # 依据（实测该课件两个文本框：单段大标题 / 两段标题，各段都写着 spcBef=50%）：
        # 页图里首段文字都从 shape.top + margin_top 起排，第 2 段才往后让出一个段前距。
        # 首段若也加，整块文字会整体下沉（单段标题实测多偏 40px，横线反而离字太远）。
        before = 0.0
        if para_index > 0:
            before = _space_emu(paragraph, pt, "spcBef") or float(
                getattr(paragraph, "space_before", 0) or 0
            )
        blocks.append((paragraph, pt, line_h, lines, before, after))
        total_h += before + max(len(lines), 1) * line_h + after

    # 第二遍：按垂直对齐确定起始位置（文字比框矮时才需要偏移）
    cursor = top + margin_top
    avail = height - margin_top - margin_bottom
    if height > 0 and avail > total_h:
        anchor = str(getattr(tf, "vertical_anchor", "") or "").upper()
        if "MIDDLE" in anchor:
            cursor += (avail - total_h) / 2
        elif "BOTTOM" in anchor:
            cursor += avail - total_h

    out: list[tuple[str, float, float, float, float]] = []
    for paragraph, pt, line_h, lines, before, after in blocks:
        cursor += before
        align = str(getattr(paragraph, "alignment", "") or "").upper() \
            or _shape_alignment(paragraph).upper()
        for line_text in lines:
            line_w = _text_width_emu(line_text, pt)
            if width:
                line_w = min(line_w, width)
            # 对齐值两种写法都要认：python-pptx 枚举给 "CENTER (2)"，
            # 而原生 XML（lstStyle 里）给的是 "ctr"。
            if "CENTER" in align or align in ("CTR", "CENTER"):
                x = left + (width - line_w) / 2
            elif "RIGHT" in align or align in ("R",):
                x = left + width - line_w
            else:                            # 左对齐 / 继承
                x = left
            out.append((
                line_text,
                _norm(x, page_w),
                _norm(cursor + line_h * _TEXT_VERTICAL_OFFSET, page_h),
                _norm(line_w, page_w),
                _norm(line_h * _TEXT_BOX_RATIO, page_h),
            ))
            cursor += line_h
        if not lines:                        # 空段落照样占一行，后续段落位置才准
            cursor += line_h
        cursor += after
    return out


# 页图候选名：p1.png / p-1.png / slide1.png / 1.png …（按页序 1-based）
_IMAGE_CANDIDATES = (
    "p{i}.png", "p{i}.jpg", "p{i}.svg",
    "p-{i}.png", "p-{i}.jpg",
    "slide{i}.png", "slide-{i}.png", "slide_{i}.png",
    "{i}.png", "{i}.jpg",
)


def _pick_page_image(media_dir: Path, page_no: int) -> str:
    """按页序（1-based）在课件目录里找对应的页面背景图。"""
    for pattern in _IMAGE_CANDIDATES:
        name = pattern.format(i=page_no)
        if (media_dir / name).is_file():
            return name
    return ""


def _norm(value: float, total: float) -> float:
    """EMU/像素 -> 归一化坐标（页面宽高为分母，与白板坐标体系一致）。"""
    if not total:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(total)))


def count_slides(path: Path) -> int:
    """只数页数：直接读 pptx（zip）里的 ``ppt/presentation.xml``，**不加载 python-pptx**。

    课件清单只需要一个页数，而 :func:`parse_pptx` 会把每页的文本框、元素框、逐行
    文字测量全算一遍 —— 242 份课件这样跑要 **~60 秒**，冷启动时学生就对着
    「加载中…」干等。这里只解一个 XML、数 ``<p:sldId>``，单份 1ms 级。

    解析不出来时返回 ``-1``，由调用方决定是退回完整解析还是跳过。
    """
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("ppt/presentation.xml").decode("utf-8", "ignore")
    except (OSError, KeyError, zipfile.BadZipFile):
        return -1
    # 只数 <p:sldId ...>，别把容器节点 <p:sldIdLst> 算进去
    return len(re.findall(r"<p:sldId\b", xml))


def parse_pptx(
    path: Path, *, deck_id: str = "", media_dir: Path | None = None
) -> SlideDeck:
    """解析 .pptx -> :class:`SlideDeck`。

    未安装 ``python-pptx`` 时抛 :class:`NotImplementedError`，
    由 loader 捕获后跳过该文件（不影响其它课件加载）。
    """
    try:
        from pptx import Presentation          # type: ignore[import-not-found]
    except ImportError as err:                  # pragma: no cover - 依赖缺失
        raise NotImplementedError(
            "未安装 python-pptx（pip install python-pptx），暂无法解析 .pptx"
        ) from err

    media_dir = media_dir or path.parent
    prs = Presentation(str(path))
    # 页宽/页高与 shape.left/top/width/height **同为 EMU**，两者相除即得归一化坐标。
    # 曾经这里把 shape 坐标先转成像素再除以 EMU 页宽，量纲不一致（1 px / 12192000 EMU
    # ≈ 3.9e-6），导致所有元素框都被压成 (0,0,0,0)、标注全部塌在页面左上角。
    # 缺失页尺寸时退回标准 16:9（12192000 x 6858000 EMU），避免除出天量数字。
    page_w = float(prs.slide_width or 12192000)
    page_h = float(prs.slide_height or 6858000)

    slides: list[Slide] = []
    for i, slide in enumerate(prs.slides):
        texts: list[str] = []
        elements: list[SlideElement] = []
        for j, shape in enumerate(slide.shapes):
            kind = _shape_kind(shape)

            # 逐段落推算真实位置（不要用 shape 高度均分，见 _layout_paragraphs）
            layout: list[tuple[str, float, float, float, float]] = []
            paragraphs: list[str] = []
            if getattr(shape, "has_text_frame", False):
                # 按硬换行拆行（\x0b 等），页文字与元素框的行边界保持一致
                paragraphs = [
                    line
                    for p in shape.text_frame.paragraphs
                    for line in _hard_lines(p.text)
                ]
                layout = _layout_paragraphs(shape, page_w, page_h)
            text = "\n".join(paragraphs)
            if text:
                texts.append(text)

            # 空文本框（页眉装饰、未填的占位符）不进目录：对「圈哪里」毫无帮助，
            # 混在元素清单里反而干扰 AI 判断。
            if not text and kind == "text":
                continue

            sid = str(getattr(shape, "shape_id", "") or f"e{i+1}_{j+1}")

            if len(layout) > 1:
                # 一个文本框装多行（诗歌、要点页最常见）：整块框住毫无指向性，
                # 学生会看到一个大框罩住大半页。逐行拆成**行级框**，
                # AI 才能精确圈「骑在人民头上」这一行，而不是整首诗。
                for k, (line_text, lx, ly, lw, lh) in enumerate(layout):
                    elements.append(SlideElement(
                        id=f"{sid}_p{k + 1}",
                        text=line_text,
                        kind=kind,
                        x=lx, y=ly, width=lw, height=lh,
                    ))
            elif layout:
                line_text, lx, ly, lw, lh = layout[0]
                elements.append(SlideElement(
                    id=sid, text=line_text, kind=kind,
                    x=lx, y=ly, width=lw, height=lh,
                ))
            else:
                # 无文字的图片 / 形状：用 shape 自身 bbox（EMU 直接归一化）
                elements.append(SlideElement(
                    id=sid,
                    text="",
                    kind=kind,
                    x=_norm(float(getattr(shape, "left", 0) or 0), page_w),
                    y=_norm(float(getattr(shape, "top", 0) or 0), page_h),
                    width=_norm(float(getattr(shape, "width", 0) or 0), page_w),
                    height=_norm(float(getattr(shape, "height", 0) or 0), page_h),
                ))

        title = texts[0] if texts else ""
        slides.append(Slide(
            index=i,
            title=title[:60],
            image=_pick_page_image(media_dir, i + 1),
            text="\n".join(texts),
            notes=_slide_notes(slide),
            elements=elements,
        ))

    return SlideDeck(
        deck_id=deck_id or path.stem,
        title=path.stem,
        source="pptx",
        slides=slides,
    )


def _shape_kind(shape: object) -> str:
    """粗略判定元素类型，供前端/标注策略区分（失败一律按 text）。"""
    try:
        if getattr(shape, "has_table", False):
            return "table"
        if getattr(shape, "has_chart", False):
            return "chart"
        if getattr(shape, "shape_type", None) is not None:
            name = str(shape.shape_type).lower()
            if "picture" in name:
                return "image"
            if "table" in name:
                return "table"
    except Exception:                            # noqa: BLE001 - 类型判定失败无害
        pass
    return "text"


def _slide_notes(slide: object) -> str:
    """取备注栏文字（讲稿）。"""
    try:
        notes_slide = slide.notes_slide          # type: ignore[attr-defined]
        tf = notes_slide.notes_text_frame
        lines = [_clean(p.text) for p in tf.paragraphs]
        return "\n".join(x for x in lines if x).strip()
    except Exception:                            # noqa: BLE001 - 无备注属正常
        return ""
