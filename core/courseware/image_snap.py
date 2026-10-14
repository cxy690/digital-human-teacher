"""页图吸附：把 PPT 解析出的元素框对齐到页面图上真实的文字行。

为什么需要它
------------
``.pptx`` 里字体的**真实字号 / 行距**往往定义在母版 / 主题里，python-pptx 的
``run.font.size`` 读不到（本项目导入的真实课件实测全部为 ``None``），解析器只能
回落到「默认 18pt × 1.2」的行高。于是误差随行号**线性累积**：实测某页到第 7 行
已经偏出 0.38 个页高——表现为「虚线框圈不准 / 圈到别的行上」。

而每页的页面图（``p4.png`` 等，由 PPT 导出）里文字位置就是 ground truth。
这里用行 / 列投影把元素框吸附到图上真实的文字带，绕开字号继承那一堆坑。

安全兜底
--------
* 只在「浅底深字」的页面上吸附（照片页 / 深色页整页放弃）；
* 只在文本框自身的范围内匹配文字带，避免抢到别的文本框的行；
* 任何一步失败都保持原样，绝不影响课件加载。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from config.logging import get_logger

if TYPE_CHECKING:  # 仅类型标注，避免运行时循环导入
    from core.courseware.models import SlideElement

log = get_logger("core.courseware.image_snap")

# ---------------------------------------------------------------- 阈值
# 深色像素判为「墨迹」的亮度阈值（黑字、红字都远低于它；浅色底纹 / 按钮不会）
_INK_LUM = 120
# 某一行墨迹占比超过它 -> 实心色块（页眉色带 / 按钮底），整条丢弃
_SOLID_DENSITY = 0.55
# 行带高度（页高占比）下限：低于它的细线（下划线 / 分隔线）不作为吸附目标
_MIN_BAND_H = 0.012
# 整页墨迹占比超过它 -> 认定不是「浅底深字」页（照片页 / 深色页），放弃吸附
_MAX_DARK_RATIO = 0.25
# 行带之间允许的墨迹空隙（页高占比）：用于把一行的上下沿连成一条带
_ROW_GAP = 0.004

# ---------------------------------------------------------------- 吸附参数
_PAD_X = 0.004          # 吸附后左右留白（归一化）
_PAD_Y = 0.006          # 吸附后上下留白（归一化）
_WIN_X = 0.06           # x 方向搜索窗：在元素估值两侧各外扩这么多
_WIN_Y = 0.01           # 文本框窗口上下外扩（归一化），用于包容导出图的细微差异
_CLUSTER_GAP = 0.02     # 列方向「字间空隙」上限：超过它算两个独立文字簇
_CONT_LEFT_EPS = 0.008  # 续行判据：左边界比首行靠左这么多，才认为是同段折行
_CONT_GAP_RATIO = 0.9   # 续行判据：行间距不超过这个比例的行距
_FULL_LINE_RATIO = 0.55  # 元素宽度达到整行墨迹的这个比例 -> 按「占满整行」处理
_MIN_SPAN = 0.015       # 吸附出的宽度小于它 -> 结果不可信，保留原框

# 页面图下采样目标宽度：够定位文字行，又不必逐像素处理整张大图
_MAX_WIDTH = 700


class PageInk:
    """一页的墨迹信息：灰度二值图 + 文字行带（供元素框吸附使用）。

    坐标一律是**像素**；对外暴露的方法再换算成 0~1 归一化。
    """

    __slots__ = ("w", "h", "ink", "bands", "pitch")

    def __init__(self, w: int, h: int, ink, bands: list[tuple[int, int]], pitch: int) -> None:
        self.w = w
        self.h = h
        self.ink = ink                    # bool 二维数组：(h, w)，True = 墨迹
        self.bands = bands                # 文字行带 [(y0, y1)]，像素、半开区间
        self.pitch = pitch                # 行距（像素），用于判断折行间距

    # ------------------------------------------------------------ 构造
    @classmethod
    def load(cls, image_path: Path, *, max_width: int = _MAX_WIDTH) -> "PageInk | None":
        """读页面图并提取文字行带；不适合作吸附（照片页 / 依赖缺失）时返回 ``None``。"""
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            log.info("缺少 numpy/Pillow，跳过页图吸附")
            return None

        image_path = Path(image_path)
        try:
            with Image.open(image_path) as im:
                # JPEG 走缩略图解码（大图快一个量级）；png 会忽略这一提示
                im.draft("L", (max_width, max_width * 2))
                im = im.convert("L")
                if im.width > max_width:
                    ratio = max_width / float(im.width)
                    im = im.resize(
                        (max_width, max(1, int(round(im.height * ratio)))),
                        Image.BILINEAR,
                    )
                gray = np.asarray(im, dtype=np.uint8)
        except Exception as err:  # noqa: BLE001 - 单页读图失败不该拖垮课件加载
            log.warning("页图读取失败，跳过吸附 {}: {}", image_path, err)
            return None

        if gray.size == 0:
            return None
        h, w = gray.shape
        ink = gray < _INK_LUM

        # 照片页 / 深色页：整页大面积深色，行投影分不出「文字行」，直接放弃
        if float(ink.mean()) > _MAX_DARK_RATIO:
            log.info("页图深色占比过高，跳过吸附 {}", image_path.name)
            return None

        row_density = ink.mean(axis=1)
        bands: list[tuple[int, int]] = []
        start: int | None = None
        gap_allow = max(1, int(_ROW_GAP * h))
        gap = 0
        for y in range(h):
            if row_density[y] > 0.004:
                if start is None:
                    start = y
                gap = 0
            elif start is not None:
                gap += 1
                if gap > gap_allow:
                    bands.append((start, y - gap + 1))
                    start = None
        if start is not None:
            bands.append((start, h))

        kept: list[tuple[int, int]] = []
        for y0, y1 in bands:
            if (y1 - y0) < _MIN_BAND_H * h:
                continue                                  # 细线：下划线 / 分隔线
            if float(row_density[y0:y1].mean()) > _SOLID_DENSITY:
                continue                                  # 实心色块：页眉色带 / 按钮底
            kept.append((y0, y1))
        if not kept:
            return None

        tops = [b[0] for b in kept]
        pitch = int(round(float(np.median(np.diff(tops))))) if len(tops) > 1 else int(0.08 * h)
        return cls(w, h, ink, kept, max(4, pitch))

    # ------------------------------------------------------------ 查询
    def band_columns(self, y0: int, y1: int, x0: int, x1: int):
        """带 ``[y0, y1)`` 内、x 窗 ``[x0, x1)`` 的逐列墨迹布尔序列（numpy 数组）。"""
        x0 = max(0, min(self.w, int(x0)))
        x1 = max(x0, min(self.w, int(x1)))
        if x1 <= x0 or y1 <= y0:
            return None
        return self.ink[y0:y1, x0:x1].any(axis=0)

    def band_bounds(self, y0: int, y1: int) -> tuple[int, int] | None:
        """整行墨迹的左右边界（像素）；该带无墨迹返回 ``None``。"""
        cols = self.band_columns(y0, y1, 0, self.w)
        if cols is None or not bool(cols.any()):
            return None
        idx = _nonzero(cols)
        return int(idx[0]), int(idx[-1]) + 1

    def cluster_near(
        self, y0: int, y1: int, x0: int, x1: int, anchor: tuple[float, float]
    ) -> tuple[int, int] | None:
        """x 窗内离 ``anchor`` 最近的一段连续文字簇（像素左右边界）。

        行内常有多个互不相干的块（如「▶知识点 1」与后面的标题、填空旁单独浮着的
        「稳定于」），按字间空隙切开后取离元素估值最近的一簇，框才落在目标文字上，
        而不是把整行都框进去。
        """
        x0 = max(0, min(self.w, int(x0)))          # 先夹回图内，返回的簇边界才对应图上坐标
        x1 = max(x0, min(self.w, int(x1)))
        cols = self.band_columns(y0, y1, x0, x1)
        if cols is None or not bool(cols.any()):
            return None
        gap_px = max(2, int(_CLUSTER_GAP * self.w))
        runs = _runs(cols, gap_px)
        if len(runs) == 1:
            return x0 + runs[0][0], x0 + runs[0][1]
        a0, a1 = anchor
        best: tuple[tuple[int, float], tuple[int, int]] | None = None
        for s, e in runs:
            cs, ce = x0 + s, x0 + e
            # 与元素估值区间的重叠长度；完全不重叠时按中心距离兜底
            overlap = min(ce, a1) - max(cs, a0)
            score = (0, float(-overlap)) if overlap > 0 else (
                1, abs((cs + ce) / 2 - (a0 + a1) / 2)
            )
            if best is None or score < best[0]:
                best = (score, (cs, ce))
        return best[1] if best else None


def _nonzero(cols):
    """布尔序列里 True 的下标（numpy 快路径，缺 numpy 时退回纯 Python）。"""
    try:
        import numpy as np

        return np.flatnonzero(cols)
    except ImportError:                          # pragma: no cover - numpy 是核心依赖
        return [i for i, v in enumerate(cols.tolist()) if v]


def _runs(cols, gap_px: int) -> list[tuple[int, int]]:
    """按给定空隙 ``gap_px`` 切分连续文字簇（空隙更大的地方断开）。"""
    runs: list[tuple[int, int]] = []
    start = None
    gap = 0
    for i, v in enumerate(cols.tolist()):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > gap_px:
                runs.append((start, i - gap + 1))
                start = None
    if start is not None:
        runs.append((start, len(cols)))
    return runs


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def snap_shape_elements(
    elements: Sequence["SlideElement"],
    ink: PageInk | None,
    *,
    top: float,
    height: float,
    page_h: float,
) -> int:
    """把一个文本框解析出的各段落元素，吸附到页图里真实的文字行上。

    ``top`` / ``height`` 是该文本框在 PPT 上的原始范围（EMU），用作匹配窗口，
    避免把别的文本框的文字带误配进来。返回成功吸附的元素个数。
    """
    if ink is None or not elements or not ink.bands or not page_h:
        return 0

    win_top = top / page_h
    win_bottom = (top + height) / page_h
    # 文本框窗口上下各放宽 `_WIN_Y`：容忍导出页图与 PPT 页尺寸的细微差异
    pool = [
        b for b in ink.bands
        if b[1] / ink.h > win_top - _WIN_Y and b[0] / ink.h < win_bottom + _WIN_Y
    ]
    if not pool:
        return 0
    pool.sort()

    done = 0
    i = 0
    for el in elements:
        if i >= len(pool):
            break
        chosen = [pool[i]]
        first_left = ink.band_bounds(*pool[i])
        # 折行续行：下一行更靠左（首行缩进的段落，续行回到左页边）+ 行距够近
        while first_left and i + 1 < len(pool):
            nxt = pool[i + 1]
            gap = nxt[0] - chosen[-1][1]
            nxt_left = ink.band_bounds(*nxt)
            if (gap <= ink.pitch * _CONT_GAP_RATIO
                    and nxt_left
                    and nxt_left[0] <= first_left[0] - _CONT_LEFT_EPS * ink.w):
                chosen.append(nxt)
                i += 1
            else:
                break
        i += 1
        if _apply(el, ink, chosen):
            done += 1
    return done


def _apply(
    el: "SlideElement",
    ink: PageInk,
    chosen: list[tuple[int, int]],
) -> bool:
    """按选中的文字带改写元素框；返回是否真的吸附成功。"""
    y0 = chosen[0][0]
    y1 = chosen[-1][1]

    line = ink.band_bounds(y0, y1)
    if line is None:
        return False
    line_span = (line[1] - line[0]) / ink.w
    if el.width >= _FULL_LINE_RATIO * line_span:
        # 元素本来就占满这一行（整行/整段）：直接吃整行墨迹，别被偏小的宽度估值切掉尾巴
        bounds = line
    else:
        # 只占行内一小块（如填空旁浮着的「稳定于」）：取离估值最近的一簇
        anchor = (el.x * ink.w, (el.x + el.width) * ink.w)
        bounds = ink.cluster_near(
            y0, y1,
            int((el.x - _WIN_X) * ink.w),
            int((el.x + el.width + _WIN_X) * ink.w),
            anchor,
        ) or line

    left, right = bounds
    nx = _clamp01(left / ink.w - _PAD_X)
    ny = _clamp01(y0 / ink.h - _PAD_Y)
    nw = (right - left) / ink.w + 2 * _PAD_X
    nh = (y1 - y0) / ink.h + 2 * _PAD_Y
    if nw < _MIN_SPAN or nh < 0.008:
        return False                         # 结果不可信：保留原框
    el.x = nx
    el.y = ny
    el.width = max(0.001, min(1.0 - nx, nw))
    el.height = max(0.001, min(1.0 - ny, nh))
    return True
