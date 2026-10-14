"""课件（PPT）数据模型。

一份课件（``SlideDeck``）= 若干页（``Slide``），一页 = 背景图 + 页面文字 + 可标注元素。

坐标体系与白板完全一致：**全部 0~1 归一化**（相对页面左上角）。
这样后续接真实 .pptx 时，python-pptx 读出的 EMU 坐标只需「除以页宽/页高」
即可直接落到这里，无需在业务层再做换算。
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field


class SlideElement(BaseModel):
    """页面上可被 AI 圈画 / 高亮的元素（文本框、图片、形状、表格等）。

    这是 PPT 讲解标注的**落点**：LLM 讲解时引用元素 id 或元素文字，
    后端据此把标注框精确对齐到页面元素上，而不是让模型凭空猜坐标。
    """

    id: str = ""                 # 元素唯一标识（pptx 可用 shape_id）
    text: str = ""               # 元素文字（LLM 靠它定位，如标题、公式、选项）
    kind: str = "text"           # text / image / shape / table / group
    x: float = 0.0               # 归一化左上角 x
    y: float = 0.0               # 归一化左上角 y
    width: float = 0.0           # 归一化宽度
    height: float = 0.0          # 归一化高度


class Slide(BaseModel):
    """课件的一页。"""

    index: int = 0
    title: str = ""                                        # 页标题（PPT 的标题占位符）
    image: str = ""                                        # 页面背景图（相对课件目录的文件名）
    text: str = ""                                         # 页面全部文字（供 LLM 讲解）
    notes: str = ""                                        # 备注 / 讲稿（来自 PPT 备注栏）
    elements: list[SlideElement] = Field(default_factory=list)

    def element_by_id(self, element_id: str) -> SlideElement | None:
        """按 id 精确取元素。"""
        if not element_id:
            return None
        for el in self.elements:
            if el.id and el.id == element_id:
                return el
        return None

    def match_element(self, keyword: str) -> SlideElement | None:
        """按关键字模糊匹配元素（LLM 常给的是元素文字而非 id）。

        匹配顺序：完全相等 → 包含 → 双向包含。取**第一个**命中，
        因为课件里元素顺序通常与阅读顺序一致（标题在前）。
        """
        key = (keyword or "").strip()
        if not key:
            return None
        for el in self.elements:
            if el.text and el.text.strip() == key:
                return el
        for el in self.elements:
            if el.text and key in el.text:
                return el
        for el in self.elements:
            if el.text and el.text in key:
                return el
        return None

    @property
    def weight(self) -> float:
        """这一页的「信息量」权重（0.15~2.2），连讲时按它分配朗读时长。"""
        return slide_weight(self)


class SlideDeck(BaseModel):
    """一份完整课件。"""

    deck_id: str = ""
    title: str = ""                                        # 课件名
    subject: str = ""                                      # 所属学科（中文名/key 均可）
    source: str = "json"                                   # 数据来源：json / pptx
    slides: list[Slide] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.slides)

    def page(self, index: int) -> Slide | None:
        """按页序取页（越界返回 ``None``，调用方负责提示“已是最后一页”）。"""
        if not self.slides:
            return None
        if index < 0 or index >= len(self.slides):
            return None
        return self.slides[index]

    def summary(self) -> dict:
        """课件摘要（列表接口用，不含页内容，避免列表响应过大）。"""
        return {
            "deck_id": self.deck_id,
            "title": self.title,
            "subject": self.subject,
            "source": self.source,
            "total": self.total,
        }


# ---------------------------------------------------------------- 页面信息量
# 真实 PPT 里「页」的价值差别极大：封面、目录、章节分隔、谢谢页扫一眼就该翻过去，
# 而重点页/例题页/图表页需要展开讲。连讲时若**按页数平均分配**时长，就会出现
# 「封面页也慢条斯理讲 40 秒」的尴尬。下面这个权重就是给连讲分配时长用的。
#
# 注意：这是**廉价启发式**（看文字长度 + 元素数 + 标题词），不调用模型、不耗时，
# 目的是把「明显是导航页」的页压下去、把「明显是重点页」的页抬上来；
# 判断不准也不会出事——最终话术仍由 LLM 按页面实际内容生成。
_LIGHT_TITLE_KEYS = (
    "封面", "目录", "索引", "contents", "前言", "导言", "引言", "序言",
    "过渡", "谢谢", "感谢", "再见", "回顾", "导入", "单元",
    "本课结束", "本课小结", "本节结束", "谢谢观看",
)

# 「模板 / 广告 / 版权页」特征。
#
# 注意判据是**密集度**而不是「出现过」：很多课件的**封面页角落**就带一行模板站
# 水印（实测：「第八章 / 第一PPT模板网-WWW.1PPT.COM」），但那页是正常封面，
# 不该被当成广告页；而真正的模板广告页会把下载地址、模板分类**铺满整页**
# （实测最后 1 页 700+ 字全是网址）——按字数会被误判成「重点页」讲两分钟。
_URL_RE = re.compile(r"https?://|www\.", re.I)
_BOILERPLATE_RE = re.compile(
    r"(1ppt|ppt模板|模板下载|公众号|扫码关注|扫码领取|版权所有"
    r"|素材网|创客贴|稻壳|iSlide|办公资源)",
    re.I,
)
_BOILERPLATE_MIN_HITS = 4          # 网址 + 模板词 出现次数达到此值才算广告页

# 章节标记（第X章 / 第X节 / 第X单元 / Unit N / 模块N）：页面里出现它**且整页文字很短**
# → 典型的「章节封面 / 分隔页」（通常只有章节名 + 拼音 + 模板水印）。
# 实测 11 份真实课件：命中此规则的 7 页**全部**是各自课件的第 1 页封面，零误伤。
# 单靠字数是抓不住它的——生物第 1 页 92 字，按字数会被判成「重点页」讲一分钟。
_CHAPTER_RE = re.compile(
    r"(第\s*[0-9一二三四五六七八九十百]+\s*[章节篇单元课]|Unit\s*\d+|模块\s*[0-9一二三四五六七八九十]+)",
    re.I,
)
_CHAPTER_MAX_CHARS = 110


def _is_boilerplate_page(slide: Slide) -> bool:
    """是不是「模板 / 广告 / 版权页」（整页水印，没有教学内容）。"""
    head = f"{slide.title or ''}\n{slide.text or ''}"
    return (len(_URL_RE.findall(head)) + len(_BOILERPLATE_RE.findall(head))
            >= _BOILERPLATE_MIN_HITS)

# 权重阈值：低于此值视为「导航页」，一句话带过即可
LIGHT_WEIGHT = 0.5
# 高于此值视为「重点页」，可以展开讲
RICH_WEIGHT = 1.3


def slide_weight(slide: Slide) -> float:
    """估算一页课件的**信息量权重**（约 0.15 ~ 2.2，1.0 为「普通页」）。

    依据（都不联网、不算模型，纯本地计算）：
    - 页面文字长度：最主要的信号（典型一页 200~400 字）；
    - 元素个数：几乎没有文字但有多个元素 = 图/表页，也要点一下，不能真跳过去；
    - 标题词：命中「目录 / 封面 / 谢谢」这类导航词且文字很少 → 再降权；
    - 讲稿备注：老师自己写了备注，说明这页值得讲 → 轻微加权。
    """
    text = (slide.text or "").strip()
    chars = len(text)

    # 模板/广告/版权页：字数再多也是水印，直接压到最低（一个字都不该讲）
    if _is_boilerplate_page(slide):
        return 0.15

    # 章节封面 / 分隔页：只有「第X章 第X节」这类目录式标题（常配拼音、水印），
    # 属于「翻过去看一眼就行」的页，不该跟正文页抢时间。
    if chars <= _CHAPTER_MAX_CHARS and _CHAPTER_RE.search(f"{slide.title or ''}\n{text}"):
        return 0.2

    if chars <= 24:
        # 只有大标题（章节封面 / 分隔页）；若页面上还有图/表，至少要点一句
        base = 0.35 if len(slide.elements) >= 2 else 0.18
    elif chars <= 80:
        base = 0.55          # 一两行要点：仍按普通页给时间，别饿死
    elif chars <= 200:
        base = 0.85          # 一屏短要点
    else:
        base = min(1.7, 0.85 + (chars - 200) / 500.0)   # 文字越多越像重点页

    title = (slide.title or "").strip().lower()
    if chars <= 160 and any(k in title for k in _LIGHT_TITLE_KEYS):
        base *= 0.6          # 「目录」「本节封面」这类导航页再降一档

    if (slide.notes or "").strip():
        base += 0.15         # 有讲稿备注：老师有话要说

    return round(max(0.15, min(2.2, base)), 3)


def is_light_slide(slide: Slide) -> bool:
    """是否「导航页」（章节封面 / 目录 / 过渡页）：连讲时一句话带过就翻页。"""
    return slide_weight(slide) < LIGHT_WEIGHT
