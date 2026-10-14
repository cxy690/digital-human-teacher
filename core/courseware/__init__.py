"""课件（PPT）模块：数据模型、仓库加载与 .pptx 解析。

对外只需用 :func:`core.courseware.loader.get_store` 拿到仓库实例，
讲解与标注逻辑不感知课件来自 JSON 还是真实 .pptx。
"""
from __future__ import annotations

from core.courseware.loader import CoursewareStore, get_store
from core.courseware.models import Slide, SlideDeck, SlideElement

__all__ = ["CoursewareStore", "Slide", "SlideDeck", "SlideElement", "get_store"]
