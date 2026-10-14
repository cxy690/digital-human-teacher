"""课件仓库：扫描 ``data/courseware/`` 下的课件目录，对外提供列表与取页服务。

目录约定（一个子目录 = 一份课件）：:

    data/courseware/
    ├── demo_math/
    │   ├── deck.json          # 课件描述（页序、页标题、页文字、元素框）
    │   ├── p1.svg             # 页背景图（png/jpg/svg 均可）
    │   └── p2.svg
    └── my_lesson/
        └── lesson.pptx        # 真实 PPT（需 python-pptx，见 pptx_source.py）

``deck.json`` 的结构即 :class:`SlideDeck` 的序列化格式（见 models.py）。
真实 .pptx 由 :mod:`core.courseware.pptx_source` 解析成同样的结构，
**上层（讲解、标注、前端渲染）不感知数据来源**。
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from config.logging import get_logger
from core.courseware.models import Slide, SlideDeck, SlideElement

log = get_logger("core.courseware")

# 页图片对外访问前缀（server/app.py 挂载同名静态目录）
MEDIA_PREFIX = "/media/courseware"

# 支持的页图片后缀（svg 便于手写占位页，png/jpg 来自 PPT 导出）
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".webp")


def courseware_root() -> Path:
    """课件根目录（``<项目根>/data/courseware``）。"""
    return Path(__file__).resolve().parents[2] / "data" / "courseware"


def _clamp01(value: float) -> float:
    """把归一化坐标夹回 [0,1]（与白板 validator 的判据保持一致）。"""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _safe_name(name: str) -> str:
    """去掉路径穿越字符，避免 ``../../`` 读到课件目录之外。"""
    return str(name or "").replace("\\", "/").split("/")[-1]


@dataclass(frozen=True)
class _Unit:
    """一份课件的磁盘来源与元信息（轻量：不含任何页内容）。"""

    path: Path
    kind: str        # "dir" | "json" | "pptx"
    deck_id: str     # 与 _load_dir / _load_json_file 推导出的 deck_id 保持一致
    subject: str     # deck.json 里的学科（中文名 / key 原样；空 = 通用）
    # 下面三个字段专供**清单**（list_decks）使用 —— 它们都能在轻量索引阶段问出来，
    # 因此清单不必再去完整解析 pptx（见 list_decks 的说明）。
    title: str = ""       # 展示用标题（deck.json 的 title，缺省落回 deck_id）
    pages: int = 0        # 页数（deck.json 的 slides 长度；pptx 走 count_slides 轻量计数）
    source: str = ""      # 与 SlideDeck.summary() 里的 source 口径一致（json / pptx / dir）


class CoursewareStore:
    """课件仓库（进程级单例，见 :func:`get_store`）。

    两级缓存，目的是**别为了一门课解析全部课件**：

    * **轻量索引**（``_units``）：只读 ``deck.json`` 问出「有哪些课件、各属哪个学科」，
      220 多份课件实测几百毫秒；
    * **已解析课件**（``_decks``）：真正解析过的 pptx / json（含页图吸附，很贵），
      按学科懒加载、跨学科累积。
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or courseware_root()
        self._decks: dict[str, SlideDeck] = {}          # 已解析课件（跨学科累积）
        self._units: dict[str, _Unit] | None = None     # 轻量索引：deck_id -> 单元
        self._tried: set[str] = set()                   # 已尝试解析的 deck_id（含失败）
        # 扫描互斥锁：没有它，并发请求会各扫一遍（见 list_decks 的说明）。
        self._scan_lock = threading.Lock()

    # ---------------------------------------------------------------- 扫描
    def reload(self) -> None:
        """重新扫描磁盘（上传新 PPT 后调用，让新课件立刻可见）。"""
        with self._scan_lock:
            self._units = None
            self._decks = {}
            self._tried = set()

    def prewarm(self) -> int:
        """只建**轻量索引**（读 ``deck.json``，不解析 pptx），返回索引到的课件数。

        启动时调用：先把「这些目录分别是哪个学科、deck_id 叫什么」问清楚，
        真正的 pptx 解析留到某个学科第一次被打开时再按需做。
        """
        return len(self._ensure_units())

    def _ensure_units(self) -> dict[str, _Unit]:
        """构建（并缓存）轻量索引：**不解析 pptx、不读页图**，只读 ``deck.json``。

        按学科懒加载的前提正是它 —— 学科写在 ``deck.json`` 里（实测 228 份课件
        目录各有一份），于是「哪些课件属于语文」能在几百毫秒内问清楚，不必先把
        每份 pptx、每页页图都解析一遍（那才是要 1~2 分钟的那部分）。
        """
        if self._units is not None:
            return self._units
        with self._scan_lock:
            if self._units is not None:      # 等锁期间别人已经建好了
                return self._units
            self._units = self._build_units()
        return self._units

    def _build_units(self) -> dict[str, _Unit]:
        """遍历课件根目录，产出「一份课件 = 一个单元」的轻量描述。"""
        units: dict[str, _Unit] = {}
        root = self.root
        if not root.is_dir():
            log.info("课件目录不存在，已按空仓库处理：{}", root)
            return units

        for child in sorted(root.iterdir()):
            if child.is_dir():
                unit = self._unit_from_dir(child)
            elif child.suffix.lower() == ".json":
                unit = self._unit_from_json(child)
            elif child.suffix.lower() == ".pptx":
                unit = _Unit(
                    child, "pptx", child.stem, "",
                    title=child.stem, pages=self._count_pages(child), source="pptx",
                )
            else:
                continue
            if unit is not None:
                units[unit.deck_id] = unit
        return units

    def _unit_from_dir(self, directory: Path) -> _Unit:
        """目录型课件：deck_id / 学科都从同目录的 deck.json 读（读不到则按约定推导）。"""
        meta_path = directory / "deck.json"
        raw = self._read_meta(meta_path) if meta_path.is_file() else None
        meta = raw if isinstance(raw, dict) else {}
        deck_id = str(meta.get("deck_id") or "").strip()
        pptx = next(iter(sorted(directory.glob("*.pptx"))), None)
        if not deck_id:
            # 与 _load_dir 的推导保持一致：没有 deck_id 就落回 pptx 文件名 / 目录名
            deck_id = pptx.stem if pptx is not None else directory.name
        # 页数：deck.json 自带 slides 就直接数它；否则**轻量**数 pptx 的页数（不解压正文）。
        # 都拿不到就是 0 —— 清单里显示 0 页，但不会因为这一份让整张表崩掉。
        slides = meta.get("slides")
        has_slides = isinstance(slides, list)
        return _Unit(
            directory, "dir", deck_id, str(meta.get("subject") or "").strip(),
            title=str(meta.get("title") or "").strip() or deck_id,
            # 没有 slides 又没有 pptx = 这个目录里其实没有课件内容 → -1（会被清单剔除），
            # 与旧行为一致（旧 _load_dir 在这种情况下返回 None）。
            pages=len(slides) if has_slides else (self._count_pages(pptx) if pptx else -1),
            source="json" if has_slides else "pptx",
        )

    def _unit_from_json(self, path: Path) -> _Unit | None:
        """顶层 .json 课件：文件本身整体就是 deck.json。"""
        raw = self._read_meta(path)
        if isinstance(raw, list):                     # 允许直接给一个「页数组」的简写
            return _Unit(path, "json", path.stem, "",
                         title=path.stem, pages=len(raw), source="json")
        if not isinstance(raw, dict):
            return None                               # 读不到 / 格式不对：_load_json_file 同样会跳过
        deck_id = str(raw.get("deck_id") or "").strip() or path.stem
        slides = raw.get("slides")
        return _Unit(
            path, "json", deck_id, str(raw.get("subject") or "").strip(),
            title=str(raw.get("title") or "").strip() or deck_id,
            pages=len(slides) if isinstance(slides, list) else 0,
            source="json",
        )

    @staticmethod
    def _read_meta(path: Path) -> Any:
        """只读元信息（学科 / deck_id）；读不到返回 ``None``，绝不影响课件加载。"""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            log.warning("课件描述解析失败，已跳过 {}: {}", path, err)
            return None

    @staticmethod
    def _count_pages(path: Path) -> int:
        """**轻量**数 pptx 页数：只读包里的 ``presentation.xml``，不解压正文。

        为什么单列这一个函数：清单只需要一个数字，而完整解析（含逐行文字测量）
        在 200+ 份课件上要 **~60 秒** —— 冷启动点「课件」就是对着「加载中…」干等一分钟。
        ``count_slides`` 正是为此写的，但早先 ``list_decks`` 并没接上它（直接走了
        ``parse_pptx``），于是清单路径把整份 pptx 全解析了一遍。

        **返回 -1 表示「这份文件数不出来」**（沿用 ``count_slides`` 的约定）——
        调用方据此把它从清单里剔除。旧实现靠「解析失败就不进 _decks」天然做到这件事，
        换成轻量计数后必须在计数层面保留这个信号，否则坏文件会出现在清单里、
        点进去却是空的。
        """
        try:
            from core.courseware.pptx_source import count_slides  # noqa: PLC0415 - 延迟导入

            return int(count_slides(path))
        except Exception as err:  # noqa: BLE001 - 数不出来标记为 -1
            log.debug("页数统计失败（将按无效课件剔除）：{} {}", path.name, err)
            return -1

    def _load_units(self, units: Sequence[_Unit]) -> None:
        """解析这批单元（跳过已尝试过的），结果并入缓存。调用方须持有 ``_scan_lock``。"""
        for unit in units:
            if unit.deck_id in self._tried:
                continue
            self._tried.add(unit.deck_id)             # 失败的也记上，避免反复重试同一份
            deck = self._load_unit(unit)
            if deck is not None:
                self._decks[deck.deck_id] = deck

    def _load_unit(self, unit: _Unit) -> SlideDeck | None:
        if unit.kind == "dir":
            return self._load_dir(unit.path)
        if unit.kind == "json":
            return self._load_json_file(unit.path)
        return self._load_pptx_file(unit.path)

    def _load_dir(self, directory: Path) -> SlideDeck | None:
        """加载一个课件目录（deck.json / *.pptx / p1.png 可自由组合）。

        优先 ``deck.json`` 自带页内容；若它只写了元信息（没有 slides），
        则页内容交给 ``*.pptx`` 解析，再由 ``deck.json`` 补上学科/标题——
        这样真实 PPT 只需旁边放一个 ``{"subject": "语文"}`` 就能指定学科。
        """
        meta_path = directory / "deck.json"
        meta = (
            self._load_json_file(meta_path, fallback_id=directory.name)
            if meta_path.is_file() else None
        )

        if meta is not None and meta.slides:
            # 未在 json 里写图片时，按文件名顺序自动补齐目录下的页图片
            if not any(s.image for s in meta.slides):
                self._autofill_images(meta, directory)
            return meta

        for pptx in sorted(directory.glob("*.pptx")):
            deck = self._load_pptx_file(pptx)
            if deck is None:
                continue
            if meta is not None:                 # 用 deck.json 补元信息（ID/学科/标题）
                # deck_id 必须跟着目录走：页图 URL 是 /media/courseware/{deck_id}/xxx，
                # 若沿用 pptx 文件名当 ID，中文文件名里的全角空格会让 URL 变得很难用。
                if meta.deck_id:
                    deck.deck_id = meta.deck_id
                if meta.subject:
                    deck.subject = meta.subject
                if meta.title:
                    deck.title = meta.title
            return deck
        return meta

    @staticmethod
    def _autofill_images(deck: SlideDeck, directory: Path) -> None:
        """目录里没有 deck.json 的 image 字段时，按文件名字典序把图片配到各页。"""
        images = sorted(
            p.name for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        for i, slide in enumerate(deck.slides):
            if i < len(images):
                slide.image = images[i]

    def _load_json_file(
        self, path: Path, *, fallback_id: str = ""
    ) -> SlideDeck | None:
        """从 JSON 描述文件加载课件。"""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            log.warning("课件描述解析失败，已跳过 {}: {}", path, err)
            return None

        if isinstance(raw, list):            # 允许直接给一个「页数组」的简写
            raw = {"slides": raw}
        if not isinstance(raw, dict):
            return None

        deck_id = str(raw.get("deck_id") or fallback_id or path.stem).strip()
        slides: list[Slide] = []
        for i, item in enumerate(raw.get("slides") or []):
            if not isinstance(item, dict):
                continue
            elements: list[SlideElement] = []
            for j, el in enumerate(item.get("elements") or []):
                if not isinstance(el, dict):
                    continue
                elements.append(SlideElement(
                    id=str(el.get("id") or f"e{i+1}_{j+1}"),
                    text=str(el.get("text") or ""),
                    kind=str(el.get("kind") or "text"),
                    x=_clamp01(el.get("x", 0.0)),
                    y=_clamp01(el.get("y", 0.0)),
                    width=_clamp01(el.get("width", 0.0)),
                    height=_clamp01(el.get("height", 0.0)),
                ))
            slides.append(Slide(
                index=i,
                title=str(item.get("title") or ""),
                image=_safe_name(str(item.get("image") or "")),
                text=str(item.get("text") or ""),
                notes=str(item.get("notes") or ""),
                elements=elements,
            ))
        return SlideDeck(
            deck_id=deck_id,
            title=str(raw.get("title") or deck_id),
            subject=str(raw.get("subject") or ""),
            source=str(raw.get("source") or "json"),
            slides=slides,
        )

    def _load_pptx_file(self, path: Path) -> SlideDeck | None:
        """加载真实 .pptx（解析尚未接入时优雅跳过，不影响其它课件）。"""
        try:
            from core.courseware.pptx_source import parse_pptx

            return parse_pptx(path, deck_id=path.stem, media_dir=path.parent)
        except NotImplementedError as err:
            log.warning("暂未接入 PPTX 解析，已跳过 {}: {}", path.name, err)
            return None
        except Exception as err:  # noqa: BLE001 - 单个课件解析失败不应拖垮仓库
            log.warning("PPTX 解析失败，已跳过 {}: {}", path.name, err)
            return None

    # ---------------------------------------------------------------- 对外
    def list_decks(self, subjects: Sequence[str] | None = None) -> list[dict]:
        """课件清单（不含页内容）。

        ``subjects`` 给定时**只解析并返回这些学科的课件**（按学科懒加载）：
        语文课就只扫语文，不陪跑另外 150 份。``None`` / 空 = 全量（首页兜底）。

        传入的应是**候选学科名列表**（中文名 + 英文 key 都行，调用方负责归一化），
        与 ``deck.json`` 里的 ``subject`` 原样比较；学科为空的「通用课件」不出现
        在学科课堂里（否则数学课可能冒出语文 PPT）。

        为什么必须在这一层过滤：以前是「先全量解析 228 份，再在结果里筛掉别的
        学科」——等于每开一门课都要等全部课件解析完；而前端下拉框处于「加载中…」
        时会重试多次，每个请求都从头扫一遍，CPU 直接打满。现在同一批请求只解析
        一次（锁 + 缓存），且只解析用得到的那几十份。
        """
        wanted = {str(item).strip() for item in (subjects or []) if str(item).strip()}
        units = self._ensure_units()                  # 轻量索引（不解析 pptx）
        # 直接从轻量索引拼清单，**不再调 _load_units**。
        #
        # 早先这里是「_ensure_units() -> _load_units(targets) -> deck.summary()」：
        # 注释写着「轻量索引（不解析 pptx）」，下一行却把每个 pptx 都完整解析了一遍
        # （_load_pptx_file -> parse_pptx，含逐行文字测量）—— 200+ 份要 ~60 秒，
        # 而清单只需要「标题 + 学科 + 页数」这几个轻量索引阶段就能问出来的字段。
        # 页数由 _count_pages 只读 presentation.xml 得到（见该函数的说明）。
        #
        # 副作用（有意为之）：清单不再顺带预热 _decks 缓存。真正的解析留给
        # get_deck() 按需进行 —— 这才是「按学科懒加载」的本意。
        decks = [
            {
                "deck_id": unit.deck_id,
                "title": unit.title or unit.deck_id,
                "subject": unit.subject,
                "source": unit.source or unit.kind,
                "total": unit.pages,
            }
            for unit in units.values()
            # pages < 0 = 这份文件数不出页数（坏 pptx / 非 zip）。
            # 旧实现靠「parse_pptx 失败就不进 _decks」天然过滤掉它们；改成轻量计数后
            # 必须显式剔除，否则坏文件会挂在清单里、点进去是空的。
            if unit.pages >= 0 and (not wanted or unit.subject in wanted)
        ]
        log.info("课件清单：{} 份（学科：{}，走轻量索引）", len(decks), "、".join(sorted(wanted)) or "全部")
        return decks

    def get_deck(self, deck_id: str) -> SlideDeck | None:
        """取完整课件（含全部页）；尚未解析的按需只解析这一份。"""
        key = str(deck_id or "").strip()
        if not key:
            return None
        if key in self._decks:
            return self._decks[key]
        unit = self._ensure_units().get(key)
        if unit is None:
            return None
        with self._scan_lock:
            self._load_units([unit])
            return self._decks.get(key)

    def page_image_url(self, deck_id: str, image: str) -> str:
        """页背景图的对外访问 URL（无图时返回空串，前端按“纯文字页”处理）。"""
        name = _safe_name(image)
        if not name:
            return ""
        return f"{MEDIA_PREFIX}/{deck_id}/{name}"


@lru_cache(maxsize=1)
def get_store() -> CoursewareStore:
    """进程级课件仓库单例。"""
    return CoursewareStore()
