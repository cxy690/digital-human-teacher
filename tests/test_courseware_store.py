"""课件仓库单测：**清单必须走轻量索引，页内容按需才解析**。

线上真事：``data/courseware`` 有 200+ 份 pptx、整体几 GB，全部完整解析要 ~60 秒
（含逐行文字测量），而「选课件」下拉只需要一个**页数**。冷启动时点「课件」就是
对着「加载中…」干等一分钟。这些用例把这条边界钉住：列表路径**一次都不许**碰
python-pptx。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from core.courseware.loader import CoursewareStore
from core.courseware.models import Slide, SlideDeck
from core.courseware.pptx_source import count_slides


def _write_fake_pptx(path: Path, pages: int) -> None:
    """造一个最小「可识别」pptx：只需 zip 里带 ``ppt/presentation.xml``。"""
    ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{i + 1}"/>' for i in range(pages))
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<p:sldIdLst>{ids}</p:sldIdLst></p:presentation>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ppt/presentation.xml", xml)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    # ① 纯 json 课件
    demo = tmp_path / "demo_math"
    demo.mkdir()
    (demo / "deck.json").write_text(
        json.dumps(
            {
                "deck_id": "demo_math", "title": "演示课件", "subject": "数学",
                "slides": [{"text": "第一页"}, {"text": "第二页"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # ② 目录 + 真实 pptx（清单只要页数，不该解析正文）
    real = tmp_path / "lesson_dir"
    real.mkdir()
    _write_fake_pptx(real / "三次函数.pptx", 5)
    # ③ 散落在根目录的 pptx 文件
    _write_fake_pptx(tmp_path / "loose.pptx", 2)
    return tmp_path


def test_count_slides_reads_page_count_only(root: Path) -> None:
    """轻量计数：不解压正文也能数对页数（且不把 <p:sldIdLst> 算成一页）。"""
    assert count_slides(root / "loose.pptx") == 2
    assert count_slides(root / "lesson_dir" / "三次函数.pptx") == 5


def test_count_slides_bad_file_returns_minus_one(tmp_path: Path) -> None:
    """不是合法 pptx 时返回 -1，调用方据此跳过（而不是为它付全量解析）。"""
    bad = tmp_path / "bad.pptx"
    bad.write_bytes(b"not a zip at all")
    assert count_slides(bad) == -1


def test_list_decks_never_touches_python_pptx(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """清单只读元信息与页数：连 python-pptx 都不许加载。"""
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("清单路径不许调用 parse_pptx（200+ 份要 60 秒）")

    monkeypatch.setattr("core.courseware.pptx_source.parse_pptx", boom)
    decks = {d["deck_id"]: d for d in CoursewareStore(root).list_decks()}

    assert decks["demo_math"]["total"] == 2
    assert decks["demo_math"]["subject"] == "数学"
    assert decks["demo_math"]["title"] == "演示课件"
    # 目录里没有 deck.json 时，ID 取 pptx 文件名（与 _load_dir 的口径一致）
    assert decks["三次函数"]["total"] == 5
    assert decks["loose"]["total"] == 2


def test_get_deck_parses_only_the_requested_one(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_parse(path: Path, *, deck_id: str = "", media_dir: Path | None = None) -> SlideDeck:
        calls.append(path.name)
        return SlideDeck(
            deck_id=deck_id or path.stem, title=path.stem, source="pptx",
            slides=[Slide(index=i) for i in range(5)],
        )

    monkeypatch.setattr("core.courseware.pptx_source.parse_pptx", fake_parse)
    store = CoursewareStore(root)

    deck = store.get_deck("三次函数")
    assert deck is not None and deck.total == 5
    assert calls == ["三次函数.pptx"], "只该解析被打开的这一份"

    store.get_deck("三次函数")
    assert calls == ["三次函数.pptx"], "第二次应走缓存，不再解析"


def test_get_deck_unknown_and_broken_are_graceful(root: Path) -> None:
    bad = root / "broken"
    bad.mkdir()
    (bad / "broken.pptx").write_bytes(b"not a zip")
    store = CoursewareStore(root)

    assert store.get_deck("不存在的课件") is None
    assert store.get_deck("") is None
    # 解析不了的 pptx 不进清单、也取不到（但不影响其它课件）
    assert "broken" not in {d["deck_id"] for d in store.list_decks()}
    assert store.get_deck("broken") is None


def test_reload_makes_new_courseware_visible(root: Path) -> None:
    """点「↻ 重新扫描课件」后新放的课件要能立刻看到（且只清缓存、不重解析全部）。"""
    store = CoursewareStore(root)
    assert "added" not in {d["deck_id"] for d in store.list_decks()}

    added = root / "added"
    added.mkdir()
    (added / "deck.json").write_text(
        json.dumps({"slides": [{"text": "新课件"}]}, ensure_ascii=False), encoding="utf-8"
    )

    store.reload()
    assert "added" in {d["deck_id"] for d in store.list_decks()}
