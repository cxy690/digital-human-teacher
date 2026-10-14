# -*- coding: utf-8 -*-
"""把「学科/学段/*.pptx」批量导入成课件（页图 + deck.json + 源 pptx）。

为什么需要
----------
课件目录的约定（见 core/courseware/loader.py）：

    data/courseware/<deck_id>/
        ├── <原名>.pptx     # python-pptx 解析文字与元素框（AI 圈哪里）
        ├── p1.png … pN.png # 页图（学生看到什么）
        └── deck.json       # {"deck_id","title","subject"}

``deck_id`` 必须等于目录名（页图 URL 按它拼），``subject`` 必须是学科中文名。
注册表就是磁盘目录本身（``CoursewareStore._scan()`` 扫描式），无需改任何 json / 数据库。

命名（让课堂菜单按学段下钻，规则见 clients/student_tablet/index.html 的 parseDeck）：

    deck_id = "<学科>-<学段>-<课时>"      例：语文-七上-1 春
    title   = "<学段> · <课时>"           例：七上 · 1 春

用法::

    python scripts/import_courseware.py --src C:\\Users\\me\\Desktop\\赵1 \
        --subjects 语文,化学,地理
    python scripts/import_courseware.py --src ... --subjects 语文 --dry   # 只列计划

特点：整个批次**只启动一个 PowerPoint 实例**（逐个启动会慢 10 倍）；
已完整的课件目录默认跳过（``--force`` 才重导），中断后可直接重跑续上。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 学段目录名（7-上 / 8-下 …）-> 菜单里显示的中文学段（与 CW_GRADE_RE 匹配）
GRADE_MAP = {
    "7-上": "七上", "7-下": "七下",
    "8-上": "八上", "8-下": "八下",
    "9-上": "九上", "9-下": "九下",
}

# Windows 目录名非法字符（Linux 只禁 '/'，两边取交集才能保证同步后路径一致）
_BAD_CHARS = re.compile(r'[\\/:*?"<>|]')
_WS = re.compile(r"\s+")


def lesson_name(pptx: Path) -> str:
    """课时名：去扩展名 -> 全角空格转半角 -> 压缩空白 -> 去掉非法字符。"""
    name = pptx.stem.replace("\u3000", " ")
    name = _WS.sub(" ", name).strip()
    return _BAD_CHARS.sub("_", name)


def deck_complete(out_dir: Path, pages: int | None) -> bool:
    """该课件是否已导出完整（用于断点续跑）。"""
    if not (out_dir / "deck.json").is_file():
        return False
    if not any(out_dir.glob("*.pptx")):
        return False
    if pages is None:
        return True
    return all((out_dir / f"p{i}.png").is_file() for i in range(1, pages + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源目录：<src>/<学科>/<学段>/*.pptx")
    ap.add_argument("--subjects", default="语文,化学,地理", help="要导入的学科，逗号分隔")
    ap.add_argument("--out", default=str(ROOT / "data" / "courseware"), help="课件根目录")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--force", action="store_true", help="已存在也重新导出")
    ap.add_argument("--dry", action="store_true", help="只打印计划，不动 PowerPoint")
    args = ap.parse_args()

    src = Path(args.src)
    out_root = Path(args.out)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not src.is_dir():
        print(f"源目录不存在：{src}")
        return 2

    # 1) 先收集任务清单（页数用 python-pptx 读，不启动 PowerPoint）
    from pptx import Presentation  # 局部导入：--dry 时也只用这个

    jobs: list[tuple[str, str, Path, int, Path]] = []
    for subject in subjects:
        subj_dir = src / subject
        if not subj_dir.is_dir():
            print(f"[warn] 缺学科目录：{subj_dir}")
            continue
        for grade_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir()):
            grade = GRADE_MAP.get(grade_dir.name, grade_dir.name)
            for pptx in sorted(grade_dir.glob("*.pptx")):
                lesson = lesson_name(pptx)
                deck_id = f"{subject}-{grade}-{lesson}"
                try:
                    pages = len(Presentation(str(pptx)).slides._sldIdLst)
                except Exception as err:  # noqa: BLE001
                    print(f"[warn] 无法读取页数，跳过：{pptx} ({err})")
                    continue
                jobs.append((subject, grade, pptx, pages, out_root / deck_id))

    total_pages = sum(j[3] for j in jobs)
    print(f"[plan] 学科={'/'.join(subjects)} 课件 {len(jobs)} 份，共 {total_pages} 页 -> {out_root}")
    if args.dry:
        for subject, grade, pptx, pages, out_dir in jobs[:10]:
            print(f"  {out_dir.name}  ({pages} 页)")
        print("  …（--dry 只展示前 10 条）")
        return 0

    import win32com.client  # 只在真正导出时才 import

    app = win32com.client.Dispatch("PowerPoint.Application")
    done = failed = skipped = 0
    t0 = time.time()
    try:
        for idx, (subject, grade, pptx, pages, out_dir) in enumerate(jobs, 1):
            if not args.force and deck_complete(out_dir, pages):
                skipped += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pptx, out_dir / pptx.name)
            try:
                pres = app.Presentations.Open(str(pptx), -1, 0, 0)  # 只读 / 无窗口
                try:
                    for i in range(1, pages + 1):
                        pres.Slides(i).Export(
                            str(out_dir / f"p{i}.png"), "PNG", args.width, args.height
                        )
                finally:
                    pres.Close()
            except Exception as err:  # noqa: BLE001 - 单份失败不中断整批
                failed += 1
                print(f"[fail] {out_dir.name}: {err}", flush=True)
                continue
            meta = {"deck_id": out_dir.name, "title": f"{grade} · {pptx.stem.replace(chr(0x3000), ' ')}",
                    "subject": subject}
            (out_dir / "deck.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            done += 1
            print(f"[{idx}/{len(jobs)}] {out_dir.name}  {pages} 页  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        app.Quit()

    print(f"[done] 新导出 {done} 跳过(已完整) {skipped} 失败 {failed}  "
          f"耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
