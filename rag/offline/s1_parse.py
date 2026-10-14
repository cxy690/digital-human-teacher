#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 · 解析（rag/offline/s1_parse.py）
=====================================
把「原始素材」变成「纯文本 + 结构信息」，输出统一的 JSON 中间件，
下游 s2_clean / s3_chunk 只认这一个格式，不再关心素材来自 PPT 还是录音。

支持的素材类型与依赖
--------------------
================  ==========================================  ==============================
类型              首选库                                       缺失时的行为
================  ==========================================  ==============================
``.pptx``          ``python-pptx``                             抛 ``MissingDependencyError`` 并说明安装命令
``.pdf``           ``pymupdf`` (``import fitz``)               退化到 ``pdfminer.six``；都没有则报错
``.docx``          ``python-docx``                            抛 ``MissingDependencyError`` 并说明安装命令
``.json`` 转写      标准库 ``json``                             无额外依赖
``.srt`` 字幕       标准库                                         无额外依赖
================  ==========================================  ==============================

安装命令（按需，流水线本身不自动安装）::

    pip install python-pptx pymupdf python-docx

**视频/音频不在这里做转写**：ASR 属于重活，交给 FunASR 离线跑。
本模块只提供 ``build_funasr_command()`` 打印建议命令（绝不真的执行），
以及 ``parse_video_transcript()`` 解析 FunASR 的 JSON 输出。

FunASR 输出契约（本流水线假设的字段，缺字段会做兼容处理）::

    [{"start_ms": 0, "end_ms": 2500, "text": "同学们好", "confidence": 0.98}, ...]

统一输出格式（``data/parsed/<name>.json``）::

    {
      "source_path": "data/raw/x.pptx",
      "source_name": "x.pptx",
      "source_type": "pptx",          # pptx|pdf|docx|transcript
      "source_hash": "…",             # 素材 sha256 前 16 位，溯源用
      "parser": "python-pptx",        # 实际使用的解析器，便于排查"同一素材两次结果不同"
      "parsed_at": "2026-…+08:00",
      "blocks": [                     # 有序块，s2/s3 按块处理
        {"locator": "p.3",            # 人类可读定位串：页码 / 段落号 / 时间码
         "type": "slide|page|paragraph|utterance",
         "text": "…",
         "start_ms": null, "end_ms": null,
         "confidence": null,
         "notes": null}               # PPT 备注等旁路信息
      ],
      "meta": {...},                  # 类型相关元信息（页数、总时长等）
      "warnings": []                  # 解析期的降级/异常提示，s2 会汇总
    }

CLI::

    python rag/offline/s1_parse.py --in data/raw --out data/parsed
    python rag/offline/s1_parse.py --in data/raw --out data/parsed --funasr-hint audio.m4a
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ---- 既支持 `python rag/offline/s1_parse.py`，也支持 `python -m rag.offline.s1_parse` ----
try:  # pragma: no cover - 取决于调用方式
    from . import common
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore

log = common.get_logger("s1_parse")

CST = timezone(timedelta(hours=8))

#: 支持的文件后缀 → 内部 source_type
SUPPORTED_SUFFIXES = {
    ".pptx": "pptx",
    ".pdf": "pdf",
    ".docx": "docx",
    ".json": "transcript",
    ".jsonl": "transcript",
    ".srt": "transcript",
}


class MissingDependencyError(RuntimeError):
    """可选第三方库缺失。消息里必须包含可直接复制的安装命令。"""


# --------------------------------------------------------------------------- #
# 可选依赖探测
# --------------------------------------------------------------------------- #
def _try_import(module: str):
    """尝试导入可选依赖，失败返回 ``(None, 原因)``，成功返回 ``(module, None)``。"""
    try:
        return __import__(module), None
    except Exception as exc:  # noqa: BLE001 - 第三方库导入失败的形态很多
        return None, f"{type(exc).__name__}: {exc}"


def _import_pymupdf():
    """导入 PyMuPDF，**优先新模块名 ``pymupdf``**，老版本才退回 ``fitz``。

    为什么不能直接用 ``import fitz``：pymupdf 1.24 起 ``fitz`` 只是兼容别名，
    导入时会打印 ``The `fitz` API is deprecated and will be removed in future``，
    且官方计划在未来版本移除。把它当主路径等于埋一个"某天 PDF 解析整体失效"的雷，
    同时每次 ``--check-deps`` 都会刷一条无用告警。

    返回 ``(module, error)``，两者必有一个为 None。
    """
    mod, err = _try_import("pymupdf")
    if mod is not None:
        return mod, None
    legacy, legacy_err = _try_import("fitz")
    if legacy is not None:
        return legacy, None
    return None, err or legacy_err


def dependency_report() -> dict[str, bool]:
    """探测可选依赖是否可用（CLI ``--check-deps`` 与 README 都用它）。

    ``pdfminer.six`` 是 PDF 的**退路**：pymupdf 可用时它缺失不构成问题，
    调用方不该因为它为 False 就判定环境不可用（见 ``main()`` 的 ``--check-deps``）。
    """
    return {
        "python-pptx": _try_import("pptx")[0] is not None,
        "pymupdf": _import_pymupdf()[0] is not None,
        "pdfminer.six": _try_import("pdfminer.high_level")[0] is not None,
        "python-docx": _try_import("docx")[0] is not None,
    }


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def ms_to_timecode(ms: int | float | None, with_ms: bool = False) -> str:
    """毫秒 → ``HH:MM:SS``（``with_ms=True`` 时带 ``.mmm``）。样例里用的是 ``00:08:10`` 形式。"""
    if ms is None:
        return ""
    total = max(0.0, float(ms) / 1000.0)
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = total % 60
    if with_ms:
        return f"{h:02d}:{m:02d}:{int(s):02d}.{int(round((s - int(s)) * 1000)):03d}"
    return f"{h:02d}:{m:02d}:{int(s):02d}"


def timecode_range(start_ms: int | float | None, end_ms: int | float | None) -> str:
    """生成 ``00:08:10-00:18:40`` 形式的定位串（与 kp.yaml/qa.yaml 的 locator 风格一致）。"""
    return f"{ms_to_timecode(start_ms)}-{ms_to_timecode(end_ms)}"


def _clean_text(text: Any) -> str:
    """块内文本规整：统一换行、去零宽字符、压掉多余空白（不动中文标点）。"""
    s = str(text or "")
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t\u00a0]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# --------------------------------------------------------------------------- #
# PPTX
# --------------------------------------------------------------------------- #
def parse_pptx(path: str | Path) -> dict:
    """解析 ``.pptx``：逐页返回 ``{page, text, notes}``。

    实现要点：
      * 按「形状顺序」拼接文本，保留段落换行；
      * 表格文本单独抽出（PPT 里的表格常是练习题，不能丢）；
      * 备注页（notes）单独存 ``notes``，因为备注里常有"提问/停顿"等教学指令，
        它们是切分信号而不是朗读文本。

    依赖：``python-pptx``（``pip install python-pptx``）。缺失时抛
    ``MissingDependencyError``，不会静默返回空结果——静默失败比报错危险得多。
    """
    pptx, err = _try_import("pptx")
    if pptx is None:
        raise MissingDependencyError(
            "解析 .pptx 需要 python-pptx，请执行：pip install python-pptx\n"
            f"（导入失败原因：{err}）"
        )
    p = Path(path)
    prs = pptx.Presentation(str(p))  # type: ignore[attr-defined]
    blocks: list[dict] = []
    pages: list[dict] = []
    for i, slide in enumerate(prs.slides, start=1):
        texts: list[str] = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.text_frame is not None:
                for para in shape.text_frame.paragraphs:
                    t = _clean_text("".join(run.text for run in para.runs) or para.text)
                    if t:
                        texts.append(t)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:  # type: ignore[attr-defined]
                    cells = [_clean_text(c.text) for c in row.cells]
                    line = " | ".join(c for c in cells if c)
                    if line:
                        texts.append(line)
        notes = ""
        if getattr(slide, "has_notes_slide", False):
            try:
                notes = _clean_text(slide.notes_slide.notes_text_frame.text)
            except Exception:  # noqa: BLE001 - 部分文件备注页结构异常
                notes = ""
        text = "\n".join(texts)
        pages.append({"page": i, "text": text, "notes": notes})
        if text or notes:
            blocks.append({
                "locator": f"p.{i}",
                "type": "slide",
                "text": text,
                "start_ms": None,
                "end_ms": None,
                "confidence": None,
                "notes": notes or None,
            })
    return {
        "blocks": blocks,
        "meta": {"page_count": len(pages), "pages": pages},
        "parser": "python-pptx",
        "warnings": [],
    }


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def parse_pdf(path: str | Path) -> dict:
    """解析 ``.pdf``：逐页返回文本（优先 pymupdf，退化到 pdfminer.six）。

    依赖（二者取一）::

        pip install pymupdf        # 首选，快，版面还原好
        pip install pdfminer.six   # 退路，纯 Python 但慢

    扫描版 PDF（图片型）两种库都抽不出文字，会在 ``warnings`` 里显式提示需要 OCR，
    而不是产出一个空的解析结果骗过下游。
    """
    p = Path(path)
    fitz, fitz_err = _import_pymupdf()
    warnings: list[str] = []
    if fitz is not None:
        blocks: list[dict] = []
        with fitz.open(str(p)) as doc:  # type: ignore[attr-defined]
            for i, page in enumerate(doc, start=1):
                text = _clean_text(page.get_text("text"))
                blocks.append({
                    "locator": f"p.{i}", "type": "page", "text": text,
                    "start_ms": None, "end_ms": None, "confidence": None, "notes": None,
                })
        parser = "pymupdf"
    else:
        pdfminer_high, _ = _try_import("pdfminer.high_level")
        if pdfminer_high is None:
            raise MissingDependencyError(
                "解析 .pdf 需要 pymupdf 或 pdfminer.six，请执行其中之一：\n"
                "  pip install pymupdf\n"
                "  pip install pdfminer.six\n"
                f"（pymupdf 导入失败原因：{fitz_err}）"
            )
        from pdfminer.high_level import extract_pages  # type: ignore
        from pdfminer.layout import LTTextContainer  # type: ignore

        blocks = []
        for i, layout in enumerate(extract_pages(str(p)), start=1):
            texts = [
                _clean_text(el.get_text())
                for el in layout
                if isinstance(el, LTTextContainer)
            ]
            blocks.append({
                "locator": f"p.{i}", "type": "page", "text": _clean_text("\n".join(texts)),
                "start_ms": None, "end_ms": None, "confidence": None, "notes": None,
            })
        parser = "pdfminer.six"
        warnings.append("使用 pdfminer.six 降级解析，版面与阅读顺序可能与原文有偏差")

    empty = [b["locator"] for b in blocks if not b["text"]]
    if blocks and len(empty) == len(blocks):
        warnings.append("所有页面均未抽到文本，可能是扫描版 PDF；需要先做 OCR（本流水线不内置 OCR）")
    elif empty:
        warnings.append(f"以下页面无文本层（疑似图片页）：{', '.join(empty[:10])}")
    return {
        "blocks": blocks,
        "meta": {"page_count": len(blocks), "empty_pages": empty},
        "parser": parser,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def parse_docx(path: str | Path) -> dict:
    """解析 ``.docx``：按段落抽文本，表格按行拼成 ``a | b | c``。

    依赖：``python-docx``（``pip install python-docx``）。
    教案类素材的"教学环节标题"通常是独立段落，因此段落粒度必须保留，
    不能合并成一大段（s3 靠这个粒度找环节边界）。
    """
    docx, err = _try_import("docx")
    if docx is None:
        raise MissingDependencyError(
            "解析 .docx 需要 python-docx，请执行：pip install python-docx\n"
            f"（导入失败原因：{err}）"
        )
    p = Path(path)
    doc = docx.Document(str(p))  # type: ignore[attr-defined]
    blocks: list[dict] = []
    paragraphs: list[dict] = []
    idx = 0
    for para in doc.paragraphs:
        idx += 1
        text = _clean_text(para.text)
        style = getattr(getattr(para, "style", None), "name", None)
        paragraphs.append({"index": idx, "text": text, "style": style})
        if text:
            blocks.append({
                "locator": f"¶{idx}", "type": "paragraph", "text": text,
                "start_ms": None, "end_ms": None, "confidence": None, "notes": style,
            })
    for t_i, table in enumerate(doc.tables, start=1):
        rows = []
        for row in table.rows:
            cells = [_clean_text(c.text) for c in row.cells]
            line = " | ".join(c for c in cells if c)
            if line:
                rows.append(line)
        if rows:
            blocks.append({
                "locator": f"表{t_i}", "type": "paragraph", "text": "\n".join(rows),
                "start_ms": None, "end_ms": None, "confidence": None, "notes": "table",
            })
    return {
        "blocks": blocks,
        "meta": {"paragraph_count": len(paragraphs), "paragraphs": paragraphs,
                 "table_count": len(doc.tables)},
        "parser": "python-docx",
        "warnings": [],
    }


# --------------------------------------------------------------------------- #
# 教案（DOCX）分节抽取
# --------------------------------------------------------------------------- #
#: 为什么单独做这件事，而不是当普通 DOCX 平铺抽文本：
#: 教案是**唯一"教研已经用结构化方式写过一遍"的素材**——教学目标 / 重点 / 难点 /
#: 教学过程 通常都有明确小标题。平铺抽文本等于把结构扔掉，再让机器从课件里重新猜一遍。
#: 约定见 schema/source_conventions.yaml（DOCX → 教案）。
#:
#: 抽取结果直接映射到教研交付物：
#:   objectives       → kp.objectives（D1）
#:   key_points       → lesson.key_points（D2）
#:   difficult_points → lesson.difficult_points / kp.common_misconceptions（D1/D2）
#:   process_steps    → lesson.episodes 的骨架（D2）
#:   homework         → episode(TI_HOMEWORK) 内容
_HERE = Path(__file__).resolve().parent
_CONVENTIONS_CACHE: dict | None = None


def load_source_conventions() -> dict:
    """读取 ``schema/source_conventions.yaml``（唯一的角色/标题别名真源）。

    对流水线而言，约定读不到**不阻断解析**（退回内置别名表），但会记录一条
    ``CONVENTIONS_ERROR`` 供调用方提取为 warning——绝不允许"悄悄用了兜底表"
    而外界毫不知情。教案分节、角色路由都依赖这份文件，静默降级会让结果不可信。
    """
    global _CONVENTIONS_CACHE
    if _CONVENTIONS_CACHE is None:
        path = _HERE.parent / "schema" / "source_conventions.yaml"
        try:
            import yaml  # noqa: PLC0415 - 与 common.py 一致的延迟导入
            with open(path, encoding="utf-8") as f:
                _CONVENTIONS_CACHE = yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001
            _CONVENTIONS_CACHE = {"CONVENTIONS_ERROR": f"{type(exc).__name__}: {exc}"}
    return _CONVENTIONS_CACHE


def _conventions_warning() -> str | None:
    """约定文件读取失败时返回一句可挂到 warnings 的说明，否则 None。"""
    err = load_source_conventions().get("CONVENTIONS_ERROR")
    if err:
        return (f"采集约定文件读取失败，已退回内置别名表（角色路由与教案分节可能不准）：{err}")
    return None


#: 约定文件读不到时的兜底别名表（宁可退化，也不要因为一个 YAML 读不到就全盘失败）
_FALLBACK_SECTIONS = {
    "objectives": ["教学目标", "学习目标", "教学目的"],
    "key_points": ["教学重点", "重点"],
    "difficult_points": ["教学难点", "难点"],
    "process": ["教学过程", "教学环节", "教学流程", "活动过程"],
    "board_design": ["板书设计", "板书"],
    "homework": ["作业布置", "作业设计", "课后作业"],
    "reflection": ["教学反思", "课后反思"],
    "materials": ["教具准备", "教学准备"],
    "analysis": ["教材分析", "学情分析"],
}
_FALLBACK_META = {
    "subject": ["学科", "科目"],
    "grade": ["年级", "班级"],
    "textbook": ["教材", "教材版本"],
    "unit": ["单元", "章节"],
    "lesson_no": ["课时", "第几课时", "课次"],
    "teacher": ["授课教师", "执教", "教师"],
    "duration": ["课时安排", "时长", "教学时间"],
}

#: 标题行前缀噪声："一、"/"1."/"（一）"/"第一部分" 等，归一后要能匹配到别名
_HEADING_PREFIX = re.compile(
    r"^\s*(?:第[一二三四五六七八九十]+部分|"
    r"[一二三四五六七八九十]+[、.．)）]|"
    r"\(?[一二三四五六七八九十]+\)?[、.．)）]|"
    r"\d+[、.．)）]|"
    r"[（(]\s*\d+\s*[)）])\s*"
)


def _norm_heading(text: str) -> str:
    """把可能的标题行归一成"纯别名"，供精确比对。

    「一、教学目标：」→「教学目标」；「教学目标与要求」保持原样（它是另一个别名）。
    只做前缀去噪 + 去尾部冒号/空白，**不做模糊匹配**——
    教案正文里出现"教学目标"四个字的句子很常见，模糊匹配会把正文误判成标题。
    """
    t = _HEADING_PREFIX.sub("", (text or "").strip())
    return t.strip().strip("：: 。.、 \t")


def _section_aliases() -> dict[str, list[str]]:
    conv = load_source_conventions()
    return conv.get("lesson_plan_sections") or _FALLBACK_SECTIONS


#: 「裸别名」兜底：太泛的别名单独出现才算标题，带正文则视为正文
_FALLBACK_BARE_ALIASES = ["重点", "难点", "板书", "作业", "准备", "反思"]


def _bare_aliases() -> set[str]:
    conv = load_source_conventions()
    return set(conv.get("lesson_plan_bare_aliases") or _FALLBACK_BARE_ALIASES)


def _meta_aliases() -> dict[str, list[str]]:
    conv = load_source_conventions()
    return conv.get("lesson_plan_meta") or _FALLBACK_META


#: 教学过程内部的小环节编号（"环节一"/"1."/"(1)"），用来切出 episode 骨架
_STEP_PREFIX = re.compile(
    r"^\s*(?:环节\s*[一二三四五六七八九十\d]+|步骤\s*[一二三四五六七八九十\d]+|"
    r"[一二三四五六七八九十]+[、.．)）]|\d+[、.．)）]|[（(]\s*\d+\s*[)）])\s*"
)

#: 教案里最常见的环节名（用于给 process_steps 起名，命中不了就用原文首句）
_COMMON_STEP_NAMES = (
    "导入", "情境导入", "复习导入", "新课导入", "新知探究", "探究新知", "新授", "讲授",
    "例题", "例题讲解", "示范", "练习", "随堂练习", "巩固练习", "课堂小结", "总结",
    "作业布置", "布置作业", "板书设计", "拓展", "提问",
)


def _match_alias(text: str, alias_map: dict[str, list[str]]) -> str | None:
    """返回 text 命中的节键（objectives/key_points/...），未命中返回 None。"""
    n = _norm_heading(text)
    if not n or len(n) > 14:      # 标题不会很长；长句即便以别名开头也是正文
        return None
    for key, aliases in alias_map.items():
        if n in aliases:
            return key
    return None


def _split_heading_inline(text: str, alias_map: dict[str, list[str]]
                          ) -> tuple[str | None, str | None, str | None]:
    """处理「标题与正文同行」的写法，返回 ``(节键, 行内正文, 命中的别名)``。

    教案里极常见的写法是 ``二、教学重点：建立毫米和分米的长度观念。``
    —— 标题和内容挤在同一段。只按整段匹配别名会漏掉这一节，
    内容还会被错误地并进上一节（实测会把"教学重点"并进"教学目标"）。

    只切**第一个**分隔符：``教学重点：xxx：yyy`` → 节=教学重点，正文=``xxx：yyy``。

    第三个返回值是**实际命中的别名**，调用方据此判断它是不是"裸别名"
    （见 ``_bare_aliases``）——裸别名带正文时必须当正文，否则会把后面的环节全吞掉。
    """
    key = _match_alias(text, alias_map)
    if key:
        return key, None, _norm_heading(text)
    for sep in ("：", ":"):
        if sep in text:
            head, _, tail = text.partition(sep)
            k = _match_alias(head, alias_map)
            if k:
                return k, (tail.strip() or None), _norm_heading(head)
    return None, None, None


def _extract_lesson_plan_meta(texts: list[str]) -> dict:
    """从教案封面/表格里抽课题、学科、年级、教材、课时等。

    支持三种常见形态：
      ``学科：数学``                    （段落）
      ``学科 | 数学 | 年级 | 三年级``    （表格行，单元格两两配对）← parse_docx 拼成这种
      ``《毫米和分米的认识》教学设计``    （书名号兜底取课题）

    配对类形态必须成对处理：早期实现按"第一个分隔符切分"，
    结果 ``学科 | 数学 | 年级 | 三年级`` 被解析成 学科="数学 | 年级 | 三年级"，
    年级直接丢了。
    """
    meta: dict[str, str] = {}
    alias_map = _meta_aliases()
    for raw in texts:
        for line in (raw or "").splitlines():
            line = line.strip()
            if not line:
                continue

            # 形态二：表格行 "键 | 值 | 键 | 值" —— 相邻两格配对
            if "|" in line:
                cells = [c.strip() for c in line.split("|")]
                for i in range(0, len(cells) - 1, 2):
                    k = _norm_heading(cells[i])
                    v = cells[i + 1].strip()
                    for mk, aliases in alias_map.items():
                        if k in aliases and mk not in meta and v:
                            meta[mk] = v[:60]
            else:
                # 形态一：键：值
                m = re.match(r"^([^：:|]{1,8})\s*[:：]\s*(.+)$", line)
                if m:
                    key, val = _norm_heading(m.group(1)), m.group(2).strip()
                    for mk, aliases in alias_map.items():
                        if key in aliases and mk not in meta and val:
                            meta[mk] = val[:60]

            # 形态三：课题（两种写法都在同一行上找）
            if "title" not in meta:
                mt = re.search(r"课题\s*[:：|]?\s*[《【]?([^》】\n|]{2,40})", line)
                if mt:
                    meta["title"] = mt.group(1).strip()[:60]
                else:
                    # 没有"课题"字样时，退回第一个书名号内容（教案封面常见写法）
                    mq = re.search(r"[《【]([^》】]{2,40})[》】]", line)
                    if mq:
                        meta["title"] = mq.group(1).strip()[:60]
    return meta


def _split_process_steps(lines: list[str]) -> list[dict]:
    """把「教学过程」拆成环节骨架，对应 lesson.episodes（D2）。

    两种识别方式，按优先级：
      1. 显式编号（"环节一" / "1." / "（1）"）—— 最可靠
      2. 常见环节名（导入/新授/练习/总结…）出现在行首
    识别不出就返回空列表，让教研自己填——**不猜**。
    """
    steps: list[dict] = []
    cur: dict | None = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        name = None
        if _STEP_PREFIX.match(s):
            name = _STEP_PREFIX.sub("", s).strip()[:20]
        else:
            for nm in _COMMON_STEP_NAMES:
                if s.startswith(nm) and len(s) <= len(nm) + 14:
                    name = nm
                    break
        if name:
            cur = {"index": len(steps) + 1, "name": name or s[:20], "text": ""}
            steps.append(cur)
        elif cur is not None:
            cur["text"] = (cur["text"] + "\n" + s).strip()
    for st in steps:
        st["text"] = st["text"][:2000]
    return steps


def parse_lesson_plan_docx(path: str | Path) -> dict:
    """解析教案 ``.docx``：**分节抽取**，直接产出可映射到 kp / episode 的结构。

    返回结构与 ``parse_docx`` 一致（同样有 ``blocks``，下游可照常处理），
    额外在 ``meta.lesson_plan`` 里给出：

        meta.title / subject / grade / textbook / unit / lesson_no / duration
        sections{objectives, key_points, difficult_points, process, ...}
        process_steps[{index, name, text}]     ← episode 骨架
        sections_found / sections_missing      ← 缺哪些节，教研补哪些

    为什么不集成进 ``parse_docx`` 而单列一个入口：
    教案是**唯一需要"理解结构"才能用好的素材**，而课件/讲义只需要抽文本。
    把两种诉求混在一个函数里，会让两边都变脆。
    """
    base = parse_docx(path)
    paras = (base.get("meta") or {}).get("paragraphs") or []
    warnings = list(base.get("warnings") or [])
    cw = _conventions_warning()
    if cw:
        warnings.append(cw)

    alias_map = _section_aliases()
    bare = _bare_aliases()
    sections: dict[str, list[str]] = {}
    sections_found: list[str] = []
    cur_key: str | None = None

    for pa in paras:
        text = (pa.get("text") or "").strip()
        if not text:
            continue
        # 标题识别：先处理"标题：正文"同行写法，再退回纯标题匹配。
        # 样式名（Word 的 Heading/标题）能匹配上更好，但不是必需——教案多用编号而非样式。
        key, inline, matched = _split_heading_inline(text, alias_map)
        if key is None:
            style = (pa.get("style") or "") or ""
            if re.search(r"标题|Heading|head", style, re.I):
                key, inline, matched = _split_heading_inline(text, alias_map)

        # 裸别名 + 带正文 → 这是正文，不是标题。
        # 「教学过程」里的教师提示「重点：终点在 -2…」若被当成标题，
        # 会把当前小节切到 key_points，后面所有环节都被吞进去（episode 骨架从 6 缩到 3）。
        if key and inline and matched in bare:
            key, inline, matched = None, None, None

        if key:
            cur_key = key
            sections.setdefault(key, [])
            sections_found.append(f"{key}:{text[:20]}")
            if inline:                      # 标题同行的正文不能丢，它是该节的第一条内容
                sections[key].append(inline)
            continue
        if cur_key:
            sections[cur_key].append(text)

    # 表格内容也参与 meta 抽取（教案的封面信息常用表格排版）
    table_text = "\n".join(b.get("text", "") for b in base.get("blocks", [])
                           if (b.get("notes") == "table"))
    cover_texts = [pa.get("text", "") for pa in paras[:40]] + [table_text]
    lp_meta = _extract_lesson_plan_meta(cover_texts)

    required = ["objectives", "key_points", "process"]
    missing = [k for k in required if not sections.get(k)]
    if not sections:
        warnings.append(
            "未识别出任何教案小节（教学目标/教学过程…）。可能这份 DOCX 不是教案，"
            "或标题未用标准写法——请确认素材角色，或按 schema/source_conventions.yaml "
            "的 lesson_plan_sections 别名补充标题"
        )
    elif missing:
        warnings.append(
            f"教案缺少必需小节：{', '.join(missing)}。"
            f"请补写（objectives → kp.objectives，key_points → lesson.key_points）"
        )

    lp = {
        "is_lesson_plan": True,
        "title": lp_meta.get("title"),
        "subject": lp_meta.get("subject"),
        "grade": lp_meta.get("grade"),
        "textbook": lp_meta.get("textbook"),
        "unit": lp_meta.get("unit"),
        "lesson_no": lp_meta.get("lesson_no"),
        "teacher": lp_meta.get("teacher"),
        "duration": lp_meta.get("duration"),
        "sections": {k: v for k, v in sections.items()},
        "process_steps": _split_process_steps(sections.get("process", [])),
        "sections_found": sorted(set(k.split(":")[0] for k in sections_found)),
        "sections_missing": missing,
    }

    meta = dict(base.get("meta") or {})
    meta["lesson_plan"] = lp
    meta["is_lesson_plan"] = True
    return {
        "blocks": base.get("blocks") or [],
        "meta": meta,
        "parser": base.get("parser", "python-docx"),
        "warnings": warnings,
    }


def parse_docx_auto(path: str | Path) -> dict:
    """按约定自动分流：像教案 → 分节抽取；否则 → 普通段落抽取。

    判定标准很保守：至少命中 2 个教案小节别名才算教案。
    宁可把教案当普通文本（损失结构），也不要把普通讲义当教案（产出错误的 episode 骨架）。
    """
    base = parse_docx(path)
    alias_map = _section_aliases()
    hits = set()
    for pa in (base.get("meta") or {}).get("paragraphs") or []:
        k = _match_alias(pa.get("text") or "", alias_map)
        if k:
            hits.add(k)
    if len(hits) >= 2:
        return parse_lesson_plan_docx(path)
    cw = _conventions_warning()
    if cw:
        base.setdefault("warnings", []).append(cw)
    return base


# --------------------------------------------------------------------------- #
# 转写（FunASR / SRT）
# --------------------------------------------------------------------------- #
def build_funasr_command(
    audio_path: str | Path,
    out_dir: str | Path = "data/parsed",
    model: str = "paraformer-zh",
    hotword: str | None = None,
    device: str = "cpu",
) -> str:
    """返回**建议的** FunASR 命令行字符串（只打印，绝不执行）。

    为什么只打印：ASR 单文件动辄几分钟到几十分钟，且要下载模型权重（数 GB）。
    把它塞进这条轻量流水线会让 `python rag/offline/s1_parse.py` 变得不可预期。
    正确姿势是运维/算法同学在 GPU 机器上按下面的命令离线跑，再把 JSON 放回 ``data/raw/``。

    ``hotword`` 建议填本节课的知识点词表（如「毫米 分米 进率」），能显著降低
    学科词汇的错字率——这是教研能直接贡献 ASR 质量的地方。
    """
    audio = Path(audio_path)
    cmd = (
        f"funasr ++model={model} ++device={device} "
        f"++input={audio.as_posix()} "
        f"++output_dir={(Path(out_dir) / 'funasr_raw').as_posix()} "
        f"++output_timestamp=true ++batch_size_s=300"
    )
    if hotword:
        cmd += f' ++hotword="{hotword}"'
    return cmd


def _coerce_ms(value: Any, default: int = 0) -> int:
    """把 FunASR 的各种时间字段形态（ms / 秒 / 字符串）统一成整数毫秒。"""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.strip().replace("ms", "")
            if ":" in value:  # 00:01:02.500
                parts = value.split(":")
                sec = 0.0
                for part in parts:
                    sec = sec * 60 + float(part)
                return int(sec * 1000)
            value = float(value)
        num = float(value)
    except (TypeError, ValueError):
        return default
    # 小于 1000 且带小数，基本可以断定是「秒」；否则按毫秒
    if num < 1000 and not float(num).is_integer():
        return int(num * 1000)
    return int(num)


def parse_video_transcript(path: str | Path, gap_ms: int = 1500, max_chars: int = 120) -> dict:
    """解析 FunASR 输出的转写 JSON，合并成「带时间码的段落」。

    合并策略（教学场景的关键）：相邻句子若都属于同一说话人连续发言、
    且间隔小于 ``gap_ms``、且合并后不超过 ``max_chars``，就合成一个块。
    这样得到的是"一段连续讲授"，而不是一句一个块——s3 的教学感知切分需要这个粒度。

    输入支持三种形态：
      1. ``[{start_ms, end_ms, text, confidence}, ...]``（本流水线约定格式）；
      2. FunASR 原生 ``{"text": ..., "sentence_info": [...]}``（自动兼容）。
      3. 一行一句的 ``.jsonl``，兼容 ``begin_ms/end_ms`` 规范化字段。

    也支持 ``.srt`` 字幕（标准库解析，无依赖）。
    """
    p = Path(path)
    warnings: list[str] = []
    if p.suffix.lower() == ".srt":
        items = _parse_srt(p.read_text(encoding="utf-8", errors="replace"))
    elif p.suffix.lower() == ".jsonl":
        items = []
        for line_no, line in enumerate(
            p.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_no} 行不是合法 JSON：{exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL 第 {line_no} 行必须是对象")
            items.append(item)
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if "sentence_info" in raw:
                items = [
                    {"start_ms": _coerce_ms(s.get("start")), "end_ms": _coerce_ms(s.get("end")),
                     "text": s.get("text", ""), "confidence": s.get("confidence")}
                    for s in raw.get("sentence_info") or []
                ]
            elif "text" in raw:
                items = [{"start_ms": 0, "end_ms": 0, "text": raw.get("text", ""), "confidence": None}]
                warnings.append("输入只有整段 text，没有时间戳；无法生成时间码定位串")
            else:
                raise ValueError(f"无法识别的转写 JSON 结构：{sorted(raw)[:10]}")
        elif isinstance(raw, list):
            items = raw
        else:
            raise ValueError("转写 JSON 顶层必须是列表或对象")

    utterances: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = _clean_text(it.get("text"))
        if not text:
            continue
        start_ms = _coerce_ms(
            it.get("begin_ms", it.get("start_ms", it.get("begin", it.get("start"))))
        )
        utterances.append({
            "start_ms": start_ms,
            "end_ms": _coerce_ms(it.get("end_ms", it.get("end")), start_ms),
            "text": text,
            "confidence": it.get("confidence"),
            "speaker": it.get("speaker"),
        })
    utterances.sort(key=lambda x: x["start_ms"])

    merged: list[dict] = []
    for u in utterances:
        if merged:
            last = merged[-1]
            gap = u["start_ms"] - last["end_ms"]
            joined = f"{last['text']}{u['text']}"
            if (
                last.get("speaker") == u.get("speaker")
                and gap <= gap_ms
                and len(joined) <= max_chars
            ):
                confs = [c for c in (last.get("confidence"), u.get("confidence")) if isinstance(c, (int, float))]
                last.update({"text": joined, "end_ms": u["end_ms"],
                             "confidence": round(sum(confs) / len(confs), 4) if confs else None})
                continue
        merged.append(dict(u))

    blocks = []
    low_conf = 0
    for m in merged:
        conf = m.get("confidence")
        if isinstance(conf, (int, float)) and conf < 0.6:
            low_conf += 1
        blocks.append({
            "locator": timecode_range(m["start_ms"], m["end_ms"]),
            "type": "utterance",
            "text": m["text"],
            "start_ms": m["start_ms"],
            "end_ms": m["end_ms"],
            "confidence": conf,
            "speaker": m.get("speaker"),
            "notes": None,
        })
    if low_conf:
        warnings.append(f"有 {low_conf} 个段落 ASR 置信度低于 0.6，建议人工复核（常见于学科术语）")
    if not utterances:
        warnings.append("转写文件里没有任何带文本的句子")
    return {
        "blocks": blocks,
        "meta": {
            "utterance_count": len(utterances),
            "segment_count": len(blocks),
            "duration_ms": (blocks[-1]["end_ms"] if blocks else 0),
            "low_confidence_count": low_conf,
        },
        "parser": (
            "srt" if p.suffix.lower() == ".srt"
            else "funasr-jsonl" if p.suffix.lower() == ".jsonl"
            else "funasr-json"
        ),
        "warnings": warnings,
    }


def _parse_srt(text: str) -> list[dict]:
    """极简 SRT 解析（无依赖）：返回 ``[{start_ms, end_ms, text}]``。"""
    items: list[dict] = []
    pattern = re.compile(
        r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    for block in re.split(r"\n\s*\n", text.strip()):
        m = pattern.search(block)
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
        body = block[m.end():].strip()
        items.append({
            "start_ms": ((h1 * 60 + m1) * 60 + s1) * 1000 + ms1,
            "end_ms": ((h2 * 60 + m2) * 60 + s2) * 1000 + ms2,
            "text": body,
            "confidence": None,
        })
    return items


# --------------------------------------------------------------------------- #
# 调度
# --------------------------------------------------------------------------- #
PARSERS = {
    "pptx": parse_pptx,
    "pdf": parse_pdf,
    # DOCX 走自动分流：像教案就分节抽取，否则按普通段落处理。
    # 约定见 schema/source_conventions.yaml（DOCX → 教案）；判定阈值保守，宁可漏
    # 结构也不要把讲义误判成教案、产出错误的 episode 骨架。
    "docx": parse_docx_auto,
    "transcript": parse_video_transcript,
}


def parse_file(path: str | Path) -> dict:
    """按后缀选择解析器，返回统一结构的解析结果（含 source_* 元信息）。"""
    p = Path(path)
    stype = SUPPORTED_SUFFIXES.get(p.suffix.lower())
    if stype is None:
        raise ValueError(f"不支持的文件类型 {p.suffix}（支持：{sorted(SUPPORTED_SUFFIXES)}）")
    parser = PARSERS[stype]
    result = parser(p)
    # 打上素材角色（真源 schema/source_conventions.yaml）。
    # 角色是下游路由的依据：教案不该进 s3 切分、扫描件要先 OCR —— 见 s3_chunk。
    try:
        role = common.classify_role(p)
    except Exception:  # noqa: BLE001 - 约定读不到也不能让解析整体失败
        role = "unknown"
    result.update({
        "source_role": role,
        "source_path": common.rel_to_root(p),
        "source_name": p.name,
        "source_type": stype,
        "source_hash": common.sha256_file(p)[:16],
        "parsed_at": datetime.now(CST).isoformat(timespec="seconds"),
    })
    return result


def parse_dir(in_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """批量解析目录（不递归），每个素材产出 ``<name>.json``。返回成功写出的文件列表。"""
    files = common.iter_input_files(in_dir, SUPPORTED_SUFFIXES.keys())
    if not files:
        log.warning("输入目录 %s 下没有可解析素材（支持 %s）", in_dir, sorted(SUPPORTED_SUFFIXES))
        return []
    out = common.ensure_dir(out_dir)
    written: list[Path] = []
    for f in files:
        try:
            result = parse_file(f)
        except MissingDependencyError as exc:
            log.error("跳过 %s：%s", f.name, exc)
            continue
        except Exception as exc:  # noqa: BLE001 - 单个素材坏掉不应中断整批
            log.error("解析失败 %s：%s: %s", f.name, type(exc).__name__, exc)
            continue
        target = out / f"{common.slug(f.stem)}.json"
        common.write_json(target, result)
        written.append(target)
        log.info("解析完成 %s → %s（%d 块，%d 条警告）",
                 f.name, target.name, len(result["blocks"]), len(result.get("warnings") or []))
        for w in result.get("warnings") or []:
            log.warning("  %s：%s", f.name, w)
    log.info("本批共解析 %d/%d 个素材，输出目录 %s", len(written), len(files), out)
    return written


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 解析器（单独抽出来，方便测试里复用参数定义）。"""
    ap = argparse.ArgumentParser(
        prog="s1_parse",
        description="Stage1 解析：把 PPT/PDF/DOCX/转写稿统一解析为 data/parsed/<name>.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python rag/offline/s1_parse.py --in data/raw --out data/parsed\n"
            "  python rag/offline/s1_parse.py --check-deps\n"
            "  python rag/offline/s1_parse.py --funasr-hint data/raw/lesson.m4a --hotword \"毫米 分米\"\n"
        ),
    )
    common.add_in_out_args(ap, "teaching_assets/raw_materials", "data/parsed")
    ap.add_argument("--check-deps", action="store_true",
                    help="只探测可选第三方库（python-pptx/pymupdf/pdfminer/python-docx）是否可用")
    ap.add_argument("--funasr-hint", metavar="AUDIO",
                    help="打印（不执行）该音频的建议 FunASR 转写命令，用于离线 ASR")
    ap.add_argument("--hotword", default="", help="配合 --funasr-hint 的领域热词，如 \"毫米 分米 进率\"")
    ap.add_argument("--device", default="cpu", help="配合 --funasr-hint 的推理设备（cpu/cuda:0）")
    return ap


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。返回 0 表示成功（全部素材解析成功或目录为空），否则 1。"""
    args = build_parser().parse_args(argv)

    if args.check_deps:
        deps = dependency_report()
        for name, ok in deps.items():
            print(f"{'✓' if ok else '✗'} {name:14s} {'可用' if ok else '未安装'}")
        pdf_ok = deps["pymupdf"] or deps["pdfminer.six"]
        print("\n可解析格式：")
        print(f"  {'✓' if deps['python-pptx'] else '✗'} .pptx")
        print(f"  {'✓' if pdf_ok else '✗'} .pdf"
              + (f"（后端：{'pymupdf' if deps['pymupdf'] else 'pdfminer.six 降级'}）" if pdf_ok else ""))
        print(f"  {'✓' if deps['python-docx'] else '✗'} .docx")
        print("  ✓ .json / .srt（内置，无需第三方库）")
        if not deps["pdfminer.six"] and deps["pymupdf"]:
            print("\n注：pdfminer.six 未安装不构成问题——它是 PDF 的退路，pymupdf 已可用。")
        missing = [n for n, ok in deps.items() if not ok and n != "pdfminer.six"]
        if missing:
            print(f"\n建议补装：pip install {' '.join(missing)}")
        # 判定标准：**PDF 后端可用**即视为环境就绪（其余按需）。pdfminer 缺失不算失败。
        return 0 if pdf_ok else 1

    if args.funasr_hint:
        print("# 以下命令仅作建议，本脚本不会执行它。请在 GPU/离线机器上运行后，")
        print("# 把 FunASR 输出的 JSON 放回输入目录，再用本脚本解析。")
        print(build_funasr_command(args.funasr_hint, args.out_dir,
                                   hotword=args.hotword or None, device=args.device))
        return 0

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        log.error("输入目录不存在：%s（请先把原始素材放进该目录）", in_dir)
        return 1
    try:
        written = parse_dir(in_dir, args.out_dir)
    except Exception as exc:  # noqa: BLE001
        log.error("解析阶段失败：%s: %s", type(exc).__name__, exc)
        return 1
    return 0 if written or not common.iter_input_files(in_dir, SUPPORTED_SUFFIXES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
