#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 · 教学感知切分（rag/offline/rag/offline/s3_chunk.py）
=============================================
**这不是固定长度切分。** 固定长度（比如每 200 字一刀）会把"提问—等待—解答"切碎，
把一句完整的话从中间劈开，是教学语言切分的典型事故。本 stage 用两类教学信号定边界：

A. 教学环节信号（主边界）
   转写文本里的 cue 词，例如「我们来看」「接下来」「请大家」「我们总结一下」。
   它们标志着一个新的教学动作开始（讲授 → 提问 → 练习 → 总结），
   是教研最认可的自然断点。cue 词表见 ``CUE_WORDS``，可被 ``--cue-file`` 覆盖。

B. 知识点边界（长度约束）
   同一知识点内的片段不超过 ``max_chars``（默认 220 字）；超长时再按句号切，
   并优先在分号/顿号的次级边界断开。一个知识点讲不完就派生新切片，
   但**绝不跨知识点合并**——L1 渲染与 RAG 检索都以 kp 为锚点。

另外：句子（句号/问号/感叹号/分号切分）是不可再分的原子单位。宁可某段短一点，
也不从句子中间断开——一口气读不完的 TTS 片段是下游无法补救的。

输出 ``data/chunks/<lesson>.yaml``（候选 segment 草稿）+ 同名 ``.json``（机器可读，
供 s4_autolabel 直接消费）。草稿字段与 ``samples/.../segments.yaml`` 对齐，
机器不确定的一律 ``status: draft`` 且带 ``review_notes``。

CLI::

    python rag/offline/rag/offline/s3_chunk.py --in data/clean --out data/chunks \\
        --lesson LESSON-MATH-P3A-RJ-U03 [--kp samples/.../kp.yaml] [--max-chars 180]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

try:  # pragma: no cover
    from . import common
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore

log = common.get_logger("s3_chunk")

DEFAULT_MAX_CHARS = 220
DEFAULT_PACE = "MEDIUM"

# ---- P2 分块策略：default（原字段/cue 切分，行为不变）为基准，新增两种 ----
STRATEGIES = ("default", "semantic", "parent_child")
DEFAULT_STRATEGY = "default"
#: 相邻句 bge 余弦低于此值即断开。实测（bge-m3 归一化向量）：同主题相邻句 ~0.6-0.85，
#: 跨主题跳变 ~0.3-0.5。0.5 与在线检索门限一致，作为初值，可 --sem-threshold 调。
DEFAULT_SEM_THRESHOLD = 0.5
#: 父块（节）主键模板：一个 clean 块 = 一个「节」，作为 parent_child 的上下文单元。
PARENT_ID_TPL = "{lesson_id}-PB{block_idx:03d}"

#: 教学环节 cue 词 → 提示的教学意图（只作为 draft 提示，最终由 s4/教研确认）
CUE_WORDS: list[tuple[str, str]] = [
    ("我们来看", "TI_LECTURE"), ("我们看", "TI_LECTURE"), ("请看", "TI_LECTURE"),
    ("接下来", "TI_TRANSITION"), ("下面我们", "TI_TRANSITION"), ("然后我们", "TI_TRANSITION"),
    ("好，我们", "TI_TRANSITION"),
    ("请大家", "TI_PRACTICE"), ("请同学们", "TI_PRACTICE"), ("拿出练习本", "TI_PRACTICE"),
    ("自己做", "TI_PRACTICE"), ("试着做", "TI_PRACTICE"), ("做一做", "TI_PRACTICE"),
    ("我们总结一下", "TI_SUMMARY"), ("总结一下", "TI_SUMMARY"), ("这节课我们", "TI_SUMMARY"),
    ("回顾一下", "TI_SUMMARY"),
    ("谁能告诉", "TI_QUESTION"), ("谁来回答", "TI_QUESTION"), ("想一想", "TI_QUESTION"),
    ("思考一下", "TI_QUESTION"), ("谁来说说", "TI_QUESTION"),
    ("举个例子", "TI_DEMO"), ("看这道题", "TI_DEMO"), ("我们来做这道", "TI_DEMO"),
    ("老师示范", "TI_DEMO"),
    ("上节课我们", "TI_REVIEW"), ("还记得", "TI_REVIEW"), ("复习一下", "TI_REVIEW"),
    ("今天我们要", "TI_INTRO"), ("我们先看一个", "TI_INTRO"), ("情境", "TI_INTRO"),
    ("作业是", "TI_HOMEWORK"), ("今天的作业", "TI_HOMEWORK"), ("布置作业", "TI_HOMEWORK"),
    ("下课", "TI_FAREWELL"), ("同学们再见", "TI_FAREWELL"),
]
CUE_MAP: dict[str, str] = dict(CUE_WORDS)

#: 允许 cue 词出现在句首多少字以内仍算"环节边界"（避免正文中间提到"请大家"就切一刀）
CUE_LEAD_TOLERANCE = 6

#: 次级切分边界（超长句内优先在这些标点后断开）
SECONDARY_BREAKS = "；;、，,"
SENTENCE_END = "。！？!?…"

EPISODE_PREFIX_TPL = "{lesson_id}-EP{idx:03d}"
SEGMENT_ID_TPL = "{episode_id}-SG{idx:03d}"


# --------------------------------------------------------------------------- #
# 句子切分（保 offset，用于回算 source_locator）
# --------------------------------------------------------------------------- #
def split_sentences_with_offsets(text: str) -> list[dict]:
    """按句末标点切句，返回 ``[{text, start, end}]``（offset 相对整块文本）。

    连续标点（如 "……" "！？"）与右引号一并归入前一句，避免产生孤立的标点句。
    """
    out: list[dict] = []
    start = 0
    n = len(text or "")
    i = 0
    while i < n:
        ch = text[i]
        if ch in SENTENCE_END:
            j = i
            while j + 1 < n and (text[j + 1] in SENTENCE_END or text[j + 1] in "」』\"'"):
                j += 1
            seg = text[start:j + 1].strip()
            if seg:
                out.append({"text": seg, "start": start, "end": j + 1})
            start = j + 1
            i = j + 1
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        out.append({"text": tail, "start": start, "end": n})
    return out


def find_cues(text: str) -> list[str]:
    """返回该块文本里出现的 cue 词（保持首次出现顺序，去重）。"""
    found: list[str] = []
    for cue in CUE_MAP:
        if cue in text and cue not in found:
            found.append(cue)
    return found


def is_cue_boundary(sentence: str) -> tuple[bool, str | None]:
    """判断某句是否是"教学环节起始句"：cue 词出现在句首附近。"""
    body = sentence.lstrip("　 \t（(【[")
    for cue in sorted(CUE_MAP, key=len, reverse=True):
        pos = body.find(cue)
        if 0 <= pos <= CUE_LEAD_TOLERANCE:
            return True, cue
    return False, None


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """超长单句的兜底切分：优先在次级标点断开，其次硬切。

    正常教学文本不应走到硬切分支；一旦走到，说明 ASR 丢标点或素材本身是长段落，
    调用方会在 ``warnings`` 里报告，提示人工加标点。
    """
    if len(sentence) <= max_chars:
        return [sentence]
    pieces: list[str] = []
    buf = ""
    for part in re.split(rf"(?<=[{re.escape(SECONDARY_BREAKS)}])", sentence):
        if not part:
            continue
        if len(buf) + len(part) <= max_chars:
            buf += part
            continue
        if buf:
            pieces.append(buf)
        if len(part) > max_chars:
            for k in range(0, len(part), max_chars):
                pieces.append(part[k:k + max_chars])
            buf = ""
        else:
            buf = part
    if buf:
        pieces.append(buf)
    return [p for p in pieces if p]


def pack_chunks(sentences: list[dict], max_chars: int) -> list[dict]:
    """把句子打包成切片：cue 边界强制断段，段内不超过 ``max_chars``。

    返回 ``[{text, start, end, cue, forced}]``；``forced=True`` 表示该段由 cue 词起头。
    """
    chunks: list[dict] = []
    cur: dict | None = None
    for sent in sentences:
        for piece in split_long_sentence(sent["text"], max_chars):
            is_cue, cue = is_cue_boundary(piece)
            if cur is not None and (is_cue or len(cur["text"]) + len(piece) > max_chars):
                chunks.append(cur)
                cur = None
            if cur is None:
                cur = {"text": piece, "start": sent["start"], "end": sent["end"],
                       "cue": cue if is_cue else None, "forced": bool(is_cue)}
            else:
                cur["text"] += piece
                cur["end"] = sent["end"]
                if is_cue and not cur.get("cue"):
                    cur["cue"] = cue
    if cur is not None:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------- #
# 语义切分用的嵌入器（惰性加载 + 文本级缓存；仅 semantic 策略触发，单测可注入 fake）
# --------------------------------------------------------------------------- #
Embedder = Callable[[Sequence[str]], list[list[float]]]

_EMBEDDER: Embedder | None = None
_VEC_CACHE: dict[str, list[float]] = {}


def set_embedder(fn: Embedder | None) -> None:
    """注入编码器（单测用 fake；传 None 恢复为默认惰加载 bge-m3）。同时清空缓存。"""
    global _EMBEDDER
    _EMBEDDER = fn
    _VEC_CACHE.clear()


def _default_embedder(texts: Sequence[str]) -> list[list[float]]:
    """默认编码器：复用离线权威嵌入器 ``tools.store_common.embed_texts``（同模同维）。

    放在函数内延迟导入：不把 torch/pymilvus 拖进整个 stage，只在走 semantic 时加载；
    与在线 embedder 同模型（bge-m3）同维度（1024），保证向量空间一致。
    """
    root = str(common.ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools import store_common  # noqa: PLC0415  仅 semantic 策略才走这里

    return store_common.embed_texts(list(texts))


def _embed_cached(texts: Sequence[str]) -> list[list[float]]:
    """带缓存的批量编码：相同句子只编一次（任务要求“缓存复用”）。"""
    embedder = _EMBEDDER or _default_embedder
    missing = [t for t in texts if t not in _VEC_CACHE]
    if missing:
        for t, v in zip(missing, embedder(missing)):
            _VEC_CACHE[t] = list(v)
    return [_VEC_CACHE[t] for t in texts]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度（归一化向量下即点积，但这里显式除模长以兼容 fake）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def pack_semantic(sentences: list[dict], max_chars: int, *, threshold: float = DEFAULT_SEM_THRESHOLD,
                  embedder: Embedder | None = None) -> list[dict]:
    """语义切分：相邻句 embedding 余弦 < threshold 即断开；超长单句仍按次级标点兵底。

    与 :func:`pack_chunks` 同构（返回 ``[{text,start,end,cue,forced}]``，句子不截断），
    只是断点从“cue 词”换成“语义跳变”。编码器可注入（单测用 fake）。
    """
    if not sentences:
        return []
    _embed = embedder or _embed_cached
    vecs = _embed([s["text"] for s in sentences])
    out: list[dict] = []
    cur: dict | None = None
    for i, sent in enumerate(sentences):
        for k, sub in enumerate(split_long_sentence(sent["text"], max_chars)):
            if cur is None:
                cur = {"text": sub, "start": sent["start"], "end": sent["end"],
                       "cue": None, "forced": False}
                continue
            break_here = len(cur["text"]) + len(sub) > max_chars
            # 跨句边界才比相似度；同一长句内部硬切不算语义跳变。
            if not break_here and k == 0 and i > 0:
                break_here = _cosine(vecs[i - 1], vecs[i]) < threshold
            if break_here:
                out.append(cur)
                cur = {"text": sub, "start": sent["start"], "end": sent["end"],
                       "cue": None, "forced": False}
            else:
                cur["text"] += sub
                cur["end"] = sent["end"]
    if cur is not None:
        out.append(cur)
    return out


def pack_parent_child(sentences: list[dict], max_chars: int, *, parent_id: str) -> list[dict]:
    """父子切分：子块（段）= :func:`pack_chunks` 的检索单元，统一挂到父块（节）下。

    每个子块多一个 ``parent_id``，供入库时写入 ``kp_chunks_v2.parent_id``；
    “小-到大”检索（命中精确子块→用上下文父块）就是靠这个回指。
    """
    pieces = pack_chunks(sentences, max_chars)
    for p in pieces:
        p["parent_id"] = parent_id
    return pieces


# --------------------------------------------------------------------------- #
# 定位串
# --------------------------------------------------------------------------- #
def build_locator(block: dict, char_start: int, char_end: int, block_text: str) -> str:
    """根据块类型生成 ``source_locator``：转写用时间码范围，PPT 用页码（可带行内比例）。"""
    kind = block.get("type")
    if kind == "utterance" and block.get("start_ms") is not None:
        s, e = int(block["start_ms"]), int(block.get("end_ms") or block["start_ms"])
        total = max(1, len(block_text))
        a = s + int((e - s) * (char_start / total))
        b = s + int((e - s) * (char_end / total))
        return f"{_tc(a)}-{_tc(b)}"
    base = str(block.get("locator") or "")
    if kind == "slide":
        return base
    if kind == "paragraph":
        return base
    if kind == "page":
        return base
    return base


def _tc(ms: int) -> str:
    """毫秒 → ``HH:MM:SS``（与 samples 里 locator 的写法一致）。"""
    total = max(0, int(ms)) // 1000
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def slide_ref(block: dict, unit: str | None = None) -> str | None:
    """PPT 块 → ``PPT-<unit>-p<N>`` 形式的画面引用（与样例 render_hints.slide_ref 同风格）。"""
    loc = str(block.get("locator") or "")
    m = re.match(r"p\.(\d+)", loc)
    if not m:
        return None
    return f"PPT-{unit or block.get('_unit') or 'U??'}-p{m.group(1)}"


# --------------------------------------------------------------------------- #
# 知识点加载与匹配
# --------------------------------------------------------------------------- #
def load_kp_index(kp_path: str | Path | None) -> dict[str, list[str]]:
    """读 kp.yaml，返回 ``{匹配关键词: [kp_id, ...]}``。

    ``--kp`` 没给时会自动尝试 ``samples/<lesson_id>/kp.yaml``。找不到就返回空表，
    后续 kp_ids 留空并在 review_notes 里明确写"待教研挂知识点"（不猜、不编）。

    匹配词来自三处：
      1. ``name`` 全称；
      2. ``aliases`` 全部别名（教研手写的口语说法，最贴近学生真实提问）；
      3. 从 ``name`` 派生出的「核心术语」——去掉"的认识/的换算/…"这类结构后缀后的部分。
         没有第 3 条的话，"毫米的认识"匹配不上课文里的"毫米"，召回率会低得没法用。
         派生词里有泛化风险的长词（如"长度单位换算"→"长度单位"）已用黑名单排除。
    """
    if not kp_path:
        return {}
    items = common.read_yaml_list(kp_path)
    index: dict[str, list[str]] = {}
    for kp in items:
        kp_id = kp.get("kp_id")
        if not kp_id:
            continue
        keys = [kp.get("name") or ""] + list(kp.get("aliases") or []) + core_terms(kp.get("name"))
        for key in keys:
            key = str(key).strip()
            if len(key) < 2:
                continue
            index.setdefault(key, [])
            if kp_id not in index[key]:
                index[key].append(kp_id)
    return index


#: 知识点名里常见的结构后缀（"毫米的认识" → "毫米"）
STRUCT_SUFFIXES = (
    "的认识", "的换算", "的应用", "的估测与选择", "的估测", "的选择", "的测量",
    "的初步认识", "的概念", "的意义", "的性质", "的理解", "的复习", "的计算",
)
#: 派生词黑名单：这些词太泛，命中它们会把知识点认错（宁缺毋滥）
GENERIC_TERMS = {"长度单位", "单位", "认识", "换算", "测量", "估测", "计算", "图形"}


def core_terms(name: str | None) -> list[str]:
    """从知识点全名派生"核心术语"匹配词（可能返回空列表）。"""
    n = str(name or "").strip()
    if not n:
        return []
    terms: list[str] = []
    for suffix in sorted(STRUCT_SUFFIXES, key=len, reverse=True):
        if n.endswith(suffix) and len(n) - len(suffix) >= 2:
            terms.append(n[: -len(suffix)])
            break
    if "与" in n:
        head, _, tail = n.partition("与")
        if len(head) >= 2:
            terms.append(head)
        if len(tail) >= 2:
            terms.append(tail)
    return [t for t in dict.fromkeys(terms) if t not in GENERIC_TERMS]


def match_kp(text: str, kp_index: dict[str, list[str]]) -> list[str]:
    """按关键词最长优先匹配 kp_id，返回去重后的 kp_id 列表。

    长词优先很关键：先匹配「长度单位换算」再匹配「长度单位」，避免把 k003 认成 k001。
    """
    hits: list[str] = []
    for key in sorted(kp_index, key=len, reverse=True):
        if key in text:
            for kp_id in kp_index[key]:
                if kp_id not in hits:
                    hits.append(kp_id)
    return hits


def auto_find_kp(lesson_id: str) -> Path | None:
    """自动定位该课的 kp.yaml（samples 优先，其次 data/review）。"""
    candidates = [
        common.SAMPLES_DIR / lesson_id / "kp.yaml",
        common.ROOT / common.STAGE_DIRS["review"] / lesson_id / "kp.yaml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def chunk_document(
    clean_doc: dict,
    lesson_id: str,
    kp_index: dict[str, list[str]] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    episode_base: int = 2,
    style_defaults: dict | None = None,
    unit: str | None = None,
    strategy: str = DEFAULT_STRATEGY,
    sem_threshold: float = DEFAULT_SEM_THRESHOLD,
    embedder: Embedder | None = None,
) -> dict:
    """把一份 clean 文档切成候选 segment 草稿。

    ``episode_base``：机器从 EP002 开始编号（EP001 通常留给开场/点名这类交互环节，
    样板课的实际情况也是如此）。真实的 episode 归属必须由教研在 s4/审核阶段确认。

    ``strategy``：``default``（原 cue+长度切分，输出与历史逐字节一致）、
    ``semantic``（相邻句 bge 余弦低于 ``sem_threshold`` 断开）、
    ``parent_child``（子块=段仍走 default  packing，额外挂 ``parent_id``=节）。
    非 default 时，每条草稿多带 ``chunk_strategy``（parent_child 再带 ``parent_id``），
    parent_child 另在结果里产出 ``parents``（节，上下文单元）供入库版本化。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"未知切分策略 {strategy!r}，可选：{', '.join(STRATEGIES)}")
    kp_index = kp_index or {}
    style_defaults = style_defaults or {"emotion": "E_CALM", "pace": DEFAULT_PACE}
    unit = unit or (lesson_id.split("-")[-1] if lesson_id else "U??")
    warnings: list[str] = []
    drafts: list[dict] = []
    parents: list[dict] = []
    seq = 0
    ep_no = episode_base

    blocks = [b for b in (clean_doc.get("blocks") or []) if str(b.get("text") or "").strip()]
    if not blocks:
        warnings.append("clean 文档没有任何非空块，无法切分")

    for bi, block in enumerate(blocks, start=1):
        text = str(block["text"])
        sentences = split_sentences_with_offsets(text)
        if not sentences:
            continue
        parent_id = PARENT_ID_TPL.format(lesson_id=lesson_id, block_idx=bi)
        if strategy == "semantic":
            pieces = pack_semantic(sentences, max_chars, threshold=sem_threshold, embedder=embedder)
        elif strategy == "parent_child":
            pieces = pack_parent_child(sentences, max_chars, parent_id=parent_id)
            parents.append(_build_parent(block, sentences, parent_id, lesson_id, clean_doc, bi))
        else:
            pieces = pack_chunks(sentences, max_chars)
        # cue 起头的新段 → 认为进入新的教学环节
        for i, piece in enumerate(pieces):
            if piece["forced"] and i > 0:
                ep_no += 1
            seq += 1
            episode_id = EPISODE_PREFIX_TPL.format(lesson_id=lesson_id, idx=ep_no)
            kp_ids = match_kp(piece["text"], kp_index)
            chunk_text = piece["text"]
            too_long = len(chunk_text) > max_chars
            if too_long:
                warnings.append(f"切片 {seq} 长 {len(chunk_text)} 字，超过 max_chars={max_chars}（已按标点兜底切分）")
            notes = [
                "机器切分，待教研确认边界",
                f"机器推断环节：{piece.get('cue') or '按长度切分'}",
            ]
            if not kp_ids:
                notes.append("未匹配到知识点（kp_ids 留空），需教研挂接")
            if too_long:
                notes.append(f"超长（{len(chunk_text)} 字 > {max_chars}）")
            source = {"type": "LIVE_TRANSCRIPT", "ref": clean_doc.get("source_name"),
                      "locator": build_locator(block, piece["start"], piece["end"], text)}
            if block.get("confidence") is not None:
                source["asr_confidence"] = block["confidence"]
            draft = {
                "_comment": [
                    f"候选切片 {seq}（{episode_id}）：机器切分，边界/环节/知识点均需教研确认",
                    f"来源定位：{source['locator']}，原文长 {len(chunk_text)} 字",
                ],
                "schema_version": "1.0",
                "segment_id": SEGMENT_ID_TPL.format(episode_id=episode_id, idx=1),
                "episode_id": episode_id,
                "index": 1,
                "text": chunk_text,
                "text_normalized": None,
                "kp_ids": kp_ids or [],
                "style": {
                    "emotion": style_defaults.get("emotion", "E_CALM"),
                    "pace": style_defaults.get("pace", DEFAULT_PACE),
                    "emphasis": "EMP_MID",
                    "energy": 3,
                },
                "emphasis_spans": [],
                "pause_after_ms": 500,
                "render_hints": {
                    "gesture": None,
                    "board_action": None,
                    "slide_ref": slide_ref(block, unit),
                    "camera": "medium",
                },
                "est_duration_s": round(len(chunk_text) / 4.0, 1),  # 240 字/分 ≈ 4 字/秒
                "status": "draft",
                "version": 1,
                "source": source,
                "review_notes": "；".join(notes),
            }
            # 非 default 才加新键，保证 default 产物逐字节不变。
            if strategy != "default":
                draft["chunk_strategy"] = strategy
                if strategy == "parent_child":
                    draft["parent_id"] = piece.get("parent_id") or parent_id
            drafts.append(draft)

    # 重排 index / segment_id，保证同 episode 内 index 从 1 连续
    per_ep: dict[str, int] = {}
    for d in drafts:
        ep = d["episode_id"]
        per_ep[ep] = per_ep.get(ep, 0) + 1
        d["index"] = per_ep[ep]
        d["segment_id"] = SEGMENT_ID_TPL.format(episode_id=ep, idx=per_ep[ep])

    stats = {
        "block_count": len(blocks),
        "chunk_count": len(drafts),
        "episode_count": len(per_ep),
        "avg_chars": round(sum(len(d["text"]) for d in drafts) / len(drafts), 1) if drafts else 0,
        "max_chars_seen": max((len(d["text"]) for d in drafts), default=0),
        "chunks_without_kp": sum(1 for d in drafts if not d["kp_ids"]),
        "max_chars_limit": max_chars,
    }
    result = {
        "lesson_id": lesson_id,
        "source_name": clean_doc.get("source_name"),
        "source_hash": clean_doc.get("source_hash"),
        "chunks": drafts,
        "stats": stats,
        "warnings": warnings,
    }
    # 只有非 default 才写额外字段，保证 default 产物（重跑时）逐字节不变。
    if strategy != "default":
        stats["strategy"] = strategy
        stats["parent_count"] = len(parents)
        stats["children_with_parent"] = sum(1 for d in drafts if d.get("parent_id"))
        result["parents"] = parents
    return result


def _build_parent(block: dict, sentences: list[dict], parent_id: str, lesson_id: str,
                  clean_doc: dict, block_idx: int) -> dict:
    """一个 clean 块 → 一个父块（节，上下文单元）：拼接整块原文（不截断句子）。

    父块不进 dense 检索主单元（子块才是），它只在 “小-到大” 展开时作为更大上下文，
    并被计入父块覆盖率指标。
    """
    text = "".join(s["text"] for s in sentences).strip()
    return {
        "parent_id": parent_id,
        "lesson_id": lesson_id,
        "block_idx": block_idx,
        "text": text,
        "char_count": len(text),
        "sentence_count": len(sentences),
        "status": "draft",
        "source": {"type": "LIVE_TRANSCRIPT", "ref": clean_doc.get("source_name"),
                   "locator": build_locator(block, 0, len(str(block.get("text") or "")),
                                             str(block.get("text") or ""))},
    }


def load_lesson_style_defaults(lesson_id: str) -> dict:
    """从样板课的 lesson.yaml 读 stage_band，映射出建议的默认语速/情绪。

    这只是"给一个合理初值"，让草稿看起来像成品；真实风格必须由教研按语句标。
    """
    lesson_yaml = common.SAMPLES_DIR / lesson_id / "lesson.yaml"
    lesson = common.read_yaml(lesson_yaml) or {}
    band = lesson.get("stage_band")
    mapping = (common.load_taxonomy().get("style_mapping_defaults") or {}).get(band) or {}
    return {
        "emotion": "E_CALM",
        "pace": mapping.get("pace", DEFAULT_PACE),
        "stage_band": band,
        "max_sentence_chars": next(
            (e.get("max_sentence_chars") for e in common.taxonomy_entries("stage_band")
             if e.get("id") == band), None),
    }


def chunk_dir(
    in_dir: str | Path,
    out_dir: str | Path,
    lesson_id: str,
    kp_path: str | Path | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    episode_base: int = 2,
    strategy: str = DEFAULT_STRATEGY,
    sem_threshold: float = DEFAULT_SEM_THRESHOLD,
) -> list[Path]:
    """批量切分目录下的 clean 文档；每个素材产出一份 YAML + 一份 JSON。

    非 default 策略的产物文件名带策略后缀（如 ``...__stem.semantic.json``），
    **绝不覆盖** default 产物（主链 load_milvus_store 依赖那些文件）。
    """
    files = common.iter_input_files(in_dir, [".json"])
    if not files:
        log.warning("输入目录 %s 下没有 s2 清洗产物（*.json），请先跑 s2_clean.py", in_dir)
        return []
    kp_file = Path(kp_path) if kp_path else auto_find_kp(lesson_id)
    kp_index = load_kp_index(kp_file)
    if kp_file:
        log.info("已加载知识点索引 %s（%d 个关键词）", common.rel_to_root(kp_file), len(kp_index))
    else:
        log.warning("未找到 kp.yaml，切分结果的 kp_ids 将留空，需教研手工挂接（可加 --kp 指定）")
    style_defaults = load_lesson_style_defaults(lesson_id)

    out = common.ensure_dir(out_dir)
    written: list[Path] = []
    skipped: list[tuple[str, str]] = []
    for f in files:
        clean_doc = common.read_json(f)
        # ---- 按采集约定路由：s3 只处理「课堂语言」 ----
        # 关键区分：teacher_intent 描述的是"老师在某个课堂时刻做了什么"，
        # 而课件/讲义/教案是**内容材料**，不是教学动作。把它们也丢进来，
        # 只会得到 100% 低置信度、大量"未判定"的碎片，既污染候选池又浪费教研复核时间。
        # （实测：4 份素材全切全标 → 48 条候选 48 条低置信度；按角色路由后只剩 11 条
        #   真正该标的课堂语言。）
        role = clean_doc.get("source_role") or "unknown"
        route = common.route_of(role)
        chunk_unit = route.get("chunk_unit")
        only_unit = common.s3_chunk_unit()
        if chunk_unit != only_unit:
            skipped.append((f.name, f"角色 {role}（{common.role_names().get(role, role)}）的切分单位是"
                                    f"「{chunk_unit}」，不走 s3 的 cue 词切分（s3 只处理「{only_unit}」）；"
                                    f"按约定下一步是 {route.get('next', '其它')}"))
            continue
        if route.get("needs_ocr") and not any((b.get("text") or "").strip()
                                              for b in (clean_doc.get("blocks") or [])):
            skipped.append((f.name, f"角色 {role} 需先 OCR，当前无可切分文本；"
                                    f"请先 OCR 再重跑（见 docs/02 §3.2）"))
            continue

        result = chunk_document(clean_doc, lesson_id, kp_index, max_chars,
                                episode_base=episode_base, style_defaults=style_defaults,
                                strategy=strategy, sem_threshold=sem_threshold)
        suffix = "" if strategy == "default" else f".{strategy}"
        stem = f"{lesson_id}__{common.slug(f.stem)}{suffix}"
        header = [
            f"候选切片草稿（由 rag/offline/rag/offline/s3_chunk.py 自动生成，strategy={strategy}）：{clean_doc.get('source_name')}",
            f"课程：{lesson_id}；max_chars={max_chars}；共 {result['stats']['chunk_count']} 条候选",
            f"素材角色：{role}（{common.role_names().get(role, role)}）",
            "注意：本文件是【机器草稿】，status 全为 draft，未经教研审核不可入库。",
            "教研必须做的事：① 确认/合并边界；② 把 episode_id 映射到 lesson.yaml 的真实环节；",
            "③ 补齐 kp_ids；④ 逐句改成可直接朗读的成稿（去旁白、口语虚词）；⑤ 标 style/emphasis。",
        ]
        yml = common.write_yaml_list(out / f"{stem}.yaml", result["chunks"], "\n".join(header))
        js = common.write_json(out / f"{stem}.json", result)
        written += [yml, js]
        log.info("切分完成 %s → %d 条候选 / %d 个环节（%d 条未挂知识点）",
                 f.name, result["stats"]["chunk_count"], result["stats"]["episode_count"],
                 result["stats"]["chunks_without_kp"])

    for name, why in skipped:
        log.warning("按采集约定跳过切分：%s —— %s", name, why)
    log.info("本批共处理 %d 个文件（切分 %d，按约定跳过 %d），输出目录 %s",
             len(files), len(files) - len(skipped), len(skipped), out)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="s3_chunk",
        description="Stage3 教学感知切分：按教学环节 cue 词 + 知识点长度上限切分，产出候选 segment 草稿",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python rag/offline/rag/offline/s3_chunk.py --in data/clean --out data/chunks --lesson LESSON-MATH-P3A-RJ-U03\n"
            "  python rag/offline/rag/offline/s3_chunk.py --in data/clean --out data/chunks --lesson LESSON-MATH-P3A-RJ-U03 --max-chars 180\n"
            "  python rag/offline/rag/offline/s3_chunk.py --in data/clean --out data/chunks --lesson LESSON-MATH-J7A-RJ-U01 --strategy semantic --sem-threshold 0.5\n"
            "  python rag/offline/rag/offline/s3_chunk.py --in data/clean --out data/chunks --lesson LESSON-MATH-J7A-RJ-U01 --strategy parent_child\n"
        ),
    )
    common.add_in_out_args(ap, "data/clean", "data/chunks")
    ap.add_argument("--lesson", required=True, help="课程 ID，如 LESSON-MATH-P3A-RJ-U03")
    ap.add_argument("--kp", default=None, help="kp.yaml 路径（默认自动在 samples/<lesson>/ 查找）")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"单个切片的最大字数（默认 {DEFAULT_MAX_CHARS}）")
    ap.add_argument("--episode-base", type=int, default=2,
                    help="机器编号的起始 episode 序号（默认 2，EP001 通常留给开场交互）")
    ap.add_argument("--strategy", choices=STRATEGIES, default=DEFAULT_STRATEGY,
                    help="切分策略：default=原 cue+长度切分（默认）；semantic=相邻句 bge 余弦断开；"
                         "parent_child=子块（段）挂父块（节）parent_id")
    ap.add_argument("--sem-threshold", type=float, default=DEFAULT_SEM_THRESHOLD,
                    help=f"semantic 策略的相邻句断开余弦阈值（默认 {DEFAULT_SEM_THRESHOLD}）")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        log.error("输入目录不存在：%s（请先跑 python rag/offline/rag/offline/s2_clean.py）", in_dir)
        return 1
    written = chunk_dir(in_dir, args.out_dir, args.lesson, args.kp, args.max_chars,
                        episode_base=args.episode_base, strategy=args.strategy,
                        sem_threshold=args.sem_threshold)
    return 0 if written or not common.iter_input_files(in_dir, [".json"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
