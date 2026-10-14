#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5 · 问答对挖掘（rag/offline/rag/offline/s5_qa_mine.py）
============================================
从清洗后的答疑记录 / 直播转写里挖「候选问答对」，给 L3 自由问答层提供素材来源。

识别逻辑（刻意保守）
--------------------
1. **提问识别**：满足其一即视为学生提问
   - 句末是 ``？`` / ``?``；
   - 含疑问词：怎么 / 为什么 / 为啥 / 是不是 / 多少 / 能不能 / 可以吗 / 什么意思 / 哪个 / 哪里；
   - 短句 + 疑问语气助词（"…吗" / "…呢"，且长度 ≤ 20 字）。
   注意："对吗 / 对不对 / 好不好" 在真人课里绝大多数是**老师的口头确认**，不是学生提问，
   因此默认归入 ``TEACHER_CONFIRM`` 黑名单，不算提问——这是本 stage 最容易出错的地方。

2. **回答截取**：取该提问之后、满足以下条件的最长连续片段
   - 不超过 ``max_answer_chars``（默认 200 字）；
   - 不跨越下一次提问；
   - 时间跨度不超过 ``max_gap_ms``（默认 120 秒，避免把十分钟后的另一段讲授当答案）。

3. **答案定界**：答案在下一个"环节切换"处截断（复用 s3 的 cue 词表），
   因为老师一旦说"接下来我们看…"，前一个问题的回答就已经结束了。

产出 ``data/qa_candidates/<lesson>.yaml``（+ 同名 ``.json``）。

**机器只生成这四个字段**：``canonical_question`` / ``raw_answer`` / ``sources`` / ``evidence_count``。
``paraphrases`` / ``negative_examples`` / ``policy`` **必须由教研手工补齐，机器不生成**——
它们是召回率与护栏质量的命门，让模型生成就等于把质检对象交给被质检的人：
- ``paraphrases`` 要覆盖真实学生的错别字、倒装、不完整表达，模型写出来的是"标准口语"，不等于真实口语；
- ``negative_examples`` 要抓"字面相似但语义不同"的易混问法，属于教研对易错点的经验判断；
- ``policy`` 直接决定 L3 护栏行为（给不给答案、要不要上报），是安全责任，不能自动生成。

CLI::

    python rag/offline/rag/offline/s5_qa_mine.py --in data/clean --out data/qa_candidates --lesson LESSON-MATH-P3A-RJ-U03
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    from . import common
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # type: ignore

try:  # pragma: no cover
    from .s3_chunk import CUE_MAP, split_sentences_with_offsets
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from s3_chunk import CUE_MAP, split_sentences_with_offsets  # type: ignore

log = common.get_logger("s5_qa_mine")

#: 疑问词表（教研可按学科追加；加词后务必跑单测确认没有把陈述句吃进来）
QUESTION_WORDS = (
    "怎么", "为什么", "为啥", "是不是", "多少", "能不能", "可以吗", "什么意思",
    "哪个", "哪一个", "哪里", "哪些", "什么是", "有多", "什么时候", "对不对",
)
#: 语气助词问句：短句 + 以这些字结尾
QUESTION_PARTICLES = ("吗", "呢", "么")
#: 老师的口头确认，不是学生提问（避免把课堂管理语气当成问题）
TEACHER_CONFIRM = (
    "对不对", "好不好", "是不是啊", "行不行", "可以吗同学们", "明白了吗", "清楚了吗",
    "听清楚了吗", "看到了吗", "会了吗",
)
MAX_QUESTION_CHARS = 60
MIN_QUESTION_CHARS = 4


# --------------------------------------------------------------------------- #
# 提问识别
# --------------------------------------------------------------------------- #
def is_question(sentence: str) -> tuple[bool, str]:
    """判断一句是否为学生提问，返回 ``(是否, 理由)``。"""
    s = (sentence or "").strip()
    if len(s) < MIN_QUESTION_CHARS:
        return False, "过短，不构成提问"
    for bad in TEACHER_CONFIRM:
        if bad in s:
            return False, f"含教师口头确认『{bad}』，判为非提问"
    if re.search(r"[?？]\s*$", s):
        return True, "句末问号"
    for w in QUESTION_WORDS:
        if w in s:
            return True, f"含疑问词『{w}』"
    if len(s) <= 20 and s.endswith(QUESTION_PARTICLES):
        return True, "短句 + 疑问语气助词"
    return False, "无提问特征"


def is_cue_start(sentence: str) -> bool:
    """是否是教学环节起始句（用于截断上一个回答）。"""
    body = sentence.lstrip("　 \t（(【[")
    for cue in sorted(CUE_MAP, key=len, reverse=True):
        pos = body.find(cue)
        if 0 <= pos <= 6:
            return True
    return False


def canonicalize_question(text: str, max_chars: int = MAX_QUESTION_CHARS) -> str:
    """把学生原话规整成 ``canonical_question``。

    只做"去掉称呼/语气词 + 补问号 + 截断"这类机械处理；**不做改写、不做润色**，
    因为改写会引入模型幻觉，也会掩盖真实学生的口语形态（那是 paraphrase 的素材）。
    """
    s = re.sub(r"^(?:老师|老师好|那个|嗯|诶|请问|我想问一下|我想问|我问一下)[，,、\s]*", "", text.strip())
    s = re.sub(r"[。！!]+$", "？", s)
    if not s.endswith(("？", "?")):
        s += "？"
    if len(s) > max_chars:
        s = s[: max_chars - 1].rstrip("，,、 ") + "？"
    return s


# --------------------------------------------------------------------------- #
# 挖掘
# --------------------------------------------------------------------------- #
def _block_sentences(blocks: list[dict]) -> list[dict]:
    """把 clean 文档的块拆成句子流，保留块索引与时间信息。

    把 blocks 拼成一条字符流再切句，可以正确处理"一句话被 ASR 拆到两个块"的情况。
    """
    flow: list[dict] = []
    for bi, b in enumerate(blocks):
        text = str(b.get("text") or "")
        if not text:
            continue
        for sent in split_sentences_with_offsets(text):
            flow.append({
                "block_index": bi,
                "text": sent["text"],
                "start": sent["start"],
                "end": sent["end"],
                "block": b,
            })
    return flow


def _locator(block: dict, start_char: int, end_char: int, text_len: int) -> str:
    """把句子位置换算成 source_locator（转写给时间码，课件给页码）。"""
    if block.get("type") == "utterance" and block.get("start_ms") is not None:
        s = int(block["start_ms"])
        e = int(block.get("end_ms") or s)
        total = max(1, text_len)
        a = s + int((e - s) * (start_char / total))
        b = s + int((e - s) * (end_char / total))
        return f"{_tc(a)}-{_tc(b)}"
    return str(block.get("locator") or "")


def _tc(ms: int) -> str:
    total = max(0, int(ms)) // 1000
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def mine_qa_pairs(
    clean_doc: dict,
    lesson_id: str,
    max_answer_chars: int = 200,
    max_gap_ms: int = 120_000,
    min_answer_chars: int = 8,
) -> dict:
    """从一份 clean 文档挖候选问答对，返回 ``{candidates, stats, warnings}``。"""
    blocks = clean_doc.get("blocks") or []
    flow = _block_sentences(blocks)
    warnings: list[str] = []
    candidates: list[dict] = []
    seen: dict[str, dict] = {}
    asked = 0
    no_answer = 0
    duplicates = 0

    for i, item in enumerate(flow):
        ok, reason = is_question(item["text"])
        if not ok:
            continue
        asked += 1
        # ---- 截取回答 ----
        answer_parts: list[str] = []
        src_blocks: list[dict] = []
        ans_len = 0
        first_block = item["block"]
        start_ms = first_block.get("start_ms")
        for j in range(i + 1, len(flow)):
            nxt = flow[j]
            nxt_text = nxt["text"]
            if is_question(nxt_text)[0]:
                break                                  # 遇到下一个提问，当前回答结束
            if answer_parts and is_cue_start(nxt_text):
                break                                  # 环节切换，回答结束
            nb = nxt["block"]
            if (start_ms is not None and nb.get("start_ms") is not None
                    and int(nb["start_ms"]) - int(start_ms) > max_gap_ms):
                break                                  # 时间跨度过大，不太可能是同一次答疑
            if ans_len + len(nxt_text) > max_answer_chars:
                break
            answer_parts.append(nxt_text)
            ans_len += len(nxt_text)
            if not src_blocks or src_blocks[-1] is not nb:
                src_blocks.append(nb)
        raw_answer = "".join(answer_parts).strip()
        if len(raw_answer) < min_answer_chars:
            no_answer += 1                              # 无有效回答，可能只是学生自言自语
            continue

        question = canonicalize_question(item["text"])
        source = {
            "type": "LIVE_TRANSCRIPT",
            "ref": clean_doc.get("source_name"),
            "locator": _locator(first_block, item["start"], item["end"],
                                len(str(first_block.get("text") or ""))),
        }
        key = common.stable_hash(re.sub(r"[\s，,。？?！!]", "", question), 12)
        if key in seen:
            seen[key]["evidence_count"] += 1
            duplicates += 1
            if source not in seen[key]["sources"]:
                seen[key]["sources"].append(source)
            continue
        rec = {
            "_comment": [
                f"候选问答对（机器挖掘，理由：{reason}）",
                "机器只写了 canonical_question / raw_answer / sources / evidence_count 四个字段。",
                "paraphrases / negative_examples / policy 必须由教研手工补齐（机器不生成，见脚本 docstring）。",
                "status=draft；补完字段并通过 tools/lint.py 后才可改为 approved。",
            ],
            "canonical_question": question,
            "raw_answer": raw_answer,
            "sources": [source],
            "evidence_count": 1,
            "status": "draft",
            "extraction_reason": reason,
        }
        seen[key] = rec
        candidates.append(rec)
        if item["block"].get("asr_confidence") is not None or first_block.get("confidence") is not None:
            conf = first_block.get("confidence", item["block"].get("asr_confidence"))
            if isinstance(conf, (int, float)) and conf < 0.7:
                warnings.append(f"提问「{question[:16]}…」的 ASR 置信度仅 {conf:.2f}，可能识别有误")

    stats = {
        "sentence_count": len(flow),
        "question_count": asked,
        "candidate_count": len(candidates),
        "duplicate_merged": duplicates,
        "excluded_no_answer": no_answer,
    }
    if asked and not candidates:
        warnings.append("检测到提问但未取到合格回答（回答过短），建议放宽 max_answer_chars 或检查转写质量")
    if not asked:
        warnings.append("未检测到任何提问；若是答疑记录，请检查素材是否包含学生发言")
    return {
        "lesson_id": lesson_id,
        "source_name": clean_doc.get("source_name"),
        "mined_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "candidates": candidates,
        "stats": stats,
        "warnings": warnings,
    }


def mine_dir(
    in_dir: str | Path,
    out_dir: str | Path,
    lesson_id: str,
) -> list[Path]:
    """批量挖掘目录下的 clean 文档。"""
    files = common.iter_input_files(in_dir, [".json"])
    if not files:
        log.warning("输入目录 %s 下没有 s2 清洗产物（*.json），请先跑 s2_clean.py", in_dir)
        return []
    out = common.ensure_dir(out_dir)
    written: list[Path] = []
    for f in files:
        try:
            doc = common.read_json(f)
        except Exception as exc:  # noqa: BLE001
            log.error("读取失败 %s：%s", f.name, exc)
            continue
        result = mine_qa_pairs(doc, lesson_id)
        stem = f"{lesson_id}__{common.slug(f.stem)}"
        header = [
            f"候选问答对（rag/offline/rag/offline/s5_qa_mine.py 自动挖掘）：{doc.get('source_name')}",
            f"共 {result['stats']['candidate_count']} 条候选，全部 status=draft。",
            "【必须人工补齐，机器不生成】paraphrases（≥3 条，含真实口语）／",
            "negative_examples（≥1 条字面相似但语义不同）／policy（L3 护栏行为）。",
            "补齐前不要删掉 evidence_count 与 sources：它们决定这条问答的优先级与合规溯源。",
        ]
        yml = common.write_yaml_list(out / f"{stem}.yaml", result["candidates"], "\n".join(header))
        js = common.write_json(out / f"{stem}.json", result)
        written += [yml, js]
        log.info("挖掘完成 %s → %d 条候选（原始提问 %d 条，重复合并 %d 条）",
                 f.name, result["stats"]["candidate_count"], result["stats"]["question_count"],
                 result["stats"]["duplicate_merged"])
        for w in result["warnings"]:
            log.warning("  %s：%s", f.name, w)
    log.info("本批共处理 %d 个文件，输出目录 %s", len(files), out)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="s5_qa_mine",
        description="Stage5 问答挖掘：从清洗后的师生对话里挖候选问答对（只生成待人工补齐的草稿）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python rag/offline/rag/offline/s5_qa_mine.py --in data/clean --out data/qa_candidates --lesson LESSON-MATH-P3A-RJ-U03\n"
            "  python rag/offline/rag/offline/s5_qa_mine.py --in data/clean --out data/qa_candidates --lesson LESSON-XXX --max-answer-chars 300\n"
        ),
    )
    common.add_in_out_args(ap, "data/clean", "data/qa_candidates")
    ap.add_argument("--lesson", default="LESSON-MATH-P3A-RJ-U03", help="课程 ID")
    ap.add_argument("--max-answer-chars", type=int, default=200, help="单个回答的最大字数（默认 200）")
    ap.add_argument("--max-gap-ms", type=int, default=120_000,
                    help="提问与回答的最大时间跨度毫秒（默认 120000）")
    ap.add_argument("--min-answer-chars", type=int, default=8, help="回答短于此长度则丢弃（默认 8）")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        log.error("输入目录不存在：%s（请先跑 python rag/offline/rag/offline/s2_clean.py）", in_dir)
        return 1
    # 把命令行参数透传给 mine_qa_pairs
    files = common.iter_input_files(in_dir, [".json"])
    if not files:
        log.warning("输入目录 %s 下没有可处理的 *.json", in_dir)
        return 0
    out = common.ensure_dir(args.out_dir)
    written: list[Path] = []
    for f in files:
        doc = common.read_json(f)
        result = mine_qa_pairs(doc, args.lesson, args.max_answer_chars,
                               args.max_gap_ms, args.min_answer_chars)
        stem = f"{args.lesson}__{common.slug(f.stem)}"
        header = [
            f"候选问答对（rag/offline/rag/offline/s5_qa_mine.py 自动挖掘）：{doc.get('source_name')}",
            f"共 {result['stats']['candidate_count']} 条候选，全部 status=draft。",
            "【必须人工补齐，机器不生成】paraphrases（≥3 条，含真实口语）／",
            "negative_examples（≥1 条字面相似但语义不同）／policy（L3 护栏行为）。",
        ]
        written.append(common.write_yaml_list(out / f"{stem}.yaml", result["candidates"], "\n".join(header)))
        written.append(common.write_json(out / f"{stem}.json", result))
        log.info("挖掘完成 %s → %d 条候选（提问 %d 条）", f.name,
                 result["stats"]["candidate_count"], result["stats"]["question_count"])
    log.info("输出目录 %s（共 %d 个文件）", out, len(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
