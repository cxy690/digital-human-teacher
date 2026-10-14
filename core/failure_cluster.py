"""L3 进化执行层：把一批失败**聚类成可追的缺陷**，并对症给出根因结论。

它和「单维频次」的区别（这是本模块存在的全部理由）
------------------------------------------------
``core.verdict.Window`` 已经给了四张表：``top_reasons``（原因）、``by_subject``（学科）、
``by_kind``（场景）、``steps``（每步报错数）。但它们**彼此独立**，长这样::

    原因：JSON 降级 12 次、质检扣分 8 次、步骤异常 5 次
    学科：数学 30 次、物理 12 次
    步骤：answer_graph.generate 报错 5 次

人得自己在脑子里把三张表拼起来才知道「那 12 次 JSON 降级到底发生在哪、卡在哪一步」。
本模块做的事就是把它们**压成一个签名**再统计 —— 于是每一行天然就是「一个可追的缺陷」::

    [1] JSON 降级 · 卡在 answer_graph.generate      12 次（占失败 40%）
        场景：teach 11 / courseware 1      学科：数学 9 / 物理 3
        伴随：质检扣分 7                   最近：10-12 20:41
        样本：帮我讲讲二次函数的图像和性质 → 话术断在半句、无板书

它**只读** L0 落盘的裁决（``core/verdict``），不新增任何埋点，也不改主链路。

与「经验沉淀」的分工（别把两者搞混）
----------------------------------
===============  ==========================  ==============================
                 经验沉淀（现在每回合都跑）    本模块（L3 的真正职责）
===============  ==========================  ==============================
颗粒度           一条经验                    一个**系统性缺陷**
输入             单个回合的复盘              一批失败的聚类
产出             提示词里多一句话            「哪儿坏了、改哪儿」
时机             每回合（在线）              按需（离线 / 排障时）
===============  ==========================  ==============================

根因分析（:func:`analyze_cluster`）**只在被人显式调用时才跑** ——
它要调模型、要花钱花时间，不该挂在主链路上。

数据从哪来
----------
``core/verdict`` 的 ``verdict-*.jsonl``（L0 落盘）。调用 :func:`cluster` 即可，
不依赖任何新配置。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from config.logging import get_logger
from config.settings import settings
from core.llm_json import parse_json_object

log = get_logger("core.failure_cluster")

#: 一条失败记录里可能出现的 reason 形态 -> 归一化键。
#:
#: 顺序即**优先级**：一条回合常常同时有好几个原因（JSON 降级往往伴随质检扣分），
#: 取"最靠前"的那个当主签名 —— 越靠前越接近根子（异常 > 没交付 > 结构坏 > 内容差）。
#: 其余的记进簇的「伴随」里，这样能看出「谁和谁总是一起出现」。
_REASON_RULES: tuple[tuple[str, str, str], ...] = (
    ("异常：", "exception", "回合抛异常"),
    ("空回复", "empty_reply", "空回复（无话术且无板书）"),
    ("无响应产物", "no_response", "无响应产物"),
    ("JSON 降级", "json_degraded", "JSON 降级（板书整块丢弃）"),
    ("步骤异常：", "step_error", "调用链里有步骤抛异常"),
    ("质检未通过", "quality_failed", "质检未通过"),
    ("质检问题：", "quality_issue", "质检问题"),
    ("质检扣分", "quality_deduct", "质检扣分"),
)

#: 签名里那些「每个回合都不一样」的部分（分数、耗时）要抹掉，否则同一类缺陷会被拆成几十个簇。
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class FailureCluster:
    """一个失败簇 = 一类可追的缺陷。"""

    key: str                       # 签名（稳定、可比较）
    label: str                     # 给人看的中文标签
    count: int = 0
    #: 场景 / 学科 / 老师的分布（簇内偏向哪儿，一眼看出）
    by_kind: Counter = field(default_factory=Counter)
    by_subject: Counter = field(default_factory=Counter)
    by_teacher: Counter = field(default_factory=Counter)
    #: 伴随出现的其他原因（「JSON 降级总是跟着质检扣分」这类线索）
    co_reasons: Counter = field(default_factory=Counter)
    #: 卡在哪一步（步骤异常时才有；正常路径为 "-"）
    steps: Counter = field(default_factory=Counter)
    #: 真实样本（排障时最有用的东西：原问题 + 它当时坏在哪）
    samples: list[dict[str, Any]] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""

    @property
    def step(self) -> str:
        """本簇最常卡住的那一步（没有则 ``"-"``）。"""
        return self.steps.most_common(1)[0][0] if self.steps else "-"

    def as_dict(self, *, samples: int = 3) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "count": self.count,
            "step": self.step,
            "by_kind": dict(self.by_kind.most_common(5)),
            "by_subject": dict(self.by_subject.most_common(5)),
            "by_teacher": dict(self.by_teacher.most_common(5)),
            "co_reasons": dict(self.co_reasons.most_common(5)),
            "steps": dict(self.steps.most_common(5)),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "samples": self.samples[:samples],
        }


# ---------------------------------------------------------------------- 归一化
def reason_key(reason: str) -> tuple[str, str]:
    """一条 reason -> ``(主键, 细节)``。

    细节（异常名 / 具体的质检问题文本）**保留**，因为「质检问题：话术太长」和
    「质检问题：没有板书」是两种病；而分数、次数这类每个回合都不同的数字要抹掉，
    否则同一类缺陷会被拆成几十个只差一个数字的簇。
    """
    text = str(reason or "").strip()
    for prefix, key, label in _REASON_RULES:
        if text.startswith(prefix):
            detail = _NUM_RE.sub("?", text[len(prefix):]).strip(" 　:：")
            return key, (f"{label}：{detail}" if detail else label)
    return "other", text[:60] or "未分类"


def primary_reason(reasons: list[Any]) -> tuple[str, str]:
    """从一条记录的全部 reason 里挑出**主原因**（按 :data:`_REASON_RULES` 的优先级）。"""
    parsed = [reason_key(r) for r in reasons or [] if str(r or "").strip()]
    if not parsed:
        return "other", "未分类"
    order = {key: i for i, (_, key, _) in enumerate(_REASON_RULES)}
    parsed.sort(key=lambda kv: order.get(kv[0], len(order)))
    return parsed[0]


def failed_step(rec: dict[str, Any]) -> str:
    """记录里**第一个抛异常的步骤**（``file:func``）；没有返回 ``""``。

    只看 ``ok=False`` 的步骤：慢不等于坏 —— 慢会进 ``Window.slow_steps``，
    那是性能问题，不该混进失败簇。
    """
    for step in rec.get("steps") or []:
        if step.get("ok", True):
            continue
        func = str(step.get("func") or "")
        path = str(step.get("file") or "")
        return f"{path}:{func}" if path and func else (func or path or "?")
    return ""


# ---------------------------------------------------------------------- 聚类
def cluster(records: list[dict[str, Any]], *, min_count: int = 1) -> list[FailureCluster]:
    """把一批裁决记录聚成失败簇（按命中数降序）。``pass`` 的记录会被跳过。"""
    buckets: dict[str, FailureCluster] = {}
    for rec in records or []:
        if str(rec.get("verdict") or "") == "pass":
            continue
        key_cat, label = primary_reason(rec.get("reasons") or [])
        step = failed_step(rec)
        # 签名 = 主原因 +（有的话）卡住的那一步。
        # 带上步骤是刻意的：同一个「JSON 降级」发生在生成节点还是收尾节点，
        # 修法完全不同 —— 不带步骤就只能猜。
        key = f"{key_cat}@{step}" if step else key_cat
        node = buckets.get(key)
        if node is None:
            node = FailureCluster(key=key, label=label)
            buckets[key] = node
        node.count += 1
        node.by_kind[str(rec.get("kind") or "other")] += 1
        node.by_subject[str(rec.get("subject") or "未知")] += 1
        node.by_teacher[str(rec.get("teacher") or "-")] += 1
        if step:
            node.steps[step] += 1
        # 伴随原因：本记录除主原因外的其他条目（能看出「谁跟谁总一起出现」）
        for reason in rec.get("reasons") or []:
            got_key, got_label = reason_key(reason)
            if got_key != key_cat:
                node.co_reasons[got_label] += 1
        day = str(rec.get("day") or "")
        if day:
            node.first_seen = node.first_seen or day
            node.last_seen = day
        if len(node.samples) < _MAX_SAMPLES:
            node.samples.append(_sample(rec, step))

    items = [c for c in buckets.values() if c.count >= max(1, int(min_count))]
    items.sort(key=lambda c: (-c.count, c.key))
    return items


def report(days: int | None = None, *, top: int = 10) -> dict[str, Any]:
    """聚合成可直接打印 / 返回接口的结构（L3 的主入口，**不调模型**）。

    只读 L0 的裁决文件；读不到就返回空报告（排障工具不该因为没数据就报错）。
    """
    from core.verdict import verdict_store  # noqa: PLC0415 - 延迟导入，避免无谓耦合

    window = int(days if days is not None else settings.verdict_window_days)
    try:
        records = verdict_store.read(window)
    except Exception as err:  # noqa: BLE001 - 读不到就当作没数据
        log.warning("读取裁决失败（按空报告处理）：{}", err)
        records = []

    failures = [r for r in records if str(r.get("verdict") or "") != "pass"]
    clusters = cluster(records)[:top]
    total = max(1, len(failures))
    return {
        "window_days": window,
        "records": len(records),
        "failures": len(failures),
        "clusters": [
            {**c.as_dict(), "share_pct": round(100.0 * c.count / total, 1)}
            for c in clusters
        ],
    }


def describe(days: int | None = None, *, top: int = 5) -> str:
    """把报告渲染成一段人读的文本（命令行 / 排障时直接看）。"""
    data = report(days, top=top)
    if not data["failures"]:
        return f"最近 {data['window_days']} 天没有失败回合（共 {data['records']} 条记录）。"
    lines = [
        f"最近 {data['window_days']} 天：{data['records']} 个回合，"
        f"其中 {data['failures']} 个有降级/失败。"
    ]
    for i, c in enumerate(data["clusters"], 1):
        lines.append("")
        lines.append(f"[{i}] {c['label']}    {c['count']} 次（占失败 {c['share_pct']}%）")
        if c["step"] != "-":
            lines.append(f"      卡在：{c['step']}")
        if c["by_kind"]:
            lines.append("      场景：" + " / ".join(f"{k} {v}" for k, v in c["by_kind"].items()))
        if c["by_subject"]:
            lines.append("      学科：" + " / ".join(f"{k} {v}" for k, v in c["by_subject"].items()))
        if c["co_reasons"]:
            lines.append("      伴随：" + " / ".join(f"{k} {v}" for k, v in c["co_reasons"].items()))
        if c["last_seen"]:
            lines.append(f"      最近：{c['last_seen']}")
        for s in c["samples"][:2]:
            q = str(s.get("question") or "")[:34]
            lines.append(f"      样本：{q} → {'、'.join(s.get('reasons') or [])[:52]}")
    return "\n".join(lines)


#: 每个簇最多留几条真实样本（多了会撑爆报告，排障时 3 条足够看出模式）
_MAX_SAMPLES = 5


def _sample(rec: dict[str, Any], step: str) -> dict[str, Any]:
    """从一条失败记录里取「现场」：原问题 + 原因 + 卡在哪 + 产物形态。"""
    return {
        "day": str(rec.get("day") or ""),
        "session_id": str(rec.get("session_id") or ""),
        "question": str(rec.get("question") or "")[:80],
        "reasons": [str(r)[:40] for r in (rec.get("reasons") or [])][:3],
        "kind": str(rec.get("kind") or ""),
        "step": step,
        "speech_len": rec.get("speech_len"),
        "board_ops": rec.get("board_ops"),
        "elapsed_ms": rec.get("elapsed_ms"),
    }


# ---------------------------------------------------------------------- 根因分析
_ANALYZE_PROMPT = """你是这个 AI 教师项目的排障工程师。下面是一个**失败簇**的统计与真实样本。

请判断**最可能的根因**，并给出**具体该改哪里**（文件名/模块级别即可，不要泛泛而谈）。

要求：
1. 只依据给出的证据推断；证据不足时明确说「证据不足」并把置信度打低；
2. 区分「**根因**」与「**表象**」：例如「JSON 降级」是表象，根因可能是 token 预算不足、
   契约太复杂、或某条提示词让模型跑偏；
3. 建议要落到**可执行的动作**（改哪个文件的什么），不要写「优化提示词」这种空话；
4. 输出**严格 JSON**，不要 markdown 代码块、不要解释文字。

输出格式：
{"root_cause": "一句话根因", "evidence": ["证据1", "证据2"],
 "suggestions": ["改哪里、怎么改"], "confidence": 0.0}
"""


async def analyze_cluster(cluster_data: dict[str, Any], *, llm: Any = None) -> dict[str, Any]:
    """让模型读一个失败簇，产出「根因 + 证据 + 建议」（L3 的根因分析）。

    只在**被人显式调用**时跑（命令行 / 接口）：它要调模型、花几秒钟、算钱，
    挂在主链路上是错的。失败一律返回带 ``error`` 的字典，不抛 ——
    排障工具本身报错会让人以为是系统坏了。
    """
    if not settings.failure_analysis_enable:
        return {"ok": False, "error": "FAILURE_ANALYSIS_ENABLE=false，未做根因分析"}
    try:
        from models.llm import get_llm  # noqa: PLC0415 - 延迟导入

        client = llm or get_llm()
    except Exception as err:  # noqa: BLE001 - 拿不到模型就直说
        return {"ok": False, "error": f"取模型失败：{err}"}
    try:
        raw = await client.chat(
            [
                {"role": "system", "content": _ANALYZE_PROMPT},
                {"role": "user", "content": _render_evidence(cluster_data)},
            ],
            temperature=0.2,
            max_tokens=settings.failure_analysis_max_tokens,
            json_mode=True,
        )
    except Exception as err:  # noqa: BLE001 - 分析失败不该把排障流程搞挂
        log.warning("根因分析调用失败：{}", err)
        return {"ok": False, "error": f"{type(err).__name__}: {err}"}

    parsed = parse_json_object(raw)
    if not isinstance(parsed, dict) or not str(parsed.get("root_cause") or "").strip():
        return {"ok": False, "error": "模型没给出可用的 JSON 结论", "raw": str(raw)[:400]}
    return {
        "ok": True,
        "root_cause": str(parsed.get("root_cause")).strip(),
        "evidence": [str(x)[:120] for x in (parsed.get("evidence") or [])][:5],
        "suggestions": [str(x)[:160] for x in (parsed.get("suggestions") or [])][:5],
        "confidence": _confidence(parsed.get("confidence")),
    }


def _render_evidence(c: dict[str, Any]) -> str:
    """把簇压成给模型看的证据文本（只喂它真正需要的，别把整份报告塞进去）。"""
    lines = [
        f"失败簇：{c.get('label')}",
        f"命中次数：{c.get('count')}（占全部失败 {c.get('share_pct')}%）",
    ]
    if c.get("step") and c["step"] != "-":
        lines.append(f"卡住的步骤：{c['step']}")
    for name, key in (("场景分布", "by_kind"), ("学科分布", "by_subject"), ("伴随现象", "co_reasons")):
        if c.get(key):
            lines.append(f"{name}：" + "、".join(f"{k} {v} 次" for k, v in c[key].items()))
    lines.append("真实样本：")
    for i, s in enumerate(c.get("samples") or [], 1):
        lines.append(
            f"  {i}. 学生问「{s.get('question')}」；场景 {s.get('kind')}；"
            f"裁决原因 {'、'.join(s.get('reasons') or [])}；"
            f"话术 {s.get('speech_len')} 字 / 板书 {s.get('board_ops')} 条；"
            f"耗时 {s.get('elapsed_ms')}ms"
        )
    return "\n".join(lines)


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, round(float(value), 2)))
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "FailureCluster",
    "analyze_cluster",
    "cluster",
    "describe",
    "failed_step",
    "primary_reason",
    "reason_key",
    "report",
]
