"""L5 治理层：人工抽检台 —— 把「最该看一眼」的经验按**风险**排出来。

为什么是「按风险」而不是「按时间」
--------------------------------
经验是系统自己学的、立刻就用，人只能事后止损（``retire``）。整个治理环节里
**最稀缺的资源是人的注意力**：一次抽检大概只有几分钟。

按时间列「最近新增的 20 条」，这几分钟会花在绝大多数没问题的条目上；
按**风险**排序，才能让同样的几分钟落在最可能出问题的地方。

**这个闸门不阻断注入。** 新经验依旧立刻可用 —— 那是这套自我学习的核心价值，
改成「先审后用」等于把自动学习退化成人肉录入。它做的是**降低抽检成本**，
而不是把学习停下来。

四个风险信号（都能自动检测，且都指向一个具体动作）
------------------------------------------------
========================  ==========================================  ==================
信号                      含义                                        人的动作
========================  ==========================================  ==================
实测偏低                  用了 N 次、干净交付不到一半                  ``retire`` 或改措辞
高置信但从未验证          模型自评很高，一次都还没上过场               看一眼描述准不准
与另一条近重复            内容高度重合，白占一个注入名额                合并或停用一条
影响面大                  已经注入过很多次，一旦有问题波及面广          确认描述无误
========================  ==========================================  ==================

「实测偏低」和「影响面大」靠 ``ExperienceItem`` 的 ``pulls / wins / gain_rate``
（L2 的实测战绩，见 :mod:`core.skills.experience.feedback`）；
「近重复」复用 :mod:`core.skills.experience.store` 的去重归一化。
"""
from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from core.skills.experience.models import ExperienceItem
from core.skills.experience.store import (
    ExperienceStore,
    get_experience_store,
    normalize_text,
)

#: 少于这么多次实测就不下结论（样本太少的「有效率」没有意义）
MIN_PULLS_FOR_VERDICT = 5

#: 实测有效率低于它就认为「这条可能没什么用」
LOW_GAIN_RATE = 0.5

#: 注入次数到这个量级就算「影响面大」
BUSY_PULLS = 15

#: 模型自评到这个置信度、且一次都没试过 —— 值得人先扫一眼
HIGH_CONFIDENCE = 0.8


def review_queue(
    teacher_key: str = "", *, limit: int = 10, store: ExperienceStore | None = None
) -> list[dict[str, Any]]:
    """按风险降序返回「最该人工看一眼」的经验（``teacher_key`` 为空则看全部老师）。

    返回的是**视图**（在 ``to_dict`` 之上加了 ``risk`` / ``reasons`` / ``teacher``），
    可以直接丢给接口或命令行打印。
    """
    book = store or get_experience_store()
    keys = [teacher_key.strip()] if (teacher_key or "").strip() else book.list_teachers()
    rows: list[dict[str, Any]] = []
    for key in keys:
        items = [it for it in book.load(key) if it.active]
        for item in items:
            risk, reasons = risk_of(item, peers=items)
            if risk <= 0:
                continue
            rows.append(
                {
                    "teacher": key,
                    "risk": round(risk, 2),
                    "reasons": reasons,
                    "gain_rate": item.gain_rate,
                    "pulls": item.pulls,
                    "wins": item.wins,
                    "hits": item.hits,
                    "confidence": round(float(item.confidence), 2),
                    "kind": item.kind,
                    "text": item.text,
                    "id": item.id,
                }
            )
    rows.sort(key=lambda r: (-r["risk"], -r["pulls"], r["text"]))
    return rows[: max(1, int(limit))]


def risk_of(
    item: ExperienceItem, *, peers: Iterable[ExperienceItem] = ()
) -> tuple[float, list[str]]:
    """一条经验的抽检风险与理由。

    这不是「好坏」评分，而是「**人的眼睛该先落在哪一条**」的排序依据 ——
    所以同一件事只计一次、且每个理由都对应一个能立刻做的动作。
    """
    risk = 0.0
    reasons: list[str] = []

    # ① 实测说了算：上过场、干净交付率却明显偏低
    if item.pulls >= MIN_PULLS_FOR_VERDICT and item.gain_rate < LOW_GAIN_RATE:
        risk += 0.6
        reasons.append(
            f"用了 {item.pulls} 次、只有 {item.wins} 次干净交付"
            f"（有效率 {item.gain_rate:.2f}）—— 考虑停用或改措辞"
        )

    # ② 说得响但一次都没验证过：影响未知
    if item.pulls == 0 and float(item.confidence) >= HIGH_CONFIDENCE:
        risk += 0.3
        reasons.append(
            f"模型自评置信度 {float(item.confidence):.2f} 很高，但一次都还没上过场"
            " —— 先看一眼描述准不准"
        )

    # ③ 与另一条近重复：白占一个注入名额（名额只有 skill_experience_inject_limit 个）
    twin = _near_duplicate(item, peers)
    if twin is not None:
        risk += 0.45
        reasons.append(f"与「{twin.text[:16]}…」内容高度重合 —— 合并或停用其中一条")

    # ④ 影响面大：注入次数多，一旦有问题波及很多回合
    if item.pulls >= BUSY_PULLS:
        risk += 0.25
        reasons.append(f"已经注入过 {item.pulls} 次（影响面大）—— 确认描述无误")

    return min(risk, 1.0), reasons


def _near_duplicate(
    item: ExperienceItem, peers: Iterable[ExperienceItem]
) -> ExperienceItem | None:
    """在同类经验里找一条与它高度重合的（用于「白占名额」提示）。

    只比**同 kind** 的：易错点与类比写法天然相近，跨类比较会大量误报。
    """
    mine = normalize_text(item.text)
    if not mine:
        return None
    for other in peers:
        if other.id == item.id or other.kind != item.kind:
            continue
        theirs = normalize_text(other.text)
        if not theirs or theirs == mine:
            continue
        if SequenceMatcher(None, mine, theirs).ratio() >= SIMILARITY_FOR_REVIEW:
            return other
    return None


#: 「近重复」判定阈值。
#:
#: **为什么是 0.62 而不是存储层去重的 0.85**：0.85 是「判定为同一条、直接合并」，
#: 宁可漏合并也不能错合并；这里是「提醒人看一眼」，宁可多提醒。两件事的风险方向相反。
#:
#: 0.62 是实测定出来的：真实重复落在 0.71~0.76，而彼此无关的条目落在 0.07 附近
#: —— 中间隔了十倍，阈值放在中间很安全（实测数据见下方「能力边界」）。
SIMILARITY_FOR_REVIEW = 0.62

#: 本信号的能力边界（**如实标注，不假装能全抓**）：
#:
#: 字符级相似度只能抓「**字面**重复」。同样一句话换个说法写两遍，实测相似度会掉到
#: 0.35 —— 例如云端真实数据里的这两条，谁都看得出是一件事，但字面只沾一点边::
#:
#:     学生常只对一边运算，或把未跨等号的项也变号，需强调只有跨过等号的项才变号
#:     学生常只对含未知数的一边操作，忘记另一边同步变化，需强调等式如天平两边同动
#:
#: 抓这一类需要**语义**比较（embedding 或让模型判），成本高一个量级；
#: 当前环境用的是词法检索（bigram），同样够不着。
#: 好在它们仍会以「高置信但未验证」等理由出现在抽检台里 —— 只是不会被打上「重复」标签。
#: 这是个**已知缺口**，不是「已经解决」。


__all__ = [
    "BUSY_PULLS",
    "HIGH_CONFIDENCE",
    "LOW_GAIN_RATE",
    "MIN_PULLS_FOR_VERDICT",
    "SIMILARITY_FOR_REVIEW",
    "review_queue",
    "risk_of",
]
