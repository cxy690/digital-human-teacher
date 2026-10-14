"""1v1 学科辅导课流程：阶段划分、时间映射与话术模板。

设计原则（与现有架构对齐）：
* 阶段推进由**前端计时器**驱动（前端到点发 ``course_phase`` 事件），后端只做
  「已用时长 -> 阶段」的映射与话术生成，不引入后端长驻定时器；
* 复用 ``config.constants.ClassPhase`` 既有枚举，不新造状态机；
* 学科元数据统一取自 ``config.subjects``，与学科选择页同源。
"""
from __future__ import annotations

import random
from typing import Any

from config.constants import (
    TUTOR_PHASE_LABELS,
    TUTOR_PHASE_ORDER,
    TUTOR_PHASE_RATIOS,
    ClassPhase,
)
from config.settings import settings
from config.teachers import GENERIC_GREETINGS


def duration_seconds() -> int:
    """一次辅导的总时长（秒）。"""
    return max(1, int(settings.course_duration_minutes)) * 60


def tutorial_plan() -> list[dict[str, Any]]:
    """给前端的阶段表：阶段 key / 中文名 / 起止秒 / 时长秒。"""
    total = duration_seconds()
    plan: list[dict[str, Any]] = []
    acc = 0.0
    for phase in TUTOR_PHASE_ORDER:
        start = int(round(acc * total))
        acc += TUTOR_PHASE_RATIOS.get(phase, 0.0)
        end = int(round(acc * total))
        plan.append(
            {
                "phase": phase.value,
                "label": TUTOR_PHASE_LABELS.get(phase, phase.value),
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": max(0, end - start),
            }
        )
    return plan


def phase_at(elapsed_seconds: float) -> ClassPhase:
    """按已用时长定位当前阶段；超出总时长则结束。"""
    total = duration_seconds()
    if elapsed_seconds >= total:
        return ClassPhase.ENDED
    acc = 0.0
    for phase in TUTOR_PHASE_ORDER:
        acc += TUTOR_PHASE_RATIOS.get(phase, 0.0) * total
        if elapsed_seconds < acc:
            return phase
    return ClassPhase.ENDED


def to_phase(value: ClassPhase | str | None) -> ClassPhase | None:
    """把字符串安全转换为 ClassPhase；非法值返回 None。"""
    if isinstance(value, ClassPhase):
        return value
    if not value:
        return None
    try:
        return ClassPhase(str(value))
    except ValueError:
        return None


def phase_label(phase: ClassPhase | str | None) -> str:
    parsed = to_phase(phase)
    if parsed is None:
        return str(phase or "")
    return TUTOR_PHASE_LABELS.get(parsed, parsed.value)


def greeting_text(subject: str, greeting: str = "") -> str:
    """进入辅导页时教师的第一句招呼。

    ``greeting`` 为该老师档案里选中的开场白（同科多老师时用于体现所选风格）；
    留空则从**通用问候池**（``config.teachers.GENERIC_GREETINGS``）随机取一条 ——
    多条随机，避免每次开课都是同一句模板。
    """
    if greeting.strip():
        return greeting.strip()
    return random.choice(GENERIC_GREETINGS).format(subject=subject)


_PHASE_INTROS: dict[ClassPhase, str] = {
    ClassPhase.OPENING: "我们先花几分钟热热身，你可以先想想今天想弄明白哪些问题。",
    ClassPhase.LECTURE: "接下来进入知识讲解环节，遇到不懂的随时问我；我讲完还会给你出一道模拟题练一练。",
    ClassPhase.INTERACTION: "现在我们来做几道模拟题巩固一下，检验刚才的学习效果，别紧张，答错也没关系。",
    ClassPhase.SUMMARY: "时间差不多啦，我们一起来回顾今天讲到的重点；这节课就到这里，记得课后把模拟题再练一遍哦。",
    ClassPhase.ENDED: "这节课结束啦，你表现得很好，下次见！",
}


def phase_intro_text(phase: ClassPhase | str | None, subject: str) -> str:
    """阶段切换时的过渡话术（温柔、口语化）。"""
    parsed = to_phase(phase)
    if parsed is None:
        return ""
    text = _PHASE_INTROS.get(parsed, "")
    if not text:
        return ""
    if parsed in (ClassPhase.OPENING, ClassPhase.LECTURE):
        return f"{text}（今天是{subject}课）"
    return text
