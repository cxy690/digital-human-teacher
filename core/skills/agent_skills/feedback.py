"""技能战绩回写：把「本回合调用的技能有没有帮上忙」写进战绩文件。

为什么必须回写
--------------
不回写，技能注册表就退化成「一份写死的清单」：模型调不调、调了有没有用，
从来没人知道 —— 于是技能只能靠人的直觉维护，而直觉恰恰是这套机制想替代的东西。

奖励从哪来
----------
复用回合裁决（:mod:`core.verdict`），**不新增任何埋点**：``pass`` = 帮上忙（gain=1），
``degraded`` / ``failed`` = 没帮上（gain=0），异常回合不记账（基础设施抖动不该让技能背黑锅）。
判定口径与经验层（L2）共用 :func:`core.skills.experience.feedback.gain_from_verdict`。

记账对象从哪来
------------
``AgentToolkit`` 在每次技能调用时把技能名记进 ``session.scratch``（按 speech_epoch 分轮，
避免跨回合串数据）；本模块在回合收口处取走并清空（与经验注入记账
``INJECTED_KEY`` 同一套「记完就清」的纪律）。
"""
from __future__ import annotations

from typing import Any

from config.logging import get_logger
from config.settings import settings
from core.skills.agent_skills.builtin import current_teacher_key
from core.skills.agent_skills.usage import SkillUsageStore, get_skill_usage_store

log = get_logger("core.skills.agent_skills.feedback")

#: 会话 scratch 上的记账键：``{"epoch": 轮次, "names": [...]}``
USED_KEY = "_skills_used"


def mark_skill_used(session: Any, name: str) -> None:
    """记一次技能调用（``AgentToolkit`` 调）；同回合去重，跨回合按 epoch 隔离。"""
    key = str(name or "").strip()
    if not key or session is None:
        return
    try:
        scratch = getattr(session, "scratch", None) or {}
        epoch = int(getattr(session, "speech_epoch", 0) or 0)
        book = scratch.get(USED_KEY)
        if not isinstance(book, dict) or book.get("epoch") != epoch:
            book = {"epoch": epoch, "names": []}
        names = [str(n) for n in (book.get("names") or [])]
        if key not in names:
            names.append(key)
        book["names"] = names
        scratch[USED_KEY] = book
    except Exception as err:  # noqa: BLE001 - 记账是旁路，失败只告警
        log.warning("技能调用记账失败（不影响课堂）：{}", err)


def consume_used_skills(session: Any) -> list[str]:
    """取走本回合调用过的技能名（取走即清空，避免下一轮重复记账）。"""
    if session is None:
        return []
    try:
        scratch = getattr(session, "scratch", None) or {}
        book = scratch.pop(USED_KEY, None)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(book, dict):
        return []
    raw_epoch = book.get("epoch")
    # 注意不能写 ``book.get("epoch") or -1``：第一回合的 epoch 就是 0，
    # ``0 or -1`` 会把它当成「没有 epoch」而误判为跨轮残账，整轮记账全丢。
    epoch = int(raw_epoch) if raw_epoch is not None else -1
    if epoch != int(getattr(session, "speech_epoch", 0) or 0):
        return []          # 上一轮没收干净的残账：丢弃，不并进这一轮
    return [str(n) for n in (book.get("names") or []) if str(n).strip()]


def record_skill_outcome(
    session: Any, record: dict[str, Any] | None, *, store: SkillUsageStore | None = None
) -> int:
    """按本回合裁决回写技能战绩；返回更新的技能数（``0`` = 本轮没什么可记的）。

    调用点是 ``Supervisor.handle_turn`` 的收口（紧跟 ``record_turn_outcome``）——
    那里是所有分支的唯一汇合处，且裁决刚刚算出来。任何异常都在这里被吞掉：
    学习链路绝不能影响课堂。
    """
    if not (settings.agent_skills_enable and settings.agent_tools_enable):
        return 0
    names = consume_used_skills(session)
    if not names:
        return 0
    try:
        from core.skills.experience.feedback import gain_from_verdict  # noqa: PLC0415

        gain = gain_from_verdict(record)
    except Exception as err:  # noqa: BLE001
        log.warning("技能收益判定失败（不影响课堂）：{}", err)
        return 0
    if gain is None:
        return 0
    teacher_key = current_teacher_key(session)
    try:
        updated = (store or get_skill_usage_store()).record_usage(names, gain=gain)
    except Exception as err:  # noqa: BLE001
        log.warning("技能战绩回写失败（不影响课堂）：{}", err)
        return 0
    if updated:
        log.info(
            "技能战绩回写：{} 个（{}，裁决 {} gain={:.0f}）",
            updated, "、".join(names), (record or {}).get("verdict"), gain,
        )
    return updated


__all__ = ["USED_KEY", "consume_used_skills", "mark_skill_used", "record_skill_outcome"]
