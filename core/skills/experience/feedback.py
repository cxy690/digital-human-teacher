"""L2 进化决策层：把「本回合的裁决」回写成经验的**实测收益**（bandit 的奖励）。

为什么必须回写
--------------
没有回写，经验表就是一条**只进不出的单向管道**：

* 复盘不断沉淀新条目（**写**）；
* 注入按 ``hits × confidence`` 排序（**读**）；
* 但「注入了之后到底有没有帮上忙」**从来没被观测过**。

于是排序依据是什么？是「这条经验被重复提到过几次」和「模型写它时有多自信」——
两者都与**实际效果**无关。一条被反复提到、注入后毫无作用的经验，会一直霸占名额，
而真正管用的那条可能因为只被提过一次而永远排不上。

奖励从哪来
----------
复用 L0 已经落盘的裁决（:mod:`core.verdict`），**不新增任何埋点**：
``pass`` = 这批经验帮上忙了（``gain=1``），``degraded`` / ``failed`` = 没帮上（``gain=0``）。

归因的诚实说明
--------------
一个回合注入 N 条经验，**没法知道**是哪一条起了作用 —— 这种归因歧义是此类系统固有的。
本模块的处理是「整批同记」：噪声会随 ``pulls`` 增长被平均掉，而真正有用的经验会稳定地
累积高 ``wins``，后验均值自然抬起来、下一轮更容易被选中。它不精确，但**方向正确**，
而且比「完全不看效果」强得多。
"""
from __future__ import annotations

from typing import Any

from config.logging import get_logger
from config.settings import settings
from core.skills.experience.prompt import INJECTED_KEY
from core.skills.experience.store import ExperienceStore, get_experience_store

log = get_logger("core.skills.experience.feedback")


def gain_from_verdict(record: Any) -> float | None:
    """裁决记录 -> 收益；返回 ``None`` 表示**这次不记账**。

    ``None`` 的几种情形，都是「有理由不给反馈」而不是「反馈为 0」：

    * **抛了异常**：TTS 挂了 / 网络抖了 / 数据库不通 —— 这是基础设施层的抖动，
      与「经验好不好」无关。记成失败会让「恰好赶上一台机器抽风」的经验背黑锅；
    * 裁决不是已知三态（观测层关了、记录被截断）—— 没有裁决就没有奖励，
      静默跳过比瞎猜一个分数好。
    """
    if not isinstance(record, dict):
        return None
    if record.get("error"):
        return None
    verdict = str(record.get("verdict") or "")
    if verdict == "pass":
        return 1.0
    if verdict in ("degraded", "failed"):
        return 0.0
    return None


def record_turn_outcome(
    session: Any, record: dict[str, Any] | None, *, store: ExperienceStore | None = None
) -> int:
    """按本回合裁决回写经验收益；返回更新的条数（``0`` = 本轮没什么可记的）。

    调用点是 ``core/supervisor.handle_turn`` 的收口（紧跟 ``record_turn_verdict``）——
    那里是所有分支的唯一汇合处，且裁决刚刚算出来。
    """
    if not (settings.skill_self_update_enable and settings.skill_experience_feedback_enable):
        return 0
    gain = gain_from_verdict(record)
    if gain is None:
        return 0
    book = _injected(session)
    if book is None:
        return 0
    ids = [str(i) for i in (book.get("ids") or []) if str(i).strip()]
    teacher_key = str(book.get("teacher") or "").strip()
    if not ids or not teacher_key:
        return 0                      # 本回合没注入过经验：没有可归因的对象
    try:
        updated = (store or get_experience_store()).record_outcome(
            teacher_key, ids, gain=gain
        )
    except Exception as err:  # noqa: BLE001 - 学习链路任何异常都不能影响课堂
        log.warning("经验收益回写失败（不影响课堂）：{}", err)
        return 0
    _clear(session)
    if updated:
        log.info(
            "经验收益回写：{} 条（{}，裁决 {} gain={:.0f}）",
            updated, teacher_key, record.get("verdict") if record else "?", gain,
        )
    return updated


# ---------------------------------------------------------------------- 内部
def _injected(session: Any) -> dict[str, Any] | None:
    if session is None:
        return None
    try:
        book = getattr(session, "scratch", {}).get(INJECTED_KEY)
    except Exception:  # noqa: BLE001 - 记账是旁路
        return None
    return book if isinstance(book, dict) else None


def _clear(session: Any) -> None:
    """记完就清。

    为什么要清：``scratch`` 会跨回合留着，而下一轮**可能压根没注入经验**
    （比如经验被停用、或该回合走的是不需要经验的链路）——
    不清的话那批旧 id 会被再记一次，``pulls`` 虚高、后验被稀释。
    """
    try:
        session.scratch.pop(INJECTED_KEY, None)
    except Exception:  # noqa: BLE001 - 清账失败不影响课堂
        pass


__all__ = ["gain_from_verdict", "record_turn_outcome"]
