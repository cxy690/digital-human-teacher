"""把沉淀下来的经验拼成提示词片段，注入到该老师的 system prompt 里。

位置很关键：经验片段排在**人格 Skill 之后**（越靠后越能影响输出），
这样它就像真老师「教了几年之后攒下的手感」——人格是本，经验是长出来的。

注入哪几条由 **L2 bandit 定序**决定（按实测收益的后验采样，见
:meth:`ExperienceStore.select`），而不是「被提到次数最多」的静态排序；
本模块额外负责把「这一轮选中了哪几条」记在会话上，供回合结束后回写收益。
"""
from __future__ import annotations

from typing import Any

from config.logging import get_logger
from config.settings import settings
from core.skills.experience.models import KIND_LABELS, ExperienceItem
from core.skills.experience.store import ExperienceStore, get_experience_store

log = get_logger("core.skills.experience.prompt")

_HEADER = "【{name}经验（从真实课堂沉淀，讲课时优先采用；与本条冲突时以本条的底线规则为准）】"

#: 会话上的记账键：``{"epoch": 轮次, "teacher": 老师key, "ids": [...], "text": "渲染结果"}``
#:
#: 为什么要按**轮次**缓存：同一个回合里答疑链路可能把提示词装配两遍
#: （流式答疑的「话术路」与「结构路」并行跑，两条路各装配一次）。
#: 采样是随机的，不缓存就会出现「同一位老师、同一道题，两条路拿到不同的经验」——
#: 说出去的话与写到黑板上的内容对不上。缓存后一次回合内**选一次、用一致**。
INJECTED_KEY = "_experience_injected"


def build_experience_prompt(
    teacher_key: str,
    *,
    store: ExperienceStore | None = None,
    limit: int | None = None,
    name: str = "本学科",
    session: Any | None = None,
) -> str:
    """返回经验提示词片段；该老师还没有任何经验时返回空串（调用方直接跳过）。

    ``session`` 传入时会把本轮选中的经验 id 记在 ``session.scratch`` ——
    **这是 bandit 的奖励信号来源**：回合结束后由
    :func:`core.skills.experience.feedback.record_turn_outcome` 按裁决回写
    「这批经验到底有没有帮上忙」。不记账，bandit 就永远没有反馈，等于没装。
    """
    limit = int(limit if limit is not None else settings.skill_experience_inject_limit)
    if limit <= 0:
        return ""
    key = str(teacher_key or "").strip()
    book = store or get_experience_store()

    epoch = int(getattr(session, "speech_epoch", 0) or 0) if session is not None else 0
    cached = _cached(session, epoch, key)
    if cached is not None:
        return str(cached.get("text") or "")

    picked = book.select(key, limit=limit)
    if not picked:
        _remember(session, epoch, key, [], "")
        return ""
    text = render_experience(picked, name=name)
    _remember(session, epoch, key, [it.id for it in picked], text)
    return text


def render_experience(items: list[ExperienceItem], *, name: str = "本学科") -> str:
    """把经验条目按类型分组渲染成文本（空列表返回空串）。"""
    if not items:
        return ""
    grouped: dict[str, list[ExperienceItem]] = {}
    for item in items:
        grouped.setdefault(item.kind, []).append(item)

    lines = [_HEADER.format(name=name)]
    for kind, label in KIND_LABELS.items():
        bucket = grouped.get(kind)
        if not bucket:
            continue
        lines.append(f"{label}：")
        for item in bucket:
            lines.append(f"- {item.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- 记账
def _cached(session: Any, epoch: int, teacher_key: str) -> dict[str, Any] | None:
    """本回合本轮已经选过就复用（见 :data:`INJECTED_KEY` 的说明）。"""
    if session is None:
        return None
    try:
        cached = getattr(session, "scratch", {}).get(INJECTED_KEY)
    except Exception:  # noqa: BLE001 - 记账是旁路
        return None
    if not isinstance(cached, dict):
        return None
    if cached.get("epoch") != epoch or str(cached.get("teacher") or "") != teacher_key:
        return None
    return cached


def _remember(
    session: Any, epoch: int, teacher_key: str, ids: list[str], text: str
) -> None:
    """记录本轮选中的经验（失败只告警，绝不影响注入本身）。"""
    if session is None:
        return
    try:
        session.scratch[INJECTED_KEY] = {
            "epoch": epoch,
            "teacher": teacher_key,
            "ids": list(ids),
            "text": text,
        }
    except Exception as err:  # noqa: BLE001 - 记账失败不影响答疑
        log.warning("经验注入记账失败（不影响注入）：{}", err)
