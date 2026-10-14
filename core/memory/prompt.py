"""学情记忆的提示词层：把检索到的记忆渲染成可注入的 system 片段。

层序中的位置（见 ``agents/shared/teaching_layers.build_messages``）：
身份 -> 任务契约 -> 学科约束 -> 共享人格层 -> **学情记忆** -> 教师经验 -> 本轮输入。

为什么不放在人格之前：人格是「这位老师是谁」，记忆是「眼前这个学生是谁」。
越靠后影响力越强 —— 具体到这堂课的信息，应当压过抽象的教学法约定。
"""
from __future__ import annotations

from typing import Any

from config.logging import get_logger
from config.settings import settings
from core.memory.models import KIND_LABELS, MemoryItem
from core.memory.store import get_memory_store

log = get_logger("core.memory.prompt")

#: 一条记忆最多渲染多少字（注入的是提示词，不是档案）
_MAX_ITEM_CHARS = 60


def render_memory(items: list[MemoryItem], *, name: str = "这位学生") -> str:
    """把记忆渲染成分组文本；无条目时返回空串（调用方直接跳过这一层）。"""
    if not items:
        return ""
    grouped: dict[str, list[str]] = {}
    for item in items:
        text = (item.text or "").strip()
        if not text:
            continue
        grouped.setdefault(item.kind, []).append(text[:_MAX_ITEM_CHARS])

    lines = [f"【学情记忆 · {name}】以下是这位学生**此前**留下的记录（按重要度排序）："]
    for kind, label in KIND_LABELS.items():
        notes = grouped.get(kind)
        if notes:
            lines.append(f"- {label}：" + "；".join(notes))
    if len(lines) == 1:
        return ""
    lines.append(
        "使用要求：只在**确实相关**时自然带出（例如「上次你在移项上错过一次，这次我们盯住它」），"
        "不要生硬罗列，也不要据此给学生贴标签或降低要求。"
    )
    return "\n".join(lines)


def build_memory_prompt(
    student_id: str,
    *,
    query: str = "",
    subject: str = "",
    limit: int | None = None,
    name: str = "这位学生",
) -> str:
    """按学生 id 取记忆并渲染；任何异常都只告警（记忆增强不该阻断上课）。"""
    student = (student_id or "").strip()
    if not settings.memory_enable or not student:
        return ""
    cap = int(limit if limit is not None else settings.memory_inject_limit)
    if cap <= 0:
        return ""
    try:
        store = get_memory_store()
        items = store.search(student, query, subject=subject, limit=cap)
    except Exception as err:  # noqa: BLE001 - 记忆层异常不应阻断答疑
        log.warning("读取学情记忆失败（{}）：{}", student, err)
        return ""
    return render_memory(items, name=name)


def build_memory_prompt_for(session: Any) -> str:
    """从会话取学生 id / 学科 / 最近提问，渲染本回合该注入的记忆层。"""
    if not settings.memory_enable:
        return ""
    student_id = str(getattr(session, "student_id", "") or "").strip()
    if not student_id:
        return ""
    subject = str(getattr(session, "subject", "") or "").strip()
    if subject == "综合":
        subject = ""
    query = _recent_question(session)
    return build_memory_prompt(student_id, query=query, subject=subject)


def _recent_question(session: Any) -> str:
    """取最近一条学生提问作为检索关键词（记忆很多时按相关性挑最相关的几条）。"""
    history = list(getattr(session, "history", None) or [])
    for msg in reversed(history):
        if getattr(msg, "role", "") == "student" and getattr(msg, "text", ""):
            return str(msg.text)[:120]
    return ""
