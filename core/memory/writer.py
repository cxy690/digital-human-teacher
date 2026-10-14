"""学情记忆的写入方：什么时候该记住什么。

两条写入路径，成本与价值分明：

1. **规则写入（默认开，零成本）**：判题判错时把「错题 + 学生答的 + 正确答案」
   记成 ``mistake``。这件事**已经发生**（判题结果就在手里），不需要再问模型一遍，
   因此不花任何额度 —— 也是最该被记住的一类信息。
2. **模型蒸馏（默认关，``MEMORY_DISTILL_ENABLE``）**：回合结束后让模型从
   「学生问了什么 / 老师讲了什么」里提炼薄弱点、偏好。这一条要多一次 LLM 调用，
   默认关闭；开启后按 ``MEMORY_DISTILL_MIN_INTERVAL_SECONDS`` 节流。

为什么默认关：面向中小学的课堂里，第一次课就问不出「这位学生的稳定特征」，
高频蒸馏只会把额度花在噪音上。等有真实数据再开，是更省的选择。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from config.logging import get_logger
from config.settings import settings
from core.memory.models import KIND_LABELS
from core.memory.store import get_memory_store

log = get_logger("core.memory.writer")

_DISTILL_PROMPT = """你是学情分析器。请从这次一对一问答里，提炼 1~2 条**值得长期记住**的学情信息。
只输出 JSON：
{{"items": [{{"kind": "mistake|weak_point|preference|fact",
              "text": "不超过 40 字的中文描述"}}]}}

要求：
1. 只记「这位学生身上稳定的特征」：做错的题、反复卡住的知识点、明显的学习偏好；
2. 一次性的寒暄、与学习无关的内容、老师讲了什么，都不要记；
3. 没有值得记的就返回空数组。

学生问：{question}
老师讲：{answer}
"""

#: 单条错题描述的长度上限（与 MemoryItem 的上限一致）
_MAX_TEXT_CHARS = 80


class MemoryWriter:
    """把本回合的学情信息写进长期记忆。"""

    def __init__(self, store: Any | None = None) -> None:
        self._store = store

    @property
    def store(self) -> Any:
        return self._store or get_memory_store()

    # ---------------------------------------------------------------- 规则写入
    def record_mistake(
        self,
        session: Any,
        *,
        question: str,
        expected: str,
        student_answer: str,
        source: str = "judge",
    ) -> Any | None:
        """判题判错时记一条错题（零 LLM 成本）。"""
        if not settings.memory_enable:
            return None
        student_id = str(getattr(session, "student_id", "") or "").strip()
        if not student_id:
            return None
        stem = (question or "").strip()
        if not stem:
            return None
        text = f"{stem[:40]}"
        if student_answer.strip():
            text += f"（学生答「{student_answer.strip()[:16]}」）"
        if expected.strip():
            text += f"，正确答案「{expected.strip()[:16]}」"
        try:
            return self.store.observe(
                student_id, "mistake", text[:_MAX_TEXT_CHARS],
                subject=_subject_of(session), source=source,
            )
        except Exception as err:  # noqa: BLE001 - 记忆写入失败不该影响判题反馈
            log.warning("写入错题记忆失败：{}", err)
            return None

    def record_weak_point(self, session: Any, text: str, *, source: str = "manual") -> Any | None:
        """记一条薄弱知识点（接口 / 监控台 / 蒸馏都用这一条路径）。"""
        if not settings.memory_enable:
            return None
        student_id = str(getattr(session, "student_id", "") or "").strip()
        if not student_id or not (text or "").strip():
            return None
        try:
            return self.store.observe(
                student_id, "weak_point", text,
                subject=_subject_of(session), source=source,
            )
        except Exception as err:  # noqa: BLE001
            log.warning("写入薄弱点记忆失败：{}", err)
            return None

    # ---------------------------------------------------------------- 模型蒸馏
    def distill_due(self, session: Any) -> bool:
        """蒸馏是否到点（按会话节流，避免同一节课里反复调模型）。"""
        if not settings.memory_enable or not settings.memory_distill_enable:
            return False
        interval = float(settings.memory_distill_min_interval_seconds)
        if interval <= 0:
            return True
        last = float(getattr(session, "scratch", {}).get("_memory_distill_ts") or 0.0)
        return (time.time() - last) >= interval

    async def distill(self, session: Any, *, question: str, answer: str) -> int:
        """用模型从本回合里蒸馏学情信息；返回写入条数（失败返回 0）。"""
        if not self.distill_due(session):
            return 0
        student_id = str(getattr(session, "student_id", "") or "").strip()
        if not student_id or not (question or "").strip():
            return 0
        try:
            session.scratch["_memory_distill_ts"] = time.time()
        except Exception:  # noqa: BLE001 - scratch 不可写时继续，只是不节流
            pass

        try:
            from models.llm import get_llm  # noqa: PLC0415 - 避免导入期建立客户端

            data = await asyncio.wait_for(
                get_llm().chat_json(
                    [{
                        "role": "user",
                        "content": _DISTILL_PROMPT.format(
                            question=question[:300], answer=(answer or "")[:400]
                        ),
                    }],
                    temperature=0.2,
                    max_tokens=300,
                ),
                timeout=8.0,
            )
        except Exception as err:  # noqa: BLE001 - 蒸馏失败静默（记忆是增强，不是主链路）
            log.warning("学情蒸馏失败：{}", err)
            return 0

        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return 0
        written = 0
        for raw in items[:2]:
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind") or "").strip()
            text = str(raw.get("text") or "").strip()
            if not text or kind not in KIND_LABELS:
                continue
            try:
                if self.store.observe(student_id, kind, text, subject=_subject_of(session), source="distill"):
                    written += 1
            except Exception as err:  # noqa: BLE001
                log.warning("写入蒸馏记忆失败：{}", err)
        if written:
            log.info("学情蒸馏写入 {} 条记忆（student={}）", written, student_id)
        return written


def _subject_of(session: Any) -> str:
    """取当前学科（「综合」视为不分学科，避免把通用记忆绑死在某一科上）。"""
    subject = str(getattr(session, "subject", "") or "").strip()
    return "" if subject in ("", "综合") else subject
