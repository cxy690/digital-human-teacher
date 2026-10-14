"""Skill 自我更新的复盘器：把一次答疑里**可复用**的经验提炼出来沉淀。

复盘的核心判据是「**换个学生、换个数字仍然成立**」：

* ✅ 「学生常把移项符号搞错」「讲浮力时用『船为什么会漂』类比很有效」
* ❌ 「这道题的答案是 x=5」「刚才讲了第 3 步」—— 换个题就作废，沉淀下来只会污染提示词

所以提示词里明确要求模型只输出**通用经验**，并且允许它一条都不输出（宁可空手而归，
也不要为凑数编经验）。产出还会再过一遍长度／置信度／条数过滤。

**边界**：本模块只写「经验」（``core/skills/experience``），
永远不会去改人格 Skill（``core/skills/teacher_personality``）的提示词本体。
"""
from __future__ import annotations

import json
import re
from typing import Any

from config.logging import get_logger
from config.settings import settings
from core.skills.experience.models import (
    KIND_LABELS,
    MAX_TEXT_CHARS,
    ExperienceItem,
)
from core.skills.experience.store import ExperienceStore, get_experience_store
from models.base import BaseLLMClient
from models.llm import get_llm

log = get_logger("core.skills.experience.reflector")

_PROMPT = """你是一位资深教研员，正在给「{subject}」老师做课后复盘。

请从下面这次真实的答疑里，提炼**可复用的教学经验**（换一道题、换一个学生仍然成立）。

只输出 JSON：
{{"items": [{{"kind": "pitfall|analogy|must_cover|confusion",
            "text": "一句话经验，不超过{max_chars}字",
            "confidence": 0.0~1.0}}]}}

四类经验的含义：
- pitfall：学生在这类题上容易错的地方；
- analogy：讲这个概念时有效的类比或生活例子；
- must_cover：这类题必须补讲、否则学生一定卡住的点；
- confusion：学生容易产生困惑或误解的地方。

硬性要求：
1. **只写通用经验，不写本题的具体数字、答案、题号**（如"答案是 x=5"这种一律不要写）；
2. 不要复述题目内容，不要写空泛套话（如"要耐心讲解"）；
3. 最多 {max_items} 条；**没有值得沉淀的就不输出**，items 给空数组，绝不为了凑数编造；
4. 不得违反教学底线（不嘲讽学生、不给危险操作建议、不超纲）；
5. 只输出 JSON，不要额外解释。

【本次答疑】
学生提问：{question}
老师讲解：{answer}
涉及知识点：{points}
老师标出的易错点：{pitfalls}
"""

_JSON_RE = re.compile(r"\{[\s\S]*\}")


class SkillReflector:
    """课后复盘：从一次答疑里提炼经验并写入经验表。"""

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        store: ExperienceStore | None = None,
        *,
        max_items_per_turn: int | None = None,
        min_confidence: float | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._max_items = max_items_per_turn
        self._min_confidence = min_confidence

    @property
    def store(self) -> ExperienceStore:
        return self._store or get_experience_store()

    @property
    def max_items(self) -> int:
        return int(self._max_items or settings.skill_reflect_max_items)

    @property
    def min_confidence(self) -> float:
        return float(
            self._min_confidence
            if self._min_confidence is not None
            else settings.skill_reflect_min_confidence
        )

    async def reflect(
        self,
        *,
        teacher_key: str,
        subject: str = "",
        question: str = "",
        answer: str = "",
        points: list[str] | None = None,
        pitfalls: list[str] | None = None,
    ) -> list[ExperienceItem]:
        """复盘一次答疑；返回本次**新增或强化**的经验条目（失败返回空表，绝不抛）。"""
        teacher_key = (teacher_key or "").strip()
        if not teacher_key or not (answer or "").strip():
            return []

        raw = ""
        try:
            raw = await (self._llm or get_llm()).chat(
                [{"role": "user", "content": self._build_prompt(
                    subject=subject, question=question, answer=answer,
                    points=points or [], pitfalls=pitfalls or [],
                )}],
                temperature=0.2,
                max_tokens=400,
            )
        except Exception as err:  # noqa: BLE001 - 复盘失败绝不能影响课堂
            log.warning("经验复盘调用失败（忽略）：{}", err)
            return []

        saved: list[ExperienceItem] = []
        for kind, text, confidence in self._parse(raw):
            if confidence < self.min_confidence:
                log.debug("跳过低置信经验（{}）：{}", confidence, text[:20])
                continue
            item = self.store.observe(
                teacher_key,
                kind,
                text,
                source=f"{subject or '综合'}｜{question[:30]}",
                confidence=confidence,
            )
            if item is not None:
                saved.append(item)
        return saved

    # ---------------------------------------------------------------- 内部
    def _build_prompt(
        self,
        *,
        subject: str,
        question: str,
        answer: str,
        points: list[str],
        pitfalls: list[str],
    ) -> str:
        return _PROMPT.format(
            subject=subject or "综合",
            max_chars=MAX_TEXT_CHARS,
            max_items=self.max_items,
            question=(question or "（无）")[:200],
            answer=(answer or "（无）")[:600],
            points="、".join(points) or "（无）",
            pitfalls="、".join(pitfalls) or "（无）",
        )

    def _parse(self, text: str) -> list[tuple[str, str, float]]:
        """解析模型输出 -> ``[(kind, text, confidence)]``；最多 ``max_items`` 条。"""
        match = _JSON_RE.search(text or "")
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, dict):
            return []
        raw_items = data.get("items")
        if not isinstance(raw_items, list):
            return []

        parsed: list[tuple[str, str, float]] = []
        for raw in raw_items[: self.max_items]:
            if not isinstance(raw, dict):
                continue
            note = str(raw.get("text") or "").strip()
            kind = str(raw.get("kind") or "").strip()
            if not note or kind not in KIND_LABELS:
                continue
            try:
                confidence = float(raw.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            parsed.append((kind, note, max(0.0, min(1.0, confidence))))
        return parsed
