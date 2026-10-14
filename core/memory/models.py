"""学情记忆条目：跨会话记住「这位学生哪儿不会、习惯怎样」。

与「教师经验」（``core/skills/experience``）的分工，是一组正交的边界：

| | 教师经验 | 学情记忆（本模块） |
|---|---|---|
| 归属 | 按**老师**存（一位老师一份） | 按**学生**存（一位学生一份） |
| 内容 | 老师怎么讲更有效（易错点、有效类比） | 学生哪儿不会、习惯怎样（错题、薄弱点、偏好） |
| 生效 | 注入该老师的提示词 | 只有这位学生上课时才注入 |

为什么强调这条边界：经验可以跨学生复用（它描述的是教学法），
记忆**绝不能**跨学生复用（同名同班的两个孩子是两个人）。因此记忆按 ``student_id``
分文件存放，读盘时不存在「串到别人画像」的可能。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

#: 记忆类型 -> 注入提示词时的分组标题
KIND_LABELS: dict[str, str] = {
    "mistake": "做错的题",
    "weak_point": "薄弱知识点",
    "preference": "学习偏好",
    "fact": "学情事实",
}

#: 默认类型（模型给的类型不认识时归到这里：最保守，仅作背景信息）
DEFAULT_KIND = "fact"

#: 各类型的默认重要度：错题最该被老师看到，偏好只是锦上添花
KIND_WEIGHTS: dict[str, float] = {
    "mistake": 0.9,
    "weak_point": 0.8,
    "fact": 0.5,
    "preference": 0.4,
}

#: 单条记忆的长度上下限：太短没信息量，太长会把提示词撑爆
MIN_TEXT_CHARS = 2
MAX_TEXT_CHARS = 80

#: 记忆的「半衰期」（天）：久不出现的旧记忆逐渐让位给近期表现
RECENCY_HALF_LIFE_DAYS = 30.0


@dataclass
class MemoryItem:
    """一条学情记忆。"""

    id: str
    student_id: str
    kind: str
    text: str
    subject: str = ""              # 归属学科（空 = 不分学科，任何学科都注入）
    hits: int = 1                  # 被观察到几次（重复出现 = 更可信）
    weight: float = 0.5            # 重要度（错题 > 薄弱点 > 事实 > 偏好）
    status: str = "active"         # active / retired（停用即不再注入，可回滚）
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    sources: list[str] = field(default_factory=list)   # 来源摘要（便于人工追溯）

    @property
    def score(self) -> float:
        """排序分：出现越多、越重要、越近，越优先注入提示词。"""
        age_days = max(0.0, (time.time() - self.updated_at) / 86400.0)
        decay = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
        return self.hits * max(0.0, min(1.0, self.weight)) * decay

    @property
    def active(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "kind": self.kind,
            "text": self.text,
            "subject": self.subject,
            "hits": self.hits,
            "weight": round(float(self.weight), 3),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": list(self.sources),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "MemoryItem | None":
        """从 JSONL 行还原；字段缺失 / 类型不对返回 ``None``（坏行跳过，不炸整个文件）。"""
        if not isinstance(raw, dict):
            return None
        item_id = str(raw.get("id") or "").strip()
        text = str(raw.get("text") or "").strip()
        student_id = str(raw.get("student_id") or "").strip()
        if not item_id or not text or not student_id:
            return None
        kind = str(raw.get("kind") or DEFAULT_KIND).strip() or DEFAULT_KIND
        if kind not in KIND_LABELS:
            kind = DEFAULT_KIND
        try:
            hits = max(0, int(raw.get("hits") or 0))
            weight = float(raw.get("weight", KIND_WEIGHTS.get(kind, 0.5)))
            created = float(raw.get("created_at") or time.time())
            updated = float(raw.get("updated_at") or created)
        except (TypeError, ValueError):
            return None
        return cls(
            id=item_id,
            student_id=student_id,
            kind=kind,
            text=text,
            subject=str(raw.get("subject") or "").strip(),
            hits=hits,
            weight=weight,
            status=str(raw.get("status") or "active"),
            created_at=created,
            updated_at=updated,
            sources=[str(s) for s in (raw.get("sources") or []) if str(s).strip()],
        )
