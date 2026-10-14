"""教师经验条目：Skill 自我更新沉淀下来的「可复用教学经验」。

只描述**经验**，不含任何人格 / 教学法本体 —— 这是刻意的边界：
skill 的人格与底线（``core/skills/teacher_personality``）永远由人写、由人改，
自我更新只允许在这层之上**追加经验**，不碰本体。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

#: 经验类型 -> 注入提示词时的分组标题
KIND_LABELS: dict[str, str] = {
    "pitfall": "学生易错点",
    "analogy": "有效类比",
    "must_cover": "必须补讲的点",
    "confusion": "学生常见困惑",
}

#: 默认类型（模型给的类型不认识时归到这里）
DEFAULT_KIND = "pitfall"

#: 单条经验的长度上下限：太短没信息量，太长会把提示词撑爆
MIN_TEXT_CHARS = 4
MAX_TEXT_CHARS = 60

#: 先验强度上限：``hits`` 最多折算成这么多次「先验观测」。
#:
#: 为什么要封顶：``hits`` 记的是「这条经验被复盘重复提到过几次」，它衡量的是
#: **发现频率**，不是**帮助大小**。不封顶的话，一条被反复提到却毫无作用的经验
#: 会靠巨大的 ``hits`` 把先验压得极硬，实测战绩（``pulls/wins``）永远翻不了盘 ——
#: 那就又回到「靠嘴硬排序」了。封顶让它最多只是「起步时稍微占点便宜」。
PRIOR_HITS_CAP = 5


@dataclass
class ExperienceItem:
    """一条教师经验。"""

    id: str
    teacher_key: str
    kind: str
    text: str
    hits: int = 1                  # 被复盘命中过几次（越多次越可信）
    confidence: float = 0.5        # 模型自评置信度
    status: str = "active"         # active / retired（停用即不再注入，可回滚）
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    sources: list[str] = field(default_factory=list)   # 来源摘要（最近若干次，便于人工追溯）

    # ---- L2 进化决策层：这条经验「被用了之后到底有没有帮上忙」----
    # 这三个字段是**实测战绩**，与上面的 hits/confidence（发现频率 + 模型自评）正交。
    # 没有它们就永远只能按「说得响不响」排序 —— 那正是 L2 要解决的问题。
    pulls: int = 0                 # 被注入过几个回合（bandit 的分母）
    wins: int = 0                  # 其中裁决为 pass 的回合数（bandit 的分子）
    last_gain: float = 0.0         # 最近一次的收益（1 = 帮上忙，0 = 没帮上）

    @property
    def score(self) -> float:
        """**静态**排序分：命中越多、模型越确信，越靠前。

        ⚠️ 它衡量的是「这条经验被**反复发现**过几次」，**不是「它有没有帮上忙」**：
        一条被反复提到、注入后却毫无作用的经验，score 照样最高。
        真正决定「每轮注入哪几条」的是 :meth:`sample`（按实测战绩试错的 Thompson 采样）；
        这里只保留给①人工查看 ②bandit 关闭时的回退排序。
        """
        return self.hits * max(0.0, min(1.0, self.confidence))

    @property
    def active(self) -> bool:
        return self.status == "active"

    # ---------------------------------------------------------------- 收益（L2）
    def beta_params(self) -> tuple[float, float]:
        """Beta(α, β) 超参：先验由「发现次数 × 模型置信度」给，后验由实测战绩累积。

        ``hits`` 决定先验的**强度**（封顶见 :data:`PRIOR_HITS_CAP`），
        ``confidence`` 决定先验的**方向**；``pulls/wins`` 是实测观测。
        三者合起来：刚进场时听模型的，用得多了以后听数据的。
        """
        prior_n = min(max(0, int(self.hits)), PRIOR_HITS_CAP)
        prior_mean = max(0.0, min(1.0, float(self.confidence)))
        alpha = 1.0 + max(0, int(self.wins)) + prior_n * prior_mean
        beta = 1.0 + max(0, int(self.pulls) - int(self.wins)) + prior_n * (1.0 - prior_mean)
        return alpha, beta

    @property
    def gain_rate(self) -> float:
        """实测有效率（Beta 后验均值）：**人工看「哪条真有用」就看这个**。"""
        alpha, beta = self.beta_params()
        total = alpha + beta
        return round(alpha / total, 3) if total > 0 else 0.0

    def sample(self, rng: random.Random | None = None) -> float:
        """按后验做一次采样（bandit 定序用的「此刻的乐观估计」）。

        为什么是**采样**而不是排序：一条从没被注入过的经验（``pulls=0``）
        和一个被证明有效 20 次的经验，谁该上？纯排序答不了这个问题 —— 采样可以：
        ``pulls`` 少的后验**宽**，采样值偶尔会很高，于是它有机会被试到（**探索**）；
        战绩好的后验窄且均值高，多数时候会被选中（**利用**）。
        这就是「把名额从静态排序改成按实际收益试错」的全部机制。
        """
        alpha, beta = self.beta_params()
        return (rng or random).betavariate(alpha, beta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "teacher_key": self.teacher_key,
            "kind": self.kind,
            "text": self.text,
            "hits": self.hits,
            "confidence": round(float(self.confidence), 3),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": list(self.sources),
            # L2：实测战绩（老文件没有这几个字段，读盘时按 0 兜底）
            "pulls": self.pulls,
            "wins": self.wins,
            "last_gain": round(float(self.last_gain), 3),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "ExperienceItem | None":
        """从 JSONL 行还原；字段缺失／类型不对返回 ``None``（坏行跳过，不炸整个文件）。"""
        if not isinstance(raw, dict):
            return None
        item_id = str(raw.get("id") or "").strip()
        text = str(raw.get("text") or "").strip()
        teacher_key = str(raw.get("teacher_key") or "").strip()
        if not item_id or not text or not teacher_key:
            return None
        kind = str(raw.get("kind") or DEFAULT_KIND).strip() or DEFAULT_KIND
        sources = [str(s) for s in (raw.get("sources") or []) if str(s).strip()]
        try:
            hits = max(0, int(raw.get("hits") or 0))
            confidence = float(raw.get("confidence") or 0.0)
            created = float(raw.get("created_at") or time.time())
            updated = float(raw.get("updated_at") or created)
            # L2 战绩字段：老经验文件里没有这几个键，按 0 兜底即可（等价于「还没试过」），
            # 这正是我们想要的初始状态 —— 不该因为文件是旧的就把条目判为坏行丢掉。
            pulls = max(0, int(raw.get("pulls") or 0))
            wins = max(0, min(pulls, int(raw.get("wins") or 0)))
            last_gain = float(raw.get("last_gain") or 0.0)
        except (TypeError, ValueError):
            return None
        return cls(
            id=item_id,
            teacher_key=teacher_key,
            kind=kind if kind in KIND_LABELS else DEFAULT_KIND,
            text=text,
            hits=hits,
            confidence=confidence,
            status=str(raw.get("status") or "active"),
            created_at=created,
            updated_at=updated,
            sources=sources,
            pulls=pulls,
            wins=wins,
            last_gain=last_gain,
        )
