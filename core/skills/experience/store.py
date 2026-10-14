"""教师经验存储：append-only JSONL（可去重、可计数、可回滚）。

为什么是 append-only JSONL，而不是就地改写 JSON：
* **每次变更追加一行快照**（同 id 后写的生效），本身就是一份审计日志；
* 停用 / 回滚只是再追加一条 ``status`` 变更，不需要写「撤销」逻辑；
* 追加写不怕写一半崩掉，坏行读盘时跳过即可，也不会并发写冲突；
* 纯文本，人可以打开看、手工删掉明显不对的一条。

为什么放 ``data/teacher_experience/`` 而不是 ``data/teachers/<key>/``：
经验是**运行时学习产物**（与 data/cache、data/raw 同类），不属于「教师素材」。
``scripts/sync_cloud.py`` 已把它排除在同步之外 —— 否则本地一次同步就会覆盖掉
云端线上课堂学到的经验（反之亦然）。每个环境各自学习、各自积累。
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
import uuid
from collections.abc import Sequence
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import settings
from core.skills.experience.models import (
    DEFAULT_KIND,
    KIND_LABELS,
    MAX_TEXT_CHARS,
    MIN_TEXT_CHARS,
    ExperienceItem,
)

log = get_logger("core.skills.experience.store")

#: 相似度阈值：归一化后相似到这种程度就当作「同一条经验」，累加命中次数而不是新增
SIMILARITY_THRESHOLD = 0.85

#: 每条经验保留的来源摘要条数（人工追溯用，多了没意义）
MAX_SOURCES = 5

_NORMALIZE_RE = re.compile(r"[\s，。！？；：、,.!?;:（）()“”\"'·—\-]")


def normalize_text(text: str) -> str:
    """归一化：去空白与标点、统一小写，用于去重比较。"""
    return _NORMALIZE_RE.sub("", (text or "")).lower()


class ExperienceStore:
    """某位（或全部）老师的经验表。"""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        max_items: int | None = None,
        similarity: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self._root = Path(root) if root is not None else None
        self._max_items = max_items
        self.similarity = similarity
        self._cache: dict[str, tuple[float, list[ExperienceItem]]] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 路径
    @property
    def root(self) -> Path:
        """经验目录（默认取配置 ``skill_experience_dir``，便于测试改写）。"""
        return self._root or Path(settings.skill_experience_dir)

    @property
    def max_items(self) -> int:
        """生效条数上限；**``0`` = 不限**（默认，永不自动淘汰）。

        注意这里不能用 ``self._max_items or 默认值``：显式传 ``0`` 表示「不限」，
        用 ``or`` 会被当成「没传」而回落到配置值，把「不限」变成「限额」。
        """
        configured = settings.skill_experience_max_items if self._max_items is None else self._max_items
        return int(configured)

    def path_for(self, teacher_key: str) -> Path:
        return self.root / f"{teacher_key}.jsonl"

    # ---------------------------------------------------------------- 读
    def load(self, teacher_key: str) -> list[ExperienceItem]:
        """读该老师的全部经验（按 mtime 缓存，同一轮答疑里不重复读盘）。"""
        key = (teacher_key or "").strip()
        if not key:
            return []
        path = self.path_for(key)
        cache_key = self._cache_key(key)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            self._cache[cache_key] = (0.0, [])
            return []
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        items = self._read_file(path)
        self._cache[cache_key] = (mtime, items)
        return items

    def active(self, teacher_key: str, *, limit: int | None = None) -> list[ExperienceItem]:
        """生效中的经验，按 ``score`` 降序（命中多、置信高的在前）。"""
        items = [it for it in self.load(teacher_key) if it.active]
        items.sort(key=lambda it: (-it.score, it.created_at))
        return items[:limit] if limit and limit > 0 else items

    def all_items(self, teacher_key: str) -> list[ExperienceItem]:
        """含已停用的全部条目（人工排查用）。"""
        items = list(self.load(teacher_key))
        items.sort(key=lambda it: (-it.score, it.created_at))
        return items

    def select(
        self, teacher_key: str, *, limit: int, rng: random.Random | None = None
    ) -> list[ExperienceItem]:
        """选出**这一轮要注入**的若干条经验（L2 bandit 定序）。

        与 :meth:`active` 只差一件事，但那就是 L2 的全部意义：
        这里按**实测收益**的后验**采样**排序（Thompson），而不是按 ``hits × confidence``
        静态排序。前者会把名额分一些给「还没试过、但看起来有希望」的条目去试错，
        后者永远只喂「被提到次数最多」的那几条 —— 而「被反复提到」≠「帮上了忙」。

        bandit 关掉（``SKILL_EXPERIENCE_BANDIT_ENABLE=false``）时退回静态排序，
        行为与加装前完全一致。
        """
        items = [it for it in self.load(teacher_key) if it.active]
        if limit <= 0 or not items:
            return []
        if not settings.skill_experience_bandit_enable:
            items.sort(key=lambda it: (-it.score, it.created_at))
            return items[:limit]
        # 每条**恰采一次**再排序：少了这一步就会退化成「随机洗牌」，
        # 而不是「按后验的乐观程度排序」。
        picks = sorted(items, key=lambda it: (-it.sample(rng), it.created_at))
        return picks[:limit]

    def record_outcome(
        self, teacher_key: str, item_ids: Sequence[str], *, gain: float
    ) -> int:
        """回写一次实测结果：这批被注入的经验，本回合**帮上忙了没有**。

        ``gain`` ∈ [0, 1]：``1`` = 本回合裁决为 ``pass``（干净交付），``0`` = 有降级或失败。
        返回实际更新的条数。

        为什么按**整批**记账而不是逐条归因：一个回合注入了 N 条经验，本来就没法知道
        是哪一条起了作用 —— 这种归因歧义是此类系统固有的（bandit 的「批量拉动」）。
        处理方式是「都记一次」：噪声会随 ``pulls`` 增长被平均掉，而真正有用的经验
        会稳定地累积高 ``wins``，后验均值自然抬起来，下一轮更容易被选中。
        """
        key = (teacher_key or "").strip()
        wanted = {str(i).strip() for i in (item_ids or []) if str(i).strip()}
        if not key or not wanted:
            return 0
        try:
            value = max(0.0, min(1.0, float(gain)))
        except (TypeError, ValueError):
            return 0
        updated = 0
        with self._lock:
            for item in self.load(key):
                if item.id not in wanted:
                    continue
                item.pulls += 1
                if value >= 0.5:
                    item.wins += 1
                item.last_gain = value
                item.updated_at = time.time()
                self._append(key, item)
                updated += 1
            if updated:
                self._invalidate(key)
        return updated

    def list_teachers(self) -> list[str]:
        """已有经验文件的老师 key 列表。"""
        try:
            return sorted(p.stem for p in self.root.glob("*.jsonl"))
        except OSError:
            return []

    # ---------------------------------------------------------------- 写
    def observe(
        self,
        teacher_key: str,
        kind: str,
        text: str,
        *,
        source: str = "",
        confidence: float = 0.5,
    ) -> ExperienceItem | None:
        """记录一条经验：已有相似条目则**累加命中**，否则新增。

        返回落盘的条目；文本过短／过长／为空返回 ``None``（视为无价值的噪音）。
        """
        key = (teacher_key or "").strip()
        note = (text or "").strip()
        if not key or not note:
            return None
        if not (MIN_TEXT_CHARS <= len(note) <= MAX_TEXT_CHARS):
            log.debug("跳过长度不合适的经验（{} 字）：{}", len(note), note[:30])
            return None
        normalized = normalize_text(note)
        if not normalized:
            return None
        kind = kind if kind in KIND_LABELS else DEFAULT_KIND
        confidence = max(0.0, min(1.0, float(confidence)))

        with self._lock:
            items = list(self.load(key))
            matched = self._find_similar(items, normalized)
            if matched is not None:
                matched.hits += 1
                matched.confidence = max(matched.confidence, confidence)
                matched.updated_at = time.time()
                if source and source not in matched.sources:
                    matched.sources = (matched.sources + [source])[-MAX_SOURCES:]
                self._append(key, matched)
                self._invalidate(key)
                return matched

            item = ExperienceItem(
                id=f"exp-{uuid.uuid4().hex[:12]}",
                teacher_key=key,
                kind=kind,
                text=note,
                confidence=confidence,
                sources=[source] if source else [],
            )
            self._evict_if_needed(key, items + [item])
            self._append(key, item)
            self._invalidate(key)
            return item

    def retire(self, teacher_key: str, item_id: str) -> bool:
        """停用一条经验（不再注入提示词）—— 这就是「回滚」：坏经验出问题的止损动作。"""
        return self._set_status(teacher_key, item_id, "retired")

    def restore(self, teacher_key: str, item_id: str) -> bool:
        """把停用的经验恢复生效（回滚的反向操作）。"""
        return self._set_status(teacher_key, item_id, "active")

    def stats(self) -> dict[str, Any]:
        """全局概况（接口 / 运维查看用）。"""
        teachers = self.list_teachers()
        total = sum(len(self.load(k)) for k in teachers)
        active = sum(len(self.active(k)) for k in teachers)
        return {
            "dir": str(self.root),
            "teachers": len(teachers),
            "items": total,
            "active": active,
            "max_items_per_teacher": self.max_items,
        }

    # ---------------------------------------------------------------- 内部
    def _set_status(self, teacher_key: str, item_id: str, status: str) -> bool:
        key = (teacher_key or "").strip()
        target = (item_id or "").strip()
        if not key or not target:
            return False
        with self._lock:
            for item in self.load(key):
                if item.id == target:
                    item.status = status
                    item.updated_at = time.time()
                    self._append(key, item)
                    self._invalidate(key)
                    return True
        return False

    @staticmethod
    def _find_similar(items: list[ExperienceItem], normalized: str) -> ExperienceItem | None:
        """在现有条目里找语义上「同一条」的经验（归一化相同或高度相似）。"""
        for item in items:
            other = normalize_text(item.text)
            if not other:
                continue
            if other == normalized:
                return item
            if SequenceMatcher(None, other, normalized).ratio() >= SIMILARITY_THRESHOLD:
                return item
        return None

    def _evict_if_needed(self, teacher_key: str, items: list[ExperienceItem]) -> None:
        """超过条数上限时，停用**分数最低**的条目，给新经验腾位置。

        ``max_items = 0``（默认）表示**不限**：直接返回，不做任何淘汰 ——
        经验是老师长期教学攒下来的，系统不该背着自己把学过的东西停掉。
        即使显式设了上限，这里也只「停用」不「删除」：条目仍在文件里，
        人随时能 ``restore`` 或用接口看一眼。
        """
        if self.max_items <= 0:
            return
        active = [it for it in items if it.active]
        overflow = len(active) - self.max_items
        if overflow <= 0:
            return
        for victim in sorted(active, key=lambda it: (it.score, it.updated_at))[:overflow]:
            victim.status = "retired"
            victim.updated_at = time.time()
            self._append(teacher_key, victim)
            log.info("经验条数超限，停用低分条目：{} {}", teacher_key, victim.text[:20])

    def _read_file(self, path: Path) -> list[ExperienceItem]:
        """按 id 折叠快照（后写的生效），坏行跳过。"""
        latest: dict[str, ExperienceItem] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("经验文件坏行已跳过：{}:{}", path.name, line_no)
                        continue
                    item = ExperienceItem.from_dict(raw)
                    if item is not None:
                        latest[item.id] = item
        except OSError as err:
            log.warning("读取经验文件失败（{}）：{}", path, err)
            return []
        return list(latest.values())

    def _append(self, teacher_key: str, item: ExperienceItem) -> None:
        """追加一行快照（同 id 后写的生效）。"""
        path = self.path_for(teacher_key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        except OSError as err:
            log.warning("写入经验失败（{}）：{}", path, err)

    def _invalidate(self, teacher_key: str) -> None:
        self._cache.pop(self._cache_key(teacher_key), None)

    def _cache_key(self, teacher_key: str) -> str:
        """缓存键带上目录：root 可由配置改写（测试／多环境），只用 teacher_key 会串数据。"""
        return f"{self.root}::{teacher_key}"


@lru_cache(maxsize=1)
def get_experience_store() -> ExperienceStore:
    """进程级默认经验表（root 延迟到真正读写时从配置取，便于测试改写）。"""
    return ExperienceStore()
