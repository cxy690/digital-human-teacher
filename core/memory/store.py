"""学情记忆存储：按学生分文件的 append-only JSONL（可去重、可计数、可删除）。

为什么沿用「append-only JSONL」而不是数据库
------------------------------------------
与教师经验（``core/skills/experience/store.py``）同源的理由：每行一次快照、
自带审计、追加写不怕写一半崩掉、坏行读盘时跳过、纯文本可人工检查与手工修正。
教学链路的记忆读写都在**课堂上**发生，不能因为存储抖动而卡住老师开口。

为什么按学生分文件
------------------
记忆是**学生画像**，必须按 ``student_id`` 物理隔离：同名的两个孩子是两个人，
共用一个文件迟早串号（这也是最不能出错的一类错误）。
路径：``<MEMORY_DIR>/students/<student_id>.jsonl``。

隐私与合规（面向未成年人，硬性）
--------------------------------
* :meth:`MemoryStore.forget` 一次性**物理删除**某位学生的全部记忆（家长要求删除时用）；
* 目录默认在 ``data/memory``（运行时产物），``scripts/sync_cloud.py`` 与经验目录一样
  按环境隔离，本地与云端各自积累，不会互相覆盖。
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import settings
from core.memory.models import (
    DEFAULT_KIND,
    KIND_LABELS,
    KIND_WEIGHTS,
    MAX_TEXT_CHARS,
    MIN_TEXT_CHARS,
    MemoryItem,
)

log = get_logger("core.memory.store")

#: 相似度阈值：归一化后相似到这种程度就当作「同一条记忆」，累加命中次数而不是新增
SIMILARITY_THRESHOLD = 0.85

#: 每条记忆保留的来源摘要条数（人工追溯用）
MAX_SOURCES = 5

#: 关键词检索时，相关性对总分的放大权重
RELEVANCE_WEIGHT = 2.0

_NORMALIZE_RE = re.compile(r"[\s，。！？；：、,.!?;:（）()“”\"'·—\-]")
#: 字符 bigram：与 rag/retriever.py 同一套零依赖做法（记忆文本很短，够用）
_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


def normalize_text(text: str) -> str:
    """归一化：去空白与标点、统一小写，用于去重比较。"""
    return _NORMALIZE_RE.sub("", (text or "")).lower()


def bigrams(text: str) -> set[str]:
    """字符 bigram 集合（中文短文本相似度的零依赖做法）。"""
    tokens = _TOKEN_RE.findall(text or "")
    grams: set[str] = set()
    for token in tokens:
        if len(token) == 1:
            grams.add(token)
            continue
        grams.update(token[i:i + 2] for i in range(len(token) - 1))
    return grams


def _overlap(left: set[str], right: set[str]) -> float:
    """Jaccard 相似度；任一为空时返回 0。"""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class MemoryStore:
    """某位学生的学情记忆表（``root`` 可改写，便于测试与多环境）。"""

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
        self._cache: dict[str, tuple[float, list[MemoryItem]]] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 路径
    @property
    def root(self) -> Path:
        """记忆根目录（默认取配置 ``memory_dir``）。"""
        return self._root or Path(settings.memory_dir)

    @property
    def max_items(self) -> int:
        """生效条数上限；**``0`` = 不限**（默认，永不自动淘汰）。

        与教师经验同理：自动学到的学情不该由系统自己丢掉；
        注入侧另有 ``MEMORY_INJECT_LIMIT`` 控制每次带几条进提示词。
        不能用 ``self._max_items or 默认值``：显式传 0 = 不限，会被 ``or`` 吃掉。
        """
        configured = settings.memory_max_items if self._max_items is None else self._max_items
        return int(configured)

    def path_for(self, student_id: str) -> Path:
        # 学生 id 可能含路径分隔符（外部系统传入），统一收敛成安全的文件名
        safe = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]", "_", str(student_id or "").strip()) or "anonymous"
        return self.root / "students" / f"{safe}.jsonl"

    # ---------------------------------------------------------------- 读
    def load(self, student_id: str) -> list[MemoryItem]:
        """读该学生的全部记忆（按 mtime 缓存，同一回合里不重复读盘）。"""
        key = (student_id or "").strip()
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

    def active(self, student_id: str, *, subject: str = "", limit: int | None = None) -> list[MemoryItem]:
        """生效中的记忆，按 ``score`` 降序。

        ``subject`` 给定时：只保留「该学科」与「不分学科」的记忆 —— 数学课不该看到
        语文的错题（不同学科的老师不是同一节课）。
        """
        items = [it for it in self.load(student_id) if it.active]
        if subject:
            items = [it for it in items if not it.subject or it.subject == subject]
        items.sort(key=lambda it: (-it.score, it.created_at))
        return items[:limit] if limit and limit > 0 else items

    def search(
        self,
        student_id: str,
        query: str = "",
        *,
        subject: str = "",
        limit: int | None = None,
    ) -> list[MemoryItem]:
        """相关记忆检索：``score``（次数 × 重要度 × 时间衰减）+ 文本相关性。

        为什么不是纯向量检索：记忆文本只有几十字、条数也少（默认上限 200 条），
        字符 bigram 的 Jaccard 相关性足够，且与 ``rag/retriever.py`` 同一套零依赖做法 ——
        少一个必须常驻的向量库，就少一个会拖垮课堂的依赖。
        """
        items = self.active(student_id, subject=subject)
        query_grams = bigrams(query)
        if not query_grams:
            return items[:limit] if limit and limit > 0 else items
        scored = [
            (item.score + _overlap(query_grams, bigrams(item.text)) * RELEVANCE_WEIGHT, item)
            for item in items
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].created_at))
        ranked = [item for _, item in scored]
        return ranked[:limit] if limit and limit > 0 else ranked

    def list_students(self) -> list[str]:
        """已有记忆文件的学生 id 列表（运维 / 合规检查用）。"""
        try:
            return sorted(p.stem for p in (self.root / "students").glob("*.jsonl"))
        except OSError:
            return []

    # ---------------------------------------------------------------- 写
    def observe(
        self,
        student_id: str,
        kind: str,
        text: str,
        *,
        subject: str = "",
        weight: float | None = None,
        source: str = "",
    ) -> MemoryItem | None:
        """记录一条记忆：已有相似条目则**累加命中**，否则新增。

        返回落盘的条目；文本过短／过长／为空返回 ``None``（视为无价值的噪音）。
        """
        key = (student_id or "").strip()
        note = (text or "").strip()[:MAX_TEXT_CHARS]
        if not key or not note:
            return None
        if len(note) < MIN_TEXT_CHARS:
            return None
        normalized = normalize_text(note)
        if not normalized:
            return None
        kind = kind if kind in KIND_LABELS else DEFAULT_KIND
        value = KIND_WEIGHTS.get(kind, 0.5) if weight is None else float(weight)
        weight = max(0.0, min(1.0, value))

        with self._lock:
            items = list(self.load(key))
            matched = self._find_similar(items, normalized)
            if matched is not None:
                matched.hits += 1
                matched.weight = max(matched.weight, weight)
                matched.updated_at = time.time()
                if source and source not in matched.sources:
                    matched.sources = (matched.sources + [source])[-MAX_SOURCES:]
                self._append(key, matched)
                self._invalidate(key)
                return matched

            item = MemoryItem(
                id=f"mem-{uuid.uuid4().hex[:12]}",
                student_id=key,
                kind=kind,
                text=note,
                subject=(subject or "").strip(),
                weight=weight,
                sources=[source] if source else [],
            )
            self._evict_if_needed(key, items + [item])
            self._append(key, item)
            self._invalidate(key)
            return item

    def retire(self, student_id: str, item_id: str) -> bool:
        """停用一条记忆（不再注入提示词）—— 记错时的止损动作。"""
        return self._set_status(student_id, item_id, "retired")

    def restore(self, student_id: str, item_id: str) -> bool:
        """把停用的记忆恢复生效。"""
        return self._set_status(student_id, item_id, "active")

    def forget(self, student_id: str) -> bool:
        """**物理删除**某位学生的全部记忆（家长要求删除 / 数据清理）。

        合规要求：删除请求不能只是「标记停用」——标记后数据仍在盘上。
        """
        key = (student_id or "").strip()
        if not key:
            return False
        path = self.path_for(key)
        with self._lock:
            self._invalidate(key)
            try:
                path.unlink()
                return True
            except FileNotFoundError:
                return False
            except OSError as err:
                log.warning("删除学情记忆失败（{}）：{}", path, err)
                return False

    def stats(self) -> dict[str, Any]:
        """全局概况（接口 / 运维查看用）。"""
        students = self.list_students()
        total = sum(len(self.load(s)) for s in students)
        active = sum(len(self.active(s)) for s in students)
        return {
            "dir": str(self.root),
            "students": len(students),
            "items": total,
            "active": active,
            "max_items_per_student": self.max_items,
        }

    # ---------------------------------------------------------------- 内部
    def _set_status(self, student_id: str, item_id: str, status: str) -> bool:
        key = (student_id or "").strip()
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
    def _find_similar(items: list[MemoryItem], normalized: str) -> MemoryItem | None:
        """在现有条目里找「同一条」记忆（归一化相同或高度相似）。"""
        for item in items:
            if not item.active:
                continue
            other = normalize_text(item.text)
            if not other:
                continue
            if other == normalized:
                return item
            if SequenceMatcher(None, other, normalized).ratio() >= SIMILARITY_THRESHOLD:
                return item
        return None

    def _evict_if_needed(self, student_id: str, items: list[MemoryItem]) -> None:
        """超过条数上限时，停用**分数最低**的条目，给新记忆腾位置。

        ``max_items = 0``（默认）表示**不限**：直接返回，不做任何淘汰 ——
        学情是自动积累的画像，系统不该自己把「这位学生以前哪儿不会」丢掉。
        即使显式设了上限，这里也只「停用」不「删除」；真正要清空只有
        :meth:`forget`（人工触发，合规用）。
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
            self._append(student_id, victim)
            log.info("学情记忆条数超限，停用低分条目：{} {}", student_id, victim.text[:20])

    def _read_file(self, path: Path) -> list[MemoryItem]:
        """按 id 折叠快照（后写的生效），坏行跳过。"""
        latest: dict[str, MemoryItem] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("学情记忆文件坏行已跳过：{}:{}", path.name, line_no)
                        continue
                    item = MemoryItem.from_dict(raw)
                    if item is not None:
                        latest[item.id] = item
        except OSError as err:
            log.warning("读取学情记忆失败（{}）：{}", path, err)
            return []
        return list(latest.values())

    def _append(self, student_id: str, item: MemoryItem) -> None:
        """追加一行快照（同 id 后写的生效）。"""
        path = self.path_for(student_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        except OSError as err:
            log.warning("写入学情记忆失败（{}）：{}", path, err)

    def _invalidate(self, student_id: str) -> None:
        self._cache.pop(self._cache_key(student_id), None)

    def _cache_key(self, student_id: str) -> str:
        """缓存键带上目录：root 可由配置改写（测试／多环境），只用 id 会串数据。"""
        return f"{self.root}::{student_id}"


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    """进程级默认记忆表（root 延迟到真正读写时从配置取，便于测试改写）。"""
    return MemoryStore()


def clear_memory_dir(root: str | Path | None = None) -> None:
    """清空记忆目录（测试 / 数据清理用）。"""
    target = Path(root) if root is not None else Path(settings.memory_dir)
    try:
        shutil.rmtree(target, ignore_errors=True)
    except OSError as err:  # pragma: no cover - 极端情况
        log.warning("清空学情记忆目录失败（{}）：{}", target, err)
