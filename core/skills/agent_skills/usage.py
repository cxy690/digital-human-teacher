"""技能调用战绩：把「这个技能被用了之后到底有没有帮上忙」落盘。

为什么需要它
------------
技能暴露给模型之后，唯一能判断它值不值得继续存在的依据是**实战战绩**：
一个每轮都被调、但调完回合照样降级的技能，是在浪费名额与 token；
一个从没被调过的技能，可能是描述写得不好（模型不知道何时用它），也可能是真的没必要。

本模块只做采集与聚合（与 :mod:`core.verdict` 同一取舍）： append-only JSONL、
按技能名分文件、写失败静默、绝不因为「记账」把课堂搞挂。

与经验层（L2）的关系
--------------------
* 经验层记的是**一条经验**被注入后的收益（``data/teacher_experience/``）；
* 本层记的是**一个技能**被调用后的收益（``data/skill_usage/``）。
两者共用同一份裁决信号（``pass`` = 帮上忙），但记账对象不同 —— 归因歧义的处理
也一致：「整批同记、噪声随 pulls 增长被平均掉」。
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import settings

log = get_logger("core.skills.agent_skills.usage")


@dataclass
class SkillUsage:
    """一个技能的累计战绩（文件里最后一条快照生效）。"""

    name: str
    pulls: int = 0                  # 被调用过几个回合
    wins: int = 0                   # 其中裁决为 pass 的回合数
    last_gain: float = 0.0          # 最近一次收益（1 = 帮上忙，0 = 没帮上）
    status: str = "active"          # active / disabled（人工停用，可恢复）
    updated_at: float = 0.0

    @property
    def active(self) -> bool:
        return self.status == "active"

    @property
    def gain_rate(self) -> float:
        """原始有效率（``wins / pulls``）；没上过场返回 0.0。"""
        return round(self.wins / self.pulls, 3) if self.pulls > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pulls": self.pulls,
            "wins": self.wins,
            "last_gain": round(float(self.last_gain), 3),
            "status": self.status,
            "updated_at": self.updated_at,
            "gain_rate": self.gain_rate,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "SkillUsage | None":
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name") or "").strip()
        if not name:
            return None
        try:
            pulls = max(0, int(raw.get("pulls") or 0))
            wins = max(0, min(pulls, int(raw.get("wins") or 0)))
            return cls(
                name=name,
                pulls=pulls,
                wins=wins,
                last_gain=float(raw.get("last_gain") or 0.0),
                status=str(raw.get("status") or "active"),
                updated_at=float(raw.get("updated_at") or 0.0),
            )
        except (TypeError, ValueError):
            return None


class SkillUsageStore:
    """技能战绩存储（按技能名分文件，只追加）。

    ``directory`` 留空时每次读写现取 ``settings.agent_skill_usage_dir``（与
    :class:`core.verdict.VerdictStore` 同一取舍）：测试可换目录，线上可改 ``.env``。
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self._fixed_dir = Path(directory) if directory is not None else None
        self._lock = threading.Lock()

    @property
    def dir(self) -> Path:
        base = self._fixed_dir or Path(settings.agent_skill_usage_dir)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def path_for(self, name: str) -> Path:
        return self.dir / f"{str(name or '').strip()}.jsonl"

    # ---------------------------------------------------------------- 写
    def record_usage(self, names: list[str] | tuple[str, ...], *, gain: float) -> int:
        """把本回合被调用的技能各记一次收益；返回更新的技能数。

        整批同记（与经验层一致）：一个回合调了 N 个技能，没法知道是哪个起了作用；
        噪声会随 ``pulls`` 增长被平均掉，真正有用的技能会稳定累积高 ``wins``。
        """
        wanted = [str(n).strip() for n in (names or []) if str(n).strip()]
        if not wanted:
            return 0
        try:
            value = max(0.0, min(1.0, float(gain)))
        except (TypeError, ValueError):
            return 0
        updated = 0
        with self._lock:
            for name in dict.fromkeys(wanted):     # 保序去重：同回合调两次只记一次
                usage = self.load(name)
                usage.pulls += 1
                if value >= 0.5:
                    usage.wins += 1
                usage.last_gain = value
                usage.updated_at = time.time()
                try:
                    with self.path_for(name).open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(usage.to_dict(), ensure_ascii=False) + "\n")
                    updated += 1
                except OSError as err:
                    # 记账失败绝不能影响课堂（与 verdict / safety_audit 同一取舍）
                    log.warning("技能战绩落盘失败（{}）：{}", name, err)
        return updated

    def set_status(self, name: str, status: str) -> bool:
        """人工停用 / 恢复一个技能（止损入口，等价于经验的 retire/restore）。"""
        key = str(name or "").strip()
        if not key or status not in ("active", "disabled"):
            return False
        usage = self.load(key)
        usage.status = status
        usage.updated_at = time.time()
        try:
            with self.path_for(key).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(usage.to_dict(), ensure_ascii=False) + "\n")
            return True
        except OSError as err:
            log.warning("技能状态写入失败（{}）：{}", key, err)
            return False

    # ---------------------------------------------------------------- 读
    def load(self, name: str) -> SkillUsage:
        """读一个技能的战绩（最后一条合法快照；没有则返回零值）。"""
        key = str(name or "").strip()
        path = self.path_for(key) if key else self.dir / ".none"
        usage = SkillUsage(name=key)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return usage
        for raw in reversed(lines):
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = SkillUsage.from_dict(json.loads(raw))
            except json.JSONDecodeError:
                continue     # 写了一半被 kill 的行：跳过，不让整个技能的战绩不可读
            if parsed is not None:
                return parsed
        return usage

    def names(self) -> list[str]:
        """已有战绩文件的技能名。"""
        try:
            return sorted(p.stem for p in self.dir.glob("*.jsonl"))
        except OSError:
            return []

    # ---------------------------------------------------------------- 抽检
    def review_queue(self, *, limit: int = 10, min_pulls: int | None = None) -> list[dict[str, Any]]:
        """按风险降序返回「最该人工看一眼」的技能（与经验抽检台同一立场）。

        **这个闸门不阻断调用**（除非显式打开 ``AGENT_SKILL_AUTO_RETIRE``）：
        它做的是降低抽检成本，不是把技能停下来。三个风险信号都指向一个具体动作：

        ====================  =========================================  ==================
        信号                  含义                                        人的动作
        ====================  =========================================  ==================
        有效率明显偏低        用了 N 次、干净交付不到三成                  改措辞或停用
        影响面大              已经调用过很多次，一旦有问题波及很多回合      确认描述无误
        从未上过场            注册了但模型从来不调（描述可能没写清）          看描述要不要改
        ====================  =========================================  ==================
        """
        floor = int(min_pulls if min_pulls is not None else settings.agent_skill_min_pulls)
        rows: list[dict[str, Any]] = []
        for name in self.names():
            usage = self.load(name)
            if not usage.active:
                continue
            risk = 0.0
            reasons: list[str] = []
            if usage.pulls >= floor and usage.gain_rate < settings.agent_skill_min_gain_rate:
                risk += 0.6
                reasons.append(
                    f"用了 {usage.pulls} 次、只有 {usage.wins} 次干净交付"
                    f"（有效率 {usage.gain_rate:.2f}）—— 改措辞或停用"
                )
            if usage.pulls >= max(floor, 10) * 2:
                risk += 0.25
                reasons.append(f"已经调用过 {usage.pulls} 次（影响面大）—— 确认描述无误")
            if usage.pulls == 0:
                risk += 0.2
                reasons.append("注册了但一次都没被调用过 —— 看 description 有没有写清何时用")
            if risk <= 0:
                continue
            rows.append({
                "name": name,
                "risk": round(risk, 2),
                "reasons": reasons,
                "pulls": usage.pulls,
                "wins": usage.wins,
                "gain_rate": usage.gain_rate,
            })
        rows.sort(key=lambda r: (-r["risk"], -r["pulls"], r["name"]))
        return rows[: max(1, int(limit))]


#: 进程内共享单例（与 ``core.verdict.verdict_store`` 同一用法）
skill_usage_store = SkillUsageStore()


def get_skill_usage_store() -> SkillUsageStore:
    """返回进程级技能战绩存储（共用一份目录与同一把锁，避免各写各的）。"""
    return skill_usage_store
