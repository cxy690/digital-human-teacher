"""回合裁决记录（L0 数据观测层）：把「这次回答到底好不好」落盘。

为什么需要它
------------
项目里**每个回合其实都已经产生了成败信号**，但它们全部只活在内存里：

* ``response.meta["json_degraded"]``      —— 模型写坏 JSON、板书整块丢弃
  （见 ``agents/teacher/agent.py`` 的 ``_compose_response``）；
* ``response.meta["quality"]``            —— 质检扣分与问题清单
  （见 ``agents/quality/agent.py`` 的 ``QualityReport``）；
* ``core.trace`` 每步的 ``ok`` / ``ms``   —— 哪一步抛了异常、链路卡在哪；
* 本回合总耗时与最终产物形态（有无板书 / 话术长度）。

这些信号随响应发给前端后就消失了：不落盘、不计数、不聚合。后果是
**系统能学习（经验在沉淀），但没人知道它学得好不好** —— 无法回答
「上周进的 12 条经验让质量升了还是降了」「哪个学科在退化」「改完提示词有没有变慢」。

本模块只做一件事：**把已经产生的信号抄一份到文件里**，作为上层
（健康画像 / 失败聚类 / 回归门禁）的唯一数据底座。它**不新增任何埋点**，
也不改判定以外的东西。

设计取舍
--------
1. **按天一个文件**（``verdict-YYYY-MM-DD.jsonl``）：窗口聚合天然就是「读最近 N 个
   文件」，不需要索引；单文件损坏也只损失一天。
2. **一行一条 JSONL**：只追加、永不整体重写；人可以直接 ``grep`` / ``tail`` 看，
   也可以喂给任何分析脚本。
3. **写失败必须静默**：观测层绝不能因为磁盘问题把课堂搞挂
   （与 :mod:`core.safety_audit` 同一取舍）。
4. **判定与采集分离**：判定规则全部集中在 :func:`judge`，改判定口径不用碰采集逻辑。

四维健康画像的对应关系（L1 会消费这里的字段）
--------------------------------------------
============  ================================  ==========================
图里的维度      本模块提供                        来源
============  ================================  ==========================
失败率          ``verdict != pass`` 的占比         判定规则（见 judge）
覆盖           ``by_subject`` / 经验条数          学科分布
staleness      需要 L1 结合经验文件算              ``data/teacher_experience``
last_gain      需要 L4 回归门禁给出                本模块提供前后两窗口的差
============  ================================  ==========================
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import settings

log = get_logger("core.verdict")

#: 裁决三态。``pass`` 干净、``degraded`` 能用但掉了东西、``failed`` 基本没交付。
VERDICTS = ("pass", "degraded", "failed")

#: 从调用链步骤名推断回合类型（``core.trace`` 的步骤名 → 中文场景名）。
#:
#: 为什么要推断而不是让各分支自报：回合类型是在 ``_handle_turn_builtin`` 内部由
#: 意图路由决定的，要让每个分支都自报就得改一圈调用点；而 ``core.trace`` 记录的
#: 函数名**本来就带着这个信息**（``_turn_teach`` / ``_turn_chitchat`` / ...）。
#: 按「命中即取」匹配，顺序即优先级。
_KIND_BY_STEP: tuple[tuple[str, str], ...] = (
    ("_turn_crisis", "crisis"),
    ("_turn_realtime_voice", "realtime"),
    ("_play_slides", "courseware"),
    ("on_quiz_request", "quiz"),
    ("evaluate_answer", "quiz_judge"),
    ("_turn_teach", "teach"),
    ("_turn_chitchat", "chitchat"),
    ("_turn_control", "control"),
    ("on_course_start", "greeting"),
    ("on_idle", "idle_nudge"),
    ("on_start", "session_start"),
)

#: 落盘时每步只保留这三个字段（调用链全量太占地方，且分析只用得到这些）。
_STEP_KEYS = ("file", "func", "desc", "ms", "ok")


@dataclass
class Window:
    """一次聚合的口径与结果（L1 健康画像的原始输入）。"""

    days: int
    since: str
    until: str
    total: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    #: 质检扣分（<100）的回合数与均分
    quality_scored: int = 0
    quality_sum: int = 0
    quality_issues: Counter = field(default_factory=Counter)
    #: 耗时分布（毫秒）
    latencies: list[float] = field(default_factory=list)
    #: 分组统计：学科 / 场景 -> 计数与失败数
    by_subject: dict[str, dict[str, int]] = field(default_factory=dict)
    by_kind: dict[str, dict[str, int]] = field(default_factory=dict)
    #: 步骤级：名字 -> {calls, errors, ms 列表}
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: 每条记录的补充失败原因（deg/failed 才记）
    reasons: Counter = field(default_factory=Counter)

    # ---------------------------------------------------------------- 输出
    def as_dict(self) -> dict[str, Any]:
        """渲染成可直接打印 / 返回接口的结构（百分比已算好）。"""
        total = max(self.total, 1)
        passed = self.counts.get("pass", 0)
        degraded = self.counts.get("degraded", 0)
        failed = self.counts.get("failed", 0)
        lat = sorted(self.latencies)

        def pct(x: float) -> float:
            return round(100.0 * x / total, 1)

        return {
            "window": {"days": self.days, "since": self.since, "until": self.until},
            "total": self.total,
            "verdicts": {"pass": passed, "degraded": degraded, "failed": failed},
            "rates_pct": {
                "clean": pct(passed),
                "degraded": pct(degraded),
                "failed": pct(failed),
                # 失败率把「降级」也算进去：降级回合学生是能感觉到掉了东西的
                "affected": pct(degraded + failed),
            },
            "quality": {
                "scored": self.quality_scored,
                "avg_score": round(self.quality_sum / self.quality_scored, 1)
                if self.quality_scored
                else None,
                "top_issues": self.quality_issues.most_common(5),
            },
            "latency_ms": {
                "avg": round(sum(lat) / len(lat), 0) if lat else None,
                "p50": _pct(lat, 0.50),
                "p95": _pct(lat, 0.95),
                "max": round(lat[-1], 0) if lat else None,
            },
            "by_subject": _group_view(self.by_subject),
            "by_kind": _group_view(self.by_kind),
            "slow_steps": _slow_steps(self.steps),
            "top_reasons": self.reasons.most_common(5),
        }


def _pct(sorted_vals: list[float], q: float) -> float | None:
    """分位数（``sorted_vals`` 必须已升序）。样本为空返回 None。"""
    if not sorted_vals:
        return None
    idx = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return round(sorted_vals[idx], 0)


def _group_view(groups: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    """把分组计数渲染成「总数 + 受影响率」，按受影响率降序（最差的排前面）。"""
    out: dict[str, dict[str, Any]] = {}
    for key, c in groups.items():
        n = c.get("total", 0)
        if not n:
            continue
        affected = c.get("degraded", 0) + c.get("failed", 0)
        out[key] = {
            "total": n,
            "affected": affected,
            "affected_pct": round(100.0 * affected / n, 1),
        }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["affected_pct"], reverse=True))


def _slow_steps(steps: dict[str, dict[str, Any]], top: int = 8) -> list[dict[str, Any]]:
    """最慢的步骤：按 p95 降序，p95 相同看调用次数（次数多的更值得优化）。"""
    rows: list[dict[str, Any]] = []
    for name, s in steps.items():
        ms = sorted(s.get("ms") or [])
        if not ms:
            continue
        rows.append({
            "name": name,
            "calls": s.get("calls", 0),
            "errors": s.get("errors", 0),
            "p50_ms": _pct(ms, 0.50),
            "p95_ms": _pct(ms, 0.95),
        })
    rows.sort(key=lambda r: (r["p95_ms"] or 0, r["calls"]), reverse=True)
    return rows[:top]


# ---------------------------------------------------------------------- 判定
def judge(
    *,
    response: Any | None,
    error: BaseException | None = None,
    steps: list[Any] | None = None,
) -> tuple[str, list[str]]:
    """判定一个回合的成败，返回 ``(verdict, reasons)``。

    **这是全部判定口径的唯一出处** —— 想调整「什么算降级」只改这里，
    采集与聚合都不用动。

    判定优先级（从严到松）::

        failed   抛异常 / 没有响应 / 话术与板书都空
        degraded JSON 降级、质检扣分、质检不通过、调用链里有步骤抛异常
        pass     以上都没有
    """
    reasons: list[str] = []

    if error is not None:
        return "failed", [f"异常：{type(error).__name__}"]
    if response is None:
        # 返回 None 是合法路径（例如实时语音不接管后回退、控制指令无回复），
        # 但对着落盘数据我们无从区分「本就不该有回复」和「链路断了」，
        # 所以记为 degraded 而不是 failed，靠 kind 在下游区分。
        return "degraded", ["无响应产物"]

    meta = getattr(response, "meta", None) or {}
    speech = str(getattr(response, "speech", "") or "").strip()
    board = list(getattr(response, "board_ops", None) or [])

    if not speech and not board:
        return "failed", ["空回复（无话术且无板书）"]

    if meta.get("json_degraded"):
        reasons.append("JSON 降级（板书整块丢弃）")

    # 模型调用失败后走的是兜底话术（学生能听、但内容不是真回答）—— 必须记为降级。
    # 不记的话，这条链路就会「学生听到了兜底、观测层看到一切正常」，
    # L0/L3 从此看不见这类上游故障（比不兜底更糟：故障被藏起来了）。
    llm_error = str(meta.get("llm_error") or "").strip()
    if llm_error:
        reasons.append(f"模型调用失败：{llm_error}")

    quality = meta.get("quality") or {}
    if isinstance(quality, dict):
        score = quality.get("score")
        if isinstance(score, int) and score < 100:
            reasons.append(f"质检扣分 {score}")
        if quality.get("passed") is False:
            reasons.append("质检未通过")
        for issue in quality.get("issues") or []:
            reasons.append(f"质检问题：{str(issue)[:40]}")

    bad_steps = [s for s in (steps or []) if not getattr(s, "ok", True)]
    if bad_steps:
        names = ", ".join(sorted({getattr(s, "label", "?") for s in bad_steps})[:3])
        reasons.append(f"步骤异常：{names}")

    return ("degraded" if reasons else "pass"), reasons


def infer_kind(steps: list[Any] | None) -> str:
    """从调用链步骤名推断回合场景（见 :data:`_KIND_BY_STEP`）。推不出返回 ``"other"``。"""
    labels = " ".join(str(getattr(s, "label", "") or getattr(s, "func", "")) for s in (steps or []))
    for needle, kind in _KIND_BY_STEP:
        if needle in labels:
            return kind
    return "other"


def _compact_steps(steps: list[Any] | None) -> list[dict[str, Any]]:
    """把 ``core.trace.Step`` 压成落盘用的精简结构。"""
    out: list[dict[str, Any]] = []
    for s in steps or []:
        out.append({
            "file": str(getattr(s, "file", "") or ""),
            "func": str(getattr(s, "func", "") or ""),
            "desc": str(getattr(s, "desc", "") or ""),
            "ms": round(float(getattr(s, "ms", 0.0) or 0.0), 1),
            "ok": bool(getattr(s, "ok", True)),
        })
    return out


# ---------------------------------------------------------------------- 存储
class VerdictStore:
    """回合裁决的 JSONL 存储（按天分文件，只追加）。

    ``directory`` 留空时**每次读写都现取** ``settings.verdict_dir``（而不是在构造时
    固定）。这样测试可以用 monkeypatch 换目录，生产也能靠 .env 改路径 —— 模块级
    单例 :data:`verdict_store` 是在 import 期构造的，若在构造时固定目录，
    之后任何配置变更都会「看起来改了其实没生效」。
    """

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        self._fixed_dir = Path(directory) if directory is not None else None
        self._lock = threading.Lock()
        self._last_cleanup = 0.0

    @property
    def dir(self) -> Path:
        """落盘目录（现取现建，保证首次写入前目录一定存在）。"""
        base = self._fixed_dir or Path(settings.verdict_dir)
        base.mkdir(parents=True, exist_ok=True)
        return base

    # ---------------------------------------------------------------- 写
    def path_for(self, day: str) -> Path:
        """某一天的落盘文件（``verdict-YYYY-MM-DD.jsonl``）。"""
        return self.dir / f"verdict-{day}.jsonl"

    def record(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """追加一条裁决；关闭或写失败时返回 ``None``（**绝不抛**）。"""
        if not settings.verdict_enable:
            return None
        now = time.time()
        entry = {
            "ts": now,
            "ts_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            **payload,
        }
        try:
            line = json.dumps(entry, ensure_ascii=False)
            with self._lock, self.path_for(entry["day"]).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except (OSError, TypeError, ValueError) as err:
            log.warning("裁决落盘失败（不影响主链路）：{}", err)
            return None
        self._cleanup_if_due(now)
        return entry

    # ---------------------------------------------------------------- 读
    def days(self, days: int) -> list[str]:
        """最近 ``days`` 天的日期串（含今天），升序。"""
        today = datetime.now()
        return [
            (today - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(days - 1, -1, -1)
        ]

    def read(self, days: int, limit: int = 0) -> list[dict[str, Any]]:
        """读最近 ``days`` 天的记录（升序）。单行损坏只跳过该行，不放弃整个文件。"""
        out: list[dict[str, Any]] = []
        for day in self.days(days):
            path = self.path_for(day)
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as err:
                log.warning("裁决读取失败（{}）：{}", day, err)
                continue
            for raw in lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(json.loads(raw))
                except json.JSONDecodeError:
                    # 写了一半被 kill 的行不应让整天的数据不可读
                    log.debug("跳过损坏的裁决行：{}", raw[:60])
        return out[-limit:] if limit > 0 else out

    def _cleanup_if_due(self, now: float) -> None:
        """按 ``verdict_keep_days`` 清理过期文件（每小时最多做一次）。"""
        if now - self._last_cleanup < 3600:
            return
        self._last_cleanup = now
        keep = max(int(settings.verdict_keep_days or 0), 1)
        cutoff = (datetime.now() - timedelta(days=keep)).strftime("%Y-%m-%d")
        try:
            for path in self.dir.glob("verdict-*.jsonl"):
                day = path.stem.removeprefix("verdict-")
                if day < cutoff:
                    path.unlink(missing_ok=True)
                    log.info("清理过期裁决文件：{}", path.name)
        except OSError as err:  # noqa: BLE001 - 清理失败无所谓
            log.debug("裁决清理失败：{}", err)

    # ---------------------------------------------------------------- 聚合
    def summarize(self, days: int | None = None) -> Window:
        """把时间窗内的记录聚合成健康画像（L1 的原始输入，此处只做统计不做判断）。"""
        days = int(days if days is not None else settings.verdict_window_days)
        day_list = self.days(days)
        win = Window(days=days, since=day_list[0], until=day_list[-1])
        win.counts = {v: 0 for v in VERDICTS}

        for rec in self.read(days):
            win.total += 1
            verdict = str(rec.get("verdict") or "degraded")
            win.counts[verdict] = win.counts.get(verdict, 0) + 1

            for reason in rec.get("reasons") or []:
                win.reasons[str(reason)] += 1

            quality = rec.get("quality") or {}
            if isinstance(quality, dict) and isinstance(quality.get("score"), int):
                win.quality_scored += 1
                win.quality_sum += int(quality["score"])
                for issue in quality.get("issues") or []:
                    win.quality_issues[str(issue)[:40]] += 1

            ms = rec.get("elapsed_ms")
            if isinstance(ms, (int, float)) and ms > 0:
                win.latencies.append(float(ms))

            subject = str(rec.get("subject") or "未知")
            kind = str(rec.get("kind") or "other")
            for bucket, key in ((win.by_subject, subject), (win.by_kind, kind)):
                cell = bucket.setdefault(key, {"total": 0, "degraded": 0, "failed": 0})
                cell["total"] += 1
                if verdict in ("degraded", "failed"):
                    cell[verdict] += 1

            for s in rec.get("steps") or []:
                name = str(s.get("func") or s.get("file") or "?")
                if s.get("desc"):
                    name = f"{s['desc']}  {s.get('file', '')}·{s.get('func', '')}"
                cell = win.steps.setdefault(name, {"calls": 0, "errors": 0, "ms": []})
                cell["calls"] += 1
                if not s.get("ok", True):
                    cell["errors"] += 1
                ms_step = s.get("ms")
                if isinstance(ms_step, (int, float)):
                    cell["ms"].append(float(ms_step))
        return win


#: 进程内共享单例（与 ``core.safety_audit.safety_audit`` 同一用法）：
#: 共用一份目录与同一把锁，避免各会话各写一份导致交错。
verdict_store = VerdictStore()


# ---------------------------------------------------------------------- 采集
def record_turn_verdict(
    *,
    session: Any,
    turn: Any,
    response: Any | None,
    elapsed_ms: float,
    kind: str | None = None,
    error: BaseException | None = None,
) -> dict[str, Any] | None:
    """记录一个回合的裁决 —— **采集的唯一入口**，由 ``core/supervisor.handle_turn`` 调用。

    为什么放在 ``handle_turn``：它是**所有分支的唯一收口点**（危机 / 实时语音 /
    LangGraph / 内建各返回一次），且本身就是 ``@traced(root=True)`` 的回合根，
    调用链信息在这里现成可用。放在任何更靠内的分支都会漏掉别的分支。

    任何异常都在这里被吞掉：**观测层绝不能影响课堂**。
    """
    if not settings.verdict_enable:
        return None
    try:
        steps = current_steps()
        verdict, reasons = judge(response=response, error=error, steps=steps)
        meta = (getattr(response, "meta", None) or {}) if response is not None else {}
        quality = meta.get("quality") if isinstance(meta.get("quality"), dict) else {}
        board_ops = list(getattr(response, "board_ops", None) or []) if response is not None else []
        speech = str(getattr(response, "speech", "") or "") if response is not None else ""

        payload: dict[str, Any] = {
            "day": time.strftime("%Y-%m-%d"),
            "verdict": verdict,
            "reasons": reasons,
            "kind": kind or infer_kind(steps),
            "session_id": str(getattr(session, "session_id", "") or ""),
            "student_id": str(getattr(session, "student_id", "") or ""),
            "subject": str(getattr(session, "subject", "") or ""),
            "teacher": str(getattr(session, "teacher", "") or ""),
            "course_active": bool(getattr(session, "course_active", False)),
            "elapsed_ms": round(float(elapsed_ms), 1),
            "speech_len": len(speech),
            "board_ops": len(board_ops),
            "question": str(getattr(turn, "text", "") or "")[:120],
            "json_degraded": bool(meta.get("json_degraded")),
            "quality": {
                "passed": quality.get("passed"),
                "score": quality.get("score"),
                "issues": list(quality.get("issues") or [])[:5],
                "checked_by": quality.get("checked_by"),
            } if quality else None,
            "steps_count": len(steps),
        }
        if error is not None:
            payload["error"] = f"{type(error).__name__}: {error}"[:200]
        if settings.verdict_keep_steps:
            payload["steps"] = _compact_steps(steps)
        return verdict_store.record(payload)
    except Exception as err:  # noqa: BLE001 - 观测层任何异常都必须被吞掉
        log.warning("裁决采集失败（不影响主链路）：{}", err)
        return None


def current_steps() -> list[Any]:
    """取当前回合的调用链步骤；``core.trace`` 不可用时返回空列表。

    单独包一层是为了「观测层可选」：即便 trace 关了（``TRACE_ENABLE=false``），
    verdict 依然能靠 ``meta`` 里的信号给出判定，只是少了耗时维度。
    """
    try:
        from core.trace import current_steps as _cs  # noqa: PLC0415 - 避免无谓的导入期耦合

        return list(_cs())
    except Exception:  # noqa: BLE001 - 拿不到调用链不该影响记录
        return []


def summarize(days: int | None = None) -> dict[str, Any]:
    """当前窗口的健康摘要（供 CLI 与 ``/api/health/verdict`` 直接用）。"""
    return verdict_store.summarize(days).as_dict()


__all__ = [
    "VERDICTS",
    "VerdictStore",
    "Window",
    "infer_kind",
    "judge",
    "record_turn_verdict",
    "summarize",
    "verdict_store",
]
