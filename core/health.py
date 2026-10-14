"""L1 健康诊断层：判断「系统在变好还是变坏」，而不是只把数字列出来。

与 L0 的分工
------------
* **L0**（:mod:`core.verdict`）只做**采集与聚合** —— 把信号落盘、算成比例，**不做判断**；
* **L1**（本模块）做**判断** —— 什么时候该报警、哪一维在退化、什么信号先动。

为什么必须分开：判定口径（多少算退化）会随产品阶段变，但原始数据不能跟着变。
分开之后，调阈值、加维度都不用碰落盘格式。

四维健康画像
------------
============  ================================  ==========================
维度          指标                              数据源
============  ================================  ==========================
**失败率**    受影响率 / 失败率 / 降级率        ``core/verdict`` 的裁决
**覆盖**      学科覆盖 / 场景覆盖 / 日覆盖       裁决的分布
**staleness** 经验·记忆最后更新、停滞的学科      ``data/teacher_experience``
                                                ``data/memory``
**last_gain** 最新评测 vs 基线                   ``evals/baseline.json``
============  ================================  ==========================

先行退化信号（这一节是 L1 的真正价值）
--------------------------------------
**失败率是滞后指标** —— 等它涨起来，学生已经受影响了。真正该盯的是它的上游：

* **JSON 降级率** / **质检均分** —— 先于「板书丢失 / 讲得变差」被学生察觉；
* **p95 耗时** —— 先于超时与卡顿；
* **调用链步骤异常率** —— 先于整个回合失败。

所以本模块把这些单列出来做窗口对比，而不是只报一个总分。

判定规则（三条，都是为了避免「狼来了」）
----------------------------------------
1. **样本量下限**：窗口内少于 :data:`MIN_SAMPLES` 条时只报事实、**不下判断** ——
   早上只跑了 3 条就说「失败率 33%」是纯噪声；
2. **相对与绝对同时看**：相对涨幅大但绝对值极小的（0.5% → 1%）不报；
3. **影响面优先**：某学科显著差于全局（> 1.5 倍且样本足够）比全局微涨更值得先看。
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import settings

log = get_logger("core.health")

#: 窗口内记录数少于它时只报事实、不下判断（小样本的比率没有意义）
MIN_SAMPLES = 20
#: 某学科/场景的样本数下限（低于它不做「比全局差」的判断）
MIN_GROUP_SAMPLES = 8

#: 触发 warn 的相对 / 绝对变化
WARN_REL = 0.50
WARN_ABS = 0.10
#: 触发 critical 的相对 / 绝对变化
CRIT_REL = 1.00
CRIT_ABS = 0.25
#: 某学科受影响率高于全局这么多倍 → 单独提示
SUBJECT_FACTOR = 1.5
#: 经验/记忆超过这么多小时没更新 → staleness 提示
STALE_HOURS = 72.0

#: 指标定义：``(key, 中文名, 坏的方向, 单位)``。
#: 单位决定阈值是按「百分点」还是按「相对比例」比 —— 比例型看相对更合理，
#: 均分与耗时的绝对值本身就稳定，看相对即可。
METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("affected_ratio", "受影响率", "up", "ratio"),
    ("failed_ratio", "失败率", "up", "ratio"),
    ("degraded_ratio", "降级率", "up", "ratio"),
    ("json_degraded_ratio", "JSON 降级率", "up", "ratio"),
    ("step_error_ratio", "步骤异常率", "up", "ratio"),
    ("quality_avg", "质检均分", "down", "score"),
    ("latency_p95", "耗时 p95", "up", "ms"),
)


@dataclass
class Metric:
    """一个指标在两个窗口里的取值。"""

    key: str
    label: str
    unit: str
    bad_when: str
    value: float | None = None
    baseline: float | None = None
    sample: int = 0

    @property
    def delta(self) -> float | None:
        """绝对变化（本次 - 基线）。"""
        if self.value is None or self.baseline is None:
            return None
        return self.value - self.baseline

    @property
    def rel(self) -> float | None:
        """相对变化（本次 / 基线 - 1）；基线为 0 时返回 None（无法比）。"""
        if self.value is None or self.baseline is None or self.baseline == 0:
            return None
        return self.value / self.baseline - 1.0

    def worse(self) -> float:
        """变差程度（正数 = 变差，越大越差；0 = 没变差）。已按指标方向归一。"""
        d = self.delta
        if d is None:
            return 0.0
        return d if self.bad_when == "up" else -d

    def fmt(self, v: float | None) -> str:
        if v is None:
            return "—"
        if self.unit == "ratio":
            return f"{v * 100:.1f}%"
        if self.unit == "ms":
            return f"{v:.0f}ms"
        return f"{v:.1f}"


@dataclass
class Finding:
    """一条诊断结论。"""

    level: str          # ok | warn | critical
    dimension: str      # 失败率 / 覆盖 / staleness / last_gain / 先行信号 / 耗时
    title: str
    evidence: str = ""
    advice: str = ""


@dataclass
class Diagnosis:
    """一次完整诊断的结果。"""

    window_days: int
    baseline_days: int
    samples: int
    baseline_samples: int
    metrics: list[Metric] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    staleness: dict[str, Any] = field(default_factory=dict)
    last_gain: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    @property
    def worst_level(self) -> str:
        for lv in ("critical", "warn"):
            if any(f.level == lv for f in self.findings):
                return lv
        return "ok"


# ---------------------------------------------------------------------- 采集
def collect(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """从一批裁决记录里算出 :data:`METRICS` 的取值（**只算，不判断**）。"""
    n = len(records)
    if not n:
        return {k: None for k, *_ in METRICS}

    def ratio(pred) -> float:
        return sum(1 for r in records if pred(r)) / n

    quality = [
        int((r.get("quality") or {}).get("score"))
        for r in records
        if isinstance((r.get("quality") or {}).get("score"), int)
    ]
    lat = sorted(
        float(r["elapsed_ms"]) for r in records
        if isinstance(r.get("elapsed_ms"), (int, float)) and r["elapsed_ms"] > 0
    )

    def p95(vals: list[float]) -> float | None:
        if not vals:
            return None
        return vals[min(int(len(vals) * 0.95), len(vals) - 1)]

    # 步骤异常率：所有记录的所有步骤里，ok=False 的占比（细粒度于「整回合失败」）
    steps_total = steps_bad = 0
    for r in records:
        for s in r.get("steps") or []:
            steps_total += 1
            if not s.get("ok", True):
                steps_bad += 1

    return {
        "affected_ratio": ratio(lambda r: r.get("verdict") in ("degraded", "failed")),
        "failed_ratio": ratio(lambda r: r.get("verdict") == "failed"),
        "degraded_ratio": ratio(lambda r: r.get("verdict") == "degraded"),
        "json_degraded_ratio": ratio(lambda r: bool(r.get("json_degraded"))),
        "step_error_ratio": (steps_bad / steps_total) if steps_total else 0.0,
        "quality_avg": statistics.fmean(quality) if quality else None,
        "latency_p95": p95(lat),
    }


def _window(records: list[dict[str, Any]], days: int, *, offset: int) -> list[dict[str, Any]]:
    """取「最近 days 天」或「再往前 offset 个窗口」的记录。

    按 ``day`` 字段分桶而不是按 ts —— 与 L0 的按天分文件保持一致，
    也让「前一个等长窗口」这件事变得直观。
    """
    days_all = sorted({str(r.get("day") or "") for r in records if r.get("day")})
    if offset:
        days_all = days_all[:-days] if days <= len(days_all) else []
    picked = set(days_all[-days:])
    return [r for r in records if str(r.get("day") or "") in picked]


# ---------------------------------------------------------------------- 覆盖
def coverage_of(records: list[dict[str, Any]], window_days: int) -> dict[str, Any]:
    """覆盖度：有多少学科 / 场景 / 天数**真的产生了数据**。

    为什么这是独立一维：一个学科完全没有数据，和它「表现很差」是两回事 ——
    前者是**盲区**（可能压根没被用过、或链路断了），后者才是质量问题。
    只看失败率会把「零数据」误判成「零问题」。
    """
    subjects = {str(r.get("subject") or "未知") for r in records}
    kinds = {str(r.get("kind") or "other") for r in records}
    days = {str(r.get("day") or "") for r in records if r.get("day")}

    # 已知学科总数（配置里声明的），用来算覆盖率
    try:
        from config.subjects import SUBJECTS  # noqa: PLC0415 - 可选依赖

        known = {str(s.get("name") or "") for s in SUBJECTS}
    except Exception:  # noqa: BLE001 - 配置不可读就不算覆盖率
        known = set()

    return {
        "subjects": sorted(subjects),
        "subject_count": len(subjects),
        "known_subject_count": len(known),
        "missing_subjects": sorted(known - subjects) if known else [],
        "kinds": sorted(kinds),
        "days_with_data": len(days),
        "window_days": window_days,
    }


# ---------------------------------------------------------------------- staleness
def staleness_of() -> dict[str, Any]:
    """学习链路的新鲜度：经验 / 记忆最后更新于何时、有没有停滞的学科。

    为什么这一维重要：**系统「有在学习」和「还在学习」不是一回事**。
    复盘可能因为额度、异常、配置变更而静默停掉，此时失败率不会有任何变化 ——
    只有 staleness 能看出来。
    """
    out: dict[str, Any] = {"experience": {}, "memory": {}}

    exp_dir = Path(settings.skill_experience_dir)
    if exp_dir.exists():
        files = sorted(exp_dir.glob("*.jsonl"))
        newest = 0.0
        per_teacher: dict[str, int] = {}
        for p in files:
            try:
                items = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except OSError:
                continue
            per_teacher[p.stem] = len(items)
            newest = max(newest, p.stat().st_mtime)
        out["experience"] = {
            "exists": True,
            "teachers": len(files),
            "items": sum(per_teacher.values()),
            "per_teacher": per_teacher,
            "age_hours": round((time.time() - newest) / 3600, 1) if newest else None,
        }
    else:
        out["experience"] = {"exists": False, "teachers": 0, "items": 0, "age_hours": None}

    mem_dir = Path(settings.memory_dir)
    if mem_dir.exists():
        newest = 0.0
        students = 0
        items = 0
        for p in mem_dir.rglob("*.jsonl"):
            students += 1
            try:
                lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except OSError:
                continue
            items += len(lines)
            newest = max(newest, p.stat().st_mtime)
        out["memory"] = {
            "exists": True, "students": students, "items": items,
            "age_hours": round((time.time() - newest) / 3600, 1) if newest else None,
        }
    else:
        out["memory"] = {"exists": False, "students": 0, "items": 0, "age_hours": None}

    return out


# ---------------------------------------------------------------------- last_gain
def last_gain_of(evals_dir: Path | None = None) -> dict[str, Any]:
    """最近一次 L4 评测相对基线是升是降（读 ``evals/``，没有就返回空）。"""
    base_dir = evals_dir or Path("evals")
    base_file = base_dir / "baseline.json"
    runs_dir = base_dir / "runs"
    if not base_file.exists() or not runs_dir.exists():
        return {"available": False}

    try:
        base = json.loads(base_file.read_text(encoding="utf-8"))
        runs = sorted(
            (p for p in runs_dir.iterdir() if (p / "result.json").exists()),
            key=lambda p: p.name,
        )
        if not runs:
            return {"available": False}
        latest = json.loads((runs[-1] / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        log.debug("读取评测基线失败：{}", err)
        return {"available": False}

    base_total = float(base.get("summary", {}).get("total") or 0.0)
    cur_total = float(latest.get("summary", {}).get("total") or 0.0)
    base_cases = int(base.get("summary", {}).get("cases") or 0)
    cur_cases = int(latest.get("summary", {}).get("cases") or 0)
    # 可比性：题数不同的两次评测**分数不可比**。
    # 这不是吹毛求疵 —— 用 `--only 某题` 跑一次单题验证会产出 1 题的结果，
    # 若拿它去比 12 题的基线，会得出「+18 分，一切正常」这种完全错误的乐观结论。
    comparable = base_cases > 0 and base_cases == cur_cases
    return {
        "available": True,
        "comparable": comparable,
        "baseline_cases": base_cases,
        "latest_cases": cur_cases,
        "baseline_total": base_total,
        "baseline_stamp": base.get("stamp", ""),
        "latest_total": cur_total,
        "latest_stamp": latest.get("stamp", ""),
        "delta": round(cur_total - base_total, 1) if comparable else None,
    }


# ---------------------------------------------------------------------- 判定
def _judge_metric(m: Metric) -> Finding | None:
    """给单个指标出结论；没变差或样本不足时返回 ``None``。"""
    if m.value is None or m.baseline is None:
        return None
    worse = m.worse()
    if worse <= 0:
        return None

    rel = m.rel
    # 绝对变化的口径（这个划分是踩过坑才定下来的）：
    #   * ratio（0~1）→ 百分点；score（0~100）→ 分值。两者数值量级接近，共用一组阈值。
    #     **评分型必须看绝对值**：质检 95 → 60 分掉的是 35 分（很严重），
    #     但相对只有 37%，落在 50% 的 warn 阈值之下 —— 只看相对会一声不吭。
    #   * ms 只按相对比：基线 3s 还是 6s 都可能合理，「翻倍」才说明问题，
    #     绝对毫秒没有通用标准。
    has_abs = m.unit in ("ratio", "score")
    abs_delta = abs(m.delta or 0.0) if has_abs else None

    # 能走到这里说明**方向上是变差**（见上面的 worse() 判断），所以相对变化也取绝对值：
    # 均分下跌时 rel 是负数，与正阈值比大小永远不达标。
    rel_bad = abs(rel) if rel is not None else None

    level = ""
    if (rel_bad is not None and rel_bad >= CRIT_REL) or (abs_delta is not None and abs_delta >= CRIT_ABS):
        level = "critical"
    elif (rel_bad is not None and rel_bad >= WARN_REL) or (abs_delta is not None and abs_delta >= WARN_ABS):
        level = "warn"
    if not level:
        return None
    # 噪音过滤：绝对变化太小就别报 —— 0.5% → 1% 相对翻倍但只有 0.5 个百分点；
    # 质检 95 → 88 分同理，属于正常波动范围。
    if has_abs and abs_delta is not None and abs_delta < WARN_ABS:
        return None

    change = f"{m.fmt(m.baseline)} → {m.fmt(m.value)}"
    if m.unit == "ratio" and abs_delta is not None:
        change += f"（{'+' if (m.delta or 0) > 0 else ''}{(m.delta or 0) * 100:.1f}pp）"
    elif m.unit == "score" and abs_delta is not None:
        change += f"（{'+' if (m.delta or 0) > 0 else ''}{(m.delta or 0):.1f} 分）"
    elif rel is not None:
        change += f"（{rel * 100:+.0f}%）"

    dimension = "先行信号" if m.key in (
        "json_degraded_ratio", "quality_avg", "step_error_ratio",
    ) else "失败率"
    return Finding(
        level=level,
        dimension=dimension,
        title=f"{m.label}变差：{change}",
        evidence=f"本次窗口 {m.sample} 条；基线值 {m.fmt(m.baseline)}",
        advice={
            "json_degraded_ratio": "板书丢失的先行指标，去查提示词契约与 max_tokens",
            "quality_avg": "质检扣分先于学生察觉，看 top_issues 是哪一类",
            "step_error_ratio": "调用链里有步骤抛异常，用 trace 看是哪一步",
            "latency_p95": "先于超时与卡顿，看 slow_steps 定位",
        }.get(m.key, ""),
    )


def diagnose(
    *,
    days: int = 7,
    baseline_days: int | None = None,
    store: Any | None = None,
) -> Diagnosis:
    """跑一次完整诊断：当前窗口 vs 前一个等长窗口。

    ``baseline_days`` 留空 = 与紧邻的前一个等长窗口比（滚动对比，开箱即用、零配置）。
    """
    baseline_days = baseline_days or days
    if store is None:
        from core.verdict import verdict_store  # noqa: PLC0415 - 避免导入期耦合

        store = verdict_store

    span = max(days, baseline_days) * 2
    all_records = store.read(span)
    cur = _window(all_records, days, offset=0)
    base = _window(all_records, baseline_days, offset=1)

    cur_vals = collect(cur)
    base_vals = collect(base)

    metrics = [
        Metric(key=k, label=lb, unit=u, bad_when=d,
               value=cur_vals.get(k), baseline=base_vals.get(k), sample=len(cur))
        for k, lb, d, u in METRICS
    ]

    diag = Diagnosis(
        window_days=days,
        baseline_days=baseline_days,
        samples=len(cur),
        baseline_samples=len(base),
        metrics=metrics,
        coverage=coverage_of(cur, days),
        staleness=staleness_of(),
        last_gain=last_gain_of(),
    )

    findings: list[Finding] = []

    # --- 样本量门禁：不够就不下判断（这是最容易被忽略、也最容易误报的一条）---
    if len(cur) < MIN_SAMPLES:
        # 单列「样本」维度：它不是一条失败率结论，而是「本轮**放弃**了失败率判断」。
        # 混进失败率维度会让下游（含测试）误以为真的产出了一条失败率结论。
        findings.append(Finding(
            level="ok",
            dimension="样本",
            title=f"样本不足（{len(cur)}/{MIN_SAMPLES}），本轮只报事实、不下结论",
            evidence="小样本的比率没有统计意义，强行判断只会产生误报",
        ))
    else:
        for m in metrics:
            f = _judge_metric(m)
            if f:
                findings.append(f)

    # --- 影响面：某学科显著差于全局（比全局微涨更值得先看）---
    global_affected = cur_vals.get("affected_ratio") or 0.0
    if len(cur) >= MIN_SAMPLES and global_affected > 0:
        by_subject: dict[str, list[dict]] = {}
        for r in cur:
            by_subject.setdefault(str(r.get("subject") or "未知"), []).append(r)
        for name, rows in by_subject.items():
            if len(rows) < MIN_GROUP_SAMPLES:
                continue
            sub = collect(rows).get("affected_ratio") or 0.0
            if sub >= global_affected * SUBJECT_FACTOR and sub - global_affected >= WARN_ABS:
                findings.append(Finding(
                    level="warn",
                    dimension="失败率",
                    title=f"学科「{name}」受影响率 {sub * 100:.1f}%，是全局 "
                          f"{global_affected * 100:.1f}% 的 {sub / global_affected:.1f} 倍",
                    evidence=f"该学科 {len(rows)} 条记录（全局 {len(cur)} 条）",
                    advice="先看这个学科的经历记录里出现了什么失败原因",
                ))

    # --- 覆盖：有学科/场景完全没有数据 = 盲区（区别于「表现差」）---
    cov = diag.coverage
    if cov.get("missing_subjects"):
        findings.append(Finding(
            level="warn",
            dimension="覆盖",
            title=f"{len(cov['missing_subjects'])} 个学科在窗口内没有任何数据",
            evidence="、".join(cov["missing_subjects"][:8]),
            advice="要么没被用到，要么链路断了 —— 两者都不会体现在失败率上",
        ))
    if cov.get("days_with_data", 0) < min(days, 7):
        findings.append(Finding(
            level="warn",
            dimension="覆盖",
            title=f"窗口 {days} 天里只有 {cov.get('days_with_data', 0)} 天有数据",
            evidence="没有数据的日子无法判断好坏，但也不代表没问题",
        ))

    # --- staleness：学习链路是否还在跑 ---
    exp = diag.staleness.get("experience") or {}
    if not exp.get("exists"):
        findings.append(Finding(
            level="warn",
            dimension="staleness",
            title="教师经验库不存在（学习链路可能从未产出）",
            evidence=f"目录：{settings.skill_experience_dir}",
            advice="检查 SKILL_SELF_UPDATE_ENABLE / 复盘是否报错",
        ))
    elif exp.get("age_hours") is not None and exp["age_hours"] > STALE_HOURS:
        findings.append(Finding(
            level="warn",
            dimension="staleness",
            title=f"教师经验已 {exp['age_hours']:.0f} 小时没有更新",
            evidence=f"共 {exp.get('items', 0)} 条 / {exp.get('teachers', 0)} 位老师",
            advice="系统可能停学了 —— 这类停滞不会体现在失败率上",
        ))

    # --- last_gain：L4 评测的升降 ---
    gain = diag.last_gain
    if gain.get("available") and not gain.get("comparable"):
        findings.append(Finding(
            level="ok",
            dimension="last_gain",
            title=f"最近一次评测只跑了 {gain.get('latest_cases')} 题，"
                  f"与基线（{gain.get('baseline_cases')} 题）不可比",
            evidence="题数不同的两次评测，分数不能直接相减",
            advice="跑一次完整评测再判断：python scripts/eval_agent.py run --judge",
        ))
    elif gain.get("available"):
        d = float(gain.get("delta") or 0.0)
        if d <= -3.0:
            findings.append(Finding(
                level="warn",
                dimension="last_gain",
                title=f"最近一次评测比基线低 {abs(d)} 分",
                evidence=f"基线 {gain['baseline_total']}（{gain['baseline_stamp']}）"
                         f" → 本次 {gain['latest_total']}",
                advice="跑 scripts/eval_agent.py gate 看是哪几道题掉的",
            ))
        else:
            findings.append(Finding(
                level="ok",
                dimension="last_gain",
                title=f"评测分与基线持平或更好（{d:+.1f}）",
                evidence=f"基线 {gain['baseline_total']} → 本次 {gain['latest_total']}",
            ))

    if not findings:
        findings.append(Finding(level="ok", dimension="失败率", title="窗口内没有发现退化信号"))

    diag.findings = findings
    return diag


def as_dict(diag: Diagnosis) -> dict[str, Any]:
    """渲染成可打印 / 可返回接口的结构。"""
    return {
        "window": {
            "days": diag.window_days,
            "baseline_days": diag.baseline_days,
            "samples": diag.samples,
            "baseline_samples": diag.baseline_samples,
        },
        "level": diag.worst_level,
        "metrics": [
            {
                "key": m.key, "label": m.label, "unit": m.unit,
                "value": m.value, "baseline": m.baseline,
                "delta": m.delta, "rel": m.rel,
            }
            for m in diag.metrics
        ],
        "coverage": diag.coverage,
        "staleness": diag.staleness,
        "last_gain": diag.last_gain,
        "findings": [
            {"level": f.level, "dimension": f.dimension, "title": f.title,
             "evidence": f.evidence, "advice": f.advice}
            for f in diag.findings
        ],
    }


__all__ = [
    "METRICS",
    "MIN_SAMPLES",
    "Diagnosis",
    "Finding",
    "Metric",
    "as_dict",
    "collect",
    "coverage_of",
    "diagnose",
    "last_gain_of",
    "staleness_of",
]
