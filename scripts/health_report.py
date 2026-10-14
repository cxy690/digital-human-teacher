"""健康诊断报告（L1）：判断系统在变好还是变坏，并指出该先看哪里。

用法（项目根目录）::

    python scripts/health_report.py                  # 最近 7 天 vs 前 7 天
    python scripts/health_report.py --days 1         # 今天 vs 昨天
    python scripts/health_report.py --json           # 出原始 JSON（喂给别的工具）

与 ``scripts/verdict_report.py`` 的区别（两者互补，不是替代）
------------------------------------------------------------
============  ==========================  ============================
脚本          回答的问题                   做什么
============  ==========================  ============================
verdict_report 「数据长什么样」             列分布、列分组、列最慢步骤
health_report  「**要紧吗**、该先看哪里」    对比基线、出结论与建议
============  ==========================  ============================

前者的输出是**事实**，后者是**判断** —— 判断口径（多少算退化）改起来很频繁，
所以刻意不写进落盘格式里（见 :mod:`core.health` 开头的分工说明）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.health import METRICS, MIN_SAMPLES, as_dict, diagnose  # noqa: E402

_LEVEL_STYLE = {
    "critical": ("🔴", "CRITICAL"),
    "warn": ("🟡", "WARN    "),
    "ok": ("🟢", "OK      "),
}


def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass


def render(diag, metrics_by_key: dict) -> str:
    """把诊断渲染成给人读的文本。"""
    L: list[str] = []
    w = diag
    L.append("")
    L.append("=" * 78)
    L.append(f"  健康诊断 · 最近 {w.window_days} 天（对比前 {w.baseline_days} 天）")
    L.append("=" * 78)
    L.append("")
    L.append(f"  样本：本次 {w.samples} 条  基线 {w.baseline_samples} 条"
             f"   {'⚠️ 低于 ' + str(MIN_SAMPLES) + ' 条，本轮不下结论' if w.samples < MIN_SAMPLES else ''}")

    # ---- 四维画像 ----
    L.append("")
    L.append("  ── 四维画像 " + "─" * 64)

    def fmt(m) -> str:
        return f"{m.fmt(m.baseline)} → {m.fmt(m.value)}"

    def mark(m) -> str:
        if m.value is None or m.baseline is None:
            return "·"
        return "▼" if m.worse() > 0 else ("▲" if m.worse() < 0 else "=")

    affected = metrics_by_key.get("affected_ratio")
    if affected:
        L.append(f"    ① 失败率  受影响率 {fmt(affected)}  {mark(affected)}")
    cov = w.coverage
    miss = cov.get("missing_subjects") or []
    L.append(f"    ② 覆盖    {cov.get('subject_count', 0)}/{cov.get('known_subject_count') or '?'} 学科"
             f" / {len(cov.get('kinds') or [])} 类场景"
             f" / {cov.get('days_with_data', 0)} 天有数据"
             + (f"   缺：{'、'.join(miss[:4])}" if miss else ""))
    exp = w.staleness.get("experience") or {}
    mem = w.staleness.get("memory") or {}
    exp_age = f"{exp['age_hours']:.0f}h 前" if exp.get("age_hours") is not None else "无"
    mem_age = f"{mem['age_hours']:.0f}h 前" if mem.get("age_hours") is not None else "无"
    L.append(f"    ③ 新鲜度  经验 {exp.get('items', 0)} 条（最后更新 {exp_age}）"
             f" / 学情记忆 {mem.get('items', 0)} 条（{mem_age}）")
    gain = w.last_gain
    if gain.get("available") and gain.get("comparable"):
        L.append(f"    ④ last_gain  评测 {gain['baseline_total']} → {gain['latest_total']}"
                 f"（{gain['delta']:+.1f}）")
    elif gain.get("available"):
        L.append(f"    ④ last_gain  最近一次只跑 {gain.get('latest_cases')} 题，"
                 f"与基线（{gain.get('baseline_cases')} 题）不可比")
    else:
        L.append("    ④ last_gain  无评测数据（跑 scripts/eval_agent.py run 后可见）")

    # ---- 指标对比表 ----
    L.append("")
    L.append("  ── 指标对比（本次 vs 基线） " + "─" * 50)
    L.append(f"    {'指标':<14}{'基线':>10}{'本次':>10}{'变化':>12}   判定")
    for key, label, _dir, unit in METRICS:
        m = metrics_by_key.get(key)
        if m is None or m.value is None:
            continue
        delta = ""
        if m.delta is not None:
            if unit == "ratio":
                delta = f"{m.delta * 100:+.1f}pp"
            elif unit == "ms":
                delta = f"{m.delta:+.0f}ms"
            else:
                delta = f"{m.delta:+.1f}"
        worse = m.worse()
        verdict = "—" if worse == 0 else ("**变差**" if worse > 0 else "变好")
        L.append(f"    {label:<14}{m.fmt(m.baseline):>10}{m.fmt(m.value):>10}{delta:>12}   {mark(m)} {verdict}")

    # ---- 结论 ----
    L.append("")
    L.append("  ── 诊断结论 " + "─" * 64)
    for f in w.findings:
        icon, name = _LEVEL_STYLE.get(f.level, ("·", f.level))
        L.append(f"    {icon} [{name}] 【{f.dimension}】{f.title}")
        if f.evidence:
            L.append(f"         依据：{f.evidence}")
        if f.advice:
            L.append(f"         建议：{f.advice}")
    L.append("")
    L.append(f"  总体：{_LEVEL_STYLE.get(w.worst_level, ('·', ''))[1].strip()}")
    L.append("")
    L.append("=" * 78)
    L.append("")
    return "\n".join(L)


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="L1 健康诊断报告")
    parser.add_argument("--days", type=int, default=7, help="当前窗口天数（默认 7）")
    parser.add_argument("--baseline-days", type=int, default=0,
                        help="基线窗口天数（默认与当前窗口等长）")
    parser.add_argument("--json", action="store_true", help="只输出原始 JSON")
    args = parser.parse_args()

    diag = diagnose(days=args.days, baseline_days=args.baseline_days or None)
    payload = as_dict(diag)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if diag.worst_level != "critical" else 2

    metrics_by_key = {m.key: m for m in diag.metrics}
    print(render(diag, metrics_by_key))
    # 退出码可被 CI 直接用：critical=2 / warn=1 / ok=0
    return {"critical": 2, "warn": 1, "ok": 0}.get(diag.worst_level, 0)


if __name__ == "__main__":
    raise SystemExit(main())
