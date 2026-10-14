"""回合健康摘要：把 ``core/verdict.py`` 落盘的裁决聚合成一张能看的报表。

用法（项目根目录）::

    python scripts/verdict_report.py                 # 最近 7 天
    python scripts/verdict_report.py --days 1        # 只看今天
    python scripts/verdict_report.py --recent 15     # 附最近 15 条「非干净」回合
    python scripts/verdict_report.py --json          # 出原始 JSON（喂给别的工具）

它是 L0 观测层的**查看入口**：回答「系统最近表现如何、哪里在退化」。
注意本脚本只做**统计**，不做判断 —— 「多少算退化」属于 L1 诊断层的事。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.verdict import verdict_store  # noqa: E402


def _force_utf8_stdout() -> None:
    """Windows 控制台默认非 UTF-8，直接 print 中文会炸或被替换成问号。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass


def _fmt(v: object, dash: str = "-") -> str:
    """把 ``None`` 显示成占位符（聚合在无样本时会返回 None）。"""
    return dash if v is None else str(v)


def _bar(pct: float, width: int = 20) -> str:
    """用字符画一个极简占比条，让「谁最差」一眼可见。"""
    filled = int(round(pct / 100.0 * width))
    return "█" * filled + "·" * (width - filled)


def render(summary: dict, recent: list[dict]) -> str:
    """把摘要渲染成给人读的文本。"""
    lines: list[str] = []
    win = summary["window"]
    lines.append("")
    lines.append("=" * 74)
    lines.append(f"  回合健康摘要 · 最近 {win['days']} 天")
    lines.append(f"  窗口：{win['since']} ~ {win['until']}")
    lines.append("=" * 74)

    if not summary["total"]:
        lines.append("")
        lines.append("  窗口内没有任何记录。可能原因：")
        lines.append("    * 还没跑过真实回合（VERDICT_ENABLE 默认开，跑一次答疑就会写）")
        lines.append("    * VERDICT_DIR 指到了别的目录")
        lines.append(f"    * 落盘目录：{verdict_store.dir}")
        lines.append("")
        return "\n".join(lines)

    rates = summary["rates_pct"]
    v = summary["verdicts"]
    lines.append("")
    lines.append(f"  总回合数：{summary['total']}")
    lines.append("")
    lines.append("  ── 判定分布 " + "─" * 60)
    lines.append(f"    ✅ 干净    {v['pass']:>5}   {rates['clean']:>5.1f}%  {_bar(rates['clean'])}")
    lines.append(f"    ⚠️  降级    {v['degraded']:>5}   {rates['degraded']:>5.1f}%  {_bar(rates['degraded'])}")
    lines.append(f"    ❌ 失败    {v['failed']:>5}   {rates['failed']:>5.1f}%  {_bar(rates['failed'])}")
    lines.append(f"    受影响合计       {rates['affected']:>5.1f}%   ← 降级 + 失败（学生感觉得到掉东西）")

    q = summary["quality"]
    lines.append("")
    lines.append("  ── 质检 " + "─" * 64)
    if q["scored"]:
        lines.append(f"    有分数 {q['scored']} 条，均分 {q['avg_score']} / 100")
        for issue, cnt in q["top_issues"]:
            lines.append(f"      · {issue}  ×{cnt}")
    else:
        lines.append("    窗口内没有带质检分数的回合")

    lat = summary["latency_ms"]
    lines.append("")
    lines.append("  ── 整回合耗时（毫秒） " + "─" * 50)
    lines.append(f"    p50 {_fmt(lat['p50'])}   p95 {_fmt(lat['p95'])}   max {_fmt(lat['max'])}   avg {_fmt(lat['avg'])}")

    if summary["by_subject"]:
        lines.append("")
        lines.append("  ── 按学科（受影响率降序，最差的先看） " + "─" * 36)
        for name, cell in summary["by_subject"].items():
            lines.append(
                f"    {name:<8} {cell['total']:>5} 回合   受影响 {cell['affected_pct']:>5.1f}%  "
                f"{_bar(cell['affected_pct'], 14)}"
            )

    if summary["by_kind"]:
        lines.append("")
        lines.append("  ── 按场景 " + "─" * 62)
        for name, cell in summary["by_kind"].items():
            lines.append(
                f"    {name:<14} {cell['total']:>5} 回合   受影响 {cell['affected_pct']:>5.1f}%"
            )

    if summary["slow_steps"]:
        lines.append("")
        lines.append("  ── 最慢的步骤（按 p95，优化从这里下手） " + "─" * 35)
        for s in summary["slow_steps"]:
            err = f"  异常 {s['errors']}" if s["errors"] else ""
            lines.append(
                f"    p95 {str(s['p95_ms']):>7}ms  p50 {str(s['p50_ms']):>6}ms  "
                f"×{s['calls']:<4} {s['name']}{err}"
            )

    if summary["top_reasons"]:
        lines.append("")
        lines.append("  ── 主要失败原因 " + "─" * 58)
        for reason, cnt in summary["top_reasons"]:
            lines.append(f"    ×{cnt:<4} {reason}")

    if recent:
        lines.append("")
        lines.append(f"  ── 最近 {len(recent)} 条「非干净」回合 " + "─" * 44)
        for rec in recent:
            ts = str(rec.get("ts_iso") or "")[5:16]      # 只看月-日 时:分
            lines.append(
                f"    {ts}  [{rec.get('verdict')}] {rec.get('kind')}/{rec.get('subject') or '?'}"
                f"  {rec.get('elapsed_ms')}ms"
            )
            q_text = str(rec.get("question") or "")[:40]
            if q_text:
                lines.append(f"              问：「{q_text}」")
            for reason in (rec.get("reasons") or [])[:3]:
                lines.append(f"              ↳ {reason}")

    lines.append("")
    lines.append("=" * 74)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="回合健康摘要（L0 观测层查看入口）")
    parser.add_argument("--days", type=int, default=None,
                        help="统计最近几天（默认取 VERDICT_WINDOW_DAYS，出厂 7）")
    parser.add_argument("--recent", type=int, default=0,
                        help="附最近 N 条「非干净」回合的明细（默认 0 = 不附）")
    parser.add_argument("--json", action="store_true", help="只输出原始 JSON")
    args = parser.parse_args()

    summary = verdict_store.summarize(args.days).as_dict()

    recent: list[dict] = []
    if args.recent > 0:
        days = args.days if args.days is not None else summary["window"]["days"]
        bad = [r for r in verdict_store.read(days) if r.get("verdict") != "pass"]
        recent = bad[-args.recent:]

    if args.json:
        print(json.dumps({**summary, "recent": recent}, ensure_ascii=False, indent=2))
        return 0

    print(render(summary, recent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
