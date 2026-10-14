"""查看节点活动日志：把 ``logs/activity/*.jsonl`` 按「一次图执行」拼成可读链路。

为什么不直接 ``cat`` JSONL：一行一条事件，一次执行被拆成 ``run_start`` / (``node_start`` +
``node``) × N / ``run_end``，人眼拼不起来。这里按 ``run_id`` 归组，直接给出
**连接顺序 + 实际执行 + 跳过/收工/失败**，用于回答「这次答疑到底走了哪几个节点、
从哪条连接进来的、第几步挂了」。

另外两行专门解决「卡在哪一步」：

* ``进行中/中断`` —— 只发了 ``node_start`` 没发 ``node`` 的节点，就是此刻正在执行的
  （长节点如 LLM 生成期间也看得到），或进程被中断时卡住的那一步；
* ``未收尾`` —— 没有 ``run_end`` 时提示「距最后一条事件多久」，用来区分
  「还活着、只是这一步慢」与「进程已经死在半路」。

用法（项目根目录）：
    python scripts/view_activity.py                      # 最近 5 次图执行
    python scripts/view_activity.py --last 20 --graph teacher.answer
    python scripts/view_activity.py --day 2026-10-07 --session ses_abc
    python scripts/view_activity.py --list               # 有哪些日期的日志
    python scripts/view_activity.py --json               # 原始归组结果
    python scripts/view_activity.py --out logs\\activity-view.txt    # 导成 txt 存档/分享

为什么 ``--out`` 由脚本自己写文件、而不是用 ``>`` 重定向：Windows 上
``python xxx > a.txt`` 会把 Python 的 stdout 按**本地代码页（cp936）**写进管道，
PowerShell 再用自己的编码拼回去，中文必然乱码。这里直接以 ``utf-8-sig`` 落盘
（带 BOM：PowerShell 5.1 的 ``Get-Content`` 无 BOM 时按 ANSI 解码，同样会花屏）。

日志默认关闭（逐节点记录量大，排障时再开）：.env 里设 ``ACTIVITY_LOG_ENABLE=true``，
落盘位置与滚动/保留策略见 ``core/activity_log.py``。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import activity_log  # noqa: E402

WIDTH = 72


def _clock(ts: Any) -> str:
    """时间戳 → ``HH:MM:SS``（日志按天分文件，日期不重复展示）。"""
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError, OSError):
        return "--:--:--"


def render_run(run: dict[str, Any], *, now: float | None = None) -> list[str]:
    """一次执行 → 若干可读行（供终端与将来的接口复用）。

    ``now``：用于计算「进行中/未收尾已经过去多久」，默认取当前时间（测试可注入固定值）。
    """
    moment = time.time() if now is None else float(now)
    total = run.get("total_ms")
    total_text = f"{float(total):.2f}ms" if total is not None else "未收尾"
    lines = [
        f"run {run.get('run_id') or '?'} · {run.get('graph') or '?'} · "
        f"session={run.get('session') or '-'} · {_clock(run.get('started_at'))} · {total_text}"
    ]
    if run.get("order"):
        lines.append("  连接顺序: " + " → ".join(run["order"]))

    nodes: list[dict[str, Any]] = list(run.get("nodes") or [])
    if nodes:
        steps = []
        for item in nodes:
            status = str(item.get("status") or activity_log.STATUS_OK)
            if status == activity_log.STATUS_SKIP:
                steps.append(f"{item.get('node')}({activity_log.status_label(status)})")
            else:
                steps.append(
                    f"{item.get('node')}({activity_log.status_label(status)} "
                    f"{float(item.get('elapsed_ms') or 0.0):.2f}ms)"
                )
        lines.append("  实际执行: " + " → ".join(steps))

    if run.get("skipped"):
        lines.append("  跳过: " + ", ".join(run["skipped"]))
    if run.get("stopped"):
        lines.append("  收工: " + ", ".join(run["stopped"]) + "（其后节点未执行）")

    # 只发 node_start 没发 node 的节点 = 此刻正在执行，或进程被中断时卡住的那一步
    for item in run.get("running") or []:
        started_at = item.get("started_at")
        elapsed = moment - float(started_at) if started_at else 0.0
        lines.append(
            f"  进行中/中断: {item.get('node')}（起 {_clock(started_at)}，已 {elapsed:.1f}s）"
        )

    for item in nodes:
        if item.get("error"):
            lines.append(
                f"  ! {item.get('node')} 失败（{float(item.get('elapsed_ms') or 0.0):.2f}ms）：{item['error']}"
            )
    if run.get("aborted"):
        lines.append("  ! 本次执行被中断：非 optional 节点失败，整图未跑完（后续节点未执行）")
    if run.get("ended_at") is None:
        last_ts = run.get("last_ts")
        gap = f"{moment - float(last_ts):.1f}s" if last_ts else "未知"
        lines.append(
            f"  ! 未收尾：没有收尾事件（进程被中断，或此刻仍在执行）；"
            f"最后一条事件距今 {gap}"
        )
    return lines


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看节点活动日志（logs/activity/*.jsonl）")
    parser.add_argument("--dir", default="", help="日志根目录（默认取配置 ACTIVITY_LOG_DIR）")
    parser.add_argument("--day", default="", help="只看某天（YYYY-MM-DD）")
    parser.add_argument("--session", default="", help="只看某个会话")
    parser.add_argument("--graph", default="", help="只看某张图（如 supervisor.turn）")
    parser.add_argument("--last", type=int, default=5, help="显示最近几次执行（0=全部，默认 5）")
    parser.add_argument("--json", action="store_true", help="输出原始归组结果（JSON）")
    parser.add_argument("--list", action="store_true", help="只列出有日志的日期")
    parser.add_argument("--out", default="", help="把结果写成 txt 文件（UTF-8 BOM，避免控制台编码乱码）")
    return parser.parse_args(argv)


def _emit(lines: list[str], args: argparse.Namespace, code: int) -> int:
    """统一出口：默认打到终端；``--out`` 时写成 txt 文件（顺带解决 Windows 编码问题）。"""
    text = "\n".join(lines)
    if not args.out:
        print(text)
        return code
    path = Path(args.out)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8-sig")
    except OSError as err:
        print(f"写入失败：{path}（{err}）")
        return 1
    print(f"已写入 {path}（UTF-8 BOM，共 {len(lines)} 行）")
    print(f'查看：Get-Content "{path}" -Tail 40    过滤：Select-String -Path "{path}" -Pattern "失败"')
    return code


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log = activity_log.ActivityLog(args.dir or None, enabled=True)

    if args.list:
        days = log.days()
        return _emit([
            f"日志目录：{log.dir}",
            "有记录的日期：" + (", ".join(days) if days else "（无）"),
        ], args, 0 if days else 1)

    if not log.files():
        return _emit([
            f"没有活动日志：{log.dir}",
            "启用方式：在 .env 里设置 ACTIVITY_LOG_ENABLE=true（默认关闭，见 core/activity_log.py）",
        ], args, 1)

    runs = log.runs(
        day=args.day or None,
        session=args.session or None,
        graph=args.graph or None,
    )
    if args.json:
        return _emit(json.dumps(runs, ensure_ascii=False, indent=2).splitlines(), args, 0)

    shown = runs[-args.last:] if args.last > 0 else runs
    scope = f"{args.day}" if args.day else "全部日期"
    lines = [f"日志目录：{log.dir}（{scope} 共 {len(runs)} 次图执行，显示最近 {len(shown)} 次）"]
    if not shown:
        lines.append("没有匹配的记录：检查 --day / --session / --graph 过滤条件")
        return _emit(lines, args, 1)
    for run in shown:
        lines.append("-" * WIDTH)
        lines.extend(render_run(run))
    lines.append("-" * WIDTH)
    lines.append("提示：--last 0 看全部；--session / --graph 过滤；--json 看原始事件；--list 看有哪些日期")
    return _emit(lines, args, 0)


if __name__ == "__main__":
    raise SystemExit(main())
