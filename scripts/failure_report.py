"""L3 失败聚类报表：把一批失败**聚成可追的缺陷**，并可选地给出根因结论。

用法（项目根目录）::

    python scripts/failure_report.py                 # 最近 7 天，前 5 个簇
    python scripts/failure_report.py --days 1        # 只看今天
    python scripts/failure_report.py --top 10        # 多看几个簇
    python scripts/failure_report.py --analyze       # 对最大那个簇做根因分析（调 LLM，几秒）
    python scripts/failure_report.py --json          # 出原始 JSON（喂给别的工具）

和 ``verdict_report.py`` 的分工
------------------------------
===============  ================================  ==============================
脚本             回答                              数据形态
===============  ================================  ==============================
verdict_report   最近表现如何、哪里在退化            单维频次（原因/学科/场景各一张表）
**本脚本**       **坏了什么、该去改哪儿**           **跨维度聚成一个可追的缺陷**
===============  ================================  ==============================

``--analyze`` 会真的调模型，因此**默认不开**；不影响课堂，只是花几秒和一点点钱。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _force_utf8_stdout() -> None:
    """Windows 控制台默认非 UTF-8，直接 print 中文会炸或被替换成问号。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass


def render(data: dict, *, top: int) -> str:
    """把聚类报告渲染成给人读的文本。"""
    from core.failure_cluster import describe

    lines = ["", "=" * 74, "  L3 失败聚类 · 最近 %d 天" % data["window_days"], "=" * 74]
    body = describe(data["window_days"], top=top)
    lines.append("")
    lines.append("  " + body.replace("\n", "\n  "))
    lines.append("")
    lines.append("-" * 74)
    lines.append(
        "  「簇」= 一类可追的缺陷（失败原因 + 卡住的那一步 + 它偏向哪个场景/学科）。"
    )
    lines.append("  想看某一簇的根因：给本脚本加 --analyze")
    return "\n".join(lines)


async def _analyze(data: dict) -> dict:
    from core.failure_cluster import analyze_cluster

    clusters = data.get("clusters") or []
    if not clusters:
        return {"ok": False, "error": "窗口里没有失败簇，没什么可分析的"}
    return await analyze_cluster(clusters[0])


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="L3 失败聚类 + 根因分析")
    parser.add_argument("--days", type=int, default=None, help="统计窗口天数（默认取配置）")
    parser.add_argument("--top", type=int, default=5, help="最多显示几个簇（默认 5）")
    parser.add_argument("--analyze", action="store_true", help="对最大的簇做根因分析（调 LLM）")
    parser.add_argument("--json", action="store_true", help="只输出原始 JSON")
    args = parser.parse_args()

    from core.failure_cluster import report  # noqa: PLC0415 - 延迟导入，保证 --help 秒回

    data = report(args.days, top=max(1, args.top))

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    print(render(data, top=max(1, args.top)))

    if args.analyze:
        result = asyncio.run(_analyze(data))
        print("")
        print("=" * 74)
        print("  根因分析（最大那个簇）")
        print("=" * 74)
        if not result.get("ok"):
            print(f"  未完成：{result.get('error')}")
            return 1
        print(f"  根因：{result['root_cause']}    （置信度 {result['confidence']}）")
        if result["evidence"]:
            print("  证据：")
            for item in result["evidence"]:
                print(f"    - {item}")
        if result["suggestions"]:
            print("  建议改哪里：")
            for item in result["suggestions"]:
                print(f"    → {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
