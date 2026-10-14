"""L5 人工抽检台：按**风险**列出最该看一眼的教师经验。

用法（项目根目录）::

    python scripts/experience_review.py                   # 全部老师，前 10 条
    python scripts/experience_review.py --teacher math    # 只看某位老师
    python scripts/experience_review.py --limit 20
    python scripts/experience_review.py --json            # 出原始 JSON

为什么按风险而不是按时间
------------------------
经验是系统自己学的、**立刻就用**，人只能事后止损。所以治理环节里最稀缺的是
**人的注意力**：一次抽检大概几分钟。按时间列「最近新增的 20 条」，这几分钟会花在
绝大多数没问题的条目上；按风险排，才能落在最可能出问题的地方。

**它不阻断注入** —— 新经验照旧立刻可用。把自动学习改成「先审后用」等于退化成
人肉录入；这里做的是**降低抽检成本**。

风险信号与对应动作（判定见 ``core/skills/experience/review.py``）::

    实测偏低         用了 N 次、干净交付不到一半        → retire 或改措辞
    高置信但从未验证  模型自评很高，一次都没上过场        → 看一眼描述准不准
    与另一条近重复    内容高度重合，白占一个注入名额       → 合并或停用其中一条
    影响面大         注入过很多次，出问题波及其他回合      → 确认描述无误
"""
from __future__ import annotations

import argparse
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


def render(rows: list[dict], *, teacher: str) -> str:
    """渲染成给人读的文本（每条都要能回答「我该做什么」）。"""
    lines = ["", "=" * 74, "  教师经验抽检台" + (f" · {teacher}" if teacher else ""), "=" * 74]
    if not rows:
        lines.append("")
        lines.append("  没有需要抽检的条目 —— 说明没有「实测偏低 / 未验证 / 近重复 / 影响面大」的经验。")
        return "\n".join(lines)

    lines.append("")
    for i, row in enumerate(rows, 1):
        lines.append(f"  [{i}] 风险 {row['risk']:.2f}   {row['teacher']} / {row['kind']}")
        lines.append(f"      {row['text']}")
        lines.append(
            f"      战绩：注入 {row['pulls']} 次、干净交付 {row['wins']} 次、"
            f"有效率 {row['gain_rate']:.2f}；被复盘提到 {row['hits']} 次；"
            f"模型自评 {row['confidence']:.2f}"
        )
        for reason in row["reasons"]:
            lines.append(f"      → {reason}")
        lines.append(
            "      停用/恢复：POST /api/skills/experience/status "
            f'{{"teacher":"{row["teacher"]}","id":"{row["id"]}","status":"retired"}}'
        )
    lines.append("")
    lines.append("-" * 74)
    lines.append(f"  共 {len(rows)} 条。看完一条就 retire / 改一条，不必一次清空。")
    return "\n".join(lines)


def stats_render(teacher: str = "") -> str:
    """经验库战绩总览：**折叠后**的条数与实测战绩。

    为什么要单独看「折叠后」：经验文件是 append-only 的**版本快照**（每条经验每次
    变更追加一行，``load()`` 按 id 取最后一条），所以 ``wc -l`` 的数远大于实际条数，
    拿它当「学了多少」会严重高估。真正该看的是这里。
    """
    from core.skills.experience import get_experience_store  # noqa: PLC0415

    book = get_experience_store()
    keys = [teacher.strip()] if (teacher or "").strip() else book.list_teachers()
    lines = ["", "=" * 74, "  经验库战绩总览", "=" * 74, ""]
    if not keys:
        lines.append("  还没有任何经验。")
        return "\n".join(lines)

    lines.append(f"  {'老师':<12}{'生效':>5}{'停用':>5}{'注入':>6}{'干净':>6}{'被复盘':>7}")
    lines.append("  " + "-" * 60)
    for key in keys:
        items = book.all_items(key)
        live = [it for it in items if it.active]
        pulls = sum(it.pulls for it in live)
        wins = sum(it.wins for it in live)
        hits = sum(it.hits for it in live)
        lines.append(
            f"  {key:<12}{len(live):>5}{len(items) - len(live):>5}"
            f"{pulls:>6}{wins:>6}{hits:>7}"
        )
    lines.append("")
    lines.append("  注入 = 这条经验被选进提示词的次数；干净 = 那回合裁决为 pass 的次数。")
    lines.append("  注入长期为 0 → 「选经验 → 回写收益」这条闭环没跑起来，自我学习等于没生效。")
    return "\n".join(lines)


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="L5 教师经验抽检台（按风险排序）")
    parser.add_argument("--teacher", default="", help="只看某位老师（默认全部）")
    parser.add_argument("--limit", type=int, default=10, help="最多列几条（默认 10）")
    parser.add_argument("--json", action="store_true", help="只输出原始 JSON")
    parser.add_argument("--stats", action="store_true",
                        help="只出「经验库战绩总览」（折叠后的条数与实测战绩）")
    args = parser.parse_args()

    if args.stats:
        print(stats_render(args.teacher))
        return 0

    from core.skills.experience import review_queue  # noqa: PLC0415 - 延迟导入，--help 秒回

    rows = review_queue(args.teacher, limit=max(1, args.limit))
    if args.json:
        print(json.dumps({"teacher": args.teacher or "(全部)", "items": rows},
                         ensure_ascii=False, indent=2))
        return 0
    print(render(rows, teacher=args.teacher))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
