"""L5 语义归并：找出「说法不同、意思相同」的教师经验（要调模型，离线跑）。

用法（项目根目录）::

    python scripts/experience_dedupe.py            # 全部老师
    python scripts/experience_dedupe.py --teacher math
    python scripts/experience_dedupe.py --json     # 出原始 JSON

和 ``experience_review.py`` 的分工
----------------------------------
* ``experience_review.py`` —— **免费、快**，按风险排字面可疑的条目（含字面近重复）。
  日常抽检用它。
* 本脚本 —— **要调模型、花几秒、算钱**，抓「字面看不出来、意思却相同」的那一类。
  定期体检用它。

实测依据：云端真实数据里有一对谁都看得出是一件事的经验，字面相似度只有 0.348
（而彼此无关的条目是 0.07）—— 字面方法够不着，只能靠语义。详见
``core/skills/experience/dedupe.py``。

⚠️ 它**只给建议，不自动停用**：归并错了会永久丢掉一个独立的教学要点。
确认后请用 ``POST /api/skills/experience/status`` 停用，或抽检台里操作。
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


def render(result: dict) -> str:
    """渲染成给人读的建议（每组都要能回答「留哪条、去掉哪条、为什么」）。"""
    lines = ["", "=" * 76, "  经验语义归并建议（说法不同、意思相同）", "=" * 76, ""]
    if not result.get("ok"):
        lines.append(f"  ⚠️ 没跑成：{result.get('error')}")
        return "\n".join(lines)

    scanned = result.get("scanned") or {}
    if scanned:
        lines.append("  已比较：" + "、".join(f"{k} {v} 条" for k, v in scanned.items()))
        lines.append("")

    groups = [g for g in (result.get("groups") or []) if g.get("ids")]
    errors = [g for g in (result.get("groups") or []) if g.get("error")]
    if not groups:
        lines.append("  没有发现「同一件事」的经验。")
    for index, group in enumerate(groups, 1):
        lines.append(f"  [{index}] {group['teacher']} —— {group.get('why') or '（模型没给理由）'}")
        for item_id in group["ids"]:
            mark = "留" if item_id == group["keep"] else "去"
            lines.append(f"      [{mark}] {group['texts'].get(item_id, '')}")
            lines.append(f"           id={item_id}")
        lines.append("")
    for err in errors:
        lines.append(f"  ⚠️ {err['teacher']} 这一轮没跑成：{err['error']}")

    lines.append("-" * 76)
    lines.append("  以上只是建议 —— 确认无误后，用下面的接口停用（不可逆，请先逐条读一遍）：")
    for group in groups:
        for item_id in group["drop"]:
            lines.append(
                "    POST /api/skills/experience/status "
                f'{{"teacher":"{group["teacher"]}","id":"{item_id}","status":"retired"}}'
            )
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    from core.skills.experience import find_duplicate_groups  # noqa: PLC0415 - 延迟导入，--help 秒回

    result = await find_duplicate_groups(args.teacher)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render(result))
        if not result.get("ok"):
            return 1
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(
        description="L5 语义归并：找出说法不同、意思相同的教师经验（调模型，离线跑）"
    )
    parser.add_argument("--teacher", default="", help="只看某位老师（默认全部）")
    parser.add_argument("--json", action="store_true", help="只输出原始 JSON")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
