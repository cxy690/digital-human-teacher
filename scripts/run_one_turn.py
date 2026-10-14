"""跑一轮真实答疑并打印全过程 —— 不开前端就能验证「链路是否真的通了」。

用法（项目根目录）::

    python scripts/run_one_turn.py "这道二次函数题怎么做？"
    python scripts/run_one_turn.py "什么是数轴？" --subject 数学 --teacher math
    python scripts/run_one_turn.py "你好呀" --no-speak      # 跳过 TTS/数字人（快）

为什么要它
----------
本项目的功能常挂在**多条链路**上（内建图 / LangGraph、答疑 / 讲课、生产 Supervisor /
同名拷贝），改完一处很难靠读代码确认「生产真的走到了」。而开前端验证成本很高
（要点按钮、要等渲染）。

这个脚本用**生产那条链路**（``core.harness`` 装配的那个 Supervisor）跑一轮，然后打印：

* 话术与板书（回答到底出来了没有）
* 耗时与调用链（慢在哪一步）
* **本次是否落盘了裁决**（L0 观测层，见 ``core/verdict``）
* **本次是否触发了经验复盘 / 学情记忆**（L3 学习链路）

后两项是关键：它们让「系统有没有在学习」从「看代码猜」变成「看输出」。

注意：它会真的调用 LLM，并在默认情况下真的合成语音 / 渲染数字人（用 --no-speak 跳过）。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from core.state.models import StudentTurn  # noqa: E402
from core.state.session import SessionState  # noqa: E402


def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass


def _count_experience(teacher_key: str) -> int:
    """该老师的经验条数（读文件，失败返回 -1 表示读不了）。"""
    try:
        path = Path(settings.skill_experience_dir) / f"{teacher_key}.jsonl"
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return -1


def _count_verdicts() -> int:
    """今天已落盘的裁决条数。"""
    try:
        import time

        path = Path(settings.verdict_dir) / f"verdict-{time.strftime('%Y-%m-%d')}.jsonl"
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return -1


async def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="跑一轮真实答疑（生产链路）")
    parser.add_argument("question", help="学生的问题")
    parser.add_argument("--subject", default="数学", help="学科（默认数学）")
    parser.add_argument("--teacher", default="", help="老师 key（默认该学科默认老师）")
    parser.add_argument("--student", default="probe-student", help="学生 id")
    parser.add_argument("--no-speak", action="store_true",
                        help="跳过 TTS 与数字人渲染（快，但仍真的调 LLM）")
    args = parser.parse_args()

    # 必须从 harness 取 —— 它才是 server 装配生产 Supervisor 的入口。
    # 直接 from core.supervisor import 也可以，但用 harness 能顺带验证
    # 「生产装配入口」本身没坏。
    import core.harness as harness

    exp_before = _count_experience(args.teacher or args.subject)
    verdicts_before = _count_verdicts()

    state = SessionState(session_id="probe-session", student_id=args.student)
    state.subject = args.subject
    state.scratch["subject"] = args.subject
    if args.teacher:
        state.teacher = args.teacher
        state.scratch["teacher"] = args.teacher

    supervisor = harness.Supervisor(state)
    if args.no_speak:
        # 只跳过「合成音频 + 渲染数字人」，白板与文本照常走，因此调用链仍完整。
        async def _skip(_response, **_kw):  # type: ignore[no-untyped-def]
            return _response

        supervisor._push_audio = _skip  # type: ignore[method-assign]
        supervisor._push_avatar_video = _skip  # type: ignore[method-assign]

    print("=" * 70)
    print(f"  学科={args.subject}  老师={args.teacher or '(默认)'}  提问={args.question}")
    print(f"  Supervisor = {type(supervisor).__module__}.{type(supervisor).__name__}")
    print("=" * 70)

    import time

    started = time.perf_counter()
    response = await supervisor.handle_turn(StudentTurn(text=args.question))
    elapsed = (time.perf_counter() - started) * 1000

    print()
    print(f"总耗时：{elapsed:.0f} ms")
    if response is None:
        print("产物：None（本回合不产出回复）")
    else:
        print(f"话术（{len(response.speech or '')} 字）：{response.speech}")
        print(f"板书指令：{len(response.board_ops or [])} 条")
        print(f"知识点：{response.knowledge_points}")
        meta_keys = ", ".join(sorted(response.meta.keys()))
        print(f"meta 字段：{meta_keys}")
        if response.meta.get("json_degraded"):
            print("  ⚠️ 本回合 JSON 降级（板书整块丢弃）")

    # 等后台学习任务收尾（复盘要调一次 LLM，给足时间）
    print()
    print("等待后台学习任务（经验复盘 / 学情记忆）...")
    for _ in range(40):
        await asyncio.sleep(0.25)
        if not supervisor._bg_tasks:
            break
    pending = len([t for t in supervisor._bg_tasks if not t.done()])
    print(f"后台任务：{'仍在跑 ' + str(pending) + ' 个' if pending else '已全部完成'}")

    exp_after = _count_experience(args.teacher or args.subject)
    verdicts_after = _count_verdicts()

    print()
    print("─" * 70)
    print(f"  裁决落盘（L0）    : {verdicts_before} -> {verdicts_after} "
          f"{'✅ 新增' if verdicts_after > verdicts_before else '⚠️ 未新增'}"
          f"   [{settings.verdict_dir}]")
    print(f"  教师经验（L3）    : {exp_before} -> {exp_after} "
          f"{'✅ 新增' if exp_after > exp_before else '（本次未沉淀新经验）'}"
          f"   [{settings.skill_experience_dir}]")
    print(f"  学情记忆目录      : {settings.memory_dir} "
          f"{'存在' if Path(settings.memory_dir).exists() else '（尚未创建）'}")
    print("─" * 70)
    print()

    await supervisor.on_session_end(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
