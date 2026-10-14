"""项目自检：学科目录 / 教师档案 / 同科多老师 / 技能模块 / 关键模块导入是否连通。

用法（项目根目录）：
    python scripts/selfcheck.py

不依赖任何引擎（LLM/TTS/数字人全部可缺省），用于改完代码后快速确认没打断链路。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from config.subjects import SUBJECTS, resolve_subject  # noqa: E402
from config.teacher_skills import available_skills  # noqa: E402
from config.teachers import (  # noqa: E402
    available_keys,
    default_teacher_key,
    profiles_for_subject,
    teacher_for_subject,
)


def main() -> int:
    problems: list[str] = []

    print(f"[1] 学科目录：{len(SUBJECTS)} 科")
    print("    ", [s["key"] for s in SUBJECTS])

    print(f"\n[2] 教师档案：{len(available_keys())} 个")
    print("    ", available_keys())

    print(f"\n[3] 学科技能模块：{len(available_skills())} 个")
    print("    ", available_skills())

    print("\n[4] 学科 -> 老师（同科多老师）")
    for s in SUBJECTS:
        name = s["name"]
        profiles = profiles_for_subject(name)
        default_key = default_teacher_key(name)
        prof = teacher_for_subject(name)          # 不传 teacher -> 默认老师
        if prof is None:
            problems.append(f"{name}: 没有可用教师档案")
            print(f"    {name:<6} [缺] 无档案")
            continue
        if prof.key != default_key:
            problems.append(f"{name}: 默认老师不一致 {prof.key} != {default_key}")
        voices = {p.voice_name() for p in profiles if p.voice_engine == "edge"}
        if len(profiles) > 1 and len(voices) < len([p for p in profiles if p.voice_engine == "edge"]):
            problems.append(f"{name}: 同科 edge 音色重复（听不出区别）{sorted(voices)}")
        keys = ", ".join(p.key for p in profiles)
        print(f"    {name:<6} 默认={default_key:<16} 可选 {len(profiles)} 位：{keys}")

    print("\n[5] 关键模块导入")
    try:
        from core.skills.expression_skill import ExpressionSkill  # noqa: F401
        from models.avatar import get_avatar  # noqa: F401
        from models.tts import get_tts  # noqa: F401
        from core.avatar_idle import ensure_idle_assets, ensure_teacher_portrait  # noqa: F401
        from server.routes import router  # noqa: F401

        print("    expression_skill / avatar / tts / avatar_idle / routes 全部 OK")
    except Exception as err:  # noqa: BLE001
        problems.append(f"模块导入失败：{type(err).__name__}: {err}")
        print(f"    [失败] {type(err).__name__}: {err}")

    print(f"\n[6] 学科名归一化：{resolve_subject('数学')} / {resolve_subject('science')} / {resolve_subject('综合')}")

    print("\n[7] Agent 流程节点（agents/nodes.py：谁负责哪些节点 + 连接顺序）")
    try:
        from agents import nodes as agent_nodes  # noqa: PLC0415

        described = agent_nodes.describe_agent_nodes()
        total = 0
        for role, graphs in agent_nodes.AGENT_NODES.items():
            for graph, specs in graphs.items():
                total += len(specs)
                chain = " → ".join(spec.name for spec in specs)
                print(f"    [{role}] {graph:<28} {chain}")
                for problem in described[role][graph]["problem"]:
                    problems.append(f"{graph}: {problem}")
        print(f"    共 {len(agent_nodes.AGENT_NODES)} 个角色 / {total} 个节点")
    except Exception as err:  # noqa: BLE001
        problems.append(f"节点地图加载失败：{type(err).__name__}: {err}")
        print(f"    [失败] {type(err).__name__}: {err}")

    print("\n[8] Agent 自主性（工具调用 / 回合规划 / 学情记忆，详见 docs/agent_autonomy.md）")
    try:
        from agents.shared.toolkit import AgentToolkit  # noqa: PLC0415
        from core.memory import get_memory_store  # noqa: PLC0415
        from core.planning import get_turn_planner  # noqa: PLC0415
        from core.state.models import StudentTurn  # noqa: PLC0415
        from core.state.session import SessionState  # noqa: PLC0415

        probe = SessionState(session_id="selfcheck", student_id="selfcheck")
        toolkit = AgentToolkit(probe)
        tools = toolkit.tool_names()
        print(f"    工具：开关={settings.agent_tools_enable} 数量={len(tools)}（{'、'.join(tools)}）")

        plan = get_turn_planner().rule_plan(StudentTurn(text="解方程 3x+5=20"))
        print(
            f"    规划：开关={settings.agent_planner_enable} 规则兜底={plan.source}"
            f"（建议工具 {'、'.join(plan.tools) or '无'}）"
        )

        store = get_memory_store()
        print(
            f"    记忆：开关={settings.memory_enable} 目录={store.root}"
            f"（已有记忆的学生 {len(store.list_students())} 位）"
        )
        print(
            f"    ReAct：开关={settings.agent_react_enable}"
            f"（最多 {settings.agent_react_max_steps} 轮，超时 {settings.agent_react_timeout_seconds}s）"
        )
        if not tools:
            problems.append("Agent 工具集为空：AGENT_TOOLS_ENABLE 或注册表有问题")
    except Exception as err:  # noqa: BLE001
        problems.append(f"自主性模块加载失败：{type(err).__name__}: {err}")
        print(f"    [失败] {type(err).__name__}: {err}")

    print("\n结论：", "全部通过" if not problems else f"存在 {len(problems)} 处问题")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
