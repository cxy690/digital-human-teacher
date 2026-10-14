"""agents 节点注册表：汇总四个 Agent 各自负责的流程节点。

四个 Agent 的节点声明分散在各包的 ``nodes.py``（``agents/<角色>/nodes.py``），
本模块把它们汇到一张地图里，回答三个问题：

1. **谁负责哪些流程节点** —— :data:`AGENT_NODES`（角色 → 图 → 节点声明）；
2. **每个节点的功能 / 输入输出 / 连接顺序 / 依赖配置** —— 节点声明里的
   ``summary`` / ``inputs`` / ``outputs`` / ``deps`` / ``requires``；
3. **怎么把节点连起来跑** —— :data:`BUILDERS`（角色 → 图 → 构建函数）。

| 角色 | 图 | 节点（连接顺序） |
|---|---|---|
| ``supervisor`` | ``supervisor.turn`` | route → answer_check / multi_intent / control / chitchat / teach |
| ``supervisor`` | ``supervisor.slide`` | present_page → explain → annotate → speak → push_audio |
| ``supervisor`` | ``supervisor.slide_prepared`` | present_page → annotate → speak → flush_events |
| ``teacher`` | ``teacher.answer`` | retrieve → generate → validate → degrade → solve |
| ``teacher`` | ``teacher.judge`` | evaluate → decide → announce |
| ``teacher`` | ``teacher.quiz`` | generate → publish → announce |
| ``lecturer`` | ``lecturer.slide`` | request → explain → fallback → degrade |
| ``student`` | ``student.voice`` | guard_mic → transcribe → guard_transcript → build_turn → dispatch |
| ``student`` | ``student.control`` | camera_off / camera_on / mic_off / mic_on / clear_board |

装配与调用都在总控（``agents/supervisor/agent.py``）：各图由对应入口「就地构建 + 执行」，
节点声明与执行实现同源（``NodeSpec`` 既是文档也是建图输入，见 ``core/graph.py``）。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agents.lecturer import nodes as lecturer_nodes
from agents.protocol import ROLE_LECTURER, ROLE_STUDENT, ROLE_SUPERVISOR, ROLE_TEACHER
from agents.student import nodes as student_nodes
from agents.supervisor import nodes as supervisor_nodes
from agents.teacher import nodes as teacher_nodes
from core.graph import NodeSpec, TaskGraph

GraphBuilder = Callable[[Any], TaskGraph]

#: 角色 → 图名 → 节点声明（与实现同源，见各包 ``nodes.py`` 的 ``GRAPH_SPECS``）
AGENT_NODES: dict[str, dict[str, tuple[NodeSpec, ...]]] = {
    ROLE_SUPERVISOR: dict(supervisor_nodes.GRAPH_SPECS),
    ROLE_TEACHER: dict(teacher_nodes.GRAPH_SPECS),
    ROLE_LECTURER: dict(lecturer_nodes.GRAPH_SPECS),
    ROLE_STUDENT: dict(student_nodes.GRAPH_SPECS),
}

#: 角色 → 图名 → 构建函数（入参为总控 ``Supervisor``，图内 handler 复用它的实现）
BUILDERS: dict[str, dict[str, GraphBuilder]] = {
    ROLE_SUPERVISOR: {
        supervisor_nodes.TURN_GRAPH: supervisor_nodes.build_turn_graph,
        supervisor_nodes.SLIDE_GRAPH: supervisor_nodes.build_slide_graph,
        supervisor_nodes.PRESENTED_GRAPH: supervisor_nodes.build_presented_graph,
    },
    ROLE_TEACHER: {
        teacher_nodes.ANSWER_GRAPH: teacher_nodes.build_answer_graph,
        teacher_nodes.JUDGE_GRAPH: teacher_nodes.build_judge_graph,
        teacher_nodes.QUIZ_GRAPH: teacher_nodes.build_quiz_graph,
    },
    ROLE_LECTURER: {
        lecturer_nodes.SLIDE_GRAPH: lecturer_nodes.build_slide_graph,
    },
    ROLE_STUDENT: {
        student_nodes.VOICE_GRAPH: student_nodes.build_voice_graph,
        student_nodes.CONTROL_GRAPH: student_nodes.build_control_graph,
    },
}


def graphs_of(role: str) -> dict[str, tuple[NodeSpec, ...]]:
    """某角色负责的全部图（图名 → 节点声明）。角色未知时抛 ``KeyError``。"""
    return AGENT_NODES[role]


def specs_of(role: str, graph: str) -> tuple[NodeSpec, ...]:
    """某角色某张图的节点声明（按连接顺序）。"""
    return AGENT_NODES[role][graph]


def build_agent_graph(role: str, graph: str, supervisor: Any) -> TaskGraph:
    """按角色 + 图名构建可执行图（handler 绑定到 ``supervisor`` 的实现）。"""
    builder = BUILDERS[role][graph]
    return builder(supervisor)


def describe_agent_nodes(role: str | None = None) -> dict[str, Any]:
    """节点地图（可序列化）：不带 ``role`` 时返回全部角色的图与节点。"""
    roles = [role] if role else list(AGENT_NODES)
    out: dict[str, Any] = {}
    for name in roles:
        graphs: dict[str, Any] = {}
        for graph, specs in AGENT_NODES[name].items():
            graphs[graph] = {
                "order": [spec.name for spec in specs],
                "problem": _problems(specs),
                "nodes": [spec.to_dict() for spec in specs],
            }
        out[name] = graphs
    return out


def _problems(specs: tuple[NodeSpec, ...]) -> list[str]:
    """声明级自检：未知依赖 / 自依赖（建图前就能发现写错的连接）。"""
    names = {spec.name for spec in specs}
    problems: list[str] = []
    for spec in specs:
        for dep in spec.deps:
            if dep == spec.name:
                problems.append(f"node {spec.name} depends on itself")
            elif dep not in names:
                problems.append(f"node {spec.name} depends on unknown node {dep}")
    return problems


def render_agent_nodes(role: str | None = None) -> str:
    """把节点地图渲染成文本（``python -m agents.nodes`` 可打印）。"""
    lines: list[str] = []
    for agent_role, graphs in describe_agent_nodes(role).items():
        lines.append(f"[{agent_role}]")
        for graph, info in graphs.items():
            deps = {node["name"]: node["deps"] for node in info["nodes"]}
            chain = " → ".join(info["order"])
            lines.append(f"  {graph}: {chain}")
            for node in info["nodes"]:
                pre = f"<- {', '.join(node['deps'])}" if node["deps"] else "<- (入口)"
                requires = f" | 依赖: {', '.join(node['requires'])}" if node["requires"] else ""
                lines.append(f"    - {node['name']} {pre}: {node['summary']}{requires}")
            if info["problem"]:
                lines.append(f"    ! {info['problem']}")
            # deps 与节点清单一一对应，便于人工核对连接顺序
            lines.append(f"    (deps) {deps}")
        lines.append("")
    return "\n".join(lines).rstrip()


if __name__ == "__main__":  # pragma: no cover - 人工查看节点地图
    print(render_agent_nodes())
