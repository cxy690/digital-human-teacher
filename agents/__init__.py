"""agents 包：智能体集合（总控 + 意图分流 + 学生端代理 + AI 智能教师 + AI 讲课 + 共享能力）。

包里四个角色与实现类一一对应（调度关系见 ``agents/protocol.py``）：

| 角色（``ROLE_*``） | 实现类 | 模块 |
|---|---|---|
| ``supervisor`` | :class:`Supervisor` | ``agents/supervisor/agent.py``（总调度，唯一持有会话状态） |
| ``student`` | :class:`StudentAgent` | ``agents/student/agent.py``（麦控 / 设备状态 / 构造回合） |
| ``teacher`` | :class:`TeacherAgent` | ``agents/teacher/agent.py``（答疑 / 判题 / 出题） |
| ``lecturer`` | :class:`LecturerAgent` | ``agents/lecturer/agent.py``（课件逐页讲解 + 标注） |

**角色负责的流程节点**在各包的 ``nodes.py`` 里声明（``NodeSpec``：名称 / 功能 /
输入输出 / 前置依赖 / 所需配置），并由 ``agents/nodes.py`` 汇成一张地图；
每张图都能就地构建并执行（``build_agent_graph(角色, 图名, supervisor)``）：
``supervisor.turn`` · ``supervisor.slide`` · ``supervisor.slide_prepared`` ·
``teacher.answer`` · ``teacher.judge`` · ``teacher.quiz`` · ``lecturer.slide`` ·
``student.voice`` · ``student.control``。

``Supervisor`` 沿用 **惰性导出**：它的依赖链很长（``document_parsing`` / ``models`` 下的
OCR、ASR、TTS、数字人客户端），聚合导入会把这些全部拖进 ``import agents`` 的启动路径。
用 ``__getattr__`` 之后，``from agents import Supervisor`` 照常可用，而只想要
``TeacherAgent`` 的调用方（如 ``scripts/verify_stream_board.py``）不必付这份开销。
节点地图（``AGENT_NODES`` / ``describe_agent_nodes`` / ``build_agent_graph``）同样惰性导出 ——
它要跨包汇总四个角色，没必要为 ``import agents`` 付这份开销。
"""
from agents.base import BaseAgent
from agents.lecturer.agent import LecturerAgent
from agents.protocol import AgentBus, AgentMessage
from agents.shared.mic_control import MicController
from agents.shared.whiteboard import Whiteboard
from agents.student.agent import StudentAgent
from agents.teacher.agent import TeacherAgent

__all__ = [
    "AGENT_NODES",
    "AgentBus",
    "AgentMessage",
    "BaseAgent",
    "LecturerAgent",
    "MicController",
    "StudentAgent",
    "Supervisor",
    "TeacherAgent",
    "Whiteboard",
    "build_agent_graph",
    "describe_agent_nodes",
]


def __getattr__(name: str):
    """惰性导出总控 ``Supervisor`` 与节点地图（依赖链重，见包注释）。"""
    if name == "Supervisor":
        from agents.supervisor import Supervisor

        return Supervisor
    if name in ("AGENT_NODES", "build_agent_graph", "describe_agent_nodes"):
        from agents import nodes as agent_nodes

        return getattr(agent_nodes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
