"""agents.supervisor 包：总调度智能体（总控），存放会话闭环的唯一编排者。

四个角色的包结构现在是齐的，**包名 = 角色名**：

| 包 | 角色 | 实现类 |
|---|---|---|
| ``agents/supervisor/`` | ``supervisor`` | :class:`Supervisor`（总调度：唯一持有会话状态） |
| ``agents/teacher/`` | ``teacher`` | :class:`~agents.teacher.agent.TeacherAgent`（答疑/判题/出题） |
| ``agents/lecturer/`` | ``lecturer`` | :class:`~agents.lecturer.agent.LecturerAgent`（课件逐页讲解 + 标注） |
| ``agents/student/`` | ``student`` | :class:`~agents.student.agent.StudentAgent`（麦控 / 设备状态） |

总控是唯一**有状态**的 agent（其余三个只吃参数、吐 ``AIResponse``）：
调用顺序、抢占、异常回退、白板与语音下发都由它决定，
策略全貌见 ``docs/supervisor_orchestration.md``。

对外只暴露 :class:`Supervisor`；装配入口是 ``core/harness.py``（建会话）与
``core/loop.py``（事件分发），外部也可沿用 ``from core import Supervisor``。
"""
from agents.supervisor.agent import Supervisor

__all__ = ["Supervisor"]
