# agents — 智能体层（四个角色 + 协作协议）

智能体按**职责**切分，不按学科——学科差异由 `.agents/skills/<学科key>/SKILL.md` 一份 Markdown 承载。四个角色：

| 角色 | 包 | 负责 | 不负责 |
| --- | --- | --- | --- |
| `Supervisor` | `supervisor/` | 唯一持有会话状态：调用顺序、抢占、异常回退 | 具体讲解内容 |
| `TeacherAgent` | `teacher/` | 答疑：讲题、图片题、白板书写、追问判定、出题判题 | 课件逐页讲解 |
| `LecturerAgent` | `lecturer/` | 讲课：课件逐页话术、页面圈画/高亮 | 答疑、正文板书 |
| `StudentAgent` | `student/` | 学生端代理（麦控状态机 / 设备状态 / 学生回合生成） | — |

> 入口闸门 `core/intent.IntentAnalyzer` 是**决策层**而非 Agent（不持状态、不参与消息总线）。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户：导出四个角色类与 `ROLE_*` 常量。 |
| `base.py` | `BaseAgent` 基类：统一的 `role` / 日志 / 会话生命周期钩子（`on_session_start` / `on_session_end`）。 |
| `protocol.py` | 协作协议：角色常量、`AgentMessage`/`AgentBus`（消息→事件扇出+环形缓冲）、调度策略说明。 |
| `nodes.py` | **节点地图**：汇总四个 Agent 的流程节点（`AGENT_NODES` / `BUILDERS`），谁负责哪些节点、连接顺序、依赖配置一处可查；`scripts/render_agent_flow.py` 据此画总览图。 |
| `supervisor/` | 总控（见该目录 README）。 |
| `teacher/` | 答疑（含 ReAct 闭环）。 |
| `lecturer/` | 课件讲解。 |
| `student/` | 学生端代理。 |
| `shared/` | 跨 Agent 共用件（麦控 / 白板 / 提示词装配 / 工具包 / 重点标红 / 回复解析）。 |

## 设计要点

- **不直接互相调用**：协作只走三种机制——共享上下文（`SessionState`，唯一事实来源）、消息（`AgentMessage`/`AgentBus`）、事件（`agent_message` 下发前端，可回溯）。
- **消息总线只记转交与异常路径**（`request`/`handoff`/`fallback`/`error`，环形缓冲 80 条），常规判定痕迹走 `turn.meta["intent_decision"]`，逐回合记录会把缓冲刷满。
- **提示词两层切分**：共享层（`shared/teaching_layers.py`：这个人是谁——身份/人格/风格/经验）+ 任务层（各 agent 自持契约：这一次产出什么），保证「同一节课是同一位老师」，且两套 JSON 契约不互相打架。
- **节点声明与执行同源**：`NodeSpec` 既是文档也是建图输入（`core/graph.py`），改代码重跑脚本即同步，不会出现「图是旧的、代码是新的」。
