# docs — 架构与设计文档

承接 README「技术架构」章节的深度设计说明与架构图。与 `md_docs/`（部署 / 联调 / 运维类）分工：本目录放**设计理念与方法论**。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `supervisor_orchestration.md` | 总控调度策略全貌：调用顺序、抢占、回退、状态所有权。 |
| `agent_autonomy.md` | Agent 自主性四层（工具调用 / 自主规划 / ReAct 闭环 / 长期记忆）：开关矩阵、取舍（要自主性还是极限首响）、各层降级路径。 |
| `architecture.png` / `architecture.html` | 完整项目架构图：分层总览 · Agent 调度关系与协作 · 学科老师 Skill 加载链路；`.html` 为可交互源文件。 |
| `agent_flow.png` | 「4 个 Agent / 9 条流程线 / 40 个节点」总览图，由 `python scripts/render_agent_flow.py` 渲染。 |

## 设计要点

- **图随代码走**：`agent_flow.png` 的节点清单是从 `agents/nodes.py::describe_agent_nodes()` 读出来的——代码改了重跑一次脚本即同步，不会出现「图上是旧的、代码是新的」。
- **文档与开关一一对应**：`agent_autonomy.md` 的每个开关都能在 `.env.example` 找到同名变量与降级行为说明。
