# a2a — A2A（Agent2Agent）协议接入

把本系统的答疑能力以 A2A 协议对外暴露：外部 agent 先读 Agent Card 发现能力，再用 JSON-RPC 2.0 发起答疑任务、查询产物、取消任务。

挂载在 `server/app.py`（在静态托底挂载**之前**注册，否则会被 `/` 吞掉）：

| 路径 | 说明 |
| --- | --- |
| `GET /.well-known/agent-card.json` | Agent Card（A2A 0.3.x 发现路径） |
| `GET /.well-known/agent.json` | 旧路径（A2A 0.2.x，同一份卡片） |
| `POST /a2a` | JSON-RPC 2.0：`message/send` · `tasks/get` · `tasks/cancel` |

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户。 |
| `card.py` | 构建 Agent Card（能力声明 + 三个 skill：答疑讲解 / 意图识别拆分 / 白板板书）。 |
| `models.py` | 协议数据模型：`AgentCard` / `Task` / `Message` / `Part` / `Artifact` 等（pydantic）。 |
| `errors.py` | JSON-RPC 2.0 标准错误码与 `error_payload()`，与 `mcp_tools/protocol.py` 同构。 |
| `executor.py` | 任务执行器 `SupervisorAgentExecutor`：把 Supervisor 一次回合包成 A2A Task。 |
| `rpc.py` | JSON-RPC 分发层：`handle_rpc()`（支持单条与批量请求）。 |
| `store.py` | 内存任务表 `A2ATaskStore`（TTL 1 小时 + 上限 2000，防长跑进程只增不删）。 |
| `routes.py` | FastAPI 路由 + 进程级执行器懒建（优先复用 `server.websocket.harness`，与 WS 共用装配根）。 |

## 设计要点

- **不改 Supervisor 一行代码**：执行时临时接管 `SessionState.emit` 收集事件，回合结束翻译成 A2A Artifact —— `ai_speech` → 文本 Part，`board_ops` → data Part，`ai_ask`/`ai_quiz`/`ai_check` → data Part。
- **任务状态映射答疑场景**：讲完留下待作答追问 → `input-required`（再发同一 `taskId` 的消息即续跑判题链路）；正常结束 → `completed`；异常 → `failed`。
- **会话隔离**：一个 `contextId` 一个独立会话，session_id 前缀 `a2a-`，**与前端 WebSocket 会话完全隔离** —— 外部 agent 调进来不会打断学生浏览器里的课堂。
- **卡片与实现对齐、不吹牛**：只实现阻塞式 `message/send`，`capabilities.streaming=false` 如实声明（流式与 push notification 未实现）。
