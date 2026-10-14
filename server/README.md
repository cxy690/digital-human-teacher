# server — FastAPI 应用接入层

对外的唯一 HTTP / WebSocket 入口：挂载 REST 路由、WS 端点、MCP 与 A2A 协议端点、前端静态页面与媒体资源。启动方式：`python -m server.app`（或 `education-server` 命令，端口见 `.env` 的 `PORT`）。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包声明。 |
| `app.py` | 应用装配：项目根入 `sys.path` 并固定工作目录（三种运行方式都能读到 `.env` 与 `data/`）；lifespan 里做底片校验、建库建表、数字人子服务托管；挂载顺序有讲究——`app.mount("/")` 静态兜底必须最后注册。 |
| `routes.py` | REST 路由（`/health`、`/api/*`）：认证、学科/老师/人格、课件、课程与计费、记忆、经验、Agent 技能（清单/抽检/停用）、数字人素材、工具清单等。 |
| `websocket.py` | WS 端点 `/ws`：进程级装配根 `harness = EducationHarness()`；握手 `init`（学科/老师/令牌）→ `SessionLoop.submit`；出站由独立 sender 任务异步下发；`?token=` 鉴权。 |

## 设计要点

- **薄接入层**：本目录只做「解析请求 → 委托 Harness/SessionLoop → 序列化响应」，不含业务逻辑；`server` 只依赖 `core.harness`，不直接 new 任何模型/agent 实现。
- **启动不因外部依赖而失败**：底片校验失败打 WARNING 继续、MySQL 连不上降级（登录不可用）、数字人子服务拉起失败只记日志。
- **静态资源强制禁缓存**（`NoCacheStaticFiles`）：前端 JS 内联在 HTML 里，`no-store` 避免「改了代码却仍加载旧页面」。
- **WS 出站解耦**：`SessionState.emit` 回调是同步的、WS 发送是异步的，用发送队列 + sender 任务桥接，避免事件在回调里被丢弃。
- **挂载顺序即路由优先级**：`/health`、`/api`、`/ws`、A2A/MCP 端点在 `app.mount("/")` 之前注册，否则会被静态托底吞掉。
