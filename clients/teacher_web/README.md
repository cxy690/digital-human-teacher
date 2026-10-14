# clients/teacher_web — 监控控制台

单文件页面（`index.html`，极简）：展示服务端当前可用的工具清单（MCP 风格），用于快速确认「服务起来了、工具注册正常」。

## 页面结构

- `<header>`：标题 + 返回首页链接。
- `<table>`：工具表（工具名 + 描述），由 `load()` → `GET /api/tools` 逐行插入。

## 设计要点

- **失败提示明确**：请求失败时页内提示「无法连接服务端，请先启动 education-server」，避免出现空白页让人误判。
- **定位是运维自检入口**，不是教学界面；教学监控（白板指令流、协作轨迹）走 `agent_message` 事件与 `logs/activity/`（`scripts/view_activity.py`）。
