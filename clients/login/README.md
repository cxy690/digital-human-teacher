# clients/login — 登录 / 注册页

单文件页面（`index.html`）：登录或注册后把 `{token, user}` 写入 `localStorage["edu_auth"]`，作为**全站鉴权凭据**（学科页与学生端读同一 key）。

## 页面结构

| 分区 | 说明 |
| --- | --- |
| `#panel-auth` | 未登录面板：登录 / 注册两个 tab + 表单 + 演示账号提示。 |
| `#panel-logged` | 已登录面板：展示当前用户，提供退出与跳转入口。 |

JS 分区（按代码顺序）：`readAuth / writeAuth / clearAuth`（凭据读写）→ `setMode(login|register)`（切换 tab）→ `submit()`（提交表单）→ `renderLogged()`（渲染已登录态）→ `refresh()`（用 `/api/auth/me` 校验令牌，失效即清除）。

## 后端接口

| 接口 | 用途 |
| --- | --- |
| `POST /api/auth/login` | 登录，返回 token + 用户信息。 |
| `POST /api/auth/register` | 注册（`AUTH_ALLOW_REGISTER=false` 时关闭）。 |
| `GET /api/auth/me` | 校验当前令牌是否有效。 |

## 设计要点

- **`?next=` 参数**：登录成功后跳回来源页面，实现「辅导页 → 登录 → 回辅导页」的闭环。
- **令牌失效即清除**：`refresh()` 检测 401 会清掉本地凭据并切回未登录面板，避免带着过期令牌打一堆 401。
- **历史 bug 教训（注释保留在页内）**：退出登录时只切了 tab 没切面板，导致「退出后面板不刷新」——改 UI 状态时两个面板必须同步切换。
