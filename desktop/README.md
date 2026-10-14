# AI 教师课堂 · 桌面客户端（Electron → .exe）

## 一、为什么这样设计

学生端页面（`clients/student_tablet/index.html`）里所有网络地址都是**相对路径**：

- `fetch("/api/teacher?subject=...")`
- WebSocket：`ws(s)://` + `location.host` + `/ws`

如果 exe 用 `file://` 加载本地页面，`location.host` 为空，**所有接口与 WebSocket 全部失效**。

所以本客户端**加载远程课堂页面**（云端的 `/app/subject_select/`），让浏览器按同源规则自动解析地址：

- 前端代码 **零改动**
- 云端更新页面后，客户端**自动生效**（不用重新打包）

## 二、目录结构

```
desktop/
├── package.json    # Electron + electron-builder 配置，含打包命令
├── main.js         # 主进程：读配置 → 打开课堂页 / 首次弹出设置页
├── setup.html      # 服务器地址设置页（首次运行或配置为空时出现）
├── config.json     # 服务器地址（可改）
└── README.md
```

## 三、本地试运行

```powershell
cd desktop
npm install
npm start
```

> 若本地已开 SSH 隧道，可把 `config.json` 的 `serverUrl` 改成 `http://127.0.0.1:8001`。

## 四、打包成 .exe

```powershell
cd desktop
npm install
npm run build
```

产物在 `desktop/dist/`：

- `AI教师课堂 Setup x.x.x.exe` —— 安装版（推荐分发）
- 或 `build:dir` 生成免安装目录

打包后的 `config.json` 位于 **exe 同目录**，可直接编辑改服务器地址；删除 `serverUrl` 则会重新弹出设置页。

## 五、服务器地址怎么填

| 场景 | serverUrl | 说明 |
| --- | --- | --- |
| 你自己用（SSH 隧道） | `http://127.0.0.1:8001` | 需先运行 `python _tunnel.py` |
| 公网分发 | `https://<固定域名>` | **必须固定地址** |
| 局域网/内网部署 | `http://192.168.x.x:8001` | 同一内网即可 |

## 六、分发前的前置条件（重要）

打包给学生/客户前，**必须有一个稳定的服务器地址**：

- ngrok 免费版会 `ERR_NGROK_725`（额度用尽），且免费域名重启即变 → 不适合分发
- cloudflared 临时地址每次重启会变 → 不适合分发
- **推荐：自有域名 + Cloudflare Tunnel 固定命名隧道**，或 ngrok 付费绑定域名，或客户内网部署

## 七、可选：换成内嵌页面（离线 UI）

若希望 exe 自带页面、不依赖网络加载 UI，需要改前端：
把 `WS_PATH` 与所有 `fetch()` 改成基于可配置基址（如 `window.API_BASE`），
再把页面 `loadFile` 加载。改动量较大，且云端更新页面后客户端不会同步，
因此**不推荐**，除非确需离线 UI。
