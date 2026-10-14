# clients — 前端（原生 HTML / Canvas，零构建、与服务同源）

全部页面为原生 HTML + Canvas + 原生 JS 单文件实现，**无构建流水线**；所有网络地址为相对路径，任何域名或隧道下均可直接运行。由 `server/app.py` 挂载到 `/app`（含根路径兜底），页面与静态资源均设 `no-store`（改完刷新即生效，不会加载到旧版）。

## 页面一览

| 页面 | 路径 | 用途 |
| --- | --- | --- |
| `index.html` | `/app/index.html` | 入口导航页：产品定位说明 + 学生端 / 监控台两个入口按钮（纯静态，无 JS）。 |
| `subject_select/` | `/app/subject_select/`（根路径 `/` 也指向它） | 学科选择页：11 个学科卡片，同科多老师时弹风格浮层，选定后跳转 1v1 辅导页。 |
| `student_tablet/` | `/app/student_tablet/` | **1v1 辅导主前端**：数字人 + 白板 + 麦控 + 课件 + 出题判题（见该目录 README）。 |
| `login/` | `/app/login/` | 登录 / 注册页（JWT 凭据存入 `localStorage["edu_auth"]`）。 |
| `teacher_web/` | `/app/teacher_web/` | 监控控制台（展示可用工具清单）。 |

## 设计要点

- **鉴权贯穿全站**：登录页写 `localStorage["edu_auth"] = {token, user}`；学科页 / 学生端读同一 key。HTTP 走 `Authorization: Bearer`，WebSocket 因浏览器不能自定义 Header 而走 `?token=`（见 `student_tablet` 的 `WS_PATH`）。
- **白板三层模型**：背景图（课件页图 / 学生题目图）+ Canvas 标注层 + DOM 文字层，标注坐标全为 0~1 归一化。
- **数字人双缓冲**：讲解视频与待机画面用双缓冲交叉淡化切换；视频不可用时回退照片木偶 + 音频能量驱动口型。
- **零构建的代价与纪律**：JS 内联在页面里，改前端的自检是 `node scripts/check_frontend_js.mjs clients/<页面>/index.html`（语法解析 + HTML 注释配对检查）。
