# clients/subject_select — 学科选择页

单文件页面（`index.html`，约 700 行）：动态渲染学科卡片；点开一张卡后，若该学科有多位老师则弹出「选一位老师的风格」浮层，选定后带 `?subject=&teacher=` 跳转 1v1 辅导页。也是根路径 `/` 的默认入口页。

## 页面结构

| 分区 | 说明 |
| --- | --- |
| `.auth-entry` | 右上角账号区（读 `localStorage["edu_auth"]`，未登录显示登录入口）。 |
| `header` / `main.grid#grid` | Hero 区 + 学科卡片网格（鼠标跟随聚光效果）。 |
| `#picker` | 同科多老师风格浮层（单老师的学科直接进课堂，不弹层）。 |

JS 分区：`FALLBACK`（11 门学科兜底清单，与 `config/subjects.py` 对齐，接口不可用时页面仍可展示）→ `INFO_CARDS` → `render(subjects)` → `picker`（`renderPicker` / `openTeacherPicker` / `goClass` / `closePicker`）→ `initAuthEntry()`。

## 后端接口

| 接口 | 用途 |
| --- | --- |
| `GET /api/subjects` | 学科清单（含每科的教师人格摘要）。 |
| `GET /api/teachers?subject=<key>` | 该学科可选老师列表（`multi=false` 时不弹层）。 |
| `GET /api/auth/me` | 右上角账号区状态。 |

## 设计要点

- **学科清单以接口为准、兜底写死**：`FALLBACK` 与 `config/subjects.py` 对齐，`/api/subjects` 失败时页面不白屏。
- **同科多老师的入口收敛在浮层**：`/api/teachers` 返回 `multi` 字段，前端据此决定是否弹出风格选择——单老师学科保持旧行为（点击直进课堂）。
- **跳转协议**：`../student_tablet/index.html?subject=<中文名>[&teacher=<key>]`，中文名会被后端归一化。
- **非 HTTPS 提示**：HTTP 上下文下浏览器禁用麦克风/摄像头，页面会提示（文字提问不受影响）。
