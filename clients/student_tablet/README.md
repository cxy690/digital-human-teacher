# clients/student_tablet — 1v1 辅导主前端

单文件页面（`index.html`，约 2500+ 行）：本项目的**主前端**。学生在此完成「开始上课 → 提问（语音/文字/拍照）→ 看讲解与板书 → 互动出题 → 总结」全流程。

## 页面骨架（DOM 分区）

| 分区 | 内容 |
| --- | --- |
| 顶栏 `header` | 学科 chip、状态徽章组（连接 / 麦控 / AI / 表情）、摄像头与麦克风开关按钮、前端版本标记。 |
| `#net-banner` | 非 HTTPS 降级横幅（HTTP 下麦克风/摄像头不可用，引导改用文字提问）。 |
| 主区 `section.stage` | **左侧白板**（绿框）—— `canvas#board` 标注层 + `img#board-slide` 课件页图 + `img#board-photo` 题目图 + `#board-text` 讲解区 + 可拖动字幕条；**右侧竖列**——上方 AI 教师数字人（蓝框）、下方学生端摄像头（紫框）。 |
| 课件区 | 选课件下拉（按年级/章节分组）→ 翻页 → AI 按页讲解与圈画。 |
| 弹层 | 文字/拍照上传菜单、视频讲解录制弹窗、开始上课闸门 `start-gate`（一次点击同时解锁语音与开课）。 |

## JS 分区（行号即定位锚）

| 行（约） | 分区 |
| --- | --- |
| 360 | 启动配置：`AUTH_TOKEN` 读 `localStorage.edu_auth`；`WS_PATH = ws(s)://host/ws?token=…`；`SUBJECT`/`TEACHER` 取自 URL；HTTPS 判定。 |
| 384 | 1v1 辅导课状态（学科 / 阶段 / 倒计时）。 |
| 404 | **白板渲染**：canvas 重绘、`op==="slide"` 换页清标注、`renderEmphasis` 处理 `**…**` 标红、页图加载后重放标注。 |
| 453 | 讲解字幕条：上下拖动决定看多少，比例存 localStorage。 |
| 590 | 课件讲解：清单 / 当前课件高亮（`/api/courseware`）。 |
| 660 | 断点续讲：上报本页已播进度。 |
| 912 | **WebSocket**：下行事件分发（`status` / `board_ops` / `slide_state` / `chat_message` / `ai_speech` / `ai_avatar`…）；上行 `{type:"interrupt"}` 打断。 |
| 1080 / 1121 / 1152 | 互动追问作答卡片、辅导课模拟题卡片、卡片显示时机（等老师讲完再出现）。 |
| 1173 / 1220 | AI 数字人讲师：待机帧序列 + 讲解状态特效；音频能量驱动口型 + 视素调制 + 眨眼。 |
| 1332 / 1365 / 1376 | 浏览器朗读音色按老师性别挑选；语音播报优先真实音频；`ai_avatar` 视频逐句队列播放，播完回照片木偶。 |
| 1692 / 1761 / 1781 | 视频讲解录制（数字人 + 白板 + AI 语音）；摄像头；图片上传（等比压缩后传）。 |
| 1908 / 1974 / 2142 | 语音提问与实时对话（录音 → WAV → 服务端识别）；文字提问。 |
| 2152 / 2490 | 课程课时展示与计费；启动 / 空闲问候（`/api/idle/policy`）。 |

## 后端接口汇总

`GET /api/teacher?subject=`（老师形象与性别）、`GET /api/courseware`、`GET/POST /api/course/*`（`policy|status|bills|close|start`）、`GET /api/idle/policy`、WS `/ws`。

## 设计要点

- **不写死老师素材**：开课时由 `/api/teacher` 按学科下发，换老师前端零改动。
- **数字人双路径**：优先 `ai_avatar`（服务端逐句渲染视频，队列串联播放），不可用时退化为浏览器 TTS + 能量驱动口型 + 眨眼；过渡帧保证待机 ↔ 讲解切换不闪断。
- **白板渲染容错**：页图 `contain` 留白会偏移红框，页图 `load` 后统一 `redrawHint()` 重放；翻页（`slide` 指令）清空上一页标注。
- **鉴权走 query**：浏览器 WS 无法自定义 Header，故 `ws://…/ws?token=<JWT>`。
- **已注释停用的区块（保留可恢复）**：课程倒计时 chip / 续费结算弹窗（后端 `/api/course/*` 未动，恢复时去注释即可）。
