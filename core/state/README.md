# core/state — 会话状态与事件模型

贯穿 core / agents / server 的数据模型与状态容器：事件模型（`AIResponse` / `StudentTurn` / `WhiteboardOp`…）、单会话可变状态（`SessionState`）、多会话仓库（`SessionStore`）。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户：导出事件模型 + `SessionState` / `SessionStore`。 |
| `models.py` | pydantic 数据模型：`AIResponse`、`ChatMessage`、`StudentTurn`、`WhiteboardOp`、`QuizItem` 等。 |
| `session.py` | 单会话可变状态容器：麦控 / 白板 / 课件页码 / 历史 / 教学追问 / 1v1 课程（`subject`、`teacher`、`course_phase`、`quiz`）；`speech_epoch` 学生插话即 +1 作废未播完播报。 |
| `store.py` | 会话仓库：线程安全（`threading.Lock`）管理多个 `SessionState`；模块级 `session_store` 单例与 server 同生命周期。 |

## 设计要点

- **唯一共享黑板**：所有 Agent 读写同一个 `SessionState` 实例（不拷贝），它是会话的唯一事实来源；由 Supervisor 独占持有。
- **坐标归一化**：`WhiteboardOp` 全部使用 0~1 归一化坐标，前端按画布尺寸换算；`BoardOp.SLIDE` 翻页时前端据 `deck_id` 丢弃过期标注。
- **「画错比不画更糟」**：`StudentTurn.image_region` 非法/缺失时返回 `None`，下游不画框。
- **作废机制简单可靠**：`speech_epoch` 计数器——学生一插话 +1，正在播的旧回合检测到 epoch 变化即停止，不做复杂的状态机。
- **单机可换分布式**：`SessionStore` 内存实现开箱即用，多实例部署可平滑换 Redis。
