# agents/lecturer — AI 讲课智能体（LecturerAgent）

课件讲解的主角：按页讲解 PPT，**只做「讲 + 指」**（逐页话术 + 页面元素圈画/高亮）。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 导出 `LecturerAgent`。 |
| `agent.py` | 讲课实现：吃 `(session, deck, index)` 吐 `AIResponse`（话术 + 标注指令）；`LECTURE_CONTRACT` 契约装配；标注面积上限（过大不标，避免盖章式遮挡）。 |
| `nodes.py` | 节点声明与图构建：`lecturer.slide`（request → explain → fallback → degrade）。 |

## 设计要点

- **纯函数式**：不直接调用 TeacherAgent、不改会话状态（翻页 / 白板 push / 事件下发都由 Supervisor 负责）——可预测、可丢弃、可回退。
- **职责边界（必须守住）**：

  | 谁 | 负责 | 不负责 |
  | --- | --- | --- |
  | LecturerAgent | 课件逐页讲解话术、本页要点、页面元素圈画/高亮 | 答疑、出题判题、正文板书、作业模式分层提示、追问判定 |
  | TeacherAgent | 答疑全流程 | 课件讲解（保留兼容转发） |
  | Supervisor | 调用顺序、抢占、异常回退 | 具体讲解内容 |

- **契约隔离**：讲课时不注入答疑契约与答疑历史，只从共享层取人格/风格/学科（保证「同一位老师」）；拆分前两套契约揉在同一 system 里互相打架，是这次拆分的直接动因。
- **标注有边界**：坐标全部 0~1 归一化、引用 `SlideElement` 元素 id/文字（不让模型凭空猜坐标）；面积超过阈值宁可不标。
