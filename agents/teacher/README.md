# agents/teacher — AI 答疑智能体（TeacherAgent）

答疑链路的主角：讲题、图片题讲解、白板书写、追问判定、出题判题（含真题库选题与三级降级）。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 导出 `TeacherAgent`。 |
| `agent.py` | 答疑实现：提示词装配与 JSON 契约解析、单次生成（`_answer_plain`）、ReAct 闭环入口（`answer_with_react`）、流式话术（`speech_stream`）、白板先行两段式（`stream_board_first`）、图片题圈画（`answer_image`）、出题（`generate_quiz`，真题库 → LLM → 规则三级降级）、判题与追问。 |
| `react.py` | ReAct 闭环（「思考 → 行动 → 观察」）：驱动模型自主调工具、收敛判定与步数/超时上限。 |
| `nodes.py` | 本 Agent 的节点声明与图构建：`teacher.answer`（retrieve → generate → validate → degrade → solve）、`teacher.judge`（evaluate → decide → announce）、`teacher.quiz`（generate → publish → announce）。 |

## 设计要点

- **提示词来自共享层**：身份/人格/风格/学科约束由 `agents/shared/teaching_layers.py` 统一装配，本 Agent 只注入自己的任务契约（speech / steps / knowledge_points / pitfalls / answer / followup）。
- **作业模式分层提示**：`hint_level`（概念→方法→步骤→答案）会替换默认讲解风格（`_HINT_STYLE`），由 Supervisor 的 `_resolve_hint_level` 逐层升级。
- **流式与结构化并行**：`_turn_teach_streaming` 下话术路逐句入队直通 TTS，结构路（完整 JSON 契约）并行生成，避免长 JSON 阻塞首响。
- **出题三级降级**：真题库（`core/exercise_bank.py`）→ LLM 即兴生成 → 规则兜底选择题，任何一级失败都不中断课堂。
- **重点标红**：话术中的 `**...**` 标记由 `agents/shared/emphasis.py` 剥离后再进 TTS；白板/聊天文字保留标记供前端渲染。
