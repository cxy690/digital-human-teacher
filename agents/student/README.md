# agents/student — 学生端代理（StudentAgent）

代表学生侧的设备状态与输入：持有本会话的**麦控状态机**，驱动 VAD、生成学生回合（`StudentTurn`），并同步摄像头/麦克风开关状态。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 导出 `StudentAgent`。 |
| `agent.py` | 学生端代理：`set_camera` / `set_microphone`（下发 `status` 事件）、`on_vad`（驱动 `MicController`）、学生回合生成。 |
| `nodes.py` | 节点声明与图构建：`student.voice`（guard_mic → transcribe → guard_transcript → build_turn → dispatch）、`student.control`（camera_off / camera_on / mic_off / mic_on / clear_board）。 |

## 设计要点

- **麦控状态机在 Agent 层**：`MicState`（idle / student_speaking / student_silent / ai_speaking / barged_in）的迁移逻辑委托给 `agents/shared/mic_control.py`，本 Agent 只持有实例并按会话驱动。
- **设备状态即事件**：摄像头/麦克风开关只是 `session.push_status(...)`，前端据 `status` 事件更新徽章，不产生额外信令。
- **与 PRD 学生端功能对齐**：音视频通道开关、语音连麦、文字提问、上传题目图片四类输入都在本模块或由其派发。
