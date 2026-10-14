# agents/supervisor — 总控（Supervisor）

唯一**持有会话状态**的智能体：决定调用顺序、处理抢占与学生插话、承担异常回退。被 `core/loop.py` / `core/harness.py` 装配与分发，是「一条 WebSocket 连接 → 一个 Supervisor 实例」的入口。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 导出 `Supervisor`。 |
| `agent.py` | 总控实现（项目最大单文件）：回合决策、答疑/图片题/课件讲解调度、流式与白板先行链路、出题判题编排、TTS/数字人下发、学情记忆与经验写入、辅导课阶段推进。 |
| `nodes.py` | **builtin（自研 TaskGraph）版**的编排图构建：`supervisor.turn`（route → control / chitchat / teach / multi_intent）、`supervisor.slide`（present_page → explain → annotate → speak → push_audio）、`supervisor.slide_prepared`。 |

## 设计要点

- **状态所有权独占**：`SessionState` 由 Supervisor 独占；其它 Agent 只通过参数拿同一实例，不持有、不缓存。
- **回合决策顺序**（`_handle_turn_builtin`）：待作答追问优先 → 意图分流 → 非控制诉求 ≥2 条逐条作答 → 混装控制指令当场执行 → 控制 / 闲聊 / 答疑。两套编排（内建 TaskGraph 与 LangGraph 主图，见 `core/turn_graph.py`）读同一份 `IntentDecision`，行为由单测（`tests/test_graph_parity.py`）强绑定。
- **抢占**：学生新输入**入队瞬间**即自增 `speech_epoch` 并下发 `avatar_interrupt`（见 `core/loop.py`）；讲课中学生提问记一次 `handoff`，让位给 TeacherAgent。
- **回退链**：讲课失败 → `TeacherAgent.explain_slide` → 兜底话术；意图小模型超时/非法 → 保留规则结果；RAG 失败静默忽略；数字人渲染失败 → 照片木偶。
- **讲解口径三档**：标准管线（默认）/ 流式（接数字人时的 `_turn_teach_streaming`，话术边生成边逐句播报）/ 白板先行（`BOARD_FIRST_MODE=single_call|two_call`，先写题意理解再给解答）。
- **不记逐回合调度**：`AgentBus` 只记转交与异常路径；常规痕迹走 `turn.meta`。调度策略全貌见 `docs/supervisor_orchestration.md`。
