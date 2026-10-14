# 总控（Supervisor）调度策略

> 适用对象：`agents/supervisor/agent.py` 的 `Supervisor` —— 它继承 `agents.base.BaseAgent`（`role = supervisor`），
> 是「总控 Agent」，与它调度/对接的执行者（`StudentAgent` / `TeacherAgent` / `LecturerAgent`）同在 `agents/` 包内。
> 本文只描述**调度策略**（谁在什么时候干什么、出事退给谁），不描述教学内容 —— 讲解风格见
> [`../.agents/skills/`](../.agents/skills/)（一科一份 `SKILL.md`，运行期事实源；
> 目录约定与字段规范见 [`../.agents/notes/README.md`](../.agents/notes/README.md)）。
>
> 定位：**这是给人读的说明书**。运行时的事实来源仍是代码：
> 角色与消息类型见 [`../agents/protocol.py`](../agents/protocol.py)，
> 决策细节见 `agents/supervisor/agent.py` 的函数注释。

---

## 1. 定位与边界

一次会话一个 `Supervisor` 实例，它持有本会话的全部状态与四个执行者：

| 持有的东西 | 说明 |
|---|---|
| `self.session`（`SessionState`） | **唯一共享黑板**：白板指令、历史、麦控、课件页序、`scratch` |
| `self.student` / `self.teacher` / `self.lecturer` | 三个执行者 Agent |
| `self.router`（`core.intent.IntentAnalyzer`） | 意图识别/分流**决策层**（不是 Agent） |
| `self.validator` / `self.parser` / `self.guardrail` / `self.retriever` | 校验 / OCR 结构化 / 学科护栏 / RAG |

**职责边界（与执行者的分工）**

| 谁 | 负责 | 不负责 |
|---|---|---|
| `Supervisor` | 调用顺序、抢占、异常回退、写共享上下文、白板/语音下发 | 具体讲解内容、拼提示词 |
| `TeacherAgent` | 答疑全流程（讲题/图片题/出题判题/追问） | 课件逐页讲解、调度 |
| `LecturerAgent` | 课件逐页讲解 + 页面标注 | 答疑、写正文板书、调度 |
| `StudentAgent` | 麦控状态机、摄像头/麦克风、构造回合 | 任何内容生成 |
| `IntentAnalyzer` | 判意图 + 标复杂/多意图 + 出执行计划 | 执行、讲解 |

> 为什么总控能放进 `agents/`：它是**唯一有状态**的 agent（其余三个只吃参数、吐 `AIResponse`），
> 由 `core/loop.py` / `core/harness.py` / `a2a/executor.py` 装配调用 —— 但这三处都是**单向依赖**：
> `core.*` 不会在运行期反向 import 总控（`core/turn_graph.py` / `core/langgraph_flow.py` 只在
> `TYPE_CHECKING` 下引用），所以不构成循环导入。`agents/__init__.py` 用惰性 `__getattr__`
> 再导出 `Supervisor`：`from agents import Supervisor` 可用，但只取 `TeacherAgent` 的调用方
> 不必连带加载它的重依赖链（`document_parsing` / `models`）。

---

## 2. 输入事件表

上行事件统一由 `core/loop.py` 的 `SessionLoop._dispatch()` 分发到总控：

| 事件 `type` | 总控入口 | 说明 |
|---|---|---|
| `ping` | — | 回 `pong`，不经总控 |
| `vad` | `on_vad(has_voice, ts_ms)` | 逐帧麦控（驱动 `StudentAgent` 的状态机） |
| `voice` | `on_voice(audio)` | 音频 → ASR → 构造回合 → `handle_turn` |
| `text` | `handle_turn(turn)` | 文字提问主入口 |
| `image` | `handle_image(image_id, bytes, text)` | 拍照讲题：OCR/结构化 → 答疑链路 |
| `interrupt` | —（只 `_barge_in`） | 作废当前播报，不产生新回合 |
| `idle` | `on_idle()` | 空闲主动问候（围绕上次提问，有次数上限） |
| `course_start` | `on_course_start(subject, teacher)` | 开课：打招呼 + 阶段计时 |
| `course_phase` | `on_course_phase(phase)` | 阶段切换话术 |
| `quiz_request` | `on_quiz_request(topic)` | 出模拟题 |
| `courseware` | `handle_courseware(deck_id, index)` | 打开课件 |
| `slide_goto` / `slide_step` | `handle_slide_goto/step()` | 翻页 |
| `slide_play` / `slide_stop` | `handle_slide_play/stop()` | 连讲开始/暂停 |
| `slide_ready` | `mark_slide_ready(index)` | 前端回报「本页音频播完」（**就地处理，不入队**） |

> `slide_ready` 必须就地处理：主循环此刻正卡在 `_await_page_played` 等它，排进队列会永远轮不到
> （表现为「每页卡一个超时时长」）。同理，页序 `0` 是合法值，不能用 `int(raw or -1)` 兜底。

---

## 3. 回合决策树（`_handle_turn_builtin`）

两套编排读同一份 `IntentDecision`，行为由单测强绑定一致（`tests/test_intent_analyzer.py` 的
parity 用例）；LangGraph 版对应 `core/turn_graph.py` 的 `route` 节点。

```
handle_turn(turn)
 └─ 待答追问优先：_should_treat_as_answer(turn) → _turn_answer_check()   ← 判题，不重新路由
 └─ 否则 意图分流 decision = router.analyze_async(turn)                   ← 写 turn.meta
      ├─ plan() 里「非控制」诉求 ≥2 → _turn_multi_intent()                ← 逐条作答（见 §4）
      ├─ 混装的 control_clauses → _handle_control()                      ← 当场办
      ├─ intent == CONTROL   → _handle_control() → None
      ├─ intent == CHITCHAT  → _turn_chitchat()（不进答疑链路）
      └─ 其余                → _turn_teach() → _run_answer_pipeline()
                                └─ 答疑子图 retrieve → generate → validate → solve
                                   （generate 内按 IMAGE_QUESTION 走 teacher.answer_image）
```

**复杂（非多意图）不拆分**：走同一条 `_turn_teach`，差别只是 `delegation_notes()` 生成的
「先拆解条件再逐层推导」注入 `TeacherAgent` 的 `context_notes`。

---

**这棵树已节点化**：`agents/supervisor/nodes.py` 的 `route` 节点做分流，其余分支节点
（`answer_check` / `multi_intent` / `control` / `chitchat` / `teach`）不命中时返回
`{"__skip__": True}` 跳过；答疑子图的 `retrieve → generate → validate → degrade → solve`
在 `agents/teacher/nodes.py`，讲课件链路在 `agents/lecturer/nodes.py`，
麦控与设备控制链路在 `agents/student/nodes.py`；四个角色的节点清单汇总见 `agents/nodes.py`
（`python -m agents.nodes` 可直接打印「谁负责哪些节点 + 连接顺序 + 依赖配置」）。

---

## 4. 多意图策略

触发条件：`plan()` 里**非控制类诉求 ≥2 条**（只有 1 条提问的多意图走常规单意图链路 —— 硬拆反而多一次播报）。

1. **控制指令先办且不出声**：顺序固定为「控制 → 提问」，先把设备状态摆正，避免打断后面的讲解；
2. **提问逐条独立走完整链路**：学科护栏、RAG、白板、追问**按条生效** ——
   「数学题…另外帮我看看英语作文」里的英语那条会被单独拦下，而不是混在一段讲解里；
3. **子回合不继承上下文**：`meta` 只带 `from_voice`，不带上一题的 `intent_notes`/理解（避免串味）；
4. **追问只在最后一条之后**（`ask_followup = index == len(questions) - 1`）：否则
   `session.pending_question` 会被后一题覆盖，学生答了也判不上；
5. 白板按条推**全量**（前端先清板再渲染，天然正确），语音按条依次播报。

---

## 5. 抢占与打断

| 场景 | 机制 |
|---|---|
| 学生新输入 | `submit()` **入队瞬间**即 `_barge_in()`：`speech_epoch += 1` + 下发 `avatar_interrupt`，旧播报在下一个句边界中止。分发时会再调一次（双保险，`speech_epoch` 会多自增一次） |
| 讲课中学生提问 | `_record_lecture_handoff()` 记一条 `handoff`（`lecturer → teacher`），去重键是 `speech_epoch`（多意图逐条作答每轮只记一次）；判定依据是「有没有打开课件」，不是 `auto_playing` |
| 课程到点 | `interrupt` 事件只作废播报，不产生新回合 |

---

## 6. 失败与回退

| 场景 | 处理 |
|---|---|
| 讲课 LLM 失败 / 返回非法 JSON | `LecturerAgent` 降级为「本页要点口播」，绝不哑巴翻页（记 `response ok=false`） |
| 讲课链路整体抛异常 | `_explain_slide` 回退 `TeacherAgent.explain_slide`（等价实现，共用同一 llm/白板），记 `fallback` |
| 回退也失败 | 记 `error` + 返回兜底话术 —— 不因为一页异常打断整节课 |
| 意图小模型超时 / 报错 / JSON 不合法 | `IntentLlmClassifier` 返回 `None` → 保留规则结果，**绝不阻塞答疑** |
| RAG 检索失败 | 记 warning 后忽略，答疑退化为纯 LLM |
| 白板指令越界 | `ResponseValidator` 只丢弃有问题的白板指令，**不整块清屏**（避免「讲了半天板上一片白」） |
| 图片损坏 / OCR 失败 | `QuestionParser.parse` 返回空 `raw_text`，链路继续 |
| 后台任务（技能复盘） | fire-and-forget，进 `self._bg_tasks`；失败只记日志 |

---

## 7. 生命周期与状态所有权

```
SessionLoop.run()      → on_session_start(session)   # 初始化 StudentAgent + push ai_running
WebSocket finally      → on_session_end(session)     # 停连讲、清播报态、cancel 后台任务
```

* `on_start()` 保留为兼容入口（等价于 `on_session_start(self.session)`），历史调用方是
  `SessionLoop` / 单测 / `scripts/run_demo.py`；
* `SessionState` 由总控**独占**：其它 agent 只通过参数拿同一实例，不持有、不缓存；
* `ASR / TTS / Avatar / LLM` 客户端是进程级单例（`models/*.py` 的 `@lru_cache`），跨会话共享；
* 会话表 `session_store` 是进程级单例（多实例部署需换 Redis，见 `待补齐规划.md` P2）。

---

## 8. 可观测性

| 通道 | 内容 |
|---|---|
| `AgentBus`（`session.scratch["agent_trace"]`，环形 80 条） | 只记**转交与异常路径**：`request`（总控→讲课）、`handoff`（讲课→答疑）、`fallback`、`error` |
| 客户端事件 `agent_message` | 上述消息的实时扇出（前端调试面板可用） |
| `turn.meta["intent_decision"]` | 每回合的意图判定快照（`IntentDecision.to_meta()`，可序列化） |
| `turn.meta["intent_notes"]` | 转交上下文（多意图子问题清单 / 复杂题拆解要求） |
| `session.scratch` | 会话级临时位（如 `_lecture_handoff_epoch`） |

> **刻意不记逐回合调度**：每个回合都记一条 `supervisor → teacher` 的 `request` 会把 80 条环形缓冲刷满，
> 排障价值反而下降。常规判定的痕迹走 `turn.meta`。

---

## 9. 非目标（明确不做什么）

1. **不生成话术**：所有文本都由 `TeacherAgent` / `LecturerAgent` 产出，总控只负责下发与播报顺序；
2. **不判学科**：越界/内容安全由 `core/guardrail.py`（`TeachingGuardrail`）负责；
3. **不解析题目**：OCR 与结构化在 `document_parsing/pipeline.py`；
4. **不占用模型额度做判定**：学科护栏、白板校验、验算都是本地逻辑；
5. **不互相调用 agent**：agent 之间零直接调用，协作只靠「共享 `SessionState` + `AgentBus` 记录 + 上下文注入」。

---

## 10. 相关文件索引

| 关心什么 | 去哪看 |
|---|---|
| 角色、消息类型、三种通信机制 | `agents/protocol.py` |
| 总控全部入口与内部实现 | `agents/supervisor/agent.py` |
| 事件分发与打断时机 | `core/loop.py` |
| 意图判定（含多意图拆分） | `core/intent/analyzer.py` |
| 两套编排（内建 / LangGraph） | `agents/supervisor/agent.py:_handle_turn_builtin` · `core/turn_graph.py` |
| 各角色负责的流程节点（名称/功能/输入输出/连接顺序/依赖配置） | `agents/nodes.py` → `agents/<角色>/nodes.py` |
| 装配与会话工厂 | `core/harness.py` |
| 整体调用链与目录树 | `代码地图.md` |
