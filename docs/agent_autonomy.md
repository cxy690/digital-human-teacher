# Agent 自主性：工具调用 / 自主规划 / ReAct 闭环 / 长期记忆

> 适用对象：`agents/shared/toolkit.py`、`agents/teacher/react.py`、`core/planning/`、
> `core/memory/`，以及把它们接进链路的 `agents/supervisor/agent.py` 与
> `agents/teacher/agent.py`。
>
> 本文只讲**自主性这四层怎么接进来的、开关在哪、代价是什么**；原有的角色分工与调度策略仍见
> [`supervisor_orchestration.md`](supervisor_orchestration.md)。

---

## 0. 为什么要加这四层

加装之前，答疑链路的形态是**确定性流水线 + LLM 作为内容生成器**：

```
retrieve(RAG 写死在图上) → generate(一次生成 JSON) → validate(格式校验) → solve(事后验算)
```

这条链路的三个结构性限制：

1. **顺序写死**：模型没有机会决定「这题要不要查教材」（`retrieve` 节点无条件先跑）；
2. **没有回头路**：`validate` 只校验格式，模型算错了没有任何一步会让它重看自己的结论；
3. **不带学生记忆**：每节课都是第一次见这位学生，错过的题下周还会再错。

四层能力分别针对这三点（长期记忆针对第三点）：

| 层 | 模块 | 开关 | 解决什么 |
|---|---|---|---|
| 工具调用 | `models`（`chat_with_tools`）+ `agents/shared/toolkit.py` | `AGENT_TOOLS_ENABLE` | 模型自己决定查不查、算不算 |
| 自主规划 | `core/planning/planner.py` | `AGENT_PLANNER_ENABLE` | 开口前先定「这题怎么讲」 |
| ReAct 闭环 | `agents/teacher/react.py` | `AGENT_REACT_ENABLE` | 思考 → 行动 → 观察 → 再思考，错了能改 |
| Agent 技能 | `core/skills/agent_skills/` | `AGENT_SKILLS_ENABLE` | 模型自主调用带判断的业务能力，且能自我更新 |
| 长期记忆 | `core/memory/` | `MEMORY_ENABLE` | 跨会话记住这位学生哪儿不会 |

---

## 1. 工具调用（function calling）

### 模型怎么拿到工具

`OpenAICompatLLM.chat_with_tools()` 走标准 OpenAI 的 `tools` / `tool_choice`，
把 `agents/shared/toolkit.py` 给出的工具清单交给模型：

| 工具 | 作用 | 复用自 |
|---|---|---|
| `knowledge_search` | 查教材 / 题库片段 | `mcp_tools.registry`（MCP 对外暴露的是同一份实现） |
| `math_solve` | sympy 解方程 | 同上 |
| `math_verify` | sympy 复核等式 / 答案 | 同上 |
| `ocr_parse` | 按 `image_id` 取会话里已识别好的题干 | 同能力，接口改成 JSON 友好（不传二进制） |
| `read_board` | 读当前白板内容 | 会话状态（`board_ops`） |
| `whiteboard_note` | 追加一行补充板书 | 布局器 `Whiteboard`（只给内容，不给坐标） |

**刻意不暴露坐标级白板工具（`whiteboard_write` / `whiteboard_clear`）**：
板书版式由确定性布局器负责（行距、防重叠、归一化坐标），让模型直接下发坐标只会把版面写乱。
板上「内容」仍由模型的 JSON 契约产出，工具侧只允许追加补充行。
MCP 对外仍暴露这两个工具（外部 agent 有自己的版面），只是本项目的答疑链路不用它们。

### 降级（三条，缺一不可）

1. **供应商不支持函数调用**：`BaseLLMClient.chat_with_tools` 的默认实现直接退化为
   普通对话，`tool_calls` 恒为空 —— 调用处不需要写 `if`；
2. **工具执行失败**：`AgentToolkit.call` 把异常收敛成一段「观察」文本回给模型
   （「工具执行失败（…）。可以换一种方式，或直接用已有知识作答。」），不打断整节课；
3. **调用次数超限**：`AGENT_TOOL_MAX_CALLS`（默认 6）之后一律返回「已达上限，请直接作答」。

### 可观测

| 通道 | 内容 |
|---|---|
| `turn.meta["react"]["trace"]` | 每一步的思考 / 工具名 / 参数 / 观察（截断后） |
| `turn.meta["react"]["tool_calls"]` | 本回合工具调用总数 |
| 客户端事件 `agent_tool` | 实时扇出（调试面板可用） |
| `AgentBus` | **只记失败调用**（与「只记转交与异常路径」的既有约定一致，不刷环形缓冲） |

---

## 2. 自主规划（`core/planning`）

`core.intent` 回答「这句话是什么意图、要不要转交」（规则优先，零成本）；
`TurnPlanner` 回答「这一回合打算怎么做」，由模型给出：

```json
{"goal": "讲清移项法则",
 "steps": ["先讲什么是移项", "再演示一道题"],
 "tools": ["math_solve", "math_verify"],
 "ask_followup": true,
 "reason": "方程题要先算再讲"}
```

计划作为 `context_notes` 注入答疑链路（`Supervisor._plan_notes`），并落进
`turn.meta["turn_plan"]`。

三条约束：

* **不拖慢上课**：超时（`AGENT_PLANNER_TIMEOUT_SECONDS`，默认 6s）、关闭、
  模型输出为空 / 不合法 -> 一律回退**规则计划**（由意图判定直接派生，零额外开销）；
* **不是硬指令**：计划只进提示词；真正调不调工具、按什么顺序讲，仍由模型在闭环里决定；
* **一回合只规划一次**：结果缓存在 `turn.meta["_turn_plan"]`，流式答疑的两条路共用同一份。

---

## 3. ReAct 闭环（`agents/teacher/react.py`）

```
thought: "这题是方程，我先解出来再写讲解。"
action : math_solve({"equation": "3x+5=20"})
observe: "3x+5=20 的解：x=5"
thought: "再验算一次。"
action : math_verify({"equation": "3x+5=20", "answer": "x=5"})
observe: "结论：x=5 满足 3x+5=20。"
answer : {...最终 JSON 契约...}          ← 不再调工具，收尾
```

* **有界**：`AGENT_REACT_MAX_STEPS`（默认 4）轮 + `AGENT_TOOL_MAX_CALLS`（默认 6）次；
  步数用尽会再发一次 `tool_choice="none"` 的收尾调用，逼模型给出答复；
* **超时收敛**：`AGENT_REACT_TIMEOUT_SECONDS`（默认 45s）到点即返回空答复，
  由 `TeacherAgent.answer` 回退到**单次生成**（即加装前的行为）；
* **不支持工具的模型**：退化为一次普通生成，`stopped="no_tool_support"`。

### 一个必须说清的取舍：ReAct 与流式话术互斥

流式答疑（`_turn_teach_streaming`）的做法是**话术路与结构路并行**：话术边生成边 TTS，
好处是首句延迟低。但模型此刻还在调工具复核 —— 两个版本的话会前后打架。

因此 `AGENT_REACT_ENABLE=true` 时**不走流式话术路**，等结构化作答（含工具复核）完成后再播报：

| 你要 | 怎么配 |
|---|---|
| 让老师先算再讲（推荐，答案更可靠） | `AGENT_REACT_ENABLE=true`（默认） |
| 极限首响延迟（第一句最快出声） | `AGENT_REACT_ENABLE=false` —— 回到并行流式链路 |

---

## 4. 长期记忆（`core/memory`）

### 与「教师经验」的分工

| | 教师经验（`core/skills/experience`） | 学情记忆（`core/memory`） |
|---|---|---|
| 归属 | 按**老师**存 | 按**学生**存 |
| 内容 | 老师怎么讲更有效 | 学生哪儿不会、习惯怎样 |
| 注入 | 该老师的每次讲解 | 只有这位学生上课时 |

经验可以跨学生复用（它描述教学法），记忆**绝不能**跨学生复用 —— 因此按 `student_id`
物理分文件：`<MEMORY_DIR>/students/<student_id>.jsonl`。

### 什么时候写

| 路径 | 触发 | 成本 | 默认 |
|---|---|---|---|
| 规则写入 | 判题**首次答错** -> 记一条 `mistake`（含学生答案与正确答案） | 零 LLM 调用 | 开 |
| 模型蒸馏 | 答疑结束异步提炼薄弱点 / 偏好 | 一次 LLM 调用 | **关**（`MEMORY_DISTILL_ENABLE`） |

默认关掉蒸馏的理由：面向中小学的课堂里，前几次课问不出「稳定特征」，
高频蒸馏只会把额度花在噪音上。有真实数据再开更省。

### 什么时候读

`agents/shared/teaching_layers.build_messages` 在装配提示词时插入「学情记忆」层，
位置在**共享人格层之后、教师经验之前**（越靠后影响力越强：具体到这堂课的信息
应当压过抽象的教学法约定）。检索用「`score`（命中次数 × 重要度 × 时间衰减）
+ 字符 bigram 相关性」，与 `rag/retriever.py` 同一套零依赖做法。

### 合规（面向未成年人，硬性）

* `DELETE /api/memory/{student_id}` 是**物理删除**（不是标记停用）；
* 记忆目录 `data/memory` 是运行时产物，与经验目录一样**按环境隔离**，不随同步覆盖。

---

## 5. 开关总表

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_TOOLS_ENABLE` | `true` | 是否把工具暴露给模型 |
| `AGENT_TOOL_MAX_CALLS` | `6` | 单回合工具调用总次数上限 |
| `AGENT_REACT_ENABLE` | `true` | 答疑是否走 ReAct 闭环（关掉 = 回到并行流式单次生成） |
| `AGENT_REACT_MAX_STEPS` | `4` | 单回合最多几轮「思考 + 工具」 |
| `AGENT_REACT_TIMEOUT_SECONDS` | `45` | 闭环总超时 |
| `AGENT_PLANNER_ENABLE` | `true` | 回合开始前是否让模型出计划 |
| `AGENT_PLANNER_TIMEOUT_SECONDS` | `6` | 规划超时（超时回退规则计划） |
| `AGENT_PLANNER_MAX_TOKENS` | `400` | 计划 JSON 的 token 预算 |
| `AGENT_SKILLS_ENABLE` | `true` | 是否把内置 Agent 技能暴露给模型（见 [`agent_skills.md`](agent_skills.md)） |
| `AGENT_SKILL_AUTO_RETIRE` | `false` | 战绩过差的技能自动停用（默认只进抽检台） |
| `MEMORY_ENABLE` | `true` | 长期记忆总开关 |
| `MEMORY_DIR` | `data/memory` | 落盘目录 |
| `MEMORY_INJECT_LIMIT` | `5` | 每次讲解最多注入几条记忆 |
| `MEMORY_MAX_ITEMS` | `0` | 单个学生最多保留多少条生效记忆；**0 = 不限，系统永不自动淘汰** |
| `MEMORY_DISTILL_ENABLE` | `false` | 是否用 LLM 蒸馏薄弱点 |
| `MEMORY_DISTILL_MIN_INTERVAL_SECONDS` | `120` | 蒸馏节流 |

**四个开关全部关掉 = 完全回到加装前的行为**：单次生成 + 规则分流 + 无记忆。
这是刻意的设计 —— 新增能力都必须是**可关的增强**，而不是新的故障点。

---

## 6. 怎么验证

```bash
# 单测（新增 24 个用例，覆盖四层各自的正常路径与降级路径）
pytest -q tests/test_agent_autonomy.py

# 端到端（离线，不依赖任何密钥）：MockLLM 传 tool_script 即可跑出一次真实闭环
python -c "import asyncio; from models.llm import MockLLM; ..."
```

排障时看这几处：

| 现象 | 先看 |
|---|---|
| 老师没调工具 | `AGENT_TOOLS_ENABLE` / `AGENT_REACT_ENABLE`；模型是否支持函数调用（`llm.supports_tools`） |
| 白板多出一行奇怪的补充 | `turn.meta["react"]["trace"]` 里那一步的 `whiteboard_note` |
| 老师讲的比流式时慢了 | 正常：ReAct 开启时不走并行流式（见 §3 的取舍） |
| 记忆串到别的学生 | 不该发生 —— 记忆按 `student_id` 分文件；若出现请看 `MEMORY_DIR` 是否被指向了共享目录 |
| 记忆越来越多 / 记错了 | `GET /api/memory/{student_id}` 查看，`POST /api/memory/status` 停用，`DELETE` 整份删除 |
