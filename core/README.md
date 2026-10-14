# core — 核心编排与决策层

系统的「大脑中枢」：接入层（`server/`）→ **编排与决策（本目录）** → 智能体（`agents/`）→ 能力（`models/`）。本目录负责实时循环、意图分流、任务图执行、校验护栏、会话状态、课程计费与各类横切能力。

## 文件说明

| 文件 / 子包 | 用途 |
| --- | --- |
| `__init__.py` | 包门户；`Supervisor` 用模块级 `__getattr__` 惰性转发，避免导入期被拖重。 |
| `harness.py` | **装配根**（composition root）：进程级单例（模型 / 检索器 / 防逆向）在装配期构建，每连接按需造会话态对象。 |
| `loop.py` | 实时事件循环 `SessionLoop`：消费 WS 上行事件并分发；**入队瞬间**自增 `speech_epoch` + `avatar_interrupt`（barge-in）。 |
| `router.py` | 规则意图路由（关键词表以只读属性透出供复用）；「继续/再讲一遍」刻意不在此表（由 Supervisor 处理）。 |
| `graph.py` | **自研 TaskGraph**：`NodeSpec`（文档兼建图输入）与 `Node`（可执行）分离；`__skip__`/`__stop__` 约定实现分支与早退。 |
| `turn_graph.py` | LangGraph 版**回合主图**（`ORCHESTRATOR=langgraph` 启用）：节点直接复用 Supervisor 方法，防两套实现分叉。 |
| `langgraph_flow.py` | LangGraph 版**答疑子图**（retrieve → generate → validate）；缺 langgraph 时上层自动回退 TaskGraph。 |
| `validator.py` | 回复校验：话术长度/语言、作业模式答案泄露、白板坐标合法性、敏感词；只丢非法白板指令、保留其余。 |
| `guardrail.py` | 教学护栏与内容安全（G5）：学科学段范围限定 + 内容安全过滤 + AI 生成标识（每次输出必经）。 |
| `math_solver.py` | sympy 独立求解/验算；未装 sympy 时全部校验自动降级为「不做处理」。 |
| `exercise_bank.py` | 真题库只读访问层（`ai_teacher.exercise_item`）：只发 SELECT、只用质检干净题；一切异常 → None 由调用方回退 LLM。 |
| `homework_mode.py` | 作业助学模式：题目队列与进度、判对错、串联错题本（`HomeworkSession`）。 |
| `hint_homework.py` | 分层渐进式提示策略：`HintLevel` 四级（concept/method/step/answer）按题维度逐层放开。 |
| `course_session.py` | 1v1 辅导课阶段划分、时间映射与话术模板（阶段推进由前端计时器驱动）。 |
| `course_billing.py` | 课程倒计时与计费：课时状态在服务端（退出页面不停表）、到点自动结算；持久化 `data/courses.json`。 |
| `checkpoint.py` | 会话检查点落盘（白板/历史/题目缓存），支持断线续聊；board_ops 上限 500、history 上限 200。 |
| `resilience.py` | 弹性：重试 / 超时 / 熔断（`CircuitBreaker` 半开探活）/ 降级。 |
| `anti_rev.py` | 防逆向：HMAC-SHA256 接口签名 + 时效 TTL + 按调用方滑窗限流。 |
| `activity_log.py` | 图执行轨迹 append-only JSONL（`logs/activity/`，默认关）；`node_start` 单侧存在即可判定「中断时卡在哪一步」。 |
| `avatar_source.py` | 教师驱动视频源解析与底片规格校验；以**内容哈希**作缓存 key（换底片自动失效）。 |
| `avatar_idle.py` | 用口型引擎渲染静音素材产出待机帧（含真实闭眼帧抠取），前端据此做待机眨眼。 |
| `avatar_stream.py` | 数字人视频落盘 + `/media/avatar_stream` URL 暴露（避免大 base64 走 WS 造成卡顿）。 |
| `avatar_launcher.py` | 数字人子服务托管：探活 + detached 自动拉起（PID 状态文件保证幂等）。 |
| `avatar_bridge.py` | 数字人播报 HTTP 桥接（MuseTalk + WebRTC 服务）；服务未启动全静默跳过。 |
| `viseme.py` | 话术 → 粗略口型序列（pypinyin 可选，缺失返回空列表由前端回退能量驱动）。 |
| `audio_utils.py` | TTS 音频尾部补静音（让口型引擎以闭嘴帧收尾，切回待机更平滑）；失败原样返回。 |
| `auth/` | 认证：密码哈希 + JWT + FastAPI 依赖（见该目录 README）。 |
| `courseware/` | 课件：数据模型 + 仓库加载 + .pptx 解析（见该目录 README）。 |
| `intent/` | 意图决策层：规则 + 小模型两级判定（见该目录 README）。 |
| `memory/` | 学情长期记忆（见该目录 README）。 |
| `planning/` | 回合自主规划（见该目录 README）。 |
| `skills/` | 模型技能：表情 + 教师人格 + 经验自我更新（见该目录 README）。 |
| `state/` | 会话状态与事件模型（见该目录 README）。 |

## 设计要点

- **一切外部依赖「缺失即降级，绝不阻断」**：sympy / pypinyin / langgraph / python-pptx / opencv / mediapipe / RAG / 真题库 / 数据库缺失时均静默回退，教学主链路始终可跑。
- **分层单一职责**：本目录不实例化下层的具体实现（模型从 `models/` 的工厂取、Agent 从 `agents/` 装配），上层只依赖抽象接口。
- **运行时产物与代码分离**：`data/memory`、`data/teacher_experience`、`logs/activity` 等运行时数据被 `scripts/sync_cloud.py` 排除，本地与云端各自积累。
- **图化编排可观测**：`TaskGraph` 的 `NodeSpec` 同源文档 + `activity_log` 轨迹 + `turn.meta` 决策痕迹，三层信息可回答「这次答疑走了哪些节点、为什么」。
