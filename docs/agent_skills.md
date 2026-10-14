# Agent 技能：模型可自主调用的技能单元（含自我更新）

> 适用对象：`core/skills/agent_skills/`、`agents/shared/toolkit.py`（技能登记）、
> `agents/supervisor/agent.py`（战绩回写）、`server/routes.py`（抽检与止损接口）。
>
> 本文讲**技能怎么接进来、怎么自我更新、开关与代价**；原子工具 / 自主规划 / ReAct 闭环
> 仍见 [`agent_autonomy.md`](agent_autonomy.md)，人格与经验见 `core/skills/` 各目录 README。

---

## 0. 为什么需要它

加装之前，模型的能力入口只有两类：

| 已有 | 是什么 | 谁决定用不用 |
|---|---|---|
| 原子工具 | 查教材 / 解方程 / 读白板（无判断） | 模型自己（function calling） |
| 人格 / 经验 | 提示词层（说话方式 + 学到的手感） | 学科查表 / bandit 采样 |

缺的是中间那一层：**带判断的业务能力**。「学生答对了要不要表扬」「讲易错点前要不要
翻这位学生的旧账」「本科目按什么骨架讲」—— 这些不该由正在写讲解的模型分心判断，
也不该写死在流水线里。Agent 技能就是这一层：模型像调工具一样调它，技能自己判断
「这题适不适合用我」。

唯一有技能形状的 `core/skills/expression_skill.py` 此前**只有单测引用**（写了没接线），
本次把它包成 `praise_expression` 技能，接线与测试一起补上。

## 1. 技能长什么样

```python
SkillSpec(
    name="student_mistake_recall",            # 与工具同一命名空间，不得重名
    title="学生旧账回顾",
    description="讲解之前，查这位学生以前在类似地方出过的错……",   # 给模型：何时调用（短）
    instruction="翻旧账是为了「顺带提醒」，不是点名批评……",        # 给技能：执行纪律
    inputs={"query": {"type": "string", ...}},                     # JSON Schema 子集
    experience_kinds=("pitfall", "confusion"),                     # 关心哪几类经验
    needs_llm=False,
)
```

两类形态，同一个接口：

| 形态 | 例子 | 说明 |
|---|---|---|
| 执行型 | `praise_expression` / `student_mistake_recall` / `teacher_experience` | 内部再调 LLM 或查存储，产出「观察」文本 |
| 参考型 | `subject_playbook` | 无执行体，调用即返回正文（现读 SKILL.md 的「讲课方式」）—— 这就是**渐进披露**：学科讲法不必每轮全量注入，模型需要时才取 |

## 2. 怎么接进链路

```
TeacherAgent.answer()
  └─ ReActLoop.run(messages)                 ← 思考→行动→观察（docs/agent_autonomy.md）
       └─ AgentToolkit.schemas()             ← 6 个原子工具 + N 个技能（同一份清单）
       └─ AgentToolkit.call("praise_expression", {...})
            ├─ registry.context_for(skill, session)   ← 装配上下文（含学习增强）
            ├─ skill.run(args, ctx)                    ← 执行（失败降级成「观察」）
            └─ mark_skill_used(session, name)          ← 记账进 scratch（同回合去重）

Supervisor.handle_turn()  finally
  ├─ record_turn_verdict(...)        ← L0 裁决（pass / degraded / failed）
  ├─ record_turn_outcome(...)        ← L2 经验收益回写
  └─ record_skill_outcome(...)       ← 技能战绩回写（pulls/wins）
```

三个刻意的边界：

1. **同一命名空间、同一预算**：技能与原子工具对模型是一个清单，调用次数同样受
   `AGENT_TOOL_MAX_CALLS` 约束（技能内部可能再调 LLM，不设限会烧掉整个回合的工具预算）；
2. **失败绝不外抛**：异常 / 未命中都变成一句可读的「观察」，模型自己决定换方式还是直接作答；
3. **可关**：`AGENT_SKILLS_ENABLE=false` 时一个技能都不登记，模型看到的清单与加装前一致。

## 3. 自我更新：技能怎么自己变好

闭环与经验层（L0→L2→L5）同构，只是记账对象从「一条经验」换成「一个技能」：

| 环节 | 机制 | 落点 |
|---|---|---|
| **学** | 复盘沉淀的经验按 `experience_kinds` 选出「实战补充」，附加到技能上下文 | `registry.context_for()` → `SkillContext.learned_notes` |
| **选** | 模型自主决定调不调；技能描述（`description`）是人写的，**不被自动改写** | `registry.schemas()` |
| **用** | 调用进 `turn.meta` 轨迹 + `session.scratch` 记账（同回合去重、按 epoch 隔轮） | `AgentToolkit` / `feedback.mark_skill_used` |
| **评** | 回合裁决 `pass=1 / degraded·failed=0`，异常回合不记账；整批同记（噪声随 pulls 平均掉） | `feedback.record_skill_outcome` → `data/skill_usage/<技能>.jsonl` |
| **止损** | 战绩差 → 抽检台；`AGENT_SKILL_AUTO_RETIRE=true` 时自动停用（可人工恢复） | `usage.review_queue()` / `registry.enabled()` |

「人写基底、机器学补充」与人格 Skill 同一纪律：`spec.instruction` 永不被程序改写，
学到的经验只在其上追加 —— 技能因此会越用越准，但不会变成认不出来的另一段话。

## 4. 开关总表

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_SKILLS_ENABLE` | `true` | 是否把内置技能暴露给模型（关掉 = 回到只有工具的旧行为） |
| `AGENT_SKILL_USAGE_DIR` | `data/skill_usage` | 技能战绩落盘目录（按环境隔离，不随代码同步） |
| `AGENT_SKILL_LEARNED_MAX` | `2` | 每个技能最多附加几条学到的经验（0 = 不接经验增强） |
| `AGENT_SKILL_AUTO_RETIRE` | `false` | 战绩过差的技能是否自动停用（默认只进抽检台） |
| `AGENT_SKILL_MIN_PULLS` | `5` | 少于这么次实战不下结论 |
| `AGENT_SKILL_MIN_GAIN_RATE` | `0.3` | 有效率低于它认为「这个技能没什么用」 |

## 5. 接口（人看与止损）

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/skills/agent` | 全部技能声明 + 调用战绩 |
| GET | `/api/skills/agent/review` | 抽检台：按风险排出最该看一眼的技能 |
| POST | `/api/skills/agent/status` | 停用 / 恢复一个技能（`{"name": "...", "status": "disabled"}`） |

命令行等价物：`scripts/experience_review.py` 的同目录思路（技能侧先用接口）。

## 6. 一个必须说清的取舍：白板先行与技能互斥

`BOARD_FIRST_MODE=single_call` 走的是「流式单次调用」（`teacher.stream_board_first`），
那条路**没有工具集**，模型既调不了工具也调不了技能。因此 ReAct 开启时，
答疑会自动改走 `two_call`（同样「先讲理解再给解答」，但走 `teacher.answer`，工具集齐全）——
与「ReAct 开启时不走并行流式话术」是同一条取舍：**要自主性（先查证再开口），
就让出省往返的那一次调用**。想回到 single_call，把 `AGENT_REACT_ENABLE` 关掉即可。

## 7. 怎么验证

```bash
pytest -q tests/skills/test_agent_skills.py          # 14 条：暴露/调用/降级/自更新/可关/端到端
pytest -q tests/skills/ tests/test_agent_autonomy.py # 技能 + 自主性一起跑
```

排障时看这几处：

| 现象 | 先看 |
|---|---|
| 模型没调技能 | `AGENT_SKILLS_ENABLE`；技能是否被自动停用（`/api/skills/agent` 的 `usage.status`） |
| 技能调了但没战绩 | `handle_turn` 的 `finally` 是否走到（裁决为 `None` 时不记账）；`AGENT_SKILL_USAGE_DIR` |
| 技能总是被跳过 | `turn.meta["react"]["trace"]` 里那一步的观察文本（未命中会写清原因） |
| 讲解变慢 | 正常：技能内部可能再调一次 LLM；`needs_llm=true` 的技能尤其明显 |
