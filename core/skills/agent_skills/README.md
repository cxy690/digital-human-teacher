# core/skills/agent_skills — Agent 技能（模型可自主调用 + 自我更新）

把「一项可被模型**自主调用**的业务能力」定义成技能，并让它接上经验闭环自动进化。
加装之前项目里只有两类东西：**人格提示词**（人写，按学科注入）与**经验条目**（自动学，
bandit 注入）—— 模型没有任何「自己决定用一段带判断的能力」的入口；唯一有技能形状的
`expression_skill` 还只有单测引用（写了没接线）。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `base.py` | 技能基座：`SkillSpec`（名字/何时用/执行纪律/入参 schema）、`SkillOutcome`、`SkillContext`、`AgentSkill` 抽象。 |
| `builtin.py` | 四个内置技能：`praise_expression`（答对表扬，修复 expression_skill 的死代码）、`student_mistake_recall`（查这位学生的旧账）、`teacher_experience`（查老师的经验）、`subject_playbook`（学科讲法，现读 SKILL.md）。 |
| `registry.py` | 注册表：装配、`schemas()` 暴露给模型、`context_for()` 装上下文（含**学习增强**）、`enabled()` 按战绩过滤。 |
| `usage.py` | 技能战绩存储：append-only JSONL（`data/skill_usage/<技能>.jsonl`），`record_usage` / `review_queue` / `set_status`。 |
| `feedback.py` | 战绩回写：回合收口处按裁决把 `pulls/wins` 写进战绩文件（记账对象来自 `session.scratch`）。 |

## 自我更新闭环（对应经验闭环）

```
学：复盘沉淀经验 ──► 注册表按 kind 选出「实战补充」附加到技能上下文（人写基底 + 自动补充）
选：模型在 ReAct 闭环里自主决定调不调、传什么参数（function calling，与工具同一命名空间）
用：AgentToolkit 执行技能 ──► 调用名记进 session.scratch（同回合去重）
评：回合裁决 pass/degraded ──► 回写每个被调用技能的 pulls/wins ──► 抽检台 / 可选自动停用
```

## 设计要点

- **技能只做加法**：关掉 `AGENT_SKILLS_ENABLE` 后模型看到的工具清单与加装前完全一致；
  技能失败一律降级成一句「观察」，未命中也要说清「为什么这次没用我」。
- **人写基底、机器学补充**：`spec.instruction` 永不被程序改写，学到的经验只在其上追加
  （与人格 Skill「人格 + 经验」同一纪律）。
- **默认不自动停用**：战绩差只进抽检台（`GET /api/skills/agent/review`），
  打开 `AGENT_SKILL_AUTO_RETIRE` 才自动停用 —— 与经验层 L5 同一立场。
- **新增一项技能**：在 `builtin.py` 加一个 `AgentSkill` 子类并注册，工具暴露、学习增强、
  战绩记账都由基座负责；参考型技能（无执行体，如 `subject_playbook`）照抄它即可。

开关与接口见 `docs/agent_skills.md`；测试在 `tests/skills/test_agent_skills.py`。
