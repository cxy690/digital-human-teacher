# core/skills — 模型技能

挂在模型前后的「技能」集合：表情贴纸（学生答对时的表扬反馈）、教师人格（11 学科各自人格）、教师经验自我更新（答疑后复盘沉淀）、**Agent 技能（模型可自主调用 + 自我更新）**。人格与经验的**事实源是 Markdown**（`.agents/skills/<学科key>/SKILL.md`），本目录只做消费侧。

## 文件说明

| 文件 / 子包 | 用途 |
| --- | --- |
| `__init__.py` | 包门户（含表情技能与教师人格技能入口）。 |
| `expression_skill.py` | 表情技能：学生短回答 → LLM 判定情境（一般答对 praise / 表现出色 celebrate）→ 返回 `ExpressionDecision`（表情键 + 表扬话术）。**现在由 Agent 技能 `praise_expression` 调用**（此前只有单测引用，见 agent_skills/README）。 |
| `experience/` | 教师 Skill 的「自我更新」经验层（见该目录 README）。 |
| `teacher_personality/` | 教师人格 Skill：按学科加载 SKILL.md 并适配为接口（见该目录 README）。 |
| `agent_skills/` | **Agent 技能**：模型可自主调用的业务能力单元（function calling），战绩按回合裁决回写，实现自我更新（见该目录 README 与 `docs/agent_skills.md`）。 |

## 设计要点

- **表情技能不拦截答疑**：`_SKIP_HINTS` 先拦截疑问/求助/寒暄类输入；未命中返回 `None`，正常走答疑链路——技能只做加法，不做判决。
- **数字人素材已固定**：表情仅以贴纸（`ai_speech` 事件携带表情键）形式下发，不再切换数字人视频素材。
- **「人格是本体、经验是长出来的」**：`teacher_personality` 由人写由人改（md 事实源），`experience` 只追加可回滚的经验条目，永不改写人格本体。
