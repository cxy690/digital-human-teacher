# core/skills/teacher_personality — 教师人格 Skill（11 学科）

各学科独立人格的加载与适配。**人格事实源是 Markdown**：`.agents/skills/<学科key>/SKILL.md`——改 md 下一次答疑即生效（按 mtime 失效缓存），无需重启改码。各科互不共享人格内容，只共享接口与平台底线。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户：各学科人格类（`MathTeacherSkill` 等按类名动态注入命名空间，保持历史 import 写法可用）。 |
| `base.py` | 人格 Skill 基础接口 + `GLOBAL_TEACHER_RULES`（平台底线：通俗口语 / 不贬低 / 不编造 / 安全提醒 / 学科边界 / 保持 JSON 格式）。 |
| `md_teacher.py` | Markdown 事实源驱动的适配器：把 `SkillDoc` frontmatter + 正文小节映射到接口；`_FALLBACK` 保证九项行为**永不返回空串**。 |
| `registry.py` | 注册表：只负责查找/加载；找不到返回 `None` 由调用方回退 persona/style，**绝不误加载其他学科的 Skill**。 |

## 设计要点

- **底线不属于人格**：`GLOBAL_TEACHER_RULES` 是所有教师必须遵守的平台级要求，单独写在 `base.py`，与学科人格解耦。
- **TeacherAgent 不做学科分支**：只做通用调用，不写 `if subject == "math"`——学科差异全部由 md + 注册表承载。
- **契约兜底**：`_FALLBACK` 语义是「平台最低要求」而非某位老师人格，保证接口九项行为在缺 md 时也不返回空串。
- **事实源一致性**：目录名 = 学科 key，与 `config/subjects.py` 的 `key`、`data/teachers/<key>/` 三者一致；少一个目录 = 该科没有技能，回退 `teacher.json` 的 `persona`。
