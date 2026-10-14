# config/teacher_skills — SKILL.md 的解析与消费侧

教师技能 Markdown（`.agents/skills/<学科key>/SKILL.md`）的**读取与装配层**：解析 frontmatter 与正文小节，产出注入 system prompt 的「说话风格层」。人格行为层由 `core/skills/teacher_personality/` 消费同一份 md。

> 分工：事实源是 md 文件本身；本目录只做「解析 + 映射」，不含任何具体学科内容。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户：`load_skill()` / `available_skills()` 等入口。 |
| `base.py` | 技能数据结构与接口（`TeacherSkill`：`depth` / `speech_rate` / `teaching_style` / 口头禅 / 务必做到 / 不要 等字段）。 |
| `md.py` | md 解析器：frontmatter（PyYAML）+ 正文小节（【说话风格】【讲课方式】【深度】【口头禅】【务必做到】【不要】）映射为 `TeacherSkill`。 |

## 设计要点

- **一份 md 两层技能**：frontmatter 给风格 / 深度 / 语速，正文给人格与教学行为——合并前这两层各写一份 Python，「改一处漏一处」，合并后改 md 一处生效。
- **运行期现读现改现生效**：按 mtime 失效缓存，线上改 md 下一次答疑即生效，不需要重启或改代码。
- **键对齐纪律**：目录名 = 学科 key（与 `config/subjects.py`、`data/teachers/<key>/` 三者一致）；缺 md 的学科跳过并告警，回退 `teacher.json` 的 `persona`。
