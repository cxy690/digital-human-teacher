# `.agents/` 目录说明（数字人教师技能）

> 本目录是**数字人教师技能**（teacher skill）的落点。**运行期事实源**在 `skills/`，
> 设计稿与维护笔记在 `notes/`（本目录，运行期不读）。

```
.agents/
├── skills/                        # ★ 一科一目录，运行期从 SKILL.md 现读
│   ├── chinese/SKILL.md
│   ├── math/SKILL.md
│   │   └── references/            #   流派设计稿（人读，不注入提示词）
│   ├── english/SKILL.md
│   ├── science/SKILL.md
│   ├── physics/ chemistry/ biology/ politics/ morality/ geography/ history/ …
└── notes/                         # 设计稿 / 维护笔记（本目录）
    ├── README.md                  #   本文件
    └── teacher-skills-plan.md     #   教师技能总体方案（风格 / 语速 / 深度三维度）
```

目录名 = **学科 key**，与 `config/subjects.py` 的 `key`、`data/teachers/<key>/` 三者一致；
少一个目录 = 该科没有技能，会回退到 `teacher.json` 的 `persona`。

---

## 一、SKILL.md 格式

**一份 md 同时承载两层技能**（合并前这两层各写一份 Python，改一处漏一处）：

| 层 | 消费方 | 取的字段 |
|---|---|---|
| 说话风格层 | `config/teacher_skills.TeacherSkill` → 注入 system prompt | `depth` / `speech_rate` / `teaching_style` / 【说话风格】【讲课方式】【深度】【口头禅】【务必做到】【不要】 |
| 人格行为层 | `core/skills/teacher_personality.TeacherPersonalitySkill` | 【人格与教学】整段 + 九项行为 + 进场问候 |

### frontmatter

```yaml
---
name: "数学 · 严谨的数学老师"      # 人读标题
subject_key: "math"              # 学科 key（= 目录名）
subject_name: "数学"
skill_id: "teacher.math.yanli"   # 人格 Skill 全局唯一 id（注册表按它查）
class_name: "MathTeacherSkill"   # 生成的 Python 类名（保持旧导入可用）
display_name: "严谨的数学老师"    # 前端/问候语里的称呼
persona: "严谨清晰、不跳步、重条件与验证"      # 一句话关键词（日志/前端）
tone: "语速平稳、一句一步，关键结论放慢强调"
depth: "standard"                # elementary | standard | advanced
speech_rate: 0.9                 # 语速倍率（见 §六 已知未接通）
teaching_style: "plain"          # plain | normal | academic
source_flow: "math-modeling-studio"   # 对应的流派设计稿（reference 文件名）
greeting_default: "今天的数学课"        # 问候模板 {where} 为空时的默认课题
style_persona: |-                # 风格层【说话风格】（多行块）
  你是一位沉稳理性的数学老师……
catchphrases: ["…"]              # 常挂嘴边的话
do: ["…"]                        # 务必做到
dont: ["…"]                      # 不要做
---
```

> `persona`（一句话关键词）与 `style_persona`（成段说话风格）是两个字段，别写混：
> 前者给日志与前端展示，后者真正拼进提示词。

### 正文小节

标题必须**逐字**用下面这些（解析器按 `## ` 标题匹配，改标题即失效）：

| 小节 | 用途 |
|---|---|
| `## 说话风格` | 风格层【说话风格】（一般与 `style_persona` 一致） |
| `## 讲课方式` | 风格层【讲课方式】：拿到一道题按什么顺序讲（流派骨架写在这） |
| `## 人格与教学` | 人格层整段提示词（身份 + 人格特征 + 教学行为） |
| `## 开场方式` | `opening_style()` |
| `## 追问方式` | `questioning_style()` |
| `## 鼓励方式` | `encouragement_style()` |
| `## 表扬方式` | `praise_style()` |
| `## 纠错方式` | `correction_style()` |
| `## 学生说「我不会」时` | `confused_student_strategy()` |
| `## 学生连续答错时` | `repeated_error_strategy()` |
| `## 白板组织` | `whiteboard_style()` |
| `## 语气与节奏` | `tone_rhythm()` |
| `## 进场问候` | `greeting(where)` 模板，`{where}` 替换为当前课题 |

小节标题之外的段落（导语、`>` 引用块、`<!-- -->` 注释）**不会被注入提示词**，随便写。

---

## 二、加载链路

```
.agents/skills/<key>/SKILL.md
        │  config/teacher_skills/md.py（parse_skill_md → SkillDoc，按 mtime 失效缓存）
        ├─→ config/teacher_skills.TeacherSkill.from_doc      （风格层）
        │        └─ agents/shared/teaching_layers.build_profile_layers → system prompt
        └─→ core/skills/teacher_personality/md_teacher        （人格层）
                 └─ registry 按 md 生成一科一个类 → TeacherAgent / LecturerAgent
```

**热更新**：解析缓存按文件 **mtime** 失效，所以线上改完一份 SKILL.md，
下一次答疑即生效，**不需要重启进程、也不需要改任何 Python**。
（身份字段 `skill_id`/`class_name` 等只在进程启动时生成类，改这类字段需重启。）

自检：`python scripts/selfcheck.py`（会打印 `available_skills()`，即 `.agents/skills/` 下的目录）。

---

## 三、改一位老师

1. 编辑 `.agents/skills/<key>/SKILL.md`（frontmatter 的参数 + 对应小节的话术）；
2. 想同时改「形象 / 音色 / 标定」→ 改 `data/teachers/<key>/teacher.json`（与本目录分工不同）；
3. 跑 `python -m pytest tests/test_teacher_personality.py tests/test_skill_md.py -q` 自检。

## 四、新增一科老师

1. `cp -r .agents/skills/math .agents/skills/<新key>`，改 frontmatter 的
   `subject_key` / `skill_id` / `class_name` / `display_name` 等；
2. 在 `config/subjects.py` 的 `SUBJECTS` 里加上该学科（决定前端选课与人格查表）；
3. 准备 `data/teachers/<key>/`（`talk.mp4` + `voice_ref.wav` + `teacher.json`），见
   [`../../数字人素材规格.md`](../../数字人素材规格.md)；
4. 跑单测：技能数、学科数、人格数三者的数量关系由 `tests/test_teacher_personality.py` 钉住。

## 五、本次 md 化的来龙去脉

合并前的两份资产（已删除）：

| 旧位置 | 内容 |
|---|---|
| `config/teacher_skills/<key>.py` | 风格 / 讲法 / depth / speech_rate / 口头禅 |
| `core/skills/teacher_personality/<key>_teacher.py` | 人格提示词 + 九项行为 |
| `docs/teacher_skills_src/<流派>/` | 流派设计稿（含 references） |

现在合并为 `.agents/skills/<key>/SKILL.md`（含 `references/` 里的流派设计稿）。
迁移由 `scripts/migrate_teacher_skills_to_md.py` 一次性完成，并带**往返自校验**
（生成后立刻用 YAML 读回、逐字段与源对象比对）—— 内容零丢失是有据可查的。

## 六、已知未接通

* **`speech_rate` 尚未真正影响发声**：md 里的倍率已被读出
  （`config/teachers.py::TeacherProfile.speech_rate`），但两个 TTS 引擎还没消费它
  （CosyVoice 请求体未带 `speed`，EdgeTTS 的 `rate` 仍是固定 `+0%`）。
  即当前「语速」只在提示词层面间接起作用。接线方案见 `teacher-skills-plan.md` 阶段二。
* **`science`（科学）没有独立流派设计稿**：`references/` 为空，其风格来自本次从
  原 `.py` 迁移的内容。

## 七、相关文档

* [`teacher-skills-plan.md`](teacher-skills-plan.md) —— 风格 / 语速 / 深度三维度总体方案
* [`../../docs/supervisor_orchestration.md`](../../docs/supervisor_orchestration.md) —— 总控调度策略（人格在哪一层注入）
* [`../../代码地图.md`](../../代码地图.md) —— 目录结构与端到端调用链
