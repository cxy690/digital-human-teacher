# 教师技能（Skill）个性化计划

> 目标：让 10 位数字人老师各自拥有**说话风格、语速、讲课深浅度**三个维度的独立配置，
> 同一道题，语文老师和数学老师讲出来的"感觉"完全不同。

---

## 〇、现状更新（本节的结论优先于下文）

> 下表是**已完成**的部分；下文（§一~§六）保留原始分析，作为「为什么这么设计」的依据。

| 维度 | 现在的落点 | 状态 |
| --- | --- | --- |
| 说话风格 / 讲课方式 / 深度 | `.agents/skills/<学科key>/SKILL.md` 的 frontmatter 与正文小节 | ✅ 已完成，改 md 即生效 |
| 人格与九项教学行为 | 同一份 `SKILL.md` 的正文小节 | ✅ 已完成（原 `core/skills/teacher_personality/*.py` 已删除） |
| 语速（`speech_rate`） | md 的 frontmatter 声明 → `TeacherProfile.speech_rate` | ⚠️ **半接通**：字段已读出，但 CosyVoice / EdgeTTS 两个引擎还没消费它（见 §二·阶段二） |
| 讲课深浅度 | md 的 `depth`（elementary / standard / advanced）→ `DEPTH_GUIDE` 统一出提示词 | ✅ 已完成 |

技能维护入口与字段规范见 [`README.md`](README.md)（同目录）；
原 `docs/teacher_skills_src/` 的 10 份流派设计稿已随迁移进入
`.agents/skills/<学科key>/references/`。

---

## 一、现状盘点（计划基于的已有机制）

| 能力 | 现状 | 位置 |
| --- | --- | --- |
| 说话人格（persona） | **已支持**：`DEFAULT_PERSONAS` 10 位各有默认，`teacher.json` 的 `persona` 字段可覆盖 | `config/teachers.py` → `agents/teacher/agent.py:597-607` 拼入 system prompt |
| 讲课风格（teaching_style） | **已支持且按老师独立**：plain / normal / academic 三档，优先取教师档案 | `agents/teacher/agent.py:560-607` |
| TTS 语速 | **不支持**：CosyVoice 引擎未传 `speed`；EdgeTTS 的 `rate` 固定 `+0%`，不按老师 | `scripts/tts_engine_cosyvoice.py` / `models/tts.py:74-88` |
| 讲课深浅度 | **无独立通道**：目前只能靠 persona 文本间接描述 | — |

10 位老师当前配置：

| 老师 | 发声引擎 | teaching_style | persona |
| --- | --- | --- | --- |
| chinese 语文 | 克隆 | plain | 默认（温婉知性） |
| math 数学 | 克隆 | plain | 默认（沉稳理性） |
| english 英语 | 克隆 | plain | 默认（明快干练） |
| physics 物理 | 克隆 | plain | 默认（严谨追问） |
| chemistry 化学 | 克隆 | plain | 默认（细致耐心） |
| biology 生物 | 克隆 | plain | 默认（亲切自然） |
| politics 政治 | 克隆 | plain | 默认（端庄稳重） |
| geography 地理 | edge | plain | 默认（阳光健谈） |
| history 历史 | edge | plain | 默认（儒雅沉静） |
| morality 道德 | edge | plain | 默认（温和可信） |

---

## 二、Skill 设计：三个维度

### 维度 1：说话风格（persona）—— 纯配置，立刻可用

每位老师一段"这个人怎么说话"的提示词，回答语、比喻习惯、口头禅都在这里定义。
**已有机制支持**，只需把每位的 persona 写进 `data/teachers/<key>/teacher.json`。

### 维度 2：语速（speech_rate）—— 需要补一条代码通道

新增配置字段，两位 TTS 引擎各自落实：

```python
# config/teachers.py :: TeacherProfile 新增
speech_rate: float = 1.0    # 语速倍率：0.85 慢 / 1.0 正常 / 1.1 快
```

- **CosyVoice 3/2**：官方 `inference_zero_shot(..., speed=1.0)` 本来就支持 speed，只需把参数接进来
- **edge-tts**：`rate` 换算 `rate% = (rate-1)×100`（1.0→"+0%"，0.9→"-10%"）
- 数字人链路**无需改动**：MuseTalk 按音频时长渲染口型，音频快慢自动跟随

### 维度 3：讲课深浅度（depth）—— 并入 persona，不改结构

不新增独立字段（避免过度设计），把"讲多深"写成 persona 里的明确指令。
这与现有 `teaching_style`（大白话/规范术语）互补：**style 管"用什么词"，depth 管"讲到多细"**。

---

## 三、10 位老师的 Skill 方案（核心）

> 设计原则：**语速跟着"学生要不要思考时间"走，深度跟着"学科抽象度"走。**

| 老师 | 说话风格（persona 要点） | 语速 | 深度 | 设计理由 |
| --- | --- | --- | --- | --- |
| **语文** | 温婉知性、有画面感；多引用原文和诗词，讲词义必给生活例句；语气像陪读的姐姐 | **0.95** | 标准（重语感轻术语） | 课文朗读感需要稍慢的节奏；语法术语点到为止，重"品" |
| **数学** | 沉稳理性、不跳步；每一步先说"为什么这样做"再动笔；讲完必带一句"你自己再走一遍" | **0.90** | 标准（每步有依据） | 推导密集，学生需要跟上的时间；慢讲是数学课堂的刚需 |
| **英语** | 明快干练、多鼓励；**全程普通话讲解**，只在举例时给英文并立刻中文解释；爱用造句练习互动 | **0.95** | 浅显（重开口） | 例句和单词必须吐字清晰；语法规则讲到"会用"即可 |
| **物理** | 严谨又亲切；从生活现象追问"为什么"；**每个物理量必带单位、每个公式必讲适用条件** | **0.95** | 标准（重因果） | 概念抽象，稍慢；条件与单位是物理的核心，必须反复强调 |
| **化学** | 细致耐心；用实验现象和厨房类比讲抽象概念；**方程式逐字慢读**，提醒配平与反应条件 | **0.90** | 浅显（现象→结论） | 化学式的读法必须慢；类比降低抽象度 |
| **生物** | 亲切自然；爱用身体、饮食、自然界的例子；把零散知识点串成"规律链" | **1.00** | 浅显（多比喻） | 内容本身贴近生活，正常语速；重记忆串联 |
| **政治** | 端庄稳重、条理分明；**先摆材料再提炼观点**；术语规范（不用大白话替代概念） | **0.95** | 标准（重规范） | 唯一建议 `teaching_style=normal` 的学科：概念必须用规范表述 |
| **道德** | 温和可信赖；用校园和家庭真实情境讲道理；重视规则意识与同理心，不说教 | **0.95** | 浅显（情境化） | 面向低龄，语速放缓、情境先行 |
| **地理** | 阳光健谈、有活力；地图和旅行见闻信手拈来；**一切按"位置→气候→人文"因果链**讲 | **1.05** | 标准（重链条） | 内容偏描述性，稍快反而有课堂气氛；因果链是骨架 |
| **历史** | 儒雅沉静、会讲故事；**把事件放回发生的年代**；先叙事后分析，重史料与因果 | **0.90** | 深入（重脉络） | 讲故事需要慢节奏的沉浸感；历史价值在因果分析，可讲深 |

语速档位速查：`0.90 明显放慢（数学/化学/历史）· 0.95 略慢（语文/英语/物理/政治/道德）· 1.00 正常（生物）· 1.05 略快（地理）`

---

## 四、实现路径（三步走）

### 阶段一：配置层（零代码，当天生效）✦ 先做这个

为每份 `data/teachers/<key>/teacher.json` 写入 `persona`（风格 + 深度指令合一）。
现有机制会自动拼进 system prompt，**改完同步重启即生效**。

```json
// data/teachers/math/teacher.json（示例）
{
  "teaching_style": "plain",
  "persona": "你是一位沉稳理性的数学老师。说话不急不缓、条理清晰：每一步先说清'为什么这样做'再动笔，绝不跳步；讲完必提醒学生'你自己再走一遍'。用词朴素但精确，学生没懂就换一种说法再讲。"
}
```

### 阶段二：语速通道（小改 3 个文件）

| 文件 | 改动 |
| --- | --- |
| `config/teachers.py` | `TeacherProfile` 加 `speech_rate: float = 1.0`；`_from_dict` 读取 |
| `scripts/tts_engine_cosyvoice.py` | `TTSRequest` 加 `speed: float = 1.0` → `inference_zero_shot(..., speed=req.speed)` |
| `models/tts.py` | `CosyVoiceTTS._payload` 带 `speed`；`EdgeTTS.synthesize` 按老师换算 `rate`；新增 `_teacher_speech_rate(teacher_id)` |

### 阶段三：验收与微调

1. 同一道题（如"解方程 3x+5=20"）分别发给数学/语文/历史，对比讲解的**节奏与深浅**
2. 同一段文本各老师合成音频，对比时长验证语速（0.9 的应比 1.0 长 ~11%）
3. 学生端实际上课体验 → 微调 persona 措辞

---

## 五、执行清单

- [x] 阶段一：10 位老师的风格 + 深度指令（现落在各科 `SKILL.md`，不再写 `teacher.json`）
- [ ] 阶段一：政治老师的 `teaching_style` 改为 `normal`（唯一建议改的，见各科 md 的 frontmatter）
- [x] 阶段二：`TeacherProfile.speech_rate` 字段（值来自 md 的 `speech_rate`）
- [ ] 阶段二：两位 TTS 引擎接入 speed/rate（**当前缺口**）
- [ ] 阶段二：同步云端、重启 TTS 引擎与主应用
- [ ] 阶段三：同题对比试听验收，微调

## 六、两点说明

1. **风格与深度不用改代码**：prompt 机制现成，写好 persona 就是"skill"。
2. **语速值得做**：它是三个维度里唯一"听得出来但 prompt 控制不了"的 ——
   persona 里写"语速放慢"模型只会少说几句，不会真正放慢；真正的慢必须由 TTS 引擎的 speed/rate 控制。
