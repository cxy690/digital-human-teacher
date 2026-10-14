# 数据接入流水线（rag/offline）

把**原始素材**（直播录像 / 课件 / 答疑记录 / 题库 / 教材）加工成**教研可审校的结构化草稿**，
最终由 `tools/build_bundle.py` 编译成下游交付包。

> **本目录只负责「把原料变成待审草稿」，不负责发布。**
> 导出与门禁的真源是 `tools/build_bundle.py`，本目录的 `s6_export.py` 只做门禁编排与调用。

---

## 1. 六个 stage 与人工介入点

```
data/raw/                  原始素材（PPTX 课件、PDF 讲义/教材、DOCX 教案、录像音频、答疑导出）
   │
   │  ★ 第 0 步（强烈建议先做）★
   │  tools/audit_sources.py   盘点：格式分布、角色归类、约定符合性、PDF 文字层体检、OCR 工作量
   │                           只读不写。**先知道手上有什么，再动手做数据**
   ▼
   │  s1_parse.py          解析：PPTX/PDF/DOCX → 文本；音频 → FunASR 转写 JSON
   ▼                        ⚠️ 保留时间码/页码，这是溯源的唯一凭据
   │                        ⚠️ DOCX 走 parse_docx_auto：像教案就分节抽取（产出 D1/D2 原料）
   │                        ⚠️ PDF 扫描件/混排页会明确告警，不静默丢弃
data/parsed/<name>.json
   │
   │  s2_clean.py          清洗：去口头禅 → **隐私脱敏** → 标点归一
   ▼                        ⚠️ 脱敏是面向未成年人场景的硬性要求，入库前必须完成
data/clean/<name>.json
   │
   │  s3_chunk.py          教学感知切分：cue 词边界 + 知识点边界 + 220 字上限
   ▼                        （不是固定长度切分）
data/chunks/<lesson>/*.yaml   ← segment 草稿，全部 status=draft
   │
   │  s4_autolabel.py      机器预标注：teacher_intent / style_emotion / kp_ids
   ▼                        ⚠️ 结果一律 needs_human_review=true，绝不产出 approved
data/labels/<lesson>/*.yaml
   │
   │  s5_qa_mine.py        从答疑/转写里挖候选问答对（提问-回答配对）
   ▼
data/qa_candidates/<lesson>/qa_candidates.yaml
   │
   │  ★★★ 人工介入点 ★★★
   │  教研把上述草稿搬进 samples/<lesson>/，人工修正：
   │    · 切分边界与停顿设计
   │    · 追问链的分支（correct / wrong / no_response 各走哪条路）
   │    · paraphrases 与 negative_examples
   │    · 安全策略与护栏话术
   │    · 风格标签校准
   │
   │  s6_export.py         门禁编排：schema 校验 → tools/lint.py → tools/build_bundle.py
   ▼
data/export/<lesson_id>/    运行时交付包（7 个文件，见 docs/06）
```

**这条流水线的设计前提**：机器做覆盖，人做判断。
机器能自动做的（切分、预标注、配对）覆盖率做到 100%；
不能自动做的（追问链、负例、安全策略）明确标出来交给人工，**不假装能自动做**。

---

## 2. 各 stage 命令

```powershell
cd C:\Users\Lenovo\Desktop\AI_Teacher
# 不需要设置 PYTHONIOENCODING：各 stage 会在非交互场景自行切到 UTF-8 输出

# 0. 先探测可选依赖（python-pptx / pymupdf / python-docx 是否可用）
python rag/offline/rag/offline/s1_parse.py --check-deps

# 1. 解析
python rag/offline/rag/offline/s1_parse.py --in data/raw --out data/parsed
#   音频转写只打印建议命令、不执行（ASR 是重活，应在 GPU 机器上离线跑）
python rag/offline/rag/offline/s1_parse.py --funasr-hint data/raw/lesson.wav --hotword "毫米 分米 进率"

# 2. 清洗（含隐私脱敏）
python rag/offline/rag/offline/s2_clean.py --in data/parsed --out data/clean
#   单条排查正则时用这个（不读目录）
python rag/offline/rag/offline/s2_clean.py --text "我是三年二班的张小明，手机13812345678" --show

# 3. 教学感知切分
python rag/offline/rag/offline/s3_chunk.py --in data/clean --out data/chunks --lesson LESSON-MATH-P3A-RJ-U03

# 4. 机器预标注（rule = 纯规则，离线可跑）
python rag/offline/rag/offline/s4_autolabel.py --in data/chunks --out data/labels --labeler rule
#   想评审 LLM 标注方案时，先看看 prompt 长什么样
python rag/offline/rag/offline/s4_autolabel.py --show-prompt

# 5. 挖掘候选问答对
python rag/offline/rag/offline/s5_qa_mine.py --in data/clean --out data/qa_candidates --lesson LESSON-MATH-P3A-RJ-U03

# ★ 人工审校：把草稿整理进 samples/<lesson>/ ★

# 6. 门禁 + 导出（不实现导出逻辑，调用 tools/build_bundle.py）
python rag/offline/rag/offline/s6_export.py --lesson-dir samples/LESSON-MATH-P3A-RJ-U03 --out data/export
python rag/offline/rag/offline/s6_export.py --lesson-dir samples/LESSON-MATH-P3A-RJ-U03 --dry-run   # 只跑门禁
```

---

## 3. 每个 stage 的关键设计

### s1_parse —— 解析

**采集约定**（真源 `schema/source_conventions.yaml`）：**PPTX→课件 / PDF→讲义·教材扫描件 / DOCX→教案**。
约定决定每个 stage 怎么处理，所以先看约定再解析。

| 素材 | 工具 | 说明 |
|---|---|---|
| PPTX（课件） | `python-pptx` | 逐页 `{page, text, notes}`，**备注栏常藏教学提示，别丢** |
| PDF（讲义/教材） | PyMuPDF → pdfminer 兜底 | 逐页文本 + 页码；**扫描件/混排页会明确告警** |
| DOCX（教案） | `python-docx` + **分节抽取** | 见下方「教案分节抽取」 |
| 录像/音频 | `ffmpeg` 抽音轨 → FunASR | 输出 `[{start_ms, end_ms, text, confidence}]` |

**三条纪律**：保留时间码/页码（`locator`）、保留 ASR 置信度（低置信优先人工核对）、
**不做任何"顺手"加工**（清洗是下一层的事，分层是为了出错时能定位）。

**PDF 文字层体检**：PDF 分三种——文字版（直接可用）、图文混排（**部分页静默缺失，最危险**）、
纯扫描版（整份等于空的）。全部页为空、或部分页为空，都会写进 `warnings`，**绝不静默丢弃**。

#### 教案分节抽取（DOCX 专属，D1/D2 最省力的入口）

教案是**唯一"教研已经用结构化方式写过一遍"的素材**：教学目标、重点难点、教学过程、
作业布置通常都有明确小标题。因此 `parse_docx_auto` 对教案做**分节抽取**，
直接产出可映射到交付物的结构：

| 抽取字段 | 映射到 |
|---|---|
| `sections.objectives` | `kp.objectives`（D1） |
| `sections.key_points` | `lesson.key_points`（D2） |
| `sections.difficult_points` | `lesson.difficult_points` / `kp.common_misconceptions` |
| `process_steps[{index,name,text}]` | `lesson.episodes` 骨架（D2） |
| `sections.homework` | `episode(TI_HOMEWORK)` 内容 |

处理过两个真实形态问题，都有回归用例兜着：
- **封面表格** `学科 | 数学 | 年级 | 三年级` 必须**相邻两格配对**；
  按"第一个分隔符切分"会得到 `学科="数学 | 年级 | 三年级"`、年级直接丢失。
- **标题与正文同行** `二、教学重点：建立毫米和分米的长度观念。` 必须切出小节；
  只按整段匹配别名会漏掉该节，内容还会被错误并进上一节（实测并进了"教学目标"）。

判定阈值刻意保守：**至少命中 2 个小节别名才算教案**。
宁可把教案当普通文本（损失结构），也不要把讲义误判成教案（产出错误的 episode 骨架）。


### s2_clean —— 清洗（合规关键路径）
处理顺序是**有意设计**的：去页眉页码 → 去口头禅 → **隐私脱敏** → 标点归一。
脱敏放在去口头禅**之后**，否则删词会把手机号切成两半导致漏检。

脱敏覆盖：手机号/座机、身份证号、学生姓名、学校名、QQ/微信。

**关于学生姓名**：中文姓名没有词边界，不能"见到 2-4 个汉字就脱敏"——
那会把「各位同学」「老师」「大家一起」全打成 `{student_name}`，文本直接不可用。
所以采用**高精度上下文锚点法**：
- 面称：`<姓名>同学 / <姓名>小朋友`
- 动作框架：`<姓名>说/回答/举手/来/试试`
- **自我介绍**：`我是<…>班的<姓名>` / `我叫<姓名>`
并配 `NAME_STOPWORDS` 白名单防误伤。

> **已知局限（验收时必须知情）**：纯锚点法召回率不是 100%。
> 例如「张小明，你来读一下」这种无锚点呼名会漏掉。
> 因此清洗结束会做**漏检自检**，命中疑似人名/IM 号时输出显式 warning，**不静默放过**。
> **生产环境必须叠加姓名词典**（从花名册生成，词典本身加密存储、不入 Git），锚点法作兜底。

**清洗的边界**：清洗**不是**把口语改成书面语。
「来，我们一起数一数」清洗后应该还是「来，我们一起数一数」——
**教学语言的口语感是资产不是噪声**。把口语润色成书面语，AI 教师听起来就会像播音员。

### s3_chunk —— 三级切分
```
第一级 按教学环节切     cue 词：我们来看 / 接下来 / 请大家 / 我们总结一下 / 这道题
第二级 按知识点边界切   同一 kp 内不超过 max_chars（默认 220），跨 kp 必断
第三级 按交互点切       提问、练习、等待作答的位置 —— 决定编排引擎在哪插 L2 交互
```
**核心原则**：一个切片 = **一次连续发声，中间不插任何交互**。要插交互就断成两段。

不同素材切分单位不同：课件**按页切块**（一页=一个教学单元）、教材按自然段、
直播转写按环节→知识点、答疑记录**一问一答配对不做切分**。

所有草稿 `status: draft`，`review_notes` 标注"机器切分，待教研确认边界"。

### s4_autolabel —— 人机协同的机器侧
提供 `Labeler` 抽象基类、`RuleLabeler`（纯规则，离线可跑）、`LLMLabeler`（接入桩）。

`teacher_intent` 判定优先级（**本模块最容易出错的地方**）：
```
1. 表扬 / 纪律口令（安静 / 坐好 / 看黑板）—— 整句性质的管理与鼓励
   · 但「我们来看黑板上这道题」里的"看黑板"是引出例题的定语，靠紧邻前字（来/看/观察）判定
2. 具体教学动作（示范 / 练习 / 提问 / 讲授），按强→弱排序
3. 弱纪律信号（注意看）
4. 过渡泛化信号（我们继续 / 接下来）—— 不允许单独决定 intent，需人工指定
```
> 历史 bug（已修）：纪律旁证判定曾写成 `prefix.endswith("来看")`，
> 但关键字是「看黑板」，"看"正是它的首字，两字"来看"与关键字重叠、**永远落不进 prefix**，
> 导致该旁证从不生效——「我们来看黑板上这道题，老师先示范一遍」被误判为 `TI_MANAGE`。
> 现改为匹配紧邻关键字的单字动词，并有回归用例固化。

**所有预标注结果一律带 `prelabel_confidence` 与 `needs_human_review: true`**，
置信度低于阈值的进 `review_priority=1`（排审核队列前面）。
**机器绝不产出 `approved`。**

### s5_qa_mine —— 候选问答对挖掘
用问号 + 疑问词（怎么/为什么/是不是/多少/能不能）识别学生提问，取紧随其后的老师回答。
输出只含 `canonical_question` / `raw_answer` / `sources` / `evidence_count` / `status: draft`。

> **`paraphrases` / `negative_examples` / `policy` 必须由教研手工补齐，机器不生成。**
> 这三样是问答对质量的关键，机器生成的会污染库。

`evidence_count`（同一问题在历史数据里出现几次）**决定优先级**——
数千条候选里真正高频的可能只有一两百条，先做这些。

### s6_export —— 门禁与编排（不实现导出）
```
门禁 1  schema 校验（结构化检查）
门禁 2  tools/lint.py（66 条业务规则，0 ERROR 放行）
门禁 3  tools/build_bundle.py（引用完整性硬门禁）
```
**导出逻辑只有一份实现**：`tools/build_bundle.py`。s6 通过子进程调用它，
这样做是为了保证「导出门禁与 CI 用的是同一份规则」——
否则哪天 lint 升级了，两套标准必然漂移。

退出码：`0` 成功；`1` 入参/工具缺失；`2` 门禁未通过。

---

## 4. 目录约定

| 目录 | 内容 | 入版本控制 |
|---|---|---|
| `rag/offline/` | 代码 | ✅ |
| `data/raw/` | 原始素材 | ❌ **禁止入库**（含隐私与大文件） |
| `data/parsed/` `data/clean/` `data/chunks/` `data/labels/` `data/qa_candidates/` | 中间产物 | ❌ 可再生产物 |
| `data/export/` | 交付包 | ❌ 可再生产物 |
| `samples/` | **人工审校后的真源** | ✅ 这是真源 |

`.gitignore` 已排除 `data/` 与常见素材后缀。
**原始素材绝对不能入库**——里面含未成年人个人信息。

---

## 5. 测试

```powershell
python tests/test_pipeline.py      # 60 个用例，无需第三方库即可跑通（待搬迁：原 pipeline/tests/ → tests/）
```

覆盖：脱敏正则（手机号/身份证/姓名/IM）、口头禅清洗、教学感知切分（超长会切/短的不切）、
`RuleLabeler` 的意图与优先级边界、导出产物可被 schema 校验，
以及 **`TestWindowsEncodingRegression`**——中文 Windows 上「stdout 被管道捕获」时的编码崩溃。

**纪律**：新增或修改任何正则、cue 词表、意图规则，**必须同步补回归用例**。
本模块已有两次真实教训：
- 规则表里写了纪律旁证判定，但因字符串重叠从不生效，没有回归用例就一直没被发现；
- 编码用例全都给子进程注入了 `PYTHONIOENCODING=utf-8`，**测试"帮了忙"**，
  于是 GBK 管道下工具崩溃的问题在 52 个用例全绿的情况下依然上线了。
  **凡验证编码健壮性的用例，必须跑在未加修饰的环境里。**

---

## 6. 未实现 / 待接入（TODO）

| 项 | 说明 |
|---|---|
| **`LLMLabeler` 真实模型接入** | 目前是桩：需传入 `llm_call: (prompt) -> dict\|str`。`--show-prompt` 可先评审 prompt 方案 |
| **FunASR 自动执行** | 只打印建议命令，不执行（ASR 应在 GPU 机器上离线跑） |
| **标点恢复模型** | ASR 输出常无标点，目前靠规则补，长句标点质量有限 |
| **姓名词典** | 生产环境必须叠加（见 s2 已知局限）；词典需加密存储、不入 Git |
| **OCR（板书照片）** | 未实现 |
| **`s1_parse` 授权校验** | 应在入口校验 `consent_record` 存在，目前只在文档中约束 |
| **跨课通用话术复用率统计** | 工具已就位：`tools/scripts_mine.py --all-lessons` 自动发现多课逐字稿并在报告 `reuse_stats` 输出复用率（≥2 课才计）。当前仅 G7A-U01 一课数据，统计为 `null`；待批量下载/转写（受媒体 CDN 出口限制）产出多课后即生效 |
