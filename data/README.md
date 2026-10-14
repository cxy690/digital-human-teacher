# data — 数据与素材

运行期数据、静态素材与模型缓存的统一落点。**代码与产物分离**：本目录是被 `.gitignore` 部分排除的运行时工作区，`scripts/sync_cloud.py` 只同步其中的静态素材，运行时产物（经验 / 记忆 / 缓存 / 检查点）按环境隔离。

## 目录说明

### 静态素材（随项目分发）

| 目录 | 内容 |
| --- | --- |
| `teachers/` | **各老师素材包**：`<老师key>/talk.mp4`（主底片）、手势视频（nod/shake/shrug/point）、`teacher.json`（档案与标定）、`voice_ref.wav`（克隆音色干声）；一个学科可有多位风格变体（如 `math` / `math_strict`）。 |
| `courseware/` | 课件：一份课件一个子目录（`deck.json` + `p{i}.png` 页图），由 `scripts/export_pptx_images.ps1` 从 pptx 导出。 |
| `curriculum/` | 学科目录与教学大纲：`subjects.json`（科目→数字人映射）、`primary_all.txt` / `junior_all.txt` / `math_grade5.txt` 等。 |
| `textbook/` | 教材文本（供 `rag/retriever.py` 检索注入）。 |
| `question_bank/` | 题库文本（供检索）。 |
| `error_patterns/` | 易错点库。 |
| `teaching_materials/` | 原始教学素材（老师底片源片、素材加工输入），如「数学老师.mp4」等。 |

### 运行时产物（按环境隔离，不随同步迁移）

| 目录 | 内容 |
| --- | --- |
| `teacher_experience/` | 教师经验 JSONL（`<老师key>.jsonl`，经验自我更新沉淀），`sync_cloud.py` 已排除——本地与云端各自学习。 |
| `memory/` | 学情记忆（`<student_id>.jsonl`，按学生物理隔离）。 |
| `wrong_book/` | 错题本（作业模式串联的错题记录）。 |
| `avatar_stream/` | 数字人单句视频与待机素材（`/media/avatar_stream` 静态暴露；超上限按 mtime 清理，`idle*` 长期资源不清理）。 |
| `raw/` | 会话检查点、白板快照等原始运行数据。 |
| `cache/` | 数字人人脸预处理缓存（按 `video_hash` 分子目录，换底片自动失效；清理用 `python -m scripts.clear_cache`）。 |

### 第三方模型缓存

| 目录 | 内容 |
| --- | --- |
| `models/paddlex/` | PaddleOCR 模型缓存（PaddleX 官方模型卡与权重）。因 PaddleX C++ 层不支持非 ASCII 路径，`document_parsing/ocr.py` 把缓存强制重定向到这里，**不要手动改路径**。 |

## 设计要点

- **目录即协议**：`teachers/<key>/`、`courseware/<deck_id>/` 的目录名同时也是学科 key / 课件 id，改名会破坏三处对齐（`config/subjects.py` / `.agents/skills/` / 本目录）。
- **换素材不改代码**：换老师只替换 `teachers/` 下的视频与 `teacher.json`；人脸缓存按内容哈希自动失效。
- **缺素材不阻断**：教师素材缺失时服务启动校验回退 `AI_TEACHER_FALLBACK_PATH` 并打 WARNING；课件缺页图时该页降级为纯文字。
- **隐私数据可整删**：`memory/` 支持按学生物理删除（`DELETE /api/memory/{student_id}`），面向未成年人合规。
