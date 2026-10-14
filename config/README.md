# config — 配置与静态元数据

全部配置与「静态事实源」的落点：环境变量映射（`settings.py`）、学科清单（`subjects.py`）、教师档案（`teachers.py`）、全局常量（`constants.py`）、日志（`logging.py`）、教师技能 md 解析（`teacher_skills/`）。**运行期现读现改现生效**，配置集中于此，业务代码不散落阈值。

## 文件说明

| 文件 / 子包 | 用途 |
| --- | --- |
| `__init__.py` | 包门户。 |
| `settings.py` | **全部环境配置**：pydantic-settings 把 `.env` 映射为强类型 `Settings`（模型端点 / 音视频 / 白板 / 课程 / 记忆 / 活动日志 / 真题库 / 认证…），启动时校验，`get_settings()` 单例。 |
| `constants.py` | 全局枚举与策略常量：`MicState`（麦控状态机）、`BoardOp`（白板指令类型）、`IntentType`（意图路由类型）与课堂编排参数——阈值集中在策略层，不散落在业务代码。 |
| `subjects.py` | **11 科学科元数据**（key / 中文名 / emoji / 描述 / 学段 / 主题色）：学科选择页与后端提示词的单一事实来源；新增学科只改这里。另含 `resolve_subject()` 归一化与 `detect_other_subject()` 学科越界检测。 |
| `teachers.py` | **教师档案**：扫描 `data/teachers/<key>/teacher.json`，提供 `teacher_for_subject()`（默认老师）/ `teacher_variants()`（同科多老师）/ `default_teacher_key()` 等；`teacher.json` 的 `subject` 字段把变体 key（如 `math_strict`）归类到学科。 |
| `logging.py` | 统一日志（loguru）：`setup_logging()` 幂等初始化 + `get_logger(module)` 带模块上下文。 |
| `teacher_skills/` | `SKILL.md` 的解析与消费侧装配（见该目录 README）。 |

## 设计要点

- **一个事实源**：学科清单只有 `subjects.py` 一处（前端 `/?api/subjects` 拉取、后端提示词与护栏同源），避免「前后端各写一份清单」。
- **目录名三对齐**：`.agents/skills/<key>/` = `config/subjects.py` 的 `key` = `data/teachers/<key>/`，少一处即回退降级。
- **敏感信息只走环境变量**：`Settings` 不落任何默认密钥（`AUTH_SECRET` / `ANTI_REV_SECRET` 的默认值仅为开发占位，生产必须替换）。
- **开关即降级点**：`AGENT_*` / `MEMORY_*` / `COURSE_*` / `EXERCISE_BANK_*` / `ACTIVITY_LOG_*` 等每个开关都有明确的「关闭后的回退行为」（见 `.env.example` 逐项注释）。
