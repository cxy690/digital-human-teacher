# core/skills/experience — 教师经验自我更新

回合结束后异步复盘，把「可复用教学经验」沉淀进 `data/teacher_experience/<老师key>.jsonl` 并注入后续提示词。`SKILL_SELF_UPDATE_ENABLE` 控制。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户：`get_experience_store()` 等；边界声明——**只沉淀经验、永远不改写人格提示词本体**。 |
| `models.py` | 经验条目：kind / hits / confidence / status；`score = hits × confidence` 决定注入优先级；文本 4~60 字。 |
| `prompt.py` | 把经验渲染成提示词片段注入该老师 system prompt（排在人格 Skill 之后）。 |
| `reflector.py` | 复盘提炼：从一次答疑中提取**可复用**经验（核心判据「换个学生、换个数字仍成立」）。 |
| `store.py` | append-only JSONL 存储：每次变更加一行快照，停用/回滚只是追加 status。 |

## 设计要点

- **只追加、可回滚**：停用（retire）/ 恢复（restore）都是追加一条状态变更，不就地改写、不删文件；`GET /api/skills/experience` 与 `POST /api/skills/experience/status` 提供人工查看与止损入口。
- **注入位置在人格之后**：像真老师「教了几年攒下的手感」——人格是本、经验是长出来的，越靠后越能影响输出。
- **复盘质量纪律**：禁写本题数字/答案/题号；宁空手而归不为凑数编造；产出再过长度/置信度/条数过滤。
- **按环境隔离**：`data/teacher_experience/` 已被 `scripts/sync_cloud.py` 排除——本地与云端各自学习，互不覆盖。
- **`SKILL_EXPERIENCE_MAX_ITEMS=0` 默认不限**：自动学到的经验不主动淘汰（不会「学着学着把以前学的丢了」），提示词侧条数由 `SKILL_EXPERIENCE_INJECT_LIMIT` 单独控制，文件再涨也不撑大 prompt。
