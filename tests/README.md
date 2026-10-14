# tests — 单元与端到端测试

25 个 `test_*.py`、约 280 个用例。运行：

```bash
pip install -e ".[dev]"
pytest -q
```

配置在 `pyproject.toml`（`asyncio_mode = "auto"`，async 测试无需装饰器）。

## 文件说明

| 文件 | 覆盖内容 |
| --- | --- |
| `conftest.py` | **测试隔离**：清空密钥、强制 `LLM/ASR/TTS/AVATAR=mock`、`ORCHESTRATOR=builtin`、`BOARD_FIRST_MODE=off`、关真题库——保证测试不依赖外网与外部服务。 |
| `test_supervisor.py` | Supervisor 端到端：回合决策、分流、异常回退。 |
| `test_graph_parity.py` | **双编排链路一致性**：builtin TaskGraph 与 LangGraph 逐节点行为对照（强绑定）。 |
| `test_task_graph.py` | 自研 TaskGraph：拓扑执行、跳过 / 早退约定。 |
| `test_agent_nodes.py` | 节点地图完整性（谁负责哪些节点、连接顺序、依赖配置）。 |
| `test_lecturer_collaboration.py` | 讲课 / 答疑契约隔离与 Agent 协作协议。 |
| `test_intent_analyzer.py` | 意图识别：规则直判、小模型兜底、多意图拆分、超时回退。 |
| `test_skill_md.py` | 教师技能 md 契约（frontmatter 与正文小节解析）。 |
| `test_teacher_personality.py` | 11 科人格加载与九项行为兜底。 |
| `test_skill_experience.py` | 经验自我更新：复盘提炼、注入、停用 / 恢复。 |
| `test_teaching_emphasis.py` | 重点标红：`**...**` 解析与剥离（绝不进语音 / HTML）。 |
| `test_agent_autonomy.py` | **Agent 自主性四条链路**：工具调用与降级、ReAct 收敛与步数上限、回合规划与规则兜底、学情记忆隔离 / 去重 / 注入 / 删除。 |
| `test_teacher_quiz.py` | 出题判题：三级降级（真题库 → LLM → 规则）、判分与追问。 |
| `test_mic_control.py` | 麦控状态机：VAD 迁移、barge-in。 |
| `test_validator.py` | 回复校验：话术长度 / 答案泄露 / 坐标合法性。 |
| `test_subject_guardrail.py` | 学科护栏：范围限定、越界引导、内容安全与 AI 标识。 |
| `test_whiteboard_layout.py` | 白板指令排版与坐标处理。 |
| `test_courseware_store.py` | 课件仓库：扫描、清单、取页。 |
| `test_pptx_layout.py` | pptx 元素框与文字宽度估算。 |
| `test_document_region.py` | 题目区域定位：分块、面积上限、「宁可不画框」边界。 |
| `test_a2a.py` | A2A 协议闭环：Card、`message/send`、`tasks/get`、`input-required` 续跑。 |
| `test_mcp_server.py` | MCP 协议闭环：握手、tools/list、tools/call。 |
| `test_auth.py` | 认证：密码哈希、JWT、REST / WS 鉴权。 |
| `test_activity_log.py` | 活动日志：写入、按天滚动、损坏行跳过、`node_start` 单侧判定。 |
| `test_llm_robustness.py` | LLM 输出健壮性：JSON 修复、异常输出降级。 |
| `skills/test_expression_skill.py` | 表情技能：情境判定、`_SKIP_HINTS` 拦截、未命中不拦截答疑。 |

## 设计要点

- **默认全 Mock**：所有测试在无密钥、无外部服务的环境跑通——与系统「缺失即降级」的设计对称。
- **一致性有单测兜底**：`test_graph_parity.py` 让两套编排引擎的行为差异在 CI 即暴露，而不是线上。
- **提交前纪律**（见根 README「贡献」）：`python scripts/selfcheck.py` + `pytest -q`；改了前端内联 JS 再跑 `node scripts/check_frontend_js.mjs`。
