# rag — 知识检索（三层布局）

知识检索相关的全部代码。2026-10-10 由原 `pipeline/` 包并入重组为三层：**轻量检索（主链路在用）** + **离线数据流水线** + **在线检索增强问答运行时**（后两者为重依赖独立链路）。

## 三层布局

| 层 | 目录 | 用途 | 依赖 |
| --- | --- | --- | --- |
| 轻量检索 | `retriever.py` | 字符 bigram 相似度检索，从 `data/textbook`、`data/teaching_materials`、`data/curriculum`、`data/question_bank` 取上下文注入提示词 | 零重依赖，被 `core/harness.py` / `mcp_tools` 直接使用 |
| 离线流水线 | `offline/` | 把原始素材（直播录像 / 课件 / 答疑记录 / 题库 / 教材）加工成教研可审校的结构化草稿，六个 stage（s1_parse → s2_clean → s3_chunk → s4_autolabel → s5_qa_mine → s6_export） | pyyaml / jsonschema 等 |
| 在线问答 | `online/` | 检索增强问答运行时：查询改写 → bge-m3 编码 → Milvus 混合检索 → RRF + 重排 → 策略路由 → 引用校验生成 → 话术组装；含 Redis 多轮会话与 RAGAS 评测 | sentence_transformers / Milvus / Redis |

各子目录的完整设计见其自身 README：[`offline/README.md`](offline/README.md)、[`online/README.md`](online/README.md)。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | 包门户：顶层只导出轻量的 `Document` / `Retriever` / `build_retriever` / `load_documents`；`online` / `offline` 用 PEP 562 `__getattr__` **惰性加载**。 |
| `retriever.py` | 轻量检索实现（`Document` / `Retriever` / `build_retriever`）。 |

## 设计要点

- **顶层 import 保持轻量**：`online` / `offline` 不在 `rag/__init__.py` 自动 import——否则所有 `import rag` 的调用方（`mcp_tools/server.py`、`core/harness.py`）都要为 sentence_transformers / Milvus 付出加载成本。
- **主链路只依赖轻量检索**：`core/harness.py` 与 MCP 的 `knowledge_search` 工具只走 `rag.retriever`；重依赖链路可独立部署、独立评测（`python -m rag.online.cli`）。
- **生产可平滑替换**：轻量检索的接口（`Retriever`）与向量实现同构，替换实现不影响上层。
- **只读铁律（online/offline 共同约束）**：在线链路对 MySQL / Milvus / Neo4j 只读，写库只发生在离线链路。
