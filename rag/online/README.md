# pipeline/online — 在线 RAG 检索增强问答运行时

> 架构依据：团队 RAG 双流程设计（2026-09-20），实施按 docs/16 六步迭代。
> **铁律：在线链路对 MySQL / Milvus / Neo4j 只读。** 写库只发生在离线链路。

## 在线请求链路（与模块对应）

```
用户 query（自然语言，可能含错别字/歧义/代词）
  │
  ├─ 0. session.py         多轮会话（传 session_id 且 session.enabled 时）：载入最近 N 轮历史；
  │                          本轮与上一问 bge 余弦<阈值→判换主题、重置上下文（不换 session id）
  ├─ 1. query_rewrite.py   查询改写：规则级错别字/标点归一（默认开）+ LLM 级指代消解/补全（可配）
  │                          改写前后 bge 余弦<阈值判语义漂移→回退原 query；多轮时注入会话 history 消解代词
  ├─ 2. embedder.py        bge-m3 编码（1024 维，与离线入库同模型）
  ├─ 3. retrieval.py       混合检索：Milvus dense+sparse 混合搜索 + bm25 关键字
  │                          —— 六步法第三步启用混合；V1 纯 dense 起步
  ├─ 4. rerank.py          RRF 融合排序 →（可选）bge-reranker 精排，阈值过滤 → top-k
  │                          门控阈值分档(P4.5)：qa/kp/chunks 三路各用独立 dense 阈值
  │                          (qa_threshold/kp_threshold/chunks_threshold)，替代旧单一 score_threshold；
  │                          matched_by(exact/paraphrase/vector) 按本路阈值判定，值见 config.yaml + 校准报告
  ├─ 5. context.py         上下文组装：top-k 合并去重、父块扩展、引用登记（kp_id/页码/出处）
  ├─ 6. policy.py          策略路由（项目特有）：safety_policy_matrix / in_scope / refuse；
  │                          fixed 直接返回答案文本，不进 LLM
  ├─ 7. generation.py      提示词构建 + LLM 生成：rag_grounded 强制引用 [kp_id]，
  │                          引用校验失败重试一次，再失败降级；第 2 轮起前置"已确认上下文"摘要（近 N 轮，控 token）
  ├─ 8. compose.py         话术组装（项目特有）：utterance_template 槽位填充包装答案
  └─ 9. pipeline.py        编排器（分段计时）→ cli.py 命令行入口（--chat 多轮交互 / --json 带 session_id）
                           末尾 session.append 写回本轮（question/rewritten/answer/citations）到 Redis
```

## 其余模块

| 模块 | 职责 | 所属阶段 |
|------|------|---------|
| config.py / config.yaml | 全部配置；密钥只从环境变量读 | P0 |
| session.py | 多轮会话：标准库 socket 自实现 RESP 客户端，历史存 Redis `rag:session:{id}`(TTL 30min/最近 N 轮)，不可用时降级进程内存并告警；主题一致性(bge 余弦重置)、代词消解上下文 | P6 |
| eval/ragas_eval.py + eval_set.jsonl | RAGAS 四维评测（faithfulness / answer_relevancy / context_precision / context_recall）+ badcase 清单；每阶段同一测试集对比出指标 | P1 起 |
| eval/eval_multi.py + eval_set_multi.jsonl | 多轮会话评测（12 组/26 轮）：代词消解、主题切换重置、拒答、首轮路由、逐轮综合；产 `rag_eval_P6.md` | P6 |

## 统一返回结构

```json
{"answer": "...", "modality": "fixed|rag_grounded|template|refuse",
 "citations": [{"kp_id": "...", "evidence": "...", "page_no": 12, "score": 0.87}],
 "policy": "...", "latency_ms": {"total": 0, "retrieve": 0, "generate": 0},
 "matched_by": "exact|paraphrase|vector|no_match",
 "rewrite": {"original": "...", "rewritten": "...", "level": "none|rule|llm",
              "changed": true, "drift": false, "cosine": 0.93},
 "session": {"turn_no": 2, "store": "redis", "topic_switch": false, "context_turns": 1},
 "session_id": "..."}
```

> `citations` 固定四键（`kp_id/evidence/page_no/score`）不可增删；`rewrite` 为 P4 新增的查询改写日志（检索用 `rewritten`，策略/生成仍以 `original` 为准）。
> `session` 为 P6 新增的多轮会话日志（仅当传入 `session_id` 且 `session.enabled` 时出现）：`turn_no` 当前轮次、`store` 会话后端(`redis`/`memory`)、`topic_switch` 本轮是否判换主题、`context_turns` 实际注入的上下文轮数。不传 `session_id` 时该键缺省，行为与单轮（P4）逐字节一致。

## 硬约束

1. 只读三存储；写库只允许离线链路（且须版本化、经 WorkBuddy 验收）
2. 嵌入与离线侧同模型（bge-m3, 1024 维），禁止混用
3. 新增依赖（ragas / rank-bm25 / sentencepiece 等）必须先申请登记 requirements.txt
4. 无 LLM key 时 fixed / refuse / template 路径必须完整可用
5. 模块布局以本 README 为契约；每个模块配单测（tests/test_rag_online_*.py）
