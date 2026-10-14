# -*- coding: utf-8 -*-
"""在线链路配置加载（P0）。

设计要点（契约 pipeline/online/README.md + docs/16 P0）：
* 全部配置来自同目录 ``config.yaml``；
* **密钥只从环境变量读**：``DASHSCOPE_API_KEY`` / ``OPENAI_API_KEY`` / ``MYSQL_PASSWORD``
  等，绝不写进任何文件；yaml 里只有非密的本地 dev 默认值；
* 用 dataclass 暴露结构化字段，带合理默认，便于单测与类型提示。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class MySQLConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = "root"
    database: str = "ai_teacher"
    charset: str = "utf8mb4"


@dataclass
class MilvusConfig:
    uri: str = "http://127.0.0.1:19530"
    collections: dict[str, str] = field(default_factory=lambda: {
        "qa": "qa_semantic", "kp": "kp_semantic",
        "chunks": "kp_chunks", "utterance": "utterance_semantic"})


@dataclass
class Neo4jConfig:
    uri: str = "bolt://127.0.0.1:7687"
    user: str = "neo4j"
    password: str = "12345678"
    database: str = "neo4j"


@dataclass
class EmbeddingConfig:
    model_path: str = r"C:\Users\Lenovo\Desktop\bge-m3"
    dim: int = 1024
    normalize: bool = True


@dataclass
class RerankConfig:
    """P3 精排（可选）。bge-reranker-base 需额外依赖 + GPU；无 GPU/未获批时保持关闭，
    关闭时 RRF 序即为最终序——接口留好，enabled=True 且 model_path 有效才真正加载。"""
    enabled: bool = False
    model_path: str = ""        # bge-reranker-base 本地目录（获批后填）
    top_n: int = 5             # 送入精排的候选数（只精排融合后的头部，省算力）


@dataclass
class RewriteConfig:
    """P4 查询改写（链路第 1 步）。两级：
    * 规则级（默认开）：错别字/异体字词典归一 + 空白/标点归一（不联网、不调模型）；
    * LLM 级（可配开关，默认关）：指代消解与补全（“它怎么算”→补全指代对象），输出独立 query。
    护栏：改写前后 bge 余弦 < drift_threshold 视为语义漂移，放弃改写回退原 query。"""
    enabled: bool = True            # 规则级总开关
    typo_fix: bool = True           # 错别字/异体字词典归一
    punct_norm: bool = True         # 空白/标点归一
    llm_enabled: bool = False       # LLM 级指代消解/补全（需 Ollama）；默认关，无历史/单轮时尤其应关
    drift_threshold: float = 0.6    # 改写前后余弦低于此值→判漂移→回退原 query


@dataclass
class SessionConfig:
    """P6 多轮会话（session.py）。近 N 轮问答存 Redis，供代词消解与上下文摘要复用。
    * Redis 不可用时降级进程内存 dict（session.py 内告警），不阻断主链；
    * ``topic_reset_threshold``：本轮 query 与上一问 bge 余弦低于此值→判换主题→重置上下文
      （不开新 session id）；
    * ``ttl_seconds``/``max_turns``：会话过期时间与保留轮数。"""
    enabled: bool = True              # 关→整条链路退回单轮（不读写会话存储）
    ttl_seconds: int = 1800           # 会话 TTL，默认 30min
    max_turns: int = 5                # 只保留最近 N 轮
    key_prefix: str = "rag:session:"  # Redis key 前缀：rag:session:{id}
    topic_reset_threshold: float = 0.41   # 低于此 bge 余弦判为换主题→重置上下文
    context_summary_turns: int = 2    # 生成摘要最多附带最近 N 轮已确认上下文


@dataclass
class RetrievalConfig:
    top_k: int = 5
    final_k: int = 3
    # ---- P4.5 门控阈值分档（先测量后定值，见 data/stores/reports/gate_threshold_calibration.md）----
    # 三路各自独立稠密余弦阈值，替代此前 qa/kp/chunks 共用的单一 score_threshold。
    # 值系 34 条评测集实测「每条最高正确命中 vs 最高误命中」校准取带内保守值，margin≥0.05。
    qa_threshold: float = 0.82       # qa 路（5 条精选问答）：误匹配代价高（错误 fixed 劫持），整体上抬；a 类 top 正确≥0.8945、b/c/d top 误命中≤0.6975
    kp_threshold: float = 0.67       # kp_semantic 路证据稠密下限：c/d top 误命中 kp≤0.6069、b top 正确 kp≥0.7367（带宽充裕）
    chunks_threshold: float = 0.648  # kp_chunks 路证据稠密下限：c/d chunks 误命中上界0.6327 与 b 正确下界0.6652 带仅0.0325（取中偏保守，见校准报告局限声明）
    allowed_status: list[str] = field(default_factory=lambda: ["approved", "candidate"])
    qa_text_kinds: list[str] = field(default_factory=lambda: ["canonical", "paraphrase"])
    # P2 集合路由：kp_chunks（旧，默认）或 kp_chunks_v2（新，带 parent_id/chunk_strategy）。
    chunks_collection: str = "kp_chunks"
    # 仅对 v2 生效：只允许这些 chunk_strategy 进入检索（空=不过滤）。
    # 用于同一 v2 集合内对比 P2-semantic vs P2-parent_child（非 transcript 行标为 default）。
    chunks_strategies: list[str] = field(default_factory=list)
    # ---- P3 混合检索 ----
    hybrid_enabled: bool = True       # 关→退回纯 dense（等价 V1），便于 A/B 对照
    rrf_k: int = 60                   # RRF 平滑常数（k 越大越看重尾部名次）
    sparse_top_k: int = 10            # 稀疏路候选数（>dense top_k 以扩大召回）
    local_bm25: bool = True           # 本地 rank-bm25 兜底（kp.search_text / qa.canonical_question）
    hybrid_floor: float = 0.85        # 仅被稀疏路命中（dense 未过阈值）时的 BM25 归一化下限，才允许进入上下文
    rerank: RerankConfig = field(default_factory=RerankConfig)


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"


@dataclass
class DashScopeConfig:
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"
    # 只从环境变量读，默认 None；不落文件。
    api_key: str | None = None


@dataclass
class CitationConfig:
    require_in_evidence: bool = True
    max_retry: int = 1


@dataclass
class GenerationConfig:
    backend: str = "ollama"
    timeout_s: int = 60
    max_new_tokens: int = 512
    temperature: float = 0.2
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    dashscope: DashScopeConfig = field(default_factory=DashScopeConfig)
    citation: CitationConfig = field(default_factory=CitationConfig)


@dataclass
class Config:
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    policy: dict[str, Any] = field(default_factory=dict)
    compose: dict[str, Any] = field(default_factory=dict)
    redis_url: str = "redis://127.0.0.1:6379/0"
    session_id_default: str = "p1-local"
    raw: dict[str, Any] = field(default_factory=dict)


def _sub(d: dict[str, Any], key: str) -> dict[str, Any]:
    val = d.get(key)
    return val if isinstance(val, dict) else {}


def load_config(path: Path | str | None = None) -> Config:
    """从 yaml 读配置，并用环境变量覆盖密钥类字段（密钥不进文件）。"""
    import yaml  # 延迟导入，未装 yaml 时不拖累其它模块

    cfg_path = Path(path) if path else CONFIG_PATH
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    my = _sub(data, "mysql")
    mysql = MySQLConfig(
        host=my.get("host", "127.0.0.1"), port=int(my.get("port", 3306)),
        user=my.get("user", "root"),
        # 环境变量优先，其次 yaml 默认，最后 root。
        password=os.environ.get("MYSQL_PASSWORD") or my.get("password", "root"),
        database=my.get("database", "ai_teacher"), charset=my.get("charset", "utf8mb4"))

    mv = _sub(data, "milvus")
    milvus = MilvusConfig(uri=mv.get("uri", "http://127.0.0.1:19530"),
                          collections=_sub(mv, "collections") or MilvusConfig().collections)

    neo = _sub(data, "neo4j")
    neo4j = Neo4jConfig(uri=neo.get("uri", "bolt://127.0.0.1:7687"),
                        user=neo.get("user", "neo4j"),
                        password=os.environ.get("NEO4J_PASSWORD") or neo.get("password", "12345678"),
                        database=neo.get("database", "neo4j"))

    emb = _sub(data, "embedding")
    embedding = EmbeddingConfig(model_path=emb.get("model_path", EmbeddingConfig.model_path),
                                dim=int(emb.get("dim", 1024)),
                                normalize=bool(emb.get("normalize", True)))

    ret = _sub(data, "retrieval")
    mv_coll = milvus.collections
    rrk = _sub(ret, "rerank")
    retrieval = RetrievalConfig(top_k=int(ret.get("top_k", 5)), final_k=int(ret.get("final_k", 3)),
                                qa_threshold=float(ret.get("qa_threshold", 0.82)),
                                kp_threshold=float(ret.get("kp_threshold", 0.67)),
                                chunks_threshold=float(ret.get("chunks_threshold", 0.648)),
                                allowed_status=list(ret.get("allowed_status", ["approved", "candidate"])),
                                qa_text_kinds=list(ret.get("qa_text_kinds", ["canonical", "paraphrase"])),
                                # 优先 retrieval.chunks_collection，其次 milvus.collections.chunks。
                                chunks_collection=str(ret.get("chunks_collection")
                                                      or mv_coll.get("chunks", "kp_chunks")),
                                chunks_strategies=list(ret.get("chunks_strategies", [])),
                                # P3 混合检索
                                hybrid_enabled=bool(ret.get("hybrid_enabled", True)),
                                rrf_k=int(ret.get("rrf_k", 60)),
                                sparse_top_k=int(ret.get("sparse_top_k", 10)),
                                local_bm25=bool(ret.get("local_bm25", True)),
                                hybrid_floor=float(ret.get("hybrid_floor", 0.85)),
                                rerank=RerankConfig(enabled=bool(rrk.get("enabled", False)),
                                                    model_path=str(rrk.get("model_path", "")),
                                                    top_n=int(rrk.get("top_n", 5))))

    gen = _sub(data, "generation")
    cita = _sub(gen, "citation")
    oll = _sub(gen, "ollama")
    dsh = _sub(gen, "dashscope")
    generation = GenerationConfig(
        backend=gen.get("backend", "ollama"),
        timeout_s=int(gen.get("timeout_s", 60)),
        max_new_tokens=int(gen.get("max_new_tokens", 512)),
        temperature=float(gen.get("temperature", 0.2)),
        ollama=OllamaConfig(base_url=oll.get("base_url", "http://localhost:11434"),
                            model=oll.get("model", "qwen2.5:7b")),
        dashscope=DashScopeConfig(base_url=dsh.get("base_url", DashScopeConfig.base_url),
                                  model=dsh.get("model", "qwen-plus"),
                                  # 密钥只来自环境变量，绝不来自 yaml。
                                  api_key=os.environ.get("DASHSCOPE_API_KEY")
                                  or os.environ.get("OPENAI_API_KEY")),
        citation=CitationConfig(require_in_evidence=bool(cita.get("require_in_evidence", True)),
                                max_retry=int(cita.get("max_retry", 1))))

    pol = _sub(data, "policy")
    comp = _sub(data, "compose")
    redis = _sub(data, "redis")
    timing = _sub(data, "timing")
    rw = _sub(data, "rewrite")
    rewrite = RewriteConfig(enabled=bool(rw.get("enabled", True)),
                            typo_fix=bool(rw.get("typo_fix", True)),
                            punct_norm=bool(rw.get("punct_norm", True)),
                            llm_enabled=bool(rw.get("llm_enabled", False)),
                            drift_threshold=float(rw.get("drift_threshold", 0.6)))
    se = _sub(data, "session")
    session = SessionConfig(enabled=bool(se.get("enabled", True)),
                            ttl_seconds=int(se.get("ttl_seconds", 1800)),
                            max_turns=int(se.get("max_turns", 5)),
                            key_prefix=str(se.get("key_prefix", "rag:session:")),
                            topic_reset_threshold=float(se.get("topic_reset_threshold", 0.41)),
                            context_summary_turns=int(se.get("context_summary_turns", 2)))
    return Config(mysql=mysql, milvus=milvus, neo4j=neo4j, embedding=embedding,
                  retrieval=retrieval, rewrite=rewrite, generation=generation, session=session,
                  policy=pol, compose=comp,
                  redis_url=redis.get("url", "redis://127.0.0.1:6379/0"),
                  session_id_default=timing.get("session_id_default", "p1-local"), raw=data)


# 进程内单例（cli/pipeline 复用）。
_CACHE: Config | None = None


def get_config(reload: bool = False) -> Config:
    global _CACHE
    if _CACHE is None or reload:
        _CACHE = load_config()
    return _CACHE
