# -*- coding: utf-8 -*-
"""RAG 评测（P1）：规则化路由/答案指标 + RAGAS 四维（judge 走与生成相同的 Ollama 后端）。

用法：
    python -m rag.online.eval.ragas_eval              # 全量规则指标 + RAGAS 抽样
    python -m rag.online.eval.ragas_eval --no-ragas   # 只跑规则指标（快、确定性）
    python -m rag.online.eval.ragas_eval --limit 6     # RAGAS 只评前 6 条有参考的样本

两类指标：
  1) 规则（全 34 条）：模态路由准确率、策略码命中、fixed 答案精确、rag_grounded 证据覆盖
     （expect.kp_ids ⊆ 引用 kp_id）、超纲/安全拒答正确率、时延分位。→ badcase 来源。
  2) RAGAS（仅对能作答且有上下文的样本）：faithfulness / context_precision /
     context_recall 用 ragas collections（judge=本地 qwen2.5:7b）；answer_relevancy 用
     同后端反推问题 + bge-m3 余弦（ragas 版需额外 embeddings 依赖，这里自实现更省）。

judge/后端不可用时自动只出规则指标（降级，不阻断）。结果写 data/stores/reports/。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.online import config, embedder, generation, pipeline, store  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.jsonl"
REPORT_DIR = ROOT / "data" / "stores" / "reports"
CATEGORY = {"a": "库内QA直答", "b": "开放题需kp证据", "c": "超纲拒答", "d": "安全题"}


def load_items() -> list[dict]:
    with EVAL_SET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _kp_references() -> dict[str, str]:
    """开放题的独立参考=该 kp 的教学目标（与检索上下文非同一来源，避免自证）。"""
    try:
        return {r["kp_id"]: str(r.get("objectives") or r.get("name") or "")
                for r in store.query("SELECT kp_id, name, objectives FROM kp")}
    except Exception:  # noqa: BLE001
        return {}


def _cited_kps(res: dict) -> set[str]:
    return {c["kp_id"] for c in res.get("citations", []) if c.get("kp_id")}


def _recall_at_k(res: dict, expect_kps: list[str], k: int = 3) -> float | None:
    """recall@k：进入上下文的前 k 条证据（按融合序）覆盖了多少期望 kp。

    citations 已是 context 按 final_k 截断后的有序列表，其去重 kp_id 前 k 个即 top-k 证据。
    仅对声明了 expect.kp_ids 的样本（开放题）计算，否则返回 None。
    """
    if not expect_kps:
        return None
    ordered: list[str] = []
    for c in res.get("citations", []):
        kp = c.get("kp_id")
        if kp and kp not in ordered:
            ordered.append(kp)
    topk = set(ordered[:k])
    exp = set(expect_kps)
    return round(len(topk & exp) / len(exp), 4)


def _check(item: dict, res: dict) -> dict:
    """逐条对账：期望 vs 实际，产出是否通过 + 错误类型（供 badcase）。"""
    exp = item["expect"]
    got_mod, got_pol = res["modality"], res["policy"]
    issues = []
    if got_mod != exp["modality"]:
        issues.append(f"模态误判: 期望{exp['modality']} 实得{got_mod}")
    elif exp["policy"] and got_pol != exp["policy"]:
        issues.append(f"策略误判: 期望{exp['policy']} 实得{got_pol}")
    if exp["modality"] == "fixed" and exp["answer_ref"] and res["answer"] != exp["answer_ref"]:
        issues.append("fixed答案不符")
    if exp["modality"] == "rag_grounded" and exp["kp_ids"]:
        cited = _cited_kps(res)
        missing = [k for k in exp["kp_ids"] if k not in cited]
        if missing:
            issues.append(f"引用证据未覆盖期望kp: {missing}")
    if exp["modality"] == "refuse" and res["answer"] in ("", None):
        issues.append("拒答无话术")
    return {"qid": item["qid"], "category": CATEGORY[item["qid"][0]],
            "question": item["question"], "expect": exp, "actual": {
                "answer": res["answer"], "modality": got_mod, "policy": got_pol,
                "matched_by": res["matched_by"], "cited_kps": sorted(_cited_kps(res))},
            "rewrite": res.get("rewrite"),
            "passed": not issues, "issues": issues,
            "latency_ms": res["latency_ms"]}


def run_rules(items: list[dict], cfg) -> dict:
    refs = _kp_references()
    records = []
    for it in items:
        res = pipeline.ask(it["question"], cfg=cfg)
        rec = _check(it, res)
        rec["recall_at_3"] = _recall_at_k(res, it["expect"].get("kp_ids") or [])
        # 组装 RAGAS 样本（仅能作答且有上下文者）
        sample = None
        if res["modality"] in ("fixed", "rag_grounded") and res["citations"]:
            contexts = [c["evidence"] for c in res["citations"]]
            reference = it["expect"]["answer_ref"] or " ".join(
                refs.get(k, "") for k in (it["expect"]["kp_ids"] or []))
            sample = {"user_input": it["question"], "response": res["answer"],
                      "retrieved_contexts": contexts, "reference": reference}
        rec["ragas_sample"] = sample
        records.append(rec)
        print(f"[{ 'OK' if rec['passed'] else 'XX'}] {rec['qid']} {rec['question'][:24]}"
              f" -> {res['modality']}" + ("" if rec["passed"] else f" | {rec['issues']}"))
    return _aggregate(records)


def _aggregate(records: list[dict]) -> dict:
    total = len(records)
    by_cat: dict[str, dict] = {}
    for r in records:
        c = by_cat.setdefault(r["category"], {"n": 0, "passed": 0})
        c["n"] += 1
        c["passed"] += int(r["passed"])
    lats = [r["latency_ms"]["total"] for r in records]
    gats = [r["latency_ms"]["generate"] for r in records if r["latency_ms"]["generate"]]
    rec3 = [r["recall_at_3"] for r in records if r.get("recall_at_3") is not None]
    return {
        "n": total,
        "routing_accuracy": round(sum(r["passed"] for r in records) / total, 4),
        "recall_at_3": round(statistics.fmean(rec3), 4) if rec3 else None,
        "recall_at_3_n": len(rec3),
        "per_category": {k: {"n": v["n"], "acc": round(v["passed"] / v["n"], 4)}
                         for k, v in by_cat.items()},
        "latency_ms": {"total_p50": _pct(lats, 50), "total_p90": _pct(lats, 90),
                       "generate_p50": _pct(gats, 50)},
        "badcases": [r for r in records if not r["passed"]],
        "records": records,
    }


def _pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1))))
    return vals[k]


def _judge_llm(cfg):
    """RAGAS judge：复用生成的 Ollama，走其 OpenAI 兼容 /v1 端点。"""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    base = cfg.generation.ollama.base_url.rstrip("/")
    client = AsyncOpenAI(base_url=f"{base}/v1", api_key="ollama")
    return llm_factory(cfg.generation.ollama.model, client=client, adapter="instructor")


def _answer_relevancy(sample, backend) -> float | None:
    """自实现：同后端反推 3 个问题，与原问题做 bge-m3 余弦均值（省 ragas embeddings 依赖）。"""
    prompt = (f"请根据以下答案，反推出3个能被它回答的问题，每行一个，不要编号：\n{sample['response']}")
    try:
        lines = [ln.strip() for ln in backend.complete(prompt).splitlines() if ln.strip()][:3]
    except Exception:  # noqa: BLE001
        return None
    if not lines:
        return None
    qv = embedder.embed_one(sample["user_input"])
    return round(statistics.fmean(sum(a * b for a, b in zip(embedder.embed_one(r), qv))
                                  for r in lines) / len(lines), 4)


def run_ragas(summary: dict, cfg, limit: int) -> dict:
    samples = [(r, r["ragas_sample"]) for r in summary["records"] if r.get("ragas_sample")]
    if limit:
        samples = samples[:limit]
    if not samples:
        return {"skipped": True, "reason": "无可评样本（均需拒答或无上下文）"}
    try:
        from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

        llm = _judge_llm(cfg)
        f, cp, cr = Faithfulness(llm=llm), ContextPrecision(llm=llm), ContextRecall(llm=llm)
    except Exception as e:  # noqa: BLE001 - RAGAS/后端不可用时降级，只保留规则指标
        return {"skipped": True, "reason": f"RAGAS 初始化失败：{e}"}

    backend = generation.build_backend(cfg)
    rows = []
    for rec, s in samples:
        one = {"qid": rec["qid"]}
        try:
            one["faithfulness"] = f.score(user_input=s["user_input"], response=s["response"],
                                          retrieved_contexts=s["retrieved_contexts"]).value
            one["context_precision"] = cp.score(user_input=s["user_input"], reference=s["reference"],
                                                 retrieved_contexts=s["retrieved_contexts"]).value
            one["context_recall"] = cr.score(user_input=s["user_input"], reference=s["reference"],
                                             retrieved_contexts=s["retrieved_contexts"]).value
        except Exception as e:  # noqa: BLE001
            one["error"] = str(e)
        ar = _answer_relevancy(s, backend)
        if ar is not None:
            one["answer_relevancy"] = ar
        rows.append(one)
        print("RAGAS", one)
    dims = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    return {"n_scored": len(rows), "backend": f"ollama:{cfg.generation.ollama.model}",
            "mean": {d: round(statistics.fmean([r[d] for r in rows if d in r]), 4)
                     if any(d in r for r in rows) else None for d in dims},
            "per_item": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P1/P2 RAG 评测（同一测试集）")
    ap.add_argument("--no-ragas", action="store_true", help="只跑规则指标")
    ap.add_argument("--limit", type=int, default=8, help="RAGAS 评样本上限（0=全部）")
    ap.add_argument("--tag", default="P1", help="输出文件名后缀：eval_results_{tag}.json")
    ap.add_argument("--chunks-collection", default=None,
                    help="覆盖检索分块集合（如 kp_chunks_v2），不改动 config.yaml")
    ap.add_argument("--strategies", nargs="*", default=None,
                    help="仅 v2：允许进入检索的 chunk_strategy（如 default semantic parent_child）")
    ap.add_argument("--hybrid", choices=["on", "off"], default=None,
                    help="P3 混合检索开关；off=纯稠密（等价 P1/P2 门控），缺省用 config.yaml 值")
    ap.add_argument("--rrf-k", type=int, default=None, help="覆盖 RRF 平滑常数 k")
    ap.add_argument("--no-local-bm25", action="store_true", help="关闭本地 rank-bm25 兜底路")
    ap.add_argument("--rewrite", choices=["on", "off"], default=None,
                    help="P4 规则级改写总开关；缺省用 config.yaml 值")
    ap.add_argument("--rewrite-llm", action="store_true", help="开启 LLM 级指代消解/补全（需 Ollama）")
    ap.add_argument("--drift", type=float, default=None, help="覆盖语义漂移护栏阈值")
    args = ap.parse_args(argv)

    cfg = config.get_config()
    if args.chunks_collection:
        cfg.retrieval.chunks_collection = args.chunks_collection
    if args.strategies is not None:
        cfg.retrieval.chunks_strategies = list(args.strategies)
    if args.hybrid is not None:
        cfg.retrieval.hybrid_enabled = (args.hybrid == "on")
    if args.rrf_k is not None:
        cfg.retrieval.rrf_k = args.rrf_k
    if args.no_local_bm25:
        cfg.retrieval.local_bm25 = False
    if args.rewrite is not None:
        cfg.rewrite.enabled = (args.rewrite == "on")
    if args.rewrite_llm:
        cfg.rewrite.llm_enabled = True
    if args.drift is not None:
        cfg.rewrite.drift_threshold = args.drift
    print(f"[eval] chunks_collection={cfg.retrieval.chunks_collection} "
          f"strategies={cfg.retrieval.chunks_strategies or '(no filter)'} "
          f"hybrid={cfg.retrieval.hybrid_enabled} rrf_k={cfg.retrieval.rrf_k} "
          f"local_bm25={cfg.retrieval.local_bm25} rerank={cfg.retrieval.rerank.enabled}")
    print(f"[eval] gate: qa>={cfg.retrieval.qa_threshold} kp>={cfg.retrieval.kp_threshold} "
          f"chunks>={cfg.retrieval.chunks_threshold} hybrid_floor={cfg.retrieval.hybrid_floor}")
    print(f"[eval] rewrite: enabled={cfg.rewrite.enabled} llm={cfg.rewrite.llm_enabled} "
          f"typo={cfg.rewrite.typo_fix} punct={cfg.rewrite.punct_norm} "
          f"drift<{cfg.rewrite.drift_threshold}")
    embedder.embed_one("warmup")  # 先加载 bge-m3，避免计时噪声
    summary = run_rules(load_items(), cfg)
    # 改写命中统计（多少条被改写/触发漂移护栏）
    recs = summary["records"]
    summary["rewrite_stats"] = {
        "changed": sum(1 for r in recs if (r.get("rewrite") or {}).get("changed")),
        "drift_rejected": sum(1 for r in recs if (r.get("rewrite") or {}).get("drift")),
        "by_level": {lv: sum(1 for r in recs if (r.get("rewrite") or {}).get("level") == lv)
                     for lv in ("rule", "llm", "none")}}
    ragas = {"skipped": True, "reason": "--no-ragas"} if args.no_ragas else run_ragas(summary, cfg, args.limit)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"eval_results_{args.tag}.json"
    out.write_text(json.dumps({"meta": {
                           "chunks_collection": cfg.retrieval.chunks_collection,
                           "chunks_strategies": cfg.retrieval.chunks_strategies,
                           "hybrid_enabled": cfg.retrieval.hybrid_enabled,
                           "rrf_k": cfg.retrieval.rrf_k,
                           "local_bm25": cfg.retrieval.local_bm25,
                           "hybrid_floor": cfg.retrieval.hybrid_floor,
                           "qa_threshold": cfg.retrieval.qa_threshold,
                           "kp_threshold": cfg.retrieval.kp_threshold,
                           "chunks_threshold": cfg.retrieval.chunks_threshold,
                           "rerank_enabled": cfg.retrieval.rerank.enabled,
                           "rewrite_enabled": cfg.rewrite.enabled,
                           "rewrite_llm": cfg.rewrite.llm_enabled,
                           "drift_threshold": cfg.rewrite.drift_threshold},
                           "rules": {k: v for k, v in summary.items() if k != "records"},
                           "records": summary["records"], "ragas": ragas},
                          ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n规则指标：路由准确率 {summary['routing_accuracy']:.1%}  "
          f"recall@3 {summary.get('recall_at_3')}  "
          f"分类 {json.dumps(summary['per_category'], ensure_ascii=False)}")
    print(f"改写统计：{json.dumps(summary['rewrite_stats'], ensure_ascii=False)}")
    print(f"RAGAS：{json.dumps(ragas.get('mean', ragas), ensure_ascii=False)}")
    print(f"结果已写 {out}")
    store.reset_for_tests()
    return 0


if __name__ == "__main__":
    sys.exit(main())
