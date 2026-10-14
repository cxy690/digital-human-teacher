# -*- coding: utf-8 -*-
"""P6 多轮会话评测：跑 eval_set_multi.jsonl（多组、每组 2-4 轮），产出多轮改写正确率、
主题切换重置正确率、超纲/安全拒答与首轮路由，并留存逐轮明细供报告对照单轮（P4）。

真跑 bge-m3（漂移护栏 + 主题切换余弦）、Ollama qwen2.5:7b（LLM 级指代消解/补全）、
Redis（会话存储；不可用时 session.py 自动降级内存）。只读三存储，不写任何库。

用法：
    python -m rag.online.eval.eval_multi                      # 默认 v2 集合 + 混合 + LLM 改写
    python -m rag.online.eval.eval_multi --tag P6
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.online import config, embedder, pipeline, session  # noqa: E402

EVAL_MULTI = Path(__file__).resolve().parent / "eval_set_multi.jsonl"
REPORT_DIR = ROOT / "data" / "stores" / "reports"


def load_groups() -> list[dict]:
    lines = EVAL_MULTI.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def _check(turn: dict, res: dict) -> list[dict]:
    """逐轮对账 → 问题清单（kind ∈ modality/policy/resolution/topic），空表示该轮通过。"""
    exp = turn.get("expect") or {}
    rewritten = (res.get("rewrite") or {}).get("rewritten", "")
    sess = res.get("session") or {}
    probs: list[dict] = []
    if exp.get("modality") and res["modality"] != exp["modality"]:
        probs.append({"kind": "modality", "msg": f"期望{exp['modality']} 实得{res['modality']}"})
    if exp.get("policy") and res["policy"] != exp["policy"]:
        probs.append({"kind": "policy", "msg": f"期望{exp['policy']} 实得{res['policy']}"})
    for tok in exp.get("resolved_must_contain", []):
        if tok not in rewritten:
            probs.append({"kind": "resolution", "msg": f"改写未消解出『{tok}』"})
    for tok in exp.get("resolved_must_not_contain", []):
        if tok in rewritten:
            probs.append({"kind": "resolution", "msg": f"改写误带旧主题『{tok}』"})
    if "topic_switch" in exp and bool(sess.get("topic_switch")) != bool(exp["topic_switch"]):
        probs.append({"kind": "topic", "msg": f"期望{exp['topic_switch']} 实得{sess.get('topic_switch')}"})
    return probs


def _acc(vals: list[bool]) -> float | None:
    return round(sum(vals) / len(vals), 4) if vals else None


def run(cfg) -> dict:
    groups = load_groups()
    records: list[dict] = []
    # 各维度分桶（仅对声明了相应期望的轮次计数）
    buckets: dict[str, list[bool]] = {"resolution": [], "topic": [], "refuse": [],
                                      "first_routing": [], "overall": []}
    stores: dict[str, int] = {}
    for g in groups:
        sid = f"p6eval-{g['sid']}-{uuid.uuid4().hex[:6]}"   # 每组独立会话，互不串味
        for idx, turn in enumerate(g["turns"]):
            res = pipeline.ask(turn["question"], cfg=cfg, session_id=sid)
            probs = _check(turn, res)
            exp = turn.get("expect") or {}
            sess = res.get("session") or {}
            kinds = {p["kind"] for p in probs}
            has_res_exp = bool(exp.get("resolved_must_contain") or exp.get("resolved_must_not_contain"))
            if has_res_exp:
                buckets["resolution"].append("resolution" not in kinds)
            if "topic_switch" in exp:
                buckets["topic"].append("topic" not in kinds)
            if exp.get("modality") == "refuse":
                buckets["refuse"].append("modality" not in kinds)
            if idx == 0 and exp.get("modality"):
                buckets["first_routing"].append("modality" not in kinds)
            buckets["overall"].append(not probs)
            store = sess.get("store", "none")
            stores[store] = stores.get(store, 0) + 1
            records.append({
                "sid": g["sid"], "title": g["title"], "turn_no": idx + 1,
                "qid": turn["qid"], "question": turn["question"],
                "rewritten": (res.get("rewrite") or {}).get("rewritten"),
                "level": (res.get("rewrite") or {}).get("level"),
                "modality": res["modality"], "policy": res["policy"],
                "matched_by": res["matched_by"], "session": sess,
                "expect": exp, "passed": not probs, "issues": probs})
            flag = "OK" if not probs else "XX"
            rw = "→" + str(res.get("rewrite", {}).get("rewritten")) if (res.get("rewrite") or {}).get("changed") else ""
            print(f"[{flag}] {turn['qid']} {turn['question'][:22]} {rw}"
                  + ("" if not probs else " | " + "; ".join(p["msg"] for p in probs)))
    summary = {
        "n_groups": len(groups), "n_turns": len(records),
        "pronoun_resolution_accuracy": _acc(buckets["resolution"]),
        "n_resolution_turns": len(buckets["resolution"]),
        "topic_switch_accuracy": _acc(buckets["topic"]),
        "n_topic_turns": len(buckets["topic"]),
        "refusal_accuracy": _acc(buckets["refuse"]),
        "n_refuse_turns": len(buckets["refuse"]),
        "first_turn_routing_accuracy": _acc(buckets["first_routing"]),
        "n_first_routing_turns": len(buckets["first_routing"]),
        "overall_turn_accuracy": _acc(buckets["overall"]),
        "session_store_counts": stores,
    }
    return {"summary": summary, "records": records}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P6 多轮会话评测")
    ap.add_argument("--tag", default="P6", help="输出文件名后缀：eval_results_{tag}.json")
    ap.add_argument("--chunks-collection", default="kp_chunks_v2")
    ap.add_argument("--strategies", nargs="*", default=["default", "semantic", "parent_child"])
    ap.add_argument("--hybrid", choices=["on", "off"], default="on")
    args = ap.parse_args(argv)

    cfg = config.get_config()
    cfg.retrieval.chunks_collection = args.chunks_collection
    cfg.retrieval.chunks_strategies = list(args.strategies)
    cfg.retrieval.hybrid_enabled = args.hybrid == "on"
    cfg.rewrite.enabled = True
    cfg.rewrite.llm_enabled = True          # 多轮代词消解/补全必需（需本地 Ollama）
    cfg.session.enabled = True

    print(f"[eval_multi] chunks={cfg.retrieval.chunks_collection} strategies={cfg.retrieval.chunks_strategies} "
          f"hybrid={cfg.retrieval.hybrid_enabled} rewrite=rule+llm session={cfg.session.key_prefix}* ttl={cfg.session.ttl_seconds}s")
    embedder.embed_one("warmup")            # 先加载 bge-m3

    result = run(cfg)
    meta = {"chunks_collection": cfg.retrieval.chunks_collection,
            "chunks_strategies": cfg.retrieval.chunks_strategies,
            "hybrid_enabled": cfg.retrieval.hybrid_enabled,
            "rewrite_enabled": True, "rewrite_llm": True,
            "drift_threshold": cfg.rewrite.drift_threshold,
            "session_enabled": True, "session_ttl_seconds": cfg.session.ttl_seconds,
            "session_max_turns": cfg.session.max_turns,
            "topic_reset_threshold": cfg.session.topic_reset_threshold,
            "context_summary_turns": cfg.session.context_summary_turns}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"eval_results_{args.tag}.json"
    out.write_text(json.dumps({"meta": meta, **result}, ensure_ascii=False, indent=2), encoding="utf-8")

    s = result["summary"]
    print("\n=== P6 多轮评测汇总 ===")
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"结果已写 {out}")
    session.reset_for_tests()
    return 0


if __name__ == "__main__":
    sys.exit(main())
