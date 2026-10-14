# -*- coding: utf-8 -*-
"""命令行入口（链路第 9 步）。

单轮：``python -m rag.online.cli "什么是有理数"``（可加 ``--json``）。
多轮：``python -m rag.online.cli --chat`` —— 交互式会话，按 session_id 维持上下文，
      支持代词追问消解与主题切换重置（P6）。Ctrl-D/Z 或输入 ``exit`` 退出。
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid

from . import config, pipeline


def _render(res: dict, *, show_rewrite: bool = False) -> str:
    lat = res["latency_ms"]
    lines = [
        f"答案({res['modality']}/{res['policy']}): {res['answer']}",
        f"命中: {res['matched_by']}  会话: {res['session_id']}",
        f"耗时ms: total={lat['total']} retrieve={lat['retrieve']} generate={lat['generate']}",
    ]
    rw = res.get("rewrite") or {}
    if show_rewrite and rw.get("changed"):
        lines.append(f"改写[{rw.get('level')}]: {rw.get('original')} → {rw.get('rewritten')}"
                     + (f"（漂移回退 cos={rw.get('cosine')}）" if rw.get("drift") else ""))
    s = res.get("session")
    if s:
        lines.append(f"会话轮次: 第{s['turn_no']}轮 store={s['store']} "
                     f"上下文{s['context_turns']}轮" + ("（已换主题，上下文重置）" if s["topic_switch"] else ""))
    for c in res["citations"]:
        lines.append(f"  引用 [{c['kp_id']}] p{c['page_no']} score={c['score']}: "
                     f"{c['evidence'][:40]}")
    return "\n".join(lines)


def _chat(cfg, args) -> int:
    sid = args.session_id or uuid.uuid4().hex[:8]
    # 多轮的意义在代词/省略消解：交互模式默认开启 LLM 级改写（需本地 Ollama）。
    cfg.rewrite.llm_enabled = True
    print(f"[多轮会话 {sid}] 输入问题回车提问；exit/quit 或 Ctrl-D 退出。")
    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", ":q"):
            break
        res = pipeline.ask(line, cfg=cfg, session_id=sid)
        if args.json:
            print(json.dumps(res, ensure_ascii=False))
        else:
            print("AI> " + _render(res, show_rewrite=True))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m rag.online.cli",
                                description="在线 RAG 检索增强问答（对三存储只读）")
    p.add_argument("question", nargs="*", help="问题；留空则从 stdin 读取")
    p.add_argument("--json", action="store_true", help="输出统一结构 JSON（含 session_id，供评测/下游消费）")
    p.add_argument("--backend", choices=["ollama", "dashscope"], help="覆盖生成后端")
    p.add_argument("--session-id", dest="session_id", help="会话 ID；--chat 下延续同一会话上下文")
    p.add_argument("--chat", action="store_true", help="多轮交互模式（P6）：维持会话、代词消解、主题切换重置")
    args = p.parse_args(argv)

    cfg = config.get_config()
    if args.backend:
        cfg.generation.backend = args.backend

    if args.chat:
        return _chat(cfg, args)

    query = " ".join(args.question).strip() or sys.stdin.read().strip()
    if not query:
        p.error("问题为空")

    res = pipeline.ask(query, cfg=cfg, session_id=args.session_id)
    print(json.dumps(res, ensure_ascii=False) if args.json else _render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
