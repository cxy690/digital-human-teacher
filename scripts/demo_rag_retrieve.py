"""面试演示脚本：RAG 检索链路 + 会话检查点（历史对话落盘）。

用法（项目根执行）：
    python scripts/demo_rag_retrieve.py                 # 默认查询「分数加法」
    python scripts/demo_rag_retrieve.py 三角形的面积     # 自定义查询

展示两件事：
    1) RAG：语料加载 → 中文 bigram 检索 → top-k 命中（含来源文件）
    2) 历史对话：data/raw/checkpoints/<session_id>.json 的会话快照结构
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # 语料走 data/ 相对路径：切到项目根，保证任意 cwd 下都能跑
try:  # Windows 控制台中文输出
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from rag.retriever import build_retriever, load_documents  # noqa: E402


def show_rag(query: str) -> None:
    """展示 RAG：语料 -> 检索 -> top-k 命中。"""
    docs = load_documents()
    retriever = build_retriever()  # 按配置自动选：向量 / 词法
    backend = getattr(retriever, "backend_name", type(retriever).__name__)
    print(f"[1] RAG 语料：{len(docs)} 个文本块　|　当前检索后端：{backend}")
    print("    来源目录：data/textbook · data/teaching_materials · data/curriculum · data/question_bank")
    hits = asyncio.run(retriever.retrieve(query))
    print(f"\n[2] 检索：{query!r} → 命中 {len(hits)} 条（top_k 默认 4，来自 config/settings.py）")
    source_of: dict[str, str] = {}
    for d in docs:
        source_of.setdefault(d.text, d.meta.get("source", "?"))
    for i, text in enumerate(hits, 1):
        print(f"    {i}. {text[:72]}")
        print(f"       来源: {source_of.get(text, '?')}")
    if not hits:
        print("    （未命中：语料中暂无可匹配内容，答疑会自动退化为无 RAG 上下文）")


def show_checkpoints() -> None:
    """展示历史对话落盘：会话检查点 JSON 的结构。"""
    ckpt_dir = ROOT / "data" / "raw" / "checkpoints"
    files = sorted(ckpt_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"\n[3] 历史对话落盘：{ckpt_dir}")
    print(f"    共 {len(files)} 个会话文件（一个会话一个 JSON，断线续聊用）")
    pick: tuple[Path, dict] | None = None
    best_count = 0
    for p in files[:120]:  # 优先挑「对话最多」的会话作为示例，展示效果最好
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        count = len(data.get("history") or [])
        if count > best_count:
            best_count = count
            pick = (p, data)
    if pick is None and files:
        pick = (files[0], json.loads(files[0].read_text(encoding="utf-8")))
    if pick is None:
        print("    （目录为空：跑一次应用后这里会出现会话文件）")
        return
    path, data = pick
    history = data.get("history") or []
    print(f"\n    示例会话：{path.name}")
    print(f"      session_id : {data.get('session_id')}")
    print(f"      student_id : {data.get('student_id')}")
    print(f"      对话条数   : {len(history)}（最多保留最近 200 条）")
    print(f"      白板指令   : {len(data.get('board_ops') or [])} 条（重连后恢复板书画面）")
    for msg in history[-3:]:
        role = msg.get("role", "?")
        text = (msg.get("text") or "").replace("\n", " ")[:64]
        print(f"      [{role}] {text}")
    if not history:
        print("      （这个示例会话还没有对话：本地跑一次应用再回来看）")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "分数加法"
    show_rag(query)
    show_checkpoints()
    print(
        "\n[4] 相关代码：rag/retriever.py（检索核心）· core/harness.py（启动装配 + 降级）"
        "· core/supervisor.py:713（答疑前注入）· core/checkpoint.py（落盘）"
    )
