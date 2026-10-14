# -*- coding: utf-8 -*-
"""查看云端向量库（data/raw/rag_index）里到底存了什么数据。

只读工具，不改任何东西。用法（项目根执行；本地/云端都行）：
    python scripts/inspect_rag_index.py                  # 概览 + 均匀抽样 8 条
    python scripts/inspect_rag_index.py --n 20           # 抽样 20 条
    python scripts/inspect_rag_index.py --source data/question_bank   # 只看某来源
    python scripts/inspect_rag_index.py --keyword 打折    # 只看含关键词的块

展示内容：
    [1] 元信息（模型 / 维度 / 块数 / 生成时间 / 文件大小）
    [2] 向量矩阵统计（形状、是否已归一化）
    [3] 语料来源分布（每个来源文件贡献多少块）
    [4] 抽样内容（doc_id + 来源 + 文本前 110 字）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data/raw/rag_index"

try:  # Windows 控制台中文输出
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402


def load_index():
    """读 meta.json / docs.jsonl / vectors.npy；缺失时给出可操作的提示。"""
    meta_path = INDEX_DIR / "meta.json"
    docs_path = INDEX_DIR / "docs.jsonl"
    vec_path = INDEX_DIR / "vectors.npy"
    if not (meta_path.exists() and docs_path.exists() and vec_path.exists()):
        print(f"[x] 没找到向量库文件：{INDEX_DIR}")
        print("    先构建：python scripts/build_rag_index.py --rebuild")
        raise SystemExit(1)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    docs = [
        json.loads(line)
        for line in docs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = np.load(vec_path)
    return meta, docs, vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 RAG 向量库数据（只读）")
    parser.add_argument("--n", type=int, default=8, help="抽样条数（默认 8）")
    parser.add_argument("--source", default="", help="只看某来源前缀（如 data/question_bank）")
    parser.add_argument("--keyword", default="", help="只看文本包含该关键词的块")
    args = parser.parse_args()

    meta, docs, vectors = load_index()

    created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta.get("created_at", 0)))
    print(f"[1] 元信息：模型 {meta.get('model')} ｜ 维度 {meta.get('dim')} ｜ 块数 {meta.get('count')}"
          f" ｜ 后端 {meta.get('backend')} ｜ 生成于 {created}")
    print(f"    文件：vectors.npy {(INDEX_DIR / 'vectors.npy').stat().st_size // 1024} KB"
          f" ｜ docs.jsonl {(INDEX_DIR / 'docs.jsonl').stat().st_size // 1024} KB")
    if len(docs) != vectors.shape[0]:
        print(f"[!] 语料与向量数量不一致：{len(docs)} vs {vectors.shape[0]}（索引可能需要重建）")

    norms = np.linalg.norm(vectors, axis=1)
    print(f"[2] 向量矩阵：{vectors.shape} {vectors.dtype} ｜ 行范数 min={norms.min():.4f} max={norms.max():.4f}"
          f"（≈1 表示已 L2 归一化，内积即余弦相似度）")

    counts = Counter((d.get("meta") or {}).get("source", "(无来源)") for d in docs)
    print(f"[3] 语料来源分布（共 {len(counts)} 个来源，按块数排序，列前 15）")
    for src, cnt in counts.most_common(15):
        print(f"    {cnt:>5}  {src}")

    picked = list(range(len(docs)))
    if args.source:
        picked = [i for i in picked if str((docs[i].get("meta") or {}).get("source", "")).startswith(args.source)]
    if args.keyword:
        picked = [i for i in picked if args.keyword in str(docs[i].get("text", ""))]

    show = min(args.n, len(picked))
    print(f"[4] 抽样内容（候选 {len(picked)} 条，均匀抽 {show} 条）")
    if not picked:
        print("    （条件过滤后没有匹配的语料块）")
    elif show:
        step = max(1, len(picked) // show)
        for i in picked[::step][:show]:
            doc = docs[i]
            src = (doc.get("meta") or {}).get("source", "?")
            text = str(doc.get("text", "")).replace("\n", " ")
            print(f"    - [{doc.get('doc_id')}] {src}")
            print(f"      {text[:110]}")


if __name__ == "__main__":
    main()
