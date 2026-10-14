# -*- coding: utf-8 -*-
"""Milvus 数据导出工具（RESTful API，零额外依赖；要求 Milvus >= 2.4）。

背景：校园/实验室的内网 Milvus（如 http://10.224.10.30:19530）云端直连不通，
路线是「把数据导出 -> 传到云端 -> 在云端重建向量库」。本工具负责第一步。

用法（需要有网络能访问目标 Milvus 的机器上执行；只需 httpx，项目已有）：
    python scripts/milvus_export.py probe  --uri http://10.224.10.30:19530
    python scripts/milvus_export.py sample --uri http://10.224.10.30:19530 --collection math_qa_text
    python scripts/milvus_export.py export --uri http://10.224.10.30:19530 --collection math_qa_text

产物（export，默认输出到 data/milvus_export/<集合名>/）：
    meta.json       集合结构、索引信息、行数、字段清单（云端重建的依据）
    rows.jsonl      全部非向量字段（每行一条 JSON）
    vectors.jsonl   原向量（与 rows 顺序一一对应；--no-vectors 可跳过）
    texts.jsonl     纯文本行（可直接放进 data/question_bank 等目录当 RAG 语料）

兼容性备注（实测 10.224.10.30 那种部署）：
* 部分版本没有 /v2/vectordb/databases/list 路由 —— probe 会容忍并继续；
* describe 里的参数既有 {"params": {"dim": 1024}} 写法，也有
  {"params": [{"key": "dim", "value": "1024"}]} 写法，两种都解析；
* 主键标记既有 isPrimary 也有 primaryKey，两种都识别。

说明：REST 接口走 HTTP（与 gRPC 同在默认 19530 端口）。若报 404 且
collections/list 也不通，多半是 Milvus < 2.4 没有 REST 接口，告诉我改用
pymilvus 单独建隔离环境再导。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover
    print("[x] 需要 httpx：pip install httpx")
    raise SystemExit(1)

try:  # Windows 控制台中文输出
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 文本字段自动识别顺序（按可能性从高到低）
TEXT_FIELD_CANDIDATES = ["text", "content", "question", "sentence", "chunk", "qa", "title"]


class MilvusRestError(RuntimeError):
    """Milvus REST 调用失败（网络 / 鉴权 / 协议 / 业务错误）。"""


class MilvusRest:
    """Milvus RESTful API 极简客户端（v2，Milvus >= 2.4）。"""

    def __init__(self, uri: str, token: str = "", timeout: float = 30.0) -> None:
        self.base = uri.strip().rstrip("/")
        self.token = token
        self.timeout = timeout

    def post(self, path: str, payload: dict) -> object:
        url = f"{self.base}{path}"
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        except Exception as err:  # noqa: BLE001 - 统一收敛为可读错误
            raise MilvusRestError(f"请求失败（{url}）：{err}") from err
        if resp.status_code != 200:
            raise MilvusRestError(f"HTTP {resp.status_code}（{url}）：{resp.text[:300]}")
        data = resp.json()
        if data.get("code", 0) != 0:
            raise MilvusRestError(f"接口报错（{url}）：{data.get('message', data)}")
        return data.get("data")

    # ------------------------------------------------------------ 常用接口
    def list_databases(self) -> object:
        return self.post("/v2/vectordb/databases/list", {})

    def list_collections(self, db: str) -> object:
        return self.post("/v2/vectordb/collections/list", {"dbName": db})

    def describe(self, db: str, coll: str) -> dict:
        return self.post("/v2/vectordb/collections/describe", {"dbName": db, "collectionName": coll})  # type: ignore[return-value]

    def stats(self, db: str, coll: str) -> dict:
        return self.post("/v2/vectordb/collections/get_stats", {"dbName": db, "collectionName": coll})  # type: ignore[return-value]

    def query(self, db: str, coll: str, *, fields: list[str], limit: int, offset: int, filter: str = "") -> list[dict]:
        data = self.post(
            "/v2/vectordb/entities/query",
            {"dbName": db, "collectionName": coll, "filter": filter,
             "outputFields": fields, "limit": limit, "offset": offset},
        )
        return data if isinstance(data, list) else []  # type: ignore[return-value]

    def list_indexes(self, db: str, coll: str) -> object:
        return self.post("/v2/vectordb/indexes/list", {"dbName": db, "collectionName": coll})

    def describe_index(self, db: str, coll: str, name: str) -> dict:
        return self.post("/v2/vectordb/indexes/describe",
                         {"dbName": db, "collectionName": coll, "indexName": name})  # type: ignore[return-value]


# ------------------------------------------------------------------ 小工具
def _as_list(data: object, key: str) -> list:
    """兼容不同版本的返回包装：['a'] 或 {"databases": ['a']} 或 {"collections": [...]}。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in (key, "data", "items"):
            v = data.get(k)
            if isinstance(v, list):
                return v
        return list(data.values()) if data else []
    return []


def _fields(desc: dict) -> list[dict]:
    return list(desc.get("fields") or [])


def _is_vector(field: dict) -> bool:
    return "vector" in str(field.get("type", "")).lower()


def _is_float_vector(field: dict) -> bool:
    return "floatvector" in str(field.get("type", "")).lower()


def _is_text_like(field: dict) -> bool:
    t = str(field.get("type", "")).lower()
    return (not _is_vector(field)) and ("char" in t or t == "string")


def _param(field: dict, key: str, default: object = "?") -> object:
    """读字段参数：兼容 {"params": {"dim": 1024}} 与 {"params": [{"key": "dim", "value": "1024"}]}。"""
    params = field.get("params")
    if isinstance(params, dict):
        return params.get(key, default)
    if isinstance(params, list):
        for item in params:
            if isinstance(item, dict) and item.get("key") == key:
                return item.get("value", default)
    return default


def _dim(field: dict) -> object:
    return _param(field, "dim")


def _pick_id_field(fields: list[dict]) -> str:
    for f in fields:  # 主键标记：isPrimary（v2 REST）或 primaryKey（部分版本）
        if f.get("isPrimary") or f.get("primaryKey") or f.get("primary"):
            return f["name"]
    for f in fields:
        if not _is_vector(f):
            return f["name"]
    return "id"


def _pick_text_field(fields: list[dict], given: str) -> str:
    text_names = [f["name"] for f in fields if _is_text_like(f)]
    if given:
        if given in [f["name"] for f in fields]:
            return given
        print(f"[!] --text-field {given!r} 不存在，改为自动识别")
    for cand in TEXT_FIELD_CANDIDATES:
        if cand in text_names:
            return cand
    return text_names[0] if text_names else ""


# ------------------------------------------------------------------ 子命令
def probe(client: MilvusRest, args: argparse.Namespace) -> None:
    print(f"[1] 连接 {client.base} …")
    try:
        dbs = _as_list(client.list_databases(), "databases")
        print(f"    数据库：{dbs}")
    except MilvusRestError as err:
        print(f"    （该部署没有 databases/list 路由，跳过：{err}）")
    colls = _as_list(client.list_collections(args.db), "collections")
    print(f"[2] {args.db} 库内集合（{len(colls)} 个）")
    for name in colls[:50]:
        try:
            st = client.stats(args.db, name)
            rows = st.get("rowCount", st) if isinstance(st, dict) else st
        except MilvusRestError:
            rows = "?"
        print(f"    - {name}: {rows} 行")
    print("[3] 连通性正常。下一步：sample 看数据结构，export 导出全量")


def sample(client: MilvusRest, args: argparse.Namespace) -> None:
    desc = client.describe(args.db, args.collection)
    fields = _fields(desc)
    print(f"[1] 集合 {args.collection}：{len(fields)} 个字段"
          f"（一致性 {desc.get('consistencyLevel', '?')}）")
    for f in fields:
        extra = ""
        if _is_vector(f):
            extra = f"  dim={_dim(f)}"
        elif _is_text_like(f):
            extra = f"  max_length={_param(f, 'max_length')}"
        print(f"    - {f['name']}  {f.get('type', '?')}{extra}")
    for idx in (desc.get("indexes") or []):
        if isinstance(idx, dict):
            print(f"    索引：{idx.get('indexName')} on {idx.get('fieldName')}"
                  f"（metric {idx.get('metricType', '?')}）")
    scalar = [f["name"] for f in fields if not _is_vector(f)]
    rows = client.query(args.db, args.collection, fields=scalar, limit=max(1, args.rows), offset=0)
    print(f"[2] 前 {len(rows)} 行（向量字段未展开）：")
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))


def export(client: MilvusRest, args: argparse.Namespace) -> None:
    desc = client.describe(args.db, args.collection)
    fields = _fields(desc)
    scalar = [f["name"] for f in fields if not _is_vector(f)]
    float_vectors = [f["name"] for f in fields if _is_float_vector(f)]
    other_vectors = [f["name"] for f in fields if _is_vector(f) and f["name"] not in float_vectors]
    with_vectors = (not args.no_vectors) and bool(float_vectors)

    text_field = _pick_text_field(fields, args.text_field)
    id_field = _pick_id_field(fields)
    fetch = scalar + (float_vectors if with_vectors else [])

    try:
        st = client.stats(args.db, args.collection)
        expected = st.get("rowCount") if isinstance(st, dict) else None
    except MilvusRestError:
        expected = None

    out = Path(args.out) if args.out else Path("data/milvus_export") / args.collection
    out.mkdir(parents=True, exist_ok=True)
    print(f"[1] 导出 {args.db}.{args.collection} -> {out.resolve()}")
    print(f"    字段：标量 {len(scalar)} 个；向量 {len(float_vectors)} 个"
          f"{'（本次导出）' if with_vectors else '（跳过）'}；主键 {id_field!r}；文本字段 {text_field!r}"
          f"{'；预计 ' + str(expected) + ' 行' if expected is not None else ''}")
    if other_vectors:
        print(f"[!] 非浮点向量字段跳过（{other_vectors}）：REST 下无法可靠还原，如需导出告诉我加 pymilvus 方案")
    if not text_field:
        print("[!] 未识别到文本字段：texts.jsonl 将不生成（可加 --text-field 手动指定）")

    batch = max(1, args.batch)
    total = 0
    prev_ids: list | None = None
    started = time.time()
    rows_path = out / "rows.jsonl"
    vec_path = out / "vectors.jsonl"
    text_path = out / "texts.jsonl"
    with rows_path.open("w", encoding="utf-8") as rows_f, \
            (vec_path.open("w", encoding="utf-8") if with_vectors else _NullCtx()) as vec_f, \
            (text_path.open("w", encoding="utf-8") if text_field else _NullCtx()) as text_f:
        offset = 0
        while True:
            rows = client.query(args.db, args.collection, fields=fetch, limit=batch, offset=offset, filter=args.filter)
            if not rows:
                break
            ids = [r.get(id_field) for r in rows]
            if ids == prev_ids:
                print("[!] 服务端似乎未生效 offset（返回了重复数据），为避免重复已停止导出")
                break
            prev_ids = ids
            for r in rows:
                row_scalar = {k: v for k, v in r.items() if k not in float_vectors}
                rows_f.write(json.dumps(row_scalar, ensure_ascii=False, default=str) + "\n")
                if with_vectors:
                    vec_f.write(json.dumps({k: r.get(k) for k in float_vectors}, ensure_ascii=False) + "\n")
                if text_field:
                    t = r.get(text_field)
                    if isinstance(t, str) and t.strip():
                        text_f.write(t.replace("\n", " ").strip() + "\n")
            total += len(rows)
            offset += len(rows)
            print(f"    已导出 {total} 行…", flush=True)
            # 不用「本页不满 batch」判断结束：部分服务端对单页条数有上限，
            # 提前 break 会漏数据；改为一直拉到空页（重复页由上面的保护拦下）。

    indexes: list[dict] = []
    try:
        for name in _as_list(client.list_indexes(args.db, args.collection), "indexes"):
            if isinstance(name, str):
                try:
                    indexes.append({"name": name, "detail": client.describe_index(args.db, args.collection, name)})
                except MilvusRestError:
                    indexes.append({"name": name})
    except MilvusRestError:
        pass

    meta = {
        "uri": client.base,
        "db": args.db,
        "collection": args.collection,
        "num_rows": total,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "id_field": id_field,
        "text_field": text_field,
        "vector_fields": [{"name": f["name"], "dim": _dim(f)} for f in fields if _is_float_vector(f)],
        "describe": desc,
        "indexes": indexes,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    cost = time.time() - started
    print(f"[2] 完成：{total} 行，用时 {cost:.1f}s")
    print(f"    {rows_path.name}" + (f" + {vec_path.name}" if with_vectors else "")
          + (f" + {text_path.name}" if text_field else "") + " + meta.json")
    print(f"    目录：{out.resolve()}")
    print("    下一步：python scripts/sync_cloud.py 可把该目录传到云端，然后重建向量库接入 RAG")


class _NullCtx:
    """空上下文管理器：字段不需要导出时占位，避免到处写 if。"""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ------------------------------------------------------------------ 入口
def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus 数据导出工具（RESTful API，要求 Milvus >= 2.4）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--uri", default=os.getenv("MILVUS_URI", ""), help="Milvus 地址，如 http://10.224.10.30:19530")
        p.add_argument("--token", default=os.getenv("MILVUS_TOKEN", ""), help="鉴权 token（user:password），未开鉴权可不填")
        p.add_argument("--db", default="default", help="数据库名（默认 default）")
        p.add_argument("--timeout", type=float, default=30.0, help="单次请求超时秒数（默认 30）")

    p_probe = sub.add_parser("probe", help="探测连通性：列出数据库/集合/行数")
    add_common(p_probe)

    p_sample = sub.add_parser("sample", help="抽样看前几行数据与字段结构")
    add_common(p_sample)
    p_sample.add_argument("--collection", required=True, help="集合名（如 math_qa_text）")
    p_sample.add_argument("--rows", type=int, default=3, help="抽样行数（默认 3）")

    p_export = sub.add_parser("export", help="全量导出到本地文件")
    add_common(p_export)
    p_export.add_argument("--collection", required=True, help="集合名（如 math_qa_text）")
    p_export.add_argument("--out", default="", help="输出目录（默认 data/milvus_export/<集合名>）")
    p_export.add_argument("--batch", type=int, default=1000, help="每页拉取行数（默认 1000）")
    p_export.add_argument("--filter", default="", help="只导出满足条件的行（默认全量）")
    p_export.add_argument("--no-vectors", action="store_true", help="不导出向量，只导文本与标量字段")
    p_export.add_argument("--text-field", default="", help="指定文本字段名（默认自动识别 text/content/question 等）")

    args = parser.parse_args()
    if not args.uri:
        print("[x] 请用 --uri 指定 Milvus 地址，如：--uri http://10.224.10.30:19530")
        raise SystemExit(2)
    client = MilvusRest(args.uri, args.token, args.timeout)
    try:
        if args.cmd == "probe":
            probe(client, args)
        elif args.cmd == "sample":
            sample(client, args)
        else:
            export(client, args)
    except MilvusRestError as err:
        print(f"[x] {err}")
        if "404" in str(err):
            print("    提示：404 常见于该版本没有对应 REST 路由（如 databases/list），或 Milvus < 2.4 无 REST 接口")
        else:
            print("    排查：地址/端口是否正确、当前网络能否访问该地址、服务是否在运行、是否需要 token")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
