# -*- coding: utf-8 -*-
"""在线链路只读存储访问层（retrieval / policy / compose 共用）。

铁律落地：本层**只允许 SELECT**。:func:`query` 拒绝任何非只读语句，从入口杜绝误写库。
Milvus 客户端也只用于 search/query（读），写库永远只在离线链路。
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from . import config as _cfg

_READ_ONLY_PREFIXES = ("SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN")

_CONN = None
_MILVUS = None


def _assert_read_only(sql: str) -> None:
    head = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
    if head not in _READ_ONLY_PREFIXES:
        raise PermissionError(f"在线链路只读，拒绝执行 {head or '(空)'} 语句")


def connection():
    """惰性建立并复用一条只读 pymysql 连接。"""
    global _CONN
    import pymysql

    if _CONN is None:
        my = _cfg.get_config().mysql
        _CONN = pymysql.connect(
            host=my.host, port=my.port, user=my.user, password=my.password,
            database=my.database, charset=my.charset,
            cursorclass=pymysql.cursors.DictCursor, autocommit=True)
    else:
        _CONN.ping(reconnect=True)
    return _CONN


def query(sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
    """执行只读查询，返回字典行列表；非 SELECT/WITH/SHOW 一律拒绝。"""
    _assert_read_only(sql)
    cur = connection().cursor()
    try:
        cur.execute(sql, tuple(params) if params else None)
        return list(cur.fetchall())
    finally:
        cur.close()


def milvus():
    """惰性返回 MilvusClient（仅用于读）。"""
    global _MILVUS
    if _MILVUS is None:
        from pymilvus import MilvusClient

        _MILVUS = MilvusClient(uri=_cfg.get_config().milvus.uri)
    return _MILVUS


def placeholders(values: Iterable[Any]) -> str:
    return ", ".join(["%s"] * len(list(values)))


def reset_for_tests() -> None:
    """测试收尾释放进程级连接（不参与生产路径）。"""
    global _CONN, _MILVUS
    if _CONN is not None:
        try:
            _CONN.close()
        except Exception:  # noqa: BLE001
            pass
    _CONN = None
    _MILVUS = None
