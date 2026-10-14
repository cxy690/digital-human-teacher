"""运行时写库：会话检查点落盘的同时，把会话镜像进数据库（Navicat 实时可见）。

与「纯 JSON 检查点」的关系：JSON 仍是主存储（断线续聊行为不变），本模块只增加
一份**数据库镜像**，两条路径语义幂等（同 :mod:`db.importer`），先删后插、随时可重放。

设计约束（与项目「每一层都可降级」的原则一致）：

* 默认 ``RUNTIME_DB_WRITE=auto`` 只对 MySQL 生效 —— 云端默认的 SQLite 配置
  保持原行为（不新增任何文件与写入）；
* 写库在**后台线程**执行，同一会话只写最新快照（快速连续落盘自动合并），
  绝不阻塞事件循环；
* 任何失败（MySQL 未启动 / 密码错误 / 磁盘满）只告警一次，不影响答疑主链路。
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from config.logging import get_logger
from config.settings import settings

log = get_logger("db.runtime_sync")

_lock = threading.Lock()
_pending: dict[str, dict[str, Any]] = {}   # 每会话的最新待写快照（写落后时以最新为准）
_running: set[str] = set()                 # 正在后台写库的会话
_warned = False                            # 失败只告警一次（避免刷屏）
_db_ready = False                          # init_db 懒执行标记（幂等，重复也无害）


def enabled() -> bool:
    """当前配置下是否启用运行时写库：off 永关；auto 仅 MySQL；on 强制。"""
    mode = (settings.runtime_db_write or "auto").strip().lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return settings.database_url.startswith("mysql")


def maybe_sync(state: Any) -> None:
    """会话落盘后调用：调度一次后台写库；本函数永不抛错、永不阻塞。"""
    try:
        if not enabled():
            return
        if os.environ.get("PYTEST_CURRENT_TEST"):  # 测试运行期间绝不写真实库
            return
        data = {
            "session_id": str(state.session_id),
            "student_id": str(state.student_id or "unknown"),
            "history": [m.model_dump() for m in state.history[-200:]],
            "saved_at": time.time(),
        }
        sid = data["session_id"]
        with _lock:
            _pending[sid] = data
            if sid in _running:
                return  # 已有线程在写：新快照挂起，写完自动接着写最新版
            _running.add(sid)
        threading.Thread(target=_worker, args=(sid,), daemon=True, name=f"db-sync-{sid[:12]}").start()
    except Exception as err:  # noqa: BLE001 - 调度失败也必须静默（仅告警一次）
        _warn_once(err)


def _worker(sid: str) -> None:
    """后台写库循环：持续取该会话的最新快照写入数据库，直到没有新快照为止。"""
    while True:
        with _lock:
            data = _pending.pop(sid, None)
            if data is None:
                _running.discard(sid)
                return
        try:
            _write(data)
        except Exception as err:  # noqa: BLE001
            _warn_once(err)


def _write(data: dict[str, Any]) -> None:
    """在后台线程内开独立数据库会话写入（SQLAlchemy Session 不跨线程共享）。"""
    from db.base import SessionLocal, init_db
    from db.importer import upsert_session

    global _db_ready
    if not _db_ready:
        init_db()  # 幂等建表：本机首次运行自动建好四张表
        _db_ready = True
    with SessionLocal() as db:
        upsert_session(
            db,
            session_id=data["session_id"],
            student_id=data["student_id"],
            history=data["history"],
            saved_at=data["saved_at"],
        )
        db.commit()
    log.debug("runtime db sync ok: {} ({} msgs)", data["session_id"], len(data["history"]))


def _warn_once(err: Exception) -> None:
    global _warned
    if not _warned:
        _warned = True
        log.warning("runtime db sync failed（后续静默，不影响答疑）: {}", err)
